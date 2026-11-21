//! Port of `smartloop/agents/orchestration/model_router.py`.
//!
//! Model selection and progress reporting.
//!
//! Which LLM runs a given node, and telling the client what is happening while
//! it runs.  The two belong together: resolving a capability model is exactly
//! the moment the client needs to be told a model is loading, and the answer to
//! "which model" is the answer to "what do we report".
//!
//! This is the leaf of the orchestration graph — it depends on nothing else.

use std::collections::HashSet;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::agents::planner::ModelTier;
use crate::constants::{display_model_label, display_model_name};
use crate::llms::BaseLLM;
use crate::llms::cloud_llm::local_only_scope;
use crate::types::ModelCapability;

/// Python logger name of this module.
pub const LOGGER: &str = "smartloop.agents.orchestration.model_router";

/// Output cap for decision/classification LLM calls (yes/no, relevance, a short
/// condition question).  Their answer is a handful of tokens, but without a cap
/// the model may fail to emit a stop token and ramble up to the full context
/// window — and with several matched skills those runaways stack into a
/// multi-minute stall.  Content-generating calls (extract/analyze/synthesize)
/// are intentionally NOT capped: they stay bounded by the model's context
/// window, so capping would only risk truncating a legitimate answer.
pub const DECISION_MAX_TOKENS: usize = 64;

/// Cap for short free-text extractions (e.g. a 1-3 sentence formatting hint).
pub const SHORT_TEXT_MAX_TOKENS: usize = 256;

/// `progress_callback(step, status, message)` — the per-request progress sink
/// supplied by the streaming API layer.  Invoked from DAG worker threads, so it
/// must be thread-safe; a panic in it must never break orchestration.
pub type ProgressCallback = Arc<dyn Fn(&str, &str, &str) + Send + Sync>;

/// Loads the generic large model on demand (Python `large_llm_provider`).
pub type LargeLlmProvider =
    Arc<dyn Fn() -> anyhow::Result<Option<Arc<dyn BaseLLM>>> + Send + Sync>;

/// Resolves the model declaring a capability (Python `capability_llm_provider`).
pub type CapabilityLlmProvider =
    Arc<dyn Fn(&str) -> anyhow::Result<Option<Arc<dyn BaseLLM>>> + Send + Sync>;

/// Answers "does any selected model declare this capability?" without loading
/// anything (Python `capability_available`).
pub type CapabilityAvailable = Arc<dyn Fn(&str) -> anyhow::Result<bool> + Send + Sync>;

/// Answers "why did the last capability model not load?" (Python
/// `capability_skip_reason`, which returned a `(model_name, cause[, detail])`
/// tuple or `None`).
pub type CapabilitySkipReasonFn =
    Arc<dyn Fn() -> anyhow::Result<Option<CapabilitySkipReason>> + Send + Sync>;

/// Why a capability model was skipped: Python's `(model_name, cause, detail?)`
/// tuple.  `cause` is one of `memory`, `missing`, `faulted`, `error`; `detail`
/// carries the figures behind the cause ("needs ~4.9GB, 4.4GB free").
///
/// A record whose `cause` is empty stands for Python's malformed (too short)
/// tuple: it explains nothing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CapabilitySkipReason {
    pub model_name: String,
    pub cause: String,
    pub detail: Option<String>,
}

impl CapabilitySkipReason {
    /// Build a reason without a detail clause.
    pub fn new(model_name: impl Into<String>, cause: impl Into<String>) -> Self {
        Self { model_name: model_name.into(), cause: cause.into(), detail: None }
    }

    /// Build a reason carrying a detail clause.
    pub fn with_detail(
        model_name: impl Into<String>,
        cause: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self { model_name: model_name.into(), cause: cause.into(), detail: Some(detail.into()) }
    }
}

/// Optional collaborators of a [`ModelRouter`] (Python's keyword arguments).
#[derive(Clone, Default)]
pub struct ModelRouterOptions {
    pub large_llm: Option<Arc<dyn BaseLLM>>,
    pub large_llm_provider: Option<LargeLlmProvider>,
    pub capability_llm_provider: Option<CapabilityLlmProvider>,
    pub capability_available: Option<CapabilityAvailable>,
    pub capability_skip_reason: Option<CapabilitySkipReasonFn>,
    /// `agent_llm` is the request's one model: it runs every text node —
    /// routing, decisions, tool resolution, light tasks — local or remote.
    /// Another model is consulted only for a capability step it does not
    /// cover (see [`ModelRouter::llm_for_tier`]).
    pub pinned: bool,
}

/// Client-facing labels for routing keys: the capabilities, and the answer
/// tier.  The wire values are internal terms; map them to human-readable
/// text before surfacing in SSEs.
pub const CAPABILITY_LABELS: &[(&str, &str)] = &[
    ("text", "text"),
    ("reasoning", "reasoning"),
    ("vision", "vision"),
    ("thinking", "thinking"),
    (crate::types::ANSWER_TIER, "the answer"),
];

/// Why a capability model was skipped, phrased for the user.  `memory` is the
/// one they can act on — close something, or deselect a model — so it says so
/// plainly instead of leaving the fallback looking like their selection was
/// ignored.  `{model}` is substituted with the display model name.
pub const SKIP_REASONS: &[(&str, &str)] = &[
    ("memory", "not enough memory for {model}"),
    ("missing", "{model} is not downloaded"),
    ("faulted", "{model} ran out of memory"),
    ("error", "{model} could not be loaded"),
];

/// What the user is told a reasoning model was picked for.  `thinking` is an
/// internal routing tag — it is excluded from the capabilities the API reports
/// and it names a property of the model rather than the work, so "Selected
/// qwen3.5-2b for thinking" tells the reader nothing about what is happening.
pub const RESOLUTION_PURPOSE: &str = "tool resolution";

/// What the user is told a model was picked for when it runs a planned step's
/// decision (resolving a tool, extracting its arguments).  The step's
/// capability is usually the answer tier, and "Selected sl-mini for the
/// answer" read as though the small model had written the reply.
pub const STEP_PURPOSE: &str = "a planned step";

/// What the user is told a model was picked for when it ranks web results.
pub const WEB_SEARCH_PURPOSE: &str = "web search";

/// True when two handles point at the same model instance (Python `is`).
fn same_llm(a: &Arc<dyn BaseLLM>, b: &Arc<dyn BaseLLM>) -> bool {
    std::ptr::eq(Arc::as_ptr(a) as *const (), Arc::as_ptr(b) as *const ())
}

/// Resolves the LLM for a node and reports progress to the client.
///
/// `progress_callback(step, status, message)` — supplied per request by the
/// streaming API layer — is invoked as each step runs so the client can
/// surface live activity.  It fires from DAG worker threads, so it must be
/// thread-safe; a failure in it must never break orchestration.
pub struct ModelRouter {
    pub orchestrator_llm: Arc<dyn BaseLLM>,
    pub agent_llm: Arc<dyn BaseLLM>,
    /// Generic large model; loaded lazily from `large_llm_provider`.
    pub large_llm: Mutex<Option<Arc<dyn BaseLLM>>>,
    large_llm_provider: Option<LargeLlmProvider>,
    capability_llm_provider: Option<CapabilityLlmProvider>,
    /// Answers "does any selected model declare this capability?" without
    /// loading anything, so a load is only announced when one can happen.
    capability_available: Option<CapabilityAvailable>,
    /// Answers "why did the last capability model not load?", so a fallback
    /// can say what went wrong.
    capability_skip_reason: Option<CapabilitySkipReasonFn>,
    /// The whole graph runs on `agent_llm` (see [`ModelRouterOptions::pinned`]).
    pinned: bool,
    /// Per-request state, reset by [`ModelRouter::begin_request`].
    progress_callback: Mutex<Option<ProgressCallback>>,
    emitted_model_events: Mutex<HashSet<String>>,
}

impl ModelRouter {
    /// Build a router over the orchestrator/agent models plus optional providers.
    pub fn new(
        orchestrator_llm: Arc<dyn BaseLLM>,
        agent_llm: Arc<dyn BaseLLM>,
        options: ModelRouterOptions,
    ) -> Self {
        Self {
            orchestrator_llm,
            agent_llm,
            large_llm: Mutex::new(options.large_llm),
            large_llm_provider: options.large_llm_provider,
            capability_llm_provider: options.capability_llm_provider,
            capability_available: options.capability_available,
            capability_skip_reason: options.capability_skip_reason,
            pinned: options.pinned,
            progress_callback: Mutex::new(None),
            emitted_model_events: Mutex::new(HashSet::new()),
        }
    }

    /// True for a capability that adds an input modality rather than text
    /// work: the one reason a pinned request consults another model.
    fn augments_modality(capability: &str) -> bool {
        capability.eq_ignore_ascii_case(ModelCapability::Vision.value())
    }

    /// True when every text node runs on the request's one model.
    pub fn is_pinned(&self) -> bool {
        self.pinned
    }

    /// True when a generic large model can still be loaded on demand.
    pub fn has_large_provider(&self) -> bool {
        self.large_llm_provider.is_some()
    }

    /// The progress sink installed for the current request, if any (Python
    /// `router._progress_callback`).
    pub fn progress_callback(&self) -> Option<ProgressCallback> {
        self.progress_callback.lock().clone()
    }

    /// Reset per-request state and install this request's progress sink.
    ///
    /// The model-selection dedupe set has to be cleared per request: a
    /// capability's model is resolved once per DAG node (once per document in
    /// a fan-out, say), but the user only needs to be told which model was
    /// picked once.
    pub fn begin_request(&self, progress_callback: Option<ProgressCallback>) {
        *self.progress_callback.lock() = progress_callback;
        self.emitted_model_events.lock().clear();
    }

    /// Resolve the model for a node, reporting the choice to the client.
    ///
    /// `capability` routes; `purpose` — when given — is what the user is told
    /// the model is for.  The two are not the same thing.  Synthesis routes on
    /// the `skills` capability because a skill's answer should be written by a
    /// skills-capable model, but the user is watching the final answer being
    /// written, and "Loading model for reasoning…" names an internal routing
    /// tag rather than the work in progress.
    pub fn llm_for_tier(
        &self,
        tier: ModelTier,
        capability: Option<&str>,
        purpose: Option<&str>,
    ) -> Arc<dyn BaseLLM> {
        let capability = capability.filter(|c| !c.is_empty());
        // One model for the request: every text step — the answer tier,
        // reasoning, a planned step — stays on it at any tier.  Only a
        // capability that adds an input modality the text model may lack
        // (vision) is resolved separately.
        if self.pinned && !capability.is_some_and(Self::augments_modality) {
            return self.agent_llm.clone();
        }
        if tier == ModelTier::Large {
            if let (Some(cap), Some(provider)) = (capability, self.capability_llm_provider.as_ref()) {
                // Don't cache the capability LLM here — the provider
                // (model_service) is the authoritative cache and may evict a
                // prior step's model to fit this one. Holding our own strong
                // reference would pin that VRAM and defeat the eager release,
                // so a subsequent capability (e.g. skills after dp) can't fit.
                //
                // Only announce the load when there is something to load.
                // This event fires *before* the provider is consulted (its
                // whole point is to precede a slow load), so on a workspace
                // where no selected model declares the capability it used to
                // promise a load that never happened — "Loading model for
                // skills…" on a request with no skills-capable model, then a
                // fallback to the bootstrap model.
                if self.capability_model_exists(Some(cap)) {
                    self.emit_model_loading(Some(cap), purpose);
                }
                match provider(cap) {
                    Ok(Some(llm)) => {
                        tracing::info!(
                            target: LOGGER,
                            "[models] capability '{}' → {}",
                            cap,
                            Self::model_label(llm.as_ref())
                        );
                        self.emit_model_selected(Some(cap), llm.as_ref(), purpose);
                        return llm;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        tracing::warn!(
                            target: LOGGER,
                            "[models] capability model load failed ({}): {}",
                            cap,
                            e
                        );
                    }
                }
            }
            // No capability-specific model available — fall through to the
            // generic large model so synthesis / analysis still gets the
            // best available model rather than the small orchestrator.
            let fallback = {
                let mut large = self.large_llm.lock();
                if large.is_none() {
                    if let Some(provider) = self.large_llm_provider.as_ref() {
                        *large = match provider() {
                            Ok(llm) => llm,
                            Err(e) => {
                                tracing::warn!(
                                    target: LOGGER,
                                    "[models] large model load failed: {}",
                                    e
                                );
                                None
                            }
                        };
                    }
                }
                large.clone().unwrap_or_else(|| self.agent_llm.clone())
            };
            if let Some(cap) = capability {
                tracing::warn!(
                    target: LOGGER,
                    "[models] capability '{}' unresolved — falling back to {}",
                    cap,
                    Self::model_label(fallback.as_ref())
                );
                self.emit_model_fallback(Some(cap), fallback.as_ref(), purpose);
            }
            return fallback;
        }
        // A capability tag routes at every tier, not only LARGE.  A SMALL step
        // is a decision node — picking an MCP tool, extracting its arguments,
        // judging relevance — so it routes *best-effort*: decision_llm takes the
        // capability model only when it is already resident or comfortably
        // fits, and otherwise lands back on agent_llm without forcing a load.
        //
        // Dropping the tag here is what kept every MCP tool call on the
        // bootstrap router: pipeline_builder tags a non-skill TOOL step SMALL
        // and still sets its capability, and that capability was read by
        // nothing.  It only surfaced with two or more models selected — a
        // single selection is pinned into agent_llm upstream, so the default
        // happened to be the user's model.
        self.decision_llm(capability, purpose.or(Some(STEP_PURPOSE)))
    }

    /// A model that can reason, for a call whose judgement warrants it.
    ///
    /// MCP tool resolution is that call: one pass over every discovered tool
    /// that has to pick the tool *and* fill its required arguments, and a
    /// considered answer beats a guess.  The router model only routes, so it
    /// never makes that call while a better one is selected.  In order:
    ///
    ///   1. the best model declaring `ModelCapability::Reasoning` — the
    ///      largest, local or remote, with the local one winning a size tie;
    ///   2. a model declaring `ModelCapability::Thinking`, on device, when it
    ///      is resident or comfortably fits;
    ///   3. `fallback`, the model the step would otherwise use.
    ///
    /// Neither step forces the generic large-model load.  A `fallback` that
    /// already reasons is kept as-is — no lookup, no switch.  A candidate is
    /// announced only once it is accepted, so a thinking lookup that lands on
    /// a model that cannot think never names it as the pick.
    pub fn reasoning_llm(&self, fallback: Arc<dyn BaseLLM>) -> Arc<dyn BaseLLM> {
        // One model for the request: switching to another for a judgement
        // call costs a load (or a second provider) and the tokens/s of the
        // model already warm.
        if self.pinned || fallback.supports_thinking() {
            return fallback;
        }
        let reasoning = ModelCapability::Reasoning.value();
        if self.capability_model_exists(Some(reasoning)) {
            if let Some(llm) = self.resolve(reasoning, false, "reasoning").filter(|l| !same_llm(l, &fallback)) {
                tracing::info!(
                    target: LOGGER,
                    "[models] tool resolution → {} (reasoning)",
                    Self::model_label(llm.as_ref())
                );
                self.announce(reasoning, &llm, Some(RESOLUTION_PURPOSE));
                return llm;
            }
        }
        let thinking = ModelCapability::Thinking.value();
        if self.capability_model_exists(Some(thinking)) {
            if let Some(llm) = self.resolve(thinking, true, "decision").filter(|l| l.supports_thinking()) {
                tracing::info!(
                    target: LOGGER,
                    "[models] tool resolution → {} (thinking)",
                    Self::model_label(llm.as_ref())
                );
                self.announce(thinking, &llm, Some(RESOLUTION_PURPOSE));
                return llm;
            }
        }
        fallback
    }

    /// The model that classifies and plans a request: the best selected
    /// model declaring `ModelCapability::Reasoning` — local or remote, the
    /// largest first — or `fallback` (the on-device router) when none is
    /// selected.  Not announced: routing is not work the user is waiting on.
    pub fn route_llm(&self, fallback: Arc<dyn BaseLLM>) -> Arc<dyn BaseLLM> {
        if self.pinned {
            return self.agent_llm.clone();
        }
        let reasoning = ModelCapability::Reasoning.value();
        if !self.capability_model_exists(Some(reasoning)) {
            return fallback;
        }
        self.resolve(reasoning, false, "reasoning").unwrap_or(fallback)
    }

    /// True when some selected model declares `capability`.
    ///
    /// Unknown (no predicate wired in) counts as true, so a caller that does
    /// not supply one keeps the previous behaviour.
    fn capability_model_exists(&self, capability: Option<&str>) -> bool {
        let (Some(capability), Some(available)) =
            (capability.filter(|c| !c.is_empty()), self.capability_available.as_ref())
        else {
            return true;
        };
        match available(capability) {
            Ok(exists) => exists,
            Err(e) => {
                tracing::debug!(
                    target: LOGGER,
                    "[models] capability availability check failed for {}: {}",
                    capability,
                    e
                );
                true
            }
        }
    }

    /// Ask the provider for `capability`'s model, on device only when
    /// `local_only`.  `kind` names the lookup in the log.  `None` when there
    /// is no provider, nothing resolves, or the load fails.
    ///
    /// No local caching — see llm_for_tier: the provider owns the capability
    /// cache and may evict to fit, so pinning a reference here would prevent
    /// that VRAM from being reclaimed for the next step.
    fn resolve(&self, capability: &str, local_only: bool, kind: &str) -> Option<Arc<dyn BaseLLM>> {
        let provider = self.capability_llm_provider.as_ref()?;
        let resolved = {
            let _scope = local_only_scope(local_only);
            provider(capability)
        };
        match resolved {
            Ok(Some(llm)) => {
                tracing::info!(
                    target: LOGGER,
                    "[models] {} capability '{}' → {}",
                    kind,
                    capability,
                    Self::model_label(llm.as_ref())
                );
                Some(llm)
            }
            Ok(None) => None,
            Err(e) => {
                tracing::warn!(target: LOGGER, "[models] {} capability load failed ({}): {}", kind, capability, e);
                None
            }
        }
    }

    /// Report `llm` as the model picked for `capability`, unless it is the
    /// model this request already runs on.  Announcing that anyway puts a
    /// second "Selected …" beside the first, naming one model twice under
    /// two labels — which is what a single-model workspace saw for every MCP
    /// step, since the model that reasons is also the one that answers.
    ///
    /// The router model is not announced either: it is what every step runs
    /// on when nothing better is chosen, so naming it is not news — and on
    /// the answer tier it read as the router writing the reply.
    fn announce(&self, capability: &str, llm: &Arc<dyn BaseLLM>, purpose: Option<&str>) {
        if !same_llm(llm, &self.agent_llm) && !self.is_router_model(llm) {
            self.emit_model_selected(Some(capability), llm.as_ref(), purpose);
        }
    }

    /// Resolve a capability model for a low-value *decision* node (skill
    /// matching, multi-step planning, MCP tool selection).
    ///
    /// `purpose` is what the user is told the model is for, in place of the
    /// capability tag — see [`Self::llm_for_tier`] on why the two differ.
    ///
    /// Unlike [`Self::llm_for_tier`], this never forces a generic large-model
    /// load: it uses the capability model only when it is already resident or
    /// comfortably fits (the provider's own headroom guard decides), and
    /// otherwise falls straight back to the small orchestrator model. This
    /// keeps the better intent judgement "when available" without adding VRAM
    /// pressure — and risking an OOM — on memory-tight devices.  Decisions
    /// stay on device: a remote model is never one.
    pub fn decision_llm(
        &self,
        capability: Option<&str>,
        purpose: Option<&str>,
    ) -> Arc<dyn BaseLLM> {
        if self.pinned {
            return self.agent_llm.clone();
        }
        if let Some(cap) = capability.filter(|c| !c.is_empty()) {
            if let Some(llm) = self.resolve(cap, true, "decision") {
                self.announce(cap, &llm, purpose);
                return llm;
            }
        }
        self.agent_llm.clone()
    }

    /// Resolve a model for a light task that still benefits from a good
    /// model — reranking web results, say.
    ///
    /// Local first, then remote: any selected local model that fits does the
    /// task, even a small one, since the task is light.  Only when nothing
    /// local resolves — or the lookup lands back on the router model — does
    /// a remote model take it, ahead of the router model.  Unlike
    /// [`Self::decision_llm`] a remote model is allowed; unlike
    /// [`Self::llm_for_tier`] a miss never forces the generic large-model
    /// load — it lands on `agent_llm`.
    pub fn light_llm(&self, capability: Option<&str>, purpose: Option<&str>) -> Arc<dyn BaseLLM> {
        if self.pinned {
            return self.agent_llm.clone();
        }
        let cap = capability.filter(|c| !c.is_empty()).unwrap_or(crate::types::ANSWER_TIER);
        let picked = self
            .resolve(cap, true, "light")
            .filter(|llm| !self.is_router_model(llm))
            .or_else(|| self.resolve(cap, false, "light"));
        if let Some(llm) = picked {
            self.announce(cap, &llm, purpose);
            return llm;
        }
        self.agent_llm.clone()
    }

    /// True when `llm` is the router (bootstrap) model.  A handle to a model
    /// in another process is a fresh object each time, so it is also matched
    /// by the name it reports — only an explicit one, since the fallback
    /// label (a class name) is shared by unrelated in-process models.
    fn is_router_model(&self, llm: &Arc<dyn BaseLLM>) -> bool {
        if same_llm(llm, &self.orchestrator_llm) {
            return true;
        }
        let label = llm.model_label();
        !label.is_empty() && label == self.orchestrator_llm.model_label()
    }

    /// Name this model the way the user's model picker names it.
    ///
    /// The GGUF's `general.name` is the upstream author's label, which for our
    /// own model is `Slp Orchestra Mini` — a name that appears nowhere in the
    /// product.  This label goes out on the progress SSE, so it is mapped back
    /// to the short name (`sl-mini`) before it is shown.
    pub fn model_label(llm: &dyn BaseLLM) -> String {
        // A handle to a model owned by another process reports its own label —
        // there is no local GGUF here to read metadata from, and falling
        // through would put "RemoteLLM" on the user's progress event.
        let explicit = llm.model_label();
        if !explicit.is_empty() {
            return explicit;
        }
        if let Some(model) = llm.llama_model() {
            let meta = model.metadata();
            let name =
                display_model_label(meta.get("general.name").map(String::as_str).unwrap_or(""));
            let size = meta.get("general.size_label").map(String::as_str).unwrap_or("");
            if !name.is_empty() {
                return if size.is_empty() { name } else { format!("{name} ({size})") };
            }
        }
        llm.class_name().to_string()
    }

    /// Report an orchestration step to the optional progress callback.
    ///
    /// The callback is supplied by the streaming API layer to surface live
    /// activity as SSE `chat.status` events.  It is invoked from DAG worker
    /// threads, so it must be thread-safe; any failure here must never break
    /// orchestration.
    pub fn emit(&self, step: &str, status: &str, message: &str) {
        let Some(cb) = self.progress_callback.lock().clone() else {
            return;
        };
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cb(step, status, message)));
        if outcome.is_err() {
            tracing::debug!(target: LOGGER, "[progress] callback failed");
        }
    }

    /// Human-readable label for a capability tag.
    fn capability_label(capability: &str) -> String {
        let lower = capability.to_lowercase();
        CAPABILITY_LABELS
            .iter()
            .find(|(cap, _)| *cap == lower)
            .map(|(_, label)| (*label).to_string())
            .unwrap_or_else(|| capability.to_string())
    }

    /// True if this model SSE already fired this request (and record it).
    ///
    /// Collapses the per-node emits of a fan-out (one model resolution per
    /// document) into a single client message.
    fn model_event_seen(&self, key: &str) -> bool {
        let mut seen = self.emitted_model_events.lock();
        if seen.contains(key) {
            return true;
        }
        seen.insert(key.to_string());
        false
    }

    /// True if this model SSE already fired — without recording it.
    fn model_event_fired(&self, key: &str) -> bool {
        self.emitted_model_events.lock().contains(key)
    }

    /// `(label, key)` for a model event — what to say, and what to dedupe on.
    /// `purpose` wins over the capability tag on both counts, so two different
    /// jobs routed through the same capability each get their line.
    fn event_label(capability: Option<&str>, purpose: Option<&str>) -> (String, String) {
        if let Some(purpose) = purpose.filter(|p| !p.is_empty()) {
            return (purpose.to_string(), purpose.to_lowercase());
        }
        if let Some(capability) = capability.filter(|c| !c.is_empty()) {
            return (Self::capability_label(capability), capability.to_lowercase());
        }
        (String::new(), String::new())
    }

    /// Surface an *immediate* "loading" SSE before the (potentially slow)
    /// capability model load, so the user gets feedback right away instead of
    /// only after the model is resident.  Emitted once per capability/request.
    pub fn emit_model_loading(&self, capability: Option<&str>, purpose: Option<&str>) {
        let (label, key) = Self::event_label(capability, purpose);
        if key.is_empty() || self.model_event_seen(&format!("loading:{key}")) {
            return;
        }
        self.emit("model", "processing", &format!("Loading model for {label}…"));
    }

    /// Surface the selected capability model as an SSE.
    ///
    /// Emitted once per capability/request — a fan-out resolves the model once
    /// per document — and, where nothing announced a load, once per model.
    ///
    /// A "Selected …" line answers the "Loading model for …" line before it,
    /// so a step that promised a load always reports what it got, even when
    /// that is a model the user has already seen.  A step that promised
    /// nothing has only the switch to report, and a model already named this
    /// turn is not a switch: resolving a tool call after synthesis, on a
    /// workspace where one model serves both, read as the model being picked
    /// twice — where the second line told the user nothing the first had not.
    /// Work moving to a *different* model is still announced, every time.
    pub fn emit_model_selected(
        &self,
        capability: Option<&str>,
        llm: &dyn BaseLLM,
        purpose: Option<&str>,
    ) {
        let (label, key) = Self::event_label(capability, purpose);
        if key.is_empty() || self.model_event_seen(&format!("selected:{key}")) {
            return;
        }
        let name = Self::model_label(llm);
        // Recorded whichever way this goes: a line that answered a load has
        // named the model, so a later step that promised nothing has nothing
        // left to add about it.
        let first_mention = !self.model_event_seen(&format!("selected-model:{name}"));
        if !first_mention && !self.model_event_fired(&format!("loading:{key}")) {
            return;
        }
        self.emit("model", "processing", &format!("Selected {name} for {label}"));
    }

    /// Trailing clause explaining why no capability model was used, or `""`.
    fn skip_explanation(&self) -> String {
        let Some(reporter) = self.capability_skip_reason.as_ref() else {
            return String::new();
        };
        let skip = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| reporter())) {
            Ok(Ok(skip)) => skip,
            Ok(Err(e)) => {
                tracing::debug!(target: LOGGER, "[models] skip-reason lookup failed: {}", e);
                return String::new();
            }
            Err(_) => {
                tracing::debug!(target: LOGGER, "[models] skip-reason lookup failed");
                return String::new();
            }
        };
        let Some(skip) = skip else {
            return String::new();
        };
        // `detail` is optional and carries the figures behind the cause —
        // "needs ~4.9GB, 4.4GB free" — which is the part the user can act on.
        let Some(template) =
            SKIP_REASONS.iter().find(|(cause, _)| *cause == skip.cause).map(|(_, t)| *t)
        else {
            return String::new();
        };
        if skip.model_name.is_empty() {
            return String::new();
        }
        let explanation = template.replace("{model}", &display_model_name(&skip.model_name));
        match skip.detail.as_deref().filter(|d| !d.is_empty()) {
            Some(detail) => format!("{explanation} ({detail})"),
            None => explanation,
        }
    }

    /// Name the model that will run the step when the capability could not be
    /// resolved, and say why when the reason is known.
    ///
    /// Deliberately does not name the capability: no capability model was
    /// found, so "Selected sl-mini for reasoning" reads as though one had been
    /// chosen for that work.  Shares the selection dedupe key, so a later real
    /// resolution for the same capability does not report twice.
    pub fn emit_model_fallback(
        &self,
        capability: Option<&str>,
        llm: &dyn BaseLLM,
        purpose: Option<&str>,
    ) {
        let (_label, key) = Self::event_label(capability, purpose);
        if key.is_empty() || self.model_event_seen(&format!("selected:{key}")) {
            return;
        }
        let explanation = self.skip_explanation();
        let mut message = format!("Selected {}", Self::model_label(llm));
        if !explanation.is_empty() {
            message = format!("{message} — {explanation}");
        }
        self.emit("model", "processing", &message);
    }
}

/// A scripted LLM double shared by the orchestration-step unit tests.
#[cfg(test)]
pub(crate) mod test_llm {
    use std::any::Any;
    use std::sync::Arc;

    use parking_lot::Mutex;

    use crate::llms::base_llm::{CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream};
    use crate::llms::BaseLLM;

    type Reply = Arc<dyn Fn(&str) -> String + Send + Sync>;

    /// Replies through a scripted closure and records every prompt.
    pub(crate) struct ScriptedLlm {
        pub reply: Mutex<Reply>,
        pub prompts: Mutex<Vec<String>>,
        pub max_tokens: usize,
        pub num_tokens: usize,
    }

    impl ScriptedLlm {
        pub(crate) fn new(reply: impl Fn(&str) -> String + Send + Sync + 'static) -> Arc<Self> {
            Arc::new(Self {
                reply: Mutex::new(Arc::new(reply)),
                prompts: Mutex::new(Vec::new()),
                max_tokens: 8192,
                num_tokens: 100,
            })
        }

        pub(crate) fn calls(&self) -> usize {
            self.prompts.lock().len()
        }
    }

    impl BaseLLM for ScriptedLlm {
        fn project_id(&self) -> &str {
            ""
        }
        fn get_num_tokens(&self, _text: &str) -> usize {
            self.num_tokens
        }
        fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
            messages
        }
        fn get_max_tokens(&self) -> usize {
            self.max_tokens
        }
        fn call(&self, prompt: &str, _stop: Option<&[String]>, _o: CallOptions) -> anyhow::Result<String> {
            Ok(self.invoke(prompt, None, None))
        }
        fn stream<'a>(&'a self, prompt: &str, _o: CallOptions) -> GenerationStream<'a> {
            let text = self.invoke(prompt, None, None);
            Box::new(std::iter::once(GenerationChunk { text }))
        }
        fn invoke(&self, prompt: &str, _max_tokens: Option<usize>, _thinking: Option<bool>) -> String {
            self.prompts.lock().push(prompt.to_string());
            let reply = self.reply.lock().clone();
            reply(prompt)
        }
        fn cancelled(&self) -> Option<CancelEvent> {
            None
        }
        fn set_cancelled(&self, _event: Option<CancelEvent>) {}
        fn llm_type(&self) -> &str {
            "scripted"
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_llm::ScriptedLlm;
    use super::*;

    fn recorder() -> (ProgressCallback, Arc<Mutex<Vec<String>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        (Arc::new(move |_s: &str, _st: &str, m: &str| sink.lock().push(m.to_string())), seen)
    }

    #[test]
    fn capability_model_is_announced_once_per_request() {
        let agent = ScriptedLlm::new(|_| "x".into());
        let cap: Arc<dyn BaseLLM> = ScriptedLlm::new(|_| "y".into());
        let cap_for_provider = cap.clone();
        let router = ModelRouter::new(
            agent.clone(),
            agent,
            ModelRouterOptions {
                capability_llm_provider: Some(Arc::new(move |_c: &str| Ok(Some(cap_for_provider.clone())))),
                ..Default::default()
            },
        );
        let (cb, seen) = recorder();
        router.begin_request(Some(cb));
        for _ in 0..3 {
            let got = router.llm_for_tier(ModelTier::Large, Some("vision"), None);
            assert!(same_llm(&got, &cap));
        }
        assert_eq!(
            *seen.lock(),
            vec![
                "Loading model for vision…".to_string(),
                "Selected ScriptedLlm for vision".to_string(),
            ]
        );
    }

    #[test]
    fn decision_llm_on_the_agent_model_is_not_announced() {
        let agent: Arc<dyn BaseLLM> = ScriptedLlm::new(|_| "x".into());
        let same = agent.clone();
        let router = ModelRouter::new(
            agent.clone(),
            agent.clone(),
            ModelRouterOptions {
                capability_llm_provider: Some(Arc::new(move |_c: &str| Ok(Some(same.clone())))),
                ..Default::default()
            },
        );
        let (cb, seen) = recorder();
        router.begin_request(Some(cb));
        let got = router.llm_for_tier(ModelTier::Small, Some("reasoning"), None);
        assert!(same_llm(&got, &agent));
        assert!(seen.lock().is_empty());
    }

    #[test]
    fn large_provider_is_used_for_unresolved_capability() {
        let agent: Arc<dyn BaseLLM> = ScriptedLlm::new(|_| "x".into());
        let large: Arc<dyn BaseLLM> = ScriptedLlm::new(|_| "L".into());
        let large_for_provider = large.clone();
        let router = ModelRouter::new(
            agent.clone(),
            agent,
            ModelRouterOptions {
                large_llm_provider: Some(Arc::new(move || Ok(Some(large_for_provider.clone())))),
                ..Default::default()
            },
        );
        assert!(router.has_large_provider());
        let got = router.llm_for_tier(ModelTier::Large, None, None);
        assert!(same_llm(&got, &large));
        assert!(router.large_llm.lock().is_some());
    }

    #[test]
    fn only_decision_nodes_resolve_local_only() {
        use crate::llms::cloud_llm::local_only;

        let agent: Arc<dyn BaseLLM> = ScriptedLlm::new(|_| "x".into());
        let cap: Arc<dyn BaseLLM> = ScriptedLlm::new(|_| "y".into());
        let scopes = Arc::new(Mutex::new(Vec::new()));
        let (seen, cap_for_provider) = (scopes.clone(), cap.clone());
        let router = ModelRouter::new(
            agent.clone(),
            agent,
            ModelRouterOptions {
                capability_llm_provider: Some(Arc::new(move |_c: &str| {
                    seen.lock().push(local_only());
                    Ok(Some(cap_for_provider.clone()))
                })),
                ..Default::default()
            },
        );
        router.llm_for_tier(ModelTier::Large, Some(crate::types::ANSWER_TIER), None);
        router.llm_for_tier(ModelTier::Small, Some(crate::types::ANSWER_TIER), None);
        router.decision_llm(Some("reasoning"), None);
        assert_eq!(*scopes.lock(), vec![false, true, true]);
        assert!(!local_only());
    }

    #[test]
    fn a_light_task_may_resolve_remote_and_defaults_to_the_answer_tier() {
        use crate::llms::cloud_llm::local_only;

        let agent: Arc<dyn BaseLLM> = ScriptedLlm::new(|_| "x".into());
        let cap: Arc<dyn BaseLLM> = ScriptedLlm::new(|_| "y".into());
        let calls = Arc::new(Mutex::new(Vec::new()));
        let (seen, cap_for_provider) = (calls.clone(), cap.clone());
        let router = ModelRouter::new(
            agent.clone(),
            agent,
            ModelRouterOptions {
                capability_llm_provider: Some(Arc::new(move |c: &str| {
                    seen.lock().push((c.to_string(), local_only()));
                    Ok(Some(cap_for_provider.clone()))
                })),
                ..Default::default()
            },
        );
        let got = router.light_llm(None, Some(WEB_SEARCH_PURPOSE));
        assert!(same_llm(&got, &cap));
        // A selected local model answered on device, so the cloud was not asked.
        assert_eq!(*calls.lock(), vec![(crate::types::ANSWER_TIER.to_string(), true)]);
    }

    #[test]
    fn a_light_task_landing_on_the_router_model_asks_the_cloud() {
        use crate::llms::cloud_llm::local_only;

        let router_model: Arc<dyn BaseLLM> = ScriptedLlm::new(|_| "x".into());
        let cloud: Arc<dyn BaseLLM> = ScriptedLlm::new(|_| "y".into());
        let calls = Arc::new(Mutex::new(Vec::new()));
        let (seen, on_device, remote) = (calls.clone(), router_model.clone(), cloud.clone());
        let router = ModelRouter::new(
            router_model.clone(),
            router_model,
            ModelRouterOptions {
                capability_llm_provider: Some(Arc::new(move |_c: &str| {
                    let local = local_only();
                    seen.lock().push(local);
                    Ok(Some(if local { on_device.clone() } else { remote.clone() }))
                })),
                ..Default::default()
            },
        );
        assert!(same_llm(&router.light_llm(None, None), &cloud));
        assert_eq!(*calls.lock(), vec![true, false]);
    }

    #[test]
    fn a_light_task_with_no_capable_model_stays_on_the_agent_model() {
        let agent: Arc<dyn BaseLLM> = ScriptedLlm::new(|_| "x".into());
        let large_loads = Arc::new(Mutex::new(0));
        let loads = large_loads.clone();
        let router = ModelRouter::new(
            agent.clone(),
            agent.clone(),
            ModelRouterOptions {
                capability_llm_provider: Some(Arc::new(|_c: &str| Ok(None))),
                large_llm_provider: Some(Arc::new(move || {
                    *loads.lock() += 1;
                    Ok(None)
                })),
                ..Default::default()
            },
        );
        let got = router.light_llm(Some("coding"), None);
        assert!(same_llm(&got, &agent));
        assert_eq!(*large_loads.lock(), 0);
    }

    /// A pinned request runs every text node on its one model — routing,
    /// decisions, tool resolution, light tasks and the answer tier — and
    /// asks the provider only for vision, a modality the model may lack.
    #[test]
    fn a_pinned_request_stays_on_one_model_except_for_vision() {
        let router_model: Arc<dyn BaseLLM> = ScriptedLlm::new(|_| "r".into());
        let pinned: Arc<dyn BaseLLM> = ScriptedLlm::new(|_| "p".into());
        let vision: Arc<dyn BaseLLM> = ScriptedLlm::new(|_| "v".into());
        let asked = Arc::new(Mutex::new(Vec::new()));
        let (seen, vision_for_provider) = (asked.clone(), vision.clone());
        let router = ModelRouter::new(
            router_model,
            pinned.clone(),
            ModelRouterOptions {
                capability_llm_provider: Some(Arc::new(move |c: &str| {
                    seen.lock().push(c.to_string());
                    Ok(Some(vision_for_provider.clone()))
                })),
                capability_available: Some(Arc::new(|_c: &str| Ok(true))),
                pinned: true,
                ..Default::default()
            },
        );
        assert!(router.is_pinned());
        let answer = crate::types::ANSWER_TIER;
        assert!(same_llm(&router.route_llm(pinned.clone()), &pinned));
        assert!(same_llm(&router.decision_llm(Some("reasoning"), None), &pinned));
        assert!(same_llm(&router.light_llm(None, Some(WEB_SEARCH_PURPOSE)), &pinned));
        assert!(same_llm(&router.reasoning_llm(pinned.clone()), &pinned));
        assert!(same_llm(&router.llm_for_tier(ModelTier::Large, Some(answer), None), &pinned));
        assert!(same_llm(&router.llm_for_tier(ModelTier::Small, Some("reasoning"), None), &pinned));
        assert!(asked.lock().is_empty(), "no text node consults another model");

        assert!(same_llm(&router.llm_for_tier(ModelTier::Large, Some("vision"), None), &vision));
        assert_eq!(*asked.lock(), vec!["vision".to_string()]);
    }

    #[test]
    fn a_panicking_progress_callback_is_swallowed() {
        let agent = ScriptedLlm::new(|_| "x".into());
        let router = ModelRouter::new(agent.clone(), agent, ModelRouterOptions::default());
        router.begin_request(Some(Arc::new(|_: &str, _: &str, _: &str| panic!("boom"))));
        router.emit("step", "processing", "message");
    }
}
