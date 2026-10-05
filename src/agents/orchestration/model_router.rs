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

#![allow(unused_imports, unused_variables, dead_code)]

use std::collections::HashSet;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::agents::planner::ModelTier;
use crate::constants::{display_model_label, display_model_name};
use crate::llms::BaseLLM;
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
}

/// Client-facing labels for capabilities.  The enum values (e.g. "dp") are
/// internal terms; map them to human-readable text before surfacing in SSEs.
pub const CAPABILITY_LABELS: &[(&str, &str)] = &[
    ("general", "general"),
    ("reasoning", "reasoning"),
    ("dp", "document processing"),
    ("vision", "vision"),
    ("coding", "coding"),
    ("thinking", "thinking"),
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
/// internal routing tag and names a property of the model rather than the work.
pub const RESOLUTION_PURPOSE: &str = "tool resolution";

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
    capability_available: Option<CapabilityAvailable>,
    capability_skip_reason: Option<CapabilitySkipReasonFn>,
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
        todo!()
    }

    /// True when a generic large model can still be loaded on demand.
    pub fn has_large_provider(&self) -> bool {
        todo!()
    }

    /// Reset per-request state and install this request's progress sink.
    ///
    /// The model-selection dedupe set is cleared per request: a capability's
    /// model is resolved once per DAG node, but the user only needs to be told
    /// which model was picked once.
    pub fn begin_request(&self, progress_callback: Option<ProgressCallback>) {
        todo!()
    }

    /// Resolve the model for a node, reporting the choice to the client.
    ///
    /// `capability` routes; `purpose` — when given — is what the user is told
    /// the model is for.  LARGE falls back to the generic large model, then the
    /// agent model; SMALL routes best-effort through [`Self::decision_llm`].
    pub fn llm_for_tier(
        &self,
        tier: ModelTier,
        capability: Option<&str>,
        purpose: Option<&str>,
    ) -> Arc<dyn BaseLLM> {
        todo!()
    }

    /// A model that can reason, for a call whose judgement warrants it (MCP
    /// tool resolution).  Best-effort; a `fallback` that already reasons is
    /// kept as-is.
    pub fn reasoning_llm(&self, fallback: Arc<dyn BaseLLM>) -> Arc<dyn BaseLLM> {
        todo!()
    }

    /// True when some selected model declares `capability`.  Unknown (no
    /// predicate wired in, or the predicate failed) counts as true.
    fn capability_model_exists(&self, capability: Option<&str>) -> bool {
        todo!()
    }

    /// Resolve a capability model for a low-value *decision* node.  Never
    /// forces a generic large-model load; falls straight back to `agent_llm`.
    pub fn decision_llm(
        &self,
        capability: Option<&str>,
        purpose: Option<&str>,
    ) -> Arc<dyn BaseLLM> {
        todo!()
    }

    /// Name this model the way the user's model picker names it (explicit
    /// label, else GGUF `general.name` mapped to the product name plus
    /// `general.size_label`, else the LLM's type name).
    pub fn model_label(llm: &dyn BaseLLM) -> String {
        todo!()
    }

    /// Report an orchestration step to the optional progress callback.  Any
    /// failure in the callback is swallowed.
    pub fn emit(&self, step: &str, status: &str, message: &str) {
        todo!()
    }

    /// Human-readable label for a capability tag.
    fn capability_label(capability: &str) -> String {
        todo!()
    }

    /// True if this model SSE already fired this request (and record it).
    fn model_event_seen(&self, key: &str) -> bool {
        todo!()
    }

    /// True if this model SSE already fired — without recording it.
    fn model_event_fired(&self, key: &str) -> bool {
        todo!()
    }

    /// `(label, key)` for a model event; `purpose` wins over the capability.
    fn event_label(capability: Option<&str>, purpose: Option<&str>) -> (String, String) {
        todo!()
    }

    /// Surface an immediate "Loading model for …" SSE before a capability
    /// model load.  Emitted once per capability/request.
    pub fn emit_model_loading(&self, capability: Option<&str>, purpose: Option<&str>) {
        todo!()
    }

    /// Surface the selected capability model as an SSE ("Selected X for Y"),
    /// deduped per capability/request and per model.
    pub fn emit_model_selected(
        &self,
        capability: Option<&str>,
        llm: &dyn BaseLLM,
        purpose: Option<&str>,
    ) {
        todo!()
    }

    /// Trailing clause explaining why no capability model was used, or `""`.
    fn skip_explanation(&self) -> String {
        todo!()
    }

    /// Name the model that will run the step when the capability could not be
    /// resolved ("Selected X — reason"), sharing the selection dedupe key.
    pub fn emit_model_fallback(
        &self,
        capability: Option<&str>,
        llm: &dyn BaseLLM,
        purpose: Option<&str>,
    ) {
        todo!()
    }
}
