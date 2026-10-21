//! Per-model metadata as data, hydrated from the cloud `/v1/models` catalog.
//!
//! A [`ModelSpec`] holds every fact that differs between models — repo id,
//! size, version, capabilities, thinking default, vision handler, and for a
//! remote model its endpoint and context window.  All runtime behaviour lives
//! in [`GenericModel`](crate::model::generic::GenericModel), so a model on an
//! architecture this build already handles ships as a catalog row alone.
//!
//! [`MODEL_SPECS`] starts from [`seed_specs`] (what this build knows offline)
//! and is overlaid with catalog rows: once from the workspace's cached
//! catalog on first use, then on every catalog fetch.  A row only overrides
//! the seed fields it declares, so an older catalog leaves the seed values
//! intact, and each fetch replaces what the previous one said.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde::Deserialize;
use serde_json::Value;

use crate::constants::{canonical_model_name, BOOTSTRAP_MODEL};
use crate::model::slp_model::DEFAULT_CAPABILITIES;
use crate::platform::client::JsonObject;
use crate::types::{Attention, ModelCapability, ModelKind, ModelType};

/// What this build knows about one model.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    /// Canonical short name, e.g. `"gemma4-e2b"`.
    pub name: String,
    /// HuggingFace-style repo id, also the on-disk model directory; empty for
    /// a remote model.
    pub model_id: String,
    pub kind: ModelKind,
    /// Model family version (higher wins between same-size models).
    pub model_version: i64,
    /// Parameter count in billions; 0 when unknown.
    pub param_size_b: f64,
    /// Whether `param_size_b` is an estimate (a closed remote model's,
    /// say).  It ranks the same; only display differs.
    pub param_size_estimated: bool,
    /// Parameters active per token for a mixture-of-experts model; `None`
    /// for a dense one, where it equals `param_size_b`.
    pub active_param_size_b: Option<f64>,
    pub attention: Option<Attention>,
    /// Capabilities the model declares — never empty.
    pub capabilities: Vec<ModelCapability>,
    /// Whether reasoning starts ON for a model that supports it.
    pub thinking_default: bool,
    /// llama.cpp chat-handler name for image input, `None` for text-only.
    pub vision_chat_handler: Option<String>,
    /// Base URL a remote model's chat requests go to.
    pub url: Option<String>,
    pub context_length: Option<u32>,
}

impl ModelSpec {
    /// A local model with the base defaults: version 0, unknown size,
    /// `[General]`, thinking off, text-only.
    pub fn local(name: &str, model_id: &str) -> Self {
        Self {
            name: canonical_model_name(name),
            model_id: model_id.to_string(),
            kind: ModelKind::Local,
            model_version: 0,
            param_size_b: 0.0,
            param_size_estimated: false,
            active_param_size_b: None,
            attention: None,
            capabilities: DEFAULT_CAPABILITIES.to_vec(),
            thinking_default: false,
            vision_chat_handler: None,
            url: None,
            context_length: None,
        }
    }

    pub fn is_remote(&self) -> bool {
        self.kind == ModelKind::Remote
    }

    /// The size models are compared on: active parameters for a
    /// mixture-of-experts model (what sets its speed and per-token ability),
    /// otherwise the parameter count.
    pub fn ranking_size_b(&self) -> f64 {
        self.active_param_size_b.unwrap_or(self.param_size_b)
    }

    /// Whether the model declares `capability`.
    pub fn supports(&self, capability: ModelCapability) -> bool {
        self.capabilities.contains(&capability)
    }

    /// The input modality: multimodal exactly when an image handler is
    /// declared.
    pub fn model_type(&self) -> ModelType {
        if self.vision_chat_handler.is_some() {
            ModelType::MultiModal
        } else {
            ModelType::Text
        }
    }

    /// Capability wire values, e.g. `["general", "dp"]`.
    pub fn capability_values(&self) -> Vec<String> {
        self.capabilities.iter().map(|c| c.value().to_string()).collect()
    }

    /// `base` (or the defaults for a model this build has never seen) with
    /// every field `row` declares laid over it.
    pub fn merged(base: Option<&ModelSpec>, row: CatalogModel) -> ModelSpec {
        let name = canonical_model_name(&row.name);
        let mut spec = match base {
            Some(base) => base.clone(),
            None => ModelSpec::local(&name, ""),
        };
        spec.name = name;
        if let Some(model_id) = row.model_id {
            spec.model_id = model_id;
        }
        if let Some(kind) = row.kind {
            spec.kind = kind;
        }
        if let Some(version) = row.model_version {
            spec.model_version = version;
        }
        if let Some(size) = row.param_size_b {
            spec.param_size_b = size;
        }
        if let Some(estimated) = row.param_size_estimated {
            spec.param_size_estimated = estimated;
        }
        if let Some(active) = row.active_param_size_b {
            spec.active_param_size_b = Some(active);
        }
        if let Some(attention) = row.attention.filter(|a| !a.is_empty()) {
            match Attention::from_value(&attention) {
                Some(attention) => spec.attention = Some(attention),
                None => tracing::warn!("[models] {} declares unknown attention '{attention}' — ignoring it", spec.name),
            }
        }
        let capabilities = row.capabilities.unwrap_or_default();
        let known: Vec<ModelCapability> = capabilities
            .iter()
            .filter_map(|c| {
                let cap = ModelCapability::from_value(c);
                if cap.is_none() {
                    tracing::warn!("[models] {} declares unknown capability '{c}' — ignoring it", spec.name);
                }
                cap
            })
            .collect();
        if !known.is_empty() {
            spec.capabilities = known;
        }
        if let Some(thinking) = row.thinking_default {
            spec.thinking_default = thinking;
        }
        if let Some(handler) = row.vision_chat_handler.filter(|h| !h.is_empty()) {
            spec.vision_chat_handler = Some(handler);
        }
        if let Some(url) = row.url.filter(|u| !u.is_empty()) {
            spec.url = Some(url);
        }
        if let Some(context_length) = row.context_length {
            spec.context_length = Some(context_length);
        }
        spec
    }
}

/// One `/v1/models` row as far as the spec goes.  Every field but `name` is
/// optional: an absent or null field is "not declared" and keeps the value
/// already known, as does an empty `capabilities` list.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CatalogModel {
    pub name: String,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(rename = "type", default)]
    pub kind: Option<ModelKind>,
    #[serde(default)]
    pub model_version: Option<i64>,
    #[serde(default)]
    pub param_size_b: Option<f64>,
    #[serde(default)]
    pub param_size_estimated: Option<bool>,
    #[serde(default)]
    pub active_param_size_b: Option<f64>,
    /// Kept as text so an attention type this build doesn't know is
    /// ignored rather than failing the row.
    #[serde(default)]
    pub attention: Option<String>,
    #[serde(default)]
    pub capabilities: Option<Vec<String>>,
    #[serde(default)]
    pub thinking_default: Option<bool>,
    #[serde(default)]
    pub vision_chat_handler: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub context_length: Option<u32>,
}

impl CatalogModel {
    /// Parse a catalog row; `None` (with a warning) when it doesn't parse.
    pub fn from_row(row: &JsonObject) -> Option<Self> {
        match serde_json::from_value::<CatalogModel>(Value::Object(row.clone())) {
            Ok(model) if !model.name.is_empty() => Some(model),
            Ok(_) => None,
            Err(e) => {
                tracing::warn!("[models] skipping unparseable catalog row: {e}");
                None
            }
        }
    }
}

/// The models this build knows before any catalog is read.  They match what
/// the catalog publishes for them, so a first run or an offline machine
/// behaves exactly as an online one.
pub fn seed_specs() -> Vec<ModelSpec> {
    use ModelCapability::{Dp, General, Reasoning, Thinking, Vision};

    let spec = |name: &str,
                model_id: &str,
                model_version: i64,
                param_size_b: f64,
                capabilities: Vec<ModelCapability>,
                vision_chat_handler: Option<&str>| ModelSpec {
        model_version,
        param_size_b,
        capabilities,
        vision_chat_handler: vision_chat_handler.map(str::to_string),
        ..ModelSpec::local(name, model_id)
    };

    vec![
        // The 1B text-only orchestrator model.
        spec(BOOTSTRAP_MODEL, "smartloop-ai/slp-orchestra-mini", 3, 1.0, vec![General], None),
        // llama.cpp has no Gemma3-specific handler; Gemma 3's mmproj loads
        // through the generic mtmd path (projector auto-detected from the GGUF).
        spec("gemma3-4b", "google/gemma-3-4b-it", 3, 4.0, vec![General, Dp], Some("MTMDChatHandler")),
        // Gemma 4's chat template reads enable_thinking (defaulting it to false)
        // and emits reasoning on a <|channel>thought…<channel|> channel; THINKING
        // is declared so the picker offers the toggle, left off by default.
        spec(
            "gemma4-e2b",
            "google/gemma-4-e2b-it",
            4,
            2.0,
            vec![General, Dp, Vision, Thinking],
            Some("Gemma4ChatHandler"),
        ),
        spec(
            "gemma4-e4b",
            "google/gemma-4-e4b-it",
            4,
            4.0,
            vec![General, Dp, Vision, Thinking],
            Some("Gemma4ChatHandler"),
        ),
        // Smaller than Qwen2.5-VL-3B so it fits on GPU alongside the
        // orchestrator and on-demand agent models on memory-constrained
        // devices.  Runs via `Qwen25VLChatHandler`, which handles the
        // `qwen2vl` architecture.
        spec("qwen2-vl-2b", "Qwen/Qwen2-VL-2B-Instruct", 2, 2.0, vec![Vision], Some("Qwen25VLChatHandler")),
        // No Pixtral/mistral3-specific handler: vision (arch 'mistral3') decodes
        // through llama.cpp's generic mtmd path, so declare MTMDChatHandler.
        spec(
            "ministral3-3b",
            "mistralai/Ministral-3-3B-Instruct-2512",
            3,
            3.0,
            vec![General, Dp, Vision],
            Some("MTMDChatHandler"),
        ),
        // A hybrid-reasoning model that can think in a `<think>` block. Its
        // GGUF architecture is `qwen35`, which no arch-sniffed handler matches,
        // so the declared handler is what makes its vision work at all.  The
        // generic MTMDChatHandler renders the model's own embedded chat
        // template (keeping <think> control); Qwen25VLChatHandler would impose
        // a hardcoded ChatML prompt that predates Qwen3.5.
        spec(
            "qwen3.5-2b",
            "Qwen/Qwen3.5-2B",
            3,
            2.0,
            vec![General, Reasoning, Dp, Vision, Thinking],
            Some("MTMDChatHandler"),
        ),
    ]
}

/// Specs by canonical name, in registration order (seeds first).
#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    specs: Vec<Arc<ModelSpec>>,
    /// What catalog rows are laid over.
    seeds: Vec<Arc<ModelSpec>>,
}

impl ModelRegistry {
    /// A registry seeded with `seeds`.
    pub fn new(seeds: Vec<ModelSpec>) -> Self {
        let mut registry = Self::default();
        for spec in seeds {
            registry.insert(spec);
        }
        registry.seeds = registry.specs.clone();
        registry
    }

    /// A registry holding [`seed_specs`].
    pub fn seeded() -> Self {
        Self::new(seed_specs())
    }

    /// The spec for `name` (canonicalized), if known.
    pub fn get(&self, name: &str) -> Option<Arc<ModelSpec>> {
        let name = canonical_model_name(name);
        self.specs.iter().find(|s| s.name == name).cloned()
    }

    /// The local model whose weights live under `model_id`.
    pub fn get_by_id(&self, model_id: &str) -> Option<Arc<ModelSpec>> {
        if model_id.is_empty() {
            return None;
        }
        self.specs.iter().find(|s| s.model_id == model_id).cloned()
    }

    /// Every spec, in registration order.
    pub fn specs(&self) -> Vec<Arc<ModelSpec>> {
        self.specs.clone()
    }

    pub fn len(&self) -> usize {
        self.specs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    /// Add `spec`, replacing any spec of the same name in place.
    pub fn insert(&mut self, spec: ModelSpec) {
        let spec = Arc::new(spec);
        match self.specs.iter().position(|s| s.name == spec.name) {
            Some(i) => self.specs[i] = spec,
            None => self.specs.push(spec),
        }
    }

    /// Lay each catalog row over its model's seed, replacing whatever an
    /// earlier row said.  Models the catalog omits are kept: an anonymous
    /// catalog hides some, and the seeds must survive it.  Returns how many
    /// models were new.
    pub fn hydrate(&mut self, rows: impl IntoIterator<Item = CatalogModel>) -> usize {
        let mut added = 0;
        for row in rows {
            let name = canonical_model_name(&row.name);
            if self.get(&name).is_none() {
                added += 1;
            }
            let seed = self.seeds.iter().find(|s| s.name == name).cloned();
            self.insert(ModelSpec::merged(seed.as_deref(), row));
        }
        added
    }
}

/// The process-wide registry: seeds, then the cached catalog (read once, on
/// first use), then every fetched catalog.
pub struct ModelSpecs {
    registry: Lazy<RwLock<ModelRegistry>>,
    cache_read: AtomicBool,
}

impl ModelSpecs {
    const fn new() -> Self {
        Self { registry: Lazy::new(|| RwLock::new(ModelRegistry::seeded())), cache_read: AtomicBool::new(false) }
    }

    /// Overlay the workspace's cached catalog, once per process.  The flag
    /// is set before reading, so a lookup made while the cache loads (the
    /// workspace load resolves model names) sees the seeds instead of
    /// recursing.
    fn read_cache_once(&self) {
        if self.cache_read.swap(true, Ordering::SeqCst) {
            return;
        }
        let rows = crate::platform::cached_model_catalog();
        if !rows.is_empty() {
            self.hydrate(&rows);
        }
    }

    fn registry(&self) -> parking_lot::RwLockReadGuard<'_, ModelRegistry> {
        self.read_cache_once();
        self.registry.read()
    }

    /// The spec for `name` (canonicalized), if known.
    pub fn get(&self, name: &str) -> Option<Arc<ModelSpec>> {
        self.registry().get(name)
    }

    /// Whether `name` (canonicalized) names a known model.
    pub fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// The local model whose weights live under `model_id`.
    pub fn get_by_id(&self, model_id: &str) -> Option<Arc<ModelSpec>> {
        self.registry().get_by_id(model_id)
    }

    /// Every known spec, seeds first.
    pub fn specs(&self) -> Vec<Arc<ModelSpec>> {
        self.registry().specs()
    }

    /// Canonical names of every known model, seeds first.
    pub fn keys(&self) -> Vec<String> {
        self.specs().iter().map(|s| s.name.clone()).collect()
    }

    /// Lay catalog rows (as `/v1/models` returns them) over the known specs;
    /// returns how many models were new.
    pub fn hydrate(&self, rows: &[JsonObject]) -> usize {
        let rows: Vec<CatalogModel> = rows.iter().filter_map(CatalogModel::from_row).collect();
        self.registry.write().hydrate(rows)
    }
}

/// Every model this process knows about.
pub static MODEL_SPECS: ModelSpecs = ModelSpecs::new();
