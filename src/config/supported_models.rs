//! Registry of the built-in model classes, keyed by canonical short name.
//!
//! Port of `smartloop/config/supported_models.py`.  Python maps names to model
//! *classes* and calls their static methods; Rust maps names to a
//! [`ModelClass`] descriptor holding those static methods as fn pointers
//! (resolved from each model struct's associated functions of the same name).
//! Lookups canonicalize the key, so a legacy name resolves to the same class;
//! iteration yields only canonical names, in insertion order.

use once_cell::sync::Lazy;

#[allow(unused_imports)]
use crate::constants::{canonical_model_name, BOOTSTRAP_MODEL};
use crate::types::ModelCapability;

/// Static description of a built-in model class (the Python class object).
///
/// Two descriptors are "the same class" when their `class_name` matches
/// (Python's `is` comparison on classes).
#[derive(Debug, Clone, Copy)]
pub struct ModelClass {
    /// The Python/Rust struct name, e.g. `"GemmaItTextModel"`; used by
    /// `ModelManager` to construct the matching model.
    pub class_name: &'static str,
    /// HuggingFace-style repo id (also the on-disk model directory).
    pub model_id: fn() -> String,
    /// Model family version (higher wins in selection).
    pub model_version: fn() -> i64,
    /// Parameter count in billions; 0 when unknown.
    pub param_size_in_b: fn() -> f64,
    /// Capabilities this model declares — never empty.
    pub default_capabilities: fn() -> Vec<ModelCapability>,
    /// Whether reasoning starts ON for a model that supports it.
    pub thinking_default: fn() -> bool,
    /// llama.cpp chat-handler name for image input, `None` for text-only.
    pub vision_chat_handler: fn() -> Option<String>,
}

impl PartialEq for ModelClass {
    fn eq(&self, other: &Self) -> bool {
        self.class_name == other.class_name
    }
}

impl Eq for ModelClass {}

/// Lazily-built, canonicalizing name -> [`ModelClass`] map.
pub struct LazyModelDict {
    data: Lazy<Vec<(&'static str, ModelClass)>>,
}

impl LazyModelDict {
    const fn new() -> Self {
        LazyModelDict { data: Lazy::new(load) }
    }

    /// Look up `key` (canonicalized); `None` when unknown.
    pub fn get(&self, key: &str) -> Option<ModelClass> {
        let _ = key;
        todo!()
    }

    /// Whether `key` (canonicalized) names a built-in model.
    pub fn contains(&self, key: &str) -> bool {
        let _ = key;
        todo!()
    }

    /// Canonical names in insertion order.
    pub fn keys(&self) -> Vec<&'static str> {
        todo!()
    }

    /// The model classes in insertion order.
    pub fn values(&self) -> Vec<ModelClass> {
        todo!()
    }

    /// `(canonical name, class)` pairs in insertion order.
    pub fn items(&self) -> Vec<(&'static str, ModelClass)> {
        let _ = &self.data;
        todo!()
    }

    /// Iterate canonical names (Python `__iter__`).
    pub fn iter(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.data.iter().map(|(k, _)| *k)
    }
}

/// Build the table: sl-mini, gemma3-4b, gemma4-e2b, gemma4-e4b, qwen2-vl-2b,
/// ministral3-3b, qwen3.5-2b.
fn load() -> Vec<(&'static str, ModelClass)> {
    todo!()
}

/// The built-in model registry.
pub static SUPPORTED_MODELS: LazyModelDict = LazyModelDict::new();
