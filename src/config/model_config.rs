//! Model configuration classes for the core framework.
//!
//! Port of `smartloop/config/model_config.py`.

use serde::{Deserialize, Serialize};

fn default_quantize() -> String {
    "q4_k_m".to_string()
}

/// Configuration for model initialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    /// model name
    pub model_name: String,
    /// Project id
    #[serde(default)]
    pub project_id: Option<String>,
    /// Quantization level
    #[serde(default = "default_quantize")]
    pub quantize: String,
    /// HuggingFace token
    #[serde(default)]
    pub hf_token: Option<String>,
    /// Developer token for authentication.  Never serialized to JSON; it
    /// persists in memory only (pydantic `exclude=True`).
    #[serde(default, skip_serializing)]
    pub developer_token: Option<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            model_name: String::new(),
            project_id: None,
            quantize: default_quantize(),
            hf_token: None,
            developer_token: None,
        }
    }
}

impl ModelConfig {
    /// A config for `model_name` with every other field at its default.
    pub fn new(model_name: impl Into<String>) -> Self {
        ModelConfig { model_name: model_name.into(), ..Default::default() }
    }
}
