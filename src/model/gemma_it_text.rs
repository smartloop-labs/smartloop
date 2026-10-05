//! Port of `smartloop/model/gemma_it_text.py`.

#[allow(unused_imports)]
use crate::types::{ModelCapability, ModelType};

crate::generic_model_type!(
    /// The 1B text-only orchestrator model (`sl-mini`).
    GemmaItTextModel {
        model_id: "smartloop-ai/slp-orchestra-mini",
        model_type: ModelType::Text,
        model_version: 3,
        param_size_in_b: 1.0,
        default_capabilities: crate::model::slp_model::DEFAULT_CAPABILITIES.to_vec(),
        thinking_default: false,
        vision_chat_handler: None,
    }
);
