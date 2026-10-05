//! Port of `smartloop/model/gemma_it.py`.

#[allow(unused_imports)]
use crate::types::{ModelCapability, ModelType};

crate::generic_model_type!(
    /// Gemma 3 4B instruction-tuned multimodal model.
    GemmaItModel {
        model_id: "google/gemma-3-4b-it",
        model_type: ModelType::MultiModal,
        model_version: 3,
        param_size_in_b: 4.0,
        default_capabilities: vec![ModelCapability::General, ModelCapability::Dp],
        thinking_default: false,
        // llama.cpp has no Gemma3-specific handler; Gemma 3's mmproj loads
        // through the generic mtmd path (projector auto-detected from the GGUF).
        vision_chat_handler: Some("MTMDChatHandler"),
    }
);
