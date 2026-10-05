//! Port of `smartloop/model/gemma4_2b.py`.

#[allow(unused_imports)]
use crate::types::{ModelCapability, ModelType};

crate::generic_model_type!(
    /// Gemma 4 E2B instruction-tuned multimodal model.
    Gemma4_2bModel {
        model_id: "google/gemma-4-e2b-it",
        model_type: ModelType::MultiModal,
        model_version: 4,
        param_size_in_b: 2.0,
        default_capabilities: vec![ModelCapability::General, ModelCapability::Dp, ModelCapability::Vision, ModelCapability::Thinking],
        thinking_default: false,
        // Gemma 4's chat template reads enable_thinking (defaulting it to false)
        // and emits reasoning on a <|channel>thought…<channel|> channel; THINKING
        // is declared so the picker offers the toggle, left off by default.
        vision_chat_handler: Some("Gemma4ChatHandler"),
    }
);
