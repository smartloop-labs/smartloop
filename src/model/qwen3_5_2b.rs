//! Port of `smartloop/model/qwen3_5_2b.py`.

#[allow(unused_imports)]
use crate::types::{ModelCapability, ModelType};

crate::generic_model_type!(
    /// Qwen3.5-2B — a lighter multimodal model for document OCR / vision.
    ///
    /// A hybrid-reasoning model that can think in a `<think>` block; THINKING
    /// is declared and left off by default.  Its GGUF architecture is `qwen35`,
    /// which no arch-sniffed handler matches, so the declared handler is what
    /// makes its vision work at all.
    Qwen3_5_2bModel {
        model_id: "Qwen/Qwen3.5-2B",
        model_type: ModelType::MultiModal,
        model_version: 3,
        param_size_in_b: 2.0,
        default_capabilities: vec![ModelCapability::General, ModelCapability::Reasoning, ModelCapability::Dp, ModelCapability::Vision, ModelCapability::Thinking],
        thinking_default: false,
        // The generic MTMDChatHandler renders the model's own embedded chat
        // template (keeping <think> control); Qwen25VLChatHandler would impose a
        // hardcoded ChatML prompt that predates Qwen3.5.
        vision_chat_handler: Some("MTMDChatHandler"),
    }
);
