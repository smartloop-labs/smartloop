//! Port of `smartloop/model/qwen2_vl_2b.py`.

#[allow(unused_imports)]
use crate::types::{ModelCapability, ModelType};

crate::generic_model_type!(
    /// Qwen2-VL 2B — a lighter multimodal model for document OCR / vision.
    ///
    /// Smaller than Qwen2.5-VL-3B so it fits on GPU alongside the orchestrator
    /// and on-demand agent models on memory-constrained devices.  Runs via the
    /// same `Qwen25VLChatHandler` (the `qwen2vl` architecture is handled by it).
    #[allow(non_camel_case_types)]
    Qwen2VL_2bModel {
        model_id: "Qwen/Qwen2-VL-2B-Instruct",
        model_type: ModelType::MultiModal,
        model_version: 2,
        param_size_in_b: 2.0,
        default_capabilities: vec![ModelCapability::Vision],
        thinking_default: false,
        vision_chat_handler: Some("Qwen25VLChatHandler"),
    }
);
