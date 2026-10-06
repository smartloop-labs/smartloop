//! Port of `smartloop/model/ministral_3b.py`.

#[allow(unused_imports)]
use crate::types::{ModelCapability, ModelType};

crate::generic_model_type!(
    /// Ministral 3B instruct (vision via the generic mtmd path).
    Ministral3bModel {
        model_id: "mistralai/Ministral-3-3B-Instruct-2512",
        model_type: ModelType::MultiModal,
        model_version: 3,
        param_size_in_b: 3.0,
        default_capabilities: vec![ModelCapability::General, ModelCapability::Dp, ModelCapability::Vision],
        thinking_default: false,
        // No Pixtral/mistral3-specific handler: vision (arch 'mistral3') decodes
        // through llama.cpp's generic mtmd path, so declare MTMDChatHandler.
        vision_chat_handler: Some("MTMDChatHandler"),
    }
);
