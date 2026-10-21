//! Port of `smartloop/model/__init__.py`.

pub mod downloader;
pub mod gemma4_2b;
pub mod gemma4_4b;
pub mod gemma_it;
pub mod gemma_it_text;
pub mod generic;
pub mod ministral_3b;
pub mod model_manager;
pub mod qwen2_vl_2b;
pub mod qwen3_5_2b;
pub mod slp_model;
pub mod spec;
pub mod workers;

pub use gemma4_2b::Gemma4_2bModel;
pub use gemma4_4b::Gemma4_4bModel;
pub use gemma_it::GemmaItModel;
pub use gemma_it_text::GemmaItTextModel;
pub use ministral_3b::Ministral3bModel;
pub use qwen2_vl_2b::Qwen2VL_2bModel;
pub use qwen3_5_2b::Qwen3_5_2bModel;
pub use slp_model::SLPModel;
