//! Loads the workspace's active GGUF model (plus any project LoRA adapter).
//!
//! Port of `smartloop/loader/model_loader.py`.

use std::sync::Arc;

#[allow(unused_imports)]
use crate::config::app_settings::AppSettings;
use crate::llms::llama_cpp::LlamaCpp;
#[allow(unused_imports)]
use crate::llms::llm_factory::LlmFactory;
#[allow(unused_imports)]
use crate::utils::device_utils::get_device_config;
use crate::workspace::Workspace;

/// Resolves and loads the model file for the workspace's current project.
///
/// The Python version took a `logging.Logger`; logging here goes through
/// `tracing` under target `smartloop.loader.model_loader`.
pub struct ModelLoader {
    pub workspace: Arc<Workspace>,
    pub gguf_path: Option<String>,
}

impl ModelLoader {
    /// A loader for `workspace`; `gguf_path` overrides the workspace model path.
    pub fn new(workspace: Arc<Workspace>, gguf_path: Option<String>) -> Self {
        ModelLoader { workspace, gguf_path }
    }

    /// Find the GGUF adapter file (`<project_dir>/model/<adapter>.gguf`) for
    /// the current project, if one exists.
    fn resolve_adapter_path(&self) -> Option<String> {
        todo!()
    }

    /// Load the model: errors with "Model file not found: <path>" when the
    /// resolved file is absent; caps the context at `llm_main_n_ctx_cap` when
    /// set; prints a Project/Device parameter table.
    pub fn load(&self) -> anyhow::Result<Arc<LlamaCpp>> {
        let _ = self.resolve_adapter_path();
        todo!()
    }
}
