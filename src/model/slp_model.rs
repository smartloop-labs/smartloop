//! Port of `smartloop/model/slp_model.py`.
//!
//! The abstract Smartloop model ([`SLPModel`]).  The facts that differ between
//! models live in their [`ModelSpec`](crate::model::spec::ModelSpec).

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;

use crate::llms::base_llm::BaseLLM;
use crate::model::downloader::slp_downloader::{SharedLogger, SharedWorkspace};
use crate::types::{ModelCapability, ModelType};
use crate::workspace::types::Document;

/// A document to add: a string (path or URL; Python `str`) or a filesystem
/// path (Python `Path`, never treated as a URL).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocumentSource {
    Str(String),
    Path(PathBuf),
}

impl From<&str> for DocumentSource {
    fn from(s: &str) -> Self {
        DocumentSource::Str(s.to_string())
    }
}

impl From<String> for DocumentSource {
    fn from(s: String) -> Self {
        DocumentSource::Str(s)
    }
}

impl From<PathBuf> for DocumentSource {
    fn from(p: PathBuf) -> Self {
        DocumentSource::Path(p)
    }
}

impl From<&std::path::Path> for DocumentSource {
    fn from(p: &std::path::Path) -> Self {
        DocumentSource::Path(p.to_path_buf())
    }
}

/// Progress callback for document ingestion: receives
/// `{"stage", "filename", "message"}`.
pub type DocumentProgressCallback = Arc<dyn Fn(Value) + Send + Sync>;

/// Progress callback for training / building (receives the trainer's
/// progress payload as JSON).
pub type TrainProgressCallback = Arc<dyn Fn(Value) + Send + Sync>;

/// Base-class default for `default_capabilities()`: every model does at least
/// general chat.
pub const DEFAULT_CAPABILITIES: &[ModelCapability] = &[ModelCapability::General];

/// Abstract base trait for a Smartloop model.
pub trait SLPModel: Send + Sync {
    /// The workspace this model belongs to.
    fn workspace(&self) -> &SharedWorkspace;

    /// The logger this model reports to.
    fn logger(&self) -> &SharedLogger;

    /// Initialize the model (download its weights).
    fn init(&self) -> anyhow::Result<()>;

    /// Add a new source file, directory or URL to the current project.
    fn add_document(
        &self,
        source: DocumentSource,
        progress_callback: Option<DocumentProgressCallback>,
        cwd: Option<&str>,
        generate_filename: bool,
    ) -> anyhow::Result<Vec<Document>>;

    /// Add a skill to the current project.
    fn add_skill(&self, name: &str, content: &str) -> anyhow::Result<()>;

    /// Print the current project's document status.
    fn status(&self) -> anyhow::Result<()>;

    /// Build the model.
    fn build(&self, progress_callback: Option<TrainProgressCallback>) -> anyhow::Result<()>;

    /// Train the model.
    fn train(&self, progress_callback: Option<TrainProgressCallback>) -> anyhow::Result<()>;

    /// The model's modality.
    fn model_type(&self) -> ModelType;

    /// The HuggingFace-style model id.
    fn model_id(&self) -> String;

    /// Model family version; base default 0.
    fn model_version(&self) -> i64 {
        0
    }

    /// Parameter count in billions; base default 0 (unknown).
    fn param_size_in_b(&self) -> f64 {
        0.0
    }

    /// Capabilities this model declares; base default `[General]`.
    fn default_capabilities(&self) -> Vec<ModelCapability> {
        DEFAULT_CAPABILITIES.to_vec()
    }

    /// Whether reasoning starts ON; base default off.
    fn thinking_default(&self) -> bool {
        false
    }

    /// Vision chat-handler name; `None` for non-vision models.
    fn vision_chat_handler(&self) -> Option<String> {
        None
    }

    /// Load (once) and return the LLM instance for this model.
    fn get_instance(&self) -> anyhow::Result<Arc<dyn BaseLLM>>;
}
