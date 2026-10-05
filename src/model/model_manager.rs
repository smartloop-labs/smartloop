//! Port of `smartloop/model/model_manager.py`.
//!
//! [`Model`] couples a workspace with the concrete [`SLPModel`] that operates
//! on it; [`ModelManager`] builds one, attaching the right downloader.

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use crate::auth::credential_store;
use crate::config::app_settings::AppSettings;
use crate::config::model_config::ModelConfig;
use crate::llms::base_llm::BaseLLM;
#[allow(unused_imports)]
use crate::model::downloader::{HFDownloader, SLPArtifactDownloader};
use crate::model::downloader::slp_downloader::{SharedLogger, SharedWorkspace};
use crate::model::slp_model::{DocumentProgressCallback, DocumentSource, ModelClass, SLPModel, TrainProgressCallback};
use crate::utils::log_utils::{ConsoleLogger, LoggerLike};
use crate::workspace::Workspace;
use crate::workspace::types::Document;

/// Module-level settings (`_settings = AppSettings()`).
pub fn settings() -> &'static AppSettings {
    todo!()
}

/// The persisted shape of a [`Model`] in the workspace config file
/// (`{"id": ..., "workspace": {...}}`); `slp_model` is never serialized.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecord {
    pub id: String,
    pub workspace: Workspace,
}

/// A workspace plus the model implementation operating on it.
#[derive(Clone)]
pub struct Model {
    pub id: String,
    pub workspace: SharedWorkspace,
    /// Excluded from serialization.
    pub slp_model: Option<Arc<dyn SLPModel>>,
}

impl Model {
    /// Build a model handle.
    pub fn new(id: impl Into<String>, workspace: SharedWorkspace, slp_model: Option<Arc<dyn SLPModel>>) -> Self {
        Self { id: id.into(), workspace, slp_model }
    }

    /// The serializable view of this model.
    pub fn to_record(&self) -> ModelRecord {
        todo!()
    }

    /// Persist workspace state (delegates to `Workspace::save`).
    pub fn save(&self) -> anyhow::Result<()> {
        todo!()
    }

    /// Initialize the model (download weights).
    pub fn init(&self) -> anyhow::Result<()> {
        todo!()
    }

    /// Add a source file or URL to the project, then save.
    pub fn add_source(&self, source: DocumentSource, progress_callback: Option<DocumentProgressCallback>) -> anyhow::Result<Vec<Document>> {
        let _ = (source, progress_callback);
        todo!()
    }

    /// Add a skill to the current project, then save.
    pub fn add_skill(&self, name: &str, content: &str) -> anyhow::Result<()> {
        let _ = (name, content);
        todo!()
    }

    /// Print the project status.
    pub fn status(&self) -> anyhow::Result<()> {
        todo!()
    }

    /// Train, then save.
    pub fn train(&self, progress_callback: Option<TrainProgressCallback>) -> anyhow::Result<()> {
        let _ = progress_callback;
        todo!()
    }

    /// Build.
    pub fn build(&self, progress_callback: Option<TrainProgressCallback>) -> anyhow::Result<()> {
        let _ = progress_callback;
        todo!()
    }

    /// The loaded LLM instance.
    pub fn get_instance(&self) -> anyhow::Result<Arc<dyn BaseLLM>> {
        todo!()
    }
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model").field("id", &self.id).finish_non_exhaustive()
    }
}

/// Construct the concrete model named by `t.class_name` (Python's
/// `t(workspace, downloader, logger)`); `None` for an unknown class name.
pub fn instantiate_model(
    t: &ModelClass,
    workspace: SharedWorkspace,
    downloader: Arc<dyn crate::model::downloader::SLPDownloader>,
    logger: SharedLogger,
) -> Option<Arc<dyn SLPModel>> {
    let _ = (t, workspace, downloader, logger);
    todo!()
}

/// Builds [`Model`]s.
pub struct ModelManager;

impl ModelManager {
    /// Clear `config.project_id` when it names a project not in this workspace
    /// (also on the workspace model's config and in the persisted config file);
    /// returns the (possibly updated) config.
    pub fn purge_stale_project_id(workspace: &mut Workspace, config: &mut ModelConfig, logger: &dyn LoggerLike) -> ModelConfig {
        let _ = (workspace, config, logger);
        todo!()
    }

    /// Load a model from the persisted workspace config file, merging `config`
    /// into the active workspace model; falls back to creating a new model when
    /// the file is missing, empty or corrupt.
    pub fn load_model(t: &ModelClass, config: ModelConfig) -> anyhow::Result<Model> {
        let _ = (t, config);
        todo!()
    }

    /// Return a [`Model`] backed by `workspace` (already initialised via
    /// `Workspace::init`): registers the model, attaches a downloader and an
    /// [`SLPModel`] instance.  `logger` defaults to a [`ConsoleLogger`];
    /// `config` defaults to the active workspace model's config.  Callers are
    /// responsible for calling `model.save()` after their operation succeeds.
    pub fn get_model(
        t: &ModelClass,
        workspace: SharedWorkspace,
        logger: Option<SharedLogger>,
        config: Option<ModelConfig>,
    ) -> anyhow::Result<Model> {
        let _ = (t, workspace, logger, config);
        let _: Option<(ConsoleLogger, PathBuf)> = None;
        todo!()
    }
}
