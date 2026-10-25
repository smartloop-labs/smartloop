//! Port of `smartloop/model/model_manager.py`.
//!
//! [`Model`] couples a workspace with the concrete [`SLPModel`] that operates
//! on it; [`ModelManager`] builds one, attaching the right downloader.

use std::path::Path;
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::auth::credential_store;
use crate::config::app_settings::AppSettings;
use crate::config::model_config::ModelConfig;
use crate::llms::base_llm::BaseLLM;
use crate::model::downloader::slp_downloader::{SharedLogger, SharedWorkspace};
use crate::model::downloader::{HFDownloader, SLPArtifactDownloader, SLPDownloader};
use crate::model::generic::GenericModel;
use crate::model::slp_model::{DocumentProgressCallback, DocumentSource, SLPModel, TrainProgressCallback};
use crate::model::spec::ModelSpec;
use crate::utils::log_utils::{ConsoleLogger, LoggerLike};
use crate::workspace::types::Document;
use crate::workspace::Workspace;

/// Python logger name (`logging.getLogger(__name__)`).
pub const LOG_TARGET: &str = "smartloop.model.model_manager";

static SETTINGS: Lazy<AppSettings> = Lazy::new(AppSettings::new);

/// Module-level settings (`_settings = AppSettings()`).
pub fn settings() -> &'static AppSettings {
    &SETTINGS
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
        ModelRecord { id: self.id.clone(), workspace: self.workspace.read().clone() }
    }

    /// The attached model implementation (Python would raise
    /// `AttributeError` on `None`).
    fn slp(&self) -> anyhow::Result<&Arc<dyn SLPModel>> {
        self.slp_model.as_ref().ok_or_else(|| anyhow::anyhow!("'NoneType' object has no attribute 'slp_model'"))
    }

    /// Persist workspace state (delegates to `Workspace::save`).
    pub fn save(&self) -> anyhow::Result<()> {
        self.workspace.read().save()?;
        Ok(())
    }

    /// Initialize the model (download weights).
    pub fn init(&self) -> anyhow::Result<()> {
        self.slp()?.init()
    }

    /// Add a source file or URL to the project, then save.
    pub fn add_source(
        &self,
        source: DocumentSource,
        progress_callback: Option<DocumentProgressCallback>,
    ) -> anyhow::Result<Vec<Document>> {
        let new_docs = self.slp()?.add_document(source, progress_callback, None, false)?;
        self.save()?;
        Ok(new_docs)
    }

    /// Add a skill to the current project, then save.
    pub fn add_skill(&self, name: &str, content: &str) -> anyhow::Result<()> {
        self.slp()?.add_skill(name, content)?;
        self.save()
    }

    /// Print the project status.
    pub fn status(&self) -> anyhow::Result<()> {
        self.slp()?.status()
    }

    /// Train, then save.
    pub fn train(&self, progress_callback: Option<TrainProgressCallback>) -> anyhow::Result<()> {
        self.slp()?.train(progress_callback)?;
        self.save()
    }

    /// Build.
    pub fn build(&self, progress_callback: Option<TrainProgressCallback>) -> anyhow::Result<()> {
        self.slp()?.build(progress_callback)
    }

    /// The loaded LLM instance.
    pub fn get_instance(&self) -> anyhow::Result<Arc<dyn BaseLLM>> {
        self.slp()?.get_instance()
    }
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model").field("id", &self.id).finish_non_exhaustive()
    }
}

/// Construct the model `spec` describes; `None` for a remote model, which
/// has no weights to run here.
pub fn instantiate_model(
    spec: &Arc<ModelSpec>,
    workspace: SharedWorkspace,
    downloader: Arc<dyn SLPDownloader>,
    logger: SharedLogger,
) -> Option<Arc<dyn SLPModel>> {
    if spec.is_remote() {
        return None;
    }
    Some(Arc::new(GenericModel::new(workspace, downloader, logger, Arc::clone(spec))))
}

/// `json.dump(data, f, indent=4)`: four-space indentation and, as Python's
/// `ensure_ascii` default does, every non-ASCII character escaped.
fn python_json_dump(data: &Value) -> anyhow::Result<String> {
    let mut buf = Vec::new();
    let formatter = serde_json::ser::PrettyFormatter::with_indent(b"    ");
    let mut serializer = serde_json::Serializer::with_formatter(&mut buf, formatter);
    data.serialize(&mut serializer)?;
    let text = String::from_utf8(buf)?;
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            let mut units = [0u16; 2];
            for unit in c.encode_utf16(&mut units) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    Ok(out)
}

/// Clear `project_id` on the persisted config of `model_name` in the raw
/// config file.  Best effort: any failure leaves the file as it was.
fn clear_persisted_project_id(config_file: &str, model_name: &str) -> anyhow::Result<()> {
    let raw = std::fs::read_to_string(config_file)?;
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(());
    }
    let mut data: Value = serde_json::from_str(raw)?;
    if let Some(models) = data
        .get_mut("workspace")
        .and_then(Value::as_object_mut)
        .and_then(|ws| ws.get_mut("models"))
        .and_then(Value::as_array_mut)
    {
        for model in models.iter_mut() {
            let Some(model) = model.as_object_mut() else { continue };
            if model.get("model_name").and_then(Value::as_str) == Some(model_name) {
                if let Some(cfg) = model.get_mut("config").and_then(Value::as_object_mut) {
                    cfg.insert("project_id".to_string(), Value::Null);
                    break;
                }
            }
        }
    }
    std::fs::write(config_file, python_json_dump(&data)?)?;
    Ok(())
}

/// `ModelConfig.model_dump()` of `base` with every non-`None` field of
/// `overlay` written over it.
fn merge_configs(base: &ModelConfig, overlay: &ModelConfig) -> anyhow::Result<ModelConfig> {
    let mut merged: Map<String, Value> = match serde_json::to_value(base)? {
        Value::Object(map) => map,
        _ => Map::new(),
    };
    if let Value::Object(fields) = serde_json::to_value(overlay)? {
        for (key, value) in fields {
            if !value.is_null() {
                merged.insert(key, value);
            }
        }
    }
    let mut config: ModelConfig = serde_json::from_value(Value::Object(merged))?;
    // Never serialized, so carried across by hand.
    config.developer_token = overlay.developer_token.clone().or_else(|| base.developer_token.clone());
    Ok(config)
}

/// Builds [`Model`]s.
pub struct ModelManager;

impl ModelManager {
    /// Clear `config.project_id` when it names a project not in this workspace
    /// (also on the workspace model's config and in the persisted config file);
    /// returns the (possibly updated) config.
    pub fn purge_stale_project_id(
        workspace: &mut Workspace,
        config: &mut ModelConfig,
        logger: &dyn LoggerLike,
    ) -> ModelConfig {
        let Some(project_id) = config.project_id.clone().filter(|p| !p.is_empty()) else {
            return config.clone();
        };
        if workspace.projects.is_empty() {
            return config.clone();
        }
        if workspace.projects.iter().any(|p| p.id == project_id) {
            return config.clone();
        }

        logger.warning(&format!(
            "[model] config names project {project_id}, which is not in this workspace; clearing stale project binding"
        ));
        config.project_id = None;
        if let Some(ws_model) = workspace.get_model_mut(Some(&config.model_name)) {
            if let Some(ws_config) = ws_model.config.as_mut() {
                ws_config.project_id = None;
            }
        }

        let config_file = Workspace::config_file();
        if Path::new(&config_file).exists() {
            let _ = clear_persisted_project_id(&config_file, &config.model_name);
        }
        config.clone()
    }

    /// Load a model from the persisted workspace config file, merging `config`
    /// into the active workspace model.
    ///
    /// The Python fallback for a missing, empty or corrupt file called
    /// `ModelManager._create_new_model`, which does not exist, so that path
    /// raised `AttributeError`; it fails the same way here.
    pub fn load_model(spec: &Arc<ModelSpec>, config: ModelConfig) -> anyhow::Result<Model> {
        let _ = spec;
        let create_new_model = || -> anyhow::Result<Model> {
            anyhow::bail!("type object 'ModelManager' has no attribute '_create_new_model'")
        };

        let config_file = Workspace::config_file();
        if !Path::new(&config_file).exists() {
            return create_new_model();
        }

        let content = std::fs::read_to_string(&config_file)?;
        let content = content.trim();
        if content.is_empty() {
            // "Config file is empty"
            return create_new_model();
        }
        let mut data: Value = match serde_json::from_str(content) {
            Ok(data) => data,
            Err(_) => return create_new_model(),
        };
        // Remove computed fields that shouldn't be in the constructor
        if let Some(ws) = data.get_mut("workspace").and_then(Value::as_object_mut) {
            for key in ["model_dir", "model_path", "current_project", "model_id"] {
                ws.remove(key);
            }
        }
        let record: ModelRecord = match serde_json::from_value(data) {
            Ok(record) => record,
            Err(_) => return create_new_model(),
        };
        let mut workspace = record.workspace;

        // Re-resolve workspace paths against the current base_dir
        let workspace_dir = Path::new(&settings().home_dir()).join(&workspace.id);
        workspace.workspace_dir = workspace_dir.to_string_lossy().into_owned();
        workspace.models_dir = workspace_dir.join("models").to_string_lossy().into_owned();

        // Restore runtime state from the active WorkspaceModel
        if let Some(active_ws_model) = workspace.get_model_mut(Some(&config.model_name)) {
            active_ws_model.config = Some(match &active_ws_model.config {
                Some(existing) => match merge_configs(existing, &config) {
                    Ok(merged) => merged,
                    Err(_) => return create_new_model(),
                },
                None => config.clone(),
            });
        }
        let mut config = config;
        let logger = crate::utils::logging_setup::get_logger(LOG_TARGET);
        ModelManager::purge_stale_project_id(&mut workspace, &mut config, &logger);
        // Merge config only — don't change workspace active model

        Ok(Model::new(record.id, Arc::new(RwLock::new(workspace)), None))
    }

    /// Return a [`Model`] backed by `workspace` (already initialised via
    /// `Workspace::init`): registers the model, attaches a downloader and an
    /// [`SLPModel`] instance.  `logger` defaults to a [`ConsoleLogger`];
    /// `config` defaults to the active workspace model's config.  Callers are
    /// responsible for calling `model.save()` after their operation succeeds.
    pub fn get_model(
        spec: &Arc<ModelSpec>,
        workspace: SharedWorkspace,
        logger: Option<SharedLogger>,
        config: Option<ModelConfig>,
    ) -> anyhow::Result<Model> {
        let logger: SharedLogger = logger.unwrap_or_else(|| Arc::new(ConsoleLogger::new()));
        let result = Self::build_model(spec, workspace, &logger, config);
        if let Err(e) = &result {
            logger.error(&format!("Failed to initialise model: {e}"));
        }
        result
    }

    fn build_model(
        spec: &Arc<ModelSpec>,
        workspace: SharedWorkspace,
        logger: &SharedLogger,
        config: Option<ModelConfig>,
    ) -> anyhow::Result<Model> {
        // Use the passed config; fall back to the active workspace model's config
        let config = match config {
            Some(config) => Some(config),
            None => workspace.read().get_model(None).and_then(|m| m.config),
        };
        let Some(mut config) = config else {
            anyhow::bail!(
                "No model config available — pass config explicitly or ensure workspace has an active model."
            );
        };

        {
            let mut ws = workspace.write();
            // Ensure the model is registered (without changing workspace active model)
            if ws.get_model(Some(&config.model_name)).is_none() {
                ws.add_model(&config.model_name, &spec.model_id, Some(config.clone()), true);
            }

            // A `project_id` in the config used to be applied by moving the
            // workspace's `current` flag onto that project.  It resolves
            // through the request/process binding now
            // (`Workspace::current_project`), so there is nothing to switch
            // here: the caller that named the project is already bound to it,
            // and writing the flag would move the UI's selection for every
            // other reader — a workspace-owned field a pinned process cannot
            // persist in any case.
            config = ModelManager::purge_stale_project_id(&mut ws, &mut config, logger.as_ref());
        }

        // Load model id from persisted config, or generate a new one
        let new_id = || uuid::Uuid::new_v4().to_string();
        let config_file = Workspace::config_file();
        let model_id = if Path::new(&config_file).exists() {
            let raw = std::fs::read_to_string(&config_file)?;
            let raw = raw.trim();
            if raw.is_empty() {
                new_id()
            } else {
                match serde_json::from_str::<Value>(raw) {
                    Ok(data) => match data.get("id") {
                        Some(Value::String(id)) => id.clone(),
                        Some(other) if !other.is_null() => other.to_string(),
                        _ => new_id(),
                    },
                    Err(_) => {
                        logger.warning("config.json contains invalid JSON; generating new model id");
                        new_id()
                    }
                }
            }
        } else {
            new_id()
        };

        // Attach the downloader + SLPModel instance
        let use_hf = !credential_store::has_credentials() && config.hf_token.as_deref().is_some_and(|t| !t.is_empty());
        let downloader: Arc<dyn SLPDownloader> = if use_hf {
            Arc::new(HFDownloader::new(Arc::clone(&workspace), Arc::clone(logger)))
        } else {
            Arc::new(SLPArtifactDownloader::new(Arc::clone(&workspace), Arc::clone(logger), None))
        };
        let slp_model = instantiate_model(spec, Arc::clone(&workspace), downloader, Arc::clone(logger))
            .ok_or_else(|| anyhow::anyhow!("{} is a remote model and runs behind the platform API", spec.name))?;

        Ok(Model::new(model_id, workspace, Some(slp_model)))
    }
}
