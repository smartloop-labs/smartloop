//! Builds a [`Model`] for a [`ModelConfig`].
//!
//! Port of `smartloop/model_factory.py`.

use std::sync::Arc;

use once_cell::sync::Lazy;

use crate::config::app_settings::AppSettings;
use crate::config::ModelConfig;
use crate::model::downloader::slp_downloader::SharedLogger;
use crate::model::model_manager::{Model, ModelManager};
use crate::model::spec::MODEL_SPECS;
use crate::utils::log_utils::ConsoleLogger;
use crate::workspace::Workspace;

/// Module-level settings (`_settings = AppSettings()`), read once.
static SETTINGS: Lazy<AppSettings> = Lazy::new(AppSettings::new);

/// Factory for workspace-backed models.
pub struct ModelFactory;

impl ModelFactory {
    /// Resolve the spec for `config.model_name` (error "Unknown model:
    /// <name>" when there is none), ensure the home dir exists,
    /// initialise the workspace and return the registered model.
    ///
    /// Logs through a [`ConsoleLogger`], the Python default.
    pub fn get_model(config: &ModelConfig) -> anyhow::Result<Model> {
        Self::get_model_with_logger(config, Arc::new(ConsoleLogger::new()))
    }

    /// [`ModelFactory::get_model`] with an explicit logger (the Python
    /// `logger` keyword).
    pub fn get_model_with_logger(config: &ModelConfig, logger: SharedLogger) -> anyhow::Result<Model> {
        let Some(spec) = MODEL_SPECS.get(&config.model_name) else {
            anyhow::bail!("Unknown model: {}", config.model_name);
        };

        std::fs::create_dir_all(SETTINGS.home_dir())?;

        let workspace = Workspace::init(config);

        ModelManager::get_model(&spec, workspace, Some(logger), Some(config.clone()))
    }
}
