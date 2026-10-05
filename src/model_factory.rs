//! Builds a [`Model`] for a [`ModelConfig`].
//!
//! Port of `smartloop/model_factory.py`.

#[allow(unused_imports)]
use crate::config::app_settings::AppSettings;
use crate::config::ModelConfig;
#[allow(unused_imports)]
use crate::config::SUPPORTED_MODELS;
#[allow(unused_imports)]
use crate::model::model_manager::{Model, ModelManager};
#[allow(unused_imports)]
use crate::workspace::Workspace;

/// Factory for workspace-backed models.
pub struct ModelFactory;

impl ModelFactory {
    /// Resolve the model class for `config.model_name` (error
    /// "Unknown model: <name>" when unsupported), ensure the home dir exists,
    /// initialise the workspace and return the registered model.
    pub fn get_model(config: &ModelConfig) -> anyhow::Result<Model> {
        let _ = config;
        todo!()
    }
}
