//! Port of `smartloop/model/downloader/artifact_source.py`.
//!
//! Artifact sources provide download URLs for model files.
//!
//! An [`ArtifactSource`] knows how to resolve a model identifier into a
//! `{filename: url}` map.  The actual downloading is handled separately by the
//! downloader.

use std::sync::Arc;

use indexmap::IndexMap;
use parking_lot::Mutex;
use serde_json::Value;

use crate::model::downloader::artifact_downloader::DownloadError;
use crate::platform::SLPPlatformClient;

/// `{filename: signed_url}`, in the order the platform returned them.
pub type ArtifactMap = IndexMap<String, String>;

/// Base trait for artifact URL providers.
pub trait ArtifactSource: Send + Sync {
    /// Return `{filename: signed_url}` for the given model.
    fn get_artifacts(&self, model_name: &str) -> anyhow::Result<ArtifactMap>;

    /// Whether this source requires authentication.
    fn requires_auth(&self) -> bool {
        false
    }
}

/// An artifact source that can also list the bootstrap catalog.
pub trait BootstrapArtifactSource: ArtifactSource {
    /// Fetch (and cache) the bootstrap model list (`list[dict]`).  The
    /// cached list is shared, so repeated calls hand back the same object.
    fn list_models(&self) -> anyhow::Result<Arc<Vec<Value>>>;
}

/// The slice of [`SLPPlatformClient`] the artifact sources use; a trait so
/// tests can substitute a fake platform client.
pub trait ArtifactPlatformClient: Send + Sync {
    /// `/v1/models/{name}/download` → `{filename: signed_url}`.
    fn get_model_artifacts(&self, model_name: &str) -> anyhow::Result<ArtifactMap>;
    /// `/v1/bootstrap/models` → list of model dicts.
    fn list_bootstrap_models(&self) -> anyhow::Result<Vec<Value>>;
}

/// Platform failures are passed on as [`crate::platform::PlatformError`]
/// inside the `anyhow::Error`, so callers can downcast to recover the kind
/// (`ModelAccessError`, `PermissionError`, `ConnectionError`, ...).
impl ArtifactPlatformClient for SLPPlatformClient {
    fn get_model_artifacts(&self, model_name: &str) -> anyhow::Result<ArtifactMap> {
        let pairs = SLPPlatformClient::get_model_artifacts(self, model_name).map_err(anyhow::Error::new)?;
        Ok(pairs.into_iter().collect())
    }

    fn list_bootstrap_models(&self) -> anyhow::Result<Vec<Value>> {
        let models = SLPPlatformClient::list_bootstrap_models(self).map_err(anyhow::Error::new)?;
        Ok(models.into_iter().map(Value::Object).collect())
    }
}

/// Authenticated artifact source using the `/v1/models/{name}/download` endpoint.
pub struct SLPModelSource {
    platform_client: Arc<dyn ArtifactPlatformClient>,
}

impl SLPModelSource {
    /// Wrap a platform client.
    pub fn new(platform_client: Arc<dyn ArtifactPlatformClient>) -> Self {
        Self { platform_client }
    }
}

impl ArtifactSource for SLPModelSource {
    fn get_artifacts(&self, model_name: &str) -> anyhow::Result<ArtifactMap> {
        self.platform_client.get_model_artifacts(model_name)
    }

    fn requires_auth(&self) -> bool {
        true
    }
}

/// Unauthenticated artifact source using the `/v1/bootstrap/models` endpoint.
pub struct SLPBootstrapSource {
    platform_client: Arc<dyn ArtifactPlatformClient>,
    models: Mutex<Option<Arc<Vec<Value>>>>,
}

impl SLPBootstrapSource {
    /// Wrap a platform client; the model list is fetched lazily and cached.
    pub fn new(platform_client: Arc<dyn ArtifactPlatformClient>) -> Self {
        Self { platform_client, models: Mutex::new(None) }
    }

    /// Fetch and cache the bootstrap model list.  Every call after the first
    /// returns the same cached list (`first is second` in Python).
    pub fn list_models(&self) -> anyhow::Result<Arc<Vec<Value>>> {
        let mut models = self.models.lock();
        if let Some(cached) = models.as_ref() {
            return Ok(Arc::clone(cached));
        }
        let fetched = Arc::new(self.platform_client.list_bootstrap_models()?);
        *models = Some(Arc::clone(&fetched));
        Ok(fetched)
    }
}

/// The `name` of a bootstrap catalog entry, if it has one (`m.get("name")`).
pub fn entry_name(entry: &Value) -> Option<&str> {
    entry.get("name").and_then(Value::as_str)
}

/// `", ".join(m.get("name", "?") for m in models)`.
pub fn available_names(models: &[Value]) -> String {
    models.iter().map(|m| entry_name(m).unwrap_or("?")).collect::<Vec<_>>().join(", ")
}

/// A catalog entry's `artifacts` dict as an [`ArtifactMap`] (empty when absent).
pub fn entry_artifacts(entry: &Value) -> ArtifactMap {
    entry
        .get("artifacts")
        .and_then(Value::as_object)
        .map(|artifacts| {
            artifacts
                .iter()
                .map(|(name, url)| {
                    let url = url.as_str().map(str::to_string).unwrap_or_else(|| url.to_string());
                    (name.clone(), url)
                })
                .collect()
        })
        .unwrap_or_default()
}

impl ArtifactSource for SLPBootstrapSource {
    /// Errors (ValueError in Python, [`DownloadError::Value`] here) with
    /// `"Model '<name>' not found in bootstrap. Available: a, b"` or
    /// `"No artifacts returned for bootstrap model '<name>'"`.
    fn get_artifacts(&self, model_name: &str) -> anyhow::Result<ArtifactMap> {
        let models = self.list_models()?;
        let Some(target) = models.iter().find(|m| entry_name(m) == Some(model_name)) else {
            return Err(anyhow::Error::new(DownloadError::Value(format!(
                "Model '{model_name}' not found in bootstrap. Available: {}",
                available_names(&models)
            ))));
        };
        let artifacts = entry_artifacts(target);
        if artifacts.is_empty() {
            return Err(anyhow::Error::new(DownloadError::Value(format!(
                "No artifacts returned for bootstrap model '{model_name}'"
            ))));
        }
        Ok(artifacts)
    }
}

impl BootstrapArtifactSource for SLPBootstrapSource {
    fn list_models(&self) -> anyhow::Result<Arc<Vec<Value>>> {
        SLPBootstrapSource::list_models(self)
    }
}
