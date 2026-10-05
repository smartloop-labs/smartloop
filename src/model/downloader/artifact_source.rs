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
    /// Fetch (and cache) the bootstrap model list (`list[dict]`).
    fn list_models(&self) -> anyhow::Result<Vec<Value>>;
}

/// The slice of [`SLPPlatformClient`] the artifact sources use; a trait so
/// tests can substitute a fake platform client.
pub trait ArtifactPlatformClient: Send + Sync {
    /// `/v1/models/{name}/download` → `{filename: signed_url}`.
    fn get_model_artifacts(&self, model_name: &str) -> anyhow::Result<ArtifactMap>;
    /// `/v1/bootstrap/models` → list of model dicts.
    fn list_bootstrap_models(&self) -> anyhow::Result<Vec<Value>>;
}

impl ArtifactPlatformClient for SLPPlatformClient {
    fn get_model_artifacts(&self, model_name: &str) -> anyhow::Result<ArtifactMap> {
        let _ = model_name;
        todo!()
    }

    fn list_bootstrap_models(&self) -> anyhow::Result<Vec<Value>> {
        todo!()
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
        let _ = (model_name, &self.platform_client);
        todo!()
    }

    fn requires_auth(&self) -> bool {
        true
    }
}

/// Unauthenticated artifact source using the `/v1/bootstrap/models` endpoint.
pub struct SLPBootstrapSource {
    platform_client: Arc<dyn ArtifactPlatformClient>,
    models: Mutex<Option<Vec<Value>>>,
}

impl SLPBootstrapSource {
    /// Wrap a platform client; the model list is fetched lazily and cached.
    pub fn new(platform_client: Arc<dyn ArtifactPlatformClient>) -> Self {
        Self { platform_client, models: Mutex::new(None) }
    }

    /// Fetch and cache the bootstrap model list.
    pub fn list_models(&self) -> anyhow::Result<Vec<Value>> {
        let _ = (&self.platform_client, &self.models);
        todo!()
    }
}

impl ArtifactSource for SLPBootstrapSource {
    /// Errors (ValueError in Python) with
    /// `"Model '<name>' not found in bootstrap. Available: a, b"` or
    /// `"No artifacts returned for bootstrap model '<name>'"`.
    fn get_artifacts(&self, model_name: &str) -> anyhow::Result<ArtifactMap> {
        let _ = model_name;
        todo!()
    }
}

impl BootstrapArtifactSource for SLPBootstrapSource {
    fn list_models(&self) -> anyhow::Result<Vec<Value>> {
        SLPBootstrapSource::list_models(self)
    }
}
