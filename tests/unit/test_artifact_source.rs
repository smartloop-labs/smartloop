//! Port of `tests/unit/test_artifact_source.py`.

use std::sync::Arc;

use serde_json::{json, Value};
use smartloop::model::downloader::artifact_downloader::DownloadError;
use smartloop::model::downloader::artifact_source::{
    ArtifactMap, ArtifactPlatformClient, ArtifactSource, SLPBootstrapSource, SLPModelSource,
};

/// Stub for SLPPlatformClient — no network calls.
#[derive(Default)]
struct FakePlatformClient {
    artifacts: Vec<(String, ArtifactMap)>,
    bootstrap_models: Vec<Value>,
}

impl ArtifactPlatformClient for FakePlatformClient {
    fn get_model_artifacts(&self, model_name: &str) -> anyhow::Result<ArtifactMap> {
        match self.artifacts.iter().find(|(name, _)| name == model_name) {
            Some((_, artifacts)) => Ok(artifacts.clone()),
            None => Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!("Model '{model_name}' not found on server."),
            )
            .into()),
        }
    }

    fn list_bootstrap_models(&self) -> anyhow::Result<Vec<Value>> {
        if self.bootstrap_models.is_empty() {
            return Err(DownloadError::Value("Bootstrap API returned no models".to_string()).into());
        }
        Ok(self.bootstrap_models.clone())
    }
}

fn artifact_map(pairs: &[(&str, &str)]) -> ArtifactMap {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

/// A `ValueError` raised through an artifact source.
fn is_value_error(err: &anyhow::Error) -> bool {
    matches!(err.downcast_ref::<DownloadError>(), Some(DownloadError::Value(_)))
}

/// `TestSLPModelSource`
mod test_slp_model_source {
    use super::*;


    #[test]
    fn test_get_artifacts() {
        let client = FakePlatformClient {
            artifacts: vec![(
                "slp-orchestra-mini".to_string(),
                artifact_map(&[("model.gguf", "https://storage/model.gguf")]),
            )],
            ..Default::default()
        };
        let source = SLPModelSource::new(Arc::new(client));
        let result = source.get_artifacts("slp-orchestra-mini").unwrap();
        assert_eq!(result, artifact_map(&[("model.gguf", "https://storage/model.gguf")]));
    }

    #[test]
    fn test_requires_auth() {
        let source = SLPModelSource::new(Arc::new(FakePlatformClient::default()));
        assert!(source.requires_auth());
    }

    #[test]
    fn test_unknown_model_raises() {
        let source = SLPModelSource::new(Arc::new(FakePlatformClient::default()));
        let err = source.get_artifacts("nonexistent").unwrap_err();
        let io = err.downcast_ref::<std::io::Error>().expect("a PermissionError");
        assert_eq!(io.kind(), std::io::ErrorKind::PermissionDenied);
    }
}

/// `TestSLPBootstrapSource`
mod test_slp_bootstrap_source {
    use super::*;


    fn make_source(models: Vec<Value>) -> SLPBootstrapSource {
        SLPBootstrapSource::new(Arc::new(FakePlatformClient { bootstrap_models: models, ..Default::default() }))
    }

    #[test]
    fn test_list_models() {
        let models = vec![json!({"name": "slp-orchestra-mini", "artifacts": {"m.gguf": "url"}})];
        let source = make_source(models.clone());
        assert_eq!(*source.list_models().unwrap(), models);
    }

    #[test]
    fn test_list_models_caches() {
        let models = vec![json!({"name": "slp-orchestra-mini", "artifacts": {"m.gguf": "url"}})];
        let source = make_source(models);
        let first = source.list_models().unwrap();
        let second = source.list_models().unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn test_requires_auth_false() {
        let source = make_source(Vec::new());
        assert!(!source.requires_auth());
    }

    #[test]
    fn test_get_artifacts() {
        let models = vec![json!({"name": "slp-orchestra-mini", "artifacts": {"m.gguf": "https://url"}})];
        let source = make_source(models);
        assert_eq!(source.get_artifacts("slp-orchestra-mini").unwrap(), artifact_map(&[("m.gguf", "https://url")]));
    }

    #[test]
    fn test_get_artifacts_unknown_model() {
        let models = vec![json!({"name": "slp-orchestra-mini", "artifacts": {"m.gguf": "url"}})];
        let source = make_source(models);
        let err = source.get_artifacts("nonexistent").unwrap_err();
        assert!(is_value_error(&err));
        assert!(err.to_string().contains("not found in bootstrap"));
    }

    #[test]
    fn test_get_artifacts_empty_artifacts() {
        let models = vec![json!({"name": "slp-orchestra-mini", "artifacts": {}})];
        let source = make_source(models);
        let err = source.get_artifacts("slp-orchestra-mini").unwrap_err();
        assert!(is_value_error(&err));
        assert!(err.to_string().contains("No artifacts"));
    }

    #[test]
    fn test_empty_bootstrap_raises() {
        let source = make_source(Vec::new());
        let err = source.list_models().unwrap_err();
        assert!(is_value_error(&err));
    }
}
