//! Port of `tests/api/test_models_load_errors.py`.
//!
//! `POST /v1/models/load` answers the status each failure actually is.
//!
//! It mapped every `PermissionError` to 401, and `PermissionError` is four
//! different things here: a platform 401 (sign in), a plan refusal
//! (`ModelAccessError`, 403), and — because `PermissionError` is an `OSError`
//! — any local filesystem refusal.  So a full disk and an unentitled model both
//! told the desktop client to show a sign-in prompt, for a session that was
//! already valid.

use std::sync::Arc;

use serde_json::json;
use serial_test::serial;

use smartloop::api::services::model_service::ModelServiceHooks;
use smartloop::api::services::ModelService;
use smartloop::llms::base_llm::BaseLLM;
use smartloop::platform::{ModelAccessError, PlatformError};

use crate::common::api_core::{ModelServiceOverride, NullLlm, TestClient};

/// `model_service`: a stand-in service bound as the dependency, whose
/// `load_model` the test sets.
fn model_service(
    load_model: impl Fn() -> anyhow::Result<Arc<dyn BaseLLM>> + Send + Sync + 'static,
) -> ModelServiceOverride {
    let svc = ModelService::new_bare();
    svc.set_hooks(ModelServiceHooks { load_model: Some(Arc::new(load_model)), ..Default::default() });
    ModelServiceOverride::install(svc)
}

fn platform_401() -> anyhow::Error {
    PlatformError::Permission {
        message: "Authentication failed. Token may be expired or invalid.".to_string(),
        status_code: Some(401),
    }
    .into()
}

fn plan_403() -> anyhow::Error {
    ModelAccessError::new("Model 'gemma3-12b' is not available on your current plan.").into()
}

/// No status code and not a plan refusal — a local disk problem, which
/// signing in cannot fix and upgrading cannot fix.
fn local_permission_error() -> anyhow::Error {
    std::io::Error::new(std::io::ErrorKind::PermissionDenied, "models dir is not writable").into()
}

fn connection_error() -> anyhow::Error {
    PlatformError::connection("platform unreachable").into()
}

async fn status_for_failure(make: fn() -> anyhow::Error) -> u16 {
    let client = TestClient::new();
    let _svc = model_service(move || Err(make()));
    client.post("/v1/models/load", json!({})).await.status_code
}

mod test_load_model_failure_statuses {
    use super::*;

    #[tokio::test]
    #[serial]
    async fn test_the_status_matches_the_failure_platform_401() {
        assert_eq!(status_for_failure(platform_401).await, 401);
    }

    #[tokio::test]
    #[serial]
    async fn test_the_status_matches_the_failure_plan_403() {
        assert_eq!(status_for_failure(plan_403).await, 403);
    }

    #[tokio::test]
    #[serial]
    async fn test_the_status_matches_the_failure_local_permission_error() {
        assert_eq!(status_for_failure(local_permission_error).await, 500);
    }

    #[tokio::test]
    #[serial]
    async fn test_the_status_matches_the_failure_connection_error() {
        assert_eq!(status_for_failure(connection_error).await, 502);
    }

    /// The regression in one line: 403 must not read as 401.
    #[tokio::test]
    #[serial]
    async fn test_a_plan_refusal_is_not_a_login_prompt() {
        let client = TestClient::new();
        let _svc = model_service(|| Err(plan_403()));
        let resp = client.post("/v1/models/load", json!({})).await;
        assert_eq!(resp.status_code, 403);
        assert!(resp.json()["detail"].as_str().unwrap_or_default().contains("plan"), "{}", resp.text());
    }

    #[tokio::test]
    #[serial]
    async fn test_a_successful_load_still_answers_200() {
        let client = TestClient::new();
        let _svc = model_service(|| Ok(Arc::new(NullLlm::default()) as Arc<dyn BaseLLM>));
        let resp = client.post("/v1/models/load", json!({})).await;
        assert_eq!(resp.status_code, 200);
        assert_eq!(resp.json()["status"], "ok");
    }
}
