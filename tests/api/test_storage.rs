//! Port of `tests/api/test_storage.py`.
//!
//! Tests for the /v1/storage API router.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use serial_test::serial;

use smartloop::api::routers::storage::{self, StorageRouteHooks};
use smartloop::api::services::storage_service::{
    CleanupOptions, Row, StorageError, StorageModelService,
};
use smartloop::workspace::workspace::SharedWorkspace;

use crate::common::misc_client::{get, post_json};

fn sample_usage() -> Row {
    let value = json!({
        "home_dir": "/home/.smartloop",
        "total": 1000,
        "files": 10,
        "reclaimable": 300,
        "kept": 700,
        "categories": {
            // The group subtotals partition the category: in_use + system +
            // reclaimable + overhead == bytes.  Kept realistic so this sample
            // does not drift from what compute_usage actually returns.
            "models": {
                "bytes": 800, "files": 4, "reclaimable": 300, "kept": 500, "items": [],
                "in_use": 400, "system": 90, "overhead": 10,
            },
        },
    });
    match value {
        Value::Object(map) => map,
        _ => unreachable!(),
    }
}

/// A stand-in model service (the Python `MagicMock`), bound as the router's
/// model-service dependency.
struct FakeModelService;

impl StorageModelService for FakeModelService {
    fn ensure_workspace(&self) -> anyhow::Result<Option<SharedWorkspace>> {
        Ok(None)
    }

    fn unload_model_by_name_unlocked(&self, _name: &str) -> anyhow::Result<()> {
        Ok(())
    }
}

/// The `model_service` fixture: binds a fake as the dependency, and clears
/// every router hook afterwards.
struct ModelServiceFixture {
    service: Arc<dyn StorageModelService>,
}

impl ModelServiceFixture {
    fn new() -> Self {
        let service: Arc<dyn StorageModelService> = Arc::new(FakeModelService);
        storage::set_hooks(StorageRouteHooks { model_service: Some(service.clone()), ..Default::default() });
        ModelServiceFixture { service }
    }

    /// `patch(_USAGE, ...)`: replace `compute_usage`, recording whether it was
    /// handed this fixture's service.
    fn patch_usage(
        &self,
        answer: impl Fn() -> anyhow::Result<Row> + Send + Sync + 'static,
    ) -> Arc<Mutex<Vec<bool>>> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let recorded = calls.clone();
        let expected = self.service.clone();
        let hook = Arc::new(move |svc: Option<&dyn StorageModelService>| {
            let same = svc.is_some_and(|r| {
                std::ptr::addr_eq(r as *const dyn StorageModelService, Arc::as_ptr(&expected))
            });
            recorded.lock().push(same);
            answer()
        });
        storage::set_hooks(StorageRouteHooks {
            model_service: Some(self.service.clone()),
            compute_usage: Some(hook),
            run_cleanup: None,
        });
        calls
    }

    /// `patch(_CLEANUP, ...)`: replace `run_cleanup`, recording its arguments.
    fn patch_cleanup(
        &self,
        answer: impl Fn() -> anyhow::Result<Row> + Send + Sync + 'static,
    ) -> Arc<Mutex<Vec<(bool, CleanupOptions)>>> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let recorded = calls.clone();
        let expected = self.service.clone();
        let hook = Arc::new(move |svc: Option<&dyn StorageModelService>, opts: CleanupOptions| {
            let same = svc.is_some_and(|r| {
                std::ptr::addr_eq(r as *const dyn StorageModelService, Arc::as_ptr(&expected))
            });
            recorded.lock().push((same, opts));
            answer()
        });
        storage::set_hooks(StorageRouteHooks {
            model_service: Some(self.service.clone()),
            compute_usage: None,
            run_cleanup: Some(hook),
        });
        calls
    }
}

impl Drop for ModelServiceFixture {
    fn drop(&mut self) {
        storage::set_hooks(StorageRouteHooks::default());
    }
}

fn client() -> axum::Router {
    storage::router()
}

fn row(value: Value) -> Row {
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

mod test_get_storage_usage {
    use super::*;

    #[tokio::test]
    #[serial]
    async fn test_returns_the_usage_report() {
        let model_service = ModelServiceFixture::new();
        model_service.patch_usage(|| Ok(sample_usage()));
        let resp = get(&client(), "/v1/storage/usage").await;

        assert_eq!(resp.status, 200);
        assert_eq!(resp.json(), Value::Object(sample_usage()));
    }

    #[tokio::test]
    #[serial]
    async fn test_passes_the_model_service_through() {
        let model_service = ModelServiceFixture::new();
        let usage = model_service.patch_usage(|| Ok(sample_usage()));
        get(&client(), "/v1/storage/usage").await;

        let calls = usage.lock();
        assert_eq!(calls.len(), 1);
        assert!(calls[0], "compute_usage was not handed the bound model service");
    }

    #[tokio::test]
    #[serial]
    async fn test_503_when_workspace_is_not_initialized() {
        let model_service = ModelServiceFixture::new();
        model_service.patch_usage(|| Err(StorageError::RuntimeError("Workspace not initialized".into()).into()));
        let resp = get(&client(), "/v1/storage/usage").await;

        assert_eq!(resp.status, 503);
        assert_eq!(resp.json()["detail"], json!("Workspace not initialized"));
    }

    #[tokio::test]
    #[serial]
    async fn test_500_on_unexpected_failure() {
        let model_service = ModelServiceFixture::new();
        model_service.patch_usage(|| Err(std::io::Error::other("disk gone").into()));
        let resp = get(&client(), "/v1/storage/usage").await;

        assert_eq!(resp.status, 500);
    }
}

mod test_cleanup_storage {
    use super::*;

    #[tokio::test]
    #[serial]
    async fn test_defaults_to_cleaning_nothing() {
        let model_service = ModelServiceFixture::new();
        let cleanup = model_service.patch_cleanup(|| Ok(row(json!({"freed": 0}))));
        let resp = post_json(&client(), "/v1/storage/cleanup", &json!({})).await;

        assert_eq!(resp.status, 200);
        let calls = cleanup.lock();
        let (_, kwargs) = calls.last().expect("run_cleanup was called");
        assert!(![
            kwargs.models, kwargs.versions, kwargs.logs,
            kwargs.cache, kwargs.orphan_projects, kwargs.dry_run,
        ]
        .iter()
        .any(|flag| *flag));
        assert_eq!(kwargs.model_ids, None);
    }

    #[tokio::test]
    #[serial]
    async fn test_forwards_every_flag() {
        let model_service = ModelServiceFixture::new();
        let body = json!({
            "models": true, "model_ids": ["google/gemma-3-4b-it"], "versions": true,
            "logs": true, "cache": true, "orphan_projects": true, "dry_run": true,
        });
        let cleanup = model_service.patch_cleanup(|| Ok(row(json!({"freed": 0}))));
        let resp = post_json(&client(), "/v1/storage/cleanup", &body).await;

        assert_eq!(resp.status, 200);
        let calls = cleanup.lock();
        let (same_service, kwargs) = calls.last().expect("run_cleanup was called");
        let forwarded = json!({
            "models": kwargs.models, "model_ids": kwargs.model_ids, "versions": kwargs.versions,
            "logs": kwargs.logs, "cache": kwargs.cache, "orphan_projects": kwargs.orphan_projects,
            "dry_run": kwargs.dry_run,
        });
        for (key, value) in body.as_object().unwrap() {
            assert_eq!(&forwarded[key], value, "{key}");
        }
        assert!(*same_service, "run_cleanup was not handed the bound model service");
    }

    #[tokio::test]
    #[serial]
    async fn test_returns_the_cleanup_report() {
        let model_service = ModelServiceFixture::new();
        let report = json!({
            "dry_run": false, "freed": 512, "freed_by_category": {"logs": 512},
            "deleted": [{"category": "logs", "path": "/home/.smartloop/server.log.1", "bytes": 512}],
            "failed": [], "models_marked": [], "restart_required": false,
            "usage": Value::Object(sample_usage()),
        });
        let answer = report.clone();
        model_service.patch_cleanup(move || Ok(row(answer.clone())));
        let resp = post_json(&client(), "/v1/storage/cleanup", &json!({"logs": true})).await;

        assert_eq!(resp.status, 200);
        assert_eq!(resp.json(), report);
    }

    #[tokio::test]
    #[serial]
    async fn test_400_when_a_model_id_is_not_removable() {
        let model_service = ModelServiceFixture::new();
        model_service.patch_cleanup(|| {
            Err(StorageError::ValueError("Not removable: google/gemma-4-e4b-it".into()).into())
        });
        let resp = post_json(
            &client(),
            "/v1/storage/cleanup",
            &json!({"models": true, "model_ids": ["google/gemma-4-e4b-it"]}),
        )
        .await;

        assert_eq!(resp.status, 400);
        assert!(resp.json()["detail"].as_str().unwrap_or_default().contains("Not removable"));
    }

    #[tokio::test]
    #[serial]
    async fn test_503_when_the_model_is_busy() {
        let model_service = ModelServiceFixture::new();
        model_service.patch_cleanup(|| Err(StorageError::RuntimeError("Model is busy generating".into()).into()));
        let resp = post_json(&client(), "/v1/storage/cleanup", &json!({"models": true})).await;

        assert_eq!(resp.status, 503);
    }

    #[tokio::test]
    #[serial]
    async fn test_422_on_a_malformed_body() {
        let _model_service = ModelServiceFixture::new();
        let resp = post_json(&client(), "/v1/storage/cleanup", &json!({"models": "yes please"})).await;

        assert_eq!(resp.status, 422);
    }
}
