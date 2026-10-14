//! Port of `smartloop/api/routers/storage.py`.
//!
//! Storage usage and cleanup router.
//!
//! Reports what the SLP home directory is spending disk on and reclaims the parts
//! that are safe to lose.  The desktop app used to do this itself in Electron JS;
//! these endpoints exist so the framework — which owns the workspace record and
//! the loaded models — makes those decisions instead.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use axum::routing::{get, post};
use axum::{Json, Router};
use once_cell::sync::Lazy;
use parking_lot::{Condvar, Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::services::storage_service::{
    self, CleanupOptions, InferencePermits, Row, StorageModelService,
};
use crate::api::services::{get_model_service, ModelService};
use crate::workspace::workspace::SharedWorkspace;

use super::_errors::{py_error_kind, run_blocking, HTTPException, PyErrorKind, ValidatedJson};

/// Logger name used by this router.
pub const LOGGER: &str = "smartloop.api.routers.storage";

/// Categories to reclaim.  Nothing is deleted unless asked for explicitly.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct CleanupRequest {
    /// Delete downloaded models that aren't in use.
    #[serde(default)]
    pub models: bool,
    /// Restrict the model sweep to these ids; omit for every removable model.
    #[serde(default)]
    pub model_ids: Option<Vec<String>>,
    /// Delete framework installs other than the running one.
    #[serde(default)]
    pub versions: bool,
    /// Empty server.log and delete its rotated backups.
    #[serde(default)]
    pub logs: bool,
    /// Empty the regenerable cache directories.
    #[serde(default)]
    pub cache: bool,
    /// Delete project directories the workspace no longer references.
    #[serde(default)]
    pub orphan_projects: bool,
    /// Report what would be removed without removing it.
    #[serde(default)]
    pub dry_run: bool,
}

impl From<CleanupRequest> for CleanupOptions {
    fn from(request: CleanupRequest) -> Self {
        CleanupOptions {
            models: request.models,
            model_ids: request.model_ids,
            versions: request.versions,
            logs: request.logs,
            cache: request.cache,
            orphan_projects: request.orphan_projects,
            dry_run: request.dry_run,
        }
    }
}

// ---------------------------------------------------------------------------
// The model service as the storage service sees it
// ---------------------------------------------------------------------------

/// [`ModelService`] seen through [`StorageModelService`], the slice of it the
/// storage service duck-typed in Python (`_ensure_workspace`,
/// `refresh_project_state`, the named model slots and `_cap_llm_*` caches,
/// `_task_queue`, `_inference_lock`, `_unload_model_by_name_unlocked`,
/// `_reset_llm_executor`).
pub struct ModelServiceStorage(pub Arc<ModelService>);

impl StorageModelService for ModelServiceStorage {
    fn ensure_workspace(&self) -> anyhow::Result<Option<SharedWorkspace>> {
        self.0.ensure_workspace(None).map(Some)
    }

    fn refresh_project_state(&self, project_id: &str) -> anyhow::Result<bool> {
        Ok(self.0.refresh_project_state(Some(project_id)))
    }

    fn resident_model_names(&self) -> HashSet<String> {
        let svc = &self.0;
        let mut names = HashSet::new();
        for slot in [
            &svc.model_name,
            &svc.decision_model_name,
            &svc.large_model_name,
            &svc.vision_model_name,
        ] {
            if let Some(name) = slot.read().as_ref() {
                if !name.is_empty() {
                    names.insert(name.clone());
                }
            }
        }
        // The faulted sentinel is a latch, not a live context, so it does not
        // pin the files.
        for (name, cached) in svc.cap_llms.lock().iter() {
            if !cached.is_gpu_faulted() {
                names.insert(name.clone());
            }
        }
        names
    }

    fn task_status(&self, name: &str) -> Option<Value> {
        let queue = self.0.task_queue.read().clone();
        match queue.get_status(Some(name)) {
            Value::Null => None,
            status => Some(status),
        }
    }

    fn inference_lock(&self) -> Option<Arc<dyn InferencePermits>> {
        Some(Arc::new(InferenceLockPermits { service: self.0.clone() }))
    }

    fn unload_model_by_name_unlocked(&self, name: &str) -> anyhow::Result<()> {
        self.0.unload_model_by_name_unlocked(name);
        Ok(())
    }

    fn reset_llm_executor(&self) {
        self.0.reset_llm_executor();
    }
}

/// `ModelService._inference_lock` with Python's `acquire(timeout=...)`.
///
/// The semaphore itself only blocks, so the wait happens on a helper thread
/// and the caller gives up after *timeout*.  A permit the helper obtains after
/// the caller has given up is handed straight back, so an abandoned wait
/// never leaks one.
struct InferenceLockPermits {
    service: Arc<ModelService>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AcquireState {
    Waiting,
    Acquired,
    Abandoned,
}

impl InferencePermits for InferenceLockPermits {
    fn acquire(&self, timeout: Duration) -> bool {
        let shared = Arc::new((Mutex::new(AcquireState::Waiting), Condvar::new()));
        let helper_shared = shared.clone();
        let service = self.service.clone();
        std::thread::spawn(move || {
            service.inference_lock.acquire();
            let (state, cond) = &*helper_shared;
            let mut state = state.lock();
            if *state == AcquireState::Abandoned {
                drop(state);
                service.inference_lock.release();
            } else {
                *state = AcquireState::Acquired;
                cond.notify_all();
            }
        });

        let (state, cond) = &*shared;
        let mut guard = state.lock();
        if *guard == AcquireState::Waiting {
            cond.wait_for(&mut guard, timeout);
        }
        if *guard == AcquireState::Acquired {
            true
        } else {
            *guard = AcquireState::Abandoned;
            false
        }
    }

    fn release(&self) {
        self.service.inference_lock.release();
    }
}

// ---------------------------------------------------------------------------
// Dependencies (overridable, as FastAPI's `dependency_overrides` / `patch`)
// ---------------------------------------------------------------------------

/// Replacement for `storage_service.compute_usage`.
pub type ComputeUsageFn =
    Arc<dyn Fn(Option<&dyn StorageModelService>) -> anyhow::Result<Row> + Send + Sync>;
/// Replacement for `storage_service.run_cleanup`.
pub type RunCleanupFn =
    Arc<dyn Fn(Option<&dyn StorageModelService>, CleanupOptions) -> anyhow::Result<Row> + Send + Sync>;

/// What this router calls, each `None` meaning the real thing: the model
/// service dependency and the two storage-service functions.
#[derive(Clone, Default)]
pub struct StorageRouteHooks {
    /// Stands in for `Depends(get_model_service)`.
    pub model_service: Option<Arc<dyn StorageModelService>>,
    pub compute_usage: Option<ComputeUsageFn>,
    pub run_cleanup: Option<RunCleanupFn>,
}

static HOOKS: Lazy<RwLock<StorageRouteHooks>> = Lazy::new(|| RwLock::new(StorageRouteHooks::default()));

/// Install (or with `StorageRouteHooks::default()`, clear) the router's hooks.
pub fn set_hooks(hooks: StorageRouteHooks) {
    *HOOKS.write() = hooks;
}

/// The model service dependency.
fn model_service() -> Arc<dyn StorageModelService> {
    if let Some(svc) = HOOKS.read().model_service.clone() {
        return svc;
    }
    Arc::new(ModelServiceStorage(get_model_service()))
}

fn compute_usage(model_service: &dyn StorageModelService) -> anyhow::Result<Row> {
    let hook = HOOKS.read().compute_usage.clone();
    match hook {
        Some(hook) => hook(Some(model_service)),
        None => storage_service::compute_usage(Some(model_service)).map_err(anyhow::Error::from),
    }
}

fn run_cleanup(model_service: &dyn StorageModelService, opts: CleanupOptions) -> anyhow::Result<Row> {
    let hook = HOOKS.read().run_cleanup.clone();
    match hook {
        Some(hook) => hook(Some(model_service), opts),
        None => storage_service::run_cleanup(Some(model_service), opts).map_err(anyhow::Error::from),
    }
}

/// `ValueError` → 400, `RuntimeError` → 503, anything else → 500 (logged).
fn storage_error(err: anyhow::Error, what: &str) -> HTTPException {
    match py_error_kind(&err) {
        PyErrorKind::Value => HTTPException::new(400, err.to_string()),
        PyErrorKind::Runtime => HTTPException::new(503, err.to_string()),
        _ => {
            tracing::error!(target: LOGGER, "{}: {:?}", what, err);
            HTTPException::new(500, err.to_string())
        }
    }
}

/// Routes under `/v1` (tag `storage`).
pub fn router() -> Router {
    Router::new()
        .route("/v1/storage/usage", get(get_storage_usage))
        .route("/v1/storage/cleanup", post(cleanup_storage))
}

/// `GET /v1/storage/usage` — disk usage under the SLP home directory, by category.
#[utoipa::path(
    get,
    path = "/v1/storage/usage",
    tag = "system",
    summary = "Get disk usage",
    description = "Disk usage under the Smartloop home directory, by category.",
    responses((status = 200, description = "OK", body = serde_json::Value), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn get_storage_usage() -> Result<Json<Value>, HTTPException> {
    let svc = model_service();
    let result = run_blocking(move || compute_usage(svc.as_ref())).await?;
    match result {
        Ok(row) => Ok(Json(Value::Object(row))),
        Err(e) => Err(storage_error(e, "Error computing storage usage")),
    }
}

/// `POST /v1/storage/cleanup` — reclaim disk space in the requested categories.
///
/// Returns what was removed, what failed, and a fresh usage report, so the
/// caller doesn't need a follow-up request to refresh its view.
#[utoipa::path(
    post,
    path = "/v1/storage/cleanup",
    tag = "system",
    request_body = CleanupRequest,
    summary = "Reclaim disk space",
    description = "",
    responses((status = 200, description = "OK", body = serde_json::Value), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn cleanup_storage(
    ValidatedJson(request): ValidatedJson<CleanupRequest>,
) -> Result<Json<Value>, HTTPException> {
    let svc = model_service();
    let opts = CleanupOptions::from(request);
    let result = run_blocking(move || run_cleanup(svc.as_ref(), opts)).await?;
    match result {
        Ok(row) => Ok(Json(Value::Object(row))),
        Err(e) => Err(storage_error(e, "Error cleaning up storage")),
    }
}
