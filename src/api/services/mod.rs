//! Port of `smartloop/api/services/__init__.py`.
//!
//! API services: the model service, the platform telemetry client and the
//! storage/project-io helpers, plus the process-wide singletons.

use std::sync::Arc;

use once_cell::sync::OnceCell;

pub mod api_client;
pub mod model_service;
pub mod project_io;
pub mod storage_service;

pub use api_client::SLPAPIClient;
pub use model_service::ModelService;
pub use storage_service::{compute_usage, discard_partial_download, run_cleanup};

/// Global `ModelService` singleton (Python `_model_service`).
static MODEL_SERVICE: OnceCell<Arc<ModelService>> = OnceCell::new();

/// Global `SLPAPIClient` singleton (Python `_api_client`).
static API_CLIENT: OnceCell<Arc<SLPAPIClient>> = OnceCell::new();

/// Stand-in for FastAPI's `app.dependency_overrides[get_model_service]`: when
/// set, [`get_model_service`] returns this instead of the singleton.
static MODEL_SERVICE_OVERRIDE: once_cell::sync::Lazy<parking_lot::RwLock<Option<Arc<ModelService>>>> =
    once_cell::sync::Lazy::new(|| parking_lot::RwLock::new(None));

/// Make [`get_model_service`] return `service` (or, with `None`, restore the
/// singleton).  Tests only — the analogue of overriding the dependency.
pub fn set_model_service_override(service: Option<Arc<ModelService>>) {
    *MODEL_SERVICE_OVERRIDE.write() = service;
}

/// Get the global `ModelService` singleton, creating it on first use.
pub fn get_model_service() -> Arc<ModelService> {
    if let Some(service) = MODEL_SERVICE_OVERRIDE.read().clone() {
        return service;
    }
    MODEL_SERVICE.get_or_init(ModelService::instance).clone()
}

/// Get the global `SLPAPIClient` singleton, creating it on first use.
pub fn get_api_client() -> Arc<SLPAPIClient> {
    todo!()
}
