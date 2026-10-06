//! Port of `smartloop/platform/__init__.py`.

pub mod auth;
pub mod client;
pub mod release_agent;
pub mod semver;
pub mod service;
pub mod token_agent;
pub mod workspace_db;

use std::path::PathBuf;
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::Mutex;

pub use client::{ModelAccessError, PlatformApi, PlatformError, SLPPlatformClient};
pub use service::PlatformService;
pub use workspace_db::WorkspaceDB;

use crate::config::app_settings::AppSettings;
use crate::workspace::workspace::Workspace;

/// The singleton and the DB path it is bound to (`None` = pure proxy).
struct PlatformServiceSlot {
    service: Option<Arc<PlatformService>>,
    db_path: Option<PathBuf>,
}

static PLATFORM_SERVICE: Lazy<Mutex<PlatformServiceSlot>> =
    Lazy::new(|| Mutex::new(PlatformServiceSlot { service: None, db_path: None }));

/// Test seam replacing `get_platform_service` (Python tests monkeypatch it).
static PLATFORM_SERVICE_OVERRIDE: Lazy<Mutex<Option<Arc<PlatformService>>>> =
    Lazy::new(|| Mutex::new(None));

/// Make [`get_platform_service`] return *service* (or, with `None`, restore
/// normal behaviour).
pub fn set_platform_service_override(service: Option<Arc<PlatformService>>) {
    *PLATFORM_SERVICE_OVERRIDE.lock() = service;
}

/// Return the global `PlatformService` singleton (lazy-initialized).
///
/// The DB lives inside the workspace's per-UUID directory at
/// `<home_dir>/<workspace_id>/workspace.db`.  Before a workspace exists on disk
/// the service runs as a pure proxy without caching; the singleton rebinds to
/// the workspace-scoped DB on the next call once a workspace has been created.
/// Creation is guarded by a lock so concurrent callers don't open multiple
/// SQLite connections to the same file.
pub fn get_platform_service() -> Arc<PlatformService> {
    if let Some(service) = PLATFORM_SERVICE_OVERRIDE.lock().clone() {
        return service;
    }

    let settings = AppSettings::new();
    let ws = Workspace::load();
    let db_path = ws
        .as_ref()
        .filter(|ws| !ws.workspace_dir.is_empty())
        .map(|ws| PathBuf::from(&ws.workspace_dir).join("workspace.db"));

    // The slot's mutex covers both the fast path (already bound to the right
    // DB) and creation, so concurrent callers never open two SQLite
    // connections to the same file.
    let mut slot = PLATFORM_SERVICE.lock();
    if let Some(service) = &slot.service {
        if slot.db_path == db_path {
            return service.clone();
        }
    }

    let client = SLPPlatformClient::new(&settings.slp_api_base);
    let mut bound_path = db_path.clone();
    let db = match &db_path {
        Some(path) => match WorkspaceDB::new(path) {
            Ok(db) => Some(Arc::new(db)),
            Err(e) => {
                // Serve as a pure proxy rather than failing the request, and
                // leave the slot unbound so the next call tries the DB again.
                tracing::warn!(
                    target: "smartloop.platform",
                    "Could not open workspace DB at {}: {:#}",
                    path.display(),
                    e
                );
                bound_path = None;
                None
            }
        },
        None => None,
    };
    if let (Some(db), Some(ws)) = (&db, &ws) {
        if let Err(e) = db.save_workspace(&ws.id) {
            tracing::warn!(target: "smartloop.platform", "Failed to record workspace {}: {:#}", ws.id, e);
        }
    }
    let service = Arc::new(PlatformService::new(Arc::new(client), db));
    slot.service = Some(service.clone());
    slot.db_path = bound_path;
    service
}
