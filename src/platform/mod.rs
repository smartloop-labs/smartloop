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

#[allow(unused_imports)]
use crate::config::app_settings::AppSettings;
#[allow(unused_imports)]
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
    let _ = (&*PLATFORM_SERVICE, &*PLATFORM_SERVICE_OVERRIDE);
    todo!()
}
