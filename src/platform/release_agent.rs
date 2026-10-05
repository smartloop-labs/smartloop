//! Port of `smartloop/platform/release_agent.py`.
//!
//! Background agent that keeps the local release cache warm.
//!
//! Periodically polls the SLP cloud API for the latest published release on
//! each configured channel and stores it in the workspace DB, so the Electron
//! client sees update notifications promptly even when otherwise idle.  Runs as
//! a tokio task started from the API server lifespan.

#[allow(unused_imports)]
use crate::config::AppSettings;
#[allow(unused_imports)]
use crate::platform::get_platform_service;

/// How often the release cache is refreshed (1 hour).
pub const RELEASE_POLL_INTERVAL_SECONDS: u64 = 60 * 60;

/// Refresh the cached latest release for every configured channel on an interval.
///
/// Each poll runs on the blocking pool (sync HTTP + SQLite).  Connection /
/// permission failures log a warning and retry next cycle; never returns.
pub async fn run_release_poll_loop() {
    todo!()
}
