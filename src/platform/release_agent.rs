//! Port of `smartloop/platform/release_agent.py`.
//!
//! Background agent that keeps the local release cache warm.
//!
//! Periodically polls the SLP cloud API for the latest published release on
//! each configured channel and stores it in the workspace DB, so the Electron
//! client sees update notifications promptly even when otherwise idle.  Runs as
//! a tokio task started from the API server lifespan.

use std::time::Duration;

use crate::config::AppSettings;
use crate::platform::get_platform_service;

/// Python logger name for this module.
pub const LOGGER: &str = "smartloop.platform.release_agent";

/// How often the release cache is refreshed (1 hour).
pub const RELEASE_POLL_INTERVAL_SECONDS: u64 = 60 * 60;

/// Refresh the cached latest release for every configured channel on an interval.
///
/// Each poll runs on the blocking pool (sync HTTP + SQLite).  Connection /
/// permission failures log a warning and retry next cycle; never returns.
pub async fn run_release_poll_loop() {
    let settings = AppSettings::new();

    loop {
        for channel in settings.release_channels() {
            // get_latest_release does a synchronous HTTP call (up to a 30s
            // timeout) plus sync SQLite writes.  Running it inline would block
            // the runtime — and this local server also serves chat and health
            // — so push it to the blocking pool, mirroring the model catalog
            // seeding in the init router.
            let ch = channel.clone();
            let result =
                tokio::task::spawn_blocking(move || get_platform_service().get_latest_release(&ch, true)).await;
            match result {
                Ok(Ok(_)) => {
                    tracing::info!(
                        target: "smartloop.platform.release_agent",
                        "Refreshed release cache for channel '{}'",
                        channel
                    );
                }
                Ok(Err(err)) if err.is_connection() || err.is_permission() => {
                    tracing::warn!(
                        target: "smartloop.platform.release_agent",
                        "Release poll failed for channel '{}' — will retry next cycle",
                        channel
                    );
                }
                Ok(Err(err)) => {
                    tracing::warn!(
                        target: "smartloop.platform.release_agent",
                        "Unexpected error polling releases for channel '{}': {}",
                        channel,
                        err
                    );
                }
                Err(join_err) => {
                    tracing::warn!(
                        target: "smartloop.platform.release_agent",
                        "Unexpected error polling releases for channel '{}': {}",
                        channel,
                        join_err
                    );
                }
            }
        }

        tokio::time::sleep(Duration::from_secs(RELEASE_POLL_INTERVAL_SECONDS)).await;
    }
}
