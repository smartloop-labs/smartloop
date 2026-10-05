//! Port of `smartloop/api/services/api_client.py`.
//!
//! Smartloop API client for token usage telemetry.

use std::path::PathBuf;
use std::sync::Arc;

use chrono::Utc;
use parking_lot::Mutex;
use serde_json::{Map, Value};

use crate::config::app_settings::AppSettings;
use crate::platform::auth as platform_auth;
use crate::platform::{PlatformApi, SLPPlatformClient};

/// Logger name used by the Python module.
#[allow(dead_code)]
const LOG_TARGET: &str = "smartloop.api.services.api_client";

/// `<home_dir>` (Python `_SLP_DIR`), resolved from fresh settings.
fn slp_dir() -> PathBuf {
    todo!()
}

/// `<home_dir>/token_usage.jsonl` (Python `_TOKEN_USAGE_FILE`).
fn token_usage_file() -> PathBuf {
    todo!()
}

/// HTTP client for pushing telemetry to the Smartloop platform API.
///
/// Uses the developer token set during `slp init` for authentication.
/// Self-disables gracefully when no token is configured so the rest of the
/// application continues to work without an API connection.
///
/// Every public method swallows all errors — the client never crashes the
/// host process.
pub struct SLPAPIClient {
    settings: AppSettings,
    platform_client: Arc<dyn PlatformApi>,
    pending_usage: Arc<Mutex<Vec<Map<String, Value>>>>,
}

impl std::fmt::Debug for SLPAPIClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SLPAPIClient").finish_non_exhaustive()
    }
}

impl Default for SLPAPIClient {
    fn default() -> Self {
        Self::new(None)
    }
}

impl SLPAPIClient {
    /// Build a client from `settings` (fresh `AppSettings` when `None`),
    /// talking to `settings.slp_api_base`.
    pub fn new(settings: Option<AppSettings>) -> Self {
        let _ = (Utc::now(), SLPPlatformClient::new, platform_auth::has_credentials);
        let _ = settings;
        todo!()
    }

    /// Build a client with an injected platform client (tests substitute fakes).
    pub fn with_platform_client(settings: Option<AppSettings>, platform_client: Arc<dyn PlatformApi>) -> Self {
        let _ = (settings, platform_client);
        todo!()
    }

    /// The settings this client was built with.
    pub fn settings(&self) -> &AppSettings {
        &self.settings
    }

    /// True when platform credentials are available.
    pub fn enabled(&self) -> bool {
        todo!()
    }

    /// Snapshot of records queued for the next flush.
    pub fn pending_usage(&self) -> Vec<Map<String, Value>> {
        self.pending_usage.lock().clone()
    }

    /// Record token usage locally and queue it for the next health-cycle push.
    ///
    /// The local append to `<home>/token_usage.jsonl` always happens
    /// regardless of API availability.  The HTTP POST is deferred to
    /// [`flush_token_usage`](Self::flush_token_usage), which runs during the
    /// periodic health check.
    pub fn push_token_usage(
        &self,
        project_id: Option<&str>,
        model: &str,
        prompt_tokens: i64,
        completion_tokens: i64,
    ) {
        let _ = (project_id, model, prompt_tokens, completion_tokens, &self.platform_client);
        todo!()
    }

    /// Push all pending token usage records to the platform API.
    ///
    /// Called from the health check endpoint so pushes happen on the natural
    /// polling cycle rather than inline during chat completion.  The posts run
    /// on a detached background thread.  When disabled, records are re-queued.
    pub fn flush_token_usage(&self) {
        todo!()
    }
}
