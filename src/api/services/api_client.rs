//! Port of `smartloop/api/services/api_client.py`.
//!
//! Smartloop API client for token usage telemetry.

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{Map, Value};

use crate::config::app_settings::AppSettings;
use crate::platform::auth as platform_auth;
use crate::platform::{PlatformApi, SLPPlatformClient};

use super::{pyjson, utc_now_isoformat};

/// Logger name used by the Python module.
const LOG_TARGET: &str = "smartloop.api.services.api_client";

/// Timeout (seconds) of a telemetry POST (`push_token_usage`'s default).
const PUSH_TIMEOUT_S: u64 = 10;

/// `<home_dir>` (Python `_SLP_DIR`), resolved from fresh settings.
fn slp_dir() -> PathBuf {
    PathBuf::from(AppSettings::new().home_dir())
}

/// `<home_dir>/token_usage.jsonl` (Python `_TOKEN_USAGE_FILE`).
fn token_usage_file() -> PathBuf {
    slp_dir().join("token_usage.jsonl")
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
        let settings = settings.unwrap_or_else(AppSettings::new);
        let platform_client: Arc<dyn PlatformApi> = Arc::new(SLPPlatformClient::new(&settings.slp_api_base));
        Self { settings, platform_client, pending_usage: Arc::new(Mutex::new(Vec::new())) }
    }

    /// Build a client with an injected platform client (tests substitute fakes).
    pub fn with_platform_client(settings: Option<AppSettings>, platform_client: Arc<dyn PlatformApi>) -> Self {
        let settings = settings.unwrap_or_else(AppSettings::new);
        Self { settings, platform_client, pending_usage: Arc::new(Mutex::new(Vec::new())) }
    }

    /// The settings this client was built with.
    pub fn settings(&self) -> &AppSettings {
        &self.settings
    }

    /// True when platform credentials are available.
    pub fn enabled(&self) -> bool {
        platform_auth::has_credentials()
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
        let mut record = Map::new();
        record.insert("timestamp".into(), Value::String(utc_now_isoformat()));
        record.insert(
            "project_id".into(),
            project_id.map(|p| Value::String(p.to_string())).unwrap_or(Value::Null),
        );
        record.insert("model".into(), Value::String(model.to_string()));
        record.insert("prompt_tokens".into(), Value::from(prompt_tokens));
        record.insert("completion_tokens".into(), Value::from(completion_tokens));
        record.insert("total_tokens".into(), Value::from(prompt_tokens + completion_tokens));

        // Always persist locally
        let line = pyjson::dumps(&Value::Object(record.clone())) + "\n";
        let written = std::fs::create_dir_all(slp_dir()).and_then(|_| {
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(token_usage_file())?;
            f.write_all(line.as_bytes())
        });
        if let Err(e) = written {
            tracing::warn!(target: LOG_TARGET, "Failed to write local token usage: {}", e);
        }

        self.pending_usage.lock().push(record);
    }

    /// Push all pending token usage records to the platform API.
    ///
    /// Called from the health check endpoint so pushes happen on the natural
    /// polling cycle rather than inline during chat completion.  The posts run
    /// on a detached background thread.  When disabled, records are re-queued.
    pub fn flush_token_usage(&self) {
        let records: Vec<Map<String, Value>> = {
            let mut pending = self.pending_usage.lock();
            std::mem::take(&mut *pending)
        };

        if records.is_empty() {
            return;
        }

        if !self.enabled() {
            self.pending_usage.lock().extend(records);
            return;
        }

        tracing::info!(target: LOG_TARGET, "Flushing {} pending token usage record(s)", records.len());

        let platform_client = Arc::clone(&self.platform_client);
        let post = move || {
            for record in &records {
                // The platform client swallows (and logs) its own failures;
                // a panic in it must not take the rest of the batch down.
                let pushed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    platform_client.push_token_usage(record, PUSH_TIMEOUT_S)
                }));
                match pushed {
                    Ok(()) => {
                        let total = record.get("total_tokens").cloned().unwrap_or(Value::Null);
                        let project = match record.get("project_id") {
                            Some(Value::String(p)) => p.clone(),
                            _ => "None".to_string(),
                        };
                        tracing::info!(
                            target: LOG_TARGET,
                            "Token usage pushed ({} tokens, project={})",
                            total,
                            project
                        );
                    }
                    Err(_) => {
                        tracing::debug!(target: LOG_TARGET, "Token usage push failed (non-fatal): panicked");
                    }
                }
            }
        };

        if let Err(e) = std::thread::Builder::new().name("token-usage-flush".into()).spawn(post) {
            tracing::debug!(target: LOG_TARGET, "Token usage push failed (non-fatal): {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isoformat_has_utc_offset() {
        let s = utc_now_isoformat();
        assert!(s.ends_with("+00:00"), "{s}");
        assert!(s.contains('T'));
    }
}
