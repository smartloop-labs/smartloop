//! Port of `smartloop/api/config.py`.
//!
//! API Configuration – `APISettings` extends `AppSettings` with all fields unified.

use once_cell::sync::Lazy;
use parking_lot::RwLock;

use crate::config::app_settings::AppSettings;

/// Backward-compatible alias; all settings live in [`AppSettings`].
pub type APISettings = AppSettings;

/// The process-wide API settings (`api_settings` in Python).
///
/// Mutable because the server adopts the port it actually bound (see
/// [`crate::api::main::adopt_bound_port`]) and tests patch fields.
pub static API_SETTINGS: Lazy<RwLock<APISettings>> = Lazy::new(|| RwLock::new(APISettings::new()));

/// Convenience accessor returning a snapshot clone of [`API_SETTINGS`].
pub fn api_settings() -> APISettings {
    API_SETTINGS.read().clone()
}
