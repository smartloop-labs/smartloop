//! Server lifecycle management for Smartloop.
//!
//! Port of `smartloop/server.py`.  Provides [`start_server`], [`stop_server`]
//! and [`get_status`] as importable functions.

use std::path::PathBuf;

use once_cell::sync::Lazy;
use serde_json::Value;

#[allow(unused_imports)]
use crate::api::start_server_foreground;
use crate::config::AppSettings;
#[allow(unused_imports)]
use crate::supervisor::registry::process_alive;

/// `<home_dir>/server.pid`, resolved once from [`AppSettings`].
pub static PID_FILE: Lazy<PathBuf> =
    Lazy::new(|| PathBuf::from(AppSettings::new().home_dir()).join("server.pid"));

/// Get the base URL for the API server.
pub fn get_server_url(host: &str, port: u16) -> String {
    format!("http://{host}:{port}")
}

/// Check if the API server is already running (`GET /health` == 200, 2s timeout).
pub fn is_server_running(host: &str, port: u16) -> bool {
    let _ = (host, port);
    todo!()
}

/// Read the PID from the pid file if it exists and the process is alive;
/// a stale file is removed.
pub fn read_pid_file() -> Option<u32> {
    todo!()
}

/// Write `pid` to the pid file, creating its directory.
pub fn write_pid_file(pid: u32) -> std::io::Result<()> {
    let _ = pid;
    todo!()
}

/// Stop the background API server if running (terminate, wait 10s, then
/// kill).  Returns true if the server was stopped.
pub fn stop_server() -> bool {
    todo!()
}

/// Return server status as a JSON object.
///
/// Always has `"running"`; when running also `host`, `port`, `pid`,
/// `model_loaded`, `model_name`, optional model stats, `active_project`, and
/// under a supervisor `supervised`, `supervisor`, `model_host`, `agents`.
pub fn get_status(host: &str, port: u16) -> Value {
    let _ = (host, port);
    todo!()
}

/// `{project_id: name}`, so the status table reads in project names.
fn project_names(host: &str, port: u16) -> serde_json::Map<String, Value> {
    let _ = (host, port);
    todo!()
}

/// Start the API server in the foreground (blocking).
///
/// Sets `SLP_SUPERVISOR` to `"1"`/`"0"` from `supervised`, then runs the
/// server.  `debug` is accepted for signature compatibility and unused, as in
/// Python.
pub fn start_server(
    host: &str,
    port: u16,
    debug: bool,
    service: bool,
    supervised: bool,
) -> anyhow::Result<()> {
    let _ = (host, port, debug, service, supervised);
    todo!()
}
