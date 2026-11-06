//! Port of `smartloop/api/mcp/__init__.py`.
//!
//! In-process MCP servers exposed to the orchestrator.

pub mod skill_server;
pub mod system_server;

use crate::api::config::API_SETTINGS;

/// Return the base URL of the running API server.
///
/// `0.0.0.0` isn't reachable from the client side, so it is replaced by
/// `127.0.0.1`.
pub fn get_api_base_url() -> String {
    let settings = API_SETTINGS.read();
    let port = settings.api_port;
    let mut host = settings.api_host.clone();
    // 0.0.0.0 isn't reachable from the client side — use localhost instead
    if host == "0.0.0.0" {
        host = "127.0.0.1".to_string();
    }
    format!("http://{host}:{port}")
}
