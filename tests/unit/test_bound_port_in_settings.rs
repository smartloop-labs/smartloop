//! Port of `tests/unit/test_bound_port_in_settings.py`.
//!
//! An agent started on port 0 reports the port it actually bound.
//!
//! The launcher asks for a free port with `SLP_API_PORT=0`.  The settings kept
//! saying 0 after the bind, and the MCP OAuth redirect is built from them, so a
//! provider sent the browser back to `localhost:0` and the sign-in never
//! returned.

use std::sync::Arc;

use serial_test::serial;

use smartloop::api::config::API_SETTINGS;
use smartloop::api::main::adopt_bound_port_with;
use smartloop::api::routers::mcp_servers::default_callback_url;
use smartloop::supervisor::service::Supervisor;

/// `_restore_port`: `SLP_API_PORT=0` and `api_settings.api_port = 0` for the
/// test, both restored afterwards.
struct RestorePort {
    env: Option<String>,
    port: u16,
}

impl RestorePort {
    fn new() -> Self {
        let env = std::env::var("SLP_API_PORT").ok();
        // SAFETY: these tests run #[serial].
        unsafe { std::env::set_var("SLP_API_PORT", "0") };
        let port = {
            let mut settings = API_SETTINGS.write();
            std::mem::replace(&mut settings.api_port, 0)
        };
        RestorePort { env, port }
    }
}

impl Drop for RestorePort {
    fn drop(&mut self) {
        // SAFETY: see `new`.
        unsafe {
            match &self.env {
                Some(value) => std::env::set_var("SLP_API_PORT", value),
                None => std::env::remove_var("SLP_API_PORT"),
            }
        }
        API_SETTINGS.write().api_port = self.port;
    }
}

#[test]
#[serial]
fn test_the_redirect_names_the_bound_port() {
    let _port = RestorePort::new();
    adopt_bound_port_with(52575, None);
    assert_eq!(
        default_callback_url("p1"),
        "http://localhost:52575/v1/projects/p1/mcp/oauth/callback"
    );
}

/// A child builds the redirect from the environment it was spawned with.
#[test]
#[serial]
fn test_children_inherit_the_bound_port() {
    let _port = RestorePort::new();
    adopt_bound_port_with(52575, None);
    assert_eq!(std::env::var("SLP_API_PORT").as_deref(), Ok("52575"));
    assert_eq!(API_SETTINGS.read().api_port, 52575);
}

/// The supervisor starts during lifespan startup, before the bind.
#[test]
#[serial]
fn test_the_supervisor_record_names_the_bound_port() {
    let _port = RestorePort::new();
    let home = tempfile::tempdir().unwrap();
    let supervisor = Arc::new(Supervisor::new(Some(home.path().to_string_lossy().into_owned()), None, None));

    adopt_bound_port_with(52575, Some(Arc::clone(&supervisor)));

    let record = supervisor.registry.supervisor();
    assert_eq!(record.get("pid").and_then(|v| v.as_i64()), Some(i64::from(std::process::id())));
    assert_eq!(record.get("port").and_then(|v| v.as_u64()), Some(52575));
}
