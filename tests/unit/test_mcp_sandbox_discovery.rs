//! Port of `tests/unit/test_mcp_sandbox_discovery.py`.
//!
//! The discovery error a sandboxed build reports for a local MCP server.
//!
//! Inside the macOS App Sandbox the tilde is resolved correctly (the path in
//! the error is the right one) but the sandbox refuses to exec any program
//! outside the app bundle with EPERM — "[Errno 1] Operation not permitted:
//! '/path'".  That is not a misconfiguration a retry ladder can walk back, so
//! the friendly mapping must say what it is instead of handing an OS error to
//! the user to chase.  The Developer ID build (no sandbox in the environment)
//! keeps the raw first line.

use smartloop::api::services::model_service::ModelService;

use serial_test::serial;

use crate::shared::model_service_b::global_lock;

const CONTAINER_HOME: &str = "/Users/tester/Library/Containers/ai.smartloop.studio/Data";
const EPERM: &str = "[Errno 1] Operation not permitted: '/Users/mehfuzh/.local/bin/craft'";

/// Run `body` with `HOME` (and optionally `SLP_HOME`) set, restoring both.
/// The `_clean_sandbox_env` fixture: `SLP_HOME` starts unset.
fn with_env(home: &str, slp_home: Option<&str>, body: impl FnOnce()) {
    let _guard = global_lock();
    let saved_home = std::env::var("HOME").ok();
    let saved_slp_home = std::env::var("SLP_HOME").ok();
    // SAFETY: serialized by `global_lock`; restored before returning.
    unsafe {
        std::env::set_var("HOME", home);
        match slp_home {
            Some(value) => std::env::set_var("SLP_HOME", value),
            None => std::env::remove_var("SLP_HOME"),
        }
    }
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
    unsafe {
        match saved_home {
            Some(value) => std::env::set_var("HOME", value),
            None => std::env::remove_var("HOME"),
        }
        match saved_slp_home {
            Some(value) => std::env::set_var("SLP_HOME", value),
            None => std::env::remove_var("SLP_HOME"),
        }
    }
    if let Err(panic) = outcome {
        std::panic::resume_unwind(panic);
    }
}

#[test]
#[serial]
fn test_an_eperm_under_the_sandbox_names_the_build_limitation() {
    with_env(CONTAINER_HOME, None, || {
        let msg = ModelService::friendly_discovery_error(EPERM);
        assert!(msg.contains("App Sandbox"));
        assert!(msg.contains("build from smartloop.ai"));
        assert!(msg.contains("/Users/mehfuzh/.local/bin/craft"));
    });
}

#[test]
#[serial]
fn test_an_eperm_under_the_sandbox_named_by_slp_home() {
    let slp_home = format!("{CONTAINER_HOME}/Library/Application Support/smartloop-studio/slp");
    with_env("/Users/tester", Some(&slp_home), || {
        let msg = ModelService::friendly_discovery_error(EPERM);
        assert!(msg.contains("App Sandbox"));
    });
}

/// A Developer ID build has a real HOME, no container anywhere: whatever
/// denial this is, it is a genuine problem, not a sandbox one, so the mapping
/// must not dress it up.
#[test]
#[serial]
fn test_the_same_eperm_outside_the_sandbox_stays_raw() {
    with_env("/Users/tester", None, || {
        assert_eq!(ModelService::friendly_discovery_error(EPERM), EPERM);
    });
}

#[test]
#[serial]
fn test_unrelated_discovery_errors_are_unchanged_under_the_sandbox() {
    with_env(CONTAINER_HOME, None, || {
        assert_eq!(
            ModelService::friendly_discovery_error("ModuleNotFoundError: No module named 'mcp.server.fastmcp'"),
            "This package isn't compatible with the current MCP SDK — its \
             dependencies are out of date. Try another package, or ask its \
             maintainer to update it."
        );
        assert_eq!(ModelService::friendly_discovery_error(""), "Couldn't start this connection.");
    });
}
