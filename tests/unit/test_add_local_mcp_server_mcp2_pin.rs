//! Port of `tests/unit/test_add_local_mcp_server_mcp2_pin.py`.
//!
//! Auto-fix for MCP SDK 2.x breakage in local MCP server discovery.
//!
//! `mcp` 2.0.0 removed `mcp.server.fastmcp`, so legacy PyPI servers that
//! declare an unbounded `mcp>=1.x` dependency die at startup with
//! `ModuleNotFoundError: No module named 'mcp.server.fastmcp'`.  The discovery
//! retry must inject `--with mcp<2` into the uv invocation and persist the
//! pinned args into the server config so runtime tool calls use it too.
//!
//! The Python tests patched the `LocalMCPClient` class; here the
//! `local_mcp_discovery` seam stands in for the launch, scripted with the
//! same sequence of outcomes, and records the args of every launch.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::json;
use serial_test::serial;

use smartloop::api::config::api_settings;
use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::mcp::client::{MCPClientError, MCPConnectResult, MCPDiscoveryError, MCPLaunchError};
use smartloop::workspace::{MCPServer, MCPToolDefinition, Project, Workspace};

use crate::shared::model_service_a::{ScratchConfig, attach_workspace, bare_service};

const MCP2_TRACEBACK: &str = "MCP process exited (code 1). stderr: Traceback (most recent call last):\n  File \"/Users/x/.cache/uv/archive-v0/xxx/bin/weather-mcp-pypi\", line 6, in <module>\n    from weather_mcp_pypi import main\n  File \".../weather_mcp_pypi/__init__.py\", line 2, in <module>\n    from mcp.server.fastmcp import FastMCP\nModuleNotFoundError: No module named 'mcp.server.fastmcp'";

const MCP2_MCPERROR_TRACEBACK: &str = "MCP process exited (code 1). stderr: port *\n  File \".../mcp_weather_server/tools/tools_time.py\", line 12, in <module>\n    from .. import utils\n  File \".../mcp_weather_server/utils.py\", line 7, in <module>\n    from mcp import McpError\nImportError: cannot import name 'McpError' from 'mcp' (...)";

const UVX_HINT_EXEC_MISMATCH: &str = "MCP process exited (code 1). stderr: An executable named `mcp-weather-server` is not provided by package `mcp-weather-server`.\nThe following executables are available:\n- mcp_weather_server\n\nUse `uv tool run --from mcp-weather-server mcp_weather_server` instead.";

const MCP_TYPES_MISSING: &str = "MCP process exited (code 1). stderr: Traceback (most recent call last):\n    from mcp_types import Something\nModuleNotFoundError: No module named 'mcp_types'";

/// One scripted launch outcome.
#[derive(Clone)]
enum Outcome {
    Launch(&'static str),
    Ok,
}

fn ok_result() -> MCPConnectResult {
    MCPConnectResult {
        success: true,
        tools: vec![MCPToolDefinition {
            name: "get_weather".to_string(),
            description: Some("d".to_string()),
            input_schema: json!({"type": "object"}).as_object().cloned(),
        }],
        ..Default::default()
    }
}

/// The args of every launch, in order (`mock_cls.call_args_list[i].kwargs["args"]`).
type Launches = Arc<Mutex<Vec<Vec<String>>>>;

/// A service whose workspace holds project `p1`, with registration's queueing
/// stubbed (the discovery step under test is run explicitly) and the launch
/// scripted by `outcomes`.
fn svc(outcomes: Vec<Outcome>) -> (Arc<ModelService>, Launches, ScratchConfig) {
    let config = ScratchConfig::new();
    let svc = bare_service();
    let mut workspace = Workspace::new("ws");
    workspace.projects = vec![Project {
        id: "p1".to_string(),
        project_dir: config.project_dir("p1"),
        current: true,
        ..Default::default()
    }];
    attach_workspace(&svc, workspace);
    *svc.model_name.write() = Some(api_settings().default_model);

    let launches: Launches = Arc::default();
    let record = launches.clone();
    let script = Arc::new(Mutex::new(VecDeque::from(outcomes)));
    svc.set_hooks(ModelServiceHooks {
        save_model: Some(Arc::new(|| Ok(()))),
        queue_mcp_provisioning: Some(Arc::new(|_, _| {})),
        local_mcp_discovery: Some(Arc::new(
            move |_command: &str, args: &[String], _env: &HashMap<String, String>, _cwd: Option<&str>| {
                record.lock().push(args.to_vec());
                match script.lock().pop_front() {
                    Some(Outcome::Launch(text)) => Err(MCPClientError::Launch(MCPLaunchError(text.to_string()))),
                    Some(Outcome::Ok) => Ok(ok_result()),
                    None => panic!("more launches than scripted"),
                }
            },
        )),
        ..Default::default()
    });
    (svc, launches, config)
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

/// Register a local server, then run the discovery pass.
///
/// Registration is non-blocking: it normalises the launch config, persists the
/// connection and queues provisioning.  The launch retries under test live in
/// the discovery step the MCP task runs, so both halves are exercised here the
/// way they run in sequence in production.
fn register_and_discover(svc: &ModelService, command: &str, args: &[&str]) -> (MCPServer, anyhow::Result<()>) {
    svc.add_local_mcp_server(command, Some(strings(args)), None, None, None, None, None, None, Some("p1"))
        .expect("registration");
    let mut server = {
        let ws = svc.workspace_handle().expect("workspace");
        let guard = ws.read();
        guard.project("p1").expect("p1").mcp_servers.last().cloned().expect("registered server")
    };
    let result = svc.discover_local_mcp_tools(&mut server);
    (server, result)
}

fn is_discovery_error(result: &anyhow::Result<()>) -> bool {
    matches!(result, Err(e) if e.downcast_ref::<MCPDiscoveryError>().is_some())
}

#[test]
#[serial]
fn test_mcp2_breakage_retries_with_mcp_pin() {
    let (svc, launches, _config) = svc(vec![Outcome::Launch(MCP2_TRACEBACK), Outcome::Ok]);

    let (server, result) =
        register_and_discover(&svc, "uv", &["tool", "run", "--from", "weather-mcp-pypi", "weather-mcp-pypi"]);
    result.expect("discovery succeeds on the pinned retry");

    let launches = launches.lock();
    assert_eq!(launches.len(), 2);
    let retry_args = &launches[1];
    assert_eq!(
        retry_args,
        &strings(&["tool", "run", "--with", "mcp<2", "--from", "weather-mcp-pypi", "weather-mcp-pypi"])
    );
    assert_eq!(&server.config.expect("config").args, retry_args);
}

#[test]
#[serial]
fn test_no_pin_when_with_flag_already_present() {
    let (svc, launches, _config) = svc(vec![Outcome::Launch(MCP2_TRACEBACK), Outcome::Ok]);

    let (_server, result) =
        register_and_discover(&svc, "uv", &["tool", "run", "--from", "pkg", "--with", "mcp>=1", "pkg"]);

    assert!(is_discovery_error(&result));
    assert_eq!(launches.lock().len(), 1);
}

#[test]
#[serial]
fn test_non_uv_launcher_not_pinned() {
    let (svc, launches, _config) = svc(vec![Outcome::Launch(MCP2_TRACEBACK)]);

    let (_server, result) = register_and_discover(&svc, "python", &["-m", "my_server"]);

    assert!(is_discovery_error(&result));
    assert_eq!(launches.lock().len(), 1);
}

#[test]
#[serial]
fn test_mcperror_import_breakage_retries_with_mcp_pin() {
    let (svc, launches, _config) = svc(vec![Outcome::Launch(MCP2_MCPERROR_TRACEBACK), Outcome::Ok]);

    let (server, result) =
        register_and_discover(&svc, "uv", &["tool", "run", "--from", "mcp-weather-server", "mcp-weather-server"]);
    result.expect("discovery succeeds on the pinned retry");

    let launches = launches.lock();
    assert_eq!(launches.len(), 2);
    let retry_args = &launches[1];
    assert_eq!(
        retry_args,
        &strings(&["tool", "run", "--with", "mcp<2", "--from", "mcp-weather-server", "mcp-weather-server"])
    );
    assert_eq!(&server.config.expect("config").args, retry_args);
}

#[test]
#[serial]
fn test_mcp_types_missing_not_pinned() {
    let (svc, launches, _config) = svc(vec![Outcome::Launch(MCP_TYPES_MISSING)]);

    let (_server, result) = register_and_discover(&svc, "uv", &["tool", "run", "--from", "pkg", "pkg"]);

    assert!(is_discovery_error(&result));
    assert_eq!(launches.lock().len(), 1);
}

#[test]
#[serial]
fn test_hint_fix_then_mcperror_retries_with_pin() {
    // The app registers the bare exec name, so the *first* failure is uv's
    // "Use `uv tool run --from <pkg> <exec>` instead" hint.  The hint fix
    // swaps in the right executable, but that retry then dies importing
    // McpError — the pin must be evaluated on the retry error too.
    let (svc, launches, _config) = svc(vec![
        Outcome::Launch(UVX_HINT_EXEC_MISMATCH),
        Outcome::Launch(MCP2_MCPERROR_TRACEBACK),
        Outcome::Ok,
    ]);

    let (server, result) = register_and_discover(&svc, "uv", &["tool", "run", "mcp-weather-server"]);
    result.expect("discovery succeeds on the third launch");

    let calls = launches.lock();
    assert_eq!(calls.len(), 3);
    assert_eq!(calls[1], strings(&["tool", "run", "--from", "mcp-weather-server", "mcp_weather_server"]));
    assert_eq!(
        calls[2],
        strings(&["tool", "run", "--with", "mcp<2", "--from", "mcp-weather-server", "mcp_weather_server"])
    );
    assert_eq!(server.config.as_ref().expect("config").args, calls[2]);
    assert_eq!(server.name, "mcp-weather-server");
}

/// The other shape of the 2.x break, and the one the import patterns miss.
///
/// 2.x kept the low-level `Server` class and removed the decorators every 1.x
/// server hangs its handlers off, so a package imports cleanly and then dies
/// registering its tools.  The traceback names no module at all.
#[test]
fn test_removed_decorator_api_is_recognised_as_mcp2_breakage() {
    assert!(ModelService::mcp2_incompat_error(
        "AttributeError: 'Server' object has no attribute 'list_tools'"
    ));
    assert!(ModelService::mcp2_incompat_error(
        "AttributeError: 'FastMCP' object has no attribute 'call_tool'"
    ));
}

#[test]
fn test_an_unrelated_attribute_error_is_not_mcp2_breakage() {
    // Pinning mcp<2 for every AttributeError would hide real bugs behind a
    // dependency downgrade that has nothing to do with them.
    assert!(!ModelService::mcp2_incompat_error(
        "AttributeError: 'Server' object has no attribute 'frobnicate'"
    ));
    assert!(!ModelService::mcp2_incompat_error(
        "AttributeError: 'Widget' object has no attribute 'list_tools'"
    ));
}
