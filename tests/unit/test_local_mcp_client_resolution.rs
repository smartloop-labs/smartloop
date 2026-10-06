//! Port of `tests/unit/test_local_mcp_client_resolution.py`.
//!
//! Which program a local MCP server's command actually resolves to.
//!
//! A definition can name a server the app ships with — a copy next to the
//! `slp` binary (the app bundle when packaged), or one a project keeps under
//! the framework home.  Resolution has to find it from a bare name or a path
//! relative to those anchors, and inside the macOS App Sandbox it has to
//! *prefer* it: only the bundle and the container may be exec'd, so the user's
//! own install at the same basename is refused with EPERM however the path was
//! written.
//!
//! Outside the sandbox an absolute command still wins outright — the app does
//! not silently redirect a path the user configured.

#![cfg(unix)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serial_test::serial;

use smartloop::mcp::local_client::LocalMCPClient;

use crate::shared::mcp::EnvGuard;

const CONTAINER_HOME: &str = "/Users/tester/Library/Containers/ai.smartloop.studio/Data";

fn exe(path: PathBuf) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, "#!/bin/sh\nexit 0\n").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

/// The fixtures: `tmp_path`, `slp_dir` (`tmp/slp-bin`) and `home`
/// (`tmp/slp-home`).
struct Fixture {
    tmp: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        Self { tmp: tempfile::tempdir().unwrap() }
    }
    fn path(&self) -> &Path {
        self.tmp.path()
    }
    fn slp_dir(&self) -> PathBuf {
        self.path().join("slp-bin")
    }
    fn home(&self) -> PathBuf {
        self.path().join("slp-home")
    }

    /// Resolution reads the running executable's directory as the slp
    /// binary's; pin it at a temp dir so the test never consults a real
    /// install (Python: `monkeypatch.setattr(sys, "executable", ...)`).
    /// `home` is the App Sandbox container or an ordinary home.
    fn environment(&self, sandboxed: bool) -> EnvGuard {
        let slp = self.slp_dir().join("slp").to_string_lossy().into_owned();
        let home = self.home().to_string_lossy().into_owned();
        EnvGuard::set(&[
            ("SLP_EXE", Some(&slp)),
            ("HOME", Some(if sandboxed { CONTAINER_HOME } else { "/Users/tester" })),
            ("SLP_HOME", Some(&home)),
            ("SNAP_NAME", None),
        ])
    }
}

fn empty_path() -> HashMap<String, String> {
    HashMap::from([("PATH".to_string(), String::new())])
}

fn s(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[test]
#[serial]
fn test_the_sandbox_prefers_the_copy_next_to_the_slp_binary() {
    let fx = Fixture::new();
    let _env = fx.environment(true);
    let bundled = exe(fx.slp_dir().join("craft"));
    let installed = exe(fx.path().join("user/.local/bin/craft"));
    let client = LocalMCPClient::new(s(&installed), vec!["mcp".into()]);
    assert_eq!(client.resolve_command(&empty_path()), s(&bundled));
}

#[test]
#[serial]
fn test_the_sandbox_falls_back_to_a_copy_under_the_framework_home() {
    let fx = Fixture::new();
    let _env = fx.environment(true);
    let in_home = exe(fx.home().join("craft"));
    let installed = exe(fx.path().join("user/.local/bin/craft"));
    let client = LocalMCPClient::new(s(&installed), vec!["mcp".into()]);
    assert_eq!(client.resolve_command(&empty_path()), s(&in_home));
}

#[test]
#[serial]
fn test_the_sandbox_keeps_the_configured_path_without_a_bundled_copy() {
    // Nothing to redirect to: the caller keeps the path, the spawn fails with
    // EPERM, and the discovery message explains the sandbox.  Silently running
    // *some other* `craft` would be worse than the honest failure.
    let fx = Fixture::new();
    let _env = fx.environment(true);
    let installed = exe(fx.path().join("user/.local/bin/craft"));
    let client = LocalMCPClient::new(s(&installed), vec!["mcp".into()]);
    assert_eq!(client.resolve_command(&empty_path()), s(&installed));
}

#[test]
#[serial]
fn test_the_developer_build_keeps_an_absolute_command() {
    // A bundled copy next to slp must not shadow the path a user configured
    // when the app is not sandboxed.
    let fx = Fixture::new();
    let _env = fx.environment(false);
    exe(fx.slp_dir().join("craft"));
    let installed = exe(fx.path().join("user/.local/bin/craft"));
    let client = LocalMCPClient::new(s(&installed), vec!["mcp".into()]);
    assert_eq!(client.resolve_command(&empty_path()), s(&installed));
}

#[test]
#[serial]
fn test_a_bundled_copy_is_found_from_a_bare_name() {
    let fx = Fixture::new();
    let _env = fx.environment(false);
    let bundled = exe(fx.slp_dir().join("craft"));
    let client = LocalMCPClient::new("craft", vec!["mcp".into()]);
    assert_eq!(client.resolve_command(&empty_path()), s(&bundled));
}

#[test]
#[serial]
fn test_a_path_relative_to_the_framework_home_is_resolved() {
    let fx = Fixture::new();
    let _env = fx.environment(false);
    let in_home = exe(fx.home().join("bin/craft"));
    let client = LocalMCPClient::new("bin/craft", vec!["mcp".into()]);
    assert_eq!(client.resolve_command(&empty_path()), s(&in_home));
}

#[test]
#[serial]
fn test_the_path_wins_over_the_anchors_for_a_bare_name() {
    // Anchors are the fallback, not an override: a `craft` the user put on
    // PATH is still the one that runs in a build that can launch it.
    let fx = Fixture::new();
    let _env = fx.environment(false);
    exe(fx.slp_dir().join("craft"));
    let on_path = exe(fx.path().join("path-bin/craft"));
    let client = LocalMCPClient::new("craft", vec!["mcp".into()]);
    let env = HashMap::from([("PATH".to_string(), s(on_path.parent().unwrap()))]);
    assert_eq!(client.resolve_command(&env), s(&on_path));
}
