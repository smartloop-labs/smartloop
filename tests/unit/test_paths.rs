//! Port of `tests/unit/test_paths.py`.
//!
//! The user's home, read past the sandbox each store puts us in.
//!
//! Both stores we ship through rewrite `$HOME` to a private per-app
//! directory, and a tilde expanded against it names a path the user has
//! never put a file in: on the Mac App Store build every local MCP server
//! failed with an ENOENT under `~/Library/Containers/<bundle-id>/Data`, and a
//! strict snap would put the framework home at
//! `~/snap/smartloop/<rev>/.smartloop` instead of `~/.smartloop`.  What
//! neither rewrites is the passwd record, and snapd says it outright in
//! `SNAP_REAL_HOME`.

#![cfg(unix)]

use std::ffi::OsString;

use serial_test::serial;
use smartloop::paths::{expand_home, expand_home_with, expanduser, real_home};

/// `monkeypatch` for environment variables: every change is undone on drop.
struct Env {
    saved: Vec<(String, Option<OsString>)>,
}

impl Env {
    /// The autouse `clean_env` fixture: start every case outside any
    /// confinement, whatever the runner is in.
    fn clean() -> Self {
        let mut env = Env { saved: Vec::new() };
        env.delenv("SNAP_REAL_HOME");
        env.delenv("SNAP_NAME");
        env
    }

    fn remember(&mut self, key: &str) {
        if !self.saved.iter().any(|(k, _)| k == key) {
            self.saved.push((key.to_string(), std::env::var_os(key)));
        }
    }

    fn setenv(&mut self, key: &str, value: &str) {
        self.remember(key);
        // SAFETY: every test that touches the environment is `#[serial]`.
        unsafe { std::env::set_var(key, value) };
    }

    fn delenv(&mut self, key: &str) {
        self.remember(key);
        // SAFETY: every test that touches the environment is `#[serial]`.
        unsafe { std::env::remove_var(key) };
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        for (key, value) in self.saved.drain(..).rev() {
            // SAFETY: every test that touches the environment is `#[serial]`.
            unsafe {
                match value {
                    Some(v) => std::env::set_var(&key, v),
                    None => std::env::remove_var(&key),
                }
            }
        }
    }
}

/// The `passwd_home` fixture: `pwd.getpwuid(os.getuid()).pw_dir`.
fn passwd_home() -> String {
    smartloop::paths::passwd_home().expect("a passwd record for the current uid")
}

mod test_real_home {
    use super::*;

    #[test]
    #[serial]
    fn test_an_ordinary_home_env_var_is_honoured() {
        let mut env = Env::clean();
        // A service unit or a test that sets $HOME means it; only the two
        // confinement spellings are ever overridden.
        env.setenv("HOME", "/home/tester");
        assert_eq!(real_home(), "/home/tester");
    }

    #[test]
    #[serial]
    fn test_an_app_sandbox_container_reads_through_to_passwd() {
        let mut env = Env::clean();
        let passwd_home = passwd_home();
        env.setenv("HOME", "/Users/tester/Library/Containers/ai.smartloop.studio/Data");
        assert_eq!(real_home(), passwd_home);
    }

    #[test]
    #[serial]
    fn test_snapd_is_taken_at_its_word() {
        let mut env = Env::clean();
        // SNAP_REAL_HOME (snapd >= 2.46) is the answer stated outright, so
        // nothing has to be inferred from the shape of $HOME.
        env.setenv("HOME", "/home/tester/snap/smartloop/42");
        env.setenv("SNAP_NAME", "smartloop");
        env.setenv("SNAP_REAL_HOME", "/home/tester");
        assert_eq!(real_home(), "/home/tester");
    }

    #[test]
    #[serial]
    fn test_an_older_snapd_falls_back_to_passwd() {
        let mut env = Env::clean();
        let passwd_home = passwd_home();
        // Before SNAP_REAL_HOME existed there is only $SNAP_USER_DATA, whose
        // shape says where the real home is — and passwd says it exactly.
        env.setenv("HOME", "/home/tester/snap/smartloop/42");
        env.setenv("SNAP_NAME", "smartloop");
        assert_eq!(real_home(), passwd_home);
    }

    #[test]
    #[serial]
    fn test_a_snaps_environment_leaking_into_a_child_is_ignored() {
        let mut env = Env::clean();
        // A snap that launches something else — a terminal emulator starting
        // a shell — hands its whole SNAP_* block to a child that is not
        // confined at all.  $HOME is what says whether we are the confined
        // process, so an inherited SNAP_REAL_HOME must not override a home
        // that is already real (and, in a test, already steered).
        env.setenv("HOME", "/home/tester");
        env.setenv("SNAP_NAME", "alacritty");
        env.setenv("SNAP_REAL_HOME", "/home/someone-else");
        assert_eq!(real_home(), "/home/tester");
    }

    #[test]
    #[serial]
    fn test_a_snap_shaped_home_outside_a_snap_is_left_alone() {
        let mut env = Env::clean();
        // `~/snap/x/y` is a perfectly ordinary directory to have as a home.
        // Without SNAP_NAME we are not in a snap and $HOME is not a rewrite.
        env.setenv("HOME", "/home/tester/snap/smartloop/42");
        assert_eq!(real_home(), "/home/tester/snap/smartloop/42");
    }
}

mod test_expand_home {
    use super::*;

    #[test]
    #[serial]
    fn test_a_tilde_expands_against_the_real_home() {
        let _env = Env::clean();
        let real_home = || "/Users/tester".to_string();
        assert_eq!(expand_home_with("~/.smartloop", real_home), "/Users/tester/.smartloop");
        assert_eq!(expand_home_with("~", real_home), "/Users/tester");
    }

    #[test]
    #[serial]
    fn test_an_absolute_path_is_returned_unchanged() {
        let _env = Env::clean();
        assert_eq!(expand_home("/opt/smartloop"), "/opt/smartloop");
    }

    #[test]
    #[serial]
    fn test_another_users_home_is_left_to_expanduser() {
        let _env = Env::clean();
        // `~root/...` names somebody else's home, which only the passwd
        // lookup behind expanduser knows how to resolve.
        let real_home = || "/Users/tester".to_string();
        assert_eq!(expand_home_with("~root/x", real_home), expanduser("~root/x"));
    }
}
