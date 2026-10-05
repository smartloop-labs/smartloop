//! Port of `tests/conftest.py` (shared fixtures).

// ===== BEGIN api_routes_misc =====================================================

/// Isolation the Python `conftest.py` applied to every test, for the tests
/// that touch credentials or process-wide state: the credential store is
/// redirected to a temp directory (`_isolate_credential_store`), the process
/// pin is cleared around the test (`_reset_project_pin`), and the hosted
/// search status cache is forgotten.
///
/// Tests using it must be `#[serial]`: the redirection is an environment
/// variable (`SLP_HOME`), which is process-global.
pub mod misc_isolation {
    use smartloop::search::providers;
    use smartloop::workspace::context::pin_project;

    /// Environment variables the isolation sets or clears, restored on drop.
    const ISOLATED_VARS: &[&str] = &["SLP_HOME", "SLP_DEVELOPER_TOKEN", "SLP_CONFIG_FILE"];

    /// A temp SLP home for the duration of a test.
    pub struct IsolatedHome {
        pub dir: tempfile::TempDir,
        saved: Vec<(&'static str, Option<String>)>,
    }

    impl IsolatedHome {
        /// Redirect `SLP_HOME` to a fresh temp directory and clear the
        /// developer-token / config-file overrides.
        pub fn new() -> Self {
            let dir = tempfile::tempdir().expect("create temp SLP home");
            let saved = ISOLATED_VARS.iter().map(|k| (*k, std::env::var(k).ok())).collect();
            // SAFETY: tests that use this fixture run serially (`#[serial]`),
            // so no other thread reads the environment concurrently.
            unsafe {
                std::env::set_var("SLP_HOME", dir.path());
                std::env::remove_var("SLP_DEVELOPER_TOKEN");
                std::env::remove_var("SLP_CONFIG_FILE");
            }
            pin_project(None);
            providers::reset_status_cache();
            IsolatedHome { dir, saved }
        }

        /// The temp home as a string.
        pub fn path(&self) -> String {
            self.dir.path().to_string_lossy().into_owned()
        }
    }

    impl Default for IsolatedHome {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Drop for IsolatedHome {
        fn drop(&mut self) {
            pin_project(None);
            providers::reset_status_cache();
            // SAFETY: see `new`.
            unsafe {
                for (key, value) in &self.saved {
                    match value {
                        Some(v) => std::env::set_var(key, v),
                        None => std::env::remove_var(key),
                    }
                }
            }
        }
    }
}

// ===== END api_routes_misc =======================================================
