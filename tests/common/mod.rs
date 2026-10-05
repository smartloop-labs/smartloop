//! Port of `tests/conftest.py` (shared fixtures).

// ======================= BEGIN unit: core ========================
/// Test support shared by the core unit's tests: the conftest autouse
/// `_isolate_credential_store`, which redirects the credential store to a
/// temp directory so no test ever touches real credentials.
pub mod core_support {
    use std::path::Path;

    use once_cell::sync::Lazy;
    use parking_lot::{Mutex, MutexGuard};
    use smartloop::auth::credential_store;

    /// The credential store's location is process-global, so every test that
    /// redirects it runs exclusively of the others.
    static CREDENTIAL_STORE_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    /// Redirects the credential store to a fresh temp directory for its
    /// lifetime (`monkeypatch.setattr(credential_store, "_slp_home", ...)`).
    ///
    /// `SLP_DEVELOPER_TOKEN` is cleared while the guard lives — it is step 3
    /// of the resolution order, so a value inherited from the developer's
    /// shell would otherwise leak into every assertion — and restored on
    /// drop, along with any `save_creds` observer a test installed.
    pub struct CredentialStoreGuard {
        dir: tempfile::TempDir,
        saved_env: Option<std::ffi::OsString>,
        _lock: MutexGuard<'static, ()>,
    }

    impl CredentialStoreGuard {
        /// The temp directory standing in for `SLP_HOME` (pytest's `tmp_path`).
        pub fn path(&self) -> &Path {
            self.dir.path()
        }

        /// `monkeypatch.setenv("SLP_DEVELOPER_TOKEN", value)`; restored when
        /// the guard drops.
        pub fn set_developer_token_env(&self, value: &str) {
            // SAFETY: see `Drop`.
            unsafe { std::env::set_var("SLP_DEVELOPER_TOKEN", value) };
        }
    }

    impl Drop for CredentialStoreGuard {
        fn drop(&mut self) {
            credential_store::set_save_observer(None);
            credential_store::set_slp_home_override(None);
            // SAFETY: tests touching the environment hold this guard's lock
            // (and are `#[serial]`).
            unsafe {
                match &self.saved_env {
                    Some(value) => std::env::set_var("SLP_DEVELOPER_TOKEN", value),
                    None => std::env::remove_var("SLP_DEVELOPER_TOKEN"),
                }
            }
        }
    }

    /// Start isolating the credential store (see [`CredentialStoreGuard`]).
    pub fn isolate_credential_store() -> CredentialStoreGuard {
        let lock = CREDENTIAL_STORE_LOCK.lock();
        let dir = tempfile::tempdir().expect("temp dir for the credential store");
        credential_store::set_slp_home_override(Some(dir.path().to_path_buf()));
        let saved_env = std::env::var_os("SLP_DEVELOPER_TOKEN");
        // SAFETY: see `Drop`.
        unsafe { std::env::remove_var("SLP_DEVELOPER_TOKEN") };
        CredentialStoreGuard { dir, saved_env, _lock: lock }
    }
}
// ======================== END unit: core =========================

// ===== BEGIN unit: utils =====
// Port of the `_restore_root_logging` autouse fixture.

/// Snapshot of the root logger (handlers + level), restored on drop.
///
/// A test that points the root logger at a queue and leaves it installed
/// turns every later test's log call into a write nobody reads.  Keep such a
/// leak local to the test that made it:
///
/// ```ignore
/// let _root = crate::shared::restore_root_logging();
/// ```
pub struct RootLoggingGuard {
    handlers: Vec<smartloop::utils::logging_setup::RootHandler>,
    level: i32,
}

/// Take a [`RootLoggingGuard`] for the current test.
pub fn restore_root_logging() -> RootLoggingGuard {
    use smartloop::utils::logging_setup::{get_logger, root_handlers};
    RootLoggingGuard { handlers: root_handlers(), level: get_logger("").level() }
}

impl Drop for RootLoggingGuard {
    fn drop(&mut self) {
        use smartloop::utils::logging_setup::{get_logger, set_root_handlers};
        set_root_handlers(std::mem::take(&mut self.handlers));
        get_logger("").set_level(self.level);
    }
}
// ===== END unit: utils =====
