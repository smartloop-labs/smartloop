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

// ===================== BEGIN unit: workspace =====================
/// Test support shared by the workspace unit's tests: environment guards
/// (`monkeypatch.setenv` / `delenv`), a `caplog` stand-in capturing `tracing`
/// events, and the conftest autouse resets of process-global project state.
pub mod workspace_support {
    use std::sync::Arc;

    use parking_lot::Mutex;
    use tracing::field::{Field, Visit};
    use tracing_subscriber::layer::{Context, SubscriberExt};
    use tracing_subscriber::Layer;

    /// Sets / removes an environment variable, restoring the previous value
    /// on drop (`monkeypatch.setenv` / `monkeypatch.delenv`).
    pub struct EnvGuard {
        key: String,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        pub fn set(key: &str, value: impl AsRef<std::ffi::OsStr>) -> Self {
            let previous = std::env::var_os(key);
            // SAFETY: tests touching the environment are `#[serial]`.
            unsafe { std::env::set_var(key, value) };
            Self { key: key.to_string(), previous }
        }

        pub fn remove(key: &str) -> Self {
            let previous = std::env::var_os(key);
            // SAFETY: tests touching the environment are `#[serial]`.
            unsafe { std::env::remove_var(key) };
            Self { key: key.to_string(), previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: tests touching the environment are `#[serial]`.
            unsafe {
                match &self.previous {
                    Some(value) => std::env::set_var(&self.key, value),
                    None => std::env::remove_var(&self.key),
                }
            }
        }
    }

    /// One captured log record.
    #[derive(Debug, Clone)]
    pub struct LogRecord {
        pub level: tracing::Level,
        pub target: String,
        pub message: String,
    }

    impl LogRecord {
        /// `logging.LogRecord.getMessage()`.
        pub fn get_message(&self) -> &str {
            &self.message
        }
    }

    #[derive(Default)]
    struct MessageVisitor {
        message: String,
    }

    impl Visit for MessageVisitor {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" {
                self.message = format!("{value:?}");
            }
        }

        fn record_str(&mut self, field: &Field, value: &str) {
            if field.name() == "message" {
                self.message = value.to_string();
            }
        }
    }

    struct CaptureLayer {
        records: Arc<Mutex<Vec<LogRecord>>>,
    }

    impl<S: tracing::Subscriber> Layer<S> for CaptureLayer {
        fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
            let mut visitor = MessageVisitor::default();
            event.record(&mut visitor);
            self.records.lock().push(LogRecord {
                level: *event.metadata().level(),
                target: event.metadata().target().to_string(),
                message: visitor.message,
            });
        }
    }

    /// `caplog`: captures every `tracing` event emitted on this thread while
    /// alive.
    pub struct LogCapture {
        records: Arc<Mutex<Vec<LogRecord>>>,
        _guard: tracing::subscriber::DefaultGuard,
    }

    impl LogCapture {
        pub fn start() -> Self {
            let records = Arc::new(Mutex::new(Vec::new()));
            let subscriber =
                tracing_subscriber::registry().with(CaptureLayer { records: Arc::clone(&records) });
            let guard = tracing::subscriber::set_default(subscriber);
            Self { records, _guard: guard }
        }

        /// Every record captured so far.
        pub fn records(&self) -> Vec<LogRecord> {
            self.records.lock().clone()
        }

        /// Records at WARNING or above from `target` (`caplog.set_level("WARNING",
        /// logger=target)` followed by `caplog.records`).
        pub fn warnings_from(&self, target: &str) -> Vec<LogRecord> {
            self.records()
                .into_iter()
                .filter(|r| r.target == target && r.level <= tracing::Level::WARN)
                .collect()
        }

        /// `caplog.text`: every captured message, one per line.
        pub fn text(&self) -> String {
            self.records().iter().map(|r| format!("{}\n", r.message)).collect()
        }
    }

    /// The conftest autouse `_reset_project_pin` plus the per-module
    /// `delegate_projects(False)` teardowns: clears the process pin and the
    /// delegation flag now and again on drop.
    pub struct ProcessScopeReset;

    impl ProcessScopeReset {
        pub fn new() -> Self {
            smartloop::workspace::context::pin_project(None);
            smartloop::workspace::context::delegate_projects(false);
            Self
        }
    }

    impl Default for ProcessScopeReset {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Drop for ProcessScopeReset {
        fn drop(&mut self) {
            smartloop::workspace::context::pin_project(None);
            smartloop::workspace::context::delegate_projects(false);
        }
    }

    /// The `home` fixture shared by the project-state tests: `SLP_HOME`
    /// pointed at a fresh temp dir (resolved the way it is live), the legacy
    /// layout switch and any config-file override cleared, the workspace
    /// singleton dropped, and the process pin / delegation reset — all undone
    /// on drop.
    pub struct TempHome {
        pub dir: tempfile::TempDir,
        _env: Vec<EnvGuard>,
        _reset: ProcessScopeReset,
    }

    impl TempHome {
        pub fn new() -> Self {
            use smartloop::workspace::workspace::{Workspace, set_settings_home_dir};

            let reset = ProcessScopeReset::new();
            let dir = tempfile::tempdir().expect("temp home");
            let env = vec![
                EnvGuard::set("SLP_HOME", dir.path()),
                EnvGuard::remove("SLP_LEGACY_CONFIG"),
                EnvGuard::remove("SLP_CONFIG_FILE"),
            ];
            Workspace::set_config_file(None);
            set_settings_home_dir(None);
            Workspace::reset();
            Self { dir, _env: env, _reset: reset }
        }

        /// The home directory.
        pub fn path(&self) -> &std::path::Path {
            self.dir.path()
        }

        /// `str(home / part)`.
        pub fn join(&self, part: &str) -> String {
            self.dir.path().join(part).to_string_lossy().into_owned()
        }
    }

    impl Default for TempHome {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Drop for TempHome {
        fn drop(&mut self) {
            use smartloop::workspace::workspace::{Workspace, set_settings_home_dir};

            Workspace::set_config_file(None);
            set_settings_home_dir(None);
            Workspace::reset();
        }
    }
}
// ====================== END unit: workspace ======================
