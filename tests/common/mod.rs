//! Port of `tests/conftest.py` (shared fixtures).

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
