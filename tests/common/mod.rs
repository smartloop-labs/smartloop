//! Port of `tests/conftest.py` (shared fixtures).

// ===================== process-wide home isolation =====================
/// Pytest isolated every test through autouse fixtures; Rust has no
/// equivalent, so a test that forgot to redirect `SLP_HOME` used to read and
/// write the developer's real `~/.smartloop` (config.json included).  Before
/// any test in this binary runs, point `HOME` and `SLP_HOME` at a private
/// temp directory.  Tests that set `SLP_HOME` themselves still do; their
/// guards restore it to this directory, never to the real home.
#[ctor::ctor(unsafe)]
fn isolate_process_home() {
    let root = tempfile::Builder::new()
        .prefix("slp-test-home-")
        .tempdir()
        .expect("create isolated test home")
        .keep();
    let slp_home = root.join(".smartloop");
    std::fs::create_dir_all(&slp_home).expect("create isolated SLP_HOME");
    // SAFETY: runs before main, while the process is still single-threaded.
    unsafe {
        std::env::set_var("HOME", &root);
        std::env::set_var("SLP_HOME", &slp_home);
        std::env::remove_var("SLP_CONFIG_FILE");
        std::env::remove_var("SNAP_NAME");
        std::env::remove_var("SNAP_REAL_HOME");
    }
}

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

// ===================================================================== db unit
// Shared fixtures added by the `db` unit: a recording `VectorStore` (the
// Python tests' `MagicMock()` store) and a tracing capture (the Python tests'
// `MagicMock()` logger / `caplog`).

pub mod db {
    use std::sync::Arc;

    use parking_lot::Mutex;
    use serde_json::{Map, Value};
    use smartloop::db::store::{Chunk, ChunkResult, VectorStore};

    /// A `VectorStore` that records every call and stores nothing — the
    /// Python tests' `MagicMock()` store.
    #[derive(Default)]
    pub struct RecordingStore {
        /// Every `add_chunks` call's chunk list, in call order.
        pub added: Mutex<Vec<Vec<Chunk>>>,
        /// Every `delete_document` call's document id, in call order.
        pub deleted: Mutex<Vec<String>>,
        /// Every `query_with_metadata` call's query text.
        pub queries: Mutex<Vec<String>>,
    }

    impl RecordingStore {
        pub fn new() -> Arc<Self> {
            Arc::new(Self::default())
        }

        /// The chunks of the most recent `add_chunks` call (`call_args[0][0]`).
        pub fn last_added(&self) -> Option<Vec<Chunk>> {
            self.added.lock().last().cloned()
        }

        /// Number of `add_chunks` calls.
        pub fn add_count(&self) -> usize {
            self.added.lock().len()
        }

        /// Number of `delete_document` calls.
        pub fn delete_count(&self) -> usize {
            self.deleted.lock().len()
        }
    }

    impl VectorStore for RecordingStore {
        fn project_dir(&self) -> &str {
            ""
        }

        fn collection_name(&self) -> &str {
            "documents"
        }

        fn base_dir(&self) -> Option<&str> {
            None
        }

        fn add_chunks(&self, chunks: &[Chunk]) -> anyhow::Result<()> {
            self.added.lock().push(chunks.to_vec());
            Ok(())
        }

        fn query(&self, query_text: &str, _n_results: Option<usize>) -> anyhow::Result<Vec<String>> {
            self.queries.lock().push(query_text.to_string());
            Ok(Vec::new())
        }

        fn query_with_metadata(
            &self,
            query_text: &str,
            _n_results: Option<usize>,
            _max_distance: f64,
            _where: Option<&Map<String, Value>>,
        ) -> anyhow::Result<Vec<ChunkResult>> {
            self.queries.lock().push(query_text.to_string());
            Ok(Vec::new())
        }

        fn delete_document(&self, document_id: &str) -> anyhow::Result<()> {
            self.deleted.lock().push(document_id.to_string());
            Ok(())
        }
    }

    /// One captured tracing event.
    #[derive(Debug, Clone, PartialEq)]
    pub struct LogRecord {
        pub level: tracing::Level,
        pub target: String,
        pub message: String,
    }

    /// Captured events, shared with the subscriber.
    #[derive(Clone, Default)]
    pub struct LogCapture {
        records: Arc<Mutex<Vec<LogRecord>>>,
    }

    impl LogCapture {
        /// All captured records.
        pub fn records(&self) -> Vec<LogRecord> {
            self.records.lock().clone()
        }

        /// Messages logged at `level` (optionally restricted to `target`).
        pub fn messages_at(&self, level: tracing::Level, target: Option<&str>) -> Vec<String> {
            self.records
                .lock()
                .iter()
                .filter(|r| r.level == level && target.is_none_or(|t| r.target == t))
                .map(|r| r.message.clone())
                .collect()
        }

        /// `logger.info.assert_any_call(message)`.
        pub fn has_info(&self, message: &str) -> bool {
            self.messages_at(tracing::Level::INFO, None).iter().any(|m| m == message)
        }
    }

    struct MessageVisitor<'a>(&'a mut String);

    impl tracing::field::Visit for MessageVisitor<'_> {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" {
                *self.0 = format!("{value:?}");
            }
        }

        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            if field.name() == "message" {
                *self.0 = value.to_string();
            }
        }
    }

    struct CaptureLayer {
        records: Arc<Mutex<Vec<LogRecord>>>,
    }

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for CaptureLayer {
        fn on_event(&self, event: &tracing::Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
            let mut message = String::new();
            event.record(&mut MessageVisitor(&mut message));
            self.records.lock().push(LogRecord {
                level: *event.metadata().level(),
                target: event.metadata().target().to_string(),
                message,
            });
        }
    }

    /// Run `f` with every tracing event on this thread captured.
    pub fn capture_logs<R>(f: impl FnOnce() -> R) -> (R, LogCapture) {
        use tracing_subscriber::layer::SubscriberExt;

        let capture = LogCapture::default();
        let subscriber = tracing_subscriber::registry()
            .with(CaptureLayer { records: capture.records.clone() });
        let result = tracing::subscriber::with_default(subscriber, f);
        (result, capture)
    }
}
// ================================================================= end db unit

// ===================================================================== mcp
// Fixtures added by the `mcp` unit.  Other units: add yours in your own
// delimited block below rather than editing this one.

/// Fakes for the MCP transport layer.
///
/// [`mcp::FakeHttp`] stands in for `httpx.Client`: installed process-wide with
/// [`smartloop::mcp::client::HttpTransportOverride`] it is what every
/// `MCPClient` / owned `MCPOAuthService` talks to (the Python tests'
/// `patch("httpx.Client", ...)`), and passed to `MCPOAuthService::with_transport`
/// it is the injected client (`MCPOAuthService(http_client=fake)`).  Tests that
/// install the override (or touch any other process-wide MCP state: the
/// router's OAuth sessions and hooks, refresh locks) must be
/// `#[serial(mcp_global)]`.  Tests that build a [`mcp::ServiceEnv`] also touch the
/// process-scope pin and the settings home shared with default-key tests,
/// so they hold both keys: `#[serial] #[serial(mcp_global)]`.
pub mod mcp {
    use std::sync::Arc;

    use parking_lot::Mutex;
    use serde_json::{Map, Value, json};

    use smartloop::agents::mcp::mcp_tool_caller::MCPToolCaller;
    use smartloop::mcp::client::{HttpBody, HttpResponse, HttpTransport, MCPToolCallResult};
    use smartloop::mcp::read_only_calls::transport_only_tool_caller;

    /// One request the fake saw.
    #[derive(Debug, Clone)]
    pub struct Recorded {
        pub method: String,
        pub url: String,
        pub body: Option<HttpBody>,
        pub headers: Vec<(String, String)>,
    }

    impl Recorded {
        /// The posted form / JSON object as a map (`data=` / `json=`).
        pub fn form(&self) -> Map<String, Value> {
            self.body.as_ref().map(HttpBody::form_map).unwrap_or_default()
        }
    }

    /// `(method, url, body) -> response`.  `Err` is a transport failure.
    pub type Handler =
        Arc<dyn Fn(&str, &str, Option<&HttpBody>) -> Result<HttpResponse, String> + Send + Sync>;

    /// A scripted `httpx.Client` that records every request.
    pub struct FakeHttp {
        handler: Handler,
        pub calls: Mutex<Vec<Recorded>>,
    }

    impl FakeHttp {
        /// A fake answering through `handler`.
        pub fn new(
            handler: impl Fn(&str, &str, Option<&HttpBody>) -> Result<HttpResponse, String> + Send + Sync + 'static,
        ) -> Arc<Self> {
            Arc::new(Self { handler: Arc::new(handler), calls: Mutex::new(Vec::new()) })
        }

        /// Exact URL -> response (any method); unmapped URLs answer 404.
        pub fn routes(routes: Vec<(&str, HttpResponse)>) -> Arc<Self> {
            let routes: Vec<(String, HttpResponse)> =
                routes.into_iter().map(|(u, r)| (u.to_string(), r)).collect();
            Self::new(move |_, url, _| {
                Ok(routes
                    .iter()
                    .find(|(u, _)| u == url)
                    .map(|(_, r)| r.clone())
                    .unwrap_or_else(|| HttpResponse::new(404, "")))
            })
        }

        /// Every request answers `response`.
        pub fn always(response: HttpResponse) -> Arc<Self> {
            Self::new(move |_, _, _| Ok(response.clone()))
        }

        /// URLs of the GET requests, in order.
        pub fn get_calls(&self) -> Vec<String> {
            self.calls.lock().iter().filter(|c| c.method == "GET").map(|c| c.url.clone()).collect()
        }

        /// `(url, form)` of the POST requests, in order.
        pub fn posts(&self) -> Vec<(String, Map<String, Value>)> {
            self.calls
                .lock()
                .iter()
                .filter(|c| c.method == "POST")
                .map(|c| (c.url.clone(), c.form()))
                .collect()
        }

        /// Number of requests seen.
        pub fn call_count(&self) -> usize {
            self.calls.lock().len()
        }
    }

    impl HttpTransport for FakeHttp {
        fn get(&self, url: &str) -> Result<HttpResponse, String> {
            self.calls.lock().push(Recorded {
                method: "GET".into(),
                url: url.into(),
                body: None,
                headers: Vec::new(),
            });
            (self.handler)("GET", url, None)
        }

        fn post(&self, url: &str, body: &HttpBody, headers: &[(String, String)]) -> Result<HttpResponse, String> {
            self.calls.lock().push(Recorded {
                method: "POST".into(),
                url: url.into(),
                body: Some(body.clone()),
                headers: headers.to_vec(),
            });
            (self.handler)("POST", url, Some(body))
        }
    }

    /// A `200` JSON response.
    pub fn ok_json(value: Value) -> HttpResponse {
        HttpResponse::json(200, &value)
    }

    /// A JSON-RPC `result` envelope answering request `id`.
    pub fn rpc_result(result: Value) -> HttpResponse {
        ok_json(json!({"jsonrpc": "2.0", "id": "1", "result": result}))
    }

    /// `MCPToolCaller(None, None)`: a caller with no LLM behind it.
    pub fn tool_caller() -> MCPToolCaller {
        transport_only_tool_caller()
    }

    /// A successful tool result carrying one text block.
    pub fn text_result(body: &str) -> MCPToolCallResult {
        MCPToolCallResult {
            success: true,
            content: Some(vec![json!({"type": "text", "text": body})]),
            ..Default::default()
        }
    }

    /// A `MakeWriter` appending formatted log lines to a shared buffer.
    #[derive(Clone, Default)]
    pub struct LogBuffer(pub Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for LogBuffer {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogBuffer {
        type Writer = LogBuffer;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Run `f` with every event at `level` or above (on this thread) captured
    /// — pytest's `caplog.at_level(...)`.  Returns `f`'s value and the text.
    pub fn capture_logs<R>(level: tracing::Level, f: impl FnOnce() -> R) -> (R, String) {
        let buffer = LogBuffer::default();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(level)
            .with_ansi(false)
            .with_writer(buffer.clone())
            .finish();
        let out = tracing::subscriber::with_default(subscriber, f);
        let text = String::from_utf8_lossy(&buffer.0.lock()).into_owned();
        (out, text)
    }

    /// The 401 a transport reports for an expired token.
    pub fn unauthorized() -> MCPToolCallResult {
        MCPToolCallResult {
            success: false,
            error: Some("Authentication failed (401).".into()),
            auth_expired: true,
            ..Default::default()
        }
    }

    /// Run a future to completion on a fresh current-thread runtime
    /// (`asyncio.run`).
    pub fn block_on<F: std::future::Future>(fut: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a tokio runtime")
            .block_on(fut)
    }

    /// Sets environment variables for its lifetime and restores the previous
    /// values on drop (`monkeypatch.setenv` / `delenv`).  Tests using it must
    /// be `#[serial]`.
    pub struct EnvGuard {
        saved: Vec<(String, Option<std::ffi::OsString>)>,
    }

    impl EnvGuard {
        /// `Some(value)` sets the variable, `None` removes it.
        pub fn set(vars: &[(&str, Option<&str>)]) -> Self {
            let mut saved = Vec::with_capacity(vars.len());
            for (key, value) in vars {
                saved.push((key.to_string(), std::env::var_os(key)));
                // SAFETY: callers are serialized (`#[serial]`), so no other
                // thread of this test binary reads the environment meanwhile.
                unsafe {
                    match value {
                        Some(v) => std::env::set_var(key, v),
                        None => std::env::remove_var(key),
                    }
                }
            }
            Self { saved }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (key, value) in self.saved.drain(..).rev() {
                // SAFETY: see `EnvGuard::set`.
                unsafe {
                    match value {
                        Some(v) => std::env::set_var(&key, v),
                        None => std::env::remove_var(&key),
                    }
                }
            }
        }
    }

    /// A tool definition with an optional input schema.
    pub fn tool(name: &str, schema: Option<Value>) -> smartloop::workspace::MCPToolDefinition {
        smartloop::workspace::MCPToolDefinition {
            name: name.to_string(),
            description: Some(String::new()),
            input_schema: schema.and_then(|s| s.as_object().cloned()),
        }
    }

    /// A remote OAuth connection, as the workspace stores one.
    pub fn oauth_server(
        id: &str,
        name: &str,
        server_url: &str,
        token_endpoint: Option<&str>,
        access_token: Option<&str>,
        refresh_token: Option<&str>,
    ) -> smartloop::workspace::MCPServer {
        use smartloop::workspace::workspace::{
            MCPOAuthCredentials, MCPServer, MCPServerType, RemoteMCPServerConfig,
        };
        MCPServer {
            id: id.to_string(),
            name: name.to_string(),
            server_type: MCPServerType::Remote,
            remote_config: Some(RemoteMCPServerConfig {
                server_url: server_url.to_string(),
                token_endpoint: token_endpoint.map(str::to_string),
                ..Default::default()
            }),
            oauth_credentials: Some(MCPOAuthCredentials {
                client_id: "c".to_string(),
                access_token: access_token.map(str::to_string),
                refresh_token: refresh_token.map(str::to_string),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// The form a token endpoint was posted, by URL.
    pub fn token_posts(fake: &FakeHttp, token_endpoint: &str) -> Vec<Map<String, Value>> {
        fake.posts().into_iter().filter(|(u, _)| u == token_endpoint).map(|(_, f)| f).collect()
    }

    /// A model service bound to an in-memory workspace holding one project,
    /// with the workspace's files under a temp home.
    ///
    /// The Rust stand-in for the Python tests' `MagicMock()` model service in
    /// the router tests that need registration to *succeed*: what the mock's
    /// `add_remote_mcp_server` was called with is read back here from the
    /// connection the real method stored on the project.
    ///
    /// Nothing it starts reaches the network: every HTTP request made while it
    /// lives (a background provisioning pass, say) answers 404.
    pub struct ServiceEnv {
        pub model_service: Arc<smartloop::api::services::model_service::ModelService>,
        pub workspace: smartloop::workspace::workspace::SharedWorkspace,
        pub home: tempfile::TempDir,
        /// What every request made while the env lives was answered by.
        pub http: Arc<FakeHttp>,
        _http: smartloop::mcp::client::HttpTransportOverride,
    }

    impl ServiceEnv {
        pub fn new(project_id: &str) -> Self {
            use smartloop::api::services::model_service::ModelService;
            use smartloop::model::model_manager::Model;
            use smartloop::workspace::workspace::{Project, Workspace};

            // conftest's autouse `_reset_project_pin`: an unpinned process
            // writes every project's file.
            smartloop::workspace::context::pin_project(None);
            let fake = FakeHttp::always(HttpResponse::new(404, ""));
            let http = smartloop::mcp::client::HttpTransportOverride::install(fake.clone());
            let home = tempfile::tempdir().expect("a temp home");
            let root = home.path().to_string_lossy().into_owned();
            Workspace::set_config_file(Some(format!("{root}/config.json")));
            smartloop::workspace::workspace::set_settings_home_dir(Some(root.clone()));

            let mut workspace = Workspace::new("w1");
            workspace.workspace_dir = format!("{root}/w1");
            workspace.models_dir = format!("{root}/w1/models");
            workspace.projects.push(Project {
                id: project_id.to_string(),
                name: Some("proj".to_string()),
                current: true,
                project_dir: format!("{root}/w1/{project_id}"),
                ..Default::default()
            });
            let workspace = Arc::new(parking_lot::RwLock::new(workspace));
            let model_service = ModelService::new_bare();
            *model_service.model.write() = Some(Arc::new(Model::new("m", workspace.clone(), None)));
            // So the `_ensure_model` / `_initialize_workspace` guards see the
            // injected model as the loaded default and keep it.
            *model_service.model_name.write() =
                Some(smartloop::api::config::api_settings().default_model);
            Self { model_service, workspace, home, http: fake, _http: http }
        }

        /// Store `server` on the project.
        pub fn add_server(&self, server: smartloop::workspace::MCPServer) {
            self.workspace.write().projects[0].mcp_servers.push(server);
        }

        /// Register an MCP task whose provisioner only records what it was
        /// handed (Python: `svc._queue_mcp_provisioning` as a mock), so a
        /// queued pass never runs for real behind the test's back.
        pub fn record_provisioning(&self) -> Arc<Mutex<Vec<(String, String)>>> {
            use smartloop::tasks::base_task::TaskOptions;
            use smartloop::tasks::mcp_task::MCPTask;

            let queued: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));
            let seen = queued.clone();
            let task = MCPTask::new(
                Arc::new(QuietLogger),
                Some(Arc::new(move |project_id: &str, server_id: &str| {
                    seen.lock().push((project_id.to_string(), server_id.to_string()));
                    Ok(json!({"status": "skipped"}))
                })),
                Some(Arc::new(|| Ok(Vec::new()))),
                TaskOptions::default(),
            );
            self.model_service.task_queue.read().register(task);
            queued
        }

        /// The connections stored on the first project.
        pub fn servers(&self) -> Vec<smartloop::workspace::MCPServer> {
            self.workspace.read().projects[0].mcp_servers.clone()
        }
    }

    impl Drop for ServiceEnv {
        fn drop(&mut self) {
            smartloop::workspace::workspace::Workspace::set_config_file(None);
            smartloop::workspace::workspace::set_settings_home_dir(None);
        }
    }

    /// A task logger that drops everything.
    pub struct QuietLogger;

    impl smartloop::utils::log_utils::LoggerLike for QuietLogger {
        fn info(&self, _msg: &str) {}
        fn warning(&self, _msg: &str) {}
        fn debug(&self, _msg: &str) {}
        fn error(&self, _msg: &str) {}
    }

    /// Wait up to `timeout` for `done()`.
    pub fn wait_for(timeout: std::time::Duration, done: impl Fn() -> bool) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        while std::time::Instant::now() < deadline {
            if done() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        done()
    }

    /// Reset the router's process-wide OAuth state and test hooks.
    pub fn reset_router_state() {
        use smartloop::api::routers::mcp_servers::{
            set_hooks, McpRouteHooks, COMPLETED_OAUTH_RESPONSES, PENDING_OAUTH_SESSIONS,
        };
        PENDING_OAUTH_SESSIONS.lock().clear();
        COMPLETED_OAUTH_RESPONSES.lock().clear();
        set_hooks(McpRouteHooks::default());
    }

    /// What the register endpoint's OAuth branch produced.
    pub struct OAuthRegistration {
        pub response: Result<
            smartloop::api::model::mcp::MCPRegisterResponse,
            smartloop::api::routers::mcp_catalog::HTTPException,
        >,
        /// The authorization request's query (`build_authorization_url`'s
        /// keyword arguments, as they reached the URL).
        pub authorize: Map<String, Value>,
        /// The pending session the callback will read.
        pub session: Map<String, Value>,
        pub http: Arc<FakeHttp>,
    }

    /// Drive the register endpoint down its OAuth path.
    ///
    /// The Python tests patch `MCPOAuthService` with a fake whose discovery
    /// returns `prm` (or raises when `None`) and the authorization server's
    /// metadata.  Here the real service runs against a fake HTTP layer serving
    /// exactly those documents, and the unauthenticated probe fails through
    /// the router's discovery hook (`MCPClient().__enter__` raising "401").
    pub fn register_via_oauth(
        prm: Option<Value>,
        as_metadata: Value,
        request: smartloop::api::model::mcp::MCPRegisterRequest,
        curated: Option<&smartloop::api::routers::mcp_catalog::MCPCatalogEntry>,
    ) -> OAuthRegistration {
        use smartloop::api::routers::mcp_servers::{
            register_remote_mcp_server, set_hooks, McpRouteHooks, PENDING_OAUTH_SESSIONS,
        };
        use smartloop::api::services::model_service::ModelService;
        use smartloop::mcp::client::HttpTransportOverride;

        reset_router_state();
        let registration_endpoint = as_metadata
            .get("registration_endpoint")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let http = FakeHttp::new(move |method, url, _| {
            if method == "GET" && url.contains("/.well-known/oauth-protected-resource") {
                return Ok(match &prm {
                    Some(doc) => ok_json(doc.clone()),
                    None => HttpResponse::new(404, ""),
                });
            }
            if method == "GET"
                && (url.contains("/.well-known/oauth-authorization-server")
                    || url.contains("/.well-known/openid-configuration"))
            {
                return Ok(ok_json(as_metadata.clone()));
            }
            if method == "POST" && !registration_endpoint.is_empty() && url == registration_endpoint {
                return Ok(ok_json(json!({"client_id": "cid", "client_secret": null})));
            }
            Ok(HttpResponse::new(404, ""))
        });
        let _transport = HttpTransportOverride::install(http.clone());
        set_hooks(McpRouteHooks {
            // `_try_register_smartloop_mcp_server` → None: no /info here.
            probe_info: Some(Arc::new(|_| Err(anyhow::anyhow!("no /info endpoint")))),
            // The direct, unauthenticated connect has to fail for OAuth to run.
            discover_remote_tools: Some(Arc::new(|_, _, _| Err(anyhow::anyhow!("401")))),
            ..Default::default()
        });

        let response = block_on(register_remote_mcp_server(&request, ModelService::new_bare(), "p1", curated));
        set_hooks(McpRouteHooks::default());

        let session = PENDING_OAUTH_SESSIONS.lock().values().next().cloned().unwrap_or_default();
        let authorize = response
            .as_ref()
            .ok()
            .and_then(|r| r.auth_url.as_deref())
            .map(query_params)
            .unwrap_or_default();
        OAuthRegistration { response, authorize, session, http }
    }

    /// Deliver a provider redirect to the unscoped callback.
    ///
    /// `session`, when given, is stored under its `state` first.  The code
    /// exchange answers `token` and records each call (the Python tests'
    /// `_FakeOAuth.exchange_code_for_tokens`); discovery answers
    /// `connect_result` (their patched `MCPClient`).  Returns the response and
    /// every exchange that was attempted.
    pub fn run_callback(
        session: Option<Value>,
        query: smartloop::api::routers::mcp_servers::OAuthCallbackQuery,
        connect_result: smartloop::mcp::client::MCPConnectResult,
        token: smartloop::mcp::oauth::TokenResponse,
        model_service: Arc<smartloop::api::services::model_service::ModelService>,
    ) -> (
        smartloop::api::routers::mcp_servers::OAuthCallbackResponse,
        Vec<smartloop::api::routers::mcp_servers::TokenExchangeParams>,
    ) {
        use smartloop::api::routers::mcp_servers::{
            mcp_oauth_callback, set_hooks, McpRouteHooks, PENDING_OAUTH_SESSIONS,
        };

        reset_router_state();
        if let Some(Value::Object(session)) = session {
            let state = session.get("state").and_then(Value::as_str).unwrap_or("st").to_string();
            PENDING_OAUTH_SESSIONS.lock().insert(state, session);
        }
        let exchanged = Arc::new(Mutex::new(Vec::new()));
        let seen = exchanged.clone();
        set_hooks(McpRouteHooks {
            exchange_oauth_code: Some(Arc::new(move |params| {
                seen.lock().push(params);
                Ok(token.clone())
            })),
            discover_remote_tools: Some(Arc::new(move |_, _, _| Ok(connect_result.clone()))),
            probe_info: None,
            ..Default::default()
        });
        let response = block_on(mcp_oauth_callback(query, model_service));
        set_hooks(McpRouteHooks::default());
        let exchanged = exchanged.lock().clone();
        (response, exchanged)
    }

    /// A token response (`MagicMock(access_token="at", ...)`).
    pub fn token_response(refresh_token: Option<&str>, expires_in: Option<i64>, scope: Option<&str>) -> smartloop::mcp::oauth::TokenResponse {
        smartloop::mcp::oauth::TokenResponse {
            access_token: "at".to_string(),
            token_type: "Bearer".to_string(),
            expires_in,
            refresh_token: refresh_token.map(str::to_string),
            scope: scope.map(str::to_string),
        }
    }

    /// One call the transport saw: `(server name, tool, arguments, access token)`.
    pub type SeenCall = (String, String, Map<String, Value>, Option<String>);

    /// Replace `caller`'s transport (Python: `caller.call_tool =
    /// MagicMock(side_effect=...)`).  `answer` gets `(server, tool, arguments)`;
    /// every call is recorded, in order, in the returned log.
    pub fn stub_transport(
        caller: &MCPToolCaller,
        answer: impl Fn(&smartloop::workspace::MCPServer, &str, &Map<String, Value>) -> MCPToolCallResult
            + Send
            + Sync
            + 'static,
    ) -> Arc<Mutex<Vec<SeenCall>>> {
        let log: Arc<Mutex<Vec<SeenCall>>> = Arc::new(Mutex::new(Vec::new()));
        let seen = log.clone();
        caller.set_call_tool_hook(Some(Arc::new(move |server, tool, arguments, token, _cwd| {
            let arguments = arguments.cloned().unwrap_or_default();
            seen.lock().push((server.name.clone(), tool.to_string(), arguments.clone(), token.map(str::to_string)));
            answer(server, tool, &arguments)
        })));
        log
    }

    /// Answer `results` in order, repeating the last one once they run out
    /// (`side_effect=[...]` / `return_value=...`).
    pub fn in_order(
        results: Vec<MCPToolCallResult>,
    ) -> impl Fn(&smartloop::workspace::MCPServer, &str, &Map<String, Value>) -> MCPToolCallResult + Send + Sync + 'static
    {
        let queue = Mutex::new(std::collections::VecDeque::from(results));
        move |_, _, _| {
            let mut q = queue.lock();
            if q.len() > 1 { q.pop_front().unwrap() } else { q.front().cloned().expect("a result") }
        }
    }

    /// A token endpoint, installed process-wide, that answers every refresh
    /// with `access_token` (or with `status` when that is not 200), counting
    /// what it was sent.
    pub fn token_endpoint(status: u16, access_token: &str) -> (Arc<FakeHttp>, smartloop::mcp::client::HttpTransportOverride) {
        let body = json!({"access_token": access_token, "token_type": "Bearer", "expires_in": 3600});
        let fake = if status == 200 {
            FakeHttp::always(ok_json(body))
        } else {
            FakeHttp::always(HttpResponse::new(status, "upstream unavailable"))
        };
        let guard = smartloop::mcp::client::HttpTransportOverride::install(fake.clone());
        (fake, guard)
    }

    /// A tool resolver answering through a closure (Python:
    /// `ex.mcp_agent.resolve.side_effect = ...`); counts its calls.
    pub struct FakeResolver {
        pub answer: Box<
            dyn Fn(&smartloop::workspace::MCPServer) -> Option<smartloop::agents::mcp::mcp_agent::ResolvedCall>
                + Send
                + Sync,
        >,
        pub calls: Mutex<usize>,
    }

    impl smartloop::agents::mcp::explorer::ToolResolver for FakeResolver {
        fn resolve(
            &self,
            _directive: &str,
            mcp_servers: &[smartloop::workspace::MCPServer],
            _context: Option<&str>,
            _llm: Option<&dyn smartloop::llms::BaseLLM>,
        ) -> Option<smartloop::agents::mcp::mcp_agent::ResolvedCall> {
            *self.calls.lock() += 1;
            mcp_servers.first().and_then(|s| (self.answer)(s))
        }
    }

    /// Web search that is switched off (`web_search_agent.is_enabled() ->
    /// False`).
    pub struct NoWebSearch;

    impl smartloop::agents::mcp::explorer::WebSearcher for NoWebSearch {
        fn is_enabled(&self) -> bool {
            false
        }
        fn search(
            &self,
            _query: &str,
            _progress: Option<smartloop::agents::mcp::explorer::ProgressCallback>,
        ) -> anyhow::Result<String> {
            Ok(String::new())
        }
        fn last_sources(&self) -> Vec<String> {
            Vec::new()
        }
        fn last_source_extracts(&self) -> std::collections::HashMap<String, String> {
            std::collections::HashMap::new()
        }
    }

    /// An explorer over `caller` with no model behind it (Python:
    /// `Explorer.__new__(Explorer)` with the collaborators set by hand).
    pub fn explorer(
        caller: Arc<MCPToolCaller>,
        resolver: Arc<FakeResolver>,
        follow_up_calls: smartloop::agents::mcp::explorer::FollowUpPlanner,
    ) -> smartloop::agents::mcp::explorer::Explorer {
        smartloop::agents::mcp::explorer::Explorer {
            llm: None,
            tool_caller: Some(caller),
            arg_extractor: None,
            use_llm_history: false,
            fetch_agent: Arc::new(smartloop::agents::fetch_agent::FetchAgent::new()),
            web_search_agent: Arc::new(NoWebSearch),
            mcp_agent: resolver,
            follow_up_calls,
        }
    }

    /// The query parameters of a URL, decoded.
    pub fn query_params(url: &str) -> Map<String, Value> {
        let parsed = url::Url::parse(url).expect("a URL");
        parsed
            .query_pairs()
            .map(|(k, v)| (k.into_owned(), Value::String(v.into_owned())))
            .collect()
    }
}
// =================================================================== /mcp

// ======================= BEGIN unit: model =======================
/// Downloader test doubles (scripted HTTP, test logger).
pub mod model_fakes;
// ======================== END unit: model ========================

// ===================== BEGIN unit: platform ======================
/// The `tests/conftest.py` autouse fixtures, plus the helpers the platform
/// unit's tests share (a canned-response HTTP stub server and a clock guard).
///
/// pytest applies the conftest fixtures to every test automatically; in Rust a
/// test opts in by holding the guard [`platform_support::autouse`] returns for
/// its whole body (and should be `#[serial]`, since everything these fixtures
/// reset is process-global).
pub mod platform_support {
    use std::collections::{BTreeMap, HashMap};
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread::JoinHandle;

    use once_cell::sync::Lazy;
    use parking_lot::{Mutex, MutexGuard};
    use serde_json::Value;
    use smartloop::platform::service::{set_clock, system_now};
    use smartloop::platform::token_agent;
    use smartloop::workspace::context::pin_project;

    /// Everything the autouse fixtures touch is process-global, so tests that
    /// hold a [`Conftest`] run one at a time even without `#[serial]`.
    static CONFTEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    /// The autouse fixtures of `tests/conftest.py`, active while this lives:
    ///
    /// * `_isolate_credential_store` — the credential store lives in a fresh
    ///   temp directory (`SLP_HOME`), and `SLP_DEVELOPER_TOKEN` is cleared so
    ///   a value from the developer's shell cannot leak into assertions;
    /// * `_reset_refresh_cooldown` — the failed-refresh cooldown is forgotten
    ///   on entry and on exit;
    /// * `_reset_project_pin` — the process is unpinned on entry and on exit.
    ///
    /// `_reset_mcp_refresh_state` is [`reset_mcp_refresh_state`] (call it from
    /// tests that exercise MCP redemption), and `_restore_root_logging` has no
    /// Rust counterpart: `tracing` has no root logger a test could leave
    /// pointed somewhere else.
    pub struct Conftest {
        dir: tempfile::TempDir,
        saved_home: Option<std::ffi::OsString>,
        saved_dev_token: Option<std::ffi::OsString>,
        _lock: MutexGuard<'static, ()>,
    }

    impl Conftest {
        /// The temp directory standing in for the credential store's home
        /// (pytest's `tmp_path`).
        pub fn tmp_path(&self) -> &Path {
            self.dir.path()
        }
    }

    impl Drop for Conftest {
        fn drop(&mut self) {
            token_agent::reset_failure_cooldown();
            pin_project(None);
            // SAFETY: tests holding a `Conftest` are serialized by
            // CONFTEST_LOCK (and `#[serial]`).
            unsafe {
                restore_env("SLP_HOME", &self.saved_home);
                restore_env("SLP_DEVELOPER_TOKEN", &self.saved_dev_token);
            }
        }
    }

    unsafe fn restore_env(key: &str, saved: &Option<std::ffi::OsString>) {
        match saved {
            Some(value) => unsafe { std::env::set_var(key, value) },
            None => unsafe { std::env::remove_var(key) },
        }
    }

    /// Enter the conftest autouse fixtures (see [`Conftest`]).
    pub fn autouse() -> Conftest {
        let lock = CONFTEST_LOCK.lock();
        let dir = tempfile::tempdir().expect("temp dir for the credential store");
        let saved_home = std::env::var_os("SLP_HOME");
        let saved_dev_token = std::env::var_os("SLP_DEVELOPER_TOKEN");
        // SAFETY: see `Drop`.
        unsafe {
            std::env::set_var("SLP_HOME", dir.path());
            std::env::remove_var("SLP_DEVELOPER_TOKEN");
        }
        token_agent::reset_failure_cooldown();
        pin_project(None);
        Conftest { dir, saved_home, saved_dev_token, _lock: lock }
    }

    /// `_reset_mcp_refresh_state`: forget MCP redemption locks and cooldowns
    /// left by another test.
    pub fn reset_mcp_refresh_state() {
        smartloop::agents::mcp::mcp_tool_caller::reset_refresh_state();
    }

    /// `monkeypatch.setattr(time, "time", lambda: original_time() + offset)`;
    /// the real clock is restored when this drops.
    pub struct ClockGuard;

    impl Drop for ClockGuard {
        fn drop(&mut self) {
            set_clock(None);
        }
    }

    /// Shift the platform clock *offset* seconds into the future.
    pub fn advance_clock(offset: f64) -> ClockGuard {
        set_clock(Some(Arc::new(move || system_now() + offset)));
        ClockGuard
    }

    /// A JSON object literal as a `serde_json::Map`.
    pub fn obj(value: Value) -> serde_json::Map<String, Value> {
        match value {
            Value::Object(map) => map,
            other => panic!("expected a JSON object, got {other}"),
        }
    }

    /// One request the stub server received.
    #[derive(Debug, Clone)]
    pub struct RecordedRequest {
        pub method: String,
        /// The path without its query string.
        pub path: String,
        /// The full request target, query string included.
        pub target: String,
        /// Header names lower-cased (HTTP header names are case-insensitive).
        pub headers: BTreeMap<String, String>,
        pub body: Vec<u8>,
    }

    impl RecordedRequest {
        /// A header by (case-insensitive) name.
        pub fn header(&self, name: &str) -> Option<&str> {
            self.headers.get(&name.to_ascii_lowercase()).map(String::as_str)
        }

        /// The body decoded as JSON.
        pub fn json(&self) -> Value {
            serde_json::from_slice(&self.body).expect("request body is JSON")
        }
    }

    #[derive(Default)]
    struct StubState {
        routes: HashMap<String, Vec<(u16, Value)>>,
        requests: Vec<RecordedRequest>,
    }

    /// Minimal HTTP server that serves canned responses (the Python
    /// `_StubHandler` + `stub_server` fixture).
    ///
    /// A route's value is a list of `(status, body)` served one per request —
    /// which is how a test says "401 first, then 200".  The last entry repeats
    /// once the list runs out, so a retry that should not have happened still
    /// gets an answer rather than hanging the test.  An unknown path is a 404
    /// with no body.  Every request is recorded so a test can assert on what
    /// went out, not just on what came back.
    pub struct StubServer {
        base_url: String,
        state: Arc<Mutex<StubState>>,
        stop: Arc<AtomicBool>,
        addr: std::net::SocketAddr,
        thread: Option<JoinHandle<()>>,
    }

    impl StubServer {
        /// Start a server on an ephemeral 127.0.0.1 port.
        pub fn start() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind stub server");
            let addr = listener.local_addr().expect("stub server address");
            let state = Arc::new(Mutex::new(StubState::default()));
            let stop = Arc::new(AtomicBool::new(false));
            let thread = {
                let state = state.clone();
                let stop = stop.clone();
                std::thread::spawn(move || {
                    for stream in listener.incoming() {
                        if stop.load(Ordering::SeqCst) {
                            break;
                        }
                        if let Ok(stream) = stream {
                            let _ = serve_connection(stream, &state);
                        }
                    }
                })
            };
            StubServer {
                base_url: format!("http://{addr}"),
                state,
                stop,
                addr,
                thread: Some(thread),
            }
        }

        /// `http://127.0.0.1:<port>`.
        pub fn base_url(&self) -> &str {
            &self.base_url
        }

        /// Serve `(status, body)` for every request to *path*.
        pub fn route(&self, path: &str, status: u16, body: Value) {
            self.route_seq(path, vec![(status, body)]);
        }

        /// Serve the responses in turn (the last one repeats).
        pub fn route_seq(&self, path: &str, responses: Vec<(u16, Value)>) {
            assert!(!responses.is_empty(), "a route needs at least one response");
            self.state.lock().routes.insert(path.to_string(), responses);
        }

        /// Every request received so far, oldest first.
        pub fn requests(&self) -> Vec<RecordedRequest> {
            self.state.lock().requests.clone()
        }
    }

    impl Drop for StubServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            // Unblock `accept`.
            let _ = TcpStream::connect(self.addr);
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }

    fn serve_connection(stream: TcpStream, state: &Mutex<StubState>) -> std::io::Result<()> {
        stream.set_read_timeout(Some(std::time::Duration::from_secs(10)))?;
        let mut reader = BufReader::new(stream.try_clone()?);
        let mut writer = stream;
        // One request per connection; the response says `Connection: close`.
        let mut request_line = String::new();
        if reader.read_line(&mut request_line)? == 0 {
            return Ok(());
        }
        let mut parts = request_line.trim_end().splitn(3, ' ');
        let method = parts.next().unwrap_or_default().to_string();
        let target = parts.next().unwrap_or_default().to_string();

        let mut headers = BTreeMap::new();
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            let line = line.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                break;
            }
            if let Some((name, value)) = line.split_once(':') {
                headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
            }
        }
        let length: usize = headers.get("content-length").and_then(|v| v.parse().ok()).unwrap_or(0);
        let mut body = vec![0u8; length];
        if length > 0 {
            reader.read_exact(&mut body)?;
        }

        let path = target.split('?').next().unwrap_or_default().to_string();
        let response = {
            let mut st = state.lock();
            st.requests.push(RecordedRequest {
                method,
                path: path.clone(),
                target: target.clone(),
                headers,
                body,
            });
            match st.routes.get_mut(&path) {
                None => None,
                Some(entries) => {
                    if entries.len() == 1 {
                        Some(entries[0].clone())
                    } else {
                        Some(entries.remove(0))
                    }
                }
            }
        };

        let raw = match response {
            None => "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string(),
            Some((status, payload)) => {
                let body = payload.to_string();
                let reason = reqwest_reason(status);
                format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
            }
        };
        writer.write_all(raw.as_bytes())?;
        writer.flush()
    }

    fn reqwest_reason(status: u16) -> &'static str {
        match status {
            200 => "OK",
            401 => "Unauthorized",
            403 => "Forbidden",
            404 => "Not Found",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            502 => "Bad Gateway",
            503 => "Service Unavailable",
            _ => "Status",
        }
    }
}
// ====================== END unit: platform =======================

// ===== BEGIN unit: supervisor =====================================================
//
// Stand-in processes for the supervisor tests.  The Python tests ran small
// `python -c` / script children (a stub project agent, a stray agent, a probe);
// here the test binary re-executes itself into `supervisor_stub_entry` below,
// which plays the requested part.  The part is passed as a libtest filter
// argument (`slp-stub:<mode>:<arg>`) that matches no test, so the child runs
// exactly the one ignored entry point and nothing else.

/// Prefix of the argument that selects a stub's behaviour.
pub const SUPERVISOR_STUB_PREFIX: &str = "slp-stub:";

/// The libtest path of the stub entry point inside every target that mounts
/// this module as `shared`.
pub const SUPERVISOR_STUB_TEST: &str = "shared::supervisor_stub_entry";

/// argv that re-executes this test binary as a stub process.
///
/// Modes:
/// * `agent:<project_dir>` — the stub project agent: binds a loopback port,
///   answers every GET with 200 `{"status": "healthy", "path": ...}` and
///   echoes POST bodies, then publishes `agent.pid` / `agent.port` beside the
///   project and serves until killed;
/// * `wedged:<project_dir>` — publishes a port but serves nothing on it;
/// * `silent:<project_dir>` — publishes port `1` (nothing listens there);
/// * `probe:<path>` — writes its session/process-group/stdin/fd facts as JSON;
/// * `sleep:` — blocks until killed;
/// * `exit:<code>` — exits at once with `code`.
pub fn supervisor_stub_command(mode: &str, arg: &str) -> Vec<String> {
    let exe = std::env::current_exe().expect("test binary path");
    vec![
        exe.to_string_lossy().into_owned(),
        "--exact".into(),
        SUPERVISOR_STUB_TEST.into(),
        "--ignored".into(),
        "--nocapture".into(),
        "--test-threads=1".into(),
        "-q".into(),
        format!("{SUPERVISOR_STUB_PREFIX}{mode}:{arg}"),
    ]
}

/// `STUB_AGENT`: the stand-in project agent for `project_dir`.
pub fn supervisor_stub_agent(project_dir: &str) -> Vec<String> {
    supervisor_stub_command("agent", project_dir)
}

/// Spawn a stub as an ordinary child (the Python fixtures' `subprocess.Popen`).
pub fn supervisor_spawn_stub(mode: &str, arg: &str) -> std::process::Child {
    let command = supervisor_stub_command(mode, arg);
    std::process::Command::new(&command[0])
        .args(&command[1..])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn stub")
}

/// A real process we can kill, standing in for a supervisor.
pub fn supervisor_sacrificial() -> std::process::Child {
    std::process::Command::new("sleep")
        .arg("300")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn sleep")
}

/// Kill and reap `child`, ignoring errors (fixture teardown).
pub fn supervisor_reap(child: &mut std::process::Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// Poll `predicate` until it holds or `timeout` seconds pass.
pub fn supervisor_wait_for(timeout: f64, mut predicate: impl FnMut() -> bool) -> bool {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(timeout);
    while std::time::Instant::now() < deadline {
        if predicate() {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    predicate()
}

/// `threading.Event` for callbacks the tests hand to the code under test.
#[derive(Debug, Default)]
pub struct SupervisorFlag {
    flag: std::sync::Mutex<bool>,
    cv: std::sync::Condvar,
}

impl SupervisorFlag {
    pub fn new() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self::default())
    }

    pub fn set(&self) {
        *self.flag.lock().unwrap() = true;
        self.cv.notify_all();
    }

    pub fn is_set(&self) -> bool {
        *self.flag.lock().unwrap()
    }

    /// Wait up to `timeout` seconds; returns whether the flag is set.
    pub fn wait(&self, timeout: f64) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(timeout);
        let mut flag = self.flag.lock().unwrap();
        while !*flag {
            let now = std::time::Instant::now();
            if now >= deadline {
                break;
            }
            flag = self.cv.wait_timeout(flag, deadline - now).unwrap().0;
        }
        *flag
    }
}

/// HTTP GET from a test, off any async runtime: the status code, or `None`.
pub fn supervisor_http_status(url: &str, timeout: f64) -> Option<u16> {
    let url = url.to_string();
    std::thread::spawn(move || {
        reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs_f64(timeout))
            .no_proxy()
            .build()
            .ok()?
            .get(&url)
            .send()
            .ok()
            .map(|r| r.status().as_u16())
    })
    .join()
    .ok()
    .flatten()
}

/// HTTP GET from a test, off any async runtime: `(status, body)`.
pub fn supervisor_http_get(url: &str, timeout: f64) -> Option<(u16, String)> {
    let url = url.to_string();
    std::thread::spawn(move || {
        let response = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs_f64(timeout))
            .no_proxy()
            .build()
            .ok()?
            .get(&url)
            .send()
            .ok()?;
        let status = response.status().as_u16();
        Some((status, response.text().ok()?))
    })
    .join()
    .ok()
    .flatten()
}

fn supervisor_stub_publish(project_dir: &str, pid: u32, port: u16) {
    std::fs::create_dir_all(project_dir).expect("project dir");
    std::fs::write(std::path::Path::new(project_dir).join("agent.pid"), pid.to_string()).unwrap();
    std::fs::write(std::path::Path::new(project_dir).join("agent.port"), port.to_string()).unwrap();
}

fn supervisor_stub_block() -> ! {
    loop {
        std::thread::park();
    }
}

fn supervisor_stub_serve(listener: std::net::TcpListener) {
    use std::io::{BufRead, BufReader, Read, Write};

    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            std::thread::spawn(move || {
                let mut reader = BufReader::new(stream);
                let mut request_line = String::new();
                if reader.read_line(&mut request_line).is_err() {
                    return;
                }
                let mut length = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        break;
                    }
                    let line = line.trim_end();
                    if line.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = line.split_once(':') {
                        if name.trim().eq_ignore_ascii_case("content-length") {
                            length = value.trim().parse().unwrap_or(0);
                        }
                    }
                }
                let mut parts = request_line.split_whitespace();
                let method = parts.next().unwrap_or("").to_string();
                let path = parts.next().unwrap_or("").to_string();
                let body = if method == "POST" {
                    let mut body = vec![0u8; length];
                    let _ = reader.read_exact(&mut body);
                    body
                } else {
                    format!("{{\"status\": \"healthy\", \"path\": \"{path}\"}}").into_bytes()
                };
                let mut stream = reader.into_inner();
                let head = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(&body);
                let _ = stream.flush();
            });
        }
    });
}

fn supervisor_stub_probe(path: &str) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        use std::os::unix::io::FromRawFd;

        // SAFETY: plain libc queries about this process.
        let (sid, pgid, pid) = unsafe { (libc::getsid(0), libc::getpgid(0), libc::getpid()) };
        // SAFETY: fd 0 is borrowed only for fstat and never closed.
        let stdin = std::mem::ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(0) });
        let devnull = std::fs::OpenOptions::new().read(true).write(true).open("/dev/null").unwrap();
        let stdin_devnull = stdin.metadata().map(|m| m.rdev()).ok() == devnull.metadata().map(|m| m.rdev()).ok();
        let mut fds: Vec<i64> = std::fs::read_dir("/dev/fd")
            .unwrap()
            .filter_map(|e| e.ok()?.file_name().to_str()?.parse().ok())
            .collect();
        fds.sort();
        let payload = serde_json::json!({
            "sid": sid, "pgid": pgid, "pid": pid,
            "stdin_devnull": stdin_devnull, "fds": fds,
        });
        std::fs::write(path, payload.to_string()).unwrap();
        drop(devnull);
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
}

/// The stub process entry point (see [`supervisor_stub_command`]).  A no-op
/// when run without a stub argument.
#[test]
#[ignore = "helper process for the supervisor tests; runs only when re-executed"]
fn supervisor_stub_entry() {
    let Some(spec) = std::env::args().find_map(|a| a.strip_prefix(SUPERVISOR_STUB_PREFIX).map(str::to_string))
    else {
        return;
    };
    let (mode, arg) = spec.split_once(':').unwrap_or((spec.as_str(), ""));
    match mode {
        "agent" => {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            supervisor_stub_serve(listener);
            supervisor_stub_publish(arg, std::process::id(), port);
            supervisor_stub_block();
        }
        "wedged" => {
            let port = {
                let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
                listener.local_addr().unwrap().port()
            };
            supervisor_stub_publish(arg, std::process::id(), port);
            supervisor_stub_block();
        }
        "silent" => {
            supervisor_stub_publish(arg, std::process::id(), 1);
            supervisor_stub_block();
        }
        "probe" => {
            supervisor_stub_probe(arg);
            std::process::exit(0);
        }
        "sleep" => supervisor_stub_block(),
        "exit" => std::process::exit(arg.parse().unwrap_or(1)),
        other => panic!("unknown stub mode {other}"),
    }
}

// ===== END unit: supervisor =======================================================

// ===== agents_core (begin) =====
pub mod agents_core;
// ===== agents_core (end) =====

// ============================================================ agents_mcp_explorer
// Fixtures added by the `agents_mcp_explorer` unit.  Other units: add yours in
// your own delimited block rather than editing this one.

/// Fakes for driving `smartloop::agents::mcp::explorer::Explorer` without a
/// model, a network or an MCP transport (Python: `Explorer.__new__(Explorer)`
/// with the collaborators set by hand, plus `MagicMock`s).
pub mod explorer_fixtures {
    use std::any::Any;
    use std::collections::HashMap;
    use std::sync::Arc;

    use parking_lot::Mutex;
    use serde_json::{Map, Value};

    use smartloop::agents::fetch_agent::FetchAgent;
    use smartloop::agents::mcp::argument_extractor::MCPArgumentExtractor;
    use smartloop::agents::mcp::explorer::{
        ConversationTurn, DocumentGroups, Explorer, ExplorerSearchService, FollowUpPlanner,
        ProgressCallback, ToolResolver, WebSearcher,
    };
    use smartloop::agents::mcp::mcp_agent::ResolvedCall;
    use smartloop::agents::mcp::mcp_tool_caller::MCPToolCaller;
    use smartloop::agents::mcp::result_mapper::MCPResultMapper;
    use smartloop::llms::base_llm::{CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream};
    use smartloop::llms::BaseLLM;
    use smartloop::mcp::client::MCPToolCallResult;
    use smartloop::parsers::DocumentChunk;
    use smartloop::workspace::workspace::MCPServer;

    /// One `progress(step, status, message)` call.
    pub type ProgressEvent = (String, String, String);

    /// A model that answers every `invoke` with a fixed reply and records the
    /// prompts it was given.
    pub struct ScriptedLLM {
        pub reply: String,
        pub prompts: Mutex<Vec<String>>,
    }

    impl ScriptedLLM {
        pub fn replying(reply: &str) -> Arc<Self> {
            Arc::new(Self { reply: reply.to_string(), prompts: Mutex::new(Vec::new()) })
        }
    }

    impl BaseLLM for ScriptedLLM {
        fn project_id(&self) -> &str {
            ""
        }
        fn get_num_tokens(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }
        fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
            messages
        }
        fn get_max_tokens(&self) -> usize {
            4096
        }
        fn call(&self, prompt: &str, _stop: Option<&[String]>, _options: CallOptions) -> anyhow::Result<String> {
            self.prompts.lock().push(prompt.to_string());
            Ok(self.reply.clone())
        }
        fn stream<'a>(&'a self, prompt: &str, _options: CallOptions) -> GenerationStream<'a> {
            self.prompts.lock().push(prompt.to_string());
            Box::new(std::iter::once(GenerationChunk { text: self.reply.clone() }))
        }
        fn invoke(&self, prompt: &str, _max_tokens: Option<usize>, _thinking: Option<bool>) -> String {
            self.prompts.lock().push(prompt.to_string());
            self.reply.clone()
        }
        fn cancelled(&self) -> Option<CancelEvent> {
            None
        }
        fn set_cancelled(&self, _event: Option<CancelEvent>) {}
        fn llm_type(&self) -> &str {
            "scripted"
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// The calls a stubbed transport saw: `(server name, tool, arguments)`.
    pub type TransportLog = Arc<Mutex<Vec<(String, String, Map<String, Value>)>>>;

    /// `MCPToolCaller(None, None)` with its transport replaced by `answer`
    /// (Python: `caller.call_tool = MagicMock(...)`).  Every call is logged.
    pub fn stubbed_caller(
        answer: impl Fn(&MCPServer, &str, &Map<String, Value>) -> MCPToolCallResult + Send + Sync + 'static,
    ) -> (Arc<MCPToolCaller>, TransportLog) {
        let llm: Arc<dyn BaseLLM> = ScriptedLLM::replying("");
        let caller = MCPToolCaller::new(
            Arc::new(MCPArgumentExtractor::new(llm)),
            Arc::new(Mutex::new(MCPResultMapper::new())),
            None,
        );
        let log: TransportLog = Arc::new(Mutex::new(Vec::new()));
        let seen = log.clone();
        caller.set_call_tool_hook(Some(Arc::new(move |server, tool, arguments, _token, _cwd| {
            let arguments = arguments.cloned().unwrap_or_default();
            seen.lock().push((server.name.clone(), tool.to_string(), arguments.clone()));
            answer(server, tool, &arguments)
        })));
        (Arc::new(caller), log)
    }

    /// A successful result carrying one text block.
    pub fn text_result(text: &str) -> MCPToolCallResult {
        MCPToolCallResult {
            success: true,
            content: Some(vec![serde_json::json!({"type": "text", "text": text})]),
            ..Default::default()
        }
    }

    /// `mcp_agent.resolve.return_value = ...`; counts its calls.
    pub struct ScriptedResolver {
        pub answer: Option<ResolvedCall>,
        pub calls: Mutex<usize>,
    }

    impl ScriptedResolver {
        pub fn returning(answer: Option<ResolvedCall>) -> Arc<Self> {
            Arc::new(Self { answer, calls: Mutex::new(0) })
        }
    }

    impl ToolResolver for ScriptedResolver {
        fn resolve(
            &self,
            _directive: &str,
            _mcp_servers: &[MCPServer],
            _context: Option<&str>,
            _llm: Option<&dyn BaseLLM>,
        ) -> Option<ResolvedCall> {
            *self.calls.lock() += 1;
            self.answer.clone()
        }
    }

    /// A web search agent double (`MagicMock()` with `is_enabled`,
    /// `last_sources` and `search` set); records each search and the
    /// progress sink it was handed.
    pub struct FakeWebSearch {
        pub enabled: bool,
        pub result: String,
        pub sources: Vec<String>,
        pub searched_with: Mutex<Vec<(String, Option<ProgressCallback>)>>,
    }

    impl FakeWebSearch {
        pub fn new(enabled: bool, result: &str, sources: &[&str]) -> Arc<Self> {
            Arc::new(Self {
                enabled,
                result: result.to_string(),
                sources: sources.iter().map(|s| s.to_string()).collect(),
                searched_with: Mutex::new(Vec::new()),
            })
        }
    }

    impl WebSearcher for FakeWebSearch {
        fn is_enabled(&self) -> bool {
            self.enabled
        }
        fn search(&self, query: &str, progress: Option<ProgressCallback>) -> anyhow::Result<String> {
            self.searched_with.lock().push((query.to_string(), progress));
            Ok(self.result.clone())
        }
        fn last_sources(&self) -> Vec<String> {
            self.sources.clone()
        }
        fn last_source_extracts(&self) -> HashMap<String, String> {
            HashMap::new()
        }
    }

    /// A model service whose searches all come back empty (Python patched
    /// `_search_vector` → `None`, `_search_feedback` / `_search_workspace` →
    /// `""`).
    #[derive(Default)]
    pub struct EmptySearch;

    impl ExplorerSearchService for EmptySearch {
        fn get_document_groups(
            &self,
            _query: Option<&str>,
            _skill_instruction: Option<&str>,
            _conversation_history: Option<&[ConversationTurn]>,
        ) -> anyhow::Result<Option<DocumentGroups>> {
            Ok(None)
        }
        fn search_feedback(&self, _query: &str, _n_results: usize) -> anyhow::Result<Vec<DocumentChunk>> {
            Ok(Vec::new())
        }
        fn search_workspace(
            &self,
            _query: &str,
            _work_dir: &str,
            _n_results: usize,
            _conversation_history: Option<&[ConversationTurn]>,
        ) -> anyhow::Result<Vec<DocumentChunk>> {
            Ok(Vec::new())
        }
    }

    /// Follow-up planner returning `calls` whatever it is asked
    /// (`monkeypatch.setattr("smartloop.mcp.call_plans.follow_up_calls", ...)`).
    pub fn planned(calls: Vec<(&'static str, Value)>) -> FollowUpPlanner {
        Arc::new(move |_, _, _| {
            calls
                .iter()
                .map(|(tool, args)| (tool.to_string(), args.as_object().cloned().unwrap_or_default()))
                .collect()
        })
    }

    /// An explorer with every collaborator injected and no model unless one
    /// is given.
    pub fn bare_explorer(
        llm: Option<Arc<dyn BaseLLM>>,
        tool_caller: Option<Arc<MCPToolCaller>>,
        resolver: Arc<dyn ToolResolver>,
        web_search: Arc<dyn WebSearcher>,
        follow_up_calls: FollowUpPlanner,
    ) -> Explorer {
        Explorer {
            llm,
            tool_caller,
            arg_extractor: None,
            use_llm_history: false,
            fetch_agent: Arc::new(FetchAgent::new()),
            web_search_agent: web_search,
            mcp_agent: resolver,
            follow_up_calls,
        }
    }

    /// A progress sink recording every event.
    pub fn recording_sink() -> (ProgressCallback, Arc<Mutex<Vec<ProgressEvent>>>) {
        let events: Arc<Mutex<Vec<ProgressEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let seen = events.clone();
        let sink: ProgressCallback = Arc::new(move |step: &str, status: &str, message: &str| {
            seen.lock().push((step.to_string(), status.to_string(), message.to_string()));
        });
        (sink, events)
    }

    /// The messages of the recorded events (`c.args[2]` per call).
    pub fn messages(events: &Arc<Mutex<Vec<ProgressEvent>>>) -> Vec<String> {
        events.lock().iter().map(|(_, _, m)| m.clone()).collect()
    }
}
// =========================================================== /agents_mcp_explorer

// ===================================================================== agents_mcp_args
// Fixtures added by the `agents_mcp_args` unit.
pub mod agents_mcp_args {
    //! A scripted stand-in for `MagicMock()` LLMs: `get_num_tokens` returns a
    //! fixed count and `invoke` replays queued replies (then a default).

    use std::any::Any;
    use std::collections::VecDeque;

    use parking_lot::Mutex;
    use smartloop::llms::base_llm::{CallOptions, CancelEvent, ChatMessage, GenerationStream};
    use smartloop::llms::BaseLLM;

    /// `MagicMock` LLM: `get_num_tokens -> num_tokens`, `invoke -> next reply`.
    pub struct ScriptedLLM {
        replies: Mutex<VecDeque<String>>,
        default_reply: String,
        num_tokens: usize,
        /// Every prompt `invoke` received, in order.
        pub prompts: Mutex<Vec<String>>,
        cancelled: Mutex<Option<CancelEvent>>,
    }

    impl ScriptedLLM {
        /// An LLM whose every `invoke` returns `reply`.
        pub fn new(reply: impl Into<String>) -> Self {
            Self {
                replies: Mutex::new(VecDeque::new()),
                default_reply: reply.into(),
                num_tokens: 10,
                prompts: Mutex::new(Vec::new()),
                cancelled: Mutex::new(None),
            }
        }

        /// An LLM that replays `replies` in order, then returns `""`.
        pub fn with_replies<I, S>(replies: I) -> Self
        where
            I: IntoIterator<Item = S>,
            S: Into<String>,
        {
            let llm = Self::new("");
            *llm.replies.lock() = replies.into_iter().map(Into::into).collect();
            llm
        }

        /// How many times `invoke` was called.
        pub fn invoke_count(&self) -> usize {
            self.prompts.lock().len()
        }
    }

    impl BaseLLM for ScriptedLLM {
        fn project_id(&self) -> &str {
            "test"
        }
        fn get_num_tokens(&self, _text: &str) -> usize {
            self.num_tokens
        }
        fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
            messages
        }
        fn get_max_tokens(&self) -> usize {
            8192
        }
        fn call(&self, prompt: &str, _stop: Option<&[String]>, _options: CallOptions) -> anyhow::Result<String> {
            Ok(self.invoke(prompt, None, None))
        }
        fn stream<'a>(&'a self, prompt: &str, _options: CallOptions) -> GenerationStream<'a> {
            let text = self.invoke(prompt, None, None);
            Box::new(std::iter::once(smartloop::llms::base_llm::GenerationChunk { text }))
        }
        fn invoke(&self, prompt: &str, _max_tokens: Option<usize>, _thinking: Option<bool>) -> String {
            self.prompts.lock().push(prompt.to_string());
            self.replies
                .lock()
                .pop_front()
                .unwrap_or_else(|| self.default_reply.clone())
        }
        fn cancelled(&self) -> Option<CancelEvent> {
            self.cancelled.lock().clone()
        }
        fn set_cancelled(&self, event: Option<CancelEvent>) {
            *self.cancelled.lock() = event;
        }
        fn llm_type(&self) -> &str {
            "scripted"
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
}
// ================================================================= end agents_mcp_args

// ======================================================================
// BEGIN agents_mcp_agent fixtures
// ======================================================================

/// Shared fixtures for the MCP agent / tool caller tests (unit
/// `agents_mcp_agent`): a scriptable LLM standing in for Python's
/// `MagicMock()` handles.
pub mod agents_mcp_agent {
    use std::any::Any;
    use std::sync::Arc;

    use parking_lot::Mutex;
    use smartloop::llms::base_llm::{
        BaseLLM, CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream,
    };

    /// `(prompt, max_tokens, thinking) -> reply`.
    pub type ReplyFn = Arc<dyn Fn(&str, Option<usize>, Option<bool>) -> String + Send + Sync>;

    /// One recorded `invoke` call (Python's `call_args`).
    #[derive(Debug, Clone, PartialEq)]
    pub struct InvokeCall {
        pub prompt: String,
        pub max_tokens: Option<usize>,
        pub thinking: Option<bool>,
    }

    /// A scriptable LLM.
    ///
    /// `get_max_tokens` defaults to 0 — "no window reported" — which is what a
    /// `MagicMock` with non-numeric token arithmetic amounted to in Python: the
    /// budget helpers fall back to what they asked for.
    pub struct StubLLM {
        reply: Mutex<ReplyFn>,
        calls: Mutex<Vec<InvokeCall>>,
        supports_thinking: Mutex<bool>,
        model_label: Mutex<String>,
        max_tokens: Mutex<usize>,
        num_tokens: Mutex<usize>,
        cancelled: Mutex<Option<CancelEvent>>,
    }

    impl StubLLM {
        /// An LLM that always answers `reply`.
        pub fn new(reply: &str) -> Arc<Self> {
            let reply = reply.to_string();
            Self::with_reply_fn(Arc::new(move |_, _, _| reply.clone()))
        }

        /// An LLM whose answers come from `reply`.
        pub fn with_reply_fn(reply: ReplyFn) -> Arc<Self> {
            Arc::new(Self {
                reply: Mutex::new(reply),
                calls: Mutex::new(Vec::new()),
                supports_thinking: Mutex::new(false),
                model_label: Mutex::new(String::new()),
                max_tokens: Mutex::new(0),
                num_tokens: Mutex::new(0),
                cancelled: Mutex::new(None),
            })
        }

        /// A labelled LLM (Python's `llm.model.metadata["general.name"]`).
        pub fn labelled(label: &str) -> Arc<Self> {
            let stub = Self::new("");
            stub.set_model_label(label);
            stub
        }

        pub fn set_reply(&self, reply: &str) {
            let reply = reply.to_string();
            *self.reply.lock() = Arc::new(move |_, _, _| reply.clone());
        }

        pub fn set_reply_fn(&self, reply: ReplyFn) {
            *self.reply.lock() = reply;
        }

        pub fn set_supports_thinking(&self, value: bool) {
            *self.supports_thinking.lock() = value;
        }

        pub fn set_model_label(&self, label: &str) {
            *self.model_label.lock() = label.to_string();
        }

        pub fn set_max_tokens(&self, value: usize) {
            *self.max_tokens.lock() = value;
        }

        pub fn set_num_tokens(&self, value: usize) {
            *self.num_tokens.lock() = value;
        }

        /// Every `invoke` so far, oldest first.
        pub fn calls(&self) -> Vec<InvokeCall> {
            self.calls.lock().clone()
        }

        pub fn call_count(&self) -> usize {
            self.calls.lock().len()
        }

        /// The most recent `invoke` (Python's `call_args`); panics if none.
        pub fn last_call(&self) -> InvokeCall {
            self.calls.lock().last().cloned().expect("invoke was called")
        }

        /// This stub as the trait object the code under test takes.
        pub fn as_llm(self: &Arc<Self>) -> Arc<dyn BaseLLM> {
            self.clone()
        }
    }

    impl BaseLLM for StubLLM {
        fn project_id(&self) -> &str {
            "test-project"
        }

        fn get_num_tokens(&self, _text: &str) -> usize {
            *self.num_tokens.lock()
        }

        fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
            messages
        }

        fn get_max_tokens(&self) -> usize {
            *self.max_tokens.lock()
        }

        fn call(&self, prompt: &str, _stop: Option<&[String]>, options: CallOptions) -> anyhow::Result<String> {
            Ok(self.invoke(prompt, options.max_tokens, options.thinking))
        }

        fn stream<'a>(&'a self, prompt: &str, options: CallOptions) -> GenerationStream<'a> {
            let text = self.invoke(prompt, options.max_tokens, options.thinking);
            Box::new(std::iter::once(GenerationChunk { text }))
        }

        fn invoke(&self, prompt: &str, max_tokens: Option<usize>, thinking: Option<bool>) -> String {
            self.calls.lock().push(InvokeCall {
                prompt: prompt.to_string(),
                max_tokens,
                thinking,
            });
            let reply = self.reply.lock().clone();
            reply(prompt, max_tokens, thinking)
        }

        fn supports_thinking(&self) -> bool {
            *self.supports_thinking.lock()
        }

        fn model_label(&self) -> String {
            self.model_label.lock().clone()
        }

        fn cancelled(&self) -> Option<CancelEvent> {
            self.cancelled.lock().clone()
        }

        fn set_cancelled(&self, event: Option<CancelEvent>) {
            *self.cancelled.lock() = event;
        }

        fn llm_type(&self) -> &str {
            "StubLLM"
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Whether two handles are the same model (Python's `is`).
    pub fn same_llm(a: &Arc<dyn BaseLLM>, b: &Arc<dyn BaseLLM>) -> bool {
        std::ptr::eq(Arc::as_ptr(a) as *const (), Arc::as_ptr(b) as *const ())
    }
}

// ======================================================================
// END agents_mcp_agent fixtures
// ======================================================================

// ===================== BEGIN orch_exec =====================
/// Fixtures shared by the orch_exec unit's tests
/// (`test_mcp_dead_grant`, `test_mcp_resolution_fills_arguments`).
pub mod orch_exec {
    use std::any::Any;
    use std::sync::Arc;

    use parking_lot::Mutex;

    use smartloop::llms::base_llm::{CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream};
    use smartloop::llms::BaseLLM;

    /// Stand-in for the `MagicMock()` LLMs the Python tests pass around: it
    /// answers every prompt with a fixed reply and records what it was asked.
    pub struct StubLLM {
        pub reply: String,
        pub prompts: Mutex<Vec<String>>,
        cancelled: Mutex<Option<CancelEvent>>,
    }

    impl StubLLM {
        /// A stub that replies with `reply` to every prompt.
        pub fn new(reply: impl Into<String>) -> Self {
            Self { reply: reply.into(), prompts: Mutex::new(Vec::new()), cancelled: Mutex::new(None) }
        }

        /// The stub as the trait object every service takes.
        pub fn shared(reply: impl Into<String>) -> Arc<dyn BaseLLM> {
            Arc::new(Self::new(reply))
        }
    }

    impl BaseLLM for StubLLM {
        fn project_id(&self) -> &str {
            "test"
        }
        fn get_num_tokens(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }
        fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
            messages
        }
        fn get_max_tokens(&self) -> usize {
            4096
        }
        fn call(&self, prompt: &str, _stop: Option<&[String]>, _options: CallOptions) -> anyhow::Result<String> {
            self.prompts.lock().push(prompt.to_string());
            Ok(self.reply.clone())
        }
        fn stream<'a>(&'a self, prompt: &str, _options: CallOptions) -> GenerationStream<'a> {
            self.prompts.lock().push(prompt.to_string());
            Box::new(std::iter::once(GenerationChunk { text: self.reply.clone() }))
        }
        fn invoke(&self, prompt: &str, _max_tokens: Option<usize>, _thinking: Option<bool>) -> String {
            self.prompts.lock().push(prompt.to_string());
            self.reply.clone()
        }
        fn cancelled(&self) -> Option<CancelEvent> {
            self.cancelled.lock().clone()
        }
        fn set_cancelled(&self, event: Option<CancelEvent>) {
            *self.cancelled.lock() = event;
        }
        fn llm_type(&self) -> &str {
            "stub"
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
}
// ====================== END orch_exec ======================

// ===================================================================== //
// BEGIN orch_steps fixtures                                             //
// ===================================================================== //

/// LLM test doubles for the orchestration-step tests (stand-ins for the
/// Python tests' `MagicMock()` LLMs whose `model.metadata` was set).
pub mod orch_steps {
    use std::any::Any;
    use std::collections::HashMap;
    use std::sync::Arc;

    use parking_lot::Mutex;

    use smartloop::llms::base_llm::{
        CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream,
    };
    use smartloop::llms::llama::{
        ChatCompletionOutput, ChatCompletionRequest, LlamaModel, LlamaState, LlamaToken,
    };
    use smartloop::llms::BaseLLM;

    /// A llama.cpp model whose only readable surface is its GGUF metadata.
    pub struct MetadataModel {
        pub metadata: HashMap<String, String>,
        state: LlamaState,
    }

    impl MetadataModel {
        pub fn new(metadata: HashMap<String, String>) -> Self {
            Self { metadata, state: LlamaState::default() }
        }
    }

    impl LlamaModel for MetadataModel {
        fn state(&self) -> &LlamaState {
            &self.state
        }

        fn n_ctx(&self) -> usize {
            0
        }

        fn metadata(&self) -> HashMap<String, String> {
            self.metadata.clone()
        }

        fn tokenize(&self, _text: &[u8]) -> anyhow::Result<Vec<LlamaToken>> {
            Ok(Vec::new())
        }

        fn detokenize(&self, _tokens: &[LlamaToken]) -> anyhow::Result<Vec<u8>> {
            Ok(Vec::new())
        }

        fn create_chat_completion(
            &self,
            _request: ChatCompletionRequest,
        ) -> anyhow::Result<ChatCompletionOutput<'_>> {
            anyhow::bail!("MetadataModel cannot generate")
        }
    }

    /// Reply strategy of a [`MetadataLlm`]'s `invoke`.
    pub type InvokeFn = Arc<dyn Fn(&str, Option<usize>) -> String + Send + Sync>;

    /// An LLM double: optional GGUF metadata (no metadata = the model is
    /// unreadable, like a mock whose `model` property raises), a scripted
    /// `invoke`, and a record of every prompt it was invoked with.
    pub struct MetadataLlm {
        pub model: Option<Arc<MetadataModel>>,
        pub label: String,
        pub max_tokens: usize,
        pub num_tokens: usize,
        pub thinking: bool,
        pub reply: Mutex<InvokeFn>,
        pub prompts: Mutex<Vec<String>>,
        cancelled: Mutex<Option<CancelEvent>>,
    }

    impl MetadataLlm {
        /// A model whose GGUF reports `general.name` / `general.size_label`.
        pub fn named(name: &str, size: &str) -> Self {
            let mut meta = HashMap::new();
            meta.insert("general.name".to_string(), name.to_string());
            meta.insert("general.size_label".to_string(), size.to_string());
            Self::with_model(Some(Arc::new(MetadataModel::new(meta))))
        }

        /// A model whose metadata cannot be read at all.
        pub fn unreadable() -> Self {
            Self::with_model(None)
        }

        fn with_model(model: Option<Arc<MetadataModel>>) -> Self {
            Self {
                model,
                label: String::new(),
                max_tokens: 8192,
                num_tokens: 100,
                thinking: false,
                reply: Mutex::new(Arc::new(|_p: &str, _m: Option<usize>| "OK".to_string())),
                prompts: Mutex::new(Vec::new()),
                cancelled: Mutex::new(None),
            }
        }

        /// Script `invoke` (Python `llm.invoke = MagicMock(side_effect=...)`).
        pub fn set_reply(&self, reply: impl Fn(&str, Option<usize>) -> String + Send + Sync + 'static) {
            *self.reply.lock() = Arc::new(reply);
        }

        /// Number of `invoke` calls so far.
        pub fn invoke_count(&self) -> usize {
            self.prompts.lock().len()
        }
    }

    impl BaseLLM for MetadataLlm {
        fn project_id(&self) -> &str {
            ""
        }

        fn get_num_tokens(&self, _text: &str) -> usize {
            self.num_tokens
        }

        fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
            messages
        }

        fn get_max_tokens(&self) -> usize {
            self.max_tokens
        }

        fn call(&self, prompt: &str, _stop: Option<&[String]>, options: CallOptions) -> anyhow::Result<String> {
            Ok(self.invoke(prompt, options.max_tokens, options.thinking))
        }

        fn stream<'a>(&'a self, prompt: &str, options: CallOptions) -> GenerationStream<'a> {
            let text = self.invoke(prompt, options.max_tokens, options.thinking);
            Box::new(std::iter::once(GenerationChunk { text }))
        }

        fn invoke(&self, prompt: &str, max_tokens: Option<usize>, _thinking: Option<bool>) -> String {
            self.prompts.lock().push(prompt.to_string());
            let reply = self.reply.lock().clone();
            reply(prompt, max_tokens)
        }

        fn supports_thinking(&self) -> bool {
            self.thinking
        }

        fn model_label(&self) -> String {
            self.label.clone()
        }

        fn cancelled(&self) -> Option<CancelEvent> {
            self.cancelled.lock().clone()
        }

        fn set_cancelled(&self, event: Option<CancelEvent>) {
            *self.cancelled.lock() = event;
        }

        fn llm_type(&self) -> &str {
            "metadata-llm"
        }

        fn llama_model(&self) -> Option<Arc<dyn LlamaModel>> {
            self.model.clone().map(|m| m as Arc<dyn LlamaModel>)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// A progress sink that records every `(step, status, message)` it gets
    /// (Python `router.begin_request(MagicMock())`).
    #[derive(Clone, Default)]
    pub struct ProgressRecorder {
        pub calls: Arc<Mutex<Vec<(String, String, String)>>>,
    }

    impl ProgressRecorder {
        pub fn new() -> Self {
            Self::default()
        }

        /// The callback to hand to `ModelRouter::begin_request`.
        pub fn callback(&self) -> smartloop::agents::orchestration::model_router::ProgressCallback {
            let calls = self.calls.clone();
            Arc::new(move |step: &str, status: &str, message: &str| {
                calls.lock().push((step.to_string(), status.to_string(), message.to_string()));
            })
        }

        /// The message argument of every call, in order.
        pub fn messages(&self) -> Vec<String> {
            self.calls.lock().iter().map(|(_, _, m)| m.clone()).collect()
        }
    }
}

// ===================================================================== //
// END orch_steps fixtures                                               //
// ===================================================================== //

// ============================================================================
// api_core — `_reset_project_pin`.
// ============================================================================
#[allow(dead_code)]
pub mod api_core {
    use smartloop::workspace::context::pin_project;

    /// Unpin the process around a test (`_reset_project_pin`).
    ///
    /// The pin is process-wide by design — an agent child sets it once at
    /// spawn — so a test that pins and does not clean up leaks into every test
    /// that runs after it.  That failure is quiet in the worst way: a leaked
    /// pin makes the workspace-owner checks answer false, so assertions about
    /// "did not do the owner's work" keep passing for the wrong reason.
    pub struct ProjectPinReset;

    impl ProjectPinReset {
        pub fn new() -> Self {
            pin_project(None);
            ProjectPinReset
        }
    }

    impl Default for ProjectPinReset {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Drop for ProjectPinReset {
        fn drop(&mut self) {
            if !std::thread::panicking() {
                pin_project(None);
            } else {
                let _ = std::panic::catch_unwind(|| pin_project(None));
            }
        }
    }
}
// ============================================================================
// end api_core
// ============================================================================

// ============================================================ api_chat (begin)
// Shared by the chat-router tests (tests/unit/test_stream_*.rs,
// test_orchestrate_timeout_unload.rs): a `ModelService` stand-in, the
// Python tests' `MagicMock` service, with the real `InProcessModelClient` put
// in front of it so the tests drive exactly the path production takes.
pub mod api_chat {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use parking_lot::{Condvar, Mutex};

    use smartloop::api::routers::chat::{
        ChatService, CitationDict, HistoryTurn, ImageAttachment, OrchestrateArgs,
        OrchestrateOutcome, ProgressCallback,
    };
    use smartloop::llms::base_llm::{CancelEvent, GenerationChunk};
    use smartloop::model_host::client::{InProcessModelClient, LocalModelService, ModelClient};
    use smartloop::types::ModelCapability;
    use smartloop::workspace::types::Skill;

    /// Python's `threading.Event`.
    #[derive(Clone, Default)]
    pub struct Event {
        inner: Arc<(Mutex<bool>, Condvar)>,
    }

    impl Event {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn set(&self) {
            let (flag, cv) = &*self.inner;
            *flag.lock() = true;
            cv.notify_all();
        }

        pub fn is_set(&self) -> bool {
            *self.inner.0.lock()
        }

        /// Block until set or `timeout_s` elapses; returns whether it is set.
        pub fn wait(&self, timeout_s: f64) -> bool {
            let (flag, cv) = &*self.inner;
            let deadline = Instant::now() + Duration::from_secs_f64(timeout_s);
            let mut guard = flag.lock();
            while !*guard {
                if cv.wait_until(&mut guard, deadline).timed_out() {
                    break;
                }
            }
            *guard
        }
    }

    /// Runs a callback when dropped — a generator's `finally:` block.
    pub struct OnDrop(Option<Box<dyn FnOnce() + Send>>);

    impl OnDrop {
        pub fn new(f: impl FnOnce() + Send + 'static) -> Self {
            Self(Some(Box::new(f)))
        }
    }

    impl Drop for OnDrop {
        fn drop(&mut self) {
            if let Some(f) = self.0.take() {
                f();
            }
        }
    }

    /// `GenerationChunk(text=...)`.
    pub fn chunk(text: &str) -> GenerationChunk {
        GenerationChunk { text: text.to_string() }
    }

    /// The chunks a fake `stream_completion` produces.
    pub type ChunkIter = Box<dyn Iterator<Item = anyhow::Result<GenerationChunk>>>;

    /// A fake `stream_completion(prompt, cancelled=...)`: a generator function
    /// — nothing in the body runs until the returned iterator is advanced.
    pub type StreamFn = Arc<dyn Fn(&str, Option<CancelEvent>) -> ChunkIter + Send + Sync>;

    /// The device-owning half of the fake: what `InProcessModelClient` calls.
    pub struct FakeDevice {
        pub stream_fn: Mutex<Option<StreamFn>>,
        /// `generate_completion.return_value`.
        pub generate_return: Mutex<String>,
        /// Prompts `generate_completion` was called with.
        pub generate_prompts: Mutex<Vec<String>>,
        /// `include_embedding` of every `unload_auxiliary_models` call.
        pub unload_calls: Mutex<Vec<bool>>,
        /// `unload_auxiliary_models.side_effect`: fail with this message.
        pub unload_failure: Mutex<Option<String>>,
        pub unload_embedding_calls: Mutex<usize>,
    }

    impl FakeDevice {
        fn record_unload(&self, include_embedding: bool) {
            self.unload_calls.lock().push(include_embedding);
            let failure = self.unload_failure.lock().clone();
            if let Some(message) = failure {
                panic!("{message}");
            }
        }
    }

    impl LocalModelService for FakeDevice {
        fn generate_completion(
            &self,
            prompt: &str,
            _model_name: Option<&str>,
            _capability: Option<&str>,
            _max_tokens: Option<usize>,
            _thinking: Option<bool>,
        ) -> anyhow::Result<String> {
            self.generate_prompts.lock().push(prompt.to_string());
            Ok(self.generate_return.lock().clone())
        }

        fn stream_completion<'a>(
            &'a self,
            prompt: &str,
            _model_name: Option<&str>,
            cancelled: Option<CancelEvent>,
            _capability: Option<&str>,
            _thinking: Option<bool>,
        ) -> anyhow::Result<Box<dyn Iterator<Item = anyhow::Result<GenerationChunk>> + 'a>> {
            let stream_fn = self.stream_fn.lock().clone();
            match stream_fn {
                Some(f) => Ok(f(prompt, cancelled)),
                None => Ok(Box::new(std::iter::empty())),
            }
        }

        fn is_loaded(&self) -> bool {
            true
        }

        fn resident_model_name(&self) -> Option<String> {
            Some("sl-mini".to_string())
        }

        fn response_model_label(&self, _capability: Option<&str>) -> String {
            "sl-mini".to_string()
        }

        fn count_tokens(&self, _text: &str) -> usize {
            10
        }

        fn get_max_context_tokens(&self) -> usize {
            4096
        }

        fn get_response_max_context_tokens(&self, _capability: Option<&str>) -> usize {
            4096
        }

        fn has_capability_model(&self, _capability: &str) -> bool {
            false
        }

        fn idle_seconds(&self) -> f64 {
            0.0
        }

        fn load_model(&self) -> anyhow::Result<()> {
            Ok(())
        }

        fn unload_auxiliary_models(&self, include_embedding: bool) {
            self.record_unload(include_embedding);
        }

        fn unload_embedding_model(&self) {
            *self.unload_embedding_calls.lock() += 1;
        }
    }

    /// The `MagicMock` `ModelService` of the chat tests.
    pub struct FakeChatService {
        pub device: Arc<FakeDevice>,
        /// `orchestrate.return_value`.
        pub orchestrate_outcome: Mutex<OrchestrateOutcome>,
        /// User messages `orchestrate` was called with.
        pub orchestrate_calls: Mutex<Vec<String>>,
        /// `_model.workspace.workspace_dir`.
        pub workspace_dir: Mutex<Option<String>>,
        /// `(session_id, role, content)` of every `append_to_session`.
        pub session_writes: Mutex<Vec<(String, String, String)>>,
    }

    impl Default for FakeChatService {
        fn default() -> Self {
            Self {
                device: Arc::new(FakeDevice {
                    stream_fn: Mutex::new(None),
                    generate_return: Mutex::new(String::new()),
                    generate_prompts: Mutex::new(Vec::new()),
                    unload_calls: Mutex::new(Vec::new()),
                    unload_failure: Mutex::new(None),
                    unload_embedding_calls: Mutex::new(0),
                }),
                orchestrate_outcome: Mutex::new(OrchestrateOutcome::default()),
                orchestrate_calls: Mutex::new(Vec::new()),
                workspace_dir: Mutex::new(None),
                session_writes: Mutex::new(Vec::new()),
            }
        }
    }

    impl FakeChatService {
        /// A service whose `stream_completion` is `stream_fn`.
        pub fn with_stream<F>(stream_fn: F) -> Self
        where
            F: Fn(&str, Option<CancelEvent>) -> ChunkIter + Send + Sync + 'static,
        {
            let svc = Self::default();
            *svc.device.stream_fn.lock() = Some(Arc::new(stream_fn));
            svc
        }

        /// `include_embedding` of every `unload_auxiliary_models` call.
        pub fn unload_calls(&self) -> Vec<bool> {
            self.device.unload_calls.lock().clone()
        }

        /// `unload_auxiliary_models.call_count`.
        pub fn unload_call_count(&self) -> usize {
            self.device.unload_calls.lock().len()
        }

        /// Prompts `generate_completion` was called with.
        pub fn generate_prompts(&self) -> Vec<String> {
            self.device.generate_prompts.lock().clone()
        }
    }

    impl ChatService for FakeChatService {
        fn is_loaded(&self) -> bool {
            true
        }
        fn load_model(&self) -> anyhow::Result<()> {
            Ok(())
        }
        fn get_relevant_history(&self, _session_id: &str, _user_query: &str) -> Vec<HistoryTurn> {
            Vec::new()
        }
        fn get_project_skills(&self) -> Vec<Skill> {
            Vec::new()
        }
        fn get_asset_images(&self, _asset_ids: &[String]) -> Vec<ImageAttachment> {
            Vec::new()
        }
        fn resolve_attachment_context(&self, _asset_ids: &[String]) -> String {
            String::new()
        }
        fn orchestrate(&self, args: OrchestrateArgs) -> anyhow::Result<OrchestrateOutcome> {
            self.orchestrate_calls.lock().push(args.user_message);
            Ok(self.orchestrate_outcome.lock().clone())
        }
        fn get_project_id(&self) -> Option<String> {
            Some("proj".to_string())
        }
        fn get_project_info(&self) -> Option<String> {
            None
        }
        fn count_tokens(&self, _text: &str) -> usize {
            10
        }
        fn get_max_context_tokens(&self) -> usize {
            4096
        }
        fn get_response_max_context_tokens(&self, _capability: Option<ModelCapability>) -> usize {
            4096
        }
        fn begin_response_request(&self, _progress_callback: Option<ProgressCallback>) {}
        fn announce_response_model(&self, _capability: Option<ModelCapability>) {}
        fn append_to_session(
            &self,
            session_id: &str,
            role: &str,
            content: &str,
            _citations: Option<&[CitationDict]>,
            _message_id: Option<&str>,
        ) {
            self.session_writes.lock().push((
                session_id.to_string(),
                role.to_string(),
                content.to_string(),
            ));
        }
        fn maybe_write_output(
            &self,
            _user_message: &str,
            _llm_response: &str,
            _cwd: Option<&str>,
            _session_id: Option<&str>,
        ) -> Option<String> {
            None
        }
        fn model_client(&self) -> Arc<dyn ModelClient> {
            Arc::new(InProcessModelClient::new(self.device.clone()))
        }
        fn unload_auxiliary_models(&self, include_embedding: bool) {
            self.device.record_unload(include_embedding);
        }
        fn unload_embedding_model(&self) {
            self.device.unload_embedding_model();
        }
        fn workspace_dir(&self) -> Option<String> {
            self.workspace_dir.lock().clone()
        }
        fn workspace_project_ids(&self) -> Option<Vec<String>> {
            None
        }
        fn current_project_dir(&self) -> Option<String> {
            None
        }
        /// The `_no_telemetry` fixture: the usage tail stays offline.
        fn push_token_usage(
            &self,
            _project_id: Option<&str>,
            _model: &str,
            _prompt_tokens: usize,
            _completion_tokens: usize,
        ) {
        }
    }

    /// The content deltas of raw SSE frames, in order.
    pub fn content(raw: &[String]) -> Vec<String> {
        let mut out = Vec::new();
        for chunk in raw {
            for line in chunk.lines() {
                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };
                if data == "[DONE]" {
                    continue;
                }
                let event: serde_json::Value =
                    serde_json::from_str(data).expect("SSE data is JSON");
                if let Some(choices) = event.get("choices").and_then(|c| c.as_array()) {
                    for choice in choices {
                        if let Some(text) = choice
                            .get("delta")
                            .and_then(|d| d.get("content"))
                            .and_then(|c| c.as_str())
                            .filter(|t| !t.is_empty())
                        {
                            out.push(text.to_string());
                        }
                    }
                }
            }
        }
        out
    }

    /// Sets an env var for the life of the guard (`monkeypatch.setenv`).
    pub struct EnvGuard {
        key: String,
        previous: Option<String>,
    }

    impl EnvGuard {
        pub fn set(key: &str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            // SAFETY: the tests that touch the environment are `#[serial]`.
            unsafe { std::env::set_var(key, value) };
            Self { key: key.to_string(), previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: see `EnvGuard::set`.
            unsafe {
                match &self.previous {
                    Some(v) => std::env::set_var(&self.key, v),
                    None => std::env::remove_var(&self.key),
                }
            }
        }
    }
}
// ============================================================== api_chat (end)

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

// ==== BEGIN model_service_a fixtures ====
/// Fixtures shared by the `ModelService` tests (unit `model_service_a`):
/// LLM test doubles, a progress-event recorder, log capture, and a service
/// wired to an in-memory workspace.
pub mod model_service_a {
    use std::any::Any;
    use std::collections::HashMap;
    use std::io::Write;
    use std::sync::Arc;

    use parking_lot::{Mutex, RwLock};

    use smartloop::agents::orchestration::model_router::ProgressCallback;
    use smartloop::api::services::model_service::ModelService;
    use smartloop::llms::base_llm::{BaseLLM, CallOptions, CancelEvent, ChatMessage, GenerationStream};
    use smartloop::llms::llama::{ChatCompletionOutput, ChatCompletionRequest, LlamaModel, LlamaState, LlamaToken};
    use smartloop::model::model_manager::Model;
    use smartloop::model_host::client::{InProcessModelClient, LocalModelService, ModelClient};
    use smartloop::workspace::Workspace;

    /// A llama.cpp model stand-in that only carries GGUF metadata.
    pub struct FakeLlamaModel {
        state: LlamaState,
        metadata: HashMap<String, String>,
    }

    impl FakeLlamaModel {
        pub fn new(metadata: HashMap<String, String>) -> Self {
            Self { state: LlamaState::default(), metadata }
        }
    }

    impl LlamaModel for FakeLlamaModel {
        fn state(&self) -> &LlamaState {
            &self.state
        }

        fn n_ctx(&self) -> usize {
            4096
        }

        fn metadata(&self) -> HashMap<String, String> {
            self.metadata.clone()
        }

        fn tokenize(&self, text: &[u8]) -> anyhow::Result<Vec<LlamaToken>> {
            Ok(text.iter().map(|b| *b as LlamaToken).collect())
        }

        fn detokenize(&self, tokens: &[LlamaToken]) -> anyhow::Result<Vec<u8>> {
            Ok(tokens.iter().map(|t| *t as u8).collect())
        }

        fn create_chat_completion(&self, _request: ChatCompletionRequest) -> anyhow::Result<ChatCompletionOutput<'_>> {
            anyhow::bail!("FakeLlamaModel does not generate")
        }
    }

    /// An LLM test double (Python's `MagicMock()` LLM): a name for debugging,
    /// optional GGUF metadata, and a fixed context size.
    pub struct FakeLLM {
        pub name: String,
        model: Option<Arc<dyn LlamaModel>>,
        pub max_tokens: usize,
        cancelled: Mutex<Option<CancelEvent>>,
    }

    impl FakeLLM {
        /// A double with no metadata.
        pub fn named(name: &str) -> Arc<dyn BaseLLM> {
            Arc::new(Self { name: name.to_string(), model: None, max_tokens: 4096, cancelled: Mutex::new(None) })
        }

        /// `llm.model.metadata = {"general.name": name, "general.size_label": ""}`.
        pub fn with_metadata(name: &str) -> Arc<dyn BaseLLM> {
            let mut meta = HashMap::new();
            meta.insert("general.name".to_string(), name.to_string());
            meta.insert("general.size_label".to_string(), String::new());
            Arc::new(Self {
                name: name.to_string(),
                model: Some(Arc::new(FakeLlamaModel::new(meta))),
                max_tokens: 4096,
                cancelled: Mutex::new(None),
            })
        }
    }

    impl BaseLLM for FakeLLM {
        fn project_id(&self) -> &str {
            ""
        }

        fn get_num_tokens(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }

        fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
            messages
        }

        fn get_max_tokens(&self) -> usize {
            self.max_tokens
        }

        fn call(&self, _prompt: &str, _stop: Option<&[String]>, _options: CallOptions) -> anyhow::Result<String> {
            Ok(String::new())
        }

        fn stream<'a>(&'a self, _prompt: &str, _options: CallOptions) -> GenerationStream<'a> {
            Box::new(std::iter::empty())
        }

        fn invoke(&self, _prompt: &str, _max_tokens: Option<usize>, _thinking: Option<bool>) -> String {
            String::new()
        }

        fn cancelled(&self) -> Option<CancelEvent> {
            self.cancelled.lock().clone()
        }

        fn set_cancelled(&self, event: Option<CancelEvent>) {
            *self.cancelled.lock() = event;
        }

        fn llm_type(&self) -> &str {
            "fake"
        }

        fn llama_model(&self) -> Option<Arc<dyn LlamaModel>> {
            self.model.clone()
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Python `is` for two LLM handles.
    pub fn same(a: &Arc<dyn BaseLLM>, b: &Arc<dyn BaseLLM>) -> bool {
        std::ptr::eq(Arc::as_ptr(a) as *const (), Arc::as_ptr(b) as *const ())
    }

    /// A `progress_callback` that records every `(step, status, message)`.
    #[derive(Clone, Default)]
    pub struct ProgressRecorder {
        pub calls: Arc<Mutex<Vec<(String, String, String)>>>,
    }

    impl ProgressRecorder {
        pub fn new() -> Self {
            Self::default()
        }

        /// The callback to install.
        pub fn callback(&self) -> ProgressCallback {
            let calls = self.calls.clone();
            Arc::new(move |step: &str, status: &str, message: &str| {
                calls.lock().push((step.to_string(), status.to_string(), message.to_string()));
            })
        }

        /// `[c.args[2] for c in callback.call_args_list]`.
        pub fn messages(&self) -> Vec<String> {
            self.calls.lock().iter().map(|(_, _, m)| m.clone()).collect()
        }
    }

    /// A writer appending to a shared buffer (log capture).
    #[derive(Clone, Default)]
    pub struct LogBuffer(pub Arc<Mutex<Vec<u8>>>);

    impl Write for LogBuffer {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogBuffer {
        type Writer = LogBuffer;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Run `f` with a thread-local subscriber capturing everything at
    /// `level` and above (pytest's `caplog.at_level`); returns `f`'s result
    /// and the captured text.
    pub fn capture_logs<R>(level: tracing::Level, f: impl FnOnce() -> R) -> (R, String) {
        let buffer = LogBuffer::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(buffer.clone())
            .with_max_level(level)
            .with_ansi(false)
            .finish();
        let result = tracing::subscriber::with_default(subscriber, f);
        let text = String::from_utf8_lossy(&buffer.0.lock()).into_owned();
        (result, text)
    }

    /// Keep the service in-process (`_remote_device()` → `None`) without
    /// resolving a client from the environment.
    pub fn keep_in_process(svc: &Arc<ModelService>) {
        let local: Arc<dyn LocalModelService> = svc.clone();
        let client: Arc<dyn ModelClient> = Arc::new(InProcessModelClient::new(local));
        svc.set_model_client(Some(client));
    }

    /// `svc._model = Model(workspace=workspace)` (no SLP model attached).
    pub fn attach_workspace(svc: &Arc<ModelService>, workspace: Workspace) -> Arc<RwLock<Workspace>> {
        let shared = Arc::new(RwLock::new(workspace));
        *svc.model.write() = Some(Arc::new(Model::new("model", shared.clone(), None)));
        shared
    }

    /// Points `Workspace::config_file` at a scratch directory for the test's
    /// duration, so nothing a test saves can reach the machine's real
    /// `config.json`.  Process-global: tests using it must be `#[serial]`.
    pub struct ScratchConfig {
        pub dir: tempfile::TempDir,
    }

    impl ScratchConfig {
        pub fn new() -> Self {
            let dir = tempfile::tempdir().expect("tempdir");
            Workspace::set_config_file(Some(dir.path().join("config.json").to_string_lossy().into_owned()));
            ScratchConfig { dir }
        }

        /// A project directory inside the scratch directory.
        pub fn project_dir(&self, project_id: &str) -> String {
            self.dir.path().join(project_id).to_string_lossy().into_owned()
        }
    }

    impl Drop for ScratchConfig {
        fn drop(&mut self) {
            Workspace::set_config_file(None);
        }
    }

    /// A bare service (`object.__new__(ModelService)`) kept in-process.
    pub fn bare_service() -> Arc<ModelService> {
        let svc = ModelService::new_bare();
        keep_in_process(&svc);
        svc
    }
}
// ==== END model_service_a fixtures ====

// ==== model_service_b ====
/// Test doubles for the `ModelService` tests owned by unit `model_service_b`:
/// a recording LLM (standing in for the `MagicMock` LLMs the Python tests
/// built), a llama.cpp model carrying GGUF metadata, a workspace-backed
/// service, and a lock serializing tests that touch process-global state.
pub mod model_service_b {
    use std::any::Any;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use parking_lot::{Mutex, MutexGuard, RwLock};

    use smartloop::api::config::api_settings;
    use smartloop::api::services::model_service::ModelService;
    use smartloop::db::store::{Chunk, ChunkResult, VectorStore};
    use smartloop::llms::base_llm::{BaseLLM, CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream};
    use smartloop::llms::llama::{ChatCompletionOutput, ChatCompletionRequest, LlamaModel, LlamaState, LlamaToken};
    use smartloop::model::model_manager::Model;
    use smartloop::workspace::workspace::{Project, Workspace, WorkspaceModel};

    /// Serializes tests that touch process-global state (the project pin,
    /// delegation flag, the workspace config-file override, `HOME`).
    pub fn global_lock() -> MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock()
    }

    /// A llama.cpp model that only carries GGUF metadata.
    pub struct FakeLlamaModel {
        state: LlamaState,
        pub metadata: HashMap<String, String>,
        pub closed: AtomicBool,
    }

    impl FakeLlamaModel {
        pub fn new(general_name: &str, size_label: &str) -> Self {
            let mut metadata = HashMap::new();
            metadata.insert("general.name".to_string(), general_name.to_string());
            metadata.insert("general.size_label".to_string(), size_label.to_string());
            Self { state: LlamaState::default(), metadata, closed: AtomicBool::new(false) }
        }
    }

    impl LlamaModel for FakeLlamaModel {
        fn state(&self) -> &LlamaState {
            &self.state
        }

        fn n_ctx(&self) -> usize {
            4096
        }

        fn metadata(&self) -> HashMap<String, String> {
            self.metadata.clone()
        }

        fn tokenize(&self, text: &[u8]) -> anyhow::Result<Vec<LlamaToken>> {
            Ok(text.split(|b| *b == b' ').map(|_| 1).collect())
        }

        fn detokenize(&self, _tokens: &[LlamaToken]) -> anyhow::Result<Vec<u8>> {
            Ok(Vec::new())
        }

        fn create_chat_completion(&self, _request: ChatCompletionRequest) -> anyhow::Result<ChatCompletionOutput<'_>> {
            anyhow::bail!("FakeLlamaModel cannot generate")
        }

        fn close(&self) {
            self.closed.store(true, Ordering::SeqCst);
        }
    }

    /// A recording LLM: every prompt it is asked to complete is kept, and it
    /// answers with a fixed reply.
    pub struct FakeLLM {
        pub name: String,
        pub reply: RwLock<String>,
        pub prompts: Mutex<Vec<String>>,
        pub model: Option<Arc<FakeLlamaModel>>,
        pub cancelled: Mutex<Option<CancelEvent>>,
        pub stream_chunks: RwLock<Vec<String>>,
        pub panic_on_generate: AtomicBool,
        pub streams_started: AtomicUsize,
    }

    impl FakeLLM {
        /// A bare recording LLM named `name` (no GGUF metadata).
        pub fn new(name: &str) -> Arc<FakeLLM> {
            Arc::new(Self::build(name, None))
        }

        /// A recording LLM whose GGUF metadata names it `general_name`.
        pub fn named(general_name: &str) -> Arc<FakeLLM> {
            Self::labelled(general_name, "")
        }

        /// A recording LLM whose GGUF metadata carries a name and size label.
        pub fn labelled(general_name: &str, size_label: &str) -> Arc<FakeLLM> {
            Arc::new(Self::build(general_name, Some(Arc::new(FakeLlamaModel::new(general_name, size_label)))))
        }

        fn build(name: &str, model: Option<Arc<FakeLlamaModel>>) -> FakeLLM {
            FakeLLM {
                name: name.to_string(),
                reply: RwLock::new(r#"{"description": "Weather lookups", "tags": ["weather", "forecast"]}"#.to_string()),
                prompts: Mutex::new(Vec::new()),
                model,
                cancelled: Mutex::new(None),
                stream_chunks: RwLock::new(vec!["hello".to_string(), " world".to_string()]),
                panic_on_generate: AtomicBool::new(false),
                streams_started: AtomicUsize::new(0),
            }
        }

        /// How many generations this LLM was asked for.
        pub fn calls(&self) -> usize {
            self.prompts.lock().len()
        }

        /// This LLM as the trait object the service stores.
        pub fn dyn_llm(self: &Arc<Self>) -> Arc<dyn BaseLLM> {
            self.clone()
        }
    }

    impl BaseLLM for FakeLLM {
        fn project_id(&self) -> &str {
            ""
        }

        fn get_num_tokens(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }

        fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
            messages
        }

        fn get_max_tokens(&self) -> usize {
            4096
        }

        fn call(&self, prompt: &str, _stop: Option<&[String]>, _options: CallOptions) -> anyhow::Result<String> {
            self.prompts.lock().push(prompt.to_string());
            if self.panic_on_generate.load(Ordering::SeqCst) {
                panic!("{} ran out of device memory", self.name);
            }
            Ok(self.reply.read().clone())
        }

        fn stream<'a>(&'a self, prompt: &str, _options: CallOptions) -> GenerationStream<'a> {
            self.prompts.lock().push(prompt.to_string());
            self.streams_started.fetch_add(1, Ordering::SeqCst);
            if self.panic_on_generate.load(Ordering::SeqCst) {
                panic!("{} ran out of device memory", self.name);
            }
            let chunks: Vec<GenerationChunk> =
                self.stream_chunks.read().iter().map(|t| GenerationChunk { text: t.clone() }).collect();
            Box::new(chunks.into_iter())
        }

        fn invoke(&self, prompt: &str, max_tokens: Option<usize>, thinking: Option<bool>) -> String {
            let options = CallOptions { max_tokens, thinking };
            match self.call(prompt, None, options) {
                Ok(text) => text,
                Err(e) => format!("Error generating response: {e}"),
            }
        }

        fn cancelled(&self) -> Option<CancelEvent> {
            self.cancelled.lock().clone()
        }

        fn set_cancelled(&self, event: Option<CancelEvent>) {
            *self.cancelled.lock() = event;
        }

        fn llm_type(&self) -> &str {
            "fake"
        }

        fn llama_model(&self) -> Option<Arc<dyn LlamaModel>> {
            self.model.clone().map(|m| m as Arc<dyn LlamaModel>)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// An in-memory vector store answering every query with fixed chunks.
    pub struct FakeStore {
        pub label: String,
        pub chunks: Vec<ChunkResult>,
    }

    impl FakeStore {
        pub fn new(label: &str, chunks: Vec<ChunkResult>) -> Arc<FakeStore> {
            Arc::new(FakeStore { label: label.to_string(), chunks })
        }
    }

    impl VectorStore for FakeStore {
        fn project_dir(&self) -> &str {
            ""
        }

        fn collection_name(&self) -> &str {
            "documents"
        }

        fn base_dir(&self) -> Option<&str> {
            None
        }

        fn add_chunks(&self, _chunks: &[Chunk]) -> anyhow::Result<()> {
            Ok(())
        }

        fn query(&self, _query_text: &str, _n_results: Option<usize>) -> anyhow::Result<Vec<String>> {
            Ok(self.chunks.iter().map(|c| c.text.clone()).collect())
        }

        fn query_with_metadata(
            &self,
            _query_text: &str,
            _n_results: Option<usize>,
            _max_distance: f64,
            _where: Option<&serde_json::Map<String, serde_json::Value>>,
        ) -> anyhow::Result<Vec<ChunkResult>> {
            Ok(self.chunks.clone())
        }

        fn get_by_document_ids(&self, document_ids: &[String]) -> anyhow::Result<Vec<ChunkResult>> {
            Ok(self.chunks.iter().filter(|c| document_ids.contains(&c.document_id)).cloned().collect())
        }

        fn delete_document(&self, _document_id: &str) -> anyhow::Result<()> {
            Ok(())
        }
    }

    /// Identity comparison of two LLM handles (Python `is`).
    pub fn same(a: &Arc<dyn BaseLLM>, b: &Arc<dyn BaseLLM>) -> bool {
        std::ptr::addr_eq(Arc::as_ptr(a), Arc::as_ptr(b))
    }

    /// The bootstrap/orchestrator model name.
    pub fn default_model() -> String {
        api_settings().default_model
    }

    /// A workspace model the user can select.
    pub fn ws_model(name: &str, enabled: bool, downloaded: bool) -> WorkspaceModel {
        WorkspaceModel {
            model_name: name.to_string(),
            model_id: format!("org/{name}"),
            enabled,
            downloaded,
            ..WorkspaceModel::default()
        }
    }

    /// A project rooted at `project_dir`.
    pub fn project(id: &str, project_dir: &str, current: bool) -> Project {
        Project {
            id: id.to_string(),
            project_dir: project_dir.to_string(),
            current,
            name: Some(id.to_string()),
            ..Project::default()
        }
    }

    /// A bare service whose `_model` holds a workspace with `projects` and
    /// `models`, and whose `_model_name` is the default model so the
    /// `_ensure_model` / `_initialize_workspace` guards see it as loaded.
    pub fn service_with_workspace(
        workspace_dir: &str,
        projects: Vec<Project>,
        models: Vec<WorkspaceModel>,
    ) -> Arc<ModelService> {
        let svc = ModelService::new_bare();
        let mut workspace = Workspace::new("ws-test");
        workspace.workspace_dir = workspace_dir.to_string();
        workspace.models_dir = format!("{workspace_dir}/models");
        workspace.projects = projects;
        workspace.models = models;
        let shared = Arc::new(RwLock::new(workspace));
        *svc.model.write() = Some(Arc::new(Model::new("model-test", shared, None)));
        *svc.model_name.write() = Some(default_model());
        svc
    }
}
// ==== end model_service_b ====
