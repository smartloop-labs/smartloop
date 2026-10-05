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
/// `#[serial(mcp_global)]`.
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
