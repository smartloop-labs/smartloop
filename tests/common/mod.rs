//! Port of `tests/conftest.py` (shared fixtures).

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
