//! Port of `tests/conftest.py` (shared fixtures).

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
/// install the override must be `#[serial(http_transport)]`.
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
}
// =================================================================== /mcp
