//! Port of `smartloop/supervisor/routing.py`.
//!
//! Getting a request from :38540 to the child that owns its project.
//!
//! The RFC left the granularity open: proxy the request bodies, or redirect the
//! client to the child's port?  This proxies, for two reasons.  The client
//! contract is supposed to be unchanged, and a redirect is not unchanged — the
//! desktop app and CLI would have to follow to a second origin, which moves
//! CORS, auth and the "everything is on :38540" promise onto every client.  And
//! one origin means one auth surface: the child's loopback port never has to be
//! safe to expose, because nothing outside the supervisor ever addresses it.
//!
//! The cost a redirect would have avoided is streaming a token-by-token SSE
//! response through two hops.  That is answered by never buffering: the
//! response is forwarded as raw bytes as they arrive ([`proxy`]), so a chunk
//! crosses the supervisor and reaches the client without waiting for the next.
//!
//! Which requests are project-scoped:
//!
//! * a project id in the path (`/v1/projects/<id>`, with or without a
//!   sub-resource after it) — read from the request head, no body needed;
//! * a `project_id` in the query string — same;
//! * a `project_id` in a JSON body — chat, essentially.  The body has to be
//!   read to forward it anyway, so reading it to route costs nothing extra.
//!
//! Everything else is workspace-level — models, releases, auth, users, the
//! project list — and the supervisor answers it from its own app.
//!
//! Two of those need saying out loud, because a path that merely *contains* a
//! project-shaped segment used to be enough to forward it:
//!
//! * **a delete** is the one operation on a project that its own agent cannot
//!   carry out.  Removing a project means removing it from a list that lives
//!   only in the workspace record, which a pinned save never writes — and
//!   could not write here even in principle, since the project it is pinned to
//!   is the one leaving the list.  Every *other* write to a project record
//!   leaves the project in existence, so it still goes to the child;
//! * **`/v1/projects/import`** names a verb, not a project.  Read as an id it
//!   had the supervisor try to start an agent for a project called "import"
//!   (see `crate::api::middleware::COLLECTION_ACTIONS`).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use axum::Router;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use regex::Regex;

use crate::api::middleware;
use crate::supervisor::registry::AgentRecord;
use crate::utils::timing;

/// Logger name of this module.
const LOGGER: &str = "smartloop.supervisor.routing";

/// Hop-by-hop headers, which describe one connection and must not be forwarded
/// onto another (RFC 9110 §7.6.1).  `content-length` goes too: the body is
/// re-sent, and a streamed response has none.
pub const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
];

/// Paths whose body may name a project.  Kept explicit so the supervisor never
/// reads a body it has no reason to.
///
/// `/v1/feedback` is here because its POST carries `project_id` in the body
/// while its GET and DELETE carry it in the query string — so without this the
/// three halves of one endpoint were answered by two different processes.
pub const BODY_ROUTED_PATHS: &[&str] = &["/v1/chat/completions", "/v1/feedback"];

/// `/v1/projects/<id>` — the project record itself, with or without a
/// sub-resource after it.
pub const PROJECT_PATH_RE: &str = r"/projects/([^/]+)(?P<rest>/.+)?$";

static PROJECT_PATH: Lazy<Regex> = Lazy::new(|| Regex::new(PROJECT_PATH_RE).expect("valid regex"));

/// Methods on a project record that its own agent cannot serve.  Only the
/// delete: it is the one that takes the project out of the workspace record.
pub const WORKSPACE_ONLY_ITEM_METHODS: &[&str] = &["DELETE"];

/// The parts of an ASGI scope routing looks at.
///
/// `path` is the request path as the client sent it; `query_string` the raw
/// query bytes; `headers` the raw `(lowercase name, value)` pairs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RequestScope {
    pub method: String,
    pub path: String,
    pub query_string: Vec<u8>,
    pub headers: Vec<(Vec<u8>, Vec<u8>)>,
}

impl RequestScope {
    /// Build a scope from an incoming axum request's head.
    pub fn from_parts(parts: &http::request::Parts) -> Self {
        Self {
            method: parts.method.as_str().to_string(),
            path: parts.uri.path().to_string(),
            query_string: parts.uri.query().unwrap_or("").as_bytes().to_vec(),
            headers: parts
                .headers
                .iter()
                .map(|(name, value)| (name.as_str().as_bytes().to_vec(), value.as_bytes().to_vec()))
                .collect(),
        }
    }
}

/// What routing needs from a pool: a live agent for a project (blocking).
pub trait AgentPool: Send + Sync {
    /// Return a live agent for `project_id`, starting one if needed.
    fn ensure(&self, project_id: &str) -> anyhow::Result<Arc<AgentRecord>>;
}

impl AgentPool for crate::supervisor::pool::ProjectAgentPool {
    fn ensure(&self, project_id: &str) -> anyhow::Result<Arc<AgentRecord>> {
        crate::supervisor::pool::ProjectAgentPool::ensure(self, project_id).map_err(Into::into)
    }
}

/// Supplies the pool at request time; `None` makes the middleware inert.
pub type PoolProvider = Arc<dyn Fn() -> Option<Arc<dyn AgentPool>> + Send + Sync>;

/// Bytes as latin-1 text (every byte one char), as ASGI decodes headers.
fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|b| char::from(*b)).collect()
}

/// Latin-1 text back to bytes (chars above U+00FF cannot occur from [`latin1`]).
fn latin1_bytes(text: &str) -> Vec<u8> {
    text.chars().map(|c| u32::from(c).min(0xFF) as u8).collect()
}

/// `%XX`-decode a path segment (UTF-8, lossy), as the ASGI server decodes a path.
fn unquote(segment: &str) -> String {
    let bytes = segment.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).ok();
            if let Some(value) = hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(value);
                index += 3;
                continue;
            }
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Drain a request body into bytes.
pub async fn read_body(body: Body) -> anyhow::Result<Bytes> {
    Ok(axum::body::to_bytes(body, usize::MAX).await?)
}

/// A body that serves `body` once, for a request already read.
///
/// Needed when routing decides the supervisor should handle a request after
/// all: its body has been consumed, and the downstream app has to see it.
pub fn replay_body(body: Bytes) -> Body {
    Body::from(body)
}

/// The `project_id` in a JSON body, if there is one (non-empty string).
pub fn project_from_body(body: &[u8]) -> Option<String> {
    if body.is_empty() {
        return None;
    }
    let payload: serde_json::Value = serde_json::from_slice(body).ok()?;
    match payload.as_object()?.get("project_id")? {
        serde_json::Value::String(project_id) if !project_id.is_empty() => Some(project_id.clone()),
        _ => None,
    }
}

/// Whether this request's project can only be found in its body.
pub fn routes_by_body(scope: &RequestScope) -> bool {
    BODY_ROUTED_PATHS.contains(&scope.path.as_str())
}

/// The project this request should be *forwarded* to, or `None`.
///
/// Narrower than `crate::api::middleware::project_from_scope`, which answers a
/// different question — which project a request is *about*, for binding — and
/// still binds the ones answered here.
pub fn project_to_proxy(scope: &RequestScope) -> Option<String> {
    let path = scope.path.as_str();
    if middleware::COLLECTION_ACTIONS.contains(&path) {
        return None;
    }

    if let Some(captures) = PROJECT_PATH.captures(path) {
        if captures.name("rest").is_none() && WORKSPACE_ONLY_ITEM_METHODS.contains(&scope.method.as_str()) {
            return None;
        }
        let project_id = unquote(captures.get(1).map(|m| m.as_str()).unwrap_or(""));
        return if project_id.is_empty() { None } else { Some(project_id) };
    }
    middleware::project_from_query(scope)
}

/// Request headers minus the ones that describe *this* connection.
pub fn forwardable_headers(raw_headers: &[(Vec<u8>, Vec<u8>)]) -> HashMap<String, String> {
    raw_headers
        .iter()
        .map(|(name, value)| (latin1(name), latin1(value)))
        .filter(|(name, _)| !HOP_BY_HOP.contains(&name.to_lowercase().as_str()))
        .collect()
}

/// Response headers to pass back, minus hop-by-hop ones (order preserved).
pub fn response_headers(headers: &reqwest::header::HeaderMap) -> Vec<(Vec<u8>, Vec<u8>)> {
    headers
        .iter()
        .filter(|(name, _)| !HOP_BY_HOP.contains(&name.as_str().to_lowercase().as_str()))
        .map(|(name, value)| (name.as_str().as_bytes().to_vec(), value.as_bytes().to_vec()))
        .collect()
}

/// Forward one request to `base_url` and stream the answer back unbuffered.
///
/// Raw bytes, unbuffered, in both directions: an SSE response reaches the
/// client chunk by chunk exactly as the child produced it, which is what keeps
/// the second hop from turning a streaming answer into a pause and then a wall
/// of text.
pub async fn proxy(
    scope: &RequestScope,
    body: Bytes,
    base_url: &str,
    client: &reqwest::Client,
) -> Result<Response, reqwest::Error> {
    let mut url = format!("{base_url}{}", scope.path);
    if !scope.query_string.is_empty() {
        url.push('?');
        url.push_str(&latin1(&scope.query_string));
    }
    let method = reqwest::Method::from_bytes(scope.method.as_bytes()).unwrap_or(reqwest::Method::GET);

    let mut headers = reqwest::header::HeaderMap::new();
    for (name, value) in forwardable_headers(&scope.headers) {
        let (Ok(name), Ok(value)) = (
            reqwest::header::HeaderName::from_bytes(name.as_bytes()),
            reqwest::header::HeaderValue::from_bytes(&latin1_bytes(&value)),
        ) else {
            continue;
        };
        headers.insert(name, value);
    }

    let upstream = client.request(method, url).headers(headers).body(body).send().await?;

    let mut response = Response::builder().status(upstream.status().as_u16());
    for (name, value) in response_headers(upstream.headers()) {
        response = response.header(name, value);
    }
    let stream = upstream.bytes_stream();
    Ok(response
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| error(502, "project agent sent an invalid response")))
}

/// Send each project-scoped request to that project's own agent.
///
/// Sits outermost on the supervisor's app.  A request that names no project
/// falls through to the app itself, so every workspace-level endpoint —
/// models, releases, auth, the project list — is served here as before and
/// the client cannot tell which requests were forwarded.
///
/// Inert when there is no pool — which is a child serving its own project, and
/// the `--no-supervised` fallback while that still exists.
pub struct ProjectRoutingMiddleware {
    pool_provider: PoolProvider,
    client: Mutex<Option<reqwest::Client>>,
}

impl ProjectRoutingMiddleware {
    /// A middleware asking `pool_provider` for the pool on every request.
    pub fn new(pool_provider: PoolProvider) -> Self {
        Self { pool_provider, client: Mutex::new(None) }
    }

    /// The proxy's HTTP client: no total timeout, 10s connect, no redirects.
    pub fn client(&self) -> reqwest::Client {
        let mut client = self.client.lock();
        if let Some(existing) = client.as_ref() {
            return existing.clone();
        }
        // No total timeout: a long orchestration legitimately streams for
        // minutes, and cutting it off at the proxy would look to the client
        // like the model stopped mid-answer.  Bytes pass through untouched (no
        // transparent decompression), and the loopback hop never goes through
        // a system proxy.
        let built = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::none())
            .no_gzip()
            .no_proxy()
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        *client = Some(built.clone());
        built
    }

    /// Drop the HTTP client.
    pub async fn aclose(&self) {
        self.client.lock().take();
    }

    /// Route one request (the ASGI `__call__`).
    pub async fn call(self: Arc<Self>, request: Request, next: Next) -> Response {
        // Only plain HTTP is routed (the ASGI `scope["type"] == "http"` check):
        // a WebSocket upgrade is the supervisor's own to answer.
        if is_websocket_upgrade(request.headers()) {
            return next.run(request).await;
        }
        let Some(pool) = (self.pool_provider)() else {
            return next.run(request).await;
        };

        let (parts, body) = request.into_parts();
        let mut scope = RequestScope::from_parts(&parts);
        let mut project_id = project_to_proxy(&scope);
        let mut read: Option<Bytes> = None;
        let mut body = Some(body);
        if project_id.is_none() && routes_by_body(&scope) {
            let bytes = read_body(body.take().unwrap_or_default()).await.unwrap_or_default();
            project_id = project_from_body(&bytes);
            read = Some(bytes);
        }

        let Some(project_id) = project_id.filter(|id| !id.is_empty()) else {
            let body = match read {
                Some(bytes) => replay_body(bytes),
                None => body.take().unwrap_or_default(),
            };
            return next.run(Request::from_parts(parts, body)).await;
        };

        let timed = begin_forward(&mut scope, &project_id);
        timing::scope(timed, self.forward(pool, scope, project_id, read, body)).await
    }

    /// Start `project_id`'s agent if it is not running, then proxy the
    /// request to it.
    async fn forward(
        &self,
        pool: Arc<dyn AgentPool>,
        scope: RequestScope,
        project_id: String,
        read: Option<Bytes>,
        mut body: Option<Body>,
    ) -> Response {
        // `ensure` is blocking and, for a cold project, can wait for minutes
        // on a child's port and health check.  Run it off the async workers —
        // otherwise that wait freezes the whole supervisor, including every
        // other project's already-running requests, for as long as the cold
        // start takes.
        let wanted = project_id.clone();
        let ensure_timer = timing::timer("supervisor.ensure");
        let carried = timing::carry();
        let ensured = tokio::task::spawn_blocking(move || {
            let _timing = carried.enter();
            pool.ensure(&wanted)
        })
        .await;
        drop(ensure_timer);
        let record = match ensured {
            Ok(Ok(record)) => record,
            Ok(Err(exc)) => {
                // A project that cannot be started is the supervisor's problem
                // to report, not a reason to answer from the wrong process.
                tracing::error!(target: LOGGER, "[supervisor] could not start project {}: {}", project_id, exc);
                return error(503, &format!("project agent unavailable: {exc}"));
            }
            Err(exc) => {
                tracing::error!(target: LOGGER, "[supervisor] could not start project {}: {}", project_id, exc);
                return error(503, &format!("project agent unavailable: {exc}"));
            }
        };

        let mut forwarded = read.unwrap_or_default();
        if forwarded.is_empty() && !matches!(scope.method.as_str(), "GET" | "HEAD" | "DELETE") {
            if let Some(body) = body.take() {
                forwarded = read_body(body).await.unwrap_or_default();
            }
        }

        // Until the child's response headers are back; a streamed body
        // goes on after.
        let proxy_timer = timing::timer("supervisor.proxy");
        let proxied = proxy(&scope, forwarded, &record.base_url(), &self.client()).await;
        drop(proxy_timer);
        match proxied {
            Ok(response) => response,
            Err(exc) => {
                tracing::warn!(target: LOGGER, "[supervisor] project {}: proxy failed: {}", project_id, exc);
                error(502, &format!("project agent did not answer: {exc}"))
            }
        }
    }

    /// Wrap `router` with this middleware as its outermost layer.
    ///
    /// The router is mounted as the fallback of an outer router that carries
    /// the middleware, so every request — including ones the inner router has
    /// no route for — passes through routing first.
    pub fn layer(self: Arc<Self>, router: Router) -> Router {
        let middleware = self;
        Router::new().fallback_service(router).layer(axum::middleware::from_fn(
            move |request: Request, next: Next| {
                let middleware = Arc::clone(&middleware);
                async move { middleware.call(request, next).await }
            },
        ))
    }
}

/// Under `SLP_TIMING`, time a forwarded chat turn here, under the id its
/// caller sent or a fresh one, and pass that id on so the child's timing
/// joins it.
fn begin_forward(scope: &mut RequestScope, project_id: &str) -> Option<Arc<timing::Request>> {
    if !timing::on() || !scope.path.ends_with("/chat/completions") {
        return None;
    }
    let sent = scope
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(timing::REQUEST_ID_HEADER.as_bytes()))
        .map(|(_, value)| latin1(value));
    let timed = timing::begin(sent.as_deref(), "forward", serde_json::json!({"project": project_id}))?;
    if sent.is_none() {
        scope.headers.push((timing::REQUEST_ID_HEADER.as_bytes().to_vec(), timed.id().as_bytes().to_vec()));
    }
    Some(timed)
}

/// Whether a request asks to switch to the WebSocket protocol.
fn is_websocket_upgrade(headers: &http::HeaderMap) -> bool {
    headers
        .get_all(http::header::UPGRADE)
        .iter()
        .any(|value| value.to_str().is_ok_and(|v| v.trim().eq_ignore_ascii_case("websocket")))
}

/// Wrap `router` so project-scoped requests are proxied to their agents.
pub fn with_project_routing(router: Router, pool_provider: PoolProvider) -> Router {
    Arc::new(ProjectRoutingMiddleware::new(pool_provider)).layer(router)
}

/// `json.dumps` of a string: ASCII-only, `\uXXXX` escapes (surrogate pairs
/// above the BMP).
fn json_string_ascii(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7F => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Answer with the same JSON shape the API's own errors use (`{"detail": ...}`).
fn error(status: u16, detail: &str) -> Response {
    let payload = format!("{{\"detail\": {}}}", json_string_ascii(detail));
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .header("content-length", payload.len().to_string())
        .body(Body::from(payload))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}
