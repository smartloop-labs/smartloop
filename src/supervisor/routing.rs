//! Port of `smartloop/supervisor/routing.py`.
//!
//! Getting a request from :38540 to the child that owns its project.
//!
//! This proxies rather than redirects: the client contract stays unchanged and
//! one origin means one auth surface.  Responses are forwarded as raw bytes as
//! they arrive ([`proxy`]), so an SSE stream crosses the supervisor chunk by
//! chunk.
//!
//! Which requests are project-scoped:
//!
//! * a project id in the path (`/v1/projects/<id>`, with or without a
//!   sub-resource after it);
//! * a `project_id` in the query string;
//! * a `project_id` in a JSON body (chat and feedback POSTs).
//!
//! Everything else is workspace-level and the supervisor answers it from its
//! own app.  A `DELETE /v1/projects/<id>` is never forwarded (only the
//! workspace can remove a project from its list), and `/v1/projects/import`
//! names a verb, not a project (see `crate::api::middleware::COLLECTION_ACTIONS`).

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use axum::Router;
use parking_lot::Mutex;

#[allow(unused_imports)]
use crate::api::middleware;
use crate::supervisor::registry::AgentRecord;

/// Hop-by-hop headers, which describe one connection and must not be forwarded
/// onto another (RFC 9110 §7.6.1), plus `content-length` and `host`.
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

/// Paths whose body may name a project.
pub const BODY_ROUTED_PATHS: &[&str] = &["/v1/chat/completions", "/v1/feedback"];

/// `/v1/projects/<id>` — the project record itself, with or without a
/// sub-resource after it.
pub const PROJECT_PATH_RE: &str = r"/projects/([^/]+)(?P<rest>/.+)?$";

/// Methods on a project record that its own agent cannot serve.
pub const WORKSPACE_ONLY_ITEM_METHODS: &[&str] = &["DELETE"];

/// The parts of an ASGI scope routing looks at.
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
        let _ = parts;
        todo!()
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

/// Drain a request body into bytes.
pub async fn read_body(body: Body) -> anyhow::Result<Bytes> {
    let _ = body;
    todo!()
}

/// A body that serves `body` once, for a request already read.
pub fn replay_body(body: Bytes) -> Body {
    Body::from(body)
}

/// The `project_id` in a JSON body, if there is one (non-empty string).
pub fn project_from_body(body: &[u8]) -> Option<String> {
    let _ = body;
    todo!()
}

/// Whether this request's project can only be found in its body.
pub fn routes_by_body(scope: &RequestScope) -> bool {
    BODY_ROUTED_PATHS.contains(&scope.path.as_str())
}

/// The project this request should be *forwarded* to, or `None`.
pub fn project_to_proxy(scope: &RequestScope) -> Option<String> {
    let _ = scope;
    todo!()
}

/// Request headers minus the ones that describe *this* connection.
pub fn forwardable_headers(raw_headers: &[(Vec<u8>, Vec<u8>)]) -> HashMap<String, String> {
    let _ = raw_headers;
    todo!()
}

/// Response headers to pass back, minus hop-by-hop ones (order preserved).
pub fn response_headers(headers: &reqwest::header::HeaderMap) -> Vec<(Vec<u8>, Vec<u8>)> {
    let _ = headers;
    todo!()
}

/// Forward one request to `base_url` and stream the answer back unbuffered.
pub async fn proxy(
    scope: &RequestScope,
    body: Bytes,
    base_url: &str,
    client: &reqwest::Client,
) -> Result<Response, reqwest::Error> {
    let _ = (scope, body, base_url, client);
    todo!()
}

/// Send each project-scoped request to that project's own agent.
///
/// Sits outermost on the supervisor's app.  A request that names no project
/// falls through to the app itself.  Inert when there is no pool.
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
        todo!()
    }

    /// Drop the HTTP client.
    pub async fn aclose(&self) {
        todo!()
    }

    /// Route one request (the ASGI `__call__`).
    pub async fn call(self: Arc<Self>, request: Request, next: Next) -> Response {
        let _ = (request, next, &self.pool_provider);
        todo!()
    }

    /// Wrap `router` with this middleware as its outermost layer.
    pub fn layer(self: Arc<Self>, router: Router) -> Router {
        let _ = router;
        todo!()
    }
}

/// Wrap `router` so project-scoped requests are proxied to their agents.
pub fn with_project_routing(router: Router, pool_provider: PoolProvider) -> Router {
    Arc::new(ProjectRoutingMiddleware::new(pool_provider)).layer(router)
}

/// Answer with the same JSON shape the API's own errors use (`{"detail": ...}`).
fn error(status: u16, detail: &str) -> Response {
    let _ = (status, detail);
    todo!()
}
