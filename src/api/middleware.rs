//! Port of `smartloop/api/middleware.py`.
//!
//! Middleware binding each request to the project it addresses.
//!
//! Most routers already take a `project_id` and pass it down explicitly, but the
//! service layer still falls back to "the current project" in plenty of places
//! (and did so, until now, by reading a process-wide flag that any concurrent
//! request could move).  Binding the project for the request's whole lifetime
//! makes that fallback resolve to the project the caller actually asked about.
//!
//! The binding must cover the response body as it streams, not just the
//! handler that returned it: a streaming response is polled by the server
//! *after* the handler has returned, so the response body is wrapped in the
//! same project binding as the handler future (re-established around every
//! poll of the body).
//!
//! The chat endpoint is the exception — it carries `project_id` in the request
//! body, which cannot be read here without consuming the stream, so it binds
//! itself (`chat::bind_request_project`).

use std::pin::Pin;
use std::task::{Context, Poll};

use axum::body::{Body, Bytes};
use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use axum::Router;
use hyper::body::{Body as HttpBody, Frame, SizeHint};
use once_cell::sync::Lazy;
use regex::Regex;

use crate::supervisor::routing::RequestScope;
use crate::workspace::context::{scope_project, use_project};

/// `/v1/projects/<id>/...` — the id segment as the routers declare it.
pub const PATH_PROJECT_RE: &str = r"/projects/([^/]+)";

static PATH_PROJECT: Lazy<Regex> = Lazy::new(|| Regex::new(PATH_PROJECT_RE).expect("valid regex"));

/// Verbs on the project collection.  They sit in the same slot in the path as
/// a project id and are emphatically not one, so the pattern above has to be
/// told about them: reading `import` as a project bound a request to a
/// project called "import", and the first thing that resolved the binding
/// warned about a workspace member that could never exist.
///
/// Listed rather than pattern-matched so adding one is a decision, not an
/// accident.  The supervisor's routing reads the same set to decide what never
/// to forward (see `crate::supervisor::routing::project_to_proxy`) — one list,
/// because a path that names no project must neither bind one nor be proxied
/// to one.
pub const COLLECTION_ACTIONS: &[&str] = &["/v1/projects/import"];

/// Bytes as latin-1 text (every byte one char), as ASGI decodes a query.
fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|b| char::from(*b)).collect()
}

/// Python's `urllib.parse.unquote` over text: `%XX` runs are decoded as UTF-8
/// (invalid sequences replaced), anything else — including a `%` not followed
/// by two hex digits — is kept as-is.
fn unquote(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut pending: Vec<u8> = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        if chars[index] == '%' && index + 2 < chars.len() {
            let hi = chars[index + 1].to_digit(16);
            let lo = chars[index + 2].to_digit(16);
            if let (Some(hi), Some(lo)) = (hi, lo) {
                pending.push((hi * 16 + lo) as u8);
                index += 3;
                continue;
            }
        }
        if !pending.is_empty() {
            out.push_str(&String::from_utf8_lossy(&pending));
            pending.clear();
        }
        out.push(chars[index]);
        index += 1;
    }
    if !pending.is_empty() {
        out.push_str(&String::from_utf8_lossy(&pending));
    }
    out
}

/// Python's `urllib.parse.parse_qs(query)` (default flags) restricted to what
/// this module needs: the values of `key`, in order.
///
/// Pairs are split on `&`; a pair without `=` is skipped, as is one with an
/// empty value (`keep_blank_values=False`); `+` is a space and `%XX` escapes
/// are decoded.
fn parse_qs_values(query: &str, key: &str) -> Vec<String> {
    let mut values = Vec::new();
    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let Some((name, value)) = pair.split_once('=') else {
            continue;
        };
        if value.is_empty() {
            continue;
        }
        let name = unquote(&name.replace('+', " "));
        if name != key {
            continue;
        }
        values.push(unquote(&value.replace('+', " ")));
    }
    values
}

/// The `project_id` in a request's query string, if there is one.
///
/// Split out because the supervisor's routing asks a narrower question about
/// the path — which project a request should be *forwarded* to, which is not
/// every project a request is about — but the same one about the query.  See
/// `crate::supervisor::routing::project_to_proxy`.
///
/// Decoded as latin-1 like the ASGI original; a malformed query string yields
/// `None` rather than an error.
pub fn project_from_query(scope: &RequestScope) -> Option<String> {
    if scope.query_string.is_empty() {
        return None;
    }
    let query = latin1(&scope.query_string);
    parse_qs_values(&query, "project_id").into_iter().next()
}

/// The project this request addresses, from its path or query string.
pub fn project_from_scope(scope: &RequestScope) -> Option<String> {
    let path = scope.path.as_str();
    if COLLECTION_ACTIONS.contains(&path) {
        return None;
    }
    if let Some(captures) = PATH_PROJECT.captures(path) {
        // The ASGI path the Python read was already percent-decoded.
        let project_id = unquote(captures.get(1).map(|m| m.as_str()).unwrap_or(""));
        return if project_id.is_empty() { None } else { Some(project_id) };
    }
    project_from_query(scope)
}

/// A response body that re-establishes the request's project binding around
/// every poll, so a streaming body generated after the handler returned still
/// resolves the project the request addressed.
struct ProjectScopedBody {
    inner: Body,
    project_id: String,
}

impl HttpBody for ProjectScopedBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        let _binding = use_project(Some(&this.project_id));
        Pin::new(&mut this.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

/// Whether this request is a WebSocket handshake (an ASGI `websocket` scope,
/// which the Python middleware passed through unbound).
fn is_websocket(request: &Request) -> bool {
    request
        .headers()
        .get(http::header::UPGRADE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.eq_ignore_ascii_case("websocket"))
}

/// Bind the addressed project for the duration of each HTTP request
/// (including a streaming response body).
#[derive(Debug, Clone, Copy, Default)]
pub struct ProjectScopeMiddleware;

impl ProjectScopeMiddleware {
    /// The middleware function (`axum::middleware::from_fn`).
    pub async fn call(request: Request, next: Next) -> Response {
        if is_websocket(&request) {
            return next.run(request).await;
        }
        let (parts, body) = request.into_parts();
        let project_id = project_from_scope(&RequestScope::from_parts(&parts));
        let request = Request::from_parts(parts, body);
        match project_id {
            None => next.run(request).await,
            Some(project_id) => {
                let response = scope_project(Some(project_id.clone()), next.run(request)).await;
                let (parts, body) = response.into_parts();
                Response::from_parts(parts, Body::new(ProjectScopedBody { inner: body, project_id }))
            }
        }
    }

    /// Wrap `router` with this middleware (`app.add_middleware(...)`).
    pub fn layer<S>(router: Router<S>) -> Router<S>
    where
        S: Clone + Send + Sync + 'static,
    {
        router.layer(axum::middleware::from_fn(Self::call))
    }
}

/// Convenience: `router` wrapped in [`ProjectScopeMiddleware`].
pub fn with_project_scope<S>(router: Router<S>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    ProjectScopeMiddleware::layer(router)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unquote_matches_python() {
        assert_eq!(unquote("a%20b"), "a b");
        assert_eq!(unquote("%%"), "%%");
        assert_eq!(unquote("%E2%82%AC"), "€");
        assert_eq!(unquote("%ff"), "\u{fffd}");
        assert_eq!(unquote("100%"), "100%");
    }

    #[test]
    fn parse_qs_skips_blank_and_bare_pairs() {
        assert_eq!(parse_qs_values("project_id=&project_id=x", "project_id"), vec!["x"]);
        assert_eq!(parse_qs_values("project_id", "project_id"), Vec::<String>::new());
        assert_eq!(parse_qs_values("a=1&project_id=a+b", "project_id"), vec!["a b"]);
    }
}
