//! Port of `smartloop/api/middleware.py`.
//!
//! Middleware binding each request to the project it addresses.
//!
//! Most routers already take a `project_id` and pass it down explicitly, but the
//! service layer still falls back to "the current project" in plenty of places.
//! Binding the project for the request's whole lifetime makes that fallback
//! resolve to the project the caller actually asked about.
//!
//! The binding must cover the response body as it streams, not just the
//! handler that returned it, so the response body is wrapped in the same
//! project scope as the handler future.
//!
//! The chat endpoint is the exception — it carries `project_id` in the request
//! body, which cannot be read here without consuming the stream, so it binds
//! itself (`chat::bind_request_project`).

use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use axum::Router;

use crate::supervisor::routing::RequestScope;
#[allow(unused_imports)]
use crate::workspace::context::{scope_project, use_project};

/// `/v1/projects/<id>/...` — the id segment as the routers declare it.
pub const PATH_PROJECT_RE: &str = r"/projects/([^/]+)";

/// Verbs on the project collection.  They sit in the same slot in the path as
/// a project id and are emphatically not one: reading `import` as a project
/// bound a request to a project called "import".
///
/// Listed rather than pattern-matched so adding one is a decision, not an
/// accident.  The supervisor's routing reads the same set to decide what never
/// to forward (see `crate::supervisor::routing::project_to_proxy`).
pub const COLLECTION_ACTIONS: &[&str] = &["/v1/projects/import"];

/// The `project_id` in a request's query string, if there is one.
///
/// Decoded as latin-1 like the ASGI original; a malformed query string yields
/// `None` rather than an error.
pub fn project_from_query(scope: &RequestScope) -> Option<String> {
    let _ = scope;
    todo!()
}

/// The project this request addresses, from its path or query string.
pub fn project_from_scope(scope: &RequestScope) -> Option<String> {
    let _ = scope;
    todo!()
}

/// Bind the addressed project for the duration of each HTTP request
/// (including a streaming response body).
#[derive(Debug, Clone, Copy, Default)]
pub struct ProjectScopeMiddleware;

impl ProjectScopeMiddleware {
    /// The middleware function (`axum::middleware::from_fn`).
    pub async fn call(request: Request, next: Next) -> Response {
        let _ = (request, next);
        todo!()
    }

    /// Wrap `router` with this middleware (`app.add_middleware(...)`).
    pub fn layer<S>(router: Router<S>) -> Router<S>
    where
        S: Clone + Send + Sync + 'static,
    {
        let _ = router;
        todo!()
    }
}

/// Convenience: `router` wrapped in [`ProjectScopeMiddleware`].
pub fn with_project_scope<S>(router: Router<S>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    ProjectScopeMiddleware::layer(router)
}
