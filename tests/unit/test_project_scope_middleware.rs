//! Port of `tests/unit/test_project_scope_middleware.py`.
//!
//! Requests that name a project in the URL are bound to it for their lifetime.
//!
//! The chat endpoint carries `project_id` in the body and binds itself; every
//! other router takes it in the path or query string, which the middleware can
//! read from the request head without touching the request stream.

use axum::body::{Body, Bytes};
use axum::http::{Request, StatusCode};
use axum::response::Response;
use axum::routing::get;
use axum::{Json, Router};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

use smartloop::api::middleware::{project_from_scope, ProjectScopeMiddleware, COLLECTION_ACTIONS};
use smartloop::supervisor::routing::{project_to_proxy, RequestScope};
use smartloop::workspace::context::active_project_id;

fn scope(path: &str, query: &str) -> RequestScope {
    RequestScope {
        method: "GET".to_string(),
        path: path.to_string(),
        query_string: query.as_bytes().to_vec(),
        headers: Vec::new(),
    }
}

mod test_reading_the_project_off_the_scope {
    use super::*;

    #[test]
    fn test_from_the_path() {
        assert_eq!(project_from_scope(&scope("/v1/projects/abc/documents", "")).as_deref(), Some("abc"));
    }

    #[test]
    fn test_from_a_nested_path() {
        assert_eq!(project_from_scope(&scope("/v1/projects/abc/mcp/tools/count", "")).as_deref(), Some("abc"));
    }

    #[test]
    fn test_from_the_query_string() {
        assert_eq!(project_from_scope(&scope("/v1/feedback", "project_id=xyz")).as_deref(), Some("xyz"));
    }

    #[test]
    fn test_the_path_wins_over_the_query() {
        let s = scope("/v1/projects/abc/documents", "project_id=xyz");
        assert_eq!(project_from_scope(&s).as_deref(), Some("abc"));
    }

    #[test]
    fn test_no_project_named() {
        assert_eq!(project_from_scope(&scope("/v1/models", "")), None);
    }

    #[test]
    fn test_the_collection_itself_names_no_project() {
        assert_eq!(project_from_scope(&scope("/v1/projects", "")), None);
    }

    /// `import` sits where an id sits and is not one.
    ///
    /// Bound as a project it produced "bound project import is not in this
    /// workspace" the moment anything resolved it — a warning about a
    /// workspace member that cannot exist, in the middle of an import that had
    /// otherwise succeeded.
    #[test]
    fn test_a_collection_verb_is_not_a_project() {
        assert_eq!(project_from_scope(&scope("/v1/projects/import", "")), None);
    }

    /// Binding and routing agree on what is not a project.
    ///
    /// They disagreed once: the supervisor stopped forwarding `import` while
    /// the middleware went on binding it.  The next verb added must not be able
    /// to land in one layer only — routing reads this module's
    /// `COLLECTION_ACTIONS` itself, so there is one list by construction, and
    /// both layers must answer "no project" for every entry.
    #[test]
    fn test_the_two_layers_read_the_same_list() {
        assert!(!COLLECTION_ACTIONS.is_empty());
        for path in COLLECTION_ACTIONS {
            assert_eq!(project_from_scope(&scope(path, "")), None);
            let mut post = scope(path, "");
            post.method = "POST".to_string();
            assert_eq!(project_to_proxy(&post), None);
        }
    }

    #[test]
    fn test_a_malformed_query_string_is_not_fatal() {
        let s = RequestScope {
            method: "GET".to_string(),
            path: "/v1/models".to_string(),
            query_string: b"\xff\xfe=%%".to_vec(),
            headers: Vec::new(),
        };
        assert_eq!(project_from_scope(&s), None);
    }
}

/// The test application: a path-scoped probe, an unscoped probe and a
/// streaming body that reads the binding only once it is being sent.
fn app() -> Router {
    let routes = Router::new()
        .route("/v1/projects/{project_id}/probe", get(|| async { Json(json!({"bound": active_project_id()})) }))
        .route("/v1/unscoped", get(|| async { Json(json!({"bound": active_project_id()})) }))
        .route(
            "/v1/projects/{project_id}/stream",
            get(|| async {
                // Reads while the response is streaming, i.e. after the
                // handler has already returned — the case a binding scoped to
                // the handler alone would miss.
                let body = async_stream::stream! {
                    let bound = active_project_id().unwrap_or_else(|| "none".to_string());
                    yield Ok::<Bytes, std::io::Error>(Bytes::from(bound));
                };
                Response::builder()
                    .header("content-type", "text/plain")
                    .body(Body::from_stream(body))
                    .unwrap()
            }),
        );
    ProjectScopeMiddleware::layer(routes)
}

async fn get_body(app: Router, uri: &str) -> (StatusCode, Vec<u8>) {
    let response = app.oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap()).await.unwrap();
    let status = response.status();
    let body = response.into_body().collect().await.unwrap().to_bytes().to_vec();
    (status, body)
}

async fn get_json(app: Router, uri: &str) -> Value {
    let (status, body) = get_body(app, uri).await;
    assert_eq!(status, StatusCode::OK);
    serde_json::from_slice(&body).unwrap()
}

mod test_the_binding_reaches_the_handler {
    use super::*;

    #[tokio::test]
    #[serial_test::serial]
    async fn test_a_path_scoped_request_is_bound() {
        assert_eq!(get_json(app(), "/v1/projects/abc/probe").await, json!({"bound": "abc"}));
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_an_unscoped_request_is_not() {
        assert_eq!(get_json(app(), "/v1/unscoped").await, json!({"bound": null}));
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_the_binding_survives_into_a_streaming_body() {
        let (status, body) = get_body(app(), "/v1/projects/abc/stream").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(String::from_utf8(body).unwrap(), "abc");
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_the_binding_does_not_leak_between_requests() {
        let client = app();
        assert_eq!(get_json(client.clone(), "/v1/projects/abc/probe").await, json!({"bound": "abc"}));
        assert_eq!(get_json(client, "/v1/unscoped").await, json!({"bound": null}));
        assert_eq!(active_project_id(), None);
    }
}
