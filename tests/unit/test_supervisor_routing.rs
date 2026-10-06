//! Port of `tests/unit/test_supervisor_routing.py`.
//!
//! Getting a request from :38540 to the child that owns its project.
//!
//! The client contract is supposed to be unchanged, which is why this proxies
//! rather than redirecting: one origin, one auth surface, and the desktop app
//! never learns that a second port exists.  The cost a redirect would have
//! avoided — streaming SSE through two hops — is paid off by never buffering,
//! which is what the streaming tests here are for.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{Path, Request};
use axum::http::{HeaderMap, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use futures::StreamExt;
use http_body_util::BodyExt;
use parking_lot::Mutex;
use serde_json::{json, Value};
use smartloop::supervisor::registry::AgentRecord;
use smartloop::supervisor::routing::{
    forwardable_headers, project_from_body, project_to_proxy, read_body, replay_body, response_headers,
    routes_by_body, AgentPool, PoolProvider, ProjectRoutingMiddleware, RequestScope,
};
use tower::ServiceExt;

/// Stands in for the agent pool: records what was routed where.
struct FakePool {
    port: u16,
    requested: Mutex<Vec<String>>,
    fail_with: Mutex<Option<String>>,
}

impl FakePool {
    fn new(base_url: &str) -> Arc<Self> {
        let port = base_url.rsplit(':').next().unwrap().parse().unwrap();
        Arc::new(Self { port, requested: Mutex::new(Vec::new()), fail_with: Mutex::new(None) })
    }

    fn requested(&self) -> Vec<String> {
        self.requested.lock().clone()
    }
}

impl AgentPool for FakePool {
    fn ensure(&self, project_id: &str) -> anyhow::Result<Arc<AgentRecord>> {
        self.requested.lock().push(project_id.to_string());
        if let Some(message) = self.fail_with.lock().clone() {
            anyhow::bail!(message);
        }
        Ok(Arc::new(AgentRecord::new(project_id, 0, self.port)))
    }
}

fn provider(pool: &Arc<FakePool>) -> PoolProvider {
    let pool = Arc::clone(pool);
    Arc::new(move || Some(Arc::clone(&pool) as Arc<dyn AgentPool>))
}

fn no_pool() -> PoolProvider {
    Arc::new(|| None)
}

fn with_routing(router: Router, pool_provider: PoolProvider) -> Router {
    Arc::new(ProjectRoutingMiddleware::new(pool_provider)).layer(router)
}

/// A running server for `app` on a real loopback port.
///
/// A real socket, not `oneshot`: an in-process call assembles the whole
/// response before handing it back, so a streamed body would look buffered no
/// matter what the proxy did.
struct Served {
    base_url: String,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for Served {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn serve(app: Router) -> Served {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let task = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Served { base_url: format!("http://127.0.0.1:{port}"), task }
}

/// A stand-in project agent, on a real port, echoing what it received.
async fn child() -> Served {
    async fn documents(Path(project_id): Path<String>) -> Json<Value> {
        Json(json!({"served_by": "child", "project_id": project_id}))
    }

    async fn probe(method: Method, headers: HeaderMap, request: Request) -> Json<Value> {
        let query = request.uri().query().unwrap_or("").to_string();
        let body = request.into_body().collect().await.unwrap().to_bytes();
        Json(json!({
            "method": method.as_str(),
            "query": query,
            "body": String::from_utf8_lossy(&body),
            "saw_header": headers.get("x-custom").map(|v| v.to_str().unwrap().to_string()),
        }))
    }

    async fn echo_project(Json(payload): Json<Value>) -> Json<Value> {
        Json(json!({"served_by": "child", "project_id": payload.get("project_id")}))
    }

    async fn stream(Path(_project_id): Path<String>) -> Response {
        let body = async_stream::stream! {
            for index in 0..5 {
                yield Ok::<_, std::convert::Infallible>(format!("data: chunk-{index}\n\n"));
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        };
        ([("content-type", "text/event-stream")], Body::from_stream(body)).into_response()
    }

    let app = Router::new()
        .route("/v1/projects/{project_id}/documents", get(documents))
        .route("/v1/probe", get(probe).post(probe))
        .route("/v1/chat/completions", post(echo_project))
        .route("/v1/feedback", post(echo_project))
        .route("/v1/projects/{project_id}/stream", get(stream));
    serve(app).await
}

/// A supervisor app: its own routes, plus routing to the child.
struct SupervisorApp {
    app: Router,
    pool: Arc<FakePool>,
    _child: Served,
}

async fn supervisor_app() -> SupervisorApp {
    let child = child().await;
    let pool = FakePool::new(&child.base_url);

    async fn here() -> Json<Value> {
        Json(json!({"served_by": "supervisor"}))
    }

    async fn delete_project(Path(project_id): Path<String>) -> Json<Value> {
        Json(json!({"served_by": "supervisor", "project_id": project_id}))
    }

    let app = Router::new()
        .route("/v1/models", get(here))
        .route("/v1/projects", get(here))
        .route("/v1/projects/{project_id}", delete(delete_project))
        .route("/v1/projects/import", post(here))
        .route("/v1/chat/completions", post(here))
        .route("/health", get(here));
    SupervisorApp { app: with_routing(app, provider(&pool)), pool, _child: child }
}

/// `TestClient(app).<method>(uri)` — one in-process request.
async fn call(app: &Router, method: &str, uri: &str, body: Body, headers: &[(&str, &str)]) -> (StatusCode, Value) {
    let mut request = Request::builder().method(method).uri(uri);
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    let response = app.clone().oneshot(request.body(body).unwrap()).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

async fn get_json(app: &Router, uri: &str) -> Value {
    call(app, "GET", uri, Body::empty(), &[]).await.1
}

async fn post_json(app: &Router, uri: &str, payload: Value) -> Value {
    call(app, "POST", uri, Body::from(payload.to_string()), &[("content-type", "application/json")]).await.1
}

fn scope(path: &str, method: &str, query: &str) -> RequestScope {
    RequestScope {
        method: method.to_string(),
        path: path.to_string(),
        query_string: query.as_bytes().to_vec(),
        headers: Vec::new(),
    }
}

fn path_scope(path: &str) -> RequestScope {
    scope(path, "GET", "")
}

mod test_reading_the_project_from_a_body {
    use super::*;

    #[test]
    fn test_a_chat_request_names_its_project_in_the_body() {
        assert_eq!(project_from_body(br#"{"project_id": "alpha"}"#), Some("alpha".to_string()));
    }

    #[test]
    fn test_a_body_without_one() {
        assert_eq!(project_from_body(br#"{"model": "sl-mini"}"#), None);
    }

    #[test]
    fn test_an_empty_body() {
        assert_eq!(project_from_body(b""), None);
    }

    #[test]
    fn test_a_non_json_body_is_not_an_error() {
        assert_eq!(project_from_body(b"not json at all"), None);
    }

    #[test]
    fn test_a_json_array_is_not_an_error() {
        assert_eq!(project_from_body(b"[1, 2, 3]"), None);
    }

    #[test]
    fn test_an_empty_project_id_does_not_count() {
        assert_eq!(project_from_body(br#"{"project_id": ""}"#), None);
    }

    #[test]
    fn test_chat_is_routed_by_its_body() {
        assert!(routes_by_body(&path_scope("/v1/chat/completions")));
    }

    /// Its POST names the project in the body; its GET/DELETE in the query.
    #[test]
    fn test_feedback_is_routed_by_its_body() {
        assert!(routes_by_body(&path_scope("/v1/feedback")));
    }

    #[test]
    fn test_nothing_else_is() {
        assert!(!routes_by_body(&path_scope("/v1/models")));
        assert!(!routes_by_body(&path_scope("/v1/projects")));
    }
}

/// A delete is the one thing a project's own agent cannot do to itself.
///
/// Removing a project means removing it from a list that lives only in
/// `config.json`, which a pinned save never writes — and could not write here
/// anyway, since the pinned project is the one leaving the list.  Every other
/// write to a project record leaves the project in existence, so it still goes
/// to the child that owns it.
mod test_what_only_the_workspace_can_answer {
    use super::*;

    #[test]
    fn test_a_sub_resource_belongs_to_the_child() {
        assert_eq!(project_to_proxy(&scope("/v1/projects/abc/documents", "GET", "")), Some("abc".into()));
    }

    #[test]
    fn test_a_nested_sub_resource_too() {
        assert_eq!(project_to_proxy(&scope("/v1/projects/abc/mcp/tools/count", "GET", "")), Some("abc".into()));
    }

    #[test]
    fn test_a_delete_is_not_forwarded() {
        assert_eq!(project_to_proxy(&scope("/v1/projects/abc", "DELETE", "")), None);
    }

    /// Deleting a document is the child's; only the project itself is not.
    #[test]
    fn test_a_delete_of_a_sub_resource_still_is() {
        assert_eq!(project_to_proxy(&scope("/v1/projects/abc/documents/d1", "DELETE", "")), Some("abc".into()));
    }

    /// It writes `web_search_enabled` and skills — project state, not the index.
    #[test]
    fn test_a_rename_still_belongs_to_the_child() {
        assert_eq!(project_to_proxy(&scope("/v1/projects/abc", "PATCH", "")), Some("abc".into()));
    }

    #[test]
    fn test_the_collection_is_not_forwarded() {
        assert_eq!(project_to_proxy(&scope("/v1/projects", "GET", "")), None);
    }

    #[test]
    fn test_import_is_a_verb_not_a_project_id() {
        assert_eq!(project_to_proxy(&scope("/v1/projects/import", "POST", "")), None);
    }

    #[test]
    fn test_the_query_string_still_names_a_project() {
        assert_eq!(project_to_proxy(&scope("/v1/feedback", "GET", "project_id=xyz")), Some("xyz".into()));
    }

    #[test]
    fn test_a_malformed_query_string_is_not_fatal() {
        let mut scope = scope("/v1/models", "GET", "");
        scope.query_string = b"\xff\xfe=%%".to_vec();
        assert_eq!(project_to_proxy(&scope), None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_a_delete_is_answered_by_the_supervisor() {
        let app = supervisor_app().await;
        let (_, body) = call(&app.app, "DELETE", "/v1/projects/alpha", Body::empty(), &[]).await;

        assert_eq!(body, json!({"served_by": "supervisor", "project_id": "alpha"}));
        // Not merely answered here — no agent was started to answer it.
        assert!(app.pool.requested().is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_an_import_does_not_start_an_agent_called_import() {
        let app = supervisor_app().await;
        let (_, body) = call(&app.app, "POST", "/v1/projects/import", Body::empty(), &[]).await;

        assert_eq!(body, json!({"served_by": "supervisor"}));
        assert!(app.pool.requested().is_empty());
    }
}

mod test_header_handling {
    use super::*;

    #[test]
    fn test_hop_by_hop_headers_are_not_forwarded() {
        let raw: Vec<(Vec<u8>, Vec<u8>)> = [
            ("host", "localhost:38540"),
            ("connection", "keep-alive"),
            ("content-length", "12"),
            ("transfer-encoding", "chunked"),
            ("authorization", "Bearer t"),
            ("x-custom", "kept"),
        ]
        .iter()
        .map(|(k, v)| (k.as_bytes().to_vec(), v.as_bytes().to_vec()))
        .collect();
        let headers = forwardable_headers(&raw);
        let expected: std::collections::HashMap<String, String> =
            [("authorization", "Bearer t"), ("x-custom", "kept")].iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        assert_eq!(headers, expected);
    }

    #[test]
    fn test_response_hop_by_hop_headers_are_dropped() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.append("content-type", "text/event-stream".parse().unwrap());
        headers.append("transfer-encoding", "chunked".parse().unwrap());
        headers.append("cache-control", "no-cache".parse().unwrap());
        assert_eq!(
            response_headers(&headers),
            vec![
                (b"content-type".to_vec(), b"text/event-stream".to_vec()),
                (b"cache-control".to_vec(), b"no-cache".to_vec()),
            ]
        );
    }
}

mod test_replaying_a_body {
    use super::*;

    #[tokio::test]
    async fn test_a_body_read_for_routing_can_still_be_served() {
        let mut body = replay_body(bytes::Bytes::from_static(br#"{"a": 1}"#));
        let frame = body.frame().await.unwrap().unwrap();
        assert_eq!(frame.into_data().unwrap(), bytes::Bytes::from_static(br#"{"a": 1}"#));
        assert!(body.frame().await.is_none(), "the body ends once it has been served");
    }

    #[tokio::test]
    async fn test_a_chunked_body_is_reassembled() {
        let chunks = futures::stream::iter(vec![
            Ok::<_, std::io::Error>(bytes::Bytes::from_static(b"one")),
            Ok(bytes::Bytes::from_static(b"two")),
        ]);
        assert_eq!(read_body(Body::from_stream(chunks)).await.unwrap(), bytes::Bytes::from_static(b"onetwo"));
    }
}

mod test_what_gets_forwarded {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_a_project_scoped_path_goes_to_the_child() {
        let app = supervisor_app().await;
        let body = get_json(&app.app, "/v1/projects/alpha/documents").await;

        assert_eq!(body, json!({"served_by": "child", "project_id": "alpha"}));
        assert_eq!(app.pool.requested(), vec!["alpha"]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_a_workspace_level_path_is_answered_here() {
        let app = supervisor_app().await;
        let body = get_json(&app.app, "/v1/models").await;

        assert_eq!(body, json!({"served_by": "supervisor"}));
        assert!(app.pool.requested().is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_the_project_list_itself_is_workspace_level() {
        let app = supervisor_app().await;
        assert_eq!(get_json(&app.app, "/v1/projects").await, json!({"served_by": "supervisor"}));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_health_is_the_supervisors_own() {
        let app = supervisor_app().await;
        assert_eq!(get_json(&app.app, "/health").await, json!({"served_by": "supervisor"}));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_a_chat_request_is_routed_by_its_body() {
        let app = supervisor_app().await;
        let body = post_json(&app.app, "/v1/chat/completions", json!({"project_id": "beta", "model": "sl-mini"})).await;

        assert_eq!(body, json!({"served_by": "child", "project_id": "beta"}));
        assert_eq!(app.pool.requested(), vec!["beta"]);
    }

    /// The POST used to be answered here, indexing into the wrong project.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_a_feedback_post_reaches_the_project_it_names() {
        let app = supervisor_app().await;
        let body = post_json(
            &app.app,
            "/v1/feedback",
            json!({"session_id": "s1", "message_id": "m1", "project_id": "gamma", "rating": "up"}),
        )
        .await;

        assert_eq!(body, json!({"served_by": "child", "project_id": "gamma"}));
        assert_eq!(app.pool.requested(), vec!["gamma"]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_a_chat_request_naming_no_project_stays_here() {
        let app = supervisor_app().await;
        let body = post_json(&app.app, "/v1/chat/completions", json!({"model": "sl-mini"})).await;

        assert_eq!(body, json!({"served_by": "supervisor"}));
        assert!(app.pool.requested().is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_the_query_string_survives_the_hop() {
        let app = supervisor_app().await;
        let body = get_json(&app.app, "/v1/probe?project_id=alpha&limit=5").await;
        assert_eq!(body["query"], "project_id=alpha&limit=5");
    }

    /// A `project_id` in the query proxies *any* path, workspace ones included.
    ///
    /// Worth pinning because a good deal of reasoning elsewhere depends on it.
    /// These endpoints read and write workspace-level state — the model
    /// catalogue, the credential store — and it is tempting to assume routing
    /// keeps them on the supervisor because they carry no project in their
    /// path.  It does not: the query parameter is enough, so the handlers
    /// themselves have to know which process they are in.
    #[test]
    fn test_a_workspace_endpoint_naming_a_project_is_proxied_too() {
        for path in ["/v1/models", "/v1/auth/status", "/v1/users/me"] {
            assert_eq!(project_to_proxy(&scope(path, "GET", "project_id=alpha")), Some("alpha".into()), "{path}");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_the_body_and_headers_survive_the_hop() {
        let app = supervisor_app().await;
        let (_, body) = call(
            &app.app,
            "POST",
            "/v1/probe?project_id=alpha",
            Body::from(r#"{"hello": "world"}"#),
            &[("x-custom", "kept")],
        )
        .await;
        assert_eq!(body["method"], "POST");
        assert_eq!(body["body"], r#"{"hello": "world"}"#);
        assert_eq!(body["saw_header"], "kept");
    }
}

/// The cost a redirect would have avoided, paid off by never buffering.
///
/// Driven over real sockets on both hops: an in-process call assembles the
/// whole response first, which would make a buffering proxy pass.
mod test_streaming_through_two_hops {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_an_sse_response_arrives_chunk_by_chunk() {
        let app = supervisor_app().await;
        let served = serve(app.app.clone()).await;

        let client = reqwest::Client::builder().no_proxy().timeout(Duration::from_secs(30)).build().unwrap();
        let started = Instant::now();
        let response = client.get(format!("{}/v1/projects/alpha/stream", served.base_url)).send().await.unwrap();
        assert!(response.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream"));

        let mut arrivals: Vec<f64> = Vec::new();
        let mut chunks: Vec<String> = Vec::new();
        let mut pending = String::new();
        let mut stream = response.bytes_stream();
        while let Some(piece) = stream.next().await {
            pending.push_str(&String::from_utf8_lossy(&piece.unwrap()));
            while let Some(newline) = pending.find('\n') {
                let line: String = pending.drain(..=newline).collect();
                let line = line.trim_end_matches(['\r', '\n']);
                if line.starts_with("data:") {
                    arrivals.push(started.elapsed().as_secs_f64());
                    chunks.push(line.to_string());
                }
            }
        }

        assert_eq!(chunks.len(), 5);
        // The child sleeps 50ms between chunks.  Had the supervisor buffered
        // the response, every chunk would have landed at the same moment.
        assert!(arrivals[arrivals.len() - 1] - arrivals[0] > 0.1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_every_chunk_arrives_intact() {
        let app = supervisor_app().await;
        let served = serve(app.app.clone()).await;

        let client = reqwest::Client::builder().no_proxy().timeout(Duration::from_secs(30)).build().unwrap();
        let body = client
            .get(format!("{}/v1/projects/alpha/stream", served.base_url))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();

        for index in 0..5 {
            assert!(body.contains(&format!("data: chunk-{index}")));
        }
    }
}

mod test_when_a_child_cannot_be_reached {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_a_project_that_will_not_start_is_a_503() {
        let app = supervisor_app().await;
        *app.pool.fail_with.lock() = Some("index is corrupt".into());

        let (status, body) = call(&app.app, "GET", "/v1/projects/alpha/documents", Body::empty(), &[]).await;

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(body["detail"].as_str().unwrap().contains("index is corrupt"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_an_agent_that_does_not_answer_is_a_502() {
        let pool = FakePool::new("http://127.0.0.1:1"); // nothing listens there
        async fn models() -> Json<Value> {
            Json(json!({"served_by": "supervisor"}))
        }
        let app = with_routing(Router::new().route("/v1/models", get(models)), provider(&pool));

        let (status, body) = call(&app, "GET", "/v1/projects/alpha/documents", Body::empty(), &[]).await;

        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert!(body["detail"].as_str().unwrap().contains("did not answer"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_the_supervisor_keeps_serving_its_own_routes() {
        let app = supervisor_app().await;
        *app.pool.fail_with.lock() = Some("nope".into());

        let (status, _) = call(&app.app, "GET", "/v1/projects/alpha/documents", Body::empty(), &[]).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(get_json(&app.app, "/v1/models").await, json!({"served_by": "supervisor"}));
    }
}

/// Every unsupervised install: the middleware must be invisible.
mod test_without_a_supervisor {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_no_pool_means_nothing_is_routed() {
        let _child = child().await;
        async fn documents(Path(project_id): Path<String>) -> Json<Value> {
            Json(json!({"served_by": "single-process", "project_id": project_id}))
        }
        let app = with_routing(Router::new().route("/v1/projects/{project_id}/documents", get(documents)), no_pool());

        assert_eq!(
            get_json(&app, "/v1/projects/alpha/documents").await,
            json!({"served_by": "single-process", "project_id": "alpha"})
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_a_chat_body_is_not_even_read() {
        let _child = child().await;
        async fn chat(request: Request) -> Json<Value> {
            let body = request.into_body().collect().await.unwrap().to_bytes();
            Json(json!({"body": String::from_utf8_lossy(&body)}))
        }
        let app = with_routing(Router::new().route("/v1/chat/completions", post(chat)), no_pool());

        let body = post_json(&app, "/v1/chat/completions", json!({"project_id": "alpha"})).await;
        let inner: Value = serde_json::from_str(body["body"].as_str().unwrap()).unwrap();
        assert_eq!(inner["project_id"], "alpha");
    }
}
