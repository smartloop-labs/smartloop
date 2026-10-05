//! Port of `tests/api/conftest.py`.

// ===== BEGIN api_routes_misc =====================================================

/// An in-process HTTP client over an axum `Router` — the stand-in for
/// FastAPI's `TestClient`: no socket, every request goes through
/// `tower::ServiceExt::oneshot`, and redirects are never followed.
#[allow(dead_code)]
pub mod misc_client {
    use axum::body::{Body, Bytes};
    use axum::http::{HeaderMap, Method, Request};
    use axum::Router;
    use http_body_util::BodyExt;
    use serde_json::Value;
    use tower::ServiceExt;

    /// What a request answered.
    #[derive(Debug, Clone)]
    pub struct TestResponse {
        pub status: u16,
        pub headers: HeaderMap,
        pub body: Bytes,
    }

    impl TestResponse {
        /// The body parsed as JSON (`resp.json()`).
        pub fn json(&self) -> Value {
            serde_json::from_slice(&self.body)
                .unwrap_or_else(|e| panic!("response is not JSON ({e}): {:?}", self.text()))
        }

        /// The body as text (`resp.text`).
        pub fn text(&self) -> String {
            String::from_utf8_lossy(&self.body).into_owned()
        }

        /// A response header, if present.
        pub fn header(&self, name: &str) -> Option<String> {
            self.headers.get(name).and_then(|v| v.to_str().ok()).map(str::to_owned)
        }
    }

    /// Send one request to *router*.
    pub async fn request(
        router: &Router,
        method: Method,
        uri: &str,
        content_type: Option<&str>,
        body: Body,
    ) -> TestResponse {
        let mut builder = Request::builder().method(method).uri(uri);
        if let Some(ct) = content_type {
            builder = builder.header("content-type", ct);
        }
        let request = builder.body(body).expect("build request");
        let response = router.clone().oneshot(request).await.expect("router is infallible");
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        let body = response.into_body().collect().await.expect("read body").to_bytes();
        TestResponse { status, headers, body }
    }

    /// `client.get(uri)`.
    pub async fn get(router: &Router, uri: &str) -> TestResponse {
        request(router, Method::GET, uri, None, Body::empty()).await
    }

    /// `client.delete(uri)`.
    pub async fn delete(router: &Router, uri: &str) -> TestResponse {
        request(router, Method::DELETE, uri, None, Body::empty()).await
    }

    /// `client.post(uri, json=body)`.
    pub async fn post_json(router: &Router, uri: &str, body: &Value) -> TestResponse {
        let bytes = serde_json::to_vec(body).expect("serialize body");
        request(router, Method::POST, uri, Some("application/json"), Body::from(bytes)).await
    }

    /// `client.patch(uri, json=body)`.
    pub async fn patch_json(router: &Router, uri: &str, body: &Value) -> TestResponse {
        let bytes = serde_json::to_vec(body).expect("serialize body");
        request(router, Method::PATCH, uri, Some("application/json"), Body::from(bytes)).await
    }

    /// `/path?k=v&...` with the parameters URL-encoded (`params=`).
    pub fn with_query(path: &str, params: &[(&str, &str)]) -> String {
        let query = url::form_urlencoded::Serializer::new(String::new())
            .extend_pairs(params.iter())
            .finish();
        if query.is_empty() { path.to_string() } else { format!("{path}?{query}") }
    }

    /// The `data: ` payloads of an SSE body, parsed as JSON.
    pub fn sse_frames(text: &str) -> Vec<Value> {
        text.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|data| serde_json::from_str(data).ok())
            .collect()
    }
}

// ===== END api_routes_misc =======================================================
