//! The OpenAPI document and its Swagger UI.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::Value;
use smartloop::api::openapi;
use tower::ServiceExt;

async fn get(uri: &str) -> (StatusCode, Vec<u8>) {
    let response = openapi::router()
        .oneshot(Request::get(uri).header("host", "127.0.0.1:8001").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    (status, response.into_body().collect().await.unwrap().to_bytes().to_vec())
}

#[tokio::test]
async fn serves_the_openapi_document() {
    let (status, body) = get(openapi::OPENAPI_PATH).await;
    assert_eq!(status, StatusCode::OK);
    let doc: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(doc["info"]["version"], smartloop::VERSION);
    assert_eq!(doc["servers"][0]["url"], "http://127.0.0.1:8001");
    let paths = doc["paths"].as_object().unwrap();
    for path in ["/v1/chat/completions", "/v1/chat/summary", "/v1/projects", "/v1/projects/{project_id}/mcp/{server_id}", "/health"] {
        assert!(paths.contains_key(path), "{path} missing from the OpenAPI document");
    }
    let operations: usize = paths.values().map(|item| item.as_object().unwrap().len()).sum();
    assert_eq!(operations, 67, "every route's handler should be listed in ApiDoc");
}

#[tokio::test]
async fn serves_the_scalar_reference_offline() {
    let (status, body) = get(openapi::DOCS_PATH).await;
    assert_eq!(status, StatusCode::OK);
    let page = String::from_utf8(body).unwrap();
    assert!(page.contains(openapi::SCALAR_JS_PATH) && page.contains(openapi::OPENAPI_PATH));
    // Everything the page loads is served from here.
    assert!(!page.contains("https://"), "the reference page must not load anything remote");

    let (status, body) = get(openapi::SCALAR_JS_PATH).await;
    assert_eq!(status, StatusCode::OK);
    assert!(String::from_utf8_lossy(&body).contains("createApiReference"));
}
