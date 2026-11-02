//! `CloudLLM` with `SLP_CLOUD_DIRECT`: a session key from the platform, the
//! chat sent straight to the provider.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};
use serial_test::serial;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

use smartloop::llms::base_llm::{BaseLLM, CallOptions};
use smartloop::llms::cloud_llm::{describe_provider_failure, CloudLLM};
use smartloop::llms::cloud_session::{self, CloudSession};
use smartloop::platform::client::{AuthHeadersFn, Headers};

use crate::shared::workspace_support::EnvGuard;

const PROMPT: &str = "<system>Be brief.</system><user>hi</user>";

fn bearer(token: &'static str) -> AuthHeadersFn {
    Arc::new(move || Headers::from([("Authorization".to_string(), format!("Bearer {token}"))]))
}

fn sse(texts: &[&str]) -> String {
    let mut body = String::from(": OPENROUTER PROCESSING\n\n");
    for text in texts {
        let chunk = json!({"choices": [{"index": 0, "delta": {"content": text}}]});
        body.push_str(&format!("data: {chunk}\n\n"));
    }
    body.push_str("data: [DONE]\n\n");
    body
}

/// A platform at `platform` handing out `keys` in turn, one per session.
async fn mount_sessions(platform: &MockServer, provider: &MockServer, keys: &'static [&'static str]) -> Arc<AtomicUsize> {
    let minted = Arc::new(AtomicUsize::new(0));
    let counter = minted.clone();
    let provider_uri = provider.uri();
    Mock::given(method("POST"))
        .and(path("/v1/chat/session"))
        .and(header("Authorization", "Bearer platform-token"))
        .respond_with(move |_: &Request| {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            let key = keys[n.min(keys.len() - 1)];
            ResponseTemplate::new(200).set_body_json(session_body_from(&provider_uri, key))
        })
        .mount(platform)
        .await;
    minted
}

fn session_body_from(provider_uri: &str, key: &str) -> Value {
    json!({
        "base_url": format!("{provider_uri}/api/v1"),
        "key": key,
        "expires_at": (chrono::Utc::now() + chrono::Duration::minutes(30)).to_rfc3339(),
        "headers": {"X-Title": "Smartloop", "Authorization": "Bearer not-this-one"},
        "models": [{
            "name": "kimi-k2",
            "provider_model": "moonshotai/kimi-k2",
            "max_tokens": 1000,
            "body": {"provider": {"sort": "price"}, "usage": {"include": true}}
        }]
    })
}

/// A direct-mode model whose platform is `platform`.
fn direct_llm(platform: &MockServer, name: &str) -> (CloudLLM, EnvGuard, EnvGuard) {
    let base = EnvGuard::set("SLP_API_BASE", platform.uri());
    let direct = EnvGuard::set("SLP_CLOUD_DIRECT", "true");
    // Mock servers reuse ports, so a session cached by an earlier test
    // could otherwise answer for this one.
    cloud_session::clear();
    let llm = CloudLLM::with_hooks("p1", name, &platform.uri(), Some(8192), Some(bearer("platform-token")), Some(Arc::new(|| false)));
    (llm, base, direct)
}

async fn drain(llm: Arc<CloudLLM>) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            llm.stream(PROMPT, CallOptions::default()).map(|c| c.text).collect::<String>()
        }))
        .map_err(|panic| {
            panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default()
        })
    })
    .await
    .unwrap()
}

#[tokio::test]
#[serial]
async fn test_chat_goes_to_the_provider_with_the_session_key() {
    let (platform, provider) = (MockServer::start().await, MockServer::start().await);
    let minted = mount_sessions(&platform, &provider, &["sk-sub-1"]).await;
    let seen: Arc<parking_lot::Mutex<Vec<(Value, Option<String>)>>> = Arc::default();
    let log = seen.clone();
    Mock::given(method("POST"))
        .and(path("/api/v1/chat/completions"))
        .respond_with(move |req: &Request| {
            let auth = req.headers.get("authorization").map(|v| v.to_str().unwrap().to_string());
            log.lock().push((serde_json::from_slice(&req.body).unwrap(), auth));
            ResponseTemplate::new(200).set_body_string(sse(&["Hel", "lo"]))
        })
        .mount(&provider)
        .await;

    let (llm, _b, _d) = direct_llm(&platform, "kimi-k2");
    assert!(llm.is_direct());
    let llm = Arc::new(llm);
    assert_eq!(drain(llm.clone()).await.unwrap(), "Hello");
    assert_eq!(drain(llm).await.unwrap(), "Hello");

    // One session serves both chats; the platform never sees a chat.
    assert_eq!(minted.load(Ordering::SeqCst), 1);
    let chats = platform.received_requests().await.unwrap();
    assert!(chats.iter().all(|r| r.url.path() == "/v1/chat/session"));

    let seen = seen.lock();
    assert_eq!(seen.len(), 2);
    let (body, auth) = &seen[0];
    // The session's own key, not the platform token, and not an extra
    // header trying to replace it.
    assert_eq!(auth.as_deref(), Some("Bearer sk-sub-1"));
    assert_eq!(body["model"], "moonshotai/kimi-k2");
    assert_eq!(body["provider"]["sort"], "price");
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"]["include_usage"], true);
    assert!(body.get("project_id").is_none(), "the project id stays home");
    // Held to the server's cap.
    assert_eq!(body["max_tokens"], 1000);
}

#[tokio::test]
#[serial]
async fn test_a_refused_key_is_renewed_once() {
    let (platform, provider) = (MockServer::start().await, MockServer::start().await);
    let minted = mount_sessions(&platform, &provider, &["sk-old", "sk-new"]).await;
    Mock::given(method("POST"))
        .and(header("Authorization", "Bearer sk-old"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({"error": {"message": "key expired"}})))
        .mount(&provider)
        .await;
    Mock::given(method("POST"))
        .and(header("Authorization", "Bearer sk-new"))
        .respond_with(ResponseTemplate::new(200).set_body_string(sse(&["ok"])))
        .mount(&provider)
        .await;

    let (llm, _b, _d) = direct_llm(&platform, "kimi-k2");
    assert_eq!(drain(Arc::new(llm)).await.unwrap(), "ok");
    assert_eq!(minted.load(Ordering::SeqCst), 2);
}

#[tokio::test]
#[serial]
async fn test_a_spent_allowance_reports_the_platforms_reason() {
    let (platform, provider) = (MockServer::start().await, MockServer::start().await);
    let provider_uri = provider.uri();
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    // The first session works; asked again, the platform says the allowance is spent.
    Mock::given(method("POST"))
        .and(path("/v1/chat/session"))
        .respond_with(move |_: &Request| match counter.fetch_add(1, Ordering::SeqCst) {
            0 => ResponseTemplate::new(200).set_body_json(session_body_from(&provider_uri, "sk-1")),
            _ => ResponseTemplate::new(429).set_body_json(json!({"detail": "Allowance used. Resets Monday."})),
        })
        .mount(&platform)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(402).set_body_json(json!({"error": {"message": "limit"}})))
        .mount(&provider)
        .await;

    let (llm, _b, _d) = direct_llm(&platform, "kimi-k2");
    assert_eq!(drain(Arc::new(llm)).await.unwrap_err(), "Allowance used. Resets Monday.");
}

#[tokio::test]
#[serial]
async fn test_a_model_outside_the_session_is_refused_locally() {
    let (platform, provider) = (MockServer::start().await, MockServer::start().await);
    mount_sessions(&platform, &provider, &["sk-1"]).await;

    let (llm, _b, _d) = direct_llm(&platform, "not-on-plan");
    let err = drain(Arc::new(llm)).await.unwrap_err();
    assert_eq!(err, "not-on-plan isn't available on your current plan.");
    assert!(provider.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
#[serial]
async fn test_signed_out_session_asks_to_sign_in() {
    let (platform, provider) = (MockServer::start().await, MockServer::start().await);
    Mock::given(method("POST"))
        .and(path("/v1/chat/session"))
        .respond_with(ResponseTemplate::new(401))
        .mount(&platform)
        .await;

    let (llm, _b, _d) = direct_llm(&platform, "kimi-k2");
    assert_eq!(drain(Arc::new(llm)).await.unwrap_err(), "Sign in to use kimi-k2.");
    assert!(provider.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
#[serial]
async fn test_an_insecure_provider_endpoint_is_refused() {
    let (platform, provider) = (MockServer::start().await, MockServer::start().await);
    let mut body = session_body_from(&provider.uri(), "sk-1");
    body["base_url"] = json!("http://provider.example.com/api/v1");
    Mock::given(method("POST"))
        .and(path("/v1/chat/session"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&platform)
        .await;

    let (llm, _b, _d) = direct_llm(&platform, "kimi-k2");
    assert_eq!(drain(Arc::new(llm)).await.unwrap_err(), "The cloud session named an insecure endpoint.");
}

#[test]
#[serial]
fn test_direct_is_the_default() {
    let _env = EnvGuard::set("SLP_API_BASE", "https://api.smartloop.ai");
    let _unset = EnvGuard::remove("SLP_CLOUD_DIRECT");
    assert!(CloudLLM::new("p1", "kimi-k2", "https://api.smartloop.ai", None).is_direct());
    // A model served from elsewhere has no session to call it with.
    assert!(!CloudLLM::new("p1", "other", "https://models.example.com", None).is_direct());
}

#[test]
#[serial]
fn test_the_proxy_can_still_be_chosen() {
    let _env = EnvGuard::set("SLP_API_BASE", "https://api.smartloop.ai");
    let _off = EnvGuard::set("SLP_CLOUD_DIRECT", "false");
    assert!(!CloudLLM::new("p1", "kimi-k2", "https://api.smartloop.ai", None).is_direct());
}

#[test]
fn test_session_debug_hides_the_key() {
    let session: CloudSession = serde_json::from_value(json!({
        "base_url": "https://openrouter.ai/api/v1",
        "key": "sk-or-secret",
        "expires_at": "2030-01-01T00:00:00Z",
        "models": []
    }))
    .unwrap();
    let shown = format!("{session:?}");
    assert!(!shown.contains("sk-or-secret"));
    assert!(shown.contains("<redacted>"));
}

#[test]
fn test_provider_failures_read_as_plain_messages() {
    assert_eq!(describe_provider_failure(402, "", "m"), "You've used your cloud allowance for now.");
    assert_eq!(describe_provider_failure(404, "", "m"), "m isn't available on your current plan.");
    assert_eq!(describe_provider_failure(400, r#"{"error":{"message":"prompt too long"}}"#, "m"), "prompt too long");
    assert_eq!(describe_provider_failure(503, "", "m"), "m did not answer (HTTP 503).");
}
