//! `CloudLLM`: a remote catalog model answered by the platform's
//! `/v1/chat/completions`.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};
use serial_test::serial;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

use smartloop::llms::base_llm::{BaseLLM, CallOptions};
use smartloop::llms::cloud_llm::{
    credentials_allowed, describe_failure, CloudLLM, DEFAULT_CONTEXT_LENGTH, DEFAULT_REASONING_BUDGET_TOKENS,
};
use smartloop::platform::client::{AuthHeadersFn, Headers};

use crate::shared::workspace_support::EnvGuard;

const PROMPT: &str = "<system>Be brief.</system><user>hi</user>";

fn sse(chunks: &[Value]) -> String {
    let mut body = String::from(": keep-alive\n\n");
    for chunk in chunks {
        body.push_str(&format!("data: {chunk}\n\n"));
    }
    body.push_str("data: [DONE]\n\n");
    body
}

fn delta(text: &str) -> Value {
    json!({"model": "kimi-k2", "choices": [{"index": 0, "delta": {"content": text}}]})
}

fn bearer(token: &'static str) -> AuthHeadersFn {
    Arc::new(move || Headers::from([("Authorization".to_string(), format!("Bearer {token}"))]))
}

/// A model served by `server`, treated as the platform API so credentials go
/// to it.
fn llm(server: &MockServer, auth: Option<AuthHeadersFn>) -> (CloudLLM, EnvGuard) {
    let env = EnvGuard::set("SLP_API_BASE", server.uri());
    let llm = CloudLLM::with_hooks("p1", "kimi-k2", &server.uri(), Some(8192), auth, Some(Arc::new(|| false)));
    (llm, env)
}

/// Drain a stream on a blocking thread, catching the panic a failure raises.
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
async fn test_streams_the_answer_from_the_platform() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer token-1"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse(&[delta("Hel"), delta("lo"), json!({"choices": [], "usage": {}})])),
        )
        .expect(1)
        .mount(&server)
        .await;
    let (llm, _env) = llm(&server, Some(bearer("token-1")));

    assert_eq!(drain(Arc::new(llm)).await.unwrap(), "Hello");

    let sent: Value = server.received_requests().await.unwrap()[0].body_json().unwrap();
    assert_eq!(sent["model"], "kimi-k2");
    assert_eq!(sent["stream"], true);
    assert_eq!(sent["project_id"], "p1");
    assert_eq!(
        sent["messages"],
        json!([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}])
    );
}

#[tokio::test]
#[serial]
async fn test_call_returns_the_whole_answer() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}}],
        })))
        .mount(&server)
        .await;
    let (llm, _env) = llm(&server, Some(bearer("t")));
    let llm = Arc::new(llm);

    let answer = tokio::task::spawn_blocking(move || {
        llm.call(PROMPT, None, CallOptions { max_tokens: Some(64), thinking: None })
    })
    .await
    .unwrap()
    .unwrap();

    assert_eq!(answer, "Hello!");
    let sent: Value = server.received_requests().await.unwrap()[0].body_json().unwrap();
    assert_eq!(sent["stream"], false);
    // The cap sizes the answer; the reasoning budget comes on top.
    assert_eq!(sent["max_tokens"], 64 + DEFAULT_REASONING_BUDGET_TOKENS);
}

#[tokio::test]
#[serial]
async fn test_the_catalog_reasoning_budget_replaces_the_default() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        })))
        .mount(&server)
        .await;
    let (llm, _env) = llm(&server, Some(bearer("t")));
    let llm = Arc::new(llm.with_reasoning_budget(Some(4096)));

    tokio::task::spawn_blocking(move || llm.call(PROMPT, None, CallOptions { max_tokens: Some(256), thinking: None }))
        .await
        .unwrap()
        .unwrap();

    let sent: Value = server.received_requests().await.unwrap()[0].body_json().unwrap();
    assert_eq!(sent["max_tokens"], 256 + 4096);
}

#[tokio::test]
#[serial]
async fn test_an_uncapped_request_stays_uncapped() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        })))
        .mount(&server)
        .await;
    let (llm, _env) = llm(&server, Some(bearer("t")));
    let llm = Arc::new(llm);

    tokio::task::spawn_blocking(move || llm.call(PROMPT, None, CallOptions::default())).await.unwrap().unwrap();

    let sent: Value = server.received_requests().await.unwrap()[0].body_json().unwrap();
    assert!(sent.get("max_tokens").is_none());
}

#[tokio::test]
#[serial]
async fn test_a_401_retries_once_with_refreshed_credentials() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(header("authorization", "Bearer stale"))
        .respond_with(ResponseTemplate::new(401))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(header("authorization", "Bearer fresh"))
        .respond_with(ResponseTemplate::new(200).set_body_string(sse(&[delta("ok")])))
        .mount(&server)
        .await;

    let refreshed = Arc::new(AtomicBool::new(false));
    let auth: AuthHeadersFn = {
        let refreshed = Arc::clone(&refreshed);
        Arc::new(move || {
            let token = if refreshed.load(Ordering::SeqCst) { "fresh" } else { "stale" };
            Headers::from([("Authorization".to_string(), format!("Bearer {token}"))])
        })
    };
    let refresh = {
        let refreshed = Arc::clone(&refreshed);
        Arc::new(move || {
            refreshed.store(true, Ordering::SeqCst);
            true
        })
    };
    let _env = EnvGuard::set("SLP_API_BASE", server.uri());
    let llm = CloudLLM::with_hooks("p1", "kimi-k2", &server.uri(), None, Some(auth), Some(refresh));

    assert_eq!(drain(Arc::new(llm)).await.unwrap(), "ok");
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
#[serial]
async fn test_credentials_stay_off_a_foreign_host() {
    let server = MockServer::start().await;
    let saw_auth = Arc::new(AtomicUsize::new(0));
    {
        let saw_auth = Arc::clone(&saw_auth);
        Mock::given(method("POST"))
            .respond_with(move |req: &Request| {
                if req.headers.contains_key("authorization") {
                    saw_auth.fetch_add(1, Ordering::SeqCst);
                }
                ResponseTemplate::new(200).set_body_string(sse(&[delta("ok")]))
            })
            .mount(&server)
            .await;
    }
    // The platform API is somewhere else, so the mock is a foreign host.
    let _env = EnvGuard::set("SLP_API_BASE", "https://api.smartloop.ai");
    let llm = CloudLLM::with_hooks("p1", "kimi-k2", &server.uri(), None, Some(bearer("secret")), None);

    assert_eq!(drain(Arc::new(llm)).await.unwrap(), "ok");
    assert_eq!(saw_auth.load(Ordering::SeqCst), 0);
}

#[tokio::test]
#[serial]
async fn test_a_refusal_is_reported_in_words_the_user_can_act_on() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(403).set_body_json(json!({"detail": "Upgrade required"})))
        .mount(&server)
        .await;
    let (llm, _env) = llm(&server, Some(bearer("t")));

    let err = drain(Arc::new(llm)).await.unwrap_err();
    assert_eq!(err, "kimi-k2 isn't available on your current plan.");
}

#[tokio::test]
#[serial]
async fn test_an_error_chunk_mid_stream_fails_the_stream() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(sse(&[
            delta("Hel"),
            json!({"error": {"message": "The model stopped answering."}}),
        ])))
        .mount(&server)
        .await;
    let (llm, _env) = llm(&server, Some(bearer("t")));

    let err = drain(Arc::new(llm)).await.unwrap_err();
    assert_eq!(err, "The model stopped answering.");
}

#[tokio::test]
#[serial]
async fn test_cancellation_stops_reading() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(sse(&[delta("a"), delta("b"), delta("c")])))
        .mount(&server)
        .await;
    let (llm, _env) = llm(&server, Some(bearer("t")));
    let cancel = Arc::new(AtomicBool::new(true));
    llm.set_cancelled(Some(cancel));

    assert_eq!(drain(Arc::new(llm)).await.unwrap(), "");
}

#[test]
#[serial]
fn test_context_length_comes_from_the_catalog() {
    let _env = EnvGuard::set("SLP_API_BASE", "https://api.smartloop.ai");
    let with = CloudLLM::new("p1", "kimi-k2", "https://api.smartloop.ai", Some(131072));
    let without = CloudLLM::new("p1", "kimi-k2", "https://api.smartloop.ai", None);
    assert_eq!(with.get_max_tokens(), 131072);
    assert_eq!(without.get_max_tokens(), DEFAULT_CONTEXT_LENGTH);
    assert!(with.llama_model().is_none());
    assert_eq!(with.model_label(), "kimi-k2");
}

#[test]
fn test_credentials_allowed_only_on_platform_hosts() {
    let base = "https://api.smartloop.ai";
    assert!(credentials_allowed("https://api.smartloop.ai", base));
    assert!(credentials_allowed("https://edge.smartloop.ai/v2", base));
    assert!(credentials_allowed("http://localhost:8888", "http://localhost:8888"));
    assert!(!credentials_allowed("https://smartloop.ai.evil.com", base));
    assert!(!credentials_allowed("https://openrouter.ai", base));
    assert!(!credentials_allowed("not a url", base));
}

#[test]
fn test_failure_messages() {
    assert_eq!(describe_failure(401, "", "kimi-k2"), "Sign in to use kimi-k2.");
    assert_eq!(describe_failure(429, "", "kimi-k2"), "kimi-k2 is busy right now. Try again shortly.");
    assert_eq!(
        describe_failure(429, r#"{"error": "Rate limit exceeded: 60 per 1 minute"}"#, "kimi-k2"),
        "kimi-k2 is busy right now. Try again shortly."
    );
    assert_eq!(
        describe_failure(
            429,
            r#"{"detail": "Cloud model usage limit reached on the paid plan. It resets on 2026-11-01."}"#,
            "kimi-k2"
        ),
        "Cloud model usage limit reached on the paid plan. It resets on 2026-11-01."
    );
    assert_eq!(describe_failure(400, r#"{"detail": "context length exceeded"}"#, "kimi-k2"), "context length exceeded");
    assert_eq!(describe_failure(502, "", "kimi-k2"), "kimi-k2 did not answer (HTTP 502).");
}

fn answer(content: &str, finish: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
    }))
}

#[tokio::test]
#[serial]
async fn test_a_truncated_answer_is_retried_with_twice_the_room() {
    let server = MockServer::start().await;
    Mock::given(method("POST")).respond_with(answer("Partial", "length")).up_to_n_times(1).mount(&server).await;
    Mock::given(method("POST")).respond_with(answer("The whole answer.", "stop")).mount(&server).await;
    let (llm, _env) = llm(&server, Some(bearer("t")));
    let llm = Arc::new(llm);

    let reply = tokio::task::spawn_blocking(move || {
        llm.call(PROMPT, None, CallOptions { max_tokens: Some(256), thinking: None })
    })
    .await
    .unwrap()
    .unwrap();

    assert_eq!(reply, "The whole answer.");
    let sent = server.received_requests().await.unwrap();
    assert_eq!(sent.len(), 2);
    let first = 256 + DEFAULT_REASONING_BUDGET_TOKENS;
    assert_eq!(sent[0].body_json::<Value>().unwrap()["max_tokens"], first);
    assert_eq!(sent[1].body_json::<Value>().unwrap()["max_tokens"], first * 2);
}

#[tokio::test]
#[serial]
async fn test_an_answer_still_truncated_comes_back_empty() {
    let server = MockServer::start().await;
    Mock::given(method("POST")).respond_with(answer("Cut off mid-", "length")).mount(&server).await;
    let (llm, _env) = llm(&server, Some(bearer("t")));
    let llm = Arc::new(llm);

    let reply = tokio::task::spawn_blocking(move || {
        llm.call(PROMPT, None, CallOptions { max_tokens: Some(256), thinking: None })
    })
    .await
    .unwrap()
    .unwrap();

    assert_eq!(reply, "");
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
#[serial]
async fn test_an_uncapped_answer_is_not_retried() {
    let server = MockServer::start().await;
    Mock::given(method("POST")).respond_with(answer("Long answer", "length")).mount(&server).await;
    let (llm, _env) = llm(&server, Some(bearer("t")));
    let llm = Arc::new(llm);

    let reply = tokio::task::spawn_blocking(move || llm.call(PROMPT, None, CallOptions::default())).await.unwrap().unwrap();

    assert_eq!(reply, "Long answer");
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}
