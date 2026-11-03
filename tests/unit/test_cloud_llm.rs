//! `CloudLLM`: a remote catalog model, answered by its provider with a
//! session key from the platform.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};
use serial_test::serial;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

use smartloop::llms::base_llm::{BaseLLM, CallOptions};
use smartloop::llms::cloud_llm::{describe_failure, CloudLLM, DEFAULT_ANSWER_TOKENS, DEFAULT_CONTEXT_LENGTH, DEFAULT_REASONING_BUDGET_TOKENS};
use smartloop::llms::cloud_session;
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

/// The session key the mock platform hands out.
pub(crate) const SESSION_KEY: &str = "sk-session";

/// Make `server` the platform: it hands out a session for `models` (each its
/// own provider id) whose provider is `provider`, at `/chat/completions`.
/// Sessions cached by an earlier test are dropped first: mock servers reuse
/// ports, so one could otherwise answer for this test.
pub(crate) async fn mount_session_at(server: &MockServer, provider: &str, models: &[&str]) {
    cloud_session::clear();
    let models: Vec<Value> = models.iter().map(|m| json!({"name": m, "provider_model": m})).collect();
    Mock::given(method("POST"))
        .and(path("/v1/chat/session"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "base_url": provider,
            "key": SESSION_KEY,
            "expires_at": (chrono::Utc::now() + chrono::Duration::minutes(30)).to_rfc3339(),
            "models": models,
        })))
        // Ahead of each test's catch-all provider mock.
        .with_priority(1)
        .mount(server)
        .await;
}

/// [`mount_session_at`] with `server` as the provider too.
pub(crate) async fn mount_session(server: &MockServer, models: &[&str]) {
    mount_session_at(server, &server.uri(), models).await;
}

/// The chats `server` received as the provider, without the session request.
pub(crate) async fn chats(server: &MockServer) -> Vec<Request> {
    server
        .received_requests()
        .await
        .unwrap()
        .into_iter()
        .filter(|r| r.url.path() != "/v1/chat/session")
        .collect()
}

/// A model whose platform and provider are both `server`.
async fn llm(server: &MockServer, auth: Option<AuthHeadersFn>) -> (CloudLLM, EnvGuard) {
    let env = EnvGuard::set("SLP_API_BASE", server.uri());
    mount_session(server, &["kimi-k2"]).await;
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
async fn test_streams_the_answer_from_the_provider() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer sk-session"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse(&[delta("Hel"), delta("lo"), json!({"choices": [], "usage": {}})])),
        )
        .expect(1)
        .mount(&server)
        .await;
    let (llm, _env) = llm(&server, Some(bearer("token-1"))).await;

    assert_eq!(drain(Arc::new(llm)).await.unwrap(), "Hello");

    let sent: Value = chats(&server).await[0].body_json().unwrap();
    assert_eq!(sent["model"], "kimi-k2");
    assert_eq!(sent["stream"], true);
    assert!(sent.get("project_id").is_none(), "the project id stays home");
    // The user's own token went to the platform, for the session.
    let session = &server.received_requests().await.unwrap()[0];
    assert_eq!(session.url.path(), "/v1/chat/session");
    assert_eq!(session.headers.get("authorization").unwrap(), "Bearer token-1");
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
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}}],
        })))
        .mount(&server)
        .await;
    let (llm, _env) = llm(&server, Some(bearer("t"))).await;
    let llm = Arc::new(llm);

    let answer = tokio::task::spawn_blocking(move || {
        llm.call(PROMPT, None, CallOptions { max_tokens: Some(64), thinking: None })
    })
    .await
    .unwrap()
    .unwrap();

    assert_eq!(answer, "Hello!");
    let sent: Value = chats(&server).await[0].body_json().unwrap();
    assert_eq!(sent["stream"], false);
    // The cap sizes the answer; the reasoning budget comes on top.
    assert_eq!(sent["max_tokens"], 64 + DEFAULT_REASONING_BUDGET_TOKENS);
}

#[tokio::test]
#[serial]
async fn test_the_catalog_reasoning_budget_replaces_the_default() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        })))
        .mount(&server)
        .await;
    let (llm, _env) = llm(&server, Some(bearer("t"))).await;
    let llm = Arc::new(llm.with_reasoning_budget(Some(4096)));

    tokio::task::spawn_blocking(move || llm.call(PROMPT, None, CallOptions { max_tokens: Some(256), thinking: None }))
        .await
        .unwrap()
        .unwrap();

    let sent: Value = chats(&server).await[0].body_json().unwrap();
    assert_eq!(sent["max_tokens"], 256 + 4096);
}

#[tokio::test]
#[serial]
async fn test_an_uncapped_request_gets_the_default_cap() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        })))
        .mount(&server)
        .await;
    let (llm, _env) = llm(&server, Some(bearer("t"))).await;
    let llm = Arc::new(llm);

    tokio::task::spawn_blocking(move || llm.call(PROMPT, None, CallOptions::default())).await.unwrap().unwrap();

    // The endpoint's own default would cut a long answer short, so the
    // request carries room for one, and the reasoning on top.
    let sent: Value = chats(&server).await[0].body_json().unwrap();
    assert_eq!(sent["max_tokens"], DEFAULT_ANSWER_TOKENS + DEFAULT_REASONING_BUDGET_TOKENS);
}

#[tokio::test]
#[serial]
async fn test_a_401_for_the_session_retries_once_with_refreshed_credentials() {
    let server = MockServer::start().await;
    cloud_session::clear();
    Mock::given(method("POST"))
        .and(path("/v1/chat/session"))
        .and(header("authorization", "Bearer stale"))
        .respond_with(ResponseTemplate::new(401))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/session"))
        .and(header("authorization", "Bearer fresh"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "base_url": server.uri(),
            "key": SESSION_KEY,
            "expires_at": (chrono::Utc::now() + chrono::Duration::minutes(30)).to_rfc3339(),
            "models": [{"name": "kimi-k2", "provider_model": "kimi-k2"}],
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
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
    assert_eq!(server.received_requests().await.unwrap().len(), 3, "two session requests, one chat");
}

/// The user's platform token goes to the platform alone; the provider only
/// ever sees the session key.
#[tokio::test]
#[serial]
async fn test_platform_credentials_never_reach_the_provider() {
    let platform = MockServer::start().await;
    let provider = MockServer::start().await;
    mount_session_at(&platform, &provider.uri(), &["kimi-k2"]).await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(sse(&[delta("ok")])))
        .mount(&provider)
        .await;
    let _env = EnvGuard::set("SLP_API_BASE", platform.uri());
    // Whatever url the catalog names, the session decides where chats go.
    let llm = CloudLLM::with_hooks("p1", "kimi-k2", "https://models.example.com", None, Some(bearer("secret")), None);

    assert_eq!(drain(Arc::new(llm)).await.unwrap(), "ok");
    let sent = provider.received_requests().await.unwrap();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].headers.get("authorization").unwrap(), "Bearer sk-session");
    assert_eq!(chats(&platform).await.len(), 0, "no chat reaches the platform");
}

#[tokio::test]
#[serial]
async fn test_a_refusal_is_reported_in_words_the_user_can_act_on() {
    let server = MockServer::start().await;
    cloud_session::clear();
    Mock::given(method("POST"))
        .and(path("/v1/chat/session"))
        .respond_with(ResponseTemplate::new(403).set_body_json(json!({"detail": "Upgrade required"})))
        .mount(&server)
        .await;
    let _env = EnvGuard::set("SLP_API_BASE", server.uri());
    let llm = CloudLLM::with_hooks("p1", "kimi-k2", &server.uri(), None, Some(bearer("t")), Some(Arc::new(|| false)));

    let err = drain(Arc::new(llm)).await.unwrap_err();
    assert_eq!(err, "kimi-k2 isn't available on your current plan.");
    assert_eq!(chats(&server).await.len(), 0);
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
    let (llm, _env) = llm(&server, Some(bearer("t"))).await;

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
    let (llm, _env) = llm(&server, Some(bearer("t"))).await;
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
    let (llm, _env) = llm(&server, Some(bearer("t"))).await;
    let llm = Arc::new(llm);

    let reply = tokio::task::spawn_blocking(move || {
        llm.call(PROMPT, None, CallOptions { max_tokens: Some(256), thinking: None })
    })
    .await
    .unwrap()
    .unwrap();

    assert_eq!(reply, "The whole answer.");
    let sent = chats(&server).await;
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
    let (llm, _env) = llm(&server, Some(bearer("t"))).await;
    let llm = Arc::new(llm);

    let reply = tokio::task::spawn_blocking(move || {
        llm.call(PROMPT, None, CallOptions { max_tokens: Some(256), thinking: None })
    })
    .await
    .unwrap()
    .unwrap();

    assert_eq!(reply, "");
    assert_eq!(chats(&server).await.len(), 2);
}

#[tokio::test]
#[serial]
async fn test_an_uncapped_answer_cut_off_is_retried_with_more_room() {
    let server = MockServer::start().await;
    Mock::given(method("POST")).respond_with(answer("Long answer", "length")).mount(&server).await;
    let (llm, _env) = llm(&server, Some(bearer("t"))).await;
    let llm = Arc::new(llm);

    let reply = tokio::task::spawn_blocking(move || llm.call(PROMPT, None, CallOptions::default())).await.unwrap().unwrap();

    // Cut off at the default cap, it's asked again with twice the room; cut
    // off there too, the truncated answer is discarded, as a capped one is.
    assert_eq!(reply, "");
    let sent = chats(&server).await;
    assert_eq!(sent.len(), 2);
    let second: Value = sent[1].body_json().unwrap();
    assert_eq!(second["max_tokens"], (DEFAULT_ANSWER_TOKENS + DEFAULT_REASONING_BUDGET_TOKENS) * 2);
}

/// Refused for asking more answer tokens than the endpoint allows, the
/// request is sent again within the cap it named, and the cap holds for
/// requests after. The context window is another number and unaffected.
#[tokio::test]
#[serial]
async fn test_an_answer_cap_the_endpoint_names_is_learned() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(400).set_body_string("RuntimeError: max_tokens is at most 16384 for minimax-m3."))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST")).respond_with(answer("ok", "stop")).mount(&server).await;
    // Its own model name: a learned cap is kept for the process by endpoint
    // and model, and a mock server's port can come round again.
    let _env = EnvGuard::set("SLP_API_BASE", server.uri());
    mount_session(&server, &["minimax-m3"]).await;
    let llm = Arc::new(CloudLLM::with_hooks("p1", "minimax-m3", &server.uri(), Some(8192), Some(bearer("t")), Some(Arc::new(|| false))));

    let first = llm.clone();
    let reply = tokio::task::spawn_blocking(move || first.call(PROMPT, None, CallOptions::default())).await.unwrap().unwrap();
    assert_eq!(reply, "ok");
    let sent = chats(&server).await;
    assert_eq!(sent.len(), 2);
    let refused: Value = sent[0].body_json().unwrap();
    let within: Value = sent[1].body_json().unwrap();
    assert_eq!(refused["max_tokens"], DEFAULT_ANSWER_TOKENS + DEFAULT_REASONING_BUDGET_TOKENS);
    assert_eq!(within["max_tokens"], 16384);

    tokio::task::spawn_blocking(move || llm.call(PROMPT, None, CallOptions::default())).await.unwrap().unwrap();
    let sent = chats(&server).await;
    assert_eq!(sent.len(), 3);
    let next: Value = sent[2].body_json().unwrap();
    assert_eq!(next["max_tokens"], 16384);
}

/// The first token reaches the caller while the provider is still answering:
/// nothing between the socket and the stream waits for the whole body.  The
/// provider here sends one chunk, then holds the rest back until the caller
/// has that chunk; a client that buffered would never get it.
#[tokio::test]
#[serial]
async fn test_the_first_token_arrives_before_the_answer_ends() {
    use std::io::{Read, Write};
    use std::sync::mpsc;
    use std::time::Duration;

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let provider = format!("http://{}", listener.local_addr().unwrap());
    let (go, wait) = mpsc::channel::<()>();
    let released_in_time = Arc::new(AtomicBool::new(false));
    let server = {
        let released_in_time = Arc::clone(&released_in_time);
        std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            // Read the request: headers, then the body they announce.
            let mut request = Vec::new();
            let mut buf = [0u8; 4096];
            let body_start = loop {
                let n = socket.read(&mut buf).unwrap();
                request.extend_from_slice(&buf[..n]);
                if let Some(at) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                    break at + 4;
                }
            };
            let head = String::from_utf8_lossy(&request[..body_start]).to_lowercase();
            let length: usize = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length:").map(|v| v.trim().parse().unwrap()))
                .unwrap_or(0);
            while request.len() - body_start < length {
                let n = socket.read(&mut buf).unwrap();
                request.extend_from_slice(&buf[..n]);
            }
            let frame = |data: &str| {
                let event = format!("data: {data}\n\n");
                format!("{:x}\r\n{event}\r\n", event.len())
            };
            write!(
                socket,
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n{}",
                frame(&delta("Hel").to_string())
            )
            .unwrap();
            socket.flush().unwrap();
            // The rest only once the caller has the first chunk.
            released_in_time.store(wait.recv_timeout(Duration::from_secs(5)).is_ok(), Ordering::SeqCst);
            write!(socket, "{}{}0\r\n\r\n", frame(&delta("lo").to_string()), frame("[DONE]")).unwrap();
            socket.flush().unwrap();
        })
    };

    let platform = MockServer::start().await;
    mount_session_at(&platform, &provider, &["kimi-k2"]).await;
    let _env = EnvGuard::set("SLP_API_BASE", platform.uri());
    let llm = Arc::new(CloudLLM::with_hooks("p1", "kimi-k2", &platform.uri(), None, Some(bearer("t")), None));

    let (first, rest) = tokio::task::spawn_blocking(move || {
        let mut stream = llm.stream(PROMPT, CallOptions::default());
        let first = stream.next().map(|c| c.text);
        go.send(()).unwrap();
        let rest: String = stream.map(|c| c.text).collect();
        (first, rest)
    })
    .await
    .unwrap();
    server.join().unwrap();

    assert_eq!(first.as_deref(), Some("Hel"));
    assert_eq!(rest, "lo");
    assert!(released_in_time.load(Ordering::SeqCst), "the first token waited for the end of the answer");
}
