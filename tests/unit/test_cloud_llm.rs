//! `CloudLLM`: a remote catalog model answered by the platform's
//! `/v1/chat/completions`.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};
use serial_test::serial;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

use smartloop::llms::base_llm::{BaseLLM, BatchPrompt, CallOptions};
use smartloop::types::ReasoningEffort;
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
        llm.call(PROMPT, None, CallOptions { max_tokens: Some(64), thinking: None, ..Default::default() })
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

/// A server answering every chat request "ok", and a model on it listing
/// `efforts`.
async fn effort_llm(efforts: Vec<ReasoningEffort>) -> (MockServer, Arc<CloudLLM>, EnvGuard) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        })))
        .mount(&server)
        .await;
    let (llm, env) = llm(&server, Some(bearer("t")));
    (server, Arc::new(llm.with_reasoning_efforts(efforts)), env)
}

async fn call_with_effort(llm: Arc<CloudLLM>, effort: Option<ReasoningEffort>) {
    tokio::task::spawn_blocking(move || {
        llm.call(PROMPT, None, CallOptions { max_tokens: Some(256), reasoning_effort: effort, ..Default::default() })
    })
    .await
    .unwrap()
    .unwrap();
}

#[tokio::test]
#[serial]
async fn test_a_listed_reasoning_effort_is_sent() {
    let (server, llm, _env) = effort_llm(vec![ReasoningEffort::Low, ReasoningEffort::High]).await;
    call_with_effort(llm, Some(ReasoningEffort::Low)).await;

    let sent: Value = server.received_requests().await.unwrap()[0].body_json().unwrap();
    assert_eq!(sent["reasoning"], json!({"effort": "low"}));
    assert_eq!(sent["max_tokens"], 256 + DEFAULT_REASONING_BUDGET_TOKENS);
}

#[tokio::test]
#[serial]
async fn test_an_unlisted_reasoning_effort_sends_the_nearest_listed_one() {
    use ReasoningEffort::{High, Low, Medium, Minimal, None as Off};
    for (listed, requested, sent_effort) in [
        // A model that always reasons answers "none" at its lowest.
        (vec![Minimal, Low, Medium, High], Off, "minimal"),
        (vec![Low, High], Medium, "high"),
        (vec![Low], High, "low"),
    ] {
        let (server, llm, _env) = effort_llm(listed).await;
        call_with_effort(llm, Some(requested)).await;

        let sent: Value = server.received_requests().await.unwrap()[0].body_json().unwrap();
        assert_eq!(sent["reasoning"], json!({"effort": sent_effort}), "{requested:?}");
    }
}

#[tokio::test]
#[serial]
async fn test_no_reasoning_effort_is_sent_unless_asked_and_listed() {
    for (listed, requested) in [(vec![ReasoningEffort::Low], None), (vec![], Some(ReasoningEffort::Low))] {
        let (server, llm, _env) = effort_llm(listed).await;
        call_with_effort(llm, requested).await;

        let sent: Value = server.received_requests().await.unwrap()[0].body_json().unwrap();
        assert!(sent.get("reasoning").is_none(), "{requested:?}");
        assert_eq!(sent["max_tokens"], 256 + DEFAULT_REASONING_BUDGET_TOKENS, "{requested:?}");
    }
}

#[tokio::test]
#[serial]
async fn test_reasoning_off_adds_no_reasoning_budget() {
    let (server, llm, _env) = effort_llm(vec![ReasoningEffort::None, ReasoningEffort::Low]).await;
    call_with_effort(llm, Some(ReasoningEffort::None)).await;

    let sent: Value = server.received_requests().await.unwrap()[0].body_json().unwrap();
    assert_eq!(sent["reasoning"], json!({"effort": "none"}));
    assert_eq!(sent["max_tokens"], 256);
}

#[tokio::test]
#[serial]
async fn test_a_batch_prompt_carries_its_reasoning_effort() {
    let (server, llm, _env) = effort_llm(vec![ReasoningEffort::Minimal]).await;
    let prompts =
        vec![BatchPrompt::new(PROMPT, Some(256)).with_reasoning_effort(Some(ReasoningEffort::Minimal))];
    let replies = tokio::task::spawn_blocking(move || llm.invoke_batch(&prompts, None)).await.unwrap();

    assert_eq!(replies, vec!["ok".to_string()]);
    let sent: Value = server.received_requests().await.unwrap()[0].body_json().unwrap();
    assert_eq!(sent["reasoning"], json!({"effort": "minimal"}));
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

    tokio::task::spawn_blocking(move || {
        llm.call(PROMPT, None, CallOptions { max_tokens: Some(256), thinking: None, ..Default::default() })
    })
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
        llm.call(PROMPT, None, CallOptions { max_tokens: Some(256), thinking: None, ..Default::default() })
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
        llm.call(PROMPT, None, CallOptions { max_tokens: Some(256), thinking: None, ..Default::default() })
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

const EXTRACT: &str = "<system>Extract what is relevant.</system><user>Document {n}</user>";

fn extract_prompts(count: usize, cap: usize) -> Vec<BatchPrompt> {
    (1..=count).map(|n| BatchPrompt::new(EXTRACT.replace("{n}", &n.to_string()), Some(cap))).collect()
}

/// Answers each request with `reply(user message)`.
async fn answering(server: &MockServer, reply: fn(&str) -> String) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |request: &Request| {
            let body: Value = request.body_json().unwrap();
            let user = body["messages"][1]["content"].as_str().unwrap_or_default().to_string();
            ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply(&user)}, "finish_reason": "stop"}],
            }))
        })
        .mount(server)
        .await;
}

/// Each `<task id="N">` of a packed request answered as `answer N`, in
/// reverse order; a lone prompt answered `alone`.
fn answer_tasks(user: &str) -> String {
    let ids: Vec<usize> = user
        .match_indices("<task id=\"")
        .map(|(at, _)| user[at + 10..].split('"').next().unwrap().parse().unwrap())
        .collect();
    if ids.is_empty() {
        return "alone".to_string();
    }
    ids.iter().rev().map(|id| format!("<answer id=\"{id}\">answer {id}</answer>")).collect::<Vec<_>>().join("\n")
}

async fn run_batch(llm: CloudLLM, prompts: Vec<BatchPrompt>) -> Vec<String> {
    let llm = Arc::new(llm);
    tokio::task::spawn_blocking(move || llm.invoke_batch(&prompts, None)).await.unwrap()
}

#[tokio::test]
#[serial]
async fn test_a_batch_sharing_instructions_goes_out_as_one_request() {
    let server = MockServer::start().await;
    answering(&server, answer_tasks).await;
    let (llm, _env) = llm(&server, Some(bearer("t")));

    let replies = run_batch(llm, extract_prompts(3, 256)).await;

    assert_eq!(replies, vec!["answer 1", "answer 2", "answer 3"]);
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let sent: Value = requests[0].body_json().unwrap();
    let system = sent["messages"][0]["content"].as_str().unwrap();
    assert!(system.starts_with("Extract what is relevant."));
    assert_eq!(system.matches("Extract what is relevant.").count(), 1);
    let user = sent["messages"][1]["content"].as_str().unwrap();
    assert!(user.contains("<task id=\"3\">\nDocument 3\n</task>"));
    assert_eq!(sent["max_tokens"], 3 * 256 + DEFAULT_REASONING_BUDGET_TOKENS);
}

#[tokio::test]
#[serial]
async fn test_a_prompt_the_packed_reply_leaves_out_is_asked_alone() {
    let server = MockServer::start().await;
    answering(&server, |user| {
        if user.contains("<task") {
            "<answer id=\"1\">first</answer>\n<answer id=\"2\">cut o".to_string()
        } else {
            "alone".to_string()
        }
    })
    .await;
    let (llm, _env) = llm(&server, Some(bearer("t")));

    let replies = run_batch(llm, extract_prompts(2, 256)).await;

    assert_eq!(replies, vec!["first", "alone"]);
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    let retried: Value = requests[1].body_json().unwrap();
    assert_eq!(retried["messages"][1]["content"], "Document 2");
}

#[tokio::test]
#[serial]
async fn test_a_batch_splits_where_the_answer_budget_runs_out() {
    let server = MockServer::start().await;
    answering(&server, answer_tasks).await;
    let _env = EnvGuard::set("SLP_API_BASE", server.uri());
    let llm = CloudLLM::with_hooks("p1", "kimi-k2", &server.uri(), Some(1_000_000), Some(bearer("t")), None);
    let cap = smartloop::llms::cloud_llm::PACK_MAX_ANSWER_TOKENS / 2;

    let replies = run_batch(llm, extract_prompts(3, cap)).await;

    assert_eq!(replies, vec!["answer 1", "answer 2", "alone"]);
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
#[serial]
async fn test_a_batch_splits_where_the_window_runs_out() {
    let server = MockServer::start().await;
    answering(&server, answer_tasks).await;
    let _env = EnvGuard::set("SLP_API_BASE", server.uri());
    // Room for the reasoning budget and one task's text, not two.
    let llm = CloudLLM::with_hooks("p1", "kimi-k2", &server.uri(), Some(1600), Some(bearer("t")), None);
    let long = "x".repeat(900);
    let prompts: Vec<BatchPrompt> = (1..=2)
        .map(|n| BatchPrompt::new(format!("<system>Extract.</system><user>{n}{long}</user>"), Some(64)))
        .collect();

    let replies = run_batch(llm, prompts).await;

    assert_eq!(replies, vec!["alone", "alone"]);
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
#[serial]
async fn test_prompts_with_different_instructions_go_out_apart() {
    let server = MockServer::start().await;
    answering(&server, answer_tasks).await;
    let (llm, _env) = llm(&server, Some(bearer("t")));
    let prompts = vec![
        BatchPrompt::new("<system>One.</system><user>a</user>", Some(64)),
        BatchPrompt::new("<system>Two.</system><user>b</user>", Some(64)),
        BatchPrompt::new("<system>Two.</system><user>c</user>", None),
    ];

    let replies = run_batch(llm, prompts).await;

    assert_eq!(replies, vec!["alone", "alone", "alone"]);
    assert_eq!(server.received_requests().await.unwrap().len(), 3);
}

#[tokio::test]
#[serial]
async fn test_a_failed_packed_request_answers_each_prompt_with_the_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({"detail": "Allowance spent."})))
        .mount(&server)
        .await;
    let (llm, _env) = llm(&server, Some(bearer("t")));

    let replies = run_batch(llm, extract_prompts(2, 256)).await;

    assert_eq!(replies, vec!["Error generating response: Allowance spent."; 2]);
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
#[serial]
async fn test_a_cancelled_batch_sends_nothing() {
    let server = MockServer::start().await;
    answering(&server, answer_tasks).await;
    let (llm, _env) = llm(&server, Some(bearer("t")));
    let llm = Arc::new(llm);
    let prompts = extract_prompts(2, 256);
    let cancelled = Arc::new(AtomicBool::new(true));

    let replies = tokio::task::spawn_blocking(move || llm.invoke_batch(&prompts, Some(cancelled))).await.unwrap();

    assert_eq!(replies, vec![String::new(), String::new()]);
    assert!(server.received_requests().await.unwrap().is_empty());
}
