//! Client tools: a coding client (docent) sends its own `bash` with the turn
//! and runs the calls itself.  Smartloop offers the tools to any response
//! model that calls functions natively, streams back the calls it makes, and takes
//! the results on the same turn without re-planning it.

use std::sync::Arc;

use futures::StreamExt;
use serde_json::{json, Value};
use serial_test::serial;

use smartloop::api::model::chat::{ChatCompletionRequest, ChatMessage, Role};
use parking_lot::Mutex;
use smartloop::api::routers::chat::{reply_tool_calls, split_tool_rounds, stream_chunks, streamable_end, ChatService};
use smartloop::model::spec::MODEL_SPECS;

use crate::shared::api_chat::{chunk, FakeChatService};

/// Publish the catalog rows these tests answer with (the registry is
/// process-wide, so the names are their own).
fn publish_models() {
    let rows = [
        json!({"name": "ct-cloud", "type": "remote", "capabilities": ["text", "tools"]}),
        json!({"name": "ct-small", "type": "local", "model_id": "Org/ct-small", "param_size_b": 4.0}),
        json!({"name": "ct-large", "type": "local", "model_id": "Org/ct-large", "param_size_b": 27.0}),
    ]
    .map(|v| v.as_object().unwrap().clone());
    MODEL_SPECS.hydrate(&rows);
}

fn bash_tool() -> Value {
    json!({"type": "function", "function": {"name": "bash", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}})
}

fn message(role: Role, content: &str) -> ChatMessage {
    ChatMessage { role, content: Some(content.to_string()), ..Default::default() }
}

fn first_turn() -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "sl-mini".to_string(),
        messages: vec![message(Role::User, "list the project files")],
        session_id: Some("s1".to_string()),
        stream: true,
        tools: Some(vec![bash_tool()]),
        ..ChatCompletionRequest::default()
    }
}

/// The first turn plus one round: the model called `bash`, the client ran it.
fn second_round() -> ChatCompletionRequest {
    let mut request = first_turn();
    request.messages.push(ChatMessage {
        role: Role::Assistant,
        content: Some("Let me look.".to_string()),
        tool_calls: Some(vec![json!({"id": "call_0", "type": "function",
            "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}})]),
        ..Default::default()
    });
    request.messages.push(ChatMessage {
        role: Role::Tool,
        content: Some("exit 0\nCargo.toml\nsrc".to_string()),
        tool_call_id: Some("call_0".to_string()),
        ..Default::default()
    });
    request
}

/// A service whose response model is `label`, calling tools natively or not,
/// that streams a tools turn's `reply` in pieces of 7 characters (recording
/// its prompt in `prompts`) and answers a plain turn with "plain".
fn service(label: &str, native_tools: bool, reply: &str) -> (Arc<FakeChatService>, Arc<Mutex<Vec<String>>>) {
    publish_models();
    let prompts: Arc<Mutex<Vec<String>>> = Arc::default();
    let (seen, reply) = (prompts.clone(), reply.to_string());
    let svc = Arc::new(FakeChatService::with_stream(move |prompt, _| {
        if !prompt.starts_with("<slp_tools>") {
            return Box::new(vec![Ok(chunk("plain"))].into_iter());
        }
        seen.lock().push(prompt.to_string());
        let pieces: Vec<_> = reply.chars().collect::<Vec<_>>().chunks(7).map(|c| Ok(chunk(&c.iter().collect::<String>()))).collect();
        Box::new(pieces.into_iter())
    }));
    *svc.device.label.lock() = Some(label.to_string());
    *svc.device.native_tools.lock() = native_tools;
    (svc, prompts)
}

async fn run(request: ChatCompletionRequest, svc: &Arc<FakeChatService>) -> Vec<Value> {
    let events: Vec<String> = stream_chunks(request, svc.clone() as Arc<dyn ChatService>).collect().await;
    events
        .iter()
        .filter_map(|e| e.strip_prefix("data: "))
        .filter_map(|e| serde_json::from_str(e.trim()).ok())
        .collect()
}

fn deltas(events: &[Value]) -> Vec<&Value> {
    events.iter().filter_map(|e| e.pointer("/choices/0/delta")).collect()
}

fn text(events: &[Value]) -> String {
    deltas(events).iter().filter_map(|d| d["content"].as_str()).collect()
}

fn finish(events: &[Value]) -> Option<String> {
    events.iter().filter_map(|e| e.pointer("/choices/0/finish_reason")?.as_str().map(str::to_string)).last()
}

fn tool_calls(events: &[Value]) -> Vec<Value> {
    deltas(events).iter().filter_map(|d| d["tool_calls"].as_array()).flatten().cloned().collect()
}

#[test]
fn test_a_request_without_tool_rounds_is_left_whole() {
    let (request, rounds) = split_tool_rounds(first_turn());
    assert_eq!(request.messages.len(), 1);
    assert!(rounds.is_empty());
}

#[test]
fn test_tool_rounds_become_openai_messages() {
    let (request, rounds) = split_tool_rounds(second_round());
    assert_eq!(request.messages.len(), 1, "the rounds leave the flattened turn");
    assert_eq!(rounds[0]["role"], "assistant");
    // Arguments as an object, for a local chat template.
    assert_eq!(rounds[0]["tool_calls"][0]["function"]["arguments"], json!({"command": "ls"}));
    assert_eq!(rounds[1], json!({"role": "tool", "tool_call_id": "call_0", "name": "bash",
        "content": "exit 0\nCargo.toml\nsrc"}));
}

#[test]
fn test_a_reply_splits_into_text_and_openai_calls() {
    let (text, calls) =
        reply_tool_calls("Let me look.\n<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>");
    assert_eq!(text, "Let me look.");
    assert_eq!(calls, vec![json!({"index": 0, "id": "call_0", "type": "function",
        "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}})]);

    let (text, calls) = reply_tool_calls("All done.");
    assert_eq!(text, "All done.");
    assert!(calls.is_empty());
}

#[tokio::test]
#[serial]
async fn test_a_cloud_model_gets_the_tools_and_its_calls_stream_back() {
    let (svc, prompts) = service("ct-cloud", true,
        "Let me look.\n<tool_call>{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\", \"id\": 1}}</tool_call>");
    let events = run(first_turn(), &svc).await;

    // Streamed in pieces, the markup never shown.
    assert!(deltas(&events).iter().filter(|d| d["content"].is_string()).count() > 1, "streamed, not in one piece");
    assert_eq!(text(&events).trim_end(), "Let me look.");
    assert!(!text(&events).contains('<'));
    let calls = tool_calls(&events);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["function"]["name"], "bash");
    assert_eq!(finish(&events).as_deref(), Some("tool_calls"));

    let prompts = prompts.lock().clone();
    assert!(prompts[0].starts_with("<slp_tools>"), "the tools ride with the prompt");
    assert_eq!(svc.orchestrate_calls.lock().len(), 1, "the first round is planned as usual");
    // The turn isn't over: only the user's message is stored so far.
    let writes = svc.session_writes.lock().clone();
    assert_eq!(writes.iter().map(|w| w.1.as_str()).collect::<Vec<_>>(), vec!["user"]);
}

#[tokio::test]
#[serial]
async fn test_a_tool_result_continues_the_turn_without_replanning() {
    let (svc, prompts) = service("ct-large", true, "There are two entries: Cargo.toml and src.");
    let events = run(second_round(), &svc).await;

    assert_eq!(text(&events), "There are two entries: Cargo.toml and src.");
    assert!(tool_calls(&events).is_empty());
    assert_eq!(finish(&events).as_deref(), Some("stop"));
    assert!(svc.orchestrate_calls.lock().is_empty(), "a tool round isn't re-planned");

    let prompt = prompts.lock()[0].clone();
    assert!(prompt.contains("\"tool_call_id\":\"call_0\""), "the result reaches the model");
    // The answer completes the turn; the user's message was stored with
    // the first round.
    let writes = svc.session_writes.lock().clone();
    assert_eq!(writes.iter().map(|w| w.1.as_str()).collect::<Vec<_>>(), vec!["assistant"]);
}

#[tokio::test]
#[serial]
async fn test_a_small_local_model_that_calls_tools_is_offered_them() {
    let (svc, prompts) = service("ct-small", true, "unused");
    run(first_turn(), &svc).await;

    assert_eq!(prompts.lock().len(), 1, "tools offered to a 4B model that takes them");
}

#[tokio::test]
#[serial]
async fn test_a_model_without_native_tools_answers_without_them() {
    let (svc, _) = service("ct-cloud", false, "unused");
    let events = run(first_turn(), &svc).await;

    assert_eq!(text(&events), "plain");
    assert!(tool_calls(&events).is_empty());
}

#[test]
fn test_markup_and_a_tail_that_may_start_it_are_held_back() {
    assert_eq!(streamable_end("Let me look."), 12);
    assert_eq!(streamable_end("Let me look.\n<tool_call>{"), 13);
    assert_eq!(streamable_end("Let me look.\n<too"), 13, "may be the start of <tool_call>");
    assert_eq!(streamable_end("a < b"), 5, "a plain < is shown");
    assert_eq!(streamable_end("x<function=bash>"), 1);
}

/// A model answering without the tools that writes a call in its own markup
/// anyway: the client gets it as `tool_calls`, never the markup.
async fn written_call(reply: &'static str) -> Vec<Value> {
    publish_models();
    let svc = Arc::new(FakeChatService::with_stream(move |_, _| {
        let pieces: Vec<_> = reply.chars().collect::<Vec<_>>().chunks(5).map(|c| Ok(chunk(&c.iter().collect::<String>()))).collect();
        Box::new(pieces.into_iter())
    }));
    *svc.device.label.lock() = Some("ct-cloud".to_string());
    *svc.device.native_tools.lock() = false;
    run(first_turn(), &svc).await
}

#[tokio::test]
#[serial]
async fn test_a_gemma_call_written_without_tools_comes_back_as_tool_calls() {
    let events = written_call("Let me look.\n<|tool_call>call:bash{command:<|\"|>git status<|\"|>}<tool_call|>").await;

    assert_eq!(text(&events).trim(), "Let me look.");
    let calls = tool_calls(&events);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["function"]["name"], "bash");
    assert_eq!(calls[0]["function"]["arguments"], r#"{"command":"git status"}"#);
    assert_eq!(finish(&events).as_deref(), Some("tool_calls"));
}

#[tokio::test]
#[serial]
async fn test_text_that_only_looks_like_a_call_still_shows() {
    let events = written_call("Use the <function= syntax.").await;

    assert_eq!(text(&events), "Use the <function= syntax.");
    assert!(tool_calls(&events).is_empty());
    assert_eq!(finish(&events).as_deref(), Some("stop"));
}
