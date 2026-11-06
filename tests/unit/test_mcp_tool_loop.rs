//! A native model's own tool loop: it calls, reads each result as a
//! `role: tool` message, and calls again until it answers in text.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use serial_test::serial;

use smartloop::agents::mcp::mcp_agent::{clear_resolution_cache, MCPAgent};
use smartloop::agents::mcp::tool_loop::{run_tool_loop, MAX_TURNS};
use smartloop::llms::tool_calls::split_messages;
use smartloop::llms::tool_calls::split_tools;
use smartloop::workspace::workspace::{MCPServer, MCPToolDefinition};

use crate::shared::agents_mcp_agent::StubLLM;

fn obj(v: Value) -> Map<String, Value> {
    v.as_object().cloned().unwrap()
}

fn tool(name: &str, properties: Value, required: &[&str]) -> MCPToolDefinition {
    MCPToolDefinition {
        name: name.into(),
        description: Some(format!("{name} in Linear")),
        input_schema: Some(obj(json!({"type": "object", "properties": properties, "required": required}))),
    }
}

fn linear() -> MCPServer {
    MCPServer {
        id: "linear-1".into(),
        name: "Linear".into(),
        enabled: true,
        tools: vec![
            tool("list_users", json!({"query": {"type": "string"}}), &[]),
            tool(
                "save_issue",
                json!({"title": {"type": "string"}, "description": {"type": "string"}, "assigneeId": {"type": "string"}}),
                &["title"],
            ),
        ],
        ..Default::default()
    }
}

/// A model that answers each turn in order.
fn scripted(turns: Vec<&'static str>) -> Arc<StubLLM> {
    let turns = Arc::new(Mutex::new(turns.into_iter()));
    let llm = StubLLM::with_reply_fn(Arc::new(move |_, _, _| turns.lock().next().unwrap_or("done").to_string()));
    llm.set_supports_native_tools(true);
    llm
}

/// The conversation the model was shown on its `n`th turn.
fn conversation(llm: &StubLLM, n: usize) -> Vec<Value> {
    let prompt = llm.calls()[n].prompt.clone();
    let (rest, tools) = split_tools(&prompt);
    assert!(tools.is_some(), "every turn carries the tools");
    split_messages(rest).expect("every turn carries the conversation")
}

#[test]
#[serial]
fn test_the_model_looks_up_then_acts_on_what_it_found() {
    clear_resolution_cache();
    let llm = scripted(vec![
        r#"<tool_call>{"id": "c1", "name": "list_users", "arguments": {"query": "Chris"}}</tool_call>"#,
        r#"<tool_call>{"id": "c2", "name": "save_issue", "arguments": {"title": "Slow cloud models page", "assigneeId": "u-42"}}</tool_call>"#,
        "Created ENG-7 and assigned it to Chris.",
    ]);
    let agent = MCPAgent::new(llm.as_llm());
    let calls: Arc<Mutex<Vec<(String, Map<String, Value>)>>> = Arc::default();
    let seen = calls.clone();
    let call = move |_: &MCPServer, tool: &str, args: &Map<String, Value>| {
        seen.lock().push((tool.to_string(), args.clone()));
        Ok(match tool {
            "list_users" => r#"{"users":[{"id":"u-42","name":"Chris Coveney"}]}"#.to_string(),
            _ => r#"{"issue":{"identifier":"ENG-7"}}"#.to_string(),
        })
    };

    let outcome = run_tool_loop(&agent, &*llm, "Assign an issue to Chris: the cloud models page is slow", &[linear()], &call, &|_| {})
        .expect("the loop runs");

    assert_eq!(outcome.turns, 3);
    assert_eq!(outcome.final_text, "Created ENG-7 and assigned it to Chris.");
    let calls = calls.lock();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].0, "list_users");
    assert_eq!(calls[1], ("save_issue".to_string(), obj(json!({"title": "Slow cloud models page", "assigneeId": "u-42"}))));

    // The second turn saw the first call and its result, answered by id.
    let second = conversation(&llm, 1);
    assert_eq!(second[2]["role"], "assistant");
    assert_eq!(second[2]["tool_calls"][0]["id"], "c1");
    assert_eq!(second[3]["role"], "tool");
    assert_eq!(second[3]["tool_call_id"], "c1");
    assert!(second[3]["content"].as_str().unwrap().contains("u-42"));

    let transcript = outcome.transcript();
    assert!(transcript.contains("Linear.save_issue"));
    assert!(transcript.contains("Outcome: Created ENG-7"));
    assert_eq!(outcome.last_ok().unwrap().tool_name, "save_issue");
}

/// Calls without ids (local templates) are numbered so results can answer them.
#[test]
#[serial]
fn test_calls_without_ids_are_numbered() {
    let llm = scripted(vec!["<tool_call>\n<function=list_users>\n<parameter=query>\nChris\n</parameter>\n</function>\n</tool_call>", "ok"]);
    let agent = MCPAgent::new(llm.as_llm());
    let call = |_: &MCPServer, _: &str, _: &Map<String, Value>| Ok("[]".to_string());
    run_tool_loop(&agent, &*llm, "find chris", &[linear()], &call, &|_| {}).unwrap();
    let second = conversation(&llm, 1);
    assert_eq!(second[2]["tool_calls"][0]["id"], "call_1_1");
    assert_eq!(second[3]["tool_call_id"], "call_1_1");
}

/// A tool the catalog does not have is answered with an error, not called.
#[test]
#[serial]
fn test_an_unknown_tool_is_reported_back_to_the_model() {
    let llm = scripted(vec![r#"<tool_call>{"name": "delete_everything", "arguments": {}}</tool_call>"#, "Sorry."]);
    let agent = MCPAgent::new(llm.as_llm());
    let call = |_: &MCPServer, _: &str, _: &Map<String, Value>| -> Result<String, String> { panic!("must not be called") };
    let outcome = run_tool_loop(&agent, &*llm, "x", &[linear()], &call, &|_| {}).unwrap();
    assert!(!outcome.calls[0].ok);
    assert!(conversation(&llm, 1)[3]["content"].as_str().unwrap().contains("no tool named"));
}

/// A first turn that fails means the loop never started: the caller falls back.
#[test]
#[serial]
fn test_a_failed_first_turn_falls_back() {
    let llm = scripted(vec!["Error generating response: 429 nearly out"]);
    let agent = MCPAgent::new(llm.as_llm());
    let call = |_: &MCPServer, _: &str, _: &Map<String, Value>| Ok(String::new());
    assert!(run_tool_loop(&agent, &*llm, "x", &[linear()], &call, &|_| {}).is_none());
}

/// A model that never stops calling is stopped.
#[test]
#[serial]
fn test_the_loop_is_bounded() {
    let llm = StubLLM::with_reply_fn(Arc::new(|_, _, _| {
        r#"<tool_call>{"name": "list_users", "arguments": {}}</tool_call>"#.to_string()
    }));
    llm.set_supports_native_tools(true);
    let agent = MCPAgent::new(llm.as_llm());
    let call = |_: &MCPServer, _: &str, _: &Map<String, Value>| Ok("[]".to_string());
    let outcome = run_tool_loop(&agent, &*llm, "x", &[linear()], &call, &|_| {}).unwrap();
    assert_eq!(outcome.turns, MAX_TURNS);
    assert_eq!(llm.call_count(), MAX_TURNS);
    assert!(outcome.transcript().contains("stopped after"));
}

/// A failed call shows the model the tool's own error, and the same failure
/// twice stops the loop instead of running it to the turn cap.
#[test]
#[serial]
fn test_a_repeated_failure_stops_the_loop() {
    let llm = StubLLM::with_reply_fn(Arc::new(|_, _, _| {
        r#"<tool_call>{"name": "save_issue", "arguments": {"title": "x"}}</tool_call>"#.to_string()
    }));
    llm.set_supports_native_tools(true);
    let agent = MCPAgent::new(llm.as_llm());
    let call = |_: &MCPServer, _: &str, _: &Map<String, Value>| -> Result<String, String> {
        Err("`team` is required".to_string())
    };
    let outcome = run_tool_loop(&agent, &*llm, "x", &[linear()], &call, &|_| {}).unwrap();
    assert_eq!(outcome.turns, 2);
    assert_eq!(outcome.calls.len(), 2);
    assert!(outcome.calls[0].result.contains("`team` is required"));
    assert!(conversation(&llm, 1)[3]["content"].as_str().unwrap().contains("`team` is required"));
    assert!(outcome.transcript().contains("kept failing"));
}
