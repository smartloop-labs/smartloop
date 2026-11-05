//! Port of `tests/unit/test_mcp_progress_reporting.py`.
//!
//! Collect reports the MCP calls it makes, not just that it is busy.
//!
//! A question answered from a connected server spends most of its collect
//! inside MCP: a reasoned tool pick over the server's catalogue, the call,
//! then one follow-up fetch per record.  All of it used to sit behind a single
//! "Processing..." — web search announced itself, MCP did not — so a request
//! that was reading three Notion pages was indistinguishable from a hang.
//!
//! Python replaced `exp._call_mcp_tool` / `exp._fetch_record_details` with
//! mocks.  Here the call runs through a real `MCPToolCaller` whose transport is
//! stubbed (the refresh-and-retry every MCP path shares lives on the caller,
//! not at each call site), answering with the text the mock returned, and the
//! follow-up plan is replaced so it plans nothing.

use std::sync::Arc;

use serde_json::{json, Map, Value};

use smartloop::agents::mcp::explorer::{Explorer, ProgressCallback};
use smartloop::agents::mcp::mcp_agent::ResolvedCall;
use smartloop::workspace::workspace::MCPServer;

use crate::shared::explorer_fixtures::{
    bare_explorer, messages, planned, recording_sink, stubbed_caller, text_result, FakeWebSearch,
    ScriptedResolver,
};

/// An explorer whose resolution answers `picked` and whose transport
/// answers every call with `transport_text`.
fn explorer(picked: Option<ResolvedCall>, transport_text: &'static str, follow_ups: Vec<(&'static str, Value)>) -> Explorer {
    let (caller, _log) = stubbed_caller(move |_, _, _| text_result(transport_text));
    bare_explorer(
        None,
        Some(caller),
        ScriptedResolver::returning(picked),
        FakeWebSearch::new(false, "", &[]),
        planned(follow_ups),
    )
}

fn server(name: &str) -> MCPServer {
    MCPServer { id: name.to_string(), name: name.to_string(), ..Default::default() }
}

fn call(tool_name: &str, arguments: Value) -> Option<ResolvedCall> {
    let arguments: Map<String, Value> = arguments.as_object().cloned().unwrap_or_default();
    Some(ResolvedCall { server_id: "Notion".to_string(), tool_name: tool_name.to_string(), arguments, reasoned: false, native: false, also: Vec::new() })
}

// ------------------------------------------------- the resolved call is reported

#[test]
fn test_the_long_resolution_leg_announces_itself() {
    let exp = explorer(None, "", vec![]);
    let (sink, seen) = recording_sink();
    exp.run_mcp_server("q", &server("Notion"), None, Some(sink));

    let seen = seen.lock();
    assert!(!seen.is_empty(), "resolution reported nothing");
    let (step, status, message) = &seen[0];
    assert_eq!((step.as_str(), status.as_str()), ("tool", "processing"));
    assert!(message.contains("Notion"), "{message}");
}

#[test]
fn test_the_tool_and_its_arguments_are_named() {
    // "Calling Notion" cannot tell reading a page from searching for one,
    // and which ran is the thing a user watching this wants to know.
    let exp = explorer(call("notion-search", json!({"query": "founders agreement"})), "some records", vec![]);

    let (sink, seen) = recording_sink();
    exp.run_mcp_server("q", &server("Notion"), None, Some(sink));

    let messages = messages(&seen);
    assert!(messages.iter().any(|m| m.contains("notion-search")), "{messages:?}");
    assert!(messages.iter().any(|m| m.contains("founders agreement")), "{messages:?}");
}

#[test]
fn test_a_write_tool_left_to_an_execute_step_reports_no_call() {
    let exp = explorer(call("notion-create-pages", json!({})), "", vec![]);

    let (sink, seen) = recording_sink();
    exp.run_mcp_server("q", &server("Notion"), None, Some(sink));

    assert!(!messages(&seen).iter().any(|m| m.contains("Calling")));
}

// ------------------------------------------------ the follow-up fetches are reported

#[test]
fn test_each_record_read_is_reported() {
    let exp = explorer(
        None,
        "the page body",
        vec![("notion-fetch", json!({"id": "aaa"})), ("notion-fetch", json!({"id": "bbb"}))],
    );

    let (sink, seen) = recording_sink();
    exp.fetch_record_details(&server("Notion"), "notion-search", "payload", Some(sink));

    let messages = messages(&seen);
    assert_eq!(messages.len(), 2, "{messages:?}");
    assert!(messages[0].contains("1 of 2") && messages[1].contains("2 of 2"), "{messages:?}");
    assert!(messages.iter().all(|m| m.contains("notion-fetch")), "{messages:?}");
}

// ------------------------------------------------ reporting never breaks collect

#[test]
fn test_a_raising_callback_does_not_fail_the_call() {
    // The callback hops into the request's event loop, so it can fail long
    // after the client has gone.  Collect's job is the context.
    let exp = explorer(call("notion-search", json!({"query": "x"})), "records", vec![]);

    let boom: ProgressCallback = Arc::new(|_: &str, _: &str, _: &str| panic!("client is gone"));

    let out = exp.run_mcp_server("q", &server("Notion"), None, Some(boom));
    assert!(out.contains("notion-search"), "{out}");
}

#[test]
fn test_no_callback_is_fine() {
    let exp = explorer(None, "", vec![]);
    assert_eq!(exp.run_mcp_server("q", &server("Notion"), None, None), "");
}
