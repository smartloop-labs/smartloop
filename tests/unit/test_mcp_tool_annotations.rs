//! Tool annotations and titles survive discovery and project.json.
//!
//! Servers describe each tool's behaviour in `annotations` (`readOnlyHint`,
//! `destructiveHint`, ...) and give it a display `title`, at the top level,
//! inside `annotations`, or both; the spec gives the top-level one
//! precedence.  Discovery keeps both, and a project.json written before they
//! existed still loads.

use std::sync::Arc;

use serde_json::{json, Value};

use smartloop::mcp::client::{HttpBody, HttpResponse, MCPClient};
use smartloop::workspace::{MCPToolAnnotations, MCPToolDefinition};

use crate::shared::mcp::{ok_json, FakeHttp};

const SERVER_URL: &str = "https://mcp.example.com/mcp";

/// An MCP server whose `tools/list` returns `tools` as given.
fn server_listing(tools: Value) -> Arc<FakeHttp> {
    FakeHttp::new(move |_, _, body| {
        let method = match body {
            Some(HttpBody::Json(v)) => v.get("method").and_then(Value::as_str).unwrap_or("").to_string(),
            _ => String::new(),
        };
        Ok(match method.as_str() {
            "initialize" => ok_json(json!({"jsonrpc": "2.0", "id": "1", "result": {"serverInfo": {"name": "x"}}})),
            "tools/list" => ok_json(json!({"jsonrpc": "2.0", "id": "2", "result": {"tools": tools.clone()}})),
            _ => HttpResponse::new(202, ""),
        })
    })
}

fn discover(tools: Value) -> Vec<MCPToolDefinition> {
    let mut client = MCPClient::with_transport(server_listing(tools));
    let result = client.connect_and_discover_tools(SERVER_URL, None, "Bearer");
    assert!(result.success, "{:?}", result.error);
    result.tools
}

#[test]
fn test_top_level_title_and_hints_are_kept() {
    let tools = discover(json!([{
        "name": "list_issues",
        "title": "List issues",
        "inputSchema": {"type": "object"},
        "annotations": {
            "readOnlyHint": true,
            "destructiveHint": false,
            "idempotentHint": true,
            "openWorldHint": false,
        },
    }]));

    assert_eq!(tools[0].title.as_deref(), Some("List issues"));
    assert_eq!(
        tools[0].annotations,
        Some(MCPToolAnnotations {
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        })
    );
}

/// Trello sends the title only inside `annotations`.
#[test]
fn test_a_title_inside_annotations_is_used() {
    let tools = discover(json!([{
        "name": "trelloReadCard",
        "annotations": {"title": "Read card", "readOnlyHint": true},
    }]));

    assert_eq!(tools[0].title.as_deref(), Some("Read card"));
    assert_eq!(tools[0].annotations.unwrap().read_only_hint, Some(true));
}

#[test]
fn test_the_top_level_title_wins() {
    let tools = discover(json!([{
        "name": "getJob",
        "title": "Get job",
        "annotations": {"title": "Job lookup"},
    }]));

    assert_eq!(tools[0].title.as_deref(), Some("Get job"));
}

/// A hint the server leaves out is not the same as one it says is false.
#[test]
fn test_missing_hints_stay_unset() {
    let tools = discover(json!([{"name": "whoami", "annotations": {"readOnlyHint": false}}]));

    let hints = tools[0].annotations.unwrap();
    assert_eq!(hints.read_only_hint, Some(false));
    assert_eq!(hints.destructive_hint, None);
}

#[test]
fn test_no_hints_means_no_annotations() {
    let tools = discover(json!([
        {"name": "plain"},
        {"name": "title_only", "annotations": {"title": "Only a title"}},
        {"name": "malformed", "annotations": "read-only", "title": 7},
        {"name": "wrong_types", "annotations": {"readOnlyHint": "yes"}},
    ]));

    for tool in &tools {
        assert_eq!(tool.annotations, None, "{}", tool.name);
    }
    assert_eq!(tools[0].title, None);
    assert_eq!(tools[1].title.as_deref(), Some("Only a title"));
    assert_eq!(tools[2].title, None);
}

/// A project.json written before titles and annotations were kept.
#[test]
fn test_an_old_tool_entry_still_loads_and_round_trips() {
    let old = json!({"name": "search", "description": "Search", "input_schema": {"type": "object"}});

    let tool: MCPToolDefinition = serde_json::from_value(old.clone()).unwrap();

    assert_eq!(tool.title, None);
    assert_eq!(tool.annotations, None);
    assert_eq!(serde_json::to_value(&tool).unwrap(), old);
}

#[test]
fn test_stored_annotations_round_trip_without_unset_hints() {
    let tool = MCPToolDefinition {
        name: "save_issue".into(),
        title: Some("Save issue".into()),
        annotations: Some(MCPToolAnnotations { read_only_hint: Some(false), ..Default::default() }),
        ..Default::default()
    };

    let stored = serde_json::to_value(&tool).unwrap();

    assert_eq!(stored["title"], "Save issue");
    assert_eq!(stored["annotations"], json!({"read_only_hint": false}));
    assert_eq!(serde_json::from_value::<MCPToolDefinition>(stored).unwrap(), tool);
}
