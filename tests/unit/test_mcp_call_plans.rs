//! Port of `tests/unit/test_mcp_call_plans.py`.
//!
//! The second call: fetching what a listing only named.
//!
//! `notion-search` answers with page titles and ids; the pitch document's text
//! is behind `notion-fetch`.  Nothing in either schema says the two go
//! together, so the pairing is declared per server and applied to the call a
//! question just made — the surviving half of the crawl plans, now that
//! nothing is crawled.
//!
//! What matters here: a plan is found for the server that has one and nobody
//! else's, the detail tool and its id argument are resolved against what the
//! server actually declares, ids are read out of whatever envelope the payload
//! used, and every "no follow-up" case is silent rather than an error.

use serde_json::{json, Map, Value};

use smartloop::mcp::call_plans::{
    coerce_argument, filter_arguments, follow_up_calls, plan_for, plans, record_ids, resolve_argument,
    resolve_tool, CallPlan, DEFAULT_MAX_RECORDS,
};
use smartloop::workspace::{MCPServer, MCPToolDefinition};

fn obj(value: Value) -> Map<String, Value> {
    value.as_object().cloned().expect("an object")
}

fn search_tool() -> MCPToolDefinition {
    MCPToolDefinition {
        name: "notion-search".into(),
        description: None,
        input_schema: Some(obj(json!({"type": "object", "properties": {"query": {"type": "string"}}}))),
        ..Default::default()
    }
}

fn fetch_tool() -> MCPToolDefinition {
    MCPToolDefinition {
        name: "notion-fetch".into(),
        description: None,
        input_schema: Some(obj(json!({
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "include_discussions": {"type": "boolean"},
            },
            "required": ["id"],
        }))),
        ..Default::default()
    }
}

fn server_with(name: &str, catalog_slug: Option<&str>, tools: Vec<MCPToolDefinition>) -> MCPServer {
    MCPServer {
        name: name.into(),
        catalog_slug: catalog_slug.map(str::to_string),
        tools,
        ..Default::default()
    }
}

fn server() -> MCPServer {
    server_with("Notion", Some("notion"), vec![search_tool(), fetch_tool()])
}

const HITS: &str =
    r#"{"results": [{"id": "abc-123", "title": "Pitch document"}, {"id": "def-456", "title": "Roadmap"}]}"#;

fn call(tool: &str, args: Value) -> (String, Map<String, Value>) {
    (tool.to_string(), obj(args))
}

fn ids(payload: &str) -> Vec<String> {
    record_ids(payload, DEFAULT_MAX_RECORDS)
}

// ------------------------------------------------------------ TestTheBundledPlans

#[test]
fn test_the_shipped_plans_load() {
    let loaded = plans(None);
    for slug in ["notion", "linear", "trello", "canva", "monday"] {
        assert!(loaded.contains_key(slug), "missing bundled plan '{slug}'");
    }
}

#[test]
fn test_the_catalog_slug_finds_the_plan() {
    assert_eq!(plan_for(&server()).expect("a plan").slug, "notion");
}

#[test]
fn test_a_hand_added_connection_matches_on_its_name() {
    let found = plan_for(&server_with("my notion workspace", None, vec![search_tool(), fetch_tool()]));
    assert!(found.is_some_and(|p| p.slug == "notion"));
}

#[test]
fn test_a_server_with_no_plan_has_none() {
    assert!(plan_for(&server_with("weather", None, vec![])).is_none());
}

// ------------------------------------------------------------ TestFollowUpCalls

#[test]
fn test_each_listed_record_gets_its_body_fetched() {
    let calls = follow_up_calls(&server(), "notion-search", HITS);
    assert_eq!(
        calls,
        vec![
            call("notion-fetch", json!({"include_discussions": true, "id": "abc-123"})),
            call("notion-fetch", json!({"include_discussions": true, "id": "def-456"})),
        ]
    );
}

#[test]
fn test_the_detail_call_does_not_follow_up_on_itself() {
    assert!(follow_up_calls(&server(), "notion-fetch", HITS).is_empty());
}

#[test]
fn test_a_server_without_the_detail_tool_gets_none() {
    let server = server_with("Notion", Some("notion"), vec![search_tool()]);
    assert!(follow_up_calls(&server, "notion-search", HITS).is_empty());
}

#[test]
fn test_a_payload_with_no_records_is_not_an_error() {
    assert!(follow_up_calls(&server(), "notion-search", "nothing to see").is_empty());
    assert!(follow_up_calls(&server(), "notion-search", "").is_empty());
}

#[test]
fn test_a_connection_with_no_plan_is_not_an_error() {
    let server = server_with("weather", None, vec![search_tool(), fetch_tool()]);
    assert!(follow_up_calls(&server, "notion-search", HITS).is_empty());
}

#[test]
fn test_the_number_of_follow_ups_is_bounded() {
    // Twenty round-trips to answer a question about one page is a worse
    // answer, later.
    let many = format!(
        r#"{{"results": [{}]}}"#,
        (0..20).map(|i| format!(r#"{{"id": "id-{i}"}}"#)).collect::<Vec<_>>().join(", ")
    );
    assert_eq!(follow_up_calls(&server(), "notion-search", &many).len(), 3);
}

#[test]
fn test_a_fixed_argument_the_tool_rejects_is_dropped() {
    let plain = MCPToolDefinition {
        name: "notion-fetch".into(),
        description: None,
        input_schema: Some(obj(json!({
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        }))),
        ..Default::default()
    };
    let calls = follow_up_calls(&server_with("Notion", Some("notion"), vec![search_tool(), plain]), "notion-search", HITS);
    assert_eq!(calls[0], call("notion-fetch", json!({"id": "abc-123"})));
}

// ------------------------------------------------------------ TestReadingRecordsOutOfAPayload

#[test]
fn test_the_envelope_does_not_matter() {
    let cases: &[(&str, &[&str])] = &[
        (r#"{"results": [{"id": "a"}]}"#, &["a"]),
        (r#"[{"id": "a"}, {"id": "b"}]"#, &["a", "b"]),
        (r#"{"data": {"items": [{"id": "a"}]}}"#, &["a"]),
        (r#"{"edges": [{"node": {"id": "a"}}]}"#, &["a"]),
        (r#"{"id": "a", "title": "one record on its own"}"#, &["a"]),
        ("Here are your pages:\n{\"results\": [{\"id\": \"a\"}]}", &["a"]),
    ];
    for (payload, expected) in cases {
        assert_eq!(ids(payload), expected.to_vec(), "payload: {payload}");
    }
}

#[test]
fn test_a_url_is_an_id_of_last_resort() {
    // Hosted Notion listings carry no id; the URL is the only handle.
    assert_eq!(ids(r#"{"results": [{"url": "https://notion.so/p/1"}]}"#), vec!["https://notion.so/p/1"]);
}

#[test]
fn test_the_id_key_a_server_actually_uses_is_found() {
    assert_eq!(ids(r#"{"items": [{"design_id": "DAF1"}]}"#), vec!["DAF1"]);
}

#[test]
fn test_duplicates_are_not_fetched_twice() {
    assert_eq!(ids(r#"[{"id": "a"}, {"id": "a"}, {"id": "b"}]"#), vec!["a", "b"]);
}

#[test]
fn test_unparseable_output_yields_nothing() {
    assert!(ids("the server said no").is_empty());
    assert!(ids("{not json").is_empty());
}

// ------------------------------------------------------------ TestResolvingAgainstWhatTheServerDeclares

#[test]
fn test_a_plan_name_pairs_with_the_server_spelling() {
    // A plan says `fetch`; the server calls it `notion-fetch`.
    assert_eq!(resolve_tool(&["notion-search", "notion-fetch"], &["fetch"]).as_deref(), Some("notion-fetch"));
}

#[test]
fn test_candidate_order_decides() {
    let names = ["get_page", "notion-fetch"];
    assert_eq!(resolve_tool(&names, &["notion-fetch", "get_page"]).as_deref(), Some("notion-fetch"));
}

#[test]
fn test_a_tool_the_server_lacks_costs_nothing() {
    assert_eq!(resolve_tool(&["notion-search"], &["notion-fetch"]), None);
}

#[test]
fn test_the_id_argument_comes_from_the_schema() {
    let schema = obj(json!({"properties": {"pageId": {"type": "string"}}}));
    assert_eq!(resolve_argument(&schema, &["page_id", "pageId"]).as_deref(), Some("pageId"));
}

#[test]
fn test_a_lone_required_argument_takes_the_id_whatever_it_is_called() {
    let schema = obj(json!({"properties": {"ref": {"type": "string"}}, "required": ["ref"]}));
    assert_eq!(resolve_argument(&schema, &["id", "page_id"]).as_deref(), Some("ref"));
}

#[test]
fn test_arguments_the_tool_does_not_declare_are_dropped() {
    let schema = obj(json!({"properties": {"id": {"type": "string"}}}));
    assert_eq!(filter_arguments(&schema, &obj(json!({"id": "a", "nonsense": 1}))), obj(json!({"id": "a"})));
}

#[test]
fn test_an_empty_schema_is_no_evidence_so_nothing_is_dropped() {
    assert_eq!(filter_arguments(&Map::new(), &obj(json!({"id": "a"}))), obj(json!({"id": "a"})));
}

#[test]
fn test_a_numeric_id_argument_is_cast() {
    // monday returns board ids as strings and declares them as numbers.
    let schema = obj(json!({"properties": {"boardId": {"type": "number"}}}));
    assert_eq!(coerce_argument(&schema, "boardId", json!("12345")), json!(12345));
    assert_eq!(coerce_argument(&schema, "boardId", json!("not-a-number")), json!("not-a-number"));
}

#[test]
fn test_a_string_id_is_left_alone() {
    let schema = obj(json!({"properties": {"id": {"type": "string"}}}));
    assert_eq!(coerce_argument(&schema, "id", json!("abc-123")), json!("abc-123"));
}

// ------------------------------------------------------------ TestPlanValidation

#[test]
fn test_a_misspelled_field_is_rejected() {
    let err = CallPlan::from_raw(json!({"slug": "x", "detail_tool": ["get"]})).expect_err("must be rejected");
    assert!(err.to_string().contains("detail_tool"), "{err}");
}
