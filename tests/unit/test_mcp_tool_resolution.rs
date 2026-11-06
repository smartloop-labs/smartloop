//! Port of `tests/unit/test_mcp_tool_resolution.py`.
//!
//! Resolving an MCP step against what discovery already found.
//!
//! Selection used to be a vector search over embedded tool descriptions, which
//! answered "which tool reads like this sentence" and then handed the pick to a
//! second call that had to work out the arguments on its own.  Resolution is
//! now one call over the tools discovery recorded per server: it names the tool
//! *and* fills its required arguments, and a reasoning model is given an output
//! budget big enough to think first.
//!
//! These pin the parts that are easy to regress: the catalog comes from the
//! servers in memory (no index, no project dir), the reply is parsed in the
//! shapes models actually emit, fabricated values are dropped, and a thinking
//! model asks for a budget above the floor below which the wrapper suppresses
//! the block.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use serial_test::serial;

use smartloop::agents::mcp::mcp_agent::{
    clear_resolution_cache, format_call, missing_required, MCPAgent, ResolvedCall, RESOLVE_MAX_TOKENS,
};
use smartloop::agents::planner::SYSTEM_TOOLS_SERVER_ID;
use smartloop::llms::llama_cpp::THINKING_MIN_BUDGET;
use smartloop::workspace::workspace::{MCPServer, MCPToolDefinition};

use crate::shared::agents_mcp_agent::StubLLM;

/// One question resolves once per request — including across these tests.
struct FreshCache;

impl FreshCache {
    fn new() -> Self {
        clear_resolution_cache();
        FreshCache
    }
}

impl Drop for FreshCache {
    fn drop(&mut self) {
        clear_resolution_cache();
    }
}

/// An LLM that answers `reply`.
///
/// `can_think` is what both real handles report: `LlamaCpp` from the flag
/// bound onto its model at load, `RemoteLLM` by asking the host.  Its *value*
/// — the user's setting for visible answers — is deliberately not represented
/// here, because resolution does not consult it.
fn llm_with(reply: &str, can_think: bool) -> Arc<StubLLM> {
    let llm = StubLLM::new(reply);
    llm.set_supports_thinking(can_think);
    llm.set_model_label("test-model");
    // get_max_tokens stays at 0 ("no window reported"): the budget helper falls
    // back to what it asked for, which is what a caller with no real tokenizer
    // should see.
    llm
}

fn llm(reply: &str) -> Arc<StubLLM> {
    llm_with(reply, true)
}

fn obj(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(m) => m,
        other => panic!("expected an object, got {other}"),
    }
}

fn tool_named(name: &str, description: &str, required: &[&str]) -> MCPToolDefinition {
    MCPToolDefinition {
        name: name.to_string(),
        description: Some(description.to_string()),
        input_schema: Some(obj(json!({
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The issue title"},
                "teamId": {"type": "string", "description": "Team the issue belongs to"},
                "after": {"type": "string", "description": "Pagination cursor"},
            },
            "required": required,
        }))),
    }
}

fn tool() -> MCPToolDefinition {
    tool_named("create_issue", "Create an issue", &["title"])
}

fn server_with(server_id: &str, name: &str, tools: Vec<MCPToolDefinition>, enabled: bool) -> MCPServer {
    MCPServer {
        id: server_id.to_string(),
        name: name.to_string(),
        enabled,
        system: false,
        tools,
        ..Default::default()
    }
}

fn server_tools(tools: Vec<MCPToolDefinition>) -> MCPServer {
    server_with("srv-1", "linear", tools, true)
}

fn server() -> MCPServer {
    server_tools(vec![tool()])
}

fn agent(llm: &Arc<StubLLM>) -> MCPAgent {
    MCPAgent::new(llm.as_llm())
}

// ------------------------------------------------------- TestTheCatalogIsDiscovery

#[test]
#[serial]
fn test_the_prompt_lists_every_discovered_tool_with_its_required_args() {
    let _cache = FreshCache::new();
    let llm = llm(r#"{"tool": 1, "arguments": {"title": "Login is broken"}}"#);
    let agent = agent(&llm);

    let resolved = agent.resolve(
        "file an issue: Login is broken",
        &[server_tools(vec![tool(), tool_named("list_issues", "Create an issue", &[])])],
        None,
        None,
    );

    let prompt = llm.last_call().prompt;
    assert!(prompt.contains("linear.create_issue"));
    assert!(prompt.contains("linear.list_issues"));
    assert!(prompt.contains("title (string): The issue title"));
    // A tool with no required arguments says so rather than listing the
    // optional ones — an optional argument the model can see is one it can
    // fabricate.
    assert!(prompt.contains("(no required arguments)"));
    assert!(!prompt.contains("after (string)"));
    assert_eq!(resolved.unwrap().tool_name, "create_issue");
}

/// No vector store, no embeddings, nothing on disk — just the servers.
#[test]
#[serial]
fn test_it_resolves_with_no_project_dir_and_no_index() {
    let _cache = FreshCache::new();
    let agent = agent(&llm(r#"{"tool": 1, "arguments": {"title": "x"}}"#));

    let resolved = agent.resolve("file an issue", &[server()], None, None);

    assert_eq!(resolved.unwrap().server_id, "srv-1");
}

#[test]
#[serial]
fn test_system_and_disabled_servers_are_left_out() {
    let _cache = FreshCache::new();
    let llm = llm(r#"{"tool": 0}"#);
    let agent = agent(&llm);

    agent.resolve(
        "anything",
        &[
            server_with(
                SYSTEM_TOOLS_SERVER_ID,
                "smartloop",
                vec![tool_named("bash", "Create an issue", &["title"])],
                true,
            ),
            server_with("srv-2", "off", vec![tool_named("get_page", "Create an issue", &["title"])], false),
            server(),
        ],
        None,
        None,
    );

    let prompt = llm.last_call().prompt;
    assert!(!prompt.contains("bash"));
    assert!(!prompt.contains("off.get_page"));
    assert!(prompt.contains("linear.create_issue"));
}

#[test]
#[serial]
fn test_nothing_discovered_means_no_call_and_no_inference() {
    let _cache = FreshCache::new();
    let llm = llm(r#"{"tool": 1}"#);
    let agent = agent(&llm);

    assert!(agent.resolve("anything", &[], None, None).is_none());
    assert!(agent.resolve("anything", &[server_tools(vec![])], None, None).is_none());
    assert_eq!(llm.call_count(), 0);
}

#[test]
#[serial]
fn test_a_lone_argument_free_tool_needs_no_model_at_all() {
    let _cache = FreshCache::new();
    let llm = llm(r#"{"tool": 1}"#);
    let agent = agent(&llm);

    let resolved = agent
        .resolve(
            "what is on my board",
            &[server_tools(vec![tool_named("list_issues", "Create an issue", &[])])],
            None,
            None,
        )
        .unwrap();

    assert_eq!(resolved.tool_name, "list_issues");
    assert!(resolved.arguments.is_empty());
    assert_eq!(llm.call_count(), 0);
}

// ------------------------------------------------ TestParsingWhatModelsActuallyEmit

fn resolve_reply(reply: &str) -> Option<ResolvedCall> {
    let agent = agent(&llm(reply));
    agent.resolve(
        "file an issue: Login is broken",
        &[server_tools(vec![tool_named("list_issues", "Create an issue", &[]), tool()])],
        None,
        None,
    )
}

#[test]
#[serial]
fn test_the_json_object_it_was_asked_for() {
    let _cache = FreshCache::new();
    let resolved = resolve_reply(r#"{"tool": 2, "arguments": {"title": "Login is broken"}}"#).unwrap();
    assert_eq!(resolved.tool_name, "create_issue");
    assert_eq!(resolved.arguments, obj(json!({"title": "Login is broken"})));
}

#[test]
#[serial]
fn test_a_fenced_object() {
    let _cache = FreshCache::new();
    let resolved = resolve_reply("```json\n{\"tool\": 2, \"arguments\": {\"title\": \"t\"}}\n```").unwrap();
    assert_eq!(resolved.tool_name, "create_issue");
}

#[test]
#[serial]
fn test_the_tool_named_instead_of_numbered() {
    let _cache = FreshCache::new();
    let resolved = resolve_reply(r#"{"tool": "linear.create_issue", "arguments": {"title": "t"}}"#).unwrap();
    assert_eq!(resolved.tool_name, "create_issue");
}

#[test]
#[serial]
fn test_a_bare_number_from_a_model_that_ignored_the_shape() {
    let _cache = FreshCache::new();
    let resolved = resolve_reply("2").unwrap();
    assert_eq!(resolved.tool_name, "create_issue");
    assert!(resolved.arguments.is_empty());
}

#[test]
#[serial]
fn test_zero_means_none_of_them_fit() {
    let _cache = FreshCache::new();
    assert!(resolve_reply(r#"{"tool": 0}"#).is_none());
}

#[test]
#[serial]
fn test_an_out_of_range_number_is_not_a_pick() {
    let _cache = FreshCache::new();
    assert!(resolve_reply(r#"{"tool": 9}"#).is_none());
}

#[test]
#[serial]
fn test_unparseable_output_is_not_a_pick() {
    let _cache = FreshCache::new();
    assert!(resolve_reply("I could not find a suitable tool.").is_none());
}

/// `BaseLLM::invoke` never raises: a failed generation comes back as its
/// error text, which must resolve to nothing rather than to a pick.
#[test]
#[serial]
fn test_a_failed_call_is_a_skip_not_a_raise() {
    let _cache = FreshCache::new();
    let llm = llm("Error generating response: oom");
    let agent = agent(&llm);

    assert!(agent.resolve("anything", &[server()], None, None).is_none());
}

// ---------------------------------------------------------- TestArgumentsAreCleaned

fn arguments_for(raw_arguments: &str) -> Map<String, Value> {
    let agent = agent(&llm(&format!(r#"{{"tool": 1, "arguments": {raw_arguments}}}"#)));
    let resolved = agent
        .resolve("file an issue: Login is broken", &[server()], None, None)
        .unwrap();
    resolved.arguments
}

#[test]
#[serial]
fn test_undeclared_and_unlisted_arguments_are_dropped() {
    let _cache = FreshCache::new();
    // `teamId` is declared but not required, `nonsense` is not declared at
    // all: neither was shown to the model, so neither is passed on.
    assert_eq!(
        arguments_for(r#"{"title": "Login is broken", "teamId": "T-1", "nonsense": 1}"#),
        obj(json!({"title": "Login is broken"}))
    );
}

#[test]
#[serial]
fn test_placeholders_and_empties_are_dropped() {
    let _cache = FreshCache::new();
    assert!(arguments_for(r#"{"title": "MISSING"}"#).is_empty());
    clear_resolution_cache();
    assert!(arguments_for(r#"{"title": ""}"#).is_empty());
    clear_resolution_cache();
    assert!(arguments_for(r#"{"title": null}"#).is_empty());
}

#[test]
#[serial]
fn test_an_echoed_field_description_is_not_a_value() {
    let _cache = FreshCache::new();
    assert!(arguments_for(r#"{"title": "The issue title"}"#).is_empty());
}

#[test]
#[serial]
fn test_a_cursor_is_never_accepted_from_the_model() {
    let _cache = FreshCache::new();
    let agent = agent(&llm(r#"{"tool": 1, "arguments": {"after": "cursor-123"}}"#));
    let resolved = agent
        .resolve(
            "next page",
            &[server_tools(vec![tool_named("create_issue", "Create an issue", &["after"])])],
            None,
            None,
        )
        .unwrap();
    assert!(resolved.arguments.is_empty());
}

// ------------------------------------------------------ TestReasoningGetsRoomToThink

/// `thinking=Some(true)` travels with the request, and the budget clears
/// THINKING_MIN_BUDGET — below that the wrapper suppresses the block, so a
/// resolve call would spend its budget on a thought it never finishes.
#[test]
#[serial]
fn test_the_call_asks_for_reasoning_and_room_to_do_it() {
    let _cache = FreshCache::new();
    let llm = llm(r#"{"tool": 1, "arguments": {"title": "Login is broken"}}"#);
    let agent = agent(&llm);

    let resolved = agent.resolve("file an issue", &[server()], None, None).unwrap();

    let call = llm.last_call();
    assert_eq!(call.thinking, Some(true));
    assert_eq!(call.max_tokens, Some(RESOLVE_MAX_TOKENS));
    assert!(RESOLVE_MAX_TOKENS > THINKING_MIN_BUDGET);
    assert!(resolved.reasoned);
}

/// The thinking toggle governs the visible answer; this call's output is a
/// JSON tool call whose monologue is stripped either way.  So the flag is
/// passed per call rather than read off the model.
#[test]
#[serial]
fn test_reasoning_is_asked_for_whatever_the_users_setting_says() {
    let _cache = FreshCache::new();
    let llm = llm(r#"{"tool": 1, "arguments": {"title": "Login is broken"}}"#);
    let agent = agent(&llm);

    agent.resolve("file an issue", &[server()], None, None);

    // Nothing about the model's own setting is consulted — no flag flipped:
    // the request carries it.
    assert_eq!(llm.last_call().thinking, Some(true));
}

#[test]
#[serial]
fn test_a_model_that_cannot_reason_still_resolves() {
    let _cache = FreshCache::new();
    let llm = llm_with(r#"{"tool": 1, "arguments": {"title": "Login is broken"}}"#, false);
    let agent = agent(&llm);

    let resolved = agent.resolve("file an issue", &[server()], None, None).unwrap();

    assert_eq!(resolved.tool_name, "create_issue");
    assert!(!resolved.reasoned);
}

/// A handle that ignores the per-call flag must not lose the pick.  (Python
/// retried without the keyword on a `TypeError`; in Rust every handle takes
/// the flag, so what remains to pin is that the answer and the budget do not
/// depend on the flag being honoured.)
#[test]
#[serial]
fn test_an_llm_without_the_per_call_flag_is_still_asked() {
    let _cache = FreshCache::new();
    let calls: Arc<Mutex<Vec<Option<usize>>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded = calls.clone();
    let llm = llm("");
    llm.set_reply_fn(Arc::new(move |_prompt, max_tokens, _thinking| {
        recorded.lock().push(max_tokens);
        r#"{"tool": 1, "arguments": {"title": "Login is broken"}}"#.to_string()
    }));
    let agent = agent(&llm);

    let resolved = agent.resolve("file an issue", &[server()], None, None).unwrap();

    assert_eq!(resolved.tool_name, "create_issue");
    assert_eq!(*calls.lock(), vec![Some(RESOLVE_MAX_TOKENS)]);
}

#[test]
#[serial]
fn test_the_budget_never_exceeds_what_is_left_of_the_context() {
    let _cache = FreshCache::new();
    let llm = llm(r#"{"tool": 1, "arguments": {"title": "t"}}"#);
    llm.set_max_tokens(2048);
    llm.set_num_tokens(1800);
    let agent = agent(&llm);

    agent.resolve("file an issue", &[server()], None, None);

    assert_eq!(llm.last_call().max_tokens, Some(2048 - 1800 - 64));
}

// ---------------------------------------------------- TestTheCallIsShownToThePerson
// A tool step's decision is which call to make; until this was surfaced the
// client only saw which model had been selected.

#[test]
fn test_it_renders_the_call_as_a_signature() {
    assert_eq!(
        format_call("linear", "create_issue", &obj(json!({"title": "Login is broken"}))),
        "linear.create_issue(title='Login is broken')"
    );
}

#[test]
fn test_no_arguments_still_reads_as_a_call() {
    assert_eq!(format_call("linear", "list_issues", &Map::new()), "linear.list_issues()");
}

#[test]
fn test_a_long_value_is_truncated_not_pasted() {
    let shown = format_call("notion", "create-pages", &obj(json!({"content": "x".repeat(500)})));
    assert!(shown.chars().count() < 120 && shown.ends_with("…')"));
}

#[test]
fn test_extra_arguments_are_counted_rather_than_listed() {
    let mut arguments = Map::new();
    for i in 0..7 {
        arguments.insert(i.to_string(), json!(i));
    }
    let shown = format_call("a", "b", &arguments);
    assert!(shown.ends_with("+3 more)"));
}

// ------------------------------------------------------------- TestMissingRequired

#[test]
fn test_it_names_what_the_tool_still_needs() {
    let schema = obj(json!({"required": ["title", "teamId"]}));
    assert_eq!(missing_required(&schema, &obj(json!({"title": "t"}))), vec!["teamId"]);
    assert_eq!(
        missing_required(&schema, &obj(json!({"title": "t", "teamId": ""}))),
        vec!["teamId"]
    );
    assert!(missing_required(&schema, &obj(json!({"title": "t", "teamId": "T-1"}))).is_empty());
}

#[test]
fn test_a_tool_with_no_required_arguments_is_never_missing_any() {
    assert!(missing_required(&Map::new(), &Map::new()).is_empty());
}

// ------------------------------------------------------- TestOneQuestionResolvesOnce
// The collect path and the planned step both resolve the same question over
// the same catalog; the second would spend seconds of reasoning on an answer
// already in hand.

#[test]
#[serial]
fn test_the_second_caller_reuses_the_first_resolution() {
    let _cache = FreshCache::new();
    let llm = llm(r#"{"tool": 1, "arguments": {"title": "Login is broken"}}"#);
    let servers = [server()];

    let first = MCPAgent::new(llm.as_llm()).resolve("file an issue", &servers, None, None);
    // A different MCPAgent instance — the two callers hold their own.
    let second = MCPAgent::new(llm.as_llm()).resolve("file an issue", &servers, None, None);

    assert_eq!(second, first);
    assert_eq!(llm.call_count(), 1);
}

#[test]
#[serial]
fn test_a_different_question_resolves_again() {
    let _cache = FreshCache::new();
    let llm = llm(r#"{"tool": 1, "arguments": {"title": "Login is broken"}}"#);
    let agent = agent(&llm);

    agent.resolve("file an issue", &[server()], None, None);
    agent.resolve("file a different issue", &[server()], None, None);

    assert_eq!(llm.call_count(), 2);
}

#[test]
#[serial]
fn test_a_changed_catalog_resolves_again() {
    let _cache = FreshCache::new();
    let llm = llm(r#"{"tool": 1, "arguments": {"title": "Login is broken"}}"#);
    let agent = agent(&llm);

    agent.resolve("file an issue", &[server()], None, None);
    agent.resolve(
        "file an issue",
        &[server_tools(vec![tool(), tool_named("x", "Create an issue", &["title"])])],
        None,
        None,
    );

    assert_eq!(llm.call_count(), 2);
}

/// A question no tool serves must not be re-reasoned by the next caller
/// either.
#[test]
#[serial]
fn test_no_tool_is_remembered_too() {
    let _cache = FreshCache::new();
    let llm = llm(r#"{"tool": 0}"#);
    let servers = [server()];

    assert!(MCPAgent::new(llm.as_llm()).resolve("what time is it", &servers, None, None).is_none());
    assert!(MCPAgent::new(llm.as_llm()).resolve("what time is it", &servers, None, None).is_none());
    assert_eq!(llm.call_count(), 1);
}

// --------------------------------------------------- TestADeclinedReplyIsSalvaged
// A 7.5B answered `{"tool": null, "arguments": {"context_note": "the tools are
// Notion APIs…"}}` to a question its Notion server plainly served.  Null is not
// a refusal the prompt offers — 0 is — so a name in the reply is taken rather
// than throwing the whole reasoning pass away.

fn notion_entries() -> Vec<MCPServer> {
    vec![server_tools(vec![
        tool_named("notion-search", "Create an issue", &["title"]),
        tool_named("notion-create-pages", "Create an issue", &["title"]),
    ])]
}

#[test]
#[serial]
fn test_a_null_pick_that_names_a_tool_uses_it() {
    let _cache = FreshCache::new();
    let agent = agent(&llm(
        r#"{"tool": null, "arguments": {"title": "the pitch"}, "note": "notion-search would find it"}"#,
    ));

    let resolved = agent
        .resolve("outline the pitch from notion", &notion_entries(), None, None)
        .unwrap();

    assert_eq!(resolved.tool_name, "notion-search");
    assert_eq!(resolved.arguments, obj(json!({"title": "the pitch"})));
}

#[test]
#[serial]
fn test_an_explicit_zero_is_still_a_refusal() {
    let _cache = FreshCache::new();
    // The model is allowed to say no; only an unusable answer is salvaged.
    let agent = agent(&llm(r#"{"tool": 0, "arguments": {}}"#));
    assert!(agent.resolve("what is 2 + 2", &notion_entries(), None, None).is_none());
}

#[test]
#[serial]
fn test_prose_naming_no_tool_resolves_nothing() {
    let _cache = FreshCache::new();
    let agent = agent(&llm("None of the available tools can do this."));
    assert!(agent.resolve("what is 2 + 2", &notion_entries(), None, None).is_none());
}

// ---------------------------------------------------- TestTheCatalogIsSizedToTheModel

fn many_tools() -> Vec<MCPToolDefinition> {
    (0..42)
        .map(|i| tool_named(&format!("notion-tool-{i}"), &"Does a thing. ".repeat(12), &["title"]))
        .collect()
}

/// Descriptions are what the pick is made on; a bare name list is the
/// fallback for a small window, not the default.
#[test]
#[serial]
fn test_a_roomy_context_keeps_the_descriptions() {
    let _cache = FreshCache::new();
    let llm = llm(r#"{"tool": 1, "arguments": {"title": "Login is broken"}}"#);
    llm.set_max_tokens(13056);
    MCPAgent::new(llm.as_llm()).resolve("file an issue", &[server_tools(many_tools())], None, None);

    let prompt = llm.last_call().prompt;
    assert!(prompt.contains("Does a thing."));
    assert!(prompt.contains("notion-tool-41"));
}

#[test]
#[serial]
fn test_a_small_context_falls_back_to_call_shapes() {
    let _cache = FreshCache::new();
    let llm = llm(r#"{"tool": 1, "arguments": {"title": "Login is broken"}}"#);
    llm.set_max_tokens(2048);
    MCPAgent::new(llm.as_llm()).resolve("file an issue", &[server_tools(many_tools())], None, None);

    let prompt = llm.last_call().prompt;
    assert!(!prompt.contains("Does a thing."));
    // Every tool is still listed — names are dropped last, descriptions first.
    assert!(prompt.contains("notion-tool-41"));
}

// ------------------------------------------------------- native tool calling

fn native_llm(reply: &str) -> Arc<StubLLM> {
    let llm = llm(reply);
    llm.set_supports_native_tools(true);
    llm
}

/// A model that calls functions natively gets the catalog as tool
/// definitions, schema and optional parameters included, and its own call
/// format is read back — optional arguments it filled are kept.
#[test]
#[serial]
fn test_a_native_model_resolves_through_its_own_tool_calls() {
    let _cache = FreshCache::new();
    let llm = native_llm(
        "<tool_call>\n<function=create_issue>\n<parameter=title>\nLogin is broken\n</parameter>\n<parameter=teamId>\nENG\n</parameter>\n</function>\n</tool_call>",
    );
    let agent = agent(&llm);

    let resolved = agent
        .resolve("file an issue for team ENG: Login is broken", &[server()], None, None)
        .expect("a native call resolves");

    assert!(resolved.native);
    assert_eq!(resolved.tool_name, "create_issue");
    assert_eq!(resolved.arguments, obj(json!({"title": "Login is broken", "teamId": "ENG"})));

    let call = llm.last_call();
    assert_eq!(llm.call_count(), 1);
    assert!(call.prompt.starts_with("<slp_tools>"));
    // The whole schema goes over, optional parameters included.
    assert!(call.prompt.contains("\"teamId\""));
    assert!(!call.prompt.contains("[1] linear.create_issue"));
    assert_eq!(call.thinking, Some(true));
}

/// Fabricated values are still dropped from a native call.
#[test]
#[serial]
fn test_a_native_call_still_drops_cursors_and_placeholders() {
    let _cache = FreshCache::new();
    let llm = native_llm(
        r#"<tool_call>{"name": "create_issue", "arguments": {"title": "Login is broken", "after": "cursor-123", "teamId": "MISSING"}}</tool_call>"#,
    );
    let resolved = agent(&llm).resolve("file an issue: Login is broken", &[server()], None, None).unwrap();
    assert_eq!(resolved.arguments, obj(json!({"title": "Login is broken"})));
}

/// A native reply that names no listed tool falls back to the prompt-based
/// resolution on the same model.
#[test]
#[serial]
fn test_an_unusable_native_reply_falls_back_to_the_prompt() {
    let _cache = FreshCache::new();
    let llm = native_llm("");
    llm.set_reply_fn(Arc::new(|prompt, _, _| {
        if prompt.starts_with("<slp_tools>") {
            "NONE".to_string()
        } else {
            r#"{"tool": 1, "arguments": {"title": "Login is broken"}}"#.to_string()
        }
    }));

    let resolved = agent(&llm).resolve("file an issue: Login is broken", &[server()], None, None).unwrap();

    assert_eq!(llm.call_count(), 2);
    assert!(!resolved.native);
    assert_eq!(resolved.arguments, obj(json!({"title": "Login is broken"})));
}

/// A model without native tool calling never sees a tools block.
#[test]
#[serial]
fn test_a_model_without_native_tools_keeps_the_prompt() {
    let _cache = FreshCache::new();
    let llm = llm(r#"{"tool": 1, "arguments": {"title": "Login is broken"}}"#);
    let resolved = agent(&llm).resolve("file an issue: Login is broken", &[server()], None, None).unwrap();
    assert!(!resolved.native);
    assert!(!llm.last_call().prompt.contains("<slp_tools>"));
}

/// The same tool name on two servers is qualified so each stays callable.
#[test]
#[serial]
fn test_native_names_are_qualified_only_on_a_clash() {
    let agent = agent(&llm(""));
    let servers = [
        server_with("srv-1", "linear", vec![tool(), tool_named("list_issues", "List", &[])], true),
        server_with("srv-2", "jira", vec![tool()], true),
    ];
    let (_, names) = agent.native_tools(&agent.catalog(&servers));
    assert_eq!(names, vec!["linear__create_issue", "list_issues", "jira__create_issue"]);
}

/// A native model plans "find the user, then file the issue" as two calls.
/// The step makes the first; the rest are kept so the write is not lost.
#[test]
#[serial]
fn test_a_native_multi_call_answer_keeps_the_later_calls() {
    let _cache = FreshCache::new();
    let llm = native_llm(concat!(
        "<tool_call>{\"name\": \"list_users\", \"arguments\": {}}</tool_call>\n",
        "<tool_call>{\"name\": \"create_issue\", \"arguments\": {\"title\": \"Slow pages\"}}</tool_call>",
    ));
    let servers = [server_tools(vec![tool_named("list_users", "Find users", &[]), tool()])];

    let resolved = agent(&llm).resolve("assign an issue to Chris: slow pages", &servers, None, None).unwrap();

    assert_eq!(resolved.tool_name, "list_users");
    assert_eq!(resolved.also.len(), 1);
    assert_eq!(resolved.also[0].tool_name, "create_issue");
    assert_eq!(resolved.also[0].arguments, obj(json!({"title": "Slow pages"})));
}
