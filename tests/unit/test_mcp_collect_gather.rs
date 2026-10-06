//! Port of `tests/unit/test_mcp_collect_gather.py`.
//!
//! Every connected server is asked, and a listing is followed to its bodies.
//!
//! Collect puts a question to documents, workspace files, feedback and the
//! web at once; a connected server is a source like any of them.  It used to
//! resolve one tool across the whole workspace, so a question spanning two
//! servers came back with half its context — and the one call it did make
//! returned a list of titles, which is how "outline the pitch doc" got
//! answered from a search hit's headline instead of the page.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use serial_test::serial;

use smartloop::agents::mcp::explorer::{
    ConversationTurn, DocumentGroups, Explorer, ExplorerSearchService, FollowUpPlanner, SkillDict,
};
use smartloop::agents::mcp::mcp_agent::{clear_resolution_cache, ResolvedCall};
use smartloop::mcp::client::MCPToolCallResult;
use smartloop::parsers::DocumentChunk;
use smartloop::workspace::MCPServer;

use crate::shared::mcp::{explorer, stub_transport, tool, tool_caller, FakeResolver, SeenCall};

fn server(name: &str, tool_name: &str) -> MCPServer {
    MCPServer {
        id: name.to_string(),
        name: name.to_string(),
        enabled: true,
        system: false,
        catalog_slug: None,
        pkg: None,
        remote_config: None,
        config: None,
        tools: vec![tool(tool_name, Some(json!({})))],
        ..Default::default()
    }
}

fn args(value: Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap()
}

/// The call-plan follow-ups, replaced (Python: `monkeypatch.setattr(
/// "smartloop.mcp.call_plans.follow_up_calls", ...)`).
fn follow_ups(calls: Vec<(&'static str, Value)>) -> FollowUpPlanner {
    Arc::new(move |_, _, _| calls.iter().map(|(t, a)| (t.to_string(), args(a.clone()))).collect())
}

struct Fixture {
    ex: Explorer,
    calls: Arc<Mutex<Vec<SeenCall>>>,
    resolver: Arc<FakeResolver>,
}

/// An Explorer whose tool calls return `results` keyed by (server, tool).
///
/// A real caller with only the transport stubbed: the refresh-and-retry the
/// collect path shares with every other MCP path lives on the caller.
fn explorer_with(
    results: Vec<((&str, &str), &str)>,
    resolved: impl Fn(&MCPServer) -> Option<ResolvedCall> + Send + Sync + 'static,
    follow_up_calls: FollowUpPlanner,
    broken: Option<&'static str>,
) -> Fixture {
    clear_resolution_cache();
    let results: HashMap<(String, String), String> =
        results.into_iter().map(|((s, t), text)| ((s.to_string(), t.to_string()), text.to_string())).collect();
    let caller = tool_caller();
    let calls = stub_transport(&caller, move |server, tool_name, _| {
        if Some(server.name.as_str()) == broken {
            // `get_access_token` raising for this server in Python: the
            // server's own work fails, the others must not notice.
            panic!("token gone");
        }
        let text = results.get(&(server.name.clone(), tool_name.to_string())).cloned();
        MCPToolCallResult {
            success: text.is_some(),
            content: Some(match &text {
                Some(t) if !t.is_empty() => vec![json!({"type": "text", "text": t})],
                _ => vec![],
            }),
            is_error: false,
            error: None,
            auth_expired: false,
        }
    });
    let resolver = Arc::new(FakeResolver { answer: Box::new(resolved), calls: Default::default() });
    let ex = explorer(Arc::new(caller), resolver.clone(), follow_up_calls);
    Fixture { ex, calls, resolver }
}

fn first_tool(s: &MCPServer, arguments: Value) -> Option<ResolvedCall> {
    Some(ResolvedCall {
        server_id: s.id.clone(),
        tool_name: s.tools[0].name.clone(),
        arguments: args(arguments),
        reasoned: false,
    })
}

fn run(fx: &Fixture, query: &str, servers: &[MCPServer]) -> String {
    fx.ex.run_mcp(query, Some(servers), Some("/tmp"), None, None)
}

// ------------------------------------------------------------ every server is asked

#[test]
#[serial(mcp_global)]
fn test_two_servers_both_contribute() {
    let servers = [server("Notion", "notion-search"), server("Linear", "list_issues")];
    let fx = explorer_with(
        vec![(("Notion", "notion-search"), "the pitch page"), (("Linear", "list_issues"), "ENG-1 open")],
        |s| first_tool(s, json!({"query": "x"})),
        follow_ups(vec![]),
        None,
    );

    let out = run(&fx, "what is on my plate", &servers);

    assert!(out.contains("MCP TOOL (Notion.notion-search)"), "{out}");
    assert!(out.contains("the pitch page"), "{out}");
    assert!(out.contains("MCP TOOL (Linear.list_issues)"), "{out}");
    assert!(out.contains("ENG-1 open"), "{out}");
}

#[test]
#[serial(mcp_global)]
fn test_one_server_failing_does_not_lose_the_others() {
    let servers = [server("Broken", "search"), server("Linear", "list_issues")];
    let fx = explorer_with(
        vec![(("Linear", "list_issues"), "ENG-1 open")],
        |s| first_tool(s, json!({})),
        follow_ups(vec![]),
        Some("Broken"),
    );

    let out = run(&fx, "what is on my plate", &servers);

    assert!(out.contains("ENG-1 open"), "{out}");
    assert!(!out.contains("Broken"), "{out}");
}

#[test]
#[serial(mcp_global)]
fn test_a_write_tool_is_left_to_a_planned_step() {
    let servers = [server("Notion", "notion-create-pages")];
    let fx = explorer_with(
        vec![(("Notion", "notion-create-pages"), "created")],
        |s| first_tool(s, json!({})),
        follow_ups(vec![]),
        None,
    );

    assert_eq!(run(&fx, "write it down", &servers), "");
    assert!(fx.calls.lock().is_empty(), "a write tool was called during collect");
}

#[test]
#[serial(mcp_global)]
fn test_a_server_that_resolves_to_nothing_is_skipped() {
    let servers = [server("Notion", "notion-search"), server("Linear", "list_issues")];
    let fx = explorer_with(
        vec![(("Linear", "list_issues"), "ENG-1 open")],
        |s| if s.name == "Notion" { None } else { first_tool(s, json!({})) },
        follow_ups(vec![]),
        None,
    );

    let out = run(&fx, "my issues", &servers);
    assert!(out.contains("ENG-1 open") && !out.contains("Notion"), "{out}");
}

#[test]
#[serial(mcp_global)]
fn test_the_number_of_servers_asked_is_bounded() {
    let servers: Vec<MCPServer> = (0..8).map(|i| server(&format!("srv{i}"), "search")).collect();
    let names: Vec<String> = (0..8).map(|i| format!("srv{i}")).collect();
    let hits: Vec<String> = (0..8).map(|i| format!("hit {i}")).collect();
    let fx = explorer_with(
        names.iter().zip(&hits).map(|(n, h)| ((n.as_str(), "search"), h.as_str())).collect(),
        |s| Some(ResolvedCall::new(s.id.clone(), "search")),
        follow_ups(vec![]),
        None,
    );

    let out = run(&fx, "anything", &servers);

    assert_eq!(out.matches("MCP TOOL").count(), Explorer::MAX_SERVERS_PER_QUESTION);
}

#[test]
#[serial(mcp_global)]
fn test_disabled_and_system_servers_are_not_asked() {
    let mut disabled = server("Off", "search");
    disabled.enabled = false;
    let mut system = server("smartloop", "bash");
    system.system = true;
    let fx = explorer_with(
        vec![(("Off", "search"), "x"), (("smartloop", "bash"), "x")],
        |s| first_tool(s, json!({})),
        follow_ups(vec![]),
        None,
    );

    assert_eq!(run(&fx, "anything", &[disabled, system]), "");
}

// ------------------------------------------------- the listing is followed to its bodies

#[test]
#[serial(mcp_global)]
fn test_a_search_hit_becomes_the_page_behind_it() {
    let servers = [server("Notion", "notion-search")];
    let fx = explorer_with(
        vec![
            (("Notion", "notion-search"), r#"{"results": [{"id": "abc"}]}"#),
            (("Notion", "notion-fetch"), r"# Pitch\nThe actual page text."),
        ],
        |s| Some(ResolvedCall { arguments: args(json!({"query": "pitch"})), ..ResolvedCall::new(s.id.clone(), "notion-search") }),
        follow_ups(vec![("notion-fetch", json!({"id": "abc"}))]),
        None,
    );

    let out = run(&fx, "outline the pitch", &servers);

    assert!(out.contains("The actual page text."), "{out}");
}

#[test]
#[serial(mcp_global)]
fn test_no_plan_means_the_listing_is_the_answer() {
    let servers = [server("Weather", "get_forecast")];
    let fx = explorer_with(
        vec![(("Weather", "get_forecast"), "18C and clear")],
        |s| Some(ResolvedCall::new(s.id.clone(), "get_forecast")),
        follow_ups(vec![]),
        None,
    );

    let out = run(&fx, "weather?", &servers);

    assert!(out.contains("18C and clear"), "{out}");
    assert_eq!(fx.calls.lock().len(), 1);
}

#[test]
#[serial(mcp_global)]
fn test_a_failed_follow_up_keeps_the_listing() {
    let servers = [server("Notion", "notion-search")];
    let fx = explorer_with(
        vec![(("Notion", "notion-search"), r#"{"results": [{"id": "abc"}]}"#)],
        |s| Some(ResolvedCall { arguments: args(json!({"query": "pitch"})), ..ResolvedCall::new(s.id.clone(), "notion-search") }),
        follow_ups(vec![("notion-fetch", json!({"id": "abc"}))]),
        None,
    );

    let out = run(&fx, "outline the pitch", &servers);

    assert!(out.contains(r#""id": "abc""#), "{out}");
}

// ------------------------------------------------ a matched skill owns its tool calls

/// The searches collect fans out over, each answering nothing and counting
/// how often it was asked (Python mocks `_search_vector` and friends).
#[derive(Default)]
struct QuietSearch {
    vector: Mutex<usize>,
}

impl ExplorerSearchService for QuietSearch {
    fn get_document_groups(
        &self,
        _query: Option<&str>,
        _skill_instruction: Option<&str>,
        _conversation_history: Option<&[ConversationTurn]>,
    ) -> anyhow::Result<Option<DocumentGroups>> {
        *self.vector.lock() += 1;
        Ok(None)
    }
    fn search_feedback(&self, _query: &str, _n_results: usize) -> anyhow::Result<Vec<DocumentChunk>> {
        Ok(Vec::new())
    }
    fn search_workspace(
        &self,
        _query: &str,
        _work_dir: &str,
        _n_results: usize,
        _conversation_history: Option<&[ConversationTurn]>,
    ) -> anyhow::Result<Vec<DocumentChunk>> {
        Ok(Vec::new())
    }
}

/// With a skill matched, the servers are not asked up front.
///
/// The plan built from the skill calls its tools as steps.  Asked here
/// instead, a server got the whole request as one search — for a bare
/// `/job-match` that was the skill's own name,
/// `searchCleraRoles(query='job-match')`.
fn collect(matched_skills: Option<Vec<SkillDict>>) -> (Fixture, Arc<QuietSearch>) {
    let fx = explorer_with(vec![], |s| first_tool(s, json!({})), follow_ups(vec![]), None);
    let search = Arc::new(QuietSearch::default());
    let servers = [server("Clera", "search")];
    fx.ex.collect_parallel(
        "Return a list of job matches",
        Some(search.clone()),
        None,
        None,
        matched_skills.as_deref(),
        Some(&servers),
        None,
        None,
        None,
        None,
        None,
        None,
    );
    (fx, search)
}

#[test]
#[serial(mcp_global)]
fn test_the_servers_are_not_asked_when_a_skill_matched() {
    let skill: SkillDict = HashMap::from([
        ("name".to_string(), "job-match".to_string()),
        ("content".to_string(), "...".to_string()),
    ]);
    let (fx, search) = collect(Some(vec![skill]));
    assert_eq!(*fx.resolver.calls.lock(), 0, "the servers were asked up front");
    assert!(fx.calls.lock().is_empty());
    assert_eq!(*search.vector.lock(), 1);
}

#[test]
#[serial(mcp_global)]
fn test_they_are_asked_when_no_skill_matched() {
    let (fx, _search) = collect(None);
    assert_eq!(*fx.resolver.calls.lock(), 1);
}
