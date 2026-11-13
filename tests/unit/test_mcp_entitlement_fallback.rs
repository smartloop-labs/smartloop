//! Port of `tests/unit/test_mcp_entitlement_fallback.py`.
//!
//! A tool the account cannot call must not answer for the document.
//!
//! A paywalled MCP tool does not fail.  Notion's `notion-ai-search` on a
//! non-Business plan returns HTTP 200, `success=True`, and a body reading
//! "This tool requires a Business plan or higher" with an upsell link.  Every
//! layer above treated that as retrieved content, so the model was handed an
//! advertisement in place of the founders agreement and wrote its answer from
//! it — "the provided data does not contain a Founders Agreement", followed by
//! an invented template.  Discovery cannot warn about this: it lists what a
//! server *has*, not what the plan is entitled to.
//!
//! The Python tests replaced `Explorer._call_mcp_tool` with a mock; here the
//! tool caller's transport hook plays that role, and the resolver is a
//! scripted [`ToolResolver`].

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use serial_test::serial;
use smartloop::agents::fetch_agent::FetchAgent;
use smartloop::agents::mcp::argument_extractor::MCPArgumentExtractor;
use smartloop::agents::mcp::explorer::{Explorer, ProgressCallback, ToolResolver, WebSearcher};
use smartloop::agents::mcp::mcp_agent::{
    self as mcp_agent, forget_unavailable_tools, is_entitlement_notice, mark_tool_unavailable,
    unavailable_tools, MCPAgent, ResolvedCall,
};
use smartloop::agents::mcp::mcp_tool_caller::MCPToolCaller;
use smartloop::agents::mcp::result_mapper::MCPResultMapper;
use smartloop::llms::BaseLLM;
use smartloop::mcp::MCPToolCallResult;
use smartloop::workspace::workspace::{
    MCPServer, MCPServerType, MCPToolDefinition, RemoteMCPServerConfig,
};

use crate::shared::agents_mcp_args::ScriptedLLM;

/// The real payload, from the agent log.
const NOTION_UPSELL: &str = "This tool requires a Business plan or higher. Learn how to access the \
     full Notion MCP, including a free trial: \
     https://app.notion.com/notion-mcp?source=mcp_tool_upsell&tool=ai_search\
     &product=business&mcpRequestId=5c38eb6a-b115-4a7e-bd3b-2de99746f35c";

/// The autouse `_clean_registry` fixture.
struct CleanRegistry;

impl CleanRegistry {
    fn new() -> Self {
        forget_unavailable_tools();
        Self
    }
}

impl Drop for CleanRegistry {
    fn drop(&mut self) {
        forget_unavailable_tools();
    }
}

// ------------------------------------------------------------------ doubles

/// `exp.mcp_agent.resolve` as a `MagicMock`: a `side_effect` queue, then a
/// `return_value`.
#[derive(Default)]
struct ScriptedResolver {
    queue: Mutex<VecDeque<ResolvedCall>>,
    return_value: Mutex<Option<ResolvedCall>>,
    calls: AtomicUsize,
}

impl ScriptedResolver {
    fn side_effect(picks: Vec<ResolvedCall>) -> Arc<Self> {
        Arc::new(Self { queue: Mutex::new(picks.into()), ..Default::default() })
    }

    fn return_value(pick: ResolvedCall) -> Arc<Self> {
        Arc::new(Self { return_value: Mutex::new(Some(pick)), ..Default::default() })
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl ToolResolver for ScriptedResolver {
    fn resolve(
        &self,
        _directive: &str,
        _mcp_servers: &[MCPServer],
        _context: Option<&str>,
        _llm: Option<&dyn BaseLLM>,
    ) -> Option<ResolvedCall> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(next) = self.queue.lock().pop_front() {
            return Some(next);
        }
        self.return_value.lock().clone()
    }
}

/// Web search is not part of these tests.
struct NoWebSearch;

impl WebSearcher for NoWebSearch {
    fn is_enabled(&self) -> bool {
        false
    }
    fn search(&self, _query: &str, _progress: Option<ProgressCallback>) -> anyhow::Result<String> {
        Ok(String::new())
    }
    fn last_sources(&self) -> Vec<String> {
        Vec::new()
    }
    fn last_source_extracts(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

/// What the tool returns, call after call (`_call_mcp_tool`'s side effect).
enum ToolReplies {
    Sequence(Vec<&'static str>),
    Always(&'static str),
}

fn explorer(resolver: Arc<ScriptedResolver>, replies: ToolReplies) -> Explorer {
    let llm: Arc<dyn BaseLLM> = Arc::new(ScriptedLLM::new(""));
    let extractor = Arc::new(MCPArgumentExtractor::new(llm));
    let tool_caller = Arc::new(MCPToolCaller::new(
        extractor.clone(),
        Arc::new(Mutex::new(MCPResultMapper::new())),
        None,
    ));

    let replies = Mutex::new(match replies {
        ToolReplies::Sequence(v) => (VecDeque::from(v), None),
        ToolReplies::Always(text) => (VecDeque::new(), Some(text)),
    });
    tool_caller.set_call_tool_hook(Some(Arc::new(move |_server: &MCPServer, _tool: &str, _args: Option<&Map<String, Value>>, _token: Option<&str>, _cwd: Option<&str>| {
        let mut guard = replies.lock();
        let text = guard.0.pop_front().or(guard.1).expect("tool called more often than scripted");
        MCPToolCallResult {
            success: true,
            content: Some(vec![json!({"type": "text", "text": text})]),
            ..Default::default()
        }
    })));

    Explorer {
        llm: None,
        tool_caller: Some(tool_caller),
        arg_extractor: Some(extractor),
        use_llm_history: false,
        fetch_agent: Arc::new(FetchAgent::new()),
        web_search_agent: Arc::new(NoWebSearch),
        mcp_agent: resolver,
        // `_fetch_record_details` returns "": no follow-up plan.
        follow_up_calls: Arc::new(|_server: &MCPServer, _tool: &str, _payload: &str| Vec::new()),
    }
}

fn tool(name: &str, description: &str) -> MCPToolDefinition {
    MCPToolDefinition {
        name: name.to_string(),
        description: Some(description.to_string()),
        input_schema: Some(Map::new()),
        ..Default::default()
    }
}

fn server_named(name: &str, server_id: &str) -> MCPServer {
    MCPServer {
        id: server_id.to_string(),
        name: name.to_string(),
        server_type: MCPServerType::Remote,
        remote_config: Some(RemoteMCPServerConfig {
            server_url: "https://mcp.notion.com/mcp".to_string(),
            ..Default::default()
        }),
        tools: vec![
            tool("notion-ai-search", "AI search"),
            tool("notion-search", "Search pages"),
        ],
        ..Default::default()
    }
}

fn server() -> MCPServer {
    server_named("Notion", "notion-1")
}

fn pick(tool_name: &str, query: &str) -> ResolvedCall {
    let mut call = ResolvedCall::new("notion-1", tool_name);
    call.arguments.insert("query".to_string(), Value::from(query));
    call
}

// ------------------------------------------------------------------ tests

mod test_recognising_the_notice {
    use super::*;

    #[test]
    fn test_the_real_notion_upsell_is_recognised() {
        assert!(is_entitlement_notice(NOTION_UPSELL));
    }

    #[test]
    fn test_other_phrasings_are_recognised() {
        for text in [
            "This tool requires a Business plan or higher.",
            "That requires a subscription.",
            "AI search is not available on your current plan.",
            "Please upgrade your plan to continue.",
        ] {
            assert!(is_entitlement_notice(text), "not recognised: {text}");
        }
    }

    #[test]
    fn test_a_long_document_mentioning_plans_is_not_a_notice() {
        // The failure mode to avoid: discarding a real page about pricing,
        // which is exactly the content someone would ask for.
        let page = "Pricing page. The Business plan requires a seat minimum of five. ".repeat(60);
        assert!(!is_entitlement_notice(&page));
    }

    #[test]
    fn test_ordinary_content_is_not_a_notice() {
        assert!(!is_entitlement_notice(
            "The founders agreement was signed on 1 March by both parties."
        ));
    }

    #[test]
    fn test_nothing_is_not_a_notice() {
        assert!(!is_entitlement_notice(""));
        // `None` has no Rust counterpart: the parameter is a `&str`.
    }
}

mod test_the_same_turn_falls_to_a_tool_that_works {
    use super::*;

    #[test]
    #[serial]
    fn test_a_paywalled_pick_is_retried_and_the_second_tool_answers() {
        let _clean = CleanRegistry::new();
        let resolver = ScriptedResolver::side_effect(vec![
            pick("notion-ai-search", "founders agreement"),
            pick("notion-search", "founders agreement"),
        ]);
        let exp = explorer(
            resolver.clone(),
            ToolReplies::Sequence(vec![NOTION_UPSELL, "the real page text"]),
        );

        let out = exp.run_mcp_server("outline the founders agreement", &server(), None, None);

        assert!(out.contains("the real page text"));
        assert!(!out.contains("Business plan"));
        assert_eq!(resolver.call_count(), 2);
    }

    #[test]
    #[serial]
    fn test_the_upsell_never_becomes_context() {
        let _clean = CleanRegistry::new();
        // Even with no working alternative, an advertisement must not reach the
        // model as though it were the document.
        let resolver = ScriptedResolver::return_value(pick("notion-ai-search", "x"));
        let exp = explorer(resolver, ToolReplies::Always(NOTION_UPSELL));

        let out = exp.run_mcp_server("outline the founders agreement", &server(), None, None);

        assert_eq!(out, "");
    }

    #[test]
    #[serial]
    fn test_the_retry_is_bounded() {
        let _clean = CleanRegistry::new();
        let resolver = ScriptedResolver::return_value(pick("notion-ai-search", "x"));
        let exp = explorer(resolver.clone(), ToolReplies::Always(NOTION_UPSELL));

        exp.run_mcp_server("q", &server(), None, None);

        assert_eq!(resolver.call_count(), 2);
    }

    #[test]
    #[serial]
    fn test_a_working_tool_is_called_once() {
        let _clean = CleanRegistry::new();
        let resolver = ScriptedResolver::return_value(pick("notion-search", "x"));
        let exp = explorer(resolver.clone(), ToolReplies::Always("real records"));

        let out = exp.run_mcp_server("q", &server(), None, None);

        assert!(out.contains("real records"));
        assert_eq!(resolver.call_count(), 1);
    }

    #[test]
    #[serial]
    fn test_the_user_is_told_the_tool_needs_a_higher_plan() {
        let _clean = CleanRegistry::new();
        let resolver = ScriptedResolver::side_effect(vec![
            pick("notion-ai-search", "x"),
            pick("notion-search", "x"),
        ]);
        let exp = explorer(resolver, ToolReplies::Sequence(vec![NOTION_UPSELL, "real text"]));

        let seen: Arc<Mutex<Vec<(String, String, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let progress: ProgressCallback = Arc::new(move |step: &str, status: &str, message: &str| {
            sink.lock().push((step.to_string(), status.to_string(), message.to_string()));
        });
        exp.run_mcp_server("q", &server(), None, Some(progress));

        let messages: Vec<String> = seen.lock().iter().map(|(_, _, m)| m.clone()).collect();
        assert!(messages.iter().any(|m| m.contains("higher plan")), "{messages:?}");
    }
}

mod test_the_tool_is_not_offered_again {
    use super::*;

    #[test]
    #[serial]
    fn test_a_paywalled_tool_is_recorded() {
        let _clean = CleanRegistry::new();
        let resolver = ScriptedResolver::side_effect(vec![
            pick("notion-ai-search", "x"),
            pick("notion-search", "x"),
        ]);
        let exp = explorer(resolver, ToolReplies::Sequence(vec![NOTION_UPSELL, "real text"]));

        exp.run_mcp_server("q", &server_named("Notion", "notion-1"), None, None);

        assert!(unavailable_tools().contains(&("notion-1".to_string(), "notion-ai-search".to_string())));
    }

    #[test]
    #[serial]
    fn test_the_catalogue_stops_offering_it() {
        let _clean = CleanRegistry::new();
        // Without this the resolver keeps picking the same paywalled tool every
        // turn — it is still in discovery and still the best-sounding name.
        let agent = MCPAgent::new(Arc::new(ScriptedLLM::new("")));
        let mut server = server();
        server.system = false;
        server.enabled = true;

        let before: HashSet<String> = agent
            .catalog(std::slice::from_ref(&server))
            .into_iter()
            .map(|e| e.tool_name)
            .collect();
        assert_eq!(
            before,
            ["notion-ai-search", "notion-search"].iter().map(|s| s.to_string()).collect()
        );

        mark_tool_unavailable("notion-1", "notion-ai-search");

        let after: HashSet<String> = agent
            .catalog(std::slice::from_ref(&server))
            .into_iter()
            .map(|e| e.tool_name)
            .collect();
        assert_eq!(after, ["notion-search".to_string()].into_iter().collect());
    }

    #[test]
    #[serial]
    fn test_it_is_scoped_to_the_server_that_declined() {
        let _clean = CleanRegistry::new();
        mark_tool_unavailable("notion-1", "notion-ai-search");
        assert!(!unavailable_tools().contains(&("notion-2".to_string(), "notion-ai-search".to_string())));
    }

    #[test]
    #[serial]
    fn test_the_registry_can_be_cleared_for_reauthentication() {
        let _clean = CleanRegistry::new();
        mark_tool_unavailable("notion-1", "notion-ai-search");
        assert!(!unavailable_tools().is_empty());
        mcp_agent::forget_unavailable_tools();
        assert!(unavailable_tools().is_empty());
    }
}
