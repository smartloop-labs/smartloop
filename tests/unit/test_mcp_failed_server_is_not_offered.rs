//! Port of `tests/unit/test_mcp_failed_server_is_not_offered.py`.
//!
//! A connection whose grant has died must not be offered by any consumer.
//!
//! Three places build a list of connections for the model: the planner's
//! catalog (`StepExecutor::build_available_servers`), the collect pass
//! (`Explorer::run_mcp`) and the MCP agent's tool catalog
//! (`MCPAgent::catalog`).  Each grew its own filter and only the planner's
//! checked `status`, so a dead grant went on being offered by the other two —
//! it keeps its cached tool list, so it looks healthy, and every call 401s
//! into a silent skip.
//!
//! Driven from one fixture on purpose: the bug was three copies disagreeing,
//! so the test that matters is the one that asks all three the same question.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use serial_test::serial;

use smartloop::agents::mcp::argument_extractor::MCPArgumentExtractor;
use smartloop::agents::mcp::explorer::{Explorer, ToolResolver};
use smartloop::agents::mcp::mcp_agent::{clear_resolution_cache, MCPAgent, ResolvedCall};
use smartloop::agents::mcp::mcp_tool_caller::MCPToolCaller;
use smartloop::agents::mcp::result_mapper::MCPResultMapper;
use smartloop::agents::orchestration::step_executor::StepExecutor;
use smartloop::agents::planner::mcp_connection_is_usable;
use smartloop::llms::BaseLLM;
use smartloop::mcp::MCPToolCallResult;
use smartloop::workspace::workspace::{MCPServer, MCPServerStatus, MCPServerType, MCPToolDefinition};

use crate::shared::agents_mcp_agent::StubLLM;

/// A remote connection with one discovered tool — the dead-grant shape.
///
/// A connection whose authorization died keeps everything discovery found, so
/// the only thing separating it from a healthy one is `status`.
fn server_with(status: MCPServerStatus, enabled: bool, server_id: &str) -> MCPServer {
    MCPServer {
        id: server_id.to_string(),
        name: "notion".to_string(),
        enabled,
        system: false,
        status,
        last_error: None,
        description: Some("Notion".to_string()),
        summary: None,
        server_type: MCPServerType::Remote,
        catalog_slug: None,
        pkg: None,
        remote_config: None,
        config: None,
        tools: vec![MCPToolDefinition {
            name: "search".to_string(),
            description: Some("Search pages".to_string()),
            input_schema: Some(Map::new()),
        }],
        ..Default::default()
    }
}

fn server() -> MCPServer {
    server_with(MCPServerStatus::Ready, true, "s1")
}

fn failed() -> MCPServer {
    server_with(MCPServerStatus::Failed, true, "s1")
}

// --------------------------------------------------------------- the predicate

#[test]
fn test_a_failed_connection_is_not_usable() {
    assert!(!mcp_connection_is_usable(&failed()));
}

#[test]
fn test_a_ready_connection_is() {
    assert!(mcp_connection_is_usable(&server()));
}

#[test]
fn test_a_disabled_connection_is_not() {
    assert!(!mcp_connection_is_usable(&server_with(MCPServerStatus::Ready, false, "s1")));
}

/// Provisioning has not finished; nothing says it is broken.
#[test]
fn test_a_pending_connection_still_is() {
    assert!(mcp_connection_is_usable(&server_with(MCPServerStatus::Pending, true, "s1")));
}

/// Records written before the field existed, and test stand-ins.
#[test]
fn test_a_server_with_no_status_is() {
    let bare = json!({"id": "s1", "name": "notion", "enabled": true});
    assert!(mcp_connection_is_usable(&bare));
}

/// Status arrives as the enum in the app and as a string in dict-shaped
/// records.
#[test]
fn test_the_enum_and_its_bare_value_agree() {
    assert!(!mcp_connection_is_usable(&failed()));
    assert!(mcp_connection_is_usable(&server()));
    let as_value = |status: &str| json!({"id": "s1", "name": "notion", "enabled": true, "status": status});
    assert!(!mcp_connection_is_usable(&as_value(MCPServerStatus::Failed.as_str())));
    assert!(mcp_connection_is_usable(&as_value(MCPServerStatus::Ready.as_str())));
}

// --------------------------------------------------------------- the consumers

#[test]
fn test_the_planner_catalog_leaves_it_out() {
    assert_ne!(StepExecutor::build_available_servers(&[server()]), Vec::<Value>::new());
    assert_eq!(StepExecutor::build_available_servers(&[failed()]), Vec::<Value>::new());
}

#[test]
#[serial]
fn test_the_mcp_agent_catalog_leaves_it_out() {
    clear_resolution_cache();
    let agent = MCPAgent::new(StubLLM::new("").as_llm());
    assert!(!agent.catalog(&[server()]).is_empty());
    assert!(agent.catalog(&[failed()]).is_empty());
    clear_resolution_cache();
}

/// Counts every resolution it is asked for (Python's `MagicMock().resolve`).
#[derive(Default)]
struct RecordingResolver {
    calls: AtomicUsize,
}

impl ToolResolver for RecordingResolver {
    fn resolve(
        &self,
        _directive: &str,
        _mcp_servers: &[MCPServer],
        _context: Option<&str>,
        _llm: Option<&dyn BaseLLM>,
    ) -> Option<ResolvedCall> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        None
    }
}

/// The path the dead-grant commit called the main one.
#[test]
#[serial]
fn test_the_collect_pass_leaves_it_out() {
    clear_resolution_cache();
    let llm = StubLLM::new("").as_llm();
    let arg_extractor = Arc::new(MCPArgumentExtractor::new(llm.clone()));
    let tool_caller = Arc::new(MCPToolCaller::new(
        arg_extractor.clone(),
        Arc::new(Mutex::new(MCPResultMapper::new())),
        None,
    ));
    let tool_calls = Arc::new(AtomicUsize::new(0));
    let counted = tool_calls.clone();
    tool_caller.set_call_tool_hook(Some(Arc::new(move |_server, _tool, _args, _token, _cwd| {
        counted.fetch_add(1, Ordering::SeqCst);
        MCPToolCallResult::default()
    })));
    let resolver = Arc::new(RecordingResolver::default());

    let mut explorer = Explorer::new(llm, Some(tool_caller), Some(arg_extractor), false);
    explorer.mcp_agent = resolver.clone();

    let result = explorer.run_mcp("what is in notion?", Some(&[failed()]), None, None, None);

    assert_eq!(result, "");
    // Not merely an empty answer — it must never have been asked.
    assert_eq!(resolver.calls.load(Ordering::SeqCst), 0);
    assert_eq!(tool_calls.load(Ordering::SeqCst), 0);
}
