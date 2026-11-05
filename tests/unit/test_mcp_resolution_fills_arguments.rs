//! Port of `tests/unit/test_mcp_resolution_fills_arguments.py`.
//!
//! One call resolves the tool AND its arguments; the extractor is the
//! fallback.
//!
//! Resolution and extraction used to be two calls: retrieval picked a tool
//! from embedded descriptions, then a second prompt worked out the arguments
//! for it with no knowledge of why it had been picked.  `MCPAgent::resolve`
//! answers both at once, so the step runs what came back instead of asking
//! again — and only falls back to `MCPArgumentExtractor` when what came back
//! does not cover what the tool requires.
//!
//! Python stubbed the tool caller and dependency analyzer with mocks; here the
//! real ones run against a real server record (a local server, so no URL is
//! needed, and a schema with no id parameters, so no dependencies), and the
//! two LLM-backed calls — resolution and extraction — are replaced through the
//! executor's hooks.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Map, Value};

use smartloop::agents::mcp::argument_extractor::MCPArgumentExtractor;
use smartloop::agents::mcp::dependency_analyzer::MCPDependencyAnalyzer;
use smartloop::agents::mcp::explorer::Explorer;
use smartloop::agents::mcp::mcp_agent::{MCPAgent, ResolvedCall};
use smartloop::agents::mcp::mcp_tool_caller::MCPToolCaller;
use smartloop::agents::mcp::result_mapper::MCPResultMapper;
use smartloop::agents::orchestration::context::OrchestratorContext;
use smartloop::agents::orchestration::response_builder::ResponseBuilder;
use smartloop::agents::orchestration::step_executor::StepExecutor;
use smartloop::agents::planner::{PlanStep, StepType};
use smartloop::utils::mcp_result_formatter::ResultFormatter;
use smartloop::workspace::workspace::{MCPServer, MCPServerType, MCPToolDefinition};

use crate::shared::orch_exec::StubLLM;

fn schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "The issue title"},
            "priority": {"type": "number", "description": "1-4, lower is higher"},
        },
        "required": ["title"],
    })
    .as_object()
    .cloned()
    .unwrap()
}

/// What `mcp_agent.resolve` was called with: `(directive, context)`.
type ResolveCalls = Arc<Mutex<Vec<(String, Option<String>)>>>;

/// A StepExecutor with everything but resolution and extraction real.
struct Wired {
    executor: StepExecutor,
    ctx: OrchestratorContext,
    step: PlanStep,
    server: MCPServer,
    /// What resolution answers (`mcp_agent.resolve.return_value`).
    resolve_returns: Arc<Mutex<Option<ResolvedCall>>>,
    resolve_calls: ResolveCalls,
    /// How many times the extractor's LLM pass ran (`arg_extractor.invoke`).
    extract_calls: Arc<Mutex<usize>>,
}

fn wired() -> Wired {
    let server = MCPServer {
        id: "srv-1".into(),
        name: "linear".into(),
        enabled: true,
        server_type: MCPServerType::Local,
        tools: vec![MCPToolDefinition {
            name: "create_issue".into(),
            description: Some("Create an issue".into()),
            input_schema: Some(schema()),
        }],
        ..Default::default()
    };

    let llm = StubLLM::shared("");
    // Real extractor (its URL/id helpers are used inline) with the LLM pass
    // spied on, so "did it ask a second time?" is answerable.
    let arg_extractor = Arc::new(MCPArgumentExtractor::new(llm.clone()));
    let tool_caller = Arc::new(MCPToolCaller::new(
        arg_extractor.clone(),
        Arc::new(Mutex::new(MCPResultMapper::new())),
        None,
    ));

    let executor = StepExecutor::new(
        llm.clone(),
        tool_caller.clone(),
        arg_extractor.clone(),
        Arc::new(MCPDependencyAnalyzer::new()),
        Arc::new(ResultFormatter::new(llm.clone())),
        Arc::new(Explorer::new(llm.clone(), Some(tool_caller), Some(arg_extractor), false)),
        Arc::new(MCPAgent::new(llm.clone())),
        Arc::new(ResponseBuilder::new(llm)),
    );

    let resolve_returns: Arc<Mutex<Option<ResolvedCall>>> = Arc::new(Mutex::new(None));
    let resolve_calls: ResolveCalls = Arc::new(Mutex::new(Vec::new()));
    {
        let returns = resolve_returns.clone();
        let calls = resolve_calls.clone();
        executor.set_resolve_hook(Some(Arc::new(move |directive, _servers, context, _llm| {
            calls.lock().push((directive.to_string(), context.map(str::to_string)));
            returns.lock().clone()
        })));
    }
    let extract_calls = Arc::new(Mutex::new(0usize));
    {
        let calls = extract_calls.clone();
        executor.set_extract_hook(Some(Arc::new(move |_input, _tool, _schema, _prev| {
            *calls.lock() += 1;
            json!({"title": "from the extractor"}).as_object().cloned().unwrap()
        })));
    }

    let mut ctx = OrchestratorContext::new(
        "file an issue: Login is broken",
        vec![server.clone()],
        vec![serde_json::to_value(&server).unwrap()],
        "/tmp",
    );
    ctx.project_dir = Some("/tmp/project".into());
    let step = PlanStep {
        step: 1,
        step_type: StepType::Tool,
        instruction: "file an issue: Login is broken".into(),
        ..Default::default()
    };

    Wired { executor, ctx, step, server, resolve_returns, resolve_calls, extract_calls }
}

fn args(v: Value) -> Map<String, Value> {
    v.as_object().cloned().unwrap()
}

#[test]
fn test_a_complete_resolution_is_called_as_returned() {
    let mut wired = wired();
    *wired.resolve_returns.lock() = Some(ResolvedCall {
        server_id: "srv-1".into(),
        tool_name: "create_issue".into(),
        arguments: args(json!({"title": "Login is broken"})),
        reasoned: true,
        native: false,
        also: Vec::new(),
    });

    let resolved = wired.executor.resolve_tool_args(&wired.ctx, &mut wired.step, None, None);

    assert_eq!(
        resolved,
        Some((wired.server.clone(), "create_issue".to_string(), args(json!({"title": "Login is broken"}))))
    );
    assert_eq!(*wired.extract_calls.lock(), 0);
}

/// A required argument still missing is the one case worth a second pass.
#[test]
fn test_a_partial_resolution_falls_back_to_the_extractor() {
    let mut wired = wired();
    *wired.resolve_returns.lock() = Some(ResolvedCall::new("srv-1", "create_issue"));

    let resolved = wired.executor.resolve_tool_args(&wired.ctx, &mut wired.step, None, None);

    assert_eq!(
        resolved,
        Some((wired.server.clone(), "create_issue".to_string(), args(json!({"title": "from the extractor"}))))
    );
    assert_eq!(*wired.extract_calls.lock(), 1);
}

/// A chained call takes its arguments from the prior result, so the prior
/// result has to reach the call that picks the tool.
#[test]
fn test_resolution_reads_what_earlier_steps_returned() {
    let mut wired = wired();
    *wired.resolve_returns.lock() = Some(ResolvedCall {
        server_id: "srv-1".into(),
        tool_name: "create_issue".into(),
        arguments: args(json!({"title": "Login is broken"})),
        reasoned: false,
        native: false,
        also: Vec::new(),
    });

    let previous = vec![json!({"tool_name": "list_teams", "result": "team ENG"})];
    wired.executor.resolve_tool_args(&wired.ctx, &mut wired.step, Some(&previous), None);

    let calls = wired.resolve_calls.lock();
    let context = calls.last().expect("resolution was asked").1.clone().unwrap_or_default();
    assert!(context.contains("list_teams") && context.contains("team ENG"));
}

#[test]
fn test_an_unresolvable_step_is_skipped() {
    let mut wired = wired();
    *wired.resolve_returns.lock() = None;
    // (Python also made `tool_caller.find_server` return None; the step names
    // no server, so the real lookup finds none either.)

    assert_eq!(wired.executor.resolve_tool_args(&wired.ctx, &mut wired.step, None, None), None);
}

/// A native tool call saw the whole schema, so the optional arguments it
/// filled (a ticket's priority) reach the tool.
#[test]
fn test_a_native_resolution_keeps_optional_arguments() {
    let mut wired = wired();
    *wired.resolve_returns.lock() = Some(ResolvedCall {
        server_id: "srv-1".into(),
        tool_name: "create_issue".into(),
        arguments: args(json!({"title": "Login is broken", "priority": 1})),
        reasoned: true,
        native: true,
        also: Vec::new(),
    });

    let resolved = wired.executor.resolve_tool_args(&wired.ctx, &mut wired.step, None, None);

    assert_eq!(
        resolved,
        Some((
            wired.server.clone(),
            "create_issue".to_string(),
            args(json!({"title": "Login is broken", "priority": 1}))
        ))
    );
    assert_eq!(*wired.extract_calls.lock(), 0);
}

/// The prompt-based resolution still calls with required arguments only.
#[test]
fn test_a_prompt_resolution_drops_optional_arguments() {
    let mut wired = wired();
    *wired.resolve_returns.lock() = Some(ResolvedCall {
        server_id: "srv-1".into(),
        tool_name: "create_issue".into(),
        arguments: args(json!({"title": "Login is broken", "priority": 1})),
        reasoned: true,
        native: false,
        also: Vec::new(),
    });

    let resolved = wired.executor.resolve_tool_args(&wired.ctx, &mut wired.step, None, None);

    assert_eq!(
        resolved,
        Some((wired.server.clone(), "create_issue".to_string(), args(json!({"title": "Login is broken"}))))
    );
}
