//! Port of `tests/unit/test_tool_call_progress.py`.
//!
//! The client is told which call is being made, not just which model ran.
//!
//! A TOOL step's whole decision is *which call*: server, tool, arguments.
//! The progress stream carried "Selected <model> for the response" and
//! nothing else, so a resolution that reasoned for several seconds looked
//! like a stall, and a call that went out with the wrong arguments was
//! visible only in the server log.
//!
//! Python built the pipeline over mocks and stubbed the step executor's
//! `resolve_tool_args` / `call_tool`.  Here the pipeline is the real one: the
//! step resolves against a registered "linear" server whose model fills the
//! arguments, the call goes out through the tool caller's transport seam, and
//! the router's progress sink records what the client is told.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::json;

use smartloop::agents::mcp::explorer::SearchContext;
use smartloop::agents::nodes::NodeResult;
use smartloop::agents::orchestration::step_executor::StepExecutor;
use smartloop::agents::orchestration::OrchestratorContext;
use smartloop::agents::orchestrator::{SLPOrchestrator, SharedCell, SharedOrchestratorContext, SharedSearchContext};
use smartloop::agents::planner::{PlanStep, StepType};
use smartloop::workspace::workspace::MCPServer;

use crate::shared::agents_core::{orchestrator_over, remote_server, text_result, tool, EmitSink, FakeLlm};

struct Built {
    orch: SLPOrchestrator,
    llm: Arc<FakeLlm>,
    sink: EmitSink,
    ctx: SharedOrchestratorContext,
    step: PlanStep,
}

fn linear_server() -> MCPServer {
    remote_server(
        "srv-1",
        "linear",
        vec![tool("create_issue", "Create an issue", json!({"title": {"type": "string"}}), &["title"])],
    )
}

/// The `built` fixture.  `servers` are the request's MCP connections.
fn built(servers: Vec<MCPServer>) -> Built {
    // The capability model reads as reasoning-capable.
    let llm = FakeLlm::new(&json!({"title": "Login is broken"}).to_string());
    llm.set_supports_thinking(true);
    let orch = orchestrator_over(&llm);
    let sink = EmitSink::default();
    orch.models().begin_request(Some(sink.callback()));

    let available = StepExecutor::build_available_servers(&servers);
    let ctx = OrchestratorContext::new("file an issue: Login is broken", servers, available, "/tmp");
    let step = PlanStep {
        step: 1,
        step_type: StepType::Tool,
        tool_name: Some("create_issue".to_string()),
        server_id: Some("srv-1".to_string()),
        server_name: Some("linear".to_string()),
        instruction: "file an issue: Login is broken".to_string(),
        ..Default::default()
    };
    Built { orch, llm, sink, ctx: SharedCell::wrap(ctx), step }
}

fn search_ctx() -> SharedSearchContext {
    SharedCell::wrap(SearchContext::default())
}

#[test]
fn test_resolution_reports_the_call_it_settled_on() {
    let b = built(vec![linear_server()]);
    let handler = b.orch.pipeline().make_resolve_handler(&b.ctx, &b.step, &search_ctx(), true);

    handler(HashMap::new()).unwrap();

    assert!(
        b.sink.messages().contains(&"Resolved linear.create_issue(title='Login is broken')".to_string()),
        "messages: {:?}",
        b.sink.messages()
    );
}

/// A reasoning resolution takes seconds; say what is being decided.
#[test]
fn test_the_wait_is_explained_before_the_call_is_known() {
    let b = built(vec![linear_server()]);
    let handler = b.orch.pipeline().make_resolve_handler(&b.ctx, &b.step, &search_ctx(), true);

    handler(HashMap::new()).unwrap();

    // The model here is reasoning-capable, so the wait is named as reasoning.
    assert_eq!(b.sink.messages()[0], "Reasoning about which tool to call");
    let _ = &b.llm;
}

#[test]
fn test_a_step_that_resolves_to_nothing_says_so() {
    // No connection serves the step, so nothing can be resolved.
    let b = built(vec![]);
    let handler = b.orch.pipeline().make_resolve_handler(&b.ctx, &b.step, &search_ctx(), true);

    let result = handler(HashMap::new()).unwrap();

    assert_eq!(result.data["skip"], json!(true));
    assert!(b.sink.messages().iter().any(|m| m.contains("No tool call could be resolved")));
}

#[test]
fn test_the_call_is_reported_as_it_goes_out() {
    let b = built(vec![linear_server()]);
    b.orch
        .tool_caller
        .set_call_tool_hook(Some(Arc::new(|_server, _tool, _args, _token, _cwd| text_result("formatted"))));
    let handler = b.orch.pipeline().make_execute_handler(&b.ctx, &b.step);

    handler(HashMap::from([(
        "resolve".to_string(),
        NodeResult::with_data(
            "resolve",
            json!({
                "skip": false,
                "tool_name": "create_issue",
                "server_id": "srv-1",
                "arguments": {"title": "Login is broken"},
            }),
        ),
    )]))
    .unwrap();

    assert!(
        b.sink.messages().contains(&"Calling linear.create_issue(title='Login is broken')".to_string()),
        "messages: {:?}",
        b.sink.messages()
    );
}
