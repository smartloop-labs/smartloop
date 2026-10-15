//! A conversational step drafts no reply when the answer is remote.
//!
//! The chat handler writes the final answer from the step's context.  When
//! that answer comes from a pinned remote model, a draft from the step would
//! be a second upstream call whose text only feeds the first.

use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use smartloop::agents::mcp::argument_extractor::MCPArgumentExtractor;
use smartloop::agents::mcp::dependency_analyzer::MCPDependencyAnalyzer;
use smartloop::agents::mcp::explorer::Explorer;
use smartloop::agents::mcp::mcp_agent::MCPAgent;
use smartloop::agents::mcp::mcp_tool_caller::MCPToolCaller;
use smartloop::agents::mcp::result_mapper::MCPResultMapper;
use smartloop::agents::orchestration::context::OrchestratorContext;
use smartloop::agents::orchestration::response_builder::ResponseBuilder;
use smartloop::agents::orchestration::step_executor::StepExecutor;
use smartloop::agents::planner::{PlanStep, StepType};
use smartloop::llms::BaseLLM;
use smartloop::utils::mcp_result_formatter::ResultFormatter;

use crate::shared::orch_exec::StubLLM;

fn executor(answers_remotely: bool) -> StepExecutor {
    let llm = StubLLM::shared("");
    let arg_extractor = Arc::new(MCPArgumentExtractor::new(llm.clone()));
    let tool_caller = Arc::new(MCPToolCaller::new(
        arg_extractor.clone(),
        Arc::new(Mutex::new(MCPResultMapper::new())),
        None,
    ));
    StepExecutor::new(
        llm.clone(),
        tool_caller.clone(),
        arg_extractor.clone(),
        Arc::new(MCPDependencyAnalyzer::new()),
        Arc::new(ResultFormatter::new(llm.clone())),
        Arc::new(Explorer::new(llm.clone(), Some(tool_caller), Some(arg_extractor), false)),
        Arc::new(MCPAgent::new(llm.clone())),
        Arc::new(ResponseBuilder::new(llm)),
    )
    .with_remote_answer(answers_remotely)
}

/// Run one conversational step on `answer`; returns the step's output.
fn converse(answers_remotely: bool, answer: Arc<StubLLM>) -> Option<(String, String)> {
    let ctx = Arc::new(RwLock::new(OrchestratorContext::new("hello there", Vec::new(), Vec::new(), "/tmp")));
    let step = PlanStep {
        step: 1,
        step_type: StepType::Conversational,
        instruction: "hello there".into(),
        capability: Some("general".into()),
        ..Default::default()
    };
    let llm: Arc<dyn BaseLLM> = answer;
    executor(answers_remotely).execute(&ctx, &step, None, None, Some(llm))
}

#[test]
fn test_a_local_answer_is_drafted_by_the_step() {
    let answer = Arc::new(StubLLM::new("Hi!"));
    let out = converse(false, answer.clone());
    assert_eq!(out.map(|(text, _)| text).as_deref(), Some("Hi!"));
    assert_eq!(answer.prompts.lock().len(), 1);
}

#[test]
fn test_a_remote_answer_is_not_drafted_by_the_step() {
    let answer = Arc::new(StubLLM::new("Hi!"));
    let out = converse(true, answer.clone());
    assert!(answer.prompts.lock().is_empty());
    let (text, _) = out.expect("the step still passes its context through");
    assert!(text.contains("hello there"), "context: {text}");
}
