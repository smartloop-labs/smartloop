//! Port of `tests/unit/test_analyze_node.py`.
//!
//! Unit tests for the ANALYZE node in the skill pipeline.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::json;

use smartloop::agents::mcp::explorer::SearchContext;
use smartloop::agents::mcp::server_selector::EXPLORE_TOOL;
use smartloop::agents::nodes::NodeResult;
use smartloop::agents::orchestration::OrchestratorContext;
use smartloop::agents::orchestrator::{SLPOrchestrator, SharedCell, SharedOrchestratorContext, SharedSearchContext};
use smartloop::agents::planner::{PlanStep, StepType};

use crate::shared::agents_core::{orchestrator_over, FakeLlm};

/// `mock_llm` fixture: replies `""`, a 4096-token window, every text 100
/// tokens.
fn mock_llm() -> Arc<FakeLlm> {
    FakeLlm::new("")
}

/// `orchestrator` fixture.
fn orchestrator(llm: &Arc<FakeLlm>) -> SLPOrchestrator {
    orchestrator_over(llm)
}

/// `analyze_step` fixture.
fn analyze_step() -> PlanStep {
    PlanStep {
        step: 2,
        step_type: StepType::Analyze,
        tool_name: None,
        server_id: Some(String::new()),
        server_name: Some(String::new()),
        instruction: "Assess relevance of search results to user request.".to_string(),
        ..Default::default()
    }
}

/// `ctx` fixture: a minimal request context.
fn ctx() -> SharedOrchestratorContext {
    let mut ctx = OrchestratorContext::new("proof read my document", vec![], vec![], "/tmp");
    ctx.skill_report = None;
    SharedCell::wrap(ctx)
}

/// `search_ctx` fixture.
fn search_ctx() -> SharedSearchContext {
    SharedCell::wrap(SearchContext {
        vector_context: String::new(),
        workspace_context: String::new(),
        feedback_context: String::new(),
        citations: Some(vec![]),
        ..Default::default()
    })
}

/// Build a deps map simulating upstream explore output.
fn make_deps(content: &str) -> HashMap<String, NodeResult> {
    HashMap::from([(
        "skill_1_explore".to_string(),
        NodeResult::with_data(
            "skill_1_explore",
            json!({
                "tool_name": EXPLORE_TOOL,
                "content": content,
                "raw": content,
            }),
        ),
    )])
}

/// Tests for the ANALYZE branch of the step handler.
mod analyze_handler {
    use super::*;

    /// ANALYZE handler should call the LLM with the explore output.
    #[test]
    fn test_analyze_invokes_llm_with_explore_context() {
        let llm = mock_llm();
        let orchestrator = orchestrator(&llm);
        llm.set_reply("The document contains a summary of Q3 results.");

        let ctx = ctx();
        let handler = orchestrator.pipeline().make_step_handler(&ctx, &analyze_step(), &search_ctx());
        let result = handler(make_deps("Q3 financial results: revenue up 15%.")).unwrap();

        assert_eq!(llm.call_count(), 1);
        let prompt_arg = &llm.calls()[0];
        assert!(prompt_arg.contains("Q3 financial results"));
        assert!(prompt_arg.contains("proof read"));
        assert!(result.success);
        assert_ne!(result.data["content"], json!(""));
    }

    /// When the LLM deems context irrelevant, handler passes through raw
    /// explore data.
    #[test]
    fn test_analyze_irrelevant_context_passes_through_raw() {
        let llm = mock_llm();
        let orchestrator = orchestrator(&llm);
        llm.set_reply("NOT_RELEVANT");

        let ctx = ctx();
        let handler = orchestrator.pipeline().make_step_handler(&ctx, &analyze_step(), &search_ctx());
        let result = handler(make_deps("Unrelated cooking recipe data.")).unwrap();

        assert!(result.success);
        // Raw explore content is passed through so downstream steps can use it
        assert_eq!(result.data["content"], json!("Unrelated cooking recipe data."));
        assert!(ctx.with_ref(|c| c.skill_report.clone()).unwrap_or_default().is_empty());
    }

    /// Relevant analysis should be stored on the request context.
    #[test]
    fn test_analyze_relevant_context_stores_skill_report() {
        let llm = mock_llm();
        let orchestrator = orchestrator(&llm);
        let analysis = "The document has several grammatical errors in paragraphs 2 and 5.";
        llm.set_reply(analysis);

        let ctx = ctx();
        let handler = orchestrator.pipeline().make_step_handler(&ctx, &analyze_step(), &search_ctx());
        let result = handler(make_deps("Draft document with typos and grammar issues.")).unwrap();

        assert_eq!(ctx.with_ref(|c| c.skill_report.clone()), Some(analysis.to_string()));
        assert_eq!(result.data["content"], json!(analysis));
        assert_eq!(result.data["raw"], json!(analysis));
    }

    /// When explore produced no content, ANALYZE should skip the LLM call.
    #[test]
    fn test_analyze_empty_deps_returns_empty() {
        let llm = mock_llm();
        let orchestrator = orchestrator(&llm);

        let ctx = ctx();
        let handler = orchestrator.pipeline().make_step_handler(&ctx, &analyze_step(), &search_ctx());
        let result = handler(make_deps("")).unwrap();

        assert_eq!(llm.call_count(), 0);
        assert_eq!(result.data["content"], json!(""));
    }

    /// If the LLM fails, ANALYZE passes through raw explore data.
    #[test]
    fn test_analyze_llm_failure_returns_empty() {
        let llm = mock_llm();
        let orchestrator = orchestrator(&llm);
        llm.set_failure("model error");

        let ctx = ctx();
        let handler = orchestrator.pipeline().make_step_handler(&ctx, &analyze_step(), &search_ctx());
        let result = handler(make_deps("Some explore content.")).unwrap();

        assert!(result.success);
        // Raw explore content is passed through on failure
        assert_eq!(result.data["content"], json!("Some explore content."));
    }
}
