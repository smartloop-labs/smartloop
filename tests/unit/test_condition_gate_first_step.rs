//! Port of `tests/unit/test_condition_gate_first_step.py`.
//!
//! A first tool step must not be skipped by a skill-extracted condition.
//!
//! Regression: a lone fetch step matched via the fetch-url skill set
//! `ctx.skill_context`; the resolver then read the skill's applicability
//! clause ("when the user requests a URL, fetch it") as a runtime condition
//! and skipped the only step, so the answer was synthesized from no fetched
//! content.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::json;

use smartloop::agents::mcp::explorer::SearchContext;
use smartloop::agents::orchestration::OrchestratorContext;
use smartloop::agents::orchestrator::{SLPOrchestrator, SharedCell, SharedOrchestratorContext, SharedSearchContext};
use smartloop::agents::planner::{PlanStep, StepType};
use smartloop::workspace::workspace::MCPServer;

use crate::shared::agents_core::{fetch_server, orchestrator_over, FakeLlm};

/// Text only the condition extractor's prompt carries.
const CONDITION_PROMPT_MARKER: &str = "is conditional on some prior result";

/// `mock_llm` fixture: answers "YES" to everything.
fn mock_llm() -> Arc<FakeLlm> {
    FakeLlm::new("YES")
}

fn orchestrator(llm: &Arc<FakeLlm>) -> SLPOrchestrator {
    orchestrator_over(llm)
}

/// `_ctx(skill_context)`.
fn ctx(skill_context: &str, mcp_servers: Vec<MCPServer>) -> SharedOrchestratorContext {
    let mut ctx = OrchestratorContext::new("summarize https://example.com/post", mcp_servers, vec![], "/tmp");
    ctx.conversation_history = Some(vec![]);
    ctx.has_attachments = false;
    ctx.skill_context = Some(skill_context.to_string());
    SharedCell::wrap(ctx)
}

/// `_fetch_step()`.
fn fetch_step(server_id: &str) -> PlanStep {
    PlanStep {
        step: 1,
        step_type: StepType::Tool,
        tool_name: Some("fetch".to_string()),
        server_id: Some(server_id.to_string()),
        server_name: Some("fetch".to_string()),
        instruction: "Fetch the URL and extract its content.".to_string(),
        ..Default::default()
    }
}

fn search_ctx() -> SharedSearchContext {
    SharedCell::wrap(SearchContext::default())
}

mod extractor_hardening {
    use super::*;

    /// A reply with no '?' is malformed → no condition.
    #[test]
    fn test_bare_yes_is_not_a_condition() {
        let llm = mock_llm();
        let orchestrator = orchestrator(&llm);
        assert_eq!(orchestrator.skills().extract_tool_condition("when user requests a url use fetch tool", "fetch"), None);
    }

    #[test]
    fn test_real_question_is_a_condition() {
        let llm = mock_llm();
        let orchestrator = orchestrator(&llm);
        llm.set_reply("Were discrepancies found in the results?");
        let cond = orchestrator
            .skills()
            .extract_tool_condition("after analysis, if discrepancies found call fetch", "fetch");
        assert_eq!(cond, Some("llm:Were discrepancies found in the results?".to_string()));
    }
}

mod first_step_not_gated {
    use super::*;

    /// has_upstream=false → the step resolves instead of being skipped.
    ///
    /// Python stubbed `resolve_tool_args` (returning a resolved fetch call)
    /// and spied on `extract_tool_condition`.  Here the step resolves for
    /// real against a registered fetch server: the model fills the `url`
    /// argument, and answers any condition-extraction prompt with a real
    /// question — so a consulted extractor would gate (and skip) the step,
    /// and its prompt would show up in the model's call log.
    #[test]
    fn test_first_step_runs_despite_skill_condition() {
        let llm = mock_llm();
        let orchestrator = orchestrator(&llm);
        llm.set_responder(|prompt| {
            if prompt.contains(CONDITION_PROMPT_MARKER) {
                Ok("Were discrepancies found?".to_string())
            } else {
                Ok(json!({"url": "https://example.com/post"}).to_string())
            }
        });
        let ctx = ctx(
            "when the user requests a url use fetch tool to fetch it",
            vec![fetch_server("srv", "fetch")],
        );
        let step = fetch_step("srv");

        let handler = orchestrator.pipeline().make_resolve_handler(&ctx, &step, &search_ctx(), false);
        let result = handler(HashMap::new()).unwrap();

        // The extractor is never consulted for a first step.
        assert!(!llm.calls().iter().any(|p| p.contains(CONDITION_PROMPT_MARKER)));
        assert_eq!(result.data.get("skip"), Some(&json!(false)));
        assert_eq!(result.data.get("arguments"), Some(&json!({"url": "https://example.com/post"})));
    }

    /// has_upstream=true with an extracted condition and no prior results →
    /// the step is skipped (existing behavior preserved).
    #[test]
    fn test_downstream_step_still_gated_when_no_prior_results() {
        let llm = mock_llm();
        let orchestrator = orchestrator(&llm);
        let ctx = ctx("after analysis, if discrepancies found call fetch", vec![]);
        llm.set_reply("Were discrepancies found?");
        let step = fetch_step("");

        let handler = orchestrator.pipeline().make_resolve_handler(&ctx, &step, &search_ctx(), true);
        let result = handler(HashMap::new()).unwrap();

        assert_eq!(result.data.get("skip"), Some(&json!(true)));
    }
}
