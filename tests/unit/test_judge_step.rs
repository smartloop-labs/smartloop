//! Port of `tests/unit/test_judge_step.py`.
//!
//! Tests for `JudgeStep` — relevance evaluation of context.
//!
//! Every piece of retrieved context and every step output is judged against
//! the user's request before it is appended to the answer context.  Without
//! that check an unrelated document reaches the prompt fenced as DOCUMENT
//! CONTEXT, and the model answers that it has no context for the question
//! instead of answering the question.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};

use smartloop::agents::mcp::explorer::SearchContext;
use smartloop::agents::nodes::{LLMDecisionNode, NodeResult};
use smartloop::agents::orchestration::context as octx;
use smartloop::agents::orchestration::judge_step::JudgeStep;
use smartloop::agents::orchestration::model_router::{ModelRouter, ModelRouterOptions};
use smartloop::agents::orchestration::step::Step;
use smartloop::agents::orchestration::OrchestratorContext;
use smartloop::agents::orchestrator::{SLPOrchestrator, SharedCell, SharedOrchestratorContext, SharedSearchContext};
use smartloop::agents::planner::{PlanStep, StepType};

use crate::shared::agents_core::{capture_logs, class_name, llm_8k, orchestrator_over, FakeLlm};

/// `_make_orchestrator()`: an orchestrator over a model that judges
/// everything RELEVANT.  The model is returned so tests can re-program it
/// (`orch.agent_llm.invoke = ...`).
fn make_orchestrator() -> (SLPOrchestrator, Arc<FakeLlm>) {
    let llm = llm_8k("RELEVANT");
    (orchestrator_over(&llm), llm)
}

/// `_ctx(orch)`.
fn ctx_value(user_message: &str) -> OrchestratorContext {
    let mut ctx = OrchestratorContext::new(user_message, vec![], vec![], "/tmp");
    ctx.conversation_history = Some(vec![]);
    ctx.has_attachments = false;
    ctx
}

fn madrid_ctx() -> OrchestratorContext {
    ctx_value("what are things to do in madrid")
}

/// Route the judge's reply off whichever fixture text is in the prompt.
fn verdicts(llm: &FakeLlm, mapping: &[(&str, &str)], default: &str) {
    let mapping: Vec<(String, String)> = mapping.iter().map(|(n, v)| (n.to_string(), v.to_string())).collect();
    let default = default.to_string();
    llm.set_responder(move |prompt| {
        for (needle, verdict) in &mapping {
            if prompt.contains(needle.as_str()) {
                return Ok(verdict.clone());
            }
        }
        Ok(default.clone())
    });
}

/// A judge wired to its own router, with no orchestrator around it.
fn judge() -> (JudgeStep, Arc<FakeLlm>) {
    let llm = FakeLlm::new("RELEVANT");
    let router = ModelRouter::new(llm.shared(), llm.shared(), ModelRouterOptions::default());
    (JudgeStep::new(Arc::new(router)), llm)
}

// ------------------------------------------------------------------ #
// JudgeStep.serves_intent                                            #
// ------------------------------------------------------------------ #

mod serves_intent {
    use super::*;

    #[test]
    fn test_relevant_content_is_kept() {
        let (judge, llm) = judge();
        llm.set_reply("RELEVANT");
        assert!(judge.serves_intent("things to do in madrid", "documents", "Retiro Park and the Prado"));
    }

    #[test]
    fn test_irrelevant_content_is_rejected() {
        let (judge, llm) = judge();
        llm.set_reply("NOT_RELEVANT");
        assert!(!judge.serves_intent("things to do in madrid", "documents", "Q3 invoice for Adam Hart"));
    }

    #[test]
    fn test_empty_content_is_rejected_without_an_llm_call() {
        let (judge, llm) = judge();
        assert!(!judge.serves_intent("madrid", "documents", "   "));
        assert_eq!(llm.call_count(), 0);
    }

    #[test]
    fn test_llm_failure_keeps_the_context() {
        let (judge, llm) = judge();
        llm.set_failure("no model");
        assert!(judge.serves_intent("madrid", "documents", "text"));
    }

    #[test]
    fn test_unparseable_verdict_keeps_the_context() {
        let (judge, llm) = judge();
        llm.set_reply("hmm, maybe?");
        assert!(judge.serves_intent("madrid", "documents", "text"));
    }

    /// The reply that broke monday.com retrieval.
    ///
    /// The judge's own prompt says "Reply RELEVANT or NOT_RELEVANT", and a
    /// small model quoting it back was read as a rejection by the old
    /// substring check — so a 0.111-distance match for "list my boards on
    /// monday.com" was discarded and answered with "I don't have access to
    /// your Monday.com boards".
    #[test]
    fn test_an_echoed_instruction_is_not_a_rejection() {
        let (judge, llm) = judge();
        llm.set_reply("Reply RELEVANT or NOT_RELEVANT");
        assert!(judge.serves_intent("list my boards on monday.com", "MCP content", "boards: Tickets"));
    }

    #[test]
    fn test_a_prose_answer_that_means_relevant_is_kept() {
        let (judge, llm) = judge();
        llm.set_reply("The context is RELEVANT, not NOT_RELEVANT.");
        assert!(judge.serves_intent("madrid", "documents", "Retiro Park"));
    }

    /// A well-formed answer leads with the verdict and then reasons, and the
    /// reasoning naming the other label must not turn the verdict into a keep.
    #[test]
    fn test_a_rejection_that_explains_itself_still_rejects() {
        let (judge, llm) = judge();
        llm.set_reply("NOT_RELEVANT — this is about Trello, not relevant to the monday.com question");
        assert!(!judge.serves_intent("list my boards on monday.com", "MCP content", "name: Pay Christy"));
    }

    #[test]
    fn test_a_decorated_verdict_still_counts() {
        let (judge, llm) = judge();
        llm.set_reply("**NOT_RELEVANT**");
        assert!(!judge.serves_intent("madrid", "documents", "invoice"));
    }

    /// Without the reply in the log, a misjudging model and a misread verdict
    /// produce the same line.
    #[test]
    fn test_a_dropped_source_logs_what_the_model_replied() {
        let (judge, llm) = judge();
        llm.set_reply("NOT_RELEVANT, wrong subject");
        let (_, logs) = capture_logs(|| judge.serves_intent("madrid", "documents", "invoice"));
        assert!(logs.contains("wrong subject"));
    }

    #[test]
    fn test_a_verdictless_reply_is_logged_as_kept() {
        let (judge, llm) = judge();
        llm.set_reply("MONDAY.COM");
        let (kept, logs) = capture_logs(|| judge.serves_intent("madrid", "documents", "text"));
        assert!(kept);
        assert!(logs.contains("no verdict"));
        assert!(logs.contains("MONDAY.COM"));
    }

    #[test]
    fn test_verdict_is_memoized_per_request() {
        let (judge, llm) = judge();
        llm.set_reply("NOT_RELEVANT");
        for _ in 0..3 {
            judge.serves_intent("madrid", "documents", "invoice");
        }
        assert_eq!(llm.call_count(), 1);
    }

    /// Verdicts are scoped to one request — a new one re-judges.
    #[test]
    fn test_begin_request_clears_the_memo() {
        let (judge, llm) = judge();
        llm.set_reply("NOT_RELEVANT");
        judge.serves_intent("madrid", "documents", "invoice");
        judge.begin_request();
        judge.serves_intent("madrid", "documents", "invoice");
        assert_eq!(llm.call_count(), 2);
    }
}

// ------------------------------------------------------------------ #
// JudgeStep.evaluate_sources                                         #
// ------------------------------------------------------------------ #

mod evaluate_sources {
    use super::*;

    /// One irrelevant source must not take the relevant ones with it.
    #[test]
    fn test_each_source_is_judged_on_its_own() {
        let (orch, llm) = make_orchestrator();
        let ctx = madrid_ctx();
        verdicts(&llm, &[("OLD INVOICE", "NOT_RELEVANT")], "RELEVANT");
        let mut search_ctx = SearchContext {
            workspace_context: "OLD INVOICE for Adam Hart".to_string(),
            feedback_context: "Madrid trip notes: visit the Prado".to_string(),
            ..Default::default()
        };

        let dropped = orch.judge().evaluate_sources(&ctx, &mut search_ctx);

        assert_eq!(dropped, vec!["workspace files".to_string()]);
        assert_eq!(search_ctx.workspace_context, "");
        assert_eq!(search_ctx.feedback_context, "Madrid trip notes: visit the Prado");
    }

    #[test]
    fn test_all_irrelevant_sources_are_cleared() {
        let (orch, llm) = make_orchestrator();
        let ctx = madrid_ctx();
        llm.set_reply("NOT_RELEVANT");
        let mut search_ctx = SearchContext {
            workspace_context: "unrelated".to_string(),
            feedback_context: "unrelated".to_string(),
            mcp_context: "unrelated".to_string(),
            ..Default::default()
        };

        let dropped = orch.judge().evaluate_sources(&ctx, &mut search_ctx);

        assert_eq!(dropped, vec!["workspace files".to_string(), "feedback".to_string()]);
        // The MCP result is exempt — it is the user's connected-app data.
        assert_eq!(search_ctx.mcp_context, "unrelated");
    }

    /// A tool call is deliberate and its result is definite — not recall.
    ///
    /// The judge runs on the small decision model and fails toward dropping,
    /// so gating tool output means one bad verdict discards the data the
    /// request went and fetched.  A wrong tool is a routing problem; binning
    /// its result does not produce the right one.
    #[test]
    fn test_a_live_tool_result_is_never_judged() {
        let (orch, llm) = make_orchestrator();
        let ctx = madrid_ctx();
        llm.set_reply("NOT_RELEVANT");
        let mut search_ctx = SearchContext {
            mcp_context: "sprint backlog: move model hosting to R2".to_string(),
            web_context: "The Hasselblad X2D II has a 100MP sensor.".to_string(),
            ..Default::default()
        };

        let dropped = orch.judge().evaluate_sources(&ctx, &mut search_ctx);

        assert!(dropped.is_empty());
        assert_eq!(search_ctx.mcp_context, "sprint backlog: move model hosting to R2");
        // Web results were fetched for this query — never judged, never dropped.
        assert_eq!(search_ctx.web_context, "The Hasselblad X2D II has a 100MP sensor.");
    }

    /// Nothing judgeable left, so the judge stays off the critical path.
    #[test]
    fn test_mcp_context_alone_does_not_start_the_judge() {
        let (orch, _llm) = make_orchestrator();
        let ctx = madrid_ctx();
        let search_ctx = SearchContext { mcp_context: "open tickets = 4".to_string(), ..Default::default() };

        assert!(!orch.judge().applies_to(&ctx, Some(&search_ctx)));
    }

    #[test]
    fn test_an_empty_search_context_never_reaches_the_llm() {
        let (orch, llm) = make_orchestrator();
        let ctx = madrid_ctx();
        llm.set_reply("");

        assert!(orch.judge().evaluate_sources(&ctx, &mut SearchContext::default()).is_empty());
        assert_eq!(llm.call_count(), 0);
    }

    /// Web results were fetched for this query — on-topic by construction.
    #[test]
    fn test_web_context_is_never_judged() {
        let (orch, llm) = make_orchestrator();
        let ctx = madrid_ctx();
        llm.set_reply("NOT_RELEVANT");
        let mut search_ctx = SearchContext { web_context: "live page text".to_string(), ..Default::default() };

        assert!(orch.judge().evaluate_sources(&ctx, &mut search_ctx).is_empty());
        assert_eq!(search_ctx.web_context, "live page text");
        assert_eq!(llm.call_count(), 0);
    }

    /// Documents get a per-document judge in the fan-out — not re-judged.
    #[test]
    fn test_document_context_is_left_to_the_fanout_judge() {
        let (orch, llm) = make_orchestrator();
        let ctx = madrid_ctx();
        llm.set_reply("NOT_RELEVANT");
        let mut search_ctx = SearchContext { vector_context: "some document text".to_string(), ..Default::default() };

        assert!(orch.judge().evaluate_sources(&ctx, &mut search_ctx).is_empty());
        assert_eq!(search_ctx.vector_context, "some document text");
        assert_eq!(llm.call_count(), 0);
    }
}

// ------------------------------------------------------------------ #
// Eval-step placement in the DAG                                      #
// ------------------------------------------------------------------ #

mod eval_step_placement {
    use super::*;

    fn explore_step() -> PlanStep {
        PlanStep { step: 1, step_type: StepType::Explore, instruction: "explore".to_string(), ..Default::default() }
    }

    #[test]
    fn test_eval_step_runs_on_the_planned_step_path() {
        let (orch, _llm) = make_orchestrator();
        let ctx: SharedOrchestratorContext = SharedCell::wrap(madrid_ctx());
        let search_ctx: SharedSearchContext =
            SharedCell::wrap(SearchContext { workspace_context: "some workspace file".to_string(), ..Default::default() });

        let parent = LLMDecisionNode::new(&[]);
        let nodes = orch.pipeline().build_step_nodes(&ctx, &[explore_step()], &search_ctx, Some(&parent));

        // eval → explore → reason, with the eval step ahead of the step that
        // reads these sources off search_ctx.
        let names: Vec<&str> = nodes.iter().map(|e| class_name(&e.node)).collect();
        assert_eq!(names, vec!["LLMDecisionNode", "SearchNode", "TransformNode"]);
    }

    #[test]
    fn test_no_eval_step_when_there_is_nothing_to_judge() {
        let (orch, _llm) = make_orchestrator();
        let ctx: SharedOrchestratorContext = SharedCell::wrap(madrid_ctx());
        let search_ctx: SharedSearchContext = SharedCell::wrap(SearchContext::default());

        let parent = LLMDecisionNode::new(&[]);
        let nodes = orch.pipeline().build_step_nodes(&ctx, &[explore_step()], &search_ctx, Some(&parent));

        let names: Vec<&str> = nodes.iter().map(|e| class_name(&e.node)).collect();
        assert_eq!(names, vec!["SearchNode", "TransformNode"]);
    }

    /// A skill's own context is deliberate — the judge must not filter it.
    #[test]
    fn test_no_eval_step_when_a_skill_supplies_the_context() {
        let (orch, _llm) = make_orchestrator();
        let mut ctx = madrid_ctx();
        ctx.skill_context = Some("Follow this procedure".to_string());
        let search_ctx = SearchContext { workspace_context: "some workspace file".to_string(), ..Default::default() };

        assert!(!orch.judge().applies_to(&ctx, Some(&search_ctx)));
    }
}

// ------------------------------------------------------------------ #
// Per-step judging in the reason handler                              #
// ------------------------------------------------------------------ #

fn dep(node_id: &str, tool_name: &str, content: &str, is_tool: bool) -> (String, NodeResult) {
    (
        node_id.to_string(),
        NodeResult::with_data(
            node_id,
            json!({
                "tool_name": tool_name,
                "content": content,
                "raw": content,
                "is_tool_result": is_tool,
            }),
        ),
    )
}

/// Run the reason handler over `deps` with an empty search context.
fn reason(orch: &SLPOrchestrator, ctx: OrchestratorContext, deps: Vec<(String, NodeResult)>) -> NodeResult {
    let ctx: SharedOrchestratorContext = SharedCell::wrap(ctx);
    let search_ctx: SharedSearchContext = SharedCell::wrap(SearchContext::default());
    let handler = orch.pipeline().make_reason_handler(&ctx, &search_ctx, false);
    handler(deps.into_iter().collect::<HashMap<_, _>>()).unwrap()
}

fn contents(result: &NodeResult, key: &str) -> Vec<String> {
    result.data[key]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .map(|e| e["content"].as_str().unwrap_or("").to_string())
        .collect()
}

mod reason_handler_judging {
    use super::*;

    #[test]
    fn test_step_output_that_misses_the_intent_is_not_appended() {
        let (orch, llm) = make_orchestrator();
        verdicts(&llm, &[("OLD INVOICE", "NOT_RELEVANT")], "RELEVANT");

        let result = reason(
            &orch,
            madrid_ctx(),
            vec![
                dep("a", "explore", "Madrid: the Prado, Retiro Park", false),
                dep("b", "file-read", "OLD INVOICE for Adam Hart", false),
            ],
        );

        assert_eq!(contents(&result, "accumulated"), vec!["Madrid: the Prado, Retiro Park".to_string()]);
    }

    #[test]
    fn test_relevant_step_outputs_are_all_kept() {
        let (orch, llm) = make_orchestrator();
        llm.set_reply("RELEVANT");

        let result = reason(
            &orch,
            madrid_ctx(),
            vec![dep("a", "explore", "Madrid: the Prado", false), dep("b", "web_search", "Madrid: Retiro Park", false)],
        );

        assert_eq!(result.data["accumulated"].as_array().map(Vec::len), Some(2));
    }

    /// An all-reject verdict is far likelier to be a misfiring judge.
    #[test]
    fn test_rejecting_every_output_keeps_them_all() {
        let (orch, llm) = make_orchestrator();
        llm.set_reply("NOT_RELEVANT");

        let result = reason(
            &orch,
            madrid_ctx(),
            vec![dep("a", "explore", "some content", false), dep("b", "file-read", "other content", false)],
        );

        assert_eq!(result.data["accumulated"].as_array().map(Vec::len), Some(2));
    }

    #[test]
    fn test_tool_results_survive_the_accumulated_split() {
        let (orch, llm) = make_orchestrator();
        llm.set_reply("RELEVANT");

        let result = reason(
            &orch,
            madrid_ctx(),
            vec![dep("a", "explore", "Madrid: the Prado", false), dep("b", "get_weather", "18C and sunny", true)],
        );

        assert_eq!(contents(&result, "tool_results"), vec!["18C and sunny".to_string()]);
        let accumulated = result.data["accumulated"].as_array().cloned().unwrap_or_default();
        assert!(accumulated.iter().all(|e: &Value| e.get("_is_tool").is_none()));
    }

    /// The other half of the tool exemption: a TOOL step in the plan.
    ///
    /// Same reasoning as the live explore call — the plan resolved a tool,
    /// the server ran it, and the result is what the request went to get.
    #[test]
    fn test_a_tool_step_output_is_never_judged() {
        let (orch, llm) = make_orchestrator();
        llm.set_reply("NOT_RELEVANT");

        let result = reason(&orch, madrid_ctx(), vec![dep("a", "list_boards", "boards: Tickets, Roadmap", true)]);

        assert_eq!(contents(&result, "tool_results"), vec!["boards: Tickets, Roadmap".to_string()]);
    }

    /// The exemption is for tool output only — retrieval is still judged.
    #[test]
    fn test_a_rejected_retrieval_step_still_drops_alongside_a_tool_result() {
        let (orch, llm) = make_orchestrator();
        verdicts(&llm, &[("OLD INVOICE", "NOT_RELEVANT")], "RELEVANT");

        let result = reason(
            &orch,
            madrid_ctx(),
            vec![dep("a", "file-read", "OLD INVOICE for Adam Hart", false), dep("b", "get_weather", "18C and sunny", true)],
        );

        assert!(!contents(&result, "accumulated").contains(&"OLD INVOICE for Adam Hart".to_string()));
        assert_eq!(contents(&result, "tool_results"), vec!["18C and sunny".to_string()]);
    }

    /// A skill's own steps are its procedure — not candidates to filter.
    #[test]
    fn test_skill_context_bypasses_the_judge() {
        let (orch, llm) = make_orchestrator();
        let mut ctx = madrid_ctx();
        ctx.skill_context = Some("Follow this procedure".to_string());
        llm.set_reply("NOT_RELEVANT");

        let result = reason(&orch, ctx, vec![dep("a", "explore", "step output", false)]);

        assert_eq!(result.data["accumulated"].as_array().map(Vec::len), Some(1));
    }
}

// ------------------------------------------------------------------ #
// Document fan-out: all documents judged irrelevant                   #
// ------------------------------------------------------------------ #

mod doc_reduce_all_irrelevant {
    use super::*;

    /// Run the reduce handler over one batch result; returns the reduced
    /// search context (the result's payload).
    fn reduce(orch: &SLPOrchestrator, search_ctx: SearchContext, batch: NodeResult) -> SearchContext {
        let ctx: octx::SharedOrchestratorContext = SharedCell::wrap(madrid_ctx());
        let shared = std::sync::Arc::new(parking_lot::Mutex::new(search_ctx));
        let handler = orch.doc_fanout().make_reduce_handler(&ctx, shared, vec!["batch_0".to_string()]);
        let result = handler(HashMap::from([("batch_0".to_string(), batch)])).unwrap();
        result
            .payload_as::<smartloop::agents::orchestration::step::SharedSearchContext>()
            .map(|shared| shared.lock().clone())
            .or_else(|| result.payload_as::<SearchContext>().cloned())
            .expect("the reduced search context")
    }

    #[test]
    fn test_clears_document_context_when_every_document_is_dropped() {
        let (orch, _llm) = make_orchestrator();
        let search_ctx = SearchContext {
            vector_context: "RAW: Q3 invoice for Adam Hart".to_string(),
            citations: Some(vec![json!({"document_name": "invoice.pdf"}).as_object().cloned().unwrap()]),
            ..Default::default()
        };

        // An empty extract is the batch handler's "NONE" — the document was
        // judged irrelevant to the request.
        let reduced = reduce(
            &orch,
            search_ctx,
            NodeResult::with_data("batch_0", json!({"doc_names": ["invoice.pdf"], "content": ""})),
        );

        assert_eq!(reduced.vector_context, "");
        assert_eq!(reduced.citations, None);
        assert_eq!(reduced.cited_documents, None);
    }

    #[test]
    fn test_keeps_the_merged_extract_when_a_document_survives() {
        let (orch, _llm) = make_orchestrator();
        let search_ctx = SearchContext { vector_context: "RAW".to_string(), ..Default::default() };

        let reduced = reduce(
            &orch,
            search_ctx,
            NodeResult::with_data("batch_0", json!({"doc_names": ["madrid.pdf"], "content": "Visit the Prado"})),
        );

        assert!(reduced.vector_context.contains("Visit the Prado"));
        assert_eq!(reduced.cited_documents, Some(vec!["madrid.pdf".to_string()]));
    }
}
