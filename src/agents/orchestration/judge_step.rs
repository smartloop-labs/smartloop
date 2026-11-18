//! Port of `smartloop/agents/orchestration/judge_step.py`.
//!
//! JudgeStep — LLM-as-judge relevance evaluation of context before it is used.
//!
//! Retrieval is recall-oriented — a vector store returns its nearest
//! neighbours whether or not any of them is actually about the question — so
//! without a check an unrelated document reaches the prompt fenced as DOCUMENT
//! CONTEXT under instructions to answer from it.  The model then correctly
//! reports that it cannot ("I don't have context") instead of falling through
//! to what it knows.
//!
//! This module is that check.  It runs at two points: as a step in the DAG,
//! judging each retrieval source before any later step reads it, and inline in
//! the reason handler, judging each step's output before it is appended.
//!
//! It never gates the user's connected-app data.  Tool results — a live MCP
//! call during explore, a tool step's output in the plan — are exempt at both
//! points: they were fetched on purpose rather than returned by a similarity
//! search, so there is no recall noise to filter, and a wrong verdict throws
//! away the very data the request went to get.  See
//! [`JudgeStep::JUDGED_SOURCES`].

use std::collections::HashMap;
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use regex::Regex;
use serde_json::{Map, Value};

use crate::agents::mcp::explorer::SearchContext;
use crate::agents::nodes::{AgentNode, LLMDecisionNode, NodeHandler, NodeResult};
use crate::agents::orchestration::context::{OrchestratorContext, SharedOrchestratorContext};
use crate::agents::orchestration::model_router::{ModelRouter, DECISION_MAX_TOKENS};
use crate::agents::orchestration::step::{SharedSearchContext, Step, StepNodes};

/// Python logger name of this module.
pub const LOGGER: &str = "smartloop.agents.orchestration.judge_step";

/// Node id of the judge step's result.
pub const JUDGE_STEP_NODE: &str = "judge_step";

/// A verdict token, matched whole: `RELEVANT` inside `NOT_RELEVANT` is not a
/// separate word, and the alternation puts the negative first so the space-
/// separated spelling is read as one token rather than as a bare `RELEVANT`.
pub static VERDICT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b(NOT[_ ]RELEVANT|RELEVANT)\b").unwrap());

/// Leading decoration a small model puts in front of its answer: markdown
/// emphasis, a bullet, a "Verdict:" colon.  Stripped so a verdict that opens
/// the reply is still recognised as opening it.
pub const VERDICT_PREFIX: &str = "*_#>-—: \t\n\"'`";

/// How much of an undecided or rejecting reply is logged.  Long enough to show
/// the model's own words, short enough not to dump a paragraph per request.
pub const REPLY_LOG_CHARS: usize = 200;

/// The first `n` characters of `text` (Python `text[:n]`).
fn char_prefix(text: &str, n: usize) -> &str {
    match text.char_indices().nth(n) {
        Some((idx, _)) => &text[..idx],
        None => text,
    }
}

/// The text of one judged [`SearchContext`] field (Python
/// `getattr(search_ctx, attr, "")`).
fn source_text<'a>(search_ctx: &'a SearchContext, attr: &str) -> &'a str {
    match attr {
        "workspace_context" => &search_ctx.workspace_context,
        "feedback_context" => &search_ctx.feedback_context,
        "vector_context" => &search_ctx.vector_context,
        "web_context" => &search_ctx.web_context,
        "mcp_context" => &search_ctx.mcp_context,
        _ => "",
    }
}

/// Clear one judged [`SearchContext`] field (Python
/// `setattr(search_ctx, attr, "")`).
fn clear_source(search_ctx: &mut SearchContext, attr: &str) {
    match attr {
        "workspace_context" => {
            search_ctx.workspace_context.clear();
            search_ctx.workspace_citations.clear();
        }
        "feedback_context" => search_ctx.feedback_context.clear(),
        "vector_context" => search_ctx.vector_context.clear(),
        "web_context" => search_ctx.web_context.clear(),
        "mcp_context" => {
            search_ctx.mcp_context.clear();
            search_ctx.tool_citations.clear();
        }
        _ => {}
    }
}

/// `Some(true)`/`Some(false)` for a decisive verdict, `None` when there isn't
/// one.
///
/// The check used to be `"NOT_RELEVANT" in reply`, which reads any reply
/// *mentioning* the word as a rejection — including the two things a small
/// model most often says instead of answering: it quotes the instruction back
/// ("Reply RELEVANT or NOT_RELEVANT"), or it answers in prose ("the context is
/// RELEVANT, not NOT_RELEVANT").  Both dropped the context.
///
/// So: whole tokens, and a reply that names *both* labels without leading with
/// one is treated as no verdict at all.  A verdict that opens the reply is
/// still decisive when the reasoning after it names the other label, which is
/// the normal shape of a well-formed answer ("NOT_RELEVANT — this is about
/// Trello, not relevant to the question").
pub fn parse_verdict(reply: &str) -> Option<bool> {
    let upper = reply.trim().to_uppercase();
    let text = upper.trim_start_matches(|c: char| VERDICT_PREFIX.contains(c)).trim();
    let matches: Vec<(usize, String)> = VERDICT_RE
        .captures_iter(text)
        .filter_map(|caps| caps.get(1))
        .map(|m| (m.start(), m.as_str().replace(' ', "_")))
        .collect();
    let (first_start, first_label) = matches.first()?;
    if *first_start == 0 {
        return Some(first_label != "NOT_RELEVANT");
    }
    let has_negative = matches.iter().any(|(_, label)| label == "NOT_RELEVANT");
    let has_positive = matches.iter().any(|(_, label)| label == "RELEVANT");
    if has_negative && has_positive {
        return None;
    }
    Some(!has_negative)
}

/// Judges whether a piece of context serves the user's request.
///
/// One decision-model call per distinct piece of context, memoized for the
/// life of a request.  Every verdict fails open: an LLM error, an empty reply
/// or an unparseable answer all keep the context, because dropping real
/// context is the worse failure and this runs on the small model.
///
/// Cheap to clone (shared state behind `Arc`), so DAG handlers capture a clone.
#[derive(Clone)]
pub struct JudgeStep {
    models: Arc<ModelRouter>,
    /// Verdicts memoized for the life of a request, keyed by `(label, text)`:
    /// the same piece of context is judged at more than one point in the DAG
    /// (the eval step, then again as a step output), so this keeps each piece
    /// to one call.
    judge_cache: Arc<Mutex<HashMap<(String, String), bool>>>,
}

impl JudgeStep {
    /// How much of a piece of context the judge reads.  The opening of a
    /// retrieved chunk is what decides its subject; feeding the whole thing
    /// would blow the decision model's window for no extra signal.
    pub const JUDGE_SAMPLE_CHARS: usize = 2000;

    /// Retrieval sources judged individually before they reach the answer
    /// context, as `(SearchContext field, human label)`.
    ///
    /// What belongs here is *retrieval*: content a similarity search handed us
    /// because it was nearest, not because it was asked for.
    ///
    /// `mcp_context` is deliberately absent — the user's connected-app data is
    /// never gated by this judge.  It is a live tool result, and a tool call is
    /// not recall: something decided to invoke it, the server ran it, and what
    /// came back is the answer to that call.  A tool that should not have been
    /// called is a routing problem, and gating its output cannot fix it.
    ///
    /// `vector_context` is deliberately absent: retrieved documents already
    /// get their own per-document judge in the fan-out, where each one is
    /// extracted against the request and dropped on a `NONE` reply (see
    /// [`crate::agents::orchestration::document_fanout::DocumentFanoutStep`]).
    ///
    /// `web_context` is absent for a different reason: it was fetched for THIS
    /// query moments ago, so it is on-topic by construction.
    pub const JUDGED_SOURCES: &'static [(&'static str, &'static str)] = &[
        ("workspace_context", "workspace files"),
        ("feedback_context", "feedback"),
    ];

    /// Step name.
    pub const NAME: &'static str = "judge";

    /// Build the judge over the shared model router.
    pub fn new(models: Arc<ModelRouter>) -> Self {
        Self { models, judge_cache: Arc::new(Mutex::new(HashMap::new())) }
    }

    /// True when `content` is worth appending to the answer context.
    ///
    /// Retrieval is recall-oriented, so an unrelated document reaches the
    /// prompt fenced as DOCUMENT CONTEXT under instructions to answer from it.
    /// This judge is the check that keeps such a chunk out.
    ///
    /// Fails open: an LLM error, an empty reply or an unparseable verdict all
    /// keep the context.  Dropping real context is the worse failure, and the
    /// judge runs on the small decision model.
    ///
    /// Verdicts are memoized per request, so a piece of context judged at the
    /// retrieval gate is not re-judged when it resurfaces as a step output.
    pub fn serves_intent(&self, user_message: &str, label: &str, content: &str) -> bool {
        let text = content.trim();
        if text.is_empty() {
            return false;
        }

        let key = (label.to_string(), text.to_string());
        if let Some(cached) = self.judge_cache.lock().get(&key) {
            return *cached;
        }

        let sample = char_prefix(text, Self::JUDGE_SAMPLE_CHARS);
        let prompt = format!(
            "<system>
You are a relevance judge in a retrieval pipeline. One piece of retrieved
context is shown below. Decide whether it serves the user's request.

Reply with exactly one word:
RELEVANT — it contains information that helps answer the request.
NOT_RELEVANT — it is about a different subject and would not help.

Judge the subject matter, not the wording. Context that merely shares a few
words with the request but is about something else is NOT_RELEVANT.
</system>
<user>
User request: {user_message}

Retrieved context ({label}):
{sample}

Does this context serve the user's request? Reply RELEVANT or NOT_RELEVANT.
</user>"
        );

        let mut serves = true;
        let mut reply = String::new();
        let agent_llm = self.models.agent_llm.clone();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            agent_llm.invoke(&prompt, Some(DECISION_MAX_TOKENS), None)
        }));
        match outcome {
            Ok(raw) => {
                reply = raw.trim().to_string();
                match parse_verdict(&reply) {
                    None => {
                        // No verdict in the reply.  Kept — but said out loud,
                        // because a judge that cannot answer its own prompt is
                        // a thing to know about rather than a silent pass.
                        let shown = char_prefix(&reply, REPLY_LOG_CHARS);
                        tracing::info!(
                            target: LOGGER,
                            "[judge] {}: no verdict in reply ({}) — keeping context",
                            label,
                            if shown.is_empty() { "empty" } else { shown }
                        );
                    }
                    Some(verdict) => serves = verdict,
                }
            }
            Err(_) => {
                tracing::debug!(
                    target: LOGGER,
                    "[judge] {}: LLM call failed — keeping context",
                    label
                );
            }
        }

        if !serves {
            // The reply travels with the decision: a misjudging model and a
            // misread verdict produce the same log line without it.
            tracing::info!(
                target: LOGGER,
                "[judge] {} ({} chars) does not serve the request — dropped (reply: {})",
                label,
                text.chars().count(),
                char_prefix(&reply, REPLY_LOG_CHARS)
            );
        }
        self.judge_cache.lock().insert(key, serves);
        serves
    }

    /// Judge each retrieval source on its own, clearing the ones that don't
    /// serve the request.  Returns the labels dropped.
    ///
    /// Judged per source rather than over the concatenation: one verdict for
    /// the whole blob meant a single relevant source kept three irrelevant
    /// ones alive, and a single irrelevant one could sink the rest.
    pub fn evaluate_sources(
        &self,
        ctx: &OrchestratorContext,
        search_ctx: &mut SearchContext,
    ) -> Vec<String> {
        let mut dropped: Vec<String> = Vec::new();
        for (attr, label) in Self::JUDGED_SOURCES {
            let text = source_text(search_ctx, attr).to_string();
            if text.is_empty() {
                continue;
            }
            if self.serves_intent(&ctx.user_message, label, &text) {
                continue;
            }
            clear_source(search_ctx, attr);
            dropped.push((*label).to_string());
        }
        self.announce_dropped(&dropped);
        dropped
    }

    /// [`Self::evaluate_sources`] over the shared search context, without
    /// holding its lock across the LLM calls.
    fn evaluate_shared_sources(
        &self,
        user_message: &str,
        search_ctx: &SharedSearchContext,
    ) -> Vec<String> {
        let mut dropped: Vec<String> = Vec::new();
        for (attr, label) in Self::JUDGED_SOURCES {
            let text = source_text(&search_ctx.lock(), attr).to_string();
            if text.is_empty() {
                continue;
            }
            if self.serves_intent(user_message, label, &text) {
                continue;
            }
            clear_source(&mut search_ctx.lock(), attr);
            dropped.push((*label).to_string());
        }
        self.announce_dropped(&dropped);
        dropped
    }

    /// Tell the client which sources were put aside, if any.
    fn announce_dropped(&self, dropped: &[String]) {
        if !dropped.is_empty() {
            self.models.emit("filter", "processing", &Self::set_aside_message(dropped));
        }
    }

    /// Say what was put aside, in the voice the user is reading in.
    ///
    /// "Set aside context that does not match the request: MCP content" reads
    /// like a filter's own log line — a colon-delimited list of internal
    /// source names.  The person watching is waiting on an answer, not
    /// auditing the pipeline, so it names the sources in a sentence and says
    /// why, which is also the part that keeps a dropped source from looking
    /// lost.
    pub fn set_aside_message(dropped: &[String]) -> String {
        let (items, pronoun) = match dropped {
            [only] => (only.clone(), "it"),
            [rest @ .., last] => (format!("{} and {}", rest.join(", "), last), "they"),
            [] => (String::new(), "they"),
        };
        format!("Set aside {items} — {pronoun} didn't seem relevant to your question")
    }

    /// True when at least one judged retrieval source has content.
    pub fn has_judgeable_sources(&self, search_ctx: &SearchContext) -> bool {
        Self::JUDGED_SOURCES.iter().any(|(attr, _)| !source_text(search_ctx, attr).is_empty())
    }

    /// Create the handler that judges and clears irrelevant sources.
    pub fn make_handler(
        &self,
        ctx: &SharedOrchestratorContext,
        search_ctx: SharedSearchContext,
    ) -> NodeHandler {
        let judge = self.clone();
        let ctx = ctx.clone();
        Arc::new(move |_deps: HashMap<String, NodeResult>| {
            let user_message = ctx.read().user_message.clone();
            judge.evaluate_shared_sources(&user_message, &search_ctx);
            Ok(NodeResult::with_data(JUDGE_STEP_NODE, Value::Object(Map::new())))
        })
    }
}

impl Step for JudgeStep {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    /// Clear memoized verdicts — they are scoped to one user request.
    fn begin_request(&self) {
        self.judge_cache.lock().clear();
    }

    /// True when there is retrieved context worth judging.
    ///
    /// False when a matched skill supplies its own context deliberately and
    /// must not have it filtered out from under it, or when no judged source
    /// has content — keeping the judge off the critical path of requests it
    /// could not affect.
    fn applies_to(&self, ctx: &OrchestratorContext, search_ctx: Option<&SearchContext>) -> bool {
        if ctx.skill_context.as_deref().is_some_and(|s| !s.is_empty()) {
            return false;
        }
        search_ctx.is_some_and(|sc| self.has_judgeable_sources(sc))
    }

    /// Build the single node that judges every retrieval source.
    fn build(
        &self,
        ctx: &SharedOrchestratorContext,
        search_ctx: Option<&SharedSearchContext>,
        parent: &AgentNode,
    ) -> StepNodes {
        let node = LLMDecisionNode::new(&[parent]).labelled("judge");
        // With no search context there is nothing to judge; the node still
        // runs so the graph keeps the shape the caller asked for.
        let search_ctx = search_ctx.cloned().unwrap_or_default();
        let handler = self.make_handler(ctx, search_ctx);
        StepNodes::new(vec![(node.clone(), handler)], node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::orchestration::model_router::test_llm::ScriptedLlm;
    use crate::agents::orchestration::model_router::ModelRouterOptions;

    fn judge(reply: &'static str) -> (JudgeStep, Arc<ScriptedLlm>) {
        let llm = ScriptedLlm::new(move |_| reply.to_string());
        let router = ModelRouter::new(llm.clone(), llm.clone(), ModelRouterOptions::default());
        (JudgeStep::new(Arc::new(router)), llm)
    }

    #[test]
    fn verdicts_parse_as_whole_tokens() {
        assert_eq!(parse_verdict("RELEVANT"), Some(true));
        assert_eq!(parse_verdict("NOT_RELEVANT"), Some(false));
        assert_eq!(parse_verdict("not relevant"), Some(false));
        assert_eq!(parse_verdict("**NOT_RELEVANT**"), Some(false));
        assert_eq!(parse_verdict("Verdict: RELEVANT"), Some(true));
        assert_eq!(parse_verdict("Reply RELEVANT or NOT_RELEVANT"), None);
        assert_eq!(parse_verdict("The context is RELEVANT, not NOT_RELEVANT."), None);
        assert_eq!(
            parse_verdict("NOT_RELEVANT — this is about Trello, not relevant to the question"),
            Some(false)
        );
        assert_eq!(parse_verdict("hmm, maybe?"), None);
        assert_eq!(parse_verdict(""), None);
        assert_eq!(parse_verdict("it is NOT_RELEVANT"), Some(false));
        assert_eq!(parse_verdict("it is relevant"), Some(true));
    }

    #[test]
    fn serves_intent_follows_the_verdict_and_memoizes() {
        let (judge, llm) = judge("NOT_RELEVANT");
        for _ in 0..3 {
            assert!(!judge.serves_intent("madrid", "documents", "invoice"));
        }
        assert_eq!(llm.calls(), 1);
        judge.begin_request();
        assert!(!judge.serves_intent("madrid", "documents", "invoice"));
        assert_eq!(llm.calls(), 2);
        assert!(!judge.serves_intent("madrid", "documents", "   "));
        assert_eq!(llm.calls(), 2);
    }

    #[test]
    fn llm_failure_and_echoes_keep_the_context() {
        let (judge, _) = judge("Reply RELEVANT or NOT_RELEVANT");
        assert!(judge.serves_intent("monday boards", "MCP content", "boards: Tickets"));
        let llm = ScriptedLlm::new(|_| panic!("no model"));
        let router = ModelRouter::new(llm.clone(), llm, ModelRouterOptions::default());
        let judge = JudgeStep::new(Arc::new(router));
        assert!(judge.serves_intent("madrid", "documents", "text"));
    }

    #[test]
    fn evaluate_sources_clears_only_judged_sources() {
        let (judge, llm) = judge("NOT_RELEVANT");
        let ctx = OrchestratorContext::new("madrid", vec![], vec![], "");
        let mut sc = SearchContext {
            workspace_context: "ws".into(),
            feedback_context: "fb".into(),
            mcp_context: "unrelated".into(),
            web_context: "web".into(),
            vector_context: "docs".into(),
            ..Default::default()
        };
        let dropped = judge.evaluate_sources(&ctx, &mut sc);
        assert_eq!(dropped, vec!["workspace files".to_string(), "feedback".to_string()]);
        assert_eq!(sc.workspace_context, "");
        assert_eq!(sc.feedback_context, "");
        assert_eq!(sc.mcp_context, "unrelated");
        assert_eq!(sc.web_context, "web");
        assert_eq!(sc.vector_context, "docs");
        assert_eq!(llm.calls(), 2);
    }

    #[test]
    fn the_handler_clears_the_shared_search_context() {
        let (judge, _) = judge("NOT_RELEVANT");
        let ctx: SharedOrchestratorContext = Arc::new(parking_lot::RwLock::new(
            OrchestratorContext::new("madrid", vec![], vec![], ""),
        ));
        let sc: SharedSearchContext = Arc::new(Mutex::new(SearchContext {
            workspace_context: "ws".into(),
            ..Default::default()
        }));
        assert!(judge.applies_to(&ctx.read(), Some(&sc.lock())));
        let parent = AgentNode::default();
        let built = judge.build(&ctx, Some(&sc), &parent);
        assert_eq!(built.pairs.len(), 1);
        assert!(built.parent.is_llm_decision());
        let result = (built.pairs[0].1)(HashMap::new()).unwrap();
        assert_eq!(result.node_id, JUDGE_STEP_NODE);
        assert_eq!(sc.lock().workspace_context, "");
    }

    #[test]
    fn a_matched_skill_turns_the_judge_off() {
        let (judge, _) = judge("RELEVANT");
        let mut ctx = OrchestratorContext::new("q", vec![], vec![], "");
        ctx.skill_context = Some("skill".into());
        let sc = SearchContext { workspace_context: "ws".into(), ..Default::default() };
        assert!(!judge.applies_to(&ctx, Some(&sc)));
        ctx.skill_context = None;
        assert!(judge.applies_to(&ctx, Some(&sc)));
        assert!(!judge.applies_to(&ctx, Some(&SearchContext::default())));
    }

    #[test]
    fn set_aside_reads_as_a_sentence() {
        assert_eq!(
            JudgeStep::set_aside_message(&["feedback".to_string()]),
            "Set aside feedback — it didn't seem relevant to your question"
        );
        assert_eq!(
            JudgeStep::set_aside_message(&["a".into(), "b".into(), "c".into()]),
            "Set aside a, b and c — they didn't seem relevant to your question"
        );
    }
}
