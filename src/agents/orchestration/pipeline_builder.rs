//! Building the execution DAG for a request.
//!
//! Port of `smartloop/agents/orchestration/pipeline_builder.py`.
//!
//! Every request runs the same shape: collect what is already indexed, classify
//! the intent, plan the steps, then wire one node per step into a graph that ends
//! at a single reason node.  Whether a skill matched changes what the planner is
//! told, not which path is taken — a matched skill contributes guiding context,
//! not its own control flow.
//!
//! The graph is built lazily.  The route node cannot know what steps exist until
//! classification has run, so it returns its downstream nodes as `_extend_dag`
//! (here [`NodeResult::extend_dag`]) and the executor grafts them on mid-walk.
//!
//! ## The shared search context
//!
//! Python threaded one mutable `SearchContext` object through every handler:
//! the document reduce node rewrote its `vector_context`, the judge cleared the
//! sources that did not serve the request, a WEB_SEARCH step appended its
//! sources, and the EXPLORE step and reason node read the result.  Here that
//! object is a [`SharedSearchContext`] created by the route handler and
//! captured by every handler built from it.  Node results carry it as a typed
//! payload:
//!
//! * collect → a [`SearchContext`] (what was gathered),
//! * route → the live [`SharedSearchContext`],
//! * reason → a [`SearchContext`] snapshot of the final state (what the
//!   response builder reads).
//!
//! A step contributed ahead of the plan (the document fan-out) hands its
//! updated context downstream as a [`SearchContext`] payload; the first node
//! built after it adopts that payload into the shared context.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use regex::Regex;
use serde_json::{json, Map, Value};

use crate::agents::fetch_agent::UrlFetcher;
use crate::agents::hybrid_executor::LLMExecutor;
use crate::agents::mcp::explorer::{
    EnrichedContext, Explorer, ExplorerSearchService, ProgressCallback,
    SkillDict as ExplorerSkillDict, SearchContext,
};
use crate::agents::mcp::mcp_agent::{format_call, MCPAgent};
use crate::agents::mcp::mcp_tool_caller::MCPToolCaller;
use crate::agents::nodes::{
    AgentNode, AgentNodeName, DagExtension, LLMDecisionNode, NodeHandler, NodeResult, SearchNode,
    ToolCallNode, TransformNode,
};
pub use crate::agents::orchestration::context::SharedOrchestratorContext;
use crate::agents::orchestration::context::OrchestratorContext;
use crate::agents::orchestration::document_fanout::DocumentFanoutStep;
use crate::agents::orchestration::judge_step::{JudgeStep, JUDGE_STEP_NODE};
use crate::agents::orchestration::model_router::ModelRouter;
use crate::agents::orchestration::replies::is_placeholder_output;
use crate::agents::orchestration::response_builder::ResponseBuilder;
use crate::agents::orchestration::skill_processor::{
    char_prefix, py_repr, try_call, truthy, SkillDict, SkillProcessor,
};
use crate::agents::orchestration::step::Step;
use crate::agents::orchestration::step_executor::{plan_step_result, StepExecutor};
use crate::agents::orchestration::tool_fanout::ToolFanout;
use crate::agents::planner::{ClassifyOptions, IntentType, ModelTier, PlanStep, Planner, StepType};
use crate::agents::web_search_agent::WebSearcher;
use crate::llms::base_llm::BaseLLM;
use crate::types::ModelCapability;

/// Python logger name.
const LOG_TARGET: &str = "smartloop.agents.orchestration.pipeline_builder";

/// Node id of the final reason node.
pub const REASON_NODE: AgentNodeName = AgentNodeName::Reason;

/// Shared, mutable search context captured by node handlers (a WEB_SEARCH
/// step appends its sources to it, the judge clears sources from it).
pub type SharedSearchContext = Arc<Mutex<SearchContext>>;

/// A URL in a step instruction.
static URL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"https?://[^\s<>'")\]]+"#).unwrap());

/// `Some(s)` unless `s` is empty (Python truthiness of an optional string).
fn non_empty(s: &str) -> Option<&str> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// `step.tool_name or step.step_type.value`.
fn step_label(step: &PlanStep) -> String {
    match step.tool_name.as_deref() {
        Some(t) if !t.is_empty() => t.to_string(),
        _ => step.step_type.value().to_string(),
    }
}

/// Python `f"{value}"` of an optional string (`None` prints as `None`).
fn or_none(value: Option<&str>) -> &str {
    value.unwrap_or("None")
}

/// `f"/{capability}" if capability else ""`.
fn capability_suffix(capability: Option<&str>) -> String {
    match capability {
        Some(c) if !c.is_empty() => format!("/{c}"),
        _ => String::new(),
    }
}

/// `deps` in a stable order: the ids listed in `order` first (the node's
/// declared dependency order, which Python's dict preserved), then any others
/// by id.
fn ordered_deps<'a>(
    deps: &'a HashMap<String, NodeResult>,
    order: &[String],
) -> Vec<(&'a String, &'a NodeResult)> {
    let mut out: Vec<(&String, &NodeResult)> = Vec::with_capacity(deps.len());
    let mut seen: HashSet<&str> = HashSet::new();
    for id in order {
        if let Some((k, v)) = deps.get_key_value(id) {
            if seen.insert(k.as_str()) {
                out.push((k, v));
            }
        }
    }
    let mut rest: Vec<(&String, &NodeResult)> =
        deps.iter().filter(|(k, _)| !seen.contains(k.as_str())).collect();
    rest.sort_by(|a, b| a.0.cmp(b.0));
    out.extend(rest);
    out
}

/// String field of a node result's data (`data.get(key, "")`).
fn data_str(data: &Value, key: &str) -> String {
    match data.get(key) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

/// Parsed skills from a skill-match result's `data["matched_skills"]`.
fn matched_skills_from(result: Option<&NodeResult>) -> Option<Vec<SkillDict>> {
    let list = result?.data.get("matched_skills")?.as_array()?;
    Some(list.iter().filter_map(|v| v.as_object().cloned()).collect())
}

/// The explorer's string-only view of the matched skills.
fn explorer_skills(skills: &[SkillDict]) -> Vec<ExplorerSkillDict> {
    skills
        .iter()
        .map(|s| {
            s.iter()
                .filter_map(|(k, v)| v.as_str().map(|t| (k.clone(), t.to_string())))
                .collect()
        })
        .collect()
}

/// The search context a node result carries, if any.
fn search_ctx_from(result: &NodeResult) -> Option<SearchContext> {
    if let Some(sc) = result.payload_as::<SearchContext>() {
        return Some(sc.clone());
    }
    if let Some(shared) = result.payload_as::<SharedSearchContext>() {
        return Some(shared.lock().clone());
    }
    result
        .data
        .get("search_ctx")
        .filter(|v| v.is_object())
        .and_then(|v| serde_json::from_value(v.clone()).ok())
}

/// Adopt a search context handed downstream by a dependency (the document
/// reduce node returns its updated context as a payload) into `shared`.
/// The live shared context itself is never re-adopted.
fn adopt_search_ctx(deps: &HashMap<String, NodeResult>, shared: &SharedSearchContext) {
    let mut keys: Vec<&String> = deps.keys().collect();
    keys.sort();
    for key in keys {
        let result = &deps[key];
        if let Some(sc) = result.payload_as::<SearchContext>() {
            *shared.lock() = sc.clone();
        } else if let Some(other) = result.payload_as::<SharedSearchContext>() {
            if !Arc::ptr_eq(other, shared) {
                let snapshot = other.lock().clone();
                *shared.lock() = snapshot;
            }
        }
    }
}

/// Wrap `handler` so that, the first time any handler sharing `once` runs, it
/// adopts an upstream search context into `shared` before running.
fn adopting(handler: NodeHandler, shared: SharedSearchContext, once: Arc<Mutex<bool>>) -> NodeHandler {
    Arc::new(move |deps: HashMap<String, NodeResult>| {
        {
            let mut done = once.lock();
            if !*done {
                adopt_search_ctx(&deps, &shared);
                *done = true;
            }
        }
        handler(deps)
    })
}

/// A skip marker from a resolve node.
fn skip_result(node_id: &str, tool_name: String) -> NodeResult {
    NodeResult::with_data(node_id, json!({ "skip": true, "tool_name": tool_name }))
}

/// Turns a request into the DAG of nodes that answers it.
pub struct PipelineBuilder {
    pub(crate) models: Arc<ModelRouter>,
    pub planner: Arc<Planner>,
    pub explorer: Arc<Explorer>,
    pub(crate) executor: Arc<LLMExecutor>,
    pub tool_caller: Arc<MCPToolCaller>,
    /// The HTTP fetcher (replaceable by the orchestrator).
    fetch_agent: Mutex<Arc<dyn UrlFetcher>>,
    /// The web searcher (replaceable by the orchestrator).
    web_search_agent: Mutex<Arc<dyn WebSearcher>>,
    pub(crate) judge: Arc<JudgeStep>,
    pub(crate) doc_fanout: Arc<DocumentFanoutStep>,
    pub(crate) skills: Arc<SkillProcessor>,
    pub(crate) steps: Arc<StepExecutor>,
    pub(crate) tool_fanout: Arc<ToolFanout>,
    /// Held for parity with Python; its helpers used here are associated
    /// functions.
    #[allow(dead_code)]
    pub(crate) response: Arc<ResponseBuilder>,
}

impl PipelineBuilder {
    /// Ceiling on items per fan-out (kept for parity with the Python class).
    pub const FANOUT_MAX_ITEMS: usize = 20;

    /// Wire the builder to the services it coordinates.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        models: Arc<ModelRouter>,
        planner: Arc<Planner>,
        explorer: Arc<Explorer>,
        executor: Arc<LLMExecutor>,
        tool_caller: Arc<MCPToolCaller>,
        fetch_agent: Arc<dyn UrlFetcher>,
        web_search_agent: Arc<dyn WebSearcher>,
        judge: Arc<JudgeStep>,
        doc_fanout: Arc<DocumentFanoutStep>,
        skills: Arc<SkillProcessor>,
        steps: Arc<StepExecutor>,
        tool_fanout: Arc<ToolFanout>,
        response: Arc<ResponseBuilder>,
    ) -> Self {
        Self {
            models,
            planner,
            explorer,
            executor,
            tool_caller,
            fetch_agent: Mutex::new(fetch_agent),
            web_search_agent: Mutex::new(web_search_agent),
            judge,
            doc_fanout,
            skills,
            steps,
            tool_fanout,
            response,
        }
    }

    /// The decision model, owned by the router.
    pub fn agent_llm(&self) -> Arc<dyn BaseLLM> {
        self.models.agent_llm.clone()
    }

    /// Current HTTP fetcher.
    pub fn get_fetch_agent(&self) -> Arc<dyn UrlFetcher> {
        self.fetch_agent.lock().clone()
    }

    /// Replace the HTTP fetcher (Python: `orchestrator._fetch_agent = ...`).
    pub fn set_fetch_agent(&self, agent: Arc<dyn UrlFetcher>) {
        *self.fetch_agent.lock() = agent;
    }

    /// Current web searcher.
    pub fn get_web_search_agent(&self) -> Arc<dyn WebSearcher> {
        self.web_search_agent.lock().clone()
    }

    /// Replace the web searcher (Python: `orchestrator._web_search_agent = ...`).
    pub fn set_web_search_agent(&self, agent: Arc<dyn WebSearcher>) {
        *self.web_search_agent.lock() = agent;
    }

    /// `progress(step, status, message)` forwarding to the router's `emit`.
    fn progress(&self) -> ProgressCallback {
        let models = Arc::clone(&self.models);
        Arc::new(move |step: &str, status: &str, message: &str| models.emit(step, status, message))
    }

    /// Vector + workspace + feedback search.
    ///
    /// Returns a result whose payload is the gathered [`SearchContext`]
    /// (Python `data={"search_ctx": search_ctx}`; the JSON form is also kept
    /// under `data["search_ctx"]`).
    pub fn handle_collect(
        &self,
        ctx: &SharedOrchestratorContext,
        deps: &HashMap<String, NodeResult>,
        node: &AgentNode,
        skill_match_node: &AgentNode,
    ) -> NodeResult {
        tracing::info!(
            target: LOG_TARGET,
            "[models] COLLECT → no LLM (vector/feedback/workspace search)"
        );
        let matched_skills = matched_skills_from(deps.get(&skill_match_node.node_id));
        let matched: Option<Vec<ExplorerSkillDict>> = matched_skills
            .as_deref()
            .filter(|m| !m.is_empty())
            .map(explorer_skills);
        let c = ctx.read().clone();
        let model_service: Option<Arc<dyn ExplorerSearchService>> = c
            .model_service
            .clone()
            .map(|m| m as Arc<dyn ExplorerSearchService>);
        let search_ctx = self.explorer.collect_parallel(
            &c.user_message,
            model_service,
            non_empty(&c.effective_cwd),
            c.enriched.as_ref(),
            matched.as_deref(),
            Some(c.mcp_servers.as_slice()).filter(|s| !s.is_empty()),
            Some(c.mcp_tool_contexts.as_slice()).filter(|s| !s.is_empty()),
            c.conversation_history.as_deref(),
            Some(self.executor.io_pool()),
            c.project_dir.as_deref(),
            Some(self.progress()),
            // The live MCP call made here resolves its tool the same way a
            // planned step does, so it gets the same reasoning model.
            Some(self.models.reasoning_llm(self.agent_llm())),
        );
        let data = json!({
            "search_ctx": serde_json::to_value(&search_ctx).unwrap_or(Value::Null),
        });
        NodeResult::with_data(node.node_id.clone(), data).with_payload(search_ctx)
    }

    /// Decision: build the unified pipeline. Dynamically adds downstream nodes
    /// (returned in [`NodeResult::extend_dag`]).  The result's payload is the
    /// live [`SharedSearchContext`] every downstream handler shares.
    pub fn handle_route(
        self: &Arc<Self>,
        ctx: &SharedOrchestratorContext,
        deps: &HashMap<String, NodeResult>,
        node: &AgentNode,
        skill_match_node: &AgentNode,
        collect_node: &AgentNode,
    ) -> NodeResult {
        tracing::info!(
            target: LOG_TARGET,
            "[models] ROUTE (classify+plan) → agent_llm: {}",
            ModelRouter::model_label(&*self.agent_llm())
        );
        let matched_skills = matched_skills_from(deps.get(&skill_match_node.node_id));
        let search_ctx: SearchContext = deps
            .get(&collect_node.node_id)
            .and_then(search_ctx_from)
            .unwrap_or_default();
        let enrich_ctx: EnrichedContext = ctx.read().enriched.clone().unwrap_or_default();
        let skill_names = matched_skills
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .filter_map(|s| s.get("name").and_then(Value::as_str))
            .filter(|n| !n.is_empty())
            .collect::<Vec<_>>()
            .join(", ");
        self.models.emit(
            "plan",
            "processing",
            &if !skill_names.is_empty() {
                format!("Planning the response using skill(s): {skill_names}")
            } else {
                "Planning the response and selecting tools".to_string()
            },
        );

        if let Some(matched) = matched_skills.as_deref().filter(|m| !m.is_empty()) {
            let bodies: Vec<String> = matched
                .iter()
                .map(|s| {
                    SkillProcessor::strip_frontmatter(
                        s.get("content").and_then(Value::as_str).unwrap_or(""),
                    )
                })
                .collect();
            let mut c = ctx.write();
            c.skill_context = Some(bodies.join("\n\n"));
            c.skill_bodies = bodies;
        }

        let mut extend_dag: Vec<DagExtension> = Vec::new();
        let mut parent: AgentNode = node.clone();

        let applies = {
            let c = ctx.read();
            self.doc_fanout.applies_to(&c, Some(&search_ctx))
        };
        if applies {
            let contributed = self.doc_fanout.build(ctx, Some(&search_ctx), &parent);
            extend_dag.extend(
                contributed.pairs.into_iter().map(|(n, h)| DagExtension::new(n, h)),
            );
            parent = contributed.parent;
        }

        let shared: SharedSearchContext = Arc::new(Mutex::new(search_ctx));
        extend_dag.extend(self.build_unified_pipeline(
            ctx,
            matched_skills.as_deref(),
            &shared,
            &enrich_ctx,
            Some(&parent),
        ));

        let data = json!({
            "search_ctx": serde_json::to_value(&*shared.lock()).unwrap_or(Value::Null),
            "enrich_ctx": serde_json::to_value(&enrich_ctx).unwrap_or(Value::Null),
        });
        let mut result = NodeResult::with_data(node.node_id.clone(), data).with_payload(shared);
        result.extend_dag = extend_dag;
        result
    }

    /// Build a single pipeline regardless of whether skills matched.
    ///
    /// Both paths produce a uniform `Vec<PlanStep>` that feeds into the same
    /// node builders and handlers — no branching downstream.
    pub fn build_unified_pipeline(
        self: &Arc<Self>,
        ctx: &SharedOrchestratorContext,
        _matched_skills: Option<&[SkillDict]>,
        search_ctx: &SharedSearchContext,
        enrich_ctx: &EnrichedContext,
        parent: Option<&AgentNode>,
    ) -> Vec<DagExtension> {
        let plan_steps = self.derive_steps(ctx, search_ctx, enrich_ctx);
        if !plan_steps.is_empty() {
            tracing::info!(
                target: LOG_TARGET,
                "[plan] {} step(s): {}",
                plan_steps.len(),
                plan_steps
                    .iter()
                    .map(|s| format!(
                        "{}:{}[{}]",
                        s.step_type.value(),
                        or_none(s.tool_name.as_deref()),
                        s.model_tier.value()
                    ))
                    .collect::<Vec<_>>()
                    .join(" → ")
            );
        }

        self.build_step_nodes(ctx, &plan_steps, search_ctx, parent)
    }

    /// Derive execution steps for any request through one pipeline.
    ///
    /// Whether or not the prompt matched a skill, the request runs the same
    /// classify → create_plan flow.  An LLM-matched skill is injected as
    /// planning context — its instructions guide step generation — so the
    /// planner produces and executes the skill's procedure, instead of taking
    /// a separate derivation path.
    fn derive_steps(
        &self,
        ctx: &SharedOrchestratorContext,
        search_ctx: &SharedSearchContext,
        enrich_ctx: &EnrichedContext,
    ) -> Vec<PlanStep> {
        let c = ctx.read().clone();
        let sc = search_ctx.lock().clone();
        let vector_context = sc.vector_context.clone();

        // The LLM skill-match already decided which skills (if any) the prompt
        // maps to.  Feed each matched skill's stripped instructions to the
        // planner as a distinct entry so it can plan across multiple matched
        // skills and accumulate their outputs into the final result.  If
        // nothing matched, plan from the user prompt alone — do NOT dump every
        // workspace skill back in.
        let planner_skills: Option<Vec<String>> =
            Some(c.skill_bodies.clone()).filter(|b| !b.is_empty());

        let document_context: Option<String> = if !vector_context.is_empty() {
            Some(vector_context.clone())
        } else {
            c.document_context.clone().filter(|d| !d.is_empty())
        };

        // Planning is always driven by the user prompt's own classification.
        // A matched skill only contributes guiding context (planner_skills) —
        // it does not change the control flow.
        let mut classification = self.planner.classify(
            &c.user_message,
            &c.mcp_tool_contexts,
            ClassifyOptions {
                skills: planner_skills.clone(),
                document_context: document_context.clone(),
                has_attachments: c.has_attachments,
                history: non_empty(&enrich_ctx.history_insight).map(str::to_string),
                cwd: non_empty(&c.effective_cwd).map(str::to_string),
                available_capabilities: Some(c.available_capabilities.clone())
                    .filter(|a| !a.is_empty()),
            },
        );
        let pre_promotion_intent = classification.intent;

        // Indexed context that can ground an EXPLORE answer.  All four
        // sources count equally — a workspace-file or feedback match is just
        // as real as a document match (each is only present when it cleared
        // its search relevance threshold).  Counting only documents/skill-tools
        // here caused workspace-only matches to be discarded and the query
        // answered by the small model from raw guesswork.
        let mut matched_sources: Vec<&str> = Vec::new();
        if !vector_context.is_empty() {
            matched_sources.push("documents");
        }
        if !sc.workspace_context.is_empty() {
            matched_sources.push("workspace");
        }
        if !sc.feedback_context.is_empty() {
            matched_sources.push("feedback");
        }
        if !sc.web_context.is_empty() {
            matched_sources.push("web");
        }
        if !sc.mcp_context.is_empty() {
            matched_sources.push("mcp");
        }
        let has_indexed_context = !matched_sources.is_empty();

        if classification.intent == IntentType::Conversation {
            if has_indexed_context {
                classification.intent = IntentType::Explore;
                classification.reasoning = format!(
                    "indexed context available ({}) — exploring",
                    matched_sources.join(", ")
                );
            }
        } else if classification.intent == IntentType::Explore {
            // Small classifier LLM often hallucinates EXPLORE for plain
            // question queries.  If there's nothing indexed to explore,
            // treat it as conversation — otherwise we'd waste a search
            // step and route the answer through the small model.
            if !has_indexed_context {
                classification.intent = IntentType::Conversation;
                classification.reasoning =
                    "no indexed context to explore — treating as conversation".to_string();
            }
        }

        if !classification.detected_urls.is_empty()
            && classification.intent == IntentType::Conversation
        {
            classification.intent = IntentType::ToolCall;
            classification.reasoning = format!(
                "URL detected — deterministic fetch: {}",
                classification.detected_urls[0]
            );
        }

        tracing::info!(
            target: LOG_TARGET,
            "[classify] {}: {}",
            classification.intent.value(),
            classification.reasoning
        );
        if classification.intent != pre_promotion_intent {
            tracing::info!(
                target: LOG_TARGET,
                "[classify] intent adjusted {} → {} ({})",
                pre_promotion_intent.value(),
                classification.intent.value(),
                classification.reasoning
            );
        }

        let plan_context = self.planner.build_context(
            &c.user_message,
            &c.available_servers,
            non_empty(&c.effective_cwd),
            planner_skills,
            document_context.as_deref(),
        );
        // Multi-step tool selection is complex reasoning — prefer the large
        // 'dp' model when it is resident or fits, else fall back to the mini.
        // Resolved lazily, only for MULTI_STEP, so simple requests never touch
        // the 4B and a memory-tight device is never pushed into an OOM.
        let mut plan_llm: Option<Arc<dyn BaseLLM>> = None;
        if classification.intent == IntentType::MultiStep {
            let llm = self.models.decision_llm(Some(ModelCapability::Dp.value()), None);
            tracing::info!(
                target: LOG_TARGET,
                "[models] PLAN (multi-step) → {}",
                ModelRouter::model_label(&*llm)
            );
            plan_llm = Some(llm);
        }

        let mut plan = self.planner.create_plan(&classification, &plan_context, plan_llm.as_deref());
        if !plan.steps.is_empty() {
            let available: Vec<Map<String, Value>> = c
                .available_servers
                .iter()
                .filter_map(|v| v.as_object().cloned())
                .collect();
            plan.steps = self.explorer.resolve_fetch_steps(
                std::mem::take(&mut plan.steps),
                &available,
                &c.user_message,
            );
        }

        // Skills are indexed and matched by user intent; their plain-text body
        // guides the LLM directly. Tool execution is driven by the planner /
        // intent-matched MCP tools, not by parsing skill code blocks.
        let has_skill = c.skill_context.as_deref().is_some_and(|s| !s.is_empty());

        let detected_capability: Option<String> =
            classification.capability.clone().filter(|c| !c.is_empty());
        for step in plan.steps.iter_mut() {
            // Planner-emitted CONVERSATIONAL steps carry their own tier +
            // capability (LARGE / knowledge) — don't clobber them.
            if step.step_type == StepType::Conversational
                && step.capability.as_deref().is_some_and(|c| !c.is_empty())
            {
                continue;
            }
            if has_skill && step.step_type == StepType::Tool {
                step.model_tier = ModelTier::Large;
                step.capability = Some(
                    detected_capability
                        .clone()
                        .unwrap_or_else(|| ModelCapability::Reasoning.value().to_string()),
                );
            } else {
                step.model_tier = ModelTier::Small;
                if let Some(cap) = &detected_capability {
                    step.capability = Some(cap.clone());
                }
            }
        }

        plan.steps
    }

    /// Build DAG nodes from plan steps.
    ///
    /// One code path regardless of whether steps came from a skill or from the
    /// classify-then-plan pipeline.  Handlers use `ctx.skill_context` (set by
    /// the route handler when skills matched) for enrichment — no branching on
    /// matched_skills.
    pub fn build_step_nodes(
        self: &Arc<Self>,
        ctx: &SharedOrchestratorContext,
        plan_steps: &[PlanStep],
        search_ctx: &SharedSearchContext,
        parent: Option<&AgentNode>,
    ) -> Vec<DagExtension> {
        let is_conversation = plan_steps.is_empty();
        let mut parent: Option<AgentNode> = parent.cloned();

        // An upstream step (the document fan-out) may hand its updated search
        // context downstream; whichever node runs first after it adopts it.
        let adopted = Arc::new(Mutex::new(false));

        // The judge runs on both routes.  An EXPLORE step reads the retrieval
        // sources straight off search_ctx, so judging them only on the
        // conversation route let unrelated workspace / feedback / MCP content
        // walk into the answer context whenever the planner emitted any step.
        let mut nodes_and_handlers: Vec<DagExtension> = Vec::new();
        let judge_applies = {
            let c = ctx.read();
            let sc = search_ctx.lock();
            self.judge.applies_to(&c, Some(&sc))
        };
        if judge_applies {
            // Python: `self._judge.build(ctx, search_ctx, parent)` — one
            // LLMDecisionNode whose handler runs `evaluate_sources` on the
            // shared search context.  Built here against the shared context so
            // the sources it clears are cleared for every later reader.
            let deps: Vec<&AgentNode> = parent.iter().collect();
            let node = LLMDecisionNode::new(&deps);
            let handler = self.make_judge_handler(ctx, search_ctx);
            nodes_and_handlers.push(DagExtension::new(
                node.clone(),
                adopting(handler, Arc::clone(search_ctx), Arc::clone(&adopted)),
            ));
            parent = Some(node);
        }

        if is_conversation {
            let deps: Vec<&AgentNode> = parent.iter().collect();
            let reason_node = TransformNode::with_id(REASON_NODE.as_str(), &deps);
            let reason_handler = self.make_reason_handler_ordered(
                ctx,
                search_ctx,
                is_conversation,
                reason_node.depends_on.clone(),
            );
            nodes_and_handlers.push(DagExtension::new(
                reason_node,
                adopting(reason_handler, Arc::clone(search_ctx), Arc::clone(&adopted)),
            ));
            return nodes_and_handlers;
        }

        let mut prev_nodes: Vec<AgentNode> = parent.into_iter().collect();

        // Skill-derived steps run sequentially — they often have implicit
        // dependencies (e.g. conditions extracted from skill text at resolve
        // time) that _group_independent_steps cannot detect.
        let has_skill = ctx.read().skill_context.as_deref().is_some_and(|s| !s.is_empty());
        let groups: Vec<Vec<PlanStep>> = if has_skill {
            plan_steps.iter().map(|s| vec![s.clone()]).collect()
        } else {
            StepExecutor::group_independent_steps(plan_steps)
        };

        let mut pre_step_nodes: Vec<AgentNode> = Vec::new();

        for (gi, group) in groups.iter().enumerate() {
            let first_group = gi == 0;
            // Nodes of the first group sit directly under the parent, so they
            // are the ones that can see an upstream search-context payload.
            let wrap = |handler: NodeHandler| -> NodeHandler {
                if first_group {
                    adopting(handler, Arc::clone(search_ctx), Arc::clone(&adopted))
                } else {
                    handler
                }
            };
            let prev_refs: Vec<&AgentNode> = prev_nodes.iter().collect();
            let prev_ids: Vec<String> = prev_nodes.iter().map(|n| n.node_id.clone()).collect();
            let mut group_nodes: Vec<AgentNode> = Vec::new();
            for step in group {
                if step.step_type == StepType::Tool {
                    if step.for_each == Some(true) {
                        let fanout_node = LLMDecisionNode::new(&prev_refs);
                        let fanout_handler = self.tool_fanout.make_handler(ctx, step, search_ctx);
                        nodes_and_handlers
                            .push(DagExtension::new(fanout_node.clone(), wrap(fanout_handler)));
                        group_nodes.push(fanout_node.clone());
                        pre_step_nodes.push(fanout_node);
                    } else {
                        let resolve_node = LLMDecisionNode::new(&prev_refs);
                        let resolve_handler = self.make_resolve_handler(
                            ctx,
                            step,
                            search_ctx,
                            !pre_step_nodes.is_empty(),
                        );
                        nodes_and_handlers
                            .push(DagExtension::new(resolve_node.clone(), wrap(resolve_handler)));

                        let execute_node = ToolCallNode::new(&[&resolve_node]);
                        let execute_handler = self.make_execute_handler(ctx, step);
                        nodes_and_handlers
                            .push(DagExtension::new(execute_node.clone(), execute_handler));

                        group_nodes.push(execute_node.clone());
                        pre_step_nodes.push(execute_node);
                    }
                } else {
                    let node = match step.step_type {
                        StepType::Explore => SearchNode::new(&prev_refs),
                        StepType::Analyze => LLMDecisionNode::new(&prev_refs),
                        _ => ToolCallNode::new(&prev_refs),
                    };

                    let handler =
                        self.make_step_handler_ordered(ctx, step, search_ctx, prev_ids.clone());
                    nodes_and_handlers.push(DagExtension::new(node.clone(), wrap(handler)));
                    group_nodes.push(node.clone());
                    pre_step_nodes.push(node);
                }
            }

            prev_nodes = group_nodes;
        }

        let pre_refs: Vec<&AgentNode> = pre_step_nodes.iter().collect();
        let reason_node = TransformNode::with_id(REASON_NODE.as_str(), &pre_refs);
        let reason_handler = self.make_reason_handler_ordered(
            ctx,
            search_ctx,
            is_conversation,
            reason_node.depends_on.clone(),
        );
        nodes_and_handlers.push(DagExtension::new(reason_node, reason_handler));
        nodes_and_handlers
    }

    /// The judge node's handler: judge each retrieval source on the shared
    /// search context, clearing the ones that don't serve the request.
    fn make_judge_handler(
        &self,
        ctx: &SharedOrchestratorContext,
        search_ctx: &SharedSearchContext,
    ) -> NodeHandler {
        let judge = Arc::clone(&self.judge);
        let ctx = Arc::clone(ctx);
        let search_ctx = Arc::clone(search_ctx);
        Arc::new(move |_deps: HashMap<String, NodeResult>| {
            let c = ctx.read().clone();
            let mut sc = search_ctx.lock().clone();
            judge.evaluate_sources(&c, &mut sc);
            *search_ctx.lock() = sc;
            Ok(NodeResult::with_data(JUDGE_STEP_NODE, json!({})))
        })
    }

    /// Create a handler for a non-TOOL step node.
    ///
    /// Works uniformly for both skill-derived and prompt-derived steps.  Uses
    /// `ctx.skill_context` for instruction enrichment when available.
    pub fn make_step_handler(
        self: &Arc<Self>,
        ctx: &SharedOrchestratorContext,
        step: &PlanStep,
        search_ctx: &SharedSearchContext,
    ) -> NodeHandler {
        self.make_step_handler_ordered(ctx, step, search_ctx, Vec::new())
    }

    /// [`Self::make_step_handler`] with the node's dependency order (used to
    /// concatenate upstream output in a stable order).
    fn make_step_handler_ordered(
        self: &Arc<Self>,
        ctx: &SharedOrchestratorContext,
        step: &PlanStep,
        search_ctx: &SharedSearchContext,
        dep_order: Vec<String>,
    ) -> NodeHandler {
        let this = Arc::clone(self);
        let ctx = Arc::clone(ctx);
        let step = step.clone();
        let search_ctx = Arc::clone(search_ctx);
        Arc::new(move |deps: HashMap<String, NodeResult>| {
            Ok(this.run_step(&ctx, &step, &search_ctx, &deps, &dep_order))
        })
    }

    /// Body of a non-TOOL step handler.
    fn run_step(
        &self,
        ctx: &SharedOrchestratorContext,
        step: &PlanStep,
        search_ctx: &SharedSearchContext,
        deps: &HashMap<String, NodeResult>,
        dep_order: &[String],
    ) -> NodeResult {
        let prev_results = ResponseBuilder::collect_raw_from_deps(deps);
        let node_id = format!("step_{}_{}", step.step, step_label(step));

        // FETCH: run the FetchAgent (HTTP GET → markdown) in-process
        if step.step_type == StepType::Fetch {
            let mut url = step
                .default_args
                .as_ref()
                .and_then(|a| a.get("url").cloned())
                .unwrap_or_default();
            if url.is_empty() {
                url = URL_RE
                    .find(&step.instruction)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default();
            }
            if url.is_empty() {
                tracing::warn!(target: LOG_TARGET, "[fetch] No URL to fetch for step {}", step.step);
                return plan_step_result(&node_id, step, "", None);
            }
            self.models.emit("fetch", "processing", &format!("Fetching {url}"));
            let content = self.get_fetch_agent().fetch(&url, usize::MAX, 0, false);
            // Return empty on failure so downstream reasoning falls back to
            // any explore/vector context instead of surfacing a fetch error.
            if content.is_empty() {
                tracing::info!(target: LOG_TARGET, "[fetch] step {}: empty result from {}", step.step, url);
                return plan_step_result(&node_id, step, "", None);
            }
            tracing::info!(
                target: LOG_TARGET,
                "[fetch] step {}: {} chars from {}",
                step.step,
                content.chars().count(),
                url
            );
            return plan_step_result(&node_id, step, &content, Some(&content));
        }

        // WEB_SEARCH: run the WebSearchAgent (latest info/events) in-process
        if step.step_type == StepType::WebSearch {
            let query = step
                .default_args
                .as_ref()
                .and_then(|a| a.get("query").cloned())
                .filter(|q| !q.is_empty())
                .or_else(|| Some(step.instruction.clone()).filter(|i| !i.is_empty()))
                .unwrap_or_else(|| ctx.read().user_message.clone());
            self.models.emit(
                "web_search",
                "processing",
                "Searching the web for the latest information",
            );
            let agent = self.get_web_search_agent();
            let progress = self.progress();
            let content = agent.search(&query, None, None, Some(&progress));
            // Record the URLs on the shared search context, exactly as the
            // explore path does (see Explorer.collect).  _build_final_result
            // reads them back as citations, so without this a plan that
            // reaches the web through a WEB_SEARCH step answers from live
            // pages while surfacing no reference links at all.
            let extracts = agent.last_source_extracts();
            let last_sources = agent.last_sources();
            {
                let mut sc = search_ctx.lock();
                for url in &last_sources {
                    if !sc.web_sources.contains(url) {
                        sc.web_sources.push(url.clone());
                    }
                    if let Some(extract) = extracts.get(url) {
                        sc.web_source_extracts.insert(url.clone(), extract.clone());
                    }
                }
            }
            tracing::info!(
                target: LOG_TARGET,
                "[web_search] step {}: {} chars, {} source(s)",
                step.step,
                content.chars().count(),
                last_sources.len()
            );
            return plan_step_result(&node_id, step, &content, Some(&content));
        }

        // EXPLORE: return pre-fetched search context (already collected)
        if step.step_type == StepType::Explore {
            // Local sources first, the web last. Everything above the web
            // is the user's own material — their documents, workspace,
            // connected servers and prior feedback — and a model reading
            // top-down should meet that before a public page. Web results
            // used to sit third, ahead of the MCP result, which put
            // someone else's page in front of the user's own data.
            let sc = search_ctx.lock().clone();
            let mut parts: Vec<String> = Vec::new();
            if !sc.vector_context.is_empty() {
                parts.push(sc.vector_context.clone());
            }
            if !sc.workspace_context.is_empty() {
                parts.push(sc.workspace_context.clone());
            }
            if !sc.mcp_context.is_empty() {
                parts.push(sc.mcp_context.clone());
            }
            if !sc.feedback_context.is_empty() {
                parts.push(sc.feedback_context.clone());
            }
            if !sc.web_context.is_empty() {
                parts.push(ResponseBuilder::frame_web_results(&sc.web_context));
            }
            let content = parts.join("\n\n---\n\n");
            if let Some(citations) = sc.citations.as_ref().filter(|c| !c.is_empty()) {
                ctx.write().explore_citations =
                    Some(citations.iter().cloned().map(Value::Object).collect());
            }
            return plan_step_result(&node_id, step, &content, None);
        }

        // ANALYZE: assess relevance of explore output via LLM
        if step.step_type == StepType::Analyze {
            let mut explore_content = String::new();
            for (_, dep_result) in ordered_deps(deps, dep_order) {
                if dep_result.success && dep_result.data.is_object() {
                    let raw = data_str(&dep_result.data, "raw");
                    if !raw.is_empty() {
                        explore_content.push_str(&raw);
                        explore_content.push_str("\n\n");
                    }
                }
            }
            let explore_content = explore_content.trim().to_string();

            if explore_content.is_empty() {
                return plan_step_result(&node_id, step, "", None);
            }

            let user_message = ctx.read().user_message.clone();
            let analyze_prompt = format!(
                "<system>\n\
You are a relevance assessment assistant.\n\
Given a user request and search results, determine if the search results contain information relevant to the request.\n\
If relevant, produce a concise analysis of the useful information.\n\
If NOT relevant, respond with exactly: NOT_RELEVANT\n\
</system>\n\
<user>\n\
User request: {user_message}\n\
\n\
Search results:\n\
{}\n\
\n\
Assess whether these search results are relevant to the user's request. If relevant, summarize the key findings. If not relevant, respond with NOT_RELEVANT.\n\
</user>",
                char_prefix(&explore_content, 6000)
            );

            let outcome = try_call(|| {
                let step_llm = self.models.llm_for_tier(step.model_tier, step.capability.as_deref(), None);
                tracing::info!(
                    target: LOG_TARGET,
                    "[models] ANALYZE({}) → {}{}: {}",
                    step.step,
                    step.model_tier.value(),
                    capability_suffix(step.capability.as_deref()),
                    ModelRouter::model_label(&*step_llm)
                );
                step_llm.invoke(&analyze_prompt, None, None).trim().to_string()
            });
            return match outcome {
                Ok(result) => {
                    if result == "NOT_RELEVANT"
                        || result.chars().count() < 20
                        || is_placeholder_output(&result)
                    {
                        tracing::info!(
                            target: LOG_TARGET,
                            "[analyze] context deemed not relevant — passing through raw"
                        );
                        return plan_step_result(&node_id, step, &explore_content, Some(&explore_content));
                    }
                    tracing::info!(
                        target: LOG_TARGET,
                        "[analyze] produced {} chars",
                        result.chars().count()
                    );
                    ctx.write().skill_report = Some(result.clone());
                    plan_step_result(&node_id, step, &result, Some(&result))
                }
                Err(e) => {
                    tracing::warn!(target: LOG_TARGET, "[analyze] LLM analysis failed: {}", e);
                    plan_step_result(&node_id, step, &explore_content, Some(&explore_content))
                }
            };
        }

        // Evaluate condition
        if let Some(condition) = step.condition.as_deref().filter(|c| !c.is_empty()) {
            if !self.steps.evaluate_condition(Some(condition), &prev_results) {
                tracing::info!(
                    target: LOG_TARGET,
                    "[step] {} SKIPPED — condition not met: {}",
                    step.step,
                    condition
                );
                return plan_step_result(&node_id, step, "", None);
            }
        }

        let c = ctx.read().clone();

        // Enrich instruction with skill context when available
        let mut exec_step = step.clone();
        if let Some(skill_context) = c.skill_context.as_deref().filter(|s| !s.is_empty()) {
            let enriched_instruction = if !step.instruction.is_empty() {
                format!("{}\n\n{}", step.instruction, skill_context)
            } else {
                skill_context.to_string()
            };
            // Python rebuilt the step from these fields only, so everything
            // else takes the PlanStep defaults.
            exec_step = PlanStep {
                step: step.step,
                step_type: step.step_type,
                tool_name: step.tool_name.clone(),
                server_id: step.server_id.clone(),
                server_name: step.server_name.clone(),
                instruction: enriched_instruction,
                condition: step.condition.clone(),
                ..Default::default()
            };
        }

        let exec_llm = self.models.llm_for_tier(step.model_tier, step.capability.as_deref(), None);
        tracing::info!(
            target: LOG_TARGET,
            "[models] EXEC({}/{}) → {}{}: {}",
            step.step,
            step.step_type.value(),
            step.model_tier.value(),
            capability_suffix(step.capability.as_deref()),
            ModelRouter::model_label(&*exec_llm)
        );

        // Prefer the reduced document context (the DP model's extraction,
        // written to search_ctx.vector_context by the doc-reduce node) so a
        // skill step running on a *different* model still sees it — matching
        // the plan-build path. Fall back to the original doc context.
        let vector_context = search_ctx.lock().vector_context.clone();
        let document_context: Option<String> = if !vector_context.is_empty() {
            Some(vector_context)
        } else {
            c.document_context.clone()
        };

        let result = self.steps.execute(
            &c,
            &exec_step,
            if prev_results.is_empty() { None } else { Some(prev_results.as_slice()) },
            document_context.as_deref(),
            Some(exec_llm),
        );

        match result {
            Some((formatted, raw)) => {
                if step.step_type == StepType::Analyze && !raw.is_empty() {
                    ctx.write().skill_report = Some(raw.clone());
                }
                plan_step_result(&node_id, step, &formatted, Some(&raw))
            }
            None => plan_step_result(&node_id, step, "", None),
        }
    }

    /// Create a resolve handler for a TOOL step (parameter resolution).
    ///
    /// Works uniformly for both skill-derived and prompt-derived steps.  Uses
    /// `ctx.skill_context` for condition extraction when available.
    ///
    /// `has_upstream` is false for the first tool step in the plan. A
    /// condition gates a tool on a *prior* result, so it is meaningless for a
    /// step with nothing upstream — evaluating one there would always skip (no
    /// prior result can satisfy it). This is exactly what happens for a lone
    /// fetch step whose skill text reads "when the user requests a URL, fetch
    /// it": the extractor misreads that applicability clause as a runtime
    /// condition and drops the only step. Skip condition gating when there is
    /// no upstream.  (Python default: `has_upstream=True`.)
    pub fn make_resolve_handler(
        self: &Arc<Self>,
        ctx: &SharedOrchestratorContext,
        step: &PlanStep,
        _search_ctx: &SharedSearchContext,
        has_upstream: bool,
    ) -> NodeHandler {
        let step_llm = self.models.llm_for_tier(step.model_tier, step.capability.as_deref(), None);
        // Resolution is one call that picks the tool and fills its arguments,
        // so it goes to a reasoning model when the workspace has one enabled —
        // see ModelRouter.reasoning_llm and MCPAgent.
        let step_llm = self.models.reasoning_llm(step_llm);
        tracing::info!(
            target: LOG_TARGET,
            "[models] RESOLVE({}) → {}{}: {}",
            or_none(step.tool_name.as_deref()),
            step.model_tier.value(),
            capability_suffix(step.capability.as_deref()),
            ModelRouter::model_label(&*step_llm)
        );

        let this = Arc::clone(self);
        let ctx = Arc::clone(ctx);
        let step = step.clone();
        Arc::new(move |deps: HashMap<String, NodeResult>| {
            let prev_results = ResponseBuilder::collect_raw_from_deps(&deps);
            let node_id = format!("step_{}_{}_resolve", step.step, step_label(&step));
            let tool_display = or_none(step.tool_name.as_deref());

            let c = ctx.read().clone();
            let mut condition: Option<String> = if has_upstream {
                step.condition.clone().filter(|s| !s.is_empty())
            } else {
                None
            };
            if has_upstream && condition.is_none() {
                if let Some(skill_context) = c.skill_context.as_deref().filter(|s| !s.is_empty()) {
                    condition = this.skills.extract_tool_condition(
                        skill_context,
                        step.tool_name.as_deref().unwrap_or(""),
                    );
                }
            }
            if let Some(condition) = condition.as_deref().filter(|s| !s.is_empty()) {
                if prev_results.is_empty() {
                    tracing::info!(
                        target: LOG_TARGET,
                        "[resolve] SKIPPED {} — no upstream context for condition",
                        tool_display
                    );
                    return Ok(skip_result(&node_id, step_label(&step)));
                }
                if !this.steps.evaluate_condition(Some(condition), &prev_results) {
                    tracing::info!(
                        target: LOG_TARGET,
                        "[resolve] SKIPPED {} — condition not met",
                        tool_display
                    );
                    return Ok(skip_result(&node_id, step_label(&step)));
                }
            }

            tracing::info!(
                target: LOG_TARGET,
                "[resolve] {} [{}] {}/{}: {}",
                step.step,
                step.step_type.value(),
                or_none(step.server_name.as_deref()),
                tool_display,
                step.instruction
            );

            // Say that a tool call is being worked out, and on what — a
            // resolution that reasons takes seconds, and "Selected <model> for
            // the response" is not an account of what is happening.
            this.models.emit(
                "tool",
                "processing",
                if MCPAgent::can_think(&*step_llm) {
                    "Reasoning about which tool to call"
                } else {
                    "Working out which tool to call"
                },
            );

            let mut resolve_step = step.clone();
            let resolved = this.steps.resolve_tool_args(
                &c,
                &mut resolve_step,
                if prev_results.is_empty() { None } else { Some(prev_results.as_slice()) },
                Some(Arc::clone(&step_llm)),
            );

            let Some((server, tool_name, arguments)) = resolved else {
                tracing::info!(target: LOG_TARGET, "[resolve] {} — could not resolve args", tool_display);
                let what = match step.tool_name.as_deref() {
                    Some(t) if !t.is_empty() => t.to_string(),
                    _ => char_prefix(&step.instruction, 60).to_string(),
                };
                this.models.emit(
                    "tool",
                    "processing",
                    &format!("No tool call could be resolved for {what}"),
                );
                return Ok(skip_result(&node_id, step_label(&step)));
            };

            // The decision itself, in the shape it will be called: which
            // server, which tool, and the arguments it goes out with.
            this.models.emit(
                "tool",
                "processing",
                &format!("Resolved {}", format_call(&server.name, &tool_name, &arguments)),
            );
            tracing::info!(
                target: LOG_TARGET,
                "[resolve] {} resolved args: {}",
                tool_name,
                py_repr(&serde_json::Value::Object(arguments.clone()))
            );
            let tool_schema = this
                .tool_caller
                .find_tool_schema(&server, &tool_name)
                .map(Value::Object)
                .unwrap_or(Value::Null);
            Ok(NodeResult::with_data(
                node_id,
                json!({
                    "skip": false,
                    "tool_name": tool_name,
                    "server_id": server.id,
                    "arguments": arguments,
                    "tool_schema": tool_schema,
                }),
            ))
        })
    }

    /// Create an execute handler for a TOOL step (tool invocation).
    ///
    /// Reads resolved arguments from the resolve node and calls the tool.  No
    /// LLM needed — just an API call.
    pub fn make_execute_handler(
        self: &Arc<Self>,
        ctx: &SharedOrchestratorContext,
        step: &PlanStep,
    ) -> NodeHandler {
        tracing::info!(
            target: LOG_TARGET,
            "[models] EXECUTE({}) → no LLM (MCP tool call)",
            or_none(step.tool_name.as_deref())
        );
        let this = Arc::clone(self);
        let ctx = Arc::clone(ctx);
        let step = step.clone();
        Arc::new(move |deps: HashMap<String, NodeResult>| {
            let node_id = format!("step_{}_{}_execute", step.step, step_label(&step));
            let has_skill = step.skill_name.as_deref().is_some_and(|s| !s.is_empty());

            // Get the resolve node's output
            let resolve_data: Option<&Map<String, Value>> = ordered_deps(&deps, &[])
                .into_iter()
                .find(|(_, r)| r.success && r.data.is_object())
                .and_then(|(_, r)| r.data.as_object());

            // `if not resolve_data or resolve_data.get("skip")`
            let skipped = resolve_data
                .map_or(true, |d| d.is_empty() || d.get("skip").is_some_and(truthy));
            if skipped {
                let tool_name = match resolve_data.and_then(|d| d.get("tool_name")) {
                    Some(v) => crate::agents::orchestration::skill_processor::py_str(v),
                    None => step_label(&step),
                };
                let fail_msg = if has_skill {
                    format!("Tool '{tool_name}' skipped — could not resolve arguments.")
                } else {
                    String::new()
                };
                return Ok(plan_step_result(&node_id, &step, &fail_msg, None));
            }
            let resolve_data = resolve_data.expect("checked above");

            let tool_name = match resolve_data.get("tool_name") {
                Some(Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
                None => anyhow::bail!("KeyError: 'tool_name'"),
            };
            let server_id = match resolve_data.get("server_id") {
                Some(Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
                None => anyhow::bail!("KeyError: 'server_id'"),
            };
            let arguments: Map<String, Value> = match resolve_data.get("arguments") {
                Some(Value::Object(m)) => m.clone(),
                Some(Value::Null) => Map::new(),
                Some(other) => anyhow::bail!("arguments is not a mapping: {other}"),
                None => anyhow::bail!("KeyError: 'arguments'"),
            };

            // Find the server object
            let c = ctx.read().clone();
            let Some(server) = this.tool_caller.find_server(&c.mcp_servers, &server_id, &server_id).cloned()
            else {
                tracing::warn!(
                    target: LOG_TARGET,
                    "[execute] Server not found for {} (id={})",
                    tool_name,
                    server_id
                );
                let fail_msg = if has_skill {
                    format!("Tool '{tool_name}' failed: server not found.")
                } else {
                    String::new()
                };
                return Ok(plan_step_result(&node_id, &step, &fail_msg, None));
            };

            this.models.emit(
                "tool",
                "processing",
                &format!("Calling {}", format_call(&server.name, &tool_name, &arguments)),
            );
            let result =
                this.steps.call_tool(&server, &tool_name, &arguments, non_empty(&c.effective_cwd));
            let Some((formatted, raw)) = result else {
                let fail_msg = if has_skill {
                    format!("Tool '{tool_name}' call failed — no result returned.")
                } else {
                    String::new()
                };
                return Ok(plan_step_result(&node_id, &step, &fail_msg, None));
            };

            Ok(plan_step_result(&node_id, &step, &formatted, Some(&raw)))
        })
    }

    /// Create the reason handler — the final DAG node.
    ///
    /// Collects all upstream step outputs.  When skill context is available,
    /// synthesizes them into one coherent document.  The result's payload is
    /// a [`SearchContext`] snapshot of the shared context's final state.
    pub fn make_reason_handler(
        self: &Arc<Self>,
        ctx: &SharedOrchestratorContext,
        search_ctx: &SharedSearchContext,
        is_conversation: bool,
    ) -> NodeHandler {
        self.make_reason_handler_ordered(ctx, search_ctx, is_conversation, Vec::new())
    }

    /// [`Self::make_reason_handler`] with the reason node's dependency order,
    /// so step outputs accumulate in plan order.
    fn make_reason_handler_ordered(
        self: &Arc<Self>,
        ctx: &SharedOrchestratorContext,
        search_ctx: &SharedSearchContext,
        is_conversation: bool,
        dep_order: Vec<String>,
    ) -> NodeHandler {
        tracing::info!(
            target: LOG_TARGET,
            "[models] REASON → agent_llm: {}",
            ModelRouter::model_label(&*self.agent_llm())
        );
        let this = Arc::clone(self);
        let ctx = Arc::clone(ctx);
        let search_ctx = Arc::clone(search_ctx);
        Arc::new(move |deps: HashMap<String, NodeResult>| {
            let c: OrchestratorContext = ctx.read().clone();
            let has_skill = c.skill_context.as_deref().is_some_and(|s| !s.is_empty());

            // (tool_name, content, is_tool)
            let mut all_accumulated: Vec<(String, String, bool)> = Vec::new();

            // Retrieval-shaped step outputs are judged against the user's
            // intent before they are appended.  A plan routinely runs steps
            // that turn out not to bear on the question, and each one that
            // lands here becomes part of the context the answer is told to
            // work from.
            //
            // Tool results are exempt.  A tool call is a deliberate act with a
            // definite answer, not a similarity search that might have drifted
            // — the plan resolved a tool, the server ran it, and what came
            // back is the data the request went to get.  The judge runs on the
            // small decision model and fails toward dropping, so gating tool
            // output means a bad verdict discards the one thing that certainly
            // was about the request.  If the wrong tool ran, that is a routing
            // problem upstream; throwing away its result does not recover the
            // right one.
            let mut rejected: Vec<(String, String, bool)> = Vec::new();
            for (_, dep_result) in ordered_deps(&deps, &dep_order) {
                if !dep_result.success || !dep_result.data.is_object() {
                    continue;
                }
                let data = &dep_result.data;
                let content = data_str(data, "content");
                let raw = data_str(data, "raw");
                let tool_name = data_str(data, "tool_name");
                let is_tool = data.get("is_tool_result").is_some_and(truthy);
                let text = ResponseBuilder::strip_tool_code_blocks(if !content.is_empty() {
                    &content
                } else {
                    &raw
                });
                if text.is_empty() {
                    continue;
                }
                let label = if tool_name.is_empty() { "step output" } else { tool_name.as_str() };
                if is_tool || has_skill || this.judge.serves_intent(&c.user_message, label, &text) {
                    all_accumulated.push((tool_name, text, is_tool));
                } else {
                    rejected.push((tool_name, text, is_tool));
                }
            }

            // If the judge rejected every step output, keep them all.  Losing
            // one output among several is the judge doing its job; losing the
            // lot after the plan ran is far more likely to be the judge itself
            // misfiring, and the request would answer from nothing at all.
            // (Retrieval that is entirely irrelevant is dropped earlier, at the
            // gate — there the empty result is the correct answer and falls
            // through to a plain conversational reply.)
            if !rejected.is_empty() && all_accumulated.is_empty() {
                tracing::info!(
                    target: LOG_TARGET,
                    "[judge] all {} step output(s) rejected — keeping them rather than answering from nothing",
                    rejected.len()
                );
                all_accumulated = rejected;
            }

            let entry = |name: &str, content: &str| json!({ "tool_name": name, "content": content });
            let accumulated_json: Vec<Value> =
                all_accumulated.iter().map(|(n, t, _)| entry(n, t)).collect();
            let tool_results: Vec<Value> = all_accumulated
                .iter()
                .filter(|(_, _, is_tool)| *is_tool)
                .map(|(n, t, _)| entry(n, t))
                .collect();
            let citations = c.explore_citations.clone();

            let mut synthesized: Option<String> = None;
            if has_skill && !all_accumulated.is_empty() {
                let mut parts: Vec<String> = all_accumulated
                    .iter()
                    .filter(|(_, t, _)| !t.is_empty())
                    .map(|(_, t, _)| t.clone())
                    .collect();
                // Include the reduced document context (the DP model's extraction)
                // in the synthesis input so a skill run on a *different* model
                // still grounds its answer in the processed documents. Skip when a
                // step already carried it forward to avoid duplicating the text.
                let doc_ctx = search_ctx.lock().vector_context.clone();
                if !doc_ctx.is_empty() && !parts.iter().any(|p| p.contains(&doc_ctx)) {
                    parts.insert(0, doc_ctx);
                }
                if !parts.is_empty() {
                    this.models.emit(
                        "synthesize",
                        "processing",
                        &format!("Synthesizing the final answer from {} result(s)", parts.len()),
                    );
                    let raw_combined = parts.join("\n\n---\n\n");
                    synthesized = Some(this.skills.synthesize(
                        &c.user_message,
                        &raw_combined,
                        c.skill_context.as_deref().unwrap_or(""),
                    ));
                }
            }

            let snapshot = search_ctx.lock().clone();
            let data = json!({
                "accumulated": accumulated_json,
                "tool_results": tool_results,
                "synthesized": synthesized,
                "search_ctx": serde_json::to_value(&snapshot).unwrap_or(Value::Null),
                "is_conversation": is_conversation,
                "citations": citations,
            });
            Ok(NodeResult::with_data(REASON_NODE.as_str(), data).with_payload(snapshot))
        })
    }
}
