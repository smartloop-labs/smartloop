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

#![allow(unused_imports, unused_variables, dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{Map, Value};

use crate::agents::fetch_agent::FetchAgent;
use crate::agents::hybrid_executor::LLMExecutor;
use crate::agents::mcp::explorer::{EnrichedContext, Explorer, SearchContext};
use crate::agents::mcp::mcp_agent::{format_call, MCPAgent};
use crate::agents::mcp::mcp_tool_caller::MCPToolCaller;
use crate::agents::nodes::{
    AgentNode, AgentNodeName, CollectNode, DagExtension, LLMDecisionNode, NodeHandler,
    NodeResult, RouteNode, SearchNode, SkillMatchNode, ToolCallNode, TransformNode,
};
use crate::agents::orchestration::context::OrchestratorContext;
use crate::agents::orchestration::document_fanout::DocumentFanoutStep;
use crate::agents::orchestration::judge_step::JudgeStep;
use crate::agents::orchestration::model_router::ModelRouter;
use crate::agents::orchestration::replies::is_placeholder_output;
use crate::agents::orchestration::response_builder::ResponseBuilder;
use crate::agents::orchestration::skill_processor::{SkillDict, SkillProcessor};
use crate::agents::orchestration::step_executor::{plan_step_result, StepExecutor};
use crate::agents::orchestration::tool_fanout::ToolFanout;
use crate::agents::planner::{IntentType, ModelTier, PlanStep, Planner, StepType};
use crate::agents::web_search_agent::WebSearchAgent;
use crate::llms::base_llm::BaseLLM;
use crate::types::ModelCapability;

/// Python logger name.
const LOG_TARGET: &str = "smartloop.agents.orchestration.pipeline_builder";

/// Node id of the final reason node.
pub const REASON_NODE: AgentNodeName = AgentNodeName::Reason;

/// Shared, mutable per-request context captured by node handlers.
pub type SharedOrchestratorContext = Arc<Mutex<OrchestratorContext>>;

/// Shared, mutable search context captured by node handlers (a WEB_SEARCH
/// step appends its sources to it).
pub type SharedSearchContext = Arc<Mutex<SearchContext>>;


/// Turns a request into the DAG of nodes that answers it.
pub struct PipelineBuilder {
    pub(crate) models: Arc<ModelRouter>,
    pub planner: Arc<Planner>,
    pub explorer: Arc<Explorer>,
    pub(crate) executor: Arc<LLMExecutor>,
    pub tool_caller: Arc<MCPToolCaller>,
    /// The HTTP fetcher (replaceable by the orchestrator).
    pub fetch_agent: Arc<Mutex<Arc<FetchAgent>>>,
    pub(crate) web_search_agent: Arc<WebSearchAgent>,
    pub(crate) judge: Arc<JudgeStep>,
    pub(crate) doc_fanout: Arc<DocumentFanoutStep>,
    pub(crate) skills: Arc<SkillProcessor>,
    pub(crate) steps: Arc<StepExecutor>,
    pub(crate) tool_fanout: Arc<ToolFanout>,
    pub(crate) response: Arc<ResponseBuilder>,
}

impl PipelineBuilder {
    /// Wire the builder to the services it coordinates.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        models: Arc<ModelRouter>,
        planner: Arc<Planner>,
        explorer: Arc<Explorer>,
        executor: Arc<LLMExecutor>,
        tool_caller: Arc<MCPToolCaller>,
        fetch_agent: Arc<FetchAgent>,
        web_search_agent: Arc<WebSearchAgent>,
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
            fetch_agent: Arc::new(Mutex::new(fetch_agent)),
            web_search_agent,
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
        todo!()
    }

    /// Current HTTP fetcher.
    pub fn get_fetch_agent(&self) -> Arc<FetchAgent> {
        self.fetch_agent.lock().clone()
    }

    /// Replace the HTTP fetcher (Python: `orchestrator.fetch_agent = ...`).
    pub fn set_fetch_agent(&self, agent: Arc<FetchAgent>) {
        *self.fetch_agent.lock() = agent;
    }

    /// Vector + workspace + feedback search.
    ///
    /// Returns a result whose payload is the [`SharedSearchContext`]
    /// (Python `data={"search_ctx": search_ctx}`).
    pub fn handle_collect(
        &self,
        ctx: &SharedOrchestratorContext,
        deps: &HashMap<String, NodeResult>,
        node: &AgentNode,
        skill_match_node: &AgentNode,
    ) -> NodeResult {
        todo!()
    }

    /// Decision: build the unified pipeline. Dynamically adds downstream nodes
    /// (returned in [`NodeResult::extend_dag`]).
    pub fn handle_route(
        self: &Arc<Self>,
        ctx: &SharedOrchestratorContext,
        deps: &HashMap<String, NodeResult>,
        node: &AgentNode,
        skill_match_node: &AgentNode,
        collect_node: &AgentNode,
    ) -> NodeResult {
        todo!()
    }

    /// Build a single pipeline regardless of whether skills matched.
    ///
    /// Both paths produce a uniform `Vec<PlanStep>` that feeds into the same
    /// node builders and handlers — no branching downstream.
    pub fn build_unified_pipeline(
        self: &Arc<Self>,
        ctx: &SharedOrchestratorContext,
        matched_skills: Option<&[SkillDict]>,
        search_ctx: &SharedSearchContext,
        enrich_ctx: &EnrichedContext,
        parent: Option<&AgentNode>,
    ) -> Vec<DagExtension> {
        todo!()
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
        todo!()
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
        todo!()
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
        todo!()
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
        search_ctx: &SharedSearchContext,
        has_upstream: bool,
    ) -> NodeHandler {
        todo!()
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
        todo!()
    }

    /// Ceiling on items per fan-out (kept for parity with the Python class).
    pub const FANOUT_MAX_ITEMS: usize = 20;

    /// Create the reason handler — the final DAG node.
    ///
    /// Collects all upstream step outputs.  When skill context is available,
    /// synthesizes them into one coherent document.
    pub fn make_reason_handler(
        self: &Arc<Self>,
        ctx: &SharedOrchestratorContext,
        search_ctx: &SharedSearchContext,
        is_conversation: bool,
    ) -> NodeHandler {
        todo!()
    }
}
