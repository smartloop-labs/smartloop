//! SLPOrchestrator — the coordinator for a user request.
//!
//! Port of `smartloop/agents/orchestrator.py`.
//!
//! Three phases, and little else: build the request context, walk the DAG the
//! pipeline builder assembles, then hand the accumulated results to the
//! response builder.  Everything the phases actually do lives in
//! [`crate::agents::orchestration`], one concern per module.
//!
//! The DAG is what makes the phases possible: each stage (skill match,
//! collect, route, per-step, reason) is a node with explicit dependencies, so
//! the executor runs independent branches in parallel and sequential ones in
//! order, without this type knowing the shape in advance.

use std::collections::HashMap;
use std::sync::Arc;

use indexmap::IndexMap;

use crate::agents::agent::SharedLLM;
use crate::agents::dag_executor::DAGExecutor;
use crate::agents::fetch_agent::UrlFetcher;
use crate::agents::hybrid_executor::LLMExecutor;
#[allow(unused_imports)]
use crate::agents::mcp::argument_extractor::MCPArgumentExtractor;
#[allow(unused_imports)]
use crate::agents::mcp::dependency_analyzer::MCPDependencyAnalyzer;
#[allow(unused_imports)]
use crate::agents::mcp::explorer::Explorer;
#[allow(unused_imports)]
use crate::agents::mcp::mcp_agent::MCPAgent;
#[allow(unused_imports)]
use crate::agents::mcp::mcp_tool_caller::MCPToolCaller;
#[allow(unused_imports)]
use crate::agents::mcp::result_mapper::MCPResultMapper;
#[allow(unused_imports)]
use crate::agents::mcp::server_selector::EXPLORE_TOOL;
use crate::agents::nodes::{AgentNode, AgentNodeName, NodeHandler, NodeResult, DAG};
#[allow(unused_imports)]
use crate::agents::orchestration::context::{MCPExecutionResult, OrchestratorContext};
#[allow(unused_imports)]
use crate::agents::orchestration::document_fanout::DocumentFanoutStep;
#[allow(unused_imports)]
use crate::agents::orchestration::judge_step::JudgeStep;
#[allow(unused_imports)]
use crate::agents::orchestration::model_router::ModelRouter;
#[allow(unused_imports)]
use crate::agents::orchestration::pipeline_builder::PipelineBuilder;
#[allow(unused_imports)]
use crate::agents::orchestration::response_builder::ResponseBuilder;
#[allow(unused_imports)]
use crate::agents::orchestration::skill_processor::SkillProcessor;
#[allow(unused_imports)]
use crate::agents::orchestration::step_executor::StepExecutor;
#[allow(unused_imports)]
use crate::agents::orchestration::tool_fanout::ToolFanout;
#[allow(unused_imports)]
use crate::agents::planner::{MCPToolContext, Planner};
use crate::agents::web_search_agent::{ProgressCallback, WebSearcher};
#[allow(unused_imports)]
use crate::api::services::model_service::ModelService;
#[allow(unused_imports)]
use crate::paths::expand_home;
#[allow(unused_imports)]
use crate::utils::mcp_result_formatter::ResultFormatter;
use crate::workspace::workspace::MCPServer;

/// Node id of the final reason node.
pub const REASON_NODE: AgentNodeName = AgentNodeName::Reason;

/// Lazily loads the large model (`large_llm_provider`).
pub type LargeLlmProvider = Arc<dyn Fn() -> Option<SharedLLM> + Send + Sync>;
/// Loads the model for a capability (`capability_llm_provider`).
pub type CapabilityLlmProvider = Arc<dyn Fn(&str) -> Option<SharedLLM> + Send + Sync>;
/// Whether a capability model is available (`capability_available`).
pub type CapabilityAvailable = Arc<dyn Fn(&str) -> bool + Send + Sync>;
/// Why no capability model was used: `(model_name, cause, detail)`
/// (`capability_skip_reason`).
pub type CapabilitySkipReason =
    Arc<dyn Fn() -> Option<(String, String, Option<String>)> + Send + Sync>;

/// Optional constructor arguments of [`SLPOrchestrator::new`] (Python kwargs).
#[derive(Clone, Default)]
pub struct SLPOrchestratorOptions {
    /// Decision model; defaults to the orchestrator model.
    pub agent_llm: Option<SharedLLM>,
    pub large_llm: Option<SharedLLM>,
    pub large_llm_provider: Option<LargeLlmProvider>,
    pub capability_llm_provider: Option<CapabilityLlmProvider>,
    pub capability_available: Option<CapabilityAvailable>,
    pub capability_skip_reason: Option<CapabilitySkipReason>,
    pub model_path: Option<String>,
    pub project_id: Option<String>,
    /// Shared, long-lived I/O executor; a per-instance one is created when
    /// absent (e.g. in tests).
    pub executor: Option<Arc<LLMExecutor>>,
}

/// Optional arguments of [`SLPOrchestrator::execute`] (Python kwargs).
#[derive(Clone, Default)]
pub struct ExecuteOptions {
    pub skills: Option<Vec<String>>,
    pub cwd: Option<String>,
    pub document_context: Option<String>,
    pub has_attachments: bool,
    pub project_dir: Option<String>,
    pub model_service: Option<Arc<ModelService>>,
    pub conversation_history: Option<Vec<HashMap<String, String>>>,
    pub available_capabilities: Option<Vec<String>>,
    /// `progress_callback(step, status, message)`; fires from DAG workers.
    pub progress_callback: Option<ProgressCallback>,
    pub requested_skill: Option<String>,
    /// The client's wall-clock date ("Month DD, YYYY").
    pub current_date: Option<String>,
}

/// Coordinates plan → run → reason for one user message.
///
/// Owns the collaborators and the orchestration components, and nothing
/// else: every phase delegates.  `orchestrator_llm` and `agent_llm` are
/// write-through accessors over [`ModelRouter`], where the models live.
pub struct SLPOrchestrator {
    models: Arc<ModelRouter>,
    pub planner: Arc<Planner>,
    pub arg_extractor: Arc<MCPArgumentExtractor>,
    pub dep_analyzer: Arc<MCPDependencyAnalyzer>,
    pub result_mapper: Arc<MCPResultMapper>,
    pub tool_caller: Arc<MCPToolCaller>,
    pub result_formatter: Arc<ResultFormatter>,
    pub explorer: Arc<Explorer>,
    mcp_agent: Arc<MCPAgent>,
    response: Arc<ResponseBuilder>,
    judge: Arc<JudgeStep>,
    doc_fanout: Arc<DocumentFanoutStep>,
    skills: Arc<SkillProcessor>,
    steps: Arc<StepExecutor>,
    tool_fanout: Arc<ToolFanout>,
    executor: Arc<LLMExecutor>,
    dag_executor: DAGExecutor,
    pipeline: Arc<PipelineBuilder>,
}

impl SLPOrchestrator {
    /// Build the orchestrator and all its collaborators.
    pub fn new(orchestrator_llm: SharedLLM, options: SLPOrchestratorOptions) -> Self {
        let _ = (orchestrator_llm, options);
        todo!()
    }

    /// The HTTP fetcher.  Owned by [`PipelineBuilder`].
    pub fn fetch_agent(&self) -> Arc<dyn UrlFetcher> {
        todo!()
    }

    /// Replace the HTTP fetcher (Python `_fetch_agent` setter).
    pub fn set_fetch_agent(&self, agent: Arc<dyn UrlFetcher>) {
        let _ = agent;
        todo!()
    }

    /// The web searcher.  Owned by [`PipelineBuilder`].
    pub fn web_search_agent(&self) -> Arc<dyn WebSearcher> {
        todo!()
    }

    /// Replace the web searcher (Python `_web_search_agent` setter).
    pub fn set_web_search_agent(&self, agent: Arc<dyn WebSearcher>) {
        let _ = agent;
        todo!()
    }

    /// The response model.  Owned by [`ModelRouter`].
    pub fn orchestrator_llm(&self) -> SharedLLM {
        todo!()
    }

    /// Replace the response model.
    pub fn set_orchestrator_llm(&self, llm: SharedLLM) {
        let _ = llm;
        todo!()
    }

    /// The decision model.  Owned by [`ModelRouter`].
    pub fn agent_llm(&self) -> SharedLLM {
        todo!()
    }

    /// Replace the decision model.
    pub fn set_agent_llm(&self, llm: SharedLLM) {
        let _ = llm;
        todo!()
    }

    /// The model router (shared with the orchestration components).
    pub fn models(&self) -> &Arc<ModelRouter> {
        &self.models
    }

    /// Top-level entry point: plan → run → reason.  Any failure is logged and
    /// yields `MCPExecutionResult { handled: false, .. }`.
    pub fn execute(
        &self,
        user_message: &str,
        mcp_servers: &[MCPServer],
        options: ExecuteOptions,
    ) -> MCPExecutionResult {
        let _ = (user_message, mcp_servers, options);
        todo!()
    }

    /// Build the shared orchestrator context from input parameters.
    pub fn build_context(
        &self,
        user_message: &str,
        mcp_servers: &[MCPServer],
        options: &ExecuteOptions,
    ) -> anyhow::Result<OrchestratorContext> {
        let _ = (user_message, mcp_servers, options);
        todo!()
    }

    /// Phase 1: build the execution DAG (skill match → collect → route) and
    /// its handler map.
    pub fn plan(&self, ctx: &Arc<OrchestratorContext>) -> (DAG, HashMap<String, NodeHandler>) {
        let _ = ctx;
        todo!()
    }

    /// Phase 2: walk the DAG and accumulate raw results from each node.
    pub fn run(
        &self,
        ctx: &Arc<OrchestratorContext>,
        dag: &mut DAG,
        handlers: &mut HashMap<String, NodeHandler>,
    ) -> IndexMap<String, NodeResult> {
        let _ = (ctx, dag, handlers, &self.dag_executor);
        todo!()
    }

    /// Phase 3: analyze accumulated results and produce the final response.
    pub fn reason(
        &self,
        ctx: &Arc<OrchestratorContext>,
        results: &IndexMap<String, NodeResult>,
    ) -> MCPExecutionResult {
        let _ = (ctx, results);
        todo!()
    }

    /// Build the handler map for the initial DAG nodes.
    fn build_handlers(
        &self,
        ctx: &Arc<OrchestratorContext>,
        skill_match_node: &AgentNode,
        collect_node: &AgentNode,
        route_node: &AgentNode,
    ) -> HashMap<String, NodeHandler> {
        let _ = (ctx, skill_match_node, collect_node, route_node);
        let _ = (
            &self.mcp_agent,
            &self.response,
            &self.judge,
            &self.doc_fanout,
            &self.skills,
            &self.steps,
            &self.tool_fanout,
            &self.executor,
            &self.pipeline,
        );
        todo!()
    }
}
