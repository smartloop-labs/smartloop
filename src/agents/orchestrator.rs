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

use std::collections::{HashMap, HashSet};
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use indexmap::IndexMap;
use parking_lot::{Mutex, RwLock};
use serde_json::{Map, Value};

use crate::llms::base_llm::{BaseLLM, CancelEvent};
use crate::agents::dag_executor::DAGExecutor;
use crate::agents::fetch_agent::{FetchAgent, UrlFetcher};
use crate::agents::hybrid_executor::LLMExecutor;
use crate::agents::mcp::argument_extractor::MCPArgumentExtractor;
use crate::agents::mcp::dependency_analyzer::MCPDependencyAnalyzer;
use crate::agents::mcp::explorer::{Explorer, SearchContext};
use crate::agents::mcp::mcp_agent::MCPAgent;
use crate::agents::mcp::mcp_tool_caller::MCPToolCaller;
use crate::agents::mcp::result_mapper::MCPResultMapper;
use crate::agents::mcp::server_selector::EXPLORE_TOOL;
use crate::agents::nodes::{AgentNode, AgentNodeName, CollectNode, NodeHandler, NodeResult, RouteNode, SkillMatchNode, DAG};
pub use crate::agents::orchestration::context::MCPExecutionResult;
pub use crate::agents::orchestration::context::OrchestratorContext;
use crate::agents::orchestration::document_fanout::DocumentFanoutStep;
use crate::agents::orchestration::judge_step::JudgeStep;
pub use crate::agents::orchestration::model_router::{
    CapabilityAvailable, CapabilityLlmProvider, CapabilitySkipReasonFn, LargeLlmProvider,
};
use crate::agents::orchestration::model_router::{ModelRouter, ModelRouterOptions};
use crate::agents::orchestration::pipeline_builder::PipelineBuilder;
/// The per-request context as the pipeline's handlers share it.
pub use crate::agents::orchestration::pipeline_builder::{SharedOrchestratorContext, SharedSearchContext};
use crate::agents::orchestration::response_builder::ResponseBuilder;
use crate::agents::orchestration::skill_processor::SkillProcessor;
use crate::agents::orchestration::step::Step;
use crate::agents::orchestration::step_executor::StepExecutor;
use crate::agents::orchestration::tool_fanout::ToolFanout;
use crate::agents::planner::{MCPToolContext, Planner};
use crate::agents::web_search_agent::{ProgressCallback, WebSearchAgent, WebSearcher};
use crate::api::services::model_service::ModelService;
use crate::paths::expand_home;
use crate::utils::mcp_result_formatter::ResultFormatter;
use crate::workspace::workspace::MCPServer;

/// Python logger name.
const LOG_TARGET: &str = "smartloop.agents.orchestrator";

/// Node id of the final reason node.
pub const REASON_NODE: AgentNodeName = AgentNodeName::Reason;

/// Uniform access to a shared, lock-guarded value, whichever lock the
/// pipeline shares its per-request state behind (`Arc<Mutex<_>>` or
/// `Arc<RwLock<_>>`).
pub trait SharedCell<T>: Clone + Send + Sync + 'static {
    /// Put `value` behind a fresh lock.
    fn wrap(value: T) -> Self;
    /// Read the value.
    fn with_ref<R>(&self, f: impl FnOnce(&T) -> R) -> R;
    /// Mutate the value.
    fn with_mut<R>(&self, f: impl FnOnce(&mut T) -> R) -> R;
}

impl<T: Send + 'static> SharedCell<T> for Arc<Mutex<T>> {
    fn wrap(value: T) -> Self {
        Arc::new(Mutex::new(value))
    }
    fn with_ref<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        f(&self.lock())
    }
    fn with_mut<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        f(&mut self.lock())
    }
}

impl<T: Send + Sync + 'static> SharedCell<T> for Arc<RwLock<T>> {
    fn wrap(value: T) -> Self {
        Arc::new(RwLock::new(value))
    }
    fn with_ref<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        f(&self.read())
    }
    fn with_mut<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        f(&mut self.write())
    }
}

/// Replaces the collect phase's explore (`explorer.collect_parallel`) for a
/// request: given the request context, returns the search context the route
/// node reads.  A test seam — the Python tests patch
/// `orchestrator.explorer.collect_parallel`.
pub type CollectOverride = Arc<dyn Fn(&OrchestratorContext) -> SharedSearchContext + Send + Sync>;

/// Optional constructor arguments of [`SLPOrchestrator::new`] (Python kwargs).
#[derive(Clone, Default)]
pub struct SLPOrchestratorOptions {
    /// Decision model; defaults to the orchestrator model.
    pub agent_llm: Option<Arc<dyn BaseLLM>>,
    pub large_llm: Option<Arc<dyn BaseLLM>>,
    pub large_llm_provider: Option<LargeLlmProvider>,
    pub capability_llm_provider: Option<CapabilityLlmProvider>,
    pub capability_available: Option<CapabilityAvailable>,
    pub capability_skip_reason: Option<CapabilitySkipReasonFn>,
    pub model_path: Option<String>,
    pub project_id: Option<String>,
    /// The final answer comes from a pinned remote model, which writes it
    /// from the gathered context; no step drafts it first.
    pub answers_remotely: bool,
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
    /// Set when the client goes away; long steps (the document fan-out)
    /// stop at their next check.
    pub cancelled: Option<CancelEvent>,
}

/// Coordinates plan → run → reason for one user message.
///
/// Owns the collaborators and the orchestration components, and nothing
/// else: every phase delegates.  `orchestrator_llm` and `agent_llm` read
/// through to [`ModelRouter`], which is where the models actually live.
pub struct SLPOrchestrator {
    models: Arc<ModelRouter>,
    // -- Collaborators shared by the components below --
    pub planner: Arc<Planner>,
    pub arg_extractor: Arc<MCPArgumentExtractor>,
    pub dep_analyzer: Arc<MCPDependencyAnalyzer>,
    pub result_mapper: Arc<Mutex<MCPResultMapper>>,
    pub tool_caller: Arc<MCPToolCaller>,
    pub result_formatter: Arc<ResultFormatter>,
    pub explorer: Arc<Explorer>,
    mcp_agent: Arc<MCPAgent>,
    // -- Orchestration components --
    response: Arc<ResponseBuilder>,
    judge: Arc<JudgeStep>,
    doc_fanout: Arc<DocumentFanoutStep>,
    skills: Arc<SkillProcessor>,
    steps: Arc<StepExecutor>,
    tool_fanout: Arc<ToolFanout>,
    executor: Arc<LLMExecutor>,
    dag_executor: DAGExecutor,
    pipeline: Arc<PipelineBuilder>,
    /// Test seam replacing the collect phase's explore.
    collect_override: Mutex<Option<CollectOverride>>,
}

impl SLPOrchestrator {
    /// Build the orchestrator and all its collaborators.
    pub fn new(orchestrator_llm: Arc<dyn BaseLLM>, options: SLPOrchestratorOptions) -> Self {
        // ModelRouter owns the models; `orchestrator_llm()` / `agent_llm()`
        // read through to it so there is exactly one place they live.
        let agent_llm = options.agent_llm.clone().unwrap_or_else(|| Arc::clone(&orchestrator_llm));
        let models = Arc::new(ModelRouter::new(
            Arc::clone(&orchestrator_llm),
            Arc::clone(&agent_llm),
            ModelRouterOptions {
                large_llm: options.large_llm.clone(),
                large_llm_provider: options.large_llm_provider.clone(),
                capability_llm_provider: options.capability_llm_provider.clone(),
                capability_available: options.capability_available.clone(),
                capability_skip_reason: options.capability_skip_reason.clone(),
            },
        ));

        // -- Collaborators shared by the components below --
        let planner = Arc::new(Planner::new(Arc::clone(&agent_llm)));
        let arg_extractor = Arc::new(MCPArgumentExtractor::new(Arc::clone(&agent_llm)));
        let dep_analyzer = Arc::new(MCPDependencyAnalyzer::new());
        let result_mapper = Arc::new(Mutex::new(MCPResultMapper::new()));
        let tool_caller =
            Arc::new(MCPToolCaller::new(Arc::clone(&arg_extractor), Arc::clone(&result_mapper), None));
        let result_formatter = Arc::new(ResultFormatter::new(Arc::clone(&orchestrator_llm)));
        let explorer = Arc::new(Explorer::new(
            Arc::clone(&agent_llm),
            Some(Arc::clone(&tool_caller)),
            Some(Arc::clone(&arg_extractor)),
            false,
        ));
        let fetch_agent = Arc::new(FetchAgent::new());
        let web_search_agent = Arc::new(WebSearchAgent::new(None, 3, 10, Some(Arc::clone(&agent_llm))));
        // MCPAgent: reasons over the tools discovery recorded for every
        // connected server and answers with one call, required arguments
        // filled. This is the resolution path for external MCP tool steps (no
        // separate server selector, no keyword-scan fallback).
        let mcp_agent = Arc::new(MCPAgent::new(Arc::clone(&agent_llm)));

        // -- Orchestration components, constructed in dependency order --
        let response = Arc::new(ResponseBuilder::new(Arc::clone(&orchestrator_llm)));
        let judge = Arc::new(JudgeStep::new(Arc::clone(&models)));
        let doc_fanout = Arc::new(DocumentFanoutStep::new(Arc::clone(&models)));
        let skills = Arc::new(SkillProcessor::new(Arc::clone(&models), Arc::clone(&explorer)));
        let steps = Arc::new(StepExecutor::new(
            Arc::clone(&orchestrator_llm),
            Arc::clone(&tool_caller),
            Arc::clone(&arg_extractor),
            Arc::clone(&dep_analyzer),
            Arc::clone(&result_formatter),
            Arc::clone(&explorer),
            Arc::clone(&mcp_agent),
            Arc::clone(&response),
        )
        .with_remote_answer(options.answers_remotely));
        let tool_fanout = Arc::new(ToolFanout::new(
            Arc::clone(&models),
            Arc::clone(&steps),
            Arc::clone(&response),
            Arc::clone(&tool_caller),
        ));

        // Prefer a shared, long-lived executor.  Fall back to a per-instance
        // one when none is supplied (e.g. in tests).
        let executor = options.executor.clone().unwrap_or_else(|| Arc::new(LLMExecutor::new(6)));
        let dag_executor = DAGExecutor::new(6);

        let pipeline = Arc::new(PipelineBuilder::new(
            Arc::clone(&models),
            Arc::clone(&planner),
            Arc::clone(&explorer),
            Arc::clone(&executor),
            Arc::clone(&tool_caller),
            fetch_agent,
            web_search_agent,
            Arc::clone(&judge),
            Arc::clone(&doc_fanout),
            Arc::clone(&skills),
            Arc::clone(&steps),
            Arc::clone(&tool_fanout),
            Arc::clone(&response),
        ));

        Self {
            models,
            planner,
            arg_extractor,
            dep_analyzer,
            result_mapper,
            tool_caller,
            result_formatter,
            explorer,
            mcp_agent,
            response,
            judge,
            doc_fanout,
            skills,
            steps,
            tool_fanout,
            executor,
            dag_executor,
            pipeline,
            collect_override: Mutex::new(None),
        }
    }

    /// The HTTP fetcher.  Owned by [`PipelineBuilder`].
    pub fn fetch_agent(&self) -> Arc<dyn UrlFetcher> {
        self.pipeline.get_fetch_agent()
    }

    /// Replace the HTTP fetcher (Python `_fetch_agent` setter).
    pub fn set_fetch_agent(&self, agent: Arc<dyn UrlFetcher>) {
        self.pipeline.set_fetch_agent(agent);
    }

    /// The web searcher.  Owned by [`PipelineBuilder`].
    pub fn web_search_agent(&self) -> Arc<dyn WebSearcher> {
        self.pipeline.get_web_search_agent()
    }

    /// The response model.  Owned by [`ModelRouter`].
    pub fn orchestrator_llm(&self) -> Arc<dyn BaseLLM> {
        Arc::clone(&self.models.orchestrator_llm)
    }

    /// The decision model.  Owned by [`ModelRouter`].
    pub fn agent_llm(&self) -> Arc<dyn BaseLLM> {
        Arc::clone(&self.models.agent_llm)
    }

    /// The model router (shared with the orchestration components).
    pub fn models(&self) -> &Arc<ModelRouter> {
        &self.models
    }

    /// The DAG builder (Python `_pipeline`).
    pub fn pipeline(&self) -> &Arc<PipelineBuilder> {
        &self.pipeline
    }

    /// The response builder (Python `_response`).
    pub fn response(&self) -> &Arc<ResponseBuilder> {
        &self.response
    }

    /// The relevance judge (Python `_judge`).
    pub fn judge(&self) -> &Arc<JudgeStep> {
        &self.judge
    }

    /// The document fan-out step (Python `_doc_fanout`).
    pub fn doc_fanout(&self) -> &Arc<DocumentFanoutStep> {
        &self.doc_fanout
    }

    /// The skill processor (Python `_skills`).
    pub fn skills(&self) -> &Arc<SkillProcessor> {
        &self.skills
    }

    /// The step executor (Python `_steps`).
    pub fn steps(&self) -> &Arc<StepExecutor> {
        &self.steps
    }

    /// The tool fan-out service (Python `_tool_fanout`).
    pub fn tool_fanout(&self) -> &Arc<ToolFanout> {
        &self.tool_fanout
    }

    /// The MCP tool resolver (Python `_mcp_agent`).
    pub fn mcp_agent(&self) -> &Arc<MCPAgent> {
        &self.mcp_agent
    }

    /// The I/O executor (Python `_executor`).
    pub fn executor(&self) -> &Arc<LLMExecutor> {
        &self.executor
    }

    /// Replace (or, with `None`, restore) the collect phase's explore — see
    /// [`CollectOverride`].
    pub fn set_collect_override(&self, f: Option<CollectOverride>) {
        *self.collect_override.lock() = f;
    }

    /// Top-level entry point: plan → run → reason.
    ///
    /// `progress_callback(step, status, message)` — when supplied — is invoked
    /// as each step runs so the streaming layer can surface live activity to
    /// the client.  It fires from DAG worker threads.
    ///
    /// Any failure is logged and yields `MCPExecutionResult { handled: false, .. }`.
    pub fn execute(
        &self,
        user_message: &str,
        mcp_servers: &[MCPServer],
        options: ExecuteOptions,
    ) -> MCPExecutionResult {
        self.models.begin_request(options.progress_callback.clone());
        self.judge.begin_request();
        let large = match self.models.large_llm.lock().as_ref() {
            Some(llm) => ModelRouter::model_label(llm.as_ref()),
            None if self.models.has_large_provider() => "(lazy)".to_string(),
            None => "(none)".to_string(),
        };
        tracing::info!(
            target: LOG_TARGET,
            "[models] response={}  agent={}  large={}",
            ModelRouter::model_label(self.models.orchestrator_llm.as_ref()),
            ModelRouter::model_label(self.models.agent_llm.as_ref()),
            large
        );

        let attempt = || -> anyhow::Result<MCPExecutionResult> {
            let ctx = self.build_context(user_message, mcp_servers, &options)?;
            let (mut dag, mut handlers) = self.plan(&ctx);
            let results = self.run(&ctx, &mut dag, &mut handlers);
            Ok(self.reason(&ctx, &results))
        };
        match std::panic::catch_unwind(AssertUnwindSafe(attempt)) {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => {
                tracing::error!(target: LOG_TARGET, "MCP orchestration failed: {:?}", e);
                MCPExecutionResult { handled: false, ..Default::default() }
            }
            Err(payload) => {
                let message = payload
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "orchestration panicked".to_string());
                tracing::error!(target: LOG_TARGET, "MCP orchestration failed: {}", message);
                MCPExecutionResult { handled: false, ..Default::default() }
            }
        }
    }

    /// Build the shared orchestrator context from input parameters.
    ///
    /// `options.current_date` and `options.progress_callback` are not part of
    /// the context: the callback is installed on the router by `execute`, and
    /// (as in the Python source) the date is not forwarded onto the context.
    pub fn build_context(
        &self,
        user_message: &str,
        mcp_servers: &[MCPServer],
        options: &ExecuteOptions,
    ) -> anyhow::Result<SharedOrchestratorContext> {
        let available_servers = StepExecutor::build_available_servers(mcp_servers);
        let base_dir = match (options.cwd.as_deref(), options.project_dir.as_deref()) {
            (Some(cwd), _) if !cwd.is_empty() => cwd.to_string(),
            (_, Some(dir)) if !dir.is_empty() => dir.to_string(),
            _ => std::env::current_dir()?.to_string_lossy().to_string(),
        };
        let effective_cwd = expand_home(&base_dir);

        let mut mcp_tool_contexts: Vec<MCPToolContext> = Vec::new();
        for srv in &available_servers {
            // Include the system tools server ("smartloop") here so its
            // built-in tools (file-read, file-write, bash, …) resolve through
            // the same ```mcp-block path as third-party servers like fetch.
            // Execution is server-agnostic: find_server/call_tool dispatch
            // against the full mcp_servers list regardless of server type.
            let server_id = match srv.get("id") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Null) | None => String::new(),
                Some(other) => other.to_string(),
            };
            let server_name = match srv.get("name") {
                Some(Value::String(s)) => s.clone(),
                None => "Unknown".to_string(),
                Some(Value::Null) => String::new(),
                Some(other) => other.to_string(),
            };
            let tools: Vec<Map<String, Value>> = srv
                .get("tools")
                .and_then(Value::as_array)
                .map(|tools| tools.iter().filter_map(|t| t.as_object().cloned()).collect())
                .unwrap_or_default();
            mcp_tool_contexts.push(MCPToolContext { server_id, server_name, tools });
        }

        let server_maps: Vec<Map<String, Value>> =
            available_servers.iter().filter_map(|s| s.as_object().cloned()).collect();
        let enriched = self.explorer.enrich(Some(&server_maps), options.conversation_history.as_deref());

        let mut ctx = OrchestratorContext::new(
            user_message,
            mcp_servers.to_vec(),
            available_servers,
            effective_cwd,
        );
        ctx.skills = options.skills.clone().unwrap_or_default();
        ctx.requested_skill = options.requested_skill.clone();
        ctx.document_context = options.document_context.clone();
        ctx.project_dir = options.project_dir.clone();
        ctx.model_service = options.model_service.clone();
        ctx.conversation_history = options.conversation_history.clone();
        ctx.has_attachments = options.has_attachments;
        ctx.mcp_tool_contexts = mcp_tool_contexts;
        ctx.enriched = Some(enriched);
        ctx.available_capabilities = options.available_capabilities.clone().unwrap_or_default();
        ctx.cancelled = options.cancelled.clone();
        Ok(SharedCell::wrap(ctx))
    }

    /// Phase 1: build the execution DAG (skill match → collect → route) and
    /// its handler map.
    pub fn plan(&self, ctx: &SharedOrchestratorContext) -> (DAG, HashMap<String, NodeHandler>) {
        let mut dag = DAG::new();

        let skill_match_node = SkillMatchNode::new(&[]);
        dag.add_node(skill_match_node.clone());

        let collect_node = CollectNode::new(&[&skill_match_node]);
        dag.add_node(collect_node.clone());

        let route_node = RouteNode::new(&[&skill_match_node, &collect_node]);
        dag.add_node(route_node.clone());

        let handlers = self.build_handlers(ctx, &skill_match_node, &collect_node, &route_node);
        (dag, handlers)
    }

    /// Phase 2: walk the DAG and accumulate raw results from each node.
    pub fn run(
        &self,
        _ctx: &SharedOrchestratorContext,
        dag: &mut DAG,
        handlers: &mut HashMap<String, NodeHandler>,
    ) -> IndexMap<String, NodeResult> {
        self.dag_executor.execute(dag, handlers)
    }

    /// Phase 3: analyze accumulated results and produce the final response.
    pub fn reason(
        &self,
        ctx: &SharedOrchestratorContext,
        results: &IndexMap<String, NodeResult>,
    ) -> MCPExecutionResult {
        let unhandled = || MCPExecutionResult { handled: false, ..Default::default() };
        let Some(fmt) = results.get(REASON_NODE.as_str()) else {
            return unhandled();
        };
        let Some(data) = fmt.data.as_object().filter(|_| fmt.success) else {
            return unhandled();
        };

        let accumulated: Vec<Map<String, Value>> = data
            .get("accumulated")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|e| e.as_object().cloned()).collect())
            .unwrap_or_default();
        let is_conversation = data.get("is_conversation").map(truthy).unwrap_or(false);
        let synthesized = data.get("synthesized").and_then(Value::as_str).filter(|s| !s.is_empty());

        let Some(search_ctx) = search_ctx_of(fmt) else {
            return unhandled();
        };

        let tool_name_of = |r: &Map<String, Value>| r.get("tool_name").and_then(Value::as_str).unwrap_or("").to_string();
        let content_of = |r: &Map<String, Value>| r.get("content").and_then(Value::as_str).unwrap_or("").to_string();

        let tool_names_used: Vec<String> = accumulated.iter().map(tool_name_of).collect();
        let mut explore_citations: Option<Vec<Value>> = search_ctx
            .citations
            .as_ref()
            .map(|cs| cs.iter().cloned().map(Value::Object).collect());

        if let Some(dag_citations) = data.get("citations").and_then(Value::as_array).filter(|a| !a.is_empty()) {
            explore_citations = Some(dag_citations.clone());
        }

        if let (Some(cited_docs), Some(citations)) = (search_ctx.cited_documents.as_ref(), explore_citations.as_ref()) {
            if !cited_docs.is_empty() && !citations.is_empty() {
                let allowed: HashSet<&String> = cited_docs.iter().collect();
                explore_citations = Some(
                    citations
                        .iter()
                        .filter(|c| match c.get("document_name") {
                            Some(Value::String(name)) if !name.is_empty() => allowed.contains(name),
                            Some(Value::Null) | None => true,
                            Some(other) => !truthy(other),
                        })
                        .cloned()
                        .collect(),
                );
            }
        }

        let non_explore: Vec<String> =
            accumulated.iter().filter(|r| tool_name_of(r) != EXPLORE_TOOL).map(content_of).collect();
        let explore_parts: Vec<String> =
            accumulated.iter().filter(|r| tool_name_of(r) == EXPLORE_TOOL).map(content_of).collect();
        let has_explore_output = explore_parts.iter().any(|c| !c.is_empty());

        // Citations track which indexed documents contributed to the
        // response.  Include them when the explore step produced content
        // (documents were read and fed into the answer).  Suppress them only
        // when tool output is the sole data source and no documents were part
        // of the pipeline.
        let keep_citations = has_explore_output;

        if let Some(synthesized) = synthesized {
            return ctx.with_mut(|c| {
                self.response.build(
                    c,
                    &search_ctx,
                    &tool_names_used,
                    None,
                    synthesized,
                    if keep_citations { explore_citations.clone() } else { None },
                    is_conversation,
                    true,
                )
            });
        }

        let tool_context = non_explore.join("\n\n---\n\n");

        if !tool_context.is_empty() {
            return ctx.with_mut(|c| {
                self.response.build(c, &search_ctx, &tool_names_used, None, &tool_context, None, is_conversation, false)
            });
        }

        let mut vector_context = search_ctx.vector_context.clone();
        if !explore_parts.is_empty() && vector_context.is_empty() {
            vector_context = explore_parts.join("\n\n---\n\n");
        }

        ctx.with_mut(|c| {
            self.response.build(
                c,
                &search_ctx,
                &tool_names_used,
                Some(&vector_context),
                &tool_context,
                explore_citations.clone(),
                is_conversation,
                false,
            )
        })
    }

    /// Build the handler map for the initial DAG nodes.
    fn build_handlers(
        &self,
        ctx: &SharedOrchestratorContext,
        skill_match_node: &AgentNode,
        collect_node: &AgentNode,
        route_node: &AgentNode,
    ) -> HashMap<String, NodeHandler> {
        let mut handlers: HashMap<String, NodeHandler> = HashMap::new();

        {
            let skills = Arc::clone(&self.skills);
            let ctx = ctx.clone();
            let node = skill_match_node.clone();
            handlers.insert(
                skill_match_node.node_id.clone(),
                Arc::new(move |_deps| Ok(ctx.with_mut(|c| skills.match_skills(c, &node)))),
            );
        }

        {
            let pipeline = Arc::clone(&self.pipeline);
            let ctx = ctx.clone();
            let node = collect_node.clone();
            let skill_match_node = skill_match_node.clone();
            let collect_override = self.collect_override.lock().clone();
            handlers.insert(
                collect_node.node_id.clone(),
                Arc::new(move |deps| {
                    if let Some(collect) = collect_override.as_ref() {
                        let search_ctx = ctx.with_ref(|c| collect(c));
                        return Ok(NodeResult::ok(node.node_id.clone()).with_payload(search_ctx));
                    }
                    Ok(pipeline.handle_collect(&ctx, &deps, &node, &skill_match_node))
                }),
            );
        }

        {
            let pipeline = Arc::clone(&self.pipeline);
            let ctx = ctx.clone();
            let node = route_node.clone();
            let skill_match_node = skill_match_node.clone();
            let collect_node = collect_node.clone();
            handlers.insert(
                route_node.node_id.clone(),
                Arc::new(move |deps| Ok(pipeline.handle_route(&ctx, &deps, &node, &skill_match_node, &collect_node))),
            );
        }

        handlers
    }
}

/// Python truthiness of a JSON value.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// The search context a reason result carries (Python `data["search_ctx"]`):
/// the shared context as its payload, a plain [`SearchContext`] payload, or
/// the context serialized under `data["search_ctx"]`.
fn search_ctx_of(result: &NodeResult) -> Option<SearchContext> {
    if let Some(shared) = result.payload_as::<SharedSearchContext>() {
        return Some(shared.with_ref(|s| s.clone()));
    }
    if let Some(plain) = result.payload_as::<SearchContext>() {
        return Some(plain.clone());
    }
    match result.data.get("search_ctx") {
        Some(Value::Null) | None => None,
        Some(v) => serde_json::from_value::<SearchContext>(v.clone()).ok(),
    }
}
