//! Port of `smartloop/agents/orchestration/step_executor.py`.
//!
//! Running one planned step: resolve arguments, then call.
//!
//! The step types split two ways.  A TOOL step needs its arguments worked out
//! first — which server, which tool, what values — and
//! [`StepExecutor::resolve_tool_args`] does that before
//! [`StepExecutor::call_tool`] runs it.  Every other type (explore, fetch,
//! conversational, action) executes directly in [`StepExecutor::execute`].
//!
//! Argument resolution is the hard half and most of the volume here: values
//! have to come from the user's message and upstream results, never from the
//! tool's own description, or a small model copies the schema back as data.

#![allow(unused_imports, unused_variables)]

use std::sync::Arc;

use serde_json::{Map, Value};

use crate::agents::mcp::argument_extractor::MCPArgumentExtractor;
use crate::agents::mcp::dependency_analyzer::MCPDependencyAnalyzer;
use crate::agents::mcp::explorer::Explorer;
use crate::agents::mcp::mcp_agent::{missing_required, MCPAgent};
use crate::agents::mcp::mcp_tool_caller::MCPToolCaller;
use crate::agents::mcp::server_selector::{
    BASH_TOOL, EXPLORE_TOOL, FILE_READ_TOOL, FILE_WRITE_TOOL, GREP_TOOL,
};
use crate::agents::nodes::NodeResult;
use crate::agents::orchestration::context::OrchestratorContext;
use crate::agents::orchestration::model_router::DECISION_MAX_TOKENS;
use crate::agents::orchestration::response_builder::ResponseBuilder;
use crate::agents::planner::{
    mcp_connection_is_usable, PlanStep, StepType, SYSTEM_TOOLS_SERVER_ID,
    SYSTEM_TOOLS_SERVER_NAME,
};
use crate::llms::BaseLLM;
use crate::mcp::client::MCPToolCallResult;
use crate::paths::expand_home;
use crate::utils::mcp_result_formatter::ResultFormatter;
use crate::workspace::workspace::{MCPServer, MCPServerType};

/// Python logger name.
pub const LOGGER: &str = "smartloop.agents.orchestration.step_executor";

/// Build the NodeResult for one executed plan step.
///
/// Each node stores its own output — the next node reads only its direct
/// dependency's `raw` field, which is what keeps the context focused instead
/// of dragging the whole accumulated chain forward.
///
/// `data` = `{"tool_name", "content", "raw", "is_tool_result"}`.
pub fn plan_step_result(
    node_id: &str,
    step: &PlanStep,
    content: &str,
    raw: Option<&str>,
) -> NodeResult {
    todo!()
}

/// Resolves and runs individual plan steps.
pub struct StepExecutor {
    pub orchestrator_llm: Arc<dyn BaseLLM>,
    pub tool_caller: Arc<MCPToolCaller>,
    pub arg_extractor: Arc<MCPArgumentExtractor>,
    pub dep_analyzer: Arc<MCPDependencyAnalyzer>,
    pub result_formatter: Arc<ResultFormatter>,
    pub explorer: Arc<Explorer>,
    mcp_agent: Arc<MCPAgent>,
    response: Arc<ResponseBuilder>,
}

impl StepExecutor {
    /// Wire a step executor to the services it calls into.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        orchestrator_llm: Arc<dyn BaseLLM>,
        tool_caller: Arc<MCPToolCaller>,
        arg_extractor: Arc<MCPArgumentExtractor>,
        dep_analyzer: Arc<MCPDependencyAnalyzer>,
        result_formatter: Arc<ResultFormatter>,
        explorer: Arc<Explorer>,
        mcp_agent: Arc<MCPAgent>,
        response: Arc<ResponseBuilder>,
    ) -> Self {
        Self {
            orchestrator_llm,
            tool_caller,
            arg_extractor,
            dep_analyzer,
            result_formatter,
            explorer,
            mcp_agent,
            response,
        }
    }

    /// Thin forwarding to [`MCPToolCaller::call_tool`].
    pub fn call_tool_for_server(
        &self,
        server: &MCPServer,
        tool_name: &str,
        arguments: Option<&Map<String, Value>>,
        access_token: Option<&str>,
        cwd: Option<&str>,
    ) -> MCPToolCallResult {
        todo!()
    }

    /// Evaluate a step condition against previous results.
    ///
    /// Supported formats:
    /// - `result_contains:<text>` — case-insensitive substring check on any
    ///   previous result text
    /// - `result_contains_any:<w1,w2,w3>` — true if ANY keyword matches
    /// - `llm:<question>` — sends previous results + question to LLM, expects
    ///   YES/NO
    /// - empty — returns `true` (always execute)
    pub fn evaluate_condition(&self, condition: Option<&str>, previous_results: &[Value]) -> bool {
        todo!()
    }

    /// Group plan steps into batches that can run in parallel.
    ///
    /// A step is independent when it has no `condition`, no explicit
    /// dependency on previous results, and is not inherently sequential
    /// (e.g. write/summarize actions depend on prior context).
    ///
    /// Consecutive independent steps are grouped together; steps with
    /// conditions, dependencies, or inherently-sequential types form
    /// singleton groups that flush any accumulated parallel batch.
    pub fn group_independent_steps(steps: &[PlanStep]) -> Vec<Vec<PlanStep>> {
        todo!()
    }

    /// Run one non-TOOL step and return `(formatted, raw)` or `None`.
    ///
    /// Everything constant for the request comes off `ctx`; only what varies
    /// per step (the step itself, its upstream results, the document context
    /// in force, and the resolved model) is passed in.
    pub fn execute(
        &self,
        ctx: &OrchestratorContext,
        step: &PlanStep,
        previous_results: Option<&[Value]>,
        document_context: Option<&str>,
        llm: Option<Arc<dyn BaseLLM>>,
    ) -> Option<(String, String)> {
        todo!()
    }

    /// Resolve tool arguments: find server, run deps, extract args.
    ///
    /// Returns `(server, tool_name, arguments)` or `None` if resolution
    /// fails.  May re-pin a mis-assigned system tool onto the smartloop
    /// server, which is why `step` is mutable.
    pub fn resolve_tool_args(
        &self,
        ctx: &OrchestratorContext,
        step: &mut PlanStep,
        previous_results: Option<&[Value]>,
        llm: Option<Arc<dyn BaseLLM>>,
    ) -> Option<(MCPServer, String, Map<String, Value>)> {
        todo!()
    }

    /// Call an MCP tool with pre-resolved arguments.
    ///
    /// `cwd` is forwarded to local (stdio) servers so built-in system tools
    /// resolve relative paths against the project work dir.
    ///
    /// Returns `(formatted_content, raw_content)` or `None` on failure.
    pub fn call_tool(
        &self,
        server: &MCPServer,
        tool_name: &str,
        arguments: &Map<String, Value>,
        cwd: Option<&str>,
    ) -> Option<(String, String)> {
        todo!()
    }

    /// The planner-facing catalogue of usable connections: every enabled,
    /// non-failed server with tools, as
    /// `{"id", "name", "description", "summary", "server_type", "tools": [{"name", "description", "input_schema"}]}`.
    ///
    /// Reads nothing off `self` (Python tests call it unbound), so it is an
    /// associated function.
    pub fn build_available_servers(mcp_servers: &[MCPServer]) -> Vec<Value> {
        todo!()
    }
}
