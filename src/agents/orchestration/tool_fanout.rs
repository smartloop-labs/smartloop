//! Tool fan-out — calling one tool once per item.
//!
//! Port of `smartloop/agents/orchestration/tool_fanout.py`.
//!
//! When a plan step is marked `for_each` ("for each fund, get the ticker
//! info"), the list of items has to be recovered from upstream context before
//! the tool can run at all.  That extraction is the awkward part: upstream
//! results are free text or JSON from another tool, and a small model asked for
//! a list will happily return prose, a numbered list, or a JSON array.
//!
//! All the calls for one step collapse into a single DAG node.  They share the
//! same tool and differ only by argument, so fanning them across nodes would buy
//! nothing and cost a node each.

#![allow(unused_imports, unused_variables, dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{Map, Value};

use crate::agents::mcp::explorer::SearchContext;
use crate::agents::mcp::mcp_tool_caller::MCPToolCaller;
use crate::agents::nodes::{NodeHandler, NodeResult};
use crate::agents::orchestration::context::OrchestratorContext;
use crate::agents::orchestration::model_router::ModelRouter;
use crate::agents::orchestration::response_builder::ResponseBuilder;
use crate::agents::orchestration::step_executor::{plan_step_result, StepExecutor};
use crate::agents::planner::PlanStep;
use crate::workspace::workspace::MCPServer;

/// Python logger name.
const LOG_TARGET: &str = "smartloop.agents.orchestration.tool_fanout";

/// Runs one tool once per item extracted from upstream context.
pub struct ToolFanout {
    pub(crate) models: Arc<ModelRouter>,
    pub(crate) steps: Arc<StepExecutor>,
    pub(crate) response: Arc<ResponseBuilder>,
    pub tool_caller: Arc<MCPToolCaller>,
}

impl ToolFanout {
    /// Ceiling on items per fan-out.  The list comes from an LLM reading
    /// upstream context, so a degenerate answer could otherwise queue an
    /// unbounded number of tool calls.
    pub const FANOUT_MAX_ITEMS: usize = 20;

    /// Build the fan-out service.
    pub fn new(
        models: Arc<ModelRouter>,
        steps: Arc<StepExecutor>,
        response: Arc<ResponseBuilder>,
        tool_caller: Arc<MCPToolCaller>,
    ) -> Self {
        Self { models, steps, response, tool_caller }
    }

    /// Create a handler that extracts items and calls a tool once per item.
    ///
    /// Three phases: extract item list, resolve+execute per item, aggregate.
    ///
    /// When all arguments are pre-filled (common case: single inferred arg
    /// like `symbol`), the tool is called directly — no LLM-based dependency
    /// analysis or argument extraction per item.
    pub fn make_handler(
        self: &Arc<Self>,
        ctx: &Arc<Mutex<OrchestratorContext>>,
        step: &PlanStep,
        search_ctx: &Arc<Mutex<SearchContext>>,
    ) -> NodeHandler {
        todo!()
    }

    /// Locate the MCP server for a fan-out step.
    pub fn find_fanout_server(
        &self,
        step: &PlanStep,
        mcp_servers: &[MCPServer],
        available_servers: &[Map<String, Value>],
    ) -> Option<MCPServer> {
        todo!()
    }

    /// Extract iteration items from upstream context.
    ///
    /// Uses a structured LLM prompt that asks for one value per line (more
    /// reliable with small models than asking for JSON arrays).  Previous step
    /// results are included so the LLM can see the already-extracted data.
    ///
    /// Items are strings, or objects merged into the tool arguments.
    pub fn extract_fanout_items(
        &self,
        ctx: &OrchestratorContext,
        step: &PlanStep,
        prev_results: &[Map<String, Value>],
        search_ctx: &SearchContext,
    ) -> Vec<Value> {
        todo!()
    }

    /// Parse fan-out extraction response: one value per line or JSON array.
    pub fn parse_fanout_response(response: &str) -> Vec<String> {
        todo!()
    }
}
