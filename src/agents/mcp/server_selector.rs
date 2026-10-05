//! Port of `smartloop/agents/mcp/server_selector.py`.
//!
//! MCP Server Selector Agent - Selects the appropriate MCP tool based on user
//! request.  Builds a unified context (user input + skills + available tools)
//! and lets the LLM decide whether the request maps to a tool call and, if so,
//! which tool to use.

#![allow(dead_code, unused_variables, unused_imports)]

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::llms::BaseLLM;

/// Logger name of the Python module.
pub const LOGGER: &str = "smartloop.agents.mcp.server_selector";

// Tool name constants for built-in tools (registered via System Tools server).
pub const BASH_TOOL: &str = "bash";
pub const GREP_TOOL: &str = "grep";
pub const FILE_READ_TOOL: &str = "file-read";
pub const FILE_WRITE_TOOL: &str = "file-write";
pub const EXPLORE_TOOL: &str = "explore";

fn default_tool_index() -> i64 {
    -1
}

/// MCP tool selection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolSelection {
    /// Execution order of the tool.
    pub order: i64,
    /// Unique identifier of the MCP server.
    #[serde(default)]
    pub server_id: String,
    /// Name of the MCP server.
    pub server_name: String,
    /// Name of the tool to execute.
    pub tool_name: String,
    /// Index of the tool in the server's tool list.
    #[serde(default = "default_tool_index")]
    pub tool_index: i64,
    /// Confidence score (0.0..=1.0).
    #[serde(default)]
    pub confidence: f64,
}

impl Default for ToolSelection {
    fn default() -> Self {
        Self {
            order: 0,
            server_id: String::new(),
            server_name: String::new(),
            tool_name: String::new(),
            tool_index: -1,
            confidence: 0.0,
        }
    }
}

/// MCP tool selection result.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ToolSelectionResult {
    /// List of selected tools with details.
    pub selections: Vec<ToolSelection>,
}

/// One numbered catalogue entry: which server/tool the number stands for.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ToolLookupEntry {
    pub server_id: String,
    pub server_name: String,
    pub tool_name: String,
    pub tool_index: i64,
}

/// `(tools_list, tool_lookup, server_summaries)` from
/// [`MCPServerSelector::build_tool_catalogue`].
pub type ToolCatalogue = (Vec<String>, HashMap<i64, ToolLookupEntry>, Vec<String>);

/// Agent that analyzes user requests and selects the most appropriate MCP
/// server and tool, with a single prompt layering
/// user input → skills / instructions → available MCP tools.
pub struct MCPServerSelector {
    pub llm: Arc<dyn BaseLLM>,
}

impl MCPServerSelector {
    pub fn new(llm: Arc<dyn BaseLLM>) -> Self {
        Self { llm }
    }

    /// Build the numbered tool catalogue from available server dicts
    /// (`id`, `name`, `description`, `summary`, `tools`).
    pub fn build_tool_catalogue(&self, available_servers: &[Value], skills: Option<&[String]>) -> ToolCatalogue {
        todo!()
    }

    /// Analyze the user's message and select the most appropriate MCP tool.
    /// Returns zero or one selections; never fails.
    pub fn invoke(
        &self,
        user_message: &str,
        available_servers: &[Value],
        skills: Option<&[String]>,
        cwd: Option<&str>,
    ) -> ToolSelectionResult {
        todo!()
    }

    /// Decide whether a follow-up tool call is needed based on accumulated
    /// results (`tool_name`, `formatted_result`).  Returns zero or one selections.
    pub fn invoke_followup(
        &self,
        user_message: &str,
        available_servers: &[Value],
        accumulated_results: &[HashMap<String, String>],
        skills: Option<&[String]>,
    ) -> ToolSelectionResult {
        todo!()
    }

    /// Extract a tool number from the LLM response (plain number, JSON
    /// `{"tool_number": N}` / list thereof, or prose containing a number).
    pub fn parse_tool_number(text: &str, tool_lookup: &HashMap<i64, ToolLookupEntry>) -> Option<ToolLookupEntry> {
        todo!()
    }
}
