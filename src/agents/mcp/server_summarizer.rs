//! Port of `smartloop/agents/mcp/server_summarizer.py`.
//!
//! MCP Server Summarizer — generates a concise summary of an MCP server and its
//! tools to help the LLM pick the right tool at selection time.

#![allow(dead_code, unused_variables, unused_imports)]

use std::sync::Arc;

use serde_json::{Map, Value};

use crate::llms::BaseLLM;
use crate::workspace::workspace::MCPServer;

/// Logger name of the Python module.
pub const LOGGER: &str = "smartloop.agents.mcp.server_summarizer";

/// Summarize an MCP server and its tools into a short paragraph.
pub struct MCPServerSummarizer {
    pub llm: Arc<dyn BaseLLM>,
}

impl MCPServerSummarizer {
    pub fn new(llm: Arc<dyn BaseLLM>) -> Self {
        Self { llm }
    }

    /// Generate a short description and topical tags for an MCP server.
    ///
    /// Returns `{"description": str, "tags": [str]}` (either key may be
    /// absent) — an empty map on failure so the caller can degrade gracefully.
    pub fn generate_metadata(&self, server: &MCPServer) -> Map<String, Value> {
        todo!()
    }

    /// Generate a concise summary for the given MCP server, or `None` if
    /// summarization fails or the server has no tools.
    pub fn invoke(&self, server: &MCPServer) -> Option<String> {
        todo!()
    }
}
