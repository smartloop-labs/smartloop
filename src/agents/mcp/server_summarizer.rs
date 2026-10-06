//! Port of `smartloop/agents/mcp/server_summarizer.py`.
//!
//! MCP Server Summarizer — generates a concise summary of an MCP server and its
//! tools to help the LLM pick the right tool at selection time.

use std::sync::Arc;

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{Map, Value};

use crate::agents::mcp::mcp_agent::py_str;
use crate::llms::BaseLLM;
use crate::workspace::workspace::MCPServer;

/// Logger name of the Python module.
pub const LOGGER: &str = "smartloop.agents.mcp.server_summarizer";

/// The first `n` characters of `s`.
fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// `"- name: description"` per tool, `no description` when it has none.
fn tools_description(server: &MCPServer) -> String {
    server
        .tools
        .iter()
        .map(|t| {
            let desc = t.description.as_deref().filter(|d| !d.is_empty()).unwrap_or("no description");
            format!("- {}: {}", t.name, desc)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

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
    /// Used to enrich hand-added connections (e.g. `uvx` packages) that
    /// connect without advertising a description.  Returns
    /// `{"description": str, "tags": [str]}` (either key may be absent) — an
    /// empty map on failure so the caller can degrade gracefully.  Mirrors the
    /// JSON-parsing approach used for skill metadata generation.
    pub fn generate_metadata(&self, server: &MCPServer) -> Map<String, Value> {
        if server.tools.is_empty() {
            return Map::new();
        }

        let tools_desc = tools_description(server);
        let prompt = format!(
            "<system>\n\
             You describe MCP tool servers. Reply with ONLY a JSON object — \
             no markdown, no explanation.\n\
             </system>\n\
             <user>\n\
             Given this MCP server and its tools, output a JSON object with:\n\
             - \"description\": one clear sentence describing what this server does \
             and what its tools let the assistant do, based only on the tools \
             listed below.\n\
             - \"tags\": an array of 1 to 3 short, lowercase, single-word topical \
             tags derived from this server's tools.\n\n\
             Name: {name}\n\
             Package: {pkg}\n\
             Existing description: {description}\n\
             Tools:\n{tools_desc}\n\
             </user>",
            name = server.name,
            pkg = server.pkg.as_deref().filter(|p| !p.is_empty()).unwrap_or("none"),
            description = server.description.as_deref().filter(|d| !d.is_empty()).unwrap_or("none"),
        );

        let raw = self.llm.invoke(&prompt, None, None).trim().to_string();
        tracing::info!(
            target: LOGGER,
            "MCP metadata LLM response for '{}': {}",
            server.name,
            take_chars(&raw, 200)
        );

        match Self::parse_metadata(&raw) {
            Ok(result) => result,
            Err(e) => {
                tracing::warn!(
                    target: LOGGER,
                    "Failed to generate metadata for MCP server '{}': {}",
                    server.name,
                    e
                );
                Map::new()
            }
        }
    }

    /// Pull `description`/`tags` out of the model's reply.
    fn parse_metadata(raw: &str) -> anyhow::Result<Map<String, Value>> {
        static BAD_ESCAPE: Lazy<Regex> =
            Lazy::new(|| Regex::new(r#"\\([^"\\/bfnrtu])"#).expect("valid regex"));

        let mut raw = raw.to_string();
        // Strip markdown fences
        if raw.starts_with("```") {
            raw = raw
                .split('\n')
                .filter(|ln| !ln.trim().starts_with("```"))
                .collect::<Vec<_>>()
                .join("\n")
                .trim()
                .to_string();
        }

        // Parse the FIRST JSON object from the first "{", ignoring any
        // trailing text or extra objects the model appends (a streaming decode
        // stops at the end of the first value — a whole-text parse would fail
        // with "trailing characters" on those).
        let Some(start) = raw.find('{') else {
            return Ok(Map::new());
        };
        let raw = &raw[start..];
        // Fix invalid JSON escapes that LLMs produce (e.g. \_ \*)
        let fixed = BAD_ESCAPE.replace_all(raw, "$1");
        let meta: Value = serde_json::Deserializer::from_str(&fixed)
            .into_iter::<Value>()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Expecting value"))??;

        let mut result = Map::new();
        if let Some(Value::String(description)) = meta.get("description") {
            if !description.trim().is_empty() {
                // Collapse newlines/control chars/runs of whitespace into single
                // spaces so the stored description is always clean one-line text.
                let clean = description.split_whitespace().collect::<Vec<_>>().join(" ");
                result.insert("description".to_string(), Value::String(clean));
            }
        }
        if let Some(Value::Array(tags)) = meta.get("tags") {
            let clean: Vec<Value> = tags
                .iter()
                .map(|t| py_str(t).trim().to_string())
                .filter(|t| !t.is_empty())
                .map(|t| Value::String(t.to_lowercase()))
                .take(3)
                .collect();
            if !clean.is_empty() {
                result.insert("tags".to_string(), Value::Array(clean));
            }
        }
        Ok(result)
    }

    /// Generate a concise summary for the given MCP server, or `None` if
    /// summarization fails or the server has no tools.
    pub fn invoke(&self, server: &MCPServer) -> Option<String> {
        if server.tools.is_empty() {
            return None;
        }

        let tools_desc = tools_description(server);
        let prompt = format!(
            "<system>You summarize MCP tool servers in one concise paragraph. \
             Focus on what the server does and what its tools can accomplish. \
             Keep it under 100 words.</system>\n\
             <user>Summarize this MCP server:\n\
             Name: {name}\n\
             Description: {description}\n\
             Tools:\n{tools_desc}</user>",
            name = server.name,
            description = server.description.as_deref().filter(|d| !d.is_empty()).unwrap_or("none"),
        );

        let prompt_tokens = self.llm.get_num_tokens(&prompt);
        tracing::info!(
            target: LOGGER,
            "Server summary prompt ({} tokens) for '{}'",
            prompt_tokens,
            server.name
        );

        let response = self.llm.invoke(&prompt, None, None);
        if response.is_empty() {
            return None;
        }
        let summary = response.trim().to_string();
        if !summary.is_empty() {
            let response_tokens = self.llm.get_num_tokens(&summary);
            tracing::info!(
                target: LOGGER,
                "Generated summary for MCP server '{}' ({} tokens): {}",
                server.name,
                response_tokens,
                take_chars(&summary, 100)
            );
        }
        Some(summary)
    }
}
