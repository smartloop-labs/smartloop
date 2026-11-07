//! Port of `smartloop/agents/mcp/server_selector.py`.
//!
//! MCP Server Selector Agent - Selects the appropriate MCP tool based on user
//! request.  Builds a unified context (user input + skills + available tools)
//! and lets the LLM decide whether the request maps to a tool call and, if so,
//! which tool to use.

use std::collections::HashMap;
use std::sync::Arc;

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agents::mcp::mcp_agent::py_str;
use crate::llms::BaseLLM;

/// Logger name of the Python module.
pub const LOGGER: &str = "smartloop.agents.mcp.server_selector";

// Tool names: `explore` is the in-process explore step; bash, grep,
// file-read and file-write name the built-in tools smartloop no longer ships
// (the harness runs commands and writes files), kept to recognise them.
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

/// `d.get(key, default)` rendered the way an f-string renders it.
fn get_display(item: &Value, key: &str, default: &str) -> String {
    match item.get(key) {
        Some(v) => py_str(v),
        None => default.to_string(),
    }
}

/// Python truthiness of `d.get(key, "")`.
fn get_truthy(item: &Value, key: &str) -> bool {
    match item.get(key) {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// The first `n` characters of `s`.
fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Python's `int(value)` for the JSON shapes a model emits: ints, floats
/// (truncated), booleans, and strings holding an integer literal.
fn py_int(value: &Value) -> Option<i64> {
    match value {
        Value::Bool(b) => Some(i64::from(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(i)
            } else {
                let f = n.as_f64()?;
                if f.is_finite() {
                    Some(f.trunc() as i64)
                } else {
                    None
                }
            }
        }
        Value::String(s) => {
            let t = s.trim();
            let (sign, digits) = match t.strip_prefix('-') {
                Some(rest) => (-1, rest),
                None => (1, t.strip_prefix('+').unwrap_or(t)),
            };
            if digits.is_empty()
                || digits.starts_with('_')
                || digits.ends_with('_')
                || digits.contains("__")
                || !digits.chars().all(|c| c.is_ascii_digit() || c == '_')
            {
                return None;
            }
            digits.replace('_', "").parse::<i64>().ok().map(|n| sign * n)
        }
        _ => None,
    }
}

/// Agent that analyzes user requests and selects the most appropriate MCP
/// server and tool.
///
/// The selection flow builds up context in layers:
///   user input  →  skills / instructions  →  available MCP tools
/// and sends a single prompt to the LLM which decides whether the request
/// qualifies as a tool call and, if so, which tool to invoke.
pub struct MCPServerSelector {
    pub llm: Arc<dyn BaseLLM>,
}

impl MCPServerSelector {
    pub fn new(llm: Arc<dyn BaseLLM>) -> Self {
        Self { llm }
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Build the numbered tool catalogue from available server dicts
    /// (`id`, `name`, `description`, `summary`, `tools`).
    ///
    /// Returns `(tools_list, tool_lookup, server_summaries)` where
    /// `tools_list` is a list of formatted label strings, `tool_lookup` maps
    /// index to server/tool info, and `server_summaries` is a list of summary
    /// strings.
    pub fn build_tool_catalogue(&self, available_servers: &[Value], _skills: Option<&[String]>) -> ToolCatalogue {
        let mut tools_list: Vec<String> = Vec::new();
        let mut tool_lookup: HashMap<i64, ToolLookupEntry> = HashMap::new();
        let mut server_summaries: Vec<String> = Vec::new();
        let mut idx: i64 = 0;

        for server in available_servers {
            let server_id = get_display(server, "id", "");
            let server_name = get_display(server, "name", "Unknown");
            let server_desc = get_display(server, "description", "");
            let has_desc = get_truthy(server, "description");
            if get_truthy(server, "summary") {
                let server_summary = get_display(server, "summary", "");
                server_summaries.push(format!("- {server_name}: {server_summary}"));
            }
            let tools: &[Value] = match server.get("tools") {
                Some(Value::Array(t)) => t.as_slice(),
                _ => &[],
            };
            for (ti, tool) in tools.iter().enumerate() {
                idx += 1;
                let t_name = get_display(tool, "name", "unknown");
                tool_lookup.insert(
                    idx,
                    ToolLookupEntry {
                        server_id: server_id.clone(),
                        server_name: server_name.clone(),
                        tool_name: t_name.clone(),
                        tool_index: ti as i64,
                    },
                );
                let mut label = format!("[{idx}] {server_name} — {t_name}");
                if get_truthy(tool, "description") {
                    label.push_str(&format!(": {}", get_display(tool, "description", "")));
                } else if has_desc {
                    label.push_str(&format!(" ({server_desc})"));
                }
                tools_list.push(label);
            }
        }

        (tools_list, tool_lookup, server_summaries)
    }

    /// Analyze the user's message and select the most appropriate MCP tool.
    ///
    /// `available_servers` are server dicts with `name`, `tools` keys;
    /// `skills` are optional skill content strings that provide routing hints
    /// or define special actions (e.g. bash).  Returns zero or one
    /// selections; never fails.
    pub fn invoke(
        &self,
        user_message: &str,
        available_servers: &[Value],
        skills: Option<&[String]>,
        cwd: Option<&str>,
    ) -> ToolSelectionResult {
        let skills = skills.unwrap_or(&[]);
        if available_servers.is_empty() && skills.is_empty() {
            tracing::info!(target: LOGGER, "No MCP tools or skills available for selection");
            return ToolSelectionResult::default();
        }

        // --- Build numbered tool catalogue ---
        let (tools_list, tool_lookup, server_summaries) =
            self.build_tool_catalogue(available_servers, Some(skills));

        if tools_list.is_empty() {
            return ToolSelectionResult::default();
        }

        let tools_text = tools_list.join("\n");

        // --- Build skills section ---
        let mut skills_section = String::new();
        if !skills.is_empty() {
            let skills_block = skills.iter().map(|s| format!("- {s}")).collect::<Vec<_>>().join("\n");
            skills_section = format!(
                "Instructions / Skills (use these to decide which tool fits best, \
                 or whether the request requires a tool at all):\n{skills_block}\n\n"
            );
        }

        // --- Build server summaries section ---
        let mut summaries_section = String::new();
        if !server_summaries.is_empty() {
            summaries_section = format!("Server summaries:\n{}\n\n", server_summaries.join("\n"));
        }

        // --- Build cwd section ---
        let mut cwd_section = String::new();
        if let Some(cwd) = cwd.filter(|c| !c.is_empty()) {
            cwd_section = format!(
                "Working directory: {cwd}\n\
                 IMPORTANT: When the user refers to local files, directories, \
                 README, codebase, or project structure, use the smartloop \
                 tools (file-read, grep, explore) — NOT fetch or web tools. \
                 Only use fetch/web tools for URLs (http/https).\n\n"
            );
        }

        // --- Single LLM call with full context ---
        let prompt = format!(
            "User request: {user_message}\n\n\
             {cwd_section}\
             {skills_section}\
             {summaries_section}\
             Available tools:\n{tools_text}\n\n\
             Based on the user request and the instructions above, which tool \
             number should be used? If no tool is needed, reply with 0.\n\n\
             Reply with ONLY the tool number."
        );

        let prompt_tokens = self.llm.get_num_tokens(&prompt);
        tracing::info!(
            target: LOGGER,
            "Tool selection prompt ({} tools, {} skills, {} tokens):\n{}",
            tools_list.len(),
            skills.len(),
            prompt_tokens,
            prompt
        );

        let response = self.llm.invoke(&prompt, None, None);
        let response_text = response.trim().to_string();
        let response_tokens = self.llm.get_num_tokens(&response_text);

        tracing::info!(
            target: LOGGER,
            "Tool selection LLM response ({} tokens): {}",
            response_tokens,
            response_text
        );

        // Parse the response — try extracting a tool number
        let Some(selected) = Self::parse_tool_number(&response_text, &tool_lookup) else {
            tracing::info!(target: LOGGER, "LLM selected no tool from response: {}", response_text);
            return ToolSelectionResult::default();
        };

        let selection = ToolSelection {
            order: 1,
            server_id: selected.server_id,
            server_name: selected.server_name,
            tool_name: selected.tool_name,
            tool_index: selected.tool_index,
            confidence: 0.85,
        };
        tracing::info!(
            target: LOGGER,
            "Selected tool: {}/{} (confidence={})",
            selection.server_name,
            selection.tool_name,
            selection.confidence
        );
        ToolSelectionResult { selections: vec![selection] }
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
        if available_servers.is_empty() {
            return ToolSelectionResult::default();
        }

        let (tools_list, tool_lookup, _server_summaries) =
            self.build_tool_catalogue(available_servers, skills);

        if tools_list.is_empty() {
            return ToolSelectionResult::default();
        }

        let tools_text = tools_list.join("\n");

        // Build accumulated results section (truncate each to ~2000 chars)
        let mut results_parts: Vec<String> = Vec::new();
        let mut already_called: Vec<String> = Vec::new();
        for r in accumulated_results {
            let t_name = r.get("tool_name").cloned().unwrap_or_else(|| "unknown".to_string());
            if !already_called.contains(&t_name) {
                already_called.push(t_name.clone());
            }
            let mut result_text = r.get("formatted_result").cloned().unwrap_or_default();
            if result_text.chars().count() > 2000 {
                result_text = take_chars(&result_text, 2000) + "\n... (truncated)";
            }
            results_parts.push(format!("Tool: {t_name}\nResult:\n{result_text}"));
        }

        let results_section = results_parts.join("\n\n");
        let already_called_text = already_called.join(", ");

        // Include skills so they guide follow-up tool decisions
        let mut skills_section = String::new();
        if let Some(skills) = skills.filter(|s| !s.is_empty()) {
            let skills_block = skills.iter().map(|s| format!("- {s}")).collect::<Vec<_>>().join("\n");
            skills_section = format!(
                "Instructions / Skills (use these to decide what to do next):\n{skills_block}\n\n"
            );
        }

        let prompt = format!(
            "User's original request: {user_message}\n\n\
             {skills_section}\
             Tools already executed and their results:\n{results_section}\n\n\
             Available tools:\n{tools_text}\n\n\
             Based on the user's request, the instructions above, and the tool \
             results, does the request need another tool call to be fully answered?\n\
             - If the instructions or results indicate a follow-up tool would add \
             meaningful value (e.g., fetching content for URLs found in results, \
             looking up details for items discovered, expanding or enriching the \
             data), reply with the tool number.\n\
             - If the results already fully answer the request, or no available \
             tool would help, reply with 0.\n\
             - Do NOT re-call tools that were already used: {already_called_text}\n\n\
             Reply with ONLY the tool number."
        );

        let prompt_tokens = self.llm.get_num_tokens(&prompt);
        tracing::info!(
            target: LOGGER,
            "Follow-up selection prompt ({} tools, {} prior results, {} tokens)",
            tools_list.len(),
            accumulated_results.len(),
            prompt_tokens
        );

        let response = self.llm.invoke(&prompt, None, None);
        let response_text = response.trim().to_string();
        tracing::info!(target: LOGGER, "Follow-up selection LLM response: {}", response_text);

        let Some(selected) = Self::parse_tool_number(&response_text, &tool_lookup) else {
            tracing::info!(target: LOGGER, "No follow-up tool needed");
            return ToolSelectionResult::default();
        };

        // Don't re-call the same tool
        if already_called.contains(&selected.tool_name) {
            tracing::info!(
                target: LOGGER,
                "Follow-up selected already-called tool '{}', skipping",
                selected.tool_name
            );
            return ToolSelectionResult::default();
        }

        let selection = ToolSelection {
            order: 1,
            server_id: selected.server_id,
            server_name: selected.server_name,
            tool_name: selected.tool_name,
            tool_index: selected.tool_index,
            confidence: 0.80,
        };
        tracing::info!(
            target: LOGGER,
            "Follow-up tool selected: {}/{}",
            selection.server_name,
            selection.tool_name
        );
        ToolSelectionResult { selections: vec![selection] }
    }

    // ------------------------------------------------------------------
    // Internals
    // ------------------------------------------------------------------

    /// Extract a tool number from the LLM response.
    ///
    /// Handles multiple response formats:
    /// - Plain number: `3`
    /// - JSON: `[{"tool_number": 3}]` or `{"tool_number": 3}`
    /// - Prose with number: `The best tool is [3]` / `Tool number: 3`
    /// - Zero or no number means no match.
    pub fn parse_tool_number(text: &str, tool_lookup: &HashMap<i64, ToolLookupEntry>) -> Option<ToolLookupEntry> {
        let mut text = text.trim().to_string();
        if text.is_empty() {
            return None;
        }

        // Strip markdown fences
        if text.starts_with("```") {
            text = text
                .split('\n')
                .filter(|ln| !ln.trim().starts_with("```"))
                .collect::<Vec<_>>()
                .join("\n")
                .trim()
                .to_string();
        }

        // Try JSON first (handles [{"tool_number": N}] and bare numbers).  A
        // `tool_number` that is not an integer abandons the JSON reading and
        // falls through to the prose scan, as Python's ValueError/TypeError did.
        if let Ok(parsed) = serde_json::from_str::<Value>(&text) {
            let json_pick = || -> Result<Option<Option<ToolLookupEntry>>, ()> {
                let from_number = |num: i64| if num > 0 { tool_lookup.get(&num).cloned() } else { None };
                match &parsed {
                    // `bool` is an `int` in Python: `true` reads as 1.
                    Value::Bool(b) => Ok(Some(from_number(i64::from(*b)))),
                    Value::Number(n) if n.is_i64() || n.is_u64() => {
                        Ok(Some(n.as_i64().and_then(from_number)))
                    }
                    Value::Array(items) => {
                        for item in items {
                            if let Value::Object(obj) = item {
                                if let Some(raw) = obj.get("tool_number") {
                                    let num = py_int(raw).ok_or(())?;
                                    return Ok(Some(from_number(num)));
                                }
                            }
                        }
                        Ok(Some(None))
                    }
                    Value::Object(obj) => match obj.get("tool_number") {
                        Some(raw) => {
                            let num = py_int(raw).ok_or(())?;
                            Ok(Some(from_number(num)))
                        }
                        None => Ok(None),
                    },
                    _ => Ok(None),
                }
            };
            if let Ok(Some(decided)) = json_pick() {
                return decided;
            }
        }

        // Fallback: extract first integer from the response text
        static NUMBERS: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(\d+)\b").expect("valid regex"));
        for cap in NUMBERS.captures_iter(&text) {
            if let Ok(num) = cap[1].parse::<i64>() {
                if let Some(entry) = tool_lookup.get(&num) {
                    return Some(entry.clone());
                }
            }
        }

        None
    }
}
