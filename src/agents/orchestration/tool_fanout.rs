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

use std::collections::HashMap;
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use regex::Regex;
use serde_json::{Map, Value};

use crate::agents::mcp::explorer::SearchContext;
use crate::agents::mcp::mcp_tool_caller::MCPToolCaller;
use crate::agents::nodes::{NodeHandler, NodeResult};
use crate::agents::orchestration::context::{OrchestratorContext, SharedOrchestratorContext};
use crate::agents::orchestration::model_router::ModelRouter;
use crate::agents::orchestration::response_builder::ResponseBuilder;
use crate::agents::orchestration::skill_processor::{char_prefix, py_repr, py_str, try_call};
use crate::agents::orchestration::step_executor::{plan_step_result, StepExecutor};
use crate::agents::planner::PlanStep;
use crate::workspace::workspace::MCPServer;

/// Python logger name.
const LOG_TARGET: &str = "smartloop.agents.orchestration.tool_fanout";

/// A JSON array anywhere in the reply.
static JSON_ARRAY_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)\[.*\]").unwrap());

/// A leading list number (`1.` / `2)`).
static NUMBERING_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\d+[\.\)]\s*").unwrap());

/// Test seam replacing [`ToolFanout::extract_fanout_items`]:
/// `(ctx, step, prev_results, search_ctx) -> items`.
pub type ExtractItemsHook = Arc<
    dyn Fn(&OrchestratorContext, &PlanStep, &[Value], &SearchContext) -> Vec<Value> + Send + Sync,
>;

/// Test seam replacing [`ToolFanout::find_fanout_server`]:
/// `(step, mcp_servers, available_servers) -> server`.
pub type FindServerHook =
    Arc<dyn Fn(&PlanStep, &[MCPServer], &[Value]) -> Option<MCPServer> + Send + Sync>;

/// Split on every line boundary Python's `str.splitlines()` recognises.
fn split_lines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0usize;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let is_break = matches!(
            c,
            '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}'
                | '\u{2028}' | '\u{2029}'
        );
        if !is_break {
            continue;
        }
        lines.push(&text[start..i]);
        let mut next = i + c.len_utf8();
        if c == '\r' {
            if let Some(&(j, '\n')) = chars.peek() {
                chars.next();
                next = j + 1;
            }
        }
        start = next;
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// Runs one tool once per item extracted from upstream context.
pub struct ToolFanout {
    pub(crate) models: Arc<ModelRouter>,
    pub(crate) steps: Arc<StepExecutor>,
    /// Held for parity with Python; its helpers used here are associated
    /// functions.
    #[allow(dead_code)]
    pub(crate) response: Arc<ResponseBuilder>,
    pub tool_caller: Arc<MCPToolCaller>,
    /// Replaces [`Self::extract_fanout_items`] when set (test seam).
    extract_items_hook: RwLock<Option<ExtractItemsHook>>,
    /// Replaces [`Self::find_fanout_server`] when set (test seam).
    find_server_hook: RwLock<Option<FindServerHook>>,
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
        Self {
            models,
            steps,
            response,
            tool_caller,
            extract_items_hook: RwLock::new(None),
            find_server_hook: RwLock::new(None),
        }
    }

    /// Replace item extraction (Python tests `patch.object(..., '_extract_fanout_items')`).
    pub fn set_extract_items_hook(&self, hook: Option<ExtractItemsHook>) {
        *self.extract_items_hook.write() = hook;
    }

    /// Replace server lookup (Python tests `patch.object(..., '_find_fanout_server')`).
    pub fn set_find_server_hook(&self, hook: Option<FindServerHook>) {
        *self.find_server_hook.write() = hook;
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
        ctx: &SharedOrchestratorContext,
        step: &PlanStep,
        search_ctx: &Arc<Mutex<SearchContext>>,
    ) -> NodeHandler {
        let this = Arc::clone(self);
        let ctx = Arc::clone(ctx);
        let step = step.clone();
        let search_ctx = Arc::clone(search_ctx);
        Arc::new(move |deps: HashMap<String, NodeResult>| {
            let tool_label = step.tool_name.as_deref().unwrap_or("None");
            let node_id = format!("step_{}_{}_fanout", step.step, tool_label);
            let prev_results = ResponseBuilder::collect_raw_from_deps(&deps);

            let ctx_snap = ctx.read().clone();
            let search_snap = search_ctx.lock().clone();
            let items = this.extract_items(&ctx_snap, &step, &prev_results, &search_snap);
            if items.is_empty() {
                tracing::warn!(target: LOG_TARGET, "[fanout] No items extracted for {}", tool_label);
                return Ok(plan_step_result(&node_id, &step, "", None));
            }

            tracing::info!(
                target: LOG_TARGET,
                "[fanout] Extracted {} item(s) for {}",
                items.len(),
                tool_label
            );

            let Some(server) =
                this.find_server(&step, &ctx_snap.mcp_servers, &ctx_snap.available_servers)
            else {
                tracing::warn!(target: LOG_TARGET, "[fanout] Server not found for {}", tool_label);
                return Ok(plan_step_result(&node_id, &step, "", None));
            };

            let tool_name = step.tool_name.clone().unwrap_or_default();
            let cwd = Some(ctx_snap.effective_cwd.as_str()).filter(|c| !c.is_empty());

            // (item, formatted, raw) per successful call.
            let mut all_results: Vec<(Value, String, String)> = Vec::new();
            for (i, item_value) in items.iter().enumerate() {
                let mut arguments: Map<String, Value> = step
                    .default_args
                    .as_ref()
                    .map(|d| d.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect())
                    .unwrap_or_default();

                match item_value {
                    Value::Object(obj) => {
                        for (k, v) in obj {
                            arguments.insert(k.clone(), v.clone());
                        }
                    }
                    Value::String(s) => {
                        if let Some(inferred) = step.inferred_args.as_ref().filter(|a| a.len() == 1) {
                            arguments.insert(inferred[0].clone(), Value::String(s.clone()));
                        }
                    }
                    _ => {}
                }

                tracing::info!(
                    target: LOG_TARGET,
                    "[fanout] Calling {} [{}/{}]: {}",
                    tool_name,
                    i + 1,
                    items.len(),
                    py_repr(&serde_json::Value::Object(arguments.clone()))
                );

                if let Some((formatted, raw)) =
                    this.steps.call_tool(&server, &tool_name, &arguments, cwd)
                {
                    all_results.push((item_value.clone(), formatted, raw));
                }
            }

            if all_results.is_empty() {
                return Ok(plan_step_result(&node_id, &step, "", None));
            }

            let combined_content = all_results
                .iter()
                .map(|(item, content, _)| format!("[{} for {}]:\n{}", tool_name, py_str(item), content))
                .collect::<Vec<_>>()
                .join("\n\n---\n\n");
            let combined_raw = all_results
                .iter()
                .map(|(item, _, raw)| format!("[{} for {}]:\n{}", tool_name, py_str(item), raw))
                .collect::<Vec<_>>()
                .join("\n\n---\n\n");

            tracing::info!(
                target: LOG_TARGET,
                "[fanout] {} completed {}/{} calls",
                tool_label,
                all_results.len(),
                items.len()
            );
            Ok(plan_step_result(&node_id, &step, &combined_content, Some(&combined_raw)))
        })
    }

    /// Item extraction, honouring the test seam.
    fn extract_items(
        &self,
        ctx: &OrchestratorContext,
        step: &PlanStep,
        prev_results: &[Value],
        search_ctx: &SearchContext,
    ) -> Vec<Value> {
        let hook = self.extract_items_hook.read().clone();
        match hook {
            Some(h) => h(ctx, step, prev_results, search_ctx),
            None => self.extract_fanout_items(ctx, step, prev_results, search_ctx),
        }
    }

    /// Server lookup, honouring the test seam.
    fn find_server(
        &self,
        step: &PlanStep,
        mcp_servers: &[MCPServer],
        available_servers: &[Value],
    ) -> Option<MCPServer> {
        let hook = self.find_server_hook.read().clone();
        match hook {
            Some(h) => h(step, mcp_servers, available_servers),
            None => self.find_fanout_server(step, mcp_servers, available_servers),
        }
    }

    /// Locate the MCP server for a fan-out step.
    pub fn find_fanout_server(
        &self,
        step: &PlanStep,
        mcp_servers: &[MCPServer],
        _available_servers: &[Value],
    ) -> Option<MCPServer> {
        let tool_name = step.tool_name.as_deref().unwrap_or("");
        if let Some(server_id) = step.server_id.as_deref().filter(|s| !s.is_empty()) {
            let server_name = step.server_name.as_deref().unwrap_or("");
            if let Some(server) = self.tool_caller.find_server(mcp_servers, server_id, server_name) {
                return Some(server.clone());
            }
        }
        for srv in mcp_servers {
            if !srv.enabled || srv.tools.is_empty() {
                continue;
            }
            if srv.tools.iter().any(|t| t.name == tool_name) {
                return Some(srv.clone());
            }
        }
        None
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
        prev_results: &[Value],
        search_ctx: &SearchContext,
    ) -> Vec<Value> {
        let mut context_parts: Vec<String> = Vec::new();
        for r in prev_results {
            let result_text = match r.get("result") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Null) | None => String::new(),
                Some(other) => py_str(other),
            };
            let result_text = char_prefix(&result_text, 4000);
            if !result_text.is_empty() {
                context_parts.push(result_text.to_string());
            }
        }
        if !search_ctx.vector_context.is_empty() {
            context_parts.push(char_prefix(&search_ctx.vector_context, 6000).to_string());
        }

        let context = context_parts.join("\n\n");
        if context.is_empty() {
            return Vec::new();
        }

        let arg_name = step
            .inferred_args
            .as_ref()
            .and_then(|a| a.first())
            .cloned()
            .unwrap_or_default();
        let arg_label = if arg_name.is_empty() { "item" } else { arg_name.as_str() };

        let skill_hint = match ctx.skill_context.as_deref().filter(|s| !s.is_empty()) {
            Some(sc) => format!("Skill instruction: {}\n\n", char_prefix(sc, 500)),
            None => String::new(),
        };

        let prompt = format!(
            "<system>\n\
             Your task: read the context and list every distinct '{arg_label}' \
             value that should be passed to the tool '{}'.\n\n\
             Output format: one value per line, nothing else.\n\
             Do not add numbering, bullets, or explanation.\n\
             Do not skip any values — list ALL of them.\n\
             </system>\n\
             <user>\n\
             {skill_hint}\
             Context:\n{}\n\n\
             List every '{arg_label}' value, one per line:\n\
             </user>",
            step.tool_name.as_deref().unwrap_or("None"),
            char_prefix(&context, 8000)
        );

        let outcome = try_call(|| {
            // Fan-out item extraction is argument extraction for the same tool,
            // so it follows the step's capability like the single-call path.
            let fanout_llm = self.models.decision_llm(step.capability.as_deref(), None);
            fanout_llm.invoke(&prompt, None, None).trim().to_string()
        });
        match outcome {
            Ok(response) => {
                tracing::info!(
                    target: LOG_TARGET,
                    "[fanout-extract] LLM response: {}",
                    char_prefix(&response, 300)
                );

                let mut items = Self::parse_fanout_response(&response);
                if !items.is_empty() {
                    if items.len() > Self::FANOUT_MAX_ITEMS {
                        tracing::warn!(
                            target: LOG_TARGET,
                            "[fanout-extract] Truncated {} items to {}",
                            items.len(),
                            Self::FANOUT_MAX_ITEMS
                        );
                    }
                    items.truncate(Self::FANOUT_MAX_ITEMS);
                    return items.into_iter().map(Value::String).collect();
                }
            }
            Err(e) => {
                tracing::warn!(target: LOG_TARGET, "[fanout-extract] extraction failed: {}", e);
            }
        }

        Vec::new()
    }

    /// Parse fan-out extraction response: one value per line or JSON array.
    pub fn parse_fanout_response(response: &str) -> Vec<String> {
        let text = response.trim();
        if text.is_empty() {
            return Vec::new();
        }

        // Try JSON array first (in case the LLM returns one)
        if let Some(m) = JSON_ARRAY_RE.find(text) {
            if let Ok(Value::Array(items)) = serde_json::from_str::<Value>(m.as_str()) {
                if !items.is_empty() {
                    return items
                        .iter()
                        .filter(|v| crate::agents::orchestration::skill_processor::truthy(v))
                        .map(py_str)
                        .collect();
                }
            }
        }

        // Parse one-value-per-line format
        let mut items: Vec<String> = Vec::new();
        for line in split_lines(text) {
            let val = line.trim().trim_matches(|c| matches!(c, '-' | '•' | '*')).trim();
            let val = NUMBERING_RE.replace(val, "");
            let val = val.trim();
            if val.is_empty() || val.chars().count() > 200 {
                continue;
            }
            let lower = val.to_lowercase();
            if ["here ", "the ", "these ", "below", "value"].iter().any(|p| lower.starts_with(p)) {
                continue;
            }
            items.push(val.to_string());
        }

        items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lines_json_and_lists() {
        assert_eq!(ToolFanout::parse_fanout_response("AAPL\nMSFT\nGOOGL"), ["AAPL", "MSFT", "GOOGL"]);
        assert_eq!(ToolFanout::parse_fanout_response("[\"AAPL\", \"MSFT\"]"), ["AAPL", "MSFT"]);
        assert_eq!(
            ToolFanout::parse_fanout_response("1. AAPL\n2. MSFT\n- GOOGL\n• VYM"),
            ["AAPL", "MSFT", "GOOGL", "VYM"]
        );
        assert_eq!(
            ToolFanout::parse_fanout_response("Here are the values:\nAAPL\r\nMSFT"),
            ["AAPL", "MSFT"]
        );
        assert!(ToolFanout::parse_fanout_response("").is_empty());
        assert_eq!(ToolFanout::parse_fanout_response("[1, 0, \"x\"]"), ["1", "x"]);
    }
}
