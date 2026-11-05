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

use std::collections::HashSet;
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use regex::Regex;
use serde_json::{json, Map, Value};
use tracing::{info, warn};

use crate::agents::mcp::argument_extractor::MCPArgumentExtractor;
use crate::agents::mcp::dependency_analyzer::MCPDependencyAnalyzer;
use crate::agents::mcp::explorer::{Explorer, ExplorerSearchService};
use crate::agents::mcp::mcp_agent::{missing_required, MCPAgent, ResolvedCall};
use crate::agents::mcp::mcp_tool_caller::MCPToolCaller;
use crate::agents::mcp::server_selector::{
    BASH_TOOL, EXPLORE_TOOL, FILE_READ_TOOL, FILE_WRITE_TOOL, GREP_TOOL,
};
use crate::agents::nodes::NodeResult;
use crate::agents::orchestration::context::{OrchestratorContext, SharedOrchestratorContext};
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

/// The reason node's id (`AgentNodeName::Reason`), as it appears in
/// `previous_results[*].tool_name`.
const REASON_NODE: &str = "reason";

/// Built-in tools live on the smartloop system server.
const SYSTEM_TOOLS: [&str; 5] = [FILE_WRITE_TOOL, FILE_READ_TOOL, GREP_TOOL, BASH_TOOL, EXPLORE_TOOL];

/// An instruction that refers back to earlier output depends on it.
static DEP_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\bstep\s*\d|\bprevious\b|\bresult\s*of\b").expect("valid dependency regex")
});

/// Action kinds that inherently depend on prior results.
const SEQUENTIAL_ACTIONS: [&str; 2] = ["write", "summarize"];

/// Placeholder values the LLM generates for optional params (e.g. "Not
/// specified", "needs to be provided by the user").
static PLACEHOLDER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)not\s+(?:specified|provided|available|given)|needs?\s+to\s+be\s+provided|n/?a\b|unknown",
    )
    .expect("valid placeholder regex")
});

/// Build the NodeResult for one executed plan step.
///
/// Each node stores its own output — the next node reads only its direct
/// dependency's `raw` field, which is what keeps the context focused instead
/// of dragging the whole accumulated chain forward.
///
/// `data` = `{"tool_name", "content", "raw", "is_tool_result", "server_id",
/// "server_name"}` — the server fields name where a tool step's result came
/// from, so the answer can cite it; `null` for every other step.
pub fn plan_step_result(
    node_id: &str,
    step: &PlanStep,
    content: &str,
    raw: Option<&str>,
) -> NodeResult {
    let tool_name = match step.tool_name.as_deref() {
        Some(t) if !t.is_empty() => t.to_string(),
        _ => step.step_type.value().to_string(),
    };
    let raw_value = match raw {
        Some(r) if !r.is_empty() => r,
        _ => content,
    };
    let is_tool = step.step_type == StepType::Tool;
    NodeResult::with_data(
        node_id,
        json!({
            "tool_name": tool_name,
            "content": content,
            "raw": raw_value,
            "is_tool_result": is_tool,
            "server_id": if is_tool { step.server_id.clone() } else { None },
            "server_name": if is_tool { step.server_name.clone() } else { None },
        }),
    )
}

/// Test seam replacing [`MCPAgent::resolve`]:
/// `(directive, mcp_servers, context, llm) -> resolved call`.
pub type ResolveHook = Arc<
    dyn Fn(&str, &[MCPServer], Option<&str>, Option<Arc<dyn BaseLLM>>) -> Option<ResolvedCall>
        + Send
        + Sync,
>;

/// Test seam replacing the LLM pass of [`MCPArgumentExtractor::invoke`]:
/// `(user_input, tool_name, input_schema, previous_tool_results) -> arguments`.
pub type ExtractHook = Arc<
    dyn Fn(&str, &str, &Value, Option<&[Value]>) -> Map<String, Value> + Send + Sync,
>;

/// Resolves and runs individual plan steps.
pub struct StepExecutor {
    pub orchestrator_llm: Arc<dyn BaseLLM>,
    pub tool_caller: Arc<MCPToolCaller>,
    pub arg_extractor: Arc<MCPArgumentExtractor>,
    pub dep_analyzer: Arc<MCPDependencyAnalyzer>,
    pub result_formatter: Arc<ResultFormatter>,
    pub explorer: Arc<Explorer>,
    mcp_agent: Arc<MCPAgent>,
    /// Held for parity with the Python wiring; the one helper used from it
    /// (`strip_tool_code_blocks`) is an associated function.
    #[allow(dead_code)]
    response: Arc<ResponseBuilder>,
    resolve_hook: Mutex<Option<ResolveHook>>,
    extract_hook: Mutex<Option<ExtractHook>>,
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
            resolve_hook: Mutex::new(None),
            extract_hook: Mutex::new(None),
        }
    }

    /// Replace (or restore, with `None`) the MCPAgent resolution call (tests).
    pub fn set_resolve_hook(&self, hook: Option<ResolveHook>) {
        *self.resolve_hook.lock() = hook;
    }

    /// Replace (or restore, with `None`) the argument extractor's LLM pass
    /// (tests).
    pub fn set_extract_hook(&self, hook: Option<ExtractHook>) {
        *self.extract_hook.lock() = hook;
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
        self.tool_caller.call_tool(server, tool_name, arguments, access_token, cwd)
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
    ///
    /// (As in the Python implementation, `result_contains_any:` is not matched
    /// by the `result_contains:` prefix and so falls through to the
    /// unknown-format branch, which executes unconditionally.)
    pub fn evaluate_condition(&self, condition: Option<&str>, previous_results: &[Value]) -> bool {
        let condition = match condition {
            Some(c) if !c.is_empty() => c.trim(),
            _ => return true,
        };
        let lower = condition.to_lowercase();

        // result_contains:<text>
        if lower.starts_with("result_contains:") {
            let needle = condition["result_contains:".len()..].trim().to_lowercase();
            for r in previous_results {
                let text = result_or_content(r);
                if text.to_lowercase().contains(&needle) {
                    return true;
                }
            }
            return false;
        }

        // llm:<condition clause>
        if lower.starts_with("llm:") {
            let clause = condition[4..].trim();
            // Build a short summary of previous results
            let mut combined = String::new();
            for r in previous_results {
                let text = result_or_content(r);
                combined.push_str(&head(&text, 1500));
                combined.push('\n');
            }
            let combined = head(combined.trim(), 3000);

            if combined.is_empty() {
                info!(target: LOGGER, "[condition] No previous results — defaulting to YES");
                return true;
            }

            let prompt = format!("Results:\n{combined}\n\nCondition: {clause}\nYES or NO?");
            let response = self
                .orchestrator_llm
                .invoke(&prompt, Some(DECISION_MAX_TOKENS), None)
                .trim()
                .to_uppercase();
            // Fail-open: only skip if the LLM explicitly says NO without any
            // YES.  This handles ambiguous responses by defaulting to execute.
            let has_no = response.contains("NO");
            let has_yes =
                response.contains("YES") || response.starts_with('Y') || response.contains("TRUE");
            let result = has_yes || !has_no;
            info!(
                target: LOGGER,
                "[condition] LLM: '{}' → {}",
                head(&response, 80),
                py_bool(result)
            );
            return result;
        }

        // Unknown format — execute unconditionally
        true
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
        let mut groups: Vec<Vec<PlanStep>> = Vec::new();
        let mut current_parallel: Vec<PlanStep> = Vec::new();

        for step in steps {
            let action_kind = step.action_kind.as_deref().unwrap_or("");
            let has_condition = step.condition.as_deref().is_some_and(|c| !c.is_empty());
            let is_independent = !has_condition
                && !DEP_RE.is_match(&step.instruction)
                && !SEQUENTIAL_ACTIONS.contains(&action_kind)
                && step.step_type != StepType::Conversational;
            if is_independent {
                current_parallel.push(step.clone());
            } else {
                // Flush any accumulated parallel group
                if !current_parallel.is_empty() {
                    groups.push(std::mem::take(&mut current_parallel));
                }
                groups.push(vec![step.clone()]);
            }
        }

        if !current_parallel.is_empty() {
            groups.push(current_parallel);
        }

        groups
    }

    /// Run one non-TOOL step and return `(formatted, raw)` or `None`.
    ///
    /// Everything constant for the request comes off `ctx`; only what varies
    /// per step (the step itself, its upstream results, the document context
    /// in force, and the resolved model) is passed in.
    ///
    /// `ctx` is the shared per-request context: an EXPLORE step records the
    /// citations it drew on (`ctx.explore_citations`), and a TOOL step whose
    /// call refreshed the server's credentials writes the refreshed server
    /// back into `ctx.mcp_servers`.  The context lock is only taken briefly,
    /// so it must not be held by the caller.
    pub fn execute(
        &self,
        ctx: &SharedOrchestratorContext,
        step: &PlanStep,
        previous_results: Option<&[Value]>,
        document_context: Option<&str>,
        llm: Option<Arc<dyn BaseLLM>>,
    ) -> Option<(String, String)> {
        // Unpacked once rather than read through `ctx` throughout: the body
        // below is long and branches per step type, and the plain names keep
        // the individual branches readable.
        let (
            user_message,
            mcp_servers,
            available_servers,
            cwd,
            project_dir,
            model_service,
        ) = {
            let c = ctx.read();
            (
                c.user_message.clone(),
                c.mcp_servers.clone(),
                c.available_servers.clone(),
                c.effective_cwd.clone(),
                c.project_dir.clone(),
                c.model_service.clone(),
            )
        };

        let mut tool_name = step.tool_name.clone().unwrap_or_default();
        let has_explicit_tool = !tool_name.is_empty();
        let directive = if step.instruction.is_empty() {
            user_message.clone()
        } else {
            step.instruction.clone()
        };
        let work_dir = expand_home(&first_non_empty(&[
            Some(cwd.as_str()),
            project_dir.as_deref(),
            Some(current_dir().as_str()),
        ]));
        let prior_count = previous_results.map_or(0, <[Value]>::len);

        info!(
            target: LOGGER,
            "[step] {}/{}, cwd={}, prior_results={}",
            py_opt(step.server_name.as_deref()),
            tool_name,
            work_dir,
            prior_count
        );

        let step_type = step.step_type;

        // -- ANALYZE --
        if step_type == StepType::Analyze {
            warn!(target: LOGGER, "[step] analyze step — no enriched analysis available");
            return None;
        }

        // -- CONVERSATIONAL --
        // Collect the context only.  The chat handler writes the answer from
        // it, streaming token by token and closing with the cited sources;
        // drafting it here would make the client wait for a whole reply that
        // only feeds a second generation.
        if step_type == StepType::Conversational {
            let mut parts: Vec<String> = Vec::new();
            if let Some(prev) = previous_results {
                for r in prev {
                    parts.push(format!("[{}]: {}", field_str(r, "tool_name"), field_str(r, "result")));
                }
            }
            if let Some(doc) = document_context.filter(|d| !d.is_empty()) {
                if !directive.contains("DOCUMENT CONTEXT") {
                    parts.push(format!("DOCUMENT CONTEXT:\n{doc}"));
                }
            }
            if !directive.is_empty() {
                parts.push(directive.clone());
            }
            let raw = if parts.is_empty() { directive.clone() } else { parts.join("\n\n") };
            info!(
                target: LOGGER,
                "[step] conversational collected {} chars of raw context",
                raw.chars().count()
            );
            return Some((raw.clone(), raw));
        }

        // -- EXPLORE --
        if step_type == StepType::Explore {
            let search_service: Option<Arc<dyn ExplorerSearchService>> =
                model_service.map(|m| m as Arc<dyn ExplorerSearchService>);
            let (content, explore_citations) = self.explorer.explore(
                &user_message,
                &work_dir,
                project_dir.as_deref(),
                Some(&mcp_servers),
                search_service,
                document_context.unwrap_or(""),
                None,
                None,
                None,
                None,
                None,
            );
            if content.is_empty() {
                return None;
            }
            if let Some(citations) = explore_citations.filter(|c| !c.is_empty()) {
                ctx.write().explore_citations =
                    Some(citations.into_iter().map(Value::Object).collect());
            }
            return Some((content.clone(), content));
        }

        // -- ACTION --
        // Only the internal `summarize` kind remains here. The built-in
        // file/grep/bash tools now run as TOOL steps through the smartloop
        // MCP server (resolved/executed via the resolve+execute handlers).
        if step_type == StepType::Action {
            let action_kind = step.action_kind.as_deref().unwrap_or("");

            if action_kind == "summarize" {
                // Collect raw context; the final streaming LLM handles reasoning.
                let mut parts: Vec<String> = Vec::new();
                if let Some(prev) = previous_results {
                    for r in prev {
                        parts.push(format!(
                            "[{}]: {}",
                            field_str(r, "tool_name"),
                            field_str(r, "result")
                        ));
                    }
                }
                if !directive.is_empty() {
                    parts.push(directive.clone());
                }
                let raw = if parts.is_empty() { directive.clone() } else { parts.join("\n\n") };
                info!(
                    target: LOGGER,
                    "[step] summarize collected {} chars of raw context",
                    raw.chars().count()
                );
                return Some((raw.clone(), raw));
            }

            warn!(target: LOGGER, "[step] Unknown action_kind: {}", action_kind);
            return None;
        }

        // -- TOOL: Call external MCP tool --
        let step_server_id = step.server_id.as_deref().unwrap_or("");
        let step_server_name = step.server_name.as_deref().unwrap_or("");
        let mut server: Option<MCPServer> = None;
        if !step_server_id.is_empty() {
            server = self
                .tool_caller
                .find_server(&mcp_servers, step_server_id, step_server_name)
                .cloned();
        }
        if server.is_none() {
            // Catalogue scan: find the server that owns this tool
            let mut real_server_id: Option<String> = None;
            'scan: for srv in &available_servers {
                if srv.get("id").and_then(Value::as_str) == Some(SYSTEM_TOOLS_SERVER_ID) {
                    continue;
                }
                for t in srv.get("tools").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]) {
                    if t.get("name").and_then(Value::as_str) == Some(tool_name.as_str()) {
                        let id = value_str(srv.get("id").unwrap_or(&Value::Null));
                        info!(
                            target: LOGGER,
                            "[step] Found tool '{}' in server '{}' (id={})",
                            tool_name,
                            srv.get("name").and_then(serde_json::Value::as_str).unwrap_or(""),
                            id
                        );
                        real_server_id = Some(id);
                        break 'scan;
                    }
                }
            }

            // Fuzzy server name match (when tool not found by exact name)
            if real_server_id.is_none() && !step_server_name.is_empty() {
                let hint = step_server_name.to_lowercase();
                for srv in &available_servers {
                    if srv.get("id").and_then(Value::as_str) == Some(SYSTEM_TOOLS_SERVER_ID) {
                        continue;
                    }
                    let srv_name = srv.get("name").and_then(Value::as_str).unwrap_or("");
                    if srv_name.to_lowercase() == hint || srv_name.to_lowercase().contains(&hint) {
                        real_server_id = Some(value_str(srv.get("id").unwrap_or(&Value::Null)));
                        info!(
                            target: LOGGER,
                            "[step] Matched server '{}' via name hint '{}'",
                            srv_name,
                            step_server_name
                        );
                        break;
                    }
                }
            }

            // Fuzzy: check directive for server name hints
            if real_server_id.is_none() {
                let directive_lower = directive.to_lowercase();
                for srv in &available_servers {
                    if srv.get("id").and_then(Value::as_str) == Some(SYSTEM_TOOLS_SERVER_ID) {
                        continue;
                    }
                    let srv_name = srv.get("name").and_then(Value::as_str).unwrap_or("");
                    if !srv_name.is_empty() && directive_lower.contains(&srv_name.to_lowercase()) {
                        real_server_id = Some(value_str(srv.get("id").unwrap_or(&Value::Null)));
                        info!(target: LOGGER, "[step] Matched server '{}' from directive", srv_name);
                        break;
                    }
                }
            }

            if let Some(real_id) = real_server_id.filter(|id| !id.is_empty()) {
                server = self.tool_caller.find_server(&mcp_servers, &real_id, &real_id).cloned();
                if let Some(srv) = server.as_ref() {
                    info!(
                        target: LOGGER,
                        "[step] Resolved server for tool '{}' via catalogue scan → '{}' (id={})",
                        tool_name,
                        srv.name,
                        srv.id
                    );
                    if self.tool_caller.find_tool_schema(srv, &tool_name).is_none() {
                        if !has_explicit_tool {
                            if let Some(matched_tool) =
                                self.tool_caller.find_tool_by_keyword(srv, &directive)
                            {
                                tool_name = matched_tool;
                                info!(
                                    target: LOGGER,
                                    "[step] Resolved tool '{}' → '{}' via keyword match on server '{}'",
                                    py_opt(step.tool_name.as_deref()),
                                    tool_name,
                                    srv.name
                                );
                            }
                        } else {
                            warn!(
                                target: LOGGER,
                                "[step] Explicit tool '{}' not found on server '{}' — skipping keyword fallback",
                                tool_name,
                                srv.name
                            );
                        }
                    }
                }
            }
        }
        // Last resort: try every non-system server for the tool
        if server.is_none() {
            for srv_obj in &mcp_servers {
                if !srv_obj.enabled || srv_obj.tools.is_empty() {
                    continue;
                }
                if srv_obj.id == SYSTEM_TOOLS_SERVER_ID {
                    continue;
                }
                // Check exact match
                if srv_obj.tools.iter().any(|t| t.name == tool_name) {
                    server = Some(srv_obj.clone());
                    info!(
                        target: LOGGER,
                        "[step] Found tool '{}' via direct MCP server scan → '{}'",
                        tool_name,
                        srv_obj.name
                    );
                }
                // Check keyword match
                if server.is_none() && !has_explicit_tool {
                    if let Some(matched_tool) = self.tool_caller.find_tool_by_keyword(srv_obj, &tool_name) {
                        server = Some(srv_obj.clone());
                        tool_name = matched_tool;
                        info!(
                            target: LOGGER,
                            "[step] Resolved '{}' → '{}' via keyword on '{}'",
                            py_opt(step.tool_name.as_deref()),
                            tool_name,
                            srv_obj.name
                        );
                    }
                }
                if server.is_some() {
                    break;
                }
            }
        }

        let Some(mut server) = server else {
            warn!(
                target: LOGGER,
                "[step] Server not found for tool '{}': {} (id={})",
                tool_name,
                py_opt(step.server_name.as_deref()),
                py_opt(step.server_id.as_deref())
            );
            return None;
        };

        let mut tool_schema = self.tool_caller.find_tool_schema(&server, &tool_name);
        if tool_schema.is_none() && !has_explicit_tool {
            // Fallback: LLM may have invented a tool name.  Try to resolve the
            // correct tool via keyword match from the step instruction.
            if let Some(resolved_tool) = self.tool_caller.find_tool_by_keyword(&server, &directive) {
                info!(
                    target: LOGGER,
                    "[step] Tool '{}' not found — resolved to '{}' via keyword match on instruction",
                    tool_name,
                    resolved_tool
                );
                tool_name = resolved_tool;
                tool_schema = self.tool_caller.find_tool_schema(&server, &tool_name);
            }

            // If still not found, check if the instruction mentions a known tool name
            if tool_schema.is_none() {
                let mentioned = server
                    .tools
                    .iter()
                    .map(|t| t.name.clone())
                    .find(|n| !n.is_empty() && directive.contains(n.as_str()));
                if let Some(t_name) = mentioned {
                    info!(
                        target: LOGGER,
                        "[step] Tool '{}' not found — resolved to '{}' (mentioned in instruction)",
                        tool_name,
                        t_name
                    );
                    tool_name = t_name;
                    tool_schema = self.tool_caller.find_tool_schema(&server, &tool_name);
                }
            }

            if tool_schema.is_none() {
                warn!(
                    target: LOGGER,
                    "[step] Tool schema not found: {} on server '{}'",
                    tool_name,
                    server.name
                );
                return None;
            }
        }

        let input_schema = schema_of(tool_schema.as_ref());
        let input_schema_value = Value::Object(input_schema.clone());
        let server_url = MCPToolCaller::get_server_url(&server).filter(|u| !u.is_empty());

        if server_url.is_none() && server.server_type != MCPServerType::Local {
            warn!(
                target: LOGGER,
                "[step] No server URL configured for: {}",
                py_opt(step.server_name.as_deref())
            );
            return None;
        }

        let all_tools = tool_catalog(&server);
        let mut dep_context = directive.clone();
        if let Some(prev) = previous_results.filter(|p| !p.is_empty()) {
            let prior_text = prior_results_text(prev);
            dep_context = format!("{directive}\n\nPrevious step results:\n{prior_text}");
        }

        let dep_analysis =
            self.dep_analyzer.invoke(&dep_context, &tool_name, &input_schema_value, &all_tools);
        if dep_analysis.has_dependencies && !dep_analysis.can_execute {
            info!(
                target: LOGGER,
                "[step] '{}' has unresolvable dependencies; falling through",
                tool_name
            );
            return None;
        }

        let mut resolved_ids = Map::new();
        if dep_analysis.has_dependencies && dep_analysis.can_execute {
            let prev_maps = previous_results.map(result_maps);
            resolved_ids = self.tool_caller.execute_prerequisites(
                &dep_analysis,
                &mut server,
                &directive,
                &input_schema,
                prev_maps.as_deref(),
            );
        }

        // Use the user's original message as primary input so the LLM
        // extracts values from it, not from skill guidance metadata.
        // NOTE: Do NOT inject the tool description into the extraction input.
        // Small models copy it verbatim into same-named params (e.g. a tool
        // description "Create or update a Linear issue." leaking into a
        // `description` field). The extractor prompt already names the tool;
        // values must come from the user message + previous results only.
        let mut extraction_input = user_message.clone();
        if let Some(prev) = previous_results.filter(|p| !p.is_empty()) {
            let prior_data = prev
                .iter()
                .map(|r| head(&field_str(r, "result"), 4000))
                .collect::<Vec<_>>()
                .join("\n");
            extraction_input = format!(
                "{user_message}\n\n---\nData from previous step (use this to fill parameter values):\n{prior_data}\n---"
            );
        }

        // NOTE: Do NOT append step.instruction to extraction_input.  The
        // instruction may contain skill body text (descriptions, tool names,
        // etc.) that small models pick up as parameter values.  The tool
        // schema already tells the LLM what parameters are needed; the user
        // message + previous results provide the values.

        let mut arguments = if !resolved_ids.is_empty() {
            self.arg_extractor.invoke_with_resolved_ids(
                &extraction_input,
                &tool_name,
                &input_schema_value,
                &resolved_ids,
                llm.as_deref(),
            )
        } else {
            self.extract(&extraction_input, &tool_name, &input_schema_value, previous_results, llm.as_deref())
        };

        let missing = arguments.remove("_missing_params");
        if let Some(missing) = missing.filter(is_truthy) {
            info!(
                target: LOGGER,
                "[step] '{}' missing required params ({}); skipping",
                tool_name,
                join_values(&missing)
            );
            return None;
        }

        let required_params = string_list(input_schema.get("required"));
        let properties = object_of(input_schema.get("properties"));
        let mut still_missing: Vec<String> = Vec::new();
        for param_name in &required_params {
            if is_unset(arguments.get(param_name)) {
                let param_info = properties.get(param_name).cloned().unwrap_or_else(empty_object);
                // Only fill URL params from extracted URLs — never invent values
                if self.arg_extractor.is_url_parameter(param_name, &param_info) {
                    let mut urls = self.arg_extractor.extract_urls_from_text(&directive);
                    if urls.is_empty() {
                        urls = self.arg_extractor.extract_urls_from_text(&user_message);
                    }
                    if let Some(url) = urls.into_iter().next() {
                        arguments.insert(param_name.clone(), Value::String(url));
                    } else {
                        still_missing.push(param_name.clone());
                    }
                } else {
                    still_missing.push(param_name.clone());
                }
            }
        }
        if !still_missing.is_empty() {
            info!(
                target: LOGGER,
                "[step] '{}' still missing required params ({}); skipping",
                tool_name,
                still_missing.join(", ")
            );
            return None;
        }

        if !self.drop_invalid(&mut arguments, &properties, &required_params, &tool_name, "step") {
            return None;
        }

        if !properties.is_empty() {
            let unknown: Vec<String> =
                arguments.keys().filter(|k| !properties.contains_key(*k)).cloned().collect();
            if !unknown.is_empty() {
                info!(
                    target: LOGGER,
                    "[step] '{}' removing unknown params: {}",
                    tool_name,
                    py_list(&unknown)
                );
                for k in &unknown {
                    arguments.remove(k);
                }
            }
        }

        // Strip placeholder values the LLM generates for optional params
        // (e.g. "Not specified", "needs to be provided by the user").
        strip_placeholders(&mut arguments, &required_params, &tool_name, "step");

        info!(
            target: LOGGER,
            "[step] Calling '{}' with arguments: {}",
            tool_name,
            py_dict(&arguments)
        );
        let before = server.clone();
        let (result, refreshed) =
            self.tool_caller.call_with_refresh(&mut server, &tool_name, Some(&arguments), None);
        if refreshed || server != before {
            // Python's server object was shared with the request's server list,
            // so refreshed credentials were visible to every later step; keep
            // that true for the cloned record.
            write_back_server(ctx, &server);
        }

        self.finish_call(&result, &tool_name, "step")
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
        // See `execute`: unpacked for readability across a long body.
        let user_message = ctx.user_message.as_str();
        let mcp_servers = ctx.mcp_servers.as_slice();
        let cwd = ctx.effective_cwd.as_str();

        let mut tool_name = step.tool_name.clone().unwrap_or_default();
        let has_explicit_tool = !tool_name.is_empty();
        let directive = if step.instruction.is_empty() {
            user_message.to_string()
        } else {
            step.instruction.clone()
        };
        let work_dir = expand_home(&first_non_empty(&[Some(cwd), Some(current_dir().as_str())]));

        info!(
            target: LOGGER,
            "[resolve] {}/{}, cwd={}, prior_results={}",
            py_opt(step.server_name.as_deref()),
            tool_name,
            work_dir,
            previous_results.map_or(0, <[serde_json::Value]>::len)
        );

        // Built-in tools live on the smartloop system server. If a plan
        // mis-assigned one to another server, pin it back to smartloop so it
        // resolves against the local MCP server like any other tool.
        let is_system_tool = SYSTEM_TOOLS.contains(&tool_name.as_str());
        if is_system_tool && step.server_id.as_deref() != Some(SYSTEM_TOOLS_SERVER_ID) {
            info!(
                target: LOGGER,
                "[resolve] '{}' is a system tool — pinning to the smartloop server (was '{}')",
                tool_name,
                py_opt(step.server_name.as_deref())
            );
            step.server_id = Some(SYSTEM_TOOLS_SERVER_ID.to_string());
            step.server_name = Some(SYSTEM_TOOLS_SERVER_NAME.to_string());
        }

        let is_system = step.server_id.as_deref() == Some(SYSTEM_TOOLS_SERVER_ID) || is_system_tool;

        // What earlier steps returned, shared by tool resolution below and the
        // dependency analysis further down: a chained call ("comment on the
        // design you just listed") takes its arguments from here, so the two
        // must read the same text.
        let prior_text = match previous_results {
            Some(prev) if !prev.is_empty() => prior_results_text(prev),
            _ => String::new(),
        };

        // Find server + tool.
        let step_server_id = step.server_id.clone().unwrap_or_default();
        let step_server_name = step.server_name.clone().unwrap_or_default();
        let mut server: Option<MCPServer> = None;
        if !step_server_id.is_empty() {
            server = self
                .tool_caller
                .find_server(mcp_servers, &step_server_id, &step_server_name)
                .cloned();
        }

        // Arguments the resolution below filled while picking the tool. Used
        // instead of a second extraction pass when they cover what the tool
        // requires — one call that saw the tool list, the request and the
        // prior results together beats two that each saw part of it.
        let mut resolved_arguments: Map<String, Value> = Map::new();
        // Whether those came from the model's native tool calling, which saw
        // the whole schema and may have filled optional parameters too.
        let mut resolved_native = false;

        if !is_system {
            // External MCP tools resolve through the MCPAgent: it reasons over
            // the tools discovery recorded for every connected server and
            // answers with one call, required arguments filled. There is no
            // keyword-scan fallback. An explicit, already-resolvable plan
            // choice (server present + exact tool schema) is honoured first so
            // a well-specified plan skips resolution.
            let explicit_ok = match server.as_ref() {
                Some(srv) => {
                    has_explicit_tool && self.tool_caller.find_tool_schema(srv, &tool_name).is_some()
                }
                None => false,
            };
            if !explicit_ok {
                // Same model that extracts the arguments below resolves the
                // call — the pick is the harder judgement of the two, so
                // leaving it on the default while upgrading extraction only
                // got the arguments right for the wrong tool.
                let picked = self.resolve(&directive, mcp_servers, Some(&prior_text), llm.clone());
                if let Some(picked) = picked {
                    server = self
                        .tool_caller
                        .find_server(mcp_servers, &picked.server_id, &picked.server_id)
                        .cloned();
                    if let Some(srv) = server.as_ref() {
                        // (Python also marked the tool explicit here; nothing
                        // below reads that flag again.)
                        tool_name = picked.tool_name.clone();
                        resolved_arguments = picked.arguments.clone();
                        resolved_native = picked.native;
                        let mut keys: Vec<String> = resolved_arguments.keys().cloned().collect();
                        keys.sort();
                        info!(
                            target: LOGGER,
                            "[resolve] MCPAgent resolved '{}' on server '{}' (thinking={}, native={}, args={})",
                            tool_name,
                            srv.name,
                            py_bool(picked.reasoned),
                            py_bool(picked.native),
                            py_list(&keys)
                        );
                    }
                }
            }
        }
        let Some(mut server) = server else {
            warn!(
                target: LOGGER,
                "[resolve] Server not found for tool '{}': {} (id={})",
                tool_name,
                py_opt(step.server_name.as_deref()),
                py_opt(step.server_id.as_deref())
            );
            return None;
        };

        // Tool schema — exact lookup only (no keyword guessing).
        let Some(tool_schema) = self.tool_caller.find_tool_schema(&server, &tool_name) else {
            warn!(
                target: LOGGER,
                "[resolve] Tool '{}' not found on server '{}' — skipping step",
                tool_name,
                server.name
            );
            return None;
        };

        let full_schema = schema_of(Some(&tool_schema));
        let mut input_schema = full_schema.clone();

        // When the step comes from an ```mcp block, scope extraction to only
        // the declared args.  inferred_args are extracted by the LLM;
        // default_args are filled in later.  This prevents the LLM from
        // hallucinating values for undeclared optional parameters.
        if let Some(inferred) = step.inferred_args.as_ref() {
            // Only expose inferred_args to the LLM for extraction.
            // default_args are fixed values — they are merged unconditionally
            // after extraction, so the LLM must not see (and potentially
            // override) them.
            let allowed_keys: HashSet<&str> = inferred.iter().map(String::as_str).collect();
            let properties = object_of(input_schema.get("properties"));
            let scoped_props: Map<String, Value> = properties
                .into_iter()
                .filter(|(k, _)| allowed_keys.contains(k.as_str()))
                .collect();
            let scoped_required: Vec<Value> = string_list(input_schema.get("required"))
                .into_iter()
                .filter(|r| allowed_keys.contains(r.as_str()))
                .map(Value::String)
                .collect();
            input_schema.insert("properties".into(), Value::Object(scoped_props));
            input_schema.insert("required".into(), Value::Array(scoped_required));
            let mut sorted: Vec<String> = allowed_keys.iter().map(|s| s.to_string()).collect();
            sorted.sort();
            info!(
                target: LOGGER,
                "[resolve] Scoped '{}' schema to inferred args: {}",
                tool_name,
                py_list(&sorted)
            );
        } else {
            // Prompt-derived step: only extract REQUIRED arguments. Optional
            // params are left unset so the tool applies its own defaults —
            // this stops the LLM fabricating optional filters (e.g. Linear
            // list_issues' priority/query/includeArchived) that wrongly narrow
            // or empty the result. Tools with no required params get no
            // inferred arguments at all.
            let required_raw = input_schema.get("required").cloned().unwrap_or_else(|| json!([]));
            let required = string_list(Some(&required_raw));
            let properties = object_of(input_schema.get("properties"));
            let scoped_props: Map<String, Value> =
                properties.into_iter().filter(|(k, _)| required.contains(k)).collect();
            input_schema.insert("properties".into(), Value::Object(scoped_props));
            input_schema.insert("required".into(), required_raw);
            let mut sorted = required.clone();
            sorted.sort();
            info!(
                target: LOGGER,
                "[resolve] Scoped '{}' schema to required args only: {}",
                tool_name,
                py_list(&sorted)
            );
        }
        let input_schema_value = Value::Object(input_schema.clone());

        let server_url = MCPToolCaller::get_server_url(&server).filter(|u| !u.is_empty());

        if server_url.is_none() && server.server_type != MCPServerType::Local {
            warn!(
                target: LOGGER,
                "[resolve] No server URL configured for: {}",
                py_opt(step.server_name.as_deref())
            );
            return None;
        }

        // Dependency analysis
        let all_tools = tool_catalog(&server);
        let dep_context = if prior_text.is_empty() {
            directive.clone()
        } else {
            format!("{directive}\n\nPrevious step results:\n{prior_text}")
        };

        let dep_analysis =
            self.dep_analyzer.invoke(&dep_context, &tool_name, &input_schema_value, &all_tools);
        if dep_analysis.has_dependencies && !dep_analysis.can_execute {
            info!(
                target: LOGGER,
                "[resolve] '{}' has unresolvable dependencies; falling through",
                tool_name
            );
            return None;
        }

        let mut resolved_ids = Map::new();
        if dep_analysis.has_dependencies && dep_analysis.can_execute {
            let prev_maps = previous_results.map(result_maps);
            resolved_ids = self.tool_caller.execute_prerequisites(
                &dep_analysis,
                &mut server,
                &directive,
                &input_schema,
                prev_maps.as_deref(),
            );
        }

        // Extract arguments — use the user's original message as the primary
        // input so the LLM extracts values from it, not from skill guidance
        // text that may contain metadata like skill names.
        //
        // For ```mcp block steps, use the user message plus the reasoning
        // report from the upstream reason node.  The report contains the
        // structured analysis that should inform tool arguments (e.g. issue
        // title/description derived from the analysis).  Raw explore dumps
        // and tool descriptions are excluded — they introduce noise that
        // small models pick up as parameter values.
        let has_prev = previous_results.is_some_and(|p| !p.is_empty());
        let extraction_input = if step.inferred_args.is_some() {
            // The reasoning report flows through prev_results from the reason
            // node (tool_name=REASON_NODE).
            let reasoning_report = previous_results.and_then(|prev| {
                prev.iter()
                    .find(|pr| pr.get("tool_name").and_then(Value::as_str) == Some(REASON_NODE))
                    .map(|pr| field_str(pr, "result"))
            });
            match reasoning_report.filter(|r| !r.is_empty()) {
                Some(report) => format!(
                    "{user_message}\n\n---\nAnalysis report:\n{}\n---",
                    head(&report, 4000)
                ),
                None => user_message.to_string(),
            }
        } else {
            // NOTE: Do NOT inject the tool description into the extraction
            // input — small models copy it verbatim into same-named params
            // (e.g. a `description` field). Values come from the user message
            // + upstream results only.
            //
            // When a reasoning report exists from upstream steps, use it as
            // the primary data source so the LLM derives parameter values from
            // actual findings — not from the raw user message.
            match ctx.skill_report.as_deref().filter(|r| !r.is_empty()) {
                Some(skill_report) if has_prev => format!(
                    "User request: {user_message}\n\n---\nAnalysis report (use these findings to fill parameter values):\n{}\n---",
                    ResponseBuilder::strip_tool_code_blocks(&head(skill_report, 4000))
                ),
                _ if has_prev => {
                    let prior_data = previous_results
                        .unwrap_or(&[])
                        .iter()
                        .map(|r| head(&field_str(r, "result"), 4000))
                        .collect::<Vec<_>>()
                        .join("\n");
                    format!(
                        "User request: {user_message}\n\n---\nData from previous step (use this to fill parameter values):\n{prior_data}\n---"
                    )
                }
                _ => user_message.to_string(),
            }
        };

        // NOTE: Do NOT append step.instruction to extraction_input.  The
        // instruction may contain skill body text (descriptions, tool names,
        // etc.) that small models pick up as parameter values.  The tool
        // schema already tells the LLM what parameters are needed; the user
        // message + previous results provide the values.

        // Try to fill all required params deterministically before invoking
        // the LLM.  When every required param is a URL and the URL is in the
        // user message, skip the LLM extraction entirely.
        let required_params_pre = string_list(input_schema.get("required"));
        let properties_pre = object_of(input_schema.get("properties"));
        let default_keys: HashSet<String> = step
            .default_args
            .as_ref()
            .map(|d| d.keys().cloned().collect())
            .unwrap_or_default();
        let mut deterministic_args = Map::new();
        let mut all_deterministic = !required_params_pre.is_empty();
        for pn in &required_params_pre {
            if default_keys.contains(pn) {
                continue;
            }
            let pi = properties_pre.get(pn).cloned().unwrap_or_else(empty_object);
            if self.arg_extractor.is_url_parameter(pn, &pi) {
                let mut urls = self.arg_extractor.extract_urls_from_text(user_message);
                if urls.is_empty() {
                    urls = self.arg_extractor.extract_urls_from_text(&directive);
                }
                if let Some(url) = urls.into_iter().next() {
                    deterministic_args.insert(pn.clone(), Value::String(url));
                    continue;
                }
            }
            all_deterministic = false;
            break;
        }

        let mut arguments: Map<String, Value> = if properties_pre.is_empty() {
            // No required/declared params to extract — call the tool with no
            // inferred arguments and let it apply its own defaults.
            info!(target: LOGGER, "[resolve] '{}' has no extractable params — no inferred arguments", tool_name);
            Map::new()
        } else if all_deterministic && !deterministic_args.is_empty() {
            info!(
                target: LOGGER,
                "[resolve] '{}' all required params filled deterministically: {}",
                tool_name,
                py_dict(&deterministic_args)
            );
            deterministic_args
        } else if !resolved_arguments.is_empty()
            && missing_required(&input_schema, &resolved_arguments).is_empty()
        {
            // Resolution already filled every required argument, in the call
            // that chose the tool — a second extraction pass would ask a
            // smaller question with less context and can only lose.  A partial
            // fill falls through to the extractor below rather than being
            // patched up here.
            // A native call keeps the optional arguments it filled — the
            // reason to call natively — unless an ```mcp block scoped the
            // step to its declared arguments.
            let properties_now = if resolved_native && step.inferred_args.is_none() {
                object_of(full_schema.get("properties"))
            } else {
                object_of(input_schema.get("properties"))
            };
            let args: Map<String, Value> = resolved_arguments
                .into_iter()
                .filter(|(k, _)| properties_now.contains_key(k))
                .collect();
            info!(
                target: LOGGER,
                "[resolve] '{}' arguments came from tool resolution: {}",
                tool_name,
                py_dict(&args)
            );
            args
        } else if !resolved_ids.is_empty() {
            self.arg_extractor.invoke_with_resolved_ids(
                &extraction_input,
                &tool_name,
                &input_schema_value,
                &resolved_ids,
                llm.as_deref(),
            )
        } else {
            self.extract(&extraction_input, &tool_name, &input_schema_value, previous_results, llm.as_deref())
        };

        // Merge skill-level default args.  When the step originates from an
        // ```mcp block, default_args are fixed values declared by the skill
        // author — they always win.  For other steps (skill_map / LLM-derived)
        // they act as fallbacks for missing values.
        if let Some(defaults) = step.default_args.as_ref().filter(|d| !d.is_empty()) {
            let force = step.inferred_args.is_some(); // mcp block → fixed
            for (dk, dv) in defaults {
                if force || !arguments.get(dk).is_some_and(is_truthy) {
                    arguments.insert(dk.clone(), Value::String(dv.clone()));
                    info!(target: LOGGER, "[resolve] '{}' applying default: {}={}", tool_name, dk, dv);
                }
            }
        }

        // Drop the extractor's bookkeeping key. The required-params check
        // below re-derives what is actually missing and can still satisfy URL
        // params deterministically from the directive — so we don't skip on
        // the extractor's report alone (e.g. a fetch whose URL is stated in
        // the instruction but which the LLM failed to extract).
        arguments.remove("_missing_params");

        let required_params = string_list(input_schema.get("required"));
        let properties = object_of(input_schema.get("properties"));

        // Deterministically fill a missing file-write `content` from the data
        // accumulated by upstream steps.  The content to write IS the prior
        // step output (fetched page, generated text, analysis report) — asking
        // a small model to regurgitate a large blob back into a JSON string is
        // unreliable and was leaving `content` empty, so the step got skipped.
        // Passing the accumulated data through directly is the whole point of
        // the fetch → generate → write chain.
        if tool_name == FILE_WRITE_TOOL
            && required_params.iter().any(|p| p == "content")
            && !arguments.get("content").is_some_and(is_truthy)
            && has_prev
        {
            let prev = previous_results.unwrap_or(&[]);
            let accumulated = prev
                .iter()
                .map(|r| field_str(r, "result").trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n");
            if !accumulated.is_empty() {
                info!(
                    target: LOGGER,
                    "[resolve] '{}' filled 'content' from {} accumulated step result(s) ({} chars)",
                    tool_name,
                    prev.len(),
                    accumulated.chars().count()
                );
                arguments.insert("content".into(), Value::String(accumulated));
            }
        }

        let mut still_missing: Vec<String> = Vec::new();
        for param_name in &required_params {
            if is_unset(arguments.get(param_name)) {
                let param_info = properties.get(param_name).cloned().unwrap_or_else(empty_object);
                if self.arg_extractor.is_url_parameter(param_name, &param_info) {
                    let mut urls = self.arg_extractor.extract_urls_from_text(&directive);
                    if urls.is_empty() {
                        urls = self.arg_extractor.extract_urls_from_text(user_message);
                    }
                    if let Some(url) = urls.into_iter().next() {
                        arguments.insert(param_name.clone(), Value::String(url));
                    } else {
                        still_missing.push(param_name.clone());
                    }
                } else {
                    still_missing.push(param_name.clone());
                }
            }
        }
        if !still_missing.is_empty() {
            info!(
                target: LOGGER,
                "[resolve] '{}' still missing required params ({}); skipping",
                tool_name,
                still_missing.join(", ")
            );
            return None;
        }

        if !self.drop_invalid(&mut arguments, &properties, &required_params, &tool_name, "resolve") {
            return None;
        }

        // Remove params not in the tool schema.  default_args keys are exempt
        // — they were declared by the skill author and may not be in the
        // scoped schema used for LLM extraction.
        let full_properties = object_of(full_schema.get("properties"));
        if !full_properties.is_empty() {
            let unknown: Vec<String> = arguments
                .keys()
                .filter(|k| !full_properties.contains_key(*k) && !default_keys.contains(*k))
                .cloned()
                .collect();
            if !unknown.is_empty() {
                info!(
                    target: LOGGER,
                    "[resolve] '{}' removing unknown params: {}",
                    tool_name,
                    py_list(&unknown)
                );
                for k in &unknown {
                    arguments.remove(k);
                }
            }
        }

        // Strip placeholder values
        strip_placeholders(&mut arguments, &required_params, &tool_name, "resolve");

        info!(
            target: LOGGER,
            "[resolve] '{}' resolved arguments: {}",
            tool_name,
            py_dict(&arguments)
        );
        Some((server, tool_name, arguments))
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
        info!(
            target: LOGGER,
            "[execute] Calling '{}' with arguments: {}",
            tool_name,
            py_dict(&arguments)
        );
        // A refresh writes the new credentials through the caller's persist
        // hook; the local copy only carries them for the retry.
        let mut server = server.clone();
        let (result, _refreshed) =
            self.tool_caller.call_with_refresh(&mut server, tool_name, Some(arguments), cwd);
        self.finish_call(&result, tool_name, "execute")
    }

    /// Run `llm`'s own tool loop for `instruction` over `mcp_servers` (see
    /// [`crate::agents::mcp::tool_loop`]), calling tools the way a planned
    /// step does.  The model is fed each call's raw result, which carries the
    /// ids its next call needs.  `None` when the loop could not start.
    pub fn run_tool_loop(
        &self,
        llm: &dyn BaseLLM,
        instruction: &str,
        mcp_servers: &[MCPServer],
        cwd: Option<&str>,
        emit: &dyn Fn(&str),
    ) -> Option<crate::agents::mcp::tool_loop::LoopOutcome> {
        let call = |server: &MCPServer, tool: &str, arguments: &Map<String, Value>| {
            self.call_tool(server, tool, arguments, cwd)
                .map(|(formatted, raw)| if raw.trim().is_empty() { formatted } else { raw })
        };
        crate::agents::mcp::tool_loop::run_tool_loop(&self.mcp_agent, llm, instruction, mcp_servers, &call, emit)
    }

    /// The planner-facing catalogue of usable connections: every enabled,
    /// non-failed server with tools, as
    /// `{"id", "name", "description", "summary", "server_type", "tools": [{"name", "description", "input_schema"}]}`.
    ///
    /// Reads nothing off `self` (Python tests call it unbound), so it is an
    /// associated function.
    pub fn build_available_servers(mcp_servers: &[MCPServer]) -> Vec<Value> {
        let mut available = Vec::new();
        for server in mcp_servers {
            if !mcp_connection_is_usable(server) {
                continue;
            }
            if server.tools.is_empty() {
                continue;
            }

            available.push(json!({
                "id": server.id,
                "name": server.name,
                "description": server.description,
                "summary": server.summary,
                "server_type": server.server_type.as_str(),
                "tools": tool_catalog(server),
            }));
        }

        available
    }

    // ------------------------------------------------------------------ helpers

    /// [`MCPAgent::resolve`], or the installed test hook.
    fn resolve(
        &self,
        directive: &str,
        mcp_servers: &[MCPServer],
        context: Option<&str>,
        llm: Option<Arc<dyn BaseLLM>>,
    ) -> Option<ResolvedCall> {
        let hook = self.resolve_hook.lock().clone();
        match hook {
            Some(hook) => hook(directive, mcp_servers, context, llm),
            None => self.mcp_agent.resolve(directive, mcp_servers, context, llm),
        }
    }

    /// [`MCPArgumentExtractor::invoke`], or the installed test hook.
    fn extract(
        &self,
        user_input: &str,
        tool_name: &str,
        input_schema: &Value,
        previous_tool_results: Option<&[Value]>,
        llm: Option<&dyn BaseLLM>,
    ) -> Map<String, Value> {
        let hook = self.extract_hook.lock().clone();
        match hook {
            Some(hook) => hook(user_input, tool_name, input_schema, previous_tool_results),
            None => self.arg_extractor.invoke(user_input, tool_name, input_schema, previous_tool_results, llm),
        }
    }

    /// Validate argument values: an invalid *required* value fails the step
    /// (returns `false`); an invalid optional one is dropped.
    fn drop_invalid(
        &self,
        arguments: &mut Map<String, Value>,
        properties: &Map<String, Value>,
        required_params: &[String],
        tool_name: &str,
        tag: &str,
    ) -> bool {
        let invalid = self.tool_caller.validate_argument_values(arguments, properties);
        if invalid.is_empty() {
            return true;
        }
        let hard_invalid: Vec<String> = invalid
            .iter()
            .filter(|msg| required_params.iter().any(|p| msg.contains(&format!("'{p}'"))))
            .cloned()
            .collect();
        if !hard_invalid.is_empty() {
            info!(
                target: LOGGER,
                "[{}] '{}' invalid required argument values: {}; falling through",
                tag,
                tool_name,
                py_list(&hard_invalid)
            );
            return false;
        }
        for msg in &invalid {
            let keys: Vec<String> = arguments.keys().cloned().collect();
            for p in keys {
                if msg.contains(&format!("'{p}'")) && !required_params.contains(&p) {
                    info!(
                        target: LOGGER,
                        "[{}] '{}' dropping invalid optional param '{}': {}",
                        tag,
                        tool_name,
                        p,
                        msg
                    );
                    arguments.remove(&p);
                }
            }
        }
        true
    }

    /// Turn a tool-call result into `(formatted, raw)`, or `None` when the
    /// call failed or came back empty / as an error.
    fn finish_call(
        &self,
        result: &MCPToolCallResult,
        tool_name: &str,
        tag: &str,
    ) -> Option<(String, String)> {
        if !result.success {
            warn!(
                target: LOGGER,
                "[{}] '{}' failed: {}",
                tag,
                tool_name,
                py_opt(result.error.as_deref())
            );
            return None;
        }
        let content = result.content.as_deref().unwrap_or(&[]);
        if content.is_empty() || result.is_error {
            let mut error_detail = String::new();
            if !content.is_empty() {
                let raw = content
                    .iter()
                    .map(|block| match block {
                        Value::Object(o) => o.get("text").map(value_str).unwrap_or_default(),
                        other => value_str(other),
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                let raw = raw.trim();
                if !raw.is_empty() {
                    error_detail = format!(": {}", head(raw, 300));
                }
            }
            warn!(
                target: LOGGER,
                "[{}] '{}' returned empty or error result{}",
                tag,
                tool_name,
                error_detail
            );
            return None;
        }

        let formatted_content =
            self.result_formatter.format_tool_result(tool_name, Some(content), result.is_error);
        let raw_content = content
            .iter()
            .map(|block| match block {
                Value::Object(o) if o.get("type").and_then(Value::as_str) == Some("text") => {
                    o.get("text").map(value_str).unwrap_or_default()
                }
                other => value_str(other),
            })
            .collect::<Vec<_>>()
            .join("\n");

        info!(
            target: LOGGER,
            "[{}] '{}' returned {} chars ({} tokens)",
            tag,
            tool_name,
            formatted_content.chars().count(),
            self.orchestrator_llm.get_num_tokens(&formatted_content)
        );
        info!(target: LOGGER, "[{}] '{}' raw result: {}", tag, tool_name, head(&raw_content, 1000));
        Some((formatted_content, raw_content))
    }
}

// ---------------------------------------------------------------------- //
// Free helpers
// ---------------------------------------------------------------------- //

/// Replace the server with `server.id` in the shared context's server list.
fn write_back_server(ctx: &SharedOrchestratorContext, server: &MCPServer) {
    let mut c = ctx.write();
    if let Some(slot) = c.mcp_servers.iter_mut().find(|s| s.id == server.id) {
        *slot = server.clone();
    }
}

/// Drop optional params whose value is an LLM placeholder ("Not specified").
fn strip_placeholders(
    arguments: &mut Map<String, Value>,
    required_params: &[String],
    tool_name: &str,
    tag: &str,
) {
    let keys: Vec<String> = arguments.keys().cloned().collect();
    for k in keys {
        let Some(Value::String(val)) = arguments.get(&k) else {
            continue;
        };
        if PLACEHOLDER_RE.is_match(val) && !required_params.contains(&k) {
            info!(
                target: LOGGER,
                "[{}] '{}' dropping placeholder optional param '{}': {}",
                tag,
                tool_name,
                k,
                head(val, 60)
            );
            arguments.remove(&k);
        }
    }
}

/// `[{name, description, input_schema}]` for every tool on `server`.
fn tool_catalog(server: &MCPServer) -> Vec<Value> {
    server
        .tools
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            })
        })
        .collect()
}

/// `tool_schema.get('input_schema', {}) or {}` as an object.
fn schema_of(tool_schema: Option<&Map<String, Value>>) -> Map<String, Value> {
    tool_schema.map(|s| object_of(s.get("input_schema"))).unwrap_or_default()
}

/// The object at `v`, or an empty one.
fn object_of(v: Option<&Value>) -> Map<String, Value> {
    match v {
        Some(Value::Object(m)) => m.clone(),
        _ => Map::new(),
    }
}

fn empty_object() -> Value {
    Value::Object(Map::new())
}

/// The strings in the array at `v` (non-strings are skipped).
fn string_list(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::Array(items)) => items.iter().filter_map(|i| i.as_str().map(str::to_string)).collect(),
        _ => Vec::new(),
    }
}

/// The previous results that are objects, as maps.
fn result_maps(prev: &[Value]) -> Vec<Map<String, Value>> {
    prev.iter().filter_map(|v| v.as_object().cloned()).collect()
}

/// `"\n".join(f"[{tool_name}]: {str(result)[:500]}")` over prior results.
fn prior_results_text(prev: &[Value]) -> String {
    prev.iter()
        .map(|r| {
            let tool = match r.get("tool_name") {
                Some(v) => value_str(v),
                None => "tool".to_string(),
            };
            format!("[{}]: {}", tool, head(&field_str(r, "result"), 500))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `str(r.get(key, ""))`.
fn field_str(r: &Value, key: &str) -> String {
    r.get(key).map(value_str).unwrap_or_default()
}

/// `str(r.get("result", "") or r.get("content", ""))`.
fn result_or_content(r: &Value) -> String {
    match r.get("result") {
        Some(v) if is_truthy(v) => value_str(v),
        _ => r.get("content").map(value_str).unwrap_or_default(),
    }
}

/// Python `str(value)`: strings verbatim, `None` as `"None"`, booleans as
/// `True`/`False`, everything else as JSON.
fn value_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        Value::Bool(b) => py_bool(*b).to_string(),
        other => other.to_string(),
    }
}

/// Python truthiness of a JSON value.
fn is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `val is None or val == "" or val == "null"`.
fn is_unset(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => true,
        Some(Value::String(s)) => s.is_empty() || s == "null",
        _ => false,
    }
}

/// `", ".join(missing)` for a `_missing_params` value.
fn join_values(v: &Value) -> String {
    match v {
        Value::Array(items) => items.iter().map(value_str).collect::<Vec<_>>().join(", "),
        other => value_str(other),
    }
}

/// The first `n` characters of `s` (Python `s[:n]`).
fn head(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Python repr of a JSON value: `{'k': 'v'}`, `None`, `True`.
fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(b) => py_bool(*b).to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'")),
        Value::Array(items) => {
            format!("[{}]", items.iter().map(py_repr).collect::<Vec<_>>().join(", "))
        }
        Value::Object(map) => py_dict(map),
    }
}

/// Python repr of a dict (log lines print argument dicts this way).
fn py_dict(map: &Map<String, Value>) -> String {
    let inner = map
        .iter()
        .map(|(k, v)| format!("{}: {}", py_repr(&Value::String(k.clone())), py_repr(v)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{{inner}}}")
}

/// Python `bool` repr.
fn py_bool(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

/// Python repr of an optional string in an f-string (`None` when unset).
fn py_opt(s: Option<&str>) -> String {
    s.map(str::to_string).unwrap_or_else(|| "None".to_string())
}

/// Python repr of a list of strings: `['a', 'b']`.
fn py_list(items: &[String]) -> String {
    let inner = items.iter().map(|s| format!("'{s}'")).collect::<Vec<_>>().join(", ");
    format!("[{inner}]")
}

/// The first non-empty option (`a or b or c`).
fn first_non_empty(options: &[Option<&str>]) -> String {
    options
        .iter()
        .flatten()
        .find(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_default()
}

/// `os.getcwd()`.
fn current_dir() -> String {
    std::env::current_dir().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(n: i64, instruction: &str) -> PlanStep {
        PlanStep { step: n, instruction: instruction.into(), ..Default::default() }
    }

    #[test]
    fn independent_steps_batch_and_dependent_ones_flush() {
        let mut conditioned = step(3, "check");
        conditioned.condition = Some("result_contains:x".into());
        let mut summarize = step(5, "sum up");
        summarize.step_type = StepType::Action;
        summarize.action_kind = Some("summarize".into());
        let steps = vec![
            step(1, "list issues"),
            step(2, "list teams"),
            conditioned,
            step(4, "use the result of step 1"),
            summarize,
            step(6, "fetch docs"),
        ];

        let groups = StepExecutor::group_independent_steps(&steps);
        let numbers: Vec<Vec<i64>> =
            groups.iter().map(|g| g.iter().map(|s| s.step).collect()).collect();
        assert_eq!(numbers, vec![vec![1, 2], vec![3], vec![4], vec![5], vec![6]]);
    }

    #[test]
    fn plan_step_result_falls_back_to_content_and_step_type() {
        let mut s = step(1, "x");
        s.step_type = StepType::Explore;
        let r = plan_step_result("n1", &s, "body", None);
        assert!(r.success);
        assert_eq!(
            r.data,
            json!({
                "tool_name": "explore",
                "content": "body",
                "raw": "body",
                "is_tool_result": false,
                "server_id": null,
                "server_name": null,
            })
        );

        s.step_type = StepType::Tool;
        s.tool_name = Some("create_issue".into());
        s.server_id = Some("linear-1".into());
        s.server_name = Some("Linear".into());
        let r = plan_step_result("n1", &s, "fmt", Some("raw"));
        assert_eq!(r.data["tool_name"], "create_issue");
        assert_eq!(r.data["raw"], "raw");
        assert_eq!(r.data["is_tool_result"], true);
        // Names the server so the answer can cite the tool's source.
        assert_eq!(r.data["server_id"], "linear-1");
        assert_eq!(r.data["server_name"], "Linear");
    }

    #[test]
    fn placeholder_regex_matches_python() {
        assert!(PLACEHOLDER_RE.is_match("Not specified"));
        assert!(PLACEHOLDER_RE.is_match("needs to be provided by the user"));
        assert!(PLACEHOLDER_RE.is_match("N/A"));
        assert!(PLACEHOLDER_RE.is_match("Unknown"));
        assert!(!PLACEHOLDER_RE.is_match("Login is broken"));
    }

    #[test]
    fn python_reprs() {
        let m = json!({"a": "it's", "b": null, "c": [1, true]});
        assert_eq!(py_dict(m.as_object().unwrap()), r"{'a': 'it\'s', 'b': None, 'c': [1, True]}");
        assert_eq!(py_list(&["x".into(), "y".into()]), "['x', 'y']");
        assert!(is_unset(Some(&json!("null"))));
        assert!(!is_unset(Some(&json!(0))));
        assert!(!is_truthy(&json!([])));
    }
}
