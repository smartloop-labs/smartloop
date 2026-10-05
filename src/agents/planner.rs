//! Port of `smartloop/agents/planner.py`.
//!
//! Planner — Classify → Plan pipeline for the MCP orchestration system.
//!
//! Phase 1 (classify): Determines what kind of work a user message requires
//!     — tool call, built-in action, explore, multi-step, or conversation.
//! Phase 2 (create_plan): Builds a concrete step list from the classification.
//!
//! The orchestrator handles Phase 3 (execute) and Phase 4 (return results).

use std::collections::BTreeMap;
use std::sync::Arc;

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[allow(unused_imports)]
use crate::llms::BaseLLM;
#[allow(unused_imports)]
use crate::paths::expand_home;
#[allow(unused_imports)]
use crate::types::ModelCapability;
use crate::workspace::workspace::{MCPServer, MCPServerStatus};

// Tool name constants — duplicated here to avoid circular imports with
// smartloop.agents.mcp.server_selector.
pub const BASH_TOOL: &str = "bash";
pub const GREP_TOOL: &str = "grep";
pub const FILE_READ_TOOL: &str = "file-read";
pub const FILE_WRITE_TOOL: &str = "file-write";
pub const EXPLORE_TOOL: &str = "explore";

/// Python logger name for this module.
pub const LOGGER: &str = "smartloop.agents.planner";

// ---------------------------------------------------------------------- //
// Models                                                                  //
// ---------------------------------------------------------------------- //

pub const SYSTEM_TOOLS_SERVER_ID: &str = "__system_tools__";
pub const SYSTEM_TOOLS_SERVER_NAME: &str = "smartloop";
pub const FETCH_SERVER_ID: &str = "__fetch__";

/// Duck-typed view of a registered MCP connection, read by
/// [`mcp_connection_is_usable`].
///
/// Python read `enabled` (default `True`) and `status` (an `MCPServerStatus`
/// or its bare string value, default `None`) via `getattr`.
pub trait McpConnectionState {
    /// Whether the connection is enabled (`True` when the record has no field).
    fn connection_enabled(&self) -> bool {
        true
    }
    /// The status wire value (`"pending"`, `"ready"`, `"failed"`), or `None`
    /// when the record has no status.
    fn connection_status(&self) -> Option<String> {
        None
    }
}

impl McpConnectionState for MCPServer {
    fn connection_enabled(&self) -> bool {
        self.enabled
    }
    fn connection_status(&self) -> Option<String> {
        Some(self.status.as_str().to_string())
    }
}

impl McpConnectionState for MCPServerStatus {
    fn connection_status(&self) -> Option<String> {
        Some(self.as_str().to_string())
    }
}

/// Dict-shaped records: reads the `enabled` and `status` keys.
impl McpConnectionState for Value {
    fn connection_enabled(&self) -> bool {
        self.get("enabled").and_then(Value::as_bool).unwrap_or(true)
    }
    fn connection_status(&self) -> Option<String> {
        self.get("status").and_then(Value::as_str).map(str::to_string)
    }
}

impl<T: McpConnectionState + ?Sized> McpConnectionState for &T {
    fn connection_enabled(&self) -> bool {
        (**self).connection_enabled()
    }
    fn connection_status(&self) -> Option<String> {
        (**self).connection_status()
    }
}

/// Whether a registered connection can still be called at all.
///
/// Three places build a list of connections to offer the model — the
/// planner's catalog, the collect pass and the MCP agent's tool catalog — and
/// each grew its own filter.  They drifted: only the planner's checked whether
/// the connection could still authenticate, so a connection whose grant the
/// provider had retired went on being offered by the other two.  It keeps its
/// cached tool list, so it looks healthy, and every call 401s into a silent
/// skip — which reads as "that app has nothing in it" rather than "reconnect
/// it".
///
/// Only the two conditions that are genuinely common live here.  Which
/// connections a caller cares about — whether the system-tools server belongs
/// in its list, whether a server with no tools is worth an entry — is that
/// caller's own policy and stays at the call site.
pub fn mcp_connection_is_usable<S: McpConnectionState + ?Sized>(server: &S) -> bool {
    let _ = server;
    todo!()
}

/// Which model tier a step should run on.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, Hash)]
pub enum ModelTier {
    /// Routing, classification, arg extraction (1B)
    #[default]
    #[serde(rename = "small")]
    Small,
    /// Skills, synthesis, complex reasoning (4B+)
    #[serde(rename = "large")]
    Large,
}

impl ModelTier {
    /// The enum's wire value.
    pub fn value(&self) -> &'static str {
        match self {
            ModelTier::Small => "small",
            ModelTier::Large => "large",
        }
    }
}

/// Semantic action categories available in an execution plan.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, Hash)]
pub enum StepType {
    /// LLM reasons about user intent and context before planning
    #[serde(rename = "analyze")]
    Analyze,
    /// Call an external MCP tool
    #[default]
    #[serde(rename = "tool")]
    Tool,
    /// Fetch a URL → markdown via the FetchAgent
    #[serde(rename = "fetch")]
    Fetch,
    /// Pull latest info/events via the WebSearchAgent
    #[serde(rename = "web_search")]
    WebSearch,
    /// Walk the local filesystem tree + vector search
    #[serde(rename = "explore")]
    Explore,
    /// Built-in action: bash, write, read, grep, summarize
    #[serde(rename = "action")]
    Action,
    /// LLM generates content from knowledge
    #[serde(rename = "conversational")]
    Conversational,
}

impl StepType {
    /// The enum's wire value.
    pub fn value(&self) -> &'static str {
        match self {
            StepType::Analyze => "analyze",
            StepType::Tool => "tool",
            StepType::Fetch => "fetch",
            StepType::WebSearch => "web_search",
            StepType::Explore => "explore",
            StepType::Action => "action",
            StepType::Conversational => "conversational",
        }
    }

    /// `StepType(value)` — parse a wire value; `None` for an unknown value.
    pub fn from_value(value: &str) -> Option<StepType> {
        match value {
            "analyze" => Some(StepType::Analyze),
            "tool" => Some(StepType::Tool),
            "fetch" => Some(StepType::Fetch),
            "web_search" => Some(StepType::WebSearch),
            "explore" => Some(StepType::Explore),
            "action" => Some(StepType::Action),
            "conversational" => Some(StepType::Conversational),
            _ => None,
        }
    }
}

/// High-level intent categories produced by classification.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, Hash)]
pub enum IntentType {
    /// Single MCP tool invocation
    #[serde(rename = "tool_call")]
    ToolCall,
    /// Single built-in action (bash, read, write, grep)
    #[serde(rename = "action")]
    Action,
    /// Filesystem exploration + vector search
    #[serde(rename = "explore")]
    Explore,
    /// 2+ steps needed
    #[serde(rename = "multi_step")]
    MultiStep,
    /// Pure LLM response, no tools
    #[default]
    #[serde(rename = "conversation")]
    Conversation,
}

impl IntentType {
    /// The enum's wire value.
    pub fn value(&self) -> &'static str {
        match self {
            IntentType::ToolCall => "tool_call",
            IntentType::Action => "action",
            IntentType::Explore => "explore",
            IntentType::MultiStep => "multi_step",
            IntentType::Conversation => "conversation",
        }
    }
}

/// Deterministic signals detected in user input.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Signal {
    /// MCP tool keyword match
    #[serde(rename = "tools")]
    Tools,
    /// File or directory path(s)
    #[serde(rename = "paths")]
    Paths,
    /// Bash code block
    #[serde(rename = "bash")]
    Bash,
    /// Write intent keywords
    #[serde(rename = "write")]
    Write,
    /// Grep/search keywords
    #[serde(rename = "grep")]
    Grep,
    /// HTTP(S) URL detected
    #[serde(rename = "url")]
    Url,
}

/// An MCP tool matched by keyword detection.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MatchedTool {
    pub server_id: String,
    pub server_name: String,
    pub tool_name: String,
}

/// Output of the classify phase — carries all detected signals.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ClassificationResult {
    #[serde(default)]
    pub intent: IntentType,
    #[serde(default)]
    pub detected_urls: Vec<String>,
    #[serde(default)]
    pub detected_paths: Vec<String>,
    #[serde(default)]
    pub matched_tools: Vec<MatchedTool>,
    #[serde(default)]
    pub has_write_intent: bool,
    #[serde(default)]
    pub has_grep_intent: bool,
    #[serde(default)]
    pub bash_commands: Vec<String>,
    #[serde(default)]
    pub action_kind: Option<String>,
    #[serde(default)]
    pub has_document_context: bool,
    #[serde(default)]
    pub reasoning: String,
    /// Best-matching model capability for this request (e.g. coding, finance, vision)
    #[serde(default)]
    pub capability: Option<String>,
}

/// A single step in the LLM-generated execution plan.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct PlanStep {
    /// Step number (1-based)
    pub step: i64,
    #[serde(default)]
    pub step_type: StepType,
    /// Detailed, self-contained instruction for this step — specifies what to
    /// look for, which file/URL to use, and what to do with the result.
    pub instruction: String,
    // For tool steps
    /// Exact name of the tool to call
    #[serde(default)]
    pub tool_name: Option<String>,
    /// Server ID
    #[serde(default)]
    pub server_id: Option<String>,
    /// Human-readable server name
    #[serde(default)]
    pub server_name: Option<String>,
    // For action steps
    /// Kind of built-in action: bash, write, read, grep, summarize
    #[serde(default)]
    pub action_kind: Option<String>,
    // Conditional execution & provenance
    /// Condition for execution: 'result_contains:X' or 'llm:question'
    #[serde(default)]
    pub condition: Option<String>,
    /// Name of the skill this step originated from
    #[serde(default)]
    pub skill_name: Option<String>,
    /// Fixed argument values specified in the skill body (e.g. team=DEV)
    #[serde(default)]
    pub default_args: Option<BTreeMap<String, String>>,
    /// Bare argument names the LLM should extract from user input
    #[serde(default)]
    pub inferred_args: Option<Vec<String>>,
    /// Iterative tool calling — when true, call the tool once per item
    /// extracted from upstream results
    #[serde(default)]
    pub for_each: Option<bool>,
    /// Which model tier this step should run on
    #[serde(default)]
    pub model_tier: ModelTier,
    /// Requested model capability for this step (e.g. coding, finance, vision)
    #[serde(default)]
    pub capability: Option<String>,
}

/// An LLM-generated ordered plan of tool calls to fulfil a user request.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ExecutionPlan {
    /// Ordered steps to execute
    #[serde(default)]
    pub steps: Vec<PlanStep>,
    /// Brief explanation of why these steps were chosen
    #[serde(default)]
    pub reasoning: String,
}

/// Describes one MCP server and its tools for the planning prompt.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MCPToolContext {
    pub server_id: String,
    pub server_name: String,
    #[serde(default)]
    pub tools: Vec<Map<String, Value>>,
}

/// Structured context assembled for the planner.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct PlanContext {
    pub user_message: String,
    #[serde(default)]
    pub skills: Option<Vec<String>>,
    #[serde(default)]
    pub mcp_tools: Vec<MCPToolContext>,
    #[serde(default)]
    pub document_context: Option<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub detected_paths: Vec<String>,
    #[serde(default)]
    pub detected_urls: Vec<String>,
    #[serde(default)]
    pub skill_maps: Vec<Map<String, Value>>,
}

/// Optional keyword arguments of [`Planner::classify`].
#[derive(Debug, Clone, Default)]
pub struct ClassifyOptions {
    pub skills: Option<Vec<String>>,
    pub document_context: Option<String>,
    pub has_attachments: bool,
    pub history: Option<String>,
    pub cwd: Option<String>,
    pub available_capabilities: Option<Vec<String>>,
}

// ---------------------------------------------------------------------- //
// Planner                                                                 //
// ---------------------------------------------------------------------- //

/// Regex to find file/directory-like paths in user messages.
pub static PATH_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)(?:^|\s)([~/.][\w./\-]+|/[\w./\-]+)").unwrap());

/// Regex to detect a ```bash ... ``` fenced code block in user text.
pub static BASH_BLOCK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)```bash\s*\n(.+?)```").unwrap());

/// Multi-word phrases that signal file-write intent.
/// Single words like "create" or "make" are too ambiguous (e.g. "create a recipe")
/// and cause false positives — they're better handled by the LLM classifier.
pub const WRITE_KEYWORDS: &[&str] = &[
    "write to",
    "write a file",
    "write file",
    "write it to",
    "save to",
    "save as",
    "save file",
    "save it to",
    "create file",
    "create a file",
];

/// Keywords that signal a search/grep intent.
pub const GREP_KEYWORDS: &[&str] = &[
    "search for",
    "grep",
    "find in files",
    "look for",
    "search across",
];

/// URL extraction pattern (simple but effective).
pub static URL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"https?://[^\s<>'")\]]+"#).unwrap());

/// Keywords that identify a URL-fetching tool by name or description.
pub const FETCH_TOOL_KEYWORDS: &[&str] =
    &["fetch", "browse", "scrape", "web", "http", "navigate", "get_url"];

/// Classifies user intent and creates execution plans.
#[derive(Clone)]
pub struct Planner {
    pub llm: Arc<dyn BaseLLM>,
}

impl Planner {
    /// Create a planner backed by `llm`.
    pub fn new(llm: Arc<dyn BaseLLM>) -> Self {
        Self { llm }
    }

    // ------------------------------------------------------------------ //
    // Phase 1: Classification                                            //
    // ------------------------------------------------------------------ //

    /// Classify user intent into one of the [`IntentType`] categories.
    ///
    /// Tier 1 (deterministic): Pattern-match URLs, paths, bash blocks,
    /// MCP tool keywords, write/grep intent.  No LLM call.
    ///
    /// Tier 2 (LLM fallback): Compact prompt when Tier 1 finds no signals.
    pub fn classify(
        &self,
        user_message: &str,
        mcp_tools: &[MCPToolContext],
        options: ClassifyOptions,
    ) -> ClassificationResult {
        let _ = (user_message, mcp_tools, options);
        todo!()
    }

    /// Pick a capability from planning signals in priority order.
    ///
    /// Priority:
    ///   1. `vision`    — attachments imply image/document OCR work.
    ///   2. `reasoning` — a matched skill defines the procedure to run.
    ///   3. `dp`        — indexed document context: data-processing model.
    ///   4. `general`   — default for open-ended chat / general answers.
    ///
    /// Returns the first capability whose signal fires *and* is present
    /// in `capabilities`, or `None` when nothing matches.
    pub fn match_capability(
        capabilities: &[String],
        has_attachments: bool,
        has_matched_skills: bool,
        has_document_context: bool,
    ) -> Option<String> {
        let _ = (capabilities, has_attachments, has_matched_skills, has_document_context);
        todo!()
    }

    /// Extract all deterministic signals from the user message.
    pub fn detect_signals(
        &self,
        user_message: &str,
        mcp_tools: &[MCPToolContext],
        cwd: Option<&str>,
        document_context: Option<&str>,
        has_attachments: bool,
    ) -> ClassificationResult {
        let _ = (user_message, mcp_tools, cwd, document_context, has_attachments);
        todo!()
    }

    /// Derive intent from detected signals.  Returns `None` when ambiguous.
    pub fn resolve_intent(&self, result: &mut ClassificationResult) -> Option<IntentType> {
        let _ = result;
        todo!()
    }

    /// Lightweight LLM call to classify ambiguous messages.
    fn llm_classify(
        &self,
        user_message: &str,
        mcp_tools: &[MCPToolContext],
        skills: Option<&[String]>,
        history: Option<&str>,
        result: ClassificationResult,
    ) -> ClassificationResult {
        let _ = (user_message, mcp_tools, skills, history, result);
        todo!()
    }

    // ------------------------------------------------------------------ //
    // Phase 2: Plan Creation                                             //
    // ------------------------------------------------------------------ //

    /// Assemble a [`PlanContext`] from raw inputs.
    ///
    /// `available_servers` are dict-shaped server records (`id`, `name`,
    /// `tools`); the system-tools and fetch servers are skipped.
    pub fn build_context(
        &self,
        user_message: &str,
        available_servers: &[Value],
        cwd: Option<&str>,
        skills: Option<Vec<String>>,
        document_context: Option<&str>,
    ) -> PlanContext {
        let _ = (user_message, available_servers, cwd, skills, document_context);
        todo!()
    }

    /// Build an execution plan from a classification result.
    ///
    /// `llm` overrides the model used for LLM-based multi-step planning —
    /// the caller passes a stronger model (e.g. the large 'dp' model) for the
    /// complex tool-selection reasoning. Falls back to `self.llm`.
    ///
    /// For CONVERSATION with a matched capability (e.g. `general`) →
    /// one CONVERSATIONAL step routed to that capability's model.
    /// For CONVERSATION without a capability → empty plan (router streams
    /// via the small model directly).
    /// For TOOL_CALL, ACTION, EXPLORE → single-step plan (no LLM call).
    /// For MULTI_STEP → LLM-generated plan with fallback to signal-based steps.
    pub fn create_plan(
        &self,
        classification: &ClassificationResult,
        context: &PlanContext,
        llm: Option<&dyn BaseLLM>,
    ) -> ExecutionPlan {
        let _ = (classification, context, llm);
        todo!()
    }

    /// Single MCP tool step.
    fn plan_tool_call(&self, classification: &ClassificationResult, context: &PlanContext) -> ExecutionPlan {
        let _ = (classification, context);
        todo!()
    }

    /// Single built-in system tool step.
    ///
    /// System tools run as ordinary TOOL steps against the local `smartloop`
    /// MCP server (like fetch).  Where the classifier already extracted a
    /// concrete value (bash command, file path) it is carried as
    /// `default_args` so resolution stays deterministic; otherwise the
    /// argument extractor fills the args from the user message.
    fn plan_action(&self, classification: &ClassificationResult, context: &PlanContext) -> ExecutionPlan {
        let _ = (classification, context);
        todo!()
    }

    /// Single explore step — execution handles fetch → vector → files internally.
    fn plan_explore(&self, classification: &ClassificationResult, context: &PlanContext) -> ExecutionPlan {
        let _ = (classification, context);
        todo!()
    }

    /// Build a multi-step plan.
    ///
    /// Tries the LLM for step ordering; falls back to building steps
    /// directly from the classification's detected signals.
    fn plan_multi_step(
        &self,
        classification: &ClassificationResult,
        context: &PlanContext,
        llm: Option<&dyn BaseLLM>,
    ) -> ExecutionPlan {
        let _ = (classification, context, llm);
        todo!()
    }

    /// Call LLM to produce a step ordering for multi-step requests.
    fn llm_plan(
        &self,
        classification: &ClassificationResult,
        context: &PlanContext,
        llm: Option<&dyn BaseLLM>,
    ) -> ExecutionPlan {
        let _ = (classification, context, llm);
        todo!()
    }

    /// Build plan steps directly from classification signals (no LLM).
    fn build_steps_from_signals(
        &self,
        classification: &ClassificationResult,
        context: &PlanContext,
    ) -> ExecutionPlan {
        let _ = (classification, context);
        todo!()
    }

    // ------------------------------------------------------------------ //
    // Helpers                                                            //
    // ------------------------------------------------------------------ //

    /// Build a FETCH step that the orchestrator runs via the FetchAgent.
    ///
    /// Fetching is no longer an MCP tool call — the FetchAgent performs the
    /// HTTP GET and converts HTML/PDF to markdown in-process.  The URL is
    /// carried in `default_args` so no argument resolution is needed.
    fn make_fetch_step(&self, url: &str, context: &PlanContext) -> PlanStep {
        let _ = (url, context);
        todo!()
    }

    // ------------------------------------------------------------------ //
    // Response parsing                                                   //
    // ------------------------------------------------------------------ //

    /// Parse the raw LLM response into an [`ExecutionPlan`].
    pub fn parse_plan_response(
        &self,
        response: &str,
        user_message: &str,
        classification: Option<&ClassificationResult>,
    ) -> ExecutionPlan {
        let _ = (response, user_message, classification);
        todo!()
    }

    /// Derive [`StepType`] from `tool_name` and `server_id`.
    ///
    /// `explore` is the only internal pseudo-step; the other built-in
    /// tools (file/grep/bash) run as ordinary TOOL steps against the local
    /// smartloop MCP server.
    pub fn infer_step_type(tool_name: &str, server_id: &str) -> StepType {
        let _ = (tool_name, server_id);
        todo!()
    }
}

// ---------------------------------------------------------------------- //
// Module-level helpers                                                    //
// ---------------------------------------------------------------------- //

/// Return real filesystem paths found in `text` (expanded, made absolute
/// against `cwd`, normalised, de-duplicated, existing only).
pub fn extract_paths(text: &str, cwd: &str) -> Vec<String> {
    let _ = (text, cwd);
    todo!()
}
