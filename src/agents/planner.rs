//! Port of `smartloop/agents/planner.py`.
//!
//! Planner — Classify → Plan pipeline for the MCP orchestration system.
//!
//! Phase 1 (classify): Determines what kind of work a user message requires
//!     — tool call, built-in action, explore, multi-step, or conversation.
//! Phase 2 (create_plan): Builds a concrete step list from the classification.
//!
//! The orchestrator handles Phase 3 (execute) and Phase 4 (return results).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::llms::BaseLLM;
use crate::paths::expand_home;
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
    if !server.connection_enabled() {
        return false;
    }
    server.connection_status().as_deref() != Some("failed")
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
    /// The request is programming work (see [`CODING_RE`]).  A modifier, like
    /// write/grep: it picks the model, not the intent.
    #[serde(default)]
    pub has_coding_intent: bool,
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

/// Files at a folder's top that make it a code project.
pub const PROJECT_MARKERS: &[&str] = &[
    "Cargo.toml",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "Gemfile",
    "composer.json",
    "Package.swift",
    "CMakeLists.txt",
    "mix.exs",
];

/// Whether `dir` is a code project: one of [`PROJECT_MARKERS`] sits at its
/// top.
pub fn is_code_project(dir: &str) -> bool {
    let dir = expand_home(dir);
    let dir = Path::new(&dir);
    PROJECT_MARKERS.iter().any(|marker| dir.join(marker).is_file())
}

/// What makes a request coding work once the turn is in a code project:
/// a verb that changes something, or a noun of the code itself.  Looser
/// than [`CODING_RE`], which has to stand on its own anywhere.
pub static IN_PROJECT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concat!(
        r"(?i)\b(?:add|fix|create|implement|write|change|update|refactor|rename|remove|delete|move|build|run|",
        r"test|debug|generate|scaffold|wire|hook|support|make|extract|split|merge|migrate|upgrade|bump|",
        r"lint|format|compile|deploy|release|install)\b",
        r"|\b(?:function|fn|method|module|struct|class|trait|enum|interface|type|file|crate|package|",
        r"dependency|dependencies|import|endpoint|route|handler|api|cli|command|flag|option|config|",
        r"test|tests|bug|error|panic|crash|exception|warning|compile|build|lint|readme|changelog)s?\b",
    ))
    .unwrap()
});

/// Signals that the request is programming work, so it routes to a
/// `coding` model.  Kept to phrases that rarely mean anything else: a
/// fenced code block, a source-file name, a language name, a coding verb
/// bound to a code noun, or a language/framework naming a project ("a rust
/// project", "react app").  Bare words like "code" or "compile" are left out —
/// "zip code" and "compile a list" are not programming.
pub static CODING_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concat!(
        r"(?i)```",
        r"|\b[\w\-]+\.(?:rs|py|ts|tsx|js|jsx|go|java|kt|swift|rb|cpp|cc|hpp|cs|php|scala|sh)\b",
        r"|\b(?:python|javascript|typescript|golang|kotlin|rustlang|c\+\+|c#)",
        r"|\b(?:refactor\w*|debug(?:ging)?|stack ?trace|traceback|segfault|syntax error|compiler|compil(?:e|ation) error",
        r"|unit tests?|regex|regular expression|code ?review|codebase|source code|code snippet",
        r"|(?:this|my|following|your) code|sql|bash script|shell script)\b",
        r"|\b(?:write|implement|refactor|optimi[sz]e) (?:a |an |the |this |my )?",
        r"(?:function|method|class|program|algorithm|endpoint|api|cli|unit test)\b",
        r"|\bfix (?:a |the |this |my )?bug\b",
        r"|\b(?:rust|go|golang|node(?:\.?js)?|deno|react|vue|svelte|angular|next\.?js|django|flask|fastapi|rails|",
        r"java|kotlin|python|typescript|javascript|c\+\+|c#|elixir|haskell|zig)",
        r"\s+(?:project|app|application|crate|package|library|module|cli|service|server|api|website|starter)\b",
    ))
    .unwrap()
});

/// URL extraction pattern (simple but effective).
pub static URL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"https?://[^\s<>'")\]]+"#).unwrap());

/// Keywords that identify a URL-fetching tool by name or description.
pub const FETCH_TOOL_KEYWORDS: &[&str] =
    &["fetch", "browse", "scrape", "web", "http", "navigate", "get_url"];

/// The classify intent words the LLM fallback may answer with, in the
/// order they are searched for in its reply.
const INTENT_MAP: &[(&str, IntentType)] = &[
    ("TOOL_CALL", IntentType::ToolCall),
    ("ACTION", IntentType::Action),
    ("EXPLORE", IntentType::Explore),
    ("DOCUMENT", IntentType::Explore),
    ("MULTI_STEP", IntentType::MultiStep),
    ("CONVERSATION", IntentType::Conversation),
];

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
        let ClassifyOptions {
            skills,
            document_context,
            has_attachments,
            history,
            cwd,
            available_capabilities,
        } = options;

        let mut result = self.detect_signals(
            user_message,
            mcp_tools,
            cwd.as_deref(),
            document_context.as_deref(),
            has_attachments,
        );

        // Resolve intent from detected signals
        match self.resolve_intent(&mut result) {
            Some(intent) => {
                result.intent = intent;
                let tools: Vec<&str> =
                    result.matched_tools.iter().map(|t| t.tool_name.as_str()).collect();
                tracing::info!(
                    target: "smartloop.agents.planner",
                    "[classify] deterministic → {} (tools={}, urls={}, paths={}, write={}, grep={}, bash={})",
                    intent.value(),
                    py_str_list(&tools),
                    result.detected_urls.len(),
                    result.detected_paths.len(),
                    py_bool(result.has_write_intent),
                    py_bool(result.has_grep_intent),
                    result.bash_commands.len(),
                );
            }
            None => {
                // Tier 2: LLM fallback
                result = self.llm_classify(
                    user_message,
                    mcp_tools,
                    skills.as_deref(),
                    history.as_deref(),
                    result,
                );
                tracing::info!(
                    target: "smartloop.agents.planner",
                    "[classify] LLM → {}: {}",
                    result.intent.value(),
                    result.reasoning
                );
            }
        }

        if let Some(capabilities) = available_capabilities.as_deref() {
            if !capabilities.is_empty() {
                result.capability = Self::match_capability(
                    capabilities,
                    has_attachments,
                    result.has_coding_intent,
                    skills.as_ref().is_some_and(|s| !s.is_empty()),
                    result.has_document_context,
                );
                if let Some(cap) = result.capability.as_deref().filter(|c| !c.is_empty()) {
                    tracing::info!(target: "smartloop.agents.planner", "[classify] capability → {}", cap);
                }
            }
        }

        result
    }

    /// Pick a capability from planning signals in priority order.
    ///
    /// Priority:
    ///   1. `vision`    — attachments imply image/document OCR work.
    ///   2. `coding`    — programming work, which a coding model writes
    ///      better than a skill or document model does.
    ///   3. `reasoning` — a matched skill defines the procedure to run.
    ///   4. `dp`        — indexed document context: data-processing model.
    ///   5. `general`   — default for open-ended chat / general answers.
    ///
    /// Returns the first capability whose signal fires *and* is present
    /// in `capabilities`, or `None` when nothing matches.
    pub fn match_capability(
        capabilities: &[String],
        has_attachments: bool,
        has_coding_intent: bool,
        has_matched_skills: bool,
        has_document_context: bool,
    ) -> Option<String> {
        let priority = [
            (ModelCapability::Vision, has_attachments),
            (ModelCapability::Coding, has_coding_intent),
            (ModelCapability::Reasoning, has_matched_skills),
            (ModelCapability::Dp, has_document_context),
            (ModelCapability::General, true),
        ];
        for (cap, signal) in priority {
            if signal && capabilities.iter().any(|c| c == cap.value()) {
                return Some(cap.value().to_string());
            }
        }
        None
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
        let mut result = ClassificationResult::default();
        let msg_lower = user_message.to_lowercase();

        // MCP tool keyword match
        for srv in mcp_tools {
            for tool in &srv.tools {
                let t_name = tool.get("name").and_then(Value::as_str).unwrap_or("");
                let words: Vec<String> = tool_name_words(t_name)
                    .into_iter()
                    .filter(|w| w.chars().count() > 2)
                    .map(|w| w.to_lowercase())
                    .collect();
                if !words.is_empty() && words.iter().all(|w| msg_lower.contains(w.as_str())) {
                    result.matched_tools.push(MatchedTool {
                        server_id: srv.server_id.clone(),
                        server_name: srv.server_name.clone(),
                        tool_name: t_name.to_string(),
                    });
                    break; // one match per server
                }
            }
        }

        // URLs
        result.detected_urls = find_urls(user_message);

        // File / directory paths
        if let Some(cwd) = cwd.filter(|c| !c.is_empty()) {
            result.detected_paths = extract_paths(user_message, cwd);
        }

        // Bash blocks
        for caps in BASH_BLOCK_RE.captures_iter(user_message) {
            result.bash_commands.push(caps[1].trim().to_string());
        }

        // Write intent
        result.has_write_intent = WRITE_KEYWORDS.iter().any(|kw| msg_lower.contains(kw));

        // Grep intent
        result.has_grep_intent = GREP_KEYWORDS.iter().any(|kw| msg_lower.contains(kw));

        // Coding intent: the message says so outright, or the turn works
        // in a code project and the message asks for a change or about the
        // code, as [`IN_PROJECT_RE`] reads it.  A question about a document
        // that happens to sit in a code project stays general.
        let in_project = cwd.filter(|c| !c.is_empty()).is_some_and(is_code_project);
        result.has_coding_intent =
            CODING_RE.is_match(user_message) || (in_project && IN_PROJECT_RE.is_match(user_message));

        // Document context from vector search / uploads / attachments
        result.has_document_context =
            has_attachments || document_context.is_some_and(|d| !d.trim().is_empty());

        result
    }

    /// Derive intent from detected signals.  Returns `None` when ambiguous.
    pub fn resolve_intent(&self, result: &mut ClassificationResult) -> Option<IntentType> {
        // Build set of active signals
        let mut active: BTreeSet<Signal> = BTreeSet::new();
        if !result.matched_tools.is_empty() {
            active.insert(Signal::Tools);
        }
        if !result.detected_paths.is_empty() {
            active.insert(Signal::Paths);
        }
        if !result.bash_commands.is_empty() {
            active.insert(Signal::Bash);
        }
        if result.has_write_intent {
            active.insert(Signal::Write);
        }
        if result.has_grep_intent {
            active.insert(Signal::Grep);
        }
        if !result.detected_urls.is_empty() {
            active.insert(Signal::Url);
        }

        if active.is_empty() {
            // Document context present → treat as EXPLORE so the model
            // answers from the attached/uploaded content instead of
            // falling through to the LLM classifier (which often picks
            // CONVERSATION and ignores the documents).
            if result.has_document_context {
                result.reasoning = "document context available — exploring".to_string();
                return Some(IntentType::Explore);
            }
            return None;
        }

        // Action signals (exclude modifiers like write/grep)
        let action_signals = [Signal::Tools, Signal::Paths, Signal::Bash, Signal::Url]
            .iter()
            .filter(|s| active.contains(s))
            .count();

        // Multiple action signal types → multi-step
        if action_signals > 1 {
            result.reasoning = "multiple action signals detected".to_string();
            return Some(IntentType::MultiStep);
        }

        // URL + write modifier → fetch then write
        if active.contains(&Signal::Url) && active.contains(&Signal::Write) {
            result.reasoning = "URL + write intent".to_string();
            return Some(IntentType::MultiStep);
        }

        // Single tool + write/grep modifier → still multi-step
        if active.contains(&Signal::Tools)
            && (active.contains(&Signal::Write) || active.contains(&Signal::Grep))
        {
            // MCP tool match with write intent is still just the tool call
            // (the MCP tool IS the action, e.g. "create issue" matches write keyword)
        } else if active.contains(&Signal::Paths) && active.contains(&Signal::Write) {
            result.reasoning = "file paths + write intent".to_string();
            return Some(IntentType::MultiStep);
        }

        // Single signal type
        if active.contains(&Signal::Url) {
            result.reasoning = format!("URL detected: {}", result.detected_urls[0]);
            return Some(IntentType::ToolCall);
        }

        if active.contains(&Signal::Tools) {
            result.reasoning = format!("MCP tool keyword match: {}", result.matched_tools[0].tool_name);
            return Some(IntentType::ToolCall);
        }

        if active.contains(&Signal::Bash) {
            result.action_kind = Some("bash".to_string());
            result.reasoning = "bash code block detected".to_string();
            return Some(IntentType::Action);
        }

        if active.contains(&Signal::Paths) {
            // Check if any path is a directory → explore; otherwise read
            if let Some(dir) = result.detected_paths.iter().find(|p| Path::new(p.as_str()).is_dir()) {
                result.reasoning = format!("directory path detected: {dir}");
                return Some(IntentType::Explore);
            }
            result.action_kind = Some("read".to_string());
            result.reasoning = format!("file path detected: {}", result.detected_paths[0]);
            return Some(IntentType::Action);
        }

        if active.contains(&Signal::Grep) {
            result.action_kind = Some("grep".to_string());
            result.reasoning = "grep/search keywords detected".to_string();
            return Some(IntentType::Action);
        }

        if active.contains(&Signal::Write) {
            // Write intent alone (no paths, no URLs) — needs content generation + write
            result.reasoning = "write intent without target".to_string();
            return Some(IntentType::MultiStep);
        }

        None // shouldn't reach here, but safe fallback
    }

    /// Lightweight LLM call to classify ambiguous messages.
    fn llm_classify(
        &self,
        user_message: &str,
        mcp_tools: &[MCPToolContext],
        skills: Option<&[String]>,
        history: Option<&str>,
        mut result: ClassificationResult,
    ) -> ClassificationResult {
        // Build compact skill/server list
        let mut skill_lines = String::new();
        if let Some(skills) = skills.filter(|s| !s.is_empty()) {
            let lines: Vec<String> = skills
                .iter()
                .take(5)
                .map(|s| format!("  - {}", truncate_chars(s, 200)))
                .collect();
            skill_lines = format!("Skills:\n{}\n", lines.join("\n"));
        }

        let mut server_lines = String::new();
        if !mcp_tools.is_empty() {
            let lines: Vec<String> = mcp_tools
                .iter()
                .map(|srv| {
                    let tool_names: Vec<&str> = srv
                        .tools
                        .iter()
                        .take(5)
                        .map(|t| t.get("name").and_then(Value::as_str).unwrap_or(""))
                        .collect();
                    format!("  {}: {}", srv.server_name, tool_names.join(", "))
                })
                .collect();
            server_lines = format!("MCP servers:\n{}\n", lines.join("\n"));
        }

        let mut history_line = String::new();
        if let Some(history) = history.filter(|h| !h.is_empty()) {
            history_line = format!("Recent context: {}\n", truncate_chars(history, 600));
        }

        let mut doc_line = "";
        if result.has_document_context {
            doc_line = concat!(
                "Document context is available from uploaded documents.\n",
                "Use EXPLORE for any question about the user's documents or their contents.\n",
                "Use CONVERSATION if the question is general knowledge (e.g. recommendations, facts, opinions).\n",
                "Do NOT use ACTION unless the message contains an explicit local file path, ",
                "shell command, or grep target — a plain question about content is EXPLORE, not ACTION.\n",
            );
        }

        let mut prompt = format!(
            concat!(
                "<system>Classify the user's intent. Reply with exactly one word:\n",
                "TOOL_CALL — needs an external MCP tool (API call, fetch, etc.)\n",
                "ACTION — needs a local action on an EXPLICIT target the user named: read a file ",
                "at a given path, run a given bash command, grep for a given pattern, or write a file. ",
                "A plain natural-language question with no path/command is NOT an ACTION.\n",
                "EXPLORE — needs to browse/explore local files or search uploaded documents\n",
                "MULTI_STEP — needs multiple steps\n",
                "CONVERSATION — can be answered from general knowledge or general conversation ",
                "(e.g. recommendations, trivia, opinions, greetings)\n\n",
                "{doc_line}",
                "{skill_lines}",
                "{server_lines}",
                "{history_line}",
                "</system>",
                "<user>{user_message}</user>",
            ),
            doc_line = doc_line,
            skill_lines = skill_lines,
            server_lines = server_lines,
            history_line = history_line,
            user_message = user_message,
        );

        // Ensure the prompt fits.  (Python wrapped this and the call in a
        // `try/except` that defaulted to CONVERSATION; the Rust `BaseLLM`
        // surface is infallible — `invoke` reports errors in its reply text —
        // so there is no failure to catch here.)
        let max_tokens = self.llm.get_max_tokens();
        let token_budget = (max_tokens as f64 * 0.8) as usize;
        let prompt_tokens = self.llm.get_num_tokens(&prompt);
        if prompt_tokens > token_budget {
            // Drop skills and history to fit
            prompt = format!(
                concat!(
                    "<system>Classify the user's intent. Reply with exactly one word:\n",
                    "TOOL_CALL, ACTION, EXPLORE, MULTI_STEP, or CONVERSATION</system>",
                    "<user>{}</user>",
                ),
                user_message
            );
        }

        let response = self.llm.invoke(&prompt, None, None).trim().to_uppercase();
        // Extract the intent word from the response
        for (key, intent) in INTENT_MAP {
            if response.contains(key) {
                result.intent = *intent;
                result.reasoning = format!("LLM classified as {key}");
                return result;
            }
        }

        // Could not parse → default to conversation
        result.intent = IntentType::Conversation;
        result.reasoning = format!(
            "LLM response unparseable ({}), defaulting to conversation",
            truncate_chars(&response, 50)
        );
        result
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
        let mut mcp_tools: Vec<MCPToolContext> = Vec::new();
        for srv in available_servers {
            let id = srv.get("id").and_then(Value::as_str);
            if matches!(id, Some(SYSTEM_TOOLS_SERVER_ID) | Some(FETCH_SERVER_ID)) {
                continue;
            }
            let tools = srv
                .get("tools")
                .and_then(Value::as_array)
                .map(|tools| tools.iter().filter_map(|t| t.as_object().cloned()).collect())
                .unwrap_or_default();
            mcp_tools.push(MCPToolContext {
                server_id: id.unwrap_or("").to_string(),
                server_name: srv.get("name").and_then(Value::as_str).unwrap_or("Unknown").to_string(),
                tools,
            });
        }

        let detected_paths = match cwd.filter(|c| !c.is_empty()) {
            Some(cwd) => extract_paths(user_message, cwd),
            None => Vec::new(),
        };
        let detected_urls = find_urls(user_message);

        PlanContext {
            user_message: user_message.to_string(),
            skills,
            mcp_tools,
            document_context: document_context.map(str::to_string),
            cwd: cwd.map(str::to_string),
            detected_paths,
            detected_urls,
            skill_maps: Vec::new(),
        }
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
        match classification.intent {
            IntentType::Conversation => {
                // Web search is no longer keyword-routed here: when enabled on the
                // project it runs as a parallel source in the collect step (see
                // Explorer.collect_parallel), so latest-info context is gathered for
                // every query without a brittle keyword gate.
                if let Some(cap) = classification.capability.as_deref().filter(|c| !c.is_empty()) {
                    return ExecutionPlan {
                        steps: vec![PlanStep {
                            step: 1,
                            step_type: StepType::Conversational,
                            tool_name: Some("conversation".to_string()),
                            server_id: Some(SYSTEM_TOOLS_SERVER_ID.to_string()),
                            server_name: Some(SYSTEM_TOOLS_SERVER_NAME.to_string()),
                            instruction: context.user_message.clone(),
                            model_tier: ModelTier::Large,
                            capability: Some(cap.to_string()),
                            ..Default::default()
                        }],
                        reasoning: format!("conversation → capability '{cap}'"),
                    };
                }
                ExecutionPlan { steps: Vec::new(), reasoning: "conversation (no capability)".to_string() }
            }
            IntentType::ToolCall => self.plan_tool_call(classification, context),
            IntentType::Action => self.plan_action(classification, context),
            IntentType::Explore => self.plan_explore(classification, context),
            IntentType::MultiStep => self.plan_multi_step(classification, context, llm),
        }
    }

    /// Single MCP tool step.
    fn plan_tool_call(&self, classification: &ClassificationResult, context: &PlanContext) -> ExecutionPlan {
        if let Some(tool) = classification.matched_tools.first() {
            return ExecutionPlan {
                steps: vec![PlanStep {
                    step: 1,
                    step_type: StepType::Tool,
                    tool_name: Some(tool.tool_name.clone()),
                    server_id: Some(tool.server_id.clone()),
                    server_name: Some(tool.server_name.clone()),
                    instruction: context.user_message.clone(),
                    ..Default::default()
                }],
                reasoning: format!("MCP tool: {}", tool.tool_name),
            };
        }

        // URL fetch — run the FetchAgent as a dedicated step
        if let Some(url) = classification.detected_urls.first() {
            let mut fetch_step = self.make_fetch_step(url, context);
            fetch_step.step = 1;
            return ExecutionPlan { steps: vec![fetch_step], reasoning: format!("fetch URL: {url}") };
        }

        ExecutionPlan { steps: Vec::new(), reasoning: "tool_call but no tool matched".to_string() }
    }

    /// Single built-in system tool step.
    ///
    /// System tools run as ordinary TOOL steps against the local `smartloop`
    /// MCP server (like fetch).  Where the classifier already extracted a
    /// concrete value (bash command, file path) it is carried as
    /// `default_args` so resolution stays deterministic; otherwise the
    /// argument extractor fills the args from the user message.
    fn plan_action(&self, classification: &ClassificationResult, context: &PlanContext) -> ExecutionPlan {
        let action_kind = classification.action_kind.as_deref().filter(|k| !k.is_empty()).unwrap_or("read");

        if action_kind == "bash" {
            if let Some(cmd) = classification.bash_commands.first() {
                return ExecutionPlan {
                    steps: vec![system_tool_step(
                        1,
                        BASH_TOOL,
                        format!("Run: {cmd}"),
                        Some(single_arg("command", cmd)),
                    )],
                    reasoning: "bash command".to_string(),
                };
            }
        }

        if action_kind == "read" {
            if let Some(path) = classification.detected_paths.first() {
                return ExecutionPlan {
                    steps: vec![system_tool_step(
                        1,
                        FILE_READ_TOOL,
                        format!("Read the file {path}"),
                        Some(single_arg("path", path)),
                    )],
                    reasoning: format!("read file: {path}"),
                };
            }
        }

        if action_kind == "grep" {
            return ExecutionPlan {
                steps: vec![system_tool_step(
                    1,
                    GREP_TOOL,
                    "Search files for the requested pattern.".to_string(),
                    None,
                )],
                reasoning: "grep search".to_string(),
            };
        }

        // Generic system-tool fallback
        let tool_name = match action_kind {
            "read" => FILE_READ_TOOL,
            "write" => FILE_WRITE_TOOL,
            "grep" => GREP_TOOL,
            "bash" => BASH_TOOL,
            _ => FILE_READ_TOOL,
        };
        ExecutionPlan {
            steps: vec![system_tool_step(1, tool_name, context.user_message.clone(), None)],
            reasoning: format!("action: {action_kind}"),
        }
    }

    /// Single explore step — execution handles fetch → vector → files internally.
    fn plan_explore(&self, classification: &ClassificationResult, context: &PlanContext) -> ExecutionPlan {
        let target = match classification.detected_paths.first() {
            Some(p) => p.clone(),
            None => context.cwd.clone().filter(|c| !c.is_empty()).unwrap_or_else(|| ".".to_string()),
        };
        ExecutionPlan {
            steps: vec![explore_step(1, &target)],
            reasoning: format!("explore: {target}"),
        }
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
        // Try LLM-based planning
        let plan = self.llm_plan(classification, context, llm);
        if !plan.steps.is_empty() {
            return plan;
        }

        // Fallback: build steps from signals
        self.build_steps_from_signals(classification, context)
    }

    /// Call LLM to produce a step ordering for multi-step requests.
    fn llm_plan(
        &self,
        classification: &ClassificationResult,
        context: &PlanContext,
        llm: Option<&dyn BaseLLM>,
    ) -> ExecutionPlan {
        // Build a focused prompt — classification already tells us what signals exist
        let mut signals_hint: Vec<String> = Vec::new();
        for t in &classification.matched_tools {
            signals_hint.push(format!("MCP tool available: {} (server: {})", t.tool_name, t.server_name));
        }
        if !classification.detected_urls.is_empty() {
            let urls: Vec<&str> = classification.detected_urls.iter().take(3).map(String::as_str).collect();
            signals_hint.push(format!("URLs: {}", urls.join(", ")));
        }
        if !classification.detected_paths.is_empty() {
            let paths: Vec<&str> = classification.detected_paths.iter().take(3).map(String::as_str).collect();
            signals_hint.push(format!("Paths: {}", paths.join(", ")));
        }
        if classification.has_write_intent {
            signals_hint.push("Write intent detected".to_string());
        }
        if classification.has_grep_intent {
            signals_hint.push("Grep/search intent detected".to_string());
        }
        if let Some(cmd) = classification.bash_commands.first() {
            signals_hint.push(format!("Bash: {}", truncate_chars(cmd, 100)));
        }
        let signals_block: String =
            signals_hint.iter().map(|s| format!("  - {s}")).collect::<Vec<_>>().join("\n");

        // Skills section
        let mut skills_section = String::new();
        if let Some(skills) = context.skills.as_ref().filter(|s| !s.is_empty()) {
            skills_section = format!(
                "SKILLS (use these to decide which server to call):\n{}\n\n",
                skills.iter().map(|s| format!("  - {s}")).collect::<Vec<_>>().join("\n")
            );
        }

        // Skill maps section (explicit step skeletons from skill_map)
        let mut skill_maps_section = String::new();
        if !context.skill_maps.is_empty() {
            let mut sm_lines: Vec<String> = Vec::new();
            for sm in &context.skill_maps {
                sm_lines.push(format!(
                    "  Skill: {}",
                    sm.get("skill_name").map(py_str).unwrap_or_default()
                ));
                let steps = sm.get("steps").and_then(Value::as_array);
                for step_entry in steps.into_iter().flatten() {
                    let get = |key: &str| step_entry.get(key).filter(|v| py_truthy(v));
                    let mut parts = vec![format!(
                        "    step {}",
                        step_entry.get("step").map(py_str).unwrap_or_else(|| "?".to_string())
                    )];
                    if let Some(v) = get("tool") {
                        parts.push(format!("tool={}", py_str(v)));
                    }
                    if let Some(v) = get("server") {
                        parts.push(format!("server={}", py_str(v)));
                    }
                    if let Some(v) = get("condition") {
                        parts.push(format!("condition=\"{}\"", py_str(v)));
                    }
                    if let Some(v) = get("instruction") {
                        parts.push(format!("— {}", py_str(v)));
                    }
                    sm_lines.push(parts.join(" "));
                }
            }
            skill_maps_section = format!(
                "SKILL STEPS (use these as step skeleton — preserve tool, server, and condition):\n{}\n\n",
                sm_lines.join("\n")
            );
        }

        // Servers section — include actual tool names so the LLM uses exact
        // names instead of guessing / inventing non-existent ones.
        let mut server_lines: Vec<String> = Vec::new();
        for srv in &context.mcp_tools {
            let tool_names: Vec<&str> = srv
                .tools
                .iter()
                .take(10)
                .filter_map(|t| t.get("name").filter(|v| py_truthy(v)).and_then(Value::as_str))
                .collect();
            let tools_list = if tool_names.is_empty() { "(no tools)".to_string() } else { tool_names.join(", ") };
            server_lines.push(format!(
                "  \"{}\" (server_id=\"{}\"): tools=[{}]",
                srv.server_name, srv.server_id, tools_list
            ));
        }
        let servers_section = if server_lines.is_empty() {
            String::new()
        } else {
            format!("MCP SERVERS (use ONLY these exact tool names):\n{}\n\n", server_lines.join("\n"))
        };

        let mut prompt = format!(
            concat!(
                "<system>",
                "You are a task planner. Return a JSON array of steps.\n\n",
                "{skills_section}",
                "{skill_maps_section}",
                "{servers_section}",
                "SYSTEM TOOLS (server_id='__system_tools__'):\n",
                "  explore — walk the local file tree\n",
                "  file-read — read a local file\n",
                "  file-write — write a local file\n",
                "  grep — search text in local files\n",
                "  bash — run a shell command\n\n",
                "DETECTED SIGNALS:\n",
                "{signals_block}",
                "\n\n",
                "STEP FORMAT:\n",
                "{{\"step\":<n>,\"step_type\":\"tool|action|explore\",\"tool_name\":\"<name>\",",
                "\"server_id\":\"<id>\",\"server_name\":\"<name>\",\"instruction\":\"<what to do>\",",
                "\"action_kind\":\"read|write|grep|bash (action steps only)\",",
                "\"condition\":\"result_contains:X or llm:question (optional)\",",
                "\"skill_name\":\"originating skill name (optional)\"}}\n\n",
                "RULES:\n",
                "  1. tool_name MUST be an exact name from the MCP SERVERS or SYSTEM TOOLS lists above. ",
                "Never invent or guess tool names.\n",
                "  2. If a skill matches the request → step_type='tool', use the server from skills.\n",
                "  3. For local file operations → step_type='action' with system tools.\n",
                "  4. Max 5 steps.\n",
                "  5. If SKILL STEPS are provided, preserve their tool, server, and condition values.\n",
                "  6. If skill instructions contain conditional language (e.g. 'if there are differences', ",
                "'only when', 'in case of'), emit a 'condition' field on the conditional step:\n",
                "     - Use 'result_contains:<keyword>' for simple text checks on previous results.\n",
                "     - Use 'llm:<yes/no question>' for complex conditions requiring reasoning.\n",
                "  7. Each step can use results from previous steps. Write instructions that reference ",
                "prior output (e.g. 'Using the invoice summary from step 1, ...').\n",
                "  8. NEVER use file-read to read uploaded documents (PDFs, DOCX, etc.). ",
                "They are already vectorized and available via the document context. ",
                "Use step_type='explore' if you need to search/analyze documents.\n",
                "</system>",
                "<user>",
                "{user_message}\n\n",
                "Reply with ONLY the JSON array.",
                "</user>",
            ),
            skills_section = skills_section,
            skill_maps_section = skill_maps_section,
            servers_section = servers_section,
            signals_block = signals_block,
            user_message = context.user_message,
        );

        let plan_llm: &dyn BaseLLM = match llm {
            Some(l) => l,
            None => self.llm.as_ref(),
        };

        // Budget check
        let max_tokens = plan_llm.get_max_tokens();
        let token_budget = (max_tokens as f64 * 0.8) as usize;
        let prompt_tokens = plan_llm.get_num_tokens(&prompt);
        if prompt_tokens > token_budget {
            // Trim skills: rebuild the prompt without skills (or skill maps
            // and rules).
            if context.skills.as_ref().is_some_and(|s| !s.is_empty()) {
                prompt = format!(
                    concat!(
                        "<system>You are a task planner. Return a JSON array of steps.\n\n",
                        "{servers_section}",
                        "DETECTED SIGNALS:\n",
                        "{signals_block}\n\n",
                        "STEP FORMAT:\n",
                        "{{\"step\":<n>,\"step_type\":\"tool|action|explore\",\"tool_name\":\"<name>\",",
                        "\"server_id\":\"<id>\",\"server_name\":\"<name>\",\"instruction\":\"<what to do>\",",
                        "\"action_kind\":\"read|write|grep|bash (action steps only)\"}}\n\n",
                        "</system>",
                        "<user>{user_message}\n\nReply with ONLY the JSON array.</user>",
                    ),
                    servers_section = servers_section,
                    signals_block = signals_block,
                    user_message = context.user_message,
                );
            }
        }

        // (Python caught exceptions from the LLM here and returned an empty
        // plan with the error as reasoning; `BaseLLM::invoke` is infallible
        // and reports failures in its reply, which then fails to parse.)
        let response = plan_llm.invoke(&prompt, None, None).trim().to_string();
        tracing::info!(
            target: "smartloop.agents.planner",
            "[plan] Raw LLM plan ({} chars): {}",
            response.chars().count(),
            truncate_chars(&response, 500)
        );
        self.parse_plan_response(&response, &context.user_message, Some(classification))
    }

    /// Build plan steps directly from classification signals (no LLM).
    fn build_steps_from_signals(
        &self,
        classification: &ClassificationResult,
        context: &PlanContext,
    ) -> ExecutionPlan {
        let mut steps: Vec<PlanStep> = Vec::new();
        let mut step_num: i64 = 1;

        // Fetch URLs first (context gathering)
        for url in &classification.detected_urls {
            let mut fetch_step = self.make_fetch_step(url, context);
            fetch_step.step = step_num;
            steps.push(fetch_step);
            step_num += 1;
        }

        // MCP tool calls
        for tool in &classification.matched_tools {
            steps.push(PlanStep {
                step: step_num,
                step_type: StepType::Tool,
                tool_name: Some(tool.tool_name.clone()),
                server_id: Some(tool.server_id.clone()),
                server_name: Some(tool.server_name.clone()),
                instruction: context.user_message.clone(),
                ..Default::default()
            });
            step_num += 1;
        }

        // File paths → read or explore
        for path in &classification.detected_paths {
            if Path::new(path).is_dir() {
                steps.push(explore_step(step_num, path));
            } else {
                steps.push(system_tool_step(
                    step_num,
                    FILE_READ_TOOL,
                    format!("Read the file {path}"),
                    Some(single_arg("path", path)),
                ));
            }
            step_num += 1;
        }

        // Bash blocks
        for cmd in &classification.bash_commands {
            steps.push(system_tool_step(
                step_num,
                BASH_TOOL,
                format!("Run: {cmd}"),
                Some(single_arg("command", cmd)),
            ));
            step_num += 1;
        }

        // Grep
        if classification.has_grep_intent {
            steps.push(system_tool_step(
                step_num,
                GREP_TOOL,
                "Search files for the requested pattern.".to_string(),
                None,
            ));
            step_num += 1;
        }

        // Write intent → conversational generation + file write
        if classification.has_write_intent {
            steps.push(PlanStep {
                step: step_num,
                step_type: StepType::Conversational,
                tool_name: Some("conversation".to_string()),
                server_id: Some(SYSTEM_TOOLS_SERVER_ID.to_string()),
                server_name: Some(SYSTEM_TOOLS_SERVER_NAME.to_string()),
                instruction: "Generate the requested content based on the user request and any fetched context."
                    .to_string(),
                ..Default::default()
            });
            step_num += 1;
            steps.push(system_tool_step(
                step_num,
                FILE_WRITE_TOOL,
                "Write the generated content to a file.".to_string(),
                None,
            ));
        }

        let reasoning = format!("signal-based: {}", classification.reasoning);
        steps.truncate(5);
        ExecutionPlan { steps, reasoning }
    }

    // ------------------------------------------------------------------ //
    // Helpers                                                            //
    // ------------------------------------------------------------------ //

    /// Build a FETCH step that the orchestrator runs via the FetchAgent.
    ///
    /// Fetching is no longer an MCP tool call — the FetchAgent performs the
    /// HTTP GET and converts HTML/PDF to markdown in-process.  The URL is
    /// carried in `default_args` so no argument resolution is needed.
    fn make_fetch_step(&self, url: &str, _context: &PlanContext) -> PlanStep {
        PlanStep {
            step: 0,
            step_type: StepType::Fetch,
            tool_name: Some("fetch".to_string()),
            server_id: Some(String::new()),
            server_name: Some(String::new()),
            instruction: format!("Fetch {url} and extract its main content."),
            default_args: Some(single_arg("url", url)),
            ..Default::default()
        }
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
        // Strip markdown fences
        let mut response = response.to_string();
        if response.starts_with("```") {
            let lines: Vec<&str> =
                response.split('\n').filter(|ln| !ln.trim().starts_with("```")).collect();
            response = lines.join("\n").trim().to_string();
        }

        let steps_data: Value = match serde_json::from_str(&response) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(target: "smartloop.agents.planner", "[plan] JSON parse error: {}", e);
                return ExecutionPlan { steps: Vec::new(), reasoning: format!("Parse error: {e}") };
            }
        };

        let Value::Array(steps_data) = steps_data else {
            tracing::warn!(target: "smartloop.agents.planner", "[plan] Plan response is not a JSON array");
            return ExecutionPlan { steps: Vec::new(), reasoning: "Parse error: not a list".to_string() };
        };

        if steps_data.is_empty() {
            tracing::info!(
                target: "smartloop.agents.planner",
                "[plan] Planner returned empty plan — treating as conversation"
            );
            return ExecutionPlan { steps: Vec::new(), reasoning: "conversation".to_string() };
        }

        let mut steps: Vec<PlanStep> = Vec::new();
        for (i, s) in steps_data.iter().take(5).enumerate() {
            // hard cap at 5 steps
            let Some(s) = s.as_object() else {
                continue;
            };

            let raw_step_type = s.get("step_type").and_then(Value::as_str);
            // Backward compatibility: map old step_type values. System tools
            // (read/write/grep/bash) now run as ordinary TOOL steps against
            // the local smartloop MCP server, like fetch.
            let mapped = raw_step_type.and_then(|raw| match raw {
                "tool_call" | "mcp" => Some(StepType::Tool),
                "web_search" | "search" => Some(StepType::WebSearch),
                "conversation" => Some(StepType::Conversational),
                "document" => Some(StepType::Explore),
                "read" | "write" | "grep" | "bash" => Some(StepType::Tool),
                _ => None,
            });

            let mut step_type = match mapped.or_else(|| raw_step_type.and_then(StepType::from_value)) {
                Some(t) => t,
                None => Self::infer_step_type(
                    s.get("tool_name").and_then(Value::as_str).unwrap_or(""),
                    s.get("server_id").and_then(Value::as_str).unwrap_or(SYSTEM_TOOLS_SERVER_ID),
                ),
            };

            // Correct misclassified steps: if the LLM said "action" or
            // "conversational" but the server_id is NOT __system_tools__,
            // it's an external MCP tool call — override to TOOL.
            let mut t_server_id: Option<String> = match s.get("server_id") {
                None => Some(SYSTEM_TOOLS_SERVER_ID.to_string()),
                Some(v) => opt_string(v),
            };
            if matches!(step_type, StepType::Action | StepType::Conversational)
                && t_server_id.as_deref() != Some(SYSTEM_TOOLS_SERVER_ID)
                && s.get("tool_name").is_some_and(py_truthy)
            {
                step_type = StepType::Tool;
            }

            // The built-in file/grep/bash tools resolve through the smartloop
            // MCP server, so they run as TOOL steps pinned to that server —
            // even if the model mislabelled the step as "action". Only
            // `explore` is an internal pseudo-step.
            let t_name = s.get("tool_name").and_then(Value::as_str).unwrap_or("");
            if t_name == EXPLORE_TOOL {
                step_type = StepType::Explore;
                t_server_id = Some(SYSTEM_TOOLS_SERVER_ID.to_string());
            } else if [FILE_READ_TOOL, FILE_WRITE_TOOL, GREP_TOOL, BASH_TOOL].contains(&t_name) {
                step_type = StepType::Tool;
                t_server_id = Some(SYSTEM_TOOLS_SERVER_ID.to_string());
            }

            // Infer action_kind for the remaining ACTION steps (summarize).
            let action_kind = s.get("action_kind").and_then(opt_string);

            let server_name = match s.get("server_name") {
                None => Some(SYSTEM_TOOLS_SERVER_NAME.to_string()),
                Some(v) => opt_string(v),
            };

            steps.push(PlanStep {
                step: s.get("step").and_then(py_int).unwrap_or(i as i64 + 1),
                step_type,
                tool_name: s.get("tool_name").and_then(opt_string),
                // `t_server_id` already falls back to the raw `server_id`.
                server_id: t_server_id,
                server_name,
                instruction: s
                    .get("instruction")
                    .filter(|v| py_truthy(v))
                    .and_then(opt_string)
                    .unwrap_or_else(|| user_message.to_string()),
                action_kind,
                condition: s.get("condition").filter(|v| py_truthy(v)).and_then(opt_string),
                skill_name: s.get("skill_name").filter(|v| py_truthy(v)).and_then(opt_string),
                ..Default::default()
            });
        }

        if let Some(classification) = classification {
            let mut path_idx = 0usize;
            let mut cmd_idx = 0usize;
            for step in steps.iter_mut() {
                if step.default_args.as_ref().is_some_and(|a| !a.is_empty()) {
                    continue;
                }
                let tool_name = step.tool_name.as_deref();
                if tool_name == Some(FILE_READ_TOOL) && !classification.detected_paths.is_empty() {
                    if path_idx < classification.detected_paths.len() {
                        step.default_args =
                            Some(single_arg("path", &classification.detected_paths[path_idx]));
                        path_idx += 1;
                    }
                } else if tool_name == Some(BASH_TOOL) && !classification.bash_commands.is_empty() {
                    if cmd_idx < classification.bash_commands.len() {
                        step.default_args =
                            Some(single_arg("command", &classification.bash_commands[cmd_idx]));
                        cmd_idx += 1;
                    }
                }
            }
        }

        let summary: Vec<String> = steps
            .iter()
            .map(|s| format!("{}:{}:{}", s.step, s.step_type.value(), s.tool_name.as_deref().unwrap_or("None")))
            .collect();
        let summary_refs: Vec<&str> = summary.iter().map(String::as_str).collect();
        tracing::info!(
            target: "smartloop.agents.planner",
            "[plan] Created {}-step plan: {}",
            steps.len(),
            py_str_list(&summary_refs)
        );
        ExecutionPlan { steps, reasoning: String::new() }
    }

    /// Derive [`StepType`] from `tool_name` and `server_id`.
    ///
    /// `explore` is the only internal pseudo-step; the other built-in
    /// tools (file/grep/bash) run as ordinary TOOL steps against the local
    /// smartloop MCP server.
    pub fn infer_step_type(tool_name: &str, _server_id: &str) -> StepType {
        if tool_name == EXPLORE_TOOL {
            return StepType::Explore;
        }
        StepType::Tool
    }
}

// ---------------------------------------------------------------------- //
// Module-level helpers                                                    //
// ---------------------------------------------------------------------- //

/// Return real filesystem paths found in `text` (expanded, made absolute
/// against `cwd`, normalised, de-duplicated, existing only).
pub fn extract_paths(text: &str, cwd: &str) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for caps in PATH_RE.captures_iter(text) {
        let raw = caps[1].trim_end_matches(['.', ',', ';', ':', '!', '?']);
        // `expand_home` leaves anything not starting with `~` untouched.
        let mut expanded = if raw.starts_with('~') { expand_home(raw) } else { raw.to_string() };
        if !expanded.starts_with('/') {
            expanded = posix_join(cwd, &expanded);
        }
        let expanded = normpath(&expanded);
        if Path::new(&expanded).exists() && !found.contains(&expanded) {
            found.push(expanded);
        }
    }
    found
}

/// `_URL_RE.findall(text)`.
fn find_urls(text: &str) -> Vec<String> {
    URL_RE.find_iter(text).map(|m| m.as_str().to_string()).collect()
}

/// Split a tool name into its words: on `_`/`-`, then on camelCase humps
/// (a space is inserted before every ASCII capital, then whitespace-split).
fn tool_name_words(t_name: &str) -> Vec<String> {
    let mut words = Vec::new();
    for part in t_name.split(['_', '-']) {
        let mut spaced = String::with_capacity(part.len() * 2);
        for ch in part.chars() {
            if ch.is_ascii_uppercase() {
                spaced.push(' ');
            }
            spaced.push(ch);
        }
        words.extend(spaced.split_whitespace().map(str::to_string));
    }
    words
}

/// A system-tools-server TOOL step.
fn system_tool_step(
    step: i64,
    tool_name: &str,
    instruction: String,
    default_args: Option<BTreeMap<String, String>>,
) -> PlanStep {
    PlanStep {
        step,
        step_type: StepType::Tool,
        tool_name: Some(tool_name.to_string()),
        server_id: Some(SYSTEM_TOOLS_SERVER_ID.to_string()),
        server_name: Some(SYSTEM_TOOLS_SERVER_NAME.to_string()),
        instruction,
        default_args,
        ..Default::default()
    }
}

/// An EXPLORE step over `target`.
fn explore_step(step: i64, target: &str) -> PlanStep {
    PlanStep {
        step,
        step_type: StepType::Explore,
        tool_name: Some(EXPLORE_TOOL.to_string()),
        server_id: Some(SYSTEM_TOOLS_SERVER_ID.to_string()),
        server_name: Some(SYSTEM_TOOLS_SERVER_NAME.to_string()),
        instruction: format!("Explore directory {target}"),
        ..Default::default()
    }
}

/// `{key: value}`.
fn single_arg(key: &str, value: &str) -> BTreeMap<String, String> {
    BTreeMap::from([(key.to_string(), value.to_string())])
}

/// First `n` characters of `s` (Python `s[:n]`).
fn truncate_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

/// Python `str(bool)`.
fn py_bool(b: bool) -> &'static str {
    if b { "True" } else { "False" }
}

/// Python `repr(list_of_str)` for log lines.
fn py_str_list(items: &[&str]) -> String {
    let inner: Vec<String> = items.iter().map(|s| format!("'{s}'")).collect();
    format!("[{}]", inner.join(", "))
}

/// Python `str(value)` for a JSON value.
fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        Value::Bool(b) => py_bool(*b).to_string(),
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

/// Python truthiness of a JSON value.
fn py_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// An optional string field: `null` → `None`, strings as-is, other scalars
/// in their Python `str()` form.
fn opt_string(v: &Value) -> Option<String> {
    match v {
        Value::Null => None,
        other => Some(py_str(other)),
    }
}

/// Pydantic's lax `int` coercion: integers, integral floats and numeric
/// strings.
fn py_int(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n
            .as_i64()
            .or_else(|| n.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64)),
        Value::String(s) => s.trim().parse().ok(),
        Value::Bool(b) => Some(*b as i64),
        _ => None,
    }
}

/// `os.path.join(a, b)` for POSIX paths.
fn posix_join(a: &str, b: &str) -> String {
    if b.starts_with('/') || a.is_empty() {
        b.to_string()
    } else if a.ends_with('/') {
        format!("{a}{b}")
    } else {
        format!("{a}/{b}")
    }
}

/// `posixpath.normpath`: collapse redundant separators and `.`/`..`
/// components lexically.
fn normpath(path: &str) -> String {
    if path.is_empty() {
        return ".".to_string();
    }
    // POSIX allows one or two initial slashes, but treats three or more
    // as a single slash.
    let initial_slashes = if path.starts_with('/') {
        if path.starts_with("//") && !path.starts_with("///") { 2 } else { 1 }
    } else {
        0
    };
    let mut comps: Vec<&str> = Vec::new();
    for comp in path.split('/') {
        if comp.is_empty() || comp == "." {
            continue;
        }
        if comp != ".."
            || (initial_slashes == 0 && comps.is_empty())
            || comps.last() == Some(&"..")
        {
            comps.push(comp);
        } else if !comps.is_empty() {
            comps.pop();
        }
    }
    let joined = comps.join("/");
    let out = format!("{}{}", "/".repeat(initial_slashes), joined);
    if out.is_empty() { ".".to_string() } else { out }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normpath_matches_posixpath() {
        assert_eq!(normpath("/a/b/../c/./d/"), "/a/c/d");
        assert_eq!(normpath("//a//b"), "//a/b");
        assert_eq!(normpath("///a"), "/a");
        assert_eq!(normpath("../a/.."), "..");
        assert_eq!(normpath("a/../.."), "..");
        assert_eq!(normpath("/.."), "/");
        assert_eq!(normpath(""), ".");
        assert_eq!(normpath("./"), ".");
    }

    #[test]
    fn tool_name_words_split_snake_kebab_and_camel() {
        assert_eq!(tool_name_words("create-issue"), vec!["create", "issue"]);
        assert_eq!(tool_name_words("listPullRequests_v2"), vec!["list", "Pull", "Requests", "v2"]);
    }

    #[test]
    fn usable_connection_rules() {
        assert!(mcp_connection_is_usable(&serde_json::json!({})));
        assert!(!mcp_connection_is_usable(&serde_json::json!({"enabled": false})));
        assert!(!mcp_connection_is_usable(&serde_json::json!({"status": "failed"})));
        assert!(mcp_connection_is_usable(&serde_json::json!({"status": "pending"})));
    }
}
