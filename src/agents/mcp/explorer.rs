//! Port of `smartloop/agents/mcp/explorer.py`.
//!
//! Explorer — unified system tool for exploring skills, vector search,
//! conversation history/context, filesystem walk, and URL fetch.
//!
//! The collaborators the Python tests monkeypatch are injectable here:
//! the MCP tool resolver ([`ToolResolver`]), the web search agent
//! ([`WebSearcher`]), the model service searches ([`ExplorerSearchService`])
//! and the call-plan follow-up planner ([`FollowUpPlanner`]) are all public
//! fields holding trait objects.
#![allow(clippy::too_many_arguments)]

use std::collections::{HashMap, HashSet};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::agents::fetch_agent::{FetchAgent, UrlFetcher};
use crate::agents::mcp::argument_extractor::MCPArgumentExtractor;
use crate::agents::mcp::mcp_agent::{
    format_call, is_entitlement_notice, mark_tool_unavailable, missing_required, MCPAgent,
    ResolvedCall,
};
use crate::agents::mcp::mcp_tool_caller::MCPToolCaller;
use crate::agents::mcp::file_ops::FileOperations;
use crate::agents::planner::{mcp_connection_is_usable, MCPToolContext, MatchedTool, PlanStep, StepType};
use crate::agents::web_search_agent::{LlmResolver, WebSearchAgent};
use crate::api::model::CitationSource;
use crate::api::services::model_service::ModelService;
use crate::llms::BaseLLM;
use crate::parsers::DocumentChunk;
use crate::workspace::context::{PoolFuture, ProjectBoundExecutor};
use crate::workspace::workspace::{MCPServer, MCPToolDefinition};

/// Logger name used by the Python module.
pub const LOGGER: &str = "smartloop.agents.mcp.explorer";

// ------------------------------------------------------------------ constants

/// Tool-name fragments that mark a planned tool step as a URL fetch.
pub const FETCH_TOOL_KEYWORDS: &[&str] =
    &["fetch", "browse", "scrape", "web", "http", "navigate", "get_url"];

/// The most of one named file an explore step reads (see
/// [`Explorer::read_file`]).
pub const MAX_READ_FILE_CHARS: usize = 100_000;

/// How much of one fetched record body reaches the prompt.  A Notion page can
/// be very long and three of them share the answer's context with everything
/// else collect gathered.
pub const MAX_RECORD_BODY_CHARS: usize = 4000;

pub const MAX_SKILL_FILE_CHARS: usize = 4_000;
pub const MAX_TREE_ENTRIES: usize = 10;
pub const MAX_HISTORY_TURNS: usize = 8;
pub const MAX_PROJECT_FILES: usize = 30;
pub const MAX_PROJECT_SNIPPET_LINES: usize = 30;
pub const MAX_PROJECT_FILES_CHARS: usize = 8_000;
pub const MAX_FILE_SNIPPET_CHARS: usize = 3_000;

/// Directories skipped by the workspace tree / project-file scans.
pub const SKIP_DIRS: &[&str] = &[
    ".git", "__pycache__", "node_modules", "venv", ".venv", "dist", "build", ".tox",
    ".mypy_cache", ".ruff_cache", ".out",
];

/// Extensions the project-file scan reads.
pub const PROJECT_EXTENSIONS: &[&str] = &[
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".rb", ".php", ".swift", ".kt", ".scala", ".yaml", ".yml", ".toml", ".json", ".md",
    ".txt", ".cfg", ".ini", ".html", ".css", ".sql", ".proto", ".graphql", ".sh",
];

/// Words dropped from the user message before snippet matching.
pub const QUERY_STOP_WORDS: &[&str] = &[
    "the", "and", "for", "that", "this", "with", "from", "what", "how", "does", "project",
    "explore", "analyze", "explain", "summarize", "understand", "codebase", "code",
];

/// Verbs that mark a tool as read-only (safe to auto-run during collect).
pub const READ_ONLY_VERBS: &[&str] = &[
    "get", "list", "search", "read", "fetch", "find", "describe", "view", "show", "query",
    "lookup", "retrieve", "count", "check", "status", "browse", "scrape", "download",
];
/// Verbs that mark a tool as side-effecting (never auto-run in collect).
pub const WRITE_VERBS: &[&str] = &[
    "create", "update", "delete", "send", "post", "write", "remove", "set", "add", "edit",
    "modify", "put", "patch", "insert", "cancel", "pay", "deploy", "merge", "close", "assign",
    "run", "execute", "trigger", "save", "retire", "restore", "share", "unshare", "mark",
    "submit", "resolve", "move", "duplicate", "upload", "prepare", "spawn", "stop",
];

/// What calling a tool does, as far as collect can tell.
///
/// Only [`ToolAccess::Read`] runs unprompted during collect; every other
/// variant, `Unknown` included, is treated as a write and left to a planned
/// step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolAccess {
    Read,
    Write,
    Destructive,
    Unknown,
}

impl ToolAccess {
    /// A tool's access: the server's annotation when it gives one, otherwise
    /// the verbs in its name.
    ///
    /// A server that says a tool is not read-only is believed even when the
    /// name reads like a lookup.  Following the spec, such a tool is
    /// destructive unless the server says `destructiveHint: false`.
    pub fn of(tool: &MCPToolDefinition) -> Self {
        if let Some(hints) = tool.annotations {
            match (hints.read_only_hint, hints.destructive_hint) {
                (Some(true), _) => return Self::Read,
                (Some(false), Some(false)) => return Self::Write,
                (Some(false), _) | (None, Some(true)) => return Self::Destructive,
                (None, _) => {}
            }
        }
        Self::from_name(&tool.name)
    }

    /// A tool's access judged from the verbs in its name alone.
    ///
    /// Every word in the name is considered, not just the first.  Servers
    /// name tools after the product as often as after the verb —
    /// `notion-search`, `API-get-users`, `trelloReadCard` — so a leading-verb
    /// rule would read "notion" as an unknown verb.  Any write word anywhere
    /// wins: `notion-create-pages` is a write however it is spelled.  A name
    /// with no recognised verb is `Unknown`.
    pub fn from_name(tool_name: &str) -> Self {
        // camelCase is a word boundary too: `trelloReadCard` is three words.
        let mut spaced = String::with_capacity(tool_name.len() + 4);
        let mut previous: Option<char> = None;
        for c in tool_name.trim().chars() {
            if let Some(p) = previous {
                if (p.is_ascii_lowercase() || p.is_ascii_digit()) && c.is_ascii_uppercase() {
                    spaced.push(' ');
                }
            }
            spaced.push(c);
            previous = Some(c);
        }
        let lowered = spaced.to_lowercase();
        let words: HashSet<&str> = TOOL_NAME_SEPARATORS.split(&lowered).collect();
        if words.iter().any(|w| WRITE_VERBS.contains(w)) {
            Self::Write
        } else if words.iter().any(|w| READ_ONLY_VERBS.contains(w)) {
            Self::Read
        } else {
            Self::Unknown
        }
    }

    /// The access of `tool_name` on `server`, judged by name alone when the
    /// server's saved tool list does not have it.
    pub fn on(server: &MCPServer, tool_name: &str) -> Self {
        match server.tools.iter().find(|t| t.name == tool_name) {
            Some(tool) => Self::of(tool),
            None => Self::from_name(tool_name),
        }
    }

    /// True only for a read; an unknown tool counts as a write.
    pub fn is_read(self) -> bool {
        self == Self::Read
    }
}

// ------------------------------------------------------------------ seams

/// One conversation turn (`{"role": ..., "content": ...}`).
pub type ConversationTurn = HashMap<String, String>;

/// A parsed skill (`{"name", "description", "content", ...}`).
pub type SkillDict = HashMap<String, String>;

/// `progress(step, status, message)` — reports a step to the client.  A
/// callback that panics never fails the work (see [`Explorer::emit`]).
pub type ProgressCallback = Arc<dyn Fn(&str, &str, &str) + Send + Sync>;

/// `smartloop.mcp.call_plans.follow_up_calls`: the detail calls that fetch the
/// bodies behind the records a listing returned.
pub type FollowUpPlanner =
    Arc<dyn Fn(&MCPServer, &str, &str) -> Vec<(String, Map<String, Value>)> + Send + Sync>;

/// `(documents, citations)` from a vector search.
pub type DocumentGroups = (Vec<Map<String, Value>>, Vec<Map<String, Value>>);

/// Resolves a directive to one tool call over the servers' discovered tools
/// (`MCPAgent.resolve`).
pub trait ToolResolver: Send + Sync {
    fn resolve(
        &self,
        directive: &str,
        mcp_servers: &[MCPServer],
        context: Option<&str>,
        llm: Option<&dyn BaseLLM>,
    ) -> Option<ResolvedCall>;

    /// [`ToolResolver::resolve`] with the model override held as a shared
    /// handle — what the explorer calls.  The default borrows it for
    /// [`ToolResolver::resolve`]; [`MCPAgent`] overrides it because its own
    /// resolution keeps the model past the call (its resolution cache and
    /// thinking budget work on the shared handle).
    fn resolve_shared(
        &self,
        directive: &str,
        mcp_servers: &[MCPServer],
        context: Option<&str>,
        llm: Option<Arc<dyn BaseLLM>>,
    ) -> Option<ResolvedCall> {
        self.resolve(directive, mcp_servers, context, llm.as_deref())
    }
}

impl ToolResolver for MCPAgent {
    /// Resolution through a borrowed model override.  `MCPAgent::resolve`
    /// needs the override as a shared handle, which a borrow cannot provide,
    /// so this resolves with the agent's own model; the explorer goes through
    /// [`ToolResolver::resolve_shared`], which forwards the override.
    fn resolve(
        &self,
        directive: &str,
        mcp_servers: &[MCPServer],
        context: Option<&str>,
        llm: Option<&dyn BaseLLM>,
    ) -> Option<ResolvedCall> {
        let _ = llm;
        MCPAgent::resolve(self, directive, mcp_servers, context, None)
    }

    fn resolve_shared(
        &self,
        directive: &str,
        mcp_servers: &[MCPServer],
        context: Option<&str>,
        llm: Option<Arc<dyn BaseLLM>>,
    ) -> Option<ResolvedCall> {
        MCPAgent::resolve(self, directive, mcp_servers, context, llm)
    }
}

/// The hosted web search the explorer runs during collect (`WebSearchAgent`).
pub trait WebSearcher: Send + Sync {
    /// Whether web search is enabled for the active project.
    fn is_enabled(&self) -> bool;
    /// Search and return the fetched result content (`""` on any failure).
    fn search(&self, query: &str, progress: Option<ProgressCallback>) -> anyhow::Result<String>;
    /// URLs of the sources the last search used.
    fn last_sources(&self) -> Vec<String>;
    /// url -> fetched page text for the last search's sources.
    fn last_source_extracts(&self) -> HashMap<String, String>;
}

impl WebSearcher for WebSearchAgent {
    fn is_enabled(&self) -> bool {
        WebSearchAgent::is_enabled()
    }
    fn search(&self, query: &str, progress: Option<ProgressCallback>) -> anyhow::Result<String> {
        // The agent's own top_k / num_candidates defaults apply.
        Ok(WebSearchAgent::search(self, query, None, None, progress.as_ref()))
    }
    fn last_sources(&self) -> Vec<String> {
        self.last_sources.read().clone()
    }
    fn last_source_extracts(&self) -> HashMap<String, String> {
        self.last_source_extracts.read().clone()
    }
}

/// The model-service searches collect fans out over (duck-typed
/// `model_service` in Python).
pub trait ExplorerSearchService: Send + Sync {
    /// Vector search → `(documents, citations)`, or `None` when nothing matched.
    fn get_document_groups(
        &self,
        query: Option<&str>,
        skill_instruction: Option<&str>,
        conversation_history: Option<&[ConversationTurn]>,
    ) -> anyhow::Result<Option<DocumentGroups>>;
    /// Stored feedback Q&A search.
    fn search_feedback(&self, query: &str, n_results: usize) -> anyhow::Result<Vec<DocumentChunk>>;
    /// Workspace-files search.
    ///
    /// A returned chunk's `document_id` names the file it came from (the
    /// store's `document_name or document_id`, which is all collect reads).
    fn search_workspace(
        &self,
        query: &str,
        work_dir: &str,
        n_results: usize,
        conversation_history: Option<&[ConversationTurn]>,
    ) -> anyhow::Result<Vec<DocumentChunk>>;
}

/// A store hit as the explorer reads it: the text, the distance, and the file
/// it came from (`document_name or document_id`) carried as `document_id`.
fn chunk_from_result(chunk: crate::db::store::ChunkResult) -> DocumentChunk {
    let name = chunk.document_name.filter(|n| !n.is_empty()).unwrap_or(chunk.document_id);
    DocumentChunk {
        document_id: name,
        text: chunk.text,
        distance: Some(chunk.distance),
        metadata: if chunk.metadata.is_empty() {
            None
        } else {
            Some(Value::Object(chunk.metadata).to_string())
        },
        ..Default::default()
    }
}

/// The JSON objects of a list (non-object entries cannot be documents).
fn objects(values: Vec<Value>) -> Vec<Map<String, Value>> {
    values
        .into_iter()
        .filter_map(|v| match v {
            Value::Object(m) => Some(m),
            _ => None,
        })
        .collect()
}

impl ExplorerSearchService for ModelService {
    fn get_document_groups(
        &self,
        query: Option<&str>,
        skill_instruction: Option<&str>,
        conversation_history: Option<&[ConversationTurn]>,
    ) -> anyhow::Result<Option<DocumentGroups>> {
        Ok(ModelService::get_document_groups(self, query, skill_instruction, conversation_history)
            .map(|(documents, citations)| (objects(documents), objects(citations))))
    }
    fn search_feedback(&self, query: &str, n_results: usize) -> anyhow::Result<Vec<DocumentChunk>> {
        Ok(ModelService::search_feedback(self, query, n_results)
            .into_iter()
            .map(chunk_from_result)
            .collect())
    }
    fn search_workspace(
        &self,
        query: &str,
        work_dir: &str,
        n_results: usize,
        conversation_history: Option<&[ConversationTurn]>,
    ) -> anyhow::Result<Vec<DocumentChunk>> {
        Ok(ModelService::search_workspace(self, query, work_dir, n_results, conversation_history)
            .into_iter()
            .map(chunk_from_result)
            .collect())
    }
}

// ------------------------------------------------------------------ models

/// Collected explore results: vector, feedback, workspace, and web search.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct SearchContext {
    #[serde(default)]
    pub vector_context: String,
    #[serde(default)]
    pub feedback_context: String,
    #[serde(default)]
    pub workspace_context: String,
    /// Citations for the folder files in `workspace_context`, one per file,
    /// in the shape document citations take.
    #[serde(default)]
    pub workspace_citations: Vec<Map<String, Value>>,
    /// Latest-info content from the hosted web search (Pro feature).
    #[serde(default)]
    pub web_context: String,
    /// URLs of the web sources actually used (top reranked links).
    #[serde(default)]
    pub web_sources: Vec<String>,
    /// url -> the fetched page text that source contributed to the prompt.
    #[serde(default)]
    pub web_source_extracts: HashMap<String, String>,
    /// Result of an auto-run read-only MCP tool (a "collect" step).
    #[serde(default)]
    pub mcp_context: String,
    /// Write tools the collect pass resolved for this request but left
    /// uncalled (see [`Explorer::run_mcp_with_deferred`]).  The resolution
    /// was made by the reasoning model over the server's whole catalog, so it
    /// is the strongest evidence the request needs that tool — the planner
    /// promotes it to a TOOL step rather than trusting a small classifier
    /// that never saw the tool list.
    #[serde(default)]
    pub deferred_tools: Vec<MatchedTool>,
    /// Read-only tools the collect pass called for this request, whose
    /// results are in `mcp_context`.  A request the router misreads as a
    /// local ACTION with nothing to act on is planned as a step on the server
    /// these came from (see `planner::promote_deferred_tool`).
    #[serde(default)]
    pub consulted_tools: Vec<MatchedTool>,
    /// One citation per MCP tool whose result reached the context — the
    /// read-only call made during explore and every planned tool step — so
    /// the answer credits the connected app it came from, the way web pages
    /// and documents are credited.  See [`mcp_tool_citation`].
    #[serde(default)]
    pub tool_citations: Vec<Map<String, Value>>,
    #[serde(default)]
    pub citations: Option<Vec<Map<String, Value>>>,
    /// Per-document search results (`{"document_name", "content"}`), ordered
    /// by relevance, consumed by the orchestrator's document fan-out.
    #[serde(default)]
    pub documents: Vec<Map<String, Value>>,
    /// Document names the fan-out kept; `None` = no fan-out filtering applied.
    #[serde(default)]
    pub cited_documents: Option<Vec<String>>,
}

/// The tools the collect pass resolved, beyond the text it gathered.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CollectPicks {
    /// Write tools resolved and left uncalled for a planned step.
    pub deferred: Vec<MatchedTool>,
    /// Read-only tools resolved and called.
    pub consulted: Vec<MatchedTool>,
}

/// One server's pick: the tool, and whether it was left uncalled (a write).
#[derive(Debug, Clone, PartialEq)]
pub struct CollectPick {
    pub tool: MatchedTool,
    pub deferred: bool,
}

/// Header that opens each server's section of the explore-time MCP context:
/// `MCP TOOL (<server>.<tool>):`.  Written by `run_mcp_server_inner` and read
/// back by [`mcp_context_citations`].
const MCP_TOOL_HEADER_PREFIX: &str = "MCP TOOL (";
const MCP_TOOL_HEADER_SUFFIX: &str = "):";

/// The citation for one MCP tool call: `document_id` keys it (so repeated
/// calls of the same tool collapse to one reference) and `document_name` is
/// what a client shows.
///
/// It carries no `_extract`, so answer verification always keeps it.  That is
/// deliberate and matches the judge's policy on tool output: the call was made
/// on purpose and its result was handed to the model as the data to answer
/// from, and tool payloads are usually JSON, which an embedding comparison
/// against a prose answer would score as unrelated.
pub fn mcp_tool_citation(server_name: &str, tool_name: &str) -> Map<String, Value> {
    let server = server_name.trim();
    let tool = tool_name.trim();
    let (id, name) = if server.is_empty() {
        (format!("mcp:{tool}"), tool.to_string())
    } else {
        (format!("mcp:{server}.{tool}"), format!("{server} · {tool}"))
    };
    let mut citation = Map::new();
    citation.insert("document_id".into(), Value::String(id));
    citation.insert("document_name".into(), Value::String(name));
    citation.insert("source".into(), Value::String(CitationSource::Mcp.as_str().into()));
    if !server.is_empty() {
        citation.insert("server_name".into(), Value::String(server.to_string()));
    }
    citation.insert("tool_name".into(), Value::String(tool.to_string()));
    citation
}

/// Citations for the tools named in an explore-time MCP context, one per
/// `MCP TOOL (<server>.<tool>):` section, in order.
///
/// The header joins server and tool with a dot; tool names don't contain one,
/// server names might, so it splits on the last.
pub fn mcp_context_citations(mcp_context: &str) -> Vec<Map<String, Value>> {
    mcp_context
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_prefix(MCP_TOOL_HEADER_PREFIX)?
                .strip_suffix(MCP_TOOL_HEADER_SUFFIX)
        })
        .filter_map(|call| {
            let (server, tool) = call.rsplit_once('.').unwrap_or(("", call));
            (!tool.trim().is_empty()).then(|| mcp_tool_citation(server, tool))
        })
        .collect()
}

impl SearchContext {
    /// Record `citation` unless one with the same `document_id` is already
    /// there.
    pub fn add_tool_citation(&mut self, citation: Map<String, Value>) {
        let id = citation.get("document_id").cloned();
        if !self.tool_citations.iter().any(|c| c.get("document_id").cloned() == id) {
            self.tool_citations.push(citation);
        }
    }

    /// True when any source contributed context.
    pub fn has_context(&self) -> bool {
        !(self.vector_context.is_empty()
            && self.feedback_context.is_empty()
            && self.workspace_context.is_empty()
            && self.web_context.is_empty()
            && self.mcp_context.is_empty())
    }
}

/// Aggregated exploration output consumed by the orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct EnrichedContext {
    #[serde(default)]
    pub skill_contents: Vec<String>,
    #[serde(default)]
    pub workspace_tree: String,
    #[serde(default)]
    pub mcp_catalogue: String,
    #[serde(default)]
    pub history_insight: String,
    #[serde(default)]
    pub project_files: String,
    #[serde(default)]
    pub analysis: String,
    #[serde(default)]
    pub exploration_steps: Vec<String>,
}

impl EnrichedContext {
    /// True when any enrichment was gathered.
    pub fn has_content(&self) -> bool {
        !(self.skill_contents.is_empty()
            && self.workspace_tree.is_empty()
            && self.mcp_catalogue.is_empty()
            && self.history_insight.is_empty()
            && self.project_files.is_empty()
            && self.analysis.is_empty())
    }
}


// ------------------------------------------------------------------ explorer

/// Unified explorer: context enrichment + filesystem/vector/URL exploration.
///
/// Cheap to clone (every collaborator is shared), so the parallel fan-out can
/// move a copy into each pool job.
#[derive(Clone)]
pub struct Explorer {
    /// `None` only for test doubles built without a model.
    pub llm: Option<Arc<dyn BaseLLM>>,
    pub tool_caller: Option<Arc<MCPToolCaller>>,
    pub arg_extractor: Option<Arc<MCPArgumentExtractor>>,
    pub use_llm_history: bool,
    pub fetch_agent: Arc<FetchAgent>,
    /// Web search (Pro) is part of explore.
    pub web_search_agent: Arc<dyn WebSearcher>,
    /// MCP tool resolution for the collect phase (read-only tools only).
    pub mcp_agent: Arc<dyn ToolResolver>,
    /// Call-plan follow-ups (defaults to `crate::mcp::call_plans::follow_up_calls`).
    pub follow_up_calls: FollowUpPlanner,
}

/// `re.sub(r'[\s_\-]+', ' ', ...)` for skill-name matching.
static NAME_SEPARATORS: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\s_\-]+").expect("valid regex"));
/// Below this size (billions of parameters) a skill-matching model's pick
/// must share a word with the skill (see [`shares_a_word`]).
const SMALL_MATCHER_B: f64 = 3.0;

/// Whether `llm` is a small model by its spec; a model without a known size
/// is judged on intent alone, as before.
fn is_small_matcher(llm: &dyn BaseLLM) -> bool {
    crate::model::spec::MODEL_SPECS
        .get(&llm.model_label())
        .is_some_and(|spec| spec.ranking_size_b() > 0.0 && spec.ranking_size_b() < SMALL_MATCHER_B)
}

/// Whether `message` shares a meaningful word with `skill`'s name or
/// description: a word of four letters or more, compared on its first five
/// (so "meeting" meets "meetings"), outside a few that say nothing.
fn shares_a_word(message: &str, skill: &SkillDict) -> bool {
    const SAY_NOTHING: [&str; 16] = [
        "with", "from", "this", "that", "what", "when", "your", "have", "into", "about", "show", "make", "help",
        "please", "using", "them",
    ];
    let stems = |text: &str| -> HashSet<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .map(str::to_lowercase)
            .filter(|w| w.chars().count() >= 4 && !SAY_NOTHING.contains(&w.as_str()))
            .map(|w| w.chars().take(5).collect())
            .collect()
    };
    let skill_text = format!(
        "{} {}",
        skill_name(skill).replace(['-', '_'], " "),
        skill.get("description").map(String::as_str).unwrap_or("")
    );
    !stems(message).is_disjoint(&stems(&skill_text))
}

/// Signed integers in the skill-match reply.
static SIGNED_INT: Lazy<Regex> = Lazy::new(|| Regex::new(r"-?\d+").expect("valid regex"));
/// Word separators in a tool name.
static TOOL_NAME_SEPARATORS: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[_\-\s.]+").expect("valid regex"));
/// Query terms for project-file snippet matching.
static QUERY_TERM: Lazy<Regex> = Lazy::new(|| Regex::new(r"\w{3,}").expect("valid regex"));
/// URLs in a user message.
static URL_IN_MESSAGE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"https?://[^\s<>'")\]]+"#).expect("valid regex"));

impl Explorer {
    /// How many transcript turns the follow-up gate sees.  The decision only
    /// needs the shape of the exchange — what was asked, what came back — not
    /// the full text of a long answer.
    pub const GATE_TURNS: usize = 4;
    pub const GATE_TURN_CHARS: usize = 400;
    pub const GATE_MAX_TOKENS: usize = 8;
    /// How many connected servers are asked for one question.  Each costs a
    /// reasoned resolution and a network round-trip, and a workspace with a
    /// dozen connections would put all of them in front of every answer.
    pub const MAX_SERVERS_PER_QUESTION: usize = 4;

    /// Build an explorer with its fetch agent, web search agent and MCP agent.
    pub fn new(
        llm: Arc<dyn BaseLLM>,
        tool_caller: Option<Arc<MCPToolCaller>>,
        arg_extractor: Option<Arc<MCPArgumentExtractor>>,
        use_llm_for_history: bool,
    ) -> Self {
        let fetch_agent = Arc::new(FetchAgent::new());
        // Web search (Pro) is part of explore: it reuses the fetch agent to
        // pull each result link's content, combined with
        // vector/workspace/local.
        let fetcher: Arc<dyn UrlFetcher> = fetch_agent.clone();
        let web_search_agent = WebSearchAgent::new(Some(fetcher), 3, 10, Some(llm.clone()));
        // MCP tool retrieval (vector) for the collect phase. Only read-only
        // tools are auto-run here; write tools stay planned execute steps.
        let mcp_agent = MCPAgent::new(llm.clone());
        Self {
            llm: Some(llm),
            tool_caller,
            arg_extractor,
            use_llm_history: use_llm_for_history,
            fetch_agent,
            web_search_agent: Arc::new(web_search_agent),
            mcp_agent: Arc::new(mcp_agent),
            follow_up_calls: Arc::new(crate::mcp::call_plans::follow_up_calls),
        }
    }

    /// Rerank web results on a model resolved per search (see
    /// [`LlmResolver`]) instead of the explorer's own model.
    pub fn with_web_llm_resolver(mut self, resolver: LlmResolver) -> Self {
        let fetcher: Arc<dyn UrlFetcher> = self.fetch_agent.clone();
        let agent = WebSearchAgent::new(Some(fetcher), 3, 10, self.llm.clone()).with_llm_resolver(resolver);
        self.web_search_agent = Arc::new(agent);
        self
    }

    // ================================================================ skills

    /// Match a user message to skills, unioning two complementary signals.
    ///
    /// Keyword overlap alone is brittle — a real request ("what is the
    /// purpose of the funds?") may share only one word with the skill, and a
    /// complex prompt may share none.  So the LLM is the primary qualifier
    /// and handles indirect / multi-part phrasing; a cheap deterministic
    /// *direct* topic match against the curated skill name backs it up so an
    /// obvious hit ("funds" ↔ `fund-info`) is never lost to a flaky small
    /// model.
    ///
    /// `llm` overrides the model used for the semantic qualifier — the caller
    /// passes the `skills`-capability model, which is far better at the
    /// intent judgement than the small orchestrator model.  Falls back to
    /// `self.llm` when not given.
    pub fn match_skills(
        &self,
        user_message: &str,
        parsed_skills: &[SkillDict],
        llm: Option<&dyn BaseLLM>,
    ) -> Vec<SkillDict> {
        let matchable: Vec<SkillDict> =
            parsed_skills.iter().filter(|s| !skill_name(s).starts_with("__")).cloned().collect();
        if matchable.is_empty() {
            return Vec::new();
        }

        // A skill the user names outright is the intent signal; asking the
        // model as well only adds a call before the answer can start.
        let direct = Self::direct_name_matches(user_message, &matchable);
        let llm_matched =
            if direct.is_empty() { self.llm_match_skills(user_message, &matchable, llm) } else { Vec::new() };

        // Union, preserving the presentation order; dedup by skill name.
        let keep: HashSet<&str> =
            direct.iter().chain(llm_matched.iter()).map(|s| skill_name(s)).collect();
        let matched: Vec<SkillDict> =
            matchable.iter().filter(|s| keep.contains(skill_name(s))).cloned().collect();

        if !matched.is_empty() {
            tracing::info!(
                target: LOGGER,
                "[skill-match] Matched {} skill(s): {} (direct={})",
                matched.len(),
                py_list(matched.iter().map(|s| skill_name(s))),
                py_list(direct.iter().map(|s| skill_name(s))),
            );
        }
        matched
    }

    /// Ask the LLM which skills apply — the primary, semantic matcher.
    ///
    /// Handles complex or indirectly-worded prompts that share few or no
    /// literal keywords with the skill.  Returns `[]` when the model matches
    /// nothing or the call fails (the deterministic direct-name match in
    /// [`Explorer::match_skills`] still stands on its own).
    pub fn llm_match_skills(
        &self,
        user_message: &str,
        matchable: &[SkillDict],
        llm: Option<&dyn BaseLLM>,
    ) -> Vec<SkillDict> {
        let skill_list = matchable
            .iter()
            .enumerate()
            .map(|(i, s)| {
                format!(
                    "  {}. {}: {}",
                    i + 1,
                    skill_name(s),
                    s.get("description").map(String::as_str).unwrap_or("")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "<system>You decide which instruction sets (skills), if any, apply \
             to the user's request. Judge by INTENT, not topic: match a skill \
             only when the request asks you to perform the task the skill is \
             for. A general or definitional question ('what is X', 'why does X \
             matter', 'what is the purpose of X') is NOT a match for a skill \
             that acts on X (e.g. summarizes or reports X).\n\n\
             Answer with the matching skill number(s), comma-separated. If none \
             apply, answer exactly -1. Prefer -1 when unsure; never pick a \
             skill just because it is the only one listed.\n\n\
             Examples — given skill '1. sales-report: build a summary table of \
             sales figures':\n  \
             request 'summarize this quarter's sales' -> 1\n  \
             request 'what is a sales report?' -> -1\n  \
             request 'what is the purpose of sales?' -> -1\n\n\
             Skills:\n{skill_list}</system>\
             <user>{user_message}</user>"
        );

        let match_llm: Option<&dyn BaseLLM> = llm.or(self.llm.as_deref());
        // Skill matching only needs the model to emit a few skill numbers
        // (e.g. "1, 3"). Cap the output so a missing stop token can't trigger
        // runaway greedy generation that stalls the request.
        let response = match guarded_invoke(match_llm, &prompt, Some(16)) {
            Ok(text) => {
                let response = text.trim().to_lowercase();
                tracing::info!(target: LOGGER, "[skill-match] LLM response: {}", take_chars(&response, 100));
                response
            }
            Err(e) => {
                tracing::warn!(target: LOGGER, "[skill-match] LLM failed: {}", e);
                return Vec::new();
            }
        };

        // Explicit no-match sentinels. Handle these before digit parsing so
        // "-1" isn't misread as skill 1 (`\d+` would match the "1").
        if response.contains("none") && char_len(&response) < 10 {
            return Vec::new();
        }

        let mut matched: Vec<SkillDict> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        // Parse signed integers (handles "1", "1, 3", "-1"). Only positive
        // 1-based indices select a skill; -1 / 0 / out-of-range are ignored,
        // so a "-1" reply yields no matches.
        for token in SIGNED_INT.find_iter(&response) {
            let Some(number) = parse_py_int(token.as_str()) else { continue };
            let idx = number - 1; // 1-based in prompt
            if idx >= 0 && (idx as u128) < matchable.len() as u128 {
                let s = &matchable[idx as usize];
                if seen.insert(skill_name(s).to_string()) {
                    matched.push(s.clone());
                }
            }
        }

        // Fallback: the model sometimes echoes skill names instead of numbers.
        if matched.is_empty() {
            for s in matchable {
                let name = skill_name(s);
                if response.contains(&name.to_lowercase()) && !seen.contains(name) {
                    seen.insert(name.to_string());
                    matched.push(s.clone());
                }
            }
        }

        // A small model picks the only skill listed whatever the request
        // (the on-device router matched a calendar skill to "commit and
        // push" on every turn), so its pick has to share a word with the
        // skill too.  A larger model's judgement stands on intent alone.
        if match_llm.is_some_and(is_small_matcher) {
            matched.retain(|s| shares_a_word(user_message, s));
        }

        matched
    }

    /// Match skills the user *names explicitly* — a direct invocation.
    ///
    /// This is intentionally strict: it fires only when the skill's full name
    /// appears as a phrase in the message (e.g. "run the fund-info skill",
    /// "give me the fund info").  A bare topic mention ("what is the purpose
    /// of the funds?") is NOT a direct match — that judgement belongs to the
    /// LLM qualifier, which weighs the user's intent against the skill's
    /// described use case.
    pub fn direct_name_matches(user_message: &str, skills: &[SkillDict]) -> Vec<SkillDict> {
        let lowered = user_message.to_lowercase();
        let msg = NAME_SEPARATORS.replace_all(&lowered, " ");
        let msg = msg.trim();

        let mut matched = Vec::new();
        for s in skills {
            let name = skill_name(s);
            let lowered_name = name.to_lowercase();
            let phrase = NAME_SEPARATORS.replace_all(&lowered_name, " ");
            let phrase = phrase.trim();
            if phrase.is_empty() {
                continue;
            }
            // Word-boundary phrase match so "fund info" hits but "funding fo"
            // does not.
            let pattern = format!(r"\b{}\b", regex::escape(phrase));
            let hit = Regex::new(&pattern).map(|re| re.is_match(msg)).unwrap_or(false);
            if hit {
                tracing::debug!(target: LOGGER, "[skill-match] direct name invocation: '{}'", name);
                matched.push(s.clone());
            }
        }
        matched
    }

    // ================================================================ searches

    /// Vector/document search — thread-safe, I/O bound.
    ///
    /// Returns `(text, citations, documents)` where `documents` is the
    /// per-document list used for fan-out, or `None` when no results.
    pub fn search_vector(
        query: &str,
        model_service: &dyn ExplorerSearchService,
        skill_instruction: Option<&str>,
        conversation_history: Option<&[ConversationTurn]>,
    ) -> Option<(String, Vec<Map<String, Value>>, Vec<Map<String, Value>>)> {
        match model_service.get_document_groups(Some(query), skill_instruction, conversation_history) {
            Ok(Some((documents, citations))) => {
                let text = documents
                    .iter()
                    .map(|d| {
                        format!(
                            "=== Document: {} ===\n{}",
                            py_str_opt(d.get("document_name")),
                            py_str_opt(d.get("content"))
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n---\n\n");
                tracing::info!(
                    target: LOGGER,
                    "[collect] Vector search returned {} chars from {} document(s)",
                    char_len(&text),
                    documents.len()
                );
                return Some((text, citations, documents));
            }
            Ok(None) => {
                tracing::info!(target: LOGGER, "[collect] Vector search returned no results");
            }
            Err(e) => {
                tracing::warn!(target: LOGGER, "[collect] Vector search failed: {}", e);
            }
        }
        None
    }

    /// Feedback Q&A search — thread-safe, I/O bound.
    pub fn search_feedback(query: &str, model_service: &dyn ExplorerSearchService) -> String {
        match model_service.search_feedback(query, 5) {
            Ok(feedback_chunks) => {
                if !feedback_chunks.is_empty() {
                    let text = feedback_chunks
                        .iter()
                        .map(|c| c.text.as_str())
                        .collect::<Vec<_>>()
                        .join("\n---\n");
                    tracing::info!(
                        target: LOGGER,
                        "[collect] Feedback search returned {} Q&A pair(s)",
                        feedback_chunks.len()
                    );
                    return text;
                }
            }
            Err(e) => {
                tracing::warn!(target: LOGGER, "[collect] Feedback search failed: {}", e);
            }
        }
        String::new()
    }

    /// True when `query` needs fresh web results rather than the transcript.
    ///
    /// A follow-up that only reworks what the assistant already produced
    /// ("now create a table", "summarise that", "translate it") is answerable
    /// from the conversation.  Searching again for it is pure noise: the
    /// query carries none of the original subject, so the results come back
    /// about tables rather than about whatever was being tabulated, and those
    /// pages land in the context and as citations.
    ///
    /// The first message of a session always searches — there is nothing to
    /// rework yet.  Fails open: an LLM error or an unreadable verdict
    /// searches, so this can only suppress a search it is confident about.
    pub fn needs_web_search(
        &self,
        query: &str,
        conversation_history: Option<&[ConversationTurn]>,
    ) -> bool {
        let history = match conversation_history {
            Some(h) if !h.is_empty() => h,
            _ => return true,
        };
        let Some(llm) = self.llm.as_deref() else { return true };

        let turns: Vec<&ConversationTurn> =
            history.iter().filter(|t| t.get("content").is_some_and(|c| !c.is_empty())).collect();
        if turns.is_empty() {
            return true;
        }
        let start = turns.len().saturating_sub(Self::GATE_TURNS);
        let transcript = turns[start..]
            .iter()
            .map(|t| {
                format!(
                    "{}: {}",
                    t.get("role").map(String::as_str).unwrap_or("user"),
                    take_chars(t.get("content").map(String::as_str).unwrap_or(""), Self::GATE_TURN_CHARS)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "<system>
You decide whether a new web search is needed to answer the user's latest message.

Answer SEARCH when the message asks for information that is not already in the
conversation: a new topic, different subject, more recent data, or a detail the
previous answer did not cover.

Answer REUSE when the message only asks to rework what the assistant already
produced — reformat it (table, list, bullets), shorten it, expand it, sort it,
translate it, or explain it. Reworking needs no new information.

Respond with exactly one word: SEARCH or REUSE.
</system>
<user>
Conversation so far:
{transcript}

Latest message: {query}

Does answering the latest message require a new web search?
</user>"
        );

        let verdict = match guarded_invoke(Some(llm), &prompt, Some(Self::GATE_MAX_TOKENS)) {
            Ok(text) => text.trim().to_uppercase(),
            Err(exc) => {
                tracing::warn!(target: LOGGER, "[collect] Web-search gate failed, searching: {}", exc);
                return true;
            }
        };

        if verdict.contains("REUSE") && !verdict.contains("SEARCH") {
            tracing::info!(
                target: LOGGER,
                "[collect] Skipping web search — {} reworks the previous answer, no new information needed",
                py_repr_str(take_chars(query, 80)),
            );
            return false;
        }
        true
    }

    /// Hosted web search (Pro), part of explore — thread-safe, I/O bound.
    ///
    /// Runs the WebSearchAgent, which fetches each result link's content. Any
    /// failure is a silent skip (returns `""`) so the model falls back to its
    /// own knowledge — no error/upsell is injected.
    ///
    /// `progress` is forwarded so each page is reported as it is fetched.
    pub fn search_web(&self, query: &str, progress: Option<ProgressCallback>) -> String {
        let agent = self.web_search_agent.clone();
        let outcome = catch_unwind(AssertUnwindSafe(|| agent.search(query, progress)));
        match outcome {
            Ok(Ok(content)) => {
                if !content.is_empty() {
                    tracing::info!(
                        target: LOGGER,
                        "[collect] Web search returned {} chars",
                        char_len(&content)
                    );
                }
                content
            }
            Ok(Err(e)) => {
                tracing::warn!(target: LOGGER, "[collect] Web search failed: {}", e);
                String::new()
            }
            Err(panic) => {
                tracing::warn!(target: LOGGER, "[collect] Web search failed: {}", panic_message(panic.as_ref()));
                String::new()
            }
        }
    }

    /// Report one step to the client, never failing the work over it.
    ///
    /// A progress callback crosses a thread boundary into the request's event
    /// loop (see `chat.rs`), so it is the kind of thing that can fail long
    /// after the client has gone.  Collect's job is the context, not the
    /// commentary.
    pub fn emit(progress: Option<&ProgressCallback>, step: &str, message: &str) {
        let Some(progress) = progress else { return };
        if catch_unwind(AssertUnwindSafe(|| progress(step, "processing", message))).is_err() {
            tracing::debug!(target: LOGGER, "[collect] progress callback failed");
        }
    }

    /// Gather what the connected servers know about `query`.
    ///
    /// Every server is asked, not just the best-matching one: a question is
    /// put to documents, workspace files, feedback and the web all at once,
    /// and a connected server is a source like any of them.  Resolving over
    /// the whole catalog instead picked one tool for the whole workspace, so
    /// a question spanning two servers came back with half its context — and
    /// on a large catalog the pick was made from a list too long to describe.
    ///
    /// Per server: resolve a call over *its* tools, run it when the tool is
    /// read-only, then follow the connection's call plan to fetch the bodies
    /// behind the records it returned (see `crate::mcp::call_plans`).
    /// Write/side-effecting tools are left for a planned execute step.  A
    /// failure anywhere is a silent skip for that server, never for the rest.
    ///
    /// The argument extractor is only consulted when a resolution leaves a
    /// required argument unfilled, so an explorer without one still asks the
    /// servers — with the arguments resolution chose.
    pub fn run_mcp(
        &self,
        query: &str,
        mcp_servers: Option<&[MCPServer]>,
        project_dir: Option<&str>,
        llm: Option<Arc<dyn BaseLLM>>,
        progress: Option<ProgressCallback>,
    ) -> String {
        self.run_mcp_with_deferred(query, mcp_servers, project_dir, llm, progress).0
    }

    /// [`Self::run_mcp`], plus what it resolved on each server (one pick per
    /// server at most): the write tools it left uncalled and the read tools
    /// it called, for the planner.
    pub fn run_mcp_with_deferred(
        &self,
        query: &str,
        mcp_servers: Option<&[MCPServer]>,
        project_dir: Option<&str>,
        llm: Option<Arc<dyn BaseLLM>>,
        progress: Option<ProgressCallback>,
    ) -> (String, CollectPicks) {
        let _ = project_dir;
        let mut picks = CollectPicks::default();
        let Some(mcp_servers) = mcp_servers.filter(|s| !s.is_empty()) else { return (String::new(), picks) };
        if self.tool_caller.is_none() {
            return (String::new(), picks);
        }

        let mut servers: Vec<&MCPServer> = mcp_servers
            .iter()
            .filter(|s| {
                mcp_connection_is_usable(*s) && !s.system && !s.tools.is_empty()
            })
            .collect();
        if servers.is_empty() {
            return (String::new(), picks);
        }
        if servers.len() > Self::MAX_SERVERS_PER_QUESTION {
            tracing::info!(
                target: LOGGER,
                "[collect] {} connected servers — asking the first {}",
                servers.len(),
                Self::MAX_SERVERS_PER_QUESTION
            );
            servers.truncate(Self::MAX_SERVERS_PER_QUESTION);
        }

        let mut gathered = Vec::new();
        for server in servers {
            let (text, server_picks) = self.run_mcp_server_with_deferred(query, server, llm.clone(), progress.clone());
            for pick in server_picks {
                if pick.deferred {
                    picks.deferred.push(pick.tool);
                } else {
                    // A read that came back empty-handed still says which
                    // server the request is about.
                    picks.consulted.push(pick.tool);
                }
            }
            if !text.is_empty() {
                gathered.push(text);
            }
        }
        (gathered.join("\n\n"), picks)
    }

    /// One server's answer to `query`, or `""`.
    pub fn run_mcp_server(
        &self,
        query: &str,
        server: &MCPServer,
        llm: Option<Arc<dyn BaseLLM>>,
        progress: Option<ProgressCallback>,
    ) -> String {
        self.run_mcp_server_with_deferred(query, server, llm, progress).0
    }

    /// [`Self::run_mcp_server`], plus the tools resolution picked: the one it
    /// called or left for a step, and a write the model planned after it.
    pub fn run_mcp_server_with_deferred(
        &self,
        query: &str,
        server: &MCPServer,
        llm: Option<Arc<dyn BaseLLM>>,
        progress: Option<ProgressCallback>,
    ) -> (String, Vec<CollectPick>) {
        // The server's credentials may be refreshed by a call; the follow-up
        // fetches below must see the renewed token, as they did when Python
        // mutated the one server object in place.
        let mut working = server.clone();
        let mut picks: Vec<CollectPick> = Vec::new();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            self.run_mcp_server_inner(query, &mut working, llm, progress.as_ref(), &mut picks)
        }));
        match outcome {
            Ok(text) => (text, picks),
            Err(panic) => {
                tracing::warn!(
                    target: LOGGER,
                    "[collect] MCP call on '{}' failed, skipping: {}",
                    server.name,
                    panic_message(panic.as_ref())
                );
                (String::new(), Vec::new())
            }
        }
    }

    fn run_mcp_server_inner(
        &self,
        query: &str,
        server: &mut MCPServer,
        llm: Option<Arc<dyn BaseLLM>>,
        progress: Option<&ProgressCallback>,
        picks: &mut Vec<CollectPick>,
    ) -> String {
        let server_name = if server.name.is_empty() { "the server".to_string() } else { server.name.clone() };

        // Resolution is the long leg — a reasoned pick over this server's
        // whole tool list, tens of seconds on a large catalog — and it runs
        // before anything is called, so with no event here the client sits
        // on "Processing..." through all of it and the work reads as a hang.
        // Named per server, because a workspace with two connections pays
        // this once each.
        //
        // Two attempts, because the first pick can turn out to be a tool this
        // account cannot call.  Discovery lists what the server has, not what
        // the plan is entitled to, so a paywalled tool is indistinguishable
        // from a usable one until it answers — and it answers with an
        // advertisement, successfully.  Retrying once with it struck off the
        // catalogue is what lets the same turn fall to the tool that does
        // work (`notion-search` where `notion-ai-search` is Business-only)
        // instead of handing the upsell to the model as though it were the
        // document.
        let mut raw = String::new();
        let mut tool_name = String::new();
        for attempt in 1..=2 {
            Self::emit(progress, "tool", &format!("Choosing a {} tool", server_name));
            let Some(picked) =
                self.mcp_agent.resolve_shared(query, std::slice::from_ref(&*server), None, llm.clone())
            else {
                return String::new();
            };
            tool_name = picked.tool_name.clone();
            if !ToolAccess::on(server, &tool_name).is_read() {
                tracing::info!(
                    target: LOGGER,
                    "[collect] MCP tool '{}' is not read-only — leaving to a planned execute step",
                    tool_name
                );
                picks.push(CollectPick {
                    tool: MatchedTool {
                        server_id: server.id.clone(),
                        server_name: server_name.clone(),
                        tool_name: tool_name.clone(),
                    },
                    deferred: true,
                });
                return String::new();
            }

            // The tool and its arguments, not just the server: "Calling
            // Notion" is indistinguishable between reading a page and
            // searching for one, and which of the two ran is the thing a user
            // watching this actually wants to know.
            Self::emit(
                progress,
                "tool",
                &format!("Calling {}", format_call(&server_name, &tool_name, &picked.arguments)),
            );
            picks.retain(|p| p.deferred);
            picks.push(CollectPick {
                tool: MatchedTool {
                    server_id: server.id.clone(),
                    server_name: server_name.clone(),
                    tool_name: tool_name.clone(),
                },
                deferred: false,
            });
            // A write the model planned after this read ("find the user, then
            // file the issue") is the step the request is really for: set it
            // aside for the planner, as a write resolved first would be.
            if let Some(write) = picked.also.iter().find(|c| !ToolAccess::on(server, &c.tool_name).is_read()) {
                if !picks.iter().any(|p| p.deferred && p.tool.tool_name == write.tool_name) {
                    tracing::info!(
                        target: LOGGER,
                        "[collect] model planned {} after {} — leaving it to a planned execute step",
                        write.tool_name,
                        tool_name
                    );
                    picks.push(CollectPick {
                        tool: MatchedTool {
                            server_id: server.id.clone(),
                            server_name: server_name.clone(),
                            tool_name: write.tool_name.clone(),
                        },
                        deferred: true,
                    });
                }
            }
            raw = self.call_mcp_tool_on(server, &tool_name, &picked.arguments, query);
            if raw.is_empty() {
                return String::new();
            }

            if !is_entitlement_notice(&raw) {
                break;
            }

            // Struck off before the retry, so the same resolution cannot come
            // back with the same answer.
            mark_tool_unavailable(&server.id, &tool_name);
            tracing::warn!(
                target: LOGGER,
                "[collect] {}.{} is not available on this account — {}. Dropping it from the catalogue{}",
                server_name,
                tool_name,
                take_chars(raw.trim().lines().next().unwrap_or(""), 160),
                if attempt == 1 { " and re-resolving" } else { "" },
            );
            Self::emit(
                progress,
                "tool",
                &format!("{}.{} needs a higher plan — trying another tool", server_name, tool_name),
            );
            raw = String::new();
        }
        if raw.is_empty() {
            return String::new();
        }

        // A listing answers "which records", not "what do they say".  The
        // connection's plan names the tool that fetches a record's body, so a
        // search hit becomes the page behind it rather than its title.
        let detail = self.fetch_record_details_on(server, &tool_name, &raw, progress);
        let body = if detail.is_empty() { raw } else { format!("{}\n\n{}", raw, detail) };

        tracing::info!(
            target: LOGGER,
            "[collect] {}.{} returned {} chars",
            server.name,
            tool_name,
            char_len(&body)
        );
        // The payload, not just its size.  A short answer here is the
        // difference between "the tool found nothing", "the tool answered in
        // prose that carries no record ids to follow up on" and "the
        // follow-up plan missed" — three different bugs that a char count
        // cannot tell apart, and the execute path has always logged its raw
        // result for exactly this reason.
        tracing::info!(
            target: LOGGER,
            "[collect] {}.{} raw result: {}",
            server.name,
            tool_name,
            take_chars(&body, 1000)
        );
        format!(
            "{MCP_TOOL_HEADER_PREFIX}{}.{}{MCP_TOOL_HEADER_SUFFIX}\n{}",
            server.name, tool_name, body
        )
    }

    /// Run one read-only tool and return its text, or `""`.
    pub fn call_mcp_tool(
        &self,
        server: &MCPServer,
        tool_name: &str,
        arguments: &Map<String, Value>,
        query: &str,
    ) -> String {
        let mut working = server.clone();
        self.call_mcp_tool_on(&mut working, tool_name, arguments, query)
    }

    fn call_mcp_tool_on(
        &self,
        server: &mut MCPServer,
        tool_name: &str,
        arguments: &Map<String, Value>,
        query: &str,
    ) -> String {
        let Some(tool_caller) = self.tool_caller.as_ref() else { return String::new() };
        let input_schema: Map<String, Value> = server
            .tools
            .iter()
            .find(|t| t.name == tool_name)
            .and_then(|t| t.input_schema.clone())
            .unwrap_or_default();

        // Resolution filled the required arguments in the same call that
        // picked the tool; the extractor is only needed when it could not.
        let mut arguments = arguments.clone();
        if !missing_required(&input_schema, &arguments).is_empty() {
            if let Some(extractor) = self.arg_extractor.as_ref() {
                arguments =
                    extractor.invoke(query, tool_name, &Value::Object(input_schema.clone()), None, None);
                arguments.remove("_missing_params");
            }
        }

        // The call itself, in the log, before it goes out: a resolution that
        // picked a tool and a call that actually reached the server are
        // different events, and only the second one can fail.
        tracing::info!(target: LOGGER, "[collect] calling {}", format_call(&server.name, tool_name, &arguments));
        let (result, _refreshed) = tool_caller.call_with_refresh(server, tool_name, Some(&arguments), None);
        let content = result.content.as_deref().unwrap_or(&[]);
        if !result.success || content.is_empty() {
            tracing::info!(
                target: LOGGER,
                "[collect] {} returned nothing ({})",
                tool_name,
                result.error.as_deref().filter(|e| !e.is_empty()).unwrap_or("empty result")
            );
            return String::new();
        }
        content_text(content)
    }

    /// Bodies behind the records `payload` lists, per the connection's plan.
    ///
    /// Returns `""` when the connection has no plan, the plan names no detail
    /// tool the server has, or the payload carries no record ids — every one
    /// of which is the ordinary case for a server whose listing is already
    /// the answer.
    pub fn fetch_record_details(
        &self,
        server: &MCPServer,
        tool_name: &str,
        payload: &str,
        progress: Option<ProgressCallback>,
    ) -> String {
        let mut working = server.clone();
        self.fetch_record_details_on(&mut working, tool_name, payload, progress.as_ref())
    }

    fn fetch_record_details_on(
        &self,
        server: &mut MCPServer,
        tool_name: &str,
        payload: &str,
        progress: Option<&ProgressCallback>,
    ) -> String {
        let planner = self.follow_up_calls.clone();
        let calls = {
            let server_ref: &MCPServer = server;
            match catch_unwind(AssertUnwindSafe(|| planner(server_ref, tool_name, payload))) {
                Ok(calls) => calls,
                Err(panic) => {
                    tracing::debug!(
                        target: LOGGER,
                        "[collect] no follow-up plan for {}: {}",
                        tool_name,
                        panic_message(panic.as_ref())
                    );
                    return String::new();
                }
            }
        };
        if calls.is_empty() {
            return String::new();
        }
        let Some(tool_caller) = self.tool_caller.as_ref() else { return String::new() };

        let server_name = if server.name.is_empty() { "the server".to_string() } else { server.name.clone() };
        let mut bodies = Vec::new();
        // Announced per record, not once for the batch: a search hit list
        // turns into one fetch each, and a question that matched five pages
        // spends most of its collect here.  Reported with the same call
        // format as the resolved tool, so the follow-ups read as the calls
        // they are rather than as a single opaque step.
        let total = calls.len();
        for (index, (detail_tool, arguments)) in calls.iter().enumerate() {
            let index = index + 1;
            tracing::info!(
                target: LOGGER,
                "[collect] following up with {}",
                format_call(&server.name, detail_tool, arguments)
            );
            Self::emit(
                progress,
                "tool",
                &format!(
                    "Reading {} of {} from {}: {}",
                    index,
                    total,
                    server_name,
                    format_call("", detail_tool, arguments)
                ),
            );
            let (result, _refreshed) = tool_caller.call_with_refresh(server, detail_tool, Some(arguments), None);
            let content = result.content.as_deref().unwrap_or(&[]);
            if !result.success || content.is_empty() {
                continue;
            }
            let text = content_text(content);
            if !text.is_empty() {
                bodies.push(take_chars(&text, MAX_RECORD_BODY_CHARS).to_string());
            }
        }
        if bodies.is_empty() {
            return String::new();
        }
        tracing::info!(
            target: LOGGER,
            "[collect] fetched {} record body/bodies from {}",
            bodies.len(),
            server.name
        );
        bodies.join("\n\n")
    }

    /// True for skill/system files that are instructions or tool metadata,
    /// not user data — they must never leak into retrieved context.
    ///
    /// Covers skill files (`skills/` dirs, `SKILL.md`), MCP catalogue files
    /// (`mcp-*`), and internal files whose basename is `__`-prefixed.
    /// Without this a fund/table prompt can sweep tool descriptions in as
    /// fake data rows (e.g. "Smartloop | SMTL | N/A").
    pub fn is_non_data_file(fname: &str) -> bool {
        if fname.is_empty() {
            return false;
        }
        // `os.path.basename`: everything after the last separator.
        let base = fname.rsplit('/').next().unwrap_or(fname);
        fname.contains("/skills/")
            || fname.starts_with("skills/")
            || fname.to_uppercase().ends_with("SKILL.MD")
            || fname.starts_with("mcp-")
            || base.starts_with("mcp-")
            || base.starts_with("__")
    }

    /// Workspace file search — thread-safe, I/O bound.
    ///
    /// Returns the context text and a citation per file in it.  A citation
    /// carries the file's path under `cwd`, so a client can open it, and the
    /// text it put in front of the model as `_extract`, which the answer is
    /// checked against before the file is credited.
    pub fn search_workspace(
        query: &str,
        model_service: &dyn ExplorerSearchService,
        cwd: &str,
        conversation_history: Option<&[ConversationTurn]>,
    ) -> (String, Vec<Map<String, Value>>) {
        match model_service.search_workspace(query, cwd, 10, conversation_history) {
            Ok(ws_chunks) => {
                if !ws_chunks.is_empty() {
                    // Best (nearest) chunk per file, in first-seen order.
                    let mut seen_files: Vec<(String, DocumentChunk)> = Vec::new();
                    for chunk in ws_chunks {
                        let fname = chunk.document_id.clone();
                        if !fname.is_empty() && Self::is_non_data_file(&fname) {
                            continue;
                        }
                        match seen_files.iter().position(|(f, _)| *f == fname) {
                            None => seen_files.push((fname, chunk)),
                            Some(i) => {
                                if chunk.distance.unwrap_or(0.0) < seen_files[i].1.distance.unwrap_or(0.0) {
                                    seen_files[i].1 = chunk;
                                }
                            }
                        }
                    }
                    let extracts: Vec<(&String, String)> = seen_files
                        .iter()
                        .map(|(fname, chunk)| (fname, take_chars(&chunk.text, 2000).to_string()))
                        .collect();
                    let text = extracts
                        .iter()
                        .map(|(fname, extract)| format!("--- {} ---\n{}", fname, extract))
                        .collect::<Vec<_>>()
                        .join("\n");
                    let citations = extracts
                        .iter()
                        .map(|(fname, extract)| {
                            let mut citation = Map::new();
                            citation.insert("document_id".into(), Value::String((*fname).clone()));
                            citation.insert("document_name".into(), Value::String((*fname).clone()));
                            let path = Path::new(cwd).join(fname.as_str());
                            citation.insert("path".into(), Value::String(path.to_string_lossy().into_owned()));
                            citation.insert("source".into(), Value::String(CitationSource::Document.as_str().into()));
                            citation.insert("_extract".into(), Value::String(extract.clone()));
                            citation.insert("_chunks".into(), Value::Array(Vec::new()));
                            citation.insert("_vectors".into(), Value::Array(Vec::new()));
                            citation
                        })
                        .collect();
                    tracing::info!(
                        target: LOGGER,
                        "[collect] Workspace search returned {} file(s)",
                        seen_files.len()
                    );
                    return (text, citations);
                }
            }
            Err(e) => {
                tracing::warn!(target: LOGGER, "[collect] Workspace search failed: {}", e);
            }
        }
        (String::new(), Vec::new())
    }

    /// Fan-out vector/feedback/workspace searches in parallel.
    ///
    /// The I/O-bound searches run concurrently on a [`ProjectBoundExecutor`]
    /// (the caller's `executor`, or a pool of six owned by this call).  Skill
    /// tool execution still runs afterward in the DAG step nodes (tools may
    /// depend on search results).
    ///
    /// `progress(step, status, message)` — when supplied — reports the web
    /// search leg.  The local searches are fast and finish together; the web
    /// search is the one the user waits on.
    pub fn collect_parallel(
        &self,
        query: &str,
        model_service: Option<Arc<dyn ExplorerSearchService>>,
        cwd: Option<&str>,
        enriched: Option<&EnrichedContext>,
        matched_skills: Option<&[SkillDict]>,
        mcp_servers: Option<&[MCPServer]>,
        mcp_tool_contexts: Option<&[MCPToolContext]>,
        conversation_history: Option<&[ConversationTurn]>,
        executor: Option<&ProjectBoundExecutor>,
        project_dir: Option<&str>,
        progress: Option<ProgressCallback>,
        mcp_llm: Option<Arc<dyn BaseLLM>>,
    ) -> SearchContext {
        let _ = (enriched, mcp_tool_contexts);
        let mut ctx = SearchContext::default();

        let Some(model_service) = model_service else { return ctx };

        // Project-bound: the fan-out searches below resolve against the
        // requesting project (see crate::workspace::context).
        let owned = if executor.is_none() { Some(DetachedPool::new(6)) } else { None };
        let pool: &ProjectBoundExecutor = match (executor, owned.as_ref()) {
            (Some(pool), _) => pool,
            (None, Some(owned)) => owned.get(),
            (None, None) => unreachable!("an owned pool is built when none is passed"),
        };

        let skill_matched = matched_skills.is_some_and(|s| !s.is_empty());

        // When a skill matched, the pipeline is self-contained — do not bias
        // the vector search with previous conversation turns.  The current
        // user message is the only input that matters.
        let history_for_search: Option<Vec<ConversationTurn>> =
            if skill_matched { None } else { conversation_history.map(<[ConversationTurn]>::to_vec) };

        let f_vec = {
            let (query, service, history) = (query.to_string(), model_service.clone(), history_for_search.clone());
            pool.submit(move || Self::search_vector(&query, service.as_ref(), None, history.as_deref()))
        };
        let f_fb = {
            let (query, service) = (query.to_string(), model_service.clone());
            pool.submit(move || Self::search_feedback(&query, service.as_ref()))
        };
        let f_ws = match cwd.filter(|c| !c.is_empty()) {
            Some(cwd) => {
                let (query, service, cwd, history) =
                    (query.to_string(), model_service.clone(), cwd.to_string(), history_for_search.clone());
                Some(pool.submit(move || {
                    Self::search_workspace(&query, service.as_ref(), &cwd, history.as_deref())
                }))
            }
            None => None,
        };
        // MCP (read-only): part of explore, and on the same footing as every
        // other source — the connected servers are searched for this question
        // alongside documents, workspace files, feedback and the web, in the
        // same fan-out.  The tool is resolved by reasoning over what discovery
        // found (MCPAgent) and run only if it is read-only; write tools stay
        // planned execute steps.
        //
        // This used to be gated on a vectorized copy of the server's content:
        // when that local index matched, the live call was skipped and the
        // answer came from a snapshot. The snapshot is gone, so the source of
        // truth is the server, asked now.
        //
        // Not when a skill matched.  A skill is a procedure, and the plan
        // built from it makes its tool calls as steps with the skill's
        // instructions in hand.  Asked here, the server gets the whole
        // request as one search — for a bare `/job-match` that was the
        // skill's name (`searchCleraRoles(query='job-match')`), and with the
        // body it is a multi-step procedure no single call answers.
        let f_mcp = match mcp_servers.filter(|s| !s.is_empty()) {
            Some(servers) if !skill_matched => {
                let this = self.clone();
                let query = query.to_string();
                let servers = servers.to_vec();
                let project_dir = project_dir.map(str::to_string);
                let (llm, progress) = (mcp_llm.clone(), progress.clone());
                Some(pool.submit(move || {
                    this.run_mcp_with_deferred(&query, Some(&servers), project_dir.as_deref(), llm, progress)
                }))
            }
            _ => None,
        };

        // Collect primary results (local retrieval + MCP tools)
        if let Some((text, citations, documents)) = join(f_vec) {
            ctx.vector_context = text;
            ctx.citations = Some(citations);
            ctx.documents = documents;
        }

        ctx.feedback_context = join(f_fb);

        if let Some(f_ws) = f_ws {
            (ctx.workspace_context, ctx.workspace_citations) = join(f_ws);
        }

        if let Some(f_mcp) = f_mcp {
            let (text, picks) = join(f_mcp);
            ctx.mcp_context = text;
            ctx.deferred_tools = picks.deferred;
            ctx.consulted_tools = picks.consulted;
            for citation in mcp_context_citations(&ctx.mcp_context) {
                ctx.add_tool_citation(citation);
            }
        }

        // Web search runs alongside local retrieval, not behind it.  Gating
        // it on "did anything match locally?" made a single indexed document
        // or stored feedback Q&A — retrieval always returns its nearest
        // neighbours, related or not — silently cancel the search, leaving
        // the model to answer a camera question from cat notes.  Deciding
        // which of the collected sources actually answers the question is the
        // synthesis step's job; collect just gathers.
        //
        // That applies to a live MCP tool result too. It used to cancel the
        // web search outright, on the reasoning that a deliberately invoked
        // tool had already answered — but "deliberately invoked" only means
        // the router picked it, not that its output answered the question,
        // and a query spanning both ("how does our roadmap compare to what
        // they announced") lost half its context. Both are gathered now and
        // the ordering above puts the local one first.
        //
        // The remaining exception is a follow-up that only reworks the answer
        // already on screen ("now make it a table") — see
        // [`Explorer::needs_web_search`].
        if self.web_search_agent.is_enabled() && self.needs_web_search(query, conversation_history) {
            // Announce the search before it starts.  It is the longest leg of
            // collect — a provider round-trip plus one fetch per result — and
            // it runs on this thread while the local searches above have
            // already returned, so with no event here the client shows the
            // previous step until the whole search is done.  A skipped search
            // stays silent, as the rest of this path is.
            Self::emit(progress.as_ref(), "web_search", "Searching the web for the latest information");
            ctx.web_context = self.search_web(query, progress.clone());
            let sources = self.web_search_agent.last_sources();
            if !sources.is_empty() {
                ctx.web_sources = sources;
                ctx.web_source_extracts = self.web_search_agent.last_source_extracts();
            }
        }

        // Skill tool execution runs in the DAG step nodes — do NOT run them
        // here during collection.  Eager execution here would bypass argument
        // extraction, validation, and result accumulation in reason().
        ctx
    }

    // ================================================================ enrichment

    /// Pre-planning enrichment: MCP catalogue and history insight.
    pub fn enrich(
        &self,
        available_mcp_servers: Option<&[Map<String, Value>]>,
        conversation_history: Option<&[ConversationTurn]>,
    ) -> EnrichedContext {
        let mut ctx = EnrichedContext::default();

        if let Some(servers) = available_mcp_servers.filter(|s| !s.is_empty()) {
            ctx.mcp_catalogue = self.catalogue_mcp(servers);
            if !ctx.mcp_catalogue.is_empty() {
                let server_count = ctx.mcp_catalogue.matches('\n').count() + 1;
                ctx.exploration_steps.push(format!("MCP: catalogued {} server(s)", server_count));
                tracing::info!(
                    target: LOGGER,
                    "[explorer] MCP catalogue: {} chars",
                    char_len(&ctx.mcp_catalogue)
                );
            }
        }

        if let Some(history) = conversation_history.filter(|h| !h.is_empty()) {
            ctx.history_insight = Self::format_history(history);
            if !ctx.history_insight.is_empty() {
                ctx.exploration_steps.push(format!("History: {} relevant turn(s)", history.len()));
                tracing::info!(
                    target: LOGGER,
                    "[explorer] history insight: {} chars",
                    char_len(&ctx.history_insight)
                );
            }
        }

        ctx
    }

    /// LLM route analysis (SKILL / DOCUMENT / TOOL / CONVERSATIONAL); `""` on
    /// failure.
    pub fn analyze(
        &self,
        user_message: &str,
        enriched: &EnrichedContext,
        conversation_history: Option<&[ConversationTurn]>,
        skills: Option<&[String]>,
    ) -> String {
        let _ = conversation_history;
        let mut all_skills: Vec<String> = skills.map(<[String]>::to_vec).unwrap_or_default();
        for sc in &enriched.skill_contents {
            if !all_skills.contains(sc) {
                all_skills.push(sc.clone());
            }
        }

        let mut skill_summaries = Vec::new();
        for sc in &all_skills {
            let (mut name, mut desc) = (String::new(), String::new());
            for line in sc.split('\n') {
                let stripped = line.trim();
                if let Some(rest) = stripped.strip_prefix("name:") {
                    name = strip_quotes(rest.trim()).to_string();
                } else if let Some(rest) = stripped.strip_prefix("description:") {
                    desc = strip_quotes(rest.trim()).to_string();
                }
            }
            if !name.is_empty() {
                skill_summaries.push(if desc.is_empty() {
                    format!("  - {}", name)
                } else {
                    format!("  - {}: {}", name, desc)
                });
            }
        }
        let skills_block =
            if skill_summaries.is_empty() { "None".to_string() } else { skill_summaries.join("\n") };

        let or_none = |s: &str| if s.is_empty() { "None".to_string() } else { s.to_string() };
        let mcp_block = or_none(&enriched.mcp_catalogue);
        let history_block = or_none(&enriched.history_insight);
        let project_block = or_none(take_chars(&enriched.project_files, 3000));

        let mut prompt = format!(
            "<system>You are analyzing a user request. You have gathered \
             context from indexed documents and skills.\n\n\
             Skills are instructions that define how to handle specific types \
             of requests. Each skill has a name and describes when it applies. \
             Match the user request against ALL skills before deciding.\n\n\
             Decide the route:\n\
             - SKILL: A skill matches the user request. The skill's own \
             instructions say how to respond (it may be conversational or \
             may need a tool call).\n\
             - DOCUMENT: The gathered document context answers the question. \
             The skill may refine how to use the documents.\n\
             - TOOL: An external MCP tool call is needed (fetch URL, API, etc.).\n\
             - CONVERSATIONAL: General knowledge is sufficient.</system>\n\
             <user>\n\
             User request: {user_message}\n\n\
             ---\n\
             SKILLS:\n{skills_block}\n\n\
             ---\n\
             DOCUMENT CONTEXT:\n{project_block}\n\n\
             ---\n\
             MCP TOOLS:\n{mcp_block}\n\n\
             ---\n\
             HISTORY:\n{history_block}\n"
        );

        prompt.push_str(
            "\n---\n\
             Answer briefly:\n\
             1. What does the user want?\n\
             2. Does a skill match? If yes, which skill name?\n\
             3. Is the document context relevant to this request? (yes/no)\n\
             4. Route: SKILL, DOCUMENT, TOOL, or CONVERSATIONAL\n\
             </user>",
        );

        match guarded_invoke(self.llm.as_deref(), &prompt, None) {
            Ok(text) => {
                let result = text.trim().to_string();
                tracing::info!(
                    target: LOGGER,
                    "[analyze] LLM analysis ({} chars): {}",
                    char_len(&result),
                    take_chars(&result, 300)
                );
                result
            }
            Err(e) => {
                tracing::warn!(target: LOGGER, "[analyze] Analysis failed (non-fatal): {}", e);
                String::new()
            }
        }
    }

    /// Read `SKILL.md` (recursively) and other top-level `*.md` skill files
    /// under `skills_dir`, each trimmed to [`MAX_SKILL_FILE_CHARS`].
    pub fn scan_skills(&self, skills_dir: &str) -> Vec<String> {
        let mut contents = Vec::new();
        let skills_path = Path::new(skills_dir);
        if !skills_path.is_dir() {
            return contents;
        }

        // `rglob("SKILL.md")` + `glob("*.md")` minus the top-level SKILL.md.
        let mut candidates: Vec<PathBuf> = walkdir::WalkDir::new(skills_path)
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.depth() > 0 && e.file_name() == "SKILL.md")
            .map(|e| e.into_path())
            .collect();
        if let Ok(entries) = std::fs::read_dir(skills_path) {
            for entry in entries.filter_map(Result::ok) {
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.ends_with(".md") && name != "SKILL.md" {
                    candidates.push(entry.path());
                }
            }
        }
        // `sorted(paths)`: pathlib orders by path components.
        candidates.sort_by(|a, b| {
            let pa: Vec<_> = a.components().collect();
            let pb: Vec<_> = b.components().collect();
            pa.cmp(&pb)
        });

        for md_path in candidates {
            match read_text_lossy(&md_path) {
                Ok(text) => {
                    let mut trimmed = take_chars(&text, MAX_SKILL_FILE_CHARS).to_string();
                    if char_len(&text) > MAX_SKILL_FILE_CHARS {
                        trimmed.push_str("\n... (truncated)");
                    }
                    if !trimmed.trim().is_empty() {
                        contents.push(trimmed);
                    }
                }
                Err(exc) => {
                    tracing::debug!(
                        target: LOGGER,
                        "[explorer] could not read skill {}: {}",
                        md_path.display(),
                        exc
                    );
                }
            }
        }

        contents
    }

    /// A short indented tree of `cwd`.
    pub fn scan_workspace(&self, cwd: &str) -> String {
        let mut lines: Vec<String> = Vec::new();
        let cwd = realpath(cwd);
        let cwd_str = cwd.to_string_lossy().into_owned();

        walk(&cwd, &mut |root, dirs, files| {
            let mut kept: Vec<String> =
                dirs.iter().filter(|d| !d.starts_with('.') && !SKIP_DIRS.contains(&d.as_str())).cloned().collect();
            kept.sort();
            *dirs = kept;

            let rel = relpath(root, &cwd);
            let level = if rel == "." { 0 } else { rel.matches(std::path::MAIN_SEPARATOR).count() + 1 };
            let indent = "  ".repeat(level);
            let folder = if rel != "." { basename(root) } else { cwd_str.clone() };
            lines.push(format!("{}{}/", indent, folder));

            let sub_indent = "  ".repeat(level + 1);
            let mut sorted_files = files.to_vec();
            sorted_files.sort();
            for fname in sorted_files.iter().take(20) {
                lines.push(format!("{}{}", sub_indent, fname));
            }

            if lines.len() >= MAX_TREE_ENTRIES {
                lines.push("  ... (truncated)".to_string());
                return false;
            }
            true
        });

        lines.join("\n")
    }

    /// `server: tool1, tool2, …` per server with tools.
    pub fn catalogue_mcp(&self, servers: &[Map<String, Value>]) -> String {
        let mut lines = Vec::new();
        for srv in servers {
            let srv_name = [srv.get("name"), srv.get("id")]
                .into_iter()
                .flatten()
                .find(|v| py_truthy(v))
                .map(py_str)
                .unwrap_or_else(|| "unknown".to_string());
            let tools: Vec<Value> = match srv.get("tools") {
                Some(Value::Array(tools)) => tools.clone(),
                _ => Vec::new(),
            };
            if tools.is_empty() {
                continue;
            }
            let mut names: Vec<String> = tools
                .iter()
                .take(10)
                .map(|t| t.get("name").map(py_str).unwrap_or_else(|| "?".to_string()))
                .collect();
            if tools.len() > 10 {
                names.push(format!("… +{} more", tools.len() - 10));
            }
            lines.push(format!("{}: {}", srv_name, names.join(", ")));
        }
        lines.join("\n")
    }

    /// Format recent history turns into a concise text block.
    ///
    /// The history is already selected by the caller (recency-based, via
    /// `ModelService::get_relevant_history`), so this simply renders each
    /// turn as `role: content` (newlines flattened) without any keyword
    /// extraction or summarization.
    pub fn format_history(history: &[ConversationTurn]) -> String {
        if history.is_empty() {
            return String::new();
        }
        history
            .iter()
            .map(|turn| {
                let role = turn.get("role").map(String::as_str).unwrap_or("unknown");
                let content = turn.get("content").map(String::as_str).unwrap_or("").trim().replace('\n', " ");
                format!("{}: {}", role, content)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Query-matched snippets from project source files under `cwd`.
    pub fn scan_project_files(&self, cwd: &str, user_message: &str) -> String {
        let cwd = realpath(cwd);

        let mut candidates: Vec<String> = Vec::new();
        walk(&cwd, &mut |root, dirs, files| {
            let mut kept: Vec<String> = dirs
                .iter()
                .filter(|d| !d.starts_with('.') && !SKIP_DIRS.contains(&d.as_str()) && !d.ends_with(".egg-info"))
                .cloned()
                .collect();
            kept.sort();
            *dirs = kept;
            let mut sorted_files = files.to_vec();
            sorted_files.sort();
            for fname in sorted_files {
                let ext = splitext_ext(&fname).to_lowercase();
                if !PROJECT_EXTENSIONS.contains(&ext.as_str()) {
                    continue;
                }
                candidates.push(relpath(&root.join(&fname), &cwd));
                if candidates.len() >= MAX_PROJECT_FILES {
                    break;
                }
            }
            candidates.len() < MAX_PROJECT_FILES
        });

        if candidates.is_empty() {
            return String::new();
        }

        let lowered = user_message.to_lowercase();
        let query_terms: Vec<&str> = QUERY_TERM
            .find_iter(&lowered)
            .map(|m| m.as_str())
            .filter(|w| !QUERY_STOP_WORDS.contains(w))
            .collect();

        let mut parts: Vec<String> = Vec::new();
        let mut total_chars = 0usize;
        for fpath in &candidates {
            let full_path = cwd.join(fpath);
            let Ok(content) = read_text_lossy(&full_path) else { continue };

            if content.trim().is_empty() {
                continue;
            }

            let mut snippet_lines: Vec<String> = Vec::new();
            if !query_terms.is_empty() {
                let lines: Vec<&str> = content.split('\n').collect();
                let mut matched_indices: std::collections::BTreeSet<usize> = Default::default();
                for (i, line) in lines.iter().enumerate() {
                    let line_lower = line.to_lowercase();
                    if query_terms.iter().any(|term| line_lower.contains(term)) {
                        for j in i.saturating_sub(1)..(i + 2).min(lines.len()) {
                            matched_indices.insert(j);
                        }
                    }
                }
                for i in matched_indices {
                    if snippet_lines.len() >= MAX_PROJECT_SNIPPET_LINES {
                        break;
                    }
                    snippet_lines.push(format!("{}: {}", i + 1, lines[i]));
                }
            }

            let snippet = if !snippet_lines.is_empty() {
                snippet_lines.join("\n")
            } else {
                let mut snippet = take_chars(&content, MAX_FILE_SNIPPET_CHARS).to_string();
                if char_len(&content) > MAX_FILE_SNIPPET_CHARS {
                    snippet.push_str("\n... (truncated)");
                }
                snippet
            };

            let entry = format!("\n--- {} ---\n{}", fpath, snippet);
            let entry_len = char_len(&entry);
            if total_chars + entry_len > MAX_PROJECT_FILES_CHARS {
                break;
            }
            parts.push(entry);
            total_chars += entry_len;
        }

        parts.concat()
    }

    // ================================================================ execution

    /// Explore: URL fetch + collect (if not already done).
    ///
    /// When `vector_context` is already provided (caller ran collect
    /// earlier), this only adds URL fetching and assembles the raw context.
    /// Otherwise it runs [`Explorer::collect_parallel`] internally.
    ///
    /// URL fetching and collect run concurrently when both are needed.
    ///
    /// Returns `(raw_context, citations)` — no LLM synthesis.  The
    /// orchestrator's ready-context formatting handles labelling and the
    /// final streaming LLM produces the user-facing response.
    pub fn explore(
        &self,
        user_message: &str,
        work_dir: &str,
        project_dir: Option<&str>,
        mcp_servers: Option<&[MCPServer]>,
        model_service: Option<Arc<dyn ExplorerSearchService>>,
        vector_context: &str,
        citations: Option<Vec<Map<String, Value>>>,
        enriched: Option<&EnrichedContext>,
        skill_instruction: Option<&str>,
        conversation_history: Option<&[ConversationTurn]>,
        executor: Option<&ProjectBoundExecutor>,
    ) -> (String, Option<Vec<Map<String, Value>>>) {
        let _ = (project_dir, enriched, skill_instruction);
        tracing::info!(target: LOGGER, "[explore] Starting exploration in {}", work_dir);
        // The SLP home is the framework's own state, not a folder to search.
        let ws_dir = Some(work_dir).filter(|d| !crate::paths::in_slp_home(d));

        let urls: Vec<String> = URL_IN_MESSAGE.find_iter(user_message).map(|m| m.as_str().to_string()).collect();
        let need_fetch = !urls.is_empty() && mcp_servers.is_some_and(|s| !s.is_empty());
        let need_collect = vector_context.is_empty();

        let mut vector_context = vector_context.to_string();
        let mut citations = citations;

        // ── Parallel: fetch + collect when both needed ──
        let mut fetch_context = String::new();
        let mut feedback_context = String::new();
        let mut summaries_text = String::new();
        let could_not_fetch = |url: &str| {
            format!(
                "Could not fetch content from {}. The page may require JavaScript or the fetch service is unavailable.",
                url
            )
        };

        if need_fetch && need_collect {
            let owned = if executor.is_none() { Some(DetachedPool::new(4)) } else { None };
            let pool: &ProjectBoundExecutor = match (executor, owned.as_ref()) {
                (Some(pool), _) => pool,
                (None, Some(owned)) => owned.get(),
                (None, None) => unreachable!("an owned pool is built when none is passed"),
            };
            // The fetch runs on the pool while collect fans out from this
            // thread over the same pool.
            let f_fetch = {
                let this = self.clone();
                let url = urls[0].clone();
                pool.submit(move || this.fetch_url(&url, None))
            };
            let search_ctx = self.collect_parallel(
                user_message,
                model_service,
                ws_dir,
                None,
                None,
                None,
                None,
                conversation_history,
                Some(pool),
                None,
                None,
                None,
            );
            fetch_context = join(f_fetch);
            if !search_ctx.vector_context.is_empty() {
                vector_context = search_ctx.vector_context;
                citations = search_ctx.citations;
            }
            feedback_context = search_ctx.feedback_context;
            summaries_text = search_ctx.workspace_context;
            tracing::info!(target: LOGGER, "[explore] Ran fetch + collect_parallel concurrently");

            if fetch_context.is_empty() {
                fetch_context = could_not_fetch(&urls[0]);
            }
        } else {
            // ── Sequential paths (one or neither needed) ──
            if need_fetch {
                fetch_context = self.fetch_url(&urls[0], mcp_servers);
                if fetch_context.is_empty() {
                    fetch_context = could_not_fetch(&urls[0]);
                    tracing::info!(target: LOGGER, "[explore] URL fetch returned empty");
                }
            }

            if need_collect {
                let search_ctx = self.collect_parallel(
                    user_message,
                    model_service,
                    ws_dir,
                    None,
                    None,
                    None,
                    None,
                    conversation_history,
                    executor,
                    None,
                    None,
                    None,
                );
                if !search_ctx.vector_context.is_empty() {
                    vector_context = search_ctx.vector_context;
                    citations = search_ctx.citations;
                }
                feedback_context = search_ctx.feedback_context;
                summaries_text = search_ctx.workspace_context;
                tracing::info!(target: LOGGER, "[explore] Ran collect_parallel() for vector/feedback/workspace");
            } else {
                tracing::info!(target: LOGGER, "[explore] Skipping collect() — vector_context already provided");
            }
        }

        // ── Assemble raw context ──
        let context_parts: Vec<&str> = [&fetch_context, &vector_context, &feedback_context, &summaries_text]
            .into_iter()
            .filter(|p| !p.is_empty())
            .map(String::as_str)
            .collect();

        if context_parts.is_empty() {
            return ("No relevant content found in documents or project files.".to_string(), None);
        }

        let raw = context_parts.join("\n\n---\n\n");
        tracing::info!(
            target: LOGGER,
            "[explore] Returning raw context ({} chars, {} source(s))",
            char_len(&raw),
            context_parts.len()
        );
        (raw, citations)
    }

    /// Read a file the request named, in-process: text as it is, a binary
    /// document (PDF, DOCX, PPTX, CSV) through its parser
    /// ([`FileOperations::parse_binary_file`]).  Returns the content under a
    /// `--- <path> ---` header, or `None` when the file can't be read.
    pub fn read_file(&self, path: &str) -> Option<String> {
        let file = Path::new(path);
        if !file.is_file() {
            return None;
        }
        let ext = splitext_ext(path).to_lowercase();
        let content = if FileOperations::BINARY_EXTENSIONS.contains(&ext.as_str()) {
            let Some(llm) = self.llm.clone() else {
                tracing::warn!(target: LOGGER, "[explore] No model to parse {}", path);
                return None;
            };
            FileOperations::new(llm).parse_binary_file(path)?
        } else {
            match read_text_lossy(file) {
                Ok(content) => content,
                Err(e) => {
                    tracing::warn!(target: LOGGER, "[explore] Could not read {}: {}", path, e);
                    return None;
                }
            }
        };
        let content = if char_len(&content) > MAX_READ_FILE_CHARS {
            format!("{}\n... (truncated)", take_chars(&content, MAX_READ_FILE_CHARS))
        } else {
            content
        };
        tracing::info!(target: LOGGER, "[explore] Read {} ({} chars)", path, char_len(&content));
        Some(format!("--- {} ---\n{}", path, content))
    }

    // ================================================================ fetch

    /// Fetch a URL as markdown via the in-process FetchAgent.
    ///
    /// `mcp_servers` is accepted for backward compatibility but unused —
    /// fetching is no longer an MCP tool call.
    pub fn fetch_url(&self, url: &str, mcp_servers: Option<&[MCPServer]>) -> String {
        let _ = mcp_servers;
        tracing::info!(target: LOGGER, "[explore] Fetching URL via FetchAgent: {}", url);
        self.fetch_agent.fetch(url, usize::MAX, 0, false)
    }

    /// Rewrite fetch-intended tool steps that carry a URL into FETCH steps.
    pub fn resolve_fetch_steps(
        &self,
        steps: Vec<PlanStep>,
        available_servers: &[Map<String, Value>],
        user_message: &str,
    ) -> Vec<PlanStep> {
        let _ = available_servers;
        const FETCH_INTENT_WORDS: [&str; 7] = ["fetch", "browse", "scrape", "navigate", "retrieve", "get url", "http"];
        let extract_urls = |text: &str| -> Vec<String> {
            self.arg_extractor.as_ref().map(|x| x.extract_urls_from_text(text)).unwrap_or_default()
        };

        let mut resolved = Vec::with_capacity(steps.len());
        for step in steps {
            if step.step_type == StepType::Tool {
                let instr_lower = step.instruction.to_lowercase();
                let tool_lower = step.tool_name.as_deref().unwrap_or("").to_lowercase();
                let is_fetch_intended = FETCH_TOOL_KEYWORDS.iter().any(|kw| tool_lower.contains(kw))
                    || FETCH_INTENT_WORDS.iter().any(|kw| instr_lower.contains(kw))
                    || instr_lower.contains("url");
                if is_fetch_intended {
                    let mut urls = extract_urls(user_message);
                    if urls.is_empty() {
                        urls = extract_urls(&step.instruction);
                    }
                    if let Some(url) = urls.first() {
                        tracing::info!(
                            target: LOGGER,
                            "[resolve] Step {}: URL '{}' → FetchAgent step",
                            step.step,
                            url
                        );
                        let mut default_args = std::collections::BTreeMap::new();
                        default_args.insert("url".to_string(), url.clone());
                        resolved.push(PlanStep {
                            step: step.step,
                            step_type: StepType::Fetch,
                            tool_name: Some("fetch".to_string()),
                            server_id: Some(String::new()),
                            server_name: Some(String::new()),
                            instruction: format!(
                                "Fetch {} and extract its main content. Context: {}",
                                url, step.instruction
                            ),
                            default_args: Some(default_args),
                            ..Default::default()
                        });
                        continue;
                    }
                }
            }
            resolved.push(step);
        }
        resolved
    }
}

// ------------------------------------------------------------------ helpers

/// A pool owned by one call.  Python shut it down with `wait=False`, so the
/// caller never waits on stragglers: on drop it stops taking work and is
/// joined from a background thread.
struct DetachedPool(Option<ProjectBoundExecutor>);

impl DetachedPool {
    fn new(max_workers: usize) -> Self {
        Self(Some(ProjectBoundExecutor::new(max_workers, None)))
    }

    fn get(&self) -> &ProjectBoundExecutor {
        self.0.as_ref().expect("pool is live until dropped")
    }
}

impl Drop for DetachedPool {
    fn drop(&mut self) {
        if let Some(pool) = self.0.take() {
            pool.shutdown(false);
            // Joining happens in the pool's own drop; keep it off this thread.
            let _ = std::thread::Builder::new()
                .name("explorer-pool-reaper".to_string())
                .spawn(move || drop(pool));
        }
    }
}

/// `future.result()`: the job's value, re-raising a job's panic.
fn join<R>(future: PoolFuture<R>) -> R {
    match future.result() {
        Ok(value) => value,
        Err(panic) => resume_unwind(panic),
    }
}

/// `llm.invoke(prompt, max_tokens=...)` inside Python's `try/except
/// Exception`: a missing model or a model call that panics is an error.
fn guarded_invoke(llm: Option<&dyn BaseLLM>, prompt: &str, max_tokens: Option<usize>) -> Result<String, String> {
    let Some(llm) = llm else { return Err("no model is available".to_string()) };
    catch_unwind(AssertUnwindSafe(|| llm.invoke(prompt, max_tokens, None))).map_err(|p| panic_message(p.as_ref()))
}

/// The message carried by a panic payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown error".to_string()
    }
}

/// A skill's `name` (`""` when absent).
fn skill_name(skill: &SkillDict) -> &str {
    skill.get("name").map(String::as_str).unwrap_or("")
}

/// `int(token)` for a `-?\d+` match; `None` when it does not fit (or is
/// spelled in non-ASCII digits, which a skill-number reply never is).
fn parse_py_int(token: &str) -> Option<i128> {
    token.parse::<i128>().ok()
}

/// `s[:n]` by characters.
fn take_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

/// `len(s)` in characters.
fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// `.strip("'\"")`.
fn strip_quotes(s: &str) -> &str {
    s.trim_matches(|c| c == '\'' || c == '"')
}

/// Python `repr()` of a string.
fn py_repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python `repr()` of a list of strings (`['a', 'b']`).
fn py_list<'a>(items: impl Iterator<Item = &'a str>) -> String {
    format!("[{}]", items.map(py_repr_str).collect::<Vec<_>>().join(", "))
}

/// Python `repr()` of a JSON value.
fn py_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => py_repr_str(s),
        Value::Array(items) => format!("[{}]", items.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter().map(|(k, v)| format!("{}: {}", py_repr_str(k), py_repr(v))).collect::<Vec<_>>().join(", ")
        ),
    }
}

/// Python `str()` of a JSON value (a string is itself).
fn py_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

/// `str()` of an optional value, `None` for a missing one.
fn py_str_opt(value: Option<&Value>) -> String {
    value.map(py_str).unwrap_or_else(|| "None".to_string())
}

/// Python truthiness of a JSON value.
fn py_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// The text of an MCP result's content blocks, one per line, stripped.
fn content_text(content: &[Value]) -> String {
    content
        .iter()
        .map(|block| match block {
            Value::Object(map) if map.get("type").and_then(Value::as_str) == Some("text") => match map.get("text") {
                None | Some(Value::Null) => String::new(),
                Some(text) => py_str(text),
            },
            other => py_str(other),
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// `os.path.realpath`: the canonical path, or the absolute path when it does
/// not exist.
fn realpath(path: &str) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| {
        let p = Path::new(path);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            std::env::current_dir().map(|d| d.join(p)).unwrap_or_else(|_| p.to_path_buf())
        }
    })
}

/// `os.path.relpath(path, start)` for a `path` under `start`.
fn relpath(path: &Path, start: &Path) -> String {
    match path.strip_prefix(start) {
        Ok(rel) if rel.as_os_str().is_empty() => ".".to_string(),
        Ok(rel) => rel.to_string_lossy().into_owned(),
        Err(_) => path.to_string_lossy().into_owned(),
    }
}

/// `os.path.basename`.
fn basename(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// `os.path.splitext(name)[1]`: the extension with its dot, `""` for none
/// (leading dots do not start an extension: `.bashrc` has none).
fn splitext_ext(name: &str) -> &str {
    let base = name.rsplit('/').next().unwrap_or(name);
    let stem_start = base.len() - base.trim_start_matches('.').len();
    match base[stem_start..].rfind('.') {
        Some(i) => &base[stem_start + i..],
        None => "",
    }
}

/// `open(path, encoding="utf-8", errors="replace").read()`: undecodable bytes
/// become U+FFFD and newlines are universal (`\r\n` / `\r` → `\n`).
fn read_text_lossy(path: &Path) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    let text = String::from_utf8_lossy(&bytes);
    Ok(text.replace("\r\n", "\n").replace('\r', "\n"))
}

/// `os.walk(top, topdown=True)`.  `visit(root, dirs, files)` may prune or
/// reorder `dirs` (the `dirs[:] = ...` idiom) and returns `false` to stop
/// the walk.  Unreadable directories are skipped, symlinked directories are
/// listed but not entered (`followlinks=False`).
fn walk(top: &Path, visit: &mut dyn FnMut(&Path, &mut Vec<String>, &[String]) -> bool) -> bool {
    let Ok(entries) = std::fs::read_dir(top) else { return true };
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    for entry in entries {
        let Ok(entry) = entry else { break };
        let name = entry.file_name().to_string_lossy().into_owned();
        // `entry.is_dir()` follows symlinks.
        if std::fs::metadata(entry.path()).map(|m| m.is_dir()).unwrap_or(false) {
            dirs.push(name);
        } else {
            files.push(name);
        }
    }
    if !visit(top, &mut dirs, &files) {
        return false;
    }
    for dir in &dirs {
        let path = top.join(dir);
        let is_link = std::fs::symlink_metadata(&path).map(|m| m.file_type().is_symlink()).unwrap_or(false);
        if is_link {
            continue;
        }
        if !walk(&path, visit) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tool_citation_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn explore_mcp_context_yields_one_citation_per_call() {
        let ctx = "MCP TOOL (Notion.notion-search):\n{\"results\": []}\n\n\
                   MCP TOOL (my.server.io.get_page):\nbody";
        let cites = mcp_context_citations(ctx);
        assert_eq!(cites.len(), 2);
        assert_eq!(cites[0]["document_id"], json!("mcp:Notion.notion-search"));
        assert_eq!(cites[0]["document_name"], json!("Notion · notion-search"));
        // Split on the last dot: server names may contain one, tools don't.
        assert_eq!(cites[1]["server_name"], json!("my.server.io"));
        assert_eq!(cites[1]["tool_name"], json!("get_page"));
    }

    #[test]
    fn plain_text_has_no_tool_citations() {
        assert!(mcp_context_citations("").is_empty());
        assert!(mcp_context_citations("just some text\nMCP TOOL mention").is_empty());
    }
}
