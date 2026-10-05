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

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::agents::fetch_agent::FetchAgent;
use crate::agents::mcp::argument_extractor::MCPArgumentExtractor;
#[allow(unused_imports)]
use crate::agents::mcp::mcp_agent::{
    format_call, is_entitlement_notice, mark_tool_unavailable, missing_required, MCPAgent,
    ResolvedCall,
};
use crate::agents::mcp::mcp_tool_caller::MCPToolCaller;
#[allow(unused_imports)]
use crate::agents::mcp::server_selector::{
    BASH_TOOL, EXPLORE_TOOL, FILE_READ_TOOL, FILE_WRITE_TOOL, GREP_TOOL,
};
use crate::agents::orchestration::context::MCPExecutionResult;
#[allow(unused_imports)]
use crate::agents::planner::{
    mcp_connection_is_usable, MCPToolContext, PlanStep, StepType, SYSTEM_TOOLS_SERVER_ID,
};
use crate::agents::web_search_agent::WebSearchAgent;
use crate::api::services::model_service::ModelService;
use crate::llms::BaseLLM;
use crate::parsers::DocumentChunk;
use crate::workspace::context::ProjectBoundExecutor;
use crate::workspace::workspace::MCPServer;

/// Logger name used by the Python module.
pub const LOGGER: &str = "smartloop.agents.mcp.explorer";

// ------------------------------------------------------------------ constants

/// Tool-name fragments that mark a planned tool step as a URL fetch.
pub const FETCH_TOOL_KEYWORDS: &[&str] =
    &["fetch", "browse", "scrape", "web", "http", "navigate", "get_url"];

/// File extensions `explore_and_read` considers.
pub const EXPLORE_EXTENSIONS: &[&str] = &[
    ".yaml", ".yml", ".toml", ".json", ".cfg", ".ini", ".md", ".txt", ".rst", ".html", ".css",
    ".sql", ".proto", ".graphql", ".sh",
];
/// Directory names `explore_and_read` never descends into.
pub const EXPLORE_SKIP_DIRS: &[&str] = &[
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".mypy_cache",
    ".pytest_cache", "dist", "build", ".eggs", "*.egg-info", ".out",
];
pub const MAX_EXPLORE_FILES: usize = 50;
pub const MAX_FILE_CHARS: usize = 2000;

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
    "lookup", "retrieve", "count", "check", "status", "browse", "scrape",
];
/// Verbs that mark a tool as side-effecting (never auto-run in collect).
pub const WRITE_VERBS: &[&str] = &[
    "create", "update", "delete", "send", "post", "write", "remove", "set", "add", "edit",
    "modify", "put", "patch", "insert", "cancel", "pay", "deploy", "merge", "close", "assign",
    "run", "execute", "trigger",
];

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
}

impl ToolResolver for MCPAgent {
    fn resolve(
        &self,
        directive: &str,
        mcp_servers: &[MCPServer],
        context: Option<&str>,
        llm: Option<&dyn BaseLLM>,
    ) -> Option<ResolvedCall> {
        let _ = (directive, mcp_servers, context, llm);
        todo!()
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
        todo!()
    }
    fn search(&self, query: &str, progress: Option<ProgressCallback>) -> anyhow::Result<String> {
        let _ = (query, progress);
        todo!()
    }
    fn last_sources(&self) -> Vec<String> {
        todo!()
    }
    fn last_source_extracts(&self) -> HashMap<String, String> {
        todo!()
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
    fn search_workspace(
        &self,
        query: &str,
        work_dir: &str,
        n_results: usize,
        conversation_history: Option<&[ConversationTurn]>,
    ) -> anyhow::Result<Vec<DocumentChunk>>;
}

impl ExplorerSearchService for ModelService {
    fn get_document_groups(
        &self,
        query: Option<&str>,
        skill_instruction: Option<&str>,
        conversation_history: Option<&[ConversationTurn]>,
    ) -> anyhow::Result<Option<DocumentGroups>> {
        let _ = (query, skill_instruction, conversation_history);
        todo!()
    }
    fn search_feedback(&self, query: &str, n_results: usize) -> anyhow::Result<Vec<DocumentChunk>> {
        let _ = (query, n_results);
        todo!()
    }
    fn search_workspace(
        &self,
        query: &str,
        work_dir: &str,
        n_results: usize,
        conversation_history: Option<&[ConversationTurn]>,
    ) -> anyhow::Result<Vec<DocumentChunk>> {
        let _ = (query, work_dir, n_results, conversation_history);
        todo!()
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

impl SearchContext {
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

impl Explorer {
    /// How many transcript turns the follow-up gate sees.
    pub const GATE_TURNS: usize = 4;
    pub const GATE_TURN_CHARS: usize = 400;
    pub const GATE_MAX_TOKENS: usize = 8;
    /// How many connected servers are asked for one question.
    pub const MAX_SERVERS_PER_QUESTION: usize = 4;

    /// Build an explorer with its fetch agent, web search agent and MCP agent.
    pub fn new(
        llm: Arc<dyn BaseLLM>,
        tool_caller: Option<Arc<MCPToolCaller>>,
        arg_extractor: Option<Arc<MCPArgumentExtractor>>,
        use_llm_for_history: bool,
    ) -> Self {
        let _ = (llm, tool_caller, arg_extractor, use_llm_for_history);
        todo!()
    }

    // ================================================================ skills

    /// Match a user message to skills, unioning the LLM qualifier with the
    /// deterministic direct-name match; `__`-prefixed skills never match.
    /// `llm` overrides the model used for the semantic qualifier.
    pub fn match_skills(
        &self,
        user_message: &str,
        parsed_skills: &[SkillDict],
        llm: Option<&dyn BaseLLM>,
    ) -> Vec<SkillDict> {
        let _ = (user_message, parsed_skills, llm);
        todo!()
    }

    /// Ask the LLM which skills apply — the primary, semantic matcher.
    pub fn llm_match_skills(
        &self,
        user_message: &str,
        matchable: &[SkillDict],
        llm: Option<&dyn BaseLLM>,
    ) -> Vec<SkillDict> {
        let _ = (user_message, matchable, llm);
        todo!()
    }

    /// Match skills the user names explicitly (whole-phrase, word-boundary).
    pub fn direct_name_matches(user_message: &str, skills: &[SkillDict]) -> Vec<SkillDict> {
        let _ = (user_message, skills);
        todo!()
    }

    // ================================================================ searches

    /// Vector/document search → `(text, citations, documents)`, or `None`.
    pub fn search_vector(
        query: &str,
        model_service: &dyn ExplorerSearchService,
        skill_instruction: Option<&str>,
        conversation_history: Option<&[ConversationTurn]>,
    ) -> Option<(String, Vec<Map<String, Value>>, Vec<Map<String, Value>>)> {
        let _ = (query, model_service, skill_instruction, conversation_history);
        todo!()
    }

    /// Feedback Q&A search.
    pub fn search_feedback(query: &str, model_service: &dyn ExplorerSearchService) -> String {
        let _ = (query, model_service);
        todo!()
    }

    /// True when `query` needs fresh web results rather than the transcript.
    /// Fails open (searches) on any LLM error or unreadable verdict.
    pub fn needs_web_search(
        &self,
        query: &str,
        conversation_history: Option<&[ConversationTurn]>,
    ) -> bool {
        let _ = (query, conversation_history);
        todo!()
    }

    /// Hosted web search (Pro); any failure is a silent `""`.
    pub fn search_web(&self, query: &str, progress: Option<ProgressCallback>) -> String {
        let _ = (query, progress);
        todo!()
    }

    /// True only when a tool name clearly reads (no side effects).
    pub fn is_read_only_tool(tool_name: &str) -> bool {
        let _ = tool_name;
        todo!()
    }

    /// Report one step to the client, never failing the work over it.
    pub fn emit(progress: Option<&ProgressCallback>, step: &str, message: &str) {
        let _ = (progress, step, message);
        todo!()
    }

    /// Gather what the connected servers know about `query` (read-only tools,
    /// at most [`Self::MAX_SERVERS_PER_QUESTION`] servers).
    pub fn run_mcp(
        &self,
        query: &str,
        mcp_servers: Option<&[MCPServer]>,
        project_dir: Option<&str>,
        llm: Option<Arc<dyn BaseLLM>>,
        progress: Option<ProgressCallback>,
    ) -> String {
        let _ = (query, mcp_servers, project_dir, llm, progress);
        todo!()
    }

    /// One server's answer to `query`, or `""`.
    pub fn run_mcp_server(
        &self,
        query: &str,
        server: &MCPServer,
        llm: Option<Arc<dyn BaseLLM>>,
        progress: Option<ProgressCallback>,
    ) -> String {
        let _ = (query, server, llm, progress);
        todo!()
    }

    /// Run one read-only tool and return its text, or `""`.
    pub fn call_mcp_tool(
        &self,
        server: &MCPServer,
        tool_name: &str,
        arguments: &Map<String, Value>,
        query: &str,
    ) -> String {
        let _ = (server, tool_name, arguments, query);
        todo!()
    }

    /// Bodies behind the records `payload` lists, per the connection's plan.
    pub fn fetch_record_details(
        &self,
        server: &MCPServer,
        tool_name: &str,
        payload: &str,
        progress: Option<ProgressCallback>,
    ) -> String {
        let _ = (server, tool_name, payload, progress);
        todo!()
    }

    /// True for skill/system files that are instructions or tool metadata.
    pub fn is_non_data_file(fname: &str) -> bool {
        let _ = fname;
        todo!()
    }

    /// Workspace file search (best chunk per file).
    pub fn search_workspace(
        query: &str,
        model_service: &dyn ExplorerSearchService,
        cwd: &str,
        conversation_history: Option<&[ConversationTurn]>,
    ) -> String {
        let _ = (query, model_service, cwd, conversation_history);
        todo!()
    }

    /// Fan-out vector/feedback/workspace/MCP searches in parallel, then the
    /// web search (announced via `progress` before it starts).
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
        let _ = (
            query,
            model_service,
            cwd,
            enriched,
            matched_skills,
            mcp_servers,
            mcp_tool_contexts,
            conversation_history,
            executor,
            project_dir,
            progress,
            mcp_llm,
        );
        todo!()
    }

    // ================================================================ enrichment

    /// Pre-planning enrichment: MCP catalogue and history insight.
    pub fn enrich(
        &self,
        available_mcp_servers: Option<&[Map<String, Value>]>,
        conversation_history: Option<&[ConversationTurn]>,
    ) -> EnrichedContext {
        let _ = (available_mcp_servers, conversation_history);
        todo!()
    }

    /// LLM route analysis (SKILL / DOCUMENT / TOOL / CONVERSATIONAL); `""` on failure.
    pub fn analyze(
        &self,
        user_message: &str,
        enriched: &EnrichedContext,
        conversation_history: Option<&[ConversationTurn]>,
        skills: Option<&[String]>,
    ) -> String {
        let _ = (user_message, enriched, conversation_history, skills);
        todo!()
    }

    /// Read `SKILL.md` / `*.md` skill files under `skills_dir`.
    pub fn scan_skills(&self, skills_dir: &str) -> Vec<String> {
        let _ = skills_dir;
        todo!()
    }

    /// A short indented tree of `cwd`.
    pub fn scan_workspace(&self, cwd: &str) -> String {
        let _ = cwd;
        todo!()
    }

    /// `server: tool1, tool2, …` per server with tools.
    pub fn catalogue_mcp(&self, servers: &[Map<String, Value>]) -> String {
        let _ = servers;
        todo!()
    }

    /// Render history turns as `role: content` lines.
    pub fn format_history(history: &[ConversationTurn]) -> String {
        let _ = history;
        todo!()
    }

    /// Query-matched snippets from project source files under `cwd`.
    pub fn scan_project_files(&self, cwd: &str, user_message: &str) -> String {
        let _ = (cwd, user_message);
        todo!()
    }

    // ================================================================ execution

    /// Explore: URL fetch + collect (if not already done).  Returns
    /// `(raw_context, citations)` — no LLM synthesis.
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
        let _ = (
            user_message,
            work_dir,
            project_dir,
            mcp_servers,
            model_service,
            vector_context,
            citations,
            enriched,
            skill_instruction,
            conversation_history,
            executor,
        );
        todo!()
    }

    /// Explore -> pick best file -> read -> return content.
    pub fn explore_and_read(&self, user_message: &str, cwd: &str) -> MCPExecutionResult {
        let _ = (user_message, cwd);
        todo!()
    }

    // ================================================================ fetch

    /// Fetch a URL as markdown via the in-process FetchAgent.  `mcp_servers`
    /// is accepted for backward compatibility but unused.
    pub fn fetch_url(&self, url: &str, mcp_servers: Option<&[MCPServer]>) -> String {
        let _ = (url, mcp_servers);
        todo!()
    }

    /// Rewrite fetch-intended tool steps that carry a URL into FETCH steps.
    pub fn resolve_fetch_steps(
        &self,
        steps: Vec<PlanStep>,
        available_servers: &[Map<String, Value>],
        user_message: &str,
    ) -> Vec<PlanStep> {
        let _ = (steps, available_servers, user_message);
        todo!()
    }
}
