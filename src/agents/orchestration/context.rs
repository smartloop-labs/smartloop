//! Port of `smartloop/agents/orchestration/context.py`.
//!
//! Per-request context and result types shared across orchestration
//! components.
//!
//! These live in their own module so every component can import them without
//! importing the orchestrator itself — the orchestrator depends on all of the
//! components, so any component importing it back would be a cycle.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agents::mcp::explorer::EnrichedContext;
use crate::agents::planner::MCPToolContext;
use crate::api::services::model_service::ModelService;
use crate::llms::base_llm::CancelEvent;
use crate::workspace::workspace::MCPServer;

/// Shared parameters passed to DAG handler methods (Python
/// `_OrchestratorContext`).
#[derive(Clone, Default)]
pub struct OrchestratorContext {
    pub user_message: String,
    /// Every registered MCP connection for the request.
    pub mcp_servers: Vec<MCPServer>,
    /// Planner-facing catalogue of usable servers (the dicts built by
    /// [`crate::agents::orchestration::step_executor::StepExecutor::build_available_servers`]).
    pub available_servers: Vec<Value>,
    pub effective_cwd: String,
    /// Raw skill file contents.
    pub skills: Vec<String>,
    /// Skill the user explicitly asked for (e.g. the API resolved a leading
    /// `/fund-info` slash-command to a skill name). When set, the request is
    /// fulfilled with this skill directly, skipping the LLM skill-match guess.
    pub requested_skill: Option<String>,
    pub document_context: Option<String>,
    pub project_dir: Option<String>,
    pub model_service: Option<Arc<ModelService>>,
    pub conversation_history: Option<Vec<HashMap<String, String>>>,
    pub has_attachments: bool,
    /// Built once during context construction.
    pub enriched: Option<EnrichedContext>,
    /// Populated by handlers as they run.
    pub mcp_tool_contexts: Vec<MCPToolContext>,
    /// Parsed skill records (`{"name", "description", "content", "skill_map"}`).
    pub parsed_skills: Vec<Value>,
    /// Stripped skill body text (set by route handler when skills match).
    /// `skill_context` is the joined form (used downstream for instruction
    /// enrichment); `skill_bodies` keeps each matched skill separate so the
    /// planner can plan across multiple matched skills.
    pub skill_context: Option<String>,
    pub skill_bodies: Vec<String>,
    pub available_capabilities: Vec<String>,

    // ── Per-request scratch state ──
    //
    // Written by one component and read by another (an ANALYZE step writes
    // `skill_report`, the response builder reads it), so it belongs on the
    // object already threaded through every handler rather than on whichever
    // component happens to run first.
    /// The client's wall-clock date ("Month DD, YYYY"), used to ground
    /// date-aware prompts in the user's clock instead of the server's.
    pub current_date: Option<String>,
    /// An ANALYZE step's findings, surfaced ahead of raw context in the answer.
    pub skill_report: Option<String>,
    /// Citations from the documents the explore step actually drew on.
    pub explore_citations: Option<Vec<Value>>,
    /// The capability the planner classified the request as (e.g. coding),
    /// carried back so the answer is written by a model declaring it.
    pub request_capability: Option<String>,
    /// Whether the planner read the request as programming work, which is
    /// answered on the document-processing model whatever else was found.
    pub request_coding: bool,
    /// Whether the request brings the files it's about, so the documents
    /// found for it needn't be extracted from.
    pub request_carries_files: bool,
    /// The request's cancellation signal, set when the client goes away.
    pub cancelled: Option<CancelEvent>,
}

impl OrchestratorContext {
    /// True once the client has gone away.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.as_ref().is_some_and(|c| c.load(std::sync::atomic::Ordering::SeqCst))
    }
}

impl OrchestratorContext {
    /// Build a context with the four required fields; everything else takes
    /// its default.
    pub fn new(
        user_message: impl Into<String>,
        mcp_servers: Vec<MCPServer>,
        available_servers: Vec<Value>,
        effective_cwd: impl Into<String>,
    ) -> Self {
        Self {
            user_message: user_message.into(),
            mcp_servers,
            available_servers,
            effective_cwd: effective_cwd.into(),
            ..Default::default()
        }
    }
}

/// The context as threaded through DAG handlers: handlers are closures that
/// outlive `build`, and some of them write scratch state back (Python mutated
/// the shared pydantic object in place).
pub type SharedOrchestratorContext = Arc<RwLock<OrchestratorContext>>;

/// Result of MCP tool orchestration.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MCPExecutionResult {
    /// Whether the message was handled by an MCP tool
    pub handled: bool,
    /// Formatted result content
    #[serde(default)]
    pub content: Option<String>,
    /// Matched skill instruction to follow
    #[serde(default)]
    pub skill_instruction: Option<String>,
    /// Name of the tool that was executed
    #[serde(default)]
    pub tool_name: Option<String>,
    /// Name of the MCP server used
    #[serde(default)]
    pub server_name: Option<String>,
    /// Error message if execution failed
    #[serde(default)]
    pub error: Option<String>,
    /// All tools executed in the chain
    #[serde(default)]
    pub tool_names: Vec<String>,
    /// Source documents cited
    #[serde(default)]
    pub citations: Option<Vec<Value>>,
    /// Whether this was a mutation (write) operation — skip LLM, return
    /// content directly
    #[serde(default)]
    pub mutation: bool,
    /// The capability the planner classified the request as (e.g. dp)
    #[serde(default)]
    pub capability: Option<String>,
    /// Whether the planner read the request as programming work.
    #[serde(default)]
    pub coding: bool,
}
