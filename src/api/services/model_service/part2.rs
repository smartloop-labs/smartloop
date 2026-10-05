//! Second half of `smartloop/api/services/model_service.py` (from
//! `_llm_for_response` onward): response-model routing and generation,
//! documents, projects, model unloading, MCP server registration and
//! provisioning, orchestration, auto-write, assets and vision Q&A.
//!
//! The `ModelService` struct, its fields and the first half of its methods
//! live in the parent module (`super`).

#![allow(unused_imports)]

use super::*;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{Map, Value};

use crate::agents::agent::SharedLLM;
use crate::agents::hybrid_executor::LLMExecutor;
use crate::agents::orchestration::model_router::ProgressCallback;
use crate::llms::base_llm::{CancelEvent, GenerationChunk};
use crate::model::downloader::slp_downloader::ProgressCallback as DownloadProgressCallback;
use crate::model::slp_model::DocumentProgressCallback;
use crate::tasks::{DownloaderTask, MCPTask};
use crate::workspace::workspace::{MCPServer, Project, SharedWorkspace};
use crate::workspace::{DocumentInfo, Skill};

/// Callback fired with the name of the vision model chosen for a request.
pub type ModelSelectedCallback = Arc<dyn Fn(&str) + Send + Sync>;

/// `orchestrate` result:
/// `(context, matched, error, citations, skill_instruction, mutation)`.
///
/// * `context` — the formatted tool output;
/// * `matched` — true when the orchestrator produced content;
/// * `error` — a user-facing error message (or `None` on success);
/// * `citations` — source document dicts (or `None`);
/// * `skill_instruction` — the matched skill content (or `None`);
/// * `mutation` — the router should stream `context` verbatim instead of
///   running it through another LLM.
pub type OrchestrateOutcome = (
    Option<String>,
    bool,
    Option<String>,
    Option<Vec<Value>>,
    Option<String>,
    bool,
);

/// Signals that an image-attachment request wants to *use* indexed documents
/// (cross-validate / answer with them), not just read the image. When none of
/// these appear, an attached image is treated as the sole source and the
/// retrieved documents are discarded so they can't mis-attribute its contents.
pub static IMAGE_CROSS_REF_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concat!(
        r"(?i)\b(compare|comparison|match(?:es|ing)?|validate|verif(?:y|ies|ication)|",
        r"cross[-\s]?(?:check|validat\w*|referenc\w*)|reconcile|consistent|discrepanc\w*|",
        r"recommend|suggest|against (?:our|the|my)|based on (?:our|the|my)|",
        r"using (?:our|the|my)|according to (?:our|the|my)|",
        r"(?:in|from|with) (?:our|the|my|these|those) (?:record|document|doc|file|data|report)s?)\b",
    ))
    .expect("valid IMAGE_CROSS_REF_RE")
});

/// Inline write intent in a user message ("create", "write", "export", ...).
pub static WRITE_INTENT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(create|generate|write|save|export|produce|make|build|draft)\b")
        .expect("valid WRITE_INTENT_RE")
});

/// Matches bare "write" / "save" / "write it" / "save that" commands.
pub static BARE_WRITE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)^\s*(write|save|write\s+(it|that|this)|save\s+(it|that|this))\s*[.!]?\s*$")
        .expect("valid BARE_WRITE_RE")
});

/// Conversational preamble ("Sure, here's ...") stripped before auto-write.
pub static PREAMBLE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concat!(
        r"(?i)^(?:(?:sure|okay|of course|absolutely|certainly|here(?:'s| is| are| you go))",
        r"[^\n]*\n+)+",
    ))
    .expect("valid PREAMBLE_RE")
});

/// Conversational postamble ("Let me know ...") stripped before auto-write.
pub static POSTAMBLE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concat!(
        r"(?i)\n+(?:(?:i hope|let me know|feel free|enjoy|bon appétit|happy|hope this|",
        r"if you (?:have|need|want)|don't hesitate|",
        r"would you like|do you want|shall i|",
        r"you (?:can|could|may|might) (?:also|adjust|modify|change|customize|tweak)|",
        r"this (?:can be|could be|is just|is a) (?:adjusted|modified|customized|starting)|",
        r"note:|please note|disclaimer|tips?:)[^\n]*\n*)+\s*$",
    ))
    .expect("valid POSTAMBLE_RE")
});

/// System-message steering for direct vision Q&A.
pub const VISION_QA_GUIDANCE: &str = "Answer using only what is actually visible in the image. If the answer \
is not visible, say so plainly instead of guessing. Do not invent \
people, text, or details that are not there.";

/// Perception prompt for the orchestrator's image path; `{request}` is
/// replaced with the user's message.  Asks the vision model only to describe
/// what it sees, scoped to the request, because the perception is fed back
/// into the text pipeline which does the actual answering.
pub const VISION_PERCEPTION_PROMPT: &str = "Describe what is visible in this image, focusing on the details that \
are relevant to the following request. Be concrete and specific about \
colors, objects, and any visible text. Describe only what is visible; \
do not answer the request or make recommendations.\n\nRequest: {request}";

/// RAII guard for `ModelService::inference_slot` (Python `@contextmanager
/// _inference_slot`): holds one inference permit and counts the model as in
/// use; on drop it decrements the active count and stamps the last-inference
/// time read by `idle_seconds`.
pub struct InferenceSlot<'a> {
    pub(crate) service: &'a ModelService,
}

impl Drop for InferenceSlot<'_> {
    fn drop(&mut self) {
        let _ = self.service;
    }
}

impl ModelService {
    // ------------------------------------------------------ response routing

    /// Return the LLM to use for the final response.
    ///
    /// With exactly one model selected, that model writes the response (falling
    /// back to the default model when it cannot be loaded).  Otherwise, when
    /// `capability` is set: capability-tagged model → largest downloaded model
    /// → default orchestrator LLM.
    pub fn llm_for_response(&self, capability: Option<&str>) -> Option<SharedLLM> {
        let _ = capability;
        todo!()
    }

    /// The default router LLM to fall back to when a larger response model
    /// fails during generation, or `None` when the failed model already IS the
    /// default.
    pub fn default_fallback_llm(&self, failed_llm: &SharedLLM) -> Option<SharedLLM> {
        let _ = failed_llm;
        todo!()
    }

    /// Stop routing responses to a model that faulted during generation
    /// (e.g. GPU OOM): drops the large model (latching the attempt) or marks
    /// the capability slot as GPU-faulted.
    pub fn evict_response_llm(&self, llm: &SharedLLM) {
        let _ = llm;
        todo!()
    }

    /// Stream completion from the response LLM, holding an inference slot.
    ///
    /// `model_name` is unused (caller compatibility).  `cancelled` is checked
    /// between chunks.  When the response model fails before emitting anything
    /// the request is retried on the default model.  Errors with
    /// `"Model not loaded. Call load_model() first."` when no model is loaded.
    pub fn stream_completion<'a>(
        &'a self,
        prompt: &str,
        model_name: Option<&str>,
        cancelled: Option<CancelEvent>,
        capability: Option<&str>,
        thinking: Option<bool>,
    ) -> anyhow::Result<Box<dyn Iterator<Item = anyhow::Result<GenerationChunk>> + 'a>> {
        let _ = (prompt, model_name, cancelled, capability, thinking);
        todo!()
    }

    /// Generate a non-streaming completion on the response LLM, falling back
    /// to the default model when a larger one fails.
    pub fn generate_completion(
        &self,
        prompt: &str,
        model_name: Option<&str>,
        capability: Option<&str>,
        max_tokens: Option<usize>,
        thinking: Option<bool>,
    ) -> anyhow::Result<String> {
        let _ = (prompt, model_name, capability, max_tokens, thinking);
        todo!()
    }

    // --------------------------------------------------------------- documents

    /// Add a document source (file path or URL) to a project (by ID or
    /// current), binding the project for the call.  Returns `{"added": [...]}`.
    pub fn add_document(
        &self,
        source: &str,
        project_id: Option<&str>,
        progress_callback: Option<DocumentProgressCallback>,
    ) -> anyhow::Result<Value> {
        let _ = (source, project_id, progress_callback);
        todo!()
    }

    /// Register a document with PENDING status, enqueue it on the indexer and
    /// return immediately (`{"status": "pending"|"skipped", ...}`).
    pub fn add_document_async(
        &self,
        source: &str,
        project_id: Option<&str>,
        cwd: Option<&str>,
        generate_filename: bool,
    ) -> anyhow::Result<Value> {
        let _ = (source, project_id, cwd, generate_filename);
        todo!()
    }

    /// Drop an uploaded file into the project's documents inbox folder
    /// (de-duplicating the name).  Returns `{"status": "accepted", ...}`.
    pub fn upload_document_to_inbox(
        &self,
        project_id: &str,
        filename: &str,
        file_bytes: &[u8],
        cwd: Option<&str>,
    ) -> anyhow::Result<Value> {
        let _ = (project_id, filename, file_bytes, cwd);
        todo!()
    }

    /// Fresh status for each document ID, re-reading the workspace from disk;
    /// unknown IDs get status `"unknown"`.
    pub fn get_document_statuses(
        &self,
        doc_ids: &[String],
        project_id: Option<&str>,
    ) -> anyhow::Result<Vec<DocumentInfo>> {
        let _ = (doc_ids, project_id);
        todo!()
    }

    /// Register (once) and return the downloader task.
    pub fn ensure_downloader_task(
        &self,
        model_id: &str,
        progress_callback: Option<DownloadProgressCallback>,
    ) -> anyhow::Result<Arc<DownloaderTask>> {
        let _ = (model_id, progress_callback);
        todo!()
    }

    /// Set up all background tasks for the workspace (Chroma migration,
    /// indexer, MCP, build, train) and start the auto tasks.  Called during
    /// API server startup.
    pub fn setup_tasks(&self, workspace: Option<SharedWorkspace>) -> anyhow::Result<()> {
        let _ = workspace;
        todo!()
    }

    /// Gracefully shut down all tasks.  Called during API server shutdown.
    pub fn shutdown_tasks(&self) {
        todo!()
    }

    /// Delete a document by ID from a project (vector store, parquet, source
    /// file, manifest, config).
    pub fn delete_document(&self, document_id: &str, project_id: Option<&str>) -> anyhow::Result<Value> {
        let _ = (document_id, project_id);
        todo!()
    }

    /// List all documents in the current project.
    pub fn get_documents(&self) -> anyhow::Result<Vec<Value>> {
        todo!()
    }

    /// List documents for a project: files in its `documents/` folder enriched
    /// with workspace status (unregistered files appear as pending, `id: null`),
    /// followed by registered documents not on disk.
    pub fn get_documents_for_project(&self, project_id: &str) -> anyhow::Result<Vec<Value>> {
        let _ = project_id;
        todo!()
    }

    // ---------------------------------------------------------------- projects

    /// Create a new project with the given name and description.
    pub fn create_project(
        &self,
        name: &str,
        description: Option<&str>,
        system: bool,
        skills: Option<Vec<Skill>>,
        cwd: Option<&str>,
    ) -> anyhow::Result<Value> {
        let _ = (name, description, system, skills, cwd);
        todo!()
    }

    /// Update an existing project (label, description, skills on disk,
    /// web-search toggle).
    pub fn update_project(
        &self,
        project_id: &str,
        name: Option<&str>,
        description: Option<&str>,
        skills: Option<Vec<Skill>>,
        web_search_enabled: Option<bool>,
    ) -> anyhow::Result<Value> {
        let _ = (project_id, name, description, skills, web_search_enabled);
        todo!()
    }

    /// Convert a Project to a dict for API responses.
    pub fn project_to_dict(&self, p: &Project) -> Value {
        let _ = p;
        todo!()
    }

    /// List all projects in the workspace.
    pub fn list_projects(&self) -> anyhow::Result<Vec<Value>> {
        todo!()
    }

    /// Delete a project (system projects cannot be deleted).
    pub fn delete_project(&self, project_id: &str) -> anyhow::Result<Value> {
        let _ = project_id;
        todo!()
    }

    /// Mark *project_id* as the workspace's current project.
    pub fn set_current_project(&self, project_id: &str) -> anyhow::Result<()> {
        let _ = project_id;
        todo!()
    }

    // ---------------------------------------------------------- model lifecycle

    /// Release one LLM's native resources.
    pub fn free_llm(&self, llm: SharedLLM) {
        let _ = llm;
        todo!()
    }

    /// Record that *model_name* finished downloading in the workspace.
    pub fn mark_model_downloaded(&self, model_name: &str, workspace: Option<SharedWorkspace>) -> anyhow::Result<()> {
        let _ = (model_name, workspace);
        todo!()
    }

    /// Run garbage collection / release cached GPU memory.
    pub fn cleanup_gpu_cache(&self) {
        todo!()
    }

    /// Unload the main model (and everything hanging off it).
    pub fn unload_model(&self) {
        todo!()
    }

    /// Release the on-demand vision/OCR model and return its VRAM.
    pub fn unload_vision_model(&self) {
        todo!()
    }

    /// Release the on-demand large agent/skills model and return its VRAM.
    pub fn unload_large_model(&self) {
        todo!()
    }

    /// Unload cached capability models except *keep_model_name*; returns true
    /// when anything was freed.
    pub fn evict_capability_models(&self, keep_model_name: Option<&str>) -> bool {
        let _ = keep_model_name;
        todo!()
    }

    /// Release the embedding model.
    pub fn unload_embedding_model(&self) {
        todo!()
    }

    /// Release every model except the main orchestrator model (optionally
    /// keeping the embedding model).
    pub fn unload_auxiliary_models(&self, include_embedding: bool) {
        let _ = include_embedding;
        todo!()
    }

    /// Unload whichever loaded slot holds *model_name* (takes the inference
    /// lock); returns true when something was unloaded.
    pub fn unload_model_by_name(&self, model_name: &str) -> bool {
        let _ = model_name;
        todo!()
    }

    /// [`Self::unload_model_by_name`] without taking the inference lock.
    pub fn unload_model_by_name_unlocked(&self, model_name: &str) -> bool {
        let _ = model_name;
        todo!()
    }

    // ------------------------------------------------------------- MCP servers

    /// Register a remote (HTTP) MCP server and queue its provisioning.
    #[allow(clippy::too_many_arguments)]
    pub fn add_remote_mcp_server(
        &self,
        name: &str,
        server_url: &str,
        auth_type: &str,
        tools: Option<Vec<Value>>,
        token: Option<&str>,
        token_endpoint: Option<&str>,
        authorization_endpoint: Option<&str>,
        resource: Option<&str>,
        client_id: Option<&str>,
        client_secret: Option<&str>,
        access_token: Option<&str>,
        refresh_token: Option<&str>,
        expires_at: Option<f64>,
        token_type: &str,
        scope: Option<&str>,
        project_id: Option<&str>,
        server_info: Option<Map<String, Value>>,
        description: Option<&str>,
        tags: Option<Vec<String>>,
        catalog_slug: Option<&str>,
        pkg: Option<&str>,
    ) -> anyhow::Result<Value> {
        let _ = (
            name, server_url, auth_type, tools, token, token_endpoint, authorization_endpoint,
            resource, client_id, client_secret, access_token, refresh_token, expires_at,
            token_type, scope, project_id, server_info, description, tags, catalog_slug, pkg,
        );
        todo!()
    }

    /// True when a launch failure message indicates an MCP 2.x incompatibility.
    pub fn mcp2_incompat_error(msg: &str) -> bool {
        let _ = msg;
        todo!()
    }

    /// Rewrite `npx`-style args to pin a 1.x release of the package, or
    /// `None` when the command can't be pinned.
    pub fn pin_mcp1x_args(command: &str, args: &[String]) -> Option<Vec<String>> {
        let _ = (command, args);
        todo!()
    }

    /// Register a local (stdio) MCP server and queue its provisioning.
    #[allow(clippy::too_many_arguments)]
    pub fn add_local_mcp_server(
        &self,
        command: &str,
        args: Option<Vec<String>>,
        name: Option<&str>,
        env: Option<HashMap<String, String>>,
        cwd: Option<&str>,
        description: Option<&str>,
        tags: Option<Vec<String>>,
        catalog_slug: Option<&str>,
        project_id: Option<&str>,
    ) -> anyhow::Result<Value> {
        let _ = (command, args, name, env, cwd, description, tags, catalog_slug, project_id);
        todo!()
    }

    /// Launch a local server, list its tools and fill in its config/info.
    /// Errors with `MCPLaunchError` / `MCPDiscoveryError`.
    pub fn discover_local_mcp_tools(&self, server: &mut MCPServer) -> anyhow::Result<()> {
        let _ = server;
        todo!()
    }

    /// The registered MCP background task, if any.
    pub fn mcp_task(&self) -> Option<Arc<MCPTask>> {
        todo!()
    }

    /// Hand a newly registered server to the MCP task for provisioning.
    pub fn queue_mcp_provisioning(&self, project: &Project, server: &MCPServer) {
        let _ = (project, server);
        todo!()
    }

    /// `(project_id, server_id)` for PENDING servers this process may
    /// provision (its own pinned project only; nothing when delegated).
    pub fn pending_mcp_servers(&self) -> anyhow::Result<Vec<(String, String)>> {
        todo!()
    }

    /// Discover and enrich one registered server; returns
    /// `{"status": "ready"|"failed"|"skipped", ...}`.
    pub fn provision_mcp_server(&self, project_id: &str, server_id: &str) -> anyhow::Result<Value> {
        let _ = (project_id, server_id);
        todo!()
    }

    /// Probe the account behind an authenticated server (best effort).
    pub fn probe_mcp_account(&self, server: &mut MCPServer) {
        let _ = server;
        todo!()
    }

    /// Write one server's provisioning result back to its project on disk.
    pub fn persist_mcp_server(&self, server: &MCPServer) -> anyhow::Result<()> {
        let _ = server;
        todo!()
    }

    /// Connect to a server and record its tools/info.  Errors with
    /// `MCPDiscoveryError` / `MCPAuthExpiredError`.
    pub fn discover_mcp_tools(&self, server: &mut MCPServer) -> anyhow::Result<()> {
        let _ = server;
        todo!()
    }

    /// Hold an inference slot and record that the model is in use until the
    /// returned guard drops.
    pub fn inference_slot(&self) -> InferenceSlot<'_> {
        todo!()
    }

    /// Seconds since the model last finished answering; `0.0` while it is,
    /// `f64::INFINITY` when nothing has run this session.
    pub fn idle_seconds(&self) -> f64 {
        todo!()
    }

    /// Re-run provisioning for a server; returns as soon as it is queued.
    pub fn retry_mcp_server(&self, server_id: &str, project_id: Option<&str>) -> anyhow::Result<Value> {
        let _ = (server_id, project_id);
        todo!()
    }

    /// Provisioning fields of a server for API responses.
    pub fn mcp_provisioning_state(server: &MCPServer) -> Value {
        let _ = server;
        todo!()
    }

    /// Every MCP server of a project with its provisioning status.
    pub fn list_mcp_servers_with_status(&self, project_id: &str) -> anyhow::Result<Vec<Value>> {
        let _ = project_id;
        todo!()
    }

    /// One server's provisioning status.
    pub fn mcp_server_status(&self, server_id: &str, project_id: Option<&str>) -> anyhow::Result<Value> {
        let _ = (server_id, project_id);
        todo!()
    }

    /// Count servers and tools for a project.
    pub fn count_mcp_tools(&self, project_id: Option<&str>) -> anyhow::Result<Value> {
        let _ = project_id;
        todo!()
    }

    /// LLM-enrich a server's description/tags and write its MCP doc.
    pub fn enrich_mcp_server(&self, server: &mut MCPServer) -> anyhow::Result<()> {
        let _ = server;
        todo!()
    }

    /// Turn a raw discovery error into a user-facing message.
    pub fn friendly_discovery_error(raw: &str) -> String {
        let _ = raw;
        todo!()
    }

    /// `{"enabled": ...}` web-search config for a project.
    pub fn get_web_search_config(&self, project_id: &str) -> anyhow::Result<Value> {
        let _ = project_id;
        todo!()
    }

    /// Toggle web search for a project.
    pub fn set_web_search_enabled(&self, project_id: &str, enabled: bool) -> anyhow::Result<Value> {
        let _ = (project_id, enabled);
        todo!()
    }

    /// Write a server's markdown doc into the project; true on success.
    pub fn write_mcp_doc(&self, project: &Project, server: &MCPServer, content: &str) -> bool {
        let _ = (project, server, content);
        todo!()
    }

    /// Convert an MCPServer to a dict for API responses.
    pub fn mcp_server_to_dict(s: &MCPServer) -> Value {
        let _ = s;
        todo!()
    }

    /// MCP servers of the current project.
    pub fn list_mcp_servers(&self) -> anyhow::Result<Vec<Value>> {
        todo!()
    }

    /// Delete an MCP server (and its definition file) from a project.
    pub fn delete_mcp_server(&self, server_id: &str, project_id: Option<&str>) -> anyhow::Result<Value> {
        let _ = (server_id, project_id);
        todo!()
    }

    /// Remove the on-disk definition file of a server, if any.
    pub fn remove_mcp_definition_file(&self, server: &MCPServer) {
        let _ = server;
        todo!()
    }

    // ------------------------------------------------------------ orchestration

    /// The shared LLMExecutor (I/O pool), created lazily.
    pub fn get_llm_executor(&self) -> Arc<LLMExecutor> {
        todo!()
    }

    /// Shut down the shared executor so it's rebuilt for the new model.
    pub fn reset_llm_executor(&self) {
        todo!()
    }

    /// Fire a progress/SSE event, swallowing any callback panic.
    pub fn emit_progress(progress_callback: Option<&ProgressCallback>, step: &str, status: &str, message: &str) {
        let _ = (progress_callback, step, status, message);
        todo!()
    }

    /// Full orchestrator entry point for a user message: classification,
    /// context enrichment, DAG execution, capability routing and MCP calls.
    /// See [`OrchestrateOutcome`] for the result.
    #[allow(clippy::too_many_arguments)]
    pub fn orchestrate(
        &self,
        user_message: &str,
        skills: Option<Vec<Skill>>,
        cwd: Option<&str>,
        conversation_history: Option<Vec<Value>>,
        has_attachments: bool,
        images: Option<Vec<(Vec<u8>, String)>>,
        progress_callback: Option<ProgressCallback>,
        cancelled: Option<CancelEvent>,
        requested_skill: Option<&str>,
        current_date: Option<&str>,
    ) -> OrchestrateOutcome {
        let _ = (
            user_message, skills, cwd, conversation_history, has_attachments, images,
            progress_callback, cancelled, requested_skill, current_date,
        );
        todo!()
    }

    // --------------------------------------------------------------- auto-write

    /// Write LLM output to `<cwd>/.out/<name>.md` when the user asked to
    /// create/generate/write (or issued a bare "write"/"save").  Returns the
    /// written path, or `None` when no write was needed.
    pub fn maybe_write_output(
        &self,
        user_message: &str,
        llm_response: &str,
        cwd: Option<&str>,
        session_id: Option<&str>,
    ) -> Option<String> {
        let _ = (user_message, llm_response, cwd, session_id);
        todo!()
    }

    /// The last assistant message from session history.
    pub fn get_last_assistant_response(&self, session_id: &str) -> Option<String> {
        let _ = session_id;
        todo!()
    }

    /// Strip conversational preamble/postamble, keeping only the document body.
    pub fn extract_document_content(text: &str) -> String {
        let _ = text;
        todo!()
    }

    // ------------------------------------------------------------------- assets

    /// Return (and create if needed) `<home_dir>/assets`.
    pub fn assets_dir(&self) -> anyhow::Result<PathBuf> {
        todo!()
    }

    /// Save an uploaded file as an asset, converting to markdown when
    /// possible.  Returns `{asset_id, filename, content_type, size, markdown}`.
    pub fn upload_asset(&self, file_bytes: &[u8], filename: &str, content_type: Option<&str>) -> anyhow::Result<Value> {
        let _ = (file_bytes, filename, content_type);
        todo!()
    }

    /// Metadata (and markdown) of a stored asset.
    pub fn get_asset(&self, asset_id: &str) -> anyhow::Result<Value> {
        let _ = asset_id;
        todo!()
    }

    /// Delete a stored asset.
    pub fn delete_asset(&self, asset_id: &str) -> anyhow::Result<Value> {
        let _ = asset_id;
        todo!()
    }

    /// Concatenated markdown context for the given attachments.
    pub fn resolve_attachment_context(&self, asset_ids: &[String]) -> String {
        let _ = asset_ids;
        todo!()
    }

    /// `(bytes, mime_type)` for every image attachment.
    pub fn get_asset_images(&self, asset_ids: &[String]) -> Vec<(Vec<u8>, String)> {
        let _ = asset_ids;
        todo!()
    }

    /// A concrete description of `images` scoped to `user_message`
    /// (the orchestrator's image entry point).
    pub fn vision_perceive(
        &self,
        user_message: &str,
        images: &[(Vec<u8>, String)],
        max_tokens: usize,
        on_model_selected: Option<ModelSelectedCallback>,
    ) -> anyhow::Result<String> {
        let _ = (user_message, images, max_tokens, on_model_selected);
        todo!()
    }

    /// A vision-model answer to `prompt` about `images`; errors with
    /// `"No vision-capable model available"` when none can be loaded.
    pub fn vision_answer(
        &self,
        prompt: &str,
        images: &[(Vec<u8>, String)],
        max_tokens: usize,
        on_model_selected: Option<ModelSelectedCallback>,
    ) -> anyhow::Result<String> {
        let _ = (prompt, images, max_tokens, on_model_selected);
        todo!()
    }
}
