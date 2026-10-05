//! Port of `smartloop/api/routers/chat.py`.
//!
//! Chat completion router with OpenAI-compatible endpoints
//! (`POST /v1/chat/completions`, streaming SSE and non-streaming JSON).

#![allow(unused_imports, dead_code)]

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures::Stream;
use ndarray::Array1;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use regex::Regex;
use serde_json::{Map, Value};

use crate::agents::orchestration::skill_processor::SkillProcessor;
use crate::api::model::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    ChatStatusChunk, Choice, ChunkChoice, Citation, DeltaContent, Usage,
};
use crate::api::services::{get_api_client, get_model_service, ModelService};
// The same relevance criteria retrieval uses to decide a chunk is about the
// user's question — reused below to decide a source is about the answer, so
// the two ends of the citation contract agree on what "related" means.
use crate::api::services::model_service::MIN_BEST_DISTANCE;
use crate::config::app_settings::AppSettings;
use crate::constants::{display_model_name, resolve_current_date, OPENING_TOKENS, SYSTEM_PROMPT};
use crate::llms::base_llm::CancelEvent;
use crate::model_host::client::ModelClient;
use crate::types::ModelCapability;
use crate::workspace::context::{bind, set_active_project, ProjectBoundExecutor};
use crate::workspace::types::Skill;

/// Logger name of the Python module (`logging.getLogger(__name__)`).
pub const LOG_TARGET: &str = "smartloop.api.routers.chat";

// ---------------------------------------------------------------- types

/// A source citation as produced by retrieval / the orchestrator: a free-form
/// dict with public keys (`document_id`, `document_name`, ...) and private
/// `_`-prefixed working keys (`_chunks`, `_vectors`, `_extract`) that are
/// stripped before anything leaves the server.
pub type CitationDict = Map<String, Value>;

/// Citations shared between the source pre-embedding task (which writes
/// `_chunks`/`_vectors` in place) and verification (which reads them).
pub type SharedCitations = Arc<Mutex<Vec<CitationDict>>>;

/// One prior conversation turn: `{"role": ..., "content": ...}`.
pub type HistoryTurn = HashMap<String, String>;

/// A decoded image attachment: `(bytes, mime_type)`.
pub type ImageAttachment = (Vec<u8>, String);

/// `(step, status, message)` progress sink, callable from worker threads.
pub type ProgressCallback = Arc<dyn Fn(&str, &str, &str) + Send + Sync>;

/// A boxed SSE line stream (each item is a complete `data: ...\n\n` frame).
pub type SseStream = Pin<Box<dyn Stream<Item = String> + Send + 'static>>;

/// Arguments to [`ChatService::orchestrate`] (mirrors the keyword arguments of
/// `ModelService.orchestrate`).
#[derive(Clone, Default)]
pub struct OrchestrateArgs {
    pub user_message: String,
    pub skills: Vec<Skill>,
    pub cwd: Option<String>,
    pub conversation_history: Vec<HistoryTurn>,
    pub has_attachments: bool,
    pub images: Vec<ImageAttachment>,
    pub progress_callback: Option<ProgressCallback>,
    pub cancelled: Option<CancelEvent>,
    pub requested_skill: Option<String>,
    pub current_date: Option<String>,
}

/// Result of [`ChatService::orchestrate`]:
/// `(context, matched, error, citations, skill_instruction, mutation)`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OrchestrateOutcome {
    /// The formatted tool output.
    pub context: Option<String>,
    /// True when the orchestrator produced content.
    pub matched: bool,
    /// A user-facing error message (or `None` on success).
    pub error: Option<String>,
    /// Source document dicts (or `None`).
    pub citations: Option<Vec<CitationDict>>,
    /// The matched skill content (or `None`).
    pub skill_instruction: Option<String>,
    /// Stream `context` verbatim instead of running it through another LLM.
    pub mutation: bool,
}

/// The subset of `ModelService` the chat router uses.
///
/// A trait so tests can stand in a fake service (the Python tests drive
/// `stream_chunks` with a `MagicMock`); `ModelService` implements it by
/// delegating to its own methods.
pub trait ChatService: Send + Sync {
    /// `ModelService.is_loaded`.
    fn is_loaded(&self) -> bool;
    /// `ModelService.load_model`.
    fn load_model(&self) -> anyhow::Result<()>;
    /// `ModelService.get_relevant_history(session_id, user_query)`.
    fn get_relevant_history(&self, session_id: &str, user_query: &str) -> Vec<HistoryTurn>;
    /// `ModelService.get_project_skills`.
    fn get_project_skills(&self) -> Vec<Skill>;
    /// `ModelService.get_asset_images`.
    fn get_asset_images(&self, asset_ids: &[String]) -> Vec<ImageAttachment>;
    /// `ModelService.resolve_attachment_context`.
    fn resolve_attachment_context(&self, asset_ids: &[String]) -> String;
    /// `ModelService.orchestrate`.
    fn orchestrate(&self, args: OrchestrateArgs) -> anyhow::Result<OrchestrateOutcome>;
    /// `ModelService.get_project_id`.
    fn get_project_id(&self) -> Option<String>;
    /// `ModelService.get_project_info`.
    fn get_project_info(&self) -> Option<String>;
    /// `ModelService.count_tokens`.
    fn count_tokens(&self, text: &str) -> usize;
    /// `ModelService.get_max_context_tokens`.
    fn get_max_context_tokens(&self) -> usize;
    /// `ModelService.get_response_max_context_tokens`.
    fn get_response_max_context_tokens(&self, capability: Option<ModelCapability>) -> usize;
    /// `ModelService.begin_response_request`.
    fn begin_response_request(&self, progress_callback: Option<ProgressCallback>);
    /// `ModelService.announce_response_model`.
    fn announce_response_model(&self, capability: Option<ModelCapability>);
    /// `ModelService.append_to_session`.
    fn append_to_session(
        &self,
        session_id: &str,
        role: &str,
        content: &str,
        citations: Option<&[CitationDict]>,
        message_id: Option<&str>,
    );
    /// `ModelService.maybe_write_output`; returns the written path.
    fn maybe_write_output(
        &self,
        user_message: &str,
        llm_response: &str,
        cwd: Option<&str>,
        session_id: Option<&str>,
    ) -> Option<String>;
    /// `ModelService.model_client`: how this process reaches the generative model.
    fn model_client(&self) -> Arc<dyn ModelClient>;
    /// `ModelService.unload_auxiliary_models`.
    fn unload_auxiliary_models(&self, include_embedding: bool);
    /// `ModelService.unload_embedding_model`.
    fn unload_embedding_model(&self);
    /// `model_service._model.workspace.workspace_dir` (where the embedding
    /// model is cached), or `None` when there is no workspace.
    fn workspace_dir(&self) -> Option<String>;
    /// Ids of `model_service._model.workspace.projects`, or `None` when there
    /// is no workspace.
    fn workspace_project_ids(&self) -> Option<Vec<String>>;
    /// `model_service._model.workspace.current_project.project_dir`.
    fn current_project_dir(&self) -> Option<String>;
    /// `get_api_client().push_token_usage(...)` — a seam so tests can suppress
    /// telemetry (the Python tests patch `get_api_client`).
    fn push_token_usage(
        &self,
        project_id: Option<&str>,
        model: &str,
        prompt_tokens: usize,
        completion_tokens: usize,
    ) {
        let _ = (project_id, model, prompt_tokens, completion_tokens);
        todo!()
    }
}

impl ChatService for ModelService {
    fn is_loaded(&self) -> bool {
        todo!()
    }
    fn load_model(&self) -> anyhow::Result<()> {
        todo!()
    }
    fn get_relevant_history(&self, session_id: &str, user_query: &str) -> Vec<HistoryTurn> {
        let _ = (session_id, user_query);
        todo!()
    }
    fn get_project_skills(&self) -> Vec<Skill> {
        todo!()
    }
    fn get_asset_images(&self, asset_ids: &[String]) -> Vec<ImageAttachment> {
        let _ = asset_ids;
        todo!()
    }
    fn resolve_attachment_context(&self, asset_ids: &[String]) -> String {
        let _ = asset_ids;
        todo!()
    }
    fn orchestrate(&self, args: OrchestrateArgs) -> anyhow::Result<OrchestrateOutcome> {
        let _ = args;
        todo!()
    }
    fn get_project_id(&self) -> Option<String> {
        todo!()
    }
    fn get_project_info(&self) -> Option<String> {
        todo!()
    }
    fn count_tokens(&self, text: &str) -> usize {
        let _ = text;
        todo!()
    }
    fn get_max_context_tokens(&self) -> usize {
        todo!()
    }
    fn get_response_max_context_tokens(&self, capability: Option<ModelCapability>) -> usize {
        let _ = capability;
        todo!()
    }
    fn begin_response_request(&self, progress_callback: Option<ProgressCallback>) {
        let _ = progress_callback;
        todo!()
    }
    fn announce_response_model(&self, capability: Option<ModelCapability>) {
        let _ = capability;
        todo!()
    }
    fn append_to_session(
        &self,
        session_id: &str,
        role: &str,
        content: &str,
        citations: Option<&[CitationDict]>,
        message_id: Option<&str>,
    ) {
        let _ = (session_id, role, content, citations, message_id);
        todo!()
    }
    fn maybe_write_output(
        &self,
        user_message: &str,
        llm_response: &str,
        cwd: Option<&str>,
        session_id: Option<&str>,
    ) -> Option<String> {
        let _ = (user_message, llm_response, cwd, session_id);
        todo!()
    }
    fn model_client(&self) -> Arc<dyn ModelClient> {
        todo!()
    }
    fn unload_auxiliary_models(&self, include_embedding: bool) {
        let _ = include_embedding;
        todo!()
    }
    fn unload_embedding_model(&self) {
        todo!()
    }
    fn workspace_dir(&self) -> Option<String> {
        todo!()
    }
    fn workspace_project_ids(&self) -> Option<Vec<String>> {
        todo!()
    }
    fn current_project_dir(&self) -> Option<String> {
        todo!()
    }
}

/// FastAPI's `HTTPException`: answered as `{"detail": ...}` with its status.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{status_code}: {detail}")]
pub struct HttpException {
    pub status_code: u16,
    pub detail: String,
}

impl HttpException {
    /// Build an exception with the given status and detail message.
    pub fn new(status_code: u16, detail: impl Into<String>) -> Self {
        Self { status_code, detail: detail.into() }
    }
}

impl IntoResponse for HttpException {
    fn into_response(self) -> Response {
        todo!()
    }
}

// ---------------------------------------------------------------- constants

/// Format a status event as an SSE data line.
///
/// Serializes a [`ChatStatusChunk`] (`object: "chat.status"`) so clients can
/// distinguish these from OpenAI-compatible `chat.completion.chunk` content
/// lines and track them against a schema.
pub fn status_event(step: &str, status: &str, message: &str) -> String {
    let _ = (step, status, message);
    todo!()
}

/// Single-worker pool for running sync LLM operations (`_executor`).
pub static EXECUTOR: Lazy<ProjectBoundExecutor> =
    Lazy::new(|| ProjectBoundExecutor::new(1, Some("chat")));

/// A leading `/<skill-name>` in the prompt is an explicit request to fulfil the
/// turn with that skill (e.g. `/fund-info summarise Q3`). Captures the token
/// and the remainder so the token can be matched against the project's skills.
pub static SLASH_SKILL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)^\s*/([A-Za-z0-9][\w-]*)\s*(.*)$").unwrap());

/// Chunks of the answer that go out before the source encode is dispatched.
/// The embedder loads onto the same device the answer decodes on, so it waits
/// for the opening — see the dispatch in [`stream_chunks`].
pub const PREP_SOURCES_AFTER_CHUNKS: usize = OPENING_TOKENS;

/// Literal role tags and their escaped forms (ordered as in Python).
pub const TAG_ESCAPE: &[(&str, &str)] = &[
    ("<system>", "&lt;system&gt;"),
    ("</system>", "&lt;/system&gt;"),
    ("<user>", "&lt;user&gt;"),
    ("</user>", "&lt;/user&gt;"),
    ("<assistant>", "&lt;assistant&gt;"),
    ("</assistant>", "&lt;/assistant&gt;"),
];

/// A decode that failed for want of memory.  llama.cpp reports it as a
/// non-zero `llama_decode` return rather than a typed error, so the message
/// is all there is.
pub static DECODE_OOM_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)llama_decode returned -?\d+|out of memory|failed to allocate").unwrap()
});

/// A shared run of this many consecutive words is the evidence that an answer
/// actually drew on a source.
pub const PHRASE_LEN: usize = 3;

/// Function words carry no attribution; a run built only out of these is a
/// coincidence of grammar rather than a trace of one text in another.
pub static STOPWORDS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    "a an the and or but if then than that this these those there here \
     is are was were be been being am do does did doing done \
     have has had having will would shall should can could may might must \
     of in on at to for from by with without into onto over under about as \
     it its it's they them their we us our you your he she his her i me my \
     not no nor so such only just also very more most much many few some any \
     each every other another same own all both which who whom whose what when \
     where why how while during before after above below up down out off again \
     one two three first second next last new old good great best"
        .split_whitespace()
        .collect()
});

/// How many words in a shared run must be outside [`STOPWORDS`].
pub const MIN_CONTENT_WORDS: usize = 2;

/// A word in any script.
pub static WORD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\w[\w.,%/-]*").unwrap());

/// Scripts written without spaces between words (kana, CJK ideographs,
/// Thai): their runs are counted in characters instead.
pub static CONTINUOUS_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[\u{3040}-\u{30ff}\u{3400}-\u{4dbf}\u{4e00}-\u{9fff}\u{f900}-\u{faff}\u{0e00}-\u{0e7f}]{5,}")
        .unwrap()
});

/// Character n-gram length for continuous scripts.
pub const CHAR_GRAM: usize = 5;

/// A number in text.
pub static NUMBER_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b\d[\d.,]*\b").unwrap());

/// Figures below this many significant digits are not evidence of anything.
pub const FIGURE_MIN_DIGITS: usize = 3;

/// Window size (characters) for semantic comparison.
pub const WINDOW_CHARS: usize = 1200;
/// Overlap between consecutive windows.
pub const WINDOW_OVERLAP: usize = 200;
/// Ceiling on the windows taken from any one text.
pub const MAX_WINDOWS: usize = 8;

/// Distance above which the answer is not about this source (retrieval's own
/// quality floor).
pub const CITATION_MAX_DISTANCE: f64 = MIN_BEST_DISTANCE;
/// Margin within which a source still counts against the closest one.
pub const CITATION_CUTOFF_MARGIN: f64 = 0.30;

/// RPC budget for the verification-time encode calls against the model host
/// (seconds).
pub const VERIFY_ENCODE_TIMEOUT_S: f64 = 30.0;

/// Chunk size used to stream a mutation's synthesized content.
pub const STREAM_CHUNK_SIZE: usize = 12;

/// Alphanumeric characters that end the buffered opening of a tool-synthesis
/// stream.
pub const ALNUM_STREAM_THRESHOLD: usize = 20;

// ---------------------------------------------------------------- helpers

/// Resolve a leading `/<skill-name>` slash-command against `skills`.
///
/// Returns `(Some(skill_name), cleaned_message)` when the prompt opens with a
/// slash-command naming one of the project's skills (matched case- and
/// separator-insensitively). When the user typed only the token, the skill's
/// own body (frontmatter stripped) stands in, falling back to
/// `Run the '<name>' skill.`.  Returns `(None, message)` unchanged otherwise.
pub fn resolve_requested_skill(message: &str, skills: &[Skill]) -> (Option<String>, String) {
    let _ = (message, skills);
    todo!()
}

/// Extract the raw tool payload from the orchestrator's wrapped context.
pub fn extract_tool_content(orchestrator_context: &str) -> String {
    let _ = orchestrator_context;
    todo!()
}

/// True when `exc` is generation running out of device memory.
pub fn is_decode_oom(exc: &dyn std::fmt::Display) -> bool {
    let _ = exc;
    todo!()
}

/// A user turn asking the model to finish an answer it was cut off in.
pub fn continuation_turn(partial: &str) -> String {
    let _ = partial;
    todo!()
}

/// Escape literal role tags in `text` so they don't break the regex in
/// `create_messages_from_prompt`.
pub fn escape_role_tags(text: &str) -> String {
    let _ = text;
    todo!()
}

/// Convert OpenAI-style messages to the prompt format expected by the LLM.
///
/// When `context` is provided (pre-formatted by the orchestrator), it is
/// injected into the last user message, followed by a restatement of the
/// question.
pub fn format_messages_to_prompt(
    messages: &[ChatMessage],
    context: Option<&str>,
    cwd: Option<&str>,
    project_info: Option<&str>,
    current_date: Option<&str>,
) -> String {
    let _ = (messages, context, cwd, project_info, current_date);
    todo!()
}

/// The [`PHRASE_LEN`]-word runs in `text` that could evidence attribution.
pub fn phrases(text: &str) -> HashSet<String> {
    let _ = text;
    todo!()
}

/// Precise numbers in `text`, normalised so formatting doesn't matter.
pub fn figures(text: &str) -> HashSet<String> {
    let _ = text;
    todo!()
}

/// What a finished answer contains, for checking sources against it.
#[derive(Debug, Clone, Default)]
pub struct AnswerEvidence {
    pub blank: bool,
    pub phrases: HashSet<String>,
    pub figures: HashSet<String>,
}

impl AnswerEvidence {
    /// Analyse `response`.
    pub fn new(response: &str) -> Self {
        let _ = response;
        todo!()
    }

    /// True when the answer shows evidence of having used `source`.
    pub fn drew_on(&self, source: &str) -> bool {
        let _ = source;
        todo!()
    }
}

/// `text` split into overlapping windows the embedder can take whole,
/// sampled evenly when there are more than [`MAX_WINDOWS`].
pub fn windows(text: &str) -> Vec<String> {
    let _ = text;
    todo!()
}

/// `(vectors, texts)` — the passages this citation is judged on.
pub fn citation_sources(citation: &CitationDict) -> (Vec<Array1<f32>>, Vec<String>) {
    let _ = citation;
    todo!()
}

/// The source *text* for the fallback check, which cannot use vectors.
pub fn lexical_sources(citation: &CitationDict) -> Vec<String> {
    let _ = citation;
    todo!()
}

/// Answer-to-source cosine distance per citation, closest passage wins.
/// `None` (whole batch or per citation) means "could not be judged".
pub fn semantic_distances(
    citations: &[CitationDict],
    response: &str,
    base_dir: Option<&str>,
) -> Option<Vec<Option<f64>>> {
    let _ = (citations, response, base_dir);
    todo!()
}

/// True when `response` actually drew on this citation's source (lexical
/// check only).
pub fn citation_used(citation: &CitationDict, response: &str) -> bool {
    let _ = (citation, response);
    todo!()
}

/// Deduplicate by document and drop sources the answer never used.
///
/// Never fails: on error the sources are kept as-is, deduplicated.
/// `embed = false` forces the lexical fallback.
pub fn verified_citations(
    citations: &[CitationDict],
    response: &str,
    base_dir: Option<&str>,
    embed: bool,
) -> Vec<CitationDict> {
    let _ = (citations, response, base_dir, embed);
    todo!()
}

/// One entry per document, with the private `_` keys stripped.
pub fn dedupe_citations(citations: &[CitationDict]) -> Vec<CitationDict> {
    let _ = citations;
    todo!()
}

/// One entry per document, private keys intact, retrieval order kept.
pub fn unique_citations(citations: &[CitationDict]) -> Vec<CitationDict> {
    let _ = citations;
    todo!()
}

/// The fallible core of [`verified_citations`].
pub fn verify_citations(
    citations: &[CitationDict],
    response: &str,
    base_dir: Option<&str>,
    embed: bool,
) -> anyhow::Result<Vec<CitationDict>> {
    let _ = (citations, response, base_dir, embed);
    todo!()
}

/// Where the embedding model is cached — the workspace, as retrieval uses.
pub fn embedding_base_dir(model_service: &dyn ChatService) -> Option<String> {
    let _ = model_service;
    todo!()
}

/// Embed every source now, while the answer is still being generated.
/// Normalises every citation to `_chunks` plus index-aligned `_vectors`.
/// Best effort throughout.
pub fn prepare_citation_sources(citations: &SharedCitations, model_service: &dyn ChatService) {
    let _ = (citations, model_service);
    todo!()
}

/// [`verified_citations`] off the async runtime, after waiting on the
/// `prepared` pre-embedding task.
pub async fn verified_citations_async(
    citations: SharedCitations,
    response: String,
    model_service: Arc<dyn ChatService>,
    prepared: Option<tokio::task::JoinHandle<()>>,
) -> Vec<CitationDict> {
    let _ = (citations, response, model_service, prepared);
    todo!()
}

/// Bind this request to the project it named, for the rest of its life.
pub fn bind_request_project(request: &ChatCompletionRequest, model_service: &dyn ChatService) {
    let _ = (request, model_service);
    todo!()
}

/// The working directory for this request — always framework-level (the
/// client-sent `cwd` is deliberately ignored).
pub fn request_cwd(model_service: &dyn ChatService) -> Option<String> {
    let _ = model_service;
    todo!()
}

/// Resolve the current date from the client's `additional_message_params`.
pub fn request_current_date(request: &ChatCompletionRequest) -> String {
    let _ = request;
    todo!()
}

/// Return a new message list where attachment asset IDs are resolved to
/// markdown prepended to the message text.
pub fn resolve_message_attachments(
    messages: Vec<ChatMessage>,
    model_service: &dyn ChatService,
) -> Vec<ChatMessage> {
    let _ = (messages, model_service);
    todo!()
}

/// Trim oldest conversation turns when the message list exceeds the context
/// window.  Python defaults: `reserved_tokens = 2048`, `context_tokens = 0`,
/// `max_context = None` (the default model's window).
pub fn compact_messages(
    messages: Vec<ChatMessage>,
    model_service: &dyn ChatService,
    reserved_tokens: usize,
    context_tokens: usize,
    max_context: Option<usize>,
) -> Vec<ChatMessage> {
    let _ = (messages, model_service, reserved_tokens, context_tokens, max_context);
    todo!()
}

/// Prepend already-loaded conversation `history` to the request messages.
pub fn build_messages_with_history(
    request: &ChatCompletionRequest,
    history: &[HistoryTurn],
) -> Vec<ChatMessage> {
    let _ = (request, history);
    todo!()
}

/// Free the auxiliary models once `work` (an abandoned worker's completion)
/// has finished.  The unload itself runs on a blocking worker thread and any
/// failure is logged and swallowed.
pub fn unload_when_idle<F>(work: F, model_service: Arc<dyn ChatService>)
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let _ = (work, model_service);
    todo!()
}

/// True when the model produced no meaningful content (an unclosed code
/// fence, whitespace, or fewer than 3 alphanumeric characters).
pub fn is_degenerate(text: &str) -> bool {
    let _ = text;
    todo!()
}

/// Convert sync LLM streaming to async SSE.
///
/// Produces the exact SSE frames of the Python generator: `chat.status`
/// events, `chat.completion.chunk` deltas, `chat.message_id`, `chat.usage`,
/// `chat.citations` and the final `data: [DONE]`.
pub fn stream_chunks(request: ChatCompletionRequest, model_service: Arc<dyn ChatService>) -> SseStream {
    let _ = (request, model_service);
    todo!()
}

/// Router for `/v1` chat endpoints.
pub fn router() -> Router {
    Router::new().route("/v1/chat/completions", post(create_chat_completion))
}

/// `POST /v1/chat/completions` — create a chat completion (OpenAI-compatible).
/// Supports both streaming and non-streaming modes.
pub async fn create_chat_completion(Json(request): Json<ChatCompletionRequest>) -> Response {
    let _ = request;
    todo!()
}

/// [`create_chat_completion`] against an explicit service (test seam).
pub async fn create_chat_completion_with(
    request: ChatCompletionRequest,
    model_service: Arc<dyn ChatService>,
) -> Result<Response, HttpException> {
    let _ = (request, model_service);
    todo!()
}

/// The non-streaming half of `create_chat_completion`.
pub async fn complete_without_streaming(
    request: ChatCompletionRequest,
    model_service: Arc<dyn ChatService>,
) -> Result<ChatCompletionResponse, HttpException> {
    let _ = (request, model_service);
    todo!()
}
