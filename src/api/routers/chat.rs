//! Port of `smartloop/api/routers/chat.py`.
//!
//! Chat completion router with OpenAI-compatible endpoints
//! (`POST /v1/chat/completions`, streaming SSE and non-streaming JSON).

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures::{Stream, StreamExt};
use ndarray::{Array1, Array2};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use regex::Regex;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::task::JoinHandle;

use crate::agents::orchestration::skill_processor::SkillProcessor;
use crate::api::model::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    ChatStatusChunk, Choice, ChunkChoice, Citation, DeltaContent, Usage,
};
use crate::api::model::chat::{ChatStatus, Role};
use crate::api::services::{get_api_client, get_model_service, ModelService};
// The same relevance criteria retrieval uses to decide a chunk is about the
// user's question — reused below to decide a source is about the answer, so
// the two ends of the citation contract agree on what "related" means.
use crate::api::services::model_service::MIN_BEST_DISTANCE;
use crate::api::services::output_writer::WriteOutcome;
use crate::config::app_settings::AppSettings;
use crate::constants::{display_model_name, resolve_current_date, OPENING_TOKENS, SYSTEM_PROMPT};
use crate::db::embeddings;
use crate::llms::base_llm::CancelEvent;
use crate::model_host::client::{ModelClient, ModelHostError, TextStream};
use crate::tasks::IndexerTask;
use crate::types::ModelCapability;
use crate::workspace::context::{
    active_project_id, bind, scope_project, set_active_project, use_project, ProjectBoundExecutor,
};
use crate::workspace::types::Skill;

/// Logger name of the Python module (`logging.getLogger(__name__)`).
pub const LOG_TARGET: &str = "smartloop.api.routers.chat";

// ---------------------------------------------------------------- types

/// A source citation as produced by retrieval / the orchestrator: a free-form
/// dict with public keys (`document_id`, `document_name`, ...) and private
/// `_`-prefixed working keys (`_chunks`, `_vectors`, `_extract`) that are
/// stripped before anything leaves the server.
///
/// `_vectors` entries are the raw embedding BLOBs (Python `bytes`), carried in
/// the JSON dict as an array of byte values (`null` when a chunk has no
/// vector); a base64 string is accepted on read as well.
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
/// `(context, matched, error, citations, skill_instruction, mutation, capability)`.
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
    /// What the planner classified the request as (e.g. coding), if it ran.
    pub capability: Option<String>,
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
    /// `ModelService.maybe_write_output`; returns what was written.
    fn maybe_write_output(
        &self,
        user_message: &str,
        llm_response: &str,
        cwd: Option<&str>,
        session_id: Option<&str>,
    ) -> WriteOutcome;
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
    /// The user's folder the current project is bound to, or `None`.
    fn current_work_dir(&self) -> Option<String> {
        None
    }
    /// Bring the index of the current project's folder up to date, blocking
    /// until it is; `false` when there is nothing to index or it failed.
    fn index_work_dir(&self) -> bool {
        false
    }
    /// What the folder index is doing right now (`Reading <file>`,
    /// `Indexing <file>`), if anything.
    fn indexing_activity(&self) -> Option<String> {
        None
    }
    /// Bind the current project to `dir`, already checked by the caller;
    /// `false` when there is no project to bind or it couldn't be saved.
    fn bind_work_dir(&self, dir: &str) -> bool {
        let _ = dir;
        false
    }
    /// Whether the answer comes from a remote model, whose usage the
    /// platform records itself.
    fn answers_remotely(&self) -> bool {
        false
    }
    /// `get_api_client().push_token_usage(...)` — a seam so tests can suppress
    /// telemetry (the Python tests patch `get_api_client`).  Skipped for a
    /// remote answer, which the platform has already counted.
    fn push_token_usage(
        &self,
        project_id: Option<&str>,
        model: &str,
        prompt_tokens: usize,
        completion_tokens: usize,
    ) {
        if self.answers_remotely() {
            return;
        }
        get_api_client().push_token_usage(
            project_id,
            model,
            prompt_tokens as i64,
            completion_tokens as i64,
        );
    }
}

impl ChatService for ModelService {
    fn answers_remotely(&self) -> bool {
        ModelService::answers_remotely(self)
    }

    fn is_loaded(&self) -> bool {
        ModelService::is_loaded(self)
    }
    fn load_model(&self) -> anyhow::Result<()> {
        ModelService::load_model(self).map(|_| ())
    }
    fn get_relevant_history(&self, session_id: &str, user_query: &str) -> Vec<HistoryTurn> {
        // Python default: ``max_turns=4``.
        ModelService::get_relevant_history(self, session_id, user_query, 4)
    }
    fn get_project_skills(&self) -> Vec<Skill> {
        ModelService::get_project_skills(self)
    }
    fn get_asset_images(&self, asset_ids: &[String]) -> Vec<ImageAttachment> {
        ModelService::get_asset_images(self, asset_ids)
    }
    fn resolve_attachment_context(&self, asset_ids: &[String]) -> String {
        ModelService::resolve_attachment_context(self, asset_ids)
    }
    fn orchestrate(&self, args: OrchestrateArgs) -> anyhow::Result<OrchestrateOutcome> {
        let history: Vec<Value> = args
            .conversation_history
            .iter()
            .map(|turn| {
                let mut obj = Map::new();
                for key in ["role", "content"] {
                    if let Some(v) = turn.get(key) {
                        obj.insert(key.to_string(), Value::String(v.clone()));
                    }
                }
                for (k, v) in turn {
                    if !obj.contains_key(k) {
                        obj.insert(k.clone(), Value::String(v.clone()));
                    }
                }
                Value::Object(obj)
            })
            .collect();
        let (context, matched, error, citations, skill_instruction, mutation, capability) =
            ModelService::orchestrate(
                self,
                &args.user_message,
                Some(args.skills),
                args.cwd.as_deref(),
                Some(history),
                args.has_attachments,
                Some(args.images),
                args.progress_callback,
                args.cancelled,
                args.requested_skill.as_deref(),
                args.current_date.as_deref(),
            );
        let citations = citations.map(|list| {
            list.into_iter()
                .filter_map(|c| match c {
                    Value::Object(map) => Some(map),
                    _ => None,
                })
                .collect()
        });
        Ok(OrchestrateOutcome { context, matched, error, citations, skill_instruction, mutation, capability })
    }
    fn get_project_id(&self) -> Option<String> {
        ModelService::get_project_id(self)
    }
    fn get_project_info(&self) -> Option<String> {
        ModelService::get_project_info(self)
    }
    fn count_tokens(&self, text: &str) -> usize {
        ModelService::count_tokens(self, text)
    }
    fn get_max_context_tokens(&self) -> usize {
        ModelService::get_max_context_tokens(self)
    }
    fn get_response_max_context_tokens(&self, capability: Option<ModelCapability>) -> usize {
        ModelService::get_response_max_context_tokens(self, capability.map(|c| c.value()))
    }
    fn begin_response_request(&self, progress_callback: Option<ProgressCallback>) {
        ModelService::begin_response_request(self, progress_callback)
    }
    fn announce_response_model(&self, capability: Option<ModelCapability>) {
        ModelService::announce_response_model(self, capability.map(|c| c.value()))
    }
    fn append_to_session(
        &self,
        session_id: &str,
        role: &str,
        content: &str,
        citations: Option<&[CitationDict]>,
        message_id: Option<&str>,
    ) {
        let citations =
            citations.map(|list| list.iter().cloned().map(Value::Object).collect::<Vec<_>>());
        if let Err(e) =
            ModelService::append_to_session(self, session_id, role, content, citations, message_id)
        {
            tracing::error!(
                target: LOG_TARGET,
                "Failed to append to session {}: {}", session_id, e
            );
        }
    }
    fn maybe_write_output(
        &self,
        user_message: &str,
        llm_response: &str,
        cwd: Option<&str>,
        session_id: Option<&str>,
    ) -> WriteOutcome {
        ModelService::maybe_write_output(self, user_message, llm_response, cwd, session_id)
    }
    fn model_client(&self) -> Arc<dyn ModelClient> {
        match ModelService::model_client(self) {
            Ok(client) => client,
            // The Python property raises where it is read, which lands inside
            // the caller's error handling; a client that fails every call
            // surfaces the same error at the same point.
            Err(e) => Arc::new(UnavailableModelClient(e)),
        }
    }
    fn unload_auxiliary_models(&self, include_embedding: bool) {
        ModelService::unload_auxiliary_models(self, include_embedding)
    }
    fn unload_embedding_model(&self) {
        ModelService::unload_embedding_model(self)
    }
    fn workspace_dir(&self) -> Option<String> {
        let model = self.model.read().clone()?;
        let dir = model.workspace.read().workspace_dir.clone();
        Some(dir)
    }
    fn workspace_project_ids(&self) -> Option<Vec<String>> {
        let model = self.model.read().clone()?;
        let ids = model.workspace.read().projects.iter().map(|p| p.id.clone()).collect();
        Some(ids)
    }
    fn current_project_dir(&self) -> Option<String> {
        let model = self.model.read().clone()?;
        let ws = model.workspace.read();
        ws.current_project().map(|p| p.project_dir.clone())
    }
    fn current_work_dir(&self) -> Option<String> {
        let model = self.model.read().clone()?;
        let ws = model.workspace.read();
        ws.current_project().and_then(|p| p.work_dir())
    }
    fn index_work_dir(&self) -> bool {
        let Some(indexer) = self.task_queue.read().get_task_as::<IndexerTask>("indexer") else { return false };
        match indexer.index_workspace_now() {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!(target: LOG_TARGET, "[cwd] indexing the project's folder failed: {}", e);
                false
            }
        }
    }
    fn indexing_activity(&self) -> Option<String> {
        self.task_queue.read().get_task_as::<IndexerTask>("indexer")?.workspace_activity()
    }
    fn bind_work_dir(&self, dir: &str) -> bool {
        let project_id = self.model.read().clone().and_then(|model| {
            let ws = model.workspace.read();
            ws.current_project().map(|p| p.id.clone())
        });
        let Some(project_id) = project_id else { return false };
        let bound = self
            .update_project_record(&project_id, |p| p.cwd = Some(dir.to_string()))
            .and_then(|()| self.persist_model());
        match bound {
            Ok(()) => {
                tracing::info!(target: LOG_TARGET, "[cwd] project {} bound to {}", project_id, dir);
                true
            }
            Err(e) => {
                tracing::warn!(target: LOG_TARGET, "[cwd] could not bind project {} to {}: {}", project_id, dir, e);
                false
            }
        }
    }
}

/// A model client standing in for one that could not be resolved: every call
/// fails with the resolution error.
struct UnavailableModelClient(ModelHostError);

impl UnavailableModelClient {
    fn err<T>(&self) -> anyhow::Result<T> {
        Err(anyhow::Error::new(self.0.clone()))
    }
}

impl ModelClient for UnavailableModelClient {
    fn generate(
        &self,
        _prompt: &str,
        _model_name: Option<&str>,
        _capability: Option<&str>,
        _max_tokens: Option<usize>,
        _thinking: Option<bool>,
    ) -> anyhow::Result<String> {
        self.err()
    }
    fn stream<'a>(
        &'a self,
        _prompt: &str,
        _model_name: Option<&str>,
        _capability: Option<&str>,
        _cancelled: Option<CancelEvent>,
        _thinking: Option<bool>,
    ) -> anyhow::Result<TextStream<'a>> {
        self.err()
    }
    fn encode_texts(
        &self,
        _texts: &[String],
        _base_dir: Option<&str>,
    ) -> anyhow::Result<Option<Array2<f32>>> {
        self.err()
    }
    fn is_loaded(&self) -> anyhow::Result<bool> {
        self.err()
    }
    fn model_name(&self) -> anyhow::Result<Option<String>> {
        self.err()
    }
    fn model_label(&self, _capability: Option<&str>) -> anyhow::Result<String> {
        self.err()
    }
    fn count_tokens(&self, _text: &str) -> anyhow::Result<usize> {
        self.err()
    }
    fn max_context_tokens(&self) -> anyhow::Result<usize> {
        self.err()
    }
    fn response_max_context_tokens(&self, _capability: Option<&str>) -> anyhow::Result<usize> {
        self.err()
    }
    fn has_capability_model(&self, _capability: &str) -> anyhow::Result<bool> {
        self.err()
    }
    fn idle_seconds(&self) -> anyhow::Result<f64> {
        self.err()
    }
    fn load_model(&self) -> anyhow::Result<Option<String>> {
        self.err()
    }
    fn unload_auxiliary_models(&self, _include_embedding: bool) -> anyhow::Result<()> {
        self.err()
    }
    fn unload_embedding_model(&self) -> anyhow::Result<()> {
        self.err()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
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
        let status =
            StatusCode::from_u16(self.status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, Json(json!({ "detail": self.detail }))).into_response()
    }
}

// ---------------------------------------------------------------- constants

/// Format a status event as an SSE data line.
///
/// Serializes a [`ChatStatusChunk`] (`object: "chat.status"`) so clients can
/// distinguish these from OpenAI-compatible `chat.completion.chunk` content
/// lines and track them against a schema.  A status outside the schema's
/// `processing`/`completed`/`error` is reported as `processing`.
pub fn status_event(step: &str, status: &str, message: &str) -> String {
    try_status_event(step, status, message)
        .unwrap_or_else(|| try_status_event(step, "processing", message).unwrap_or_default())
}

/// [`status_event`] that refuses a status outside the schema (pydantic would
/// raise there), so a malformed progress event is dropped rather than sent.
fn try_status_event(step: &str, status: &str, message: &str) -> Option<String> {
    let status: ChatStatus = serde_json::from_value(Value::String(status.to_string())).ok()?;
    let chunk = ChatStatusChunk {
        object: "chat.status".to_string(),
        step: step.to_string(),
        status,
        message: message.to_string(),
    };
    let payload = serde_json::to_string(&chunk).ok()?;
    Some(format!("data: {payload}\n\n"))
}

/// How long a turn waits for its project's folder to be indexed before it
/// answers from what is ready.
const INDEX_WAIT: Duration = Duration::from_secs(180);

/// Single-worker pool for running sync LLM operations (`_executor`).
pub static EXECUTOR: Lazy<ProjectBoundExecutor> =
    Lazy::new(|| ProjectBoundExecutor::new(1, Some("chat")));

/// A leading `/<skill-name>` in the prompt is an explicit request to fulfil the
/// turn with that skill (e.g. `/fund-info summarise Q3`). Captures the token
/// and the remainder so the token can be matched against the project's skills.
pub static SLASH_SKILL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)^\s*/([A-Za-z0-9][\w-]*)\s*(.*)$").unwrap());

/// Separator-insensitive skill-name normalisation (`[\s_\-]+` removed).
static SKILL_NAME_SEPARATORS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\s_\-]+").unwrap());

/// Chunks of the answer that go out before the source encode is dispatched.
/// The embedder loads onto the same device the answer decodes on, so it waits
/// for the opening — see the dispatch in [`stream_chunks`].  The same window
/// the decode rate is reported over, so the log and this decision are talking
/// about the same opening.
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
/// non-zero `llama_decode` return (no KV slot, or a compute buffer that could
/// not be allocated) rather than a typed error, so the message is all there is.
pub static DECODE_OOM_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)llama_decode returned -?\d+|out of memory|failed to allocate").unwrap()
});

/// A shared run of this many consecutive words is the evidence that an answer
/// actually drew on a source.  Single shared words prove nothing: measured
/// against real transcripts, a workout answer and an unrelated fund prospectus
/// share 7-12 ordinary words ("available", "different", "perform", "three"),
/// which is why a one-word rule credited five Vanguard PDFs to a `good-morning`
/// routine.  No pair of unrelated texts in that sample shared a 3-word run.
///
/// Runs of ordinary grammar are excluded on top of that — see [`STOPWORDS`].
/// The length stays at 3 rather than rising to 4: a four-word rule loses the
/// short reused phrase that is often the only trace a summarising answer keeps
/// of its source ("engaged for contract work" against "for contract work
/// services" shares three words and no four).
pub const PHRASE_LEN: usize = 3;

/// Function words carry no attribution: every English text has them, so a run
/// built only out of these is a coincidence of grammar rather than a trace of
/// one text in another.  "here are the", "in the last", "as well as", "one of
/// the" — each is a three-word run, and each used to be enough to credit a
/// document the answer had never opened.  A run must carry at least
/// [`MIN_CONTENT_WORDS`] words from outside this set to count as evidence.
///
/// This list is English and only English, which is a real limit and not a
/// hidden one: against a French or Japanese answer nothing here matches, every
/// word reads as a content word, and the rule falls back to the plain
/// three-word run it refines.
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

/// How many words in a shared run must be outside [`STOPWORDS`].  Two, not
/// one: a single content word surrounded by grammar ("here are the **jobs**")
/// is the shape of an incidental collision, while two content words in a
/// three-word window ("for **contract work**") is phrasing one text took from
/// another.
pub const MIN_CONTENT_WORDS: usize = 2;

/// A word in any script.  ASCII-only matching did not weaken the check outside
/// English so much as switch it off: Arabic, Chinese, Hebrew, Cyrillic,
/// Devanagari and Thai text yielded no words at all, and accented Latin was
/// mangled into fragments that counted as content words.
pub static WORD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\w[\w.,%/-]*").unwrap());

/// Scripts written without spaces between words: Japanese kana, CJK
/// ideographs, Thai.  Splitting these on whitespace yields one enormous "word"
/// and no runs, so their runs are counted in characters instead — [`CHAR_GRAM`]
/// of them is a comparable unit to a three-word English run.
pub static CONTINUOUS_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[\u{3040}-\u{30ff}\u{3400}-\u{4dbf}\u{4e00}-\u{9fff}\u{f900}-\u{faff}\u{0e00}-\u{0e7f}]{5,}")
        .unwrap()
});

/// Character n-gram length for continuous scripts.
pub const CHAR_GRAM: usize = 5;

/// A number in text.
pub static NUMBER_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b\d[\d.,]*\b").unwrap());

/// Figures below this many significant digits are not evidence of anything: an
/// answer saying "3 sets of 10 reps" would otherwise match any document that
/// happens to contain a 3.
pub const FIGURE_MIN_DIGITS: usize = 3;

/// Window size (characters) for semantic comparison.
///
/// A citation is scored on the chunks it was retrieved for (`_chunks`), not on
/// its `_extract` — that is the whole document, and scoring a whole document
/// credits almost anything.  `_extract` stays as the fallback for citations
/// built elsewhere (web results, document fan-out) that carry no chunks.
/// Either way the text is compared a window at a time and scored on its
/// closest window.
pub const WINDOW_CHARS: usize = 1200;
/// Overlap between consecutive windows.
pub const WINDOW_OVERLAP: usize = 200;
/// Ceiling on the windows taken from any one text.  Verification runs after
/// the answer has streamed, so a 200-page PDF must not turn it into a minute
/// of embedding.
pub const MAX_WINDOWS: usize = 8;

/// Distance above which the answer is not about this source: retrieval's own
/// quality floor — the same embedding model and the same `1 - cosine` scale.
pub const CITATION_MAX_DISTANCE: f64 = MIN_BEST_DISTANCE;
/// Margin within which a source still counts against the closest one.  Not
/// retrieval's 0.15: sources an answer genuinely used spread wider than chunks
/// matching a query do, and 0.30 spans that spread while still landing well
/// clear of unrelated documents.
pub const CITATION_CUTOFF_MARGIN: f64 = 0.30;

/// RPC budget for the verification-time encode calls against the model host
/// (seconds).  Verification is a nicety layered on top of an already-streamed
/// answer, so it must never hold the pause hostage.
pub const VERIFY_ENCODE_TIMEOUT_S: f64 = 30.0;

/// Chunk size used to stream a mutation's synthesized content.
pub const STREAM_CHUNK_SIZE: usize = 12;

/// Alphanumeric characters that end the buffered opening of a tool-synthesis
/// stream.
pub const ALNUM_STREAM_THRESHOLD: usize = 20;

const ORCHESTRATION_TIMEOUT_CONTENT: &str = concat!(
    "**Request timed out.**\n\n",
    "The assistant took too long analyzing and ",
    "running the matched skills/tools and was ",
    "stopped. This often means the loaded model is ",
    "too small to synthesize the gathered context. ",
    "Try a simpler request, or enable a larger ",
    "model (`/models` in the terminal, **Settings → Models** in Studio)."
);

const NO_RESPONSE_GUIDANCE: &str = concat!(
    "**No response generated.**\n\n",
    "The model answered with nothing, twice. Either it ",
    "ran out of room before the answer began, or it ",
    "can't work from the gathered content.\n\n",
    "Try again, or a simpler request; or enable a larger ",
    "model, one tagged **DP** (document processing) or ",
    "**coding** for code (`/models` in the terminal, ",
    "**Settings → Models** in Studio)."
);

// ---------------------------------------------------------------- small utils

/// Python truthiness of an optional string.
fn truthy(s: &Option<String>) -> bool {
    s.as_deref().is_some_and(|s| !s.is_empty())
}

/// The capability the final answer is written on.  Programming work goes to
/// a `coding` model whether or not tools gathered context; anything else with
/// gathered context goes to the best `general` model, and a bare chat stays on
/// the default model.  The planner's other picks (`dp`, `vision`, …) choose
/// the model for its steps, not for the answer.
pub fn response_capability(context: &Option<String>, classified: Option<&str>) -> Option<ModelCapability> {
    if classified == Some(ModelCapability::Coding.value()) {
        Some(ModelCapability::Coding)
    } else if truthy(context) {
        Some(ModelCapability::General)
    } else {
        None
    }
}

/// Python truthiness of an optional list of asset ids.
fn has_items(list: &Option<Vec<String>>) -> bool {
    list.as_ref().is_some_and(|l| !l.is_empty())
}

/// The first `n` characters of `s` (Python `s[:n]`).
fn prefix_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Python `repr()` of a string, for log lines that used `%r`.
pub(crate) fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `%r` of an optional JSON value (`None` for missing/null).
fn py_repr_value(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => "None".to_string(),
        Some(Value::String(s)) => py_repr(s),
        Some(other) => other.to_string(),
    }
}

/// `serde_json` formatter reproducing Python's `json.dumps` defaults:
/// `", "`/`": "` separators and `ensure_ascii=True`.
struct PyJsonFormatter;

impl serde_json::ser::Formatter for PyJsonFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first { Ok(()) } else { writer.write_all(b", ") }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first { Ok(()) } else { writer.write_all(b", ") }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }

    fn write_string_fragment<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> std::io::Result<()> {
        for ch in fragment.chars() {
            if ch.is_ascii() {
                writer.write_all(&[ch as u8])?;
            } else {
                let mut units = [0u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    write!(writer, "\\u{:04x}", unit)?;
                }
            }
        }
        Ok(())
    }
}

/// Python `json.dumps(value)` (default separators, `ensure_ascii=True`).
pub fn py_json_dumps(value: &Value) -> String {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PyJsonFormatter);
    serde::Serialize::serialize(value, &mut ser).expect("serializing a JSON value cannot fail");
    String::from_utf8(out).expect("json output is UTF-8")
}

/// `f"msg-{uuid.uuid4().hex[:8]}"`.
fn new_message_id() -> String {
    format!("msg-{}", &uuid::Uuid::new_v4().simple().to_string()[..8])
}

/// The message id the client asked for, or a fresh one.
fn message_id_for(request: &ChatCompletionRequest) -> String {
    match &request.message_id {
        Some(id) if !id.is_empty() => id.clone(),
        _ => new_message_id(),
    }
}

/// One `chat.completion.chunk` SSE frame with a single choice.
fn chunk_frame(model: &str, delta: DeltaContent, finish_reason: Option<&str>) -> String {
    let chunk = ChatCompletionChunk::new(
        model,
        vec![ChunkChoice { index: 0, delta, finish_reason: finish_reason.map(str::to_string) }],
    );
    format!(
        "data: {}\n\n",
        serde_json::to_string(&chunk).expect("chunk serializes")
    )
}

fn content_delta(content: impl Into<String>) -> DeltaContent {
    DeltaContent { role: None, content: Some(content.into()) }
}

/// The content of the last user message in `messages`.
fn last_user_content(messages: &[ChatMessage]) -> Option<String> {
    messages.iter().rev().find(|m| m.role == Role::User).and_then(|m| m.content.clone())
}

/// `[name](path)` for a written file — forward slashes in the URL so the
/// Markdown parser doesn't treat backslashes as escape characters (e.g.
/// `\src` → `src` on Windows).
fn file_link(written_path: &str) -> String {
    let name = Path::new(written_path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let posix =
        if cfg!(windows) { written_path.replace('\\', "/") } else { written_path.to_string() };
    format!("[{name}]({posix})")
}

/// What an auto-write put on disk, appended to the answer: a link per file
/// written, and the files left alone because they already existed.
pub fn write_summary(outcome: &WriteOutcome) -> String {
    let mut summary = String::new();
    match outcome.written.as_slice() {
        [] => {}
        [path] => summary.push_str(&format!("\n\n{}", file_link(path))),
        paths => {
            summary.push_str(&format!("\n\nWrote {} files:\n", paths.len()));
            for path in paths {
                summary.push_str(&format!("\n- {}", file_link(path)));
            }
        }
    }
    if !outcome.skipped.is_empty() {
        let names: Vec<String> = outcome.skipped.iter().map(|p| format!("`{p}`")).collect();
        summary.push_str(&format!("\n\nLeft unchanged (already exist): {}", names.join(", ")));
    }
    summary
}

/// The message of a caught panic payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "worker panicked".to_string()
    }
}

/// Run blocking `f` on the runtime's blocking pool (`run_in_executor(None,
/// ...)`), carrying the caller's project binding into the worker.  A panic in
/// `f` resumes on the caller, as the Python exception would have propagated.
async fn run_blocking<R, F>(f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    match tokio::task::spawn_blocking(bind(f)).await {
        Ok(r) => r,
        Err(e) => match e.try_into_panic() {
            Ok(payload) => std::panic::resume_unwind(payload),
            Err(e) => panic!("blocking task did not complete: {e}"),
        },
    }
}

/// Fire-and-forget `f` on a worker thread (`loop.run_in_executor(None, f)`
/// whose future nobody awaits), bound to the caller's project.
fn spawn_detached_blocking<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    let f = bind(f);
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            drop(handle.spawn_blocking(f));
        }
        Err(_) => {
            std::thread::spawn(f);
        }
    }
}

/// Run `f` on the single-worker [`EXECUTOR`] and await its result.
async fn run_on_executor<R, F>(f: F) -> anyhow::Result<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let (tx, rx) = tokio::sync::oneshot::channel();
    EXECUTOR.submit(move || {
        let _ = tx.send(f());
    });
    rx.await.map_err(|_| anyhow::anyhow!("executor worker exited without a result"))
}

// ---------------------------------------------------------------- helpers

/// Resolve a leading `/<skill-name>` slash-command against `skills`.
///
/// Returns `(Some(skill_name), cleaned_message)` when the prompt opens with a
/// slash-command that names one of the project's skills — matched case- and
/// separator-insensitively, so `/fund-info` hits a `fund_info` skill. The
/// remainder of the prompt (after the token) becomes the message; when the
/// user typed only the token, the skill's own body stands in.  It used to be a
/// one-line `Run the '<name>' skill.`, but tool arguments are extracted from
/// the message (never the skill body — see `StepExecutor`), so a bare
/// `/morning` handed its tools the skill's *name* as their input.
///
/// Returns `(None, message)` unchanged when there is no leading slash-command
/// or it names no known skill, so an ordinary prompt that merely starts with a
/// path (`/etc/hosts …`) is left untouched.
pub fn resolve_requested_skill(message: &str, skills: &[Skill]) -> (Option<String>, String) {
    if message.is_empty() {
        return (None, message.to_string());
    }
    let Some(caps) = SLASH_SKILL_RE.captures(message) else {
        return (None, message.to_string());
    };
    let token = caps.get(1).map_or("", |m| m.as_str());
    let rest = caps.get(2).map_or("", |m| m.as_str());
    let norm = |s: &str| SKILL_NAME_SEPARATORS_RE.replace_all(&s.to_lowercase(), "").into_owned();
    let wanted = norm(token);
    for s in skills {
        if norm(&s.name) == wanted {
            let rest = rest.trim();
            let cleaned = if !rest.is_empty() {
                rest.to_string()
            } else {
                let body = SkillProcessor::strip_frontmatter(&s.content);
                if !body.is_empty() { body } else { format!("Run the '{}' skill.", s.name) }
            };
            return (Some(s.name.clone()), cleaned);
        }
    }
    (None, message.to_string())
}

/// Extract the raw tool payload from the orchestrator's wrapped context.
pub fn extract_tool_content(orchestrator_context: &str) -> String {
    if orchestrator_context.is_empty() {
        return String::new();
    }
    let marker = "TOOL RESULTS:\n";
    match orchestrator_context.find(marker) {
        None => orchestrator_context.trim().to_string(),
        Some(start) => orchestrator_context[start + marker.len()..].trim().to_string(),
    }
}

/// True when `exc` is generation running out of device memory.
pub fn is_decode_oom(exc: &dyn std::fmt::Display) -> bool {
    DECODE_OOM_RE.is_match(&exc.to_string())
}

/// A user turn asking the model to finish an answer it was cut off in.
///
/// Appended to the original prompt, where it merges into the last user turn
/// (`create_messages_from_prompt` folds consecutive same-role turns).  There
/// is no assistant prefill to resume from, so the partial answer is quoted
/// back and the model is told to carry on from its last word.
pub fn continuation_turn(partial: &str) -> String {
    format!(
        "<user>You were interrupted while answering. The user has already \
         received this beginning of your answer:\n\n\
         {}\n\n\
         Continue from exactly where it stops. Do not repeat any of it, and do \
         not mention the interruption.</user>",
        escape_role_tags(partial)
    )
}

/// Escape literal role tags in `text` so they don't break the regex in
/// `create_messages_from_prompt`.  Fetched web content can contain `<user>` or
/// `</user>` (e.g. HTML/chat tutorials); leaving them unescaped truncates the
/// extracted prompt catastrophically and makes the model emit no output.
pub fn escape_role_tags(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let mut text = text.to_string();
    for (tag, esc) in TAG_ESCAPE {
        if text.contains(tag) {
            text = text.replace(tag, esc);
        }
    }
    text
}

/// Convert OpenAI-style messages to the prompt format expected by the LLM.
///
/// When `context` is provided (pre-formatted by the orchestrator), it is
/// injected into the last user message as-is.  When no orchestrator context
/// is available the message is sent without auxiliary context so the model
/// answers from its own knowledge rather than mistaking directory listings
/// or project metadata for a restrictive knowledge base.
pub fn format_messages_to_prompt(
    messages: &[ChatMessage],
    context: Option<&str>,
    cwd: Option<&str>,
    project_info: Option<&str>,
    current_date: Option<&str>,
) -> String {
    // Accepted for signature compatibility; not part of the prompt.
    let _ = (cwd, project_info);
    let mut prompt_parts: Vec<String> = Vec::new();

    let mut system_content = SYSTEM_PROMPT.to_string();
    for msg in messages {
        if msg.role == Role::System
            && let Some(content) = msg.content.as_deref().filter(|c| !c.is_empty())
        {
            system_content.push('\n');
            system_content.push_str(content);
        }
    }
    prompt_parts.push(format!("<system>{system_content}</system>"));

    // The current date is runtime grounding the model was never trained on, so
    // it rides with the current user turn (next to the question) rather than
    // buried in the system prompt where a small model is unlikely to attend to it.
    let date_str = match current_date.filter(|d| !d.is_empty()) {
        Some(d) => d.to_string(),
        None => resolve_current_date(None, None),
    };
    let context = context.filter(|c| !c.is_empty());

    let mut context_injected = false;
    let last_user_idx = messages.iter().rposition(|m| m.role == Role::User);

    for (i, msg) in messages.iter().enumerate() {
        match msg.role {
            Role::System => continue,
            Role::Assistant => {
                prompt_parts.push(format!(
                    "<assistant>{}</assistant>",
                    escape_role_tags(msg.content.as_deref().unwrap_or(""))
                ));
            }
            Role::User => {
                let mut user_parts: Vec<String> = Vec::new();
                let content = msg.content.as_deref().filter(|c| !c.is_empty());

                // Put the user's question FIRST so the model always sees a clear
                // question, then the tool/document context.  Otherwise a short
                // question (e.g. just a URL) gets buried after a large context
                // block and small models emit no response.
                if let Some(content) = content {
                    user_parts.push(escape_role_tags(content));
                }

                if !context_injected && Some(i) == last_user_idx {
                    user_parts.push(format!("[Current date: {date_str}]"));
                    if let Some(context) = context {
                        user_parts.push(escape_role_tags(context));
                        // ...and repeat it at the END. Small models weight the
                        // tail of the prompt most heavily, and the context block
                        // closes on bare directives.  With thousands of tokens
                        // between the question and the tail, the default model
                        // reads those directives as the thing to respond to and
                        // acknowledges them instead of answering. Closing the
                        // turn with the question makes the last thing it reads
                        // the thing it has to answer.
                        if let Some(content) = content {
                            user_parts.push(format!(
                                "Using the information above, answer this question: {}",
                                escape_role_tags(content)
                            ));
                        }
                    }
                    context_injected = true;
                }

                prompt_parts.push(format!("<user>{}</user>", user_parts.join("\n\n")));
            }
        }
    }

    prompt_parts.join("\n")
}

/// The [`PHRASE_LEN`]-word runs in `text` that could evidence attribution.
///
/// Normalised for comparison, and filtered: a run has to carry at least
/// [`MIN_CONTENT_WORDS`] words that are not grammar.  Both sides of the
/// comparison are built this way, so a run only ever matches a run that also
/// qualified.
pub fn phrases(text: &str) -> HashSet<String> {
    let lowered = text.to_lowercase();
    let words: Vec<&str> = WORD_RE
        .find_iter(&lowered)
        .map(|m| m.as_str().trim_matches(|c| matches!(c, '.' | ',' | '%' | '/' | '-')))
        .filter(|w| !w.is_empty())
        .collect();
    let mut out = HashSet::new();
    if words.len() >= PHRASE_LEN {
        for run in words.windows(PHRASE_LEN) {
            let content_words = run.iter().filter(|w| !STOPWORDS.contains(*w)).count();
            if content_words < MIN_CONTENT_WORDS {
                continue;
            }
            out.insert(run.join(" "));
        }
    }
    for span in CONTINUOUS_RE.find_iter(&lowered) {
        let chars: Vec<char> = span.as_str().chars().collect();
        if chars.len() >= CHAR_GRAM {
            for gram in chars.windows(CHAR_GRAM) {
                out.insert(gram.iter().collect());
            }
        }
    }
    out
}

/// Python `str(float)` for a finite, non-integral value.
fn py_float_str(value: f64) -> String {
    let abs = value.abs();
    if abs != 0.0 && !(1e-4..1e16).contains(&abs) {
        // Python switches to exponent notation here, with at least two
        // exponent digits ("1e-05").
        let s = format!("{value:e}");
        if let Some((mantissa, exp)) = s.split_once('e') {
            let (sign, digits) = match exp.strip_prefix('-') {
                Some(d) => ("-", d),
                None => ("+", exp),
            };
            return format!("{mantissa}e{sign}{digits:0>2}");
        }
        return s;
    }
    format!("{value}")
}

/// Precise numbers in `text`, normalised so formatting doesn't matter.
///
/// "$1,000.00" in a document and a bare "1000" in the answer are the same
/// figure, so both reduce to `"1000"`.  These survive rewording: a translated
/// or heavily paraphrased answer keeps no phrasing from its source but still
/// carries its numbers.
pub fn figures(text: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    for raw in NUMBER_RE.find_iter(text) {
        let core = raw.as_str().trim_end_matches(['.', ',']);
        let significant = core.chars().filter(|c| *c != '.' && *c != ',').count();
        if significant < FIGURE_MIN_DIGITS {
            continue;
        }
        let digits = core.replace(',', "");
        let Ok(value) = digits.parse::<f64>() else {
            continue;
        };
        if !value.is_finite() {
            // A digit run too long for a float (overflows to inf) is still a
            // figure worth matching — normalise it by hand.
            if digits.contains('.') {
                out.insert(digits.trim_end_matches('0').trim_end_matches('.').to_string());
            } else {
                out.insert(digits);
            }
            continue;
        }
        if value == value.trunc() {
            out.insert(format!("{value:.0}"));
        } else {
            out.insert(py_float_str(value));
        }
    }
    out
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
        Self {
            blank: response.trim().is_empty(),
            phrases: phrases(response),
            figures: figures(response),
        }
    }

    /// True when the answer shows evidence of having used `source`.
    ///
    /// Conservative in the one direction that matters: when there is nothing
    /// to check (no source text, no answer), the source is kept.  Crediting a
    /// source the answer never used is the failure being prevented here, but
    /// silently dropping a real one is also a bug — so a reworded answer that
    /// keeps none of the source's phrasing can still qualify on a shared
    /// figure.
    pub fn drew_on(&self, source: &str) -> bool {
        if self.blank || source.is_empty() {
            return true;
        }
        let source_phrases = phrases(source);
        if source_phrases.is_empty() {
            return true;
        }
        if !source_phrases.is_disjoint(&self.phrases) {
            return true;
        }
        !figures(source).is_disjoint(&self.figures)
    }
}

/// `text` split into overlapping windows the embedder can take whole.
///
/// Sampled evenly when a text yields more than [`MAX_WINDOWS`], not truncated
/// to the first few: the passage an answer drew on is as likely to be on the
/// last page of a document as the first.
pub fn windows(text: &str) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    if len <= WINDOW_CHARS {
        return vec![text.to_string()];
    }
    let step = WINDOW_CHARS - WINDOW_OVERLAP;
    let mut starts: Vec<usize> = (0..len.saturating_sub(WINDOW_OVERLAP)).step_by(step).collect();
    if starts.len() > MAX_WINDOWS {
        let last = (starts.len() - 1) as f64;
        let mut picked: Vec<usize> = (0..MAX_WINDOWS)
            .map(|i| {
                let idx = (i as f64 * last / (MAX_WINDOWS - 1) as f64).round_ties_even() as usize;
                starts[idx]
            })
            .collect();
        picked.sort_unstable();
        picked.dedup();
        starts = picked;
    }
    starts
        .into_iter()
        .map(|s| chars[s..(s + WINDOW_CHARS).min(len)].iter().collect())
        .collect()
}

/// Read a BLOB carried in a citation dict (see [`CitationDict`]): an array of
/// byte values, or a base64 string.  `None` for `null` or anything else.
fn value_to_blob(value: &Value) -> Option<Vec<u8>> {
    match value {
        Value::Array(items) => items
            .iter()
            .map(|v| v.as_u64().filter(|b| *b <= 255).map(|b| b as u8))
            .collect(),
        Value::String(s) => {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD.decode(s).ok()
        }
        _ => None,
    }
}

/// Store a BLOB in a citation dict (array of byte values).
fn blob_to_value(blob: &[u8]) -> Value {
    Value::Array(blob.iter().map(|b| Value::from(*b)).collect())
}

/// A JSON list field of a citation (`citation.get(key) or []`).
fn list_field(citation: &CitationDict, key: &str) -> Vec<Value> {
    match citation.get(key) {
        Some(Value::Array(items)) => items.clone(),
        _ => Vec::new(),
    }
}

/// A string field of a citation, `""` when missing or not a string.
fn str_field<'a>(citation: &'a CitationDict, key: &str) -> &'a str {
    citation.get(key).and_then(Value::as_str).unwrap_or("")
}

/// `(vectors, texts)` — the passages this citation is judged on.
///
/// Split by what still costs something.  Retrieval carries each matched
/// chunk's vector out of the index in `_vectors`, so most citations arrive
/// already encoded and contribute nothing to the encode batch.  Only the
/// leftovers — chunks whose vector is missing (the TF-IDF fallback path
/// stores none) and citations built outside retrieval that have an extract
/// but no chunks (web results, document fan-out) — come back as text.
pub fn citation_sources(citation: &CitationDict) -> (Vec<Array1<f32>>, Vec<String>) {
    let chunks = list_field(citation, "_chunks");
    let blobs = list_field(citation, "_vectors");
    let mut vectors = Vec::new();
    let mut texts = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let blob = blobs.get(i).and_then(value_to_blob).filter(|b| !b.is_empty());
        if let Some(blob) = blob {
            vectors.push(embeddings::blob_to_embedding(&blob));
        } else if let Some(chunk) = chunk.as_str().filter(|c| !c.trim().is_empty()) {
            texts.extend(windows(chunk));
        }
    }
    if !vectors.is_empty() || !texts.is_empty() {
        return (vectors, texts);
    }
    (Vec::new(), windows(str_field(citation, "_extract")))
}

/// The source *text* for the fallback check, which cannot use vectors.
pub fn lexical_sources(citation: &CitationDict) -> Vec<String> {
    let chunks: Vec<String> = list_field(citation, "_chunks")
        .iter()
        .filter_map(Value::as_str)
        .filter(|c| !c.trim().is_empty())
        .map(str::to_string)
        .collect();
    if !chunks.is_empty() {
        return chunks;
    }
    let extract = str_field(citation, "_extract");
    if extract.is_empty() { Vec::new() } else { vec![extract.to_string()] }
}

/// Answer-to-source cosine distance per citation, closest passage wins.
///
/// `None` — for the whole batch or for an individual citation — means "could
/// not be judged", never "unrelated": the caller falls back to the lexical
/// check rather than dropping a source it failed to measure.
///
/// One `encode_texts` call, and in the ordinary case it encodes only the
/// answer.  The source side was embedded when the documents were indexed and
/// read back a moment ago to rank this very search — re-encoding it here would
/// be paying twice for the same vectors, after the answer has already
/// streamed, which is latency the user watches.
pub fn semantic_distances(
    citations: &[CitationDict],
    response: &str,
    base_dir: Option<&str>,
) -> Option<Vec<Option<f64>>> {
    let answer_windows = windows(response);
    if answer_windows.is_empty() {
        return None;
    }

    let mut matrices: Vec<Vec<Array1<f32>>> = Vec::new(); // per citation
    let mut pending: Vec<String> = Vec::new(); // source text with no vector
    let mut owners: Vec<usize> = Vec::new(); // which citation each belongs to
    for (i, citation) in citations.iter().enumerate() {
        let (vectors, texts) = citation_sources(citation);
        matrices.push(vectors);
        owners.extend(std::iter::repeat_n(i, texts.len()));
        pending.extend(texts);
    }

    let mut batch = answer_windows.clone();
    batch.extend(pending.iter().cloned());
    let encoded = embeddings::encode_texts(&batch, base_dir, Some(VERIFY_ENCODE_TIMEOUT_S))?;
    if encoded.nrows() != answer_windows.len() + pending.len() {
        // No embedding model on this install, or encoding failed.
        return None;
    }

    if !pending.is_empty() {
        // Everything here should have been encoded during generation; anything
        // left is a source `prepare_citation_sources` could not reach, and it
        // is being paid for in the pause after the answer.
        tracing::info!(
            target: LOG_TARGET,
            "[citations] Encoded {} source passage(s) after the answer",
            pending.len()
        );
    }

    let n_answer = answer_windows.len();
    for (j, owner) in owners.iter().enumerate() {
        matrices[*owner].push(encoded.row(n_answer + j).to_owned());
    }

    let dim = encoded.ncols();
    let mut out = Vec::with_capacity(matrices.len());
    for matrix in matrices {
        // An index written by a different embedding model yields vectors the
        // answer cannot be compared against.  Decline to judge that source
        // rather than crash the response it belongs to.
        let matrix: Vec<Array1<f32>> = matrix.into_iter().filter(|v| v.len() == dim).collect();
        if matrix.is_empty() {
            out.push(None);
            continue;
        }
        let mut source = Array2::<f32>::zeros((matrix.len(), dim));
        for (r, v) in matrix.iter().enumerate() {
            source.row_mut(r).assign(v);
        }
        let mut best: Option<f64> = None;
        for r in 0..n_answer {
            let distances = embeddings::cosine_distances(encoded.row(r), source.view());
            // numpy's ``.min()`` propagates NaN.
            let min = distances.iter().fold(f32::INFINITY, |acc, d| {
                if d.is_nan() || acc.is_nan() { f32::NAN } else { acc.min(*d) }
            }) as f64;
            best = match best {
                None => Some(min),
                Some(cur) if min < cur => Some(min),
                keep => keep,
            };
        }
        out.push(best);
    }
    Some(out)
}

/// True when `response` actually drew on this citation's source.
///
/// Retrieval returning a document does not mean the answer used it: nearest
/// neighbour always returns *something*.  The lexical check only — the
/// single-citation entry point, kept for callers and tests that have no
/// embedder to hand.  [`verified_citations`] is the real path and compares in
/// embedding space.  Citations carrying no extract are always kept.
pub fn citation_used(citation: &CitationDict, response: &str) -> bool {
    AnswerEvidence::new(response).drew_on(str_field(citation, "_extract"))
}

/// Deduplicate by document and drop sources the answer never used.
///
/// Computed, not asked: "is this answer about the passages this document was
/// retrieved for?" — the question retrieval already answers about a chunk and
/// the user's prompt, asked in the other direction, with the same pieces
/// (`encode_texts`, `cosine_distances`, retrieval's own ceiling and a margin).
///
/// `embed = false` forces the lexical fallback.  That is for the interrupt
/// path, which runs while the client is being torn down.
///
/// Runs while a response is being finished, so it must not be able to fail:
/// a bug in source checking should cost a citation, never the answer the user
/// already saw streamed.  On error the sources are kept as-is, deduplicated.
pub fn verified_citations(
    citations: &[CitationDict],
    response: &str,
    base_dir: Option<&str>,
    embed: bool,
) -> Vec<CitationDict> {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
        verify_citations(citations, response, base_dir, embed)
    }));
    let out = match outcome {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => {
            tracing::error!(
                target: LOG_TARGET,
                "[citations] Verification failed — keeping all sources: {}", e
            );
            dedupe_citations(citations)
        }
        Err(payload) => {
            tracing::error!(
                target: LOG_TARGET,
                "[citations] Verification failed — keeping all sources: {}",
                panic_message(payload.as_ref())
            );
            dedupe_citations(citations)
        }
    };
    // The one number to watch: this is wall-clock the user spends looking at a
    // finished answer with no sources under it.
    tracing::info!(
        target: LOG_TARGET,
        "[citations] Verified in {:.2}s",
        started.elapsed().as_secs_f64()
    );
    out
}

/// The deduplication key of a citation: its `document_id`, else its
/// `document_name`.
fn citation_key(citation: &CitationDict) -> String {
    let key = match citation.get("document_id") {
        Some(v) if !v.is_null() => v.clone(),
        _ => citation.get("document_name").cloned().unwrap_or(Value::Null),
    };
    key.to_string()
}

/// A citation with the private `_` keys stripped.
fn public_citation(citation: &CitationDict) -> CitationDict {
    citation
        .iter()
        .filter(|(k, _)| !k.starts_with('_'))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

/// One entry per document, with the private `_` keys stripped.
pub fn dedupe_citations(citations: &[CitationDict]) -> Vec<CitationDict> {
    unique_citations(citations).iter().map(public_citation).collect()
}

/// One entry per document, private keys intact, retrieval order kept.
pub fn unique_citations(citations: &[CitationDict]) -> Vec<CitationDict> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for c in citations {
        if seen.insert(citation_key(c)) {
            out.push(c.clone());
        }
    }
    out
}

/// The fallible core of [`verified_citations`].
pub fn verify_citations(
    citations: &[CitationDict],
    response: &str,
    base_dir: Option<&str>,
    embed: bool,
) -> anyhow::Result<Vec<CitationDict>> {
    let unique = unique_citations(citations);
    if unique.is_empty() {
        return Ok(Vec::new());
    }

    let measured_distances =
        if embed { semantic_distances(&unique, response, base_dir) } else { None };

    let (distances, keep, cutoff): (Vec<Option<f64>>, Vec<bool>, f64) = match measured_distances
    {
        None => {
            // No embedder available, or the caller asked not to load one —
            // fall back to the lexical evidence check so verification degrades
            // to a weaker test rather than switching off.
            tracing::info!(target: LOG_TARGET, "[citations] Scoring sources without embeddings");
            let evidence = AnswerEvidence::new(response);
            let keep = unique
                .iter()
                .map(|c| {
                    let passages = lexical_sources(c);
                    passages.is_empty() || passages.iter().any(|p| evidence.drew_on(p))
                })
                .collect();
            (vec![None; unique.len()], keep, CITATION_MAX_DISTANCE)
        }
        Some(distances) => {
            // Retrieval's own selection criteria, applied to the finished
            // answer: a hard ceiling, then a margin around the closest source.
            // The margin is what stops a handful of documents that merely
            // cleared the ceiling riding along behind the one that actually
            // produced the answer.
            let mut cutoff = CITATION_MAX_DISTANCE;
            let mut closest: Option<f64> = None;
            for d in distances.iter().flatten() {
                closest = match closest {
                    None => Some(*d),
                    Some(cur) if *d < cur => Some(*d),
                    keep => keep,
                };
            }
            if let Some(closest) = closest {
                let margin = closest + CITATION_CUTOFF_MARGIN;
                if margin < cutoff {
                    cutoff = margin;
                }
            }
            let keep = distances.iter().map(|d| d.is_none_or(|d| d <= cutoff)).collect();
            (distances, keep, cutoff)
        }
    };

    let mut out = Vec::new();
    for ((citation, distance), kept) in unique.iter().zip(distances).zip(keep) {
        // Logged for every source, kept or not: the two thresholds are the
        // tunables here and these lines are what they should be tuned against.
        tracing::info!(
            target: LOG_TARGET,
            "[citations] {} distance={} cutoff={:.3} — {}",
            py_repr_value(citation.get("document_name")),
            distance.map_or_else(|| "n/a".to_string(), |d| format!("{d:.3}")),
            cutoff,
            if kept { "keeping" } else { "dropping, answer is not about it" }
        );
        if kept {
            out.push(public_citation(citation));
        }
    }
    Ok(out)
}

/// Where the embedding model is cached — the workspace, as retrieval uses.
pub fn embedding_base_dir(model_service: &dyn ChatService) -> Option<String> {
    model_service.workspace_dir()
}

/// Embed every source now, while the answer is still being generated.
///
/// Verification compares two things, and only one of them depends on the
/// answer.  The sources are known the moment retrieval returns — before the
/// first token — so waiting until the answer is finished to encode them puts
/// all of that work in the one place where the user is watching a spinner.
///
/// Normalises as it goes: every citation ends up as `_chunks` plus an
/// index-aligned `_vectors`, whichever shape it arrived in.
///
/// Best effort throughout.  A failure here costs the pause it was meant to
/// avoid and nothing else — verification encodes whatever it finds missing,
/// and falls back to the lexical check if there is no model at all.
pub fn prepare_citation_sources(citations: &SharedCitations, model_service: &dyn ChatService) {
    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let base_dir = embedding_base_dir(model_service);
        let mut pending: Vec<String> = Vec::new();
        let mut slots: Vec<(usize, usize)> = Vec::new();
        {
            let mut guard = citations.lock();
            for (ci, citation) in guard.iter_mut().enumerate() {
                let mut chunks = list_field(citation, "_chunks");
                let mut vectors = list_field(citation, "_vectors");
                if chunks.is_empty() {
                    // A citation built outside retrieval — a web page, a
                    // fan-out extract.  Window it once here so verification
                    // never has to.
                    chunks = windows(str_field(citation, "_extract"))
                        .into_iter()
                        .map(Value::String)
                        .collect();
                    vectors = Vec::new();
                }
                if vectors.len() < chunks.len() {
                    vectors.resize(chunks.len(), Value::Null);
                }
                for (i, chunk) in chunks.iter().enumerate() {
                    if vectors[i].is_null() {
                        slots.push((ci, i));
                        pending.push(chunk.as_str().unwrap_or("").to_string());
                    }
                }
                citation.insert("_chunks".to_string(), Value::Array(chunks));
                citation.insert("_vectors".to_string(), Value::Array(vectors));
            }
        }

        if pending.is_empty() {
            // Every source came out of the index with its vector.  Still load
            // the model, since the answer has to go through it either way.
            embeddings::warmup(base_dir.as_deref());
            return;
        }

        let Some(encoded) =
            embeddings::encode_texts(&pending, base_dir.as_deref(), Some(VERIFY_ENCODE_TIMEOUT_S))
        else {
            return;
        };
        {
            let mut guard = citations.lock();
            for ((ci, i), row) in slots.iter().zip(encoded.rows()) {
                let blob = blob_to_value(&embeddings::embedding_to_blob(row));
                if let Some(Value::Array(vectors)) =
                    guard.get_mut(*ci).and_then(|c| c.get_mut("_vectors"))
                    && let Some(slot) = vectors.get_mut(*i)
                {
                    *slot = blob;
                }
            }
        }
        tracing::info!(
            target: LOG_TARGET,
            "[citations] Pre-embedded {} source passage(s) during generation",
            pending.len()
        );
    }));
    if let Err(payload) = outcome {
        tracing::warn!(
            target: LOG_TARGET,
            "[citations] Source pre-embedding failed: {}",
            panic_message(payload.as_ref())
        );
    }
}

/// [`verified_citations`] off the async runtime.
///
/// It runs the embedder, which must not stall the runtime's worker threads.
/// The blocking pool, not [`EXECUTOR`]: that pool has a single worker, held
/// for the duration of a generation by the thread producing the stream.
///
/// `prepared` is the [`prepare_citation_sources`] task started when
/// generation began.  Waited on rather than raced: it is writing the very
/// `_vectors` this is about to read.
pub async fn verified_citations_async(
    citations: SharedCitations,
    response: String,
    model_service: Arc<dyn ChatService>,
    prepared: Option<tokio::task::JoinHandle<()>>,
) -> Vec<CitationDict> {
    if let Some(prepared) = prepared {
        // Already logged there; verification encodes what is missing.
        let _ = prepared.await;
    }
    let base_dir = embedding_base_dir(&*model_service);
    let snapshot = citations.lock().clone();
    let fallback = snapshot.clone();
    match tokio::task::spawn_blocking(bind(move || {
        verified_citations(&snapshot, &response, base_dir.as_deref(), true)
    }))
    .await
    {
        Ok(out) => out,
        Err(_) => dedupe_citations(&fallback),
    }
}

/// The project a request should be bound to: the `project_id` it named, when
/// the workspace holds it.  Logs (and returns `None`) when it does not.
pub fn request_project(
    request: &ChatCompletionRequest,
    model_service: &dyn ChatService,
) -> Option<String> {
    let project_id = request.project_id.as_deref().filter(|p| !p.is_empty())?;
    if let Some(ids) = model_service.workspace_project_ids()
        && !ids.iter().any(|id| id == project_id)
    {
        tracing::warn!(
            target: LOG_TARGET,
            "Project not found: {}, using current project", project_id
        );
        return None;
    }
    Some(project_id.to_string())
}

/// Bind this request to the project it named, for the rest of its life.
///
/// This replaces `set_current_project` as the way a chat chooses its project.
/// The old call flipped a `current` flag shared by every request in the
/// process, so two chats on different projects raced.  The binding lives in
/// the request's own context instead (see [`crate::workspace::context`]); work
/// handed to a thread pool carries it explicitly via [`bind`].
///
/// Nothing is reset: the binding dies with the request's scope.  Nor is the
/// `current` flag touched — `/v1/models/load` remains the one call that does.
/// The endpoint itself binds through [`scope_project`] so the binding cannot
/// leak into whatever else runs on the same worker thread.
pub fn bind_request_project(request: &ChatCompletionRequest, model_service: &dyn ChatService) {
    if let Some(project_id) = request_project(request, model_service) {
        let _token = set_active_project(Some(&project_id));
    }
}

/// The working directory for this request.
///
/// The client's `cwd` is honoured only inside the folder the project is
/// bound to.  A project with no folder yet — or whose folder has gone, moved
/// or deleted — is bound to the `cwd` of the first turn that names a usable
/// one: not the home folder or above it, and not the framework's own state.
/// Anything else falls back to `project_dir`, so once a project has its
/// folder a client can't point the file tools anywhere outside it.
pub fn request_cwd(request: &ChatCompletionRequest, model_service: &dyn ChatService) -> Option<String> {
    let cwd = request.cwd.as_deref().filter(|c| !c.is_empty());
    let root = model_service.current_work_dir().filter(|root| Path::new(root).is_dir());
    let bound = match (cwd, root) {
        (Some(cwd), Some(root)) => crate::paths::dir_within(cwd, &root),
        (Some(cwd), None) => crate::paths::bindable_dir(cwd)
            .filter(|dir| !is_framework_state(dir, model_service))
            .filter(|dir| model_service.bind_work_dir(dir)),
        (None, _) => None,
    };
    bound.or_else(|| model_service.current_project_dir())
}

/// Whether `dir` lies in the framework's own workspace, which holds every
/// project's state and is never a folder to work in.
fn is_framework_state(dir: &str, model_service: &dyn ChatService) -> bool {
    model_service
        .workspace_dir()
        .and_then(|ws| std::fs::canonicalize(ws).ok())
        .is_some_and(|ws| Path::new(dir).starts_with(ws))
}

/// Resolve the current date from the client's `additional_message_params`.
///
/// Uses the browser's timezone + clock when supplied so date-aware answers
/// match the user's calendar; falls back to the server date otherwise.
pub fn request_current_date(request: &ChatCompletionRequest) -> String {
    let params = request.additional_message_params.as_ref();
    resolve_current_date(
        params.and_then(|p| p.user_date),
        params.and_then(|p| p.tz.as_deref()),
    )
}

/// Return a new message list where attachment asset IDs are resolved to
/// markdown.
///
/// For each user message containing `attachments`, the markdown content of
/// each referenced asset is prepended to the message text so the model can
/// see the full document context inline.
pub fn resolve_message_attachments(
    messages: Vec<ChatMessage>,
    model_service: &dyn ChatService,
) -> Vec<ChatMessage> {
    messages
        .into_iter()
        .map(|msg| {
            if msg.role == Role::User && has_items(&msg.attachments) {
                let ids = msg.attachments.clone().unwrap_or_default();
                let attachment_md = model_service.resolve_attachment_context(&ids);
                if !attachment_md.is_empty() {
                    let combined = [Some(attachment_md), msg.content.clone()]
                        .into_iter()
                        .flatten()
                        .filter(|s| !s.is_empty())
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    return ChatMessage { role: msg.role, content: Some(combined), attachments: None };
                }
            }
            msg
        })
        .collect()
}

/// Joined message text used for token counting.
fn joined_content(messages: &[ChatMessage]) -> String {
    messages.iter().map(|m| m.content.as_deref().unwrap_or("")).collect::<Vec<_>>().join("\n")
}

/// Trim oldest conversation turns when the message list exceeds the context
/// window.
///
/// Always keeps the first user message (original context) and the last user
/// message (the current question).  Removes the oldest assistant+user pairs
/// working forward from index 1 until the token count fits within
/// `max_context - reserved_tokens - context_tokens`.
///
/// `context_tokens` accounts for document_context, project_skills, and
/// mcp_context that will be injected into the final prompt.
///
/// `max_context` is the window of the model that will actually synthesize the
/// response; `None` means the default model's window.  Python defaults:
/// `reserved_tokens = 2048`, `context_tokens = 0`, `max_context = None`.
pub fn compact_messages(
    messages: Vec<ChatMessage>,
    model_service: &dyn ChatService,
    reserved_tokens: usize,
    context_tokens: usize,
    max_context: Option<usize>,
) -> Vec<ChatMessage> {
    let max_context = max_context.unwrap_or_else(|| model_service.get_max_context_tokens());
    if max_context == 0 {
        return messages;
    }

    let budget = max_context as i64 - reserved_tokens as i64 - context_tokens as i64;
    if budget <= 0 {
        return messages;
    }

    // Quick check — if we're already within budget, return as-is
    let used_tokens = model_service.count_tokens(&joined_content(&messages));

    if used_tokens as i64 <= budget {
        tracing::info!(
            target: LOG_TARGET,
            "Context window: {}/{} tokens used ({} budget, {} reserved) — no compaction needed",
            used_tokens, max_context, budget, reserved_tokens
        );
        return messages;
    }

    let original_count = messages.len();

    // Build a mutable copy
    let mut compacted = messages;

    // Remove oldest turns in pairs (assistant+user or user+assistant) starting
    // from index 1 so that roles keep alternating.  Index 0 (first user message
    // with context) and the last message are always kept.
    while compacted.len() > 2 {
        // Pop two messages at index 1 to remove a full turn pair
        compacted.remove(1);
        if compacted.len() > 2 {
            compacted.remove(1);
        }

        if model_service.count_tokens(&joined_content(&compacted)) as i64 <= budget {
            break;
        }
    }

    // Safety: ensure roles alternate after compaction.  If two consecutive
    // messages share the same role, merge them so the model never sees an
    // invalid sequence.
    let mut merged: Vec<ChatMessage> = Vec::with_capacity(compacted.len());
    for msg in compacted {
        match merged.last_mut() {
            Some(prev) if prev.role == msg.role => {
                let content = [prev.content.clone(), msg.content.clone()]
                    .into_iter()
                    .flatten()
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n");
                *prev = ChatMessage { role: msg.role, content: Some(content), attachments: None };
            }
            _ => merged.push(msg),
        }
    }
    let compacted = merged;

    let compacted_tokens = model_service.count_tokens(&joined_content(&compacted));
    let removed = original_count - compacted.len();
    tracing::info!(
        target: LOG_TARGET,
        "Context window: {}/{} tokens exceeded budget ({}), compacted {} messages — now {}/{} tokens",
        used_tokens, max_context, budget, removed, compacted_tokens, max_context
    );

    compacted
}

/// Prepend already-loaded conversation `history` to the request messages.
///
/// `history` is the session's recent turns (recency-based, alternation
/// enforced) loaded once by the caller via `get_relevant_history` and shared
/// with the orchestrator, so it is not re-queried here.  Prepending lets the
/// LLM reference earlier context (e.g. "create a table from that").
pub fn build_messages_with_history(
    request: &ChatCompletionRequest,
    history: &[HistoryTurn],
) -> Vec<ChatMessage> {
    if history.is_empty() {
        return request.messages.clone();
    }

    let mut out: Vec<ChatMessage> = history
        .iter()
        .map(|h| {
            let role_str = h.get("role").cloned().unwrap_or_default();
            let role: Role = serde_json::from_value(Value::String(role_str.clone()))
                .unwrap_or_else(|_| {
                    tracing::warn!(
                        target: LOG_TARGET,
                        "Unknown role {} in session history; treating it as user",
                        py_repr(&role_str)
                    );
                    Role::User
                });
            ChatMessage { role, content: h.get("content").cloned(), attachments: None }
        })
        .collect();
    out.extend(request.messages.iter().cloned());
    out
}

/// Free the auxiliary models once `work` (an abandoned worker's completion)
/// has finished.
///
/// Used by the timeout paths, which abandon their worker instead of awaiting
/// it: the models it is using cannot be closed until it stops touching them.
/// The unload itself goes onto a blocking worker thread — it closes llama.cpp
/// contexts and reclaims the GPU — and any failure is logged and swallowed:
/// cleanup must never surface to the client.
pub fn unload_when_idle<F>(work: F, model_service: Arc<dyn ChatService>)
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let task = async move {
        let _ = work.await;
        let unload = move || {
            let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
                model_service.unload_auxiliary_models(true)
            }));
            if let Err(payload) = outcome {
                tracing::warn!(
                    target: LOG_TARGET,
                    "[orchestrate] deferred model unload failed: {}",
                    panic_message(payload.as_ref())
                );
            }
        };
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                drop(handle.spawn_blocking(unload));
            }
            Err(e) => {
                tracing::warn!(
                    target: LOG_TARGET,
                    "[orchestrate] could not schedule model unload: {}", e
                );
            }
        }
    };
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            drop(handle.spawn(task));
        }
        Err(_) => {
            std::thread::spawn(move || {
                match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                    Ok(rt) => rt.block_on(task),
                    Err(e) => tracing::warn!(
                        target: LOG_TARGET,
                        "[orchestrate] could not schedule model unload: {}", e
                    ),
                }
            });
        }
    }
}

/// True when the model produced no meaningful content.
///
/// A response of just ```` ``` ```` (an unclosed code fence), whitespace, or
/// fewer than 3 alphanumeric characters means the model tried to open a
/// markdown block or stray token and hit EOS immediately.
pub fn is_degenerate(text: &str) -> bool {
    let stripped = text.trim().trim_matches('`').trim_matches('*').trim_matches('_').trim();
    if stripped.chars().count() < 3 {
        return true;
    }
    stripped.chars().filter(|c| c.is_alphanumeric()).count() < 3
}

// ---------------------------------------------------------------- streaming

/// What the producer thread hands the SSE generator.
enum QueueItem {
    /// A piece of the answer.
    Text(String),
    /// Generation failed with this message.
    Error(String),
    /// The stream is complete.
    End,
}

/// The producer's opening-buffer state, shared across a retry.
struct StreamState {
    collected: Vec<String>,
    buffered: bool,
    opening_chars: usize,
}

/// The sync side of [`stream_chunks`]: runs the generation on a worker thread
/// and puts its chunks on the queue.
struct Producer {
    svc: Arc<dyn ChatService>,
    prompt: String,
    model: String,
    capability: Option<ModelCapability>,
    cancelled: CancelEvent,
    orchestrator_context: Option<String>,
    last_user_msg: Option<String>,
    queue: UnboundedSender<QueueItem>,
    include_embedding: bool,
}

impl Producer {
    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    fn put(&self, item: QueueItem) {
        // The consumer may already be gone (client disconnect); nothing to do.
        let _ = self.queue.send(item);
    }

    /// Produce the stream, then free the models this thread just used.
    ///
    /// The unload has to happen *here*, not in the consumer's cleanup.
    /// `cancelled` is cooperative — it is only observed between tokens — so
    /// when the user interrupts, the consumer's cleanup runs while this thread
    /// is still inside `llama_sampler_sample`.  Freeing the model there pulls
    /// the llama.cpp context out from under a live sampler and segfaults the
    /// process.  Running it on this thread, after the generator is done,
    /// means nothing is mid-inference when the weights go away.
    fn run_sync_stream(self) {
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| self.stream_into_queue()));
        if let Err(payload) = outcome {
            let message = panic_message(payload.as_ref());
            tracing::error!(target: LOG_TARGET, "Error in sync stream: {}", message);
            self.put(QueueItem::Error(message));
            self.put(QueueItem::End);
        }
        // Frees the on-demand auxiliary models (vision + large agent), leaving
        // the orchestrator and the capability models resident between chats.
        //
        // The embedding model stays when there are sources to verify: this
        // runs the moment generation ends, which is the moment before
        // verification wants it.  Freeing it here would mean reloading the
        // GGUF seconds later, exactly where the user is waiting for
        // citations.  The consumer frees it once it is done.
        self.svc.unload_auxiliary_models(self.include_embedding);
    }

    /// One pass of the generation into the queue.
    ///
    /// Through the model client rather than the service directly: the default
    /// implementation is the same in-process call, but a project agent
    /// configured with `SLP_MODEL_HOST` reaches the shared device owner here
    /// instead.  The answer does not reason: by the time the answer is written
    /// the tool decision is made and its results are in the context, so a
    /// `<think>` block here only withholds the stream while it is produced.
    fn run(&self, prompt: &str, st: &mut StreamState) -> anyhow::Result<()> {
        let client = self.svc.model_client();
        let stream = client.stream(
            prompt,
            Some(&self.model),
            self.capability.map(|c| c.value()),
            Some(self.cancelled.clone()),
            Some(false),
        )?;
        for item in stream {
            let text = item?;
            if self.is_cancelled() {
                break;
            }
            if text.is_empty() {
                continue;
            }
            st.collected.push(text.clone());

            if !st.buffered {
                self.put(QueueItem::Text(text));
                continue;
            }

            // Still in the opening-buffer window.
            st.opening_chars += text.chars().filter(|c| c.is_alphanumeric()).count();
            if st.opening_chars >= ALNUM_STREAM_THRESHOLD {
                // Enough real content — flush buffered chunks and switch to
                // normal streaming mode for the remainder.
                for t in &st.collected {
                    self.put(QueueItem::Text(t.clone()));
                }
                st.buffered = false;
            }
        }
        Ok(())
    }

    /// Run the sync stream and put chunks in the queue.
    ///
    /// When tool context is present only the OPENING of the stream is
    /// buffered, so a degenerate output (small models often emit just ```)
    /// can be detected and replaced by a non-streaming retry.  Once enough
    /// real content is seen the buffer is flushed and the stream continues
    /// normally, so larger, capable models don't lose streaming UX.
    fn stream_into_queue(&self) {
        let mut st = StreamState {
            collected: Vec::new(),
            // Only buffer for tool-synthesis; regular chat streams immediately.
            buffered: self.orchestrator_context.is_some(),
            opening_chars: 0,
        };

        let result = match self.run(&self.prompt, &mut st) {
            Ok(()) => Ok(()),
            Err(e) if self.is_cancelled() || !is_decode_oom(&e) => Err(e),
            Err(e) => {
                // The device ran out of room mid-answer.  What else is
                // resident by now — the embedding model citation prep loads
                // alongside the answer, the vision/large/capability models of
                // earlier steps — is the room it needs, so free all of it and
                // try once more.  The response model reloads on demand with a
                // fresh context.
                tracing::warn!(
                    target: LOG_TARGET,
                    "[stream] Generation failed ({}) — evicting auxiliary models and retrying once",
                    e
                );
                match self.svc.model_client().unload_auxiliary_models(true) {
                    Err(unload_err) => Err(unload_err),
                    Ok(()) => {
                        if st.buffered {
                            // Nothing reached the client: start over.
                            st.collected.clear();
                            st.opening_chars = 0;
                            self.run(&self.prompt, &mut st)
                        } else {
                            // The client already has part of the answer and
                            // there is no taking it back, so ask for the rest.
                            let prompt = format!(
                                "{}{}",
                                self.prompt,
                                continuation_turn(&st.collected.concat())
                            );
                            self.run(&prompt, &mut st)
                        }
                    }
                }
            }
        };
        if let Err(e) = result {
            tracing::error!(target: LOG_TARGET, "Error in sync stream: {}", e);
            self.put(QueueItem::Error(e.to_string()));
            self.put(QueueItem::End);
            return;
        }

        if self.is_cancelled() {
            self.put(QueueItem::End);
            return;
        }

        let mut response = st.collected.concat();

        // If we already flushed the buffer we've been streaming — done.
        if !st.buffered {
            self.put(QueueItem::End);
            return;
        }

        let context = self.orchestrator_context.as_deref().filter(|c| !c.is_empty());
        if let Some(context) = context
            && is_degenerate(&response)
        {
            tracing::warn!(
                target: LOG_TARGET,
                "[stream] Degenerate response ({}) — retrying with simplified prompt",
                py_repr(&prefix_chars(&response, 40))
            );
            // Strip the TOOL RESULTS wrapper and its rules, ask the model to
            // answer with a minimal prompt structure.  Same-prompt retries
            // produce the same output at low temperature, so we only bother
            // with an input that's actually different.
            let raw_content = extract_tool_content(context);
            let mut retry = String::new();
            if !raw_content.is_empty() {
                let question = self
                    .last_user_msg
                    .as_deref()
                    .filter(|m| !m.is_empty())
                    .unwrap_or("Summarize the content below.");
                let simple_prompt = format!(
                    "<system>You are a helpful assistant.</system>\n<user>{}\n\nContent:\n{}</user>",
                    escape_role_tags(question),
                    escape_role_tags(&raw_content)
                );
                retry = match self.svc.model_client().generate(
                    &simple_prompt,
                    Some(&self.model),
                    self.capability.map(|c| c.value()),
                    None,
                    Some(false),
                ) {
                    Ok(text) => text,
                    Err(e) => {
                        tracing::error!(
                            target: LOG_TARGET,
                            "[stream] Simplified retry failed: {}", e
                        );
                        String::new()
                    }
                };
            }

            if !retry.is_empty() && !is_degenerate(&retry) {
                response = retry;
            } else {
                tracing::warn!(
                    target: LOG_TARGET,
                    "[stream] Simplified retry also degenerate ({}) — returning guidance message",
                    py_repr(&prefix_chars(&retry, 40))
                );
                response = NO_RESPONSE_GUIDANCE.to_string();
            }
        }

        if context.is_some() && !response.is_empty() {
            // Emit the (possibly retried) full response as one chunk.
            self.put(QueueItem::Text(response));
        }

        self.put(QueueItem::End);
    }
}

/// The streaming generator's `finally` block.
///
/// Runs exactly once: explicitly when the stream finishes on its own, or from
/// `Drop` when the client goes away mid-stream (Python's `aclose()`).  It must
/// not await anything — on the interrupt path there is nothing left to await
/// on — so the embedder unload is fired off rather than waited for.  Model
/// cleanup deliberately does NOT happen here — see
/// [`Producer::run_sync_stream`]: this runs on interrupt too, while the
/// producer thread may still be inside llama.cpp.
/// Raises a request's cancellation signal when dropped.
struct CancelOnDrop(CancelEvent);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

struct StreamFinally {
    svc: Arc<dyn ChatService>,
    cancelled: CancelEvent,
    citations: Option<SharedCitations>,
    parts: Vec<String>,
    verified: Option<Vec<CitationDict>>,
    session_id: Option<String>,
    message_id: String,
    done: bool,
}

impl StreamFinally {
    fn finish(&mut self) {
        if self.done {
            return;
        }
        self.done = true;

        // Signal cancellation so the background thread stops producing chunks
        self.cancelled.store(true, Ordering::SeqCst);

        // Save partial (or full) response to session history
        let full_response = self.parts.concat();
        tracing::info!(
            target: LOG_TARGET,
            "[stream] Streaming finished — {} chunk(s), {} char(s), preview={}",
            self.parts.len(),
            full_response.chars().count(),
            py_repr(&prefix_chars(&full_response, 200))
        );
        if self.verified.is_none()
            && let Some(citations) = &self.citations
            && !full_response.is_empty()
        {
            // Only reached when the stream did not finish on its own.  The
            // sources still get checked before being written to the session,
            // but with the lexical test: loading an embedding model to grade
            // an answer the user just stopped is work nobody is waiting for.
            let snapshot = citations.lock().clone();
            self.verified = Some(verified_citations(&snapshot, &full_response, None, false));
            // The producer left the embedding model loaded for a verification
            // pass that is no longer going to happen — nothing else will free
            // it.  Off-thread: this closes a llama.cpp context, and the
            // interrupt path must not stall on that.
            let svc = self.svc.clone();
            spawn_detached_blocking(move || svc.unload_embedding_model());
        }
        if let Some(session_id) = self.session_id.as_deref().filter(|s| !s.is_empty())
            && !full_response.is_empty()
        {
            // Persist the same verified set the client is shown — never the
            // raw list, which still carries the private `_extract` payload.
            let verified = self.verified.clone().unwrap_or_default();
            self.svc.append_to_session(
                session_id,
                "assistant",
                &full_response,
                Some(&verified),
                Some(&self.message_id),
            );
        }
    }
}

impl Drop for StreamFinally {
    fn drop(&mut self) {
        self.finish();
    }
}

/// A stream polled under a fixed project binding.
///
/// The binding is applied for the duration of each poll (and of the inner
/// stream's drop), so everything the generator does synchronously — including
/// capturing the binding for worker threads — resolves the request's project,
/// while nothing leaks into other tasks sharing the runtime thread.
struct BoundStream<S> {
    binding: Option<String>,
    inner: Option<Pin<Box<S>>>,
}

impl<S: Stream> Stream for BoundStream<S> {
    type Item = S::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let _guard = use_project(this.binding.as_deref());
        match this.inner.as_mut() {
            Some(inner) => inner.as_mut().poll_next(cx),
            None => Poll::Ready(None),
        }
    }
}

impl<S> Drop for BoundStream<S> {
    fn drop(&mut self) {
        let _guard = use_project(self.binding.as_deref());
        drop(self.inner.take());
    }
}

/// Start the source pre-embedding task if there are citations and it has not
/// started yet.
fn start_source_prep(
    prepared: &mut Option<JoinHandle<()>>,
    citations: &Option<SharedCitations>,
    svc: &Arc<dyn ChatService>,
) {
    if let Some(citations) = citations
        && prepared.is_none()
    {
        let citations = citations.clone();
        let svc = svc.clone();
        *prepared = Some(tokio::task::spawn_blocking(bind(move || {
            prepare_citation_sources(&citations, &*svc)
        })));
    }
}

/// Convert sync LLM streaming to async SSE.
///
/// Produces the exact SSE frames of the Python generator: `chat.status`
/// events, `chat.completion.chunk` deltas, `chat.message_id`, `chat.usage`,
/// `chat.citations` and the final `data: [DONE]`.  The generation runs on the
/// single-worker [`EXECUTOR`] and is bridged back through a queue.
///
/// Dropping the stream is the client disconnect: the cleanup Python ran in
/// the generator's `finally` (cancel the producer, write the partial answer to
/// the session) runs then.
pub fn stream_chunks(request: ChatCompletionRequest, model_service: Arc<dyn ChatService>) -> SseStream {
    // Belt and braces: the endpoint binds the project before this stream is
    // ever polled, but stream_chunks is also driven directly by tests and by
    // any future caller that skips the endpoint.
    let binding = request_project(&request, &*model_service).or_else(active_project_id);
    let inner = stream_chunks_inner(request, model_service);
    Box::pin(BoundStream { binding, inner: Some(Box::pin(inner)) })
}

fn stream_chunks_inner(
    request: ChatCompletionRequest,
    svc: Arc<dyn ChatService>,
) -> impl Stream<Item = String> + Send + 'static {
    async_stream::stream! {
        // The SSE chunks are read by a user-facing client, so they carry the
        // short public model name.  Telemetry (`push_token_usage`) keeps the
        // internal id it was requested with.
        let stream_model = display_model_name(&request.model);

        // Load the session's recent history ONCE and share it between the
        // prompt (prepended prior turns) and the orchestrator.
        let req_last_user = last_user_content(&request.messages);
        let conversation_history = match (&request.session_id, &req_last_user) {
            (Some(sid), Some(q)) if !sid.is_empty() && !q.is_empty() => {
                svc.get_relevant_history(sid, q)
            }
            _ => Vec::new(),
        };

        let mut messages = build_messages_with_history(&request, &conversation_history);

        // Resolve attachments (with status events if attachments exist)
        let has_attachments =
            messages.iter().any(|m| m.role == Role::User && has_items(&m.attachments));
        if has_attachments {
            yield status_event("attachments", "processing", "Resolving attachments...");
        }
        messages = resolve_message_attachments(messages, &*svc);
        if has_attachments {
            yield status_event("attachments", "completed", "Attachments resolved");
        }

        // A project bound to the user's folder answers from it, so the turn
        // waits for the folder's index to catch up: a document added or
        // changed there is read now, not on the next background pass.  The
        // wait is capped so a large first index can't hold the answer back
        // for good; it carries on in the background.
        let _ = request_cwd(&request, &*svc);
        if let Some(folder) = svc.current_work_dir() {
            yield status_event("index", "processing", &format!("Checking documents in {folder}"));
            let (done_tx, mut done_rx) = tokio::sync::oneshot::channel::<bool>();
            let worker = svc.clone();
            std::thread::spawn(bind(move || {
                let _ = done_tx.send(worker.index_work_dir());
            }));
            let started = Instant::now();
            // Each change is its own status line: `Reading <file>` while it
            // is parsed, then `Indexing <file>` while it is embedded.
            let mut shown: Option<String> = None;
            loop {
                match tokio::time::timeout(Duration::from_millis(500), &mut done_rx).await {
                    Ok(_) => {
                        if shown.is_some() {
                            yield status_event("index", "completed", "Documents indexed");
                        }
                        break;
                    }
                    Err(_) if started.elapsed() >= INDEX_WAIT => {
                        yield status_event("index", "completed", "Still indexing; answering from what's ready");
                        break;
                    }
                    Err(_) => {
                        let activity = svc.indexing_activity();
                        if let Some(message) = activity.as_deref().filter(|_| activity != shown) {
                            yield status_event("index", "processing", message);
                            shown = activity.clone();
                        }
                    }
                }
            }
        }

        // Image attachments on the current turn are handed to the orchestrator
        // so a vision-capable model reads the pixels + the user's own question,
        // instead of a pre-processed text transcription.
        let image_attachments: Vec<ImageAttachment> = match request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
        {
            Some(cur) if has_items(&cur.attachments) => {
                svc.get_asset_images(cur.attachments.as_deref().unwrap_or_default())
            }
            _ => Vec::new(),
        };

        let mut last_user_msg = last_user_content(&messages);

        let project_skills: Vec<Skill> =
            svc.get_project_skills().into_iter().filter(|s| !s.system).collect();

        // A leading `/<skill-name>` in the prompt is an explicit request to
        // fulfil this turn with that skill.  Resolve it against the project's
        // skills, strip the token from the prompt, and pass the skill name to
        // the orchestrator, which validates it against the request intent.
        let mut requested_skill: Option<String> = None;
        // What the session records as the user's turn.  Normally the same text
        // the model sees, but a bare `/<skill>` is expanded to the skill's body
        // for the model — the transcript keeps what the user actually typed.
        let mut session_user_msg = last_user_msg.clone();
        if let Some(idx) = messages.iter().rposition(|m| m.role == Role::User) {
            let typed = messages[idx].content.clone().filter(|c| !c.is_empty());
            if let Some(typed) = typed {
                let (skill, cleaned) = resolve_requested_skill(&typed, &project_skills);
                if let Some(skill) = skill {
                    messages[idx].content = Some(cleaned.clone());
                    last_user_msg = Some(cleaned);
                    let rest = SLASH_SKILL_RE
                        .captures(&typed)
                        .and_then(|c| c.get(2).map(|m| m.as_str().trim().to_string()))
                        .unwrap_or_default();
                    session_user_msg =
                        Some(if rest.is_empty() { typed.trim().to_string() } else { rest });
                    tracing::info!(
                        target: LOG_TARGET,
                        "[skill] user-selected skill: {}", py_repr(&skill)
                    );
                    requested_skill = Some(skill);
                }
            }
        }

        // Shared cancellation signal for the whole request.  Created before
        // orchestration so a client disconnect during the planning phase
        // interrupts those LLM calls too — not just the final response stream.
        let cancelled: CancelEvent = Arc::new(AtomicBool::new(false));
        // A client that disconnects drops this stream wherever it is
        // suspended.  `StreamFinally` raises the signal once the answer
        // streams, but orchestration runs before it exists: without this a
        // disconnect mid-fan-out left a worker thread extracting documents
        // for nobody for the better part of an hour.  Raising it on normal
        // completion too is harmless — everything has finished by then.
        let _cancel_on_drop = CancelOnDrop(cancelled.clone());

        // Orchestrator handles vector search, MCP tools, and explore in one step
        let mut orchestrator_context: Option<String> = None;
        let mut classified_capability: Option<String> = None;
        let mut citations: Option<Vec<CitationDict>> = None;
        let mut is_mutation = false;
        if truthy(&last_user_msg) {
            yield status_event("tools", "processing", "Processing...");

            // Bridge orchestrator step activity into the SSE stream.  The
            // orchestrator runs in a worker thread and reports each step via
            // progress_callback; those events are handed back thread-safely
            // and flushed as chat.status while it works, so the client shows
            // live progress instead of a frozen spinner.
            let (status_tx, mut status_rx) = unbounded_channel::<String>();
            let progress: ProgressCallback = Arc::new(move |step: &str, status: &str, message: &str| {
                if let Some(event) = try_status_event(step, status, message) {
                    let _ = status_tx.send(event);
                }
            });

            let args = OrchestrateArgs {
                user_message: last_user_msg.clone().unwrap_or_default(),
                skills: project_skills.clone(),
                cwd: request_cwd(&request, &*svc),
                conversation_history: conversation_history.clone(),
                has_attachments,
                images: image_attachments.clone(),
                progress_callback: Some(progress),
                cancelled: Some(cancelled.clone()),
                requested_skill: requested_skill.clone(),
                current_date: Some(request_current_date(&request)),
            };
            // `bind` carries this request's project into the worker: a thread
            // hop does not copy the binding, and everything the orchestrator
            // resolves — documents, skills, MCP servers — hangs off it.
            let orchestrate_svc = svc.clone();
            let orchestrate_task =
                tokio::task::spawn_blocking(bind(move || orchestrate_svc.orchestrate(args)));

            let orch_timeout = AppSettings::new().orchestrate_timeout_s;
            // An idle budget, not a total one: every progress event (a step
            // starting or finishing) restarts it, so a slow but advancing
            // orchestration is never cut off.
            let budget = (orch_timeout > 0.0).then(|| Duration::from_secs_f64(orch_timeout));
            let mut deadline = budget.map(|b| Instant::now() + b);

            let mut timed_out = false;
            while !orchestrate_task.is_finished() {
                if deadline.is_some_and(|d| Instant::now() > d) {
                    timed_out = true;
                    break;
                }
                match tokio::time::timeout(Duration::from_millis(100), status_rx.recv()).await {
                    Ok(Some(event)) => {
                        deadline = budget.map(|b| Instant::now() + b);
                        yield event
                    }
                    Ok(None) => tokio::time::sleep(Duration::from_millis(1)).await,
                    Err(_) => continue,
                }
            }

            if timed_out {
                // A native llama.cpp call can't be preempted mid-token, so we
                // signal cancellation (checked between chunks) and stop waiting
                // rather than hang the client.  The worker is not awaited — a
                // truly stuck call would block forever; it unwinds on its own
                // once the current chunk returns.
                cancelled.store(true, Ordering::SeqCst);
                tracing::error!(
                    target: LOG_TARGET,
                    "[orchestrate] no progress for {:.0}s — cancelling and returning a timeout to the client",
                    orch_timeout
                );
                // Drain any progress events already queued before giving up.
                while let Ok(event) = status_rx.try_recv() {
                    yield event;
                }
                yield status_event(
                    "tools",
                    "error",
                    &format!("Orchestration made no progress for {orch_timeout:.0}s"),
                );
                yield chunk_frame(
                    &stream_model,
                    DeltaContent {
                        role: Some("assistant".to_string()),
                        content: Some(ORCHESTRATION_TIMEOUT_CONTENT.to_string()),
                    },
                    Some("stop"),
                );
                yield "data: [DONE]\n\n".to_string();
                // The orchestration thread is still running and very likely
                // inside a native llama.cpp call right now.  Unloading here
                // would close the context it is decoding on — a native
                // use-after-free that takes down the whole server.  Defer it
                // until the thread actually unwinds; if it never does, the
                // auxiliary models stay resident, which beats a segfault.
                unload_when_idle(orchestrate_task, svc.clone());
                return;
            }

            let outcome = match orchestrate_task.await {
                Ok(Ok(outcome)) => outcome,
                Ok(Err(e)) => {
                    tracing::error!(target: LOG_TARGET, "Error in chat completion stream: {}", e);
                    return;
                }
                Err(e) => {
                    let message = match e.try_into_panic() {
                        Ok(payload) => panic_message(payload.as_ref()),
                        Err(e) => e.to_string(),
                    };
                    tracing::error!(
                        target: LOG_TARGET,
                        "Error in chat completion stream: {}", message
                    );
                    return;
                }
            };
            orchestrator_context = outcome.context;
            classified_capability = outcome.capability;
            citations = outcome.citations;
            is_mutation = outcome.mutation;
            let tool_error = outcome.error;

            // Flush any step events scheduled just before completion.
            while let Ok(event) = status_rx.try_recv() {
                yield event;
            }

            match tool_error.as_deref().filter(|e| !e.is_empty()) {
                Some(error) => yield status_event("tools", "error", error),
                None => yield status_event("tools", "completed", "Tools completed"),
            }
        }

        // For mutation operations (save, create, delete, etc.) return the
        // success message directly without running it through the LLM.
        if is_mutation && truthy(&orchestrator_context) {
            let context = orchestrator_context.clone().unwrap_or_default();
            let message_id = message_id_for(&request);
            yield status_event("streaming", "processing", "Generating response...");
            yield chunk_frame(
                &stream_model,
                DeltaContent { role: Some("assistant".to_string()), content: None },
                None,
            );
            yield format!(
                "data: {}\n\n",
                py_json_dumps(&json!({"object": "chat.message_id", "message_id": message_id}))
            );
            // Stream synthesized content in small chunks for progressive rendering
            let chars: Vec<char> = context.chars().collect();
            for piece in chars.chunks(STREAM_CHUNK_SIZE) {
                yield chunk_frame(
                    &stream_model,
                    content_delta(piece.iter().collect::<String>()),
                    None,
                );
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            yield chunk_frame(&stream_model, DeltaContent::default(), Some("stop"));
            yield "data: [DONE]\n\n".to_string();
            if let Some(session_id) = request.session_id.as_deref().filter(|s| !s.is_empty()) {
                if truthy(&last_user_msg) {
                    // For image turns, fold the vision model's reading of the
                    // image into the stored user turn so later turns (which no
                    // longer carry the attachment) can still reference what the
                    // image contained.
                    let typed = session_user_msg.clone().unwrap_or_default();
                    let user_record = if image_attachments.is_empty() {
                        typed
                    } else {
                        format!("{typed}\n\n[Attached image — description]:\n{context}")
                    };
                    svc.append_to_session(session_id, "user", &user_record, None, None);
                }
                svc.append_to_session(session_id, "assistant", &context, None, Some(&message_id));
            }

            let project_id = svc.get_project_id();
            let prompt_tokens = svc.count_tokens(last_user_msg.as_deref().unwrap_or(""));
            let completion_tokens = svc.count_tokens(&context);
            svc.push_token_usage(
                project_id.as_deref(),
                &request.model,
                prompt_tokens,
                completion_tokens,
            );
            return;
        }

        let project_id = svc.get_project_id();
        let project_info = svc.get_project_info();

        yield status_event("preparing", "processing", "Preparing response...");

        // The final synthesis runs on the capability/large/default model, whose
        // context window can be *smaller* than the default router model's.
        // Size the context against the model that will actually consume it —
        // otherwise web/document context overflows the response model and the
        // search content never reaches the LLM.
        let response_capability = response_capability(&orchestrator_context, classified_capability.as_deref());

        // The response model is resolved by the call below, and resolving it
        // can mean loading a model.  It reports through ModelRouter like every
        // other resolution; collect the events here and yield them, since this
        // generator cannot interleave with the synchronous call itself.  The
        // "loading" warning has to precede it, so it is announced first.
        let model_events: Arc<Mutex<Vec<(String, String, String)>>> = Arc::default();
        let sink = model_events.clone();
        svc.begin_response_request(Some(Arc::new(move |step: &str, status: &str, message: &str| {
            sink.lock().push((step.to_string(), status.to_string(), message.to_string()));
        })));

        let announce_svc = svc.clone();
        run_blocking(move || announce_svc.announce_response_model(response_capability)).await;
        let drained: Vec<(String, String, String)> = std::mem::take(&mut *model_events.lock());
        for (step, status, message) in drained {
            yield status_event(&step, &status, &message);
        }

        let max_ctx_svc = svc.clone();
        let max_ctx = run_blocking(move || {
            max_ctx_svc.get_response_max_context_tokens(response_capability)
        })
        .await;

        let drained: Vec<(String, String, String)> = std::mem::take(&mut *model_events.lock());
        for (step, status, message) in drained {
            yield status_event(&step, &status, &message);
        }

        let current_date = request_current_date(&request);
        if let Some(context) = orchestrator_context.as_mut().filter(|c| !c.is_empty())
            && max_ctx > 0
        {
            // Size the context against the response model's real window,
            // reserving room for the fixed prompt overhead (system prompt, chat
            // history, template tags) PLUS a healthy output budget.  A flat
            // reserve here left almost no output tokens once the system prompt
            // was added, so grounded responses were cut off mid-sentence.
            let output_reserve: i64 = 512;
            let overhead_prompt = format_messages_to_prompt(
                &messages,
                Some(""),
                request_cwd(&request, &*svc).as_deref(),
                project_info.as_deref(),
                Some(&current_date),
            );
            let overhead_tokens = svc.count_tokens(&overhead_prompt) as i64;
            let budget = (max_ctx as i64 - overhead_tokens - output_reserve).max(256);
            let ctx_tokens = svc.count_tokens(context) as i64;
            if ctx_tokens > budget {
                let char_count = context.chars().count() as i64;
                let chars_per_token = (char_count / ctx_tokens.max(1)).max(1);
                let keep = (budget * chars_per_token) as usize;
                *context = context.chars().take(keep).collect();
                tracing::info!(
                    target: LOG_TARGET,
                    "[context] Truncated orchestrator context from {} to {} tokens (overhead {}, output reserve {}) to fit response model window ({})",
                    ctx_tokens, budget, overhead_tokens, output_reserve, max_ctx
                );
            }
        }

        let context_tokens = svc.count_tokens(orchestrator_context.as_deref().unwrap_or(""));
        messages = compact_messages(messages, &*svc, 2048, context_tokens, Some(max_ctx));

        let prompt = format_messages_to_prompt(
            &messages,
            orchestrator_context.as_deref(),
            request_cwd(&request, &*svc).as_deref(),
            project_info.as_deref(),
            Some(&current_date),
        );

        yield status_event("preparing", "completed", "Preparing response...");

        // Store the user message in session history
        if let Some(session_id) = request.session_id.as_deref().filter(|s| !s.is_empty())
            && truthy(&last_user_msg)
        {
            svc.append_to_session(
                session_id,
                "user",
                session_user_msg.as_deref().unwrap_or(""),
                None,
                None,
            );
        }

        // `cancelled` is created earlier (before orchestration) and reused here
        // so the same signal covers both the planning phase and the response.
        let citations: Option<SharedCitations> = citations
            .filter(|c| !c.is_empty())
            .map(|c| Arc::new(Mutex::new(c)));

        let (queue_tx, mut queue_rx) = unbounded_channel::<QueueItem>();
        let producer = Producer {
            svc: svc.clone(),
            prompt: prompt.clone(),
            model: request.model.clone(),
            capability: response_capability,
            cancelled: cancelled.clone(),
            orchestrator_context: orchestrator_context.clone(),
            last_user_msg: last_user_msg.clone(),
            queue: queue_tx,
            include_embedding: citations.is_none(),
        };
        // Start the sync stream on the single-worker pool, carrying the
        // request's project binding into the worker.
        drop(EXECUTOR.submit(move || producer.run_sync_stream()));

        // Load the embedding model and encode the sources alongside generation
        // rather than after it — but not from the first token.  The embedder
        // is a second GGUF and it goes onto the same device the answer is
        // decoding on; landing that on the opening tokens is the one place it
        // is visible.  Starting it once the opening is out keeps the overlap
        // for everything after.
        let mut prepared_sources: Option<JoinHandle<()>> = None;

        // Signal that streaming has started
        yield status_event("streaming", "processing", "Generating response...");

        // Initial chunk with role (OpenAI-compatible raw data format)
        let message_id = message_id_for(&request);
        yield chunk_frame(
            &stream_model,
            DeltaContent { role: Some("assistant".to_string()), content: None },
            None,
        );

        yield format!(
            "data: {}\n\n",
            py_json_dumps(&json!({"object": "chat.message_id", "message_id": message_id}))
        );

        // Stream content chunks and collect the full response for session
        // history.  The verified citation set is computed once when the answer
        // is complete and reused for both the session record and the client.
        let mut fin = StreamFinally {
            svc: svc.clone(),
            cancelled: cancelled.clone(),
            citations: citations.clone(),
            parts: Vec::new(),
            verified: None,
            session_id: request.session_id.clone(),
            message_id: message_id.clone(),
            done: false,
        };
        // Silence budget for the producer thread — see stream_stall_timeout_s.
        let stall_timeout = AppSettings::new().stream_stall_timeout_s;
        loop {
            let item = if stall_timeout > 0.0 {
                match tokio::time::timeout(
                    Duration::from_secs_f64(stall_timeout),
                    queue_rx.recv(),
                )
                .await
                {
                    Ok(item) => item,
                    Err(_) => {
                        // Nothing at all for the whole budget.  The producer is
                        // either inside a native call that cannot be preempted
                        // or still queued behind one on the executor; either
                        // way it is not coming back on its own.  Signal
                        // cancellation and close the stream properly — an
                        // error the user can read, a finish_reason, and [DONE].
                        cancelled.store(true, Ordering::SeqCst);
                        tracing::error!(
                            target: LOG_TARGET,
                            "[stream] no output for {:.0}s — abandoning the generation ({} chunk(s) so far)",
                            stall_timeout,
                            fin.parts.len()
                        );
                        yield status_event(
                            "streaming",
                            "error",
                            &format!("Generation stalled after {stall_timeout:.0}s"),
                        );
                        yield chunk_frame(
                            &stream_model,
                            content_delta(format!(
                                "\n\n**Generation stopped.**\n\n\
                                 The model produced nothing for {stall_timeout:.0}s and the request was \
                                 abandoned. The device may still be busy with \
                                 an earlier request, or the loaded model may \
                                 be too large for this machine. Try again, or \
                                 pick a smaller model in \
                                 **Settings → Models**."
                            )),
                            None,
                        );
                        yield chunk_frame(&stream_model, DeltaContent::default(), Some("stop"));
                        yield "data: [DONE]\n\n".to_string();
                        // Persists whatever text did arrive.  Model cleanup
                        // still belongs to the producer thread — it is very
                        // likely mid-inference right now.
                        fin.finish();
                        return;
                    }
                }
            } else {
                queue_rx.recv().await
            };

            match item {
                // Stream complete
                None | Some(QueueItem::End) => break,
                Some(QueueItem::Error(error_msg)) => {
                    tracing::error!(target: LOG_TARGET, "Stream error: {}", error_msg);
                    yield status_event("error", "error", &error_msg);
                    // Surface the error as a visible content chunk so the user sees it
                    yield chunk_frame(
                        &stream_model,
                        content_delta(format!("\n\n**Error:** {error_msg}")),
                        None,
                    );
                    break;
                }
                Some(QueueItem::Text(text)) => {
                    fin.parts.push(text.clone());
                    if fin.parts.len() == PREP_SOURCES_AFTER_CHUNKS {
                        start_source_prep(&mut prepared_sources, &citations, &svc);
                    }
                    yield chunk_frame(&stream_model, content_delta(text), None);
                }
            }
        }

        // The answer is complete, so the sources can finally be checked against
        // it.  Announced as a step rather than done silently: this embeds the
        // response, which takes long enough to notice, and it lands exactly
        // where the user thinks the turn is over.  An unexplained pause there
        // reads as a hang; "Checking sources" reads as the last piece of work.
        if let Some(shared) = citations.clone()
            && !fin.parts.is_empty()
        {
            // An answer shorter than the opening window never reached the
            // dispatch above; the encode still has to happen, it just has
            // nothing left to overlap with.
            start_source_prep(&mut prepared_sources, &citations, &svc);
            yield status_event("citations", "processing", "Checking sources...");
            let verified = verified_citations_async(
                shared,
                fin.parts.concat(),
                svc.clone(),
                prepared_sources.take(),
            )
            .await;
            fin.verified = Some(verified.clone());
            // Verification held the last reference to the embedding model —
            // the producer skipped it on the way out.  Free it now, without
            // waiting: the citations event should not queue behind a GPU
            // context teardown.
            let unload_svc = svc.clone();
            spawn_detached_blocking(move || unload_svc.unload_embedding_model());
            let message = if verified.is_empty() {
                "No sources used".to_string()
            } else {
                format!("{} source(s) cited", verified.len())
            };
            yield status_event("citations", "completed", &message);
        }
        fin.finish();

        let full_response = fin.parts.concat();
        let verified_citations = fin.verified.clone();

        // Auto file-write: if user asked to create/generate/write, save LLM
        // response to file
        let out_cwd = request_cwd(&request, &*svc).filter(|c| !c.is_empty());
        if !fin.parts.is_empty()
            && let (Some(user_msg), Some(cwd)) =
                (last_user_msg.as_deref().filter(|m| !m.is_empty()), out_cwd.as_deref())
        {
            let outcome = svc.maybe_write_output(
                user_msg,
                &full_response,
                Some(cwd),
                request.session_id.as_deref(),
            );
            if !outcome.is_empty() {
                for written_path in &outcome.written {
                    yield status_event("file-write", "completed", written_path);
                }
                // Surface the file paths as visible content so the user can click them
                let summary = write_summary(&outcome);
                yield chunk_frame(&stream_model, content_delta(summary.clone()), None);
                // Persist the file links in session history
                if let Some(session_id) = request.session_id.as_deref().filter(|s| !s.is_empty()) {
                    svc.append_to_session(session_id, "assistant", &summary, None, None);
                }
            }
        }

        // Compute and emit usage info after streaming
        if !fin.parts.is_empty() {
            let prompt_tokens = svc.count_tokens(&prompt);
            let completion_tokens = svc.count_tokens(&full_response);
            let max_context = svc.get_max_context_tokens();
            let total_tokens = prompt_tokens + completion_tokens;

            svc.push_token_usage(
                project_id.as_deref(),
                &request.model,
                prompt_tokens,
                completion_tokens,
            );

            // Emit usage info so the TUI can display context utilisation
            let usage_payload = py_json_dumps(&json!({
                "object": "chat.usage",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "max_context_tokens": max_context,
            }));
            yield format!("data: {usage_payload}\n\n");
        }

        // Credit only the sources the answer actually used, since passing the
        // relevance filter means a document was *retrieved*, not that it
        // informed the response.  Matching is on document content, not the
        // filename: an earlier filename-based check dropped real sources whose
        // names say nothing about their contents ("sp46.pdf" for a fund).
        if let Some(verified) = verified_citations.filter(|v| !v.is_empty()) {
            let payload = py_json_dumps(&json!({
                "object": "chat.citations",
                "citations": verified,
            }));
            yield format!("data: {payload}\n\n");
        }

        yield status_event("streaming", "completed", "Response complete");

        // Final chunk with finish_reason (OpenAI-compatible raw data format)
        yield chunk_frame(&stream_model, DeltaContent::default(), Some("stop"));
        yield "data: [DONE]\n\n".to_string();
    }
}

// ---------------------------------------------------------------- endpoint

/// Router for `/v1` chat endpoints.
pub fn router() -> Router {
    Router::new().route("/v1/chat/completions", post(create_chat_completion))
}

/// A FastAPI-style 422 response.
fn validation_error_response(errors: Vec<Value>) -> Response {
    (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({ "detail": errors }))).into_response()
}

/// Parse and validate a `ChatCompletionRequest` body the way FastAPI does,
/// answering 422 with `{"detail": [...]}` on failure.
pub fn parse_chat_request(body: &[u8]) -> Result<ChatCompletionRequest, Response> {
    if body.iter().all(|b| b.is_ascii_whitespace()) {
        return Err(validation_error_response(vec![json!({
            "type": "missing",
            "loc": ["body"],
            "msg": "Field required",
            "input": Value::Null,
        })]));
    }
    let value: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            // Byte offset of the error, as Python's JSONDecodeError.pos.
            let text = String::from_utf8_lossy(body);
            let mut pos = 0usize;
            for (i, line) in text.split('\n').enumerate() {
                if i + 1 == e.line() {
                    pos += e.column().saturating_sub(1);
                    break;
                }
                pos += line.chars().count() + 1;
            }
            return Err(validation_error_response(vec![json!({
                "type": "json_invalid",
                "loc": ["body", pos],
                "msg": "JSON decode error",
                "input": {},
                "ctx": {"error": e.to_string()},
            })]));
        }
    };
    let Some(obj) = value.as_object() else {
        return Err(validation_error_response(vec![json!({
            "type": "model_attributes_type",
            "loc": ["body"],
            "msg": "Input should be a valid dictionary or object to extract fields from",
            "input": value,
        })]));
    };
    let missing: Vec<Value> = ["model", "messages"]
        .iter()
        .filter(|field| !obj.contains_key(**field))
        .map(|field| {
            json!({
                "type": "missing",
                "loc": ["body", field],
                "msg": "Field required",
                "input": value,
            })
        })
        .collect();
    if !missing.is_empty() {
        return Err(validation_error_response(missing));
    }
    let request: ChatCompletionRequest = match serde_json::from_value(value.clone()) {
        Ok(r) => r,
        Err(e) => {
            return Err(validation_error_response(vec![json!({
                "type": "value_error",
                "loc": ["body"],
                "msg": e.to_string(),
                "input": value,
            })]));
        }
    };
    let errors = request.validation_errors();
    if !errors.is_empty() {
        return Err(validation_error_response(errors));
    }
    Ok(request)
}

/// `POST /v1/chat/completions` — create a chat completion (OpenAI-compatible).
/// Supports both streaming and non-streaming modes.
#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    tag = "chat",
    request_body = crate::api::model::chat::ChatCompletionRequest,
    summary = "Create a chat completion",
    description = "OpenAI-compatible. Set `stream: true` to receive Server-Sent Events: `chat.completion.chunk` frames, `chat.status` progress frames, then `data: [DONE]`. A slow generation is never cut off.",
    responses((status = 200, description = "A completion, or with `stream: true` an SSE stream of chunks ending in `data: [DONE]`", content((crate::api::model::chat::ChatCompletionResponse = "application/json"), (crate::api::model::chat::ChatCompletionChunk = "text/event-stream"))), (status = 400, description = "Invalid request"), (status = 500, description = "Server error"))
)]
pub async fn create_chat_completion(body: Bytes) -> Response {
    let request = match parse_chat_request(&body) {
        Ok(request) => request,
        Err(response) => return response,
    };
    let model_service: Arc<dyn ChatService> = get_model_service();
    match create_chat_completion_with(request, model_service).await {
        Ok(response) => response,
        Err(e) => e.into_response(),
    }
}

/// [`create_chat_completion`] against an explicit service (test seam).
pub async fn create_chat_completion_with(
    request: ChatCompletionRequest,
    model_service: Arc<dyn ChatService>,
) -> Result<Response, HttpException> {
    // Bind before anything resolves a project.  The binding covers this
    // handler; the streaming body carries its own (see `stream_chunks`).
    let binding = request_project(&request, &*model_service);
    scope_project(binding, async move {
        if !model_service.is_loaded() {
            tracing::info!(target: LOG_TARGET, "Model not loaded, attempting automatic reload...");
            let svc = model_service.clone();
            if let Err(e) = run_blocking(move || svc.load_model()).await {
                tracing::error!(target: LOG_TARGET, "Failed to auto-reload model: {}", e);
                return Err(HttpException::new(
                    503,
                    "No model loaded and auto-reload failed. Call /v1/models/load first.",
                ));
            }
        }

        if request.stream {
            let stream = stream_chunks(request, model_service);
            let body = Body::from_stream(stream.map(Ok::<_, std::convert::Infallible>));
            let response = Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
                .header(header::CACHE_CONTROL, "no-cache")
                .header(header::CONNECTION, "keep-alive")
                .header("X-Accel-Buffering", "no")
                .body(body)
                .map_err(|e| HttpException::new(500, e.to_string()))?;
            return Ok(response);
        }

        // Non-streaming response
        let response = complete_without_streaming(request, model_service).await?;
        Ok(Json(response).into_response())
    })
    .await
}

/// The cleanup of `_complete_without_streaming`'s `finally`.
///
/// Free the on-demand auxiliary models (vision + large agent) after each
/// response so only the orchestrator stays resident — unless the generation
/// was abandoned by the timeout, in which case its worker is very likely
/// inside a native llama.cpp call and closing its context would crash the
/// server: wait for it to unwind instead.
struct NonStreamingCleanup {
    svc: Arc<dyn ChatService>,
    stream: bool,
    abandoned: Option<tokio::sync::oneshot::Receiver<anyhow::Result<String>>>,
}

impl Drop for NonStreamingCleanup {
    fn drop(&mut self) {
        if self.stream {
            return;
        }
        match self.abandoned.take() {
            Some(abandoned) => unload_when_idle(abandoned, self.svc.clone()),
            None => self.svc.unload_auxiliary_models(true),
        }
    }
}

/// The non-streaming half of `create_chat_completion`.
pub async fn complete_without_streaming(
    request: ChatCompletionRequest,
    model_service: Arc<dyn ChatService>,
) -> Result<ChatCompletionResponse, HttpException> {
    // See stream_chunks: the endpoint has already bound this, but the helper
    // is also called directly.
    let binding = request_project(&request, &*model_service);
    scope_project(binding, async move {
        let mut cleanup = NonStreamingCleanup {
            svc: model_service.clone(),
            stream: request.stream,
            abandoned: None,
        };
        match complete_body(&request, &model_service, &mut cleanup).await {
            Ok(response) => Ok(response),
            Err(e) => {
                tracing::error!(target: LOG_TARGET, "Error in chat completion: {:?}", e);
                Err(HttpException::new(500, e.to_string()))
            }
        }
    })
    .await
}

async fn complete_body(
    request: &ChatCompletionRequest,
    svc: &Arc<dyn ChatService>,
    cleanup: &mut NonStreamingCleanup,
) -> anyhow::Result<ChatCompletionResponse> {
    if !svc.is_loaded() {
        tracing::info!(
            target: LOG_TARGET,
            "Model not loaded (non-streaming), attempting automatic reload..."
        );
        let load_svc = svc.clone();
        if let Err(e) = run_blocking(move || load_svc.load_model()).await {
            tracing::error!(target: LOG_TARGET, "Failed to auto-reload model: {}", e);
            anyhow::bail!("No model loaded and auto-reload failed: {e}");
        }
    }

    // Load the session's recent history ONCE and share it between the prompt
    // (prepended prior turns) and the orchestrator.
    let req_last_user = last_user_content(&request.messages);
    let conversation_history = match (&request.session_id, &req_last_user) {
        (Some(sid), Some(q)) if !sid.is_empty() && !q.is_empty() => {
            svc.get_relevant_history(sid, q)
        }
        _ => Vec::new(),
    };

    let messages = build_messages_with_history(request, &conversation_history);
    let messages = resolve_message_attachments(messages, &**svc);

    let last_user_msg = last_user_content(&messages);

    let project_skills: Vec<Skill> =
        svc.get_project_skills().into_iter().filter(|s| !s.system).collect();

    let image_attachments: Vec<ImageAttachment> =
        match request.messages.iter().rev().find(|m| m.role == Role::User) {
            Some(cur) if has_items(&cur.attachments) => {
                svc.get_asset_images(cur.attachments.as_deref().unwrap_or_default())
            }
            _ => Vec::new(),
        };

    // Orchestrator handles vector search, MCP tools, and explore in one step
    let mut orchestrator_context: Option<String> = None;
    let mut classified_capability: Option<String> = None;
    let mut tool_error: Option<String> = None;
    let mut citations: Option<Vec<CitationDict>> = None;
    let mut is_mutation = false;
    if let Some(user_message) = last_user_msg.clone().filter(|m| !m.is_empty()) {
        let args = OrchestrateArgs {
            user_message,
            skills: project_skills.clone(),
            cwd: request_cwd(request, &**svc),
            conversation_history: conversation_history.clone(),
            images: image_attachments.clone(),
            current_date: Some(request_current_date(request)),
            ..OrchestrateArgs::default()
        };
        let orchestrate_svc = svc.clone();
        let outcome = run_on_executor(move || orchestrate_svc.orchestrate(args)).await??;
        orchestrator_context = outcome.context;
        classified_capability = outcome.capability;
        tool_error = outcome.error;
        citations = outcome.citations;
        is_mutation = outcome.mutation;
    }

    let zero_usage = Usage { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
    if let Some(error) = tool_error.filter(|e| !e.is_empty()) {
        let mut response = ChatCompletionResponse::new(
            request.model.clone(),
            vec![Choice {
                index: 0,
                message: ChatMessage { role: Role::Assistant, content: Some(error), attachments: None },
                finish_reason: "stop".to_string(),
            }],
        );
        response.usage = zero_usage;
        return Ok(response);
    }
    // For mutation operations return the success message directly.
    if is_mutation && truthy(&orchestrator_context) {
        let mut response = ChatCompletionResponse::new(
            request.model.clone(),
            vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: Role::Assistant,
                    content: orchestrator_context.clone(),
                    attachments: None,
                },
                finish_reason: "stop".to_string(),
            }],
        );
        response.usage = zero_usage;
        return Ok(response);
    }

    let project_id = svc.get_project_id();
    let project_info = svc.get_project_info();

    let context_tokens = svc.count_tokens(orchestrator_context.as_deref().unwrap_or(""));
    let messages = compact_messages(messages, &**svc, 2048, context_tokens, None);

    let prompt = format_messages_to_prompt(
        &messages,
        orchestrator_context.as_deref(),
        request_cwd(request, &**svc).as_deref(),
        project_info.as_deref(),
        Some(&request_current_date(request)),
    );

    // Store user message in session history
    if let Some(session_id) = request.session_id.as_deref().filter(|s| !s.is_empty())
        && let Some(user_msg) = last_user_msg.as_deref().filter(|m| !m.is_empty())
    {
        svc.append_to_session(session_id, "user", user_msg, None, None);
    }

    // Embed the sources alongside generation — see the same call in
    // stream_chunks.  The completion below runs on the single-worker
    // executor, so this goes to the blocking pool or it would queue behind
    // the very work it is meant to overlap with.
    let citations: Option<SharedCitations> =
        citations.filter(|c| !c.is_empty()).map(|c| Arc::new(Mutex::new(c)));
    let mut prepared_sources: Option<JoinHandle<()>> = None;
    start_source_prep(&mut prepared_sources, &citations, svc);

    // Run sync completion on the executor, under the same silence budget the
    // streaming path applies — a generation that wedged (or one queued behind
    // a wedged predecessor on the single-worker executor) held the request
    // open indefinitely.  `generate` takes no cancellation event, so the
    // budget can only abandon the worker: it keeps running until it unwinds,
    // and the client gets an error instead of nothing.
    //
    // A stream has chunks to reset the budget, so it only ever measures
    // silence; a non-streaming call has no progress signal at all, so the same
    // number necessarily bounds the whole generation.
    let response_capability = response_capability(&orchestrator_context, classified_capability.as_deref());
    let stall_timeout = AppSettings::new().stream_stall_timeout_s;
    let (tx, mut rx) = tokio::sync::oneshot::channel::<anyhow::Result<String>>();
    let generate_svc = svc.clone();
    let generate_prompt = prompt.clone();
    let generate_model = request.model.clone();
    drop(EXECUTOR.submit(move || {
        // No reasoning on the answer — see the streaming path.
        let result = generate_svc.model_client().generate(
            &generate_prompt,
            Some(&generate_model),
            response_capability.map(|c| c.value()),
            None,
            Some(false),
        );
        let _ = tx.send(result);
    }));
    let generation = if stall_timeout > 0.0 {
        match tokio::time::timeout(Duration::from_secs_f64(stall_timeout), &mut rx).await {
            Ok(result) => result,
            Err(_) => {
                tracing::error!(
                    target: LOG_TARGET,
                    "[chat] generation exceeded {:.0}s — abandoning the worker and returning an error",
                    stall_timeout
                );
                cleanup.abandoned = Some(rx);
                anyhow::bail!("Generation timed out after {stall_timeout:.0}s");
            }
        }
    } else {
        (&mut rx).await
    };
    let mut response_text = generation
        .map_err(|_| anyhow::anyhow!("executor worker exited without a result"))??;

    // Auto file-write: if user asked to create/generate/write, save LLM
    // response to file
    let out_cwd = request_cwd(request, &**svc).filter(|c| !c.is_empty());
    if let (Some(user_msg), Some(cwd)) =
        (last_user_msg.as_deref().filter(|m| !m.is_empty()), out_cwd.as_deref())
        && !response_text.is_empty()
    {
        let outcome =
            svc.maybe_write_output(user_msg, &response_text, Some(cwd), request.session_id.as_deref());
        response_text.push_str(&write_summary(&outcome));
    }

    // Verified once and reused for both the session record and the response.
    let mut verified: Option<Vec<CitationDict>> = None;
    if let Some(shared) = citations.clone()
        && !response_text.is_empty()
    {
        verified = Some(
            verified_citations_async(shared, response_text.clone(), svc.clone(), prepared_sources.take())
                .await,
        );
    }

    // Store assistant response in session history (after auto-write so the
    // link is included)
    let message_id = message_id_for(request);
    if let Some(session_id) = request.session_id.as_deref().filter(|s| !s.is_empty())
        && !response_text.is_empty()
    {
        let saved = verified.clone().unwrap_or_default();
        svc.append_to_session(session_id, "assistant", &response_text, Some(&saved), Some(&message_id));
    }

    let prompt_tokens = svc.count_tokens(&prompt);
    let completion_tokens = svc.count_tokens(&response_text);

    svc.push_token_usage(project_id.as_deref(), &request.model, prompt_tokens, completion_tokens);

    let mut citation_models: Option<Vec<Citation>> = None;
    if let Some(verified) = verified.filter(|v| !v.is_empty()) {
        let mut models = Vec::with_capacity(verified.len());
        for c in &verified {
            let document_id = match c.get("document_id") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Null) | None => anyhow::bail!("'document_id'"),
                Some(other) => other.to_string(),
            };
            let document_name = c.get("document_name").and_then(Value::as_str).map(str::to_string);
            models.push(Citation { document_id, document_name });
        }
        citation_models = Some(models);
    }

    let mut response = ChatCompletionResponse::new(
        request.model.clone(),
        vec![Choice {
            index: 0,
            message: ChatMessage {
                role: Role::Assistant,
                content: Some(response_text),
                attachments: None,
            },
            finish_reason: "stop".to_string(),
        }],
    );
    response.message_id = Some(message_id);
    response.usage = Usage {
        prompt_tokens: prompt_tokens as i64,
        completion_tokens: completion_tokens as i64,
        total_tokens: (prompt_tokens + completion_tokens) as i64,
    };
    response.citations = citation_models;
    Ok(response)
}
