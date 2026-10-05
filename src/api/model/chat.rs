//! Port of `smartloop/api/model/chat.py`.
//!
//! OpenAI-compatible chat completion schemas.

use serde::{Deserialize, Serialize};

/// The role of a chat message author (`Literal["system", "user", "assistant"]`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    #[default]
    User,
    Assistant,
}

impl Role {
    /// The wire string for this role.
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ChatMessage {
    /// The role of the message author.
    pub role: Role,
    /// The content of the message.
    #[serde(default)]
    pub content: Option<String>,
    /// List of asset IDs to include as context in this message.
    #[serde(default)]
    pub attachments: Option<Vec<String>>,
}

/// Client-supplied per-request context (the browser's clock/locale).
///
/// Sent so date/time-aware answers are grounded in the *user's* wall clock and
/// locale instead of the server's — see `resolve_current_date`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct AdditionalMessageParams {
    /// IANA timezone, e.g. 'America/New_York'.
    #[serde(default)]
    pub tz: Option<String>,
    /// BCP-47 locale, e.g. 'en-US'.
    #[serde(default)]
    pub locale: Option<String>,
    /// Client 'now' as a Unix timestamp in seconds.
    #[serde(default)]
    pub user_date: Option<i64>,
}

fn default_temperature() -> Option<f64> {
    Some(0.7)
}
fn default_top_p() -> Option<f64> {
    Some(1.0)
}
fn default_penalty() -> Option<f64> {
    Some(0.0)
}

/// OpenAI-compatible chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionRequest {
    /// ID of the model to use.
    pub model: String,
    /// A list of messages comprising the conversation.
    pub messages: Vec<ChatMessage>,
    /// Session ID for conversation tracking.
    #[serde(default)]
    pub session_id: Option<String>,
    /// Client-provided message ID for the assistant response.
    #[serde(default)]
    pub message_id: Option<String>,
    /// Project ID to use for this request.
    #[serde(default)]
    pub project_id: Option<String>,
    /// Current working directory for file tools.
    #[serde(default)]
    pub cwd: Option<String>,
    /// Whether to stream the response.
    #[serde(default)]
    pub stream: bool,
    /// Sampling temperature (0.0..=2.0).
    #[serde(default = "default_temperature")]
    pub temperature: Option<f64>,
    /// Maximum tokens to generate (>= 1).
    #[serde(default)]
    pub max_tokens: Option<i64>,
    /// Nucleus sampling parameter (0.0..=1.0).
    #[serde(default = "default_top_p")]
    pub top_p: Option<f64>,
    /// Frequency penalty (-2.0..=2.0).
    #[serde(default = "default_penalty")]
    pub frequency_penalty: Option<f64>,
    /// Presence penalty (-2.0..=2.0).
    #[serde(default = "default_penalty")]
    pub presence_penalty: Option<f64>,
    /// Stop sequences.
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    /// Client context (timezone/locale/current time) used to ground
    /// date-aware answers in the user's wall clock.
    #[serde(default)]
    pub additional_message_params: Option<AdditionalMessageParams>,
}

impl Default for ChatCompletionRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            messages: Vec::new(),
            session_id: None,
            message_id: None,
            project_id: None,
            cwd: None,
            stream: false,
            temperature: default_temperature(),
            max_tokens: None,
            top_p: default_top_p(),
            frequency_penalty: default_penalty(),
            presence_penalty: default_penalty(),
            stop: None,
            additional_message_params: None,
        }
    }
}

impl ChatCompletionRequest {
    /// Pydantic field-constraint validation (`ge`/`le`).  Returns FastAPI-style
    /// 422 error entries (`{"type", "loc", "msg", "input", "ctx"}`); empty when valid.
    pub fn validation_errors(&self) -> Vec<serde_json::Value> {
        todo!()
    }
}

/// Delta content for streaming responses.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DeltaContent {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
}

/// A single choice in a streaming chunk response.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ChunkChoice {
    #[serde(default)]
    pub index: i64,
    pub delta: DeltaContent,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// `chatcmpl-<8 hex chars>` — the default completion id.
pub fn new_completion_id() -> String {
    format!("chatcmpl-{}", &uuid::Uuid::new_v4().simple().to_string()[..8])
}

/// Current Unix time in whole seconds (`int(time.time())`).
pub fn now_unix() -> i64 {
    chrono::Utc::now().timestamp()
}

fn chunk_object() -> String {
    "chat.completion.chunk".to_string()
}
fn completion_object() -> String {
    "chat.completion".to_string()
}
fn status_object() -> String {
    "chat.status".to_string()
}

/// OpenAI-compatible streaming chunk response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionChunk {
    #[serde(default = "new_completion_id")]
    pub id: String,
    /// Always `"chat.completion.chunk"`.
    #[serde(default = "chunk_object")]
    pub object: String,
    #[serde(default = "now_unix")]
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

impl ChatCompletionChunk {
    /// A chunk with a fresh id and timestamp.
    pub fn new(model: impl Into<String>, choices: Vec<ChunkChoice>) -> Self {
        Self { id: new_completion_id(), object: chunk_object(), created: now_unix(), model: model.into(), choices }
    }
}

impl Default for ChatCompletionChunk {
    fn default() -> Self {
        Self::new(String::new(), Vec::new())
    }
}

/// Lifecycle of a [`ChatStatusChunk`] step.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChatStatus {
    #[default]
    Processing,
    Completed,
    Error,
}

/// Streaming step-activity event emitted while the orchestrator works.
///
/// Interleaved in the same SSE stream as [`ChatCompletionChunk`] content
/// deltas. Clients distinguish it by `object == "chat.status"` and render
/// `message` as live progress (e.g. "Searching documents", "Running fetch").
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatStatusChunk {
    /// Always `"chat.status"`.
    #[serde(default = "status_object")]
    pub object: String,
    /// Machine-readable phase: analyze, search, plan, tools, read, streaming.
    #[serde(default)]
    pub step: String,
    /// Lifecycle of this step.
    #[serde(default)]
    pub status: ChatStatus,
    /// Human-readable activity to display.
    #[serde(default)]
    pub message: String,
}

impl Default for ChatStatusChunk {
    fn default() -> Self {
        Self { object: status_object(), step: String::new(), status: ChatStatus::Processing, message: String::new() }
    }
}

fn default_finish_reason() -> String {
    "stop".to_string()
}

/// A single choice in a non-streaming response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Choice {
    #[serde(default)]
    pub index: i64,
    pub message: ChatMessage,
    #[serde(default = "default_finish_reason")]
    pub finish_reason: String,
}

impl Default for Choice {
    fn default() -> Self {
        Self { index: 0, message: ChatMessage::default(), finish_reason: default_finish_reason() }
    }
}

/// Token usage information.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: i64,
    #[serde(default)]
    pub completion_tokens: i64,
    #[serde(default)]
    pub total_tokens: i64,
}

/// Reference to a source document used in the response.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Citation {
    pub document_id: String,
    #[serde(default)]
    pub document_name: Option<String>,
}

/// OpenAI-compatible non-streaming response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionResponse {
    #[serde(default = "new_completion_id")]
    pub id: String,
    /// Always `"chat.completion"`.
    #[serde(default = "completion_object")]
    pub object: String,
    #[serde(default = "now_unix")]
    pub created: i64,
    pub model: String,
    #[serde(default)]
    pub message_id: Option<String>,
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Usage,
    #[serde(default)]
    pub citations: Option<Vec<Citation>>,
}

impl ChatCompletionResponse {
    /// A response with a fresh id and timestamp.
    pub fn new(model: impl Into<String>, choices: Vec<Choice>) -> Self {
        Self {
            id: new_completion_id(),
            object: completion_object(),
            created: now_unix(),
            model: model.into(),
            message_id: None,
            choices,
            usage: Usage::default(),
            citations: None,
        }
    }
}

impl Default for ChatCompletionResponse {
    fn default() -> Self {
        Self::new(String::new(), Vec::new())
    }
}

// --- Document models ---

/// Request to add a document source.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct AddDocumentRequest {
    /// File path or URL for the document source.
    pub source: String,
    /// Current working directory.
    #[serde(default)]
    pub cwd: Option<String>,
    /// Use LLM to generate a descriptive filename for txt files from the
    /// document content.
    #[serde(default)]
    pub generate_filename: bool,
}

fn default_pending() -> String {
    "pending".to_string()
}

/// Single document info.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default = "default_pending")]
    pub status: String,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub filename: Option<String>,
}

impl Default for DocumentResponse {
    fn default() -> Self {
        Self { id: None, path: None, status: default_pending(), summary: None, filename: None }
    }
}

/// List of documents.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DocumentListResponse {
    pub documents: Vec<DocumentResponse>,
}

/// Response after deleting a document.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DeleteDocumentResponse {
    pub id: String,
    pub deleted: bool,
    pub message: String,
}

// --- Asset models ---

/// Response after uploading an asset.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct AssetUploadResponse {
    pub asset_id: String,
    pub filename: String,
    pub content_type: String,
    /// Original file size in bytes.
    pub size: i64,
    /// True when the file was converted to markdown.
    #[serde(default)]
    pub markdown: bool,
}

/// Metadata and content for a stored asset.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct AssetResponse {
    pub asset_id: String,
    pub filename: String,
    pub content_type: String,
    pub size: i64,
    pub markdown: bool,
    /// Parsed markdown content.
    #[serde(default)]
    pub content: Option<String>,
}

/// Response after deleting an asset.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DeleteAssetResponse {
    pub asset_id: String,
    pub deleted: bool,
    pub message: String,
}

// --- Feedback models ---

/// Thumbs up or down (`Literal["up", "down"]`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Rating {
    #[default]
    Up,
    Down,
}

impl Rating {
    /// The wire string for this rating.
    pub fn as_str(&self) -> &'static str {
        match self {
            Rating::Up => "up",
            Rating::Down => "down",
        }
    }
}

/// Feedback on an assistant response.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct FeedbackRequest {
    /// Session ID of the conversation.
    pub session_id: String,
    /// Message ID of the assistant response.
    pub message_id: String,
    /// Project ID to scope the feedback to.
    #[serde(default)]
    pub project_id: Option<String>,
    /// Thumbs up or down.
    pub rating: Rating,
}

/// Result of recording feedback.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct FeedbackResponse {
    pub feedback_id: String,
    pub session_id: String,
    pub message_id: String,
    pub rating: Rating,
    /// Whether the Q&A pair was added to or removed from the vector index.
    #[serde(default)]
    pub indexed: bool,
}
