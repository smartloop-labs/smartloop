//! Port of `smartloop/api/model/chat.py`.
//!
//! OpenAI-compatible chat completion schemas.

use serde::{Deserialize, Serialize};

/// The role of a chat message author (`Literal["system", "user", "assistant"]`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, Hash, utoipa::ToSchema)]
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
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
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
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, utoipa::ToSchema)]
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
        let mut errors = Vec::new();
        float_range(&mut errors, "temperature", self.temperature, Some(0.0), Some(2.0));
        if let Some(max_tokens) = self.max_tokens
            && max_tokens < 1
        {
            errors.push(serde_json::json!({
                "type": "greater_than_equal",
                "loc": ["body", "max_tokens"],
                "msg": "Input should be greater than or equal to 1",
                "input": max_tokens,
                "ctx": {"ge": 1},
            }));
        }
        float_range(&mut errors, "top_p", self.top_p, Some(0.0), Some(1.0));
        float_range(&mut errors, "frequency_penalty", self.frequency_penalty, Some(-2.0), Some(2.0));
        float_range(&mut errors, "presence_penalty", self.presence_penalty, Some(-2.0), Some(2.0));
        errors
    }
}

/// A bound as pydantic prints it in a message: `0.0` reads "0", `0.5` "0.5".
fn format_bound(bound: f64) -> String {
    if bound.fract() == 0.0 && bound.is_finite() {
        format!("{}", bound as i64)
    } else {
        format!("{bound}")
    }
}

/// Append the pydantic `ge`/`le` error for a float field outside its range.
fn float_range(
    errors: &mut Vec<serde_json::Value>,
    field: &str,
    value: Option<f64>,
    ge: Option<f64>,
    le: Option<f64>,
) {
    let Some(value) = value else {
        return;
    };
    if let Some(ge) = ge
        && !(value >= ge)
    {
        errors.push(serde_json::json!({
            "type": "greater_than_equal",
            "loc": ["body", field],
            "msg": format!("Input should be greater than or equal to {}", format_bound(ge)),
            "input": value,
            "ctx": {"ge": ge},
        }));
        return;
    }
    if let Some(le) = le
        && !(value <= le)
    {
        errors.push(serde_json::json!({
            "type": "less_than_equal",
            "loc": ["body", field],
            "msg": format!("Input should be less than or equal to {}", format_bound(le)),
            "input": value,
            "ctx": {"le": le},
        }));
    }
}

/// Delta content for streaming responses.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct DeltaContent {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
}

/// A single choice in a streaming chunk response.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, utoipa::ToSchema)]
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
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, utoipa::ToSchema)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, utoipa::ToSchema)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, utoipa::ToSchema)]
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
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: i64,
    #[serde(default)]
    pub completion_tokens: i64,
    #[serde(default)]
    pub total_tokens: i64,
}

/// What kind of source a [`Citation`] names, and so what opening it does.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum CitationSource {
    /// A file on this machine, opened from `document_path` or `path`.
    Document,
    /// A web page, opened from `url`.
    Web,
    /// A connected app's tool call; nothing to open.
    Mcp,
}

impl CitationSource {
    /// The wire value, as a citation's `source` key carries it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Document => "document",
            Self::Web => "web",
            Self::Mcp => "mcp",
        }
    }

    /// The source a wire value names; `None` for one this build doesn't know.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "document" => Some(Self::Document),
            "web" => Some(Self::Web),
            "mcp" => Some(Self::Mcp),
            _ => None,
        }
    }
}

/// A citation's `source`, read as `None` when it names a kind this build
/// doesn't know, as another version's agent may send.
fn known_source<'de, D>(deserializer: D) -> Result<Option<CitationSource>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<String>::deserialize(deserializer)?;
    Ok(value.as_deref().and_then(CitationSource::parse))
}

/// Reference to a source used in the response, the same in a streamed
/// `chat.citations` event and a non-streamed response.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct Citation {
    /// Absent from an older agent's citation of a document by name alone.
    #[serde(default)]
    pub document_id: String,
    #[serde(default)]
    pub document_name: Option<String>,
    /// What kind of source this is.
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "known_source")]
    pub source: Option<CitationSource>,
    /// Where a web citation opens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// A document's file, relative to its project (`documents/invoice.pdf`),
    /// as the documents API gives it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_path: Option<String>,
    /// The pages of a document the answer drew on, in order, as the
    /// document's index recorded them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<Vec<i64>>)]
    pub page_numbers: Option<Vec<serde_json::Value>>,
    /// An explored file's absolute path.  Prefer `document_path`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// The connected app an MCP citation's tool belongs to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_name: Option<String>,
    /// The tool an MCP citation names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// Any other public key a citation carries, passed through as-is.
    #[serde(flatten, default)]
    #[schema(ignore)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// OpenAI-compatible non-streaming response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, utoipa::ToSchema)]
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
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, utoipa::ToSchema)]
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
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct DocumentListResponse {
    pub documents: Vec<DocumentResponse>,
}

/// Response after deleting a document.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct DeleteDocumentResponse {
    pub id: String,
    pub deleted: bool,
    pub message: String,
}

// --- Asset models ---

/// Response after uploading an asset.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
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
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
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
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct DeleteAssetResponse {
    pub asset_id: String,
    pub deleted: bool,
    pub message: String,
}

// --- Feedback models ---

/// Thumbs up or down (`Literal["up", "down"]`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, Hash, utoipa::ToSchema)]
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
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
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
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct FeedbackResponse {
    pub feedback_id: String,
    pub session_id: String,
    pub message_id: String,
    pub rating: Rating,
    /// Whether the Q&A pair was added to or removed from the vector index.
    #[serde(default)]
    pub indexed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_pydantic_model() {
        let request: ChatCompletionRequest =
            serde_json::from_value(serde_json::json!({"model": "m", "messages": []})).unwrap();
        assert_eq!(request.temperature, Some(0.7));
        assert_eq!(request.top_p, Some(1.0));
        assert_eq!(request.frequency_penalty, Some(0.0));
        assert!(!request.stream);
        assert!(request.validation_errors().is_empty());
    }

    #[test]
    fn out_of_range_fields_are_reported_like_pydantic() {
        let request = ChatCompletionRequest {
            temperature: Some(2.5),
            max_tokens: Some(0),
            top_p: Some(-0.1),
            ..Default::default()
        };
        let errors = request.validation_errors();
        assert_eq!(errors.len(), 3);
        assert_eq!(errors[0]["loc"], serde_json::json!(["body", "temperature"]));
        assert_eq!(errors[0]["msg"], "Input should be less than or equal to 2");
        assert_eq!(errors[1]["msg"], "Input should be greater than or equal to 1");
        assert_eq!(errors[2]["type"], "greater_than_equal");
    }

    #[test]
    fn chunks_serialize_with_their_object_tag() {
        let chunk = ChatCompletionChunk::new("m", vec![]);
        let value = serde_json::to_value(&chunk).unwrap();
        assert_eq!(value["object"], "chat.completion.chunk");
        let id = value["id"].as_str().unwrap();
        assert!(id.starts_with("chatcmpl-"));
        assert_eq!(id.len(), "chatcmpl-".len() + 8);
        let status = serde_json::to_value(ChatStatusChunk::default()).unwrap();
        assert_eq!(
            status,
            serde_json::json!({"object": "chat.status", "step": "", "status": "processing", "message": ""})
        );
    }
}
