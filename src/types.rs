//! Core shared types.  Port of `smartloop/types.py`.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Default LLM sampling options.
#[derive(Debug, Clone, Copy, Default)]
pub struct LLMOptions;

impl LLMOptions {
    /// Default sampling temperature.
    pub const TEMPERATURE: f64 = 0.3;
}

/// Input modality of a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "multi-modal")]
    MultiModal,
}

impl ModelType {
    /// The enum's wire value (`"text"` / `"multi-modal"`).
    pub fn value(&self) -> &'static str {
        match self {
            ModelType::Text => "text",
            ModelType::MultiModal => "multi-modal",
        }
    }
}

/// Where a model runs: weights on this machine, or behind the platform API.
///
/// The catalog's `type` field.  An absent or unknown value is `Local`, which
/// is what every model was before remote ones existed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    #[default]
    Local,
    Remote,
}

impl ModelKind {
    /// The wire value (`"local"` / `"remote"`).
    pub fn value(&self) -> &'static str {
        match self {
            ModelKind::Local => "local",
            ModelKind::Remote => "remote",
        }
    }

    pub fn is_local(&self) -> bool {
        *self == ModelKind::Local
    }
}

impl<'de> Deserialize<'de> for ModelKind {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value: Option<String> = Option::deserialize(d)?;
        Ok(match value.as_deref() {
            Some("remote") => ModelKind::Remote,
            Some("local") | None => ModelKind::Local,
            Some(other) => {
                tracing::warn!("[models] unknown model type '{other}' — treating as local");
                ModelKind::Local
            }
        })
    }
}

/// How a model's attention cost grows with context length (the catalog's
/// `attention`).  Reported only; nothing routes on it yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum Attention {
    /// Every layer attends the whole context: quadratic.
    Full,
    /// Most layers attend a local window, with global layers interleaved.
    SlidingWindow,
    /// Linear-time attention throughout.
    Linear,
    /// Linear-time layers mixed with full-attention ones.
    Hybrid,
    /// Each token attends a selected subset of the context (DeepSeek
    /// Sparse Attention, MiniMax Sparse Attention).
    Sparse,
}

impl Attention {
    /// Resolve a wire value; `None` when unknown.
    pub fn from_value(value: &str) -> Option<Self> {
        match value {
            "full" => Some(Attention::Full),
            "sliding_window" => Some(Attention::SlidingWindow),
            "linear" => Some(Attention::Linear),
            "hybrid" => Some(Attention::Hybrid),
            "sparse" => Some(Attention::Sparse),
            _ => None,
        }
    }
}

/// What a model can be routed to (or, for `Thinking`, how it answers).
///
/// Deserialization accepts the legacy spelling `"skills"` (any case, padded)
/// for `Reasoning`, mirroring the Python `_missing_` hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ModelCapability {
    /// `"general"`
    General,
    /// `"reasoning"` — the model that reasons over a set of options before
    /// acting (skill matching, MCP server/tool choice, planning).
    Reasoning,
    /// `"dp"`
    Dp,
    /// `"vision"`
    Vision,
    /// `"coding"`
    Coding,
    /// `"thinking"` — the chat template reads `enable_thinking`.  Not a
    /// routing target.
    Thinking,
}

impl ModelCapability {
    /// All capabilities in declaration order.
    pub const ALL: [ModelCapability; 6] = [
        ModelCapability::General,
        ModelCapability::Reasoning,
        ModelCapability::Dp,
        ModelCapability::Vision,
        ModelCapability::Coding,
        ModelCapability::Thinking,
    ];

    /// The enum's wire value.
    pub fn value(&self) -> &'static str {
        match self {
            ModelCapability::General => "general",
            ModelCapability::Reasoning => "reasoning",
            ModelCapability::Dp => "dp",
            ModelCapability::Vision => "vision",
            ModelCapability::Coding => "coding",
            ModelCapability::Thinking => "thinking",
        }
    }

    /// Resolve a capability string, including names it used to go by
    /// (`"skills"` -> `Reasoning`).  `None` when the value is unknown
    /// (Python raises `ValueError` there).
    pub fn from_value(value: &str) -> Option<Self> {
        if let Some(c) = Self::ALL.iter().find(|c| c.value() == value) {
            return Some(*c);
        }
        if value.trim().to_lowercase() == "skills" {
            return Some(ModelCapability::Reasoning);
        }
        None
    }
}

impl std::fmt::Display for ModelCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.value())
    }
}

impl std::str::FromStr for ModelCapability {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_value(s).ok_or_else(|| format!("'{s}' is not a valid ModelCapability"))
    }
}

impl Serialize for ModelCapability {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.value())
    }
}

impl<'de> Deserialize<'de> for ModelCapability {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// A retrieved text fragment with a confidence score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default = "new_document_id")]
    pub id: String,
}

fn new_document_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

impl Default for Document {
    fn default() -> Self {
        Document { text: String::new(), confidence: 0.0, id: new_document_id() }
    }
}

impl Document {
    /// Build a document; a missing id gets a fresh UUID4.
    pub fn new(text: impl Into<String>, confidence: f64, id: Option<String>) -> Self {
        Document { text: text.into(), confidence, id: id.unwrap_or_else(new_document_id) }
    }

    /// `{"text", "confidence", "id"}` as a JSON object.
    pub fn to_dict(&self) -> serde_json::Value {
        serde_json::json!({"text": self.text, "confidence": self.confidence, "id": self.id})
    }
}
