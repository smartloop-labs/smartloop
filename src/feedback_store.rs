//! Persistent JSONL feedback store.
//!
//! Port of `smartloop/feedback_store.py`.  Each project stores feedback as
//! `<project_dir>/feedback.jsonl` where every line is a JSON-encoded
//! [`FeedbackEntry`].

use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// One turn of the conversation captured alongside a rating.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ConversationTurn {
    pub role: String,
    pub content: String,
}

/// Thumbs up / thumbs down (`Literal["up", "down"]`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub enum Rating {
    #[default]
    #[serde(rename = "up")]
    Up,
    #[serde(rename = "down")]
    Down,
}

/// A user's rating of one assistant answer.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct FeedbackEntry {
    pub feedback_id: String,
    pub session_id: String,
    pub message_id: String,
    pub rating: Rating,
    pub question: String,
    pub answer: String,
    #[serde(default)]
    pub conversation: Vec<ConversationTurn>,
    /// Unix time in seconds (float).
    pub created_at: f64,
}

impl FeedbackEntry {
    /// New entry with a 12-hex-char id and the current time.
    pub fn create(
        session_id: &str,
        message_id: &str,
        rating: Rating,
        question: &str,
        answer: &str,
        conversation: Option<Vec<ConversationTurn>>,
    ) -> FeedbackEntry {
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        FeedbackEntry {
            feedback_id: uuid::Uuid::new_v4().simple().to_string()[..12].to_string(),
            session_id: session_id.to_string(),
            message_id: message_id.to_string(),
            rating,
            question: question.to_string(),
            answer: answer.to_string(),
            conversation: conversation.unwrap_or_default(),
            created_at,
        }
    }
}

/// JSONL store of [`FeedbackEntry`] records for one project.
#[derive(Debug, Clone)]
pub struct FeedbackStore {
    path: PathBuf,
}

impl FeedbackStore {
    /// Store backed by `<project_dir>/feedback.jsonl` (created lazily).
    pub fn new(project_dir: &str) -> Self {
        FeedbackStore { path: PathBuf::from(project_dir).join("feedback.jsonl") }
    }

    /// Append one entry, creating the parent directory if needed.
    pub fn append(&self, entry: &FeedbackEntry) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut fh = OpenOptions::new().create(true).append(true).open(&self.path)?;
        fh.write_all(format!("{}\n", to_json(entry)?).as_bytes())
    }

    /// All entries; invalid lines are skipped; empty when the file is absent.
    pub fn load(&self) -> std::io::Result<Vec<FeedbackEntry>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let mut entries = Vec::new();
        for line in BufReader::new(fs::File::open(&self.path)?).lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // Bad JSON and entries that fail validation are both skipped.
            if let Ok(entry) = serde_json::from_str::<FeedbackEntry>(line) {
                entries.push(entry);
            }
        }
        Ok(entries)
    }

    /// The entry for (`session_id`, `message_id`), if any.
    pub fn find(&self, session_id: &str, message_id: &str) -> std::io::Result<Option<FeedbackEntry>> {
        Ok(self
            .load()?
            .into_iter()
            .find(|e| e.session_id == session_id && e.message_id == message_id))
    }

    /// Rewrite the file without the entry whose id is `feedback_id`.
    pub fn remove(&self, feedback_id: &str) -> std::io::Result<()> {
        let mut out = String::new();
        for entry in self.load()?.into_iter().filter(|e| e.feedback_id != feedback_id) {
            out.push_str(&to_json(&entry)?);
            out.push('\n');
        }
        fs::write(&self.path, out)
    }
}

/// `model_dump_json()`: compact JSON.
fn to_json(entry: &FeedbackEntry) -> std::io::Result<String> {
    serde_json::to_string(entry).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}
