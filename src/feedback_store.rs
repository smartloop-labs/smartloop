//! Persistent JSONL feedback store.
//!
//! Port of `smartloop/feedback_store.py`.  Each project stores feedback as
//! `<project_dir>/feedback.jsonl` where every line is a JSON-encoded
//! [`FeedbackEntry`].

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
        let _ = (session_id, message_id, rating, question, answer, conversation);
        todo!()
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
        let _ = (entry, &self.path);
        todo!()
    }

    /// All entries; invalid lines are skipped; empty when the file is absent.
    pub fn load(&self) -> std::io::Result<Vec<FeedbackEntry>> {
        todo!()
    }

    /// The entry for (`session_id`, `message_id`), if any.
    pub fn find(&self, session_id: &str, message_id: &str) -> std::io::Result<Option<FeedbackEntry>> {
        let _ = (session_id, message_id);
        todo!()
    }

    /// Rewrite the file without the entry whose id is `feedback_id`.
    pub fn remove(&self, feedback_id: &str) -> std::io::Result<()> {
        let _ = feedback_id;
        todo!()
    }
}
