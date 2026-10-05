//! Persistent JSONL conversation store.
//!
//! Port of `smartloop/conversation_store.py`.  Each session is stored as
//! `<home_dir>/conversations/<session_id>.jsonl` where every line is a
//! JSON-encoded message.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// One chat message as persisted (and as accepted by the chat API).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MessageInput {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub message_id: Option<String>,
    #[serde(default)]
    pub rating: Option<String>,
    #[serde(default)]
    pub citations: Option<Vec<serde_json::Value>>,
}

impl MessageInput {
    /// A message with only `role` and `content` set.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        MessageInput { role: role.into(), content: content.into(), ..Default::default() }
    }
}

/// Persist chat sessions as JSONL files under `<home_dir>/conversations/`.
#[derive(Debug, Clone)]
pub struct ConversationStore {
    dir: PathBuf,
}

impl ConversationStore {
    /// Create the store, making `<home_dir>/conversations/` if needed.
    pub fn new(home_dir: &str) -> std::io::Result<Self> {
        let _ = home_dir;
        todo!()
    }

    fn path(&self, session_id: &str) -> PathBuf {
        self.dir.join(format!("{session_id}.jsonl"))
    }

    /// Return all messages for `session_id`; empty when no file exists.
    /// Lines that are not valid JSON are skipped.
    pub fn load(&self, session_id: &str) -> std::io::Result<Vec<MessageInput>> {
        let _ = self.path(session_id);
        todo!()
    }

    /// Append a single message to `session_id`'s JSONL file.
    pub fn append(&self, session_id: &str, message: &MessageInput) -> std::io::Result<()> {
        let _ = (session_id, message);
        todo!()
    }

    /// Rewrite the entire session file with `messages`.
    pub fn save(&self, session_id: &str, messages: &[MessageInput]) -> std::io::Result<()> {
        let _ = (session_id, messages);
        todo!()
    }

    /// True when a conversation file exists for `session_id`.
    pub fn exists(&self, session_id: &str) -> bool {
        let _ = session_id;
        todo!()
    }
}
