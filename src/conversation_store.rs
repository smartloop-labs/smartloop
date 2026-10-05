//! Persistent JSONL conversation store.
//!
//! Port of `smartloop/conversation_store.py`.  Each session is stored as
//! `<home_dir>/conversations/<session_id>.jsonl` where every line is a
//! JSON-encoded `{"role": ..., "content": ...}` message.  Serialization and
//! deserialization are intentionally trivial so the file can be inspected or
//! replayed with any JSON tool.

use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
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
        let dir = PathBuf::from(home_dir).join("conversations");
        fs::create_dir_all(&dir)?;
        Ok(ConversationStore { dir })
    }

    fn path(&self, session_id: &str) -> PathBuf {
        self.dir.join(format!("{session_id}.jsonl"))
    }

    /// Return all messages for `session_id`; empty when no file exists.
    ///
    /// Lines that are not valid JSON are skipped; JSON that does not validate
    /// as a message is an error, as pydantic's validation error was.
    pub fn load(&self, session_id: &str) -> std::io::Result<Vec<MessageInput>> {
        let path = self.path(session_id);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let mut messages = Vec::new();
        for line in BufReader::new(fs::File::open(&path)?).lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let message = serde_json::from_value::<MessageInput>(value)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            messages.push(message);
        }
        Ok(messages)
    }

    /// Append a single message to `session_id`'s JSONL file.
    pub fn append(&self, session_id: &str, message: &MessageInput) -> std::io::Result<()> {
        let mut fh = OpenOptions::new().create(true).append(true).open(self.path(session_id))?;
        fh.write_all(format!("{}\n", to_json(message)?).as_bytes())
    }

    /// Rewrite the entire session file with `messages`.
    pub fn save(&self, session_id: &str, messages: &[MessageInput]) -> std::io::Result<()> {
        let mut out = String::new();
        for msg in messages {
            out.push_str(&to_json(msg)?);
            out.push('\n');
        }
        fs::write(self.path(session_id), out)
    }

    /// True when a conversation file exists for `session_id`.
    pub fn exists(&self, session_id: &str) -> bool {
        self.path(session_id).exists()
    }
}

/// `model_dump_json()`: compact JSON with `None` fields as `null`.
fn to_json(message: &MessageInput) -> std::io::Result<String> {
    serde_json::to_string(message).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}
