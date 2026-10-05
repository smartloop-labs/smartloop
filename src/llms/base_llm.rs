//! Base model for all LLM classes.
//!
//! Port of `smartloop/llms/base_llm.py`.  The Python `BaseLLM` was a generic
//! LangChain `LLM` subclass; here it is a trait implemented by the local
//! llama.cpp wrapper ([`crate::llms::llama_cpp::LlamaCpp`]) and the model-host
//! handle (`crate::model_host::remote_llm::RemoteLLM`), and shared as
//! `Arc<dyn BaseLLM>`.  The LangChain plumbing it relied on (`_call`, `_stream`,
//! `invoke`, `GenerationChunk`) is hand-ported here.

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::constants::SYSTEM_PROMPT;

/// `<system>…</system>`, `<user>…</user>` and `<assistant>…</assistant>`
/// (`re.DOTALL`, non-greedy).
static ROLE_TAG_PATTERNS: Lazy<[(&'static str, Regex); 3]> = Lazy::new(|| {
    [
        ("system", Regex::new(r"(?s)<system>(.*?)</system>").expect("valid regex")),
        ("user", Regex::new(r"(?s)<user>(.*?)</user>").expect("valid regex")),
        ("assistant", Regex::new(r"(?s)<assistant>(.*?)</assistant>").expect("valid regex")),
    ]
});

/// Logger name of the LLM wrappers.
const LOG_TARGET: &str = "smartloop.llms";

/// `threading.Event` used to cancel an in-flight generation between tokens.
pub type CancelEvent = Arc<AtomicBool>;

/// One chat message in the standard `{"role": ..., "content": ...}` shape.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    /// Build a message from a role and content.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into() }
    }
}

/// LangChain `GenerationChunk`: one piece of streamed text.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct GenerationChunk {
    pub text: String,
}

/// The per-call keyword arguments `_call`/`_stream` read (`**kwargs`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct CallOptions {
    /// Cap on generated tokens; `None` uses the remaining context budget.
    pub max_tokens: Option<usize>,
    /// Force the reasoning block on/off for this call; `None` means off.
    pub thinking: Option<bool>,
}

/// A stream of generated chunks.  Not `Send`: the local implementation holds
/// the per-context lock for the lifetime of the stream, so it must be drained
/// on the thread that started it.
pub type GenerationStream<'a> = Box<dyn Iterator<Item = GenerationChunk> + 'a>;

/// LLM base trait.
pub trait BaseLLM: Send + Sync {
    /// The project this LLM serves.
    fn project_id(&self) -> &str;

    /// Whether this LLM was created for training.
    fn is_training(&self) -> bool {
        false
    }

    /// Number of tokens `text` encodes to.  LLM specific.
    fn get_num_tokens(&self, text: &str) -> usize;

    /// Format the prompt for the LLM in a standardized way: a system message
    /// (`system_message` or [`SYSTEM_PROMPT`]) followed by the user prompt.
    ///
    /// Creates a standard message format that can be used across different
    /// LLM implementations, similar to llama.cpp's approach.  An empty
    /// `system_message` falls back to the default, as Python's `or` did.
    fn format_prompt(&self, prompt: &str, system_message: Option<&str>) -> Vec<ChatMessage> {
        let system = match system_message {
            Some(s) if !s.is_empty() => s,
            _ => SYSTEM_PROMPT,
        };
        // Default implementation returns messages - implementations can override.
        vec![ChatMessage::new("system", system), ChatMessage::new("user", prompt)]
    }

    /// Convert a prompt string to messages using `<system>`/`<user>`/`<assistant>`
    /// tag-based separation, merging consecutive same-role messages; with no tags
    /// the raw prompt becomes `format_prompt("user_question: {prompt}")`.
    fn create_messages_from_prompt(&self, prompt: &str) -> Vec<ChatMessage> {
        // Find all tags with their content and position, then sort by position.
        let mut all_matches: Vec<(usize, &'static str, String)> = Vec::new();
        for (role, pattern) in ROLE_TAG_PATTERNS.iter() {
            for caps in pattern.captures_iter(prompt) {
                let whole = caps.get(0).expect("group 0 always matches");
                let content = caps.get(1).map(|m| m.as_str()).unwrap_or_default().trim().to_string();
                all_matches.push((whole.start(), role, content));
            }
        }
        all_matches.sort_by_key(|(pos, _, _)| *pos);

        // Create messages from tags; only non-empty content is kept.
        let messages: Vec<ChatMessage> = all_matches
            .into_iter()
            .filter(|(_, _, content)| !content.is_empty())
            .map(|(_, role, content)| ChatMessage::new(role, content))
            .collect();

        // If no tags found, process the raw prompt as a user message.
        if messages.is_empty() {
            return self.format_prompt(&format!("user_question: {prompt}"), None);
        }

        // Merge consecutive messages with the same role so the model always
        // sees alternating user/assistant turns.
        let mut merged: Vec<ChatMessage> = Vec::with_capacity(messages.len());
        for msg in messages {
            match merged.last_mut() {
                Some(last) if last.role == msg.role => {
                    last.content = format!("{}\n{}", last.content, msg.content);
                }
                _ => merged.push(msg),
            }
        }
        merged
    }

    /// Apply the LLM-specific chat template to the standard message format.
    fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage>;

    /// Effective context size in tokens.
    fn get_max_tokens(&self) -> usize;

    /// LangChain `_call`: run the LLM on the given input.
    fn call(&self, prompt: &str, stop: Option<&[String]>, options: CallOptions) -> anyhow::Result<String>;

    /// LangChain `_stream` / `stream`: stream the LLM on the given prompt.
    fn stream<'a>(&'a self, prompt: &str, options: CallOptions) -> GenerationStream<'a>;

    /// Invoke the LLM once.  Never fails: an error is returned as
    /// `"Error generating response: ..."`, an empty reply as
    /// `"No response generated"`, and a trailing `<|eot_id|>` is trimmed.
    ///
    /// `max_tokens` caps generated tokens — use a small value for short
    /// classification-style calls (e.g. skill matching) to avoid runaway
    /// greedy generation when the model fails to emit a stop token.
    /// `thinking` forces the reasoning block on (`true`) or off (`false`) for
    /// this call, whatever the user's setting says; `None` leaves it alone.
    fn invoke(&self, prompt: &str, max_tokens: Option<usize>, thinking: Option<bool>) -> String {
        let options = CallOptions { max_tokens, thinking };
        match self.call(prompt, None, options) {
            Ok(response) => {
                let mut cleaned = response.trim();
                if let Some(stripped) = cleaned.strip_suffix("<|eot_id|>") {
                    cleaned = stripped.trim();
                }
                if cleaned.is_empty() { "No response generated".to_string() } else { cleaned.to_string() }
            }
            Err(e) => {
                tracing::error!(target: LOG_TARGET, "LLM invocation failed: {e}");
                format!("Error generating response: {e}")
            }
        }
    }

    /// True when this model's chat template can reason.
    fn supports_thinking(&self) -> bool {
        false
    }

    /// The model's name as the user chose it, or `""`.
    fn model_label(&self) -> String {
        String::new()
    }

    /// The cancellation event checked between generated chunks (`_cancelled`).
    fn cancelled(&self) -> Option<CancelEvent>;

    /// Set or clear the cancellation event (`llm._cancelled = ...`).
    fn set_cancelled(&self, event: Option<CancelEvent>);

    /// Type of language model, used for logging only (`_llm_type`).
    fn llm_type(&self) -> &str;

    /// The underlying llama.cpp model when this LLM runs in-process
    /// (`llm.model` on a `LlamaCpp`); `None` for remote handles.
    fn llama_model(&self) -> Option<Arc<dyn crate::llms::llama_cpp::LlamaModel>> {
        None
    }

    /// Downcasting hook.
    fn as_any(&self) -> &dyn Any;
}
