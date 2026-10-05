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

use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use crate::constants::SYSTEM_PROMPT;

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
    fn format_prompt(&self, prompt: &str, system_message: Option<&str>) -> Vec<ChatMessage> {
        let _ = (prompt, system_message);
        todo!()
    }

    /// Convert a prompt string to messages using `<system>`/`<user>`/`<assistant>`
    /// tag-based separation, merging consecutive same-role messages; with no tags
    /// the raw prompt becomes `format_prompt("user_question: {prompt}")`.
    fn create_messages_from_prompt(&self, prompt: &str) -> Vec<ChatMessage> {
        let _ = prompt;
        todo!()
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
    fn invoke(&self, prompt: &str, max_tokens: Option<usize>, thinking: Option<bool>) -> String {
        let _ = (prompt, max_tokens, thinking);
        todo!()
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
