//! Llama.cpp wrapper.
//!
//! Port of `smartloop/llms/llama_cpp.py`.  The `llama_cpp.Llama` surface it
//! wraps lives in [`crate::llms::llama`] and is re-exported here.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::{Mutex, ReentrantMutex};
use serde_json::{Map, Value};

#[allow(unused_imports)]
use crate::constants::OPENING_TOKENS;
use crate::llms::base_llm::{BaseLLM, CallOptions, CancelEvent, ChatMessage, GenerationStream};
#[allow(unused_imports)]
use crate::llms::thinking::{ThinkingFilter, strip_thinking};

pub use crate::llms::llama::{
    ChatCompletionChunks, ChatCompletionOutput, ChatCompletionRequest, ChatHandler, ChatHandlerRequest, Llama,
    LlamaModel, LlamaParams, LlamaState, LlamaToken,
};

/// Output budget below which a reasoning model is told not to think.  Decision
/// calls cap output hard (8-256 tokens for a yes/no or a short label); a
/// `<think>` block would consume the whole budget and return a truncated
/// thought where the caller expected "yes".
pub const THINKING_MIN_BUDGET: usize = 512;

/// Restores the model's `_slp_thinking` flag when dropped
/// (the `_thinking_override` / `_thinking_for_budget` context managers).
pub struct ThinkingGuard<'a> {
    model: &'a dyn LlamaModel,
    /// `Some(previous)` when the flag was changed and must be restored.
    previous: Option<Option<bool>>,
}

impl Drop for ThinkingGuard<'_> {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            self.model.state().set_thinking(previous);
        }
    }
}

/// Llama wrapper.
pub struct LlamaCpp {
    /// The loaded llama.cpp context.
    pub model: Arc<dyn LlamaModel>,
    pub project_id: String,
    pub is_training: bool,
    /// Sampling temperature for every call.
    pub temperature: f32,
    /// The name this model was *selected* under, which is what the picker
    /// shows and therefore what a status event has to say.
    slp_model_name: Option<String>,
    cancelled: Mutex<Option<CancelEvent>>,
}

impl LlamaCpp {
    /// Overhead reserved for chat-template formatting when sizing prompt/output.
    pub const CTX_OVERHEAD: usize = 64;
    /// Output tokens to keep free when a prompt has to be truncated to fit.
    pub const MIN_OUTPUT_RESERVE: usize = 512;

    /// Wrap a loaded model.
    pub fn new(model: Arc<dyn LlamaModel>, project_id: impl Into<String>, is_training: bool, model_name: Option<String>) -> Self {
        Self {
            model,
            project_id: project_id.into(),
            is_training,
            temperature: 0.3,
            slp_model_name: model_name,
            cancelled: Mutex::new(None),
        }
    }

    /// The lock serializing generation on this llama.cpp context.  It lives
    /// on the model, so two wrappers over one context share one lock; it is
    /// reentrant because generation tokenizes while holding it.
    pub fn context_lock(&self) -> &ReentrantMutex<()> {
        self.model.state().context_lock()
    }

    /// The model's trained context length from GGUF metadata
    /// (`{arch}.context_length`), or 0.
    pub fn get_n_ctx_train(&self) -> usize {
        todo!()
    }

    /// Output tokens that fit after the prompt, never exceeding n_ctx (floored at 1).
    pub fn get_available_tokens(&self, prompt: &str) -> usize {
        let _ = prompt;
        todo!()
    }

    /// Truncate an over-long prompt so it fits n_ctx with room for output,
    /// keeping head (2/3) and tail with a truncation marker in between.
    pub fn fit_prompt(&self, prompt: &str) -> String {
        let _ = prompt;
        todo!()
    }

    /// `` | 512 prompt, ttft 1.8s, 12.4 tok/s`` for a generation; times are
    /// epoch seconds (`time.time()`).  Empty when nothing was generated.
    pub fn rate(
        prompt_tokens: usize,
        generated: usize,
        start_time: f64,
        first_token_at: Option<f64>,
        opening_at: Option<f64>,
    ) -> String {
        let _ = (prompt_tokens, generated, start_time, first_token_at, opening_at);
        todo!()
    }

    /// Whether this call asked to reason.  Unspecified means no.
    pub fn requested_thinking(options: &CallOptions) -> bool {
        options.thinking.unwrap_or(false)
    }

    /// Force the reasoning block on or off for one call (no-op when
    /// `thinking` is `None` or the model has no bound flag).
    pub fn thinking_override(&self, thinking: Option<bool>) -> ThinkingGuard<'_> {
        let _ = thinking;
        todo!()
    }

    /// Suppress the reasoning block for a call whose effective output budget
    /// is below [`THINKING_MIN_BUDGET`].  Only applies when the bound flag is on.
    pub fn thinking_for_budget(&self, max_tokens: Option<usize>) -> ThinkingGuard<'_> {
        let _ = max_tokens;
        todo!()
    }

    /// Extract the text of one OpenAI-shaped chunk (`delta.content`,
    /// `message.content` or `text`).
    pub fn chunk_text(chunk: &Value) -> String {
        let _ = chunk;
        todo!()
    }

    /// Return a dictionary of identifying parameters.
    pub fn identifying_params(&self) -> Map<String, Value> {
        todo!()
    }

    /// GGUF metadata of the wrapped model.
    pub fn metadata(&self) -> HashMap<String, String> {
        self.model.metadata()
    }
}

impl Drop for LlamaCpp {
    fn drop(&mut self) {
        // Model resources are released when the last `Arc` to the context
        // drops (`Llama::drop`); nothing to do per wrapper.
    }
}

impl BaseLLM for LlamaCpp {
    fn project_id(&self) -> &str {
        &self.project_id
    }

    fn is_training(&self) -> bool {
        self.is_training
    }

    /// Use llama.cpp's tokenizer to count the tokens of `text`.
    fn get_num_tokens(&self, text: &str) -> usize {
        let _ = text;
        todo!()
    }

    /// llama.cpp applies the chat template inside `create_chat_completion`,
    /// so messages are returned unchanged.
    fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        messages
    }

    /// Effective context size: the smaller of the runtime `n_ctx` and the
    /// model's trained context length.
    fn get_max_tokens(&self) -> usize {
        todo!()
    }

    fn call(&self, prompt: &str, stop: Option<&[String]>, options: CallOptions) -> anyhow::Result<String> {
        let _ = (prompt, stop, options);
        todo!()
    }

    fn stream<'a>(&'a self, prompt: &str, options: CallOptions) -> GenerationStream<'a> {
        let _ = (prompt, options);
        todo!()
    }

    fn invoke(&self, prompt: &str, max_tokens: Option<usize>, thinking: Option<bool>) -> String {
        let _ = (prompt, max_tokens, thinking);
        todo!()
    }

    /// True when a reasoning flag was bound onto the model at load.
    fn supports_thinking(&self) -> bool {
        self.model.state().thinking().is_some()
    }

    /// The selected model name via `display_model_name`, or `""`.
    fn model_label(&self) -> String {
        todo!()
    }

    fn cancelled(&self) -> Option<CancelEvent> {
        self.cancelled.lock().clone()
    }

    fn set_cancelled(&self, event: Option<CancelEvent>) {
        *self.cancelled.lock() = event;
    }

    fn llm_type(&self) -> &str {
        "smartloop"
    }

    fn llama_model(&self) -> Option<Arc<dyn LlamaModel>> {
        Some(self.model.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
