//! Llama.cpp wrapper.
//!
//! Port of `smartloop/llms/llama_cpp.py`.  The `llama_cpp.Llama` surface it
//! wraps lives in [`crate::llms::llama`] and is re-exported here.

use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, OnceLock};

use once_cell::sync::Lazy;
use parking_lot::{Mutex, ReentrantMutex, ReentrantMutexGuard};
use regex::Regex;
use serde_json::{Map, Value};

use crate::constants::OPENING_TOKENS;
use crate::llms::base_llm::{
    BaseLLM, BatchPrompt, CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream, clean_reply,
};
use crate::llms::llama::batch::capture_prompt;
use crate::llms::thinking::{ThinkingFilter, strip_thinking};
use crate::llms::tool_calls::{split_messages, split_tools, template_renders_tools};
use crate::model::spec::MODEL_SPECS;
use crate::types::ModelCapability;

pub use crate::llms::llama::{
    ChatCompletionChunks, ChatCompletionOutput, ChatCompletionRequest, ChatHandler, ChatHandlerRequest, Llama,
    LlamaModel, LlamaParams, LlamaState, LlamaToken,
};
use crate::llms::llama::decode_utf8_ignore;

/// Logger name of the wrapper (the Python code logged through the project
/// logger it was handed).
const LOG_TARGET: &str = "smartloop.llms.llama_cpp";

/// DRY strength for every generation — the commonly recommended 0.8.  The
/// 64-token `repeat_penalty` cannot see a loop with a longer cycle; DRY can.
const DRY_MULTIPLIER: f32 = 0.8;

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

/// Seconds since the epoch (`time.time()`).
fn time_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Runs of tabs removed from a finished reply.
static TABS: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\t]+").expect("valid regex"));

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
    /// [`BaseLLM::supports_native_tools`], worked out once.
    native_tools: OnceLock<bool>,
}

/// Chat handlers that render the model's own embedded template, so a
/// `tools` variable reaches it (the others impose a fixed format).
const EMBEDDED_TEMPLATE_HANDLERS: [&str; 2] = ["MTMDChatHandler", "Gemma4ChatHandler"];

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
            native_tools: OnceLock::new(),
        }
    }

    /// The lock serializing generation on this llama.cpp context.
    ///
    /// A llama.cpp context is NOT thread-safe — one KV cache, one token
    /// buffer, one sampler — so two concurrent `create_chat_completion` calls
    /// corrupt the heap and segfault the process.  Callers reach the model
    /// from many threads (the DAG's node pool, the collect fan-out in
    /// `Explorer::collect_parallel`, MCP tool selection and argument
    /// extraction) whose own locks guard different subsets of those callers,
    /// so serializing here is what makes the model safe whoever calls it.
    ///
    /// Held only around generation; tokenization runs on the vocab and needs
    /// no guard.  The lock lives on the model, so two wrappers over one
    /// context share one lock; it is reentrant because generation tokenizes
    /// (`fit_prompt`, `get_num_tokens`) while holding it.
    pub fn context_lock(&self) -> &ReentrantMutex<()> {
        self.model.state().context_lock()
    }

    /// The model's trained context length from GGUF metadata
    /// (`{arch}.context_length`), or 0.
    pub fn get_n_ctx_train(&self) -> usize {
        for (key, value) in self.model.metadata() {
            if key.ends_with(".context_length") {
                return value.trim().parse::<usize>().unwrap_or(0);
            }
        }
        0
    }

    /// Token count of `text`, failing when the tokenizer does.
    fn num_tokens(&self, text: &str) -> anyhow::Result<usize> {
        Ok(self.model.tokenize(text.as_bytes())?.len())
    }

    /// Output tokens that fit after the prompt, never exceeding n_ctx (floored at 1).
    ///
    /// `prompt + output` must stay within `n_ctx` or llama.cpp rejects the
    /// request.  Returns the true remaining budget — it must NOT force a fixed
    /// minimum, which is what previously drove `prompt + 1024 > n_ctx`
    /// overflows.
    pub fn get_available_tokens(&self, prompt: &str) -> usize {
        let n_ctx = self.get_max_tokens() as i64;
        let prompt_tokens = self.get_num_tokens(prompt) as i64;
        let available = n_ctx - prompt_tokens - Self::CTX_OVERHEAD as i64;
        available.max(1) as usize
    }

    /// Truncate an over-long prompt so it fits n_ctx with room for output,
    /// keeping head (2/3) and tail with a truncation marker in between.
    ///
    /// A prompt that alone exceeds `n_ctx` makes llama.cpp reject the request
    /// ("Requested tokens exceed context window") — no `max_tokens` value can
    /// save it.  The head (system/instructions) and tail (the user question,
    /// usually last) are kept and the middle bulk (retrieved context / web
    /// results) dropped, with a marker so the model knows content was elided.
    pub fn fit_prompt(&self, prompt: &str) -> String {
        match self.try_fit_prompt(prompt) {
            Ok(fitted) => fitted,
            Err(e) => {
                tracing::warn!(target: LOG_TARGET, "[llm] prompt-fit guard failed: {e}");
                prompt.to_string()
            }
        }
    }

    fn try_fit_prompt(&self, prompt: &str) -> anyhow::Result<String> {
        let n_ctx = self.get_max_tokens() as i64;
        let budget = n_ctx - Self::MIN_OUTPUT_RESERVE as i64 - Self::CTX_OVERHEAD as i64;
        if budget <= 0 {
            return Ok(prompt.to_string());
        }
        let budget = budget as usize;
        let tokens = self.model.tokenize(prompt.as_bytes())?;
        if tokens.len() <= budget {
            return Ok(prompt.to_string());
        }

        let marker = self.model.tokenize(b"\n\n...[context truncated to fit]...\n\n")?;
        let keep = budget as i64 - marker.len() as i64;
        let kept: Vec<LlamaToken> = if keep <= 0 {
            tokens[..budget].to_vec()
        } else {
            let keep = keep as usize;
            let head = keep * 2 / 3;
            let tail = keep - head;
            let mut kept = tokens[..head].to_vec();
            kept.extend_from_slice(&marker);
            if tail > 0 {
                kept.extend_from_slice(&tokens[tokens.len() - tail..]);
            }
            kept
        };
        let truncated = decode_utf8_ignore(&self.model.detokenize(&kept)?);
        tracing::warn!(
            target: LOG_TARGET,
            "[llm] prompt {} tok exceeds budget {} (n_ctx={}) — truncated context to fit",
            tokens.len(),
            budget,
            n_ctx
        );
        Ok(truncated)
    }

    /// `` | 512 prompt, ttft 1.8s, 12.4 tok/s`` for a generation; times are
    /// epoch seconds (`time.time()`).  Empty when nothing was generated.
    ///
    /// "Slow" is not actionable and neither is a wall-clock total: prefill
    /// (time to the first token, which the prompt pays for) and decode rate
    /// (what the weights and the offload decide) separate a long prompt from
    /// a CPU fallback.  When a generation is long enough to have two halves,
    /// `opening_at` — the moment the first `OPENING_TOKENS` were out — splits
    /// the decode rate into the opening the reader waits on and the rest.
    pub fn rate(
        prompt_tokens: usize,
        generated: usize,
        start_time: f64,
        first_token_at: Option<f64>,
        opening_at: Option<f64>,
    ) -> String {
        let Some(first_token_at) = first_token_at else {
            return String::new();
        };
        if generated == 0 {
            return String::new();
        }
        let ttft = first_token_at - start_time;
        let now = time_now();
        let decode = (now - first_token_at).max(1e-6);
        let mut rate = format!("{:.1} tok/s", generated as f64 / decode);
        if let Some(opening_at) = opening_at {
            if generated > OPENING_TOKENS {
                let opening = (opening_at - first_token_at).max(1e-6);
                let rest = (now - opening_at).max(1e-6);
                rate = format!(
                    "{rate} (first {OPENING_TOKENS}: {:.1}, then {:.1})",
                    OPENING_TOKENS as f64 / opening,
                    (generated - OPENING_TOKENS) as f64 / rest
                );
            }
        }
        format!(" | {prompt_tokens} prompt, ttft {ttft:.2}s, {rate}")
    }

    /// Whether this call asked to reason.  Unspecified means no.
    ///
    /// Reasoning is opt-in per call, not a mode the model sits in: a
    /// `<think>` block is worth its tokens where it changes the outcome
    /// (resolving an MCP tool call) and pure latency everywhere else.  The
    /// only caller that opts in is `MCPAgent::resolve`.
    pub fn requested_thinking(options: &CallOptions) -> bool {
        options.thinking.unwrap_or(false)
    }

    /// Force the reasoning block on or off for one call (no-op when
    /// `thinking` is `None` or the model has no bound flag).
    ///
    /// Safe on the shared model: generation on it is serialized by the
    /// context lock this runs inside.
    pub fn thinking_override(&self, thinking: Option<bool>) -> ThinkingGuard<'_> {
        let model: &dyn LlamaModel = &*self.model;
        let state = model.state();
        let (Some(thinking), Some(previous)) = (thinking, state.thinking()) else {
            return ThinkingGuard { model, previous: None };
        };
        state.set_thinking(Some(thinking));
        ThinkingGuard { model, previous: Some(Some(previous)) }
    }

    /// Suppress the reasoning block for a call whose effective output budget
    /// is below [`THINKING_MIN_BUDGET`].  Only applies when the bound flag is on.
    ///
    /// `max_tokens` is the call's *effective* budget, so this covers both a
    /// decision call that caps its own output and a long prompt that leaves
    /// little context headroom behind it.
    pub fn thinking_for_budget(&self, max_tokens: Option<usize>) -> ThinkingGuard<'_> {
        let model: &dyn LlamaModel = &*self.model;
        let state = model.state();
        let previous = state.thinking();
        let suppress = previous == Some(true) && max_tokens.is_some_and(|m| m < THINKING_MIN_BUDGET);
        if !suppress {
            return ThinkingGuard { model, previous: None };
        }
        state.set_thinking(Some(false));
        ThinkingGuard { model, previous: Some(previous) }
    }

    /// Extract the text of one OpenAI-shaped chunk (`delta.content`,
    /// `message.content` or `text`).
    pub fn chunk_text(chunk: &Value) -> String {
        let Some(choice) = chunk.get("choices").and_then(Value::as_array).and_then(|c| c.first()) else {
            return String::new();
        };
        let text = if let Some(c) = choice.get("delta").and_then(|d| d.get("content")) {
            c
        } else if let Some(c) = choice.get("message").and_then(|m| m.get("content")) {
            c
        } else if let Some(c) = choice.get("text") {
            c
        } else {
            return String::new();
        };
        match text {
            Value::String(s) => s.clone(),
            Value::Null => String::new(),
            other => other.to_string(),
        }
    }

    /// Text of one chunk as `_stream` reads it: a non-object chunk is used
    /// as-is (`str(chunk)`).
    fn stream_chunk_text(chunk: &Value) -> String {
        match chunk {
            Value::Object(_) => Self::chunk_text(chunk),
            Value::String(s) => s.clone(),
            other => other.to_string(),
        }
    }

    /// Return a dictionary of identifying parameters.
    pub fn identifying_params(&self) -> Map<String, Value> {
        // The model name allows users to specify custom token counting rules
        // in LLM monitoring applications.  LangChain's `name` is unset.
        let mut params = Map::new();
        params.insert("model_name".into(), Value::Null);
        params
    }

    /// GGUF metadata of the wrapped model.
    pub fn metadata(&self) -> HashMap<String, String> {
        self.model.metadata()
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.lock().as_ref().is_some_and(|e| e.load(Ordering::SeqCst))
    }

    /// The body of `_call` once the context lock is held.
    fn call_locked(&self, prompt: &str, options: CallOptions) -> anyhow::Result<String> {
        let start_time = time_now();
        // A tools block is lifted out before fitting, so truncation can never
        // cut a tool definition in half; it is rendered by the chat template.
        let (prompt, tools) = split_tools(prompt);
        let tools = tools.filter(|_| self.supports_native_tools());
        // A tool conversation is rendered by the template as it is; it is
        // never truncated, since a cut would break its JSON.
        let conversation = split_messages(prompt).filter(|_| tools.is_some());
        let prompt = if conversation.is_some() { prompt.to_string() } else { self.fit_prompt(prompt) };
        let prompt_tokens = self.num_tokens(&prompt)?;
        tracing::info!(
            target: LOG_TARGET,
            "Generating response | prompt_tokens={prompt_tokens}{}",
            tools.as_ref().map(|t| format!(" tools={}", t.len())).unwrap_or_default()
        );

        let result = self.generate(&prompt, prompt_tokens, start_time, options, tools, conversation);
        let text = match result {
            Ok(text) => text,
            Err(ex) => {
                tracing::error!(target: LOG_TARGET, "Generation failed: {ex}");
                String::new()
            }
        };
        tracing::info!(target: LOG_TARGET, "Total elapsed={:.3}s", time_now() - start_time);
        Ok(text)
    }

    /// The completion request every generation sends: `messages` with room
    /// for `max_tokens` of output.
    fn completion_request(&self, messages: &[ChatMessage], max_tokens: usize, stream: bool) -> ChatCompletionRequest {
        ChatCompletionRequest {
            messages: messages_to_values(messages),
            max_tokens: Some(max_tokens),
            temperature: self.temperature,
            repeat_penalty: 1.2,
            frequency_penalty: 0.1,
            presence_penalty: 0.1,
            dry_multiplier: DRY_MULTIPLIER,
            stream,
            ..ChatCompletionRequest::default()
        }
    }

    /// Model output as every caller takes it: the `<think>` block and tabs
    /// removed, trimmed.
    fn finish_text(text: &str) -> String {
        let text = strip_thinking(text);
        TABS.replace_all(text.trim(), "").into_owned()
    }

    fn generate(
        &self,
        prompt: &str,
        prompt_tokens: usize,
        start_time: f64,
        options: CallOptions,
        tools: Option<Vec<Value>>,
        conversation: Option<Vec<Value>>,
    ) -> anyhow::Result<String> {
        let messages = self.create_messages_from_prompt(prompt);
        let available_tokens = match options.max_tokens {
            Some(m) => m,
            None => self.get_available_tokens(prompt),
        };
        // Stream internally so generation can be interrupted between tokens
        // when the request is cancelled.  The thinking guards only have to
        // span the call itself: the chat handler renders the template (where
        // enable_thinking is read) eagerly and returns a token generator.
        let stream = {
            let _override = self.thinking_override(Some(Self::requested_thinking(&options)));
            let _budget = self.thinking_for_budget(Some(available_tokens));
            let mut request = self.completion_request(&messages, available_tokens, true);
            request.tools = tools;
            if let Some(conversation) = conversation {
                request.messages = conversation;
            }
            self.model.create_chat_completion(request)?
        };

        let mut parts: Vec<String> = Vec::new();
        let mut chunk_count = 0usize;
        let mut cancelled = false;
        let mut first_token_at: Option<f64> = None;
        let mut opening_at: Option<f64> = None;
        for chunk in stream.into_chunks() {
            // Check for cancellation between chunks.
            if self.is_cancelled() {
                cancelled = true;
                break;
            }
            let text = Self::chunk_text(&chunk);
            if !text.is_empty() {
                if first_token_at.is_none() {
                    first_token_at = Some(time_now());
                }
                parts.push(text);
                chunk_count += 1;
                if chunk_count == OPENING_TOKENS {
                    opening_at = Some(time_now());
                }
            }
        }

        let elapsed = time_now() - start_time;
        tracing::info!(
            target: LOG_TARGET,
            "{} | chunks={} elapsed={:.3}s{}",
            if cancelled { "Cancelled" } else { "Completed" },
            chunk_count,
            elapsed,
            Self::rate(prompt_tokens, chunk_count, start_time, first_token_at, opening_at)
        );
        // A reasoning model's <think> block is not the answer: every caller
        // from here on treats this string as one (JSON planning, skill
        // matching, the chat reply), so drop it.  A no-op for models that
        // don't think.
        Ok(Self::finish_text(&parts.concat()))
    }
}

/// Messages in the OpenAI dict shape `create_chat_completion` takes.
fn messages_to_values(messages: &[ChatMessage]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| {
            let mut obj = Map::new();
            obj.insert("role".into(), Value::String(m.role.clone()));
            obj.insert("content".into(), Value::String(m.content.clone()));
            Value::Object(obj)
        })
        .collect()
}

impl Drop for LlamaCpp {
    fn drop(&mut self) {
        // Model resources are released when the last `Arc` to the context
        // drops (`Llama::drop`); nothing to do per wrapper.
    }
}

/// `_stream`: generated chunks with reasoning withheld, holding the context
/// lock until the stream is drained or dropped.
struct LlamaCppStream<'a> {
    llm: &'a LlamaCpp,
    prompt: String,
    chunks: Option<ChatCompletionChunks<'a>>,
    thinking: ThinkingFilter,
    chunk_count: usize,
    start_time: f64,
    first_token_at: Option<f64>,
    opening_at: Option<f64>,
    finished: bool,
    // Declared last so it is released after the chunk iterator.
    _guard: ReentrantMutexGuard<'a, ()>,
}

impl LlamaCppStream<'_> {
    /// Release what the filter held back and log the run.
    fn finish(&mut self) -> Option<GenerationChunk> {
        self.finished = true;
        self.chunks = None;
        // Release whatever the filter held back — a trailing partial tag that
        // turned out to be ordinary text, or (when generation was cut off
        // mid-thought) the reasoning itself, which beats a blank reply.
        if self.thinking.unterminated() {
            tracing::warn!(
                target: LOG_TARGET,
                "Generation ended inside a <think> block — the reply was never reached (raise max_tokens or disable thinking)"
            );
        }
        let tail = self.thinking.flush();

        if self.chunk_count == 0 {
            tracing::warn!(target: LOG_TARGET, "No chunks received from stream");
        } else {
            tracing::info!(
                target: LOG_TARGET,
                "Streamed | chunks={} elapsed={:.3}s{}",
                self.chunk_count,
                time_now() - self.start_time,
                LlamaCpp::rate(
                    self.llm.get_num_tokens(&self.prompt),
                    self.chunk_count,
                    self.start_time,
                    self.first_token_at,
                    self.opening_at,
                )
            );
        }
        if tail.is_empty() { None } else { Some(GenerationChunk { text: tail }) }
    }
}

impl Iterator for LlamaCppStream<'_> {
    type Item = GenerationChunk;

    fn next(&mut self) -> Option<GenerationChunk> {
        if self.finished {
            return None;
        }
        loop {
            let chunk = match self.chunks.as_mut() {
                Some(chunks) => chunks.next(),
                None => None,
            };
            let Some(chunk) = chunk else {
                return self.finish();
            };
            // Check for cancellation between chunks.
            if self.llm.is_cancelled() {
                return self.finish();
            }

            self.chunk_count += 1;
            if self.first_token_at.is_none() {
                self.first_token_at = Some(time_now());
            } else if self.chunk_count == OPENING_TOKENS {
                self.opening_at = Some(time_now());
            }

            let text = LlamaCpp::stream_chunk_text(&chunk);
            // Only yield non-empty text chunks.
            let visible = if text.is_empty() { String::new() } else { self.thinking.feed(&text) };
            if !visible.is_empty() {
                return Some(GenerationChunk { text: visible });
            }
        }
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
        match self.num_tokens(text) {
            Ok(n) => n,
            Err(e) => {
                tracing::warn!(target: LOG_TARGET, "tokenization failed: {e}");
                0
            }
        }
    }

    /// llama.cpp applies the chat template inside `create_chat_completion`,
    /// so messages are returned unchanged.
    fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        messages
    }

    /// Effective context size: the smaller of the runtime `n_ctx` and the
    /// model's trained context length (architectural max).
    fn get_max_tokens(&self) -> usize {
        let n_ctx = self.model.n_ctx();
        let n_ctx_train = self.get_n_ctx_train();
        if n_ctx_train > 0 {
            return n_ctx.min(n_ctx_train);
        }
        n_ctx
    }

    fn call(&self, prompt: &str, _stop: Option<&[String]>, options: CallOptions) -> anyhow::Result<String> {
        let _guard = self.context_lock().lock();
        self.call_locked(prompt, options)
    }

    /// Every prompt is rendered exactly as [`BaseLLM::invoke`] would render it
    /// (fitted to the window, same template, sampling and thinking flags),
    /// then generated together; see [`Llama::generate_batch`].
    fn invoke_batch(&self, prompts: &[BatchPrompt], cancelled: Option<CancelEvent>) -> Vec<String> {
        let request_cancelled = || cancelled.as_ref().is_some_and(|c| c.load(Ordering::SeqCst));
        let Some(llama) = self.model.as_llama() else {
            return prompts
                .iter()
                .map(|p| if request_cancelled() { String::new() } else { self.invoke(&p.prompt, p.max_tokens, p.thinking) })
                .collect();
        };
        let _guard = self.context_lock().lock();
        let mut items = Vec::with_capacity(prompts.len());
        for p in prompts {
            let prompt = self.fit_prompt(&p.prompt);
            let messages = self.create_messages_from_prompt(&prompt);
            let max_tokens = p.max_tokens.unwrap_or_else(|| self.get_available_tokens(&prompt));
            let options = CallOptions { max_tokens: p.max_tokens, thinking: p.thinking };
            let (_, item) = capture_prompt(|| {
                let _override = self.thinking_override(Some(Self::requested_thinking(&options)));
                let _budget = self.thinking_for_budget(Some(max_tokens));
                self.model.create_chat_completion(self.completion_request(&messages, max_tokens, false))
            });
            match item {
                Some(item) => items.push(item),
                // A handler that never reached a text completion (multimodal)
                // cannot be batched.
                None => {
                    drop(_guard);
                    return prompts
                        .iter()
                        .map(|p| if request_cancelled() { String::new() } else { self.invoke(&p.prompt, p.max_tokens, p.thinking) })
                        .collect();
                }
            }
        }
        llama
            .generate_batch(items, &|| request_cancelled() || self.is_cancelled())
            .into_iter()
            .map(|result| match result {
                Ok(text) => clean_reply(&Self::finish_text(&text)),
                Err(e) => {
                    tracing::error!(target: LOG_TARGET, "Generation failed: {e}");
                    format!("Error generating response: {e}")
                }
            })
            .collect()
    }

    fn stream<'a>(&'a self, prompt: &str, options: CallOptions) -> GenerationStream<'a> {
        // A tools prompt is answered in one piece through `call`, which hands
        // the tools and the conversation to the chat template; the reply,
        // calls and all, comes back as one chunk.
        if split_tools(prompt).1.is_some() {
            let text = match self.call(prompt, None, options) {
                Ok(text) => text,
                Err(e) => std::panic::panic_any(e.to_string()),
            };
            return Box::new(std::iter::once(GenerationChunk { text }));
        }
        let guard = self.context_lock().lock();

        // create_chat_completion with stream=true applies the model's chat
        // template.
        let prompt = self.fit_prompt(prompt);
        let messages = self.create_messages_from_prompt(&prompt);
        let available_tokens = self.get_available_tokens(&prompt);

        let chunks = {
            let _override = self.thinking_override(Some(Self::requested_thinking(&options)));
            self.model.create_chat_completion(ChatCompletionRequest {
                messages: messages_to_values(&messages),
                max_tokens: Some(available_tokens),
                temperature: self.temperature,
                repeat_penalty: 1.1,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                dry_multiplier: DRY_MULTIPLIER,
                stream: true,
                ..ChatCompletionRequest::default()
            })
        };
        let chunks = match chunks {
            Ok(output) => Some(output.into_chunks()),
            Err(e) => {
                tracing::error!(target: LOG_TARGET, "Generation failed: {e}");
                None
            }
        };

        Box::new(LlamaCppStream {
            llm: self,
            prompt,
            chunks,
            // Reasoning is withheld from the stream rather than shown and then
            // retracted; the tags can split across chunks, so the filter is
            // stateful and spans the whole stream.
            thinking: ThinkingFilter::new(),
            chunk_count: 0,
            start_time: time_now(),
            first_token_at: None,
            opening_at: None,
            finished: false,
            _guard: guard,
        })
    }

    /// True when a reasoning flag was bound onto the model at load.
    ///
    /// The flag is bound only for a model whose template reads
    /// `enable_thinking` (see `LlmFactory::bind_thinking`), so its presence is
    /// the answer.
    fn supports_thinking(&self) -> bool {
        self.model.state().thinking().is_some()
    }

    /// True when the catalog declares `tools` for this model *and* the chat
    /// template it renders reads a `tools` variable.
    ///
    /// Both, because neither alone is enough: the declaration is the
    /// catalog's judgement that the model is big enough to be trusted with a
    /// whole tool catalog (a 2B template renders tools too), and the template
    /// check guards a model loaded through a fixed-format vision handler or a
    /// GGUF whose template was stripped, where the tools would never reach
    /// the model.  Cached: neither changes while the model is loaded.
    fn supports_native_tools(&self) -> bool {
        *self.native_tools.get_or_init(|| {
            let Some(spec) = self.slp_model_name.as_deref().and_then(|name| MODEL_SPECS.get(name)) else {
                return false;
            };
            if !spec.supports(ModelCapability::Tools) {
                return false;
            }
            if spec.vision_chat_handler.as_deref().is_some_and(|h| !EMBEDDED_TEMPLATE_HANDLERS.contains(&h)) {
                tracing::info!(
                    target: LOG_TARGET,
                    "[tools] {} declares tools but renders a fixed vision template — using prompt-based tool resolution",
                    spec.name
                );
                return false;
            }
            let renders = self
                .model
                .metadata()
                .get("tokenizer.chat_template")
                .is_some_and(|t| template_renders_tools(t));
            if !renders {
                tracing::info!(
                    target: LOG_TARGET,
                    "[tools] {} declares tools but its chat template does not render them — using prompt-based tool resolution",
                    spec.name
                );
            }
            renders
        })
    }

    /// The model's name as the user chose it, or `""`.
    ///
    /// A GGUF's `general.name` is the upstream author's; the selected name is
    /// the one in the picker, so it leads.  Empty means the caller falls back
    /// to the metadata.
    fn model_label(&self) -> String {
        match self.slp_model_name.as_deref() {
            None | Some("") => String::new(),
            Some(name) => crate::constants::display_model_name(name),
        }
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
