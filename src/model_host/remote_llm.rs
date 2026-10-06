//! Port of `smartloop/model_host/remote_llm.py`.
//!
//! A [`BaseLLM`] whose weights live in the model host, not this process.
//!
//! The orchestrator does far more inference than the final answer —
//! classification, skill matching, planning, every DAG decision node — and it
//! takes `BaseLLM` objects directly rather than going through the service.  It
//! does not fail loudly without one, so a child with no weights would quietly
//! lose skills, tools and planning while still answering.  A child therefore
//! hands the orchestrator one of these instead: it satisfies the same
//! interface the local llama.cpp LLM does, and every call crosses to the host.
//!
//! Capability selection stays on the host side: the `capability` this
//! instance carries is sent with each request, and the host picks the model
//! the same way it would in-process.

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;

use crate::llms::base_llm::{
    BaseLLM, CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream,
};
use crate::model_host::client::ModelClient;

/// Logger name used by the Python module.
const LOG_TARGET: &str = "smartloop.model_host.remote_llm";

/// Used when the host cannot say how big its context is.  Matches the local
/// LLM's own fallback, so prompt sizing degrades the same way in both.
pub const DEFAULT_MAX_CONTEXT_TOKENS: usize = 8192;

/// The host's model, addressed as if it were loaded here.
///
/// Deliberately thin: no prompt fitting, no context lock, no chat template.
/// All three belong to whoever owns the weights.
pub struct RemoteLLM {
    client: Arc<dyn ModelClient>,
    project_id: String,
    capability: Option<String>,
    cancelled: Mutex<Option<CancelEvent>>,
    supports_thinking: Mutex<Option<bool>>,
}

impl RemoteLLM {
    /// A handle to the host's model for `capability`, on behalf of
    /// `project_id`.
    pub fn new(
        client: Arc<dyn ModelClient>,
        project_id: impl Into<String>,
        capability: Option<String>,
    ) -> Self {
        Self {
            client,
            project_id: project_id.into(),
            capability,
            cancelled: Mutex::new(None),
            supports_thinking: Mutex::new(None),
        }
    }

    /// The model client this LLM is a handle to.
    pub fn client(&self) -> &Arc<dyn ModelClient> {
        &self.client
    }

    /// The capability sent with every request.
    pub fn capability(&self) -> Option<&str> {
        self.capability.as_deref()
    }
}

impl std::fmt::Debug for RemoteLLM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RemoteLLM(capability={:?})", self.capability)
    }
}

impl BaseLLM for RemoteLLM {
    fn project_id(&self) -> &str {
        &self.project_id
    }

    /// Count with the host's tokenizer — the only one that matches; 0 on
    /// failure.
    fn get_num_tokens(&self, text: &str) -> usize {
        match self.client.count_tokens(text) {
            Ok(count) => count,
            Err(exc) => {
                tracing::warn!(target: LOG_TARGET, "Remote token count failed: {exc}");
                0
            }
        }
    }

    /// Left to the host, which owns the model that defines the template.
    fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        messages
    }

    /// The host's response context window for this capability, or
    /// [`DEFAULT_MAX_CONTEXT_TOKENS`] when it cannot say.
    fn get_max_tokens(&self) -> usize {
        match self.client.response_max_context_tokens(self.capability.as_deref()) {
            Ok(0) => DEFAULT_MAX_CONTEXT_TOKENS,
            Ok(tokens) => tokens,
            Err(exc) => {
                tracing::warn!(target: LOG_TARGET, "Remote context size failed: {exc}");
                DEFAULT_MAX_CONTEXT_TOKENS
            }
        }
    }

    fn call(&self, prompt: &str, stop: Option<&[String]>, options: CallOptions) -> anyhow::Result<String> {
        let _ = stop;
        self.client.generate(
            prompt,
            None,
            self.capability.as_deref(),
            options.max_tokens,
            options.thinking,
        )
    }

    /// Streams from the host; the cancel event set by the caller cancels the
    /// gRPC stream, which the host observes between chunks.
    fn stream<'a>(&'a self, prompt: &str, options: CallOptions) -> GenerationStream<'a> {
        // `cancelled` is set by the caller between calls, exactly as it is on
        // the local LLM; here it cancels the gRPC stream the host is serving
        // from, which the host observes between chunks.
        let cancelled = self.cancelled();
        let texts = match self.client.stream(
            prompt,
            None,
            self.capability.as_deref(),
            cancelled.clone(),
            options.thinking,
        ) {
            Ok(texts) => texts,
            Err(exc) => {
                tracing::error!(target: LOG_TARGET, "Remote LLM stream failed: {exc}");
                return Box::new(std::iter::empty());
            }
        };
        let mut texts = texts;
        let mut done = false;
        Box::new(std::iter::from_fn(move || {
            if done {
                return None;
            }
            match texts.next() {
                Some(Ok(text)) => {
                    if cancelled.as_ref().is_some_and(|event| event.load(Ordering::SeqCst)) {
                        done = true;
                        return None;
                    }
                    Some(GenerationChunk { text })
                }
                Some(Err(exc)) => {
                    tracing::error!(target: LOG_TARGET, "Remote LLM stream failed: {exc}");
                    done = true;
                    None
                }
                None => {
                    done = true;
                    None
                }
            }
        }))
    }

    /// Generate once, matching the local LLM's forgiving contract: errors are
    /// returned as text rather than raised.
    ///
    /// The orchestrator calls this from DAG nodes that treat a bad answer as a
    /// dead branch, and an error there would take down a whole plan over one
    /// step.  `max_tokens` and `thinking` travel to the host with the request:
    /// they belong to this call, not to the loaded model.
    fn invoke(&self, prompt: &str, max_tokens: Option<usize>, thinking: Option<bool>) -> String {
        match self.call(prompt, None, CallOptions { max_tokens, thinking }) {
            Ok(response) => {
                let mut response = response.trim().to_string();
                if let Some(stripped) = response.strip_suffix("<|eot_id|>") {
                    response = stripped.trim().to_string();
                }
                if response.is_empty() {
                    "No response generated".to_string()
                } else {
                    response
                }
            }
            Err(exc) => {
                tracing::error!(target: LOG_TARGET, "Remote LLM invocation failed: {exc}");
                format!("Error generating response: {exc}")
            }
        }
    }

    /// Asked of the host, cached per handle; `false` on failure.
    ///
    /// Cached per handle: it cannot change without the host swapping models,
    /// and this is read on the resolution path of every MCP step.
    fn supports_thinking(&self) -> bool {
        let mut cached = self.supports_thinking.lock();
        if let Some(value) = *cached {
            return value;
        }
        let value = self
            .client
            .supports_thinking(self.capability.as_deref())
            .unwrap_or(false);
        *cached = Some(value);
        value
    }

    /// What the host calls the model behind this handle; resolved per call,
    /// `""` on failure.
    ///
    /// Read by both `model_label` helpers, which otherwise fall through to the
    /// type name — and this label goes out on the progress SSE.  Not cached:
    /// the host is free to load a different model for a capability between
    /// requests, and a stale name here would be worse than the round trip.
    fn model_label(&self) -> String {
        match self.client.model_label(self.capability.as_deref()) {
            Ok(label) => label,
            Err(exc) => {
                tracing::warn!(target: LOG_TARGET, "Remote model label failed: {exc}");
                String::new()
            }
        }
    }

    fn cancelled(&self) -> Option<CancelEvent> {
        self.cancelled.lock().clone()
    }

    fn set_cancelled(&self, event: Option<CancelEvent>) {
        *self.cancelled.lock() = event;
    }

    fn llm_type(&self) -> &str {
        "smartloop-remote"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
