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

use parking_lot::Mutex;

use crate::llms::base_llm::{BaseLLM, CallOptions, CancelEvent, ChatMessage, GenerationStream};
use crate::model_host::client::ModelClient;

/// Logger name used by the Python module.
#[allow(dead_code)]
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
        let _ = text;
        todo!()
    }

    /// Left to the host, which owns the model that defines the template.
    fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        messages
    }

    /// The host's response context window for this capability, or
    /// [`DEFAULT_MAX_CONTEXT_TOKENS`] when it cannot say.
    fn get_max_tokens(&self) -> usize {
        todo!()
    }

    fn call(&self, prompt: &str, stop: Option<&[String]>, options: CallOptions) -> anyhow::Result<String> {
        let _ = (prompt, stop, options);
        todo!()
    }

    /// Streams from the host; the cancel event set by the caller cancels the
    /// gRPC stream, which the host observes between chunks.
    fn stream<'a>(&'a self, prompt: &str, options: CallOptions) -> GenerationStream<'a> {
        let _ = (prompt, options);
        todo!()
    }

    /// Generate once, matching the local LLM's forgiving contract: errors are
    /// returned as text rather than raised.
    fn invoke(&self, prompt: &str, max_tokens: Option<usize>, thinking: Option<bool>) -> String {
        let _ = (prompt, max_tokens, thinking);
        todo!()
    }

    /// Asked of the host, cached per handle; `false` on failure.
    fn supports_thinking(&self) -> bool {
        let _ = &self.supports_thinking;
        todo!()
    }

    /// What the host calls the model behind this handle; resolved per call,
    /// `""` on failure.
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
        "smartloop-remote"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
