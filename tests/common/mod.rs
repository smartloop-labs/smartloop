//! Port of `tests/conftest.py` (shared fixtures).

// ======================================================================
// BEGIN agents_mcp_agent fixtures
// ======================================================================

/// Shared fixtures for the MCP agent / tool caller tests (unit
/// `agents_mcp_agent`): a scriptable LLM standing in for Python's
/// `MagicMock()` handles.
pub mod agents_mcp_agent {
    use std::any::Any;
    use std::sync::Arc;

    use parking_lot::Mutex;
    use smartloop::llms::base_llm::{
        BaseLLM, CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream,
    };

    /// `(prompt, max_tokens, thinking) -> reply`.
    pub type ReplyFn = Arc<dyn Fn(&str, Option<usize>, Option<bool>) -> String + Send + Sync>;

    /// One recorded `invoke` call (Python's `call_args`).
    #[derive(Debug, Clone, PartialEq)]
    pub struct InvokeCall {
        pub prompt: String,
        pub max_tokens: Option<usize>,
        pub thinking: Option<bool>,
    }

    /// A scriptable LLM.
    ///
    /// `get_max_tokens` defaults to 0 — "no window reported" — which is what a
    /// `MagicMock` with non-numeric token arithmetic amounted to in Python: the
    /// budget helpers fall back to what they asked for.
    pub struct StubLLM {
        reply: Mutex<ReplyFn>,
        calls: Mutex<Vec<InvokeCall>>,
        supports_thinking: Mutex<bool>,
        model_label: Mutex<String>,
        max_tokens: Mutex<usize>,
        num_tokens: Mutex<usize>,
        cancelled: Mutex<Option<CancelEvent>>,
    }

    impl StubLLM {
        /// An LLM that always answers `reply`.
        pub fn new(reply: &str) -> Arc<Self> {
            let reply = reply.to_string();
            Self::with_reply_fn(Arc::new(move |_, _, _| reply.clone()))
        }

        /// An LLM whose answers come from `reply`.
        pub fn with_reply_fn(reply: ReplyFn) -> Arc<Self> {
            Arc::new(Self {
                reply: Mutex::new(reply),
                calls: Mutex::new(Vec::new()),
                supports_thinking: Mutex::new(false),
                model_label: Mutex::new(String::new()),
                max_tokens: Mutex::new(0),
                num_tokens: Mutex::new(0),
                cancelled: Mutex::new(None),
            })
        }

        /// A labelled LLM (Python's `llm.model.metadata["general.name"]`).
        pub fn labelled(label: &str) -> Arc<Self> {
            let stub = Self::new("");
            stub.set_model_label(label);
            stub
        }

        pub fn set_reply(&self, reply: &str) {
            let reply = reply.to_string();
            *self.reply.lock() = Arc::new(move |_, _, _| reply.clone());
        }

        pub fn set_reply_fn(&self, reply: ReplyFn) {
            *self.reply.lock() = reply;
        }

        pub fn set_supports_thinking(&self, value: bool) {
            *self.supports_thinking.lock() = value;
        }

        pub fn set_model_label(&self, label: &str) {
            *self.model_label.lock() = label.to_string();
        }

        pub fn set_max_tokens(&self, value: usize) {
            *self.max_tokens.lock() = value;
        }

        pub fn set_num_tokens(&self, value: usize) {
            *self.num_tokens.lock() = value;
        }

        /// Every `invoke` so far, oldest first.
        pub fn calls(&self) -> Vec<InvokeCall> {
            self.calls.lock().clone()
        }

        pub fn call_count(&self) -> usize {
            self.calls.lock().len()
        }

        /// The most recent `invoke` (Python's `call_args`); panics if none.
        pub fn last_call(&self) -> InvokeCall {
            self.calls.lock().last().cloned().expect("invoke was called")
        }

        /// This stub as the trait object the code under test takes.
        pub fn as_llm(self: &Arc<Self>) -> Arc<dyn BaseLLM> {
            self.clone()
        }
    }

    impl BaseLLM for StubLLM {
        fn project_id(&self) -> &str {
            "test-project"
        }

        fn get_num_tokens(&self, _text: &str) -> usize {
            *self.num_tokens.lock()
        }

        fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
            messages
        }

        fn get_max_tokens(&self) -> usize {
            *self.max_tokens.lock()
        }

        fn call(&self, prompt: &str, _stop: Option<&[String]>, options: CallOptions) -> anyhow::Result<String> {
            Ok(self.invoke(prompt, options.max_tokens, options.thinking))
        }

        fn stream<'a>(&'a self, prompt: &str, options: CallOptions) -> GenerationStream<'a> {
            let text = self.invoke(prompt, options.max_tokens, options.thinking);
            Box::new(std::iter::once(GenerationChunk { text }))
        }

        fn invoke(&self, prompt: &str, max_tokens: Option<usize>, thinking: Option<bool>) -> String {
            self.calls.lock().push(InvokeCall {
                prompt: prompt.to_string(),
                max_tokens,
                thinking,
            });
            let reply = self.reply.lock().clone();
            reply(prompt, max_tokens, thinking)
        }

        fn supports_thinking(&self) -> bool {
            *self.supports_thinking.lock()
        }

        fn model_label(&self) -> String {
            self.model_label.lock().clone()
        }

        fn cancelled(&self) -> Option<CancelEvent> {
            self.cancelled.lock().clone()
        }

        fn set_cancelled(&self, event: Option<CancelEvent>) {
            *self.cancelled.lock() = event;
        }

        fn llm_type(&self) -> &str {
            "StubLLM"
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Whether two handles are the same model (Python's `is`).
    pub fn same_llm(a: &Arc<dyn BaseLLM>, b: &Arc<dyn BaseLLM>) -> bool {
        std::ptr::eq(Arc::as_ptr(a) as *const (), Arc::as_ptr(b) as *const ())
    }
}

// ======================================================================
// END agents_mcp_agent fixtures
// ======================================================================
