//! Port of `tests/conftest.py` (shared fixtures).

// ===================================================================== agents_mcp_args
// Fixtures added by the `agents_mcp_args` unit.
pub mod agents_mcp_args {
    //! A scripted stand-in for `MagicMock()` LLMs: `get_num_tokens` returns a
    //! fixed count and `invoke` replays queued replies (then a default).

    use std::any::Any;
    use std::collections::VecDeque;

    use parking_lot::Mutex;
    use smartloop::llms::base_llm::{CallOptions, CancelEvent, ChatMessage, GenerationStream};
    use smartloop::llms::BaseLLM;

    /// `MagicMock` LLM: `get_num_tokens -> num_tokens`, `invoke -> next reply`.
    pub struct ScriptedLLM {
        replies: Mutex<VecDeque<String>>,
        default_reply: String,
        num_tokens: usize,
        /// Every prompt `invoke` received, in order.
        pub prompts: Mutex<Vec<String>>,
        cancelled: Mutex<Option<CancelEvent>>,
    }

    impl ScriptedLLM {
        /// An LLM whose every `invoke` returns `reply`.
        pub fn new(reply: impl Into<String>) -> Self {
            Self {
                replies: Mutex::new(VecDeque::new()),
                default_reply: reply.into(),
                num_tokens: 10,
                prompts: Mutex::new(Vec::new()),
                cancelled: Mutex::new(None),
            }
        }

        /// An LLM that replays `replies` in order, then returns `""`.
        pub fn with_replies<I, S>(replies: I) -> Self
        where
            I: IntoIterator<Item = S>,
            S: Into<String>,
        {
            let llm = Self::new("");
            *llm.replies.lock() = replies.into_iter().map(Into::into).collect();
            llm
        }

        /// How many times `invoke` was called.
        pub fn invoke_count(&self) -> usize {
            self.prompts.lock().len()
        }
    }

    impl BaseLLM for ScriptedLLM {
        fn project_id(&self) -> &str {
            "test"
        }
        fn get_num_tokens(&self, _text: &str) -> usize {
            self.num_tokens
        }
        fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
            messages
        }
        fn get_max_tokens(&self) -> usize {
            8192
        }
        fn call(&self, prompt: &str, _stop: Option<&[String]>, _options: CallOptions) -> anyhow::Result<String> {
            Ok(self.invoke(prompt, None, None))
        }
        fn stream<'a>(&'a self, prompt: &str, _options: CallOptions) -> GenerationStream<'a> {
            let text = self.invoke(prompt, None, None);
            Box::new(std::iter::once(smartloop::llms::base_llm::GenerationChunk { text }))
        }
        fn invoke(&self, prompt: &str, _max_tokens: Option<usize>, _thinking: Option<bool>) -> String {
            self.prompts.lock().push(prompt.to_string());
            self.replies
                .lock()
                .pop_front()
                .unwrap_or_else(|| self.default_reply.clone())
        }
        fn cancelled(&self) -> Option<CancelEvent> {
            self.cancelled.lock().clone()
        }
        fn set_cancelled(&self, event: Option<CancelEvent>) {
            *self.cancelled.lock() = event;
        }
        fn llm_type(&self) -> &str {
            "scripted"
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
}
// ================================================================= end agents_mcp_args
