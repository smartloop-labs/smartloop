//! Port of `tests/conftest.py` (shared fixtures).

// ===================== BEGIN orch_exec =====================
/// Fixtures shared by the orch_exec unit's tests
/// (`test_mcp_dead_grant`, `test_mcp_resolution_fills_arguments`).
pub mod orch_exec {
    use std::any::Any;
    use std::sync::Arc;

    use parking_lot::Mutex;

    use smartloop::llms::base_llm::{CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream};
    use smartloop::llms::BaseLLM;

    /// Stand-in for the `MagicMock()` LLMs the Python tests pass around: it
    /// answers every prompt with a fixed reply and records what it was asked.
    pub struct StubLLM {
        pub reply: String,
        pub prompts: Mutex<Vec<String>>,
        cancelled: Mutex<Option<CancelEvent>>,
    }

    impl StubLLM {
        /// A stub that replies with `reply` to every prompt.
        pub fn new(reply: impl Into<String>) -> Self {
            Self { reply: reply.into(), prompts: Mutex::new(Vec::new()), cancelled: Mutex::new(None) }
        }

        /// The stub as the trait object every service takes.
        pub fn shared(reply: impl Into<String>) -> Arc<dyn BaseLLM> {
            Arc::new(Self::new(reply))
        }
    }

    impl BaseLLM for StubLLM {
        fn project_id(&self) -> &str {
            "test"
        }
        fn get_num_tokens(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }
        fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
            messages
        }
        fn get_max_tokens(&self) -> usize {
            4096
        }
        fn call(&self, prompt: &str, _stop: Option<&[String]>, _options: CallOptions) -> anyhow::Result<String> {
            self.prompts.lock().push(prompt.to_string());
            Ok(self.reply.clone())
        }
        fn stream<'a>(&'a self, prompt: &str, _options: CallOptions) -> GenerationStream<'a> {
            self.prompts.lock().push(prompt.to_string());
            Box::new(std::iter::once(GenerationChunk { text: self.reply.clone() }))
        }
        fn invoke(&self, prompt: &str, _max_tokens: Option<usize>, _thinking: Option<bool>) -> String {
            self.prompts.lock().push(prompt.to_string());
            self.reply.clone()
        }
        fn cancelled(&self) -> Option<CancelEvent> {
            self.cancelled.lock().clone()
        }
        fn set_cancelled(&self, event: Option<CancelEvent>) {
            *self.cancelled.lock() = event;
        }
        fn llm_type(&self) -> &str {
            "stub"
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
}
// ====================== END orch_exec ======================
