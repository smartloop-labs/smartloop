//! Port of `tests/conftest.py` (shared fixtures).

// ===================================================================== //
// BEGIN orch_steps fixtures                                             //
// ===================================================================== //

/// LLM test doubles for the orchestration-step tests (stand-ins for the
/// Python tests' `MagicMock()` LLMs whose `model.metadata` was set).
pub mod orch_steps {
    use std::any::Any;
    use std::collections::HashMap;
    use std::sync::Arc;

    use parking_lot::Mutex;

    use smartloop::llms::base_llm::{
        CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream,
    };
    use smartloop::llms::llama::{
        ChatCompletionOutput, ChatCompletionRequest, LlamaModel, LlamaState, LlamaToken,
    };
    use smartloop::llms::BaseLLM;

    /// A llama.cpp model whose only readable surface is its GGUF metadata.
    pub struct MetadataModel {
        pub metadata: HashMap<String, String>,
        state: LlamaState,
    }

    impl MetadataModel {
        pub fn new(metadata: HashMap<String, String>) -> Self {
            Self { metadata, state: LlamaState::default() }
        }
    }

    impl LlamaModel for MetadataModel {
        fn state(&self) -> &LlamaState {
            &self.state
        }

        fn n_ctx(&self) -> usize {
            0
        }

        fn metadata(&self) -> HashMap<String, String> {
            self.metadata.clone()
        }

        fn tokenize(&self, _text: &[u8]) -> anyhow::Result<Vec<LlamaToken>> {
            Ok(Vec::new())
        }

        fn detokenize(&self, _tokens: &[LlamaToken]) -> anyhow::Result<Vec<u8>> {
            Ok(Vec::new())
        }

        fn create_chat_completion(
            &self,
            _request: ChatCompletionRequest,
        ) -> anyhow::Result<ChatCompletionOutput<'_>> {
            anyhow::bail!("MetadataModel cannot generate")
        }
    }

    /// Reply strategy of a [`MetadataLlm`]'s `invoke`.
    pub type InvokeFn = Arc<dyn Fn(&str, Option<usize>) -> String + Send + Sync>;

    /// An LLM double: optional GGUF metadata (no metadata = the model is
    /// unreadable, like a mock whose `model` property raises), a scripted
    /// `invoke`, and a record of every prompt it was invoked with.
    pub struct MetadataLlm {
        pub model: Option<Arc<MetadataModel>>,
        pub label: String,
        pub max_tokens: usize,
        pub num_tokens: usize,
        pub thinking: bool,
        pub reply: Mutex<InvokeFn>,
        pub prompts: Mutex<Vec<String>>,
        cancelled: Mutex<Option<CancelEvent>>,
    }

    impl MetadataLlm {
        /// A model whose GGUF reports `general.name` / `general.size_label`.
        pub fn named(name: &str, size: &str) -> Self {
            let mut meta = HashMap::new();
            meta.insert("general.name".to_string(), name.to_string());
            meta.insert("general.size_label".to_string(), size.to_string());
            Self::with_model(Some(Arc::new(MetadataModel::new(meta))))
        }

        /// A model whose metadata cannot be read at all.
        pub fn unreadable() -> Self {
            Self::with_model(None)
        }

        fn with_model(model: Option<Arc<MetadataModel>>) -> Self {
            Self {
                model,
                label: String::new(),
                max_tokens: 8192,
                num_tokens: 100,
                thinking: false,
                reply: Mutex::new(Arc::new(|_p: &str, _m: Option<usize>| "OK".to_string())),
                prompts: Mutex::new(Vec::new()),
                cancelled: Mutex::new(None),
            }
        }

        /// Script `invoke` (Python `llm.invoke = MagicMock(side_effect=...)`).
        pub fn set_reply(&self, reply: impl Fn(&str, Option<usize>) -> String + Send + Sync + 'static) {
            *self.reply.lock() = Arc::new(reply);
        }

        /// Number of `invoke` calls so far.
        pub fn invoke_count(&self) -> usize {
            self.prompts.lock().len()
        }
    }

    impl BaseLLM for MetadataLlm {
        fn project_id(&self) -> &str {
            ""
        }

        fn get_num_tokens(&self, _text: &str) -> usize {
            self.num_tokens
        }

        fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
            messages
        }

        fn get_max_tokens(&self) -> usize {
            self.max_tokens
        }

        fn call(&self, prompt: &str, _stop: Option<&[String]>, options: CallOptions) -> anyhow::Result<String> {
            Ok(self.invoke(prompt, options.max_tokens, options.thinking))
        }

        fn stream<'a>(&'a self, prompt: &str, options: CallOptions) -> GenerationStream<'a> {
            let text = self.invoke(prompt, options.max_tokens, options.thinking);
            Box::new(std::iter::once(GenerationChunk { text }))
        }

        fn invoke(&self, prompt: &str, max_tokens: Option<usize>, _thinking: Option<bool>) -> String {
            self.prompts.lock().push(prompt.to_string());
            let reply = self.reply.lock().clone();
            reply(prompt, max_tokens)
        }

        fn supports_thinking(&self) -> bool {
            self.thinking
        }

        fn model_label(&self) -> String {
            self.label.clone()
        }

        fn cancelled(&self) -> Option<CancelEvent> {
            self.cancelled.lock().clone()
        }

        fn set_cancelled(&self, event: Option<CancelEvent>) {
            *self.cancelled.lock() = event;
        }

        fn llm_type(&self) -> &str {
            "metadata-llm"
        }

        fn llama_model(&self) -> Option<Arc<dyn LlamaModel>> {
            self.model.clone().map(|m| m as Arc<dyn LlamaModel>)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// A progress sink that records every `(step, status, message)` it gets
    /// (Python `router.begin_request(MagicMock())`).
    #[derive(Clone, Default)]
    pub struct ProgressRecorder {
        pub calls: Arc<Mutex<Vec<(String, String, String)>>>,
    }

    impl ProgressRecorder {
        pub fn new() -> Self {
            Self::default()
        }

        /// The callback to hand to `ModelRouter::begin_request`.
        pub fn callback(&self) -> smartloop::agents::orchestration::model_router::ProgressCallback {
            let calls = self.calls.clone();
            Arc::new(move |step: &str, status: &str, message: &str| {
                calls.lock().push((step.to_string(), status.to_string(), message.to_string()));
            })
        }

        /// The message argument of every call, in order.
        pub fn messages(&self) -> Vec<String> {
            self.calls.lock().iter().map(|(_, _, m)| m.clone()).collect()
        }
    }
}

// ===================================================================== //
// END orch_steps fixtures                                               //
// ===================================================================== //
