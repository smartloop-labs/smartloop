//! Port of `tests/conftest.py` (shared fixtures).

// ==== BEGIN model_service_a fixtures ====
/// Fixtures shared by the `ModelService` tests (unit `model_service_a`):
/// LLM test doubles, a progress-event recorder, log capture, and a service
/// wired to an in-memory workspace.
pub mod model_service_a {
    use std::any::Any;
    use std::collections::HashMap;
    use std::io::Write;
    use std::sync::Arc;

    use parking_lot::{Mutex, RwLock};

    use smartloop::agents::orchestration::model_router::ProgressCallback;
    use smartloop::api::services::model_service::ModelService;
    use smartloop::llms::base_llm::{BaseLLM, CallOptions, CancelEvent, ChatMessage, GenerationStream};
    use smartloop::llms::llama::{ChatCompletionOutput, ChatCompletionRequest, LlamaModel, LlamaState, LlamaToken};
    use smartloop::model::model_manager::Model;
    use smartloop::model_host::client::{InProcessModelClient, LocalModelService, ModelClient};
    use smartloop::workspace::Workspace;

    /// A llama.cpp model stand-in that only carries GGUF metadata.
    pub struct FakeLlamaModel {
        state: LlamaState,
        metadata: HashMap<String, String>,
    }

    impl FakeLlamaModel {
        pub fn new(metadata: HashMap<String, String>) -> Self {
            Self { state: LlamaState::default(), metadata }
        }
    }

    impl LlamaModel for FakeLlamaModel {
        fn state(&self) -> &LlamaState {
            &self.state
        }

        fn n_ctx(&self) -> usize {
            4096
        }

        fn metadata(&self) -> HashMap<String, String> {
            self.metadata.clone()
        }

        fn tokenize(&self, text: &[u8]) -> anyhow::Result<Vec<LlamaToken>> {
            Ok(text.iter().map(|b| *b as LlamaToken).collect())
        }

        fn detokenize(&self, tokens: &[LlamaToken]) -> anyhow::Result<Vec<u8>> {
            Ok(tokens.iter().map(|t| *t as u8).collect())
        }

        fn create_chat_completion(&self, _request: ChatCompletionRequest) -> anyhow::Result<ChatCompletionOutput<'_>> {
            anyhow::bail!("FakeLlamaModel does not generate")
        }
    }

    /// An LLM test double (Python's `MagicMock()` LLM): a name for debugging,
    /// optional GGUF metadata, and a fixed context size.
    pub struct FakeLLM {
        pub name: String,
        model: Option<Arc<dyn LlamaModel>>,
        pub max_tokens: usize,
        cancelled: Mutex<Option<CancelEvent>>,
    }

    impl FakeLLM {
        /// A double with no metadata.
        pub fn named(name: &str) -> Arc<dyn BaseLLM> {
            Arc::new(Self { name: name.to_string(), model: None, max_tokens: 4096, cancelled: Mutex::new(None) })
        }

        /// `llm.model.metadata = {"general.name": name, "general.size_label": ""}`.
        pub fn with_metadata(name: &str) -> Arc<dyn BaseLLM> {
            let mut meta = HashMap::new();
            meta.insert("general.name".to_string(), name.to_string());
            meta.insert("general.size_label".to_string(), String::new());
            Arc::new(Self {
                name: name.to_string(),
                model: Some(Arc::new(FakeLlamaModel::new(meta))),
                max_tokens: 4096,
                cancelled: Mutex::new(None),
            })
        }
    }

    impl BaseLLM for FakeLLM {
        fn project_id(&self) -> &str {
            ""
        }

        fn get_num_tokens(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }

        fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
            messages
        }

        fn get_max_tokens(&self) -> usize {
            self.max_tokens
        }

        fn call(&self, _prompt: &str, _stop: Option<&[String]>, _options: CallOptions) -> anyhow::Result<String> {
            Ok(String::new())
        }

        fn stream<'a>(&'a self, _prompt: &str, _options: CallOptions) -> GenerationStream<'a> {
            Box::new(std::iter::empty())
        }

        fn invoke(&self, _prompt: &str, _max_tokens: Option<usize>, _thinking: Option<bool>) -> String {
            String::new()
        }

        fn cancelled(&self) -> Option<CancelEvent> {
            self.cancelled.lock().clone()
        }

        fn set_cancelled(&self, event: Option<CancelEvent>) {
            *self.cancelled.lock() = event;
        }

        fn llm_type(&self) -> &str {
            "fake"
        }

        fn llama_model(&self) -> Option<Arc<dyn LlamaModel>> {
            self.model.clone()
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Python `is` for two LLM handles.
    pub fn same(a: &Arc<dyn BaseLLM>, b: &Arc<dyn BaseLLM>) -> bool {
        std::ptr::eq(Arc::as_ptr(a) as *const (), Arc::as_ptr(b) as *const ())
    }

    /// A `progress_callback` that records every `(step, status, message)`.
    #[derive(Clone, Default)]
    pub struct ProgressRecorder {
        pub calls: Arc<Mutex<Vec<(String, String, String)>>>,
    }

    impl ProgressRecorder {
        pub fn new() -> Self {
            Self::default()
        }

        /// The callback to install.
        pub fn callback(&self) -> ProgressCallback {
            let calls = self.calls.clone();
            Arc::new(move |step: &str, status: &str, message: &str| {
                calls.lock().push((step.to_string(), status.to_string(), message.to_string()));
            })
        }

        /// `[c.args[2] for c in callback.call_args_list]`.
        pub fn messages(&self) -> Vec<String> {
            self.calls.lock().iter().map(|(_, _, m)| m.clone()).collect()
        }
    }

    /// A writer appending to a shared buffer (log capture).
    #[derive(Clone, Default)]
    pub struct LogBuffer(pub Arc<Mutex<Vec<u8>>>);

    impl Write for LogBuffer {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogBuffer {
        type Writer = LogBuffer;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Run `f` with a thread-local subscriber capturing everything at
    /// `level` and above (pytest's `caplog.at_level`); returns `f`'s result
    /// and the captured text.
    pub fn capture_logs<R>(level: tracing::Level, f: impl FnOnce() -> R) -> (R, String) {
        let buffer = LogBuffer::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(buffer.clone())
            .with_max_level(level)
            .with_ansi(false)
            .finish();
        let result = tracing::subscriber::with_default(subscriber, f);
        let text = String::from_utf8_lossy(&buffer.0.lock()).into_owned();
        (result, text)
    }

    /// Keep the service in-process (`_remote_device()` → `None`) without
    /// resolving a client from the environment.
    pub fn keep_in_process(svc: &Arc<ModelService>) {
        let local: Arc<dyn LocalModelService> = svc.clone();
        let client: Arc<dyn ModelClient> = Arc::new(InProcessModelClient::new(local));
        svc.set_model_client(Some(client));
    }

    /// `svc._model = Model(workspace=workspace)` (no SLP model attached).
    pub fn attach_workspace(svc: &Arc<ModelService>, workspace: Workspace) -> Arc<RwLock<Workspace>> {
        let shared = Arc::new(RwLock::new(workspace));
        *svc.model.write() = Some(Arc::new(Model::new("model", shared.clone(), None)));
        shared
    }

    /// Points `Workspace::config_file` at a scratch directory for the test's
    /// duration, so nothing a test saves can reach the machine's real
    /// `config.json`.  Process-global: tests using it must be `#[serial]`.
    pub struct ScratchConfig {
        pub dir: tempfile::TempDir,
    }

    impl ScratchConfig {
        pub fn new() -> Self {
            let dir = tempfile::tempdir().expect("tempdir");
            Workspace::set_config_file(Some(dir.path().join("config.json").to_string_lossy().into_owned()));
            ScratchConfig { dir }
        }

        /// A project directory inside the scratch directory.
        pub fn project_dir(&self, project_id: &str) -> String {
            self.dir.path().join(project_id).to_string_lossy().into_owned()
        }
    }

    impl Drop for ScratchConfig {
        fn drop(&mut self) {
            Workspace::set_config_file(None);
        }
    }

    /// A bare service (`object.__new__(ModelService)`) kept in-process.
    pub fn bare_service() -> Arc<ModelService> {
        let svc = ModelService::new_bare();
        keep_in_process(&svc);
        svc
    }
}
// ==== END model_service_a fixtures ====
