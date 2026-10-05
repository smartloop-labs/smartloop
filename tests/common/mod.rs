//! Port of `tests/conftest.py` (shared fixtures).

// ==== model_service_b ====
/// Test doubles for the `ModelService` tests owned by unit `model_service_b`:
/// a recording LLM (standing in for the `MagicMock` LLMs the Python tests
/// built), a llama.cpp model carrying GGUF metadata, a workspace-backed
/// service, and a lock serializing tests that touch process-global state.
pub mod model_service_b {
    use std::any::Any;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use parking_lot::{Mutex, MutexGuard, RwLock};

    use smartloop::api::config::api_settings;
    use smartloop::api::services::model_service::ModelService;
    use smartloop::db::store::{Chunk, ChunkResult, VectorStore};
    use smartloop::llms::base_llm::{BaseLLM, CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream};
    use smartloop::llms::llama::{ChatCompletionOutput, ChatCompletionRequest, LlamaModel, LlamaState, LlamaToken};
    use smartloop::model::model_manager::Model;
    use smartloop::workspace::workspace::{Project, Workspace, WorkspaceModel};

    /// Serializes tests that touch process-global state (the project pin,
    /// delegation flag, the workspace config-file override, `HOME`).
    pub fn global_lock() -> MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock()
    }

    /// A llama.cpp model that only carries GGUF metadata.
    pub struct FakeLlamaModel {
        state: LlamaState,
        pub metadata: HashMap<String, String>,
        pub closed: AtomicBool,
    }

    impl FakeLlamaModel {
        pub fn new(general_name: &str, size_label: &str) -> Self {
            let mut metadata = HashMap::new();
            metadata.insert("general.name".to_string(), general_name.to_string());
            metadata.insert("general.size_label".to_string(), size_label.to_string());
            Self { state: LlamaState::default(), metadata, closed: AtomicBool::new(false) }
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
            Ok(text.split(|b| *b == b' ').map(|_| 1).collect())
        }

        fn detokenize(&self, _tokens: &[LlamaToken]) -> anyhow::Result<Vec<u8>> {
            Ok(Vec::new())
        }

        fn create_chat_completion(&self, _request: ChatCompletionRequest) -> anyhow::Result<ChatCompletionOutput<'_>> {
            anyhow::bail!("FakeLlamaModel cannot generate")
        }

        fn close(&self) {
            self.closed.store(true, Ordering::SeqCst);
        }
    }

    /// A recording LLM: every prompt it is asked to complete is kept, and it
    /// answers with a fixed reply.
    pub struct FakeLLM {
        pub name: String,
        pub reply: RwLock<String>,
        pub prompts: Mutex<Vec<String>>,
        pub model: Option<Arc<FakeLlamaModel>>,
        pub cancelled: Mutex<Option<CancelEvent>>,
        pub stream_chunks: RwLock<Vec<String>>,
        pub panic_on_generate: AtomicBool,
        pub streams_started: AtomicUsize,
    }

    impl FakeLLM {
        /// A bare recording LLM named `name` (no GGUF metadata).
        pub fn new(name: &str) -> Arc<FakeLLM> {
            Arc::new(Self::build(name, None))
        }

        /// A recording LLM whose GGUF metadata names it `general_name`.
        pub fn named(general_name: &str) -> Arc<FakeLLM> {
            Self::labelled(general_name, "")
        }

        /// A recording LLM whose GGUF metadata carries a name and size label.
        pub fn labelled(general_name: &str, size_label: &str) -> Arc<FakeLLM> {
            Arc::new(Self::build(general_name, Some(Arc::new(FakeLlamaModel::new(general_name, size_label)))))
        }

        fn build(name: &str, model: Option<Arc<FakeLlamaModel>>) -> FakeLLM {
            FakeLLM {
                name: name.to_string(),
                reply: RwLock::new(r#"{"description": "Weather lookups", "tags": ["weather", "forecast"]}"#.to_string()),
                prompts: Mutex::new(Vec::new()),
                model,
                cancelled: Mutex::new(None),
                stream_chunks: RwLock::new(vec!["hello".to_string(), " world".to_string()]),
                panic_on_generate: AtomicBool::new(false),
                streams_started: AtomicUsize::new(0),
            }
        }

        /// How many generations this LLM was asked for.
        pub fn calls(&self) -> usize {
            self.prompts.lock().len()
        }

        /// This LLM as the trait object the service stores.
        pub fn dyn_llm(self: &Arc<Self>) -> Arc<dyn BaseLLM> {
            self.clone()
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
            4096
        }

        fn call(&self, prompt: &str, _stop: Option<&[String]>, _options: CallOptions) -> anyhow::Result<String> {
            self.prompts.lock().push(prompt.to_string());
            if self.panic_on_generate.load(Ordering::SeqCst) {
                panic!("{} ran out of device memory", self.name);
            }
            Ok(self.reply.read().clone())
        }

        fn stream<'a>(&'a self, prompt: &str, _options: CallOptions) -> GenerationStream<'a> {
            self.prompts.lock().push(prompt.to_string());
            self.streams_started.fetch_add(1, Ordering::SeqCst);
            if self.panic_on_generate.load(Ordering::SeqCst) {
                panic!("{} ran out of device memory", self.name);
            }
            let chunks: Vec<GenerationChunk> =
                self.stream_chunks.read().iter().map(|t| GenerationChunk { text: t.clone() }).collect();
            Box::new(chunks.into_iter())
        }

        fn invoke(&self, prompt: &str, max_tokens: Option<usize>, thinking: Option<bool>) -> String {
            let options = CallOptions { max_tokens, thinking };
            match self.call(prompt, None, options) {
                Ok(text) => text,
                Err(e) => format!("Error generating response: {e}"),
            }
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
            self.model.clone().map(|m| m as Arc<dyn LlamaModel>)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// An in-memory vector store answering every query with fixed chunks.
    pub struct FakeStore {
        pub label: String,
        pub chunks: Vec<ChunkResult>,
    }

    impl FakeStore {
        pub fn new(label: &str, chunks: Vec<ChunkResult>) -> Arc<FakeStore> {
            Arc::new(FakeStore { label: label.to_string(), chunks })
        }
    }

    impl VectorStore for FakeStore {
        fn project_dir(&self) -> &str {
            ""
        }

        fn collection_name(&self) -> &str {
            "documents"
        }

        fn base_dir(&self) -> Option<&str> {
            None
        }

        fn add_chunks(&self, _chunks: &[Chunk]) -> anyhow::Result<()> {
            Ok(())
        }

        fn query(&self, _query_text: &str, _n_results: Option<usize>) -> anyhow::Result<Vec<String>> {
            Ok(self.chunks.iter().map(|c| c.text.clone()).collect())
        }

        fn query_with_metadata(
            &self,
            _query_text: &str,
            _n_results: Option<usize>,
            _max_distance: f64,
            _where: Option<&serde_json::Map<String, serde_json::Value>>,
        ) -> anyhow::Result<Vec<ChunkResult>> {
            Ok(self.chunks.clone())
        }

        fn get_by_document_ids(&self, document_ids: &[String]) -> anyhow::Result<Vec<ChunkResult>> {
            Ok(self.chunks.iter().filter(|c| document_ids.contains(&c.document_id)).cloned().collect())
        }

        fn delete_document(&self, _document_id: &str) -> anyhow::Result<()> {
            Ok(())
        }
    }

    /// Identity comparison of two LLM handles (Python `is`).
    pub fn same(a: &Arc<dyn BaseLLM>, b: &Arc<dyn BaseLLM>) -> bool {
        std::ptr::addr_eq(Arc::as_ptr(a), Arc::as_ptr(b))
    }

    /// The bootstrap/orchestrator model name.
    pub fn default_model() -> String {
        api_settings().default_model
    }

    /// A workspace model the user can select.
    pub fn ws_model(name: &str, enabled: bool, downloaded: bool) -> WorkspaceModel {
        WorkspaceModel {
            model_name: name.to_string(),
            model_id: format!("org/{name}"),
            enabled,
            downloaded,
            ..WorkspaceModel::default()
        }
    }

    /// A project rooted at `project_dir`.
    pub fn project(id: &str, project_dir: &str, current: bool) -> Project {
        Project {
            id: id.to_string(),
            project_dir: project_dir.to_string(),
            current,
            name: Some(id.to_string()),
            ..Project::default()
        }
    }

    /// A bare service whose `_model` holds a workspace with `projects` and
    /// `models`, and whose `_model_name` is the default model so the
    /// `_ensure_model` / `_initialize_workspace` guards see it as loaded.
    pub fn service_with_workspace(
        workspace_dir: &str,
        projects: Vec<Project>,
        models: Vec<WorkspaceModel>,
    ) -> Arc<ModelService> {
        let svc = ModelService::new_bare();
        let mut workspace = Workspace::new("ws-test");
        workspace.workspace_dir = workspace_dir.to_string();
        workspace.models_dir = format!("{workspace_dir}/models");
        workspace.projects = projects;
        workspace.models = models;
        let shared = Arc::new(RwLock::new(workspace));
        *svc.model.write() = Some(Arc::new(Model::new("model-test", shared, None)));
        *svc.model_name.write() = Some(default_model());
        svc
    }
}
// ==== end model_service_b ====
