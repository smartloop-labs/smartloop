//! Port of `tests/conftest.py` (shared fixtures).

// ============================================================ api_chat (begin)
// Shared by the chat-router tests (tests/unit/test_stream_*.rs,
// test_orchestrate_timeout_unload.rs): a `ModelService` stand-in, the
// Python tests' `MagicMock` service, with the real `InProcessModelClient` put
// in front of it so the tests drive exactly the path production takes.
pub mod api_chat {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use parking_lot::{Condvar, Mutex};

    use smartloop::api::routers::chat::{
        ChatService, CitationDict, HistoryTurn, ImageAttachment, OrchestrateArgs,
        OrchestrateOutcome, ProgressCallback,
    };
    use smartloop::llms::base_llm::{CancelEvent, GenerationChunk};
    use smartloop::model_host::client::{InProcessModelClient, LocalModelService, ModelClient};
    use smartloop::types::ModelCapability;
    use smartloop::workspace::types::Skill;

    /// Python's `threading.Event`.
    #[derive(Clone, Default)]
    pub struct Event {
        inner: Arc<(Mutex<bool>, Condvar)>,
    }

    impl Event {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn set(&self) {
            let (flag, cv) = &*self.inner;
            *flag.lock() = true;
            cv.notify_all();
        }

        pub fn is_set(&self) -> bool {
            *self.inner.0.lock()
        }

        /// Block until set or `timeout_s` elapses; returns whether it is set.
        pub fn wait(&self, timeout_s: f64) -> bool {
            let (flag, cv) = &*self.inner;
            let deadline = Instant::now() + Duration::from_secs_f64(timeout_s);
            let mut guard = flag.lock();
            while !*guard {
                if cv.wait_until(&mut guard, deadline).timed_out() {
                    break;
                }
            }
            *guard
        }
    }

    /// Runs a callback when dropped — a generator's `finally:` block.
    pub struct OnDrop(Option<Box<dyn FnOnce() + Send>>);

    impl OnDrop {
        pub fn new(f: impl FnOnce() + Send + 'static) -> Self {
            Self(Some(Box::new(f)))
        }
    }

    impl Drop for OnDrop {
        fn drop(&mut self) {
            if let Some(f) = self.0.take() {
                f();
            }
        }
    }

    /// `GenerationChunk(text=...)`.
    pub fn chunk(text: &str) -> GenerationChunk {
        GenerationChunk { text: text.to_string() }
    }

    /// The chunks a fake `stream_completion` produces.
    pub type ChunkIter = Box<dyn Iterator<Item = anyhow::Result<GenerationChunk>>>;

    /// A fake `stream_completion(prompt, cancelled=...)`: a generator function
    /// — nothing in the body runs until the returned iterator is advanced.
    pub type StreamFn = Arc<dyn Fn(&str, Option<CancelEvent>) -> ChunkIter + Send + Sync>;

    /// The device-owning half of the fake: what `InProcessModelClient` calls.
    pub struct FakeDevice {
        pub stream_fn: Mutex<Option<StreamFn>>,
        /// `generate_completion.return_value`.
        pub generate_return: Mutex<String>,
        /// Prompts `generate_completion` was called with.
        pub generate_prompts: Mutex<Vec<String>>,
        /// `include_embedding` of every `unload_auxiliary_models` call.
        pub unload_calls: Mutex<Vec<bool>>,
        /// `unload_auxiliary_models.side_effect`: fail with this message.
        pub unload_failure: Mutex<Option<String>>,
        pub unload_embedding_calls: Mutex<usize>,
    }

    impl FakeDevice {
        fn record_unload(&self, include_embedding: bool) {
            self.unload_calls.lock().push(include_embedding);
            let failure = self.unload_failure.lock().clone();
            if let Some(message) = failure {
                panic!("{message}");
            }
        }
    }

    impl LocalModelService for FakeDevice {
        fn generate_completion(
            &self,
            prompt: &str,
            _model_name: Option<&str>,
            _capability: Option<&str>,
            _max_tokens: Option<usize>,
            _thinking: Option<bool>,
        ) -> anyhow::Result<String> {
            self.generate_prompts.lock().push(prompt.to_string());
            Ok(self.generate_return.lock().clone())
        }

        fn stream_completion<'a>(
            &'a self,
            prompt: &str,
            _model_name: Option<&str>,
            cancelled: Option<CancelEvent>,
            _capability: Option<&str>,
            _thinking: Option<bool>,
        ) -> anyhow::Result<Box<dyn Iterator<Item = anyhow::Result<GenerationChunk>> + 'a>> {
            let stream_fn = self.stream_fn.lock().clone();
            match stream_fn {
                Some(f) => Ok(f(prompt, cancelled)),
                None => Ok(Box::new(std::iter::empty())),
            }
        }

        fn is_loaded(&self) -> bool {
            true
        }

        fn resident_model_name(&self) -> Option<String> {
            Some("sl-mini".to_string())
        }

        fn response_model_label(&self, _capability: Option<&str>) -> String {
            "sl-mini".to_string()
        }

        fn count_tokens(&self, _text: &str) -> usize {
            10
        }

        fn get_max_context_tokens(&self) -> usize {
            4096
        }

        fn get_response_max_context_tokens(&self, _capability: Option<&str>) -> usize {
            4096
        }

        fn has_capability_model(&self, _capability: &str) -> bool {
            false
        }

        fn idle_seconds(&self) -> f64 {
            0.0
        }

        fn load_model(&self) -> anyhow::Result<()> {
            Ok(())
        }

        fn unload_auxiliary_models(&self, include_embedding: bool) {
            self.record_unload(include_embedding);
        }

        fn unload_embedding_model(&self) {
            *self.unload_embedding_calls.lock() += 1;
        }
    }

    /// The `MagicMock` `ModelService` of the chat tests.
    pub struct FakeChatService {
        pub device: Arc<FakeDevice>,
        /// `orchestrate.return_value`.
        pub orchestrate_outcome: Mutex<OrchestrateOutcome>,
        /// User messages `orchestrate` was called with.
        pub orchestrate_calls: Mutex<Vec<String>>,
        /// `_model.workspace.workspace_dir`.
        pub workspace_dir: Mutex<Option<String>>,
        /// `(session_id, role, content)` of every `append_to_session`.
        pub session_writes: Mutex<Vec<(String, String, String)>>,
    }

    impl Default for FakeChatService {
        fn default() -> Self {
            Self {
                device: Arc::new(FakeDevice {
                    stream_fn: Mutex::new(None),
                    generate_return: Mutex::new(String::new()),
                    generate_prompts: Mutex::new(Vec::new()),
                    unload_calls: Mutex::new(Vec::new()),
                    unload_failure: Mutex::new(None),
                    unload_embedding_calls: Mutex::new(0),
                }),
                orchestrate_outcome: Mutex::new(OrchestrateOutcome::default()),
                orchestrate_calls: Mutex::new(Vec::new()),
                workspace_dir: Mutex::new(None),
                session_writes: Mutex::new(Vec::new()),
            }
        }
    }

    impl FakeChatService {
        /// A service whose `stream_completion` is `stream_fn`.
        pub fn with_stream<F>(stream_fn: F) -> Self
        where
            F: Fn(&str, Option<CancelEvent>) -> ChunkIter + Send + Sync + 'static,
        {
            let svc = Self::default();
            *svc.device.stream_fn.lock() = Some(Arc::new(stream_fn));
            svc
        }

        /// `include_embedding` of every `unload_auxiliary_models` call.
        pub fn unload_calls(&self) -> Vec<bool> {
            self.device.unload_calls.lock().clone()
        }

        /// `unload_auxiliary_models.call_count`.
        pub fn unload_call_count(&self) -> usize {
            self.device.unload_calls.lock().len()
        }

        /// Prompts `generate_completion` was called with.
        pub fn generate_prompts(&self) -> Vec<String> {
            self.device.generate_prompts.lock().clone()
        }
    }

    impl ChatService for FakeChatService {
        fn is_loaded(&self) -> bool {
            true
        }
        fn load_model(&self) -> anyhow::Result<()> {
            Ok(())
        }
        fn get_relevant_history(&self, _session_id: &str, _user_query: &str) -> Vec<HistoryTurn> {
            Vec::new()
        }
        fn get_project_skills(&self) -> Vec<Skill> {
            Vec::new()
        }
        fn get_asset_images(&self, _asset_ids: &[String]) -> Vec<ImageAttachment> {
            Vec::new()
        }
        fn resolve_attachment_context(&self, _asset_ids: &[String]) -> String {
            String::new()
        }
        fn orchestrate(&self, args: OrchestrateArgs) -> anyhow::Result<OrchestrateOutcome> {
            self.orchestrate_calls.lock().push(args.user_message);
            Ok(self.orchestrate_outcome.lock().clone())
        }
        fn get_project_id(&self) -> Option<String> {
            Some("proj".to_string())
        }
        fn get_project_info(&self) -> Option<String> {
            None
        }
        fn count_tokens(&self, _text: &str) -> usize {
            10
        }
        fn get_max_context_tokens(&self) -> usize {
            4096
        }
        fn get_response_max_context_tokens(&self, _capability: Option<ModelCapability>) -> usize {
            4096
        }
        fn begin_response_request(&self, _progress_callback: Option<ProgressCallback>) {}
        fn announce_response_model(&self, _capability: Option<ModelCapability>) {}
        fn append_to_session(
            &self,
            session_id: &str,
            role: &str,
            content: &str,
            _citations: Option<&[CitationDict]>,
            _message_id: Option<&str>,
        ) {
            self.session_writes.lock().push((
                session_id.to_string(),
                role.to_string(),
                content.to_string(),
            ));
        }
        fn maybe_write_output(
            &self,
            _user_message: &str,
            _llm_response: &str,
            _cwd: Option<&str>,
            _session_id: Option<&str>,
        ) -> Option<String> {
            None
        }
        fn model_client(&self) -> Arc<dyn ModelClient> {
            Arc::new(InProcessModelClient::new(self.device.clone()))
        }
        fn unload_auxiliary_models(&self, include_embedding: bool) {
            self.device.record_unload(include_embedding);
        }
        fn unload_embedding_model(&self) {
            self.device.unload_embedding_model();
        }
        fn workspace_dir(&self) -> Option<String> {
            self.workspace_dir.lock().clone()
        }
        fn workspace_project_ids(&self) -> Option<Vec<String>> {
            None
        }
        fn current_project_dir(&self) -> Option<String> {
            None
        }
        /// The `_no_telemetry` fixture: the usage tail stays offline.
        fn push_token_usage(
            &self,
            _project_id: Option<&str>,
            _model: &str,
            _prompt_tokens: usize,
            _completion_tokens: usize,
        ) {
        }
    }

    /// The content deltas of raw SSE frames, in order.
    pub fn content(raw: &[String]) -> Vec<String> {
        let mut out = Vec::new();
        for chunk in raw {
            for line in chunk.lines() {
                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };
                if data == "[DONE]" {
                    continue;
                }
                let event: serde_json::Value =
                    serde_json::from_str(data).expect("SSE data is JSON");
                if let Some(choices) = event.get("choices").and_then(|c| c.as_array()) {
                    for choice in choices {
                        if let Some(text) = choice
                            .get("delta")
                            .and_then(|d| d.get("content"))
                            .and_then(|c| c.as_str())
                            .filter(|t| !t.is_empty())
                        {
                            out.push(text.to_string());
                        }
                    }
                }
            }
        }
        out
    }

    /// Sets an env var for the life of the guard (`monkeypatch.setenv`).
    pub struct EnvGuard {
        key: String,
        previous: Option<String>,
    }

    impl EnvGuard {
        pub fn set(key: &str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            // SAFETY: the tests that touch the environment are `#[serial]`.
            unsafe { std::env::set_var(key, value) };
            Self { key: key.to_string(), previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: see `EnvGuard::set`.
            unsafe {
                match &self.previous {
                    Some(v) => std::env::set_var(&self.key, v),
                    None => std::env::remove_var(&self.key),
                }
            }
        }
    }
}
// ============================================================== api_chat (end)
