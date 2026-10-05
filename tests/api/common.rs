//! Port of `tests/api/conftest.py`.

// ============================================================================
// api_core — the in-process client, platform and model-service overrides, and
// the download guard (`no_real_downloads`, `platform_service`, `client`).
// ============================================================================
#[allow(dead_code)]
pub mod api_core {
    //! API test fixtures.
    //!
    //! Nothing here may reach the network, and one guarantee needs stating
    //! outright: **no test downloads a model.**  The routers under test build
    //! their own `SLPArtifactDownloader` rather than receiving one, so
    //! overriding the model service does not stop them — and `/v1/init` hands
    //! the transfer to a blocking worker that outlives the response.  So the
    //! download methods are stubbed process-wide (see
    //! `smartloop::model::downloader::slp_downloader::set_download_stubs`), at
    //! every downloader rather than at any one call site, and the guard proves
    //! afterwards that no weights were written.

    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::{Method, Request};
    use http_body_util::BodyExt;
    use parking_lot::Mutex;
    use serde_json::Value;
    use tower::ServiceExt;

    use smartloop::api::services::{set_model_service_override, ModelService};
    use smartloop::llms::base_llm::{BaseLLM, CallOptions, CancelEvent, ChatMessage, GenerationStream};
    use smartloop::model::downloader::slp_downloader::{set_download_stubs, DownloadStubCall, DownloadStubs};
    use smartloop::platform::client::{JsonObject, PlatformResult};
    use smartloop::platform::{set_platform_service_override, PlatformApi, PlatformService};

    /// Files a real download leaves behind, complete or interrupted.
    pub const WEIGHT_SUFFIXES: &[&str] = &["gguf", "part"];

    /// Sets environment variables for the guard's lifetime and restores the
    /// previous values afterwards (`monkeypatch.setenv`).
    pub struct EnvGuard {
        saved: Vec<(String, Option<String>)>,
    }

    impl EnvGuard {
        pub fn new() -> Self {
            Self { saved: Vec::new() }
        }

        pub fn set(&mut self, key: &str, value: &str) {
            self.saved.push((key.to_string(), std::env::var(key).ok()));
            // SAFETY: API tests that touch the environment run #[serial].
            unsafe { std::env::set_var(key, value) };
        }

        pub fn remove(&mut self, key: &str) {
            self.saved.push((key.to_string(), std::env::var(key).ok()));
            // SAFETY: see `set`.
            unsafe { std::env::remove_var(key) };
        }
    }

    impl Default for EnvGuard {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (key, value) in self.saved.drain(..).rev() {
                // SAFETY: see `set`.
                unsafe {
                    match value {
                        Some(value) => std::env::set_var(&key, value),
                        None => std::env::remove_var(&key),
                    }
                }
            }
        }
    }

    /// `no_real_downloads`: stub every path that would fetch weights, and
    /// prove none was taken.
    ///
    /// The stubs stand in for a *successful* download, so a route that goes on
    /// to register the model still behaves as it would in life.  What they do
    /// not do is touch the network or the disk.
    pub struct NoRealDownloads {
        pub tmp: tempfile::TempDir,
        pub calls: Arc<Mutex<Vec<(String, String)>>>,
    }

    impl NoRealDownloads {
        pub fn install() -> Self {
            let tmp = tempfile::tempdir().expect("tmp dir");
            let calls: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
            let fallback = tmp.path().to_path_buf();

            let seen = Arc::clone(&calls);
            let dir = fallback.clone();
            let download = Arc::new(move |call: DownloadStubCall| -> anyhow::Result<PathBuf> {
                match call {
                    DownloadStubCall::Download { model_id, models_dir } => {
                        seen.lock().push(("download".into(), model_id));
                        Ok(models_dir.unwrap_or_else(|| dir.clone()))
                    }
                    DownloadStubCall::BootstrapDownload { model_name, models_dir } => {
                        seen.lock().push(("bootstrap_download".into(), model_name.unwrap_or_default()));
                        Ok(models_dir.unwrap_or_else(|| dir.clone()))
                    }
                }
            });
            set_download_stubs(Some(DownloadStubs {
                download: Some(download.clone()),
                bootstrap_download: Some(download),
                resolve_artifacts: None,
            }));
            Self { tmp, calls }
        }

        /// Also answer `resolve_artifacts` (the Python tests replacing the
        /// router's downloader with a mock whose `resolve_artifacts` returns a
        /// signed-URL map).
        pub fn resolve_artifacts_with(&self, artifacts: &[(&str, &str)]) {
            let mut stubs = smartloop::model::downloader::slp_downloader::download_stubs().unwrap_or_default();
            let map: smartloop::model::downloader::artifact_source::ArtifactMap =
                artifacts.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
            stubs.resolve_artifacts = Some(Arc::new(move |_model_id: &str| Ok(map.clone())));
            set_download_stubs(Some(stubs));
        }

        pub fn path(&self) -> &Path {
            self.tmp.path()
        }
    }

    /// Every weight-shaped file under `root`.
    pub fn leaked_weights(root: &Path) -> Vec<PathBuf> {
        walkdir::WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
            .filter(|e| {
                e.path().extension().and_then(|x| x.to_str()).is_some_and(|x| WEIGHT_SUFFIXES.contains(&x))
            })
            .map(|e| e.path().to_path_buf())
            .collect()
    }

    impl Drop for NoRealDownloads {
        fn drop(&mut self) {
            set_download_stubs(None);
            // The stub can stop matching — a route that starts building its
            // downloader somewhere new, or a rename — and the only symptom
            // would be the disk filling again.  Assert the outcome rather than
            // trusting the patch.
            let leaked = leaked_weights(self.tmp.path());
            if !std::thread::panicking() {
                assert!(
                    leaked.is_empty(),
                    "a real model download escaped the stub and wrote {leaked:?} — check whether a new \
                     code path builds its own downloader"
                );
            }
        }
    }

    /// The platform answers a test configures (`platform_service.list_models
    /// .return_value` / `.side_effect`).
    #[derive(Default)]
    pub struct FakePlatform {
        pub list_models: Mutex<Option<PlatformResult<Vec<JsonObject>>>>,
        pub list_models_calls: Mutex<usize>,
    }

    impl FakePlatform {
        /// `platform_service.list_models.return_value = rows`.
        pub fn set_list_models(&self, rows: Value) {
            let rows = rows
                .as_array()
                .expect("a list of model rows")
                .iter()
                .map(|row| row.as_object().cloned().expect("a model row object"))
                .collect();
            *self.list_models.lock() = Some(Ok(rows));
        }

        /// `platform_service.list_models.side_effect = exc`.
        pub fn fail_list_models(&self, error: smartloop::platform::PlatformError) {
            *self.list_models.lock() = Some(Err(error));
        }
    }

    impl PlatformApi for FakePlatform {
        fn list_models(&self) -> PlatformResult<Vec<JsonObject>> {
            *self.list_models_calls.lock() += 1;
            match &*self.list_models.lock() {
                Some(Ok(rows)) => Ok(rows.clone()),
                Some(Err(e)) => Err(e.clone()),
                None => Ok(Vec::new()),
            }
        }
    }

    /// `platform_service`: a fake platform bound as the platform service for
    /// the guard's lifetime.  No real DB or HTTP I/O.
    pub struct PlatformServiceFixture {
        pub fake: Arc<FakePlatform>,
    }

    impl PlatformServiceFixture {
        pub fn install() -> Self {
            let fake = Arc::new(FakePlatform::default());
            let service = PlatformService::new(fake.clone(), None);
            set_platform_service_override(Some(Arc::new(service)));
            Self { fake }
        }
    }

    impl Drop for PlatformServiceFixture {
        fn drop(&mut self) {
            set_platform_service_override(None);
        }
    }

    /// `app.dependency_overrides[get_model_service] = lambda: svc` for the
    /// guard's lifetime.
    pub struct ModelServiceOverride {
        pub service: Arc<ModelService>,
    }

    impl ModelServiceOverride {
        pub fn install(service: Arc<ModelService>) -> Self {
            set_model_service_override(Some(service.clone()));
            Self { service }
        }
    }

    impl Drop for ModelServiceOverride {
        fn drop(&mut self) {
            set_model_service_override(None);
        }
    }

    /// A response read to the end.
    #[derive(Debug, Clone)]
    pub struct TestResponse {
        pub status_code: u16,
        pub headers: axum::http::HeaderMap,
        pub body: Vec<u8>,
    }

    impl TestResponse {
        pub fn json(&self) -> Value {
            serde_json::from_slice(&self.body)
                .unwrap_or_else(|e| panic!("response is not JSON ({e}): {}", self.text()))
        }

        pub fn text(&self) -> String {
            String::from_utf8_lossy(&self.body).into_owned()
        }
    }

    /// `client`: the full application, in process — no real server socket —
    /// with the platform bound to a fake (the Python `client` fixture depends
    /// on `platform_service`), downloads stubbed, and the framework home
    /// pointed at a temp directory so nothing touches the real install.
    pub struct TestClient {
        pub platform: PlatformServiceFixture,
        pub downloads: NoRealDownloads,
        pub home: tempfile::TempDir,
        pub env: EnvGuard,
    }

    impl TestClient {
        pub fn new() -> Self {
            let downloads = NoRealDownloads::install();
            let platform = PlatformServiceFixture::install();
            let home = tempfile::tempdir().expect("tmp home");
            let mut env = EnvGuard::new();
            env.set("SLP_HOME", &home.path().to_string_lossy());
            // Anything that slips past the stubs fails fast instead of
            // reaching the real platform.
            env.set("SLP_API_BASE", "http://127.0.0.1:9");
            Self { platform, downloads, home, env }
        }

        /// The platform fake (`platform_service`).
        pub fn platform(&self) -> &FakePlatform {
            &self.platform.fake
        }

        pub async fn request(&self, method: Method, path: &str, json: Option<Value>) -> TestResponse {
            let mut builder = Request::builder().method(method).uri(path);
            let body = match json {
                Some(json) => {
                    builder = builder.header("content-type", "application/json");
                    Body::from(json.to_string())
                }
                None => Body::empty(),
            };
            let response = smartloop::api::main::app()
                .oneshot(builder.body(body).expect("request"))
                .await
                .expect("infallible service");
            let status_code = response.status().as_u16();
            let headers = response.headers().clone();
            let body = response.into_body().collect().await.expect("body").to_bytes().to_vec();
            TestResponse { status_code, headers, body }
        }

        pub async fn get(&self, path: &str) -> TestResponse {
            self.request(Method::GET, path, None).await
        }

        pub async fn post(&self, path: &str, json: Value) -> TestResponse {
            self.request(Method::POST, path, Some(json)).await
        }
    }

    impl Default for TestClient {
        fn default() -> Self {
            Self::new()
        }
    }

    /// A loaded model that never generates — what a successful `load_model`
    /// hands back when the test only cares that loading succeeded.
    #[derive(Default)]
    pub struct NullLlm {
        cancelled: Mutex<Option<CancelEvent>>,
    }

    impl BaseLLM for NullLlm {
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
        fn call(&self, _prompt: &str, _stop: Option<&[String]>, _options: CallOptions) -> anyhow::Result<String> {
            Ok(String::new())
        }
        fn stream<'a>(&'a self, _prompt: &str, _options: CallOptions) -> GenerationStream<'a> {
            Box::new(std::iter::empty())
        }
        fn cancelled(&self) -> Option<CancelEvent> {
            self.cancelled.lock().clone()
        }
        fn set_cancelled(&self, event: Option<CancelEvent>) {
            *self.cancelled.lock() = event;
        }
        fn llm_type(&self) -> &str {
            "null"
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
}
// ============================================================================
// end api_core
// ============================================================================
