//! Port of `tests/api/conftest.py`.

// ======================= BEGIN unit: core ========================
/// The `platform_service` / `client` fixtures: a scripted platform client
/// behind a real [`PlatformService`], installed as the
/// `get_platform_service` override for the life of the fixture, plus a
/// helper that drives the axum app in-process (FastAPI's `TestClient`).
#[allow(dead_code)]
pub mod core_support {
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use http_body_util::BodyExt;
    use once_cell::sync::Lazy;
    use parking_lot::{Mutex, MutexGuard};
    use serde_json::Value;
    use smartloop::platform::client::{JsonObject, PlatformResult};
    use smartloop::platform::{set_platform_service_override, PlatformApi, PlatformError, PlatformService, WorkspaceDB};
    use tower::ServiceExt;

    use crate::shared::core_support::{isolate_credential_store, CredentialStoreGuard};

    /// One recorded call on the fake: method name and its arguments.
    #[derive(Debug, Clone, PartialEq)]
    pub struct Call {
        pub method: &'static str,
        pub args: Vec<Value>,
    }

    /// The answer a fake method gives (`return_value` / `side_effect`).
    type Scripted<T> = Mutex<Option<PlatformResult<T>>>;

    /// A platform client whose answers are scripted per method, recording
    /// every call (the `MagicMock` the Python fixture installed).  A method
    /// with no script fails, so an unexpected call cannot pass silently.
    #[derive(Default)]
    pub struct FakePlatform {
        pub calls: Mutex<Vec<Call>>,
        current_user: Scripted<JsonObject>,
        update_current_user: Scripted<JsonObject>,
        list_models: Scripted<Vec<JsonObject>>,
        list_mcp_catalog: Scripted<Vec<JsonObject>>,
        latest_release: Scripted<Option<JsonObject>>,
        verify_apple_transaction: Scripted<JsonObject>,
        checkout_session: Scripted<JsonObject>,
    }

    fn answer<T: Clone>(slot: &Scripted<T>, method: &str) -> PlatformResult<T> {
        slot.lock()
            .clone()
            .unwrap_or_else(|| Err(PlatformError::Other(format!("unexpected call to {method}"))))
    }

    /// A JSON object literal as a [`JsonObject`].
    pub fn obj(value: Value) -> JsonObject {
        match value {
            Value::Object(map) => map,
            other => panic!("expected a JSON object, got {other}"),
        }
    }

    impl FakePlatform {
        pub fn set_current_user(&self, result: PlatformResult<JsonObject>) {
            *self.current_user.lock() = Some(result);
        }
        pub fn set_update_current_user(&self, result: PlatformResult<JsonObject>) {
            *self.update_current_user.lock() = Some(result);
        }
        pub fn set_list_models(&self, result: PlatformResult<Vec<JsonObject>>) {
            *self.list_models.lock() = Some(result);
        }
        pub fn set_list_mcp_catalog(&self, result: PlatformResult<Vec<JsonObject>>) {
            *self.list_mcp_catalog.lock() = Some(result);
        }
        pub fn set_latest_release(&self, result: PlatformResult<Option<JsonObject>>) {
            *self.latest_release.lock() = Some(result);
        }
        pub fn set_verify_apple_transaction(&self, result: PlatformResult<JsonObject>) {
            *self.verify_apple_transaction.lock() = Some(result);
        }
        pub fn set_checkout_session(&self, result: PlatformResult<JsonObject>) {
            *self.checkout_session.lock() = Some(result);
        }

        /// Every recorded call to `method`, oldest first.
        pub fn calls_to(&self, method: &str) -> Vec<Call> {
            self.calls.lock().iter().filter(|c| c.method == method).cloned().collect()
        }

        fn record(&self, method: &'static str, args: Vec<Value>) {
            self.calls.lock().push(Call { method, args });
        }
    }

    impl PlatformApi for FakePlatform {
        fn list_models(&self) -> PlatformResult<Vec<JsonObject>> {
            self.record("list_models", vec![]);
            answer(&self.list_models, "list_models")
        }
        fn get_latest_release(&self, channel: &str) -> PlatformResult<Option<JsonObject>> {
            self.record("get_latest_release", vec![Value::from(channel)]);
            answer(&self.latest_release, "get_latest_release")
        }
        fn list_mcp_catalog(&self) -> PlatformResult<Vec<JsonObject>> {
            self.record("list_mcp_catalog", vec![]);
            answer(&self.list_mcp_catalog, "list_mcp_catalog")
        }
        fn get_current_user(&self) -> PlatformResult<JsonObject> {
            self.record("get_current_user", vec![]);
            answer(&self.current_user, "get_current_user")
        }
        fn update_current_user(&self, payload: &JsonObject) -> PlatformResult<JsonObject> {
            self.record("update_current_user", vec![Value::Object(payload.clone())]);
            answer(&self.update_current_user, "update_current_user")
        }
        fn verify_apple_transaction(&self, transaction_id: &str) -> PlatformResult<JsonObject> {
            self.record("verify_apple_transaction", vec![Value::from(transaction_id)]);
            answer(&self.verify_apple_transaction, "verify_apple_transaction")
        }
        fn create_checkout_session(&self) -> PlatformResult<JsonObject> {
            self.record("create_checkout_session", vec![]);
            answer(&self.checkout_session, "create_checkout_session")
        }
    }

    /// The service override is process-global, so fixtures run one at a time.
    static PLATFORM_OVERRIDE_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    /// The installed fake; the override is removed when this drops
    /// (`app.dependency_overrides.pop(get_platform_service)`).
    pub struct PlatformFixture {
        pub fake: Arc<FakePlatform>,
        pub service: Arc<PlatformService>,
        _creds: CredentialStoreGuard,
        _lock: MutexGuard<'static, ()>,
    }

    impl Drop for PlatformFixture {
        fn drop(&mut self) {
            set_platform_service_override(None);
        }
    }

    impl std::ops::Deref for PlatformFixture {
        type Target = FakePlatform;
        fn deref(&self) -> &FakePlatform {
            &self.fake
        }
    }

    /// The `platform_service` fixture: a pure-proxy service (no cache DB)
    /// over a fresh [`FakePlatform`].
    pub fn platform_service() -> PlatformFixture {
        platform_service_with_db(None)
    }

    /// [`platform_service`] backed by a local cache DB, for asserting what is
    /// (and is not) re-fetched.
    pub fn platform_service_with_db(db: Option<Arc<WorkspaceDB>>) -> PlatformFixture {
        let lock = PLATFORM_OVERRIDE_LOCK.lock();
        let creds = isolate_credential_store();
        let fake = Arc::new(FakePlatform::default());
        let service = Arc::new(PlatformService::new(fake.clone(), db));
        set_platform_service_override(Some(service.clone()));
        PlatformFixture { fake, service, _creds: creds, _lock: lock }
    }

    /// The decoded response of [`send`].
    #[derive(Debug)]
    pub struct TestResponse {
        pub status: StatusCode,
        pub body: Value,
    }

    impl TestResponse {
        pub fn status_code(&self) -> u16 {
            self.status.as_u16()
        }

        /// `resp.json()`.
        pub fn json(&self) -> &Value {
            &self.body
        }
    }

    /// Send one request through the full app (`TestClient(app)`), with an
    /// optional JSON body; a non-JSON response body decodes as a string.
    pub async fn send(method: Method, uri: &str, json: Option<Value>) -> TestResponse {
        let builder = Request::builder().method(method).uri(uri);
        let request = match json {
            Some(body) => builder
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
            None => builder.body(Body::empty()).unwrap(),
        };
        let response = smartloop::api::app().oneshot(request).await.expect("the app answers");
        let status = response.status();
        let bytes = response.into_body().collect().await.expect("a readable body").to_bytes();
        let body = serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
        TestResponse { status, body }
    }

    /// `client.get(uri)`.
    pub async fn get(uri: &str) -> TestResponse {
        send(Method::GET, uri, None).await
    }

    /// `client.post(uri, json=body)`.
    pub async fn post(uri: &str, body: Value) -> TestResponse {
        send(Method::POST, uri, Some(body)).await
    }

    /// `client.patch(uri, json=body)`.
    pub async fn patch(uri: &str, body: Value) -> TestResponse {
        send(Method::PATCH, uri, Some(body)).await
    }
}
// ======================== END unit: core =========================

// ===================== BEGIN unit: workspace =====================
/// API test support used by the workspace unit's API tests: the conftest
/// `no_real_downloads` autouse guard, the `platform_service` stand-in for the
/// `MagicMock` dependency override, and an in-process `TestClient` driving the
/// axum router with `tower::ServiceExt::oneshot`.
pub mod workspace_api {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use axum::body::Body;
    use http::{Method, Request, StatusCode};
    use parking_lot::Mutex;
    use serde_json::Value;
    use tower::ServiceExt;

    use smartloop::model::downloader::slp_downloader::{
        DownloadStubCall, DownloadStubs, set_download_stubs,
    };
    use smartloop::platform::client::PlatformApi;
    use smartloop::platform::service::PlatformService;
    use smartloop::platform::set_platform_service_override;

    /// Files a real download leaves behind, complete or interrupted.
    const WEIGHT_SUFFIXES: &[&str] = &["gguf", "part"];

    /// `no_real_downloads`: stubs every path that would fetch weights for the
    /// guard's lifetime, records what was asked for, and on drop proves no
    /// weight file landed under `tmp_path`.
    ///
    /// The stubs stand in for a *successful* download, so a route that goes
    /// on to register the model still behaves as it would in life; what they
    /// do not do is touch the network or the disk.
    pub struct NoRealDownloads {
        #[allow(dead_code)]
        pub calls: Arc<Mutex<Vec<(String, Option<String>)>>>,
        tmp_path: PathBuf,
    }

    impl NoRealDownloads {
        pub fn install(tmp_path: &Path) -> Self {
            let calls: Arc<Mutex<Vec<(String, Option<String>)>>> = Arc::new(Mutex::new(Vec::new()));
            let fallback = tmp_path.to_path_buf();
            let download = {
                let calls = Arc::clone(&calls);
                let fallback = fallback.clone();
                Arc::new(move |call: DownloadStubCall| match call {
                    DownloadStubCall::Download { model_id, models_dir } => {
                        calls.lock().push(("download".to_string(), Some(model_id)));
                        Ok(models_dir.unwrap_or_else(|| fallback.clone()))
                    }
                    DownloadStubCall::BootstrapDownload { model_name, models_dir } => {
                        calls.lock().push(("bootstrap_download".to_string(), model_name));
                        Ok(models_dir.unwrap_or_else(|| fallback.clone()))
                    }
                })
            };
            set_download_stubs(Some(DownloadStubs {
                download: Some(download.clone()),
                bootstrap_download: Some(download),
                resolve_artifacts: None,
            }));
            Self { calls, tmp_path: tmp_path.to_path_buf() }
        }
    }

    impl Drop for NoRealDownloads {
        fn drop(&mut self) {
            set_download_stubs(None);
            if std::thread::panicking() {
                return;
            }
            // The stub can stop matching — a route that starts building its
            // downloader somewhere new, or a rename — and the only symptom
            // would be the disk filling again.  Assert the outcome.
            let leaked: Vec<String> = walkdir::WalkDir::new(&self.tmp_path)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
                .filter(|e| {
                    e.path()
                        .extension()
                        .and_then(|x| x.to_str())
                        .is_some_and(|x| WEIGHT_SUFFIXES.contains(&x))
                })
                .map(|e| e.path().to_string_lossy().into_owned())
                .collect();
            assert!(
                leaked.is_empty(),
                "a real model download escaped the stub and wrote {leaked:?} — check whether a new \
                 code path builds its own downloader"
            );
        }
    }

    /// The `MagicMock()` platform service: every platform call falls through
    /// to the trait's defaults, and nothing reaches a DB or the network.
    pub struct MockPlatformApi;

    impl PlatformApi for MockPlatformApi {}

    /// `platform_service` + `client` fixtures: binds a mock platform service
    /// over `get_platform_service` and removes it on drop.
    pub struct PlatformServiceOverride {
        #[allow(dead_code)]
        pub service: Arc<PlatformService>,
    }

    impl PlatformServiceOverride {
        pub fn install() -> Self {
            let service = Arc::new(PlatformService::new(Arc::new(MockPlatformApi), None));
            set_platform_service_override(Some(Arc::clone(&service)));
            Self { service }
        }
    }

    impl Drop for PlatformServiceOverride {
        fn drop(&mut self) {
            set_platform_service_override(None);
        }
    }

    /// A response: status plus the parsed JSON body (`Value::Null` when the
    /// body is empty or not JSON).
    #[derive(Debug, Clone)]
    pub struct TestResponse {
        pub status: StatusCode,
        pub body: Value,
    }

    impl TestResponse {
        pub fn status_code(&self) -> u16 {
            self.status.as_u16()
        }

        pub fn json(&self) -> &Value {
            &self.body
        }
    }

    /// FastAPI `TestClient` stand-in — in-process, no real server socket.
    pub struct TestClient {
        router: axum::Router,
    }

    impl TestClient {
        pub fn new() -> Self {
            Self { router: smartloop::api::main::app() }
        }

        pub async fn request(&self, method: Method, uri: &str, body: Option<&Value>) -> TestResponse {
            let mut builder = Request::builder().method(method).uri(uri);
            let body = match body {
                Some(json) => {
                    builder = builder.header(http::header::CONTENT_TYPE, "application/json");
                    Body::from(serde_json::to_vec(json).unwrap())
                }
                None => Body::empty(),
            };
            let response = self.router.clone().oneshot(builder.body(body).unwrap()).await.unwrap();
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
            let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
            TestResponse { status, body }
        }

        pub async fn get(&self, uri: &str) -> TestResponse {
            self.request(Method::GET, uri, None).await
        }

        pub async fn patch(&self, uri: &str, json: Value) -> TestResponse {
            self.request(Method::PATCH, uri, Some(&json)).await
        }
    }

    impl Default for TestClient {
        fn default() -> Self {
            Self::new()
        }
    }
}
// ====================== END unit: workspace ======================

// ======================= BEGIN unit: model =======================
/// API test fixtures.
///
/// Nothing here may reach the network, and one guarantee needs stating
/// outright: **no test downloads a model.**  The routers under test build
/// their own `SLPArtifactDownloader` rather than receiving one, and `/v1/init`
/// hands the transfer to a blocking worker whose thread outlives the response.
/// A test that closed the stream to avoid "running a real download" therefore
/// left one running: the weights kept arriving, into a temp dir that would
/// later be deleted, long after the assertion had passed.  A full run wrote
/// tens of gigabytes of real GGUFs this way and filled the disk.
///
/// So the download entry points are stubbed for every test in this package,
/// process-wide rather than at any one call site — three call sites build a
/// downloader (`/v1/init`, `/v1/bootstrap`, and `GenericModel::init` by way of
/// `ModelFactory`), and a stub attached to one of them silently misses the
/// others.  Every API test holds a [`NoRealDownloads`] guard (Python's autouse
/// `no_real_downloads` fixture) and, because the stubs are process-global, is
/// `#[serial]`.
#[allow(dead_code)]
pub mod model_support {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use parking_lot::Mutex;
    use smartloop::model::downloader::artifact_source::ArtifactMap;
    use smartloop::model::downloader::slp_downloader::{
        DownloadStubCall, DownloadStubFn, DownloadStubs, ResolveStubFn, set_download_stubs,
    };

    /// Files a real download leaves behind, complete or interrupted.
    pub const WEIGHT_SUFFIXES: [&str; 2] = ["gguf", "part"];

    /// One intercepted download: `("download", model_id)` or
    /// `("bootstrap_download", model_name)`.
    pub type DownloadCall = (String, Option<String>);

    /// Stub every path that would fetch weights, and prove none was taken.
    ///
    /// Un-opt-out-able by design: a download is never what an API test is
    /// asserting, and the failure mode when one slips through is silent — the test
    /// passes, and the machine fills up minutes later from a thread nobody is
    /// watching.
    ///
    /// The stubs stand in for a *successful* download, so a route that goes on to
    /// register the model still behaves as it would in life.  What they do not do
    /// is touch the network or the disk.
    pub struct NoRealDownloads {
        tmp: tempfile::TempDir,
        calls: Arc<Mutex<Vec<DownloadCall>>>,
        stubs: Mutex<DownloadStubs>,
    }

    impl NoRealDownloads {
        /// Install the stubs (Python's `no_real_downloads` fixture).
        pub fn new() -> Self {
            let tmp = tempfile::tempdir().expect("temp dir");
            let calls: Arc<Mutex<Vec<DownloadCall>>> = Arc::new(Mutex::new(Vec::new()));
            let tmp_path = tmp.path().to_path_buf();

            let download_calls = Arc::clone(&calls);
            let download_root = tmp_path.clone();
            let fake_download: DownloadStubFn = Arc::new(move |call: DownloadStubCall| {
                match call {
                    DownloadStubCall::Download { model_id, models_dir } => {
                        download_calls.lock().push(("download".to_string(), Some(model_id)));
                        Ok(models_dir.unwrap_or_else(|| download_root.clone()))
                    }
                    DownloadStubCall::BootstrapDownload { model_name, models_dir } => {
                        download_calls.lock().push(("bootstrap_download".to_string(), model_name));
                        Ok(models_dir.unwrap_or_else(|| download_root.clone()))
                    }
                }
            });

            let stubs = DownloadStubs {
                download: Some(Arc::clone(&fake_download)),
                bootstrap_download: Some(fake_download),
                resolve_artifacts: None,
            };
            set_download_stubs(Some(stubs.clone()));
            NoRealDownloads { tmp, calls, stubs: Mutex::new(stubs) }
        }

        /// The temp directory the stubs report as the download location
        /// (`tmp_path`), also the tree checked for leaked weights on drop.
        pub fn tmp_path(&self) -> &Path {
            self.tmp.path()
        }

        /// Every download the stubs intercepted, in order.
        pub fn calls(&self) -> Vec<DownloadCall> {
            self.calls.lock().clone()
        }

        /// Replace the `download` stub (e.g. with one that raises
        /// `DownloadCancelled`), as a test's own `monkeypatch.setattr` would.
        pub fn set_download(&self, stub: DownloadStubFn) {
            let mut stubs = self.stubs.lock();
            stubs.download = Some(stub);
            set_download_stubs(Some(stubs.clone()));
        }

        /// Replace `resolve_artifacts` with `stub`.
        pub fn set_resolve_artifacts(&self, stub: ResolveStubFn) {
            let mut stubs = self.stubs.lock();
            stubs.resolve_artifacts = Some(stub);
            set_download_stubs(Some(stubs.clone()));
        }

        /// `resolve_artifacts` answering a fixed map.
        pub fn resolve_to(&self, artifacts: ArtifactMap) {
            self.set_resolve_artifacts(Arc::new(move |_model_id: &str| Ok(artifacts.clone())));
        }

        /// Weight files under the temp dir — there must be none.
        pub fn leaked(&self) -> Vec<PathBuf> {
            walkdir::WalkDir::new(self.tmp.path())
                .into_iter()
                .flatten()
                .filter(|entry| entry.file_type().is_file())
                .map(|entry| entry.into_path())
                .filter(|path| {
                    path.extension().and_then(|e| e.to_str()).is_some_and(|ext| WEIGHT_SUFFIXES.contains(&ext))
                })
                .collect()
        }
    }

    impl Default for NoRealDownloads {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Drop for NoRealDownloads {
        fn drop(&mut self) {
            set_download_stubs(None);
            // The stub can stop matching — a route that starts building its
            // downloader somewhere new, or a rename — and the only symptom would
            // be the disk filling again.  Assert the outcome rather than trusting
            // the patch.
            let leaked = self.leaked();
            if !std::thread::panicking() {
                assert!(
                    leaked.is_empty(),
                    "a real model download escaped the stub and wrote {leaked:?} — check whether a new code path \
                     builds its own downloader"
                );
            }
        }
    }

    /// Install the download stubs for the duration of a test.
    pub fn no_real_downloads() -> NoRealDownloads {
        NoRealDownloads::new()
    }
}
// ======================== END unit: model ========================

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

// ===== BEGIN api_routes_misc =====================================================

/// An in-process HTTP client over an axum `Router` — the stand-in for
/// FastAPI's `TestClient`: no socket, every request goes through
/// `tower::ServiceExt::oneshot`, and redirects are never followed.
#[allow(dead_code)]
pub mod misc_client {
    use axum::body::{Body, Bytes};
    use axum::http::{HeaderMap, Method, Request};
    use axum::Router;
    use http_body_util::BodyExt;
    use serde_json::Value;
    use tower::ServiceExt;

    /// What a request answered.
    #[derive(Debug, Clone)]
    pub struct TestResponse {
        pub status: u16,
        pub headers: HeaderMap,
        pub body: Bytes,
    }

    impl TestResponse {
        /// The body parsed as JSON (`resp.json()`).
        pub fn json(&self) -> Value {
            serde_json::from_slice(&self.body)
                .unwrap_or_else(|e| panic!("response is not JSON ({e}): {:?}", self.text()))
        }

        /// The body as text (`resp.text`).
        pub fn text(&self) -> String {
            String::from_utf8_lossy(&self.body).into_owned()
        }

        /// A response header, if present.
        pub fn header(&self, name: &str) -> Option<String> {
            self.headers.get(name).and_then(|v| v.to_str().ok()).map(str::to_owned)
        }
    }

    /// Send one request to *router*.
    pub async fn request(
        router: &Router,
        method: Method,
        uri: &str,
        content_type: Option<&str>,
        body: Body,
    ) -> TestResponse {
        let mut builder = Request::builder().method(method).uri(uri);
        if let Some(ct) = content_type {
            builder = builder.header("content-type", ct);
        }
        let request = builder.body(body).expect("build request");
        let response = router.clone().oneshot(request).await.expect("router is infallible");
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        let body = response.into_body().collect().await.expect("read body").to_bytes();
        TestResponse { status, headers, body }
    }

    /// `client.get(uri)`.
    pub async fn get(router: &Router, uri: &str) -> TestResponse {
        request(router, Method::GET, uri, None, Body::empty()).await
    }

    /// `client.delete(uri)`.
    pub async fn delete(router: &Router, uri: &str) -> TestResponse {
        request(router, Method::DELETE, uri, None, Body::empty()).await
    }

    /// `client.post(uri, json=body)`.
    pub async fn post_json(router: &Router, uri: &str, body: &Value) -> TestResponse {
        let bytes = serde_json::to_vec(body).expect("serialize body");
        request(router, Method::POST, uri, Some("application/json"), Body::from(bytes)).await
    }

    /// `client.patch(uri, json=body)`.
    pub async fn patch_json(router: &Router, uri: &str, body: &Value) -> TestResponse {
        let bytes = serde_json::to_vec(body).expect("serialize body");
        request(router, Method::PATCH, uri, Some("application/json"), Body::from(bytes)).await
    }

    /// `/path?k=v&...` with the parameters URL-encoded (`params=`).
    pub fn with_query(path: &str, params: &[(&str, &str)]) -> String {
        let query = url::form_urlencoded::Serializer::new(String::new())
            .extend_pairs(params.iter())
            .finish();
        if query.is_empty() { path.to_string() } else { format!("{path}?{query}") }
    }

    /// The `data: ` payloads of an SSE body, parsed as JSON.
    pub fn sse_frames(text: &str) -> Vec<Value> {
        text.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|data| serde_json::from_str(data).ok())
            .collect()
    }
}

// ===== END api_routes_misc =======================================================
