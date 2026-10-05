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
