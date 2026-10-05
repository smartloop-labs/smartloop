//! Port of `tests/api/conftest.py`.
//!
//! API test fixtures.
//!
//! Nothing here may reach the network, and one guarantee needs stating
//! outright: **no test downloads a model.**  The routers under test build
//! their own `SLPArtifactDownloader` rather than receiving one, and `/v1/init`
//! hands the transfer to a blocking worker whose thread outlives the response.
//! A test that closed the stream to avoid "running a real download" therefore
//! left one running: the weights kept arriving, into a temp dir that would
//! later be deleted, long after the assertion had passed.  A full run wrote
//! tens of gigabytes of real GGUFs this way and filled the disk.
//!
//! So the download entry points are stubbed for every test in this package,
//! process-wide rather than at any one call site — three call sites build a
//! downloader (`/v1/init`, `/v1/bootstrap`, and `GenericModel::init` by way of
//! `ModelFactory`), and a stub attached to one of them silently misses the
//! others.  Every API test holds a [`NoRealDownloads`] guard (Python's autouse
//! `no_real_downloads` fixture) and, because the stubs are process-global, is
//! `#[serial]`.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderMap, Method, Request, StatusCode};
use parking_lot::Mutex;
use serde_json::Value;
use smartloop::model::downloader::artifact_source::ArtifactMap;
use smartloop::model::downloader::slp_downloader::{
    DownloadStubCall, DownloadStubFn, DownloadStubs, ResolveStubFn, set_download_stubs,
};
use smartloop::platform::client::{JsonObject, PlatformApi, PlatformError, PlatformResult};
use smartloop::platform::{PlatformService, set_platform_service_override};
use tower::ServiceExt;

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

// ------------------------------------------------------------ platform_service

/// A configured answer for one `PlatformApi` method.
type Answer<T> = Mutex<Option<Box<dyn Fn() -> PlatformResult<T> + Send + Sync>>>;

/// The `MagicMock` standing in for the platform: tests configure the answers
/// (`return_value` / `side_effect`) of `get_current_user`,
/// `update_current_user` and `list_models`, and read back the calls.  An
/// unconfigured method fails loudly rather than reaching the network.
#[derive(Default)]
pub struct MockPlatformApi {
    get_current_user: Answer<JsonObject>,
    update_current_user: Mutex<Option<Box<dyn Fn(&JsonObject) -> PlatformResult<JsonObject> + Send + Sync>>>,
    list_models: Answer<Vec<JsonObject>>,
    /// `(method, argument)` for every call made.
    pub calls: Mutex<Vec<(String, Value)>>,
}

fn unconfigured(op: &str) -> PlatformError {
    PlatformError::Other(format!("MockPlatformApi.{op} was called but not configured"))
}

impl MockPlatformApi {
    /// `get_current_user.return_value` / `.side_effect`.
    pub fn on_get_current_user(&self, f: impl Fn() -> PlatformResult<JsonObject> + Send + Sync + 'static) {
        *self.get_current_user.lock() = Some(Box::new(f));
    }

    /// `update_current_user.return_value` / `.side_effect`.
    pub fn on_update_current_user(
        &self,
        f: impl Fn(&JsonObject) -> PlatformResult<JsonObject> + Send + Sync + 'static,
    ) {
        *self.update_current_user.lock() = Some(Box::new(f));
    }

    /// `list_models.return_value` / `.side_effect`.
    pub fn on_list_models(&self, f: impl Fn() -> PlatformResult<Vec<JsonObject>> + Send + Sync + 'static) {
        *self.list_models.lock() = Some(Box::new(f));
    }

    /// The recorded calls of `method`.
    pub fn calls_of(&self, method: &str) -> Vec<Value> {
        self.calls.lock().iter().filter(|(m, _)| m == method).map(|(_, arg)| arg.clone()).collect()
    }

    fn record(&self, method: &str, arg: Value) {
        self.calls.lock().push((method.to_string(), arg));
    }
}

impl PlatformApi for MockPlatformApi {
    fn get_current_user(&self) -> PlatformResult<JsonObject> {
        self.record("get_current_user", Value::Null);
        match self.get_current_user.lock().as_ref() {
            Some(f) => f(),
            None => Err(unconfigured("get_current_user")),
        }
    }

    fn update_current_user(&self, payload: &JsonObject) -> PlatformResult<JsonObject> {
        self.record("update_current_user", Value::Object(payload.clone()));
        match self.update_current_user.lock().as_ref() {
            Some(f) => f(payload),
            None => Err(unconfigured("update_current_user")),
        }
    }

    fn list_models(&self) -> PlatformResult<Vec<JsonObject>> {
        self.record("list_models", Value::Null);
        match self.list_models.lock().as_ref() {
            Some(f) => f(),
            None => Err(unconfigured("list_models")),
        }
    }
}

/// A mock `PlatformService` bound in place of `get_platform_service` (the
/// FastAPI dependency override); unbound on drop.  No real DB or HTTP I/O: the
/// service runs as a pure proxy over [`MockPlatformApi`].
pub struct PlatformServiceFixture {
    pub api: Arc<MockPlatformApi>,
    pub service: Arc<PlatformService>,
}

impl PlatformServiceFixture {
    /// Bind a fresh mock.
    pub fn new() -> Self {
        let api = Arc::new(MockPlatformApi::default());
        let service = Arc::new(PlatformService::new(api.clone(), None));
        set_platform_service_override(Some(Arc::clone(&service)));
        PlatformServiceFixture { api, service }
    }
}

impl Default for PlatformServiceFixture {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PlatformServiceFixture {
    fn drop(&mut self) {
        set_platform_service_override(None);
    }
}

/// The `platform_service` fixture.
pub fn platform_service() -> PlatformServiceFixture {
    PlatformServiceFixture::new()
}

// ----------------------------------------------------------------------- client

/// A response read in full (`TestClient`'s `Response`).
#[derive(Debug, Clone)]
pub struct TestResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

impl TestResponse {
    /// `resp.status_code`.
    pub fn status_code(&self) -> u16 {
        self.status.as_u16()
    }

    /// `resp.text`.
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    /// `resp.json()`.
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body)
            .unwrap_or_else(|e| panic!("response is not JSON ({e}): {}", self.text()))
    }
}

/// The in-process test client (FastAPI `TestClient`): requests go through
/// `tower::ServiceExt::oneshot` on the app's router — no real socket.
#[derive(Clone)]
pub struct TestClient {
    pub app: Router,
}

impl TestClient {
    /// A client over `app`.
    pub fn new(app: Router) -> Self {
        TestClient { app }
    }

    /// Send a request with an optional JSON body and read the whole response.
    pub async fn request(&self, method: Method, uri: &str, json: Option<&Value>) -> TestResponse {
        let mut builder = Request::builder().method(method).uri(uri);
        let body = match json {
            Some(json) => {
                builder = builder.header("content-type", "application/json");
                Body::from(serde_json::to_vec(json).expect("serializable body"))
            }
            None => Body::empty(),
        };
        let response = self
            .app
            .clone()
            .oneshot(builder.body(body).expect("valid request"))
            .await
            .expect("router is infallible");
        let status = response.status();
        let headers = response.headers().clone();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.expect("readable body").to_vec();
        TestResponse { status, headers, body }
    }

    /// `client.get(uri)`.
    pub async fn get(&self, uri: &str) -> TestResponse {
        self.request(Method::GET, uri, None).await
    }

    /// `client.post(uri, json=...)`.
    pub async fn post(&self, uri: &str, json: &Value) -> TestResponse {
        self.request(Method::POST, uri, Some(json)).await
    }

    /// `client.patch(uri, json=...)`.
    pub async fn patch(&self, uri: &str, json: &Value) -> TestResponse {
        self.request(Method::PATCH, uri, Some(json)).await
    }

    /// `client.put(uri, json=...)`.
    pub async fn put(&self, uri: &str, json: &Value) -> TestResponse {
        self.request(Method::PUT, uri, Some(json)).await
    }

    /// `client.delete(uri)`.
    pub async fn delete(&self, uri: &str) -> TestResponse {
        self.request(Method::DELETE, uri, None).await
    }
}

/// The `client` fixture: the FastAPI app in-process, with a mock platform
/// service bound for as long as the returned fixture lives.
pub fn client() -> (TestClient, PlatformServiceFixture) {
    let platform = platform_service();
    (TestClient::new(smartloop::api::main::app()), platform)
}
