//! Port of `tests/api/conftest.py`.

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
