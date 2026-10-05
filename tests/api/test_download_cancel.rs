//! Port of `tests/api/test_download_cancel.py`.
//!
//! Stopping a download from a second request, and what the stream then says.
//!
//! Two halves: `DELETE /v1/downloads/{name}` finding the running transfer, and
//! `/v1/init` answering the cancel with a terminal `cancelled` frame instead of
//! an error — and, more importantly, without any of the nine things it does
//! after a download completes.
//!
//! The Python tests bind a `MagicMock` model service through a FastAPI
//! dependency override.  Here the routes are driven through their explicit
//! service parameter (`cancel_download(.., model_service)`,
//! `init_with(.., model_service)`) with a real, bare `ModelService` whose model
//! sits on a temporary workspace, and the mock's "was it called" assertions
//! become checks on that workspace.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Method, Request};
use futures::StreamExt;
use parking_lot::RwLock;
use serde_json::{json, Value};
use serial_test::serial;
use smartloop::api::routers::downloads;
use smartloop::api::routers::init::{self as init_router_module, InitRequest};
use smartloop::api::services::ModelService;
use smartloop::api::services::model_service::ModelServiceError;
use smartloop::model::downloader::artifact_downloader::SLPArtifactDownloader;
use smartloop::model::downloader::artifact_source::ArtifactMap;
use smartloop::model::downloader::cancellation::{self, DownloadCancelled, Event};
use smartloop::model::downloader::slp_downloader::{DownloadStubCall, SharedLogger};
use smartloop::model::model_manager::Model;
use smartloop::utils::log_utils::LoggerLike;
use smartloop::workspace::Workspace;
use tower::ServiceExt;

use crate::common::model_support::{NoRealDownloads, no_real_downloads};

const MODEL_NAME: &str = "gemma3-4b";
const MODEL_ID: &str = "google/gemma-3-4b-it";

struct QuietLogger;

impl LoggerLike for QuietLogger {
    fn info(&self, msg: &str) {
        eprintln!("INFO {msg}");
    }
    fn warning(&self, msg: &str) {
        eprintln!("WARNING {msg}");
    }
    fn debug(&self, msg: &str) {
        eprintln!("DEBUG {msg}");
    }
    fn error(&self, msg: &str) {
        eprintln!("ERROR {msg}");
    }
}

/// The `_clean_registry` autouse fixture.
struct CleanRegistry;

impl CleanRegistry {
    fn new() -> Self {
        cancellation::reset_for_tests();
        CleanRegistry
    }
}

impl Drop for CleanRegistry {
    fn drop(&mut self) {
        cancellation::reset_for_tests();
    }
}

/// Everything one test runs against: an isolated `SLP_HOME`, the download
/// stubs, a cleared cancel registry, and a model service over a temporary
/// workspace (the `model_service` fixture).
struct Fixture {
    downloads: NoRealDownloads,
    service: Arc<ModelService>,
    workspace: Arc<RwLock<Workspace>>,
    models_dir: PathBuf,
    previous_home: Option<String>,
    _registry: CleanRegistry,
    _home: tempfile::TempDir,
}

impl Fixture {
    /// `register` puts `MODEL_NAME` in the workspace record (the mock's
    /// `get_model.return_value = _workspace_model()`); `false` is
    /// `get_model.return_value = None`.
    fn new(register: bool) -> Self {
        let registry = CleanRegistry::new();
        let downloads = no_real_downloads();
        let home = tempfile::tempdir().unwrap();
        let previous_home = std::env::var("SLP_HOME").ok();
        // SAFETY: every test that touches the environment is `#[serial]`.
        unsafe { std::env::set_var("SLP_HOME", home.path()) };
        Workspace::reset();

        let workspace_dir = home.path().join("workspace-test");
        let models_dir = workspace_dir.join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        let mut workspace = Workspace::new("workspace-test");
        workspace.workspace_dir = workspace_dir.to_string_lossy().into_owned();
        workspace.models_dir = models_dir.to_string_lossy().into_owned();
        if register {
            workspace.add_model(MODEL_NAME, MODEL_ID, None, true);
        }
        let workspace = Arc::new(RwLock::new(workspace));

        let service = ModelService::new_bare();
        *service.model.write() = Some(Arc::new(Model::new("model-test", Arc::clone(&workspace), None)));

        Fixture {
            downloads,
            service,
            workspace,
            models_dir,
            previous_home,
            _registry: registry,
            _home: home,
        }
    }

    /// The `cancelled_download` fixture: the transfer raises
    /// `DownloadCancelled`, and the signed URLs resolve without the platform.
    fn cancelled_download(&self) {
        self.downloads.set_download(Arc::new(|_call: DownloadStubCall| {
            Err(DownloadCancelled::new("download cancelled").into())
        }));
        let mut artifacts = ArtifactMap::new();
        artifacts.insert("weights.gguf".to_string(), "https://x/f".to_string());
        self.downloads.resolve_to(artifacts);
    }

    /// `client.delete(f"/v1/downloads/{name}?wait_ms=...")` against this
    /// fixture's service: `(status, body)`.
    async fn delete(&self, model_name: &str, wait_ms: i64) -> (u16, Value) {
        match downloads::cancel_download(model_name, wait_ms, Arc::clone(&self.service)).await {
            Ok(body) => (200, body),
            Err(exc) => (exc.status_code, json!({"detail": exc.detail})),
        }
    }

    /// `client.post("/v1/init", json={"model_name": MODEL_NAME})`: status and
    /// SSE body.
    async fn init(&self) -> (u16, String) {
        let request = InitRequest { model_name: Some(MODEL_NAME.to_string()), ..Default::default() };
        match init_router_module::init_with(request, Arc::clone(&self.service)).await {
            Ok(response) => {
                let status = response.status().as_u16();
                let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
                (status, String::from_utf8_lossy(&body).into_owned())
            }
            Err(exc) => (exc.status_code, String::new()),
        }
    }

    /// A partial download on disk, as a cancelled transfer leaves it.
    fn partial_download(&self) -> PathBuf {
        let model_dir = self.models_dir.join(MODEL_ID);
        std::fs::create_dir_all(&model_dir).unwrap();
        model_dir
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        Workspace::reset();
        // SAFETY: every test that touches the environment is `#[serial]`.
        unsafe {
            match &self.previous_home {
                Some(value) => std::env::set_var("SLP_HOME", value),
                None => std::env::remove_var("SLP_HOME"),
            }
        }
    }
}

/// Parse an SSE body into `[(event, data), …]`.
fn frames(text: &str) -> Vec<(String, Value)> {
    let mut out = Vec::new();
    let mut event = "message".to_string();
    for line in text.lines() {
        if let Some(name) = line.strip_prefix("event: ") {
            event = name.to_string();
        } else if let Some(data) = line.strip_prefix("data: ") {
            out.push((event.clone(), serde_json::from_str(data).unwrap()));
            event = "message".to_string();
        }
    }
    out
}

fn event_names(text: &str) -> Vec<String> {
    frames(text).into_iter().map(|(name, _)| name).collect()
}

/// `threading.Timer(delay, f).start()`.
fn timer(delay: Duration, f: impl FnOnce() + Send + 'static) {
    std::thread::spawn(move || {
        std::thread::sleep(delay);
        f();
    });
}

// ------------------------------------------------------- TestResolvingTheModel

/// `TestResolvingTheModel`
mod test_resolving_the_model {
    use super::*;

    #[tokio::test]
    #[serial]
    async fn test_it_uses_the_workspace_record() {
        let fx = Fixture::new(true);
        cancellation::begin(MODEL_ID).unwrap();
        let (status, body) = fx.delete(MODEL_NAME, 0).await;

        assert_eq!(status, 200);
        assert_eq!(body["model_id"], MODEL_ID);
    }

    /// A cancel can land before `/v1/init` has added the workspace entry.
    #[tokio::test]
    #[serial]
    async fn test_it_falls_back_to_the_catalog_before_registration() {
        let fx = Fixture::new(false);

        let (status, body) = fx.delete("sl-mini", 0).await;

        assert_eq!(status, 200);
        assert!(body["model_id"].as_str().is_some_and(|id| !id.is_empty()));
    }

    #[tokio::test]
    #[serial]
    async fn test_an_unknown_model_is_a_400() {
        let fx = Fixture::new(false);

        let (status, body) = fx.delete("not-a-model", 0).await;

        assert_eq!(status, 400);
        assert!(body["detail"].as_str().unwrap().contains("not-a-model"));
    }

    #[tokio::test]
    #[serial]
    async fn test_a_workspace_failure_is_a_503() {
        let fx = Fixture::new(true);
        fx.service.hooks.write().ensure_workspace = Some(Arc::new(|_config| {
            Err(ModelServiceError::runtime("Workspace not initialized"))
        }));

        let (status, _body) = fx.delete(MODEL_NAME, 0).await;

        assert_eq!(status, 503);
    }
}

// --------------------------------------------------------------- TestCancelling

/// `TestCancelling`
mod test_cancelling {
    use super::*;

    #[tokio::test]
    #[serial]
    async fn test_an_active_download_is_told_to_stop() {
        let fx = Fixture::new(true);
        let token = cancellation::begin(MODEL_ID).unwrap();

        let (status, body) = fx.delete(MODEL_NAME, 0).await;

        assert_eq!(status, 200);
        assert_eq!(
            body,
            json!({"cancelled": true, "status": "cancelling", "model_name": MODEL_NAME, "model_id": MODEL_ID})
        );
        assert!(token.cancelled());
    }

    /// The client races by construction: a progress frame arrives, the user
    /// clicks, the transfer finishes before the request lands.  A 404 would
    /// push the picker into an error path for an ordinary outcome.
    #[tokio::test]
    #[serial]
    async fn test_nothing_running_is_a_200_not_a_404() {
        let fx = Fixture::new(true);
        let (status, body) = fx.delete(MODEL_NAME, 0).await;

        assert_eq!(status, 200);
        assert_eq!(body["cancelled"], false);
        assert_eq!(body["status"], "not_found");
    }

    #[tokio::test]
    #[serial]
    async fn test_a_second_cancel_is_idempotent() {
        let fx = Fixture::new(true);
        cancellation::begin(MODEL_ID).unwrap();

        let first = fx.delete(MODEL_NAME, 0).await;
        let second = fx.delete(MODEL_NAME, 0).await;

        assert_eq!(first, second);
    }

    /// Cleanup belongs to the download thread — the only one that knows every
    /// worker has stopped.  Unlinking here would race an open write.
    #[tokio::test]
    #[serial]
    async fn test_it_deletes_nothing_itself() {
        let fx = Fixture::new(true);
        let model_dir = fx.partial_download();
        cancellation::begin(MODEL_ID).unwrap();

        fx.delete(MODEL_NAME, 0).await;

        // `discard_partial_download` was not called: the partial tree is
        // exactly where the transfer left it.
        assert!(model_dir.is_dir());
    }

    /// A stalled socket can take 60s to die; the route must not wait.
    #[tokio::test]
    #[serial]
    async fn test_it_returns_before_the_threads_stop() {
        let fx = Fixture::new(true);
        cancellation::begin(MODEL_ID).unwrap();

        let (_status, body) = fx.delete(MODEL_NAME, 0).await;

        assert_eq!(body["status"], "cancelling"); // not "stopped"
    }

    #[tokio::test]
    #[serial]
    async fn test_it_cancels_the_model_the_caller_named() {
        let fx = Fixture::new(true);
        let other = cancellation::begin("google/gemma-4-e4b-it").unwrap();
        let mine = cancellation::begin(MODEL_ID).unwrap();

        fx.delete(MODEL_NAME, 0).await;

        assert!(mine.cancelled());
        assert!(!other.cancelled());
    }
}

// ------------------------------------------------------ TestWaitingForItToStop

/// `TestWaitingForItToStop`
mod test_waiting_for_it_to_stop {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_it_reports_stopped_once_the_bytes_are_gone() {
        let fx = Fixture::new(true);
        let token = cancellation::begin(MODEL_ID).unwrap();
        timer(Duration::from_millis(50), move || cancellation::end(MODEL_ID, &token));

        let (_status, body) = fx.delete(MODEL_NAME, 5000).await;

        assert_eq!(body["status"], "stopped");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_it_still_reports_cancelling_when_the_wait_runs_out() {
        let fx = Fixture::new(true);
        cancellation::begin(MODEL_ID).unwrap();

        let (_status, body) = fx.delete(MODEL_NAME, 50).await;

        assert_eq!(
            body,
            json!({"cancelled": true, "status": "cancelling", "model_name": MODEL_NAME, "model_id": MODEL_ID})
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_a_negative_wait_is_rejected() {
        let _fx = Fixture::new(true);
        let request = Request::builder()
            .method(Method::DELETE)
            .uri(format!("/v1/downloads/{MODEL_NAME}?wait_ms=-1"))
            .body(Body::empty())
            .unwrap();
        let resp = downloads::router().oneshot(request).await.unwrap();
        assert_eq!(resp.status().as_u16(), 422);
    }
}

// ------------------------------------------------ TestTheInitStreamAnswersACancel

/// `TestTheInitStreamAnswersACancel`
///
/// `/v1/init`'s download raises, and the stream has to end honestly.  The
/// transfer is stubbed to raise rather than really run: the
/// `no_real_downloads` guard already replaces `download` for this whole
/// package, and what is under test is the router's handling, not the byte
/// loops (`tests/unit/test_download_cancel.rs` covers those).
mod test_the_init_stream_answers_a_cancel {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_it_ends_with_a_cancelled_event() {
        let fx = Fixture::new(true);
        fx.cancelled_download();

        let (status, text) = fx.init().await;

        assert_eq!(status, 200);
        assert!(event_names(&text).contains(&"cancelled".to_string()));
    }

    /// A cancel read as an error shows a failure toast for the user's own
    /// button.
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_it_emits_no_error_frame() {
        let fx = Fixture::new(true);
        fx.cancelled_download();

        let (_status, text) = fx.init().await;

        assert!(!event_names(&text).contains(&"error".to_string()));
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_it_emits_no_complete_frame() {
        let fx = Fixture::new(true);
        fx.cancelled_download();

        let (_status, text) = fx.init().await;

        assert!(!event_names(&text).contains(&"complete".to_string()));
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_it_discards_the_partial_bytes() {
        let fx = Fixture::new(true);
        fx.cancelled_download();
        let model_dir = fx.partial_download();

        fx.init().await;

        // `discard_partial_download(model_id=MODEL_ID)` removed exactly this
        // model's tree.
        assert!(!model_dir.exists());
    }

    /// Bytes left behind are the abandoned-download case the storage sweep
    /// already collects; the cancel itself still succeeded.
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    #[ignore = "needs a seam to make storage_service::discard_partial_download fail (the Python test \
                patches it with side_effect=OSError); the api units own it"]
    async fn test_a_failed_discard_does_not_become_an_error_frame() {
        let fx = Fixture::new(true);
        fx.cancelled_download();

        let (_status, text) = fx.init().await;

        let events = event_names(&text);
        assert!(events.contains(&"cancelled".to_string()) && !events.contains(&"error".to_string()));
    }

    /// The head of a list of nine side effects the early return skips.
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_it_does_not_mark_the_model_downloaded() {
        let fx = Fixture::new(true);
        fx.cancelled_download();

        fx.init().await;

        // `mark_model_downloaded` was never called.
        let ws_model = fx.workspace.read().get_model(Some(MODEL_NAME));
        assert!(!ws_model.is_some_and(|m| m.downloaded));
    }

    /// A model earns its place among the app-defaults by arriving.
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_it_does_not_promote_the_app_default() {
        let fx = Fixture::new(true);
        // Registered project-scoped (disabled), which is the state a
        // promotion would flip.
        fx.workspace.write().set_model_enabled(MODEL_NAME, false).unwrap();
        fx.cancelled_download();

        fx.init().await;

        // `promote_model_to_workspace_default` was never called.
        let ws_model = fx.workspace.read().get_model(Some(MODEL_NAME));
        assert!(!ws_model.is_some_and(|m| m.enabled));
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_the_registration_is_released() {
        let fx = Fixture::new(true);
        fx.cancelled_download();

        fx.init().await;

        assert!(cancellation::active_keys().is_empty());
    }

    /// The stub stands in for a successful transfer.
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_a_completed_download_also_releases_it() {
        let fx = Fixture::new(true);
        let mut artifacts = ArtifactMap::new();
        artifacts.insert("weights.gguf".to_string(), "https://x/f".to_string());
        fx.downloads.resolve_to(artifacts);

        fx.init().await;

        assert!(cancellation::active_keys().is_empty());
    }
}

// ------------------------------------------------ TestADroppedStreamIsNotACancel

/// `TestADroppedStreamIsNotACancel`
///
/// A window reload must not kill a multi-GB transfer.
///
/// This is what claiming and releasing the registration on the *download
/// thread* buys.  Hung off the SSE stream instead, the unwinding a closed
/// stream causes there would unregister a download still consuming bandwidth,
/// and the user's next cancel would be answered `not_found`.
///
/// Driven against `download_and_stream` rather than through `/v1/init`: the
/// stream is where the ownership question lives, and the route around it would
/// drag in the catalog seed, the model pipeline and the project setup for a
/// property none of them participate in.
mod test_a_dropped_stream_is_not_a_cancel {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_the_transfer_outlives_the_stream_and_still_cleans_up_after_itself() {
        let fx = Fixture::new(true);
        let started = Event::new();
        let release = Event::new();

        {
            let started = started.clone();
            let release = release.clone();
            fx.downloads.set_download(Arc::new(move |_call: DownloadStubCall| {
                started.set();
                // Bounded, so a regression cannot wedge the suite.
                release.wait(10.0);
                Ok(PathBuf::new())
            }));
        }

        let logger: SharedLogger = Arc::new(QuietLogger);
        let downloader = Arc::new(SLPArtifactDownloader::for_tests(Some(fx.models_dir.clone()), logger, None));
        let mut stream = init_router_module::download_and_stream(
            downloader,
            MODEL_ID.to_string(),
            MODEL_NAME.to_string(),
            false,
            None,
            Some(Arc::clone(&fx.service)),
        );

        let task = tokio::spawn(async move { while stream.next().await.is_some() {} });

        // Wait for the transfer to really be under way: the opening frame is
        // yielded before the blocking task is even submitted, so closing the
        // stream at that point would prove nothing.
        let waiter = started.clone();
        let began = tokio::task::spawn_blocking(move || waiter.wait(5.0)).await.unwrap();
        assert!(began, "the download never started");

        task.abort(); // the client walks away
        let _ = task.await;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let token = cancellation::find(MODEL_ID).expect("a dropped stream unregistered a live download");
            assert!(!token.cancelled(), "a dropped stream cancelled a live download");
            token
        }));

        // And the other half: the download thread releases the registration
        // on its own, though the stream that claimed it is long gone.
        release.set();
        let token = match result {
            Ok(token) => token,
            Err(panic) => std::panic::resume_unwind(panic),
        };
        let done = token.done.clone();
        let finished = tokio::task::spawn_blocking(move || done.wait(5.0)).await.unwrap();
        assert!(finished, "the transfer never released its registration");
        assert!(cancellation::find(MODEL_ID).is_none());
    }
}
