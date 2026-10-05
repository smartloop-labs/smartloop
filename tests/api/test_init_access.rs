//! Port of `tests/api/test_init_access.py`.
//!
//! /v1/init must answer a refused download with a status code, not a stream.
//!
//! Free models download with no token at all, so there is no credential gate
//! left on this route.  What remains is entitlement, and it is resolved
//! against the platform *before* the SSE stream opens: once the streaming
//! response begins, the request has committed to 200 and a refusal can only be
//! described inside a body that a client keying off the status will never read.
//!
//! Only failures that happen mid-transfer arrive as SSE error frames, tagged
//! with the status they would have been.
//!
//! The Python module drives the route with a `MagicMock` model service and
//! the router's `SLPArtifactDownloader` / `_download_and_stream` patched.  The
//! Rust route takes the same three things as seams: a model service is any
//! [`InitModelService`], and [`InitSeams`] replaces the downloader the route
//! builds and the stream it hands the transfer to.

use std::io;
use std::sync::Arc;

use futures::stream;
use parking_lot::Mutex;
use serde_json::Value;
use serial_test::serial;

use smartloop::api::routers::_errors::{status_for, HTTPException};
use smartloop::api::routers::init::{
    self, model_pipeline_missing, DownloadAndStreamFn, DownloadStreamArgs, InitDownloader, InitModelService,
    InitRequest, InitSeams, ProjectServicesHost, SseStream,
};
use smartloop::api::services::model_service::ModelServiceHooks;
use smartloop::api::services::ModelService;
use smartloop::model::downloader::artifact_downloader::{aggregate_error, DownloadError};
use smartloop::model::downloader::artifact_source::ArtifactMap;
use smartloop::model::downloader::cancellation::CancelToken;
use smartloop::model::downloader::slp_downloader::ProgressCallback;
use smartloop::platform::{ModelAccessError, PlatformError};
use smartloop::workspace::workspace::{Project, SharedWorkspace, Workspace, WorkspaceModel};

use crate::common::api_core::PlatformServiceFixture;
use crate::common::misc_client::sse_frames;
use crate::shared::misc_isolation::IsolatedHome;

const FREE_MODEL: &str = "gemma3-4b";
const PAID_MODEL: &str = "gemma3-12b";

/// `errno.ENOSPC`.
const ENOSPC: i32 = 28;

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

/// The fixture workspace: no weights on disk, the paid model registered but
/// not downloaded.  `models_dir` points into an empty temp dir, so
/// `gguf_exists` is false and every request takes the "needs download" path.
///
/// Saved under the isolated `SLP_HOME`: `save()` writes the config file that
/// resolves from `SLP_HOME`, *not* from `workspace_dir`, so without the
/// isolation the suite would overwrite the developer's own config.json.
fn fixture_workspace(home: &IsolatedHome) -> Workspace {
    let root = home.dir.path();
    let mut workspace = Workspace::new("ws-test");
    workspace.workspace_dir = root.to_string_lossy().into_owned();
    workspace.models_dir = root.join("models").to_string_lossy().into_owned();
    workspace.models.push(WorkspaceModel {
        model_name: PAID_MODEL.to_string(),
        model_id: "google/gemma-3-12b-it".to_string(),
        downloaded: false,
        ..Default::default()
    });
    workspace.save().expect("save the fixture workspace");
    workspace
}

/// The real model service over the fixture workspace (for the routes that
/// refuse before touching anything else).
fn model_service() -> (IsolatedHome, Arc<ModelService>) {
    let home = IsolatedHome::new();
    fixture_workspace(&home);
    (home, ModelService::new_bare())
}

/// The `MagicMock` model service of the Python fixtures.
///
/// `_ensure_workspace()` returns the one fixture workspace.  Attributes a
/// `MagicMock` answers with another mock behave as such a mock would: an
/// LLM is "loaded" (not `None`) under a name that is not the default model,
/// the indexer task is "registered" (not `None`), and `create_project`,
/// `setup_tasks`, `load_model` and the project setup do nothing.
struct FakeService {
    workspace: SharedWorkspace,
    /// `_model.workspace`; `None` while `_model` is `None`.
    model: Mutex<Option<SharedWorkspace>>,
    /// `mark_model_downloaded.call_args_list`: `(name, workspace=)`.
    mark_calls: Mutex<Vec<(String, Option<SharedWorkspace>)>>,
    /// `mark_model_downloaded.side_effect` runs the real method.
    real_mark: Mutex<bool>,
}

impl FakeService {
    fn new(workspace: Workspace, warm: bool) -> Arc<Self> {
        let workspace: SharedWorkspace = Arc::new(parking_lot::RwLock::new(workspace));
        let model = warm.then(|| Arc::clone(&workspace));
        Arc::new(FakeService {
            workspace,
            model: Mutex::new(model),
            mark_calls: Mutex::new(Vec::new()),
            real_mark: Mutex::new(false),
        })
    }
}

impl ProjectServicesHost for FakeService {
    fn current_project(&self) -> Option<Project> {
        let ws = self.model.lock().clone()?;
        let ws = ws.read();
        ws.current_project().cloned()
    }

    fn setup_project_tools(&self, _project: &Project) -> anyhow::Result<()> {
        Ok(())
    }

    fn mark_current_project_system_and_save(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

impl InitModelService for FakeService {
    fn ensure_workspace(&self) -> anyhow::Result<SharedWorkspace> {
        Ok(Arc::clone(&self.workspace))
    }

    fn mark_model_downloaded(&self, model_name: &str, workspace: Option<SharedWorkspace>) -> anyhow::Result<()> {
        self.mark_calls.lock().push((model_name.to_string(), workspace.clone()));
        if !*self.real_mark.lock() {
            return Ok(());
        }
        // `lambda name, workspace=None: ModelService.mark_model_downloaded(
        //     cold_service, name, workspace=workspace)` — the real method, on a
        // service whose `_model` is None and whose `_ensure_workspace` is this
        // fixture's.
        let real = ModelService::new_bare();
        let ws = Arc::clone(&self.workspace);
        real.set_hooks(ModelServiceHooks {
            ensure_workspace: Some(Arc::new(move |_cfg| Ok(Arc::clone(&ws)))),
            ..Default::default()
        });
        real.mark_model_downloaded(model_name, workspace)
    }

    fn ensure_model(&self) -> anyhow::Result<()> {
        // ``_model`` is None until ``_ensure_model()`` runs, so it is assigned
        // here rather than pretending it was always available.
        *self.model.lock() = Some(Arc::clone(&self.workspace));
        Ok(())
    }

    fn pipeline_workspace(&self) -> anyhow::Result<SharedWorkspace> {
        self.model.lock().clone().ok_or_else(model_pipeline_missing)
    }

    fn create_project(&self, _: &str, _: Option<&str>, _: bool, _: Option<&str>) -> anyhow::Result<()> {
        Ok(())
    }

    fn has_indexer_task(&self) -> bool {
        true
    }

    fn setup_tasks(&self, _workspace: SharedWorkspace) -> anyhow::Result<()> {
        Ok(())
    }

    fn llm_loaded(&self) -> bool {
        true
    }

    fn loaded_model_name(&self) -> Option<String> {
        Some("mock-model".to_string())
    }

    fn load_model(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn discard_partial_download(&self, _model_id: &str) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Everything a route test needs alive: the isolated home, a fake platform
/// (the catalog seed never reaches the network) and the model service.
struct Env {
    service: Arc<FakeService>,
    _platform: PlatformServiceFixture,
    _home: IsolatedHome,
}

/// The `model_service` fixture: a warm `MagicMock` service.
fn mock_service() -> Env {
    let home = IsolatedHome::new();
    let workspace = fixture_workspace(&home);
    Env { service: FakeService::new(workspace, true), _platform: PlatformServiceFixture::install(), _home: home }
}

/// The `cold_service` fixture: the service in the state a just-started
/// process is really in — `_model` is None until `_ensure_model()` runs.
fn cold_service() -> Env {
    let home = IsolatedHome::new();
    let workspace = fixture_workspace(&home);
    Env { service: FakeService::new(workspace, false), _platform: PlatformServiceFixture::install(), _home: home }
}

/// The downloader the stubbed router builds (`MagicMock()`).
struct FakeDownloader {
    /// `resolve_artifacts.side_effect` / `.return_value`.
    resolve: Box<dyn Fn() -> anyhow::Result<ArtifactMap> + Send + Sync>,
    /// `download.side_effect`.
    download_error: Mutex<Option<Box<dyn Fn() -> anyhow::Error + Send + Sync>>>,
    resolve_calls: Mutex<Vec<String>>,
    /// `download.call_args.kwargs["artifacts"]` of each call.
    download_calls: Mutex<Vec<Option<ArtifactMap>>>,
}

impl InitDownloader for FakeDownloader {
    fn resolve_artifacts(&self, model_id: &str) -> anyhow::Result<ArtifactMap> {
        self.resolve_calls.lock().push(model_id.to_string());
        (self.resolve)()
    }

    fn set_cancel_token(&self, _token: CancelToken) {}

    fn download(
        &self,
        _model_id: &str,
        _progress_callback: Option<ProgressCallback>,
        artifacts: Option<ArtifactMap>,
    ) -> anyhow::Result<()> {
        self.download_calls.lock().push(artifacts);
        match &*self.download_error.lock() {
            Some(raise) => Err(raise()),
            None => Ok(()),
        }
    }

    fn bootstrap_download(&self, _progress_callback: Option<ProgressCallback>) -> anyhow::Result<()> {
        Ok(())
    }

    fn list_bootstrap_models(&self) -> anyhow::Result<Vec<Value>> {
        Ok(Vec::new())
    }
}

fn signed_url() -> ArtifactMap {
    let mut artifacts = ArtifactMap::new();
    artifacts.insert("model.gguf".to_string(), "https://signed/url".to_string());
    artifacts
}

/// Replace the downloader the router builds with one under our control.
/// `side_effect` (when given) is what `resolve_artifacts` raises.
fn stub_downloader(
    side_effect: Option<fn() -> anyhow::Error>,
    artifacts: Option<ArtifactMap>,
) -> (Arc<FakeDownloader>, InitSeams) {
    let artifacts = artifacts.unwrap_or_else(signed_url);
    let resolve: Box<dyn Fn() -> anyhow::Result<ArtifactMap> + Send + Sync> = match side_effect {
        Some(raise) => Box::new(move || Err(raise())),
        None => Box::new(move || Ok(artifacts.clone())),
    };
    let made = Arc::new(FakeDownloader {
        resolve,
        download_error: Mutex::new(None),
        resolve_calls: Mutex::new(Vec::new()),
        download_calls: Mutex::new(Vec::new()),
    });
    let factory_made = Arc::clone(&made);
    let seams = InitSeams {
        downloader_factory: Some(Arc::new(move |_workspace| Arc::clone(&factory_made) as Arc<dyn InitDownloader>)),
        download_and_stream: None,
    };
    (made, seams)
}

/// `monkeypatch.setattr(init_router, "_download_and_stream", boom)`: a
/// stream that raises on its first step.
fn boom(raise: fn() -> anyhow::Error) -> DownloadAndStreamFn {
    Arc::new(move |_args: DownloadStreamArgs| -> SseStream { Box::pin(stream::once(async move { Err(raise()) })) })
}

/// `monkeypatch.setattr(init_router, "_download_and_stream", streamed)`: a
/// stream that yields one progress frame and finishes.
fn streamed() -> DownloadAndStreamFn {
    Arc::new(|_args: DownloadStreamArgs| -> SseStream {
        let frame = init::sse_event(&serde_json::json!({"status": "downloading", "message": "..."}), "progress");
        Box::pin(stream::once(async move { Ok(frame) }))
    })
}

/// What a `/v1/init` request answered: the status line and the body (the
/// SSE text, or the `{"detail": ...}` of a refusal).
struct Resp {
    status_code: u16,
    text: String,
    detail: Option<Value>,
}

impl Resp {
    fn frames(&self) -> Vec<Value> {
        sse_frames(&self.text)
    }

    fn errors(&self) -> Vec<Value> {
        self.frames().into_iter().filter(|f| f.get("status").and_then(Value::as_str) == Some("error")).collect()
    }

    fn detail(&self) -> String {
        self.detail.as_ref().and_then(Value::as_str).unwrap_or_default().to_string()
    }
}

/// `client.post("/v1/init", json={"model_name": model_name})`, read to the end.
async fn post_init(service: &Arc<FakeService>, seams: InitSeams, model_name: &str) -> Resp {
    let request = InitRequest { model_name: Some(model_name.to_string()), ..Default::default() };
    let svc: Arc<dyn InitModelService> = Arc::clone(service) as Arc<dyn InitModelService>;
    match init::init_with_seams(request, svc, seams).await {
        Ok(response) => {
            let status_code = response.status().as_u16();
            let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.expect("read the stream");
            Resp { status_code, text: String::from_utf8_lossy(&body).into_owned(), detail: None }
        }
        Err(HTTPException { status_code, detail, .. }) => Resp { status_code, text: String::new(), detail: Some(detail) },
    }
}

/// `PermissionError` / `ConnectionError` tagged with an upstream status, as the
/// platform client raises them.
fn tagged_permission(message: &str, status: u16) -> PlatformError {
    PlatformError::Permission { message: message.to_string(), status_code: Some(status) }
}

fn tagged_connection(message: &str, status: u16) -> PlatformError {
    PlatformError::Connection { message: message.to_string(), status_code: Some(status) }
}

mod test_entitlement_is_settled_before_the_stream {
    use super::*;

    #[tokio::test]
    #[serial]
    async fn test_refusal_is_a_403_not_a_200_stream() {
        let env = mock_service();
        let (_made, seams) = stub_downloader(
            Some(|| ModelAccessError::new(format!("Model '{PAID_MODEL}' is not available on your current plan.")).into()),
            None,
        );

        let resp = post_init(&env.service, seams, PAID_MODEL).await;

        assert_eq!(resp.status_code, 403);
        assert!(resp.detail().contains("current plan"), "{}", resp.detail());
    }

    #[tokio::test]
    #[serial]
    async fn test_rate_limit_surfaces_as_429() {
        // Not 403: retrying fixes this one, and calling it a plan problem would
        // send the user to a billing page instead of back in a minute.
        let env = mock_service();
        let (_made, seams) = stub_downloader(
            Some(|| tagged_connection("Too many download requests. Please wait a moment and try again.", 429).into()),
            None,
        );

        let resp = post_init(&env.service, seams, PAID_MODEL).await;

        assert_eq!(resp.status_code, 429);
    }

    #[tokio::test]
    #[serial]
    async fn test_platform_unreachable_is_502() {
        let env = mock_service();
        let (_made, seams) = stub_downloader(Some(|| PlatformError::connection("Connection error: refused").into()), None);

        let resp = post_init(&env.service, seams, PAID_MODEL).await;

        assert_eq!(resp.status_code, 502);
    }

    /// The only remaining 400 on this route.  It is not fixed by signing in,
    /// so a client must be able to tell it apart from an entitlement refusal.
    #[tokio::test]
    #[serial]
    async fn test_unknown_model_is_still_400() {
        let (_home, svc) = model_service();

        let request = InitRequest { model_name: Some("no-such-model".to_string()), ..Default::default() };
        let resp = init::init_with(request, svc).await;

        let err = resp.expect_err("an unknown model is refused before the stream");
        assert_eq!(err.status_code, 400);
        assert!(err.detail.as_str().unwrap_or_default().contains("Unknown model"));
    }

    /// The point of DEV-383: the old behaviour was a 400 ("Authentication
    /// required") for any download without a token.  A free model must now get
    /// as far as asking the platform for URLs.
    #[tokio::test]
    #[serial]
    async fn test_free_model_is_resolved_without_credentials() {
        let env = mock_service();
        let (made, seams) = stub_downloader(None, None);

        let resp = post_init(&env.service, seams, FREE_MODEL).await;
        assert_eq!(resp.status_code, 200);

        assert_eq!(made.resolve_calls.lock().len(), 1, "resolve_artifacts was not called exactly once");
    }

    /// The download route is rate-limited, so the URLs are minted once.
    #[tokio::test]
    #[serial]
    async fn test_resolved_urls_are_handed_to_the_download() {
        let env = mock_service();
        let artifacts = signed_url();
        let (made, seams) = stub_downloader(None, Some(artifacts.clone()));

        post_init(&env.service, seams, PAID_MODEL).await;

        let calls = made.download_calls.lock();
        let last = calls.last().expect("the download was never called");
        assert_eq!(last.as_ref(), Some(&artifacts));
    }
}

/// A failure after the stream opened still has to be classifiable.
///
/// The status line is 200 by then, so the code rides on the event for the
/// client to handle it the same way it handles the pre-stream status.
mod test_mid_stream_failure_carries_its_status {
    use super::*;

    #[tokio::test]
    #[serial]
    async fn test_late_plan_denial_is_tagged_403() {
        // A plan that changed between resolving the URLs and using them.
        let env = mock_service();
        let (made, seams) = stub_downloader(None, None);
        *made.download_error.lock() = Some(Box::new(|| {
            ModelAccessError::new(format!("Model '{PAID_MODEL}' is not available on your current plan.")).into()
        }));

        let resp = post_init(&env.service, seams, PAID_MODEL).await;

        assert_eq!(resp.status_code, 200);
        let errors = resp.errors();
        let error = errors.first().unwrap_or_else(|| panic!("no error frame in stream: {:?}", resp.text));
        assert_eq!(error["code"], 403);
        assert!(error["message"].as_str().unwrap_or_default().contains("not available on your current plan"));
    }

    #[tokio::test]
    #[serial]
    async fn test_expired_url_is_tagged_with_its_own_status() {
        let env = mock_service();
        let (_made, mut seams) = stub_downloader(None, None);
        seams.download_and_stream =
            Some(boom(|| tagged_permission("Download rejected (401 Unauthorized).", 401).into()));

        let resp = post_init(&env.service, seams, PAID_MODEL).await;

        assert_eq!(resp.status_code, 200); // already committed; the body carries the failure
        let errors = resp.errors();
        assert!(!errors.is_empty(), "no error frame in stream: {:?}", resp.text);
        assert_eq!(errors[0]["code"], 401);
    }
}

/// The code has to answer "would trying again help?", not just "it broke".
///
/// Every one of these used to arrive as a bare 500 or, worse, a 403 — which
/// the desktop client renders as "Sign in to download this model" for a
/// signed-out user and "Your plan doesn't include this model" for a signed-in
/// one.  A full disk telling someone to upgrade their plan is the failure this
/// guards.
mod test_mid_stream_failure_is_classified_by_what_fixes_it {
    use super::*;

    // Aggregated by `aggregate_error` once the transfer gives up: a dropped
    // socket, an expired signed URL, an upstream 5xx.
    #[test]
    fn test_status_reflects_the_remedy_connection_error() {
        let exc = DownloadError::connection("Failed to download 1 file(s): model.gguf", None);
        assert_eq!(status_for(&exc), 502);
    }

    #[test]
    fn test_status_reflects_the_remedy_model_access_error() {
        let exc = ModelAccessError::new("not on your plan");
        assert_eq!(status_for(&exc), 403);
    }

    // An OSError subclass, and emphatically not an entitlement verdict.
    #[test]
    fn test_status_reflects_the_remedy_permission_error() {
        let exc = io::Error::new(io::ErrorKind::PermissionDenied, "[Errno 13] Permission denied: '/models'");
        assert_eq!(status_for(&exc), 500);
    }

    // Out of disk: retryable (free a little and try again) but not a network
    // failure — 507, not 502/500.
    #[test]
    fn test_status_reflects_the_remedy_disk_full() {
        let exc = io::Error::from_raw_os_error(ENOSPC);
        assert_eq!(status_for(&exc), 507);
    }

    #[test]
    fn test_status_reflects_the_remedy_runtime_error() {
        let exc = DownloadError::runtime("model.gguf: MD5 verification failed", None);
        assert_eq!(status_for(&exc), 500);
    }

    /// A platform error carries its own status and keeps it.
    #[test]
    fn test_upstream_status_still_wins() {
        assert_eq!(status_for(&tagged_connection("Too many download requests.", 429)), 429);
        assert_eq!(status_for(&tagged_permission("Sign in to download this model.", 401)), 401);
    }

    #[tokio::test]
    #[serial]
    async fn test_dropped_socket_reaches_the_client_as_502() {
        let env = mock_service();
        let (_made, mut seams) = stub_downloader(None, None);
        seams.download_and_stream =
            Some(boom(|| DownloadError::connection("Failed to download 1 file(s): model.gguf", None).into()));

        let resp = post_init(&env.service, seams, PAID_MODEL).await;

        assert_eq!(resp.status_code, 200);
        let errors = resp.errors();
        assert!(!errors.is_empty(), "no error frame in stream: {:?}", resp.text);
        assert_eq!(errors[0]["code"], 502);
    }

    #[tokio::test]
    #[serial]
    async fn test_local_disk_refusal_is_not_a_sign_in_prompt() {
        let env = mock_service();
        let (_made, mut seams) = stub_downloader(None, None);
        seams.download_and_stream = Some(boom(|| {
            io::Error::new(io::ErrorKind::PermissionDenied, "[Errno 13] Permission denied: '/models'").into()
        }));

        let resp = post_init(&env.service, seams, PAID_MODEL).await;

        let errors = resp.errors();
        assert!(!errors.is_empty(), "no error frame in stream: {:?}", resp.text);
        assert_eq!(errors[0]["code"], 500);
    }

    /// The exact shape a real transfer produces: `aggregate_error` wraps the
    /// chunked path's ENOSPC, keeps it transient, and names the disk.  The
    /// client must see 507 (free some space and retry) and a message about the
    /// disk — not a bare filename list that could be a dropped socket.
    #[tokio::test]
    #[serial]
    async fn test_disk_full_reaches_the_client_as_507() {
        let env = mock_service();
        let (_made, mut seams) = stub_downloader(None, None);
        seams.download_and_stream = Some(boom(|| {
            aggregate_error(vec![("weights.gguf".to_string(), DownloadError::Io(io::Error::from_raw_os_error(ENOSPC)))])
                .into()
        }));

        let resp = post_init(&env.service, seams, PAID_MODEL).await;

        assert_eq!(resp.status_code, 200);
        let errors = resp.errors();
        assert!(!errors.is_empty(), "no error frame in stream: {:?}", resp.text);
        assert_eq!(errors[0]["code"], 507);
        let message = errors[0]["message"].as_str().unwrap_or_default();
        assert!(message.contains("no space left on device"), "{message}");
        assert!(message.contains("weights.gguf"), "{message}");
    }
}

/// /v1/init must not need a prior request to have warmed the service.
///
/// `ModelService` is a process-wide singleton whose model starts as `None`;
/// `ensure_model()` is what assigns it, and it runs *after* the download.  The
/// router used to save through the model immediately after
/// `mark_model_downloaded`, so on a fresh process the weights landed on disk
/// and the client still got an error frame.  Both branches hit that line, so
/// the retry failed identically: "model_ready", then error, forever.
mod test_init_on_a_fresh_process {
    use super::*;

    #[tokio::test]
    #[serial]
    async fn test_download_completes_without_a_warmed_model() {
        let env = cold_service();
        let (_made, mut seams) = stub_downloader(None, None);
        seams.download_and_stream = Some(streamed());

        let resp = post_init(&env.service, seams, PAID_MODEL).await;

        assert_eq!(resp.status_code, 200);
        let errors = resp.errors();
        assert!(errors.is_empty(), "fresh-process init errored: {errors:?}");
        let calls = env.service.mark_calls.lock();
        assert_eq!(calls.len(), 1, "mark_model_downloaded was not called exactly once");
        assert_eq!(calls[0].0, PAID_MODEL);
    }

    /// The model_ready branch dereferenced the same `None`.
    #[tokio::test]
    #[serial]
    async fn test_already_downloaded_also_survives_a_fresh_process() {
        let env = cold_service();
        // Where the route will look, rather than a path built to match it: the
        // workspace resolves `model_dir` from its own `models_dir`, so asking
        // it keeps the sentinel and the lookup from drifting apart.
        let model_dir = {
            let ws = env.service.workspace.read();
            std::path::PathBuf::from(ws.get_model(Some(PAID_MODEL)).expect("registered").model_dir())
        };
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("weights.gguf"), b"\0").unwrap();

        let resp = post_init(&env.service, InitSeams::default(), PAID_MODEL).await;

        let frames = resp.frames();
        assert!(frames.iter().any(|f| f.get("status").and_then(Value::as_str) == Some("model_ready")), "{frames:?}");
        assert!(resp.errors().is_empty(), "{frames:?}");
    }

    /// The flag is only useful if it is persisted, and it must not need the
    /// model.
    ///
    /// `mark_model_downloaded` resolves the workspace itself and saves it, so
    /// the flag survives a restart whether or not anything warmed the service.
    /// Python counts `Workspace.save` calls; here the persisted config.json is
    /// read back, which only shows the flag if a save ran after it was set.
    #[tokio::test]
    #[serial]
    async fn test_the_download_flag_reaches_disk() {
        let env = cold_service();
        *env.service.real_mark.lock() = true;
        let (_made, mut seams) = stub_downloader(None, None);
        seams.download_and_stream = Some(streamed());

        let resp = post_init(&env.service, seams, PAID_MODEL).await;

        assert!(resp.errors().is_empty(), "{:?}", resp.text);
        let saved = Workspace::load().expect("the workspace was never persisted");
        assert!(
            saved.get_model(Some(PAID_MODEL)).expect("registered on disk").downloaded,
            "the workspace was never persisted with the flag"
        );
        assert!(env.service.workspace.read().get_model(Some(PAID_MODEL)).expect("registered").downloaded);
    }

    /// The flag has to land on the instance the router mutated.
    ///
    /// This fixture's `ensure_workspace` returns one object, so the test above
    /// cannot tell the two instances apart — it passes whether or not the
    /// workspace is threaded through.  Asserting the hand-off is what pins it
    /// here; `test_the_flag_survives_a_second_load` covers the real
    /// two-instance behaviour.
    #[tokio::test]
    #[serial]
    async fn test_the_router_hands_over_its_own_workspace() {
        let env = cold_service();
        let (_made, mut seams) = stub_downloader(None, None);
        seams.download_and_stream = Some(streamed());

        let resp = post_init(&env.service, seams, PAID_MODEL).await;

        assert_eq!(resp.status_code, 200);
        let calls = env.service.mark_calls.lock();
        let (_, workspace) = calls.last().expect("mark_model_downloaded was never called");
        let workspace = workspace.as_ref().expect("no workspace= was handed over");
        assert!(Arc::ptr_eq(workspace, &env.service.workspace));
    }

    /// A cold `ensure_workspace` builds a *fresh* workspace every call.
    ///
    /// The router registers the model on its own instance and deliberately
    /// does not save (a refused download must strand nothing in config.json),
    /// so resolving a second instance inside `mark_model_downloaded` read the
    /// model back off disk *without* that registration, skipped the flag for a
    /// model it could not find, and saved the stale copy over it — leaving
    /// config.json with no record of the model at all.
    #[test]
    #[serial]
    fn test_the_flag_survives_a_second_load() {
        /// `monkeypatch.setattr(Workspace, "config_file", ...)`, undone on drop.
        struct ConfigFile;
        impl Drop for ConfigFile {
            fn drop(&mut self) {
                Workspace::set_config_file(None);
            }
        }

        let home = IsolatedHome::new();
        let tmp = home.dir.path();
        Workspace::set_config_file(Some(tmp.join("config.json").to_string_lossy().into_owned()));
        let _config_file = ConfigFile;
        let mut seed = Workspace::new("ws-1");
        seed.workspace_dir = tmp.join("ws-1").to_string_lossy().into_owned();
        seed.models_dir = tmp.join("ws-1").join("models").to_string_lossy().into_owned();
        seed.save().expect("save the seed workspace");

        let svc = ModelService::new_bare();

        // What the router does before the stream opens.
        let routers_workspace = svc.ensure_workspace(None).expect("cold ensure_workspace");
        routers_workspace.write().add_model(PAID_MODEL, "google/gemma-3-12b-it", None, false);
        let second = svc.ensure_workspace(None).expect("cold ensure_workspace");
        assert!(
            !Arc::ptr_eq(&second, &routers_workspace),
            "a cold _ensure_workspace is supposed to return a fresh instance"
        );

        // What the stream does after the transfer.
        svc.mark_model_downloaded(PAID_MODEL, Some(routers_workspace.clone()))
            .expect("mark the model downloaded");

        let reloaded = Workspace::load().expect("config.json is on disk");
        let names: Vec<&str> = reloaded.models.iter().map(|m| m.model_name.as_str()).collect();
        assert_eq!(names, vec![PAID_MODEL], "the router's add_model was clobbered by a stale instance");
        assert!(reloaded.get_model(Some(PAID_MODEL)).expect("registered").downloaded);
    }
}
