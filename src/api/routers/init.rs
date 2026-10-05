//! Port of `smartloop/api/routers/init.py`.
//!
//! Init router – initialise workspace and stream model download progress via
//! SSE (`POST /v1/init`, `POST /v1/bootstrap`).

#![allow(unused_imports)]

use std::sync::Arc;

use axum::body::Bytes;
use axum::response::Response;
use axum::routing::post;
use axum::{Json, Router};
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::config::api_settings;
use crate::api::routers::_errors::{status_for, HTTPException};
use crate::api::services::storage_service;
use crate::api::services::{get_model_service, ModelService};
use crate::config::model_config::ModelConfig;
use crate::config::SUPPORTED_MODELS;
use crate::constants::{canonical_model_name, display_model_name};
use crate::model::downloader::artifact_source::ArtifactMap;
use crate::model::downloader::cancellation;
use crate::model::downloader::cancellation::DownloadCancelled;
use crate::model::downloader::{DownloadProgress, SLPArtifactDownloader};
use crate::platform::get_platform_service;
use crate::workspace::context::projects_delegated;
use crate::workspace::workspace::{is_mmproj_gguf, Project, SharedWorkspace, Workspace};

/// Python logger name (`logging.getLogger(__name__)`).
pub const LOG_TARGET: &str = "smartloop.api.routers.init";

/// SSE keepalive comment — keeps the connection alive without producing a
/// client-visible event.
pub const SSE_KEEPALIVE: &str = ": keepalive\n\n";

/// Request body for workspace initialization.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct InitRequest {
    /// Model to initialize.
    #[serde(default)]
    pub model_name: Option<String>,
    /// Developer token for model download.
    #[serde(default)]
    pub developer_token: Option<String>,
    /// Working directory for the default project; defaults to
    /// `~/smartloop/<name>` when omitted.
    #[serde(default)]
    pub cwd: Option<String>,
    /// Project the download was requested for; scopes the resulting
    /// enablement to it.
    #[serde(default)]
    pub project_id: Option<String>,
}

/// A stream of already-framed SSE chunks; an `Err` item is the Python
/// generator raising (e.g. `DownloadCancelled`), which the caller handles.
pub type SseStream = BoxStream<'static, anyhow::Result<String>>;

/// The slice of `ModelService` that [`setup_project_services`] touches, as a
/// trait so tests can observe it without a real model service.
pub trait ProjectServicesHost: Send + Sync {
    /// `model_service._model.workspace.current_project` (`None` when the
    /// workspace has no current project).
    fn current_project(&self) -> Option<Project>;
    /// `model_service._setup_project_tools(project)`: install MCP servers and
    /// generate skills for `project`.
    fn setup_project_tools(&self, project: &Project) -> anyhow::Result<()>;
    /// Re-fetch the current project (setup may have replaced the model
    /// pipeline), set its `system` flag, then `model_service._model.save()`.
    fn mark_current_project_system_and_save(&self) -> anyhow::Result<()>;
}

impl ProjectServicesHost for ModelService {
    fn current_project(&self) -> Option<Project> {
        todo!()
    }

    fn setup_project_tools(&self, project: &Project) -> anyhow::Result<()> {
        let _ = project;
        todo!()
    }

    fn mark_current_project_system_and_save(&self) -> anyhow::Result<()> {
        todo!()
    }
}

/// The `/v1` router carrying the init routes (`tags=["init"]`).
pub fn router() -> Router {
    todo!()
}

/// Format a JSON object as an SSE event string
/// (`event: {event}\ndata: {json.dumps(data)}\n\n`, Python separators).
pub fn sse_event(data: &Value, event: &str) -> String {
    let _ = (data, event);
    todo!()
}

/// Terminal SSE frame for a failed stream, tagged with a status code
/// (`{"status": "error", "code": status_for(exc), "message": str(exc)}`).
pub fn error_event(exc: &anyhow::Error) -> String {
    let _ = exc;
    todo!()
}

/// Terminal SSE frame for a download the caller stopped.
///
/// Its own event name (`cancelled`) rather than an `error` frame carrying
/// 499, because to the client this is a success: the thing it asked for
/// happened.
pub fn cancelled_event(model_name: &str) -> String {
    let _ = model_name;
    todo!()
}

/// Fire-and-forget best-effort cache of the cloud model catalog.
///
/// Scheduled on a blocking worker rather than awaited: `list_models()` is a
/// sync cloud call with a 30s timeout, and blocking on it would stall the SSE
/// stream.  Failures are logged as warnings.
pub fn seed_model_catalog() {
    todo!()
}

/// Whether `model_dir` holds servable weights — not merely *a* GGUF.
///
/// Looks in `model_dir` and `model_dir/bin`.  Any leftover `.part` file
/// disqualifies the model (an unfinished transfer), and an `mmproj`
/// companion alone is not weights (see [`is_mmproj_gguf`]).
pub fn gguf_exists(model_dir: &str) -> bool {
    let _ = model_dir;
    todo!()
}

/// Download model files and stream progress.
///
/// `artifacts` carries a signed-URL map already resolved by the caller (see
/// [`SLPArtifactDownloader::resolve_artifacts`]), so entitlement is settled
/// before this stream opens.  Progress frames are throttled to one per file
/// per 0.5s (completed frames always pass); a keepalive is emitted after ~5s
/// of silence.  A non-bootstrap download claims/releases the cancel
/// registration on the download thread, and a cancelled download discards
/// its partial bytes before re-raising [`DownloadCancelled`].
pub fn download_and_stream(
    downloader: Arc<SLPArtifactDownloader>,
    model_id: String,
    model_name: String,
    is_bootstrap: bool,
    artifacts: Option<ArtifactMap>,
    model_service: Option<Arc<ModelService>>,
) -> SseStream {
    let _ = (downloader, model_id, model_name, is_bootstrap, artifacts, model_service);
    todo!()
}

/// Load the orchestrator model into memory with SSE keepalives (every 2s).
///
/// Delegates to `model_service.load_model()`, which always loads the small
/// base model (`api_settings.default_model`) as the orchestrator.
pub fn load_orchestrator_and_stream(model_service: Arc<ModelService>) -> SseStream {
    let _ = model_service;
    todo!()
}

/// Create a bare `general_chat` system project if none exist (no skills/MCP —
/// LLM not loaded yet).
pub fn ensure_project(
    model_service: Arc<ModelService>,
    workspace: SharedWorkspace,
    cwd: Option<String>,
) -> SseStream {
    let _ = (model_service, workspace, cwd);
    todo!()
}

/// Install MCP servers and generate skills — requires LLM to be loaded.
///
/// Skipped entirely (checked *before* the workspace is read) in a process
/// that has delegated its projects; skipped when there is no current project
/// or it is already `system`.  Otherwise yields `setup` and `setup_complete`.
pub fn setup_project_services<H>(model_service: Arc<H>) -> SseStream
where
    H: ProjectServicesHost + ?Sized + 'static,
{
    let _ = model_service;
    todo!()
}

/// Return the project info dict for the current project (`None` when there
/// is none): `id`, `name`, `description`, `model_name`, `current`, `system`.
pub fn get_project_info(workspace: &Workspace, model_name: &str) -> Option<Value> {
    let _ = (workspace, model_name);
    todo!()
}

/// `POST /v1/init` — initialize the workspace and download the model.
///
/// Settles entitlement (signed URLs) before the response starts, so a
/// refusal arrives as an HTTP status (`status_for`); an unknown model is a
/// 400.  Returns a `text/event-stream` response with `Cache-Control:
/// no-cache` and `X-Accel-Buffering: no`.
pub async fn init(Json(request): Json<InitRequest>) -> Result<Response, HTTPException> {
    let _ = request;
    todo!()
}

/// [`init`] with an explicit model service (the FastAPI dependency).
pub async fn init_with(
    request: InitRequest,
    model_service: Arc<ModelService>,
) -> Result<Response, HTTPException> {
    let _ = (request, model_service);
    todo!()
}

/// `POST /v1/bootstrap` — download a model and create a default project.
///
/// No authentication required; the body is optional (an empty body is
/// `None`).  Intended for first-run scenarios.  Streams SSE progress.
pub async fn bootstrap(body: Bytes) -> Result<Response, HTTPException> {
    let _ = body;
    todo!()
}

/// [`bootstrap`] with an explicit model service (the FastAPI dependency).
pub async fn bootstrap_with(
    request: Option<InitRequest>,
    model_service: Arc<ModelService>,
) -> Response {
    let _ = (request, model_service);
    todo!()
}
