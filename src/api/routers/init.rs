//! Port of `smartloop/api/routers/init.py`.
//!
//! Init router – initialise workspace and stream model download progress via
//! SSE (`POST /v1/init`, `POST /v1/bootstrap`).

use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes};
use axum::http::{header, StatusCode};
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use futures::stream::{BoxStream, Stream, StreamExt};
use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::api::config::API_SETTINGS;
use crate::api::routers::_errors::{status_for, HTTPException};
use crate::api::routers::models::{as_std_error, parse_json_body, py_error_kind_of, PyErrorKind};
use crate::api::services::storage_service::{self, StorageModelService};
use crate::api::services::{get_model_service, ModelService};
use crate::config::SUPPORTED_MODELS;
use crate::constants::{canonical_model_name, display_model_name};
use crate::model::downloader::artifact_downloader::DownloadError;
use crate::model::downloader::artifact_source::ArtifactMap;
use crate::model::downloader::cancellation::{self, CancelToken};
use crate::model::downloader::slp_downloader::{ProgressCallback, SharedLogger};
use crate::model::downloader::{DownloadProgress, SLPArtifactDownloader};
use crate::platform::get_platform_service;
use crate::utils::log_utils::LoggerLike;
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
        let model = self.model.read().clone()?;
        let ws = model.workspace.read();
        ws.current_project().cloned()
    }

    fn setup_project_tools(&self, project: &Project) -> anyhow::Result<()> {
        ModelService::setup_project_tools(self, &project.id)
    }

    fn mark_current_project_system_and_save(&self) -> anyhow::Result<()> {
        // Re-fetch: setup_project_tools may have replaced the model pipeline
        // via ensure_model(), so a project resolved earlier may belong to a
        // stale workspace.
        let model = self.model.read().clone().ok_or_else(model_pipeline_missing)?;
        {
            let mut ws = model.workspace.write();
            if let Some(refreshed) = ws.current_project_mut() {
                refreshed.system = true;
            }
        }
        model.save()
    }
}

/// The error Python raised reading `model_service._model.<attr>` with no
/// pipeline loaded.
fn model_pipeline_missing() -> anyhow::Error {
    anyhow::anyhow!("'NoneType' object has no attribute 'workspace'")
}

/// `model_service._model.workspace`.
fn pipeline_workspace(model_service: &ModelService) -> anyhow::Result<SharedWorkspace> {
    model_service
        .model
        .read()
        .as_ref()
        .map(|m| Arc::clone(&m.workspace))
        .ok_or_else(model_pipeline_missing)
}

/// The `/v1` router carrying the init routes (`tags=["init"]`).
pub fn router() -> Router {
    Router::new()
        .route("/v1/init", post(init))
        .route("/v1/bootstrap", post(bootstrap))
}

// ---------------------------------------------------------------------------
// SSE framing.
// ---------------------------------------------------------------------------

/// `json.dumps` with its default settings: `", "` / `": "` separators and
/// `ensure_ascii=True` (every non-ASCII or DEL character `\uXXXX`-escaped,
/// astral characters as surrogate pairs).
struct PyJsonFormatter;

impl serde_json::ser::Formatter for PyJsonFormatter {
    fn begin_array_value<W: ?Sized + io::Write>(&mut self, writer: &mut W, first: bool) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + io::Write>(&mut self, writer: &mut W, first: bool) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }

    fn write_string_fragment<W: ?Sized + io::Write>(&mut self, writer: &mut W, fragment: &str) -> io::Result<()> {
        for ch in fragment.chars() {
            if ch.is_ascii() && ch != '\u{7f}' {
                let mut buf = [0u8; 1];
                writer.write_all(ch.encode_utf8(&mut buf).as_bytes())?;
            } else {
                let mut units = [0u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    write!(writer, "\\u{:04x}", unit)?;
                }
            }
        }
        Ok(())
    }
}

/// Serialize `data` exactly as Python's `json.dumps(data)` would.
pub(crate) fn py_json_dumps(data: &Value) -> String {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PyJsonFormatter);
    data.serialize(&mut ser).expect("serializing a JSON value cannot fail");
    String::from_utf8(out).expect("ensure_ascii output is ASCII")
}

/// Format a JSON object as an SSE event string
/// (`event: {event}\ndata: {json.dumps(data)}\n\n`, Python separators).
pub fn sse_event(data: &Value, event: &str) -> String {
    format!("event: {event}\ndata: {}\n\n", py_json_dumps(data))
}

/// Terminal SSE frame for a failed stream, tagged with a status code
/// (`{"status": "error", "code": status_for(exc), "message": str(exc)}`).
pub fn error_event(exc: &anyhow::Error) -> String {
    sse_event(
        &json!({
            "status": "error",
            "code": status_for(as_std_error(exc)),
            "message": exc.to_string(),
        }),
        "error",
    )
}

/// Terminal SSE frame for a download the caller stopped.
///
/// Its own event name (`cancelled`) rather than an `error` frame carrying
/// 499, because to the client this is a success: the thing it asked for
/// happened.  A client that read it as an error would show a failure toast
/// for its own button.
pub fn cancelled_event(model_name: &str) -> String {
    sse_event(
        &json!({
            "status": "cancelled",
            "message": format!("Download of {} cancelled.", display_model_name(model_name)),
        }),
        "cancelled",
    )
}

/// A `text/event-stream` response with the headers both routes send.
fn sse_response<S>(stream: S) -> Response
where
    S: Stream<Item = String> + Send + 'static,
{
    let body = Body::from_stream(stream.map(|chunk| Ok::<Bytes, Infallible>(Bytes::from(chunk))));
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    let headers = response.headers_mut();
    // Starlette appends the charset to every ``text/*`` media type.
    headers.insert(header::CONTENT_TYPE, header::HeaderValue::from_static("text/event-stream; charset=utf-8"));
    headers.insert(header::CACHE_CONTROL, header::HeaderValue::from_static("no-cache"));
    headers.insert("x-accel-buffering", header::HeaderValue::from_static("no"));
    response
}

// ---------------------------------------------------------------------------
// Blocking work.
// ---------------------------------------------------------------------------

/// The message of a panic payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "blocking task panicked".to_string()
    }
}

/// Run fallible blocking work off the async workers
/// (`loop.run_in_executor(None, ...)`).  A panic in the worker comes back as
/// an error, so the stream still ends with an `error` frame the way an
/// exception in the executor did.
async fn blocking<T, F>(f: F) -> anyhow::Result<T>
where
    F: FnOnce() -> anyhow::Result<T> + Send + 'static,
    T: Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(result) => result,
        Err(join_err) => match join_err.try_into_panic() {
            Ok(payload) => Err(anyhow::anyhow!(panic_message(payload.as_ref()))),
            Err(join_err) => Err(anyhow::anyhow!("blocking task failed: {join_err}")),
        },
    }
}

/// Turn a downloader error into the exception the Python code raised: a
/// platform failure is handed back as-is (so `status_for` sees e.g. a
/// `ModelAccessError` directly) and a plain cancel is a bare
/// `DownloadCancelled`.
fn download_error_into_anyhow(err: DownloadError) -> anyhow::Error {
    match err {
        DownloadError::Source(inner) => inner,
        DownloadError::Cancelled(cancelled) => anyhow::Error::new(cancelled),
        other => anyhow::Error::new(other),
    }
}

/// `logger` (this module's) as the downloader's logger.
struct InitLogger;

impl LoggerLike for InitLogger {
    fn info(&self, msg: &str) {
        tracing::info!(target: LOG_TARGET, "{msg}");
    }

    fn warning(&self, msg: &str) {
        tracing::warn!(target: LOG_TARGET, "{msg}");
    }

    fn debug(&self, msg: &str) {
        tracing::debug!(target: LOG_TARGET, "{msg}");
    }

    fn error(&self, msg: &str) {
        tracing::error!(target: LOG_TARGET, "{msg}");
    }
}

fn init_logger() -> SharedLogger {
    Arc::new(InitLogger)
}

/// What [`storage_service::discard_partial_download`] needs from the model
/// service: the live workspace (with per-project state re-read).
struct StorageView(Arc<ModelService>);

impl StorageModelService for StorageView {
    fn ensure_workspace(&self) -> anyhow::Result<Option<SharedWorkspace>> {
        Ok(Some(self.0.ensure_workspace(None)?))
    }

    fn refresh_project_state(&self, project_id: &str) -> anyhow::Result<bool> {
        Ok(self.0.refresh_project_state(Some(project_id)))
    }

    fn unload_model_by_name_unlocked(&self, name: &str) -> anyhow::Result<()> {
        self.0.unload_model_by_name_unlocked(name);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Catalog seed.
// ---------------------------------------------------------------------------

/// Replacement for [`seed_model_catalog`] (tests patch `_seed_model_catalog`
/// to keep the real cloud call off their path).
pub type SeedModelCatalogHook = Arc<dyn Fn() + Send + Sync>;

static SEED_MODEL_CATALOG_HOOK: Lazy<RwLock<Option<SeedModelCatalogHook>>> = Lazy::new(|| RwLock::new(None));

/// Install (or with `None`, remove) a replacement for [`seed_model_catalog`].
pub fn set_seed_model_catalog_hook(hook: Option<SeedModelCatalogHook>) {
    *SEED_MODEL_CATALOG_HOOK.write() = hook;
}

/// Fire-and-forget best-effort cache of the cloud model catalog.
///
/// Scheduled on a blocking worker rather than awaited: `list_models()` is a
/// sync cloud call with a 30s timeout, and blocking on it would stall the SSE
/// stream.  Seeding is off the critical path, so we never wait on it.
/// Failures are logged as warnings.
pub fn seed_model_catalog() {
    let hook = SEED_MODEL_CATALOG_HOOK.read().clone();
    if let Some(hook) = hook {
        hook();
        return;
    }

    let handle = match tokio::runtime::Handle::try_current() {
        Ok(handle) => handle,
        Err(e) => {
            tracing::warn!(target: LOG_TARGET, "Could not schedule model catalog seed: {e}");
            return;
        }
    };
    let svc = get_platform_service();
    let future = handle.spawn_blocking(move || svc.list_models(false).map(|_| ()));
    handle.spawn(async move {
        match future.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::warn!(target: LOG_TARGET, "Could not seed model catalog cache: {e}"),
            Err(join_err) => {
                let message = match join_err.try_into_panic() {
                    Ok(payload) => panic_message(payload.as_ref()),
                    Err(join_err) => join_err.to_string(),
                };
                tracing::warn!(target: LOG_TARGET, "Could not seed model catalog cache: {message}");
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Weights on disk.
// ---------------------------------------------------------------------------

/// Whether `model_dir` holds servable weights — not merely *a* GGUF.
///
/// This decides whether `/v1/init` downloads at all, so anything it accepts
/// is something the route then reports as ready and never revisits.  Asking
/// only "is there a `.gguf` here?" accepted two states that are not a model,
/// and an interrupted multi-file download produces one or the other depending
/// on which file finished first:
///
/// * **A leftover `.part`.**  Bytes are written to that sidecar and renamed
///   only once the checksum matches, so its presence is the downloader saying
///   a file is unfinished.  A restart during the final verify once left
///   `mmproj` renamed and the 2GB weights still `.part`; because the companion
///   satisfied the old check every later `/v1/init` answered "already
///   available" and the weights could never arrive.
/// * **Only an `mmproj` companion.**  A vision projector is not weights
///   (see [`is_mmproj_gguf`]), so a directory holding one and nothing else
///   resolves to no model at all.
///
/// Looks in `model_dir` and `model_dir/bin`.  Deliberately still local and
/// cheap — no artifact list is fetched to compare against, because this runs
/// before entitlement is settled and its only job is to decide whether to ask.
pub fn gguf_exists(model_dir: &str) -> bool {
    let root = Path::new(model_dir);
    let dirs: Vec<_> = [root.to_path_buf(), root.join("bin")].into_iter().filter(|d| d.is_dir()).collect();
    if dirs.is_empty() {
        return false;
    }
    let listings: Vec<String> = dirs
        .iter()
        .filter_map(|d| std::fs::read_dir(d).ok())
        .flat_map(|entries| entries.filter_map(|e| e.ok()))
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    if listings.iter().any(|f| f.ends_with(".part")) {
        return false;
    }
    listings.iter().any(|f| f.ends_with(".gguf") && !is_mmproj_gguf(f))
}

// ---------------------------------------------------------------------------
// Stream steps.
// ---------------------------------------------------------------------------

/// `round(x, 1)`.
fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// Python's `str(float)`: always shows a fractional part.
fn py_float_str(x: f64) -> String {
    if x.is_finite() && x.fract() == 0.0 {
        format!("{x:.1}")
    } else {
        format!("{x}")
    }
}

/// Releases the cancel registration when the download thread unwinds
/// (`finally: cancellation.end(model_id, token)`), panics included.
struct Registration {
    model_id: String,
    token: CancelToken,
}

impl Drop for Registration {
    fn drop(&mut self) {
        cancellation::end(&self.model_id, &self.token);
    }
}

/// The download thread's body (`_do_download`).
fn do_download(
    downloader: &SLPArtifactDownloader,
    model_id: &str,
    is_bootstrap: bool,
    artifacts: Option<ArtifactMap>,
    model_service: Option<Arc<ModelService>>,
    on_progress: ProgressCallback,
) -> anyhow::Result<()> {
    if is_bootstrap {
        // First-run has no enable toggle to click, so bootstrap stays
        // unregistered and uncancellable for now; its downloader keeps the
        // private token its constructor gave it, which nothing ever sets.
        downloader
            .bootstrap_download(None, Some(on_progress))
            .map_err(download_error_into_anyhow)?;
        return Ok(());
    }

    // Claimed and released on this thread, never in the SSE stream.  A window
    // reload closes the stream and drops it there, and a release hung off
    // that would unregister a transfer still consuming bandwidth — the next
    // cancel would answer "not_found" for a download that is very much
    // running.  Owning both ends here is what makes "only an explicit cancel
    // stops a download" true rather than merely intended.
    let token = cancellation::begin(model_id)?;
    let _registration = Registration { model_id: model_id.to_string(), token: token.clone() };
    downloader.set_cancel_token(token);

    let result = downloader
        .download_with_artifacts(model_id, Some(on_progress), artifacts)
        .map(|_| ())
        .map_err(download_error_into_anyhow);
    if let Err(e) = &result {
        if py_error_kind_of(e) == PyErrorKind::Cancelled {
            // Before the registration is released, which is the point of
            // doing it here: the registration is what keeps a second download
            // from creating the directory into a tree that is in the middle
            // of being removed.
            let view = model_service.map(StorageView);
            let discarded = storage_service::discard_partial_download(
                view.as_ref().map(|v| v as &dyn StorageModelService),
                model_id,
            );
            if let Err(err) = discarded {
                // The cancel succeeded either way; bytes left behind are the
                // abandoned-download case the storage sweep already collects.
                tracing::error!(
                    target: LOG_TARGET,
                    "Could not discard the cancelled download of {model_id}: {err}"
                );
            }
        }
    }
    result
}

/// Download model files and stream progress.
///
/// `artifacts` carries a signed-URL map already resolved by the caller (see
/// [`SLPArtifactDownloader::resolve_artifacts`]), so entitlement is settled
/// before this stream opens and the URLs are minted once.  Progress frames
/// are throttled to one per file per 0.5s (completed frames always pass); a
/// keepalive is emitted after ~5s of silence.  A non-bootstrap download
/// claims/releases the cancel registration on the download thread, and a
/// cancelled download discards its partial bytes before re-raising
/// `DownloadCancelled`.
pub fn download_and_stream(
    downloader: Arc<SLPArtifactDownloader>,
    model_id: String,
    model_name: String,
    is_bootstrap: bool,
    artifacts: Option<ArtifactMap>,
    model_service: Option<Arc<ModelService>>,
) -> SseStream {
    Box::pin(async_stream::stream! {
        let progress_queue: Arc<Mutex<VecDeque<Value>>> = Arc::new(Mutex::new(VecDeque::new()));
        let last_progress_time: Arc<Mutex<HashMap<String, Instant>>> = Arc::new(Mutex::new(HashMap::new()));

        let on_progress: ProgressCallback = {
            let queue = Arc::clone(&progress_queue);
            let last = Arc::clone(&last_progress_time);
            Arc::new(move |progress: DownloadProgress| {
                let now = Instant::now();
                {
                    let mut last = last.lock();
                    if progress.status != "completed" {
                        if let Some(prev) = last.get(&progress.filename) {
                            if now.duration_since(*prev) < Duration::from_millis(500) {
                                return;
                            }
                        }
                    }
                    last.insert(progress.filename.clone(), now);
                }
                let mut d = progress.to_dict();
                d["status"] = json!("downloading");
                if progress.total > 0 {
                    let percent = round1(progress.downloaded as f64 / progress.total as f64 * 100.0);
                    d["percent"] = json!(percent);
                    d["message"] = json!(format!(
                        "Downloading {}... {}%",
                        progress.filename,
                        py_float_str(percent)
                    ));
                }
                queue.lock().push_back(d);
            })
        };

        yield Ok(sse_event(
            &json!({
                "status": "downloading",
                "message": format!("Downloading model {}...", display_model_name(&model_name)),
            }),
            "progress",
        ));

        let download_task = {
            let downloader = Arc::clone(&downloader);
            let model_id = model_id.clone();
            tokio::task::spawn_blocking(move || {
                do_download(&downloader, &model_id, is_bootstrap, artifacts, model_service, on_progress)
            })
        };

        // Emit a keepalive when no progress has flowed for a few seconds so
        // the SSE connection survives download stalls (e.g. a dropped
        // connection being retried) without the client or a proxy timing out
        // the idle stream.
        let mut idle_ticks = 0;
        while !download_task.is_finished() {
            let mut drained = false;
            loop {
                let next = progress_queue.lock().pop_front();
                match next {
                    Some(d) => {
                        yield Ok(sse_event(&d, "progress"));
                        drained = true;
                    }
                    None => break,
                }
            }
            if drained {
                idle_ticks = 0;
            } else {
                idle_ticks += 1;
                if idle_ticks >= 10 {
                    // ~5s of silence
                    yield Ok(SSE_KEEPALIVE.to_string());
                    idle_ticks = 0;
                }
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        loop {
            let next = progress_queue.lock().pop_front();
            match next {
                Some(d) => yield Ok(sse_event(&d, "progress")),
                None => break,
            }
        }

        let outcome = match download_task.await {
            Ok(result) => result,
            Err(join_err) => match join_err.try_into_panic() {
                Ok(payload) => Err(anyhow::anyhow!(panic_message(payload.as_ref()))),
                Err(join_err) => Err(anyhow::anyhow!("download task failed: {join_err}")),
            },
        };
        if let Err(e) = outcome {
            yield Err(e);
            return;
        }

        yield Ok(sse_event(
            &json!({
                "status": "download_complete",
                "message": format!("Model {} downloaded.", display_model_name(&model_name)),
            }),
            "progress",
        ));
    })
}

/// Load the orchestrator model into memory with SSE keepalives (every 2s).
///
/// Delegates to `model_service.load_model()`, which always loads the small
/// base model (`api_settings.default_model`) as the orchestrator/main model —
/// regardless of which model was just downloaded or is active.  A larger
/// downloaded model is loaded lazily on demand for large-tier steps.
pub fn load_orchestrator_and_stream(model_service: Arc<ModelService>) -> SseStream {
    Box::pin(async_stream::stream! {
        let name = API_SETTINGS.read().default_model.clone();
        yield Ok(sse_event(
            &json!({
                "status": "loading",
                "message": format!("Loading orchestrator model {}...", display_model_name(&name)),
            }),
            "progress",
        ));

        let load_task = {
            let svc = Arc::clone(&model_service);
            tokio::task::spawn_blocking(move || svc.load_model().map(|_| ()))
        };

        while !load_task.is_finished() {
            yield Ok(SSE_KEEPALIVE.to_string());
            tokio::time::sleep(Duration::from_secs(2)).await;
        }

        // Propagate any load error to the stream.
        let outcome = match load_task.await {
            Ok(result) => result,
            Err(join_err) => match join_err.try_into_panic() {
                Ok(payload) => Err(anyhow::anyhow!(panic_message(payload.as_ref()))),
                Err(join_err) => Err(anyhow::anyhow!("model load failed: {join_err}")),
            },
        };
        if let Err(e) = outcome {
            yield Err(e);
            return;
        }

        let loaded = model_service.model_name.read().clone().unwrap_or_else(|| "None".to_string());
        yield Ok(sse_event(
            &json!({
                "status": "model_loaded",
                "message": format!("Orchestrator model {} loaded.", display_model_name(&loaded)),
            }),
            "progress",
        ));
    })
}

/// Create a bare `general_chat` system project if none exist (no skills/MCP —
/// LLM not loaded yet).
pub fn ensure_project(
    model_service: Arc<ModelService>,
    workspace: SharedWorkspace,
    cwd: Option<String>,
) -> SseStream {
    // Unused in Python too: the check reads the pipeline's own workspace.
    let _ = workspace;
    Box::pin(async_stream::stream! {
        let has_projects = match pipeline_workspace(&model_service) {
            Ok(ws) => !ws.read().projects.is_empty(),
            Err(e) => {
                yield Err(e);
                return;
            }
        };
        if !has_projects {
            yield Ok(sse_event(
                &json!({"status": "creating_project", "message": "Creating default project..."}),
                "progress",
            ));
            let svc = Arc::clone(&model_service);
            let created = blocking(move || {
                svc.create_project(
                    "general_chat",
                    Some("Default general chat project"),
                    true,
                    None,
                    cwd.as_deref(),
                )
            })
            .await;
            if let Err(e) = created {
                yield Err(e);
                return;
            }
            yield Ok(sse_event(&json!({"status": "project_created", "message": "Project created."}), "project"));
        }
    })
}

/// Install MCP servers and generate skills — requires LLM to be loaded.
///
/// Not in a process that has delegated its projects.  The save below carries
/// both halves of the result — the servers `setup_project_tools` registered,
/// and the `system` flag that records the work as done — and it claims no
/// project, so on a delegating supervisor the workspace save drops every one
/// of them.  The flag going with them is what made this re-run on each init,
/// provisioning the same stdio servers again in a process that does not own
/// the project they belong to.
///
/// The project's own agent does it instead, and its save persists: it runs
/// `setup_project_tools` for the project it is pinned to when it loads a
/// model, and `create_project` has already installed them once under its own
/// claim.
///
/// Delegation is checked *before* the workspace is read; skipped when there is
/// no current project or it is already `system`.  Otherwise yields `setup` and
/// `setup_complete`.
pub fn setup_project_services<H>(model_service: Arc<H>) -> SseStream
where
    H: ProjectServicesHost + ?Sized + 'static,
{
    Box::pin(async_stream::stream! {
        if projects_delegated() {
            return;
        }
        let current = match model_service.current_project() {
            Some(current) if !current.system => current,
            _ => return,
        };
        yield Ok(sse_event(
            &json!({"status": "setup", "message": "Setting up project services..."}),
            "progress",
        ));
        let svc = Arc::clone(&model_service);
        let result = blocking(move || {
            svc.setup_project_tools(&current)?;
            svc.mark_current_project_system_and_save()
        })
        .await;
        if let Err(e) = result {
            yield Err(e);
            return;
        }
        yield Ok(sse_event(
            &json!({"status": "setup_complete", "message": "Project setup complete."}),
            "progress",
        ));
    })
}

/// The project info dict, with `fallback_model_name` used when the workspace
/// has no default model registered.
fn project_info(workspace: &Workspace, fallback_model_name: Value) -> Option<Value> {
    let current = workspace.current_project()?;
    let model_name = match workspace.get_model(None) {
        Some(current_model) => Value::String(current_model.model_name),
        None => fallback_model_name,
    };
    Some(json!({
        "id": current.id,
        "name": current.name,
        "description": current.description,
        "model_name": model_name,
        "current": current.current,
        "system": current.system,
    }))
}

/// Return the project info dict for the current project (`None` when there
/// is none): `id`, `name`, `description`, `model_name`, `current`, `system`.
pub fn get_project_info(workspace: &Workspace, model_name: &str) -> Option<Value> {
    project_info(workspace, Value::String(model_name.to_string()))
}

/// Background tasks (indexer, etc.), started if not already running.
///
/// Under supervision no indexer is registered at all (see
/// `ModelService::setup_tasks`), so asking whether one is running would
/// re-run the whole setup on every call — replacing the MCP task and
/// orphaning the loop it had already started.
async fn ensure_tasks(model_service: &Arc<ModelService>, workspace: &SharedWorkspace) -> anyhow::Result<()> {
    if projects_delegated() {
        return Ok(());
    }
    let svc = Arc::clone(model_service);
    let ws = Arc::clone(workspace);
    blocking(move || {
        let queue = svc.task_queue.read().clone();
        if queue.get_task("indexer").is_none() {
            svc.setup_tasks(Some(ws))?;
        }
        Ok(())
    })
    .await
}

/// `POST /v1/init` — initialize the workspace and download the model.
///
/// Downloads the GGUF via the Smartloop artifact endpoints and returns a
/// streaming SSE response with real-time download progress.  The body is
/// validated the way FastAPI validates `InitRequest` (422 in its shape).
pub async fn init(body: Bytes) -> Result<Response, HTTPException> {
    let request: InitRequest = parse_json_body(&body, None)?;
    init_with(request, get_model_service()).await
}

/// [`init`] with an explicit model service (the FastAPI dependency).
///
/// Settles entitlement (signed URLs) before the response starts, so a
/// refusal arrives as an HTTP status (`status_for`); an unknown model is a
/// 400.  Returns a `text/event-stream` response with `Cache-Control:
/// no-cache` and `X-Accel-Buffering: no`.
pub async fn init_with(
    request: InitRequest,
    model_service: Arc<ModelService>,
) -> Result<Response, HTTPException> {
    let unexpected = |e: anyhow::Error| {
        tracing::error!(target: LOG_TARGET, "Exception in /v1/init: {e:#}");
        HTTPException::new(500, e.to_string())
    };

    let workspace = {
        let svc = Arc::clone(&model_service);
        blocking(move || svc.ensure_workspace(None)).await.map_err(unexpected)?
    };
    let default_model = API_SETTINGS.read().default_model.clone();
    let target = request
        .model_name
        .as_deref()
        .map(canonical_model_name)
        .filter(|name| !name.is_empty())
        .unwrap_or(default_model);

    // Register the model in the workspace if not already present
    let existing = workspace.read().get_model(Some(&target));
    let ws_model = match existing {
        Some(ws_model) => ws_model,
        None => {
            let Some(model_cls) = SUPPORTED_MODELS.get(&target) else {
                return Err(HTTPException::new(400, format!("Unknown model: {target}")));
            };
            // Register a project-scoped download disabled: the caller's
            // follow-up PATCH enables it and promotes the app-default from
            // there, so a cancelled download strands nothing.  A project-less
            // init (CLI/TUI) sends no PATCH, so it registers enabled
            // workspace-wide.
            workspace
                .write()
                .add_model(&target, &(model_cls.model_id)(), None, request.project_id.is_none())
        }
    };

    let model_dir = ws_model.model_dir();
    let model_id = ws_model.model_id.clone();

    // Settle entitlement before the response starts.  There is no credential
    // pre-flight — FREE models download anonymously and the platform is the
    // authority on the rest — but the answer has to arrive as a status code:
    // once the streaming response begins, the request has already committed
    // to 200 and a refusal can only be described inside the body.  Resolving
    // here also means the signed URLs are minted once, which matters because
    // the download route is rate-limited per IP.
    let mut downloader: Option<Arc<SLPArtifactDownloader>> = None;
    let mut artifacts: Option<ArtifactMap> = None;
    if !gguf_exists(&model_dir) {
        let made = Arc::new(SLPArtifactDownloader::new(Arc::clone(&workspace), init_logger(), None));
        let resolver = Arc::clone(&made);
        let id = model_id.clone();
        match blocking(move || resolver.resolve_artifacts(&id).map_err(download_error_into_anyhow)).await {
            Ok(resolved) => artifacts = Some(resolved),
            Err(exc) => {
                return Err(match py_error_kind_of(&exc) {
                    // Same classifier the stream's error frame uses.  Keeping
                    // a second rule here is how the two halves of this
                    // endpoint came to disagree: the fallback was 403, so an
                    // unwritable models directory answered "not on your plan"
                    // before the stream and 500 inside it.
                    PyErrorKind::Permission | PyErrorKind::Connection => {
                        HTTPException::new(status_for(as_std_error(&exc)), exc.to_string())
                    }
                    _ => unexpected(exc),
                });
            }
        }
        downloader = Some(made);
    }

    let cwd = request.cwd.clone();
    let steps = {
        let target = target.clone();
        async_stream::try_stream! {
            // 1. Download if needed
            match downloader {
                None => {
                    yield sse_event(
                        &json!({
                            "status": "model_ready",
                            "message": format!("Model {} already available.", display_model_name(&target)),
                        }),
                        "progress",
                    );
                    let svc = Arc::clone(&model_service);
                    let (name, ws) = (target.clone(), Arc::clone(&workspace));
                    blocking(move || svc.mark_model_downloaded(&name, Some(ws))).await?;
                }
                Some(downloader) => {
                    {
                        let mut sub = download_and_stream(
                            downloader,
                            model_id.clone(),
                            target.clone(),
                            false,
                            artifacts,
                            Some(Arc::clone(&model_service)),
                        );
                        while let Some(item) = sub.next().await {
                            yield item?;
                        }
                    }

                    let svc = Arc::clone(&model_service);
                    let (name, ws) = (target.clone(), Arc::clone(&workspace));
                    blocking(move || {
                        svc.mark_model_downloaded(&name, Some(Arc::clone(&ws)))?;

                        // A model earns its place among the app-defaults by
                        // arriving, not by being switched on.  This is the
                        // only branch where a download actually ran, and it is
                        // the event the promotion belongs to: the model is new
                        // to the workspace, so "on by default from now on" is
                        // the answer the user just gave by fetching it.
                        // Switching on a model that was already here is
                        // project-scoped and deliberately does not reach this
                        // (see ``update_project_model``).
                        //
                        // It has to happen here rather than in that route for
                        // a second reason: ``/v1/init`` names no project in
                        // its path, so under supervision it is answered by the
                        // process that owns ``config.json``.  The project route
                        // is proxied to the child pinned to that project,
                        // which cannot write the workspace record at all — a
                        // promotion there was discarded at save.
                        let mut ws = ws.write();
                        ws.promote_model_to_workspace_default(&name);
                        ws.save()?;
                        Ok(())
                    })
                    .await?;
                }
            }

            // ``mark_model_downloaded`` persisted the workspace, so
            // config.json is on disk and the platform service can bind the
            // workspace DB and persist the catalog.
            seed_model_catalog();

            // 2. Ensure model pipeline
            {
                let svc = Arc::clone(&model_service);
                blocking(move || svc.ensure_model()).await?;
            }

            // 3. Create bare project (no skills yet — LLM not loaded)
            {
                let mut sub = ensure_project(Arc::clone(&model_service), Arc::clone(&workspace), cwd);
                while let Some(item) = sub.next().await {
                    yield item?;
                }
            }

            // 4. Start background tasks (indexer, etc.) if not already running.
            ensure_tasks(&model_service, &workspace).await?;

            // 5. Load model (only if a GGUF is available)
            let llm_loaded = model_service.llm.read().is_some();
            let loaded_name = model_service.model_name.read().clone();
            let default_model = API_SETTINGS.read().default_model.clone();
            if !gguf_exists(&model_dir) {
                yield sse_event(
                    &json!({
                        "status": "download_complete",
                        "message": format!(
                            "Model {} downloaded. Build a GGUF to load it.",
                            display_model_name(&target)
                        ),
                    }),
                    "progress",
                );
            } else if llm_loaded && loaded_name.as_deref() == Some(default_model.as_str()) {
                yield sse_event(
                    &json!({
                        "status": "model_loaded",
                        "message": format!(
                            "Orchestrator model {} already loaded.",
                            display_model_name(loaded_name.as_deref().unwrap_or_default())
                        ),
                    }),
                    "progress",
                );
            } else {
                {
                    let mut sub = load_orchestrator_and_stream(Arc::clone(&model_service));
                    while let Some(item) = sub.next().await {
                        yield item?;
                    }
                }

                // 6. Setup MCP servers and skills (requires LLM)
                {
                    let mut sub = setup_project_services(Arc::clone(&model_service));
                    while let Some(item) = sub.next().await {
                        yield item?;
                    }
                }
            }

            // 7. Done
            let project = {
                let ws = pipeline_workspace(&model_service)?;
                let ws = ws.read();
                get_project_info(&ws, &target)
            };
            yield sse_event(
                &json!({
                    "status": "completed",
                    "message": "Model initialized.",
                    "model_name": target,
                    "project": project,
                }),
                "complete",
            );
        }
    };

    let events = async_stream::stream! {
        let steps: BoxStream<'static, anyhow::Result<String>> = Box::pin(steps);
        let mut steps = steps;
        while let Some(item) = steps.next().await {
            match item {
                Ok(chunk) => yield chunk,
                Err(exc) => {
                    if py_error_kind_of(&exc) == PyErrorKind::Cancelled {
                        // Ahead of the catch-all, which would report the
                        // user's own button press as a 500.  Ending here is
                        // also the whole of the cancel's unwinding: every step
                        // above is sequential, so nothing downstream of the
                        // download runs — not ``mark_model_downloaded``, not
                        // the promotion, not the workspace save, the catalog
                        // seed, the pipeline, the project, the tasks, the
                        // orchestrator load or the MCP setup.  The
                        // ``add_model`` registration from before the stream is
                        // undone by ``discard_partial_download``, which clears
                        // the flags it set.
                        tracing::info!(target: LOG_TARGET, "Init cancelled for {target}");
                        yield cancelled_event(&target);
                    } else {
                        tracing::error!(target: LOG_TARGET, "Init failed: {exc:#}");
                        yield error_event(&exc);
                    }
                    break;
                }
            }
        }
    };

    Ok(sse_response(events))
}

/// `POST /v1/bootstrap` — download a model and create a default project.
///
/// No authentication required; the body is optional (an empty or `null` body
/// is `None`).  Intended for first-run scenarios (Electron installer, brew
/// install) where the user has no login token.  Streams SSE progress.
pub async fn bootstrap(body: Bytes) -> Result<Response, HTTPException> {
    let is_blank = body.iter().all(|b| b.is_ascii_whitespace());
    let request = if is_blank || serde_json::from_slice::<Value>(&body).is_ok_and(|v| v.is_null()) {
        None
    } else {
        Some(parse_json_body::<InitRequest>(&body, None)?)
    };
    Ok(bootstrap_with(request, get_model_service()).await)
}

/// The first model in `workspace` whose weights are on disk, preferring the
/// active (default) model, with its `downloaded` flag set.  Each record
/// checked has `models_dir` stamped from the workspace first.
fn find_existing_model(workspace: &SharedWorkspace) -> Option<String> {
    let mut ws = workspace.write();
    let models_dir = ws.models_dir.clone();

    if let Some(active) = ws.get_model_mut(None) {
        active.models_dir = models_dir.clone();
        let md = active.model_dir();
        if !md.is_empty() && gguf_exists(&md) {
            if !active.downloaded {
                active.downloaded = true;
            }
            return Some(active.model_name.clone());
        }
    }

    for wm in ws.models.iter_mut() {
        wm.models_dir = models_dir.clone();
        let md = wm.model_dir();
        if !md.is_empty() && gguf_exists(&md) {
            if !wm.downloaded {
                wm.downloaded = true;
            }
            return Some(wm.model_name.clone());
        }
    }
    None
}

/// [`bootstrap`] with an explicit model service (the FastAPI dependency).
pub async fn bootstrap_with(request: Option<InitRequest>, model_service: Arc<ModelService>) -> Response {
    let cwd = request.and_then(|r| r.cwd);

    let steps = async_stream::try_stream! {
        // 1. Ensure workspace
        yield sse_event(&json!({"status": "checking", "message": "Checking workspace..."}), "progress");
        let workspace = {
            let svc = Arc::clone(&model_service);
            blocking(move || svc.ensure_workspace(None)).await?
        };

        // Fast path: already loaded
        let llm_loaded = model_service.llm.read().is_some();
        let has_current = workspace.read().current_project().is_some();
        if llm_loaded && has_current {
            seed_model_catalog();
            let service_model_name = model_service.model_name.read().clone();
            let (model_name, project) = {
                let ws = workspace.read();
                let model_name = match ws.get_model(None) {
                    Some(active) => Value::String(active.model_name),
                    None => json!(service_model_name),
                };
                let project = project_info(&ws, model_name.clone());
                (model_name, project)
            };
            yield sse_event(
                &json!({
                    "status": "completed",
                    "message": "Model initialized.",
                    "model_name": model_name,
                    "project": project,
                }),
                "complete",
            );
            return;
        }

        // 2. Check for existing GGUF on disk — prefer the active model
        let existing_model_name = find_existing_model(&workspace);
        let existing_model_name = match existing_model_name {
            Some(name) => {
                yield sse_event(
                    &json!({
                        "status": "model_ready",
                        "message": format!("Model {} already available.", display_model_name(&name)),
                    }),
                    "progress",
                );
                name
            }
            None => {
                // 3. Bootstrap download (GGUF only, no build needed)
                let downloader = Arc::new(SLPArtifactDownloader::new(Arc::clone(&workspace), init_logger(), None));
                let source = downloader.bootstrap_source();
                let bootstrap_models = blocking(move || source.list_models()).await?;
                let target = bootstrap_models
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("list index out of range"))?;
                let bootstrap_name = target
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .ok_or_else(|| anyhow::anyhow!("Bootstrap model entry has no name: {target}"))?;

                let bootstrap_model_id = SLPArtifactDownloader::resolve_model_id(&bootstrap_name);
                {
                    let mut ws = workspace.write();
                    ws.add_model(&bootstrap_name, &bootstrap_model_id, None, true);
                    let models_dir = ws.models_dir.clone();
                    if let Some(ws_model) = ws.get_model_mut(Some(&bootstrap_name)) {
                        ws_model.models_dir = models_dir;
                    }
                }

                {
                    let mut sub = download_and_stream(
                        downloader,
                        bootstrap_model_id,
                        bootstrap_name.clone(),
                        true,
                        None,
                        None,
                    );
                    while let Some(item) = sub.next().await {
                        yield item?;
                    }
                }

                let svc = Arc::clone(&model_service);
                let (name, ws) = (bootstrap_name.clone(), Arc::clone(&workspace));
                blocking(move || svc.mark_model_downloaded(&name, Some(ws))).await?;
                bootstrap_name
            }
        };

        // Both branches above mutated this workspace -- a downloaded flag on
        // a model whose weights were already on disk, or a freshly registered
        // bootstrap model -- and only the download branch used to persist it,
        // so the flag the other branch set never reached disk.  Saving here
        // covers both; config.json is then on disk for the platform service
        // to bind the workspace DB against.
        {
            let ws = Arc::clone(&workspace);
            blocking(move || Ok(ws.read().save()?)).await?;
        }
        seed_model_catalog();

        // 4. Ensure model pipeline
        {
            let svc = Arc::clone(&model_service);
            blocking(move || svc.ensure_model()).await?;
        }

        // 5. Start background tasks (indexer, etc.) if not already running.
        ensure_tasks(&model_service, &workspace).await?;

        // 6. Create bare project (no skills yet — LLM not loaded)
        {
            let mut sub = ensure_project(Arc::clone(&model_service), Arc::clone(&workspace), cwd);
            while let Some(item) = sub.next().await {
                yield item?;
            }
        }

        // 7. Load the orchestrator model (always the small base model)
        {
            let mut sub = load_orchestrator_and_stream(Arc::clone(&model_service));
            while let Some(item) = sub.next().await {
                yield item?;
            }
        }

        // 8. Setup MCP servers and skills (requires LLM)
        {
            let mut sub = setup_project_services(Arc::clone(&model_service));
            while let Some(item) = sub.next().await {
                yield item?;
            }
        }

        // 9. Done
        let project = {
            let ws = pipeline_workspace(&model_service)?;
            let ws = ws.read();
            get_project_info(&ws, &existing_model_name)
        };
        yield sse_event(
            &json!({
                "status": "completed",
                "message": "Model initialized.",
                "model_name": existing_model_name,
                "project": project,
            }),
            "complete",
        );
    };

    let events = async_stream::stream! {
        let steps: BoxStream<'static, anyhow::Result<String>> = Box::pin(steps);
        let mut steps = steps;
        while let Some(item) = steps.next().await {
            match item {
                Ok(chunk) => yield chunk,
                Err(exc) => {
                    tracing::error!(target: LOG_TARGET, "Bootstrap failed: {exc:#}");
                    yield error_event(&exc);
                    break;
                }
            }
        }
    };

    sse_response(events)
}
