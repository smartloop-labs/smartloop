//! Port of `smartloop/api/routers/models.py`.
//!
//! Models management router: load/unload, the cloud catalog listing, the
//! workspace (app-default) model registry and per-project model settings.

use std::error::Error as StdError;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Path, Query};
use axum::routing::{get, patch, post};
use axum::{Json, Router};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Map, Value};

use crate::api::routers::_errors::{status_for, HTTPException};
use crate::api::services::model_service::ModelServiceError;
use crate::api::services::{get_model_service, ModelService};
use crate::config::app_settings::AppSettings;
use crate::constants::canonical_model_name;
use crate::model::downloader::artifact_downloader::DownloadError;
use crate::model::downloader::cancellation::{DownloadCancelled, DownloadInProgress};
use crate::model::spec::MODEL_SPECS;
use crate::platform::{get_platform_service, ModelAccessError, PlatformError, PlatformService};
use crate::types::{Attention, ModelCapability, ModelKind};
use crate::workspace::workspace::{Project, SharedWorkspace, Workspace, WorkspaceError};

/// Python logger name (`logging.getLogger(__name__)`).
pub const LOG_TARGET: &str = "smartloop.api.routers.models";

/// Memory-pressure bucket: at or below this share of total RAM a model is
/// "Low" pressure.  The picker draws the same tags from these numbers.
pub const LOW_PRESSURE_PCT: f64 = 10.0;
/// Recommendation: at or below this share of *free* memory a model runs
/// comfortably next to whatever else is open; the largest such model is
/// preferred.  Below [`FITS_PCT`] of free memory it can still be recommended
/// when nothing smaller covers its capability.
pub const COMFORTABLE_PCT: f64 = 50.0;
/// Memory-pressure bucket: below this share of total RAM a model fits.
pub const FITS_PCT: f64 = 80.0;

/// The `/v1` router carrying the models routes (`tags=["models"]`).
pub fn router() -> Router {
    Router::new()
        .route("/v1/models/load", post(load_model))
        .route("/v1/models/unload", post(unload_model))
        .route("/v1/models", get(list_models))
        .route("/v1/models/workspace", get(list_workspace_models))
        .route("/v1/models/workspace/{model_name}", patch(update_workspace_model_handler))
        .route("/v1/projects/{project_id}/models", get(list_project_models))
        .route("/v1/projects/{project_id}/models/{model_name}", patch(update_project_model))
}

// ---------------------------------------------------------------------------
// Python exception taxonomy.
//
// The Python routers branch on the exception *class* (`except ValueError`,
// `except PermissionError`, ...).  Rust errors arrive as `anyhow::Error` or a
// typed error; these helpers recover the Python class they stand for, so the
// same branch is taken.  Shared with the init and downloads routers.
// ---------------------------------------------------------------------------

/// The Python exception class an error stands for, as far as the routers'
/// `except` clauses care.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PyErrorKind {
    /// `DownloadCancelled`.
    Cancelled,
    /// `PermissionError` (incl. `ModelAccessError`) — also an `OSError`.
    Permission,
    /// `ConnectionError` — also an `OSError`.
    Connection,
    /// Any other `OSError` (`EnvironmentError`).
    Os,
    /// `ValueError`.
    Value,
    /// `RuntimeError`.
    Runtime,
    /// Anything else.
    Other,
}

/// Classify one error (not its causes); `None` when it says nothing.
fn classify_one(err: &(dyn StdError + 'static)) -> Option<PyErrorKind> {
    if err.downcast_ref::<DownloadCancelled>().is_some() {
        return Some(PyErrorKind::Cancelled);
    }
    if err.downcast_ref::<DownloadInProgress>().is_some() {
        return Some(PyErrorKind::Runtime);
    }
    if err.downcast_ref::<ModelAccessError>().is_some() {
        return Some(PyErrorKind::Permission);
    }
    if err.downcast_ref::<crate::tasks::base_task::PermissionError>().is_some() {
        return Some(PyErrorKind::Permission);
    }
    if let Some(e) = err.downcast_ref::<PlatformError>() {
        return match e {
            PlatformError::Connection { .. } => Some(PyErrorKind::Connection),
            PlatformError::Permission { .. } | PlatformError::ModelAccess(_) => {
                Some(PyErrorKind::Permission)
            }
            PlatformError::Value(_) => Some(PyErrorKind::Value),
            PlatformError::Other(_) => None,
        };
    }
    if let Some(e) = err.downcast_ref::<DownloadError>() {
        // Every shape of a cancel (a bare one, or one carrying its cause).
        if e.is_cancelled() {
            return Some(PyErrorKind::Cancelled);
        }
        #[allow(unreachable_patterns)]
        return match e {
            DownloadError::StaleUrl(_) | DownloadError::Connection { .. } => {
                Some(PyErrorKind::Connection)
            }
            DownloadError::Runtime { .. } => Some(PyErrorKind::Runtime),
            DownloadError::Io(io) => Some(classify_io(io)),
            DownloadError::Value(_) => Some(PyErrorKind::Value),
            DownloadError::Source(inner) => Some(py_error_kind_of(inner)),
            _ => None,
        };
    }
    if let Some(io) = err.downcast_ref::<std::io::Error>() {
        return Some(classify_io(io));
    }
    if let Some(e) = err.downcast_ref::<ModelServiceError>() {
        #[allow(unreachable_patterns)]
        return match e {
            ModelServiceError::Value(_) => Some(PyErrorKind::Value),
            ModelServiceError::Runtime(_) => Some(PyErrorKind::Runtime),
            _ => None,
        };
    }
    if err.downcast_ref::<WorkspaceError>().is_some() {
        return Some(PyErrorKind::Value);
    }
    // ``json.JSONDecodeError`` and the int/float parse failures are
    // ``ValueError`` subclasses.
    if err.downcast_ref::<serde_json::Error>().is_some()
        || err.downcast_ref::<std::num::ParseIntError>().is_some()
        || err.downcast_ref::<std::num::ParseFloatError>().is_some()
    {
        return Some(PyErrorKind::Value);
    }
    None
}

/// The `OSError` subclass an `io::Error` maps to.
fn classify_io(io: &std::io::Error) -> PyErrorKind {
    use std::io::ErrorKind as K;
    match io.kind() {
        K::PermissionDenied => PyErrorKind::Permission,
        // ``BrokenPipeError`` and the ``Connection*Error`` family subclass
        // ``ConnectionError``.
        K::ConnectionRefused | K::ConnectionReset | K::ConnectionAborted | K::BrokenPipe => {
            PyErrorKind::Connection
        }
        _ => PyErrorKind::Os,
    }
}

/// The Python exception class *err* stands for: the first error in its chain
/// that names one (context wrappers say nothing about the class).
pub(crate) fn py_error_kind_of(err: &anyhow::Error) -> PyErrorKind {
    for e in err.chain() {
        if let Some(kind) = classify_one(e) {
            return kind;
        }
    }
    PyErrorKind::Other
}

/// The `&dyn Error` view of an `anyhow::Error`, for [`status_for`].
pub(crate) fn as_std_error(err: &anyhow::Error) -> &(dyn StdError + 'static) {
    let e: &(dyn StdError + Send + Sync + 'static) = err.as_ref();
    e
}

/// Run blocking work off the async runtime (`loop.run_in_executor(None, ...)`),
/// re-raising a panic in the worker as a panic here.
pub(crate) async fn run_blocking<T, F>(f: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(value) => value,
        Err(join_err) => match join_err.try_into_panic() {
            Ok(payload) => std::panic::resume_unwind(payload),
            Err(join_err) => panic!("blocking task failed: {join_err}"),
        },
    }
}

// ---------------------------------------------------------------------------
// FastAPI request validation.
// ---------------------------------------------------------------------------

/// A FastAPI-shaped 422 for one invalid input (`{"detail": [{...}]}`).
pub(crate) fn validation_error(err_type: &str, loc: Value, msg: &str, input: Value) -> HTTPException {
    HTTPException::with_detail(
        422,
        json!([{ "type": err_type, "loc": loc, "msg": msg, "input": input }]),
    )
}

/// Parse a JSON request body the way FastAPI validates a pydantic body
/// parameter: an empty body is `default` (422 "Field required" when there is
/// none), malformed JSON or a mismatched shape is a 422 in FastAPI's
/// validation shape.
pub(crate) fn parse_json_body<T: DeserializeOwned>(
    body: &[u8],
    default: Option<T>,
) -> Result<T, HTTPException> {
    let is_blank = body.iter().all(|b| b.is_ascii_whitespace());
    if is_blank {
        return default.ok_or_else(|| {
            validation_error("missing", json!(["body"]), "Field required", Value::Null)
        });
    }
    let value: Value = serde_json::from_slice(body).map_err(|e| {
        let ctx = e.to_string();
        let mut detail = json!({
            "type": "json_invalid",
            "loc": ["body", e.column().saturating_sub(1)],
            "msg": "JSON decode error",
            "input": {},
        });
        detail["ctx"] = json!({ "error": ctx });
        HTTPException::with_detail(422, Value::Array(vec![detail]))
    })?;
    if value.is_null() {
        if let Some(default) = default {
            return Ok(default);
        }
    }
    if !value.is_object() {
        return Err(validation_error(
            "model_attributes_type",
            json!(["body"]),
            "Input should be a valid dictionary or object to extract fields from",
            value,
        ));
    }
    serde_json::from_value::<T>(value.clone()).map_err(|e| {
        validation_error("value_error", json!(["body"]), &e.to_string(), value)
    })
}

/// The capabilities a model advertises to the client.
///
/// `ModelCapability::Thinking` is internal-only — it drives the setting and
/// the picker toggle, not routing — so it is stripped from the list.
/// `ModelCapability::Tools` is shown: whether a model calls tools natively is
/// worth seeing when choosing one for MCP work.
pub fn display_capabilities<S: AsRef<str>>(caps: &[S]) -> Vec<String> {
    let thinking = ModelCapability::Thinking.value();
    caps.iter()
        .map(|c| c.as_ref())
        .filter(|c| *c != thinking)
        .map(str::to_string)
        .collect()
}

/// Body of `POST /v1/models/load` (optional; defaults to no project).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct ModelLoadRequest {
    #[serde(default)]
    pub project_id: Option<String>,
}

/// `POST /v1/models/load` — load the GGUF/quantised model for serving and
/// keep it in memory until `/v1/models/unload` is called.
///
/// Errors: `PermissionError` → `status_for`, `EnvironmentError`/
/// `ConnectionError` → 502, `ValueError` → 422, else 500.
#[utoipa::path(
    post,
    path = "/v1/models/load",
    tag = "models",
    request_body = ModelLoadRequest,
    summary = "Load the model into memory",
    description = "",
    responses((status = 200, description = "OK", body = serde_json::Value), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn load_model(body: Bytes) -> Result<Json<Value>, HTTPException> {
    let request: ModelLoadRequest = parse_json_body(&body, Some(ModelLoadRequest::default()))?;
    load_model_with(request, get_model_service()).await.map(Json)
}

/// Map a `/models/load` failure to the status the Python route answered.
fn load_error(err: &anyhow::Error) -> HTTPException {
    let kind = py_error_kind_of(err);
    match kind {
        // Not necessarily a 401.  ``PermissionError`` covers a plan refusal
        // (``ModelAccessError``, 403) and, being an ``OSError`` subclass, an
        // unwritable models directory too — answering 401 to either sent the
        // desktop client to a sign-in prompt for something signing in cannot fix.
        PyErrorKind::Permission => HTTPException::new(status_for(as_std_error(err)), err.to_string()),
        PyErrorKind::Connection | PyErrorKind::Os => HTTPException::new(502, err.to_string()),
        PyErrorKind::Value => HTTPException::new(422, err.to_string()),
        _ => {
            tracing::error!(target: LOG_TARGET, "Error in /models/load: {err:#}");
            HTTPException::new(500, err.to_string())
        }
    }
}

/// [`load_model`] with an explicit model service.
pub async fn load_model_with(
    request: ModelLoadRequest,
    model_service: Arc<ModelService>,
) -> Result<Value, HTTPException> {
    // Switch to the requested project, or load current project's model
    if let Some(project_id) = request.project_id.as_deref().filter(|p| !p.is_empty()) {
        let svc = model_service.clone();
        let pid = project_id.to_string();
        if let Err(e) = run_blocking(move || svc.set_current_project(&pid)).await {
            match py_error_kind_of(&e) {
                PyErrorKind::Value | PyErrorKind::Runtime | PyErrorKind::Other => {
                    tracing::warn!(
                        target: LOG_TARGET,
                        "Project not found: {project_id}, using current project"
                    );
                }
                _ => return Err(load_error(&e)),
            }
        }
    }

    // load_model resolves model from the current project
    let svc = model_service.clone();
    if let Err(e) = run_blocking(move || svc.load_model().map(|_| ())).await {
        return Err(load_error(&e));
    }
    let project_name = model_service
        .get_current_project_name()
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "default".to_string());
    Ok(json!({
        "status": "ok",
        "message": format!("Model loaded for inference (project: {project_name})."),
    }))
}

/// `POST /v1/models/unload` — unload the model from memory to free resources.
#[utoipa::path(
    post,
    path = "/v1/models/unload",
    tag = "models",
    summary = "Unload the model from memory",
    description = "",
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
pub async fn unload_model() -> Json<Value> {
    Json(unload_model_with(get_model_service()).await)
}

/// [`unload_model`] with an explicit model service.
pub async fn unload_model_with(model_service: Arc<ModelService>) -> Value {
    let project_name = model_service
        .get_current_project_name()
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "default".to_string());
    let svc = model_service.clone();
    run_blocking(move || svc.unload_model()).await;
    json!({
        "status": "ok",
        "message": format!("Model unloaded (project: {project_name})."),
    })
}

/// Mirrors the cloud catalog's access levels exactly (FREE/FREE_LOGIN/PRO/INTERNAL).
///
/// These values are proxied straight through from the platform API, so the
/// members have to match `ModelAccessLevel` there or a valid row fails
/// validation on arrival.  `premium` is deliberately not a member: it never
/// existed on the platform's enum.  `internal` only ever appears when
/// superuser credentials fetched the catalog.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, utoipa::ToSchema)]
pub enum AccessLevel {
    #[serde(rename = "free")]
    Free,
    /// Free, but only to a signed-in account.
    #[serde(rename = "free_login")]
    FreeLogin,
    #[serde(rename = "pro")]
    Paid,
    /// Not on any plan: staff/preview builds the catalog publishes alongside
    /// the public tiers.  Access is enforced where the weights are served.
    #[serde(rename = "internal")]
    Internal,
}

impl AccessLevel {
    /// The wire value (`"free"`, `"free_login"`, `"pro"`, `"internal"`).
    pub fn value(&self) -> &'static str {
        match self {
            AccessLevel::Free => "free",
            AccessLevel::FreeLogin => "free_login",
            AccessLevel::Paid => "pro",
            AccessLevel::Internal => "internal",
        }
    }

    /// Parse a wire value; `None` for one this build doesn't know.
    pub fn from_value(value: &str) -> Option<Self> {
        match value {
            "free" => Some(AccessLevel::Free),
            "free_login" => Some(AccessLevel::FreeLogin),
            "pro" => Some(AccessLevel::Paid),
            "internal" => Some(AccessLevel::Internal),
            _ => None,
        }
    }
}

/// Python `repr()` of a JSON value, for the warning text.
fn py_repr(value: &Value) -> String {
    match value {
        Value::String(s) => format!("'{s}'"),
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        other => other.to_string(),
    }
}

/// Drop an access level this build doesn't know instead of failing
/// (`ModelEntry._tolerate_unknown_access_level`): an unknown tier degrades to
/// `None` with a warning so the rest of the catalog still renders.
///
/// The catalog is server-side and gains tiers without a client release —
/// `internal` arrived that way and 500'd the whole picker, because one
/// unparseable entry fails the response model for every model in the list.
/// The download path enforces access regardless.
pub fn tolerate_unknown_access_level<'de, D>(deserializer: D) -> Result<Option<AccessLevel>, D::Error>
where
    D: Deserializer<'de>,
{
    let value: Option<Value> = Option::deserialize(deserializer)?;
    let Some(value) = value else {
        return Ok(None);
    };
    let level = value.as_str().and_then(AccessLevel::from_value);
    if level.is_none() {
        tracing::warn!(
            target: LOG_TARGET,
            "[models] unknown access_level {} in catalog — treating as unspecified",
            py_repr(&value)
        );
    }
    Ok(level)
}

/// One row of `GET /v1/models`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct ModelEntry {
    pub name: String,
    /// Empty for a remote model, which has no weights to name.
    #[serde(deserialize_with = "null_as_empty")]
    pub model_id: String,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default, deserialize_with = "tolerate_unknown_access_level")]
    pub access_level: Option<AccessLevel>,
    #[serde(default)]
    pub is_accessible: bool,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// Whether `capabilities` were set here, over the catalog's.
    #[serde(default)]
    pub capabilities_overridden: bool,
    /// Reasoning is a per-model setting, not a per-request one: a model the
    /// user switches thinking on for reasons on every call.  `None` for a
    /// model whose capabilities lack "thinking" — the picker shows the toggle
    /// only when the model can honor it.
    #[serde(default)]
    pub thinking: Option<bool>,
    /// Badged in the picker: light enough on this machine to load quickly and
    /// answer fast.  Decided here, where the RAM and each model's size are
    /// known, rather than re-derived by every client (see [`mark_recommended`]).
    #[serde(default)]
    pub recommended: bool,
    #[serde(rename = "type", default)]
    pub kind: ModelKind,
    /// Base URL a remote model's chat requests go to.  Filled with the
    /// platform API base when the catalog sends none.
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub context_length: Option<u32>,
    /// Parameter count in billions, local or remote; `None` when unknown.
    #[serde(default)]
    pub param_size_b: Option<f64>,
    /// Whether `param_size_b` is an estimate, for the picker to mark.
    #[serde(default)]
    pub param_size_estimated: bool,
    /// Parameters active per token for a mixture-of-experts model; `None`
    /// for a dense one.
    #[serde(default)]
    pub active_param_size_b: Option<f64>,
    #[serde(default, deserialize_with = "tolerate_unknown_attention")]
    pub attention: Option<Attention>,
}

/// An attention type this build doesn't know reads as undeclared rather
/// than failing the whole catalog row.
fn tolerate_unknown_attention<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Attention>, D::Error> {
    Ok(Option::<String>::deserialize(deserializer)?.as_deref().and_then(Attention::from_value))
}

fn null_as_empty<'de, D: Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_default())
}

fn default_true() -> bool {
    true
}

impl ModelEntry {
    /// `ModelEntry(name=..., model_id=...)` with every other field defaulted.
    pub fn new(name: impl Into<String>, model_id: impl Into<String>) -> Self {
        ModelEntry { name: name.into(), model_id: model_id.into(), enabled: true, ..Default::default() }
    }
}

/// Query string of `GET /v1/models`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
pub struct ListModelsQuery {
    /// Resolve enabled state against this project; defaults to workspace
    /// (app-default) state.
    #[serde(default)]
    pub project_id: Option<String>,
}

/// `GET /v1/models` — proxy the cloud model catalog (cached locally),
/// enriched with local state.  Runs on a blocking worker (the platform client
/// is synchronous).  A platform `ConnectionError` → 502 "Platform API
/// unavailable".
///
/// Sync on purpose in Python: the platform client is synchronous, and a 401
/// costs a refresh and a retry on top of the original call.  On the event
/// loop that stalls every in-flight chat stream.
#[utoipa::path(
    get,
    path = "/v1/models",
    tag = "models",
    params(ListModelsQuery),
    summary = "List available models",
    description = "The cloud model catalog, enriched with local download and enabled state.",
    responses((status = 200, description = "OK", body = Vec<ModelEntry>), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn list_models(Query(query): Query<ListModelsQuery>) -> Result<Json<Vec<ModelEntry>>, HTTPException> {
    let model_service = get_model_service();
    let svc = get_platform_service();
    run_blocking(move || list_models_with(&model_service, &svc, query.project_id.as_deref()))
        .await
        .map(Json)
}

/// Synchronous body of [`list_models`] with explicit dependencies.
///
/// Every model the cloud returns is returned here, whatever its access level.
/// Entitlement is described, not enforced: the cloud computes `is_accessible`
/// per-user, and the client decides what to do with a model it says no to.
/// Access is enforced where the weights are actually served (`/v1/init`).
///
/// Unparseable catalog rows are skipped with a warning; names are
/// canonicalized; the bootstrap model is omitted; workspace state fills
/// `enabled`/`capabilities`/`thinking`.  Sorted free, then free-with-login,
/// then the rest, then by
/// parameter size (both descending), then [`mark_recommended`] is applied.
///
/// Only `enabled` resolves against `project_id`; download state is
/// workspace-wide.
pub fn list_models_with(
    model_service: &ModelService,
    svc: &PlatformService,
    project_id: Option<&str>,
) -> Result<Vec<ModelEntry>, HTTPException> {
    let models = match svc.list_models(false) {
        Ok(models) => models,
        Err(e) if e.is_connection() => {
            return Err(HTTPException::new(502, "Platform API unavailable"));
        }
        Err(e) => return Err(HTTPException::new(500, e.to_string())),
    };

    let workspace: Option<SharedWorkspace> = model_service.ensure_workspace(None).ok();
    let ws_guard = workspace.as_ref().map(|ws| ws.read());
    let ws: Option<&Workspace> = ws_guard.as_deref();

    let project: Option<&Project> = match (project_id.filter(|p| !p.is_empty()), ws) {
        (Some(pid), Some(ws)) => ws.projects.iter().find(|p| p.id == pid),
        _ => None,
    };

    let settings = AppSettings::new();
    let bootstrap_model = settings.default_model;

    let mut result: Vec<ModelEntry> = Vec::new();
    for m in models {
        let raw = Value::Object(m);
        let mut entry: ModelEntry = match serde_json::from_value(raw.clone()) {
            Ok(entry) => entry,
            Err(e) => {
                // One unrecognised row must not empty the catalog. The cloud
                // can add a field or an access level ahead of a client
                // release, and a user left with no models at all is a worse
                // failure than a missing one.
                tracing::warn!(
                    target: LOG_TARGET,
                    "Skipping unparseable model catalog entry: {raw}: {e}"
                );
                continue;
            }
        };
        // The cloud catalog may still publish a model under its old name.
        entry.name = canonical_model_name(&entry.name);
        if entry.name == bootstrap_model {
            // Not an entitlement filter: the orchestrator is represented by
            // the client's "Auto" affordance, not as a model you pick.
            continue;
        }
        let ws_model = ws.and_then(|ws| ws.models.iter().rev().find(|wm| wm.model_name == entry.name));
        match (ws, ws_model) {
            (Some(ws), Some(ws_model)) => {
                entry.enabled = ws.effective_model_enabled(&entry.name, project);
                entry.capabilities = display_capabilities(&ws_model.effective_capabilities());
                entry.capabilities_overridden = ws_model.capabilities_override.is_some();
                entry.thinking = ws.effective_model_thinking(&entry.name, project);
            }
            _ => {
                if let Some(spec) = MODEL_SPECS.get(&entry.name) {
                    entry.capabilities = display_capabilities(&spec.capability_values());
                } else {
                    // A row the specs haven't seen keeps the catalog's list,
                    // minus names it can't route.
                    entry.capabilities.retain(|c| ModelCapability::from_value(c).is_some());
                    entry.capabilities = display_capabilities(&entry.capabilities);
                }
            }
        }
        if entry.url.as_deref().is_none_or(str::is_empty) {
            entry.url = Some(settings.slp_api_base.clone());
        }
        if let Some(size) = declared_param_size_b(&entry.name) {
            entry.param_size_b = Some(size);
            entry.param_size_estimated = param_size_estimated(&entry.name);
        }
        if let Some(spec) = MODEL_SPECS.get(&entry.name) {
            entry.active_param_size_b = spec.active_param_size_b;
            entry.attention = spec.attention;
        }
        result.push(entry);
    }
    drop(ws_guard);

    let sort_key = |entry: &ModelEntry| -> (u8, f64) {
        let size = local_param_size_b(&entry.name);
        let tier = match entry.access_level {
            Some(AccessLevel::Free) => 2,
            Some(AccessLevel::FreeLogin) => 1,
            _ => 0,
        };
        (tier, size)
    };
    // ``sorted(..., reverse=True)``: descending, stable for equal keys.
    result.sort_by(|a, b| {
        let (ka, kb) = (sort_key(a), sort_key(b));
        kb.0.cmp(&ka.0).then(kb.1.partial_cmp(&ka.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    mark_recommended(&mut result, free_ram_gb());
    Ok(result)
}

/// Memory free for a model right now in GiB, counting what the OS can
/// reclaim (see [`available_memory_gb`](crate::utils::device_utils::available_memory_gb)),
/// so other open apps shrink it.  `None` when it cannot be read.
pub fn free_ram_gb() -> Option<f64> {
    let (free_gb, total_gb) = crate::utils::device_utils::available_memory_gb();
    if total_gb <= 0.0 || free_gb <= 0.0 {
        return None;
    }
    Some(free_gb)
}

/// Approximate RAM for `entry`: ~1GB per billion parameters (Q4 weights plus
/// context), the same estimate the picker shows.  `None` when unknown.
pub fn footprint_gb(entry: &ModelEntry) -> Option<f64> {
    // The test seam stands in for the model specs when installed (see
    // `set_param_size_override` at the end of this file).
    let size_b = match param_size_override(&entry.name) {
        Some(size) => size.unwrap_or(0.0),
        None => local_param_size_b(&entry.name),
    };
    if size_b > 0.0 { Some(size_b) } else { None }
}

/// Parameter count the spec of `name` declares, local or remote.
fn declared_param_size_b(name: &str) -> Option<f64> {
    MODEL_SPECS.get(name).map(|spec| spec.param_size_b).filter(|size| *size > 0.0)
}

/// Whether the size the spec of `name` declares is an estimate.
fn param_size_estimated(name: &str) -> bool {
    MODEL_SPECS.get(name).is_some_and(|spec| spec.param_size_estimated)
}

/// Parameter count of the local model `name`, 0 when unknown.  A remote
/// model counts as 0: it takes no memory here and sorts after local ones.
fn local_param_size_b(name: &str) -> f64 {
    MODEL_SPECS.get(name).filter(|spec| !spec.is_remote()).map(|spec| spec.param_size_b).unwrap_or(0.0)
}

/// Set `recommended` on the models to badge for this machine.
///
/// `free_gb` is the memory free right now (see [`free_ram_gb`]), not total
/// RAM, so a machine busy with other apps is recommended smaller models.
///
/// Per capability, the largest model that runs comfortably (at or below
/// [`COMFORTABLE_PCT`] of free memory): a bigger model answers better, and
/// one that size still leaves room for the rest.  With nothing comfortable, the
/// lightest model per capability that fits, so a small machine still gets a
/// recommendation.
///
/// Reasoning weighs more than the rest: it is the model that resolves tool
/// calls, matches skills and plans, and without one those steps fall to the
/// small router.  So when no reasoning model is comfortable, the lightest
/// reasoning model that fits is recommended anyway.
///
/// Within a size only one kind of model is recommended, so the user is not
/// left to choose between equal-cost badges: a reasoning model first, then
/// the one with the most capabilities.
pub fn mark_recommended(entries: &mut [ModelEntry], free_gb: Option<f64>) {
    let free_gb = match free_gb {
        Some(gb) if gb != 0.0 => gb,
        _ => return,
    };
    // (pct, index into entries)
    let mut candidates: Vec<(f64, usize)> = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        let Some(gb) = footprint_gb(e) else { continue };
        if !e.is_accessible || e.capabilities.is_empty() {
            continue;
        }
        let pct = gb / free_gb * 100.0;
        if pct < FITS_PCT {
            candidates.push((pct, i));
        }
    }

    let reasoning = ModelCapability::Reasoning.value();
    let has_reasoning = |e: &ModelEntry| e.capabilities.iter().any(|c| c == reasoning);
    // At equal cost, the model that can reason, then the one that does the most.
    let merit = |e: &ModelEntry| (has_reasoning(e), e.capabilities.len());

    // One pick per size: drop a model when one of the same size outranks it,
    // so the user is not left to choose between equal-cost badges.  Only exact
    // ties on merit stay side by side.  Done before picking per capability so
    // a capability the dropped model covered can fall to another size.
    let mut best: Vec<(f64, (bool, usize))> = Vec::new();
    for &(pct, i) in &candidates {
        let m = merit(&entries[i]);
        match best.iter_mut().find(|(p, _)| *p == pct) {
            Some((_, b)) if m > *b => *b = m,
            Some(_) => {}
            None => best.push((pct, m)),
        }
    }
    let mut ranked: Vec<(f64, usize)> = candidates
        .into_iter()
        .filter(|&(pct, i)| best.iter().any(|&(p, b)| p == pct && b == merit(&entries[i])))
        .collect();
    // Lightest first, stable for ties.
    ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Badge each model in `order` that brings a capability not yet covered.
    let badge_per_capability = |entries: &mut [ModelEntry], order: &mut dyn Iterator<Item = usize>| {
        let mut covered: Vec<String> = Vec::new();
        for i in order {
            let e = &mut entries[i];
            if e.capabilities.iter().any(|cap| !covered.contains(cap)) {
                e.recommended = true;
                for cap in &e.capabilities {
                    if !covered.contains(cap) {
                        covered.push(cap.clone());
                    }
                }
            }
        }
    };

    let comfortable: Vec<usize> =
        ranked.iter().rev().filter(|(pct, _)| *pct <= COMFORTABLE_PCT).map(|(_, i)| *i).collect();
    if comfortable.is_empty() {
        // Nothing runs comfortably: the lightest per capability that fits.
        badge_per_capability(entries, &mut ranked.iter().map(|(_, i)| *i));
        return;
    }
    // The largest comfortable model per capability.
    badge_per_capability(entries, &mut comfortable.iter().copied());
    if !comfortable.iter().any(|&i| has_reasoning(&entries[i])) {
        if let Some(&(_, i)) = ranked.iter().find(|(_, i)| has_reasoning(&entries[*i])) {
            entries[i].recommended = true;
        }
    }
}

/// One row of `GET /v1/models/workspace`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct WorkspaceModelEntry {
    pub model_name: String,
    pub model_id: String,
    pub downloaded: bool,
    pub is_active: bool,
    pub enabled: bool,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub thinking: Option<bool>,
    #[serde(rename = "type", default)]
    pub kind: ModelKind,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub context_length: Option<u32>,
    /// Parameter count in billions, local or remote; `None` when unknown.
    #[serde(default)]
    pub param_size_b: Option<f64>,
    /// Whether `param_size_b` is an estimate, for the picker to mark.
    #[serde(default)]
    pub param_size_estimated: bool,
    /// Parameters active per token for a mixture-of-experts model; `None`
    /// for a dense one.
    #[serde(default)]
    pub active_param_size_b: Option<f64>,
    #[serde(default)]
    pub attention: Option<Attention>,
}

/// `sorted(models, key=lambda m: m._rank, reverse=True)`: best model first
/// (largest size, then newer version), stable for equal ranks.
fn sort_by_rank_desc<T>(items: &mut [T], rank: impl Fn(&T) -> (f64, i64)) {
    items.sort_by(|a, b| {
        let (ra, rb) = (rank(a), rank(b));
        rb.0.partial_cmp(&ra.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(rb.1.cmp(&ra.1))
    });
}

/// `GET /v1/models/workspace` — list models registered in the local
/// workspace (bootstrap model excluded, best model first).
#[utoipa::path(
    get,
    path = "/v1/models/workspace",
    tag = "models",
    summary = "List workspace models",
    description = "Models registered in the local workspace.",
    responses((status = 200, description = "OK", body = Vec<WorkspaceModelEntry>), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn list_workspace_models() -> Result<Json<Vec<WorkspaceModelEntry>>, HTTPException> {
    let model_service = get_model_service();
    list_workspace_models_with(&model_service).await.map(Json)
}

/// [`list_workspace_models`] with an explicit model service.
pub async fn list_workspace_models_with(
    model_service: &ModelService,
) -> Result<Vec<WorkspaceModelEntry>, HTTPException> {
    let build = || -> anyhow::Result<Vec<WorkspaceModelEntry>> {
        let settings = AppSettings::new();
        let bootstrap_model = settings.default_model;
        let workspace = model_service.ensure_workspace(None)?;
        let ws = workspace.read();
        // Best model first: largest size, then newer version (see rank).
        let mut models: Vec<_> = ws
            .models
            .iter()
            .filter(|m| canonical_model_name(&m.model_name) != bootstrap_model)
            .collect();
        sort_by_rank_desc(&mut models, |m| m.rank());
        // Remote models list after the local ones, as they route.
        models.sort_by_key(|m| m.is_remote());
        Ok(models
            .into_iter()
            .map(|m| WorkspaceModelEntry {
                model_name: m.model_name.clone(),
                model_id: m.model_id.clone(),
                downloaded: m.downloaded,
                is_active: m.is_active,
                enabled: m.enabled,
                capabilities: display_capabilities(&m.effective_capabilities()),
                thinking: ws.effective_model_thinking(&m.model_name, None),
                kind: m.kind(),
                url: m.url(),
                context_length: m.context_length(),
                param_size_b: declared_param_size_b(&m.model_name),
                param_size_estimated: param_size_estimated(&m.model_name),
                active_param_size_b: m.spec().and_then(|spec| spec.active_param_size_b),
                attention: m.spec().and_then(|spec| spec.attention),
            })
            .collect())
    };
    build().map_err(|e| {
        tracing::error!(target: LOG_TARGET, "Error listing workspace models: {e:#}");
        HTTPException::new(500, e.to_string())
    })
}

/// Body of `PATCH /v1/models/workspace/{model_name}`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct UpdateWorkspaceModelRequest {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub thinking: Option<bool>,
    /// The model's capabilities as set here, over the catalog's and what
    /// its name says, for routing: `["vision", "reasoning"]`. An empty
    /// list takes the override off; `general`, `dp` and `coding` are
    /// accepted and dropped, as text work needs no capability.
    #[serde(default)]
    pub capabilities: Option<Vec<String>>,
}

/// The slice of `ModelService` that [`update_workspace_model`] uses, as a
/// trait so tests can drive the route with a stand-in service.
pub trait WorkspaceModelHost: Send + Sync {
    /// `model_service._ensure_workspace()`.
    fn ensure_workspace(&self) -> anyhow::Result<SharedWorkspace>;
    /// Free any resident instance of `model_name`.
    fn unload_model_by_name(&self, model_name: &str);
    /// Push the stored thinking setting to an already-resident instance.
    fn apply_thinking_setting(&self, model_name: &str);
}

impl WorkspaceModelHost for ModelService {
    fn ensure_workspace(&self) -> anyhow::Result<SharedWorkspace> {
        ModelService::ensure_workspace(self, None)
    }

    fn unload_model_by_name(&self, model_name: &str) {
        ModelService::unload_model_by_name(self, model_name);
    }

    fn apply_thinking_setting(&self, model_name: &str) {
        ModelService::apply_thinking_setting(self, model_name);
    }
}

/// `PATCH /v1/models/workspace/{model_name}` — axum handler.
#[utoipa::path(
    patch,
    path = "/v1/models/workspace/{model_name}",
    tag = "models",
    params(("model_name" = String, Path, description = "Model name")),
    request_body = UpdateWorkspaceModelRequest,
    summary = "Update a workspace model",
    description = "Enable/disable a model, toggle thinking, or set its capabilities.",
    responses((status = 200, description = "OK", body = serde_json::Value), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn update_workspace_model_handler(
    Path(model_name): Path<String>,
    body: Bytes,
) -> Result<Json<Value>, HTTPException> {
    let request: UpdateWorkspaceModelRequest = parse_json_body(&body, None)?;
    let model_service = get_model_service();
    // The route body is synchronous service work (unloading a model frees
    // device memory); keep it off the async workers.
    run_blocking(move || {
        futures::executor::block_on(update_workspace_model(
            &model_name,
            request,
            model_service.as_ref(),
        ))
    })
    .await
    .map(Json)
}

/// Update a workspace model's enabled state, or whether it reasons.
///
/// `thinking` is rejected (400 `"<name> does not support thinking"`) for a
/// model whose capabilities lack "thinking": its chat template never reads
/// `enable_thinking`, so storing the preference would leave the picker
/// showing a toggle that does nothing.  An unknown model is a 404; anything
/// unexpected a 500.
///
/// `capabilities` in the request body is accepted for backward compatibility
/// but ignored — capabilities are read-only from the model spec.
pub async fn update_workspace_model<H: WorkspaceModelHost + ?Sized>(
    model_name: &str,
    request: UpdateWorkspaceModelRequest,
    model_service: &H,
) -> Result<Value, HTTPException> {
    // A status we choose deliberately (e.g. the 400 below) passes straight
    // through; everything else is classified the way the Python route did.
    let outcome = (|| -> Result<Result<Value, HTTPException>, anyhow::Error> {
        // An older client may still address the model by its previous name.
        let model_name = canonical_model_name(model_name);
        let workspace = model_service.ensure_workspace()?;
        if workspace.read().get_model(Some(&model_name)).is_none() {
            return Err(ModelServiceError::Value(format!("Model not found: {model_name}")).into());
        }
        if let Some(enabled) = request.enabled {
            {
                let mut ws = workspace.write();
                // This edits the app-default that new projects are seeded
                // from.  A project with no entry for the model reads through
                // to that default, so pin the current answer everywhere
                // before moving it — the change is meant for projects created
                // later, not existing ones.
                ws.pin_model_for_existing_projects(&model_name);
                if let Some(model) = ws.get_model_mut(Some(&model_name)) {
                    model.enabled = enabled;
                }
            }
            // Deselecting a model must also free any VRAM it still holds —
            // the enabled flag only gates future loads, not an
            // already-resident instance.
            if !enabled {
                model_service.unload_model_by_name(&model_name);
            }
        }
        if let Some(thinking) = request.thinking {
            {
                let mut ws = workspace.write();
                if !ws.supports_thinking(&model_name) {
                    return Ok(Err(HTTPException::new(
                        400,
                        format!("{model_name} does not support thinking"),
                    )));
                }
                ws.pin_model_for_existing_projects(&model_name);
                if let Some(model) = ws.get_model_mut(Some(&model_name)) {
                    model.thinking = Some(thinking);
                }
            }
            // Unlike ``enabled`` — which only gates future loads — this has
            // to reach the instance that is already resident, or the setting
            // would not take effect until the model is next loaded (for the
            // always-resident orchestrator, never).
            model_service.apply_thinking_setting(&model_name);
        }
        if let Some(capabilities) = request.capabilities {
            let mut set: Vec<String> = Vec::new();
            for value in &capabilities {
                let value = value.trim().to_lowercase();
                // `general`, `dp` and `coding`, once set here for what a turn
                // is about, read as `text`.
                let Some(cap) = ModelCapability::from_value(&value) else {
                    let known: Vec<&str> = ModelCapability::ALL.iter().map(|c| c.value()).collect();
                    return Ok(Err(HTTPException::new(
                        400,
                        format!("Unknown capability '{value}'; one of {}", known.join(", ")),
                    )));
                };
                if !set.iter().any(|s| *s == cap.value()) {
                    set.push(cap.value().to_string());
                }
            }
            let mut ws = workspace.write();
            if let Some(model) = ws.get_model_mut(Some(&model_name)) {
                model.capabilities_override = (!set.is_empty()).then_some(set);
            }
        }
        let ws = workspace.read();
        ws.save()?;
        let model = ws
            .get_model(Some(&model_name))
            .ok_or_else(|| ModelServiceError::Value(format!("Model not found: {model_name}")))?;
        Ok(Ok(json!({
            "status": "ok",
            "model_name": model_name,
            "enabled": model.enabled,
            "thinking": ws.effective_model_thinking(&model_name, None),
            "capabilities": display_capabilities(&model.effective_capabilities()),
            "capabilities_overridden": model.capabilities_override.is_some(),
        })))
    })();

    match outcome {
        Ok(result) => result,
        Err(e) => match py_error_kind_of(&e) {
            PyErrorKind::Value => Err(HTTPException::new(404, e.to_string())),
            _ => {
                tracing::error!(target: LOG_TARGET, "Error updating workspace model: {e:#}");
                Err(HTTPException::new(500, e.to_string()))
            }
        },
    }
}

// ---------------------------------------------------------------------------
// Per-project model settings.
//
// Model settings are scoped to a project (the project sidebar owns them),
// seeded from the workspace (app-default) registry at project creation.  The
// legacy `/v1/models/workspace` endpoints above remain as thin workspace-level
// (app-default) accessors and are deprecated in favor of these.
// ---------------------------------------------------------------------------

/// One row of `GET /v1/projects/{project_id}/models`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct ProjectModelEntry {
    pub model_name: String,
    pub model_id: String,
    pub downloaded: bool,
    pub is_active: bool,
    pub enabled: bool,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub settings: Map<String, Value>,
    #[serde(rename = "type", default)]
    pub kind: ModelKind,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub context_length: Option<u32>,
    /// Parameter count in billions, local or remote; `None` when unknown.
    #[serde(default)]
    pub param_size_b: Option<f64>,
    /// Whether `param_size_b` is an estimate, for the picker to mark.
    #[serde(default)]
    pub param_size_estimated: bool,
    /// Parameters active per token for a mixture-of-experts model; `None`
    /// for a dense one.
    #[serde(default)]
    pub active_param_size_b: Option<f64>,
    #[serde(default)]
    pub attention: Option<Attention>,
}

/// Body of `PATCH /v1/projects/{project_id}/models/{model_name}`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct UpdateProjectModelRequest {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub is_active: Option<bool>,
    #[serde(default)]
    pub settings: Option<Map<String, Value>>,
}

/// Return `(workspace, project_id)` for `project_id`, raising 404
/// `"Project not found: <id>"` when the project is unknown.
pub fn resolve_project(
    model_service: &ModelService,
    project_id: &str,
) -> Result<(SharedWorkspace, String), HTTPException> {
    let workspace = model_service.ensure_workspace(None).map_err(|e| {
        tracing::error!(target: LOG_TARGET, "Error resolving project {project_id}: {e:#}");
        HTTPException::new(500, e.to_string())
    })?;
    let found = workspace.read().projects.iter().any(|p| p.id == project_id);
    if !found {
        return Err(HTTPException::new(404, format!("Project not found: {project_id}")));
    }
    Ok((workspace, project_id.to_string()))
}

/// `GET /v1/projects/{project_id}/models` — list models registered for a
/// project, with per-project state.
#[utoipa::path(
    get,
    path = "/v1/projects/{project_id}/models",
    tag = "models",
    params(("project_id" = String, Path, description = "Project ID")),
    summary = "List project models",
    description = "",
    responses((status = 200, description = "OK", body = Vec<ProjectModelEntry>), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn list_project_models(
    Path(project_id): Path<String>,
) -> Result<Json<Vec<ProjectModelEntry>>, HTTPException> {
    let model_service = get_model_service();
    list_project_models_with(&project_id, &model_service).await.map(Json)
}

/// [`list_project_models`] with an explicit model service.
///
/// Entries are seeded from the workspace (app-default) registry; a project's
/// own settings override those defaults.  Capabilities are read-only from the
/// model spec, and so is `downloaded` — weights are shared workspace-wide.
pub async fn list_project_models_with(
    project_id: &str,
    model_service: &ModelService,
) -> Result<Vec<ProjectModelEntry>, HTTPException> {
    let (workspace, project_id) = resolve_project(model_service, project_id)?;
    let build = || -> anyhow::Result<Vec<ProjectModelEntry>> {
        let mut ws = workspace.write();
        ws.sync_project_models(&project_id);
        let ws = &*ws;
        let project = ws.project(&project_id);
        let mut models: Vec<_> =
            ws.models.iter().filter(|m| !ws.is_bootstrap_model(&m.model_name)).collect();
        sort_by_rank_desc(&mut models, |m| m.rank());
        // Remote models list after the local ones, as they route.
        models.sort_by_key(|m| m.is_remote());
        Ok(models
            .into_iter()
            .map(|m| ProjectModelEntry {
                model_name: m.model_name.clone(),
                model_id: m.model_id.clone(),
                downloaded: ws.effective_model_downloaded(&m.model_name),
                is_active: ws.effective_model_active(&m.model_name, project),
                enabled: ws.effective_model_enabled(&m.model_name, project),
                capabilities: display_capabilities(&m.effective_capabilities()),
                settings: ws
                    .project_model(project, &m.model_name)
                    .map(|pm| pm.settings.clone())
                    .unwrap_or_default(),
                kind: m.kind,
                url: m.url(),
                context_length: m.context_length(),
                param_size_b: declared_param_size_b(&m.model_name),
                param_size_estimated: param_size_estimated(&m.model_name),
                active_param_size_b: m.spec().and_then(|spec| spec.active_param_size_b),
                attention: m.spec().and_then(|spec| spec.attention),
            })
            .collect())
    };
    build().map_err(|e| {
        tracing::error!(target: LOG_TARGET, "Error listing project models: {e:#}");
        HTTPException::new(500, e.to_string())
    })
}

/// `PATCH /v1/projects/{project_id}/models/{model_name}` — axum handler.
#[utoipa::path(
    patch,
    path = "/v1/projects/{project_id}/models/{model_name}",
    tag = "models",
    params(("project_id" = String, Path, description = "Project ID"), ("model_name" = String, Path, description = "Model name")),
    request_body = UpdateProjectModelRequest,
    summary = "Update a project model",
    description = "Enable/disable, make active, or change per-project settings.",
    responses((status = 200, description = "OK", body = serde_json::Value), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn update_project_model(
    Path((project_id, model_name)): Path<(String, String)>,
    body: Bytes,
) -> Result<Json<Value>, HTTPException> {
    let request: UpdateProjectModelRequest = parse_json_body(&body, None)?;
    let model_service = get_model_service();
    // Synchronous service work (an unload frees device memory, the save
    // writes to disk); keep it off the async workers.
    run_blocking(move || {
        futures::executor::block_on(update_project_model_with(
            &project_id,
            &model_name,
            request,
            &model_service,
        ))
    })
    .await
    .map(Json)
}

/// Index of `model_name`'s entry in `project.model_settings`
/// (`Workspace.project_model`, which canonicalizes both sides).
fn project_model_index(project: &Project, model_name: &str) -> Option<usize> {
    let target = canonical_model_name(model_name);
    project
        .model_settings
        .iter()
        .position(|m| canonical_model_name(&m.model_name) == target)
}

/// Update a project's model settings (enabled/active/settings).
///
/// Per-project overrides take precedence over the workspace (app-default)
/// registry.  The bootstrap model is rejected with 400; an unknown model is
/// 404.  Disabling a model also unloads any resident instance — VRAM is
/// shared across projects, so the unload stays global.  Download state is not
/// settable here: weights are added and removed workspace-wide.
pub async fn update_project_model_with(
    project_id: &str,
    model_name: &str,
    request: UpdateProjectModelRequest,
    model_service: &ModelService,
) -> Result<Value, HTTPException> {
    // An older client may still address the model by its previous name.
    let model_name = canonical_model_name(model_name);
    let (workspace, project_id) = resolve_project(model_service, project_id)?;

    let internal = |e: anyhow::Error| {
        tracing::error!(target: LOG_TARGET, "Error updating project model: {e:#}");
        HTTPException::new(500, e.to_string())
    };
    let no_entry = || {
        internal(anyhow::anyhow!("'NoneType' object has no attribute 'enabled'"))
    };

    let model = {
        let mut ws = workspace.write();
        if ws.is_bootstrap_model(&model_name) {
            // The orchestrator model is always enabled and is filtered out of
            // the list responses above, so it is not a per-project setting.
            // Reject explicitly rather than writing an entry that would
            // shadow its always-on workspace state.
            return Err(HTTPException::new(
                400,
                format!(
                    "Model {model_name} is always enabled and cannot be configured per project."
                ),
            ));
        }
        let Some(model) = ws.get_model(Some(&model_name)) else {
            return Err(HTTPException::new(404, format!("Model not found: {model_name}")));
        };
        ws.sync_project_models(&project_id);

        if let Some(enabled) = request.enabled {
            // Project-scoped, and only that.  Turning on a model that is
            // already in the workspace says nothing about what other projects
            // should start with, so nothing here touches the app-default — a
            // model earns that by being *downloaded* (see /v1/init), not by
            // being switched on somewhere.
            let project = ws.project_mut(&project_id).ok_or_else(no_entry)?;
            let idx = project_model_index(project, &model_name).ok_or_else(no_entry)?;
            project.model_settings[idx].enabled = enabled;
        }
        model
    };

    if request.enabled == Some(false) {
        // Deselecting must also free any VRAM it still holds — the enabled
        // flag only gates future loads, not an already-resident instance.
        // This half stays workspace-wide: the device is.
        model_service.unload_model_by_name(&model_name);
    }

    let ws = {
        let mut ws = workspace.write();
        {
            let project = ws.project_mut(&project_id).ok_or_else(no_entry)?;
            let idx = project_model_index(project, &model_name).ok_or_else(no_entry)?;

            if let Some(is_active) = request.is_active {
                if is_active {
                    for (i, other) in project.model_settings.iter_mut().enumerate() {
                        other.is_active = i == idx;
                    }
                } else {
                    project.model_settings[idx].is_active = false;
                }
            }

            if let Some(settings) = &request.settings {
                let pm = &mut project.model_settings[idx];
                for (k, v) in settings {
                    pm.settings.insert(k.clone(), v.clone());
                }
            }
        }
        parking_lot::RwLockWriteGuard::downgrade(ws)
    };

    ws.save().map_err(|e| internal(e.into()))?;
    let project = ws.project(&project_id).ok_or_else(no_entry)?;
    let idx = project_model_index(project, &model_name).ok_or_else(no_entry)?;
    let pm = &project.model_settings[idx];
    Ok(json!({
        "status": "ok",
        "project_id": project_id,
        "model_name": model_name,
        "enabled": pm.enabled,
        "downloaded": ws.effective_model_downloaded(&model_name),
        "is_active": pm.is_active,
        "capabilities": display_capabilities(&model.effective_capabilities()),
        "settings": Value::Object(pm.settings.clone()),
    }))
}

// ===== BEGIN api_routes_misc: test seam for tests/unit/test_model_recommendation.rs =====

/// Replacement for the model-spec parameter-size lookup behind
/// [`footprint_gb`]: model name → parameter count (`None` = unknown model).
pub type ParamSizeLookup = Arc<dyn Fn(&str) -> Option<f64> + Send + Sync>;

static PARAM_SIZE_OVERRIDE: once_cell::sync::Lazy<parking_lot::RwLock<Option<ParamSizeLookup>>> =
    once_cell::sync::Lazy::new(|| parking_lot::RwLock::new(None));

/// Install (or with `None`, remove) the parameter-size lookup override.
pub fn set_param_size_override(lookup: Option<ParamSizeLookup>) {
    *PARAM_SIZE_OVERRIDE.write() = lookup;
}

/// The overridden parameter size of *name*: `None` when no override is
/// installed, `Some(None)` when the override does not know the model.
pub fn param_size_override(name: &str) -> Option<Option<f64>> {
    let lookup = PARAM_SIZE_OVERRIDE.read().clone();
    lookup.map(|lookup| lookup(name))
}

// ===== END api_routes_misc =====
