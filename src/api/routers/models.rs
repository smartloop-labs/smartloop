//! Port of `smartloop/api/routers/models.py`.
//!
//! Models management router: load/unload, the cloud catalog listing, the
//! workspace (app-default) model registry and per-project model settings.

#![allow(unused_imports)]

use std::sync::Arc;

use axum::extract::{Path, Query};
use axum::routing::{get, patch, post};
use axum::{Json, Router};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

use crate::api::routers::_errors::{status_for, HTTPException};
use crate::api::services::{get_model_service, ModelService};
use crate::config::app_settings::AppSettings;
use crate::config::SUPPORTED_MODELS;
use crate::constants::canonical_model_name;
use crate::platform::{get_platform_service, PlatformService};
use crate::types::ModelCapability;
use crate::workspace::workspace::{Project, SharedWorkspace, Workspace};

/// Python logger name (`logging.getLogger(__name__)`).
pub const LOG_TARGET: &str = "smartloop.api.routers.models";

/// Memory-pressure bucket: at or below this share of total RAM a model is
/// "Low" pressure.  The picker draws the same tags from these numbers.
pub const LOW_PRESSURE_PCT: f64 = 10.0;
/// Memory-pressure bucket: below this share of total RAM a model fits.
pub const FITS_PCT: f64 = 80.0;

/// The `/v1` router carrying the models routes (`tags=["models"]`).
pub fn router() -> Router {
    todo!()
}

/// The capabilities a model advertises to the client.
///
/// `ModelCapability::Thinking` is internal-only — it drives the setting and
/// the picker toggle, not routing — so it is stripped from the list.
pub fn display_capabilities<S: AsRef<str>>(caps: &[S]) -> Vec<String> {
    let _ = caps;
    todo!()
}

/// Body of `POST /v1/models/load` (optional; defaults to no project).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ModelLoadRequest {
    #[serde(default)]
    pub project_id: Option<String>,
}

/// `POST /v1/models/load` — load the GGUF/quantised model for serving and
/// keep it in memory until `/v1/models/unload` is called.
///
/// Errors: `PermissionError` → `status_for`, `EnvironmentError`/
/// `ConnectionError` → 502, `ValueError` → 422, else 500.
pub async fn load_model(body: axum::body::Bytes) -> Result<Json<Value>, HTTPException> {
    let _ = body;
    todo!()
}

/// [`load_model`] with an explicit model service.
pub async fn load_model_with(
    request: ModelLoadRequest,
    model_service: Arc<ModelService>,
) -> Result<Value, HTTPException> {
    let _ = (request, model_service);
    todo!()
}

/// `POST /v1/models/unload` — unload the model from memory to free resources.
pub async fn unload_model() -> Json<Value> {
    todo!()
}

/// [`unload_model`] with an explicit model service.
pub async fn unload_model_with(model_service: Arc<ModelService>) -> Value {
    let _ = model_service;
    todo!()
}

/// Mirrors the cloud catalog's access levels exactly (FREE/PRO/INTERNAL).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AccessLevel {
    #[serde(rename = "free")]
    Free,
    #[serde(rename = "pro")]
    Paid,
    /// Not on any plan: staff/preview builds the catalog publishes alongside
    /// the public tiers.  Access is enforced where the weights are served.
    #[serde(rename = "internal")]
    Internal,
}

impl AccessLevel {
    /// The wire value (`"free"`, `"pro"`, `"internal"`).
    pub fn value(&self) -> &'static str {
        match self {
            AccessLevel::Free => "free",
            AccessLevel::Paid => "pro",
            AccessLevel::Internal => "internal",
        }
    }

    /// Parse a wire value; `None` for one this build doesn't know.
    pub fn from_value(value: &str) -> Option<Self> {
        match value {
            "free" => Some(AccessLevel::Free),
            "pro" => Some(AccessLevel::Paid),
            "internal" => Some(AccessLevel::Internal),
            _ => None,
        }
    }
}

/// Drop an access level this build doesn't know instead of failing
/// (`ModelEntry._tolerate_unknown_access_level`): an unknown tier degrades to
/// `None` with a warning so the rest of the catalog still renders.
pub fn tolerate_unknown_access_level<'de, D>(deserializer: D) -> Result<Option<AccessLevel>, D::Error>
where
    D: Deserializer<'de>,
{
    let _ = deserializer;
    todo!()
}

/// One row of `GET /v1/models`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ModelEntry {
    pub name: String,
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
    /// Reasoning is a per-model setting; `None` for a model whose
    /// capabilities lack "thinking".
    #[serde(default)]
    pub thinking: Option<bool>,
    /// Badged in the picker: light enough on this machine to load quickly and
    /// answer fast (see [`mark_recommended`]).
    #[serde(default)]
    pub recommended: bool,
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
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
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
pub async fn list_models(Query(query): Query<ListModelsQuery>) -> Result<Json<Vec<ModelEntry>>, HTTPException> {
    let _ = query;
    todo!()
}

/// Synchronous body of [`list_models`] with explicit dependencies.
///
/// Unparseable catalog rows are skipped with a warning; names are
/// canonicalized; the bootstrap model is omitted; workspace state fills
/// `enabled`/`capabilities`/`thinking`.  Sorted free-tier first, then by
/// parameter size (both descending), then [`mark_recommended`] is applied.
pub fn list_models_with(
    model_service: &ModelService,
    svc: &PlatformService,
    project_id: Option<&str>,
) -> Result<Vec<ModelEntry>, HTTPException> {
    let _ = (model_service, svc, project_id);
    todo!()
}

/// Total physical RAM in GiB, `None` when it cannot be read.
pub fn total_ram_gb() -> Option<f64> {
    todo!()
}

/// Approximate RAM for `entry`: ~1GB per billion parameters (Q4 weights plus
/// context), the same estimate the picker shows.  `None` when unknown.
pub fn footprint_gb(entry: &ModelEntry) -> Option<f64> {
    // The test seam stands in for `SUPPORTED_MODELS` when installed (see
    // `set_param_size_override` at the end of this file).
    let size_b = match param_size_override(&entry.name) {
        Some(size) => size.unwrap_or(0.0),
        None => SUPPORTED_MODELS.get(&entry.name).map(|cls| (cls.param_size_in_b)()).unwrap_or(0.0),
    };
    if size_b > 0.0 { Some(size_b) } else { None }
}

/// Set `recommended` on the models to badge for this machine.
///
/// Every capable model at Low memory pressure; with nothing at Low, the
/// lightest model per capability that fits.  The lightest reasoning model
/// that fits is always recommended.  Within a size only the best-merit model
/// (reasoning first, then most capabilities) keeps the badge.
pub fn mark_recommended(entries: &mut [ModelEntry], total_gb: Option<f64>) {
    let _ = (entries, total_gb);
    todo!()
}

/// One row of `GET /v1/models/workspace`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
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
}

/// `GET /v1/models/workspace` — list models registered in the local
/// workspace (bootstrap model excluded, best model first).
pub async fn list_workspace_models() -> Result<Json<Vec<WorkspaceModelEntry>>, HTTPException> {
    todo!()
}

/// [`list_workspace_models`] with an explicit model service.
pub async fn list_workspace_models_with(
    model_service: &ModelService,
) -> Result<Vec<WorkspaceModelEntry>, HTTPException> {
    let _ = model_service;
    todo!()
}

/// Body of `PATCH /v1/models/workspace/{model_name}`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct UpdateWorkspaceModelRequest {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub thinking: Option<bool>,
    /// Accepted for backward compatibility but ignored.
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
        todo!()
    }

    fn unload_model_by_name(&self, model_name: &str) {
        let _ = model_name;
        todo!()
    }

    fn apply_thinking_setting(&self, model_name: &str) {
        let _ = model_name;
        todo!()
    }
}

/// `PATCH /v1/models/workspace/{model_name}` — axum handler.
pub async fn update_workspace_model_handler(
    Path(model_name): Path<String>,
    Json(request): Json<UpdateWorkspaceModelRequest>,
) -> Result<Json<Value>, HTTPException> {
    let _ = (model_name, request);
    todo!()
}

/// Update a workspace model's enabled state, or whether it reasons.
///
/// `thinking` is rejected (400 `"<name> does not support thinking"`) for a
/// model whose capabilities lack "thinking".  An unknown model is a 404;
/// anything unexpected a 500.
pub async fn update_workspace_model<H: WorkspaceModelHost + ?Sized>(
    model_name: &str,
    request: UpdateWorkspaceModelRequest,
    model_service: &H,
) -> Result<Value, HTTPException> {
    let _ = (model_name, request, model_service);
    todo!()
}

/// One row of `GET /v1/projects/{project_id}/models`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
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
}

/// Body of `PATCH /v1/projects/{project_id}/models/{model_name}`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
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
    let _ = (model_service, project_id);
    todo!()
}

/// `GET /v1/projects/{project_id}/models` — list models registered for a
/// project, with per-project state.
pub async fn list_project_models(
    Path(project_id): Path<String>,
) -> Result<Json<Vec<ProjectModelEntry>>, HTTPException> {
    let _ = project_id;
    todo!()
}

/// [`list_project_models`] with an explicit model service.
pub async fn list_project_models_with(
    project_id: &str,
    model_service: &ModelService,
) -> Result<Vec<ProjectModelEntry>, HTTPException> {
    let _ = (project_id, model_service);
    todo!()
}

/// `PATCH /v1/projects/{project_id}/models/{model_name}` — axum handler.
pub async fn update_project_model(
    Path((project_id, model_name)): Path<(String, String)>,
    Json(request): Json<UpdateProjectModelRequest>,
) -> Result<Json<Value>, HTTPException> {
    let _ = (project_id, model_name, request);
    todo!()
}

/// Update a project's model settings (enabled/active/settings).
///
/// The bootstrap model is rejected with 400; an unknown model is 404.
/// Disabling a model also unloads any resident instance (globally).
pub async fn update_project_model_with(
    project_id: &str,
    model_name: &str,
    request: UpdateProjectModelRequest,
    model_service: &ModelService,
) -> Result<Value, HTTPException> {
    let _ = (project_id, model_name, request, model_service);
    todo!()
}

// ===== BEGIN api_routes_misc: test seam for tests/unit/test_model_recommendation.rs =====

/// Replacement for the `SUPPORTED_MODELS` parameter-size lookup behind
/// [`footprint_gb`]: model name → `param_size_in_b()` (`None` = unknown model).
/// The Python tests patch `models.SUPPORTED_MODELS` with fake classes.
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
