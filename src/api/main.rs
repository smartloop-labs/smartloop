//! Port of `smartloop/api/main.py`.
//!
//! The axum application with lifespan management (startup/shutdown), the
//! `/health`, `/agents` and `/` endpoints, and the server entry points.

#![allow(unused_imports)]

use std::path::PathBuf;
use std::sync::Arc;

use axum::response::Json;
use axum::Router;
use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use serde_json::{Map, Value};
use tokio::task::JoinHandle;

use crate::api::config::{APISettings, API_SETTINGS};
use crate::api::middleware::ProjectScopeMiddleware;
use crate::api::routers;
use crate::api::services::{get_api_client, get_model_service, ModelService};
use crate::config::AppSettings;
use crate::llms::base_llm::BaseLLM;
use crate::model::downloader::artifact_downloader::SLPArtifactDownloader;
use crate::platform::release_agent::run_release_poll_loop;
use crate::platform::token_agent::run_token_refresh_loop;
use crate::supervisor::routing::ProjectRoutingMiddleware;
use crate::supervisor::service::{
    get_pool, get_supervisor, start_supervisor, stop_supervisor, supervision_enabled, Supervisor,
};
use crate::utils::log_utils::print_logo;
use crate::utils::logging_setup::QueueListener;
use crate::workspace::context::{owns_workspace, pinned_project_id};
use crate::workspace::migrations::{adopt_confined_home, migrate_project_state, split_project_state};
use crate::workspace::workspace::SharedWorkspace;
use crate::workspace::Workspace;

/// Logger name of this module.
pub const LOGGER: &str = "smartloop.api.main";

/// Module-level `AppSettings` with `version` set to the crate version.
pub static APP_SETTINGS: Lazy<RwLock<AppSettings>> = Lazy::new(|| {
    let mut settings = AppSettings::new();
    settings.version = Some(crate::VERSION.to_string());
    RwLock::new(settings)
});

/// Port file for persisting the actual bound port (shared in home directory):
/// `<home_dir>/server.port`.
pub static PORT_FILE: Lazy<String> = Lazy::new(|| {
    PathBuf::from(APP_SETTINGS.read().home_dir())
        .join("server.port")
        .to_string_lossy()
        .into_owned()
});

/// Matches tokens in access-log lines so they can be redacted.
pub const TOKEN_RE: &str = r"(access_token|refresh_token)=[^&\s]+";

/// Redact `access_token=` / `refresh_token=` values from an access-log line
/// (`_TokenRedactFilter`), replacing them with `<redacted>`.
pub fn redact_tokens(line: &str) -> String {
    let _ = line;
    todo!()
}

/// GGUF `general.file_type` numeric code → human-readable quantization label.
pub const GGUF_FTYPE_MAP: &[(&str, &str)] = &[
    ("0", "F32"), ("1", "F16"), ("2", "Q4_0"), ("3", "Q4_1"),
    ("7", "Q8_0"), ("8", "Q5_0"), ("9", "Q5_1"), ("10", "Q2_K"),
    ("11", "Q3_K_S"), ("12", "Q3_K_M"), ("13", "Q3_K_L"),
    ("14", "Q4_K_S"), ("15", "Q4_K_M"), ("16", "Q5_K_S"),
    ("17", "Q5_K_M"), ("18", "Q6_K"), ("19", "IQ2_XXS"),
    ("20", "IQ2_XS"), ("21", "IQ3_XXS"), ("22", "IQ1_S"),
    ("23", "IQ4_NL"), ("24", "IQ3_S"), ("25", "IQ2_S"),
    ("26", "IQ4_XS"), ("27", "IQ1_M"), ("28", "BF16"),
];

/// Check whether `llm` is a `LlamaCpp` instance.
fn is_llama_cpp_instance(llm: &dyn BaseLLM) -> bool {
    let _ = llm;
    todo!()
}

/// Return the expected GGUF model file path from the config, or `None`.
pub fn get_model_file_path() -> Option<String> {
    todo!()
}

/// Background task: poll for config and model file, then load the model.
pub async fn wait_for_config_and_load(model_service: Arc<ModelService>) {
    let _ = model_service;
    todo!()
}

/// Failure of [`bootstrap_orchestrator_model`].
#[derive(Debug, thiserror::Error)]
pub enum BootstrapError {
    /// `FileNotFoundError` in Python: the model is missing and this process
    /// does not own the workspace.
    #[error("{0}")]
    FileNotFound(String),
    /// `RuntimeError`: e.g. no bootstrap models available.
    #[error("{0}")]
    Runtime(String),
    /// Any other failure (download, workspace save, ...).
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// What the bootstrap needs from a downloader (seam for tests; implemented
/// by [`SLPArtifactDownloader`]).
pub trait OrchestratorDownloader: Send + Sync {
    /// The models offered by the unauthenticated bootstrap source.
    fn list_bootstrap_models(&self) -> anyhow::Result<Vec<Value>>;
    /// Download `model_name`'s artifacts via the bootstrap source.
    fn bootstrap_download(&self, model_name: &str) -> anyhow::Result<PathBuf>;
}

impl OrchestratorDownloader for SLPArtifactDownloader {
    fn list_bootstrap_models(&self) -> anyhow::Result<Vec<Value>> {
        todo!()
    }

    fn bootstrap_download(&self, model_name: &str) -> anyhow::Result<PathBuf> {
        let _ = model_name;
        todo!()
    }
}

/// Download the orchestrator base model via the unauthenticated bootstrap
/// source and register it in the workspace.
///
/// Used as a fallback when the orchestrator GGUF is missing on disk. Mirrors
/// the core of the `POST /v1/bootstrap` endpoint without the SSE streaming.
///
/// Workspace-owner work: a process pinned to one project refuses with
/// [`BootstrapError::FileNotFound`] ("... does not own the workspace") before
/// touching the workspace or the network.
pub fn bootstrap_orchestrator_model(model_service: &ModelService) -> Result<(), BootstrapError> {
    let _ = model_service;
    todo!()
}

/// [`bootstrap_orchestrator_model`] with injectable workspace and downloader.
///
/// `ensure_workspace` stands in for `model_service._ensure_workspace()` and
/// `make_downloader` for `SLPArtifactDownloader(workspace, logger)`; neither is
/// called when this process does not own the workspace.
pub fn bootstrap_orchestrator_model_with<W, D>(
    ensure_workspace: W,
    make_downloader: D,
) -> Result<(), BootstrapError>
where
    W: FnOnce() -> anyhow::Result<SharedWorkspace>,
    D: FnOnce(&SharedWorkspace) -> Box<dyn OrchestratorDownloader>,
{
    let _ = (ensure_workspace, make_downloader);
    todo!()
}

/// Background tasks and state held between lifespan startup and shutdown.
#[derive(Default)]
pub struct LifespanState {
    pub token_refresh_task: Option<JoinHandle<()>>,
    pub release_poll_task: Option<JoinHandle<()>>,
    pub config_watcher_task: Option<JoinHandle<()>>,
}

/// Lifespan startup: adopt a sandboxed home, load the workspace, start the
/// supervisor when enabled, run migrations, start the workspace-owner loops,
/// pre-load the model (bootstrapping it if missing) and set up the task queue.
pub async fn lifespan_startup() -> LifespanState {
    todo!()
}

/// Lifespan shutdown: cancel background loops, stop the supervisor, shut down
/// tasks, unload the model and finally tear down centralized logging.
pub async fn lifespan_shutdown(state: LifespanState) {
    let _ = state;
    todo!()
}

/// Build the application router: every API router plus `/health`, `/agents`
/// and `/`, wrapped (outermost first) in CORS → project routing → project
/// binding.  Lifespan is not run here; see [`serve`].
pub fn app() -> Router {
    todo!()
}

/// `GET /health` — model, indexing and memory-pressure status.
pub async fn health_check() -> Json<Map<String, Value>> {
    todo!()
}

/// `GET /agents` — what is actually running: the supervisor, the model host,
/// each project.  `{"supervised": false, "agents": []}` without a supervisor.
pub async fn agents() -> Json<Value> {
    todo!()
}

/// `GET /` — root endpoint.
pub async fn root() -> Json<Value> {
    todo!()
}

/// After binding: write the actual port to [`PORT_FILE`] and adopt it in the
/// settings (`_PortCapturingServer.startup`).
pub fn capture_bound_port(port: u16) -> std::io::Result<()> {
    let _ = port;
    todo!()
}

/// Make the port this server bound the one the settings report.
///
/// Started on port 0 the settings still say 0 after the bind, and they are
/// what the MCP OAuth redirect and the in-process MCP base URL are built from.
/// Sets `SLP_API_PORT` (children inherit it), `API_SETTINGS.api_port`, and the
/// supervisor's registry record when one is running.
pub fn adopt_bound_port(port: u16) {
    adopt_bound_port_with(port, get_supervisor())
}

/// [`adopt_bound_port`] with the supervisor supplied explicitly (test seam
/// for `patch.object(main, "get_supervisor", ...)`).
pub fn adopt_bound_port_with(port: u16, supervisor: Option<Arc<Supervisor>>) {
    let _ = (port, supervisor);
    todo!()
}

/// The running log listener (`app.state.log_listener`).
pub static LOG_LISTENER: Lazy<Mutex<Option<Arc<QueueListener>>>> = Lazy::new(|| Mutex::new(None));

/// Install the shared log queue, root queue handler, and parent listener;
/// stashes the listener in [`LOG_LISTENER`] so lifespan shutdown can stop it.
pub fn install_centralized_logging() {
    todo!()
}

/// Bind `host:port`, capture the bound port, run lifespan startup, serve
/// [`app`] until Ctrl-C / shutdown, then run lifespan shutdown.
pub async fn serve(host: &str, port: u16) -> anyhow::Result<()> {
    let _ = (host, port);
    todo!()
}

/// Run the server (for development/direct execution).  Writes/cleans the PID
/// and port files.  Blocking: builds its own tokio runtime.
pub fn run_server() {
    todo!()
}

/// Start the API server in the foreground (blocking; builds its own tokio
/// runtime).  With `service`, auto-restart on crash with exponential backoff
/// (1s doubling to 60s, reset after 120s of stable uptime).
pub fn start_server_foreground(host: &str, port: u16, service: bool) -> anyhow::Result<()> {
    let _ = (host, port, service);
    todo!()
}
