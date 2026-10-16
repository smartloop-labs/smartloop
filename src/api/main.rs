//! Port of `smartloop/api/main.py`.
//!
//! The axum application with lifespan management (startup/shutdown), the
//! `/health`, `/agents` and `/` endpoints, and the server entry points.

use std::net::SocketAddr;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{ConnectInfo, Request};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::get;
use axum::Router;
use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use regex::Regex;
use serde_json::{json, Map, Value};
use tokio::sync::Notify;
use tokio::task::JoinHandle;

use crate::api::config::{api_settings, API_SETTINGS};
use crate::api::middleware::ProjectScopeMiddleware;
use crate::api::routers;
use crate::api::services::{get_api_client, get_model_service, ModelService};
use crate::config::AppSettings;
use crate::llms::base_llm::BaseLLM;
use crate::llms::llama_cpp::LlamaCpp;
use crate::model::downloader::artifact_downloader::SLPArtifactDownloader;
use crate::platform::release_agent::run_release_poll_loop;
use crate::platform::token_agent::run_token_refresh_loop;
use crate::supervisor::routing::{with_project_routing, AgentPool, PoolProvider};
use crate::supervisor::service::{
    get_pool, get_supervisor, start_supervisor, stop_supervisor, supervision_enabled, Supervisor,
};
use crate::utils::device_utils;
use crate::utils::log_utils::print_logo;
use crate::utils::logging_setup::{
    get_logger, make_log_queue, setup_root_logging, shutdown_logging, start_listener, ListenerOptions,
    QueueListener,
};
use crate::workspace::context::{owns_workspace, pinned_project_id};
use crate::workspace::migrations::{adopt_confined_home, migrate_project_state, split_project_state};
use crate::workspace::workspace::SharedWorkspace;
use crate::workspace::{DocumentStatus, Workspace};
pub struct LogTarget;

impl LogTarget {
    pub const ACCESS: &str = "smartloop.api.access";
    pub const SERVER: &str = "smartloop.api.server";
    pub const MAIN: &str = "smartloop.api.main";
}

/// App title / description / version (the FastAPI metadata).  The OpenAPI
/// document and Swagger UI live in [`crate::api::openapi`].
pub const APP_TITLE: &str = "Smartloop";
pub const APP_DESCRIPTION: &str = "Smartloop local LLM inference framework - OpenAI-compatible API server";

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

static TOKEN_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(TOKEN_RE).expect("valid regex"));

/// Redact `access_token=` / `refresh_token=` values from an access-log line
/// (`_TokenRedactFilter`), replacing them with `<redacted>`.
pub fn redact_tokens(line: &str) -> String {
    TOKEN_REGEX.replace_all(line, "${1}=<redacted>").into_owned()
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

/// Quantization hint in a GGUF file name (e.g. `Q4_K_M`), case-insensitive.
static QUANT_IN_NAME: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)[_\-\.]((?:I?Q\d[_A-Z\d]*|F16|F32|BF16))").expect("valid regex"));

/// Python's `str(bool)`, for log lines that printed one.
fn py_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

/// The message of a caught panic payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Run a best-effort step, turning a panic into an error message (the
/// Python wrapped these in `try/except Exception`).
fn best_effort<R>(f: impl FnOnce() -> R) -> Result<R, String> {
    std::panic::catch_unwind(AssertUnwindSafe(f)).map_err(|p| panic_message(p.as_ref()))
}

/// Run blocking work on the blocking pool, mapping a panic to an error.
async fn blocking<R, F>(f: F) -> Result<R, String>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(value) => Ok(value),
        Err(e) if e.is_panic() => Err(panic_message(e.into_panic().as_ref())),
        Err(e) => Err(e.to_string()),
    }
}

/// Check whether `llm` is a `LlamaCpp` instance.
fn is_llama_cpp_instance(llm: &dyn BaseLLM) -> bool {
    llm.as_any().downcast_ref::<LlamaCpp>().is_some()
}

/// Return the expected GGUF model file path from the config, or `None`.
pub fn get_model_file_path() -> Option<String> {
    if !Path::new(&Workspace::config_file()).exists() {
        return None;
    }
    let workspace = Workspace::load()?;
    let active_model = workspace.models.iter().find(|m| m.is_active)?;
    Some(
        Path::new(&workspace.models_dir)
            .join(&active_model.model_id)
            .join(format!("{}.gguf", active_model.model_id))
            .to_string_lossy()
            .into_owned(),
    )
}

/// Background task: poll for config and model file, then load the model.
pub async fn wait_for_config_and_load(model_service: Arc<ModelService>) {
    tracing::info!(target: LogTarget::ACCESS, "Waiting for config and model file to be ready...");
    loop {
        if !Path::new(&Workspace::config_file()).exists() {
            tokio::time::sleep(Duration::from_secs(2)).await;
            continue;
        }
        if let Some(model_file) = get_model_file_path()
            && Path::new(&model_file).exists()
        {
            break;
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    tracing::info!(target: LogTarget::MAIN, "Config and model file detected! Loading model...");
    match blocking(move || model_service.load_model()).await {
        Ok(Ok(_)) => tracing::info!(target: LogTarget::MAIN, "Model loaded successfully."),
        Ok(Err(e)) => tracing::error!(target: LogTarget::MAIN, "Failed to load model: {}", e),
        Err(e) => tracing::error!(target: LogTarget::MAIN, "Failed to load model: {}", e),
    }
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
        self.bootstrap_source().list_models()
    }

    fn bootstrap_download(&self, model_name: &str) -> anyhow::Result<PathBuf> {
        SLPArtifactDownloader::bootstrap_download(self, Some(model_name), None).map_err(anyhow::Error::from)
    }
}

/// Download the orchestrator base model via the unauthenticated bootstrap
/// source and register it in the workspace.
///
/// Used as a fallback when the orchestrator GGUF is missing on disk (e.g. a
/// workspace created before the model was bootstrapped). Mirrors the core of
/// the `POST /v1/bootstrap` endpoint without the SSE streaming.
///
/// Workspace-owner work, like the token refresh and release polling: the
/// weights live in one shared `models_dir` and the record of them lives in
/// `config.json`, which a per-project agent never writes.  A child that ran
/// this would download a copy of the weights per project and then lose the
/// `downloaded` flag at save, so the next start would do it again.  Children
/// normally never reach here because the pool sets `SLP_API_PRELOAD_MODEL=0`
/// — but only when there is a model host to reach, so with the host disabled
/// or failed they inherit the supervisor's preload setting and do arrive.
pub fn bootstrap_orchestrator_model(model_service: &ModelService) -> Result<(), BootstrapError> {
    bootstrap_orchestrator_model_with(
        || model_service.ensure_workspace(None),
        |workspace| {
            Box::new(SLPArtifactDownloader::new(workspace.clone(), Arc::new(get_logger(LogTarget::MAIN)), None))
                as Box<dyn OrchestratorDownloader>
        },
    )
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
    if let Some(pinned) = pinned_project_id() {
        tracing::warn!(
            target: LogTarget::MAIN,
            "[bootstrap] orchestrator model is missing, but this process serves only project {}; \
             leaving the download to the workspace owner",
            pinned
        );
        return Err(BootstrapError::FileNotFound(
            "orchestrator model is not downloaded and this process does not own the workspace".to_string(),
        ));
    }

    let workspace = ensure_workspace()?;
    let downloader = make_downloader(&workspace);
    let bootstrap_models = downloader.list_bootstrap_models()?;
    let Some(target) = bootstrap_models.first() else {
        return Err(BootstrapError::Runtime(
            "No bootstrap models available from the bootstrap source".to_string(),
        ));
    };

    let name = target.get("name").and_then(Value::as_str).unwrap_or_default().to_string();
    let model_id = SLPArtifactDownloader::resolve_model_id(&name);

    tracing::info!(target: LogTarget::MAIN, "Bootstrapping orchestrator model '{}'...", name);
    downloader.bootstrap_download(&name)?;

    // Register (or reuse) the workspace entry and mark it downloaded so the
    // subsequent load_model() and config watchers find it. add_model is
    // alias-aware, so an existing legacy-named entry is reused, not duplicated.
    {
        let mut ws = workspace.write();
        ws.add_model(&name, &model_id, None, true);
        let models_dir = ws.models_dir.clone();
        if let Some(ws_model) = ws.get_model_mut(Some(&name)) {
            ws_model.models_dir = models_dir;
            ws_model.downloaded = true;
        }
        ws.save().map_err(anyhow::Error::from)?;
    }
    tracing::info!(target: LogTarget::MAIN, "Orchestrator model '{}' bootstrapped successfully", name);
    Ok(())
}

/// True when `err` is (or wraps) a "file not found" — Python's
/// `FileNotFoundError`, which `load_model` raises when the orchestrator GGUF
/// is not on disk yet.
fn is_file_not_found(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause.downcast_ref::<std::io::Error>().is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound)
            || matches!(cause.downcast_ref::<BootstrapError>(), Some(BootstrapError::FileNotFound(_)))
    })
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
    let settings = api_settings();
    tracing::info!(target: LogTarget::MAIN, "Starting Smartloop API server...");
    tracing::info!(target: LogTarget::MAIN, "Debug mode setting: {}", py_bool(settings.debug_mode));

    let home_dir = settings.home_dir();
    let api_port = settings.api_port;
    let workspace = blocking(move || {
        // The framework home is ~/.smartloop in every build, so the app and
        // the CLI share one workspace.  A build that shipped before the home
        // was resolved past the sandbox left its state under the rewritten
        // $HOME — inside the App Sandbox container, or under
        // ~/snap/smartloop/<rev> — so bring that forward before anything
        // reads the home.  Best-effort: a sandbox that denies the real home
        // leaves the install running where it already is, which is worse
        // than the move but better than not starting.
        if let Err(e) = best_effort(|| adopt_confined_home(&home_dir)) {
            tracing::error!(target: LogTarget::MAIN, "Adopting the sandboxed home failed; continuing startup: {}", e);
        }

        let mut workspace = Workspace::load();

        // When supervising, this process owns the port, the workspace record
        // and the model host — and spawns a child per project rather than
        // serving projects itself.  Off unless SLP_SUPERVISOR says otherwise,
        // so an existing install is the single server it has always been.
        if supervision_enabled()
            && let Err(e) = best_effort(|| start_supervisor(api_port, None))
        {
            tracing::error!(target: LogTarget::MAIN, "Supervisor failed to start; serving projects in-process: {}", e);
        }

        // Skills/documents moved from the user-space folder into project_dir
        // so they follow the framework's home (SLP_HOME, else ~/.smartloop).
        // Fold in anything left at the old location.  Best-effort: the old
        // path may be unreadable under the sandbox, which is precisely the
        // case this migration exists for, so it must never block startup.
        if let Err(e) = best_effort(|| migrate_project_state(workspace.as_ref())) {
            tracing::error!(target: LogTarget::MAIN, "Project state migration failed; continuing startup: {}", e);
        }

        // Break the single config.json into a workspace record plus one
        // project.json per project, so a project's state is only ever written
        // by whoever owns that project.  Idempotent, and the pre-split file is
        // kept — see split_project_state for the rollback.  Best-effort for
        // the same reason as above: a workspace that cannot be split still
        // runs.
        if let Err(e) = best_effort(|| split_project_state(workspace.as_mut())) {
            tracing::error!(target: LogTarget::MAIN, "Project state split failed; continuing startup: {}", e);
        }
        workspace
    })
    .await
    .unwrap_or_else(|e| {
        tracing::error!(target: LogTarget::MAIN, "Loading the workspace failed: {}", e);
        None
    });

    if settings.debug_mode {
        tracing::info!(target: LogTarget::MAIN, "🔧 Debug mode enabled - will load base model with LoRA adapters");
    }

    // Credentials and releases are workspace-level, and belong to whoever
    // owns the workspace.  A per-project agent skips them: N children polling
    // the same release endpoint and refreshing the same token is N times the
    // work for one answer.  Whoever owns the workspace does it once.
    //
    // This gate is about the duplicated work, not about safety.  The race it
    // used to also describe — children writing the credential store
    // underneath each other — is closed at the redemption itself, which a
    // pinned process declines outright; see refresh_access_token_now.
    let workspace_owner = owns_workspace();
    let mut state = LifespanState {
        token_refresh_task: workspace_owner.then(|| tokio::spawn(run_token_refresh_loop())),
        release_poll_task: workspace_owner.then(|| tokio::spawn(run_release_poll_loop())),
        config_watcher_task: None,
    };

    let model_service = get_model_service();

    if workspace.is_some() && !settings.preload_model {
        // Declared since the API server was written and never consulted.  It
        // earns its keep now: a per-project agent that reaches the device
        // through a model host has no reason to load weights of its own, and
        // bringing one up should not wait on (or trigger) a download.
        tracing::info!(target: LogTarget::MAIN, "Model pre-load disabled (slp_api_preload_model=0)");
    } else if workspace.is_some() {
        tracing::info!(target: LogTarget::MAIN, "Pre-loading model for current project...");
        let service = Arc::clone(&model_service);
        let outcome = blocking(move || match service.load_model() {
            Ok(_) => {
                tracing::info!(target: LogTarget::MAIN, "Model pre-loaded successfully");
            }
            Err(e) if is_file_not_found(&e) => {
                // The orchestrator base model GGUF isn't on disk yet. Pull it
                // via the unauthenticated bootstrap source and retry, rather
                // than leaving the server running with no model loaded.
                tracing::warn!(
                    target: LogTarget::MAIN,
                    "Orchestrator model missing ({}); attempting bootstrap download...",
                    e
                );
                let retried = bootstrap_orchestrator_model(&service)
                    .map_err(anyhow::Error::from)
                    .and_then(|()| service.load_model().map(|_| ()));
                match retried {
                    Ok(()) => tracing::info!(target: LogTarget::MAIN, "Model pre-loaded successfully after bootstrap"),
                    Err(e2) => tracing::error!(target: LogTarget::MAIN, "Failed to bootstrap and pre-load model: {}", e2),
                }
            }
            Err(e) => {
                tracing::error!(target: LogTarget::MAIN, "Failed to pre-load model: {}", e);
            }
        })
        .await;
        if let Err(e) = outcome {
            tracing::error!(target: LogTarget::MAIN, "Failed to pre-load model: {}", e);
        }
    } else {
        state.config_watcher_task = Some(tokio::spawn(wait_for_config_and_load(Arc::clone(&model_service))));
    }

    // Set up tasks (indexer, build, train) — the downloader task is
    // registered lazily when /v1/init is called.
    if let Some(workspace) = workspace {
        let service = Arc::clone(&model_service);
        let shared: SharedWorkspace = Arc::new(RwLock::new(workspace));
        match blocking(move || service.setup_tasks(Some(shared))).await {
            Ok(Ok(())) => tracing::info!(target: LogTarget::MAIN, "Task queue initialized"),
            Ok(Err(e)) => tracing::warn!(target: LogTarget::MAIN, "Could not initialize agents: {}", e),
            Err(e) => tracing::warn!(target: LogTarget::MAIN, "Could not initialize agents: {}", e),
        }
    }

    state
}

/// Lifespan shutdown: cancel background loops, stop the supervisor, shut down
/// tasks, unload the model and finally tear down centralized logging.
pub async fn lifespan_shutdown(state: LifespanState) {
    for task in [state.token_refresh_task, state.release_poll_task].into_iter().flatten() {
        task.abort();
    }
    if let Some(task) = state.config_watcher_task
        && !task.is_finished()
    {
        task.abort();
    }

    tracing::info!(target: LogTarget::MAIN, "Shutting down Smartloop API server...");

    // Children first: they are using the model host this also stops.
    if let Err(e) = blocking(stop_supervisor).await {
        tracing::error!(target: LogTarget::MAIN, "Supervisor shutdown failed; continuing: {}", e);
    }

    let model_service = get_model_service();
    if let Err(e) = blocking(move || {
        model_service.shutdown_tasks();
        model_service.unload_model();
    })
    .await
    {
        tracing::error!(target: LogTarget::MAIN, "Shutdown failed: {}", e);
    }

    tracing::info!(target: LogTarget::MAIN, "Model unloaded, shutdown complete");

    // Tear down centralized logging last so the lines above still flush
    // through the rotating file handler.
    let listener = LOG_LISTENER.lock().take();
    shutdown_logging(listener.as_ref());
}

// ------------------------------------------------------------------ CORS

/// Every method Starlette's CORS middleware expands `allow_methods=["*"]` to.
const CORS_ALL_METHODS: &str = "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT";
/// `Access-Control-Max-Age` (Starlette's default, 600 s).
const CORS_MAX_AGE: &str = "600";

/// Starlette's `CORSMiddleware` as configured by the Python app:
/// `allow_origins` from `cors_origins` (comma-separated, or `*`),
/// `allow_credentials=True`, every method and every header allowed.
#[derive(Debug, Clone)]
struct Cors {
    origins: Vec<String>,
    allow_all_origins: bool,
}

impl Cors {
    fn from_settings(cors_origins: &str) -> Self {
        let origins: Vec<String> = if cors_origins != "*" {
            cors_origins.split(',').map(str::to_string).collect()
        } else {
            vec!["*".to_string()]
        };
        let allow_all_origins = origins.iter().any(|o| o == "*");
        Cors { origins, allow_all_origins }
    }

    fn is_allowed_origin(&self, origin: &str) -> bool {
        self.allow_all_origins || self.origins.iter().any(|o| o == origin)
    }

    /// The answer to a preflight (`OPTIONS` with
    /// `Access-Control-Request-Method`).
    fn preflight(&self, headers: &HeaderMap) -> Response {
        let requested_origin = headers.get("origin").and_then(|v| v.to_str().ok()).unwrap_or_default();
        let requested_method =
            headers.get("access-control-request-method").and_then(|v| v.to_str().ok()).unwrap_or_default();
        let requested_headers = headers.get("access-control-request-headers").cloned();

        let mut out = HeaderMap::new();
        // allow_credentials=True makes the allowed origin explicit, so the
        // origin is set below rather than as "*".
        out.insert("vary", HeaderValue::from_static("Origin"));
        out.insert("access-control-allow-methods", HeaderValue::from_static(CORS_ALL_METHODS));
        out.insert("access-control-max-age", HeaderValue::from_static(CORS_MAX_AGE));
        out.insert("access-control-allow-credentials", HeaderValue::from_static("true"));

        let mut failures: Vec<&str> = Vec::new();
        if self.is_allowed_origin(requested_origin) {
            if let Ok(value) = HeaderValue::from_str(requested_origin) {
                out.insert("access-control-allow-origin", value);
            }
        } else {
            failures.push("origin");
        }
        if !CORS_ALL_METHODS.split(", ").any(|m| m == requested_method) {
            failures.push("method");
        }
        // Every header is allowed, so any requested ones are mirrored back.
        if let Some(requested_headers) = requested_headers {
            out.insert("access-control-allow-headers", requested_headers);
        }

        let (status, text) = if failures.is_empty() {
            (StatusCode::OK, "OK".to_string())
        } else {
            (StatusCode::BAD_REQUEST, format!("Disallowed CORS {}", failures.join(", ")))
        };
        let mut response = (status, text).into_response();
        response.headers_mut().extend(out);
        response
    }

    /// Decorate a simple (non-preflight) response.
    fn simple(&self, origin: &str, has_cookie: bool, response: &mut Response) {
        let headers = response.headers_mut();
        if self.allow_all_origins {
            headers.insert("access-control-allow-origin", HeaderValue::from_static("*"));
        }
        headers.insert("access-control-allow-credentials", HeaderValue::from_static("true"));
        // If the request includes any cookie headers, the specific origin has
        // to be answered instead of '*'; with specific origins only, an
        // allowed one is mirrored back.
        let explicit = (self.allow_all_origins && has_cookie)
            || (!self.allow_all_origins && self.is_allowed_origin(origin));
        if explicit && let Ok(value) = HeaderValue::from_str(origin) {
            headers.insert("access-control-allow-origin", value);
            let vary = match headers.get("vary").and_then(|v| v.to_str().ok()) {
                Some(existing) => format!("{existing}, Origin"),
                None => "Origin".to_string(),
            };
            if let Ok(vary) = HeaderValue::from_str(&vary) {
                headers.insert("vary", vary);
            }
        }
    }

    async fn call(self: Arc<Self>, request: Request, next: Next) -> Response {
        let Some(origin) = request.headers().get("origin").and_then(|v| v.to_str().ok()).map(str::to_string)
        else {
            return next.run(request).await;
        };
        if request.method() == Method::OPTIONS && request.headers().contains_key("access-control-request-method") {
            return self.preflight(request.headers());
        }
        let has_cookie = request.headers().contains_key("cookie");
        let mut response = next.run(request).await;
        self.simple(&origin, has_cookie, &mut response);
        response
    }
}

/// Wrap `router` (outermost) in the CORS middleware.
fn with_cors(router: Router) -> Router {
    let cors = Arc::new(Cors::from_settings(&api_settings().cors_origins));
    Router::new().fallback_service(router).layer(axum::middleware::from_fn(move |request: Request, next: Next| {
        let cors = Arc::clone(&cors);
        async move { cors.call(request, next).await }
    }))
}

/// The pool routing should use, as the routing layer's provider.
fn pool_provider() -> PoolProvider {
    Arc::new(|| get_pool().map(|pool| pool as Arc<dyn AgentPool>))
}

/// Build the application router: every API router plus `/health`, `/agents`
/// and `/`, wrapped (outermost first) in CORS → project routing → project
/// binding.  Lifespan is not run here; see [`serve`].
pub fn app() -> Router {
    let routes = Router::new()
        .merge(routers::auth_router())
        .merge(routers::chat_router())
        .merge(routers::documents_router())
        .merge(routers::projects_router())
        .merge(routers::assets_router())
        .merge(routers::mcp_servers_router())
        .merge(routers::mcp_catalog_router())
        .merge(routers::models_router())
        .merge(routers::init_router())
        .merge(routers::skills_router())
        .merge(routers::users_router())
        .merge(routers::payments_router())
        .merge(routers::feedback_router())
        .merge(routers::project_io_router())
        .merge(routers::releases_router())
        .merge(routers::web_search_router())
        .merge(routers::cloud_usage_router())
        .merge(routers::storage_router())
        .merge(routers::downloads_router())
        .merge(crate::api::openapi::router())
        .route("/health", get(health_check))
        .route("/agents", get(agents))
        .route("/", get(root))
        // Starlette's answers for an unknown path and a known path with the
        // wrong method.
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed);

    // The stack below ends up CORS → project routing → project binding →
    // routers.
    //
    // Bind each request to the project it addresses before anything resolves
    // one.  Innermost of the three: only a request this process is actually
    // going to serve needs a binding.
    let routes = ProjectScopeMiddleware::layer(routes);

    // A project-scoped request is forwarded to the child that owns the
    // project instead of reaching this app's routers at all.  Inside CORS
    // deliberately: a preflight is answered there without a handler, and
    // routing it would cold-start a project agent to answer an OPTIONS.
    // Falls straight through when there is no pool — in a child, and under
    // the --no-supervised fallback.
    let routes = with_project_routing(routes, pool_provider());

    with_cors(routes)
}

/// Fill the `LlamaCpp`-specific fields of the health report.  `None` stops
/// early, as the Python's `except Exception: pass` did around the block.
fn llama_health(llm: &dyn BaseLLM, result: &mut Map<String, Value>) -> Option<()> {
    let llama = llm.as_any().downcast_ref::<LlamaCpp>()?;
    result.insert("n_ctx".into(), json!(llm.get_max_tokens()));
    // Model file size
    let model_path = llama.model.model_path();
    if !model_path.is_empty() {
        let model_size = std::fs::metadata(&model_path).ok()?.len();
        result.insert("model_size_bytes".into(), json!(model_size));
    }
    // n_gpu_layers and flash_attn from the Llama instance
    result.insert("n_gpu_layers".into(), json!(llama.model.n_gpu_layers()));
    result.insert("flash_attn".into(), json!(llama.model.flash_attn()));
    // Quantization type from GGUF metadata.  GGUF file_type is a numeric
    // code — map to human-readable labels.
    let metadata = llama.model.metadata();
    let mut keys: Vec<&String> = metadata.keys().collect();
    keys.sort();
    for key in keys {
        if key.contains("file_type") || key.contains("ftype") {
            let raw = metadata[key].trim().to_string();
            let label = GGUF_FTYPE_MAP
                .iter()
                .find(|(code, _)| *code == raw)
                .map(|(_, label)| label.to_string())
                .unwrap_or_else(|| metadata[key].clone());
            result.insert("quantization".into(), json!(label));
            break;
        }
    }
    // Fallback: check GGUF filename for quant hint (e.g. Q4_K_M)
    if !result.contains_key("quantization") && !model_path.is_empty() {
        let basename = Path::new(&model_path).file_name().map(|n| n.to_string_lossy().into_owned());
        if let Some(captures) = basename.as_deref().and_then(|b| QUANT_IN_NAME.captures(b)) {
            result.insert("quantization".into(), json!(captures[1].to_uppercase()));
        }
    }
    Some(())
}

/// Indexer status plus the indexed documents of the current project.  `None`
/// stops early, as the Python's `except Exception: pass` did.
fn indexing_health(model_service: &ModelService, result: &mut Map<String, Value>) -> Option<()> {
    let queue = Arc::clone(&*model_service.task_queue.read());
    let indexer_status = best_effort(|| queue.get_status(Some("indexer"))).ok()?;
    if let Some(status) = indexer_status.as_object()
        && !status.is_empty()
        && !status.contains_key("error")
    {
        let mut indexing = Map::new();
        indexing.insert("status".into(), status.get("status").cloned().unwrap_or(Value::Null));
        indexing.insert("queued".into(), status.get("queued").cloned().unwrap_or(json!(0)));
        if let Some(workspace) = status.get("workspace") {
            indexing.insert("workspace".into(), workspace.clone());
        }
        result.insert("indexing".into(), Value::Object(indexing));
    }
    let shared = Workspace::get_instance()?;
    let workspace = shared.read();
    let project = workspace.current_project()?;
    let documents: Vec<Value> = project
        .documents
        .iter()
        .filter(|doc| doc.status == DocumentStatus::Indexed)
        .filter_map(|doc| serde_json::to_value(doc.to_info(Some(&project.project_dir))).ok())
        .collect();
    // A KeyError in the Python when the indexer reported nothing.
    let indexing = result.get_mut("indexing")?.as_object_mut()?;
    indexing.insert("documents".into(), Value::Array(documents));
    Some(())
}

/// How long the supervisor waits on a project agent's `/health` before
/// reporting without its indexing section.  Short: the desktop app polls
/// `/health` every few seconds, and a slow child must not stall that.
const DELEGATED_HEALTH_TIMEOUT: Duration = Duration::from_secs(2);

/// Under the supervisor the indexer runs in each project's agent, so this
/// process has no indexer task and [`indexing_health`] adds nothing.  Clients
/// that poll the bare `/health` (the desktop app does) then never see a
/// document turn indexed.  Fill the gap from the current project's agent.
///
/// Only an agent that is already running is asked — `running`, not `ensure`:
/// a health poll must not cold-start a project, and asking the child directly
/// rather than through routing leaves its idle clock alone, so the poll does
/// not keep an idle agent resident either.
///
/// Async, and called from the handler rather than [`health_status`]: that
/// runs on a blocking thread inside the runtime, where `reqwest::blocking`
/// panics as it tears its own runtime down.
async fn delegated_indexing_health() -> Option<Value> {
    let pool = get_pool()?;
    // From disk, not the singleton: this process does not own the workspace
    // under the supervisor, and an import here resets the singleton, so it is
    // empty or behind exactly when a new project has documents to index.
    let project_id = Workspace::load()?.current_project()?.id.clone();
    let record = pool.running(&project_id)?;
    let client = reqwest::Client::builder().timeout(DELEGATED_HEALTH_TIMEOUT).build().ok()?;
    let response = client.get(format!("{}/health", record.base_url())).send().await.ok()?;
    let mut child: Map<String, Value> = response.json().await.ok()?;
    child.remove("indexing")
}

/// The GGUF files this process has loaded, as `{name, path, bytes}`.
///
/// llama.cpp mmaps weights, so on macOS they sit outside each process's
/// footprint and are shared between processes that load the same file.
/// [`complete_memory_health`] counts each path once.
fn resident_model_files(model_service: &ModelService) -> Vec<Value> {
    model_service
        .resident_models()
        .into_iter()
        .filter_map(|(name, llm)| {
            let path = llm.as_any().downcast_ref::<LlamaCpp>()?.model.model_path();
            let bytes = std::fs::metadata(&path).ok()?.len();
            Some(json!({"name": name, "path": path, "bytes": bytes}))
        })
        .collect()
}

/// The loaded models each running project agent reports, read from its own
/// `/health`.  `running`, not `ensure`: a poll must not start an agent.
async fn agent_model_files() -> Vec<Value> {
    let Some(pool) = get_pool() else {
        return Vec::new();
    };
    let Ok(client) = reqwest::Client::builder().timeout(DELEGATED_HEALTH_TIMEOUT).build() else {
        return Vec::new();
    };
    let requests = pool.registry.live().into_iter().map(|record| {
        let client = client.clone();
        async move {
            let child: Value = client.get(format!("{}/health", record.base_url())).send().await.ok()?.json().await.ok()?;
            child.pointer("/memory/models")?.as_array().cloned()
        }
    });
    futures::future::join_all(requests).await.into_iter().flatten().flatten().collect()
}

/// Whether each model selected for the current project can load right now,
/// judged the way the loader judges it ([`ModelService::large_model_fits`]):
///
/// - `loaded`  — already resident.
/// - `fits`    — fits in available memory as things stand.
/// - `evicts`  — fits only once other capability models are unloaded, which
///               the loader does before giving up.
/// - `no_room` — won't fit even then; the turn falls back to the orchestrator.
///               `shortfall_bytes` is how much more must be freed for it to
///               load, and `exceeds_total` is set when it needs more than the
///               machine has at all.
///
/// Off Apple Silicon the loader doesn't gate on memory, so nothing is
/// reported as short there.  Read from disk, not the singleton: under the
/// supervisor this process does not own the workspace.
fn selected_model_fits(resident: &[Value], orchestrator: Option<&str>) -> Vec<Value> {
    let Some(workspace) = Workspace::load() else {
        return Vec::new();
    };
    let project = workspace.current_project();
    let gib = 1024f64.powi(3);
    let resident_path = |path: &str| resident.iter().any(|m| m.get("path").and_then(Value::as_str) == Some(path));
    // What the loader can free: resident capability models, not the orchestrator.
    let evictable_bytes: u64 = resident
        .iter()
        .filter(|m| m.get("name").and_then(Value::as_str) != orchestrator)
        .filter_map(|m| m.get("bytes").and_then(Value::as_u64))
        .sum();
    let device = get_model_service().device_free_total_gb();
    let free_gb = device.map(|(free, _)| free);
    let total_gb = device.map(|(_, total)| total);
    let evictable_gb = evictable_bytes as f64 / gib;
    let settings = AppSettings::new();

    workspace
        .models
        .iter()
        .filter(|m| {
            workspace.effective_model_enabled(&m.model_name, project) && workspace.effective_model_downloaded(&m.model_name)
        })
        .filter_map(|m| {
            let (gguf, _mmproj) = ModelService::find_model_ggufs(&workspace.model_dir_for(&m.model_id));
            let path = gguf?;
            let bytes = std::fs::metadata(&path).ok()?.len();
            let needed_gb = ModelService::fit_footprint_gb(bytes as f64 / gib, settings.llm_large_n_ctx_cap)
                + settings.llm_large_fit_reserve_gb;
            let status = match free_gb {
                _ if resident_path(&path) => "loaded",
                None => "fits",
                Some(free) if needed_gb <= free => "fits",
                Some(free) if needed_gb <= free + evictable_gb => "evicts",
                Some(_) => "no_room",
            };
            let mut entry = json!({
                "name": m.model_name,
                "bytes": bytes,
                "needed_bytes": (needed_gb * gib) as u64,
                "status": status,
            });
            if let (true, Some(free), Some(total)) = (status == "no_room", free_gb, total_gb) {
                entry["shortfall_bytes"] = json!(((needed_gb - free - evictable_gb).max(0.0) * gib) as u64);
                entry["exceeds_total"] = json!(needed_gb > total);
            }
            Some(entry)
        })
        .collect()
}

/// Fill `memory` with what Smartloop is actually using: every process from
/// this one down plus each loaded model file once, against system RAM.
///
/// Under the supervisor, `/health` is answered here, so the tree covers the
/// project agents and the model host; their loaded models come from
/// [`agent_model_files`].  `memory_percent` is `used / total`.
async fn complete_memory_health(result: &mut Map<String, Value>) {
    let mut models: Vec<Value> =
        result.get("memory").and_then(|m| m.get("models")).and_then(Value::as_array).cloned().unwrap_or_default();
    models.extend(agent_model_files().await);
    let mut seen = std::collections::HashSet::new();
    models.retain(|m| m.get("path").and_then(Value::as_str).is_some_and(|p| seen.insert(p.to_string())));
    let model_bytes: u64 = models.iter().filter_map(|m| m.get("bytes").and_then(Value::as_u64)).sum();

    let resident = models.clone();
    let orchestrator = result.get("model_name").and_then(Value::as_str).map(str::to_string);
    let sample = tokio::task::spawn_blocking(move || {
        let (process_bytes, processes) = device_utils::process_tree_memory(std::process::id());
        let (available_gb, total_gb) = device_utils::available_memory_gb();
        let selected = selected_model_fits(&resident, orchestrator.as_deref());
        (process_bytes, processes, available_gb, total_gb, selected)
    })
    .await;
    let Ok((process_bytes, processes, available_gb, total_gb, selected)) = sample else {
        return;
    };
    let gib = 1024f64.powi(3);
    let total_bytes = (total_gb * gib) as u64;
    if total_bytes == 0 {
        return;
    }
    let used_bytes =
        if device_utils::PROCESS_MEMORY_INCLUDES_MAPPED_FILES { process_bytes } else { process_bytes + model_bytes };

    result.insert(
        "memory".into(),
        json!({
            "used_bytes": used_bytes,
            "total_bytes": total_bytes,
            "available_bytes": (available_gb * gib) as u64,
            "process_bytes": process_bytes,
            "model_bytes": model_bytes,
            "processes": processes,
            "models": models,
            "selected": selected,
        }),
    );
    result.insert("memory_percent".into(), json!((used_bytes as f64 / total_bytes as f64 * 100.0).round() as i64));
}

/// The synchronous body of [`health_check`].
pub fn health_status() -> Map<String, Value> {
    get_api_client().flush_token_usage();

    let model_service = get_model_service();
    let cwd = match Workspace::get_instance() {
        Some(workspace) => workspace.read().workspace_dir.clone(),
        None => std::env::current_dir().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default(),
    };
    let mut result = Map::new();
    result.insert("status".into(), json!("healthy"));
    result.insert("cwd".into(), json!(cwd));
    result.insert("model_loaded".into(), json!(model_service.is_loaded()));
    result.insert("model_name".into(), json!(model_service.model_name.read().clone()));

    if let Some(llm) = model_service.get_llm()
        && is_llama_cpp_instance(llm.as_ref())
    {
        let _ = best_effort(|| llama_health(llm.as_ref(), &mut result));
    }

    // Indexer status + indexed documents
    let _ = best_effort(|| indexing_health(&model_service, &mut result));

    // This process's loaded models; the totals are filled in by
    // complete_memory_health, which can reach the project agents.
    if let Ok(models) = best_effort(|| resident_model_files(&model_service)) {
        result.insert("memory".into(), json!({"models": models}));
    }

    result
}

/// `GET /health` — model, indexing and memory-usage status.
#[utoipa::path(
    get,
    path = "/health",
    tag = "system",
    summary = "Health check",
    description = "Model, indexing and memory-usage status.",
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
pub async fn health_check() -> Json<Map<String, Value>> {
    let mut result = match tokio::task::spawn_blocking(health_status).await {
        Ok(result) => result,
        Err(e) if e.is_panic() => std::panic::resume_unwind(e.into_panic()),
        Err(e) => panic!("health check task failed: {e}"),
    };
    if !result.contains_key("indexing")
        && let Some(indexing) = delegated_indexing_health().await
    {
        result.insert("indexing".into(), indexing);
    }
    complete_memory_health(&mut result).await;
    Json(result)
}

/// `GET /agents` — what is actually running: the supervisor, the model host,
/// each project.  `{"supervised": false, "agents": []}` without a supervisor.
///
/// `slp agent status` used to describe one process, because there was one.
/// With a process per project the interesting questions are which projects
/// are resident, on which ports, and what each is costing in RSS.
#[utoipa::path(
    get,
    path = "/agents",
    tag = "system",
    summary = "List running agents",
    description = "The supervisor, the model host and each project agent.",
    responses((status = 200, description = "Supervisor status and running project agents", body = serde_json::Value))
)]
pub async fn agents() -> Json<Value> {
    let Some(supervisor) = get_supervisor() else {
        return Json(json!({"supervised": false, "agents": []}));
    };
    let status = tokio::task::spawn_blocking(move || supervisor.status())
        .await
        .unwrap_or_else(|e| std::panic::resume_unwind(e.into_panic()));
    let mut body = Map::new();
    body.insert("supervised".into(), json!(true));
    body.extend(status);
    Json(Value::Object(body))
}

/// Starlette's 404 for a path no route matches: `{"detail": "Not Found"}`.
async fn not_found() -> Response {
    json_response(StatusCode::NOT_FOUND, json!({"detail": "Not Found"}))
}

/// Starlette's 405 for a route that exists under other methods:
/// `{"detail": "Method Not Allowed"}`.
async fn method_not_allowed() -> Response {
    json_response(StatusCode::METHOD_NOT_ALLOWED, json!({"detail": "Method Not Allowed"}))
}

/// `GET /` — root endpoint.
#[utoipa::path(
    get,
    path = "/",
    tag = "system",
    summary = "Root",
    description = "",
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
pub async fn root() -> Json<Value> {
    Json(json!({
        "message": "Smartloop agent is healthy and running",
        "health": "/health",
    }))
}

/// After binding: write the actual port to [`PORT_FILE`] and adopt it in the
/// settings (`_PortCapturingServer.startup`).
pub fn capture_bound_port(port: u16) -> std::io::Result<()> {
    if let Some(parent) = Path::new(PORT_FILE.as_str()).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(PORT_FILE.as_str(), port.to_string())?;
    adopt_bound_port(port);
    Ok(())
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
    // SAFETY: called once at startup, right after the bind, before any
    // project child is spawned from this environment.
    unsafe { std::env::set_var("SLP_API_PORT", port.to_string()) };
    API_SETTINGS.write().api_port = port;
    if let Some(supervisor) = supervisor {
        supervisor.registry.set_supervisor(i64::from(std::process::id()), port);
    }
}

/// The running log listener (`app.state.log_listener`).
pub static LOG_LISTENER: Lazy<Mutex<Option<Arc<QueueListener>>>> = Lazy::new(|| Mutex::new(None));

/// Install the shared log queue, root queue handler, and parent listener;
/// stashes the listener in [`LOG_LISTENER`] so lifespan shutdown can stop it.
pub fn install_centralized_logging() {
    let queue = make_log_queue();
    // Root stays at INFO so dependencies (html5ever, llama.cpp) never log
    // at DEBUG; only "smartloop" is lifted below.
    setup_root_logging(&queue, "INFO");
    // The CLI entry point clamps "smartloop" to WARNING to keep the
    // interactive `slp` shell quiet — but the server is launched from that
    // same entry point, so it inherits that gag.  Lift it only when the user
    // has explicitly asked for DEBUG; at INFO (the default, used by the
    // shipped desktop build) the gag stays so server.log only accumulates
    // access lines + warnings/errors, not the full orchestrator / DAG /
    // planner / explorer / MCP trace.
    let mut log_level = api_settings().log_level.to_uppercase();
    if log_level.is_empty() {
        log_level = "INFO".to_string();
    }
    if log_level == "DEBUG" {
        get_logger("smartloop").set_level("DEBUG");
    }
    let home_dir = APP_SETTINGS.read().home_dir();
    let mut options = ListenerOptions::new(home_dir);
    options.file_level = log_level.as_str().into();
    match start_listener(&queue, options) {
        Ok(listener) => *LOG_LISTENER.lock() = Some(listener),
        Err(e) => eprintln!("Could not start the log listener: {e}"),
    }
}

/// Why a serve loop ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServeExit {
    /// Ctrl-C (Python's `KeyboardInterrupt`).
    Interrupted,
    /// SIGTERM.
    Terminated,
    /// A programmatic shutdown request (`server.should_exit = True`).
    Requested,
}

/// Wait for Ctrl-C, SIGTERM or `requested` (whichever comes first).
async fn shutdown_signal(requested: Option<Arc<Notify>>) -> ServeExit {
    let ctrl_c = async {
        if tokio::signal::ctrl_c().await.is_err() {
            std::future::pending::<()>().await;
        }
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    let requested = async {
        match requested {
            Some(notify) => notify.notified().await,
            None => std::future::pending::<()>().await,
        }
    };
    tokio::select! {
        _ = ctrl_c => ServeExit::Interrupted,
        _ = terminate => ServeExit::Terminated,
        _ = requested => ServeExit::Requested,
    }
}

/// Log one request in common access-log form, with tokens redacted
/// from the request line.
async fn access_log(request: Request, next: Next) -> Response {
    let client = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ConnectInfo(addr)| addr.to_string())
        .unwrap_or_else(|| "-".to_string());
    let method = request.method().clone();
    let path = request.uri().path_and_query().map(|p| p.as_str().to_string()).unwrap_or_else(|| "/".into());
    let version = match request.version() {
        http::Version::HTTP_10 => "1.0",
        http::Version::HTTP_2 => "2",
        http::Version::HTTP_3 => "3",
        _ => "1.1",
    };
    let response = next.run(request).await;
    let status = response.status();
    tracing::info!(
        target: LogTarget::ACCESS,
        "{} - \"{} {} HTTP/{}\" {} {}",
        client,
        method,
        redact_tokens(&path),
        version,
        status.as_u16(),
        status.canonical_reason().unwrap_or("")
    );
    response
}

/// Serve [`app`] on an already-bound `listener` until a shutdown signal,
/// running lifespan startup before and shutdown after.  With `capture_port`
/// the bound port is written to [`PORT_FILE`] and adopted in the settings.
pub async fn serve_listener(
    listener: tokio::net::TcpListener,
    capture_port: bool,
    shutdown: Option<Arc<Notify>>,
) -> anyhow::Result<ServeExit> {
    tracing::info!(target: LogTarget::SERVER, "Started server process [{}]", std::process::id());
    tracing::info!(target: LogTarget::SERVER, "Waiting for application startup.");
    let state = lifespan_startup().await;
    tracing::info!(target: LogTarget::SERVER, "Application startup complete.");

    let local = listener.local_addr()?;
    if capture_port && let Err(e) = capture_bound_port(local.port()) {
        lifespan_shutdown(state).await;
        return Err(e.into());
    }
    tracing::info!(
        target: LogTarget::SERVER,
        "Smartloop API running on http://{}:{} (Press CTRL+C to quit)",
        local.ip(),
        local.port()
    );

    let router = app().layer(axum::middleware::from_fn(access_log));
    let exit = Arc::new(Mutex::new(ServeExit::Requested));
    let exit_slot = Arc::clone(&exit);
    let result = axum::serve(listener, router.into_make_service_with_connect_info::<SocketAddr>())
        .with_graceful_shutdown(async move {
            let reason = shutdown_signal(shutdown).await;
            *exit_slot.lock() = reason;
            tracing::info!(target: LogTarget::SERVER, "Shutting down");
        })
        .await;

    tracing::info!(target: LogTarget::SERVER, "Waiting for application shutdown.");
    lifespan_shutdown(state).await;
    tracing::info!(target: LogTarget::SERVER, "Application shutdown complete.");
    tracing::info!(target: LogTarget::SERVER, "Finished server process [{}]", std::process::id());
    result?;
    let reason = *exit.lock();
    Ok(reason)
}

/// Bind `host:port`, capture the bound port, run lifespan startup, serve
/// [`app`] until Ctrl-C / shutdown, then run lifespan shutdown.
pub async fn serve(host: &str, port: u16) -> anyhow::Result<ServeExit> {
    serve_on(host, port, None).await
}

/// [`serve`] with an optional programmatic shutdown trigger.
pub async fn serve_on(host: &str, port: u16, shutdown: Option<Arc<Notify>>) -> anyhow::Result<ServeExit> {
    // Run the lifespan startup before binding so a supervisor started during
    // startup sees the configured port first and is corrected by the port
    // capture afterwards.
    tracing::info!(target: LogTarget::SERVER, "Started server process [{}]", std::process::id());
    tracing::info!(target: LogTarget::SERVER, "Waiting for application startup.");
    let state = lifespan_startup().await;
    tracing::info!(target: LogTarget::SERVER, "Application startup complete.");

    let listener = match tokio::net::TcpListener::bind((host, port)).await {
        Ok(listener) => listener,
        Err(e) => {
            tracing::error!(
                target: LogTarget::SERVER,
                "[Errno {}] error while attempting to bind on address ('{}', {}): {}",
                e.raw_os_error().unwrap_or(0),
                host,
                port,
                e
            );
            tracing::info!(target: LogTarget::SERVER, "Waiting for application shutdown.");
            lifespan_shutdown(state).await;
            tracing::info!(target: LogTarget::SERVER, "Application shutdown complete.");
            return Err(e.into());
        }
    };
    let local = listener.local_addr()?;
    if let Err(e) = capture_bound_port(local.port()) {
        lifespan_shutdown(state).await;
        return Err(e.into());
    }
    tracing::info!(
        target: LogTarget::SERVER,
        "Smartloop API running on http://{}:{} (Press CTRL+C to quit)",
        host,
        local.port()
    );

    let router = app().layer(axum::middleware::from_fn(access_log));
    let exit = Arc::new(Mutex::new(ServeExit::Requested));
    let exit_slot = Arc::clone(&exit);
    let result = axum::serve(listener, router.into_make_service_with_connect_info::<SocketAddr>())
        .with_graceful_shutdown(async move {
            let reason = shutdown_signal(shutdown).await;
            *exit_slot.lock() = reason;
            tracing::info!(target: LogTarget::SERVER, "Shutting down");
        })
        .await;

    tracing::info!(target: LogTarget::SERVER, "Waiting for application shutdown.");
    lifespan_shutdown(state).await;
    tracing::info!(target: LogTarget::SERVER, "Application shutdown complete.");
    tracing::info!(target: LogTarget::SERVER, "Finished server process [{}]", std::process::id());
    result?;
    let reason = *exit.lock();
    Ok(reason)
}

/// A multi-threaded tokio runtime for the blocking entry points.
fn runtime() -> anyhow::Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread().enable_all().build()?)
}

/// `<home_dir>/server.pid`.
fn pid_file_path() -> PathBuf {
    PathBuf::from(APP_SETTINGS.read().home_dir()).join("server.pid")
}

/// Write this process's pid to `pid_file`, creating its directory.
fn write_pid_file(pid_file: &Path) -> std::io::Result<()> {
    if let Some(parent) = pid_file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(pid_file, std::process::id().to_string())
}

/// Run the server (for development/direct execution).  Writes/cleans the PID
/// and port files.  Blocking: builds its own tokio runtime.
pub fn run_server() {
    install_centralized_logging();

    // Write PID file for tracking
    let pid_file = pid_file_path();
    if let Err(e) = write_pid_file(&pid_file) {
        tracing::error!(target: LogTarget::SERVER, "Could not write the PID file: {}", e);
    }

    let version = APP_SETTINGS.read().version.clone();
    print_logo(version.as_deref());

    let settings = api_settings();
    let outcome = runtime().and_then(|rt| rt.block_on(serve(&settings.api_host, settings.api_port)));
    match outcome {
        Ok(ServeExit::Interrupted) => tracing::info!(target: LogTarget::SERVER, "Interrupted — server stopped"),
        Ok(_) => {}
        Err(e) => tracing::error!(target: LogTarget::SERVER, "Server failed: {}", e),
    }

    // Clean up PID and port files on exit
    let _ = std::fs::remove_file(&pid_file);
    let _ = std::fs::remove_file(PORT_FILE.as_str());
}

/// Start the API server in the foreground (blocking; builds its own tokio
/// runtime).  With `service`, auto-restart on crash with exponential backoff
/// (1s doubling to 60s, reset after 120s of stable uptime).
///
/// The server runs in-process (no forking) because accelerator contexts do
/// not survive a fork.
pub fn start_server_foreground(host: &str, port: u16, service: bool) -> anyhow::Result<()> {
    // Print logo before starting
    let version = APP_SETTINGS.read().version.clone();
    print_logo(version.as_deref());

    // PID file for tracking
    let pid_file = pid_file_path();
    write_pid_file(&pid_file)?;

    install_centralized_logging();

    const MAX_BACKOFF: u64 = 60; // seconds
    const BACKOFF_RESET_THRESHOLD: f64 = 120.0; // reset backoff after 2 min of stable uptime
    let mut backoff: u64 = 1;

    let rt = match runtime() {
        Ok(rt) => rt,
        Err(e) => {
            let _ = std::fs::remove_file(&pid_file);
            return Err(e);
        }
    };

    let result = (|| -> anyhow::Result<()> {
        loop {
            let start_ts = Instant::now();
            match rt.block_on(serve(host, port)) {
                Ok(ServeExit::Interrupted) => {
                    // Ctrl-C is someone stopping the agent, not a crash.  The
                    // lifespan shutdown has already run by the time this
                    // arrives — so the children and the model host are down —
                    // and all that is left is to leave, certainly without the
                    // restart-with-backoff below treating a deliberate stop as
                    // something to recover from.
                    tracing::info!(target: LogTarget::SERVER, "Interrupted — agent stopped");
                    break;
                }
                Ok(ServeExit::Terminated) => break,
                Ok(ServeExit::Requested) => {}
                Err(e) => {
                    if !service {
                        return Err(e);
                    }
                    tracing::error!(target: LogTarget::SERVER, "Server crashed: {:#}", e);
                }
            }

            if !service {
                break;
            }

            // The lifespan shutdown tore logging down (it detaches the root
            // queue handler from a listener that no longer drains), so bring
            // it back up before logging anything about the restart.
            install_centralized_logging();

            let uptime = start_ts.elapsed().as_secs_f64();
            if uptime >= BACKOFF_RESET_THRESHOLD {
                backoff = 1;
            }

            tracing::warn!(
                target: LogTarget::SERVER,
                "Server exited after {:.1}s — restarting in {}s ...",
                uptime,
                backoff
            );
            std::thread::sleep(Duration::from_secs(backoff));
            backoff = (backoff * 2).min(MAX_BACKOFF);

            // Re-write PID file for the same process
            write_pid_file(&pid_file)?;
        }
        Ok(())
    })();

    // Clean up the PID file on exit
    let _ = std::fs::remove_file(&pid_file);
    result
}

/// Build a response with `status` and a JSON `body` (helper for callers that
/// answer outside a router).
pub fn json_response(status: StatusCode, body: Value) -> Response {
    let mut response = Response::new(Body::from(body.to_string()));
    *response.status_mut() = status;
    response.headers_mut().insert(http::header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_are_redacted() {
        assert_eq!(
            redact_tokens("GET /v1/x?access_token=abc&y=1&refresh_token=zz HTTP/1.1"),
            "GET /v1/x?access_token=<redacted>&y=1&refresh_token=<redacted> HTTP/1.1"
        );
    }

    #[test]
    fn quant_hint_in_file_name() {
        let caps = QUANT_IN_NAME.captures("gemma-3-4b-it-q4_k_m.gguf").unwrap();
        assert_eq!(caps[1].to_uppercase(), "Q4_K_M");
    }

    #[test]
    fn cors_preflight_mirrors_an_allowed_origin() {
        let cors = Cors::from_settings("*");
        let mut headers = HeaderMap::new();
        headers.insert("origin", HeaderValue::from_static("http://a"));
        headers.insert("access-control-request-method", HeaderValue::from_static("POST"));
        headers.insert("access-control-request-headers", HeaderValue::from_static("content-type"));
        let response = cors.preflight(&headers);
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["access-control-allow-origin"], "http://a");
        assert_eq!(response.headers()["access-control-allow-headers"], "content-type");

        let restricted = Cors::from_settings("http://b");
        let response = restricted.preflight(&headers);
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
