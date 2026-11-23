//! OpenAPI description of the HTTP API, served as JSON at `/openapi.json` and
//! rendered by Scalar at `/docs`.
//!
//! Each handler carries its own `#[utoipa::path]`; this module only collects
//! them.  A new route must be added to `paths(...)` below to appear.

use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE, HOST};
use axum::http::HeaderMap;
use axum::response::{Html, IntoResponse, Json};
use axum::routing::get;
use axum::Router;
use once_cell::sync::Lazy;
use utoipa::OpenApi;

/// Path of the API reference page.
pub const DOCS_PATH: &str = "/docs";

/// Path the page loads the Scalar bundle from.
pub const SCALAR_JS_PATH: &str = "/docs/scalar.js";

/// Path of the OpenAPI JSON document.
pub const OPENAPI_PATH: &str = "/openapi.json";

/// A multipart upload carrying a single `file`.
#[derive(utoipa::ToSchema)]
#[allow(dead_code)]
pub struct FileUpload {
    #[schema(value_type = String, format = Binary)]
    pub file: Vec<u8>,
}

/// A multipart document upload: the `file` and the `cwd` it is filed under.
#[derive(utoipa::ToSchema)]
#[allow(dead_code)]
pub struct DocumentUpload {
    #[schema(value_type = String, format = Binary)]
    pub file: Vec<u8>,
    /// Working directory the document is saved under.
    pub cwd: String,
}

#[derive(OpenApi)]
#[openapi(
    info(title = "Smartloop", description = "Smartloop local LLM inference framework - OpenAI-compatible API server"),
    paths(
        crate::api::routers::chat::create_chat_completion,
        crate::api::routers::chat_summary::summarize_task,
        crate::api::routers::projects::create_project,
        crate::api::routers::projects::list_projects,
        crate::api::routers::projects::update_project,
        crate::api::routers::projects::delete_project,
        crate::api::routers::project_io::export_project_endpoint,
        crate::api::routers::project_io::import_project_endpoint,
        crate::api::routers::web_search::get_web_search_config,
        crate::api::routers::web_search::set_web_search_enabled,
        crate::api::routers::web_search::get_web_search_status,
        crate::api::routers::cloud_usage::get_cloud_quota,
        crate::api::routers::documents::list_documents_for_project,
        crate::api::routers::documents::add_document_to_project,
        crate::api::routers::documents::upload_document_file,
        crate::api::routers::documents::delete_document_from_project,
        crate::api::routers::skills::list_skills,
        crate::api::routers::skills::add_skill,
        crate::api::routers::skills::get_skill,
        crate::api::routers::skills::update_skill,
        crate::api::routers::skills::delete_skill,
        crate::api::routers::mcp_servers::list_mcp_servers_for_project,
        crate::api::routers::mcp_servers::count_mcp_tools_for_project,
        crate::api::routers::mcp_servers::get_mcp_server,
        crate::api::routers::mcp_servers::delete_mcp_server_from_project,
        crate::api::routers::mcp_servers::refresh_mcp_server,
        crate::api::routers::mcp_servers::register_mcp_server_handler,
        crate::api::routers::mcp_servers::register_mcp_from_definition_handler,
        crate::api::routers::mcp_servers::mcp_oauth_callback_for_project_handler,
        crate::api::routers::mcp_servers::complete_mcp_oauth_handler,
        crate::api::routers::mcp_servers::mcp_oauth_callback_handler,
        crate::api::routers::mcp_catalog::list_mcp_catalog,
        crate::api::routers::models::load_model,
        crate::api::routers::models::unload_model,
        crate::api::routers::models::list_models,
        crate::api::routers::models::list_workspace_models,
        crate::api::routers::models::update_workspace_model_handler,
        crate::api::routers::models::list_project_models,
        crate::api::routers::models::update_project_model,
        crate::api::routers::downloads::cancel_download_handler,
        crate::api::routers::init::init,
        crate::api::routers::init::bootstrap,
        crate::api::routers::assets::upload_asset,
        crate::api::routers::assets::get_asset,
        crate::api::routers::assets::delete_asset,
        crate::api::routers::auth::auth_status_handler,
        crate::api::routers::auth::set_token_handler,
        crate::api::routers::auth::clear_token_handler,
        crate::api::routers::auth::oauth_callback_handler,
        crate::api::routers::auth::oauth_callback_get_handler,
        crate::api::routers::auth::start_device_login_handler,
        crate::api::routers::auth::device_login_status_handler,
        crate::api::routers::users::get_me_handler,
        crate::api::routers::users::update_me_handler,
        crate::api::routers::users::delete_me_handler,
        crate::api::routers::payments::verify_apple_handler,
        crate::api::routers::payments::checkout_handler,
        crate::api::routers::payments::refresh_handler,
        crate::api::routers::feedback::get_feedback,
        crate::api::routers::feedback::submit_feedback,
        crate::api::routers::feedback::delete_feedback,
        crate::api::routers::storage::get_storage_usage,
        crate::api::routers::storage::cleanup_storage,
        crate::api::routers::releases::latest_release,
        crate::api::main::health_check,
        crate::api::main::agents,
        crate::api::main::root,
    ),
    tags(
        (name = "chat", description = "OpenAI-compatible chat completions"),
        (name = "projects", description = "Projects, import/export and web search"),
        (name = "documents", description = "Project documents and indexing"),
        (name = "skills", description = "Project skills"),
        (name = "mcp", description = "MCP servers and the MCP catalog"),
        (name = "models", description = "Model listing, loading and downloads"),
        (name = "setup", description = "First-run initialisation"),
        (name = "assets", description = "Chat attachments"),
        (name = "auth", description = "Sign-in and the current user"),
        (name = "feedback", description = "Answer ratings"),
        (name = "system", description = "Health, storage and releases"),
    )
)]
pub struct ApiDoc;

/// `@scalar/api-reference` 1.72.2 (MIT), `dist/browser/standalone.js` from
/// the npm package.  Bundled so the page renders offline; to upgrade, replace
/// the file and the version here.
const SCALAR_JS: &str = include_str!("assets/scalar-api-reference.js");

/// The reference page.  Scalar's defaults reach the network — web fonts,
/// telemetry, the hosted AI agent and MCP panel — so each is switched off.
const DOCS_HTML: &str = r#"<!doctype html>
<html>
  <head>
    <title>Smartloop API</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
  </head>
  <body>
    <div id="app"></div>
    <script src="/docs/scalar.js"></script>
    <script>
      Scalar.createApiReference('#app', {
        url: '/openapi.json',
        withDefaultFonts: false,
        telemetry: false,
        showDeveloperTools: 'never',
        agent: { disabled: true },
        mcp: { disabled: true },
      })
    </script>
  </body>
</html>
"#;

/// The OpenAPI document, stamped with the crate version.
static DOC: Lazy<utoipa::openapi::OpenApi> = Lazy::new(|| {
    let mut doc = ApiDoc::openapi();
    doc.info.version = crate::VERSION.to_string();
    doc
});

/// `GET /openapi.json` — the document, with the server it was fetched from
/// as its only `servers` entry, so examples and "Test Request" target the
/// port this instance actually bound.
async fn openapi_json(headers: HeaderMap) -> Json<utoipa::openapi::OpenApi> {
    let mut doc = DOC.clone();
    if let Some(host) = headers.get(HOST).and_then(|h| h.to_str().ok()) {
        doc.servers = Some(vec![utoipa::openapi::Server::new(format!("http://{host}"))]);
    }
    Json(doc)
}

/// The reference page, its bundle, and the OpenAPI document it renders.
pub fn router() -> Router {
    Router::new()
        .route(DOCS_PATH, get(|| async { Html(DOCS_HTML) }))
        .route(
            SCALAR_JS_PATH,
            get(|| async {
                (
                    [(CONTENT_TYPE, "text/javascript; charset=utf-8"), (CACHE_CONTROL, "public, max-age=86400")],
                    SCALAR_JS,
                )
                    .into_response()
            }),
        )
        .route(OPENAPI_PATH, get(openapi_json))
}
