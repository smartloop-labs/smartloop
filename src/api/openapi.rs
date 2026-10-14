//! OpenAPI description of the HTTP API, served as Swagger UI at `/docs` and
//! as JSON at `/openapi.json`.
//!
//! Each handler carries its own `#[utoipa::path]`; this module only collects
//! them.  A new route must be added to `paths(...)` below to appear.

use axum::Router;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

/// Path of the Swagger UI.
pub const DOCS_PATH: &str = "/docs";

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
        crate::api::routers::projects::create_project,
        crate::api::routers::projects::list_projects,
        crate::api::routers::projects::update_project,
        crate::api::routers::projects::delete_project,
        crate::api::routers::project_io::export_project_endpoint,
        crate::api::routers::project_io::import_project_endpoint,
        crate::api::routers::web_search::get_web_search_config,
        crate::api::routers::web_search::set_web_search_enabled,
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
        crate::api::routers::users::get_me_handler,
        crate::api::routers::users::update_me_handler,
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

/// The Swagger UI and the OpenAPI document it renders.
pub fn router() -> Router {
    let mut doc = ApiDoc::openapi();
    doc.info.version = crate::VERSION.to_string();
    SwaggerUi::new(DOCS_PATH).url(OPENAPI_PATH, doc).into()
}
