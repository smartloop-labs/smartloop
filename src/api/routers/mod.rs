//! Port of `smartloop/api/routers/__init__.py`.
//!
//! API routers.  Each module exposes `pub fn router() -> axum::Router`, the
//! equivalent of the Python module-level `router = APIRouter(...)`; routes carry
//! their full path (the APIRouter prefix is baked in).

pub mod _errors;
pub mod assets;
pub mod auth;
pub mod chat;
pub mod documents;
pub mod downloads;
pub mod feedback;
pub mod init;
pub mod mcp_catalog;
pub mod mcp_servers;
pub mod models;
pub mod payments;
pub mod project_io;
pub mod projects;
pub mod releases;
pub mod skills;
pub mod storage;
pub mod users;
pub mod web_search;

pub use assets::router as assets_router;
pub use auth::router as auth_router;
pub use chat::router as chat_router;
pub use documents::router as documents_router;
pub use downloads::router as downloads_router;
pub use feedback::router as feedback_router;
pub use init::router as init_router;
pub use mcp_catalog::router as mcp_catalog_router;
pub use mcp_servers::router as mcp_servers_router;
pub use models::router as models_router;
pub use payments::router as payments_router;
pub use project_io::router as project_io_router;
pub use projects::router as projects_router;
pub use releases::router as releases_router;
pub use skills::router as skills_router;
pub use storage::router as storage_router;
pub use users::router as users_router;
pub use web_search::router as web_search_router;
