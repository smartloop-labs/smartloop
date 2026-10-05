//! Smartloop — AI agent harness for extracting information and creating new
//! content.  Port of the `smartloop` Python package; the module tree mirrors
//! `smartloop/` one-to-one.

//! Port of `smartloop/__init__.py`.

pub mod agents;
pub mod api;
pub mod auth;
pub mod config;
pub mod constants;
pub mod conversation_store;
pub mod db;
pub mod document_processor;
pub mod feedback_store;
pub mod llms;
pub mod loader;
pub mod mcp;
pub mod model;
pub mod model_factory;
pub mod model_host;
pub mod parsers;
pub mod paths;
pub mod platform;
pub mod search;
pub mod server;
pub mod skills;
pub mod supervisor;
pub mod tasks;
pub mod trainer;
pub mod types;
pub mod utils;
pub mod workspace;

/// The package version (`smartloop.__version__`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub use auth::credential_store::{
    auth_headers, clear_access_token, clear_refresh_token, clear_token, get_access_token,
    get_active_token, get_refresh_token, get_token, has_credentials, set_access_token,
    set_refresh_token, set_token,
};
pub use conversation_store::{ConversationStore, MessageInput};
pub use server::{get_server_url, get_status, is_server_running, read_pid_file, start_server, stop_server};

/// Process-wide environment set-up the Python package did at import time
/// (`TOKENIZERS_PARALLELISM=false`, `TRANSFORMERS_VERBOSITY=error`).  Call
/// once at start-up, before any threads are spawned.
pub fn init_process_env() {
    todo!()
}
