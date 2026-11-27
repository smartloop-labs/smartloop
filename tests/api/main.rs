//! `api` integration test target (mirrors tests/api/).

#[path = "../common/mod.rs"]
#[allow(dead_code)]
mod shared;

mod common;
mod test_auth;
mod test_download_cancel;
mod test_download_progress;
mod test_init_access;
mod test_init_adds_remote_model;
mod test_init_needs_whole_weights;
mod test_init_promotes_a_downloaded_model;
mod test_mcp_catalog;
mod test_models;
mod test_models_load_errors;
mod test_payments;
mod test_project_models;
mod test_releases;
mod test_request_guard;
mod test_storage;
mod test_users;
