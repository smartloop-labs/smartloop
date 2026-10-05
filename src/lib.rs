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
