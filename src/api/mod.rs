//! Port of `smartloop/api/__init__.py`.
//!
//! The HTTP API service (axum port of the FastAPI app).

pub mod config;
pub mod dependencies;
pub mod main;
pub mod mcp;
pub mod middleware;
pub mod model;
pub mod openapi;
pub mod project_agent;
pub mod routers;
pub mod services;

pub use main::{app, run_server, start_server_foreground};
