//! Port of `smartloop/mcp/__init__.py`.

pub mod account;
pub mod call_plans;
pub mod client;
pub mod definition_loader;
pub mod definitions;
pub mod local_client;
pub mod oauth;
pub mod read_only_calls;

pub use client::{MCPClient, MCPConnectResult, MCPToolCallResult};
pub use local_client::LocalMCPClient;
pub use oauth::MCPOAuthService;
