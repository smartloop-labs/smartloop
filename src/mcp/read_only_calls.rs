//! One read-only MCP tool call, for a background pass that makes one.
//!
//! The account probe ([`crate::mcp::account`]) needs a tool call that changes
//! nothing, its text back, and a failure treated as a skip rather than an
//! error — including the expired-OAuth-token retry the live tool path does.
//!
//! The transport underneath is
//! [`crate::agents::mcp::mcp_tool_caller::MCPToolCaller`], reached through the
//! [`ReadOnlyToolCaller`] trait so tests can inject a fake.
//!
//! Port of `smartloop/mcp/read_only_calls.py`.

use std::sync::Arc;

use serde_json::{Map, Value};

#[allow(unused_imports)]
use crate::agents::mcp::mcp_tool_caller::MCPToolCaller;
use crate::mcp::client::MCPToolCallResult;
use crate::workspace::MCPServer;

/// Characters kept from a failed call's reason.
pub const MAX_REASON_CHARS: usize = 500;

/// Outcome of one read-only call.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolCallResult {
    pub payload: Option<String>,
    /// True when an expired token was refreshed, so the caller can persist it.
    pub credentials_changed: bool,
    pub error: Option<String>,
}

impl ToolCallResult {
    pub fn new(payload: Option<String>, credentials_changed: bool, error: Option<String>) -> Self {
        Self { payload, credentials_changed, error }
    }

    /// `(payload, credentials_changed, error)`, for tuple-style destructuring.
    pub fn into_tuple(self) -> (Option<String>, bool, Option<String>) {
        (self.payload, self.credentials_changed, self.error)
    }
}

/// What a background pass needs from the MCP transport: one tool call with the
/// expired-token refresh-and-retry applied.
///
/// Returns `(result, credentials_changed)`.  Implemented by `MCPToolCaller`.
pub trait ReadOnlyToolCaller: Send + Sync {
    fn call_with_refresh(
        &self,
        server: &mut MCPServer,
        tool_name: &str,
        arguments: &Map<String, Value>,
    ) -> anyhow::Result<(MCPToolCallResult, bool)>;
}

/// Join an MCP tool result's content blocks into one string.
pub fn content_to_text(content: Option<&[Value]>) -> String {
    todo!()
}

/// Short reason a tool call produced nothing (error, else content, else
/// "empty result").
pub fn failure_reason(result: &MCPToolCallResult) -> String {
    todo!()
}

/// Return `existing`, or build the default transport caller lazily.
pub fn ensure_tool_caller(existing: Option<Arc<dyn ReadOnlyToolCaller>>) -> Option<Arc<dyn ReadOnlyToolCaller>> {
    todo!()
}

/// Call one read-only tool and return its text.
///
/// A failure is a skip, not an error; the reason travels back.  An expired
/// OAuth token is refreshed once and the call retried.
pub fn call_read_only(
    caller: Option<&dyn ReadOnlyToolCaller>,
    server: &mut MCPServer,
    tool_name: &str,
    arguments: &Map<String, Value>,
    label: &str,
) -> ToolCallResult {
    todo!()
}
