//! Which account a connection is linked to.
//!
//! MCP has no standard "who am I", so this resolves one of the identity tools
//! servers conventionally ship (`notion-get-self`, Linear's `get_viewer`, …),
//! calls it through the shared read-only path, and pulls a human-readable label
//! out of whatever shape comes back.  Everything here is best-effort.
//!
//! Port of `smartloop/mcp/account.py`.

use std::sync::Arc;

use serde_json::Value;

#[allow(unused_imports)]
use crate::mcp::definitions::normalize_name;
#[allow(unused_imports)]
use crate::mcp::read_only_calls::{call_read_only, ensure_tool_caller, ReadOnlyToolCaller};
use crate::workspace::MCPServer;

/// Tools that answer "who is this connection authenticated as", matched
/// *exactly* (see [`resolve_account_tool`]).
pub const ACCOUNT_TOOLS: &[&str] = &[
    "notion-get-self", "notion_get_self", "API-get-self", "get_self",
    "get_viewer", "viewer", "get_current_user", "current_user",
    "get_authenticated_user", "get_me", "users_me", "whoami", "who_am_i",
    "get_my_profile", "get_account",
];

/// Keys whose value names the account, in the order they are preferred.
pub const ACCOUNT_KEYS: &[&str] = &[
    "email", "workspace_name", "workspaceName",
    "displayName", "display_name", "username", "login", "name",
];

/// How deep to look for an account key.
pub const MAX_DEPTH: usize = 6;

/// A label longer than this is not a name.
pub const MAX_LABEL_CHARS: usize = 80;

/// Return the server's identity tool, or None (exact normalized match only).
pub fn resolve_account_tool<S: AsRef<str>>(tool_names: &[S]) -> Option<String> {
    todo!()
}

fn label_from(value: &Value) -> Option<String> {
    todo!()
}

/// Pull an account label out of an identity tool's result.
///
/// Walks the parsed payload breadth-first; key preference ranks first, depth
/// second.  A non-JSON payload's first line is the label when short enough.
pub fn extract_account(payload: Option<&str>) -> Option<String> {
    todo!()
}

/// Return `(account, credentials_changed)` for `server`.
///
/// One read-only call; never fails.  The second value says the call renewed an
/// expired token, so the caller knows to write it back.
pub fn probe_account(
    server: &mut MCPServer,
    tool_caller: Option<Arc<dyn ReadOnlyToolCaller>>,
) -> (Option<String>, bool) {
    todo!()
}
