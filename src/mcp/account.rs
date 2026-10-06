//! Which account a connection is linked to.
//!
//! A connected card in the UI used to say "Connected", which restated what the
//! presence of tools, an unlink button and a re-index button already proved.
//! What it could not answer is the question someone actually has in front of
//! two Notion connections: *whose* Notion is this?
//!
//! MCP has no standard "who am I", so this resolves one of the identity tools
//! servers conventionally ship (`notion-get-self`, Linear's `get_viewer`, …),
//! calls it through the shared read-only path, and pulls a human-readable label
//! out of whatever shape comes back.
//!
//! Everything here is best-effort: a server with no identity tool, or one that
//! refuses the call, simply has no account label and the card shows what it
//! showed before.
//!
//! Port of `smartloop/mcp/account.py`.

use std::sync::Arc;

use serde_json::{Map, Value};

use crate::mcp::definitions::normalize_name;
use crate::mcp::read_only_calls::{call_read_only, ensure_tool_caller, ReadOnlyToolCaller};
use crate::workspace::MCPServer;

/// Logger name of the Python module.
const LOGGER: &str = "smartloop.mcp.account";

/// Tools that answer "who is this connection authenticated as".
///
/// Matched *exactly* (see [`resolve_account_tool`]).  The loose matcher would
/// pair `get_me` with Trello's `get_members` — a member list is not an
/// account, and labelling a connection with the wrong person's name is worse
/// than labelling it with nothing.  So every real spelling is listed instead.
pub const ACCOUNT_TOOLS: &[&str] = &[
    "notion-get-self", "notion_get_self", "API-get-self", "get_self",
    "get_viewer", "viewer", "get_current_user", "current_user",
    "get_authenticated_user", "get_me", "users_me", "whoami", "who_am_i",
    "get_my_profile", "get_account",
];

/// Keys whose value names the account, in the order they are preferred.
///
/// Email first: it is the one label that distinguishes two connections to the
/// same product for certain.  `workspace_name` next, because a Notion
/// integration token authenticates as a bot whose `name` is the integration
/// ("Smartloop") — the workspace is the identity a person recognises there.
pub const ACCOUNT_KEYS: &[&str] = &[
    "email", "workspace_name", "workspaceName",
    "displayName", "display_name", "username", "login", "name",
];

/// How deep to look for an account key.  Identity payloads nest the useful
/// part (Notion puts the human behind `bot.owner.user.person.email`), but an
/// unbounded walk over an arbitrary result would happily label a connection
/// with the first `name` in an unrelated list.
pub const MAX_DEPTH: usize = 6;

/// A label longer than this is not a name — it is a description, a token or a
/// whole record that happened to sit under a matching key.
pub const MAX_LABEL_CHARS: usize = 80;

/// Return the server's identity tool, or None.
///
/// Exact-match only, on the shared name normalisation
/// ([`crate::mcp::definitions::normalize_name`]).  A looser match — prefix,
/// substring — would guess, and a wrong guess names the connection after
/// someone else's account.
pub fn resolve_account_tool<S: AsRef<str>>(tool_names: &[S]) -> Option<String> {
    // `{normalize_name(name): name for name in tool_names if name}` — a later
    // spelling of the same normalised name wins, as in a dict comprehension.
    let mut by_norm: Vec<(String, String)> = Vec::new();
    for name in tool_names.iter().map(|n| n.as_ref()).filter(|n| !n.is_empty()) {
        let norm = normalize_name(name);
        match by_norm.iter_mut().find(|(k, _)| *k == norm) {
            Some(slot) => slot.1 = name.to_string(),
            None => by_norm.push((norm, name.to_string())),
        }
    }
    for candidate in ACCOUNT_TOOLS {
        let target = normalize_name(candidate);
        if let Some((_, name)) = by_norm.iter().find(|(k, _)| *k == target) {
            return Some(name.clone());
        }
    }
    None
}

/// A usable label: a string or number, non-blank once stripped, and short.
fn label_from(value: &Value) -> Option<String> {
    let text = match value {
        Value::String(s) => s.trim().to_string(),
        Value::Number(n) => n.to_string(),
        // `isinstance(True, int)` holds in Python, so a bool passes too.
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        _ => return None,
    };
    if text.is_empty() || text.chars().count() > MAX_LABEL_CHARS {
        return None;
    }
    Some(text)
}

/// One candidate label: `(key, depth, label)`.
type Found = Vec<(String, usize, String)>;

fn walk(node: &Value, depth: usize, found: &mut Found) {
    if depth > MAX_DEPTH {
        return;
    }
    match node {
        Value::Object(map) => {
            for (key, value) in map {
                let Some(label) = label_from(value) else { continue };
                if !ACCOUNT_KEYS.contains(&key.as_str()) {
                    continue;
                }
                match found.iter_mut().find(|(k, _, _)| k == key) {
                    Some(slot) => {
                        if depth < slot.1 {
                            slot.1 = depth;
                            slot.2 = label;
                        }
                    }
                    None => found.push((key.clone(), depth, label)),
                }
            }
            for value in map.values() {
                if value.is_object() || value.is_array() {
                    walk(value, depth + 1, found);
                }
            }
        }
        Value::Array(items) => {
            // Only a single-element list is unambiguous: "the" account.  A list
            // of users is a directory, and picking one of them is a guess.
            if items.len() == 1 {
                walk(&items[0], depth + 1, found);
            }
        }
        _ => {}
    }
}

/// Pull an account label out of an identity tool's result.
///
/// Walks the parsed payload breadth-first so the shallowest match wins, and
/// tries every key in [`ACCOUNT_KEYS`] at each depth before going deeper — an
/// email three levels down beats a display name at the top, but only because
/// email is the better label, not because it is deeper.
pub fn extract_account(payload: Option<&str>) -> Option<String> {
    let payload = payload.filter(|p| !p.is_empty())?;
    let text = payload.trim();
    if text.is_empty() {
        return None;
    }
    let data: Value = match serde_json::from_str(text) {
        Ok(data) => data,
        Err(_) => {
            // A server that answers in prose: the whole line is the best label
            // available, when it is short enough to be one.
            let first = text.lines().next().unwrap_or("");
            return label_from(&Value::String(first.to_string()));
        }
    };

    // Collect every candidate with the depth it was found at, then rank by key
    // preference first and depth second.
    let mut found: Found = Vec::new();
    walk(&data, 0, &mut found);
    for key in ACCOUNT_KEYS {
        if let Some((_, _, label)) = found.iter().find(|(k, _, _)| k == key) {
            return Some(label.clone());
        }
    }
    None
}

/// Return `(account, credentials_changed)` for `server`.
///
/// One read-only call, made through the shared read-only path.  Never fails:
/// an unreachable server or a missing identity tool leaves the connection
/// unlabelled.
///
/// The second value says the call renewed an expired token, so the caller —
/// which owns the workspace — knows to write it back.  This used to be
/// discarded, and it is the whole reason a connection could die quietly: the
/// probe runs on every connect and re-index, so against a provider that
/// rotates refresh tokens it redeemed the stored one, received a replacement,
/// and dropped it.  The retired token stayed on disk and the connection could
/// never be renewed again.
pub fn probe_account(
    server: &mut MCPServer,
    tool_caller: Option<Arc<dyn ReadOnlyToolCaller>>,
) -> (Option<String>, bool) {
    let tool_names: Vec<String> = server
        .tools
        .iter()
        .map(|t| t.name.clone())
        .filter(|n| !n.is_empty())
        .collect();
    let Some(tool) = resolve_account_tool(&tool_names) else {
        return (None, false);
    };

    let caller = ensure_tool_caller(tool_caller);
    let (payload, credentials_changed, error) =
        call_read_only(caller.as_deref(), server, &tool, &Map::new(), "mcp-account").into_tuple();
    let payload = match (error, payload) {
        (None, Some(payload)) if !payload.is_empty() => payload,
        _ => return (None, credentials_changed),
    };

    let account = extract_account(Some(&payload));
    if let Some(account) = &account {
        tracing::info!(
            target: LOGGER,
            "[mcp-account] '{}' is linked as {} (via {})",
            server.name, account, tool
        );
    }
    (account, credentials_changed)
}
