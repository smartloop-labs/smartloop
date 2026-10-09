//! Port of `smartloop/agents/mcp/mcp_tool_caller.py`.
//!
//! MCP server communication: tool calls, server lookup, auth (with serialized
//! refresh-token redemption shared by every caller), and prerequisites.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use regex::Regex;
use serde_json::{Map, Value};

use crate::agents::mcp::argument_extractor::MCPArgumentExtractor;
use crate::agents::mcp::dependency_analyzer::ToolDependencyAnalysis;
use crate::agents::mcp::mcp_agent::{py_repr, py_str};
use crate::agents::mcp::result_mapper::MCPResultMapper;
use crate::mcp::local_client::LocalMCPClient;
use crate::mcp::oauth::{MCPOAuthError, MCPOAuthService};
use crate::mcp::{MCPClient, MCPToolCallResult};
use crate::workspace::workspace::{MCPOAuthCredentials, MCPServer, MCPServerStatus, MCPServerType};

/// Logger name of the Python module.
pub const LOGGER: &str = "smartloop.agents.mcp.mcp_tool_caller";

/// How long a refusal stands before the same refresh token is offered again.
///
/// Serializing redemptions means a token endpoint that hangs is waited on once
/// per caller in turn rather than once in parallel, so without this a fan-out
/// of four calls turns one 30-second timeout into two minutes of a chat turn.
/// It is keyed on the token that failed, so a reconnection — which mints a
/// different one — is never held back by it.
pub const REFRESH_FAILURE_COOLDOWN_S: f64 = 30.0;

/// Per-server redemption locks, shared by every caller rather than held per
/// instance.
///
/// A refresh token is a property of the connection, not of whoever is holding
/// it at the time, and callers are built freely — one per orchestrator, a fresh
/// one for each provisioning probe, another for each read-only call that is not
/// passed one.  Per-instance locks therefore serialized a single orchestrator's
/// fan-out and nothing else, so a probe and a chat turn could redeem the same
/// single-use token in parallel and invalidate the grant between them: exactly
/// what the serialization was added to prevent.
///
/// Keyed per server, so a connection waiting on a slow token endpoint does not
/// hold up every other connection's renewal.
static REFRESH_LOCKS: Lazy<Mutex<HashMap<String, Arc<Mutex<()>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// `server id -> (refresh token that was refused, when to retry it)`.
pub static REFRESH_FAILURES: Lazy<Mutex<HashMap<String, (String, Instant)>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// The redemption lock for one connection, created on first use.
fn lock_for(server_id: &str) -> Arc<Mutex<()>> {
    REFRESH_LOCKS
        .lock()
        .entry(server_id.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

/// Forget every lock and cooldown (tests, and a workspace reload).
pub fn reset_refresh_state() {
    REFRESH_LOCKS.lock().clear();
    REFRESH_FAILURES.lock().clear();
}

/// Which part of a server a write-back carries.
///
/// The server handed to the persist hook is the caller's copy, current only in
/// the part that changed; writing the rest would put stale fields back over
/// whatever another caller saved meanwhile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerChange {
    /// `oauth_credentials`, after a refresh.
    Credentials,
    /// `status` and `last_error`, after a connection fails.
    Status,
}

/// Writes refreshed credentials / a failed status back to the workspace.
/// (Python: `Callable[[], None]`; Rust passes the changed server since it is
/// not shared by reference with the workspace.)
pub type PersistFn = Arc<dyn Fn(&MCPServer, ServerChange) -> anyhow::Result<()> + Send + Sync>;

/// Reads a server's credentials as the workspace holds them now, by server id
/// (`None` when the workspace does not have it).
///
/// Callers each work on their own copy of a server, so a copy cannot see a
/// refresh another caller just made; this is how the lock holder does.
pub type LiveCredentialsFn = Arc<dyn Fn(&str) -> Option<MCPOAuthCredentials> + Send + Sync>;

/// Test seam replacing the transport of [`MCPToolCaller::call_tool`]:
/// `(server, tool_name, arguments, access_token, cwd) -> result`.
pub type CallToolHook = Arc<
    dyn Fn(&MCPServer, &str, Option<&Map<String, Value>>, Option<&str>, Option<&str>) -> MCPToolCallResult
        + Send
        + Sync,
>;

/// The first `n` characters of `s`.
fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Python's `float(text)` succeeds: surrounding whitespace, a sign, digits
/// with single underscores between them, `inf`/`infinity`/`nan`.
fn parses_as_python_float(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    let chars: Vec<char> = trimmed.chars().collect();
    let mut cleaned = String::with_capacity(trimmed.len());
    for (i, c) in chars.iter().enumerate() {
        if *c == '_' {
            let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let next_digit = chars.get(i + 1).map(|n| n.is_ascii_digit()).unwrap_or(false);
            if !(prev_digit && next_digit) {
                return false;
            }
            continue;
        }
        cleaned.push(*c);
    }
    cleaned.parse::<f64>().is_ok()
}

/// Python truthiness of a JSON value.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Seconds since the Unix epoch (`time.time()`).
fn now_epoch() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// A failed tool-call result carrying `error`.
fn failed_result(error: String) -> MCPToolCallResult {
    MCPToolCallResult {
        success: false,
        content: None,
        is_error: false,
        error: Some(error),
        auth_expired: false,
    }
}

/// Handles MCP server communication: tool calls, server lookup, auth, and prerequisites.
pub struct MCPToolCaller {
    pub arg_extractor: Arc<MCPArgumentExtractor>,
    pub result_mapper: Arc<Mutex<MCPResultMapper>>,
    /// Called after a refresh replaces a server's credentials, to write them
    /// back to the workspace.  The hook is the *only* thing that persists
    /// them: the content crawl, which used to write them through its own
    /// `on_server_updated`, was removed with indexing.
    persist: Mutex<Option<PersistFn>>,
    /// Reads the workspace's current credentials before a redemption.
    live_credentials: Mutex<Option<LiveCredentialsFn>>,
    /// Replaces the real transport when set (tests).
    call_tool_hook: Mutex<Option<CallToolHook>>,
}

impl MCPToolCaller {
    /// Create a caller.
    pub fn new(
        arg_extractor: Arc<MCPArgumentExtractor>,
        result_mapper: Arc<Mutex<MCPResultMapper>>,
        persist: Option<PersistFn>,
    ) -> Self {
        Self {
            arg_extractor,
            result_mapper,
            persist: Mutex::new(persist),
            live_credentials: Mutex::new(None),
            call_tool_hook: Mutex::new(None),
        }
    }

    /// Install the callback that writes refreshed credentials back to disk.
    pub fn set_persist(&self, persist: Option<PersistFn>) {
        *self.persist.lock() = persist;
    }

    /// Install the callback that reads a server's current credentials.
    pub fn set_live_credentials(&self, live_credentials: Option<LiveCredentialsFn>) {
        *self.live_credentials.lock() = live_credentials;
    }

    /// Replace (or restore, with `None`) the tool-call transport.
    pub fn set_call_tool_hook(&self, hook: Option<CallToolHook>) {
        *self.call_tool_hook.lock() = hook;
    }

    /// Call one tool on a local (stdio) or remote (HTTP) server.
    ///
    /// A local server that cannot be launched comes back as a failed result
    /// carrying the launch error.
    pub fn call_tool(
        &self,
        server: &MCPServer,
        tool_name: &str,
        arguments: Option<&Map<String, Value>>,
        access_token: Option<&str>,
        cwd: Option<&str>,
    ) -> MCPToolCallResult {
        let hook = self.call_tool_hook.lock().clone();
        if let Some(hook) = hook {
            return hook(server, tool_name, arguments, access_token, cwd);
        }

        if let (MCPServerType::Local, Some(config)) = (server.server_type, server.config.as_ref()) {
            let effective_cwd = cwd.filter(|c| !c.is_empty()).or(config.cwd.as_deref());
            let mut client = LocalMCPClient::new(config.command.clone(), config.args.clone())
                .with_env(Some(config.env.clone()))
                .with_cwd(effective_cwd);
            let result = match client.enter() {
                Ok(()) => {
                    let result = client.call_tool(tool_name, arguments);
                    client.exit();
                    result
                }
                Err(e) => {
                    client.exit();
                    failed_result(e.to_string())
                }
            };
            if !result.success {
                tracing::error!(
                    target: LOGGER,
                    "[tool_caller] Local tool '{}' on server '{}' failed: {}",
                    tool_name,
                    server.name,
                    result.error.as_deref().unwrap_or("None")
                );
            }
            result
        } else {
            let server_url = Self::get_server_url(server);
            let mut client = MCPClient::new();
            client.enter();
            let result = client.call_tool(
                server_url.as_deref().unwrap_or(""),
                tool_name,
                arguments,
                access_token,
                "Bearer",
            );
            client.exit();
            if !result.success {
                tracing::error!(
                    target: LOGGER,
                    "[tool_caller] Remote tool '{}' on server '{}' ({}) failed: {}",
                    tool_name,
                    server.name,
                    server_url.as_deref().unwrap_or("None"),
                    result.error.as_deref().unwrap_or("None")
                );
            }
            result
        }
    }

    /// Call a tool, renewing an expired token once and trying again.
    ///
    /// Returns `(result, refreshed)` — `refreshed` says the server's
    /// credentials changed, so a caller that owns the write-back knows there is
    /// something to write.  (The persist hook already handles the callers that
    /// do not.)
    ///
    /// Every MCP call path had its own copy of this, and they had drifted: two
    /// guarded on a server URL, one reported the rotation, one retried with the
    /// wrong tool name, and the collect path had no retry at all, so an
    /// aged-out connection looked to retrieval like an app with nothing in it.
    pub fn call_with_refresh(
        &self,
        server: &mut MCPServer,
        tool_name: &str,
        arguments: Option<&Map<String, Value>>,
        cwd: Option<&str>,
    ) -> (MCPToolCallResult, bool) {
        let access_token = Self::get_access_token(server);
        let result = self.call_tool(server, tool_name, arguments, access_token.as_deref(), cwd);
        if result.success || !result.auth_expired {
            return (result, false);
        }

        let Some(new_token) = self.refresh_access_token(server, access_token.as_deref()) else {
            return (result, false);
        };

        let result = self.call_tool(server, tool_name, arguments, Some(&new_token), cwd);
        (result, true)
    }

    /// Find a server by id, then by name, then by one of its tool names.
    pub fn find_server<'a>(
        &self,
        mcp_servers: &'a [MCPServer],
        server_id: &str,
        server_name: &str,
    ) -> Option<&'a MCPServer> {
        if let Some(s) = mcp_servers.iter().find(|s| s.id == server_id) {
            return Some(s);
        }
        if let Some(s) = mcp_servers.iter().find(|s| s.name == server_name) {
            return Some(s);
        }
        for s in mcp_servers {
            for t in &s.tools {
                if t.name == server_id || t.name == server_name {
                    tracing::info!(
                        target: LOGGER,
                        "[find_server] Resolved '{}/{}' via tool-name match on server '{}' (id={})",
                        server_id,
                        server_name,
                        s.name,
                        s.id
                    );
                    return Some(s);
                }
            }
        }
        None
    }

    /// `{"name", "description", "input_schema"}` of `tool_name` on `server`.
    pub fn find_tool_schema(&self, server: &MCPServer, tool_name: &str) -> Option<Map<String, Value>> {
        let tool = server.tools.iter().find(|t| t.name == tool_name)?;
        let mut schema = Map::new();
        schema.insert("name".to_string(), Value::String(tool.name.clone()));
        schema.insert(
            "description".to_string(),
            tool.description.clone().map(Value::String).unwrap_or(Value::Null),
        );
        schema.insert(
            "input_schema".to_string(),
            tool.input_schema.clone().map(Value::Object).unwrap_or(Value::Null),
        );
        Some(schema)
    }

    /// The tool whose name/description best matches the words of `instruction`.
    pub fn find_tool_by_keyword(&self, server: &MCPServer, instruction: &str) -> Option<String> {
        static WORD: Lazy<Regex> = Lazy::new(|| Regex::new(r"\w{3,}").expect("valid regex"));
        if instruction.is_empty() {
            return None;
        }
        let instr_lower = instruction.to_lowercase();
        let words: Vec<&str> = WORD.find_iter(&instr_lower).map(|m| m.as_str()).collect();
        let mut best_tool: Option<String> = None;
        let mut best_score = 0;
        for tool in &server.tools {
            let mut score = 0;
            let t_name = tool.name.to_lowercase();
            let t_desc = tool.description.as_deref().unwrap_or("").to_lowercase();
            for word in &words {
                if t_name.contains(word) {
                    score += 3;
                }
                if t_desc.contains(word) {
                    score += 1;
                }
            }
            if score > best_score {
                best_score = score;
                best_tool = Some(tool.name.clone());
            }
        }
        if best_score > 0 {
            best_tool
        } else {
            None
        }
    }

    /// The remote server URL, if any.
    pub fn get_server_url(server: &MCPServer) -> Option<String> {
        server.remote_config.as_ref().map(|r| r.server_url.clone())
    }

    /// The OAuth access token, else the static token on the remote config.
    pub fn get_access_token(server: &MCPServer) -> Option<String> {
        if let Some(token) = server
            .oauth_credentials
            .as_ref()
            .and_then(|c| c.access_token.as_ref())
            .filter(|t| !t.is_empty())
        {
            return Some(token.clone());
        }
        // A server registered from a URL carrying its own credential
        // (`?token=...`) keeps it on the remote config, not in OAuth
        // credentials there are none of.  Discovery already falls back here, so
        // without the same fallback a static-token connection lists its tools
        // and then 401s on every call to them.
        if let Some(token) = server
            .remote_config
            .as_ref()
            .and_then(|r| r.token.as_ref())
            .filter(|t| !t.is_empty())
        {
            return Some(token.clone());
        }
        None
    }

    /// Redeem `server`'s refresh token, storing the result on the server.
    ///
    /// The new credentials are written back through the persist hook.  Leaving
    /// them in memory was a slow-acting break: providers that rotate refresh
    /// tokens hand back a new one and retire the old, so a refresh during a
    /// chat turn left the workspace holding a refresh token the provider had
    /// already invalidated.  The connection kept working until the process
    /// restarted and then could never be renewed again.
    ///
    /// `stale_token` is the access token the caller's request failed with.
    /// Given it, a redemption that lost the race to another caller returns
    /// that caller's token instead of spending the refresh token a second time
    /// — which is the thing that kills a single-use grant outright.  The
    /// comparison is against the workspace's live credentials when the
    /// live-credentials hook is installed, since `server` is only this
    /// caller's copy.
    pub fn refresh_access_token(&self, server: &mut MCPServer, stale_token: Option<&str>) -> Option<String> {
        let lock = lock_for(&server.id);
        let _guard = lock.lock();
        self.refresh_locked(server, stale_token)
    }

    fn refresh_locked(&self, server: &mut MCPServer, stale_token: Option<&str>) -> Option<String> {
        // Adopt what the workspace holds now: a copy taken before another
        // caller's refresh still carries the refresh token that refresh spent.
        let live_credentials = self.live_credentials.lock().clone();
        if let Some(live) = live_credentials.and_then(|read| read(&server.id)) {
            server.oauth_credentials = Some(live);
        }

        if let Some(stale) = stale_token.filter(|s| !s.is_empty()) {
            if let Some(current) = server.oauth_credentials.as_ref().and_then(|c| c.access_token.as_ref()) {
                if current != stale {
                    // Another caller rotated it while this one waited for the
                    // lock.  Its result is good and already persisted;
                    // redeeming again would not improve on it and, single-use,
                    // would invalidate it.
                    tracing::info!(
                        target: LOGGER,
                        "Reusing the token another caller just refreshed for '{}'",
                        server.name
                    );
                    return Some(current.clone());
                }
            }
        }

        let refresh_token = server
            .oauth_credentials
            .as_ref()
            .and_then(|c| c.refresh_token.clone())
            .filter(|t| !t.is_empty());
        let Some(refresh_token) = refresh_token else {
            // Nothing to redeem.  Callers reach this only after a 401, so for a
            // pasted credential that 401 is the verdict on the credential
            // itself rather than a sign its token aged out.
            self.record_credential_rejected(server);
            tracing::warn!(
                target: LOGGER,
                "No refresh token available for server '{}'",
                server.name
            );
            return None;
        };

        let token_endpoint = server
            .remote_config
            .as_ref()
            .and_then(|r| r.token_endpoint.clone())
            .filter(|t| !t.is_empty());
        let Some(token_endpoint) = token_endpoint else {
            tracing::warn!(
                target: LOGGER,
                "No token endpoint configured for server '{}'",
                server.name
            );
            return None;
        };
        let resource = server.remote_config.as_ref().and_then(|r| r.resource.clone());

        {
            let failures = REFRESH_FAILURES.lock();
            if let Some((refused, retry_at)) = failures.get(&server.id) {
                if *refused == refresh_token && Instant::now() < *retry_at {
                    tracing::info!(
                        target: LOGGER,
                        "Not retrying the refused refresh token for '{}' yet",
                        server.name
                    );
                    return None;
                }
            }
        }

        let (client_id, client_secret) = {
            let creds = server.oauth_credentials.as_ref().expect("checked above");
            (creds.client_id.clone(), creds.client_secret.clone())
        };

        let mut oauth = MCPOAuthService::new(None);
        oauth.enter();
        // RFC 8707: a refresh must carry the same resource indicator as the
        // grant it descends from, or strict servers (monday.com) reject it.
        let outcome = oauth.refresh_access_token(
            &token_endpoint,
            &refresh_token,
            &client_id,
            client_secret.as_deref(),
            resource.as_deref(),
        );
        oauth.exit();

        let tokens = match outcome {
            Ok(tokens) => tokens,
            Err(MCPOAuthError::TokenExchange(e)) => {
                tracing::error!(
                    target: LOGGER,
                    "Token refresh failed for server '{}': {}",
                    server.name,
                    e
                );
                self.note_refusal(server, &refresh_token);
                if e.grant_is_dead() {
                    // The server has retired this grant.  Re-minting it cannot
                    // work again, so this is the one refusal that should cost
                    // the user their connection — a blip or a 5xx must not.
                    self.record_grant_expired(server);
                }
                return None;
            }
            Err(e) => {
                tracing::error!(
                    target: LOGGER,
                    "Token refresh failed for server '{}': {}",
                    server.name,
                    e
                );
                self.note_refusal(server, &refresh_token);
                return None;
            }
        };

        REFRESH_FAILURES.lock().remove(&server.id);
        if let Some(creds) = server.oauth_credentials.as_mut() {
            creds.access_token = Some(tokens.access_token.clone());
            if let Some(new_refresh) = tokens.refresh_token.as_ref().filter(|t| !t.is_empty()) {
                creds.refresh_token = Some(new_refresh.clone());
            }
            if let Some(expires_in) = tokens.expires_in.filter(|e| *e != 0) {
                creds.expires_at = Some(now_epoch() + expires_in as f64);
            }
            if let Some(scope) = tokens.scope.as_ref().filter(|s| !s.is_empty()) {
                // A refresh may come back narrower than the grant it descends
                // from (RFC 6749 §6 allows the server to narrow, never to
                // widen).  Record it, or the stored scope describes a grant
                // that no longer exists.
                creds.scope = Some(scope.clone());
            }
        }

        self.write_back(server, ServerChange::Credentials, "Refreshed");
        tracing::info!(
            target: LOGGER,
            "Refreshed access token for server '{}'",
            server.name
        );
        Some(tokens.access_token)
    }

    /// Hold `refresh_token` back briefly after the endpoint refused it.
    fn note_refusal(&self, server: &MCPServer, refresh_token: &str) {
        REFRESH_FAILURES.lock().insert(
            server.id.clone(),
            (
                refresh_token.to_string(),
                Instant::now() + Duration::from_secs_f64(REFRESH_FAILURE_COOLDOWN_S),
            ),
        );
    }

    /// Persist a change to `server`, logging rather than failing.
    ///
    /// The state in memory is good, so whatever this change was made for
    /// should still go ahead; it is the *next* process that pays for a failed
    /// write.
    fn write_back(&self, server: &MCPServer, change: ServerChange, what: &str) {
        let persist = self.persist.lock().clone();
        let Some(persist) = persist else {
            return;
        };
        if let Err(e) = persist(server, change) {
            tracing::warn!(
                target: LOGGER,
                "{} '{}' but could not persist the change: {}",
                what,
                server.name,
                e
            );
        }
    }

    /// Fail a connection whose pasted credential the server refused.
    ///
    /// Only for a credential that was pasted rather than granted.  An OAuth
    /// connection has a refresh to fall back on and a failed refresh can be
    /// transient, but a token the server has stopped accepting never starts
    /// working again on its own.  Left unreported it looks exactly like a tool
    /// with no data: `status: ready`, an empty answer, nothing in the log.
    ///
    /// Returns whether the server's state changed.
    pub fn record_credential_rejected(&self, server: &mut MCPServer) -> bool {
        let has_token = server
            .remote_config
            .as_ref()
            .and_then(|r| r.token.as_ref())
            .map(|t| !t.is_empty())
            .unwrap_or(false);
        if !has_token {
            return false;
        }
        self.fail(
            server,
            "The access token for this connection was rejected. Enter a new token to reconnect.",
        )
    }

    /// Fail a connection whose OAuth grant the server has retired.
    ///
    /// Reached only for `invalid_grant`/`invalid_client` — a refusal that
    /// re-minting the same grant cannot recover from.  Until this existed a
    /// dead connection reported `ready` with its full tool list forever, while
    /// every call 401'd and was swallowed.
    pub fn record_grant_expired(&self, server: &mut MCPServer) -> bool {
        self.fail(
            server,
            "This connection's authorization has expired. Reconnect it to keep using it.",
        )
    }

    fn fail(&self, server: &mut MCPServer, message: &str) -> bool {
        if server.status == MCPServerStatus::Failed && server.last_error.as_deref() == Some(message) {
            return false;
        }
        server.status = MCPServerStatus::Failed;
        server.last_error = Some(message.to_string());
        tracing::warn!(
            target: LOGGER,
            "Server '{}' can no longer authenticate: {}",
            server.name,
            message
        );
        self.write_back(server, ServerChange::Status, "Failed");
        true
    }

    /// Human-readable issues with argument values (URLs, ids, numbers, enums).
    pub fn validate_argument_values(
        &self,
        arguments: &Map<String, Value>,
        properties: &Map<String, Value>,
    ) -> Vec<String> {
        static URL_START: Lazy<Regex> = Lazy::new(|| Regex::new(r"^https?://").expect("valid regex"));
        let mut issues: Vec<String> = Vec::new();
        let empty = Value::Object(Map::new());

        for (param_name, value) in arguments {
            if value.is_null() {
                continue;
            }
            let param_info = properties.get(param_name).unwrap_or(&empty);

            if self.arg_extractor.is_url_parameter(param_name, param_info) {
                if let Value::String(s) = value {
                    if !URL_START.is_match(s.trim()) {
                        issues.push(format!(
                            "'{}' expects a URL but got plain text: {}...",
                            param_name,
                            take_chars(s, 80)
                        ));
                    }
                }
                continue;
            }

            if self.arg_extractor.is_id_parameter(param_name, param_info) {
                if let Value::String(s) = value {
                    if !self.arg_extractor.is_valid_id_value(value) {
                        issues.push(format!(
                            "'{}' expects an identifier but got: {}...",
                            param_name,
                            take_chars(s, 80)
                        ));
                    }
                }
                continue;
            }

            let param_type = param_info.get("type").and_then(Value::as_str).unwrap_or("");
            if matches!(param_type, "integer" | "number") {
                if let Value::String(s) = value {
                    if !parses_as_python_float(s) {
                        issues.push(format!(
                            "'{}' expects a {} but got: {}",
                            param_name,
                            param_type,
                            take_chars(s, 80)
                        ));
                    }
                    continue;
                }
            }

            if let Some(enum_values) = param_info.get("enum").filter(|e| truthy(e)) {
                let contained = match enum_values {
                    Value::Array(options) => options.contains(value),
                    Value::Object(options) => value.as_str().map(|k| options.contains_key(k)).unwrap_or(false),
                    Value::String(options) => value.as_str().map(|v| options.contains(v)).unwrap_or(false),
                    _ => false,
                };
                if !contained {
                    issues.push(format!(
                        "'{}' expects one of {} but got: {}",
                        param_name,
                        py_repr(enum_values),
                        py_str(value)
                    ));
                }
            }
        }

        issues
    }

    /// Resolve id parameters by calling prerequisite list tools (or from
    /// earlier results when they already contain the id).
    pub fn execute_prerequisites(
        &self,
        dep_analysis: &ToolDependencyAnalysis,
        server: &mut MCPServer,
        user_message: &str,
        input_schema: &Map<String, Value>,
        previous_results: Option<&[Map<String, Value>]>,
    ) -> Map<String, Value> {
        let mut resolved_ids = Map::new();
        let properties = match input_schema.get("properties") {
            Some(Value::Object(p)) => p.clone(),
            _ => Map::new(),
        };

        let mut prior_corpus = String::new();
        if let Some(results) = previous_results.filter(|r| !r.is_empty()) {
            prior_corpus = results
                .iter()
                .map(|r| match r.get("result") {
                    Some(v) => py_str(v),
                    None => String::new(),
                })
                .collect::<Vec<_>>()
                .join("\n");
        }

        for dep in &dep_analysis.dependencies {
            let Some(prerequisite_tool) = dep.prerequisite_tool.as_deref().filter(|t| !t.is_empty()) else {
                continue;
            };

            if !prior_corpus.is_empty() {
                let resolved = self.result_mapper.lock().get_resolved_arguments(
                    std::slice::from_ref(&dep.param_name),
                    &properties,
                    Some(&prior_corpus),
                );
                if resolved.get(&dep.param_name).map(truthy).unwrap_or(false) {
                    tracing::info!(
                        target: LOGGER,
                        "[prereq] Resolved '{}' from previous results (skipping API call to '{}')",
                        dep.param_name,
                        prerequisite_tool
                    );
                    resolved_ids.extend(resolved);
                    continue;
                }
            }

            tracing::info!(
                target: LOGGER,
                "[prereq] Calling prerequisite tool: {}",
                prerequisite_tool
            );
            let empty_args = Map::new();
            let (result, _refreshed) =
                self.call_with_refresh(server, prerequisite_tool, Some(&empty_args), None);

            match result.content.as_ref().filter(|c| result.success && !c.is_empty()) {
                Some(content) => {
                    let mut mapper = self.result_mapper.lock();
                    mapper.process_list_result(prerequisite_tool, &Value::Array(content.clone()), None);
                    let resolved = mapper.get_resolved_arguments(
                        std::slice::from_ref(&dep.param_name),
                        &properties,
                        Some(user_message),
                    );
                    drop(mapper);
                    let shown = py_repr(&Value::Object(resolved.clone()));
                    tracing::info!(target: LOGGER, "[prereq] Resolved from API call: {}", shown);
                    resolved_ids.extend(resolved);
                }
                None => {
                    tracing::warn!(
                        target: LOGGER,
                        "[prereq] Prerequisite tool '{}' failed: {}",
                        prerequisite_tool,
                        result.error.as_deref().unwrap_or("None")
                    );
                }
            }
        }

        resolved_ids
    }
}
