//! Remote MCP sessions kept across tool calls.
//!
//! A tool call used to open a client, `initialize`, call and throw the session
//! away, so one chat turn repeated the handshake for every collect, record
//! fetch and execute step.  Sessions here outlive the call and expire lazily
//! after `SLP_MCP_KEEP_ALIVE` seconds idle; there is no timer thread.
//!
//! Keyed by server id, URL and a hash of the access token.  The token keeps
//! two projects holding the same imported server id apart (an import strips
//! credentials, so each authenticates on its own), and a refreshed token gets
//! a fresh session rather than one opened under the old grant.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use once_cell::sync::Lazy;
use parking_lot::{Mutex, MutexGuard};

use crate::config::app_settings::AppSettings;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SessionKey {
    server_id: String,
    url: String,
    token_fp: u64,
}

/// What the handshake established.
#[derive(Debug, Default)]
pub(crate) struct SessionState {
    /// The `Mcp-Session-Id` the server issued, if it issues one.
    pub session_id: Option<String>,
    /// `initialize` has succeeded; servers that issue no session id are
    /// initialized once too.
    pub initialized: bool,
}

/// One server's session, shared by every call made under the same key.
#[derive(Debug)]
pub struct PooledSession {
    /// Held across `initialize`, so concurrent callers handshake once.
    state: Mutex<SessionState>,
    request_id: AtomicU64,
    last_used: Mutex<Instant>,
}

impl PooledSession {
    fn new() -> Self {
        Self {
            state: Mutex::new(SessionState::default()),
            request_id: AtomicU64::new(0),
            last_used: Mutex::new(Instant::now()),
        }
    }

    pub(crate) fn state(&self) -> MutexGuard<'_, SessionState> {
        self.state.lock()
    }

    /// The next JSON-RPC id, unique across every caller sharing the session.
    pub(crate) fn next_request_id(&self) -> u64 {
        self.request_id.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Forget the handshake if `session_id` is still the session held, so the
    /// next caller initializes again; a caller that already replaced it wins.
    pub(crate) fn forget(&self, session_id: Option<&str>) {
        let mut state = self.state.lock();
        if state.session_id.as_deref() == session_id {
            *state = SessionState::default();
        }
    }

    /// Mark the session as used now.
    pub(crate) fn touch(&self) {
        *self.last_used.lock() = Instant::now();
    }

    fn idle_for(&self) -> Duration {
        self.last_used.lock().elapsed()
    }
}

static SESSIONS: Lazy<Mutex<HashMap<SessionKey, Arc<PooledSession>>>> = Lazy::new(|| Mutex::new(HashMap::new()));

fn token_fingerprint(access_token: Option<&str>) -> u64 {
    let mut hasher = DefaultHasher::new();
    access_token.filter(|t| !t.is_empty()).hash(&mut hasher);
    hasher.finish()
}

/// The session to reuse for `server_id` at `url` under `access_token`,
/// created on first use; `None` when `SLP_MCP_KEEP_ALIVE` is `0`.
///
/// Sessions idle past the keep-alive are dropped first.
pub fn session_for(server_id: &str, url: &str, access_token: Option<&str>) -> Option<Arc<PooledSession>> {
    let keep_alive = AppSettings::new().mcp_keep_alive;
    if keep_alive == 0 {
        return None;
    }
    let key = SessionKey {
        server_id: server_id.to_string(),
        url: url.to_string(),
        token_fp: token_fingerprint(access_token),
    };
    let mut sessions = SESSIONS.lock();
    if keep_alive > 0 {
        let limit = Duration::from_secs(keep_alive as u64);
        sessions.retain(|_, s| s.idle_for() < limit);
    }
    let session = sessions.entry(key).or_insert_with(|| Arc::new(PooledSession::new())).clone();
    session.touch();
    Some(session)
}

/// Drop every session held for `server_id`, so its next call initializes
/// again (the server was removed, reconnected or re-provisioned).
pub fn invalidate_mcp_sessions(server_id: &str) {
    SESSIONS.lock().retain(|k, _| k.server_id != server_id);
}

/// Drop every session (tests, and a transport override).
pub fn reset_mcp_sessions() {
    SESSIONS.lock().clear();
}

/// Age every session by `by`, as if nothing had used it since (tests).
pub fn backdate_mcp_sessions(by: Duration) {
    for session in SESSIONS.lock().values() {
        let mut last_used = session.last_used.lock();
        *last_used = last_used.checked_sub(by).unwrap_or(*last_used);
    }
}
