//! Port of `tests/unit/test_mcp_oauth_state_eviction.py`.
//!
//! The router's in-memory OAuth state must not grow without bound.
//!
//! `PENDING_OAUTH_SESSIONS` is popped by the callback, but an abandoned flow
//! (the user closes the browser tab, the provider errors out before
//! redirecting) never produces one — and the entry holds a `client_secret` and
//! `code_verifier`.  `COMPLETED_OAUTH_RESPONSES` was never evicted at all; it
//! retains a full HTML response per registration purely to absorb the
//! browser's duplicate GET of the redirect, which arrives within seconds.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};
use serial_test::serial;

use smartloop::api::routers::mcp_servers::{self as m, HtmlPage, OAuthSession};

/// The `_clean_state` fixture: these are process globals — don't let one test
/// leak into the next.
struct CleanState;

impl CleanState {
    fn new() -> Self {
        m::PENDING_OAUTH_SESSIONS.lock().clear();
        m::COMPLETED_OAUTH_RESPONSES.lock().clear();
        CleanState
    }
}

impl Drop for CleanState {
    fn drop(&mut self) {
        m::PENDING_OAUTH_SESSIONS.lock().clear();
        m::COMPLETED_OAUTH_RESPONSES.lock().clear();
    }
}

/// `time.time()`.
fn time_now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).expect("clock after the epoch").as_secs_f64()
}

fn session(value: Value) -> OAuthSession {
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

mod test_pending_session_eviction {
    use super::*;

    #[test]
    #[serial(mcp_global)]
    fn test_abandoned_session_expires() {
        let _clean = CleanState::new();
        let now = time_now();
        m::PENDING_OAUTH_SESSIONS.lock().insert(
            "stale".to_string(),
            session(json!({"created_at": now - m::OAUTH_SESSION_TTL_S - 1.0, "client_secret": "s3cret"})),
        );
        m::PENDING_OAUTH_SESSIONS
            .lock()
            .insert("fresh".to_string(), session(json!({"created_at": now, "client_secret": "s3cret"})));

        m::purge_oauth_state(Some(now));

        let pending = m::PENDING_OAUTH_SESSIONS.lock();
        assert!(!pending.contains_key("stale"));
        assert!(pending.contains_key("fresh"));
    }

    /// An entry with no `created_at` must not read as infinitely old and
    /// vanish on the very next sweep — it gets stamped and expires one TTL on.
    #[test]
    #[serial(mcp_global)]
    fn test_untimestamped_session_is_stamped_not_dropped() {
        let _clean = CleanState::new();
        let now = time_now();
        m::PENDING_OAUTH_SESSIONS
            .lock()
            .insert("legacy".to_string(), session(json!({"client_secret": "s3cret"})));

        m::purge_oauth_state(Some(now));
        {
            let pending = m::PENDING_OAUTH_SESSIONS.lock();
            assert!(pending.contains_key("legacy"));
            assert_eq!(pending["legacy"]["created_at"], json!(now));
        }

        m::purge_oauth_state(Some(now + m::OAUTH_SESSION_TTL_S + 1.0));
        assert!(!m::PENDING_OAUTH_SESSIONS.lock().contains_key("legacy"));
    }
}

mod test_completed_response_eviction {
    use super::*;

    #[test]
    #[serial(mcp_global)]
    fn test_cached_response_expires() {
        let _clean = CleanState::new();
        let now = time_now();
        m::COMPLETED_OAUTH_RESPONSES.lock().insert("old".to_string(), (now - 1.0, HtmlPage::ok("old")));
        m::COMPLETED_OAUTH_RESPONSES.lock().insert("new".to_string(), (now + 60.0, HtmlPage::ok("new")));

        m::purge_oauth_state(Some(now));

        let completed = m::COMPLETED_OAUTH_RESPONSES.lock();
        assert!(!completed.contains_key("old"));
        assert!(completed.contains_key("new"));
    }

    #[test]
    #[serial(mcp_global)]
    fn test_cache_is_size_capped_oldest_first() {
        let _clean = CleanState::new();
        let now = time_now();
        for i in 0..(m::OAUTH_RESPONSE_MAX + 10) {
            m::COMPLETED_OAUTH_RESPONSES
                .lock()
                .insert(format!("s{i}"), (now + 3600.0, HtmlPage::ok(i.to_string())));
        }

        m::purge_oauth_state(Some(now));

        let completed = m::COMPLETED_OAUTH_RESPONSES.lock();
        assert_eq!(completed.len(), m::OAUTH_RESPONSE_MAX);
        assert!(!completed.contains_key("s0")); // oldest evicted
        assert!(completed.contains_key(&format!("s{}", m::OAUTH_RESPONSE_MAX + 9)));
    }
}
