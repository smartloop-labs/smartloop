//! Port of `tests/unit/test_mcp_refresh_shared_across_callers.py`.
//!
//! A refresh token belongs to the connection, not to whoever is holding it.
//!
//! Redemption was serialized on a lock held per `MCPToolCaller`, and callers
//! are built freely: one per orchestrator, a fresh one for every provisioning
//! probe, another for each read-only call that is not passed one.  So the lock
//! covered a single orchestrator's fan-out and nothing else — a probe and a
//! chat turn held different locks over the same token and could redeem it in
//! parallel, which with a single-use grant invalidates it between them.  That
//! is the failure the serialization was added to prevent.
//!
//! Python shares one server object between the two threads; the Rust analogue
//! of that shared object is the server behind a mutex, each caller taking it
//! for its call.  The token endpoint is a fake HTTP layer (Python patches
//! `MCPOAuthService`) that is slow enough for a second caller to arrive.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde_json::json;
use serial_test::serial;

use smartloop::agents::mcp::mcp_tool_caller::reset_refresh_state;
use smartloop::mcp::client::{HttpResponse, HttpTransportOverride};
use smartloop::workspace::workspace::MCPServerStatus;
use smartloop::workspace::MCPServer;

use crate::shared::mcp::{ok_json, oauth_server, tool_caller, FakeHttp};

const TOKEN_ENDPOINT: &str = "https://auth.notion.com/token";

fn server(server_id: &str) -> MCPServer {
    let mut s = oauth_server(
        server_id,
        "notion",
        "https://mcp.notion.com/mcp",
        Some(TOKEN_ENDPOINT),
        Some("stale"),
        Some("rt-1"),
    );
    s.status = MCPServerStatus::Ready;
    if let Some(creds) = s.oauth_credentials.as_mut() {
        creds.client_secret = Some("s".into());
    }
    s
}

/// A token endpoint that is slow enough for a second caller to arrive.
/// While `refuse` is set it answers 503 (and does not count the call).
struct Exchange {
    calls: Arc<AtomicUsize>,
    refuse: Arc<AtomicBool>,
    _guard: HttpTransportOverride,
}

impl Exchange {
    fn install(delay: Duration) -> Self {
        let calls = Arc::new(AtomicUsize::new(0));
        let refuse = Arc::new(AtomicBool::new(false));
        let (c, r) = (calls.clone(), refuse.clone());
        let fake = FakeHttp::new(move |_, _, _| {
            if r.load(Ordering::SeqCst) {
                return Ok(HttpResponse::new(503, "unavailable"));
            }
            let n = c.fetch_add(1, Ordering::SeqCst) + 1;
            std::thread::sleep(delay);
            Ok(ok_json(json!({
                "access_token": format!("fresh-{n}"),
                "refresh_token": format!("rt-{}", n + 1),
                "expires_in": 3600,
                "token_type": "Bearer",
                "scope": null,
            })))
        });
        Self { calls, refuse, _guard: HttpTransportOverride::install(fake) }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

/// Two separately-constructed callers refresh concurrently.
fn run_both(server_a: Arc<Mutex<MCPServer>>, server_b: Arc<Mutex<MCPServer>>) -> (usize, Vec<(String, Option<String>)>) {
    let exchange = Exchange::install(Duration::from_millis(50));
    let results = Arc::new(Mutex::new(Vec::new()));

    let go = |key: &'static str, server: Arc<Mutex<MCPServer>>| {
        let results = results.clone();
        std::thread::spawn(move || {
            // A fresh caller, as in prod.
            let caller = tool_caller();
            let token = caller.refresh_access_token(&mut server.lock(), Some("stale"));
            results.lock().push((key.to_string(), token));
        })
    };
    let threads = [go("a", server_a), go("b", server_b)];
    for t in threads {
        t.join().expect("a caller panicked");
    }
    let calls = exchange.calls();
    let results = results.lock().clone();
    (calls, results)
}

fn result_of(results: &[(String, Option<String>)], key: &str) -> Option<String> {
    results.iter().find(|(k, _)| k == key).and_then(|(_, t)| t.clone())
}

// ------------------------------------------------ one connection is redeemed once

#[test]
#[serial(mcp_global)]
fn test_two_callers_spend_the_refresh_token_once() {
    reset_refresh_state();
    let shared = Arc::new(Mutex::new(server("s1")));

    let (calls, results) = run_both(shared.clone(), shared);

    assert_eq!(calls, 1);
    // The loser reuses what the winner minted rather than redeeming again.
    assert_eq!(result_of(&results, "a").as_deref(), Some("fresh-1"));
    assert_eq!(result_of(&results, "b").as_deref(), Some("fresh-1"));
    reset_refresh_state();
}

#[test]
#[serial(mcp_global)]
fn test_the_connection_keeps_the_rotation() {
    reset_refresh_state();
    let shared = Arc::new(Mutex::new(server("s1")));

    run_both(shared.clone(), shared.clone());

    let creds = shared.lock().oauth_credentials.clone().unwrap();
    assert_eq!(creds.access_token.as_deref(), Some("fresh-1"));
    assert_eq!(creds.refresh_token.as_deref(), Some("rt-2"));
    reset_refresh_state();
}

// --------------------------------------------- connections do not block each other

#[test]
#[serial(mcp_global)]
fn test_two_servers_refresh_concurrently() {
    // Per-server locks: a slow token endpoint stalls its own connection.
    reset_refresh_state();
    let (calls, results) = run_both(Arc::new(Mutex::new(server("s1"))), Arc::new(Mutex::new(server("s2"))));

    assert_eq!(calls, 2);
    let mut tokens: Vec<String> = results.into_iter().filter_map(|(_, t)| t).collect();
    tokens.sort();
    assert_eq!(tokens, vec!["fresh-1".to_string(), "fresh-2".to_string()]);
    reset_refresh_state();
}

// ---------------------------------------------------- the cooldown is shared too

/// Record a refusal of `refresh_token` against `server_id` the way a caller
/// does: by having the token endpoint refuse it.  (Python writes
/// `mod._refresh_failures[server.id]` directly.)
fn record_refusal(exchange: &Exchange, server_id: &str, refresh_token: &str) {
    let mut refused = server(server_id);
    refused.oauth_credentials.as_mut().unwrap().refresh_token = Some(refresh_token.to_string());
    exchange.refuse.store(true, Ordering::SeqCst);
    assert_eq!(tool_caller().refresh_access_token(&mut refused, Some("stale")), None);
    exchange.refuse.store(false, Ordering::SeqCst);
}

#[test]
#[serial(mcp_global)]
fn test_a_refusal_recorded_by_one_caller_holds_back_another() {
    reset_refresh_state();
    let exchange = Exchange::install(Duration::ZERO);
    record_refusal(&exchange, "s1", "rt-1");

    let mut srv = server("s1");
    let token = tool_caller().refresh_access_token(&mut srv, Some("stale"));

    assert_eq!(token, None);
    assert_eq!(exchange.calls(), 0, "the refused token must not be re-offered");
    reset_refresh_state();
}

#[test]
#[serial(mcp_global)]
fn test_a_reconnection_is_not_held_back() {
    // The cooldown is keyed on the token that failed, not the server.
    reset_refresh_state();
    let exchange = Exchange::install(Duration::ZERO);
    record_refusal(&exchange, "s1", "rt-old");

    let mut srv = server("s1");
    let token = tool_caller().refresh_access_token(&mut srv, Some("stale"));

    assert_eq!(token.as_deref(), Some("fresh-1"));
    reset_refresh_state();
}
