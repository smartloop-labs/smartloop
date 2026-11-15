//! Remote MCP sessions outlive a single tool call.
//!
//! Every call used to `initialize` again, and servers that issue no
//! `Mcp-Session-Id` were re-initialized before every request.  These drive
//! `MCPToolCaller::call_tool` against a scripted server and count handshakes.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use serial_test::serial;

use smartloop::agents::mcp::mcp_tool_caller::reset_refresh_state;
use smartloop::mcp::client::{
    set_http_transport_override, shared_http_transport, HttpBody, HttpResponse, HttpTransport, HttpTransportOverride,
};
use smartloop::mcp::session_pool::{backdate_mcp_sessions, invalidate_mcp_sessions, reset_mcp_sessions};
use smartloop::workspace::MCPServer;

use crate::shared::mcp::{capture_logs, ok_json, oauth_server, tool_caller, EnvGuard, FakeHttp, ServiceEnv};

const URL: &str = "https://mcp.example.com/mcp";
const TOKEN_ENDPOINT: &str = "https://auth.example.com/token";

fn server(id: &str) -> MCPServer {
    oauth_server(id, "example", URL, None, Some("tok"), None)
}

fn rpc_method(body: Option<&HttpBody>) -> String {
    match body {
        Some(HttpBody::Json(v)) => v.get("method").and_then(Value::as_str).unwrap_or("").to_string(),
        _ => String::new(),
    }
}

/// A server answering `initialize` (with `session_id` as its
/// `Mcp-Session-Id`, if any), notifications and `tools/call`.
fn mcp_server(session_id: Option<&'static str>) -> Arc<FakeHttp> {
    FakeHttp::new(move |_, _, body| {
        let method = rpc_method(body);
        Ok(match method.as_str() {
            "initialize" => {
                let resp = HttpResponse::json(
                    200,
                    &json!({"jsonrpc": "2.0", "id": "1", "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "example", "version": "1"},
                    }}),
                );
                match session_id {
                    Some(id) => resp.with_header("mcp-session-id", id),
                    None => resp,
                }
            }
            "tools/call" => HttpResponse::json(
                200,
                &json!({"jsonrpc": "2.0", "id": "2", "result": {
                    "content": [{"type": "text", "text": "ok"}],
                }}),
            ),
            _ => HttpResponse::new(202, ""),
        })
    })
}

/// The JSON-RPC methods posted, in order.
fn methods(fake: &FakeHttp) -> Vec<String> {
    fake.calls.lock().iter().map(|c| rpc_method(c.body.as_ref())).collect()
}

fn count(fake: &FakeHttp, method: &str) -> usize {
    methods(fake).iter().filter(|m| *m == method).count()
}

/// The `Mcp-Session-Id` each `tools/call` carried.
fn call_session_ids(fake: &FakeHttp) -> Vec<Option<String>> {
    fake.calls
        .lock()
        .iter()
        .filter(|c| rpc_method(c.body.as_ref()) == "tools/call")
        .map(|c| c.headers.iter().find(|(k, _)| k == "Mcp-Session-Id").map(|(_, v)| v.clone()))
        .collect()
}

fn call(server: &MCPServer, token: Option<&str>) {
    let result = tool_caller().call_tool(server, "search", None, token, None);
    assert!(result.success, "call failed: {:?}", result.error);
}

#[test]
#[serial(mcp_global)]
fn test_calls_share_one_handshake() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", None)]);
    let fake = mcp_server(Some("sess-1"));
    let _guard = HttpTransportOverride::install(fake.clone());
    let srv = server("s1");

    call(&srv, Some("tok"));
    call(&srv, Some("tok"));
    call(&srv, Some("tok"));

    assert_eq!(count(&fake, "initialize"), 1);
    assert_eq!(count(&fake, "notifications/initialized"), 1);
    assert_eq!(count(&fake, "tools/call"), 3);
    assert_eq!(call_session_ids(&fake), vec![Some("sess-1".to_string()); 3]);
}

#[test]
#[serial(mcp_global)]
fn test_a_server_without_session_ids_is_initialized_once() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", None)]);
    let fake = mcp_server(None);
    let _guard = HttpTransportOverride::install(fake.clone());
    let srv = server("s1");

    call(&srv, Some("tok"));
    call(&srv, Some("tok"));

    assert_eq!(count(&fake, "initialize"), 1);
    assert_eq!(call_session_ids(&fake), vec![None, None]);
}

#[test]
#[serial(mcp_global)]
fn test_request_ids_keep_counting_across_calls() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", None)]);
    let fake = mcp_server(Some("sess-1"));
    let _guard = HttpTransportOverride::install(fake.clone());
    let srv = server("s1");

    call(&srv, Some("tok"));
    call(&srv, Some("tok"));

    let ids: Vec<String> = fake
        .calls
        .lock()
        .iter()
        .filter_map(|c| match &c.body {
            Some(HttpBody::Json(v)) => v.get("id").and_then(Value::as_str).map(str::to_string),
            _ => None,
        })
        .collect();
    assert_eq!(ids, vec!["1", "2", "3"]);
}

#[test]
#[serial(mcp_global)]
fn test_keep_alive_zero_initializes_every_call() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", Some("0"))]);
    let fake = mcp_server(Some("sess-1"));
    let _guard = HttpTransportOverride::install(fake.clone());
    let srv = server("s1");

    call(&srv, Some("tok"));
    call(&srv, Some("tok"));

    assert_eq!(count(&fake, "initialize"), 2);
}

#[test]
#[serial(mcp_global)]
fn test_an_idle_session_expires() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", Some("5m"))]);
    let fake = mcp_server(Some("sess-1"));
    let _guard = HttpTransportOverride::install(fake.clone());
    let srv = server("s1");

    call(&srv, Some("tok"));
    backdate_mcp_sessions(Duration::from_secs(299));
    call(&srv, Some("tok"));
    assert_eq!(count(&fake, "initialize"), 1, "used within the keep-alive");

    backdate_mcp_sessions(Duration::from_secs(301));
    call(&srv, Some("tok"));
    assert_eq!(count(&fake, "initialize"), 2, "idle past the keep-alive");
}

#[test]
#[serial(mcp_global)]
fn test_a_negative_keep_alive_never_expires() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", Some("-1"))]);
    let fake = mcp_server(Some("sess-1"));
    let _guard = HttpTransportOverride::install(fake.clone());
    let srv = server("s1");

    call(&srv, Some("tok"));
    backdate_mcp_sessions(Duration::from_secs(86_400));
    call(&srv, Some("tok"));

    assert_eq!(count(&fake, "initialize"), 1);
}

#[test]
#[serial(mcp_global)]
fn test_a_new_token_gets_its_own_session() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", None)]);
    let fake = mcp_server(Some("sess-1"));
    let _guard = HttpTransportOverride::install(fake.clone());
    let srv = server("s1");

    call(&srv, Some("tok"));
    call(&srv, Some("refreshed"));

    assert_eq!(count(&fake, "initialize"), 2);
}

#[test]
#[serial(mcp_global)]
fn test_servers_do_not_share_sessions() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", None)]);
    let fake = mcp_server(Some("sess-1"));
    let _guard = HttpTransportOverride::install(fake.clone());

    call(&server("s1"), Some("tok"));
    call(&server("s2"), Some("tok"));

    assert_eq!(count(&fake, "initialize"), 2);
}

#[test]
#[serial(mcp_global)]
fn test_concurrent_calls_handshake_once() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", None)]);
    let inner = mcp_server(Some("sess-1"));
    let slow = inner.clone();
    // A slow `initialize`, so the second caller arrives while it is in flight.
    let fake = FakeHttp::new(move |method, url, body| {
        if rpc_method(body) == "initialize" {
            std::thread::sleep(Duration::from_millis(100));
        }
        match (method, body) {
            ("POST", Some(b)) => slow.post(url, b, &[]),
            _ => slow.get(url),
        }
    });
    let _guard = HttpTransportOverride::install(fake.clone());

    let threads: Vec<_> = (0..4)
        .map(|_| {
            std::thread::spawn(|| {
                call(&server("s1"), Some("tok"));
            })
        })
        .collect();
    for t in threads {
        t.join().expect("a caller panicked");
    }

    assert_eq!(count(&fake, "initialize"), 1);
    assert_eq!(count(&fake, "tools/call"), 4);
    reset_mcp_sessions();
}

// ------------------------------------------------------- a session the server lost

/// A server that numbers its sessions and answers the next `refusals`
/// `tools/call` requests with `refusal`.
struct ExpiringServer {
    fake: Arc<FakeHttp>,
    refusals: Arc<AtomicUsize>,
}

impl ExpiringServer {
    fn install(refusal: HttpResponse) -> (Self, HttpTransportOverride) {
        let sessions = Arc::new(AtomicUsize::new(0));
        let refusals = Arc::new(AtomicUsize::new(0));
        let (n, left) = (sessions.clone(), refusals.clone());
        let fake = FakeHttp::new(move |_, _, body| {
            Ok(match rpc_method(body).as_str() {
                "initialize" => {
                    let id = n.fetch_add(1, Ordering::SeqCst) + 1;
                    HttpResponse::json(200, &json!({"jsonrpc": "2.0", "id": "1", "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "example", "version": "1"},
                    }}))
                    .with_header("mcp-session-id", format!("sess-{id}"))
                }
                "tools/call" if left.load(Ordering::SeqCst) > 0 => {
                    left.fetch_sub(1, Ordering::SeqCst);
                    refusal.clone()
                }
                "tools/call" => HttpResponse::json(
                    200,
                    &json!({"jsonrpc": "2.0", "id": "2", "result": {"content": [{"type": "text", "text": "ok"}]}}),
                ),
                _ => HttpResponse::new(202, ""),
            })
        });
        let guard = HttpTransportOverride::install(fake.clone());
        (Self { fake, refusals }, guard)
    }

    fn refuse_next(&self, times: usize) {
        self.refusals.store(times, Ordering::SeqCst);
    }
}

#[test]
#[serial(mcp_global)]
fn test_an_expired_session_is_replaced_and_the_call_retried() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", None)]);
    let (srv, _guard) = ExpiringServer::install(HttpResponse::new(404, ""));
    let s1 = server("s1");

    call(&s1, Some("tok"));
    srv.refuse_next(1);
    call(&s1, Some("tok"));
    call(&s1, Some("tok"));

    assert_eq!(count(&srv.fake, "initialize"), 2);
    assert_eq!(
        call_session_ids(&srv.fake),
        vec![Some("sess-1".into()), Some("sess-1".into()), Some("sess-2".into()), Some("sess-2".into())],
        "the refused call is retried on the new session, and later calls keep it"
    );
}

#[test]
#[serial(mcp_global)]
fn test_a_400_naming_the_session_is_a_lost_session_too() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", None)]);
    let (srv, _guard) = ExpiringServer::install(HttpResponse::new(400, "Bad Request: Session not found"));
    let s1 = server("s1");

    call(&s1, Some("tok"));
    srv.refuse_next(1);
    call(&s1, Some("tok"));

    assert_eq!(count(&srv.fake, "initialize"), 2);
}

#[test]
#[serial(mcp_global)]
fn test_a_second_refusal_fails_without_looping() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", None)]);
    let (srv, _guard) = ExpiringServer::install(HttpResponse::new(404, ""));
    let s1 = server("s1");

    call(&s1, Some("tok"));
    srv.refuse_next(usize::MAX);
    let result = tool_caller().call_tool(&s1, "search", None, Some("tok"), None);

    assert!(!result.success);
    assert_eq!(count(&srv.fake, "tools/call"), 3, "one call, then exactly one retry");
}

#[test]
#[serial(mcp_global)]
fn test_a_404_without_a_session_is_not_retried() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", None)]);
    let fake = FakeHttp::new(|_, _, body| {
        Ok(match rpc_method(body).as_str() {
            "initialize" => HttpResponse::json(200, &json!({"jsonrpc": "2.0", "id": "1", "result": {}})),
            "tools/call" => HttpResponse::new(404, ""),
            _ => HttpResponse::new(202, ""),
        })
    });
    let _guard = HttpTransportOverride::install(fake.clone());

    let result = tool_caller().call_tool(&server("s1"), "search", None, Some("tok"), None);

    assert!(!result.success);
    assert_eq!(count(&fake, "tools/call"), 1);
}

// ------------------------------------------------------------- invalidation

#[test]
#[serial(mcp_global)]
fn test_invalidating_a_server_forces_a_new_handshake() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", None)]);
    let fake = mcp_server(Some("sess-1"));
    let _guard = HttpTransportOverride::install(fake.clone());

    call(&server("s1"), Some("tok"));
    call(&server("s2"), Some("tok"));
    invalidate_mcp_sessions("s1");
    call(&server("s1"), Some("tok"));
    call(&server("s2"), Some("tok"));

    assert_eq!(count(&fake, "initialize"), 3, "only s1 handshakes again");
}

#[test]
#[serial(mcp_global)]
fn test_a_refresh_drops_the_old_session() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", None)]);
    let inner = mcp_server(Some("sess-1"));
    let fake = FakeHttp::new(move |method, url, body| {
        if url == TOKEN_ENDPOINT {
            return Ok(ok_json(json!({"access_token": "fresh", "token_type": "Bearer", "expires_in": 3600})));
        }
        match (method, body) {
            ("POST", Some(b)) => inner.post(url, b, &[]),
            _ => inner.get(url),
        }
    });
    let _guard = HttpTransportOverride::install(fake.clone());
    reset_refresh_state();
    let mut srv = oauth_server("s1", "example", URL, Some(TOKEN_ENDPOINT), Some("tok"), Some("rt"));

    call(&srv, Some("tok"));
    assert_eq!(tool_caller().refresh_access_token(&mut srv, Some("tok")).as_deref(), Some("fresh"));
    call(&srv, Some("tok"));

    assert_eq!(count(&fake, "initialize"), 2);
    reset_refresh_state();
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_deleting_a_server_drops_its_session() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", None)]);
    let env = ServiceEnv::new("p1");
    env.add_server(server("s1"));
    let fake = mcp_server(Some("sess-1"));
    let _guard = HttpTransportOverride::install(fake.clone());

    call(&server("s1"), Some("tok"));
    env.model_service.delete_mcp_server("s1", Some("p1")).expect("deleted");
    call(&server("s1"), Some("tok"));

    assert_eq!(count(&fake, "initialize"), 2);
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_reconnecting_a_server_drops_its_session() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", None)]);
    let env = ServiceEnv::new("p1");
    env.add_server(server("s1"));
    env.record_provisioning();
    let fake = mcp_server(Some("sess-1"));
    let _guard = HttpTransportOverride::install(fake.clone());

    call(&server("s1"), Some("tok"));
    env.model_service.retry_mcp_server("s1", Some("p1")).expect("retried");
    call(&server("s1"), Some("tok"));

    assert_eq!(count(&fake, "initialize"), 2);
}

// ------------------------------------------------------------------ logging

#[test]
#[serial(mcp_global)]
fn test_each_call_logs_its_time_and_how_it_got_a_session() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", None)]);
    let (srv, _guard) = ExpiringServer::install(HttpResponse::new(404, ""));
    let s1 = server("s1");

    let (_, logged) = capture_logs(tracing::Level::INFO, || {
        call(&s1, Some("tok"));
        call(&s1, Some("tok"));
        srv.refuse_next(1);
        call(&s1, Some("tok"));
        invalidate_mcp_sessions("s1");
    });

    let lines: Vec<&str> = logged.lines().filter(|l| l.contains("[mcp-remote] search on")).collect();
    assert_eq!(lines.len(), 3, "{logged}");
    assert!(lines[0].contains("new session, handshake"), "{}", lines[0]);
    assert!(lines[1].contains("(reused session)"), "{}", lines[1]);
    assert!(lines[2].contains("after the server lost the previous session"), "{}", lines[2]);
    assert!(logged.contains("[mcp-session] Dropped 1 session(s) for server s1"), "{logged}");
    assert!(!logged.contains("sess-"), "session ids stay out of the log");
}

#[test]
#[serial(mcp_global)]
fn test_expiry_is_logged() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", Some("5m"))]);
    let fake = mcp_server(Some("sess-1"));
    let _guard = HttpTransportOverride::install(fake.clone());

    call(&server("s1"), Some("tok"));
    backdate_mcp_sessions(Duration::from_secs(301));
    let (_, logged) = capture_logs(tracing::Level::INFO, || call(&server("s1"), Some("tok")));

    assert!(logged.contains("Dropped 1 session(s) idle past the 300s keep-alive"), "{logged}");
}

// ------------------------------------------------- servers that read headers

/// A request as [`HeaderServer`] saw it.
#[derive(Clone, Debug)]
struct Seen {
    method: String,
    token: Option<String>,
    session: Option<String>,
}

type HeaderHandler = dyn Fn(&str, &str, &Seen) -> HttpResponse + Send + Sync;

/// A server whose answers can depend on the bearer token and session id a
/// request carried, which `FakeHttp`'s handler is not shown.
struct HeaderServer {
    handler: Box<HeaderHandler>,
    seen: parking_lot::Mutex<Vec<Seen>>,
}

impl HeaderServer {
    fn install(
        handler: impl Fn(&str, &str, &Seen) -> HttpResponse + Send + Sync + 'static,
    ) -> (Arc<Self>, HttpTransportOverride) {
        let server = Arc::new(Self { handler: Box::new(handler), seen: parking_lot::Mutex::new(Vec::new()) });
        let guard = HttpTransportOverride::install(server.clone());
        (server, guard)
    }

    fn count(&self, method: &str) -> usize {
        self.seen.lock().iter().filter(|s| s.method == method).count()
    }
}

impl HttpTransport for HeaderServer {
    fn get(&self, _url: &str) -> Result<HttpResponse, String> {
        Ok(HttpResponse::new(405, ""))
    }

    fn post(&self, url: &str, body: &HttpBody, headers: &[(String, String)]) -> Result<HttpResponse, String> {
        let header = |name: &str| headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.clone());
        let seen = Seen {
            method: rpc_method(Some(body)),
            token: header("Authorization").and_then(|v| v.strip_prefix("Bearer ").map(str::to_string)),
            session: header("Mcp-Session-Id"),
        };
        self.seen.lock().push(seen.clone());
        Ok((self.handler)(url, &seen.method, &seen))
    }
}

fn initialized(session_id: &str) -> HttpResponse {
    HttpResponse::json(200, &json!({"jsonrpc": "2.0", "id": "1", "result": {
        "protocolVersion": "2024-11-05",
        "serverInfo": {"name": "example", "version": "1"},
    }}))
    .with_header("mcp-session-id", session_id)
}

fn tool_ok() -> HttpResponse {
    HttpResponse::json(
        200,
        &json!({"jsonrpc": "2.0", "id": "2", "result": {"content": [{"type": "text", "text": "ok"}]}}),
    )
}

// ------------------------------------------------- a failed handshake

#[test]
#[serial(mcp_global)]
fn test_a_failed_handshake_is_tried_again_next_call() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", None)]);
    let failures = Arc::new(AtomicUsize::new(1));
    let left = failures.clone();
    let (srv, _guard) = HeaderServer::install(move |_, method, _| match method {
        "initialize" if left.load(Ordering::SeqCst) > 0 => {
            left.fetch_sub(1, Ordering::SeqCst);
            HttpResponse::new(503, "unavailable")
        }
        "initialize" => initialized("sess-1"),
        "tools/call" => tool_ok(),
        _ => HttpResponse::new(202, ""),
    });
    let s1 = server("s1");

    let first = tool_caller().call_tool(&s1, "search", None, Some("tok"), None);
    assert!(!first.success);
    call(&s1, Some("tok"));
    call(&s1, Some("tok"));

    assert_eq!(srv.count("initialize"), 2, "one failed, one that then serves both calls");
    assert_eq!(srv.count("tools/call"), 2, "nothing is called on a session that never opened");
}

#[test]
#[serial(mcp_global)]
fn test_a_401_on_the_handshake_still_reads_as_expired_auth() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", None)]);
    let (_srv, _guard) = HeaderServer::install(|_, _, _| HttpResponse::new(401, ""));

    let result = tool_caller().call_tool(&server("s1"), "search", None, Some("tok"), None);

    assert!(!result.success);
    assert!(result.auth_expired);
}

// ------------------------------------------------- refresh through call_with_refresh

#[test]
#[serial(mcp_global)]
fn test_a_refreshed_call_runs_on_a_session_opened_with_the_new_token() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", None)]);
    reset_refresh_state();
    let (srv, _guard) = HeaderServer::install(|url, method, seen| {
        if url == TOKEN_ENDPOINT {
            return ok_json(json!({"access_token": "fresh", "token_type": "Bearer", "expires_in": 3600}));
        }
        match (method, seen.token.as_deref()) {
            ("initialize", Some("fresh")) => initialized("sess-fresh"),
            ("initialize", _) => initialized("sess-stale"),
            ("tools/call", Some("fresh")) => tool_ok(),
            ("tools/call", _) => HttpResponse::new(401, ""),
            _ => HttpResponse::new(202, ""),
        }
    });
    let mut s1 = oauth_server("s1", "example", URL, Some(TOKEN_ENDPOINT), Some("tok"), Some("rt"));

    let (result, refreshed) = tool_caller().call_with_refresh(&mut s1, "search", None, None);
    assert!(result.success, "{:?}", result.error);
    assert!(refreshed);
    call(&s1, Some("fresh"));

    let calls: Vec<Seen> = srv.seen.lock().iter().filter(|s| s.method == "tools/call").cloned().collect();
    assert_eq!(calls.len(), 3);
    assert_eq!(calls[1].token.as_deref(), Some("fresh"));
    assert_eq!(calls[1].session.as_deref(), Some("sess-fresh"), "not the session opened under the old token");
    assert_eq!(calls[2].session.as_deref(), Some("sess-fresh"), "and later calls reuse it");
    assert_eq!(srv.count("initialize"), 2);
    reset_refresh_state();
}

// ------------------------------------------------- two calls lose the session at once

#[test]
#[serial(mcp_global)]
fn test_callers_that_lose_the_session_together_open_one_new_one() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", None)]);
    let sessions = Arc::new(AtomicUsize::new(0));
    let stale = Arc::new(AtomicBool::new(false));
    let stale_arrivals = Arc::new(AtomicUsize::new(0));
    let (n, refusing, arrivals) = (sessions.clone(), stale.clone(), stale_arrivals.clone());
    let (srv, _guard) = HeaderServer::install(move |_, method, seen| match method {
        "initialize" => {
            let id = n.fetch_add(1, Ordering::SeqCst) + 1;
            // A slow handshake, so the other caller acts while it runs.
            std::thread::sleep(Duration::from_millis(50));
            initialized(&format!("sess-{id}"))
        }
        "tools/call" if refusing.load(Ordering::SeqCst) && seen.session.as_deref() == Some("sess-1") => {
            // Hold each stale request until both have arrived, then answer
            // the second while the first caller's new handshake is running,
            // so its caller drops the session it lost mid-replacement.
            let arrived = arrivals.fetch_add(1, Ordering::SeqCst);
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while arrivals.load(Ordering::SeqCst) < 2 && std::time::Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(1));
            }
            if arrived == 1 {
                std::thread::sleep(Duration::from_millis(20));
            }
            HttpResponse::new(404, "")
        }
        "tools/call" => tool_ok(),
        _ => HttpResponse::new(202, ""),
    });
    call(&server("s1"), Some("tok"));
    stale.store(true, Ordering::SeqCst);

    let threads: Vec<_> = (0..2).map(|_| std::thread::spawn(|| call(&server("s1"), Some("tok")))).collect();
    for t in threads {
        t.join().expect("a caller panicked");
    }

    assert_eq!(stale_arrivals.load(Ordering::SeqCst), 2);
    assert_eq!(srv.count("initialize"), 2, "the second caller adopts the first one's new session");
}

// ------------------------------------------------- the shared client

#[test]
#[serial(mcp_global)]
fn test_clients_share_one_http_client() {
    let previous = set_http_transport_override(None);

    let (a, b) = (shared_http_transport(), shared_http_transport());

    set_http_transport_override(previous);
    assert!(Arc::ptr_eq(&a, &b));
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_deleting_a_project_drops_its_servers_sessions() {
    let _env = EnvGuard::set(&[("SLP_MCP_KEEP_ALIVE", None)]);
    let env = ServiceEnv::new("p1");
    env.add_server(server("s1"));
    let fake = mcp_server(Some("sess-1"));
    let _guard = HttpTransportOverride::install(fake.clone());

    call(&server("s1"), Some("tok"));
    call(&server("other"), Some("tok"));
    env.model_service.delete_project("p1").expect("deleted");
    call(&server("s1"), Some("tok"));
    call(&server("other"), Some("tok"));

    assert_eq!(count(&fake, "initialize"), 3, "only the deleted project's server handshakes again");
}
