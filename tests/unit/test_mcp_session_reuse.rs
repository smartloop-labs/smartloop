//! Remote MCP sessions outlive a single tool call.
//!
//! Every call used to `initialize` again, and servers that issue no
//! `Mcp-Session-Id` were re-initialized before every request.  These drive
//! `MCPToolCaller::call_tool` against a scripted server and count handshakes.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use serial_test::serial;

use smartloop::mcp::client::{HttpBody, HttpResponse, HttpTransportOverride};
use smartloop::mcp::session_pool::{backdate_mcp_sessions, reset_mcp_sessions};
use smartloop::workspace::MCPServer;

use crate::shared::mcp::{oauth_server, tool_caller, EnvGuard, FakeHttp};

const URL: &str = "https://mcp.example.com/mcp";

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
        use smartloop::mcp::client::HttpTransport;
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
