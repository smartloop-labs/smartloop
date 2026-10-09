//! Connecting a catalog server through an RFC 8628 device sign-in.
//!
//! The provider's device and token endpoints are answered by a fake
//! transport; the handshake, the `/info` probe, the catalog and the model
//! service's MCP registry are stood in for through the router's hooks.

use std::collections::VecDeque;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Value};
use serial_test::serial;

use smartloop::api::model::mcp::{
    MCPDevicePollRequest, MCPDevicePollResponse, MCPDeviceStartRequest, MCPDeviceStartResponse,
};
use smartloop::api::routers::mcp_catalog::{HTTPException, MCPCatalogEntry};
use smartloop::api::routers::mcp_servers::{
    poll_device_sign_in, set_hooks, start_device_sign_in, AddLocalMcpServer, AddRemoteMcpServer,
    McpRouteHooks, McpServerRegistry, PENDING_DEVICE_SESSIONS,
};
use smartloop::api::services::model_service::ModelService;
use smartloop::mcp::client::{HttpResponse, HttpTransportOverride, MCPConnectResult};

use crate::shared::mcp::{block_on, reset_router_state, tool, FakeHttp};

const GITHUB_URL: &str = "https://api.githubcopilot.com/mcp/";
const DEVICE_URL: &str = "https://github.com/login/device/code";
const TOKEN_URL: &str = "https://github.com/login/oauth/access_token";
const GRANTED: &str = "gho_granted_by_device_flow";

/// Records each `add_remote_mcp_server` call and answers with a stored record.
#[derive(Default)]
struct Registry {
    added: Mutex<Vec<AddRemoteMcpServer>>,
}

impl McpServerRegistry for Registry {
    fn add_local_mcp_server(&self, _: AddLocalMcpServer) -> anyhow::Result<Value> {
        unimplemented!("remote registration only")
    }
    fn add_remote_mcp_server(&self, params: AddRemoteMcpServer) -> anyhow::Result<Value> {
        let record = json!({
            "id": "srv-1", "name": params.name, "enabled": true, "tools": [],
            "server_type": "remote", "args": [], "server_url": params.server_url,
        });
        self.added.lock().push(params);
        Ok(record)
    }
    fn list_mcp_servers_with_status(&self, _: &str) -> anyhow::Result<Vec<Value>> {
        unimplemented!()
    }
    fn count_mcp_tools(&self, _: &str) -> anyhow::Result<Value> {
        unimplemented!()
    }
    fn mcp_server_status(&self, _: &str, _: &str) -> anyhow::Result<Value> {
        unimplemented!()
    }
    fn retry_mcp_server(&self, _: &str, _: &str) -> anyhow::Result<Value> {
        unimplemented!()
    }
    fn delete_mcp_server(&self, _: &str, _: &str) -> anyhow::Result<Value> {
        unimplemented!()
    }
}

fn github_entry(device_flow: bool) -> MCPCatalogEntry {
    let mut remote = json!({
        "server_url": GITHUB_URL,
        "auth_type": "token",
        "credential_fields": [{"key": "token", "label": "a personal access token"}],
    });
    if device_flow {
        remote["device_flow"] = json!({
            "client_id": "Iv1.smartloop",
            "device_authorization_endpoint": DEVICE_URL,
            "token_endpoint": TOKEN_URL,
            "scopes": ["repo", "read:org", "read:user"],
        });
    }
    MCPCatalogEntry::from_value(json!({
        "slug": "github", "display_name": "GitHub", "hosted_by": "external", "remote_config": remote,
    }))
    .expect("valid entry")
}

fn connected() -> MCPConnectResult {
    MCPConnectResult { success: true, tools: vec![tool("search_issues", Some(json!({})))], ..Default::default() }
}

fn device_code_response() -> HttpResponse {
    HttpResponse::json(
        200,
        &json!({
            "device_code": "dev-123", "user_code": "WDJB-MJHT",
            "verification_uri": "https://github.com/login/device", "expires_in": 900, "interval": 5,
        }),
    )
}

/// A provider that issues a device code, then answers token polls from
/// *polls* in order (GitHub sends its pending errors with a 200).
struct Provider {
    http: Arc<FakeHttp>,
    _transport: HttpTransportOverride,
    registry: Arc<Registry>,
    tokens_seen: Arc<Mutex<Vec<Option<String>>>>,
}

impl Provider {
    fn new(entry: MCPCatalogEntry, polls: Vec<Value>, connect: MCPConnectResult) -> Self {
        reset_router_state();
        let polls = Arc::new(Mutex::new(polls.into_iter().collect::<VecDeque<_>>()));
        let http = FakeHttp::new(move |_, url, _| match url {
            DEVICE_URL => Ok(device_code_response()),
            TOKEN_URL => Ok(HttpResponse::json(200, &polls.lock().pop_front().expect("an answer per poll"))),
            _ => Ok(HttpResponse::new(404, "")),
        });
        let registry = Arc::new(Registry::default());
        let tokens_seen = Arc::new(Mutex::new(Vec::new()));
        let seen = tokens_seen.clone();
        set_hooks(McpRouteHooks {
            probe_info: Some(Arc::new(|_| Err(anyhow::anyhow!("no /info endpoint")))),
            discover_remote_tools: Some(Arc::new(move |_, token, _| {
                seen.lock().push(token.map(str::to_string));
                Ok(connect.clone())
            })),
            resolve_catalog_entry: Some(Arc::new(move |slug| (slug == "github").then(|| entry.clone()))),
            mcp_registry: Some(registry.clone()),
            ..Default::default()
        });
        Self { _transport: HttpTransportOverride::install(http.clone()), http, registry, tokens_seen }
    }

    fn start(&self) -> Result<MCPDeviceStartResponse, HTTPException> {
        block_on(start_device_sign_in("proj-1", MCPDeviceStartRequest { catalog_slug: "github".into() }))
    }

    /// Poll as soon as the session allows it.
    fn poll(&self, session_id: &str) -> Result<MCPDevicePollResponse, HTTPException> {
        if let Some(s) = PENDING_DEVICE_SESSIONS.lock().get_mut(session_id) {
            s.next_poll_at = 0.0;
        }
        self.poll_now("proj-1", session_id)
    }

    fn poll_now(&self, project_id: &str, session_id: &str) -> Result<MCPDevicePollResponse, HTTPException> {
        let request = MCPDevicePollRequest { session_id: session_id.into() };
        block_on(poll_device_sign_in(project_id, request, ModelService::new_bare()))
    }

    fn token_polls(&self) -> usize {
        self.http.posts().iter().filter(|(url, _)| url == TOKEN_URL).count()
    }
}

impl Drop for Provider {
    fn drop(&mut self) {
        reset_router_state();
    }
}

fn pending() -> Value {
    json!({"error": "authorization_pending"})
}

fn granted() -> Value {
    json!({"access_token": GRANTED, "token_type": "bearer", "scope": "repo,read:org,read:user"})
}

// ------------------------------------------------------------------ start

#[test]
#[serial]
#[serial(mcp_global)]
fn test_start_hands_out_the_user_code_and_keeps_the_device_code() {
    let provider = Provider::new(github_entry(true), vec![], connected());
    let started = provider.start().expect("started");

    assert_eq!((started.user_code.as_str(), started.interval), ("WDJB-MJHT", 5));
    assert_eq!(PENDING_DEVICE_SESSIONS.lock()[&started.session_id].device_code, "dev-123");
    assert!(!serde_json::to_string(&started).unwrap().contains("dev-123"));

    let calls = provider.http.calls.lock();
    let form = calls[0].form();
    assert_eq!(form["client_id"], json!("Iv1.smartloop"));
    assert_eq!(form["scope"], json!("repo read:org read:user"));
    assert!(calls[0].headers.iter().any(|(k, v)| k == "Accept" && v == "application/json"));
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_an_entry_without_device_flow_is_refused() {
    let provider = Provider::new(github_entry(false), vec![], connected());

    assert_eq!(provider.start().expect_err("refused").status_code, 400);
    assert_eq!(provider.http.call_count(), 0);
}

// ------------------------------------------------------------------ poll

#[test]
#[serial]
#[serial(mcp_global)]
fn test_a_grant_registers_the_server_like_a_pasted_token() {
    let provider = Provider::new(github_entry(true), vec![pending(), granted()], connected());
    let id = provider.start().unwrap().session_id;

    assert_eq!(provider.poll(&id).unwrap().status, "pending");
    let done = provider.poll(&id).expect("connected");

    assert_eq!(done.status, "connected");
    assert!(done.server.is_some());
    let added = provider.registry.added.lock();
    assert_eq!((added[0].auth_type.as_str(), added[0].token.as_deref()), ("token", Some(GRANTED)));
    assert_eq!(added[0].server_url, GITHUB_URL);
    assert_eq!(provider.tokens_seen.lock().last().cloned().flatten().as_deref(), Some(GRANTED));
    assert!(PENDING_DEVICE_SESSIONS.lock().is_empty());
    assert_eq!(provider.token_polls(), 2);
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_polling_early_does_not_reach_the_provider() {
    let provider = Provider::new(github_entry(true), vec![pending()], connected());
    let id = provider.start().unwrap().session_id;

    provider.poll(&id).unwrap();
    let early = provider.poll_now("proj-1", &id).unwrap();

    assert_eq!((early.status.as_str(), early.interval), ("pending", Some(5)));
    assert_eq!(provider.token_polls(), 1);
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_slow_down_lengthens_the_interval() {
    let provider = Provider::new(github_entry(true), vec![json!({"error": "slow_down"})], connected());
    let id = provider.start().unwrap().session_id;

    assert_eq!(provider.poll(&id).unwrap().interval, Some(10));
    assert_eq!(PENDING_DEVICE_SESSIONS.lock()[&id].interval, 10);
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_a_cancelled_sign_in_ends_the_session() {
    let provider = Provider::new(github_entry(true), vec![json!({"error": "access_denied"})], connected());
    let id = provider.start().unwrap().session_id;

    assert_eq!(provider.poll(&id).unwrap().status, "denied");
    assert_eq!(provider.poll(&id).expect_err("gone").status_code, 404);
    assert!(provider.registry.added.lock().is_empty());
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_an_expired_code_ends_the_session() {
    let provider = Provider::new(github_entry(true), vec![json!({"error": "expired_token"})], connected());
    let id = provider.start().unwrap().session_id;

    assert_eq!(provider.poll(&id).unwrap().status, "expired");
    assert!(PENDING_DEVICE_SESSIONS.lock().is_empty());
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_a_granted_token_the_server_refuses_reports_401() {
    let refused = MCPConnectResult { success: false, auth_expired: true, ..Default::default() };
    let provider = Provider::new(github_entry(true), vec![granted()], refused);
    let id = provider.start().unwrap().session_id;

    assert_eq!(provider.poll(&id).expect_err("refused").status_code, 401);
    assert!(provider.registry.added.lock().is_empty());
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_a_session_is_only_polled_from_its_project() {
    let provider = Provider::new(github_entry(true), vec![], connected());
    let id = provider.start().unwrap().session_id;

    assert_eq!(provider.poll_now("proj-2", &id).expect_err("other project").status_code, 404);
    assert_eq!(provider.poll_now("proj-1", "no-such-session").expect_err("unknown").status_code, 404);
    assert_eq!(provider.token_polls(), 0);
}
