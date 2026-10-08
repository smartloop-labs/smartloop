//! Registering remote MCP servers that connect with a pasted credential.
//!
//! Covers catalog `credential_fields`, the auth type register reports, and
//! how a supplied token that fails to connect is handled.
//! The handshake, the `/info` probe and the model service's MCP registry are
//! stood in for through the router's hooks; the registry records what
//! register asked it to store.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Value};
use serial_test::serial;

use smartloop::api::model::mcp::{MCPRegisterRequest, MCPRegisterResponse};
use smartloop::api::routers::mcp_catalog::{to_entries, HTTPException, MCPCatalogEntry, RemoteConfig};
use smartloop::api::routers::mcp_servers::{
    register_remote_mcp_server, set_hooks, AddLocalMcpServer, AddRemoteMcpServer, McpRouteHooks,
    McpServerRegistry,
};
use smartloop::api::services::model_service::ModelService;
use smartloop::mcp::client::{HttpResponse, HttpTransportOverride, MCPConnectResult};

use crate::shared::mcp::{block_on, reset_router_state, tool, FakeHttp};

const PAT: &str = "ghp_pasted_by_the_user";
const GITHUB_URL: &str = "https://api.githubcopilot.com/mcp/";

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
            "id": "srv-1", "name": if params.name.is_empty() { "GitHub" } else { &params.name },
            "enabled": true, "tools": [], "server_type": "remote", "args": [],
            "server_url": params.server_url,
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

fn connected() -> MCPConnectResult {
    MCPConnectResult {
        success: true,
        tools: vec![tool("search_issues", Some(json!({})))],
        ..Default::default()
    }
}

/// Register *request* with the handshake answering *connect* and no `/info`.
fn register(
    request: MCPRegisterRequest,
    curated: Option<&MCPCatalogEntry>,
    connect: MCPConnectResult,
) -> (Result<MCPRegisterResponse, HTTPException>, Arc<Registry>) {
    reset_router_state();
    let registry = Arc::new(Registry::default());
    set_hooks(McpRouteHooks {
        probe_info: Some(Arc::new(|_| Err(anyhow::anyhow!("no /info endpoint")))),
        discover_remote_tools: Some(Arc::new(move |_, _, _| Ok(connect.clone()))),
        mcp_registry: Some(registry.clone()),
        ..Default::default()
    });
    let response = block_on(register_remote_mcp_server(&request, ModelService::new_bare(), "proj-1", curated));
    reset_router_state();
    (response, registry)
}

fn remote(url: &str, token: Option<&str>) -> MCPRegisterRequest {
    MCPRegisterRequest {
        server_url: Some(url.into()),
        token: token.map(str::to_string),
        ..Default::default()
    }
}

// ------------------------------------------------ register reports what it stored

#[test]
#[serial]
#[serial(mcp_global)]
fn test_a_token_connect_reports_token() {
    let (resp, registry) = register(remote(GITHUB_URL, Some(PAT)), None, connected());

    assert_eq!(resp.expect("registered").auth_type, "token");
    let added = registry.added.lock();
    assert_eq!(added[0].auth_type, "token");
    assert_eq!(added[0].token.as_deref(), Some(PAT));
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_an_open_connect_still_reports_none() {
    let (resp, _) = register(remote("https://mcp.example.com/mcp", None), None, connected());

    assert_eq!(resp.expect("registered").auth_type, "none");
}

// ------------------------------------------ a curated entry can ask for a credential

fn github_entry() -> MCPCatalogEntry {
    MCPCatalogEntry::from_value(json!({
        "slug": "github",
        "display_name": "GitHub",
        "hosted_by": "external",
        "remote_config": {
            "server_url": GITHUB_URL,
            "auth_type": "token",
            "credential_fields": [{
                "key": "token",
                "label": "a personal access token",
                "help": "Needs the repo, read:org and read:user scopes.",
                "help_url": "https://github.com/settings/personal-access-tokens",
            }],
        },
    }))
    .expect("valid entry")
}

fn from_catalog(token: Option<&str>) -> MCPRegisterRequest {
    MCPRegisterRequest {
        catalog_slug: Some("github".into()),
        token: token.map(str::to_string),
        ..Default::default()
    }
}

#[test]
fn test_the_field_survives_the_platform_payload() {
    let raw = serde_json::to_value(github_entry()).unwrap();
    let entry = to_entries(vec![raw.as_object().unwrap().clone()]).unwrap().remove(0);
    let field = &entry.remote_config.as_ref().unwrap().credential_fields[0];

    assert_eq!((field.key.as_str(), field.secret, field.required), ("token", true, true));
    assert_eq!(entry.server_type.as_deref(), Some("remote"));
}

#[test]
fn test_an_entry_without_fields_serializes_an_empty_list() {
    // Clients render the form from this list, so it is always present.
    let rc = RemoteConfig { server_url: GITHUB_URL.into(), ..Default::default() };
    assert_eq!(serde_json::to_value(rc).unwrap()["credential_fields"], json!([]));
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_registering_without_it_says_what_is_missing() {
    let entry = github_entry();
    let (resp, registry) = register(from_catalog(None), Some(&entry), connected());

    let err = resp.expect_err("refused");
    assert_eq!(err.status_code, 400);
    assert!(err.detail.contains("personal access token"), "{}", err.detail);
    assert!(registry.added.lock().is_empty());
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_registering_with_it_goes_through() {
    let entry = github_entry();
    let (resp, registry) = register(from_catalog(Some(PAT)), Some(&entry), connected());

    assert_eq!(resp.expect("registered").auth_type, "token");
    assert_eq!(registry.added.lock()[0].server_url, GITHUB_URL);
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_a_token_in_the_url_counts_as_supplied() {
    let entry = MCPCatalogEntry {
        remote_config: Some(RemoteConfig {
            server_url: format!("{GITHUB_URL}?token={PAT}"),
            ..github_entry().remote_config.unwrap()
        }),
        ..github_entry()
    };
    let (resp, registry) = register(from_catalog(None), Some(&entry), connected());

    assert_eq!(resp.expect("registered").auth_type, "token");
    assert_eq!(registry.added.lock()[0].token.as_deref(), Some(PAT));
}

// ------------------------------------------------------- a refused token stops there

fn refused(auth_expired: bool) -> MCPConnectResult {
    MCPConnectResult {
        success: false,
        error: Some("Authorization failed".into()),
        auth_expired,
        ..Default::default()
    }
}

/// Register against a handshake that fails with *connect*.  Returns the error
/// and whether OAuth discovery went on to fetch any metadata.
fn register_failing(request: MCPRegisterRequest, connect: MCPConnectResult) -> (HTTPException, bool) {
    let http = FakeHttp::new(|_, _, _| Ok(HttpResponse::new(404, "")));
    let _transport = HttpTransportOverride::install(http.clone());
    let (resp, registry) = register(request, None, connect);
    assert!(registry.added.lock().is_empty());
    let oauth_ran = http.calls.lock().iter().any(|c| c.url.contains("/.well-known/"));
    (resp.expect_err("refused"), oauth_ran)
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_a_rejected_token_says_so() {
    let (err, oauth_ran) = register_failing(remote(GITHUB_URL, Some(PAT)), refused(true));

    assert_eq!(err.status_code, 401);
    assert!(err.detail.contains("access token was rejected"), "{}", err.detail);
    assert!(!oauth_ran);
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_any_other_failure_with_a_token_skips_oauth_too() {
    let (err, oauth_ran) = register_failing(remote(GITHUB_URL, Some(PAT)), refused(false));

    assert_eq!(err.status_code, 502);
    assert!(err.detail.contains("Authorization failed"), "{}", err.detail);
    assert!(!oauth_ran);
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_without_a_token_a_401_still_starts_oauth() {
    // DCR servers (Atlassian, Trello) must keep reaching registration.
    let (_, oauth_ran) = register_failing(remote("https://mcp.atlassian.com/v1/mcp", None), refused(true));

    assert!(oauth_ran);
}
