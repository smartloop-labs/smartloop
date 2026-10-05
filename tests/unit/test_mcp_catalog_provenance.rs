//! Port of `tests/unit/test_mcp_catalog_provenance.py`.
//!
//! Curated provenance has to survive the whole install, not just the request.
//!
//! `catalog_slug` is the record of which catalog entry a connection came from,
//! and two things downstream depend on it:
//!
//! * `ModelService::enrich_mcp_server` skips the LLM for a curated server.
//!   Without the slug, a 93-tool server like monday.com goes to the small model
//!   and comes back as unparseable JSON — two inference calls spent to produce
//!   nothing, and the curated description overwritten if it had succeeded.
//! * [`smartloop::mcp::definitions::definition_for`] checks the slug before it
//!   falls back to matching substrings against the server's name and URL,
//!   which is how a deleted connection's dropped-in file is found and removed.
//!
//! The install is not one call: register resolves the catalog, the OAuth grant
//! leaves and comes back, and only then is the record written and saved.
//! These tests cover the hops in between.
//!
//! Python patches `resolve_catalog_entry` inside `register_mcp_server`; here
//! the resolved entry (or the miss) is handed to `register_remote_mcp_server`,
//! the step that receives it.  What Python reads off the mocked
//! `add_remote_mcp_server` call is read here off the connection the real
//! method stored.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use serial_test::serial;

use smartloop::api::model::mcp::{MCPOAuthCallbackRequest, MCPRegisterRequest};
use smartloop::api::routers::mcp_catalog::{MCPCatalogEntry, RemoteConfig};
use smartloop::api::routers::mcp_servers::{
    complete_mcp_oauth, set_hooks, McpRouteHooks, OAuthCallbackQuery, PENDING_OAUTH_SESSIONS,
};
use smartloop::mcp::client::{MCPConnectResult, MCPServerInfo};
use smartloop::workspace::workspace::{MCPServer, MCPServerType, Project, RemoteMCPServerConfig, Workspace};

use crate::shared::mcp::{
    block_on, register_via_oauth, reset_router_state, run_callback, token_response, tool, ServiceEnv,
};

const SLUG: &str = "monday";
const CATALOG_URL: &str = "https://mcp.monday.com/mcp";
const CATALOG_NAME: &str = "monday.com";
const CATALOG_DESC: &str = "Query and update monday.com boards";
const CATALOG_TAGS: &[&str] = &["project-management", "boards", "workflows"];

/// What the vendor's own handshake says it is called.  Deliberately different
/// from the catalog's display name and empty where the catalog has copy: that
/// difference is what tells a curated write apart from a generic one.
fn handshake() -> MCPServerInfo {
    MCPServerInfo {
        name: Some("monday.com MCP Server".into()),
        title: None,
        version: Some("1.0".into()),
        description: None,
        icons: None,
    }
}

fn tags() -> Vec<String> {
    CATALOG_TAGS.iter().map(|t| t.to_string()).collect()
}

fn catalog_entry() -> MCPCatalogEntry {
    MCPCatalogEntry {
        slug: SLUG.into(),
        display_name: CATALOG_NAME.into(),
        description: Some(CATALOG_DESC.into()),
        tags: tags(),
        server_type: Some("remote".into()),
        hosted_by: "external".into(),
        remote_config: Some(RemoteConfig {
            server_url: CATALOG_URL.into(),
            auth_type: Some("oauth".into()),
            ..Default::default()
        }),
        local_config: None,
        ..Default::default()
    }
}

/// Run the register endpoint's OAuth branch and return the stored session
/// (OAuth discovery lands, so registration takes the OAuth branch).
fn register(catalog_hit: bool, slug: &str) -> Map<String, Value> {
    let entry = catalog_entry();
    let out = register_via_oauth(
        Some(json!({"resource": CATALOG_URL, "authorization_servers": ["https://auth.monday.com/mcp"]})),
        json!({
            "issuer": "https://auth.monday.com/mcp",
            "authorization_endpoint": "https://auth.monday.com/oauth2/authorize",
            "token_endpoint": "https://auth.monday.com/oauth_ms/oauth/token",
            "registration_endpoint": "https://auth.monday.com/oauth_ms/oauth/register",
        }),
        MCPRegisterRequest {
            server_url: Some(CATALOG_URL.into()),
            server_type: "remote".into(),
            catalog_slug: Some(slug.into()),
            ..Default::default()
        },
        if catalog_hit { Some(&entry) } else { None },
    );
    reset_router_state();
    assert!(out.response.is_ok(), "register failed: {:?}", out.response.as_ref().err());
    out.session
}

fn connect_result() -> MCPConnectResult {
    MCPConnectResult {
        success: true,
        tools: vec![tool("get_board_items_page", Some(json!({"type": "object"})))],
        server_info: Some(handshake()),
        ..Default::default()
    }
}

// ------------------------------------------------- registration carries the slug

#[test]
#[serial]
#[serial(mcp_global)]
fn test_the_oauth_session_holds_every_curated_field() {
    // The record is written in the callback, long after this request has
    // returned, so anything the catalog said has to be in the session.
    let session = register(true, SLUG);
    assert_eq!(session["catalog_slug"], SLUG);
    assert_eq!(session["tags"], json!(CATALOG_TAGS));
    assert_eq!(session["description"], CATALOG_DESC);
    assert_eq!(session["name"], CATALOG_NAME);
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_a_catalog_miss_still_records_what_the_client_asked_for() {
    // `resolve_catalog_entry` never fails — a stale cache, an expired platform
    // token or an outage all read as "not curated".  Provenance is the
    // client's claim about where it installed from, so losing the lookup must
    // not lose the slug.
    let session = register(false, SLUG);
    assert_eq!(session["catalog_slug"], SLUG);
    // The metadata genuinely is unknown here, unlike the provenance.
    assert_eq!(session["description"], Value::Null);
    assert_eq!(session["tags"], Value::Null);
}

// --------------------------------------------------- the callback writes the slug

#[test]
#[serial]
#[serial(mcp_global)]
fn test_the_record_keeps_the_curated_copy_over_the_handshake() {
    let session = register(true, SLUG);
    let env = ServiceEnv::new("p1");
    run_callback(
        Some(Value::Object(session.clone())),
        OAuthCallbackQuery {
            code: Some("c".into()),
            state: session["state"].as_str().map(str::to_string),
            ..Default::default()
        },
        connect_result(),
        token_response(Some("rt"), Some(3600), None),
        env.model_service.clone(),
    );
    reset_router_state();

    let saved = env.servers().into_iter().next().expect("the connection was stored");
    assert_eq!(saved.catalog_slug.as_deref(), Some(SLUG));
    assert_eq!(saved.tags, tags());
    assert_eq!(saved.description.as_deref(), Some(CATALOG_DESC));
    // The vendor calls itself "monday.com MCP Server"; the catalog's display
    // name is what the user picked in the panel.
    assert_eq!(saved.name, CATALOG_NAME);
}

// --------------------------------------------------- the client-driven completion

/// Post the grant back the way the Electron client's `completeOAuth` does.
fn complete(request: MCPOAuthCallbackRequest, session: Option<Map<String, Value>>, env: &ServiceEnv) {
    reset_router_state();
    if let Some(session) = session {
        let state = session["state"].as_str().unwrap_or_default().to_string();
        PENDING_OAUTH_SESSIONS.lock().insert(state, session);
    }
    let exchanged = Arc::new(Mutex::new(0usize));
    let seen = exchanged.clone();
    set_hooks(McpRouteHooks {
        exchange_oauth_code: Some(Arc::new(move |_| {
            *seen.lock() += 1;
            Ok(token_response(Some("rt"), Some(3600), None))
        })),
        discover_remote_tools: Some(Arc::new(|_, _, _| Ok(connect_result()))),
        probe_info: None,
        ..Default::default()
    });
    let result = block_on(complete_mcp_oauth("p1", request, env.model_service.clone()));
    reset_router_state();
    assert!(result.is_ok(), "completion failed: {:?}", result.err());
    assert_eq!(*exchanged.lock(), 1);
}

/// The Electron client has its own `completeOAuth`, which posts the grant back
/// instead of letting the browser redirect land on the GET callback.  It
/// reached `add_remote_mcp_server` with neither tags nor catalog_slug, and with
/// the vendor's handshake name — so the same install, completed the other way,
/// produced a connection with no provenance.
#[test]
#[serial]
#[serial(mcp_global)]
fn test_posting_the_grant_back_keeps_provenance() {
    let session = register(true, SLUG);
    let env = ServiceEnv::new("p1");
    let request = MCPOAuthCallbackRequest {
        server_url: CATALOG_URL.into(),
        code: "c".into(),
        state: session["state"].as_str().unwrap().to_string(),
        code_verifier: session["code_verifier"].as_str().unwrap().to_string(),
        client_id: "cid".into(),
        client_secret: Some("csec".into()),
        token_endpoint: session["token_endpoint"].as_str().unwrap().to_string(),
        authorization_endpoint: session["authorization_endpoint"].as_str().unwrap().to_string(),
        ..Default::default()
    };
    complete(request, Some(session), &env);

    let saved = env.servers().into_iter().next().expect("the connection was stored");
    assert_eq!(saved.catalog_slug.as_deref(), Some(SLUG));
    assert_eq!(saved.tags, tags());
    assert_eq!(saved.name, CATALOG_NAME);
    assert_eq!(saved.description.as_deref(), Some(CATALOG_DESC));
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_a_grant_with_no_session_still_installs() {
    // A session expires after an hour, and a completion arriving without one
    // is still a valid grant: it installs on the handshake's own terms rather
    // than failing.
    let env = ServiceEnv::new("p1");
    let request = MCPOAuthCallbackRequest {
        server_url: CATALOG_URL.into(),
        code: "c".into(),
        state: "gone".into(),
        code_verifier: "cv".into(),
        client_id: "cid".into(),
        token_endpoint: "https://auth.monday.com/oauth_ms/oauth/token".into(),
        authorization_endpoint: "https://auth.monday.com/oauth2/authorize".into(),
        ..Default::default()
    };
    complete(request, None, &env);

    let saved = env.servers().into_iter().next().expect("the connection was stored");
    assert_eq!(saved.catalog_slug, None);
    assert_eq!(Some(saved.name), handshake().name);
}

// ------------------------------------------------- provenance survives persistence

/// The last hop.  A connection installed at 15:40 read back its slug from the
/// API and had lost it on disk by the next process start, so the round trip
/// through config.json is part of the chain too.
#[test]
#[serial]
#[serial(mcp_global)]
fn test_the_slug_and_tags_come_back_after_a_save_and_reload() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().to_string_lossy().into_owned();

    let server = MCPServer {
        id: "s1".into(),
        name: CATALOG_NAME.into(),
        description: Some(CATALOG_DESC.into()),
        tags: tags(),
        catalog_slug: Some(SLUG.into()),
        server_type: MCPServerType::Remote,
        remote_config: Some(RemoteMCPServerConfig { server_url: CATALOG_URL.into(), ..Default::default() }),
        ..Default::default()
    };
    let mut workspace = Workspace::new("w1");
    workspace.workspace_dir = format!("{home}/w1");
    workspace.models_dir = format!("{home}/w1/models");
    workspace.projects.push(Project {
        id: "p1".into(),
        name: Some("proj".into()),
        current: true,
        project_dir: format!("{home}/w1/p1"),
        mcp_servers: vec![server],
        ..Default::default()
    });

    // save() writes to the class-level config path, and load() re-resolves the
    // workspace directory against the settings' home — both are pointed at the
    // temp home rather than the real one.
    smartloop::workspace::context::pin_project(None);
    let config_file = format!("{home}/config.json");
    Workspace::set_config_file(Some(config_file.clone()));
    smartloop::workspace::workspace::set_settings_home_dir(Some(home.clone()));
    let saved = workspace.save();
    let reloaded = Workspace::load();
    Workspace::set_config_file(None);
    smartloop::workspace::workspace::set_settings_home_dir(None);
    saved.expect("the workspace saved");

    let reloaded = reloaded.expect("the workspace reloads");
    let restored = &reloaded.projects[0].mcp_servers[0];
    assert_eq!(restored.catalog_slug.as_deref(), Some(SLUG));
    assert_eq!(restored.tags, tags());
    assert_eq!(restored.description.as_deref(), Some(CATALOG_DESC));

    // And the file itself, since a client reading the persisted state directly
    // is what first showed the field going missing.  Per-project state lives in
    // the project's own directory now; config.json keeps a thin index that
    // says where.
    let index: Value = serde_json::from_str(&std::fs::read_to_string(&config_file).unwrap()).unwrap();
    let index = &index["workspace"]["projects"][0];
    // The bulk stays out of config.json.
    assert!(index.get("mcp_servers").is_none());
    let project_file = format!("{}/project.json", index["project_dir"].as_str().unwrap());
    let persisted: Value = serde_json::from_str(&std::fs::read_to_string(project_file).unwrap()).unwrap();
    assert_eq!(persisted["project"]["mcp_servers"][0]["catalog_slug"], SLUG);
}
