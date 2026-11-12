//! Port of `tests/unit/test_mcp_account_probe.py`.
//!
//! Which account a connection is linked as.
//!
//! "Connected" told a user nothing they couldn't already see.  The question
//! they actually have, looking at two Notion cards, is whose Notion each one is
//! — so a connection carries an account label, probed once when it is
//! connected.
//!
//! Labelling a card with the *wrong* account is worse than leaving it blank, so
//! most of what is pinned here is the probe declining to guess.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;
use serde_json::{Map, Value, json};
use serial_test::serial;

use smartloop::agents::mcp::argument_extractor::MCPArgumentExtractor;
use smartloop::agents::mcp::mcp_tool_caller::MCPToolCaller;
use smartloop::agents::mcp::result_mapper::MCPResultMapper;
use smartloop::api::config::api_settings;
use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::mcp::account::{extract_account, probe_account, resolve_account_tool};
use smartloop::mcp::client::MCPToolCallResult;
use smartloop::mcp::read_only_calls::ReadOnlyToolCaller;
use smartloop::workspace::{MCPOAuthCredentials, MCPServer, MCPServerStatus, MCPToolDefinition, Project, Workspace};

use crate::shared::model_service_a::{FakeLLM, ScratchConfig, attach_workspace, bare_service};

fn tool(name: &str) -> MCPToolDefinition {
    MCPToolDefinition { name: name.to_string(), description: Some(name.to_string()), input_schema: Some(Map::new()), ..Default::default() }
}

fn server(tools: Vec<MCPToolDefinition>) -> MCPServer {
    MCPServer { id: "s1".to_string(), name: "notion".to_string(), tools, ..Default::default() }
}

// ── TestToolResolution ──────────────────────────────────────────────────

#[test]
fn test_the_identity_tool_is_found_under_any_spelling() {
    assert_eq!(resolve_account_tool(&["notion-get-self"]).as_deref(), Some("notion-get-self"));
    assert_eq!(resolve_account_tool(&["API-get-self"]).as_deref(), Some("API-get-self"));
    assert_eq!(resolve_account_tool(&["get_viewer", "list_issues"]).as_deref(), Some("get_viewer"));
}

#[test]
fn test_a_member_listing_is_not_an_identity_tool() {
    // A loose matcher would pair `get_me` with `get_members`; naming a
    // connection after whoever tops a member list is worse than not naming it
    // at all, so this match is exact.
    assert_eq!(resolve_account_tool(&["get_members", "get_boards"]), None);
}

#[test]
fn test_a_server_with_no_identity_tool_resolves_to_nothing() {
    assert_eq!(resolve_account_tool(&["search", "fetch"]), None);
}

// ── TestExtraction ──────────────────────────────────────────────────────

#[test]
fn test_notion_oauth_yields_the_person_behind_the_bot() {
    // Notion authenticates as a bot whose own `name` is the integration; the
    // human is several levels down.
    let payload = json!({
        "object": "user", "id": "u1", "name": "Smartloop",
        "bot": {
            "owner": {"type": "user", "user": {
                "name": "Mehfuz", "person": {"email": "mehfuz@smartloop.ai"},
            }},
            "workspace_name": "Smartloop HQ",
        },
    })
    .to_string();
    assert_eq!(extract_account(Some(&payload)).as_deref(), Some("mehfuz@smartloop.ai"));
}

#[test]
fn test_an_integration_token_falls_back_to_the_workspace() {
    // No person attached — "Smartloop" is the integration's name, which would
    // label every such connection identically.
    let payload = json!({
        "object": "user", "name": "Smartloop",
        "bot": {"workspace_name": "Smartloop HQ"},
    })
    .to_string();
    assert_eq!(extract_account(Some(&payload)).as_deref(), Some("Smartloop HQ"));
}

#[test]
fn test_email_wins_over_a_display_name_at_the_top_level() {
    let payload = json!({
        "id": "u1", "name": "Mehfuz", "displayName": "mehfuz",
        "email": "mehfuz@smartloop.ai",
    })
    .to_string();
    assert_eq!(extract_account(Some(&payload)).as_deref(), Some("mehfuz@smartloop.ai"));
}

#[test]
fn test_a_name_is_used_when_there_is_no_email() {
    let payload = json!({"id": "u1", "name": "Mehfuz"}).to_string();
    assert_eq!(extract_account(Some(&payload)).as_deref(), Some("Mehfuz"));
}

#[test]
fn test_a_directory_of_users_names_nobody() {
    // `get_users` returning everyone is not an answer to "who am I".
    let payload = json!({"results": [
        {"name": "Ada", "email": "ada@example.com"},
        {"name": "Grace", "email": "grace@example.com"},
    ]})
    .to_string();
    assert_eq!(extract_account(Some(&payload)), None);
}

#[test]
fn test_a_single_wrapped_record_still_resolves() {
    let payload = json!({"data": {"viewer": {"email": "mehfuz@smartloop.ai"}}}).to_string();
    assert_eq!(extract_account(Some(&payload)).as_deref(), Some("mehfuz@smartloop.ai"));
}

#[test]
fn test_a_paragraph_is_not_a_label() {
    let long_text = format!("You are authenticated as {}", "x".repeat(200));
    assert_eq!(extract_account(Some(&long_text)), None);
}

#[test]
fn test_empty_and_unparseable_payloads_yield_nothing() {
    assert_eq!(extract_account(None), None);
    assert_eq!(extract_account(Some("")), None);
    let unrelated = json!({"unrelated": {"count": 3}}).to_string();
    assert_eq!(extract_account(Some(&unrelated)), None);
}

// ── TestProbe ───────────────────────────────────────────────────────────

/// What the stubbed transport was asked: `(tool_name, arguments)`.
type Calls = Arc<Mutex<Vec<(String, Map<String, Value>)>>>;

/// The read-only path's view of a real `MCPToolCaller`: one call through its
/// refresh-and-retry, a panicking transport surfacing as an error (the Rust
/// analogue of one that raised).
struct ReadOnly(Arc<MCPToolCaller>);

impl ReadOnlyToolCaller for ReadOnly {
    fn call_with_refresh(
        &self,
        server: &mut MCPServer,
        tool_name: &str,
        arguments: &Map<String, Value>,
    ) -> anyhow::Result<(MCPToolCallResult, bool)> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            MCPToolCaller::call_with_refresh(&self.0, server, tool_name, Some(arguments), None)
        }))
        .map_err(|_| anyhow::anyhow!("transport failed"))
    }
}

fn transport_caller() -> Arc<MCPToolCaller> {
    Arc::new(MCPToolCaller::new(
        Arc::new(MCPArgumentExtractor::new(FakeLLM::named("extractor"))),
        Arc::new(Mutex::new(MCPResultMapper::new())),
        None,
    ))
}

/// A real caller with the transport stubbed: the refresh-and-retry that every
/// MCP path shares lives on the caller, so a double standing in for the whole
/// caller would not exercise the dispatch under test.
fn caller(payload: &str, success: bool) -> (Arc<dyn ReadOnlyToolCaller>, Calls) {
    let caller = transport_caller();
    let calls: Calls = Arc::default();
    let record = calls.clone();
    let payload = payload.to_string();
    caller.set_call_tool_hook(Some(Arc::new(
        move |_server: &MCPServer, tool_name: &str, arguments: Option<&Map<String, Value>>, _token, _cwd| {
            record.lock().push((tool_name.to_string(), arguments.cloned().unwrap_or_default()));
            MCPToolCallResult {
                success,
                content: Some(vec![json!({"type": "text", "text": payload})]),
                is_error: false,
                error: None,
                auth_expired: false,
            }
        },
    )));
    (Arc::new(ReadOnly(caller)), calls)
}

#[test]
fn test_the_identity_tool_is_called_with_no_arguments() {
    let (caller, calls) = caller(&json!({"email": "mehfuz@smartloop.ai"}).to_string(), true);
    let mut srv = server(vec![tool("notion-get-self"), tool("notion-search")]);

    assert_eq!(
        probe_account(&mut srv, Some(caller)),
        (Some("mehfuz@smartloop.ai".to_string()), false)
    );
    let calls = calls.lock();
    let (tool_name, arguments) = calls.last().expect("a call");
    assert_eq!(tool_name, "notion-get-self");
    assert!(arguments.is_empty());
}

#[test]
fn test_no_identity_tool_means_no_call_at_all() {
    let (caller, calls) = caller("{}", true);
    probe_account(&mut server(vec![tool("search")]), Some(caller));
    assert!(calls.lock().is_empty());
}

#[test]
fn test_a_failed_call_leaves_the_connection_unlabelled() {
    let (caller, _calls) = caller("", false);
    assert_eq!(
        probe_account(&mut server(vec![tool("get_self")]), Some(caller)),
        (None, false)
    );
}

#[test]
fn test_a_raising_transport_does_not_break_provisioning() {
    let caller = transport_caller();
    caller.set_call_tool_hook(Some(Arc::new(|_: &MCPServer, _: &str, _: Option<&Map<String, Value>>, _, _| {
        panic!("server went away")
    })));
    assert_eq!(
        probe_account(&mut server(vec![tool("get_self")]), Some(Arc::new(ReadOnly(caller)))),
        (None, false)
    );
}

// ── the service side ────────────────────────────────────────────────────

/// A service whose workspace really contains `servers`, and its save counter.
///
/// The `persist_mcp_*` saves re-find the server in the live workspace before
/// saving, so a bare double would not exercise the path under test.
fn svc_holding(servers: Vec<MCPServer>) -> (Arc<ModelService>, Arc<AtomicUsize>, ScratchConfig) {
    let config = ScratchConfig::new();
    let svc = bare_service();
    let mut workspace = Workspace::new("ws");
    workspace.projects =
        vec![Project { id: "p".to_string(), project_dir: config.project_dir("p"), mcp_servers: servers, ..Default::default() }];
    attach_workspace(&svc, workspace);
    *svc.model_name.write() = Some(api_settings().default_model);
    let saves = Arc::new(AtomicUsize::new(0));
    let counter = saves.clone();
    svc.set_hooks(ModelServiceHooks {
        save_model: Some(Arc::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })),
        ..Default::default()
    });
    (svc, saves, config)
}

/// `patch("smartloop.mcp.account.probe_account", return_value=...)`.
fn stub_probe(svc: &ModelService, account: Option<&str>, changed: bool) {
    let mut hooks = svc.hooks.read().clone();
    let account = account.map(str::to_string);
    hooks.probe_account = Some(Arc::new(move |_server: &mut MCPServer, _caller| (account.clone(), changed)));
    svc.set_hooks(hooks);
}

fn notion(account: Option<&str>) -> MCPServer {
    MCPServer {
        id: "s1".to_string(),
        name: "notion".to_string(),
        account: account.map(str::to_string),
        ..Default::default()
    }
}

fn live_servers(svc: &ModelService) -> Vec<MCPServer> {
    svc.workspace_handle().expect("workspace").read().projects[0].mcp_servers.clone()
}

#[test]
#[serial]
fn test_a_probed_account_is_persisted() {
    let mut server = notion(None);
    let (svc, saves, _config) = svc_holding(vec![server.clone()]);
    stub_probe(&svc, Some("mehfuz@smartloop.ai"), false);

    svc.probe_mcp_account(&mut server);

    assert_eq!(server.account.as_deref(), Some("mehfuz@smartloop.ai"));
    assert_eq!(saves.load(Ordering::SeqCst), 1);
}

#[test]
#[serial]
fn test_an_unchanged_account_is_not_re_saved() {
    let mut server = notion(Some("mehfuz@smartloop.ai"));
    let (svc, saves, _config) = svc_holding(vec![server.clone()]);
    stub_probe(&svc, Some("mehfuz@smartloop.ai"), false);

    svc.probe_mcp_account(&mut server);

    assert_eq!(saves.load(Ordering::SeqCst), 0);
}

#[test]
#[serial]
fn test_a_probe_that_finds_nothing_keeps_the_previous_label() {
    // A transient failure on re-index must not blank a card that was
    // correctly labelled a moment ago.
    let mut server = notion(Some("mehfuz@smartloop.ai"));
    let (svc, _saves, _config) = svc_holding(vec![server.clone()]);
    stub_probe(&svc, None, false);

    svc.probe_mcp_account(&mut server);

    assert_eq!(server.account.as_deref(), Some("mehfuz@smartloop.ai"));
}

// ── TestARotatedTokenIsWrittenBack ──────────────────────────────────────
//
// The probe is where a refresh token most often gets spent.  It runs on every
// connect and re-index and refreshes on a 401 through the shared read-only
// path, whose caller writes the rotation back as it happens.  The probe's own
// copy of the server is stale once another caller has refreshed, so the probe
// saves the account it found and nothing else.

#[test]
#[serial]
fn test_the_probe_does_not_save_credentials_from_its_copy() {
    let mut held = notion(None);
    held.oauth_credentials = Some(MCPOAuthCredentials { refresh_token: Some("old-rt".to_string()), ..Default::default() });
    let mut live = held.clone();
    live.oauth_credentials =
        Some(MCPOAuthCredentials { refresh_token: Some("rotated-rt".to_string()), ..Default::default() });
    let (svc, saves, _config) = svc_holding(vec![live]);
    stub_probe(&svc, Some("mehfuz@smartloop.ai"), true);

    svc.probe_mcp_account(&mut held);

    let live = &live_servers(&svc)[0];
    assert_eq!(live.account.as_deref(), Some("mehfuz@smartloop.ai"));
    assert_eq!(live.oauth_credentials.as_ref().and_then(|c| c.refresh_token.as_deref()), Some("rotated-rt"));
    assert_eq!(saves.load(Ordering::SeqCst), 1);
}

/// Belt: the probe is handed a caller that can write a rotation back.
///
/// The Python test invoked the caller's private `_persist()` directly; the
/// Rust caller keeps its persistence hook private, so this pins the two
/// halves the hook joins: a caller *is* handed to the probe, and the save it
/// persists a rotation through is the re-attaching one (not a bare save of a
/// detached object).
#[test]
#[serial]
fn test_the_probe_caller_carries_the_persistence_hook() {
    let mut server = notion(None);
    let (svc, saves, _config) = svc_holding(vec![server.clone()]);
    let seen: Arc<Mutex<Option<bool>>> = Arc::default();
    let record = seen.clone();
    let mut hooks = svc.hooks.read().clone();
    hooks.probe_account = Some(Arc::new(move |_server: &mut MCPServer, caller: Option<Arc<dyn ReadOnlyToolCaller>>| {
        *record.lock() = Some(caller.is_some());
        (None, false)
    }));
    svc.set_hooks(hooks);

    svc.probe_mcp_account(&mut server);
    assert_eq!(*seen.lock(), Some(true));

    // It persists through the re-attaching save, not through a bare one.
    svc.persist_mcp_credentials(&server).expect("persist");
    assert_eq!(saves.load(Ordering::SeqCst), 1);
}

/// Brace: probe_account forwards what the read-only path reported.
#[test]
fn test_the_flag_reaches_the_caller_from_the_read_only_path() {
    /// A caller whose call renewed the token and returned `{}`.
    struct Renewed;
    impl ReadOnlyToolCaller for Renewed {
        fn call_with_refresh(
            &self,
            _server: &mut MCPServer,
            _tool_name: &str,
            _arguments: &Map<String, Value>,
        ) -> anyhow::Result<(MCPToolCallResult, bool)> {
            Ok((
                MCPToolCallResult {
                    success: true,
                    content: Some(vec![json!({"type": "text", "text": "{}"})]),
                    is_error: false,
                    error: None,
                    auth_expired: false,
                },
                true,
            ))
        }
    }
    assert_eq!(
        probe_account(&mut server(vec![tool("get_self")]), Some(Arc::new(Renewed) as Arc<dyn ReadOnlyToolCaller>)),
        (None, true)
    );
}

// ── TestTheWriteLandsOnTheLiveWorkspace ─────────────────────────────────
//
// A pass holding a detached server must not save the rotation to it.  A pass
// can outlive the workspace instance it read its server from, so re-finding by
// id before saving is what puts the rotated refresh token on the object that
// gets serialized.

/// The object the pass holds, and the one the workspace holds now.
fn pair() -> (MCPServer, MCPServer) {
    let make = || MCPServer {
        id: "s1".to_string(),
        name: "notion".to_string(),
        oauth_credentials: Some(MCPOAuthCredentials { refresh_token: Some("old-rt".to_string()), ..Default::default() }),
        status: MCPServerStatus::Ready,
        last_error: None,
        ..Default::default()
    };
    (make(), make())
}

#[test]
#[serial]
fn test_a_rotation_is_carried_onto_the_replacement() {
    let (mut held, live) = pair();
    let (svc, saves, _config) = svc_holding(vec![live]); // the reload already happened
    held.oauth_credentials =
        Some(MCPOAuthCredentials { refresh_token: Some("rotated-rt".to_string()), ..Default::default() });

    svc.persist_mcp_credentials(&held).expect("persist");

    let live = &live_servers(&svc)[0];
    assert_eq!(
        live.oauth_credentials.as_ref().and_then(|c| c.refresh_token.as_deref()),
        Some("rotated-rt")
    );
    assert_eq!(saves.load(Ordering::SeqCst), 1);
}

#[test]
#[serial]
fn test_a_failed_connection_is_carried_too() {
    let (mut held, live) = pair();
    let (svc, _saves, _config) = svc_holding(vec![live]);
    held.status = MCPServerStatus::Failed;
    held.last_error = Some("This connection's authorization has expired.".to_string());

    svc.persist_mcp_status(&held).expect("persist");

    let live = &live_servers(&svc)[0];
    assert_eq!(live.status, MCPServerStatus::Failed);
    assert!(live.last_error.as_deref().unwrap_or_default().contains("expired"));
}

/// A failure is recorded by a caller holding a copy whose credentials may
/// predate another caller's refresh; writing them would undo that refresh.
#[test]
#[serial]
fn test_a_failure_leaves_the_live_credentials_alone() {
    let (mut held, mut live) = pair();
    live.oauth_credentials =
        Some(MCPOAuthCredentials { refresh_token: Some("rotated-rt".to_string()), ..Default::default() });
    let (svc, _saves, _config) = svc_holding(vec![live]);
    held.status = MCPServerStatus::Failed;
    held.last_error = Some("This connection's authorization has expired.".to_string());

    svc.persist_mcp_status(&held).expect("persist");

    let live = &live_servers(&svc)[0];
    assert_eq!(live.status, MCPServerStatus::Failed);
    assert_eq!(live.oauth_credentials.as_ref().and_then(|c| c.refresh_token.as_deref()), Some("rotated-rt"));
}

/// Re-finding by id must not resurrect what the user removed.
#[test]
#[serial]
fn test_a_server_deleted_mid_pass_is_left_deleted() {
    let (held, _) = pair();
    let (svc, saves, _config) = svc_holding(vec![]); // the workspace no longer has it

    svc.persist_mcp_credentials(&held).expect("persist");

    assert!(live_servers(&svc).is_empty());
    assert_eq!(saves.load(Ordering::SeqCst), 1);
}

/// No reload happened: the held object *is* the live one.
#[test]
#[serial]
fn test_the_ordinary_case_writes_through_untouched() {
    let (mut held, _) = pair();
    let (svc, saves, _config) = svc_holding(vec![held.clone()]);
    held.oauth_credentials =
        Some(MCPOAuthCredentials { refresh_token: Some("rotated-rt".to_string()), ..Default::default() });

    svc.persist_mcp_credentials(&held).expect("persist");

    assert_eq!(live_servers(&svc)[0], held);
    assert_eq!(saves.load(Ordering::SeqCst), 1);
}

/// A refresh reads the credentials the workspace holds now, not a held copy's.
#[test]
#[serial]
fn test_live_credentials_are_read_from_the_workspace() {
    let (mut held, live) = pair();
    let (svc, _saves, _config) = svc_holding(vec![live]);
    held.oauth_credentials =
        Some(MCPOAuthCredentials { refresh_token: Some("rotated-rt".to_string()), ..Default::default() });
    svc.persist_mcp_credentials(&held).expect("persist");

    let creds = svc.live_mcp_credentials("s1").expect("live credentials");
    assert_eq!(creds.refresh_token.as_deref(), Some("rotated-rt"));
    assert_eq!(svc.live_mcp_credentials("gone"), None);
}
