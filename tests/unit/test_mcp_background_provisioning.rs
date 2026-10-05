//! Port of `tests/unit/test_mcp_background_provisioning.py`.
//!
//! Registering an MCP server must not block on discovery or the LLM.
//!
//! Registration used to launch the server process and call the LLM twice for
//! description/tags/summary, all in front of the client's request.  It now
//! persists the connection and hands the rest to the MCP task; these tests pin
//! that the slow work has actually left the request path and that the
//! background pass records a state the client can act on.
//!
//! The Python tests drove the methods with a mock `self`; here the real
//! service runs over a real workspace, with the collaborators the mocks stood
//! in for replaced through `ModelServiceHooks` (each counted).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use parking_lot::Mutex;
use serde_json::{Value, json};
use serial_test::serial;

use smartloop::api::config::api_settings;
use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::mcp::client::{MCPConnectResult, MCPDiscoveryError};
use smartloop::tasks::base_task::{BaseTask, TaskOptions};
use smartloop::tasks::mcp_task::MCPTask;
use smartloop::utils::log_utils::LoggerLike;
use smartloop::workspace::context::{delegate_projects, pin_project};
use smartloop::workspace::{
    MCPServer, MCPServerStatus, MCPServerType, MCPToolDefinition, Project, RemoteMCPServerConfig, Workspace,
};

use crate::shared::model_service_a::{FakeLLM, ScratchConfig, attach_workspace, bare_service};

fn tools() -> Vec<MCPToolDefinition> {
    vec![MCPToolDefinition {
        name: "search".to_string(),
        description: Some("d".to_string()),
        input_schema: json!({"type": "object"}).as_object().cloned(),
    }]
}

/// The pin and the delegation flag are process-global; never leak one.
struct ClearProcessScope;

impl Drop for ClearProcessScope {
    fn drop(&mut self) {
        pin_project(None);
        delegate_projects(false);
    }
}

/// How often each stood-in collaborator was called.
#[derive(Clone, Default)]
struct Counts {
    launches: Arc<AtomicUsize>,
    queued: Arc<AtomicUsize>,
    enriched: Arc<AtomicUsize>,
    discovered: Arc<AtomicUsize>,
    model_loads: Arc<AtomicUsize>,
    large_loads: Arc<AtomicUsize>,
}

impl Counts {
    fn get(counter: &Arc<AtomicUsize>) -> usize {
        counter.load(Ordering::SeqCst)
    }
}

type DiscoverFn = Arc<dyn Fn(&mut MCPServer) -> anyhow::Result<()> + Send + Sync>;

/// A service holding project `p1`, whose collaborators are counted; the
/// discovery step runs `discover` (a no-op by default).
fn svc_with(discover: Option<DiscoverFn>) -> (Arc<ModelService>, Counts, ScratchConfig) {
    let config = ScratchConfig::new();
    let svc = bare_service();
    let mut workspace = Workspace::new("ws");
    workspace.projects = vec![Project {
        id: "p1".to_string(),
        name: Some("proj".to_string()),
        project_dir: config.project_dir("p1"),
        current: true,
        ..Default::default()
    }];
    attach_workspace(&svc, workspace);
    *svc.model_name.write() = Some(api_settings().default_model);

    let counts = Counts::default();
    let c = counts.clone();
    let discover: DiscoverFn = discover.unwrap_or_else(|| Arc::new(|_| Ok(())));
    svc.set_hooks(ModelServiceHooks {
        save_model: Some(Arc::new(|| Ok(()))),
        local_mcp_discovery: Some({
            let launches = c.launches.clone();
            Arc::new(move |_: &str, _: &[String], _: &HashMap<String, String>, _: Option<&str>| {
                launches.fetch_add(1, Ordering::SeqCst);
                Ok(MCPConnectResult { success: true, ..Default::default() })
            })
        }),
        queue_mcp_provisioning: Some({
            let queued = c.queued.clone();
            Arc::new(move |_: &Project, _: &MCPServer| {
                queued.fetch_add(1, Ordering::SeqCst);
            })
        }),
        enrich_mcp_server: Some({
            let enriched = c.enriched.clone();
            Arc::new(move |_: &mut MCPServer| {
                enriched.fetch_add(1, Ordering::SeqCst);
            })
        }),
        discover_mcp_tools: Some({
            let discovered = c.discovered.clone();
            Arc::new(move |server: &mut MCPServer| {
                discovered.fetch_add(1, Ordering::SeqCst);
                discover(server)
            })
        }),
        load_model: Some({
            let loads = c.model_loads.clone();
            Arc::new(move || {
                loads.fetch_add(1, Ordering::SeqCst);
                Ok(FakeLLM::named("loaded"))
            })
        }),
        get_large_llm: Some({
            let loads = c.large_loads.clone();
            Arc::new(move || {
                loads.fetch_add(1, Ordering::SeqCst);
                None
            })
        }),
        ..Default::default()
    });
    (svc, counts, config)
}

fn svc() -> (Arc<ModelService>, Counts, ScratchConfig) {
    svc_with(None)
}

fn add_local(svc: &ModelService, args: &[&str]) -> Value {
    svc.add_local_mcp_server(
        "uvx",
        Some(args.iter().map(|a| a.to_string()).collect()),
        None,
        None,
        None,
        None,
        None,
        None,
        Some("p1"),
    )
    .expect("local registration")
}

fn add_remote(svc: &ModelService, tools: Option<Vec<Value>>) -> Value {
    svc.add_remote_mcp_server(
        "notion",
        "https://mcp.notion.com/mcp",
        "none",
        tools,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "Bearer",
        None,
        Some("p1"),
        None,
        None,
        None,
        None,
        None,
    )
    .expect("remote registration")
}

fn servers(svc: &ModelService) -> Vec<MCPServer> {
    let ws = svc.workspace_handle().expect("workspace");
    let guard = ws.read();
    guard.project("p1").expect("p1").mcp_servers.clone()
}

fn set_server(svc: &ModelService, f: impl FnOnce(&mut MCPServer)) {
    let ws = svc.workspace_handle().expect("workspace");
    let mut guard = ws.write();
    f(&mut guard.project_mut("p1").expect("p1").mcp_servers[0]);
}

// ── TestRegistrationIsNonBlocking ───────────────────────────────────────

#[test]
#[serial]
fn test_local_registration_does_not_launch_the_server() {
    let (svc, counts, _config) = svc();
    add_local(&svc, &["--from", "pkg", "pkg"]);
    assert_eq!(Counts::get(&counts.launches), 0);
}

#[test]
#[serial]
fn test_local_registration_makes_no_llm_call() {
    let (svc, counts, _config) = svc();
    add_local(&svc, &["--from", "pkg", "pkg"]);
    assert_eq!(Counts::get(&counts.enriched), 0);
}

#[test]
#[serial]
fn test_local_registration_persists_pending_and_queues_the_work() {
    let (svc, counts, _config) = svc();
    let result = add_local(&svc, &["--from", "pkg", "pkg"]);
    let servers = servers(&svc);
    assert_eq!(servers.len(), 1);
    assert_eq!(servers[0].status, MCPServerStatus::Pending);
    assert_eq!(result["status"], "pending");
    assert_eq!(Counts::get(&counts.queued), 1);
}

/// The whole point: no inference, and no model load, on this path.
///
/// Enrichment loads a model when nothing is resident, so registering a
/// connection used to be able to pull a multi-GB model into memory before the
/// API could answer.
#[test]
#[serial]
fn test_registration_never_loads_a_model() {
    let (svc, counts, _config) = svc();
    add_local(&svc, &["--from", "pkg", "pkg"]);
    add_remote(&svc, None);
    assert_eq!(Counts::get(&counts.model_loads), 0);
    assert_eq!(Counts::get(&counts.large_loads), 0);
    assert!(svc.get_llm().is_none());
    assert_eq!(Counts::get(&counts.enriched), 0);
}

#[test]
#[serial]
fn test_remote_registration_defers_enrichment() {
    let (svc, counts, _config) = svc();
    let result = add_remote(&svc, Some(vec![json!({"name": "search", "description": "d"})]));
    assert_eq!(Counts::get(&counts.enriched), 0);
    assert_eq!(Counts::get(&counts.queued), 1);
    assert_eq!(result["status"], "pending");
}

#[test]
#[serial]
fn test_registration_response_carries_the_state_clients_poll() {
    let (svc, _counts, _config) = svc();
    add_remote(&svc, None);
    let result = ModelService::mcp_server_to_dict(&servers(&svc)[0]);
    assert_eq!(result["status"], "pending");
    assert_eq!(result["last_error"], Value::Null);
    assert_eq!(result["tools"], json!([]));
}

// ── TestProvisioningPass ────────────────────────────────────────────────

#[test]
#[serial]
fn test_successful_pass_discovers_and_enriches() {
    let (svc, counts, _config) = svc_with(Some(Arc::new(|s: &mut MCPServer| {
        s.tools = tools();
        Ok(())
    })));
    add_remote(&svc, None);
    let server_id = servers(&svc)[0].id.clone();

    let result = svc.provision_mcp_server("p1", &server_id).expect("provision");

    let server = &servers(&svc)[0];
    assert_eq!(server.status, MCPServerStatus::Ready);
    assert_eq!(server.last_error, None);
    assert_eq!(Counts::get(&counts.enriched), 1);
    // Discovery is the whole catalog: the tools it recorded are what MCPAgent
    // resolves a step against, so there is nothing to index and nothing to
    // crawl.
    assert_eq!(server.tools, tools());
    assert_eq!(result["status"], "ready");
    assert_eq!(result["tools"], 1);
}

#[test]
#[serial]
fn test_failed_discovery_is_recorded_on_the_server_not_raised() {
    let (svc, counts, _config) = svc_with(Some(Arc::new(|_: &mut MCPServer| {
        Err(MCPDiscoveryError("Package not found.".to_string()).into())
    })));
    add_local(&svc, &["--from", "broken", "broken"]);
    let server_id = servers(&svc)[0].id.clone();

    let result = svc.provision_mcp_server("p1", &server_id).expect("provision");

    let server = &servers(&svc)[0];
    assert_eq!(server.status, MCPServerStatus::Failed);
    assert_eq!(server.last_error.as_deref(), Some("Package not found."));
    assert_eq!(result["status"], "failed");
    // A failed discovery must not leave half-provisioned state behind.
    assert_eq!(Counts::get(&counts.enriched), 0);
}

#[test]
#[serial]
fn test_discovery_is_skipped_when_registration_supplied_the_tools() {
    let (svc, counts, _config) = svc();
    add_remote(&svc, Some(vec![json!({"name": "search", "description": "d"})]));
    let server_id = servers(&svc)[0].id.clone();

    svc.provision_mcp_server("p1", &server_id).expect("provision");

    assert_eq!(Counts::get(&counts.discovered), 0);
    assert_eq!(Counts::get(&counts.enriched), 1);
}

#[test]
#[serial]
fn test_a_deleted_server_is_a_no_op() {
    let (svc, _counts, _config) = svc();
    // A miss re-reads the workspace from disk (another process may have
    // registered the server), so give it this workspace's own record to read
    // rather than whatever the machine's real config holds.
    svc.workspace_handle().expect("workspace").read().save().expect("save scratch workspace");

    let result = svc.provision_mcp_server("p1", "gone").expect("provision");
    assert_eq!(result["status"], "skipped");
}

#[test]
#[serial]
fn test_retry_resets_the_server_to_pending_and_requeues() {
    let (svc, counts, _config) = svc();
    add_local(&svc, &["--from", "pkg", "pkg"]);
    set_server(&svc, |s| {
        s.status = MCPServerStatus::Failed;
        s.last_error = Some("boom".to_string());
    });
    let server_id = servers(&svc)[0].id.clone();

    let result = svc.retry_mcp_server(&server_id, Some("p1")).expect("retry");

    let server = &servers(&svc)[0];
    assert_eq!(server.status, MCPServerStatus::Pending);
    assert_eq!(server.last_error, None);
    assert_eq!(result["status"], "pending");
    assert_eq!(Counts::get(&counts.queued), 2);
}

#[test]
fn test_re_indexing_a_healthy_server_still_requeues_it() {
    // The "re-index" button in the connections list: nothing is wrong with
    // the server, the user just wants its content walked again.
    let (svc, counts, _config) = svc();
    add_remote(&svc, None);
    set_server(&svc, |s| s.status = MCPServerStatus::Ready);
    let server_id = servers(&svc)[0].id.clone();

    svc.retry_mcp_server(&server_id, Some("p1")).expect("retry");

    // Pending is what the card polls to show the pass as running.
    assert_eq!(servers(&svc)[0].status, MCPServerStatus::Pending);
    assert_eq!(Counts::get(&counts.queued), 2);
}

// ── TestRemoteDiscovery ─────────────────────────────────────────────────

fn remote_server(token: Option<&str>, url: &str) -> MCPServer {
    MCPServer {
        id: "s1".to_string(),
        name: "notion".to_string(),
        server_type: MCPServerType::Remote,
        remote_config: Some(RemoteMCPServerConfig {
            server_url: url.to_string(),
            token: token.map(str::to_string),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn test_remote_tools_are_discovered_in_the_background_pass() {
    let mut server = remote_server(Some("tok"), "https://mcp.notion.com/mcp");
    let svc = bare_service();
    let seen_token: Arc<Mutex<Option<Option<String>>>> = Arc::default();
    let record = seen_token.clone();
    svc.set_hooks(ModelServiceHooks {
        remote_mcp_discovery: Some(Arc::new(move |_url: &str, token: Option<&str>| {
            *record.lock() = Some(token.map(str::to_string));
            MCPConnectResult { success: true, tools: tools(), ..Default::default() }
        })),
        ..Default::default()
    });

    svc.discover_mcp_tools(&mut server).expect("discovery");

    let names: Vec<&str> = server.tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["search"]);
    assert_eq!(*seen_token.lock(), Some(Some("tok".to_string())));
}

#[test]
fn test_remote_discovery_failure_raises_a_readable_reason() {
    let mut server = remote_server(None, "https://x.test/mcp");
    let svc = bare_service();
    svc.set_hooks(ModelServiceHooks {
        remote_mcp_discovery: Some(Arc::new(|_url: &str, _token: Option<&str>| MCPConnectResult {
            success: false,
            error: Some("Connection refused".to_string()),
            ..Default::default()
        })),
        ..Default::default()
    });

    let result = svc.discover_mcp_tools(&mut server);

    assert!(matches!(&result, Err(e) if e.downcast_ref::<MCPDiscoveryError>().is_some()));
}

// ── TestMCPTaskQueue ────────────────────────────────────────────────────
//
// The task owns provisioning and nothing else — the periodic content crawl it
// used to run is gone, so a session starts without indexing.

/// `logger=MagicMock()`.
struct NullLogger;

impl LoggerLike for NullLogger {
    fn info(&self, _msg: &str) {}
    fn warning(&self, _msg: &str) {}
    fn debug(&self, _msg: &str) {}
    fn error(&self, _msg: &str) {}
}

/// Stops the task when the test ends, however it ends.
struct StopOnDrop(Arc<MCPTask>);

impl Drop for StopOnDrop {
    fn drop(&mut self) {
        self.0.stop();
    }
}

#[test]
fn test_enqueued_server_is_provisioned_off_the_caller_thread() {
    let seen: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
    let (tx, rx) = std::sync::mpsc::channel();
    let record = seen.clone();
    let tx = Mutex::new(tx);
    let task = MCPTask::new(
        Arc::new(NullLogger),
        Some(Arc::new(move |project_id: &str, server_id: &str| {
            record.lock().push((project_id.to_string(), server_id.to_string()));
            let _ = tx.lock().send(());
            Ok(json!({}))
        })),
        None,
        TaskOptions::default(),
    );
    let _stop = StopOnDrop(task.clone());

    task.enqueue("p1", "s1");
    assert!(rx.recv_timeout(Duration::from_secs(5)).is_ok());
    assert_eq!(*seen.lock(), vec![("p1".to_string(), "s1".to_string())]);
}

#[test]
fn test_pending_servers_from_a_previous_run_are_adopted() {
    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    let (tx, rx) = std::sync::mpsc::channel();
    let record = seen.clone();
    let tx = Mutex::new(tx);
    let task = MCPTask::new(
        Arc::new(NullLogger),
        Some(Arc::new(move |_project_id: &str, server_id: &str| {
            record.lock().push(server_id.to_string());
            let _ = tx.lock().send(());
            Ok(json!({}))
        })),
        Some(Arc::new(|| Ok(vec![("p1".to_string(), "left-pending".to_string())]))),
        TaskOptions::default(),
    );
    let _stop = StopOnDrop(task.clone());

    task.start_loop();
    assert!(rx.recv_timeout(Duration::from_secs(5)).is_ok());
    assert_eq!(*seen.lock(), vec!["left-pending".to_string()]);
}

#[test]
fn test_status_reports_the_queue_for_health() {
    let task = MCPTask::new(Arc::new(NullLogger), None, None, TaskOptions::default());
    assert_eq!(task.mcp_status(), json!({"queued": 0, "provisioning": 0}));
}

// ── TestAdoptionIsScopedToTheProcessSOwnProjects ────────────────────────
//
// Provisioning ends in a write, so only the owner may adopt.  A pass finishes
// by recording `ready`/`failed` on the server, which is a write to that
// project's `project.json`.  A process that cannot make that write would do
// the whole slow half of registration and then throw the answer away.

fn pending_project(pid: &str) -> Project {
    Project {
        id: pid.to_string(),
        mcp_servers: vec![MCPServer {
            id: format!("{pid}-srv"),
            system: false,
            status: MCPServerStatus::Pending,
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn svc_with_projects(projects: Vec<Project>) -> Arc<ModelService> {
    let svc = bare_service();
    let mut workspace = Workspace::new("ws");
    workspace.projects = projects;
    attach_workspace(&svc, workspace);
    *svc.model_name.write() = Some(api_settings().default_model);
    svc
}

fn pairs(items: &[(&str, &str)]) -> Vec<(String, String)> {
    items.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
}

#[test]
#[serial]
fn test_a_child_adopts_only_the_project_it_is_pinned_to() {
    let _scope = ClearProcessScope;
    let svc = svc_with_projects(vec![pending_project("alpha"), pending_project("beta")]);
    pin_project(Some("beta"));

    assert_eq!(svc.pending_mcp_servers().expect("pending"), pairs(&[("beta", "beta-srv")]));
}

/// Its children own those files; each picks its own server up.
#[test]
#[serial]
fn test_a_supervisor_adopts_nothing() {
    let _scope = ClearProcessScope;
    let svc = svc_with_projects(vec![pending_project("alpha"), pending_project("beta")]);
    delegate_projects(true);

    assert!(svc.pending_mcp_servers().expect("pending").is_empty());
}

#[test]
#[serial]
fn test_the_single_server_fallback_still_adopts_everything() {
    let _scope = ClearProcessScope;
    let svc = svc_with_projects(vec![pending_project("alpha"), pending_project("beta")]);

    assert_eq!(
        svc.pending_mcp_servers().expect("pending"),
        pairs(&[("alpha", "alpha-srv"), ("beta", "beta-srv")])
    );
}

#[test]
#[serial]
fn test_a_system_server_is_never_adopted() {
    let _scope = ClearProcessScope;
    let mut project = pending_project("alpha");
    project.mcp_servers[0].system = true;
    let svc = svc_with_projects(vec![project]);

    assert!(svc.pending_mcp_servers().expect("pending").is_empty());
}

// ── TestIdleSignal ──────────────────────────────────────────────────────
//
// What tells background indexing to stay out of the user's way.

fn idle_svc() -> Arc<ModelService> {
    // One inference permit, nothing active, never answered.
    bare_service()
}

#[test]
fn test_a_fresh_session_is_idle() {
    assert_eq!(idle_svc().idle_seconds(), f64::INFINITY);
}

#[test]
fn test_holding_an_inference_slot_reads_as_busy() {
    let svc = idle_svc();
    let _slot = svc.inference_slot();
    assert_eq!(svc.idle_seconds(), 0.0);
}

#[test]
fn test_idle_time_starts_when_the_answer_finishes() {
    let svc = idle_svc();
    {
        let _slot = svc.inference_slot();
    }
    let idle = svc.idle_seconds();
    assert!((0.0..5.0).contains(&idle));
}

#[test]
fn test_a_raising_inference_still_releases_the_slot() {
    // Otherwise one failed generation leaves the app "busy" forever and the
    // content crawl never runs again.
    let svc = idle_svc();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _slot = svc.inference_slot();
        panic!("generation failed");
    }));
    assert!(outcome.is_err());
    assert_ne!(svc.idle_seconds(), 0.0);
}
