//! Port of `smartloop/api/project_agent.py`.
//!
//! One agent process, serving exactly one project
//! (`slp __project-agent --project <id>`).
//!
//! The project is fixed at spawn and never varies for the life of the process
//! — which is the whole point.  A shared server had to decide, per request,
//! which project it was looking at, and did so by flipping a flag that any
//! other request could move.  A child has nothing to flip: it pins its project
//! (`pin_project`) before the app is built, and every `current_project` read in
//! the process resolves to that one.
//!
//! It runs the same axum app as the single-process server — same routers, same
//! contract — on an ephemeral loopback port that the supervisor proxies to.
//! What it owns is everything project-shaped: its own task queue, vector
//! index, MCP stdio children, sessions and orchestration.  What it does not own
//! is the device: generation and embedding go to the model host over
//! `SLP_MODEL_HOST` when one is configured.

use std::net::TcpListener;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Duration;

use clap::CommandFactory;
use tokio::sync::Notify;

use crate::api::main::serve_listener;
use crate::config::AppSettings;
use crate::supervisor::parent_watch::{start_parent_watch, ParentWatch};
use crate::supervisor::registry::{clear_project_files, write_project_files, AgentRecord};
use crate::utils::logging_setup::{
    get_logger, install_tracing_bridge, make_log_queue, setup_root_logging, start_listener, ListenerOptions,
    QueueListener,
};
use crate::workspace::context::pin_project;
use crate::workspace::Workspace;

/// Logger name of this module.
pub const LOGGER: &str = "smartloop.api.project_agent";

/// This project's own log, beside the project it describes.
pub const AGENT_LOG_FILE: &str = "agent.log";

/// How long an orphaned agent is given to shut down cleanly before it is taken
/// down the blunt way.  An orphan is often orphaned *because* something in it
/// is wedged — a hung MCP call, a request blocked on a model host that died
/// with the supervisor — and the lifespan shutdown that stops a healthy
/// process can then block forever, which would leave exactly the process this
/// is here to remove.
pub const ORPHAN_SHUTDOWN_GRACE_SECONDS: f64 = 20.0;

/// Command-line arguments (`argparse` in Python).
#[derive(Debug, Clone, clap::Parser)]
#[command(name = "smartloop.api.project_agent", about = "Serve a single project")]
pub struct ProjectAgentArgs {
    /// project id (default: $SLP_PROJECT_ID)
    #[arg(long)]
    pub project: Option<String>,
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// 0 (default) asks the OS for a free port
    #[arg(long, default_value_t = 0)]
    pub port: u16,
    /// default: $LOG_LEVEL, else info
    #[arg(long = "log-level")]
    pub log_level: Option<String>,
}

/// Reserve a loopback port and hand back the listener still holding it, so
/// nothing else can take the port between choosing and serving.
///
/// Asking the OS for a free port and then closing the socket before the server
/// binds leaves a window in which something else can take it.  Passing the
/// bound listener straight to the server closes that window entirely.  (The
/// standard library sets `SO_REUSEADDR` on a Unix listener, as the Python did
/// explicitly.)
pub fn bind_ephemeral_port(host: &str) -> std::io::Result<(TcpListener, u16)> {
    let listener = TcpListener::bind((host, 0))?;
    let port = listener.local_addr()?.port();
    Ok((listener, port))
}

/// Where this project's state lives, or `None` when the workspace or project
/// is unknown — without loading the whole workspace.
pub fn resolve_project_dir(project_id: &str) -> Option<String> {
    let workspace = Workspace::load()?;
    workspace.projects.iter().find(|p| p.id == project_id).map(|p| p.project_dir.clone())
}

/// Send this agent's logs to `<project_dir>/agent.log` (plus the console when
/// spawned from a terminal).  A log file that cannot be opened is logged and
/// yields `None` rather than failing.
///
/// One shared `server.log` was right when one process served every project.
/// With a process per project it would interleave every project's
/// orchestration, DAG, planner and MCP trace into a single file, and working
/// out which lines belong to the project you are actually debugging becomes
/// the hard part.  A file per project makes that question disappear.
///
/// A console handler is added on top when the agent was spawned from a
/// terminal — a child inherits the supervisor's stdout, so a debug run still
/// shows the trace live instead of only writing it to a file nobody is
/// tailing.
pub fn install_project_logging(project_dir: &str, log_level: &str) -> Option<Arc<QueueListener>> {
    let level = if log_level.is_empty() { "INFO".to_string() } else { log_level.to_uppercase() };
    let queue = make_log_queue();
    // Root stays at INFO so dependencies (html5ever, llama.cpp) never log
    // at DEBUG; only "smartloop" is lifted below.
    setup_root_logging(&queue, "INFO");
    // A packaged build re-execs through the CLI entry point, which clamps
    // "smartloop" to WARNING to keep the interactive shell quiet.  Lift it
    // when DEBUG was actually asked for, exactly as the single-process server
    // does.
    if level == "DEBUG" {
        get_logger("smartloop").set_level("DEBUG");
    }
    let mut options = ListenerOptions::new(project_dir);
    options.file_level = level.as_str().into();
    options.console_level = level.as_str().into();
    options.log_path =
        Some(std::path::Path::new(project_dir).join(AGENT_LOG_FILE).to_string_lossy().into_owned());
    match start_listener(&queue, options) {
        Ok(listener) => Some(listener),
        Err(exc) => {
            // A log file we cannot open is not a reason to refuse to serve the
            // project; the child keeps whatever logging it already had.
            tracing::warn!(target: LOGGER, "[project-agent] could not open the project log: {}", exc);
            None
        }
    }
}

/// The level this agent should log at, from `LOG_LEVEL` like everything else
/// (lower-cased, default "info").
///
/// Without this a child would sit at INFO while the supervisor ran at DEBUG —
/// and since orchestration happens in the child, `LOG_LEVEL=DEBUG` would turn
/// on verbose logging in the one process that has nothing verbose to say.
pub fn default_log_level() -> String {
    let level = AppSettings::new().log_level;
    if level.is_empty() { "info".to_string() } else { level.to_lowercase() }
}

/// Serve `project_id` until the process is stopped.
pub async fn run(project_id: &str, host: &str, port: u16, log_level: Option<&str>) -> anyhow::Result<()> {
    let log_level = match log_level {
        Some(level) if !level.is_empty() => level.to_lowercase(),
        _ => default_log_level(),
    };

    // Before anything builds the app: every current_project read in this
    // process — routers, orchestrator, background tasks — resolves through
    // this from here on.
    pin_project(Some(project_id));
    // SAFETY: set once at startup, before the server starts any worker that
    // reads the environment.
    unsafe { std::env::set_var("SLP_PROJECT_ID", project_id) };

    let mut port = port;
    let mut reserved: Option<TcpListener> = None;
    if port == 0 {
        let (listener, bound) = bind_ephemeral_port(host)?;
        reserved = Some(listener);
        port = bound;
    }

    let project_dir = resolve_project_dir(project_id);
    let mut log_listener: Option<Arc<QueueListener>> = None;
    if let Some(dir) = project_dir.as_deref() {
        write_project_files(dir, &AgentRecord::new(project_id, i64::from(std::process::id()), port));
        log_listener = install_project_logging(dir, &log_level);
    }

    tracing::info!(
        target: LOGGER,
        "[project-agent] project={} pid={} port={}",
        project_id,
        std::process::id(),
        port
    );

    let shutdown = Arc::new(Notify::new());
    let trigger = Arc::clone(&shutdown);
    // notify_one keeps a permit when nothing is waiting yet, so an orphaning
    // noticed before the server is listening is not lost.
    let watch = watch_the_supervisor(Arc::new(move || trigger.notify_one()));

    let served = async {
        let listener = match reserved {
            Some(listener) => {
                listener.set_nonblocking(true)?;
                tokio::net::TcpListener::from_std(listener)?
            }
            None => tokio::net::TcpListener::bind((host, port)).await?,
        };
        // A child is a plain server: the shared `server.port` belongs to the
        // process that owns the public port, so nothing is captured here.
        serve_listener(listener, false, Some(shutdown)).await
    }
    .await;

    if let Some(watch) = watch {
        watch.stop(5.0);
    }
    if let Some(dir) = project_dir.as_deref() {
        clear_project_files(dir);
    }
    if let Some(listener) = log_listener {
        listener.stop();
    }

    // Every way the serve loop ends — Ctrl-C included, which the Python turned
    // into `sys.exit(0)` — is a clean stop; only a failure to serve is an error.
    served.map(|_| ())
}

/// Exit if the supervisor that spawned this agent goes away.
///
/// `request_shutdown` asks the server to stop gracefully; a detached backstop
/// timer calls [`hard_exit`] after [`ORPHAN_SHUTDOWN_GRACE_SECONDS`].
///
/// A child is spawned detached, so nothing in the OS links the two: before
/// this, a supervisor that was killed rather than stopped left its children
/// serving projects that no longer had a router in front of them, holding
/// their task queues, MCP stdio children and model-host connections open
/// until the machine rebooted.
///
/// Asking the server to exit rather than exiting here: the lifespan shutdown
/// is what closes the SQLite handles, drains the task queue and terminates
/// this project's MCP grandchildren, and skipping it would trade one orphan
/// for several.  The hard exit is only the backstop for when that does not
/// return.
pub fn watch_the_supervisor(request_shutdown: Arc<dyn Fn() + Send + Sync>) -> Option<Arc<ParentWatch>> {
    let shut_down = move || {
        request_shutdown();
        // Detached, so the backstop never becomes the delay: when the
        // graceful shutdown does return, the process exits at once and takes
        // the pending timer with it instead of waiting out the grace period
        // and then hard-exiting a process that had already finished.
        let spawned = std::thread::Builder::new().name("orphan-backstop".into()).spawn(|| {
            std::thread::sleep(Duration::from_secs_f64(ORPHAN_SHUTDOWN_GRACE_SECONDS));
            hard_exit();
        });
        if let Err(e) = spawned {
            tracing::error!(target: LOGGER, "[project-agent] could not start the orphan backstop: {}", e);
        }
    };

    let home_dir = std::panic::catch_unwind(AssertUnwindSafe(|| AppSettings::new().home_dir())).ok();
    start_parent_watch(Arc::new(shut_down), home_dir)
}

/// Last resort: leave without running shutdown (`os._exit(1)`).
///
/// `_exit` because the graceful path has already been given its chance and
/// did not return — a normal exit from this timer thread would run exit
/// handlers that may themselves be what is wedged, and leave the process
/// exactly where it was.
pub fn hard_exit() {
    tracing::error!(
        target: LOGGER,
        "[project-agent] orphan shutdown did not finish in {:.0}s; exiting hard",
        ORPHAN_SHUTDOWN_GRACE_SECONDS
    );
    #[cfg(unix)]
    // SAFETY: `_exit` terminates the process immediately and never returns.
    unsafe {
        libc::_exit(1)
    }
    #[cfg(not(unix))]
    std::process::exit(1)
}

/// Entry point for the hidden `__project-agent` subcommand.  Resolves the
/// project from `--project` or `SLP_PROJECT_ID` (error when neither) and runs.
pub fn main(args: ProjectAgentArgs) -> anyhow::Result<()> {
    let project_id = args
        .project
        .clone()
        .filter(|p| !p.is_empty())
        .or_else(|| std::env::var("SLP_PROJECT_ID").ok().filter(|p| !p.is_empty()));
    let Some(project_id) = project_id else {
        ProjectAgentArgs::command()
            .error(
                clap::error::ErrorKind::MissingRequiredArgument,
                "no project given (pass --project or set SLP_PROJECT_ID)",
            )
            .exit();
    };

    let log_level = args.log_level.clone().filter(|l| !l.is_empty()).unwrap_or_else(default_log_level);
    // `logging.basicConfig(level=...)`: set the root level and route `tracing`
    // events into the logging tree.  Project logging (installed in `run`)
    // then points the root at the shared queue.
    get_logger("").set_level(log_level.to_uppercase().as_str());
    install_tracing_bridge();

    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    runtime.block_on(run(&project_id, &args.host, args.port, Some(&log_level)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ephemeral_port_is_held_until_served() {
        let (listener, port) = bind_ephemeral_port("127.0.0.1").unwrap();
        assert_ne!(port, 0);
        assert_eq!(listener.local_addr().unwrap().port(), port);
        // Still held: nothing else can bind it while the listener lives.
        assert!(TcpListener::bind(("127.0.0.1", port)).is_err());
    }

    #[test]
    fn arguments_default_like_argparse() {
        use clap::Parser;
        let args = ProjectAgentArgs::parse_from(["smartloop.api.project_agent"]);
        assert_eq!(args.project, None);
        assert_eq!(args.host, "127.0.0.1");
        assert_eq!(args.port, 0);
        assert_eq!(args.log_level, None);
        let args = ProjectAgentArgs::parse_from([
            "smartloop.api.project_agent",
            "--project",
            "p1",
            "--port",
            "8123",
            "--log-level",
            "debug",
        ]);
        assert_eq!(args.project.as_deref(), Some("p1"));
        assert_eq!(args.port, 8123);
        assert_eq!(args.log_level.as_deref(), Some("debug"));
    }
}
