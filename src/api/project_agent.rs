//! Port of `smartloop/api/project_agent.py`.
//!
//! One agent process, serving exactly one project
//! (`slp __project-agent --project <id>`).
//!
//! The project is fixed at spawn and never varies for the life of the process:
//! it pins its project (`pin_project`) before the app is built, and every
//! `current_project` read in the process resolves to that one.
//!
//! It runs the same axum app as the single-process server on an ephemeral
//! loopback port that the supervisor proxies to.  It owns everything
//! project-shaped (task queue, vector index, MCP stdio children, sessions and
//! orchestration) but not the device: generation and embedding go to the
//! model host over `SLP_MODEL_HOST` when one is configured.

#![allow(unused_imports)]

use std::net::TcpListener;
use std::sync::Arc;

use crate::config::AppSettings;
use crate::supervisor::parent_watch::{start_parent_watch, ParentWatch};
use crate::supervisor::registry::AgentRecord;
use crate::utils::logging_setup::QueueListener;
use crate::workspace::context::pin_project;
use crate::workspace::Workspace;

/// Logger name of this module.
pub const LOGGER: &str = "smartloop.api.project_agent";

/// This project's own log, beside the project it describes.
pub const AGENT_LOG_FILE: &str = "agent.log";

/// How long an orphaned agent is given to shut down cleanly before it is taken
/// down the blunt way.  An orphan is often orphaned *because* something in it
/// is wedged, and the graceful shutdown can then block forever.
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
pub fn bind_ephemeral_port(host: &str) -> std::io::Result<(TcpListener, u16)> {
    let _ = host;
    todo!()
}

/// Where this project's state lives, or `None` when the workspace or project
/// is unknown.
pub fn resolve_project_dir(project_id: &str) -> Option<String> {
    let _ = project_id;
    todo!()
}

/// Send this agent's logs to `<project_dir>/agent.log` (plus the console when
/// spawned from a terminal).  A log file that cannot be opened is logged and
/// yields `None` rather than failing.
pub fn install_project_logging(project_dir: &str, log_level: &str) -> Option<Arc<QueueListener>> {
    let _ = (project_dir, log_level);
    todo!()
}

/// The level this agent should log at, from `LOG_LEVEL` like everything else
/// (lower-cased, default "info").
pub fn default_log_level() -> String {
    todo!()
}

/// Serve `project_id` until the process is stopped.
pub async fn run(project_id: &str, host: &str, port: u16, log_level: Option<&str>) -> anyhow::Result<()> {
    let _ = (project_id, host, port, log_level);
    todo!()
}

/// Exit if the supervisor that spawned this agent goes away.
///
/// `request_shutdown` asks the server to stop gracefully; a daemon backstop
/// timer calls [`hard_exit`] after [`ORPHAN_SHUTDOWN_GRACE_SECONDS`].
pub fn watch_the_supervisor(request_shutdown: Arc<dyn Fn() + Send + Sync>) -> Option<Arc<ParentWatch>> {
    let _ = request_shutdown;
    todo!()
}

/// Last resort: leave without running shutdown (`os._exit(1)`).
pub fn hard_exit() {
    todo!()
}

/// Entry point for the hidden `__project-agent` subcommand.  Resolves the
/// project from `--project` or `SLP_PROJECT_ID` (error when neither) and runs.
pub fn main(args: ProjectAgentArgs) -> anyhow::Result<()> {
    let _ = args;
    todo!()
}
