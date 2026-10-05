//! Port of `main.py` — the `slp` CLI.
//!
//! The CLI is intentionally minimal: it manages the agent lifecycle
//! (start/stop/status/restart) only.  All other functionality is exposed
//! through the internal API server that the agent runs.

use clap::{Args, Parser, Subcommand};
use serde_json::Value;

#[allow(unused_imports)]
use smartloop::config::AppSettings;
#[allow(unused_imports)]
use smartloop::constants::SLP_PRIMARY;
#[allow(unused_imports)]
use smartloop::server::{get_status, is_server_running, start_server, stop_server};

/// Top-level parser.
#[derive(Debug, Parser)]
#[command(
    name = "slp",
    version = smartloop::VERSION,
    about = "AI Agent harness for extracting information and creating new content"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

/// Available commands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Agent management commands (start/stop/status/restart)
    Agent {
        #[command(subcommand)]
        agent_command: Option<AgentCommand>,
    },
}

/// Agent commands.
#[derive(Debug, Subcommand)]
pub enum AgentCommand {
    /// Start the agent in background
    Start(StartArgs),
    /// Stop the background agent
    Stop,
    /// Show agent status
    Status,
    /// Restart the agent
    Restart(StartArgs),
}

/// Options shared by `agent start` and `agent restart`.
#[derive(Debug, Args, Default, Clone)]
pub struct StartArgs {
    /// Disable auto-restart on crash
    #[arg(long = "no-service")]
    pub no_service: bool,
    /// API server host (default: $SLP_API_HOST or 127.0.0.1)
    #[arg(long)]
    pub host: Option<String>,
    /// API server port (default: $SLP_API_PORT or 38540)
    #[arg(long, short = 'p')]
    pub port: Option<u16>,
    /// Serve every project from one shared process (default: one process per project)
    #[arg(long = "no-supervised")]
    pub no_supervised: bool,
}

/// Human-readable RSS, or a dash when the process could not be read.
fn format_bytes(value: Option<f64>) -> String {
    let _ = value;
    todo!()
}

/// Report one process, or the whole tree when running supervised.
fn print_status(status: &Value, host: &str, port: u16) {
    let _ = (status, host, port, format_bytes(None));
    todo!()
}

/// Dispatch an `slp agent ...` subcommand.
fn cmd_agent(agent_command: Option<&AgentCommand>, host: &str, port: u16) -> anyhow::Result<()> {
    let _ = (agent_command, host, port, print_status as fn(&Value, &str, u16));
    todo!()
}

/// Run an internal module entry point, Python's `slp -m <module> [args...]`
/// (internal stdio MCP servers, the model host, agent children and workers
/// are spawned as `<current_exe> -m <module> ...`).  Checked on the raw argv
/// before clap parsing, as `main.py` did.
fn run_module(module: &str, args: Vec<String>) -> anyhow::Result<()> {
    let _ = (module, args);
    todo!()
}

fn main() -> anyhow::Result<()> {
    // [llms] The model-load crash probe (`LlmFactory::probe_model_load`)
    // re-executes this binary as `slp __probe-model <config-json>`.
    {
        use smartloop::llms::llm_factory::{LlmFactory, PROBE_SUBCOMMAND};
        let argv: Vec<String> = std::env::args().collect();
        if argv.get(1).map(String::as_str) == Some(PROBE_SUBCOMMAND) {
            std::process::exit(LlmFactory::run_probe(argv.get(2).map(String::as_str).unwrap_or("")));
        }
    }
    let _ = run_module as fn(&str, Vec<String>) -> anyhow::Result<()>;
    let _ = cmd_agent as fn(Option<&AgentCommand>, &str, u16) -> anyhow::Result<()>;
    todo!()
}
