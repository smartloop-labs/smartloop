//! Port of `main.py` — the `slp` CLI.
//!
//! The CLI is intentionally minimal: it manages the agent lifecycle
//! (start/stop/status/restart) only.  All other functionality is exposed
//! through the internal API server that the agent runs.
//!
//! The binary also hosts the internal entry points the Python build ran as
//! `<slp> -m <module>` (project agents, the stdio MCP system server, the model
//! host, the benchmark) and the hidden `__*` subcommands the Rust spawn sites
//! use for the same processes.  Both are recognised on the raw argv before
//! clap parsing, as `main.py` did.

use std::path::Path;
use std::time::Duration;

use clap::{Args, CommandFactory, Parser, Subcommand};
use serde_json::Value;

use smartloop::config::AppSettings;
use smartloop::constants::SLP_PRIMARY;
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

// ---------------------------------------------------------------------------
// Console output (the subset of rich markup main.py used)
// ---------------------------------------------------------------------------

/// Wrap `text` in an SGR sequence when stdout takes colour.
fn styled(text: &str, sgr: &str) -> String {
    if console::colors_enabled() {
        format!("\x1b[{sgr}m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

/// `[#rrggbb]text[/#rrggbb]`
fn hex_color(text: &str, hex: &str) -> String {
    let hex = hex.trim_start_matches('#');
    let channel = |i: usize| u8::from_str_radix(hex.get(i..i + 2).unwrap_or("ff"), 16).unwrap_or(255);
    styled(text, &format!("38;2;{};{};{}", channel(0), channel(2), channel(4)))
}

fn primary(text: &str) -> String {
    hex_color(text, SLP_PRIMARY)
}

fn dim(text: &str) -> String {
    styled(text, "2")
}

fn yellow(text: &str) -> String {
    styled(text, "33")
}

/// Python `str(value)` for a JSON value (`True`, `None`, ...).
fn py_display(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Python truthiness of a JSON value.
fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// `dict.get(key, default)` rendered with `str()`.
fn get_display(map: &Value, key: &str, default: &str) -> String {
    map.get(key).map(py_display).unwrap_or_else(|| default.to_string())
}

/// Human-readable RSS, or a dash when the process could not be read.
fn format_bytes(value: Option<f64>) -> String {
    let Some(value) = value.filter(|v| *v != 0.0 && !v.is_nan()) else {
        return "-".to_string();
    };
    let mut size = value;
    for unit in ["B", "KB", "MB", "GB"] {
        if size < 1024.0 || unit == "GB" {
            return if unit != "GB" { format!("{size:.0} {unit}") } else { format!("{size:.1} {unit}") };
        }
        size /= 1024.0;
    }
    format!("{size:.1} GB")
}

/// The first `n` characters of `s` (Python `s[:n]`).
fn truncate_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Report one process, or the whole tree when running supervised.
fn print_status(status: &Value, host: &str, port: u16) {
    println!("{}", primary(&format!("Agent running at http://{host}:{port}")));
    let pid = status.get("pid").filter(|p| truthy(Some(p))).map(py_display).unwrap_or_else(|| "unknown".into());
    println!("  PID: {pid}");
    println!("  Model loaded: {}", status.get("model_loaded").map(py_display).unwrap_or_else(|| "False".into()));
    println!("  Model name: {}", get_display(status, "model_name", "none"));

    if !truthy(status.get("supervised")) {
        return;
    }

    let empty = Value::Object(Default::default());
    let model_host = status.get("model_host").filter(|m| truthy(Some(m))).unwrap_or(&empty);
    println!();
    println!("{}", primary("Model host"));
    if truthy(model_host.get("address")) {
        println!("  {} (pid {})", get_display(model_host, "address", ""), get_display(model_host, "pid", "?"));
    } else {
        println!("  {}", dim("not running — agents load their own models"));
    }

    let agents: Vec<Value> = match status.get("agents") {
        Some(Value::Array(rows)) => rows.clone(),
        _ => Vec::new(),
    };
    println!();
    println!("{}", primary("Project agents"));
    if agents.is_empty() {
        println!("  {}", dim("none running — started on first request"));
        return;
    }
    println!("  {:<28}{:>8}{:>7}{:>10}{:>8}", "project", "pid", "port", "rss", "idle");
    for agent in &agents {
        let name = [agent.get("name"), agent.get("project_id")]
            .into_iter()
            .find(|v| truthy(*v))
            .flatten()
            .map(py_display)
            .unwrap_or_else(|| "?".to_string());
        let idle = match agent.get("idle_seconds") {
            Some(Value::Null) | None => "-".to_string(),
            Some(v) => format!("{:.0}s", v.as_f64().unwrap_or(0.0)),
        };
        println!(
            "  {:<28}{:>8}{:>7}{:>10}{:>8}",
            truncate_chars(&name, 27),
            get_display(agent, "pid", "?"),
            get_display(agent, "port", "?"),
            format_bytes(agent.get("rss_bytes").and_then(Value::as_f64)),
            idle,
        );
    }
}

/// Route warnings and errors to stderr for the duration of a short-lived
/// command, as Python's last-resort handler did for the CLI (the `smartloop`
/// loggers were set to WARNING).  Scoped to the calling thread, so a server
/// started afterwards installs its own logging unhindered.
fn cli_logging() -> tracing::subscriber::DefaultGuard {
    let subscriber = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(tracing::Level::WARN)
        .without_time()
        .with_target(false)
        .with_level(false)
        .finish();
    tracing::subscriber::set_default(subscriber)
}

/// Create the home directory, then start the server in the foreground.
fn start(settings: &AppSettings, args: &StartArgs, host: &str, port: u16) -> anyhow::Result<()> {
    // Supervised is the default; --no-supervised falls back to one shared
    // process for the whole workspace.
    let supervised = !args.no_supervised;
    std::fs::create_dir_all(Path::new(&settings.home_dir()))?;
    start_server(host, port, false, !args.no_service, supervised)
}

/// Dispatch an `slp agent ...` subcommand.
fn cmd_agent(agent_command: Option<&AgentCommand>, host: &str, port: u16) -> anyhow::Result<()> {
    let settings = AppSettings::new();
    match agent_command {
        Some(AgentCommand::Start(args)) => {
            if is_server_running(host, port) {
                println!("{}", primary(&format!("Agent already running at http://{host}:{port}")));
            } else {
                start(&settings, args, host, port)?;
            }
        }
        Some(AgentCommand::Stop) => {
            let _guard = cli_logging();
            stop_server();
        }
        Some(AgentCommand::Status) => {
            let status = get_status(host, port);
            if truthy(status.get("running")) {
                print_status(&status, host, port);
            } else {
                println!("{}", dim("Agent not running"));
            }
        }
        Some(AgentCommand::Restart(args)) => {
            {
                let _guard = cli_logging();
                stop_server();
            }
            std::thread::sleep(Duration::from_secs(1));
            start(&settings, args, host, port)?;
        }
        None => {
            println!("{}", yellow("Usage: slp agent {start|stop|status|restart}"));
        }
    }
    Ok(())
}

/// Build a multi-threaded tokio runtime for an async entry point.
fn runtime() -> anyhow::Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread().enable_all().build()?)
}

/// Run an internal module entry point, Python's `slp -m <module> [args...]`
/// (internal stdio MCP servers, the model host, agent children and workers
/// are spawned as `<current_exe> -m <module> ...`).  Checked on the raw argv
/// before clap parsing, as `main.py` did.
fn run_module(module: &str, args: Vec<String>) -> anyhow::Result<()> {
    use smartloop::api::project_agent::{self, ProjectAgentArgs};
    use smartloop::model_host::bench::{self, BenchArgs};
    use smartloop::model_host::entry::{self, ModelHostArgs};

    // `sys.argv = [argv0, *rest]`: the module's own parser sees its name as
    // the program and the remaining arguments.
    let argv = std::iter::once(module.to_string()).chain(args);
    match module {
        "smartloop.api.project_agent" => project_agent::main(ProjectAgentArgs::parse_from(argv)),
        "smartloop.api.mcp.system_server" => runtime()?.block_on(smartloop::api::mcp::system_server::run_stdio()),
        "smartloop.api.mcp.skill_server" => runtime()?.block_on(smartloop::api::mcp::skill_server::run_stdio()),
        "smartloop.model_host" | "smartloop.model_host.__main__" => entry::run(ModelHostArgs::parse_from(argv)),
        "smartloop.model_host.bench" => bench::run(BenchArgs::parse_from(argv)),
        "smartloop.api.main" => {
            smartloop::api::run_server();
            Ok(())
        }
        other => anyhow::bail!("No module named {other}"),
    }
}

/// The hidden `__*` subcommand a Rust spawn site used, mapped to the module
/// it stands for.
fn hidden_subcommand_module(name: &str) -> Option<&'static str> {
    match name {
        "__project-agent" => Some("smartloop.api.project_agent"),
        "__system-server" => Some("smartloop.api.mcp.system_server"),
        "__skill-server" => Some("smartloop.api.mcp.skill_server"),
        "__model-host" => Some("smartloop.model_host"),
        "__model-host-bench" => Some("smartloop.model_host.bench"),
        "__api-server" => Some("smartloop.api.main"),
        _ => None,
    }
}

fn main() -> anyhow::Result<()> {
    // Platform environment fixes (console code page, bundled Vulkan ICD) must
    // run before anything else, while the process is still single-threaded.
    smartloop::utils::runtime_env::apply_runtime_env();
    smartloop::init_process_env();
    // main.py: `logging.getLogger("httpx"/"httpcore").setLevel(WARNING)`.  The
    // Rust HTTP/gRPC stack logs through these crates instead; at DEBUG they
    // would bury the orchestration trace under connection-pool chatter.
    for name in ["reqwest", "hyper", "hyper_util", "h2", "rustls", "tonic", "tower", "want", "mio"] {
        smartloop::utils::logging_setup::get_logger(name).set_level("WARNING");
    }

    let argv: Vec<String> = std::env::args().collect();

    // In a packaged build the `slp` binary is the only executable there is,
    // so internal processes are spawned as `<slp> -m <module>`; honour that
    // the way `python -m` would.
    if argv.len() >= 3 && argv[1] == "-m" {
        return run_module(&argv[2], argv[3..].to_vec());
    }
    if argv.len() >= 2 {
        if argv[1] == smartloop::llms::llm_factory::PROBE_SUBCOMMAND {
            let config = argv.get(2).cloned().unwrap_or_default();
            std::process::exit(smartloop::llms::llm_factory::LlmFactory::run_probe(&config));
        }
        if let Some(module) = hidden_subcommand_module(&argv[1]) {
            return run_module(module, argv[2..].to_vec());
        }
    }

    let cli = Cli::parse();

    match &cli.command {
        Some(Command::Agent { agent_command }) => {
            let start_args = match agent_command {
                Some(AgentCommand::Start(a)) | Some(AgentCommand::Restart(a)) => Some(a),
                _ => None,
            };
            // Resolve configuration
            let host = start_args
                .and_then(|a| a.host.clone())
                .filter(|h| !h.is_empty())
                .unwrap_or_else(|| std::env::var("SLP_API_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()));
            let port = match start_args.and_then(|a| a.port).filter(|p| *p != 0) {
                Some(port) => port,
                None => AppSettings::new().api_port,
            };
            cmd_agent(agent_command.as_ref(), &host, port)
        }
        None => {
            Cli::command().print_help()?;
            println!();
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_bytes_matches_python() {
        assert_eq!(format_bytes(None), "-");
        assert_eq!(format_bytes(Some(0.0)), "-");
        assert_eq!(format_bytes(Some(512.0)), "512 B");
        assert_eq!(format_bytes(Some(2048.0)), "2 KB");
        assert_eq!(format_bytes(Some(300.0 * 1024.0 * 1024.0)), "300 MB");
        assert_eq!(format_bytes(Some(3.0 * 1024.0 * 1024.0 * 1024.0)), "3.0 GB");
    }

    #[test]
    fn hidden_subcommands_map_to_modules() {
        assert_eq!(hidden_subcommand_module("__project-agent"), Some("smartloop.api.project_agent"));
        assert_eq!(hidden_subcommand_module("agent"), None);
    }
}
