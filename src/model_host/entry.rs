//! Port of `smartloop/model_host/__main__.py`.
//!
//! Run the device owner.  Spawned by the supervisor (which re-executes the
//! `slp` binary as `slp -m smartloop.model_host ...`), or by hand when
//! debugging.

use clap::Parser;

use crate::model_host::host::serve;

/// Module name the binary is re-executed with (`slp -m smartloop.model_host`).
pub const MODULE: &str = "smartloop.model_host";

/// Command-line arguments of `smartloop.model_host`.
#[derive(Debug, Clone, clap::Parser)]
#[command(
    name = "smartloop.model_host",
    about = "Own the device and serve generation to project agents"
)]
pub struct ModelHostArgs {
    /// unix socket path, or host:port (default: $SLP_MODEL_HOST, else
    /// <home>/model-host.sock)
    #[arg(long)]
    pub address: Option<String>,
    #[arg(long = "log-level", default_value = "INFO")]
    pub log_level: String,
}

/// Configure logging at `args.log_level` and serve until stopped.
pub fn run(args: ModelHostArgs) -> anyhow::Result<()> {
    init_logging(&args.log_level);
    crate::utils::timing::set_process("host");
    serve(args.address.as_deref(), None)?;
    Ok(())
}

/// Entry point: parse process arguments and [`run`].
///
/// Accepts both `slp -m smartloop.model_host [args]` (the re-exec form, the
/// way `python -m` was honoured) and plain `[args]`.
pub fn main() -> anyhow::Result<()> {
    run(ModelHostArgs::parse_from(module_argv(MODULE)))
}

/// The process arguments with a leading `-m <module>` pair removed, so clap
/// sees `<prog> [args...]` whichever form launched us.
pub(crate) fn module_argv(module: &str) -> Vec<String> {
    let mut argv: Vec<String> = std::env::args().collect();
    if argv.len() >= 3 && argv[1] == "-m" && argv[2] == module {
        argv.drain(1..3);
    }
    if let Some(prog) = argv.first_mut() {
        *prog = module.to_string();
    }
    argv
}

/// `logging.basicConfig(level=..., format="%(asctime)s %(levelname)s
/// %(name)s: %(message)s")`.  A no-op when a subscriber is already installed.
pub(crate) fn init_logging(level: &str) {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let filter = match level.trim().to_ascii_uppercase().as_str() {
        "DEBUG" | "NOTSET" => tracing::level_filters::LevelFilter::DEBUG,
        "WARNING" | "WARN" => tracing::level_filters::LevelFilter::WARN,
        "ERROR" | "CRITICAL" | "FATAL" => tracing::level_filters::LevelFilter::ERROR,
        "TRACE" => tracing::level_filters::LevelFilter::TRACE,
        _ => tracing::level_filters::LevelFilter::INFO,
    };
    // Dependencies never log below INFO; only smartloop follows `level`.
    let targets = tracing_subscriber::filter::Targets::new()
        .with_default(filter.min(tracing::level_filters::LevelFilter::INFO))
        .with_target("smartloop", filter);
    let _ = tracing_subscriber::fmt()
        .with_max_level(filter)
        .with_writer(std::io::stderr)
        .event_format(BasicFormat)
        .finish()
        .with(targets)
        .try_init();
}

/// Python's `%(asctime)s %(levelname)s %(name)s: %(message)s` line format.
struct BasicFormat;

impl<S, N> tracing_subscriber::fmt::FormatEvent<S, N> for BasicFormat
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
    N: for<'a> tracing_subscriber::fmt::FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &tracing_subscriber::fmt::FmtContext<'_, S, N>,
        mut writer: tracing_subscriber::fmt::format::Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        let metadata = event.metadata();
        let level = match *metadata.level() {
            tracing::Level::ERROR => "ERROR",
            tracing::Level::WARN => "WARNING",
            tracing::Level::INFO => "INFO",
            tracing::Level::DEBUG | tracing::Level::TRACE => "DEBUG",
        };
        let now = chrono::Local::now();
        write!(
            writer,
            "{} {} {}: ",
            now.format("%Y-%m-%d %H:%M:%S,%3f"),
            level,
            metadata.target()
        )?;
        ctx.field_format().format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}
