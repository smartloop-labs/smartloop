//! Port of `smartloop/model_host/__main__.py`.
//!
//! Run the device owner.  Spawned by the supervisor (which re-executes the
//! `slp` binary with the hidden model-host subcommand), or by hand when
//! debugging.

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
    let _ = args;
    todo!()
}

/// Entry point: parse process arguments and [`run`].
pub fn main() -> anyhow::Result<()> {
    todo!()
}
