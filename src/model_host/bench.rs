//! Port of `smartloop/model_host/bench.py`.
//!
//! Measure a client against the in-process path before switching the default.
//!
//! The transport sits on the inference hot path, and a per-message overhead
//! that costs more than the token it delivers would show up as a slower
//! answer, not as an error.  So the default does not move on argument — it
//! moves on a number from here.
//!
//! Reported per client: wall-clock to the first chunk, total wall-clock,
//! chunks, characters, and characters per second.

use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use crate::model_host::client::{InProcessModelClient, ModelClient, RemoteModelClient};

/// Prompt streamed when none is given.
pub const DEFAULT_PROMPT: &str = "<system>You are a helpful assistant.</system>\n\
<user>Explain what a vector index is, in one paragraph.</user>";

/// One streamed generation, timed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Timing {
    pub first_chunk_s: f64,
    pub total_s: f64,
    pub chunks: usize,
    pub chars: usize,
}

impl Timing {
    /// Build a timing.
    pub fn new(first_chunk_s: f64, total_s: f64, chunks: usize, chars: usize) -> Self {
        Self { first_chunk_s, total_s, chunks, chars }
    }

    /// Characters per second over the whole stream (0 when instantaneous).
    pub fn chars_per_second(&self) -> f64 {
        if self.total_s != 0.0 { self.chars as f64 / self.total_s } else { 0.0 }
    }
}

/// Median summary of several runs (the Python `dict` row).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct BenchResult {
    pub client: String,
    pub runs: usize,
    pub first_chunk_s: f64,
    pub total_s: f64,
    pub chars: f64,
    pub chunks: f64,
    pub chars_per_second: f64,
}

/// Stream `prompt` once and time it.
pub fn time_one_stream(client: &dyn ModelClient, prompt: &str) -> anyhow::Result<Timing> {
    let _ = (client, prompt);
    todo!()
}

/// Stream `prompt` `runs` times and summarise (medians).
///
/// One warmup pass first: the first generation after a load pays for the
/// prompt-cache miss and, for a remote client, for opening the connection.
pub fn benchmark(
    client: &dyn ModelClient,
    prompt: &str,
    runs: usize,
    warmup: bool,
) -> anyhow::Result<BenchResult> {
    let _ = (client, prompt, runs, warmup);
    todo!()
}

/// A fixed-width table, and the verdict the decision actually turns on.
pub fn format_report(results: &[BenchResult]) -> String {
    let _ = results;
    todo!()
}

/// `1.04x slower` / `0.98x faster` — signed the way a reader expects.
#[allow(dead_code)]
fn ratio(numerator: f64, denominator: f64) -> String {
    let _ = (numerator, denominator);
    todo!()
}

/// Command-line arguments of `smartloop.model_host.bench`.
#[derive(Debug, Clone, clap::Parser)]
#[command(
    name = "smartloop.model_host.bench",
    about = "Compare a model host against the in-process path"
)]
pub struct BenchArgs {
    /// model host to measure; omit to measure only the in-process path
    #[arg(long)]
    pub address: Option<String>,
    #[arg(long, default_value = DEFAULT_PROMPT)]
    pub prompt: String,
    #[arg(long, default_value_t = 3)]
    pub runs: usize,
    /// measure only the host (no local model load)
    #[arg(long = "skip-in-process")]
    pub skip_in_process: bool,
}

/// Run the benchmark and print the report.
pub fn run(args: BenchArgs) -> anyhow::Result<()> {
    let _ = args;
    todo!()
}

/// Entry point: parse process arguments and [`run`].
pub fn main() -> anyhow::Result<()> {
    todo!()
}
