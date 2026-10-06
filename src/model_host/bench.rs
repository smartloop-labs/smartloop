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

use std::time::Instant;

use clap::Parser;

use crate::model_host::client::{InProcessModelClient, ModelClient, RemoteModelClient};

/// Module name the binary is re-executed with (`slp -m smartloop.model_host.bench`).
pub const MODULE: &str = "smartloop.model_host.bench";

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
    let started = Instant::now();
    let mut first: Option<f64> = None;
    let mut chunks = 0usize;
    let mut chars = 0usize;
    for text in client.stream(prompt, None, None, None, None)? {
        let text = text?;
        if first.is_none() {
            first = Some(started.elapsed().as_secs_f64());
        }
        chunks += 1;
        chars += text.chars().count();
    }
    let total = started.elapsed().as_secs_f64();
    Ok(Timing::new(first.unwrap_or(total), total, chunks, chars))
}

/// `statistics.median`: the middle value, or the mean of the two middle
/// values for an even count.
fn median(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut values: Vec<f64> = values.into_iter().collect();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

/// Stream `prompt` `runs` times and summarise (medians).
///
/// One warmup pass first: the first generation after a load pays for the
/// prompt-cache miss and, for a remote client, for opening the connection.
/// Including it would flatter or damn the transport for something that
/// happens once.
pub fn benchmark(
    client: &dyn ModelClient,
    prompt: &str,
    runs: usize,
    warmup: bool,
) -> anyhow::Result<BenchResult> {
    if warmup {
        time_one_stream(client, prompt)?;
    }

    let timings = (0..runs)
        .map(|_| time_one_stream(client, prompt))
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(BenchResult {
        client: client.type_name(),
        runs,
        first_chunk_s: median(timings.iter().map(|t| t.first_chunk_s)),
        total_s: median(timings.iter().map(|t| t.total_s)),
        chars: median(timings.iter().map(|t| t.chars as f64)),
        chunks: median(timings.iter().map(|t| t.chunks as f64)),
        chars_per_second: median(timings.iter().map(|t| t.chars_per_second())),
    })
}

/// A fixed-width table, and the verdict the decision actually turns on.
pub fn format_report(results: &[BenchResult]) -> String {
    let header = format!(
        "{:<24}{:>13}{:>10}{:>9}{:>9}{:>11}",
        "client", "first chunk", "total", "chunks", "chars", "chars/s"
    );
    let mut lines = vec![header.clone(), "-".repeat(header.chars().count())];
    for row in results {
        lines.push(format!(
            "{:<24}{:>12.3}s{:>9.3}s{:>9.0}{:>9.0}{:>11.1}",
            row.client, row.first_chunk_s, row.total_s, row.chunks, row.chars, row.chars_per_second
        ));
    }

    if let [base, other] = results {
        lines.push(String::new());
        lines.push(format!(
            "{} vs {}: first chunk {}, throughput {}",
            other.client,
            base.client,
            ratio(other.first_chunk_s, base.first_chunk_s),
            ratio(base.chars_per_second, other.chars_per_second),
        ));
    }
    lines.join("\n")
}

/// `1.04x slower` / `0.98x faster` — signed the way a reader expects.
fn ratio(numerator: f64, denominator: f64) -> String {
    if denominator == 0.0 {
        return "n/a".to_string();
    }
    let factor = numerator / denominator;
    format!("{:.2}x {}", factor, if factor >= 1.0 { "slower" } else { "faster" })
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
    let mut results: Vec<BenchResult> = Vec::new();

    if !args.skip_in_process {
        let service = crate::api::services::get_model_service();
        service.load_model()?;
        let client = InProcessModelClient::new(service);
        results.push(benchmark(&client, &args.prompt, args.runs, true)?);
    }

    if let Some(address) = args.address.as_deref() {
        let remote = RemoteModelClient::new(address);
        let measured = benchmark(&remote, &args.prompt, args.runs, true);
        remote.close();
        results.push(measured?);
    }

    println!("{}", format_report(&results));
    Ok(())
}

/// Entry point: parse process arguments and [`run`].
pub fn main() -> anyhow::Result<()> {
    run(BenchArgs::parse_from(crate::model_host::entry::module_argv(MODULE)))
}
