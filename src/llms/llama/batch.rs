//! Batched generation: several prompts through one multi-sequence context.
//!
//! Generating a token reads every weight of the model once, whether the step
//! advances one sequence or several, so on Apple Silicon (bound by memory
//! bandwidth, not compute) a step over four sequences costs little more than a
//! step over one.  The document fan-out sends a hundred-odd independent
//! extractions; running them one at a time left most of each step unused.
//!
//! The batch runs in its own context, created for the call and dropped after
//! it, beside the single-sequence engine every other caller uses.  llama.cpp
//! splits a context's window evenly between its sequences unless the KV cache
//! is unified, which `llama-cpp-2` does not expose; giving the engine more
//! sequences would have shrunk every ordinary chat's window.
//!
//! Prompts are rendered by the same chat handlers as a single completion:
//! [`capture_prompt`] runs the handler with generation diverted, collecting
//! the tokens and stop sequences it would have generated from.
//!
//! The slot count is sized automatically: as many as fit in free memory after
//! a reserve, but no more than still speed up a step on this device, which a
//! short benchmark measures once per loaded model.

use std::cell::RefCell;
use std::time::Instant;

use super::*;
use crate::config::app_settings::AppSettings;

/// Most sequences a batch ever runs, whatever memory allows.
pub const MAX_BATCH_SLOTS: usize = 16;
/// A doubling of slots has to lift step throughput by at least this factor to
/// be worth its memory.
pub const SLOT_GAIN_THRESHOLD: f64 = 1.2;
/// Prompt tokens each benchmark sequence holds before its steps are timed.
const BENCH_PREFILL: usize = 32;
/// Timed steps per slot count in the benchmark.
const BENCH_STEPS: usize = 3;
/// Context per benchmark sequence: the prefill plus every timed step.
const BENCH_SLOT_CTX: usize = 256;
/// Headroom on the KV estimate for the compute buffers that grow with it.
const KV_OVERHEAD: f64 = 1.1;
/// Slots on a device whose free memory cannot be read (discrete GPU VRAM).
const UNKNOWN_DEVICE_SLOTS: usize = 2;

/// One prompt rendered for generation: what a chat handler would have
/// generated from.
#[derive(Debug, Clone)]
pub struct BatchItem {
    pub prompt: Vec<LlamaToken>,
    pub request: ChatCompletionRequest,
    pub stop: Vec<String>,
}

thread_local! {
    static CAPTURE: RefCell<Option<Vec<BatchItem>>> = const { RefCell::new(None) };
}

/// Run `render` (a `create_chat_completion` call) with generation diverted:
/// the handler renders and tokenizes as usual, and the prompt it would have
/// generated from is returned instead of being generated.
///
/// `None` when the handler never reached a text completion (a multimodal
/// prompt, which batches cannot carry).
pub fn capture_prompt<R>(render: impl FnOnce() -> R) -> (R, Option<BatchItem>) {
    CAPTURE.with(|c| *c.borrow_mut() = Some(Vec::new()));
    let out = render();
    let item = CAPTURE.with(|c| c.borrow_mut().take()).and_then(|mut items| items.pop());
    (out, item)
}

/// Record a completion instead of running it, when [`capture_prompt`] is on.
pub(super) fn try_capture(prompt: &[LlamaToken], request: &ChatCompletionRequest, stop: &[String]) -> bool {
    CAPTURE.with(|c| match c.borrow_mut().as_mut() {
        Some(items) => {
            items.push(BatchItem { prompt: prompt.to_vec(), request: request.clone(), stop: stop.to_vec() });
            true
        }
        None => false,
    })
}

/// A sequence in flight.
struct Slot {
    item: usize,
    seq: i32,
    sampler: LlamaSampler,
    n_past: i32,
    last: LlamaToken,
    completion: Vec<LlamaToken>,
    stop: Vec<Vec<u8>>,
    max_tokens: usize,
}

/// What ended a sequence.
enum Step {
    Continue,
    Done(Vec<u8>),
}

impl Llama {
    /// Generate every item, batched across sequences, returning each text in
    /// item order.  `cancelled` is checked between steps; a cancelled batch
    /// returns what each sequence had produced.
    ///
    /// Falls back to one completion at a time when batching cannot help: a
    /// single item, a recurrent or hybrid model (its state is not a KV cache to
    /// split), or a device with room for only one sequence.
    pub fn generate_batch(&self, items: Vec<BatchItem>, cancelled: &dyn Fn() -> bool) -> Vec<anyhow::Result<String>> {
        if items.is_empty() {
            return Vec::new();
        }
        let slot_ctx = items
            .iter()
            .map(|it| it.prompt.len() + it.request.max_tokens.unwrap_or(self.n_ctx) + 1)
            .max()
            .unwrap_or(0)
            .min(self.n_ctx)
            .next_multiple_of(256);
        let slots = if items.len() < 2 || self.model.is_recurrent() || self.model.is_hybrid() {
            1
        } else {
            self.batch_slots(slot_ctx).min(items.len())
        };
        if slots < 2 {
            return items.into_iter().map(|it| self.generate_one(it)).collect();
        }
        match self.run_batch(&items, slots, slot_ctx, cancelled) {
            Ok(texts) => texts.into_iter().map(Ok).collect(),
            Err(e) => {
                tracing::warn!(target: "smartloop.llms.llama", "[batch] failed ({e}) — generating one at a time");
                items.into_iter().map(|it| self.generate_one(it)).collect()
            }
        }
    }

    /// One item through the ordinary single-sequence path.
    fn generate_one(&self, item: BatchItem) -> anyhow::Result<String> {
        let request = ChatCompletionRequest { stream: false, ..item.request };
        match self.create_chat_completion_from_tokens(item.prompt, &request, item.stop)? {
            ChatCompletionOutput::Response(value) => Ok(choice_text(&value)),
            ChatCompletionOutput::Stream(chunks) => Ok(chunks.map(|c| choice_text(&c)).collect()),
        }
    }

    /// How many sequences a batch with `slot_ctx` tokens each should run.
    ///
    /// The smallest of: what fits in free memory after the reserve, what still
    /// speeds a step up on this device (benchmarked once, then cached), and
    /// the `SLP_LLM_BATCH_SLOTS` cap (0 = automatic, 1 = off).
    fn batch_slots(&self, slot_ctx: usize) -> usize {
        let settings = AppSettings::new();
        let cap = match settings.llm_batch_slots {
            n if n > 0 => (n as usize).min(MAX_BATCH_SLOTS),
            _ => MAX_BATCH_SLOTS,
        };
        if cap < 2 {
            return 1;
        }
        let by_memory = self.slots_by_memory(slot_ctx, settings.llm_batch_reserve_gb).min(cap);
        if by_memory < 2 {
            return by_memory;
        }
        let by_speed = *self.batch_knee.lock().get_or_insert_with(|| self.benchmark_slots(by_memory));
        let slots = by_memory.min(by_speed);
        tracing::info!(
            target: "smartloop.llms.llama",
            "[batch] {slots} slot(s) at n_ctx={slot_ctx} each (memory allows {by_memory}, speedup stops past {by_speed})"
        );
        slots
    }

    /// KV-cache bytes one token costs (f16 keys and values, every layer).
    fn kv_bytes_per_token(&self) -> f64 {
        let n_layer = self.model.n_layer() as f64;
        let n_head = self.model.n_head().max(1) as f64;
        let n_head_kv = self.model.n_head_kv().max(1) as f64;
        let arch = self.metadata.get("general.architecture").cloned().unwrap_or_default();
        let dim = |key: &str| self.metadata.get(&format!("{arch}.attention.{key}")).and_then(|v| v.parse::<f64>().ok());
        let fallback = self.model.n_embd() as f64 / n_head;
        let k = dim("key_length").unwrap_or(fallback);
        let v = dim("value_length").unwrap_or(fallback);
        n_layer * n_head_kv * (k + v) * 2.0
    }

    /// Sequences of `slot_ctx` tokens that fit in available memory after
    /// `reserve_gb`.  Available counts what macOS would reclaim from caches
    /// (its free pages alone read 1–2GB on a busy 24GB machine), so the
    /// reserve is what keeps a batch off the edge where processes get killed.
    fn slots_by_memory(&self, slot_ctx: usize, reserve_gb: f64) -> usize {
        if !crate::utils::device_utils::is_mps() {
            return UNKNOWN_DEVICE_SLOTS;
        }
        let (free_gb, _) = crate::utils::device_utils::available_memory_gb();
        let slot_gb = self.kv_bytes_per_token() * slot_ctx as f64 * KV_OVERHEAD / 1024f64.powi(3);
        if slot_gb <= 0.0 {
            return 1;
        }
        (((free_gb - reserve_gb) / slot_gb).floor().max(1.0)) as usize
    }

    /// The largest slot count (1, 2, 4, ... up to `max`) whose step
    /// throughput still beats half as many slots by [`SLOT_GAIN_THRESHOLD`].
    fn benchmark_slots(&self, max: usize) -> usize {
        match self.measure_slots(max) {
            Ok((knee, rates)) => {
                tracing::info!(target: "smartloop.llms.llama", "[batch] step throughput by slots: {rates} → {knee}");
                knee
            }
            Err(e) => {
                tracing::warn!(target: "smartloop.llms.llama", "[batch] benchmark failed ({e}) — using 2 slots");
                2.min(max)
            }
        }
    }

    fn measure_slots(&self, max: usize) -> anyhow::Result<(usize, String)> {
        let mut ctx = self.batch_context(max, BENCH_SLOT_CTX)?;
        let probe = self.tokenize_ext(b"The quick brown fox jumps over the lazy dog. ", false, false);
        let prefill: Vec<LlamaToken> = probe.iter().copied().cycle().take(BENCH_PREFILL).collect();
        let mut batch = LlamaBatch::new(BENCH_PREFILL * max, 1);
        for seq in 0..max as i32 {
            for (i, &t) in prefill.iter().enumerate() {
                batch.add(NativeToken(t), i as i32, &[seq], false)?;
            }
        }
        ctx.decode(&mut batch).map_err(|e| anyhow::anyhow!("llama_decode returned {e}"))?;

        let mut n_past = vec![BENCH_PREFILL as i32; max];
        let mut step = |ctx: &mut LlamaContext<'_>, k: usize| -> anyhow::Result<()> {
            let mut batch = LlamaBatch::new(k, 1);
            for seq in 0..k {
                batch.add(NativeToken(prefill[0]), n_past[seq], &[seq as i32], true)?;
                n_past[seq] += 1;
            }
            ctx.decode(&mut batch).map_err(|e| anyhow::anyhow!("llama_decode returned {e}"))
        };

        let mut knee = 1;
        let mut best = 0.0f64;
        let mut rates = Vec::new();
        let mut k = 1;
        while k <= max {
            step(&mut ctx, k)?;
            let t0 = Instant::now();
            for _ in 0..BENCH_STEPS {
                step(&mut ctx, k)?;
            }
            let rate = (k * BENCH_STEPS) as f64 / t0.elapsed().as_secs_f64().max(1e-9);
            rates.push(format!("{k}:{rate:.0} tok/s"));
            if k == 1 || rate >= best * SLOT_GAIN_THRESHOLD {
                knee = k;
                best = rate;
            } else {
                break;
            }
            k *= 2;
        }
        Ok((knee, rates.join(" ")))
    }

    /// A context for `slots` sequences of `slot_ctx` tokens each, over this
    /// model's weights.
    fn batch_context(&self, slots: usize, slot_ctx: usize) -> anyhow::Result<LlamaContext<'static>> {
        // SAFETY: the model is boxed (stable address) and owned by `self`;
        // every batch context is dropped before this call returns.
        let model: &'static NativeModel = unsafe { &*(self.model.as_ref() as *const NativeModel) };
        let n_ctx = (slots * slot_ctx) as u32;
        let n_batch = self.n_batch.max(slots) as u32;
        let n_threads_batch = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_batch)
            .with_n_ubatch(n_batch.min(512))
            .with_n_seq_max(slots as u32)
            .with_n_threads(self.params.n_threads.max(1) as i32)
            .with_n_threads_batch(n_threads_batch as i32);
        model
            .new_context(llama_backend(), params)
            .map_err(|e| anyhow::anyhow!("Failed to create batch llama_context ({e})"))
    }

    /// Continuous batching: `slots` sequences run together, and a sequence
    /// that finishes hands its slot to the next waiting item.
    fn run_batch(
        &self,
        items: &[BatchItem],
        slots: usize,
        slot_ctx: usize,
        cancelled: &dyn Fn() -> bool,
    ) -> anyhow::Result<Vec<String>> {
        let started = Instant::now();
        let mut ctx = self.batch_context(slots, slot_ctx)?;
        let mut texts: Vec<Option<String>> = vec![None; items.len()];
        let mut waiting = 0..items.len();
        let mut active: Vec<Slot> = Vec::new();
        let mut free: Vec<i32> = (0..slots as i32).rev().collect();
        let mut generated = 0usize;
        let mut seed = LLAMA_DEFAULT_SEED;
        let mut prefill_secs = 0.0f64;
        let mut steps = 0usize;

        loop {
            // Admit waiting items into free slots: prefill each prompt and
            // sample its first token.
            while let Some(&seq) = free.last() {
                let Some(index) = waiting.next() else { break };
                free.pop();
                let item = &items[index];
                seed = item.request.seed.unwrap_or_else(|| derive_seed(seed));
                let mut slot = Slot {
                    item: index,
                    seq,
                    sampler: self.init_sampler(&item.request, seed),
                    n_past: 0,
                    last: 0,
                    completion: Vec::new(),
                    stop: item.stop.iter().filter(|s| !s.is_empty()).map(|s| s.as_bytes().to_vec()).collect(),
                    max_tokens: match item.request.max_tokens {
                        Some(m) if m > 0 => m,
                        _ => slot_ctx,
                    }
                    .min(slot_ctx.saturating_sub(item.prompt.len() + 1)),
                };
                let prefill_start = Instant::now();
                ctx.clear_kv_cache_seq(Some(seq as u32), None, None)?;
                let prompt = if item.prompt.is_empty() { vec![self.token_bos()] } else { item.prompt.clone() };
                let chunks: Vec<&[LlamaToken]> = prompt.chunks(self.n_batch).collect();
                let mut last_index = 0;
                for (c, chunk) in chunks.iter().enumerate() {
                    let mut batch = LlamaBatch::new(chunk.len(), 1);
                    let final_chunk = c + 1 == chunks.len();
                    for (i, &t) in chunk.iter().enumerate() {
                        let want_logits = final_chunk && i + 1 == chunk.len();
                        batch.add(NativeToken(t), slot.n_past, &[seq], want_logits)?;
                        slot.n_past += 1;
                    }
                    ctx.decode(&mut batch).map_err(|e| anyhow::anyhow!("llama_decode returned {e}"))?;
                    last_index = chunk.len() as i32 - 1;
                }
                let token = slot.sampler.sample(&ctx, last_index).0;
                prefill_secs += prefill_start.elapsed().as_secs_f64();
                generated += 1;
                match self.advance(&mut slot, token) {
                    Step::Continue => active.push(slot),
                    Step::Done(text) => {
                        texts[slot.item] = Some(decode_utf8_ignore(&text));
                        free.push(slot.seq);
                    }
                }
            }
            if active.is_empty() {
                break;
            }
            if cancelled() {
                for slot in active.drain(..) {
                    texts[slot.item] = Some(decode_utf8_ignore(&self.detokenize_ext(&slot.completion, false)));
                }
                break;
            }

            // One step for every active sequence.
            let mut batch = LlamaBatch::new(active.len(), 1);
            for slot in &mut active {
                batch.add(NativeToken(slot.last), slot.n_past, &[slot.seq], true)?;
                slot.n_past += 1;
            }
            ctx.decode(&mut batch).map_err(|e| anyhow::anyhow!("llama_decode returned {e}"))?;
            steps += 1;
            let mut still = Vec::with_capacity(active.len());
            for (i, mut slot) in active.drain(..).enumerate() {
                let token = slot.sampler.sample(&ctx, i as i32).0;
                generated += 1;
                match self.advance(&mut slot, token) {
                    Step::Continue => still.push(slot),
                    Step::Done(text) => {
                        texts[slot.item] = Some(decode_utf8_ignore(&text));
                        free.push(slot.seq);
                    }
                }
            }
            active = still;
        }

        let elapsed = started.elapsed().as_secs_f64();
        tracing::info!(
            target: "smartloop.llms.llama",
            "[batch] {} prompt(s) on {slots} slot(s) | {generated} tokens in {elapsed:.1}s (prompts {prefill_secs:.1}s, {steps} steps in {:.1}s, {:.1} tok/s generating)",
            items.len(),
            elapsed - prefill_secs,
            generated as f64 / (elapsed - prefill_secs).max(1e-9)
        );
        Ok(texts.into_iter().map(Option::unwrap_or_default).collect())
    }

    /// Take a sampled token into `slot`; the same end conditions as a single
    /// completion (end of generation, stop sequence, length, a repeating loop).
    fn advance(&self, slot: &mut Slot, token: LlamaToken) -> Step {
        if self.is_eog(token) {
            return Step::Done(self.detokenize_ext(&slot.completion, false));
        }
        slot.completion.push(token);
        slot.last = token;
        let text = self.detokenize_ext(&slot.completion, false);
        if let Some(at) = slot.stop.iter().filter_map(|s| find_bytes(&text, s)).min() {
            return Step::Done(text[..at].to_vec());
        }
        if slot.completion.len() >= slot.max_tokens {
            return Step::Done(text);
        }
        if slot.completion.len() % 16 == 0 {
            if let Some(run) = repetition_run(&slot.completion) {
                let keep = slot.completion.len() - run;
                tracing::warn!(
                    target: "smartloop.llms.llama",
                    "Generation stopped: output repeating ({run} tokens looped after {keep})"
                );
                slot.completion.truncate(keep);
                return Step::Done(self.detokenize_ext(&slot.completion, false));
            }
        }
        Step::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_collects_what_would_have_been_generated() {
        let request = ChatCompletionRequest { max_tokens: Some(64), ..Default::default() };
        let stop = vec!["</s>".to_string()];
        let (captured, item) = capture_prompt(|| try_capture(&[1, 2, 3], &request, &stop));
        assert!(captured);
        let item = item.expect("the captured prompt");
        assert_eq!(item.prompt, vec![1, 2, 3]);
        assert_eq!(item.request.max_tokens, Some(64));
        assert_eq!(item.stop, stop);
    }

    #[test]
    fn outside_a_capture_completions_run_normally() {
        assert!(!try_capture(&[1], &ChatCompletionRequest::default(), &[]));
        // A finished capture does not leak into the next completion.
        let _ = capture_prompt(|| ());
        assert!(!try_capture(&[1], &ChatCompletionRequest::default(), &[]));
    }

    #[test]
    fn a_render_that_never_completes_captures_nothing() {
        let ((), item) = capture_prompt(|| ());
        assert!(item.is_none());
    }
}
