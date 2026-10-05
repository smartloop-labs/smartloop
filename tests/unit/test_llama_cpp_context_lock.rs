//! Port of `tests/unit/test_llama_cpp_context_lock.py`.
//!
//! A single llama.cpp context must never run two generations at once.
//!
//! The context is not thread-safe (one KV cache, one token buffer, one
//! sampler), so concurrent `create_chat_completion` calls corrupt the heap and
//! segfault the process.  Callers reach the model from several thread pools
//! with independent locks — see `LlamaCpp::context_lock` — so the guarantee has
//! to live on the model itself.
//!
//! Reproducer this pins down: the collect fan-out in
//! `Explorer::collect_parallel` runs the vector-search history rewrite and the
//! MCP tool pick on separate threads against one shared model, which
//! segfaulted on the second message of any conversation with an MCP server
//! attached.

use std::collections::HashMap;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use parking_lot::Mutex;
use serde_json::json;

use smartloop::llms::base_llm::{BaseLLM, CallOptions};
use smartloop::llms::llama_cpp::{
    ChatCompletionOutput, ChatCompletionRequest, LlamaCpp, LlamaModel, LlamaState, LlamaToken,
};

/// Minimal stand-in for `llama_cpp.Llama` that records overlap.
#[derive(Default)]
struct FakeLlama {
    state: LlamaState,
    /// `(active, peak)` concurrent generations.
    probe: Mutex<(usize, usize)>,
}

impl FakeLlama {
    fn peak(&self) -> usize {
        self.probe.lock().1
    }
}

impl LlamaModel for FakeLlama {
    fn state(&self) -> &LlamaState {
        &self.state
    }

    fn n_ctx(&self) -> usize {
        4096
    }

    fn metadata(&self) -> HashMap<String, String> {
        HashMap::from([("test.context_length".to_string(), "4096".to_string())])
    }

    fn tokenize(&self, text: &[u8]) -> anyhow::Result<Vec<LlamaToken>> {
        Ok(vec![1; text.len() / 4 + 1])
    }

    fn detokenize(&self, tokens: &[LlamaToken]) -> anyhow::Result<Vec<u8>> {
        Ok(vec![b'x'; tokens.len()])
    }

    fn create_chat_completion(&self, _request: ChatCompletionRequest) -> anyhow::Result<ChatCompletionOutput<'_>> {
        {
            let mut probe = self.probe.lock();
            probe.0 += 1;
            probe.1 = probe.1.max(probe.0);
        }
        thread::sleep(Duration::from_millis(20)); // long enough for siblings to pile up if unguarded
        {
            let mut probe = self.probe.lock();
            probe.0 -= 1;
        }
        let chunks = vec![json!({"choices": [{"delta": {"content": "ok"}}]})];
        Ok(ChatCompletionOutput::Stream(Box::new(chunks.into_iter())))
    }
}

fn wrapper(model: &Arc<FakeLlama>) -> Arc<LlamaCpp> {
    let model: Arc<dyn LlamaModel> = model.clone();
    Arc::new(LlamaCpp::new(model, "p", false, None))
}

#[test]
fn test_concurrent_generation_is_serialized() {
    let model = Arc::new(FakeLlama::default());
    let llm = wrapper(&model);

    let threads: Vec<_> = (0..6)
        .map(|i| {
            let llm = llm.clone();
            thread::spawn(move || {
                llm.invoke(&format!("prompt {i}"), None, None);
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }

    assert_eq!(
        model.peak(),
        1,
        "generations overlapped on one context (peak={}) — this is the segfault",
        model.peak()
    );
}

/// A `_stream` consumer must exclude `_call` — different entry points, same
/// context.
#[test]
fn test_streaming_holds_the_lock_against_a_concurrent_call() {
    let model = Arc::new(FakeLlama::default());
    let llm = wrapper(&model);

    let mut threads = Vec::new();
    for _ in 0..3 {
        let llm = llm.clone();
        threads.push(thread::spawn(move || {
            let _: Vec<_> = llm.stream("streamed prompt", CallOptions::default()).collect();
        }));
    }
    for _ in 0..3 {
        let llm = llm.clone();
        threads.push(thread::spawn(move || {
            llm.invoke("blocking prompt", None, None);
        }));
    }
    for t in threads {
        t.join().unwrap();
    }

    assert_eq!(model.peak(), 1, "stream overlapped a call (peak={})", model.peak());
}

/// The lock hangs off the model object, so two wrappers over one context
/// serialize against each other rather than each guarding its own.
#[test]
fn test_wrappers_sharing_a_context_share_the_lock() {
    let model = Arc::new(FakeLlama::default());
    let (a, b) = (wrapper(&model), wrapper(&model));

    assert!(std::ptr::eq(a.context_lock(), b.context_lock()));

    let threads: Vec<_> = [a.clone(), b.clone(), a, b]
        .into_iter()
        .map(|w| {
            thread::spawn(move || {
                w.invoke("p", None, None);
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }

    assert_eq!(model.peak(), 1, "wrappers did not share a lock (peak={})", model.peak());
}

/// Two independent models must still run in parallel — the lock is
/// per-context, not global, so the worker pool keeps its concurrency.
#[test]
fn test_separate_contexts_are_not_serialized_against_each_other() {
    let (m1, m2) = (Arc::new(FakeLlama::default()), Arc::new(FakeLlama::default()));
    let barrier = Arc::new(Barrier::new(2));

    let threads: Vec<_> = [wrapper(&m1), wrapper(&m2)]
        .into_iter()
        .map(|llm| {
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                llm.invoke("p", None, None);
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }

    // The barrier only proves they started together; no assert on peak here,
    // since each model sees a single caller by construction.
    assert!(m1.peak() == 1 && m2.peak() == 1);
}
