//! A large or capability model gets the window it declares, as far as memory
//! allows, instead of the flat `llm_large_n_ctx_cap`.
//!
//! Capped at 16384, a 27B or 30B model declaring 128K+ had a prompt carrying a
//! codebase cut to its head and tail, with 512 tokens left over: a reasoning
//! model spent them inside `<think>` and the reply came back empty.

use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;

use serial_test::serial;

use smartloop::api::services::model_service::{ModelServiceHooks, widened_ctx_cap};
use smartloop::llms::base_llm::CallOptions;
use smartloop::llms::llama_cpp::{
    ChatCompletionOutput, ChatCompletionRequest, LlamaCpp, LlamaModel, LlamaState, LlamaToken,
};

use crate::shared::model_service_a::bare_service;

// ------------------------------------------------------------ the window

#[test]
fn test_a_roomy_machine_gets_the_declared_window() {
    // 60GB free of 64GB, 16GB of weights, 4GB reserve: room for ~300K tokens.
    assert_eq!(widened_ctx_cap(16384, Some(131072), 60.0, 64.0, 16.0, 4.0), 131072);
}

#[test]
fn test_never_wider_than_an_auto_sized_load() {
    // 262144 declared, but a 64GB machine auto-sizes to at most 131072.
    assert_eq!(widened_ctx_cap(16384, Some(262144), 60.0, 64.0, 16.0, 4.0), 131072);
}

#[test]
fn test_memory_bounds_the_window() {
    // 10.7GB spare at 0.5GB per 4096 tokens: 21 blocks.
    assert_eq!(widened_ctx_cap(16384, Some(131072), 31.4, 64.0, 16.0, 4.0), 21 * 4096);
}

#[test]
fn test_the_cap_is_a_floor() {
    assert_eq!(widened_ctx_cap(16384, Some(131072), 18.0, 64.0, 16.0, 4.0), 16384);
}

#[test]
fn test_an_undeclared_window_keeps_the_cap() {
    assert_eq!(widened_ctx_cap(16384, None, 60.0, 64.0, 16.0, 4.0), 16384);
}

/// A 1GB model with 40GB of 64GB free, declaring `declared`.
fn cap_for(base_cap: i64, declared: Option<u32>) -> i64 {
    let dir = tempfile::tempdir().expect("tempdir");
    let gguf = dir.path().join("m.gguf");
    File::create(&gguf).and_then(|f| f.set_len(1024u64.pow(3))).expect("sparse gguf");
    let svc = bare_service();
    svc.set_hooks(ModelServiceHooks {
        device_free_total_gb: Some(Arc::new(|| Some((40.0, 64.0)))),
        ..Default::default()
    });
    svc.large_ctx_cap(&gguf.to_string_lossy(), base_cap, declared, Some(4.0), "m")
}

#[test]
#[serial]
fn test_the_service_widens_to_the_declared_window() {
    assert_eq!(cap_for(16384, Some(65536)), 65536);
}

#[test]
#[serial]
fn test_uncapped_stays_uncapped() {
    assert_eq!(cap_for(0, Some(65536)), 0);
}

// ------------------------------------------------------ the output reserve

struct Fake {
    state: LlamaState,
    n_ctx: usize,
}

impl LlamaModel for Fake {
    fn state(&self) -> &LlamaState {
        &self.state
    }

    fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    fn tokenize(&self, text: &[u8]) -> anyhow::Result<Vec<LlamaToken>> {
        Ok((0..text.len() as i32).map(|t| t as LlamaToken).collect())
    }

    fn detokenize(&self, _tokens: &[LlamaToken]) -> anyhow::Result<Vec<u8>> {
        Ok(Vec::new())
    }

    fn create_chat_completion(&self, _request: ChatCompletionRequest) -> anyhow::Result<ChatCompletionOutput<'_>> {
        anyhow::bail!("not a real model")
    }
}

fn llm(flag: Option<bool>, n_ctx: usize) -> LlamaCpp {
    let state = LlamaState::default();
    state.set_thinking(flag);
    LlamaCpp::new(Arc::new(Fake { state, n_ctx }), "", false, None)
}

fn reasoning() -> CallOptions {
    CallOptions { thinking: Some(true), ..Default::default() }
}

#[test]
fn test_a_reasoning_call_keeps_room_for_its_thought_and_answer() {
    assert_eq!(llm(Some(false), 32768).output_reserve(&reasoning()), LlamaCpp::THINKING_OUTPUT_RESERVE);
}

#[test]
fn test_the_reasoning_reserve_is_bounded_by_the_window() {
    assert_eq!(llm(Some(false), 8192).output_reserve(&reasoning()), 2048);
}

#[test]
fn test_a_plain_call_keeps_the_small_reserve() {
    assert_eq!(llm(Some(true), 32768).output_reserve(&CallOptions::default()), LlamaCpp::MIN_OUTPUT_RESERVE);
}

#[test]
fn test_a_model_without_a_thinking_switch_keeps_the_small_reserve() {
    assert_eq!(llm(None, 32768).output_reserve(&reasoning()), LlamaCpp::MIN_OUTPUT_RESERVE);
}
