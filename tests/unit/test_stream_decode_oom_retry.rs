//! Port of `tests/unit/test_stream_decode_oom_retry.py`.
//!
//! A decode that runs out of device memory is retried once, after eviction.
//!
//! The answer shares the device with whatever earlier steps left resident, and
//! with the embedding model citation prep loads alongside it.  When llama.cpp
//! cannot find room it returns a non-zero `llama_decode`; freeing the auxiliary
//! models gives it that room, so the stream evicts them and tries again before
//! giving up.  What already reached the client is kept: the retry asks for the
//! rest of the answer rather than starting it over.

use std::sync::Arc;

use parking_lot::Mutex;
use serial_test::serial;

use smartloop::llms::base_llm::CancelEvent;

use crate::shared::api_chat::{chunk, ChunkIter};
use crate::test_stream_opening_buffer::{content, request, run, service};

/// A stream that yields `first` and then fails with a decode OOM on its first
/// call, and yields `second` on every later one.  Returns the prompts it saw.
fn failing_once(
    first: &[&str],
    second: &[&str],
) -> (
    impl Fn(&str, Option<CancelEvent>) -> ChunkIter + Send + Sync + 'static,
    Arc<Mutex<Vec<String>>>,
) {
    let prompts: Arc<Mutex<Vec<String>>> = Arc::default();
    let first: Vec<String> = first.iter().map(|s| s.to_string()).collect();
    let second: Vec<String> = second.iter().map(|s| s.to_string()).collect();
    let seen = prompts.clone();
    let stream_completion = move |prompt: &str, _cancelled: Option<CancelEvent>| -> ChunkIter {
        let call = {
            let mut prompts = seen.lock();
            prompts.push(prompt.to_string());
            prompts.len()
        };
        let chunks = if call == 1 { first.clone() } else { second.clone() };
        let mut items: Vec<anyhow::Result<_>> = chunks.iter().map(|t| Ok(chunk(t))).collect();
        if call == 1 {
            items.push(Err(anyhow::anyhow!("llama_decode returned -3")));
        }
        Box::new(items.into_iter())
    };
    (stream_completion, prompts)
}

#[tokio::test]
#[serial]
async fn test_an_oom_before_anything_is_sent_starts_over() {
    let (stream, prompts) = failing_once(&["Con"], &["Contoso Ltd. billed Jordan Rivera."]);
    let svc = service(stream);
    let raw = run(request(), &svc).await;

    assert_eq!(content(&raw).concat(), "Contoso Ltd. billed Jordan Rivera.");
    let prompts = prompts.lock().clone();
    assert!(prompts.len() == 2 && prompts[0] == prompts[1]);
    assert!(svc.unload_calls().contains(&true));
}

#[tokio::test]
#[serial]
async fn test_an_oom_mid_answer_continues_it() {
    let sent = "Contoso Ltd. billed Jordan Rivera ";
    let (stream, prompts) = failing_once(&[sent], &["for contract work."]);
    let svc = service(stream);
    let raw = run(request(), &svc).await;

    let text = content(&raw).concat();
    assert_eq!(text, format!("{sent}for contract work."));
    assert!(!text.contains("**Error:**"));
    // The retry quotes back what the client already has.
    let prompts = prompts.lock().clone();
    assert!(prompts[1].starts_with(&prompts[0]) && prompts[1][prompts[0].len()..].contains(sent));
}

#[tokio::test]
#[serial]
async fn test_a_second_oom_reaches_the_client() {
    let svc = service(|_, _| -> ChunkIter {
        Box::new(std::iter::once(Err(anyhow::anyhow!("llama_decode returned -3"))))
    });
    let raw = run(request(), &svc).await;
    assert!(content(&raw).concat().contains("**Error:** llama_decode returned -3"));
}

#[tokio::test]
#[serial]
async fn test_other_errors_are_not_retried() {
    let calls: Arc<Mutex<Vec<String>>> = Arc::default();
    let seen = calls.clone();
    let svc = service(move |prompt, _| -> ChunkIter {
        seen.lock().push(prompt.to_string());
        Box::new(std::iter::once(Err(anyhow::anyhow!("bad template"))))
    });
    let raw = run(request(), &svc).await;
    assert_eq!(calls.lock().len(), 1);
    assert!(content(&raw).concat().contains("**Error:** bad template"));
}
