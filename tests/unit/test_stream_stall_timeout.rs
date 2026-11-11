//! Port of `tests/unit/test_stream_stall_timeout.py`.
//!
//! A generation that never produces must not hang the client forever.
//!
//! The orchestration pass has had a wall-clock budget for a while; the
//! response stream that follows it did not.  Its consumer waited on the
//! producer thread's queue with no deadline, so a producer that never put
//! anything there — wedged inside llama.cpp, or still queued behind an earlier
//! generation on the single-worker executor — left the request open with no
//! error, no finish_reason and no `[DONE]`.
//!
//! `slp_stream_stall_timeout_s` bounds the *silence*, not the answer: every
//! chunk restarts the clock, so a long generation never trips it.

use std::sync::Arc;
use std::time::Duration;

use serial_test::serial;

use smartloop::api::model::chat::{ChatCompletionRequest, ChatMessage, Role};
use smartloop::llms::base_llm::CancelEvent;

use crate::shared::api_chat::{chunk, ChunkIter, EnvGuard, Event, FakeChatService};
use crate::test_stream_opening_buffer::run;

fn service<F>(stream_completion: F) -> Arc<FakeChatService>
where
    F: Fn(&str, Option<CancelEvent>) -> ChunkIter + Send + Sync + 'static,
{
    Arc::new(FakeChatService::with_stream(stream_completion))
}

/// Half a second of silence stands in for the shipped five minutes.
fn short_budget() -> EnvGuard {
    EnvGuard::set("SLP_STREAM_STALL_TIMEOUT_S", "0.5")
}

fn request() -> ChatCompletionRequest {
    // No user message → no orchestration; the response stream is the path
    // under test.
    ChatCompletionRequest {
        model: "sl-mini".to_string(),
        messages: vec![ChatMessage {
            role: Role::System,
            content: Some("be brief".to_string()),
            attachments: None,
            ..Default::default()
        }],
        stream: true,
        ..ChatCompletionRequest::default()
    }
}

#[tokio::test]
#[serial]
async fn test_a_producer_that_never_yields_ends_the_stream() {
    let _budget = short_budget();
    let release = Event::new(); // never set: the producer is wedged

    let r = release.clone();
    let svc = service(move |_, _| -> ChunkIter {
        let release = r.clone();
        let mut step = 0;
        Box::new(std::iter::from_fn(move || {
            step += 1;
            if step == 1 {
                release.wait(30.0);
                Some(Ok(chunk("too late")))
            } else {
                None
            }
        }))
    });

    let events = run(request(), &svc).await;
    release.set(); // let the worker unwind so the pool is usable again

    let body = events.concat();
    assert!(body.contains("Generation stopped."), "the user was told nothing");
    assert!(body.replace(' ', "").contains("\"finish_reason\":\"stop\""));
    assert_eq!(
        events.last().map(String::as_str),
        Some("data: [DONE]\n\n"),
        "the stream never terminated"
    );
}

/// Cooperative cancellation still has to be signalled on the way out.
#[tokio::test]
#[serial]
async fn test_the_wedged_producer_is_told_to_stop() {
    let _budget = short_budget();
    let release = Event::new();
    let seen_cancel = Event::new();

    let (r, s) = (release.clone(), seen_cancel.clone());
    let svc = service(move |_, cancelled| -> ChunkIter {
        let (release, seen_cancel) = (r.clone(), s.clone());
        let mut done = false;
        Box::new(std::iter::from_fn(move || {
            if done {
                return None;
            }
            done = true;
            release.wait(30.0);
            if cancelled
                .as_ref()
                .is_some_and(|c| c.load(std::sync::atomic::Ordering::SeqCst))
            {
                seen_cancel.set();
            }
            None
        }))
    });

    let _ = run(request(), &svc).await;

    release.set();
    for _ in 0..500 {
        if seen_cancel.is_set() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(seen_cancel.is_set(), "the abandoned producer was never cancelled");
}

/// Each chunk restarts the clock — only silence trips the watchdog.
#[tokio::test]
#[serial]
async fn test_a_slow_but_live_stream_is_not_cut_off() {
    let _budget = short_budget();
    let svc = service(|_, _| -> ChunkIter {
        let mut words = vec!["one ", "two ", "three ", "four "].into_iter();
        Box::new(std::iter::from_fn(move || {
            let word = words.next()?;
            // Each gap is most of the budget; the whole answer is well past it.
            Event::new().wait(0.35);
            Some(Ok(chunk(word)))
        }))
    });

    let body = run(request(), &svc).await.concat();

    assert!(!body.contains("Generation stopped."), "cut off a stream that was alive");
    for word in ["one", "two", "three", "four"] {
        assert!(body.contains(word));
    }
}

#[tokio::test]
#[serial]
async fn test_the_watchdog_can_be_disabled() {
    let _budget = EnvGuard::set("SLP_STREAM_STALL_TIMEOUT_S", "0");

    let svc = service(|_, _| -> ChunkIter {
        let mut done = false;
        Box::new(std::iter::from_fn(move || {
            if done {
                return None;
            }
            done = true;
            Event::new().wait(0.6); // longer than the budget the others use
            Some(Ok(chunk("hello")))
        }))
    });

    let body = run(request(), &svc).await.concat();
    assert!(body.contains("hello"));
    assert!(!body.contains("Generation stopped."));
}
