//! Port of `tests/unit/test_stream_interrupt_unload.py`.
//!
//! Interrupting a stream must not free models out from under llama.cpp.
//!
//! `cancelled` is cooperative — the producer thread only observes it between
//! tokens — so the consumer's cleanup runs while that thread is still inside
//! `llama_sampler_sample`.  Freeing the llama.cpp context there segfaults the
//! process:
//!
//! ```text
//! llama_sampler_sample + ffi_call_SYSV + ... + thread_run
//! EXC_BAD_ACCESS (SIGSEGV) at 0x14c23c000
//! ```
//!
//! preceded by `malloc: pointer being freed was not allocated`.  So the unload
//! has to run on the producer thread, after its generator is done.

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use serial_test::serial;

use smartloop::api::model::chat::{ChatCompletionRequest, ChatMessage, Role};
use smartloop::api::routers::chat::{stream_chunks, ChatService};
use smartloop::llms::base_llm::CancelEvent;

use crate::shared::api_chat::{chunk, ChunkIter, Event, FakeChatService, OnDrop};

/// A ModelService stand-in that reaches the producer with no orchestration.
/// Chat generates through the model client, whose default implementation is
/// the in-process one — so the real client sits in front of the fake service
/// and the test drives exactly the path production takes.  The usage tail's
/// telemetry is suppressed by the fake (`_no_telemetry`).
fn service<F>(stream_completion: F) -> Arc<FakeChatService>
where
    F: Fn(&str, Option<CancelEvent>) -> ChunkIter + Send + Sync + 'static,
{
    Arc::new(FakeChatService::with_stream(stream_completion))
}

fn request() -> ChatCompletionRequest {
    // No user message → the orchestrator block is skipped and we land on the
    // response stream directly, which is the path under test.
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

/// The unload must wait for the thread that is inside llama.cpp.
#[tokio::test]
#[serial]
async fn test_interrupt_defers_unload_until_the_producer_stops() {
    let entered = Event::new(); // producer is back inside stream_completion
    let release = Event::new(); // let it produce its second token
    let finished = Event::new(); // producer's generator is done

    let (e, r, f) = (entered.clone(), release.clone(), finished.clone());
    let svc = service(move |_, _| -> ChunkIter {
        let (entered, release) = (e.clone(), r.clone());
        let finished = f.clone();
        let finally = OnDrop::new(move || finished.set());
        let mut step = 0;
        Box::new(std::iter::from_fn(move || {
            let _finally = &finally;
            step += 1;
            match step {
                1 => Some(Ok(chunk("hello"))),
                2 => {
                    entered.set();
                    // Stands in for a blocking llama.cpp sampler call: the
                    // interrupt arrives while control is *inside* here.
                    release.wait(10.0);
                    Some(Ok(chunk(" world")))
                }
                _ => None,
            }
        }))
    });

    let mut stream = stream_chunks(request(), svc.clone() as Arc<dyn ChatService>);

    // Consume up to the first content chunk.  That leaves the stream suspended
    // past its setup — so dropping it below runs the same cleanup a real
    // client disconnect would, which is the whole point.
    while let Some(event) = stream.next().await {
        if event.contains("hello") {
            break;
        }
    }

    // ...and the producer thread is now blocked back inside stream_completion.
    for _ in 0..500 {
        if entered.is_set() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(entered.is_set(), "producer never re-entered stream_completion");

    drop(stream); // the interrupt

    assert_eq!(
        svc.unload_call_count(),
        0,
        "freed the model while the producer thread was still inside llama.cpp"
    );

    // Let the producer finish; only then may the models go.
    release.set();
    for _ in 0..500 {
        if svc.unload_call_count() > 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    assert!(finished.is_set());
    assert_eq!(svc.unload_call_count(), 1);
}

/// The happy path must keep freeing the auxiliary models.
#[tokio::test]
#[serial]
async fn test_normal_completion_still_unloads() {
    let svc = service(|_, _| -> ChunkIter { Box::new(std::iter::once(Ok(chunk("hello")))) });

    let mut stream = stream_chunks(request(), svc.clone() as Arc<dyn ChatService>);
    while stream.next().await.is_some() {}

    for _ in 0..500 {
        if svc.unload_call_count() > 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(svc.unload_call_count(), 1);
}
