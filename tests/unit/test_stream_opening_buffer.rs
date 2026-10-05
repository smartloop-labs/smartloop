//! Port of `tests/unit/test_stream_opening_buffer.py`.
//!
//! The opening of a tool-synthesis answer is buffered until it looks real.
//!
//! Small models often answer a tool payload with nothing but ``` and an EOS.
//! The retry that fixes that is only a *replacement* while nothing has reached
//! the client yet — once the fence is out, the correction can only be appended
//! after it, which reads as two answers.  So the opening is held until 20
//! alphanumeric characters say the model is actually writing, then flushed and
//! streamed normally.
//!
//! Regular chat — no tool context — streams from the first chunk either way.

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use parking_lot::Mutex;
use serial_test::serial;

use smartloop::api::model::chat::{ChatCompletionRequest, ChatMessage, Role};
use smartloop::api::routers::chat::{stream_chunks, ChatService, OrchestrateOutcome};
use smartloop::llms::base_llm::CancelEvent;

pub use crate::shared::api_chat::content;
use crate::shared::api_chat::{chunk, ChunkIter, FakeChatService};

pub const CONTEXT: &str = "TOOL RESULTS:\nContoso Ltd. billed Jordan Rivera for contract work.";

/// `_service(stream_completion, generate="", context=CONTEXT)`.
pub fn service_with<F>(stream_completion: F, generate: &str, context: Option<&str>) -> Arc<FakeChatService>
where
    F: Fn(&str, Option<CancelEvent>) -> ChunkIter + Send + Sync + 'static,
{
    let svc = FakeChatService::with_stream(stream_completion);
    *svc.device.generate_return.lock() = generate.to_string();
    // Tool context is what turns buffering on.
    *svc.orchestrate_outcome.lock() = OrchestrateOutcome {
        context: context.map(str::to_string),
        matched: false,
        error: None,
        citations: Some(Vec::new()),
        skill_instruction: None,
        mutation: false,
    };
    *svc.workspace_dir.lock() = Some("/tmp/ws".to_string());
    Arc::new(svc)
}

/// `_service(stream_completion)` with the default context.
pub fn service<F>(stream_completion: F) -> Arc<FakeChatService>
where
    F: Fn(&str, Option<CancelEvent>) -> ChunkIter + Send + Sync + 'static,
{
    service_with(stream_completion, "", Some(CONTEXT))
}

/// `_request()`.
pub fn request() -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "sl-mini".to_string(),
        messages: vec![ChatMessage {
            role: Role::User,
            content: Some("who was the invoice for?".to_string()),
            attachments: None,
        }],
        stream: true,
        ..ChatCompletionRequest::default()
    }
}

/// `[event async for event in stream_chunks(request, svc)]`.
pub async fn run(request: ChatCompletionRequest, svc: &Arc<FakeChatService>) -> Vec<String> {
    stream_chunks(request, svc.clone() as Arc<dyn ChatService>).collect().await
}

/// A stream that yields `texts` in order.
pub fn yields(texts: &[&str]) -> ChunkIter {
    let chunks: Vec<_> = texts.iter().map(|t| Ok(chunk(t))).collect();
    Box::new(chunks.into_iter())
}

/// The stray fence never reaches the client, so the retry stands alone.
#[tokio::test]
#[serial]
async fn test_a_degenerate_answer_is_replaced_not_appended() {
    let answer = "Contoso Ltd. billed Jordan Rivera.";
    let svc = service_with(|_, _| yields(&["```"]), answer, Some(CONTEXT));
    let raw = run(request(), &svc).await;

    assert_eq!(content(&raw).concat(), answer);
    // The retry asks with the tool payload and a minimal prompt, not the
    // original one.
    let prompts = svc.generate_prompts();
    let prompt = prompts.last().expect("generate_completion was called");
    assert!(prompt.contains("Contoso Ltd. billed Jordan Rivera for contract work."));
}

#[tokio::test]
#[serial]
async fn test_a_retry_that_is_also_degenerate_falls_back_to_guidance() {
    let svc = service_with(|_, _| yields(&["```"]), "``", Some(CONTEXT));
    let raw = run(request(), &svc).await;
    assert!(content(&raw).concat().contains("**No response generated.**"));
}

/// Nothing goes out on the preamble; the whole opening arrives together.
///
/// The producer pauses after a markdown preamble carrying seven alphanumerics
/// — under the threshold — and the consumer, whose loop is free for that
/// whole pause, must still have received nothing.  What follows crosses the
/// threshold, and the held chunks are flushed in order.
#[tokio::test]
#[serial]
async fn test_the_opening_is_held_then_flushed_once_the_answer_looks_real() {
    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    // What the consumer had in hand while the producer was paused.
    let at_pause: Arc<Mutex<Vec<Vec<String>>>> = Arc::default();

    let (seen_p, at_pause_p) = (seen.clone(), at_pause.clone());
    let svc = service(move |_, _| {
        let (seen, at_pause) = (seen_p.clone(), at_pause_p.clone());
        let mut step = 0;
        Box::new(std::iter::from_fn(move || {
            step += 1;
            match step {
                1 => Some(Ok(chunk("## "))),
                2 => Some(Ok(chunk("Summary"))),
                3 => Some(Ok(chunk("\n\n"))),
                4 => {
                    std::thread::sleep(Duration::from_millis(300));
                    at_pause.lock().push(seen.lock().clone());
                    Some(Ok(chunk("Contoso billed Jordan Rivera today.")))
                }
                _ => None,
            }
        }))
    });

    let mut stream = stream_chunks(request(), svc.clone() as Arc<dyn ChatService>);
    while let Some(event) = stream.next().await {
        seen.lock().extend(content(&[event]));
    }

    assert_eq!(*at_pause.lock(), vec![Vec::<String>::new()]);
    assert_eq!(
        *seen.lock(),
        vec!["## ", "Summary", "\n\n", "Contoso billed Jordan Rivera today."]
    );
}

/// No tool context, no buffering — the deltas are the generated chunks.
#[tokio::test]
#[serial]
async fn test_regular_chat_streams_from_the_first_chunk() {
    let chunks = ["The ", "invoice ", "was ", "for ", "Jordan ", "Rivera."];

    let svc = service_with(move |_, _| yields(&chunks), "", None);
    let raw = run(request(), &svc).await;
    assert_eq!(content(&raw), chunks);
}
