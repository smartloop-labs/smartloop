//! Port of `tests/unit/test_citation_step.py`.
//!
//! Checking the sources is a visible step, and never runs on the interrupt
//! path.
//!
//! Verification embeds the finished answer, which takes long enough to
//! notice, and it lands at the worst possible moment: the text has stopped
//! moving and the citations are not up yet.  Silently, that pause reads as a
//! hang.  So it is announced like every other phase of the turn —
//! `chat.status`, step `citations` — and it happens inside the stream, before
//! the client is told the turn is complete.
//!
//! It is also deliberately *not* in the generator's `finally`, where it used
//! to sit.  Announcing it means yielding around it, and that block also runs
//! on client disconnect.  An interrupted answer still gets its sources checked
//! on the way out, with the free lexical test rather than the embedder.
//!
//! Port notes: the Python tests drive `stream_chunks` with a `MagicMock`
//! service; here a recording [`FakeService`] implements `ChatService` (and
//! `LocalModelService`, so the real `InProcessModelClient` sits in front of
//! it exactly as in production).  `gen.aclose()` is dropping the stream; Rust
//! has no async drop, so the interrupt path's effects are awaited with
//! `eventually` instead of being asserted the instant the stream is dropped.
//! Where the Python tests patch `smartloop.db.embeddings.encode_texts`, these
//! install the same override with `embeddings::set_encode_texts_hook`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex as StdMutex, Weak};
use std::time::Duration;

use futures::StreamExt;
use ndarray::Array2;
use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use serial_test::serial;
use smartloop::api::model::{ChatCompletionRequest, Citation, CitationSource};
use smartloop::api::routers::chat::{
    citation_model, complete_without_streaming, stream_chunks, ChatService, CitationDict,
    HistoryTurn, ImageAttachment, OrchestrateArgs, OrchestrateOutcome, ProgressCallback,
};
use smartloop::constants::OPENING_TOKENS;
use smartloop::db::embeddings;
use smartloop::llms::base_llm::{CancelEvent, GenerationChunk};
use smartloop::model_host::client::{InProcessModelClient, LocalModelService, ModelClient};
use smartloop::workspace::types::Skill;

const DOC: &str = "invoice-contoso.pdf";
const CHUNK: &str = "Invoice from Contoso Ltd. to Jordan Rivera for contract work.";

fn citation() -> CitationDict {
    let Value::Object(map) = json!({
        "document_id": DOC, "document_name": DOC,
        "_extract": CHUNK, "_chunks": [CHUNK], "_vectors": [null],
    }) else {
        unreachable!()
    };
    map
}

fn default_answer() -> String {
    format!("The invoice was for Jordan Rivera, {CHUNK}")
}

// ------------------------------------------------------------------ helpers

/// `threading.Event`.
#[derive(Clone, Default)]
struct Event(Arc<(StdMutex<bool>, Condvar)>);

impl Event {
    fn set(&self) {
        let (lock, cond) = &*self.0;
        *lock.lock().unwrap() = true;
        cond.notify_all();
    }

    fn is_set(&self) -> bool {
        *self.0 .0.lock().unwrap()
    }

    /// Block up to `secs`; true when the event is set.
    fn wait(&self, secs: f64) -> bool {
        let (lock, cond) = &*self.0;
        let guard = lock.lock().unwrap();
        let (guard, _) = cond
            .wait_timeout_while(guard, Duration::from_secs_f64(secs), |set| !*set)
            .unwrap();
        *guard
    }
}

type ChunkIter = Box<dyn Iterator<Item = anyhow::Result<GenerationChunk>> + Send>;
type StreamFn = Arc<dyn Fn() -> ChunkIter + Send + Sync>;

fn chunk(text: &str) -> anyhow::Result<GenerationChunk> {
    Ok(GenerationChunk { text: text.to_string() })
}

/// A generator over fixed texts.
fn yields(texts: Vec<String>) -> StreamFn {
    Arc::new(move || {
        let texts = texts.clone();
        Box::new(texts.into_iter().map(|t| chunk(&t))) as ChunkIter
    })
}

/// Yield `first`, set `entered`, block until `release` (10s), yield
/// `" and more"` — the generator the interrupt tests use.
fn blocking_after_first(first: String, entered: Event, release: Event) -> StreamFn {
    Arc::new(move || {
        let (first, entered, release) = (first.clone(), entered.clone(), release.clone());
        let mut step = 0;
        Box::new(std::iter::from_fn(move || {
            step += 1;
            match step {
                1 => Some(chunk(&first)),
                2 => {
                    entered.set();
                    release.wait(10.0);
                    Some(chunk(" and more"))
                }
                _ => None,
            }
        })) as ChunkIter
    })
}

/// One recorded `append_to_session` call.
#[derive(Debug, Clone)]
#[allow(dead_code)] // the full call is recorded; tests assert on a subset
struct SessionWrite {
    session_id: String,
    role: String,
    content: String,
    citations: Option<Vec<CitationDict>>,
}

/// The `MagicMock()` service of the Python `_service` helper, recording the
/// calls the tests assert on.
struct FakeService {
    weak: Weak<FakeService>,
    stream: StreamFn,
    citations: Option<Vec<CitationDict>>,
    appended: Mutex<Vec<SessionWrite>>,
    /// Every `unload_auxiliary_models(include_embedding)` call, through the
    /// service or through its model client (the same mock in Python).
    unload_auxiliary: Mutex<Vec<bool>>,
    unload_embedding: AtomicUsize,
}

impl FakeService {
    fn new(stream: StreamFn, citations: Option<Vec<CitationDict>>) -> Arc<Self> {
        Arc::new_cyclic(|weak| FakeService {
            weak: weak.clone(),
            stream,
            citations,
            appended: Mutex::new(Vec::new()),
            unload_auxiliary: Mutex::new(Vec::new()),
            unload_embedding: AtomicUsize::new(0),
        })
    }

    /// Session writes that carried citations.
    fn saved_with_citations(&self) -> Vec<SessionWrite> {
        self.appended
            .lock()
            .iter()
            .filter(|w| w.citations.as_ref().is_some_and(|c| !c.is_empty()))
            .cloned()
            .collect()
    }

    fn last_unload_auxiliary(&self) -> Option<bool> {
        self.unload_auxiliary.lock().last().copied()
    }

    fn unload_embedding_count(&self) -> usize {
        self.unload_embedding.load(Ordering::SeqCst)
    }
}

impl ChatService for FakeService {
    fn is_loaded(&self) -> bool {
        true
    }
    fn load_model(&self) -> anyhow::Result<()> {
        Ok(())
    }
    fn get_relevant_history(&self, _session_id: &str, _user_query: &str) -> Vec<HistoryTurn> {
        Vec::new()
    }
    fn get_project_skills(&self) -> Vec<Skill> {
        Vec::new()
    }
    fn get_asset_images(&self, _asset_ids: &[String]) -> Vec<ImageAttachment> {
        Vec::new()
    }
    fn resolve_attachment_context(&self, _asset_ids: &[String]) -> String {
        String::new()
    }
    fn orchestrate(&self, _args: OrchestrateArgs) -> anyhow::Result<OrchestrateOutcome> {
        Ok(OrchestrateOutcome {
            context: Some(format!("DOCUMENT CONTEXT\n{CHUNK}")),
            matched: false,
            error: None,
            citations: self.citations.clone(),
            skill_instruction: None,
            mutation: false,
            capability: None,
            coding: false,
        })
    }
    fn get_project_id(&self) -> Option<String> {
        Some("proj".to_string())
    }
    fn get_project_info(&self) -> Option<String> {
        None
    }
    fn count_tokens(&self, _text: &str) -> usize {
        10
    }
    fn get_max_context_tokens(&self) -> usize {
        4096
    }
    fn get_response_max_context_tokens(&self, _capability: Option<&str>) -> usize {
        4096
    }
    fn begin_response_request(&self, _progress_callback: Option<ProgressCallback>) {}
    fn announce_response_model(&self, _capability: Option<&str>) {}
    fn append_to_session(
        &self,
        session_id: &str,
        role: &str,
        content: &str,
        citations: Option<&[CitationDict]>,
        _message_id: Option<&str>,
    ) {
        self.appended.lock().push(SessionWrite {
            session_id: session_id.to_string(),
            role: role.to_string(),
            content: content.to_string(),
            citations: citations.map(|c| c.to_vec()),
        });
    }
    fn model_client(&self) -> Arc<dyn ModelClient> {
        // Chat generates through the model client, whose default
        // implementation is the in-process one — so put the real client in
        // front of the fake service and the test drives exactly the path
        // production takes.
        let me: Arc<FakeService> = self.weak.upgrade().expect("service alive");
        Arc::new(InProcessModelClient::new(me))
    }
    fn unload_auxiliary_models(&self, include_embedding: bool) {
        self.unload_auxiliary.lock().push(include_embedding);
    }
    fn unload_embedding_model(&self) {
        self.unload_embedding.fetch_add(1, Ordering::SeqCst);
    }
    fn workspace_dir(&self) -> Option<String> {
        Some("/tmp/ws".to_string())
    }
    fn workspace_project_ids(&self) -> Option<Vec<String>> {
        None
    }
    fn current_project_dir(&self) -> Option<String> {
        None
    }
    /// No telemetry here (the Python autouse fixture patches
    /// `get_api_client`).
    fn push_token_usage(
        &self,
        _project_id: Option<&str>,
        _model: &str,
        _prompt_tokens: usize,
        _completion_tokens: usize,
    ) {
    }
}

impl LocalModelService for FakeService {
    fn generate_completion(
        &self,
        _prompt: &str,
        _model_name: Option<&str>,
        _capability: Option<&str>,
        _max_tokens: Option<usize>,
        _thinking: Option<bool>,
    ) -> anyhow::Result<String> {
        (self.stream)().map(|c| c.map(|c| c.text)).collect()
    }
    fn stream_completion<'a>(
        &'a self,
        _prompt: &str,
        _model_name: Option<&str>,
        _cancelled: Option<CancelEvent>,
        _capability: Option<&str>,
        _thinking: Option<bool>,
    ) -> anyhow::Result<Box<dyn Iterator<Item = anyhow::Result<GenerationChunk>> + 'a>> {
        Ok((self.stream)())
    }
    fn is_loaded(&self) -> bool {
        true
    }
    fn resident_model_name(&self) -> Option<String> {
        None
    }
    fn response_model_label(&self, _capability: Option<&str>) -> String {
        "sl-mini".to_string()
    }
    fn count_tokens(&self, _text: &str) -> usize {
        10
    }
    fn get_max_context_tokens(&self) -> usize {
        4096
    }
    fn get_response_max_context_tokens(&self, _capability: Option<&str>) -> usize {
        4096
    }
    fn has_capability_model(&self, _capability: &str) -> bool {
        false
    }
    fn idle_seconds(&self) -> f64 {
        f64::INFINITY
    }
    fn load_model(&self) -> anyhow::Result<()> {
        Ok(())
    }
    fn unload_auxiliary_models(&self, include_embedding: bool) {
        self.unload_auxiliary.lock().push(include_embedding);
    }
    fn unload_embedding_model(&self) {
        self.unload_embedding.fetch_add(1, Ordering::SeqCst);
    }
    fn unload_model_by_name(&self, _model_name: &str) -> bool {
        false
    }
}

/// Clears the `encode_texts` override however the test ends.
#[must_use]
struct EncoderGuard;

impl Drop for EncoderGuard {
    fn drop(&mut self) {
        embeddings::set_encode_texts_hook(None);
    }
}

type Calls = Arc<Mutex<Vec<Vec<String>>>>;

/// The autouse `_encoder` fixture: no embedding model here — verification
/// takes its lexical fallback.  Returned as a spy.
fn encoder_spy() -> (EncoderGuard, Calls) {
    let calls: Calls = Arc::default();
    let recorder = calls.clone();
    embeddings::set_encode_texts_hook(Some(Arc::new(move |texts: &[String], _base_dir, _timeout| {
        recorder.lock().push(texts.to_vec());
        None
    })));
    (EncoderGuard, calls)
}

/// Deterministic stand-in vectors — shape is all these tests care about.
fn fake_vectors(texts: &[String]) -> Array2<f32> {
    Array2::from_shape_fn((texts.len(), 2), |(i, j)| {
        let n = texts[i].chars().count();
        if j == 0 { (n % 7) as f32 + 1.0 } else { (n % 3) as f32 + 1.0 }
    })
}

/// An encoder recording every batch and returning [`fake_vectors`];
/// `on_call` runs on each call.
fn recording_encoder(on_call: Option<Event>) -> (EncoderGuard, Calls) {
    let calls: Calls = Arc::default();
    let recorder = calls.clone();
    embeddings::set_encode_texts_hook(Some(Arc::new(move |texts: &[String], _base_dir, _timeout| {
        recorder.lock().push(texts.to_vec());
        if let Some(event) = &on_call {
            event.set();
        }
        Some(fake_vectors(texts))
    })));
    (EncoderGuard, calls)
}

fn request(session_id: Option<&str>) -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": "sl-mini",
        "messages": [{"role": "user", "content": "who was the invoice for?"}],
        "stream": true,
        "session_id": session_id,
    }))
    .expect("valid request")
}

/// The SSE payloads, decoded, in order.
fn events(raw: &[String]) -> Vec<Value> {
    let mut out = Vec::new();
    for chunk in raw {
        for line in chunk.lines() {
            if let Some(body) = line.strip_prefix("data: ") {
                if body != "[DONE]" {
                    out.push(serde_json::from_str(body).expect("JSON event"));
                }
            }
        }
    }
    out
}

/// Wait for work that was deliberately fired without being awaited.
async fn eventually(predicate: impl Fn() -> bool, timeout: f64) -> bool {
    for _ in 0..(timeout / 0.01) as usize {
        if predicate() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    predicate()
}

fn object(e: &Value) -> Option<&str> {
    e.get("object").and_then(Value::as_str)
}

fn steps(events: &[Value]) -> Vec<(String, String)> {
    events
        .iter()
        .filter(|e| object(e) == Some("chat.status"))
        .map(|e| {
            (
                e["step"].as_str().unwrap_or_default().to_string(),
                e["status"].as_str().unwrap_or_default().to_string(),
            )
        })
        .collect()
}

fn has_step(events: &[Value], step: &str, status: &str) -> bool {
    steps(events).contains(&(step.to_string(), status.to_string()))
}

/// The truthy `choices[0].delta.content` of a completion chunk, if any.
fn delta_content(e: &Value) -> Option<&str> {
    e.get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("delta"))
        .and_then(|d| d.get("content"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

async fn run_with(
    citations: Option<Vec<CitationDict>>,
    answer: String,
    session_id: Option<&str>,
) -> (Arc<FakeService>, Vec<Value>) {
    let svc = FakeService::new(yields(vec![answer]), citations);
    let raw: Vec<String> = stream_chunks(request(session_id), svc.clone()).collect().await;
    (svc, events(&raw))
}

async fn run(citations: Option<Vec<CitationDict>>) -> (Arc<FakeService>, Vec<Value>) {
    run_with(citations, default_answer(), None).await
}

/// Read the stream until the answer text is out, wait for the generator to
/// park, then drop the stream (the interrupt).
async fn interrupt_after_answer(svc: Arc<FakeService>, entered: &Event) {
    let mut stream = stream_chunks(request(Some("sess-1")), svc);
    while let Some(event) = stream.next().await {
        if event.contains("Jordan Rivera") && !event.contains("chat.status") {
            break;
        }
    }
    for _ in 0..500 {
        if entered.is_set() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    drop(stream); // the interrupt
}

fn document_names(citations: &[CitationDict]) -> Vec<String> {
    citations
        .iter()
        .map(|c| c["document_name"].as_str().unwrap_or_default().to_string())
        .collect()
}

// -------------------------------------------------------------------- tests

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial(embeddings)]
async fn test_the_source_check_is_announced_as_a_step() {
    let (_enc, _spy) = encoder_spy();
    let (_, events) = run(Some(vec![citation()])).await;
    assert!(has_step(&events, "citations", "processing"));
    assert!(has_step(&events, "citations", "completed"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial(embeddings)]
async fn test_the_step_opens_after_the_text_and_closes_before_the_turn_does() {
    // The whole point of the step is to cover the gap between the last token
    // and the citations, so it has to sit in exactly that gap.
    let (_enc, _spy) = encoder_spy();
    let (_, events) = run(Some(vec![citation()])).await;
    let kinds: Vec<String> = events
        .iter()
        .map(|e| {
            if object(e) == Some("chat.completion.chunk") && delta_content(e).is_some() {
                "content".to_string()
            } else if object(e) == Some("chat.status") {
                e["step"].as_str().unwrap_or_default().to_string()
            } else {
                object(e).unwrap_or_default().to_string()
            }
        })
        .collect();
    let last_content = kinds.iter().rposition(|k| k == "content").expect("content");
    let opened = kinds.iter().position(|k| k == "citations").expect("citations step");
    let emitted = kinds.iter().position(|k| k == "chat.citations").expect("citations event");
    let complete = kinds.iter().rposition(|k| k == "streaming").expect("streaming step");
    assert!(last_content < opened && opened < emitted && emitted < complete, "{kinds:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial(embeddings)]
async fn test_the_closing_message_reports_what_survived() {
    let (_enc, _spy) = encoder_spy();
    let (_, events) = run(Some(vec![citation()])).await;
    let done: Vec<&Value> = events
        .iter()
        .filter(|e| object(e) == Some("chat.status") && e["step"] == "citations" && e["status"] == "completed")
        .collect();
    assert_eq!(done[0]["message"], "1 source(s) cited");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial(embeddings)]
async fn test_an_answer_that_used_nothing_says_so_and_emits_no_citations() {
    let (_enc, _spy) = encoder_spy();
    let (_, events) =
        run_with(Some(vec![citation()]), "The Hasselblad has a big sensor.".to_string(), None).await;
    let done: Vec<&Value> = events
        .iter()
        .filter(|e| object(e) == Some("chat.status") && e["step"] == "citations" && e["status"] == "completed")
        .collect();
    assert_eq!(done[0]["message"], "No sources used");
    assert!(!events.iter().any(|e| object(e) == Some("chat.citations")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial(embeddings)]
async fn test_a_turn_with_no_sources_skips_the_step_entirely() {
    let (_enc, _spy) = encoder_spy();
    let (_, events) = run(None).await;
    assert!(!steps(&events).iter().any(|s| s.0 == "citations"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial(embeddings)]
async fn test_the_verified_set_is_what_reaches_the_session() {
    let (_enc, _spy) = encoder_spy();
    let (svc, _) = run_with(Some(vec![citation()]), default_answer(), Some("sess-1")).await;
    // Only the streamed assistant turn carries citations.
    let saved = svc.saved_with_citations();
    let cits = saved[0].citations.as_ref().unwrap();
    assert_eq!(document_names(cits), vec![DOC]);
    assert!(!cits[0].keys().any(|k| k.starts_with('_')));
}

/// A stopped answer is still checked — with the free test, on the way out.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial(embeddings)]
async fn test_an_interrupt_writes_the_session_without_loading_the_embedder() {
    let (_enc, _spy) = encoder_spy();
    let entered = Event::default();
    let release = Event::default();

    let svc = FakeService::new(
        blocking_after_first(default_answer(), entered.clone(), release.clone()),
        Some(vec![citation()]),
    );
    interrupt_after_answer(svc.clone(), &entered).await;

    let written = eventually(|| !svc.saved_with_citations().is_empty(), 5.0).await;
    assert!(written, "the interrupt path lost the session write");
    let saved = svc.saved_with_citations();
    assert_eq!(document_names(saved[0].citations.as_ref().unwrap()), vec![DOC]);
    assert_eq!(saved[0].session_id, "sess-1");
    release.set();
}

// ---------------------------------------------------------------------------
// When the embedding model is loaded.
//
// From a real run: the GGUF took ~3.4s to come back, and every bit of it
// landed after the last token, where the user is waiting for citations to
// appear.  Two things caused that.  `unload_auxiliary_models` freed the
// embedder the instant generation ended — on the premise that "retrieval runs
// at the start of a turn and is finished well before the response is", which
// stopped being true when verification started needing it at the end.  And
// nothing loaded it until verification asked, by which point there was
// nothing left to overlap with.
//
//   [embeddings] Loaded llama-cpp:gpustack/bge-m3-GGUF/bge-m3-Q4_K_M.gguf
//   [citations] 'sp923.pdf' distance=0.563 cutoff=0.273 — dropping   (+3.4s)
//
// Verification cannot start before the answer is complete.  The load can.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial(embeddings)]
async fn test_the_producer_keeps_the_embedder_when_there_are_sources_to_check() {
    let (_enc, _spy) = encoder_spy();
    let (svc, _) = run(Some(vec![citation()])).await;
    assert!(eventually(|| svc.last_unload_auxiliary().is_some(), 5.0).await);
    assert_eq!(svc.last_unload_auxiliary(), Some(false));
    // ...and the consumer frees it once verification is done, so the
    // between-chats footprint is unchanged.  Fired without awaiting, so that
    // the citations event does not queue behind a GPU context teardown.
    assert!(eventually(|| svc.unload_embedding_count() == 1, 5.0).await);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial(embeddings)]
async fn test_a_turn_with_no_sources_frees_the_embedder_as_before() {
    let (_enc, _spy) = encoder_spy();
    let (svc, _) = run(None).await;
    assert!(eventually(|| svc.last_unload_auxiliary().is_some(), 5.0).await);
    assert_eq!(svc.last_unload_auxiliary(), Some(true));
}

/// The producer left it loaded for a check that is no longer happening.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial(embeddings)]
async fn test_an_interrupt_frees_the_embedder_nobody_will_use() {
    let (_enc, _spy) = encoder_spy();
    let entered = Event::default();
    let release = Event::default();

    let svc = FakeService::new(
        blocking_after_first(default_answer(), entered.clone(), release.clone()),
        Some(vec![citation()]),
    );
    interrupt_after_answer(svc.clone(), &entered).await;

    assert!(eventually(|| svc.unload_embedding_count() == 1, 5.0).await);
    release.set();
}

/// The half of verification that does not depend on the answer.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial(embeddings)]
async fn test_the_sources_are_embedded_while_the_answer_is_still_streaming() {
    let sources_done = Event::default();
    let (_enc, encoded) = recording_encoder(Some(sources_done.clone()));

    let gate = sources_done.clone();
    let stream: StreamFn = Arc::new(move || {
        let gate = gate.clone();
        let mut step = 0;
        Box::new(std::iter::from_fn(move || {
            step += 1;
            match step {
                1 => Some(chunk("The invoice ")),
                2 => {
                    gate.wait(10.0); // the sources are being encoded right now
                    Some(chunk(&format!("was for Jordan Rivera, {CHUNK}")))
                }
                _ => None,
            }
        })) as ChunkIter
    });

    let svc = FakeService::new(stream, Some(vec![citation()]));
    let _: Vec<String> = stream_chunks(request(None), svc).collect().await;

    assert!(sources_done.is_set(), "sources were not encoded until after the answer");
    assert_eq!(encoded.lock()[0], vec![CHUNK.to_string()]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial(embeddings)]
async fn test_once_the_answer_arrives_only_the_answer_is_embedded() {
    // The measured failure: three web citations carry a whole fetched page
    // and no vectors, so verification was encoding two dozen 1200-character
    // windows after the last token, with the user watching.  Pre-embedding
    // leaves this step one encode — the answer, which is the only thing that
    // could not have been done earlier.
    let (_enc, encoded) = recording_encoder(None);

    let answer = default_answer();
    run_with(Some(vec![citation()]), answer.clone(), None).await;

    assert_eq!(*encoded.lock(), vec![vec![CHUNK.to_string()], vec![answer]]);
}

/// A citation with only a fetched page, which is what a web result is.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial(embeddings)]
async fn test_a_web_style_source_is_windowed_and_embedded_up_front() {
    let (_enc, encoded) = recording_encoder(None);

    let mut page = Map::new();
    page.insert("document_id".into(), json!("https://example.test/x"));
    page.insert("url".into(), json!("https://example.test/x"));
    page.insert("document_name".into(), json!("https://example.test/x"));
    page.insert(
        "_extract".into(),
        Value::String(format!("{CHUNK}{}", " Navigation, footer, cookie banner. ".repeat(200))),
    );
    run(Some(vec![page])).await;

    let encoded = encoded.lock();
    assert_eq!(encoded.len(), 2, "the page was still being encoded after the answer");
    assert!(encoded[0].len() > 1, "a long page should window before it is encoded");
}

/// Sources may already be encoded; the answer nobody waited for is not.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial(embeddings)]
async fn test_an_interrupt_never_embeds_the_answer() {
    let entered = Event::default();
    let release = Event::default();
    let (_enc, encoded) = recording_encoder(None);

    let answer = default_answer();
    let svc = FakeService::new(
        blocking_after_first(answer.clone(), entered.clone(), release.clone()),
        Some(vec![citation()]),
    );
    {
        let mut stream = stream_chunks(request(Some("sess-1")), svc.clone());
        while let Some(event) = stream.next().await {
            if event.contains("Jordan Rivera") && !event.contains("chat.status") {
                break;
            }
        }
        let parked = entered.clone();
        assert!(eventually(move || parked.is_set(), 5.0).await);
        drop(stream);
    }
    // Let the interrupt path finish (its session write is its last effect)
    // before checking what it encoded.
    assert!(eventually(|| !svc.saved_with_citations().is_empty(), 5.0).await);

    assert!(!encoded.lock().iter().flatten().any(|t| t.contains(&answer)));
    release.set();
}

/// Overlapping the encode with generation is right; starting it first is
/// not.
///
/// The embedder is a second GGUF and it loads onto the same device the
/// answer is decoding on — about a second and a half of it, from the server
/// log:
///
/// ```text
///     21:46:47,018  [models] RESPONSE → qwen3.5-2b
///     21:46:47,020  [embeddings] Using cached GGUF … bge-m3-Q4_K_M.gguf
///     21:46:48,382  [embeddings] Loaded llama-cpp:… (n_gpu_layers=-1)
/// ```
///
/// Landing that on the first tokens is the one place it shows: the answer
/// opens slowly and then speeds up, which reads as the model warming up.
mod test_the_source_encode_waits_for_the_opening {
    use super::*;

    /// `(service, started)` — `started` is set when the source encode is
    /// reached.  The Python test patches `_prepare_citation_sources` itself;
    /// here the observable is the call it makes: the encode of the source
    /// passages alone (verification's own encode leads with the answer).
    fn service_with(chunks: Vec<String>) -> (Arc<FakeService>, Event, EncoderGuard) {
        let started = Event::default();
        let flag = started.clone();
        embeddings::set_encode_texts_hook(Some(Arc::new(move |texts: &[String], _base_dir, _timeout| {
            if texts.first().map(String::as_str) == Some(CHUNK) {
                flag.set();
            }
            None
        })));
        (FakeService::new(yields(chunks), Some(vec![citation()])), started, EncoderGuard)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[serial(embeddings)]
    async fn test_it_does_not_start_before_the_opening_is_out() {
        // Two hundred one-token chunks: long enough to have an opening and a
        // rest, like the answers this was noticed on.
        let (svc, started, _enc) = service_with((0..200).map(|_| "tok ".to_string()).collect());

        let mut stream = stream_chunks(request(None), svc);
        let mut seen = 0usize;
        while let Some(event) = stream.next().await {
            let deltas = events(std::slice::from_ref(&event))
                .iter()
                .filter(|e| delta_content(e).is_some())
                .count();
            if deltas == 0 {
                continue;
            }
            seen += 1;
            if seen == 1 {
                // One token of the answer is out; the device belongs to the
                // model that is writing it.
                let s = started.clone();
                let early = tokio::task::spawn_blocking(move || s.wait(0.25)).await.unwrap();
                assert!(!early);
            }
            if seen > OPENING_TOKENS {
                break;
            }
        }
        drop(stream);

        let s = started.clone();
        let reached = tokio::task::spawn_blocking(move || s.wait(5.0)).await.unwrap();
        assert!(reached, "the encode still has to happen");
    }

    /// Nothing left to overlap with — but the encode still has to happen.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[serial(embeddings)]
    async fn test_a_short_answer_still_gets_its_sources_encoded() {
        let (svc, started, _enc) = service_with(vec!["Jordan Rivera.".to_string()]);

        let _: Vec<String> = stream_chunks(request(None), svc).collect().await;

        assert!(started.is_set());
    }
}

fn dict(value: Value) -> CitationDict {
    let Value::Object(map) = value else { unreachable!() };
    map
}

/// The non-streamed response carries what the stream does: where the
/// document is and the pages cited, not just its id and name.
#[test]
fn test_a_non_streamed_citation_keeps_its_path_and_pages() {
    let model = citation_model(&dict(json!({
        "document_id": "9f2c", "document_name": DOC,
        "document_path": "documents/invoice-contoso.pdf", "page_numbers": [2, 5],
    })))
    .unwrap();
    assert_eq!(model.document_id, "9f2c");
    assert_eq!(model.document_path.as_deref(), Some("documents/invoice-contoso.pdf"));
    assert_eq!(model.page_numbers, Some(vec![json!(2), json!(5)]));

    let bare = citation_model(&dict(json!({"document_id": "9f2c", "page_numbers": []}))).unwrap();
    assert_eq!((&bare.document_path, &bare.page_numbers), (&None, &None));
    let wire = serde_json::to_value(&bare).unwrap();
    assert!(wire.get("document_path").is_none() && wire.get("page_numbers").is_none());
}

/// The fan-out cites a document by name alone; that names it.  One with
/// neither an id nor a name is dropped rather than failing the answer.
#[test]
fn test_a_citation_without_an_id_goes_by_its_name() {
    let model = citation_model(&dict(json!({"document_name": DOC}))).unwrap();
    assert_eq!(model.document_id, DOC);
    assert!(citation_model(&dict(json!({"page_numbers": [1]}))).is_none());
}

/// The non-streamed citation names its source as the stream does; a source
/// this build doesn't know is left off rather than failing the answer.
#[test]
fn test_a_non_streamed_citation_keeps_its_source() {
    let web = citation_model(&dict(json!({"document_id": "https://a", "source": "web"}))).unwrap();
    assert_eq!(web.source, Some(CitationSource::Web));
    assert_eq!(serde_json::to_value(&web).unwrap()["source"], "web");

    let unknown = citation_model(&dict(json!({"document_id": "x", "source": "fax"}))).unwrap();
    assert_eq!(unknown.source, None);
    assert!(serde_json::to_value(&unknown).unwrap().get("source").is_none());
}

/// A key the model doesn't name still reaches the client, but a private one
/// never does; pages go out as the index recorded them.
#[test]
fn test_a_citation_passes_unnamed_keys_through_but_not_private_ones() {
    let model = citation_model(&dict(json!({
        "document_id": "9f2c", "page_numbers": ["iv", 3], "sheet": "Q3",
        "_extract": CHUNK,
    })))
    .unwrap();
    let wire = serde_json::to_value(&model).unwrap();
    assert_eq!(wire["sheet"], "Q3");
    assert_eq!(wire["page_numbers"], json!(["iv", 3]));
    assert!(wire.get("_extract").is_none());
}

/// The citations of one turn, every kind, as their builders make them.
fn every_kind_of_citation() -> Vec<CitationDict> {
    let mut document = citation();
    document.insert("source".into(), json!("document"));
    document.insert("document_path".into(), json!("documents/invoice-contoso.pdf"));
    document.insert("page_numbers".into(), json!([2, 5]));
    let tool = dict(json!({
        "document_id": "mcp:Linear.list_issues", "document_name": "Linear · list_issues",
        "source": "mcp", "server_name": "Linear", "tool_name": "list_issues",
    }));
    vec![document, tool]
}

/// What a client receives for [`every_kind_of_citation`]: each citation's
/// public keys, exactly as they were built.
fn every_kind_of_citation_on_the_wire() -> Value {
    json!([
        {
            "document_id": DOC, "document_name": DOC, "source": "document",
            "document_path": "documents/invoice-contoso.pdf", "page_numbers": [2, 5],
        },
        {
            "document_id": "mcp:Linear.list_issues", "document_name": "Linear · list_issues",
            "source": "mcp", "server_name": "Linear", "tool_name": "list_issues",
        },
    ])
}

/// The stream and the non-streamed response carry the same citations, and
/// the stream's are its citations' public keys, unchanged.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial(embeddings)]
async fn test_streamed_and_non_streamed_citations_match() {
    let (_enc, _spy) = encoder_spy();
    let (_, events) = run(Some(every_kind_of_citation())).await;
    let streamed = events
        .iter()
        .find(|e| object(e) == Some("chat.citations"))
        .map(|e| e["citations"].clone())
        .expect("a citations event");

    let svc = FakeService::new(yields(vec![default_answer()]), Some(every_kind_of_citation()));
    let mut non_streaming = request(None);
    non_streaming.stream = false;
    let response = complete_without_streaming(non_streaming, svc).await.expect("a response");
    let not_streamed = serde_json::to_value(response.citations.expect("citations")).unwrap();

    assert_eq!(streamed, every_kind_of_citation_on_the_wire());
    assert_eq!(not_streamed, streamed);
}

/// A client reading another version's citations: one cited by name alone
/// with no id still reads, and a kind this build doesn't know reads as none
/// rather than failing the citation.
#[test]
fn test_a_citation_from_another_version_still_reads() {
    let old: Citation = serde_json::from_value(json!({"document_name": DOC})).unwrap();
    assert_eq!((old.document_id.as_str(), old.document_name.as_deref()), ("", Some(DOC)));

    let newer: Citation =
        serde_json::from_value(json!({"document_id": "x", "source": "image", "pages": 2})).unwrap();
    assert_eq!(newer.source, None);
    assert_eq!(newer.extra["pages"], 2);

    let web: Citation = serde_json::from_value(json!({"document_id": "x", "source": "web"})).unwrap();
    assert_eq!(web.source, Some(CitationSource::Web));
}
