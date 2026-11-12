//! Port of `tests/unit/test_remote_llm.py`.
//!
//! A per-project agent's orchestrator runs on the host's model, not on nothing.
//!
//! Routing the final response through a model client was only half of it.
//! The orchestrator does most of the inference — classification, skill
//! matching, planning, every DAG decision node — and it takes `BaseLLM`
//! objects directly.  It also fails *quietly* without one: `orchestrate` logs
//! "BYPASSED" and falls back to raw generation, so a child with no weights
//! would keep answering while silently losing skills, tools and planning.

use std::any::Any;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ndarray::Array2;
use parking_lot::{Mutex, RwLock};

use smartloop::agents::orchestration::model_router::{ModelRouter, ModelRouterOptions, ProgressCallback};
use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::llms::base_llm::{BaseLLM, CallOptions, CancelEvent, ChatMessage};
use smartloop::model::model_manager::Model;
use smartloop::model_host::client::{
    InProcessModelClient, LocalModelService, ModelClient, RemoteModelClient, TextStream,
};
use smartloop::model_host::remote_llm::{DEFAULT_MAX_CONTEXT_TOKENS, RemoteLLM};
use smartloop::types::ModelCapability;
use smartloop::workspace::Workspace;

use crate::shared::model_service_b::{FakeLLM, global_lock};

/// One recorded call: `(method, argument, capability)`.
type Call = (String, String, Option<String>);

/// The host, as a child sees it.
struct FakeClient {
    calls: Mutex<Vec<Call>>,
    chunks: Vec<String>,
    fail_with: RwLock<Option<String>>,
    tokens: usize,
    context: RwLock<usize>,
    max_context: RwLock<Option<usize>>,
    generate_reply: RwLock<Option<String>>,
    options: Mutex<Option<(Option<usize>, Option<bool>)>>,
    /// `None`: `is_loaded` raises; `Some(b)`: answers `b`.
    loaded: RwLock<Option<bool>>,
    /// Stands in for the "is it a RemoteModelClient" check the service makes.
    remote: RemoteModelClient,
}

impl FakeClient {
    fn new() -> Arc<FakeClient> {
        Arc::new(FakeClient {
            calls: Mutex::new(Vec::new()),
            chunks: vec!["think".into(), "ing".into(), " done".into()],
            fail_with: RwLock::new(None),
            tokens: 42,
            context: RwLock::new(8192),
            max_context: RwLock::new(None),
            generate_reply: RwLock::new(None),
            options: Mutex::new(None),
            loaded: RwLock::new(Some(true)),
            remote: RemoteModelClient::new("unix:///nonexistent/slp-test.sock"),
        })
    }

    fn record(&self, method: &str, arg: &str, capability: Option<&str>) {
        self.calls.lock().push((method.to_string(), arg.to_string(), capability.map(str::to_string)));
    }

    fn called(&self, method: &str, arg: &str, capability: Option<&str>) -> bool {
        self.calls.lock().iter().any(|(m, a, c)| m == method && a == arg && c.as_deref() == capability)
    }

    fn fail(&self) -> anyhow::Result<()> {
        match self.fail_with.read().clone() {
            Some(message) => Err(anyhow::anyhow!(message)),
            None => Ok(()),
        }
    }
}

impl ModelClient for FakeClient {
    fn generate(
        &self,
        prompt: &str,
        _model_name: Option<&str>,
        capability: Option<&str>,
        max_tokens: Option<usize>,
        thinking: Option<bool>,
    ) -> anyhow::Result<String> {
        self.record("generate", prompt, capability);
        *self.options.lock() = Some((max_tokens, thinking));
        self.fail()?;
        if let Some(reply) = self.generate_reply.read().clone() {
            return Ok(reply);
        }
        Ok(format!("answered {prompt}"))
    }

    fn stream<'a>(
        &'a self,
        prompt: &str,
        _model_name: Option<&str>,
        capability: Option<&str>,
        cancelled: Option<CancelEvent>,
        _thinking: Option<bool>,
    ) -> anyhow::Result<TextStream<'a>> {
        self.record("stream", prompt, capability);
        let mut chunks = self.chunks.clone().into_iter();
        Ok(Box::new(std::iter::from_fn(move || {
            if cancelled.as_ref().is_some_and(|c| c.load(Ordering::SeqCst)) {
                return None;
            }
            chunks.next().map(Ok)
        })))
    }

    fn encode_texts(&self, _texts: &[String], _base_dir: Option<&str>) -> anyhow::Result<Option<Array2<f32>>> {
        Ok(None)
    }

    fn is_loaded(&self) -> anyhow::Result<bool> {
        match *self.loaded.read() {
            Some(loaded) => Ok(loaded),
            None => Err(anyhow::anyhow!("gone")),
        }
    }

    fn model_name(&self) -> anyhow::Result<Option<String>> {
        Ok(Some("sl-mini".into()))
    }

    fn model_label(&self, capability: Option<&str>) -> anyhow::Result<String> {
        self.record("model_label", "", capability);
        self.fail()?;
        Ok("sl-mini (2B)".into())
    }

    fn count_tokens(&self, text: &str) -> anyhow::Result<usize> {
        self.record("count_tokens", text, None);
        self.fail()?;
        Ok(self.tokens)
    }

    fn max_context_tokens(&self) -> anyhow::Result<usize> {
        self.fail()?;
        Ok(self.max_context.read().unwrap_or(*self.context.read()))
    }

    fn response_max_context_tokens(&self, capability: Option<&str>) -> anyhow::Result<usize> {
        self.record("response_max_context_tokens", "", capability);
        self.fail()?;
        Ok(*self.context.read())
    }

    fn has_capability_model(&self, _capability: &str) -> anyhow::Result<bool> {
        Ok(true)
    }

    fn idle_seconds(&self) -> anyhow::Result<f64> {
        Ok(0.0)
    }

    fn load_model(&self) -> anyhow::Result<Option<String>> {
        Ok(Some("sl-mini".into()))
    }

    fn unload_auxiliary_models(&self, include_embedding: bool) -> anyhow::Result<()> {
        self.record("unload_auxiliary_models", &include_embedding.to_string(), None);
        self.fail()
    }

    fn unload_embedding_model(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn as_remote(&self) -> Option<&RemoteModelClient> {
        Some(&self.remote)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn client() -> Arc<FakeClient> {
    FakeClient::new()
}

fn remote(client: &Arc<FakeClient>, project_id: &str, capability: Option<&str>) -> RemoteLLM {
    RemoteLLM::new(client.clone(), project_id, capability.map(str::to_string))
}

fn llm(client: &Arc<FakeClient>) -> RemoteLLM {
    remote(client, "alpha", None)
}

// ------------------------------------------------------- it looks like an LLM

#[test]
fn test_it_is_a_base_llm() {
    let client = client();
    let handle: Arc<dyn BaseLLM> = Arc::new(llm(&client));
    assert!(handle.as_any().is::<RemoteLLM>());
}

#[test]
fn test_it_names_itself_distinctly() {
    let client = client();
    assert_eq!(llm(&client).llm_type(), "smartloop-remote");
}

#[test]
fn test_it_carries_the_project() {
    let client = client();
    assert_eq!(llm(&client).project_id(), "alpha");
}

/// The orchestrator builds tagged prompts; those must still parse.
#[test]
fn test_it_inherits_the_prompt_parsing_every_llm_shares() {
    let client = client();
    let messages = llm(&client).create_messages_from_prompt("<system>be brief</system><user>hello</user>");
    assert_eq!(messages, vec![ChatMessage::new("system", "be brief"), ChatMessage::new("user", "hello")]);
}

// ----------------------------------------------------------------- generation

#[test]
fn test_invoke_reaches_the_host() {
    let client = client();
    assert_eq!(llm(&client).invoke("who?", None, None), "answered who?");
    assert!(client.called("generate", "who?", None));
}

#[test]
fn test_an_empty_answer_matches_the_local_contract() {
    let client = client();
    *client.generate_reply.write() = Some("   ".into());
    assert_eq!(llm(&client).invoke("x", None, None), "No response generated");
}

/// A DAG node treats a bad answer as a dead branch; an error there would take
/// down a whole plan over one step.
#[test]
fn test_a_failure_is_returned_not_raised() {
    let client = client();
    *client.fail_with.write() = Some("host is gone".into());
    assert!(llm(&client).invoke("x", None, None).contains("Error generating response"));
}

#[test]
fn test_the_eot_marker_is_trimmed_as_it_is_locally() {
    let client = client();
    *client.generate_reply.write() = Some("the answer<|eot_id|>".into());
    assert_eq!(llm(&client).invoke("x", None, None), "the answer");
}

#[test]
fn test_streaming_yields_generation_chunks() {
    let client = client();
    let handle = llm(&client);
    let chunks: Vec<String> = handle.stream("go", CallOptions::default()).map(|c| c.text).collect();
    assert_eq!(chunks.concat(), "thinking done");
}

#[test]
fn test_cancellation_travels_with_the_stream() {
    let client = client();
    let handle = llm(&client);
    let cancelled: CancelEvent = Arc::new(AtomicBool::new(false));
    handle.set_cancelled(Some(cancelled.clone()));

    let mut produced = Vec::new();
    for chunk in handle.stream("go", CallOptions::default()) {
        produced.push(chunk.text);
        cancelled.store(true, Ordering::SeqCst);
    }

    assert!(produced.len() < client.chunks.len());
}

// ------------------------------------------------ capability stays on the host

#[test]
fn test_the_capability_is_sent_with_the_request() {
    let client = client();
    let handle = remote(&client, "", Some(ModelCapability::Vision.value()));

    handle.invoke("describe this", None, None);

    assert!(client.called("generate", "describe this", Some("vision")));
}

#[test]
fn test_no_capability_means_the_default_model() {
    let client = client();
    llm(&client).invoke("x", None, None);
    assert!(client.called("generate", "x", None));
}

#[test]
fn test_the_context_size_is_asked_for_this_capability() {
    let client = client();
    let handle = remote(&client, "", Some(ModelCapability::Reasoning.value()));

    handle.get_max_tokens();

    assert!(client.called("response_max_context_tokens", "", Some("reasoning")));
}

// ------------------------------------------ what the user is told the model is
//
// The progress SSE names the answering model; it must not name a type.  The
// regression: "Selected RemoteLLM for the response", because the label
// helpers read GGUF metadata off the local model — which a remote handle does
// not have — and fell through to the type name.

#[test]
fn test_the_handle_reports_the_hosts_label() {
    let client = client();
    assert_eq!(llm(&client).model_label(), "sl-mini (2B)");
}

#[test]
fn test_the_service_helper_uses_it() {
    let client = client();
    assert_eq!(ModelService::model_label(&llm(&client)), "sl-mini (2B)");
}

#[test]
fn test_the_router_helper_uses_it() {
    let client = client();
    assert_eq!(ModelRouter::model_label(&llm(&client)), "sl-mini (2B)");
}

#[test]
fn test_the_label_is_asked_for_this_capability() {
    let client = client();
    remote(&client, "", Some(ModelCapability::Vision.value())).model_label();
    assert!(client.called("model_label", "", Some("vision")));
}

#[test]
fn test_the_sse_message_names_the_model() {
    let client = client();
    let handle: Arc<dyn BaseLLM> = Arc::new(llm(&client));
    let messages: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = messages.clone();
    let callback: ProgressCallback = Arc::new(move |_step: &str, _status: &str, message: &str| {
        sink.lock().push(message.to_string());
    });
    let router = ModelRouter::new(handle.clone(), handle.clone(), ModelRouterOptions::default());
    router.begin_request(Some(callback));

    router.emit_model_selected(None, handle.as_ref(), Some("the response"));

    let messages = messages.lock().clone();
    assert_eq!(messages, vec!["Selected sl-mini (2B) for the response"]);
    assert!(!messages[0].contains("RemoteLLM"));
}

/// Identical to the single-process SSE: the label must come from the same
/// `llm_for_response` chain the answer does — pinned model, capability model,
/// large fallback, default — because that chain decides who writes it.
#[test]
fn test_the_label_is_the_model_that_actually_answers() {
    let svc = ModelService::new_bare();
    let chosen: Arc<dyn BaseLLM> = FakeLLM::labelled("Some Model", "7B").dyn_llm();
    let asked: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));

    let seen = asked.clone();
    let answer = chosen.clone();
    let mut hooks = ModelServiceHooks::default();
    hooks.llm_for_response = Some(Arc::new(move |capability: Option<&str>| {
        seen.lock().push(capability.map(str::to_string));
        Some(answer.clone())
    }));
    svc.set_hooks(hooks);

    let label = svc.response_model_label(Some(ModelCapability::Reasoning.value()));

    // Same selection call the response makes...
    assert_eq!(*asked.lock(), vec![Some("reasoning".to_string())]);
    // ...and the same label the SSE would have carried for it.
    assert_eq!(label, ModelService::model_label(chosen.as_ref()));
}

/// When the capability model is unavailable the answer falls back — and the
/// label has to follow it, or the user is told the wrong model.
#[test]
fn test_the_label_follows_the_fallback_rather_than_the_request() {
    let svc = ModelService::new_bare();
    let fallback: Arc<dyn BaseLLM> = FakeLLM::labelled("Default Model", "2B").dyn_llm();
    let answer = fallback.clone();
    let mut hooks = ModelServiceHooks::default();
    hooks.llm_for_response = Some(Arc::new(move |_capability: Option<&str>| Some(answer.clone())));
    svc.set_hooks(hooks);

    assert_eq!(
        svc.response_model_label(Some(ModelCapability::Vision.value())),
        ModelService::model_label(fallback.as_ref())
    );
}

/// The explicit label must not shadow the existing path.
#[test]
fn test_a_local_llm_still_reads_its_gguf_metadata() {
    let local = FakeLLM::labelled("Slp Orchestra Mini", "2B");
    assert_eq!(ModelRouter::model_label(local.as_ref()), "sl-mini (2B)");
}

#[test]
fn test_an_unreachable_host_falls_back_rather_than_raising() {
    let client = client();
    *client.fail_with.write() = Some("gone".into());
    assert_eq!(ModelRouter::model_label(&llm(&client)), "RemoteLLM");
}

// --------------------------------------------------------------------- sizing

#[test]
fn test_tokens_are_counted_by_the_host() {
    let client = client();
    assert_eq!(llm(&client).get_num_tokens("some text"), 42);
    assert!(client.called("count_tokens", "some text", None));
}

#[test]
fn test_the_context_window_comes_from_the_host() {
    let client = client();
    assert_eq!(llm(&client).get_max_tokens(), 8192);
}

#[test]
fn test_an_unreachable_host_counts_zero_rather_than_raising() {
    let client = client();
    *client.fail_with.write() = Some("gone".into());
    assert_eq!(llm(&client).get_num_tokens("x"), 0);
}

#[test]
fn test_an_unreachable_host_falls_back_to_a_usable_window() {
    let client = client();
    *client.fail_with.write() = Some("gone".into());
    assert_eq!(llm(&client).get_max_tokens(), DEFAULT_MAX_CONTEXT_TOKENS);
}

#[test]
fn test_a_host_reporting_nothing_falls_back() {
    let client = client();
    *client.context.write() = 0;
    assert_eq!(llm(&client).get_max_tokens(), DEFAULT_MAX_CONTEXT_TOKENS);
}

#[test]
fn test_the_template_is_left_to_whoever_owns_the_model() {
    let client = client();
    let messages = vec![ChatMessage::new("user", "hi")];
    assert_eq!(llm(&client).apply_chat_template(messages.clone()), messages);
}

// ------------------------------------------- the service hands out remote handles
//
// A child must not answer "no model" and then load one of its own.

fn service(client: &Arc<FakeClient>) -> Arc<ModelService> {
    let svc = ModelService::new_bare();
    svc.set_model_client(Some(client.clone()));
    *svc.llm.write() = None;
    *svc.model.write() = None;
    *svc.model_name.write() = None;
    svc
}

fn remote_capability(llm: &Arc<dyn BaseLLM>) -> Option<String> {
    llm.as_any().downcast_ref::<RemoteLLM>().and_then(|r| r.capability().map(str::to_string))
}

#[test]
fn test_it_recognises_a_remote_device() {
    let client = client();
    let svc = service(&client);
    let device = svc.remote_device().expect("a remote device");
    assert!(device.as_any().is::<FakeClient>());
}

#[test]
fn test_unloading_auxiliary_models_releases_them_on_the_host() {
    let client = client();
    let svc = service(&client);

    svc.unload_auxiliary_models(false);

    assert!(client.called("unload_auxiliary_models", "false", None));
}

#[test]
fn test_a_failed_host_unload_does_not_fail_the_response() {
    let client = client();
    let svc = service(&client);
    *client.fail_with.write() = Some("host gone".into());

    svc.unload_auxiliary_models(true);

    assert!(client.called("unload_auxiliary_models", "true", None));
}

#[test]
fn test_an_in_process_client_is_not_remote() {
    let svc = ModelService::new_bare();
    let local: Arc<dyn LocalModelService> = ModelService::new_bare();
    svc.set_model_client(Some(Arc::new(InProcessModelClient::new(local))));

    assert!(svc.remote_device().is_none());
}

/// The regression that produced "[orchestrator] BYPASSED (llm=False)":
/// answering with the host's half alone told the caller there was nothing to
/// load, so the orchestrator fell back to raw generation.
#[test]
fn test_is_loaded_is_false_until_this_process_holds_a_handle() {
    let client = client();
    let svc = service(&client);
    *client.loaded.write() = Some(true);
    *svc.llm.write() = None;

    assert!(!svc.is_loaded());
}

#[test]
fn test_is_loaded_is_true_once_both_halves_hold() {
    let client = client();
    let svc = service(&client);
    *client.loaded.write() = Some(true);
    *svc.llm.write() = Some(svc.remote_llm(None));

    assert!(svc.is_loaded());
}

#[test]
fn test_is_loaded_is_false_when_the_host_has_no_weights() {
    let client = client();
    let svc = service(&client);
    *client.loaded.write() = Some(false);
    *svc.llm.write() = Some(svc.remote_llm(None));

    assert!(!svc.is_loaded());
}

#[test]
fn test_is_loaded_is_false_when_the_host_is_unreachable() {
    let client = client();
    let svc = service(&client);
    *client.loaded.write() = None;
    *svc.llm.write() = Some(svc.remote_llm(None));

    assert!(!svc.is_loaded());
}

#[test]
fn test_token_counting_goes_to_the_host() {
    let client = client();
    let svc = service(&client);
    assert_eq!(svc.count_tokens("hello there"), 42);
}

#[test]
fn test_the_context_window_goes_to_the_host() {
    let client = client();
    let svc = service(&client);
    *client.max_context.write() = Some(4096);
    assert_eq!(svc.get_max_context_tokens(), 4096);
}

#[test]
fn test_a_capability_request_becomes_a_remote_handle() {
    let client = client();
    let svc = service(&client);

    let llm = svc.get_llm_for_capability(ModelCapability::Vision.value()).expect("a handle");

    assert!(llm.as_any().is::<RemoteLLM>());
    assert_eq!(remote_capability(&llm).as_deref(), Some("vision"));
}

/// A writer collecting formatted log lines.
#[derive(Clone, Default)]
struct LogBuffer(Arc<Mutex<Vec<u8>>>);

impl Write for LogBuffer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Never degrade to raw generation over a handle we can just build.
#[test]
#[serial_test::serial]
fn test_orchestration_binds_a_handle_rather_than_bypassing() {
    let _guard = global_lock();
    let tmp = tempfile::tempdir().expect("tempdir");
    Workspace::set_config_file(Some(tmp.path().join("config.json").to_string_lossy().into_owned()));

    let client = client();
    let svc = service(&client);
    // Workspace model present, LLM not.
    let workspace = Arc::new(RwLock::new(Workspace::new("ws-test")));
    *svc.model.write() = Some(Arc::new(Model::new("model-test", workspace, None)));
    *svc.llm.write() = None;
    *svc.model_name.write() = None;

    let logs = LogBuffer::default();
    let writer = logs.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    // Reach the guard at the top of orchestrate; the rest of the pipeline is
    // not what this asserts.
    tracing::subscriber::with_default(subscriber, || {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            svc.orchestrate("a question", None, None, None, false, None, None, None, None, None)
        }));
    });
    Workspace::set_config_file(None);

    let llm = svc.llm.read().clone().expect("a handle was bound");
    assert!(llm.as_any().is::<RemoteLLM>());
    let text = String::from_utf8_lossy(&logs.0.lock()).into_owned();
    assert!(!text.contains("BYPASSED — model/llm not loaded"), "{text}");
}

#[test]
fn test_the_large_model_becomes_a_remote_handle() {
    let client = client();
    let svc = service(&client);

    let llm = svc.get_large_llm().expect("a handle");

    assert!(llm.as_any().is::<RemoteLLM>());
    assert_eq!(remote_capability(&llm).as_deref(), Some("reasoning"));
}
