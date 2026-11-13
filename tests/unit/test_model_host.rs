//! Port of `tests/unit/test_model_host.py`.
//!
//! A project agent reaches the generative model through a client, not
//! directly.
//!
//! The model is the one thing that cannot be per-project: same-device weight
//! copies serialize in the driver and each carries its own KV cache, so N
//! projects would mean N resident copies competing for one device. The split
//! therefore runs between project work and inference — these tests cover the
//! seam and the gRPC transport across it.
//!
//! Process-wide state (the project pin, `SLP_MODEL_HOST`, the device-owner
//! flag and the AF_UNIX switch) is shared by every test in the binary, so the
//! tests that touch it or depend on it run `#[serial]`.

use std::any::Any;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use ndarray::{Array2, s};
use parking_lot::Mutex;
use serial_test::serial;

use smartloop::llms::base_llm::{CancelEvent, GenerationChunk};
use smartloop::model_host::client::{
    get_model_client, grpc_target, default_socket_path, parse_address, set_device_owner,
    set_global_encode_hook, set_has_unix_sockets, EncodeHook, InProcessModelClient,
    LocalModelService, ModelClient, ModelHostError, RemoteModelClient, TextStream,
    HAS_UNIX_SOCKETS,
};
use smartloop::model_host::host::{unlink_stale_socket, ModelHost, ModelHostServicer};
use smartloop::model_host::pb;
use smartloop::model_host::pb::model_host_server::ModelHost as _;

// ---------------------------------------------------------------- fakes

/// What the fake device was asked to do, in the shape the Python tuples had.
#[derive(Debug, Clone, PartialEq)]
enum Call {
    Generate(String, Option<String>, Option<String>),
    Stream(String, Option<String>, Option<String>),
    EncodeTexts(Vec<String>, Option<String>),
    LoadModel,
    UnloadAuxiliaryModels(bool),
    UnloadEmbeddingModel,
    UnloadModelByName(String),
}

type LabelFn = Box<dyn Fn(Option<&str>) -> String + Send + Sync>;
type EncodeFn = Box<dyn Fn(&[String]) -> Option<Array2<f32>> + Send + Sync>;
type RefreshFn = Box<dyn Fn(Option<&str>) -> anyhow::Result<bool> + Send + Sync>;

/// Stands in for the process that owns the GPU.
struct FakeDevice {
    calls: Mutex<Vec<Call>>,
    chunks: Mutex<Vec<String>>,
    chunk_delay: Mutex<Duration>,
    fail_with: Mutex<Option<String>>,
    loaded: AtomicBool,
    yielded: AtomicUsize,
    options: Mutex<(Option<usize>, Option<bool>)>,
    /// Replaces `model_label` (Python assigned `host.client.model_label`).
    label: Mutex<Option<LabelFn>>,
    /// Replaces `encode_texts` (Python assigned `device.encode_texts`).
    encode: Mutex<Option<EncodeFn>>,
    /// Replaces `refresh_project` (Python assigned `device.refresh_project`).
    refresh: Mutex<Option<RefreshFn>>,
}

impl FakeDevice {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            calls: Mutex::new(Vec::new()),
            chunks: Mutex::new(vec!["Hello".into(), ", ".into(), "world".into()]),
            chunk_delay: Mutex::new(Duration::ZERO),
            fail_with: Mutex::new(None),
            loaded: AtomicBool::new(true),
            yielded: AtomicUsize::new(0),
            options: Mutex::new((None, None)),
            label: Mutex::new(None),
            encode: Mutex::new(None),
            refresh: Mutex::new(None),
        })
    }

    fn calls(&self) -> Vec<Call> {
        self.calls.lock().clone()
    }

    fn set_chunks(&self, chunks: Vec<String>) {
        *self.chunks.lock() = chunks;
    }

    fn fail_with(&self, message: Option<&str>) {
        *self.fail_with.lock() = message.map(str::to_string);
    }

    fn set_label(&self, label: impl Fn(Option<&str>) -> String + Send + Sync + 'static) {
        *self.label.lock() = Some(Box::new(label));
    }
}

/// A `FakeDevice` streamed lazily, checking `cancelled` before each chunk.
struct FakeStream {
    device: Arc<FakeDevice>,
    chunks: std::vec::IntoIter<String>,
    cancelled: Option<CancelEvent>,
    delay: Duration,
}

impl Iterator for FakeStream {
    type Item = anyhow::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        let chunk = self.chunks.next()?;
        if self.cancelled.as_ref().is_some_and(|c| c.load(Ordering::SeqCst)) {
            return None;
        }
        if !self.delay.is_zero() {
            std::thread::sleep(self.delay);
        }
        self.device.yielded.fetch_add(1, Ordering::SeqCst);
        Some(Ok(chunk))
    }
}

/// `ModelClient` for an `Arc<FakeDevice>`, so a stream can hold the device.
#[derive(Clone)]
struct Device(Arc<FakeDevice>);

impl std::ops::Deref for Device {
    type Target = FakeDevice;
    fn deref(&self) -> &FakeDevice {
        &self.0
    }
}

impl ModelClient for Device {
    fn generate(
        &self,
        prompt: &str,
        model_name: Option<&str>,
        capability: Option<&str>,
        max_tokens: Option<usize>,
        thinking: Option<bool>,
    ) -> anyhow::Result<String> {
        self.calls.lock().push(Call::Generate(
            prompt.to_string(),
            model_name.map(str::to_string),
            capability.map(str::to_string),
        ));
        *self.options.lock() = (max_tokens, thinking);
        if let Some(message) = self.fail_with.lock().clone() {
            anyhow::bail!(message);
        }
        Ok(format!("answer to {prompt}"))
    }

    fn stream<'a>(
        &'a self,
        prompt: &str,
        model_name: Option<&str>,
        capability: Option<&str>,
        cancelled: Option<CancelEvent>,
        _thinking: Option<bool>,
    ) -> anyhow::Result<TextStream<'a>> {
        self.calls.lock().push(Call::Stream(
            prompt.to_string(),
            model_name.map(str::to_string),
            capability.map(str::to_string),
        ));
        if let Some(message) = self.fail_with.lock().clone() {
            anyhow::bail!(message);
        }
        Ok(Box::new(FakeStream {
            device: self.0.clone(),
            chunks: self.chunks.lock().clone().into_iter(),
            cancelled,
            delay: *self.chunk_delay.lock(),
        }))
    }

    fn encode_texts(
        &self,
        texts: &[String],
        base_dir: Option<&str>,
    ) -> anyhow::Result<Option<Array2<f32>>> {
        if let Some(encode) = self.encode.lock().as_ref() {
            return Ok(encode(texts));
        }
        self.calls
            .lock()
            .push(Call::EncodeTexts(texts.to_vec(), base_dir.map(str::to_string)));
        if texts.is_empty() {
            return Ok(None);
        }
        Ok(Some(arange(texts.len(), 3)))
    }

    fn is_loaded(&self) -> anyhow::Result<bool> {
        Ok(self.loaded.load(Ordering::SeqCst))
    }

    fn model_name(&self) -> anyhow::Result<Option<String>> {
        Ok(Some("sl-mini".to_string()))
    }

    fn model_label(&self, capability: Option<&str>) -> anyhow::Result<String> {
        if let Some(label) = self.label.lock().as_ref() {
            return Ok(label(capability));
        }
        // Deliberately not a real model name: this asserts the label crosses
        // the wire intact, not what any particular model is called.
        Ok(match capability {
            Some(cap) => format!("host-chosen-model [{cap}]"),
            None => "host-chosen-model".to_string(),
        })
    }

    fn count_tokens(&self, text: &str) -> anyhow::Result<usize> {
        Ok(text.split_whitespace().count())
    }

    fn max_context_tokens(&self) -> anyhow::Result<usize> {
        Ok(8192)
    }

    fn response_max_context_tokens(&self, capability: Option<&str>) -> anyhow::Result<usize> {
        Ok(if capability.is_some() { 4096 } else { 8192 })
    }

    fn has_capability_model(&self, capability: &str) -> anyhow::Result<bool> {
        Ok(capability == "vision")
    }

    fn idle_seconds(&self) -> anyhow::Result<f64> {
        Ok(12.5)
    }

    fn load_model(&self) -> anyhow::Result<Option<String>> {
        self.calls.lock().push(Call::LoadModel);
        Ok(Some("sl-mini".to_string()))
    }

    fn unload_auxiliary_models(&self, include_embedding: bool) -> anyhow::Result<()> {
        self.calls.lock().push(Call::UnloadAuxiliaryModels(include_embedding));
        Ok(())
    }

    fn unload_embedding_model(&self) -> anyhow::Result<()> {
        self.calls.lock().push(Call::UnloadEmbeddingModel);
        Ok(())
    }

    fn unload_model_by_name(&self, model_name: &str) -> anyhow::Result<bool> {
        self.calls.lock().push(Call::UnloadModelByName(model_name.to_string()));
        Ok(true)
    }

    fn refresh_project(&self, project_id: Option<&str>) -> anyhow::Result<bool> {
        match self.refresh.lock().as_ref() {
            Some(refresh) => refresh(project_id),
            None => Ok(false),
        }
    }

    fn type_name(&self) -> String {
        "FakeDevice".to_string()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `np.arange(rows * cols, dtype=float32).reshape(rows, cols)`.
fn arange(rows: usize, cols: usize) -> Array2<f32> {
    Array2::from_shape_vec((rows, cols), (0..rows * cols).map(|v| v as f32).collect()).unwrap()
}

fn device() -> Device {
    Device(FakeDevice::new())
}

fn as_client(device: &Device) -> Arc<dyn ModelClient> {
    Arc::new(device.clone())
}

/// The `host` + `client` fixtures: a host on a socket in a temp dir, and a
/// remote client connected to it.  Both are closed on drop.
struct Hosted {
    client: RemoteModelClient,
    host: ModelHost,
    device: Device,
    _tmp: tempfile::TempDir,
}

impl Hosted {
    fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let device = device();
        let host = ModelHost::new(as_client(&device), path_str(&tmp.path().join("model-host.sock")));
        host.start().unwrap();
        let client = RemoteModelClient::with_timeout(host.address(), 15.0);
        Self { client, host, device, _tmp: tmp }
    }
}

impl Drop for Hosted {
    fn drop(&mut self) {
        self.client.close();
        self.host.stop();
    }
}

fn path_str(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn collect(stream: anyhow::Result<TextStream<'_>>) -> anyhow::Result<Vec<String>> {
    stream?.collect()
}

fn set_env(key: &str, value: &str) {
    // SAFETY: every test touching the environment runs #[serial].
    unsafe { std::env::set_var(key, value) };
}

fn remove_env(key: &str) {
    // SAFETY: every test touching the environment runs #[serial].
    unsafe { std::env::remove_var(key) };
}

/// Restores `SLP_MODEL_HOST`, the AF_UNIX switch and the device-owner flag
/// (Python's `monkeypatch` undo).
struct GlobalsGuard {
    model_host: Option<String>,
}

impl GlobalsGuard {
    fn new() -> Self {
        Self { model_host: std::env::var("SLP_MODEL_HOST").ok() }
    }
}

impl Drop for GlobalsGuard {
    fn drop(&mut self) {
        match &self.model_host {
            Some(value) => set_env("SLP_MODEL_HOST", value),
            None => remove_env("SLP_MODEL_HOST"),
        }
        set_has_unix_sockets(HAS_UNIX_SOCKETS);
        set_global_encode_hook(None);
        smartloop::workspace::context::pin_project(None);
    }
}

fn assert_err_contains<T: std::fmt::Debug>(result: anyhow::Result<T>, needle: &str) {
    let err = result.expect_err("expected an error");
    let host_error = err.downcast_ref::<ModelHostError>();
    assert!(host_error.is_some(), "expected ModelHostError, got {err:?}");
    assert!(err.to_string().contains(needle), "{needle:?} not in {err}");
}

// ---------------------------------------------------------- TestAddresses

#[test]
fn test_parsing() {
    let cases: [(&str, (&str, Option<u32>)); 5] = [
        ("/tmp/model-host.sock", ("/tmp/model-host.sock", None)),
        ("unix:///tmp/a.sock", ("/tmp/a.sock", None)),
        ("unix:/tmp/a.sock", ("/tmp/a.sock", None)),
        ("127.0.0.1:9000", ("127.0.0.1", Some(9000))),
        (":9000", ("127.0.0.1", Some(9000))),
    ];
    for (address, (host, port)) in cases {
        assert_eq!(parse_address(address), (host.to_string(), port), "{address}");
    }
}

/// How gRPC is asked to reach an address: unix-prefixed or host:port.
#[test]
fn test_the_grpc_target() {
    let cases = [
        ("/tmp/model-host.sock", "unix:/tmp/model-host.sock"),
        ("unix:///tmp/a.sock", "unix:/tmp/a.sock"),
        ("127.0.0.1:9000", "127.0.0.1:9000"),
        (":9000", "127.0.0.1:9000"),
    ];
    for (address, target) in cases {
        assert_eq!(grpc_target(address), target, "{address}");
    }
}

#[test]
#[serial]
fn test_the_default_lives_in_the_home() {
    assert_eq!(default_socket_path("/home/x/.smartloop"), "/home/x/.smartloop/model-host.sock");
}

// ----------------------------------------------------- TestTheDefaultUsesGrpc
//
// A host address selects gRPC; its absence means there is no host to use.
// The process holding the device must never resolve a client to its own
// socket.

#[test]
#[serial]
fn test_no_host_configured_stays_in_process() {
    let _guard = GlobalsGuard::new();
    remove_env("SLP_MODEL_HOST");
    let client = get_model_client(Some(Arc::new(FakeService::new()))).unwrap();
    assert!(client.as_any().downcast_ref::<InProcessModelClient>().is_some());
}

#[test]
#[serial]
fn test_a_configured_host_selects_the_socket_client() {
    let _guard = GlobalsGuard::new();
    set_env("SLP_MODEL_HOST", "/tmp/somewhere.sock");
    let client = get_model_client(Some(Arc::new(FakeService::new()))).unwrap();
    let remote = client.as_remote().expect("a RemoteModelClient");
    assert_eq!(remote.address, "/tmp/somewhere.sock");
}

/// A blank value is how an unset variable arrives through a shell.
#[test]
#[serial]
fn test_an_empty_setting_is_no_setting() {
    let _guard = GlobalsGuard::new();
    set_env("SLP_MODEL_HOST", "   ");
    let client = get_model_client(Some(Arc::new(FakeService::new()))).unwrap();
    assert!(client.as_any().downcast_ref::<InProcessModelClient>().is_some());
}

/// No model service to fall back on: the host's own address is the answer.
#[test]
#[serial]
fn test_a_process_without_a_device_resolves_the_default_socket() {
    let _guard = GlobalsGuard::new();
    remove_env("SLP_MODEL_HOST");
    let client = get_model_client(None).unwrap();
    let remote = client.as_remote().expect("a RemoteModelClient");
    assert!(remote.address.ends_with("model-host.sock") || remote.address.starts_with("127.0.0.1:"));
}

// ---------------------------------------------------- TestTheInProcessClient
//
// It must call exactly what the caller used to call directly.

#[derive(Debug, Clone, PartialEq)]
enum ServiceCall {
    GenerateCompletion(String, Option<String>, Option<String>),
    StreamCompletion(String, Option<String>, Option<String>),
    LoadModel,
    UnloadAuxiliaryModels(bool),
    UnloadEmbeddingModel,
    UnloadModelByName(String),
    Refresh(Option<String>),
}

/// The `_Service` stand-in for `ModelService`.
struct FakeService {
    calls: Mutex<Vec<ServiceCall>>,
    options: Mutex<(Option<usize>, Option<bool>)>,
}

impl FakeService {
    fn new() -> Self {
        Self { calls: Mutex::new(Vec::new()), options: Mutex::new((None, None)) }
    }
}

impl LocalModelService for FakeService {
    fn refresh_project_state(&self, project_id: Option<&str>) -> anyhow::Result<bool> {
        self.calls.lock().push(ServiceCall::Refresh(project_id.map(str::to_string)));
        Ok(true)
    }

    fn generate_completion(
        &self,
        prompt: &str,
        model_name: Option<&str>,
        capability: Option<&str>,
        max_tokens: Option<usize>,
        thinking: Option<bool>,
    ) -> anyhow::Result<String> {
        self.calls.lock().push(ServiceCall::GenerateCompletion(
            prompt.to_string(),
            model_name.map(str::to_string),
            capability.map(str::to_string),
        ));
        *self.options.lock() = (max_tokens, thinking);
        Ok("done".to_string())
    }

    fn stream_completion<'a>(
        &'a self,
        prompt: &str,
        model_name: Option<&str>,
        _cancelled: Option<CancelEvent>,
        capability: Option<&str>,
        _thinking: Option<bool>,
    ) -> anyhow::Result<Box<dyn Iterator<Item = anyhow::Result<GenerationChunk>> + 'a>> {
        self.calls.lock().push(ServiceCall::StreamCompletion(
            prompt.to_string(),
            model_name.map(str::to_string),
            capability.map(str::to_string),
        ));
        Ok(Box::new(
            vec![
                Ok(GenerationChunk { text: "a".into() }),
                Ok(GenerationChunk { text: "b".into() }),
            ]
            .into_iter(),
        ))
    }

    fn is_loaded(&self) -> bool {
        true
    }

    fn resident_model_name(&self) -> Option<String> {
        Some("sl-mini".to_string())
    }

    fn response_model_label(&self, _capability: Option<&str>) -> String {
        "the-real-model (7B)".to_string()
    }

    fn count_tokens(&self, _text: &str) -> usize {
        7
    }

    fn get_max_context_tokens(&self) -> usize {
        8192
    }

    fn get_response_max_context_tokens(&self, _capability: Option<&str>) -> usize {
        4096
    }

    fn has_capability_model(&self, _capability: &str) -> bool {
        true
    }

    fn idle_seconds(&self) -> f64 {
        3.0
    }

    fn load_model(&self) -> anyhow::Result<()> {
        self.calls.lock().push(ServiceCall::LoadModel);
        Ok(())
    }

    fn unload_auxiliary_models(&self, include_embedding: bool) {
        self.calls.lock().push(ServiceCall::UnloadAuxiliaryModels(include_embedding));
    }

    fn unload_embedding_model(&self) {
        self.calls.lock().push(ServiceCall::UnloadEmbeddingModel);
    }

    fn unload_model_by_name(&self, model_name: &str) -> bool {
        self.calls.lock().push(ServiceCall::UnloadModelByName(model_name.to_string()));
        true
    }
}

#[test]
fn test_generate_delegates() {
    let service = Arc::new(FakeService::new());
    let client = InProcessModelClient::new(service.clone());
    assert_eq!(client.generate("hi", None, Some("vision"), None, None).unwrap(), "done");
    assert_eq!(
        *service.calls.lock(),
        vec![ServiceCall::GenerateCompletion("hi".into(), None, Some("vision".into()))]
    );
}

/// The wire carries text, so both clients yield the same shape.
#[test]
fn test_stream_yields_plain_text() {
    let client = InProcessModelClient::new(Arc::new(FakeService::new()));
    assert_eq!(collect(client.stream("hi", None, None, None, None)).unwrap(), vec!["a", "b"]);
}

#[test]
fn test_the_questions_delegate() {
    let client = InProcessModelClient::new(Arc::new(FakeService::new()));
    assert!(client.is_loaded().unwrap());
    assert_eq!(client.model_name().unwrap().as_deref(), Some("sl-mini"));
    assert_eq!(client.count_tokens("a b c").unwrap(), 7);
    assert_eq!(client.max_context_tokens().unwrap(), 8192);
    assert_eq!(client.response_max_context_tokens(Some("vision")).unwrap(), 4096);
    assert!(client.has_capability_model("vision").unwrap());
    assert_eq!(client.idle_seconds().unwrap(), 3.0);
    assert_eq!(client.model_label(None).unwrap(), "the-real-model (7B)");
}

#[test]
fn test_load_reports_the_resident_model() {
    let service = Arc::new(FakeService::new());
    let client = InProcessModelClient::new(service.clone());
    assert_eq!(client.load_model().unwrap().as_deref(), Some("sl-mini"));
    assert!(service.calls.lock().contains(&ServiceCall::LoadModel));
}

// ------------------------------------------------------ TestTheGrpcTransport

#[test]
#[serial]
fn test_generate() {
    let hosted = Hosted::new();
    assert_eq!(hosted.client.generate("who?", None, Some("general"), None, None).unwrap(), "answer to who?");
    assert!(hosted.device.calls().contains(&Call::Generate("who?".into(), None, Some("general".into()))));
}

#[test]
#[serial]
fn test_stream() {
    let hosted = Hosted::new();
    assert_eq!(
        collect(hosted.client.stream("tell me", None, None, None, None)).unwrap(),
        vec!["Hello", ", ", "world"]
    );
}

#[test]
#[serial]
fn test_streamed_text_keeps_its_newlines() {
    let hosted = Hosted::new();
    let chunks: Vec<String> =
        vec!["first\n".into(), "\nsecond".into(), "{\"json\": \"like\"}".into()];
    hosted.device.set_chunks(chunks.clone());
    assert_eq!(collect(hosted.client.stream("x", None, None, None, None)).unwrap(), chunks);
}

#[test]
#[serial]
fn test_the_questions_cross_the_wire() {
    let hosted = Hosted::new();
    let client = &hosted.client;
    assert!(client.is_loaded().unwrap());
    assert_eq!(client.model_name().unwrap().as_deref(), Some("sl-mini"));
    assert_eq!(client.model_label(None).unwrap(), "host-chosen-model");
    assert_eq!(client.model_label(Some("vision")).unwrap(), "host-chosen-model [vision]");
    assert_eq!(client.count_tokens("one two three").unwrap(), 3);
    assert_eq!(client.max_context_tokens().unwrap(), 8192);
    assert_eq!(client.response_max_context_tokens(Some("vision")).unwrap(), 4096);
    assert!(client.has_capability_model("vision").unwrap());
    assert!(!client.has_capability_model("coding").unwrap());
    assert_eq!(client.idle_seconds().unwrap(), 12.5);
}

#[test]
#[serial]
fn test_embeddings_cross_the_wire_intact() {
    let hosted = Hosted::new();
    let result = hosted
        .client
        .encode_texts(&["a".to_string(), "b".to_string()], None)
        .unwrap()
        .expect("an embedding matrix");
    assert_eq!(result, arange(2, 3));
}

/// A per-call timeout is the caller's, while the default stays the client's.
#[test]
fn test_encode_texts_can_impose_its_own_deadline() {
    let client = RemoteModelClient::with_timeout("127.0.0.1:0", 15.0); // never connects
    let seen: Arc<Mutex<Option<Option<f64>>>> = Arc::new(Mutex::new(None));

    let recorder = seen.clone();
    let hook: EncodeHook = Arc::new(move |_request, timeout| {
        *recorder.lock() = Some(timeout);
        let data: Vec<u8> = [0.1f32, 0.2, 0.3, 0.4].iter().flat_map(|v| v.to_ne_bytes()).collect();
        Ok(pb::EncodeTextsResponse { shape: vec![1, 4], data })
    });
    client.set_encode_hook(Some(hook));

    assert!(client.encode_texts_with_timeout(&["hello".to_string()], None, Some(7.5)).unwrap().is_some());
    assert_eq!(*seen.lock(), Some(Some(7.5)));

    client.encode_texts(&["default".to_string()], None).unwrap();
    assert_eq!(*seen.lock(), Some(None)); // unary applies the client-wide default
}

#[test]
#[serial]
fn test_a_non_contiguous_view_is_contiguised_by_the_host() {
    let hosted = Hosted::new();
    let full = Array2::from_shape_fn((4, 8), |(r, c)| ((r * 8 + c) as f32 * 0.37).sin());
    let original = full.slice_move(s![.., ..;2]);
    assert!(!original.is_standard_layout());
    let served = original.clone();
    *hosted.device.encode.lock() = Some(Box::new(move |_texts| Some(served.clone())));

    let client = RemoteModelClient::with_timeout(hosted.host.address(), 15.0);
    let texts: Vec<String> = ["x", "y", "z", "w"].iter().map(|t| t.to_string()).collect();
    let result = client.encode_texts(&texts, None);
    client.close();
    let result = result.unwrap().expect("an embedding matrix");
    assert_eq!(result.shape(), &[4, 4]);
    assert_eq!(result, original);
}

#[test]
#[serial]
fn test_an_unavailable_embedder_comes_back_as_none() {
    let hosted = Hosted::new();
    assert!(hosted.client.encode_texts(&[], None).unwrap().is_none());
}

#[test]
#[serial]
fn test_lifecycle_calls_reach_the_device() {
    let hosted = Hosted::new();
    assert_eq!(hosted.client.load_model().unwrap().as_deref(), Some("sl-mini"));
    hosted.client.unload_auxiliary_models(false).unwrap();
    hosted.client.unload_embedding_model().unwrap();
    assert!(hosted.client.unload_model_by_name("qwen3.5-2b").unwrap());
    let calls = hosted.device.calls();
    assert!(calls.contains(&Call::UnloadAuxiliaryModels(false)));
    assert!(calls.contains(&Call::UnloadEmbeddingModel));
    assert!(calls.contains(&Call::UnloadModelByName("qwen3.5-2b".into())));
}

#[test]
#[serial]
fn test_a_call_from_inside_an_executor_reaches_the_device() {
    let hosted = Hosted::new();
    // A route runs its body under `futures::executor::block_on`.
    let unloaded = futures::executor::block_on(async { hosted.client.unload_model_by_name("qwen3.5-2b") });
    assert!(unloaded.unwrap());
    assert!(hosted.device.calls().contains(&Call::UnloadModelByName("qwen3.5-2b".into())));
}

#[test]
#[serial]
fn test_many_requests_travel_on_one_channel() {
    let hosted = Hosted::new();
    for _ in 0..5 {
        hosted.client.generate("hi", None, None, None, None).unwrap();
    }
    let generated = hosted.device.calls().iter().filter(|c| matches!(c, Call::Generate(..))).count();
    assert_eq!(generated, 5);
}

// --------------------------------------------------------------- TestFailures

#[test]
#[serial]
fn test_a_device_error_surfaces_at_the_caller() {
    let hosted = Hosted::new();
    hosted.device.fail_with(Some("no model loaded"));
    assert_err_contains(hosted.client.generate("hi", None, None, None, None), "no model loaded");
}

#[test]
#[serial]
fn test_the_connection_survives_a_failed_request() {
    let hosted = Hosted::new();
    hosted.device.fail_with(Some("boom"));
    assert_err_contains(hosted.client.generate("hi", None, None, None, None), "boom");
    hosted.device.fail_with(None);
    assert_eq!(hosted.client.generate("again", None, None, None, None).unwrap(), "answer to again");
}

#[test]
#[serial]
fn test_a_stream_error_surfaces_at_the_caller() {
    let hosted = Hosted::new();
    hosted.device.fail_with(Some("no model loaded"));
    assert_err_contains(collect(hosted.client.stream("hi", None, None, None, None)), "no model loaded");
}

#[test]
#[serial]
fn test_an_unreachable_host_is_reported_clearly() {
    let tmp = tempfile::tempdir().unwrap();
    let client = RemoteModelClient::with_timeout(path_str(&tmp.path().join("not-there.sock")), 2.0);
    assert_err_contains(client.is_loaded(), "unreachable");
}

#[test]
#[serial]
fn test_a_restarted_host_is_reconnected_to() {
    let tmp = tempfile::tempdir().unwrap();
    let device = device();
    let address = path_str(&tmp.path().join("host.sock"));
    let first = ModelHost::new(as_client(&device), address.clone());
    first.start().unwrap();
    let client = RemoteModelClient::with_timeout(address.clone(), 15.0);

    assert_eq!(client.generate("before", None, None, None, None).unwrap(), "answer to before");
    first.stop();

    assert_err_contains(client.generate("during", None, None, None, None), "");

    let second = ModelHost::new(as_client(&device), address);
    second.start().unwrap();
    let after = client.generate("after", None, None, None, None);
    second.stop();
    client.close();
    assert_eq!(after.unwrap(), "answer to after");
}

// ----------------------------------------------------------- TestCancellation

#[test]
#[serial]
fn test_a_cancelled_stream_stops_early() {
    let hosted = Hosted::new();
    hosted.device.set_chunks((0..50).map(|i| format!("chunk-{i}")).collect());
    *hosted.device.chunk_delay.lock() = Duration::from_millis(10);
    let cancelled: CancelEvent = Arc::new(AtomicBool::new(false));

    let mut received = Vec::new();
    for text in hosted.client.stream("long answer", None, None, Some(cancelled.clone()), None).unwrap() {
        received.push(text.unwrap());
        if received.len() == 3 {
            cancelled.store(true, Ordering::SeqCst);
        }
    }

    // Both ends stop: the client stops receiving, and — the point of
    // cancelling rather than just hanging up — the device stops being asked
    // for tokens it would compute for nobody.
    assert!(received.len() < 50);
    // Give the host a moment to observe the hang-up between chunks.
    std::thread::sleep(Duration::from_millis(100));
    assert!(hosted.device.yielded.load(Ordering::SeqCst) < 50);
}

#[test]
#[serial]
fn test_the_connection_is_usable_after_a_cancellation() {
    let hosted = Hosted::new();
    hosted.device.set_chunks((0..50).map(|i| format!("chunk-{i}")).collect());
    *hosted.device.chunk_delay.lock() = Duration::from_millis(10);
    let cancelled: CancelEvent = Arc::new(AtomicBool::new(false));

    for (index, _) in hosted.client.stream("long", None, None, Some(cancelled.clone()), None).unwrap().enumerate() {
        if index == 2 {
            cancelled.store(true, Ordering::SeqCst);
        }
    }

    assert_eq!(hosted.client.generate("next", None, None, None, None).unwrap(), "answer to next");
}

// ------------------------------------------ TestSelectionIsScopedToTheAskingProject
//
// Which model answers depends on the *requesting* project's settings.

#[test]
#[serial]
fn test_the_requesting_project_reaches_the_host() {
    use smartloop::workspace::context::{active_project_id, pin_project};

    let _guard = GlobalsGuard::new();
    let hosted = Hosted::new();
    let seen: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = seen.clone();
    hosted.device.set_label(move |_capability| {
        recorder.lock().push(active_project_id());
        "label".to_string()
    });

    pin_project(Some("beta"));
    let client = RemoteModelClient::with_timeout(hosted.host.address(), 15.0);
    let result = client.model_label(None);
    client.close();
    pin_project(None);
    result.unwrap();

    assert_eq!(*seen.lock(), vec![Some("beta".to_string())]);
}

#[test]
#[serial]
fn test_two_projects_are_not_confused_for_each_other() {
    use smartloop::workspace::context::{active_project_id, pin_project};

    let _guard = GlobalsGuard::new();
    let hosted = Hosted::new();
    hosted.device.set_label(|_capability| active_project_id().unwrap_or_else(|| "none".to_string()));

    let mut answers = std::collections::BTreeMap::new();
    for project_id in ["alpha", "beta"] {
        pin_project(Some(project_id));
        let client = RemoteModelClient::with_timeout(hosted.host.address(), 15.0);
        answers.insert(project_id.to_string(), client.model_label(None).unwrap());
        client.close();
    }
    pin_project(None);

    let expected: std::collections::BTreeMap<String, String> =
        [("alpha", "alpha"), ("beta", "beta")].iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    assert_eq!(answers, expected);
}

/// The supervisor's own calls keep resolving through its selection.
#[test]
#[serial]
fn test_an_unpinned_caller_binds_nothing() {
    use smartloop::workspace::context::{active_project_id, pin_project};

    let _guard = GlobalsGuard::new();
    let hosted = Hosted::new();
    hosted.device.set_label(|_capability| active_project_id().unwrap_or_else(|| "unbound".to_string()));

    pin_project(None);
    let client = RemoteModelClient::with_timeout(hosted.host.address(), 15.0);
    let result = client.model_label(None);
    client.close();
    assert_eq!(result.unwrap(), "unbound");
}

// ------------------------------------------------------------ TestConcurrency

/// Three project agents, three channels, one device.
#[test]
#[serial]
fn test_several_agents_share_one_host() {
    let hosted = Hosted::new();
    let address = hosted.host.address();
    let results: Arc<Mutex<std::collections::BTreeMap<usize, String>>> = Arc::default();
    let errors: Arc<Mutex<Vec<String>>> = Arc::default();
    let start = Arc::new(std::sync::Barrier::new(3));

    let threads: Vec<_> = (0..3)
        .map(|n| {
            let (address, results, errors, start) =
                (address.clone(), results.clone(), errors.clone(), start.clone());
            std::thread::spawn(move || {
                let remote = RemoteModelClient::with_timeout(address, 15.0);
                start.wait();
                match remote.generate(&format!("agent-{n}"), None, None, None, None) {
                    Ok(text) => {
                        results.lock().insert(n, text);
                    }
                    Err(exc) => errors.lock().push(exc.to_string()),
                }
                remote.close();
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }

    assert!(errors.lock().is_empty(), "{:?}", errors.lock());
    let expected: std::collections::BTreeMap<usize, String> =
        (0..3).map(|n| (n, format!("answer to agent-{n}"))).collect();
    assert_eq!(*results.lock(), expected);
}

/// gRPC interleaves RPCs on one channel without corrupting any.
#[test]
#[serial]
fn test_one_client_multiplexes_its_own_requests() {
    let hosted = Hosted::new();
    let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let results: Mutex<Vec<String>> = Mutex::new(Vec::new());

    std::thread::scope(|scope| {
        for n in 0..6 {
            let (client, errors, results) = (&hosted.client, &errors, &results);
            scope.spawn(move || match client.generate(&format!("q{n}"), None, None, None, None) {
                Ok(text) => results.lock().push(text),
                Err(exc) => errors.lock().push(exc.to_string()),
            });
        }
    });

    assert!(errors.lock().is_empty(), "{:?}", errors.lock());
    let mut results = results.into_inner();
    results.sort();
    let mut expected: Vec<String> = (0..6).map(|n| format!("answer to q{n}")).collect();
    expected.sort();
    assert_eq!(results, expected);
}

// ----------------------------------------------- TestPlatformsWithoutUnixSockets
//
// Windows has no AF_UNIX; the host has to still be reachable there.

#[test]
#[serial]
fn test_the_default_address_becomes_loopback() {
    let _guard = GlobalsGuard::new();
    set_has_unix_sockets(false);
    assert_eq!(default_socket_path("/home/x/.smartloop"), "127.0.0.1:0");
}

/// The fallback has to actually work, not just be selected.
#[test]
#[serial]
fn test_loopback_binds_and_serves() {
    let device = device();
    let server = ModelHost::new(as_client(&device), "127.0.0.1:0");
    server.start().unwrap();
    let port = parse_address(&server.address()).1;
    assert!(port.is_some_and(|p| p > 0));
    let client = RemoteModelClient::with_timeout(server.address(), 15.0);
    let loaded = client.is_loaded();
    let streamed = collect(client.stream("hi", None, None, None, None));
    client.close();
    server.stop();
    assert!(loaded.unwrap());
    assert_eq!(streamed.unwrap(), vec!["Hello", ", ", "world"]);
}

#[test]
#[serial]
fn test_binding_a_path_without_af_unix_says_why() {
    let _guard = GlobalsGuard::new();
    let tmp = tempfile::tempdir().unwrap();
    set_has_unix_sockets(false);
    let err = ModelHost::new(as_client(&device()), path_str(&tmp.path().join("h.sock")))
        .bind()
        .expect_err("binding must fail");
    assert!(err.to_string().contains("no AF_UNIX"), "{err}");
}

#[test]
#[serial]
fn test_connecting_to_a_path_without_af_unix_says_why() {
    let _guard = GlobalsGuard::new();
    let tmp = tempfile::tempdir().unwrap();
    set_has_unix_sockets(false);
    let client = RemoteModelClient::with_timeout(path_str(&tmp.path().join("h.sock")), 2.0);
    assert_err_contains(client.is_loaded(), "cannot open");
}

/// Nothing to unlink when the platform never made one.
#[test]
#[serial]
fn test_the_stale_socket_sweep_is_skipped() {
    let _guard = GlobalsGuard::new();
    let tmp = tempfile::tempdir().unwrap();
    set_has_unix_sockets(false);
    let leftover = tmp.path().join("stale.sock");
    std::fs::write(&leftover, "").unwrap();

    unlink_stale_socket(&path_str(&leftover)).unwrap(); // must not raise

    assert!(leftover.exists());
}

// ------------------------------------------------------------ TestTheSocketFile

#[cfg(unix)]
#[test]
#[serial]
fn test_it_is_not_world_readable() {
    use std::os::unix::fs::PermissionsExt;

    let hosted = Hosted::new();
    let mode = std::fs::metadata(hosted.host.address()).unwrap().permissions().mode();
    assert_eq!(mode & 0o777, 0o600);
}

#[test]
#[serial]
fn test_it_is_removed_on_stop() {
    let tmp = tempfile::tempdir().unwrap();
    let server = ModelHost::new(as_client(&device()), path_str(&tmp.path().join("gone.sock")));
    server.start().unwrap();
    assert!(Path::new(&server.address()).exists());
    server.stop();
    assert!(!Path::new(&server.address()).exists());
}

#[test]
#[serial]
fn test_a_stale_socket_from_a_killed_host_is_cleared() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("stale.sock");
    std::fs::write(&path, "").unwrap(); // a file where a socket used to be
    let server = ModelHost::new(as_client(&device()), path_str(&path));
    server.start().unwrap();
    let client = RemoteModelClient::with_timeout(server.address(), 15.0);
    let loaded = client.is_loaded();
    client.close();
    server.stop();
    assert!(loaded.unwrap());
}

#[test]
#[serial]
fn test_a_live_host_is_never_unlinked_from_under_its_clients() {
    let hosted = Hosted::new();
    let err = unlink_stale_socket(&hosted.host.address()).expect_err("a live host must not be unlinked");
    assert!(err.to_string().contains("already listening"), "{err}");
}

// ------------------------------------------- TestTheEmbedderIsSharedNotCopied
//
// BGE-M3 is ~417 MB; a child must not load its own alongside the host's.

/// An embedder that must never be asked: standing in for the Python
/// `_get_model = pytest.fail(...)` patch.
struct ForbiddenEmbedder;

impl smartloop::db::embeddings::EmbeddingModel for ForbiddenEmbedder {
    fn encode(&self, _texts: &[String]) -> anyhow::Result<Array2<f32>> {
        panic!("the child loaded its own embedder");
    }
}

/// A local embedder returning zeros, as the Python test patched
/// `encode_local`.
struct ZeroEmbedder;

impl smartloop::db::embeddings::EmbeddingModel for ZeroEmbedder {
    fn encode(&self, texts: &[String]) -> anyhow::Result<Array2<f32>> {
        Ok(Array2::zeros((texts.len(), 3)))
    }
}

/// Resets the embeddings singleton around a test (the `_reset_embeddings`
/// autouse fixture).
struct EmbeddingsGuard {
    _globals: GlobalsGuard,
}

impl EmbeddingsGuard {
    fn new() -> Self {
        let globals = GlobalsGuard::new();
        smartloop::db::embeddings::set_encode_texts_hook(None);
        smartloop::db::embeddings::set_model(None);
        smartloop::db::embeddings::set_model_load_attempted(false);
        Self { _globals: globals }
    }
}

impl Drop for EmbeddingsGuard {
    fn drop(&mut self) {
        smartloop::db::embeddings::set_model(None);
        smartloop::db::embeddings::set_model_load_attempted(false);
        set_device_owner(false);
    }
}

/// Stand where a per-project agent stands: no models in this process.
///
/// The host fixture runs in-process and marks it the device owner, which is
/// right for the host and wrong for the child being simulated.
fn as_a_child() {
    set_device_owner(false);
    smartloop::db::embeddings::set_model(Some(Arc::new(ForbiddenEmbedder)));
    smartloop::db::embeddings::set_model_load_attempted(true);
}

#[test]
#[serial]
fn test_a_child_embeds_through_the_host() {
    let _guard = EmbeddingsGuard::new();
    let hosted = Hosted::new();
    set_env("SLP_MODEL_HOST", &hosted.host.address());
    as_a_child();

    let result = smartloop::db::embeddings::encode_texts(&["a".to_string(), "b".to_string()], None, None);

    assert_eq!(result, Some(arange(2, 3)));
    assert!(hosted.device.calls().iter().any(|c| matches!(c, Call::EncodeTexts(..))));
}

/// The regression: a host answering EncodeTexts by sending EncodeTexts.
///
/// Each hop took a gRPC worker and held it for the client-wide 600s deadline,
/// so one citation check parked the whole pool on calls only that pool could
/// serve.
#[test]
#[serial]
fn test_the_host_does_not_serve_an_encode_by_asking_for_one() {
    let _guard = EmbeddingsGuard::new();
    // An address is set and this process owns the device: exactly the
    // supervisor's situation, where the host runs in the serving process.
    set_env("SLP_MODEL_HOST", "/tmp/model-host.sock");
    set_device_owner(false);
    set_global_encode_hook(Some(Arc::new(|_request, _timeout| {
        panic!("the host asked a model host to embed for it")
    })));
    smartloop::db::embeddings::set_model(Some(Arc::new(ZeroEmbedder)));
    smartloop::db::embeddings::set_model_load_attempted(true);

    let served = InProcessModelClient::new(Arc::new(FakeService::new()))
        .encode_texts(&["a".to_string(), "b".to_string()], None)
        .unwrap()
        .expect("embeddings");

    assert_eq!(served.shape(), &[2, 3]);
}

/// The chat verification budget must travel down to the RPC itself.
#[test]
#[serial]
fn test_a_per_call_deadline_reaches_the_remote_host() {
    let _guard = EmbeddingsGuard::new();
    let received: Arc<Mutex<Option<(Vec<String>, String, Option<f64>)>>> = Arc::default();
    let recorder = received.clone();
    set_env("SLP_MODEL_HOST", "/tmp/model-host-deadline.sock");
    set_device_owner(false);
    set_global_encode_hook(Some(Arc::new(move |request: &pb::EncodeTextsRequest, timeout| {
        *recorder.lock() = Some((request.texts.clone(), request.base_dir.clone(), timeout));
        Ok(pb::EncodeTextsResponse::default())
    })));
    smartloop::db::embeddings::set_model(Some(Arc::new(ForbiddenEmbedder)));
    smartloop::db::embeddings::set_model_load_attempted(true);

    smartloop::db::embeddings::encode_texts(&["a".to_string()], Some("/tmp/ws"), Some(30.0));

    assert_eq!(
        *received.lock(),
        Some((vec!["a".to_string()], "/tmp/ws".to_string(), Some(30.0)))
    );
}

#[test]
#[serial]
fn test_warmup_is_a_no_op_in_a_child() {
    let _guard = EmbeddingsGuard::new();
    let hosted = Hosted::new();
    set_env("SLP_MODEL_HOST", &hosted.host.address());
    as_a_child();

    smartloop::db::embeddings::warmup(Some("/tmp/ws")); // must not raise
}

/// It *is* the model; delegating would be a call to itself.
#[test]
#[serial]
fn test_the_host_itself_always_embeds_locally() {
    let _guard = EmbeddingsGuard::new();
    let hosted = Hosted::new();
    set_env("SLP_MODEL_HOST", &hosted.host.address());
    // ModelHost marked this process the device owner when it was built.
    set_global_encode_hook(Some(Arc::new(|_request, _timeout| {
        panic!("the host delegated its own embedding")
    })));
    smartloop::db::embeddings::set_model(Some(Arc::new(ZeroEmbedder)));
    smartloop::db::embeddings::set_model_load_attempted(true);

    let result = smartloop::db::embeddings::encode_texts(&["a".to_string()], None, None);

    assert_eq!(result.map(|r| r.shape().to_vec()), Some(vec![1, 3]));
    assert!(!hosted.device.calls().iter().any(|c| matches!(c, Call::EncodeTexts(..))));
}

#[test]
#[serial]
fn test_no_host_configured_embeds_locally() {
    let _guard = EmbeddingsGuard::new();
    remove_env("SLP_MODEL_HOST");
    set_device_owner(false);
    set_global_encode_hook(Some(Arc::new(|_request, _timeout| {
        panic!("embedding was sent to a model host that was never configured")
    })));
    smartloop::db::embeddings::set_model(Some(Arc::new(ZeroEmbedder)));
    smartloop::db::embeddings::set_model_load_attempted(true);

    let result = smartloop::db::embeddings::encode_texts(&["a".to_string()], None, None);

    assert_eq!(result.map(|r| r.shape().to_vec()), Some(vec![1, 3]));
}

/// The documented contract: unavailable means None, and TF-IDF.
#[test]
#[serial]
fn test_an_unreachable_host_falls_back_rather_than_raising() {
    let _guard = EmbeddingsGuard::new();
    let tmp = tempfile::tempdir().unwrap();
    set_device_owner(false);
    set_env("SLP_MODEL_HOST", &path_str(&tmp.path().join("gone.sock")));
    assert!(smartloop::db::embeddings::encode_texts(&["a".to_string()], None, None).is_none());
}

// -------------------------------------------------------- TestTheBenchmarkReport

#[test]
fn test_it_times_a_stream() {
    use smartloop::model_host::bench::benchmark;

    let device = device();
    device.set_chunks(vec!["one".into(), "two".into(), "three".into()]);
    let result = benchmark(&device, "prompt", 2, false).unwrap();

    assert_eq!(result.client, "FakeDevice");
    assert_eq!(result.chunks, 3.0);
    assert_eq!(result.chars, "onetwothree".len() as f64);
    assert!(result.total_s >= 0.0);
}

#[test]
fn test_it_names_the_slower_transport() {
    use smartloop::model_host::bench::{format_report, BenchResult};

    let report = format_report(&[
        BenchResult {
            client: "InProcessModelClient".into(),
            runs: 3,
            first_chunk_s: 0.10,
            total_s: 1.0,
            chars: 500.0,
            chunks: 50.0,
            chars_per_second: 500.0,
        },
        BenchResult {
            client: "RemoteModelClient".into(),
            runs: 3,
            first_chunk_s: 0.12,
            total_s: 1.1,
            chars: 500.0,
            chunks: 50.0,
            chars_per_second: 450.0,
        },
    ]);

    assert!(report.contains("1.20x slower"), "{report}"); // first chunk
    assert!(report.contains("1.11x slower"), "{report}"); // throughput
    assert!(report.contains("InProcessModelClient"));
}

// ------------------------------------------ TestTheHostReadsTheProjectsOwnStateFirst
//
// The host asks its client to catch up before it binds.

#[tokio::test]
#[serial]
async fn test_the_project_is_refreshed_before_it_is_bound() {
    use smartloop::workspace::context::active_project_id;

    let _guard = GlobalsGuard::new();
    let seen: Arc<Mutex<Vec<(Option<String>, Option<String>)>>> = Arc::default();
    let device = device();
    let recorder = seen.clone();
    *device.refresh.lock() = Some(Box::new(move |project_id| {
        // Ordering is the point: refreshing after the bind would resolve this
        // operation against the copy we came in with.
        recorder.lock().push((project_id.map(str::to_string), active_project_id()));
        Ok(true)
    }));
    let servicer = ModelHostServicer::new(as_client(&device));

    servicer
        .generate(tonic::Request::new(pb::GenerateRequest {
            prompt: "hi".into(),
            project_id: "alpha".into(),
            ..Default::default()
        }))
        .await
        .unwrap();

    assert_eq!(*seen.lock(), vec![(Some("alpha".to_string()), None)]);
}

/// A project.json we cannot read is not a reason to refuse to answer.
#[tokio::test]
#[serial]
async fn test_a_refresh_failure_does_not_fail_the_operation() {
    let _guard = GlobalsGuard::new();
    let device = device();
    *device.refresh.lock() = Some(Box::new(|_project_id| {
        Err(std::io::Error::new(std::io::ErrorKind::NotFound, "no such file").into())
    }));
    let servicer = ModelHostServicer::new(as_client(&device));

    let reply = servicer
        .generate(tonic::Request::new(pb::GenerateRequest {
            prompt: "hi".into(),
            project_id: "alpha".into(),
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner();

    assert!(device.calls().contains(&Call::Generate("hi".into(), None, None)));
    assert_eq!(reply.text, "answer to hi");
}

/// The CLI and the single-server fallback name no project.
#[tokio::test]
#[serial]
async fn test_an_unprojected_request_still_works() {
    let _guard = GlobalsGuard::new();
    let device = device();
    let servicer = ModelHostServicer::new(as_client(&device));

    let reply = servicer
        .generate(tonic::Request::new(pb::GenerateRequest {
            prompt: "hi".into(),
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner();

    assert_eq!(reply.text, "answer to hi");
}

// ------------------------------------------------ TestTheInProcessClientCanRefresh

#[test]
fn test_it_asks_the_service() {
    let service = Arc::new(FakeService::new());
    assert!(InProcessModelClient::new(service.clone()).refresh_project(Some("alpha")).unwrap());
    assert_eq!(*service.calls.lock(), vec![ServiceCall::Refresh(Some("alpha".into()))]);
}

/// Over the wire the workspace lives on the other side.
#[test]
fn test_a_remote_client_has_nothing_to_refresh() {
    let client = RemoteModelClient::new("/tmp/never-connected.sock");
    assert!(!client.refresh_project(Some("alpha")).unwrap());
}
