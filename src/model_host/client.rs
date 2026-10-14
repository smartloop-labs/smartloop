//! Port of `smartloop/model_host/client.py`.
//!
//! The interface a project agent uses to reach the generative model.
//!
//! The model is always reached through the gRPC IPC transport.  The host
//! process owns the device and serves the model; project agents only ever call
//! into it through [`RemoteModelClient`].
//!
//! The in-process implementation remains only for the host's own local call
//! path, not as a user-visible fallback or a second transport mode.
//!
//! Select with `SLP_MODEL_HOST` — a unix socket path (or `host:port`); if it
//! is unset, the default host socket is used.

use std::any::Any;
use std::future::Future;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ndarray::Array2;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use tonic::{Code, Status};

use crate::api::services::model_service::ModelService;
use crate::config::AppSettings;
use crate::llms::base_llm::{CancelEvent, GenerationChunk};
use crate::model_host::pb;
use crate::workspace::context::pinned_project_id;

/// Logger name used by the Python module.
const LOG_TARGET: &str = "smartloop.model_host.client";

/// Set in the process that actually holds the weights.  Without it a host
/// started with `SLP_MODEL_HOST` pointing at its own socket would resolve a
/// remote client and try to answer its own requests.
static DEVICE_OWNER: AtomicBool = AtomicBool::new(false);

/// Whether this platform can speak AF_UNIX at all (compile-time default for
/// [`HAS_UNIX_SOCKETS_OVERRIDE`]).
pub const HAS_UNIX_SOCKETS: bool = cfg!(unix);

/// Runtime view of [`HAS_UNIX_SOCKETS`]; tests flip it to exercise the
/// loopback path a platform without AF_UNIX takes (the Python tests
/// monkeypatched the module constant).
static HAS_UNIX_SOCKETS_OVERRIDE: AtomicBool = AtomicBool::new(HAS_UNIX_SOCKETS);

/// grpcio's default cap on a received message is 4 MiB; an embedding batch can
/// exceed that.  Mirrors the server's own ceiling so a large payload is
/// refused on both sides at the same limit.
pub const MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

/// Longest a unix socket path may be. `sockaddr_un.sun_path` is 108 bytes on
/// Linux and 104 on macOS; the smaller is used so an address that works on one
/// works on the other, and one byte is left for the terminator.
pub const MAX_UNIX_SOCKET_PATH: usize = 103;

/// Default per-RPC deadline of a [`RemoteModelClient`], in seconds.
pub const DEFAULT_TIMEOUT_SECS: f64 = 600.0;

/// Whether unix sockets are usable in this process (see [`HAS_UNIX_SOCKETS`]).
pub fn has_unix_sockets() -> bool {
    HAS_UNIX_SOCKETS_OVERRIDE.load(Ordering::SeqCst)
}

/// Override [`has_unix_sockets`] (test seam for platforms without AF_UNIX).
pub fn set_has_unix_sockets(value: bool) {
    HAS_UNIX_SOCKETS_OVERRIDE.store(value, Ordering::SeqCst);
}

/// Declare that this process owns the models, not a host elsewhere.
pub fn mark_device_owner() {
    DEVICE_OWNER.store(true, Ordering::SeqCst);
}

/// Whether the models are resident in this process.
pub fn is_device_owner() -> bool {
    DEVICE_OWNER.load(Ordering::SeqCst)
}

/// Set the device-owner flag directly (test seam; Python monkeypatched
/// `_device_owner`).
pub fn set_device_owner(value: bool) {
    DEVICE_OWNER.store(value, Ordering::SeqCst);
}

/// The model host refused, failed, or could not be reached.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ModelHostError(pub String);

impl ModelHostError {
    /// Build an error from a message.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

/// Streamed text as produced by [`ModelClient::stream`].  A failure part-way
/// through surfaces as an `Err` item.
pub type TextStream<'a> = Box<dyn Iterator<Item = anyhow::Result<String>> + 'a>;

/// Everything a project agent needs from whoever owns the device.
///
/// Deliberately narrow: generation, embedding, and the questions whose answers
/// depend on which model is resident.  Project state — documents, skills, MCP
/// servers, sessions — is on the other side of the seam and never crosses it.
pub trait ModelClient: Send + Sync {
    // -- generation ---------------------------------------------------------

    /// Complete `prompt` and return the whole response.
    ///
    /// `max_tokens` caps the output; `thinking` forces the reasoning block on
    /// or off for this call whatever the user's setting says.  Both belong to
    /// the request rather than to the loaded model: a decision call caps
    /// itself so a missing stop token cannot ramble, and a call whose output
    /// is a tool call rather than an answer reasons even when the visible
    /// answer does not (see `MCPAgent`).
    fn generate(
        &self,
        prompt: &str,
        model_name: Option<&str>,
        capability: Option<&str>,
        max_tokens: Option<usize>,
        thinking: Option<bool>,
    ) -> anyhow::Result<String>;

    /// Complete `prompt`, yielding text as it is produced.
    ///
    /// `cancelled` is cooperative and observed between tokens — a native
    /// llama.cpp call cannot be interrupted mid-token, in-process or over the
    /// wire.
    fn stream<'a>(
        &'a self,
        prompt: &str,
        model_name: Option<&str>,
        capability: Option<&str>,
        cancelled: Option<CancelEvent>,
        thinking: Option<bool>,
    ) -> anyhow::Result<TextStream<'a>>;

    // -- embedding ----------------------------------------------------------

    /// Embed `texts`, or return `None` when the embedder is unavailable.
    ///
    /// Held by the host so BGE-M3 is loaded once for every project rather than
    /// ~417 MB per child.
    fn encode_texts(
        &self,
        texts: &[String],
        base_dir: Option<&str>,
    ) -> anyhow::Result<Option<Array2<f32>>>;

    // -- questions about what is resident -----------------------------------

    /// Whether a generative model is resident.
    fn is_loaded(&self) -> anyhow::Result<bool>;

    /// Whether the model serving `capability` can reason.
    ///
    /// Not required: a client that cannot answer says no, and a caller that
    /// wanted reasoning simply asks for it and gets whatever the model does.
    fn supports_thinking(&self, capability: Option<&str>) -> anyhow::Result<bool> {
        let _ = capability;
        Ok(false)
    }

    /// Name of the resident orchestrator model, if any.
    fn model_name(&self) -> anyhow::Result<Option<String>>;

    /// How the model that will answer for `capability` is named to the user.
    ///
    /// The progress SSE names the model that is answering, and this has to be
    /// the model the host's own selection lands on — not an approximation of
    /// it — or a remote caller shows a different name than the single-process
    /// server would have.
    fn model_label(&self, capability: Option<&str>) -> anyhow::Result<String>;

    /// Token count under the resident model's tokenizer.
    fn count_tokens(&self, text: &str) -> anyhow::Result<usize>;

    /// Context window of the resident model.
    fn max_context_tokens(&self) -> anyhow::Result<usize>;

    /// Context window of the model that would answer for `capability`.
    fn response_max_context_tokens(&self, capability: Option<&str>) -> anyhow::Result<usize>;

    /// Whether a model declaring `capability` is available.
    fn has_capability_model(&self, capability: &str) -> anyhow::Result<bool>;

    /// Seconds since the device last finished a generation.
    fn idle_seconds(&self) -> anyhow::Result<f64>;

    // -- lifecycle ----------------------------------------------------------

    /// Make a generative model resident; returns its name.
    fn load_model(&self) -> anyhow::Result<Option<String>>;

    /// Free the on-demand large/vision models (and optionally the embedder).
    fn unload_auxiliary_models(&self, include_embedding: bool) -> anyhow::Result<()>;

    /// Free the embedding model.
    fn unload_embedding_model(&self) -> anyhow::Result<()>;

    /// Release any transport this client holds. In-process: nothing.
    fn close(&self) {}

    /// Pick up any write to `project_id`'s own state before serving it.
    ///
    /// Default no-op: a client that reaches the device over the wire is not
    /// the process holding the workspace, so there is nothing local to
    /// refresh.  Overridden where the device and the workspace are in the
    /// same process.
    fn refresh_project(&self, project_id: Option<&str>) -> anyhow::Result<bool> {
        let _ = project_id;
        Ok(false)
    }

    /// Class name used in reports (Python `type(client).__name__`).
    fn type_name(&self) -> String {
        let full = std::any::type_name::<Self>();
        full.rsplit("::").next().unwrap_or(full).to_string()
    }

    /// The remote client when this is one (Python `isinstance(client,
    /// RemoteModelClient)`).
    fn as_remote(&self) -> Option<&RemoteModelClient> {
        None
    }

    /// Downcasting hook for callers and tests.
    fn as_any(&self) -> &dyn Any;
}

/// What [`InProcessModelClient`] calls straight through to: the subset of
/// `ModelService` the device seam needs.  A trait so tests can stand in a
/// fake service.
pub trait LocalModelService: Send + Sync {
    /// `ModelService.refresh_project_state`.
    fn refresh_project_state(&self, project_id: Option<&str>) -> anyhow::Result<bool> {
        let _ = project_id;
        Ok(false)
    }

    /// `ModelService.generate_completion`.
    fn generate_completion(
        &self,
        prompt: &str,
        model_name: Option<&str>,
        capability: Option<&str>,
        max_tokens: Option<usize>,
        thinking: Option<bool>,
    ) -> anyhow::Result<String>;

    /// `ModelService.stream_completion`.
    fn stream_completion<'a>(
        &'a self,
        prompt: &str,
        model_name: Option<&str>,
        cancelled: Option<CancelEvent>,
        capability: Option<&str>,
        thinking: Option<bool>,
    ) -> anyhow::Result<Box<dyn Iterator<Item = anyhow::Result<GenerationChunk>> + 'a>>;

    /// `ModelService.is_loaded`.
    fn is_loaded(&self) -> bool;

    /// `ModelService.supports_thinking`.
    fn supports_thinking(&self, capability: Option<&str>) -> bool {
        let _ = capability;
        false
    }

    /// `ModelService._model_name`: the resident orchestrator model.
    fn resident_model_name(&self) -> Option<String>;

    /// `ModelService.response_model_label`.
    fn response_model_label(&self, capability: Option<&str>) -> String;

    /// `ModelService.count_tokens`.
    fn count_tokens(&self, text: &str) -> usize;

    /// `ModelService.get_max_context_tokens`.
    fn get_max_context_tokens(&self) -> usize;

    /// `ModelService.get_response_max_context_tokens`.
    fn get_response_max_context_tokens(&self, capability: Option<&str>) -> usize;

    /// `ModelService.has_capability_model`.
    fn has_capability_model(&self, capability: &str) -> bool;

    /// `ModelService.idle_seconds`.
    fn idle_seconds(&self) -> f64;

    /// `ModelService.load_model`.
    fn load_model(&self) -> anyhow::Result<()>;

    /// `ModelService.unload_auxiliary_models`.
    fn unload_auxiliary_models(&self, include_embedding: bool);

    /// `ModelService.unload_embedding_model`.
    fn unload_embedding_model(&self);
}

impl LocalModelService for ModelService {
    fn refresh_project_state(&self, project_id: Option<&str>) -> anyhow::Result<bool> {
        Ok(ModelService::refresh_project_state(self, project_id))
    }

    fn generate_completion(
        &self,
        prompt: &str,
        model_name: Option<&str>,
        capability: Option<&str>,
        max_tokens: Option<usize>,
        thinking: Option<bool>,
    ) -> anyhow::Result<String> {
        ModelService::generate_completion(self, prompt, model_name, capability, max_tokens, thinking)
    }

    fn stream_completion<'a>(
        &'a self,
        prompt: &str,
        model_name: Option<&str>,
        cancelled: Option<CancelEvent>,
        capability: Option<&str>,
        thinking: Option<bool>,
    ) -> anyhow::Result<Box<dyn Iterator<Item = anyhow::Result<GenerationChunk>> + 'a>> {
        ModelService::stream_completion(self, prompt, model_name, cancelled, capability, thinking)
    }

    fn is_loaded(&self) -> bool {
        ModelService::is_loaded(self)
    }

    fn supports_thinking(&self, capability: Option<&str>) -> bool {
        ModelService::supports_thinking(self, capability)
    }

    fn resident_model_name(&self) -> Option<String> {
        self.model_name.read().clone()
    }

    fn response_model_label(&self, capability: Option<&str>) -> String {
        ModelService::response_model_label(self, capability)
    }

    fn count_tokens(&self, text: &str) -> usize {
        ModelService::count_tokens(self, text)
    }

    fn get_max_context_tokens(&self) -> usize {
        ModelService::get_max_context_tokens(self)
    }

    fn get_response_max_context_tokens(&self, capability: Option<&str>) -> usize {
        ModelService::get_response_max_context_tokens(self, capability)
    }

    fn has_capability_model(&self, capability: &str) -> bool {
        ModelService::has_capability_model(self, capability)
    }

    fn idle_seconds(&self) -> f64 {
        ModelService::idle_seconds(self)
    }

    fn load_model(&self) -> anyhow::Result<()> {
        ModelService::load_model(self).map(|_| ())
    }

    fn unload_auxiliary_models(&self, include_embedding: bool) {
        ModelService::unload_auxiliary_models(self, include_embedding)
    }

    fn unload_embedding_model(&self) {
        ModelService::unload_embedding_model(self)
    }
}

/// The device is owned by this process — call straight through.
///
/// The default. Every method here is the call the caller used to make
/// directly, so selecting this client changes nothing about how a request is
/// served.
pub struct InProcessModelClient {
    service: Arc<dyn LocalModelService>,
}

impl InProcessModelClient {
    /// Wrap the service that owns the device.
    pub fn new(service: Arc<dyn LocalModelService>) -> Self {
        Self { service }
    }

    /// The wrapped service.
    pub fn service(&self) -> &Arc<dyn LocalModelService> {
        &self.service
    }
}

impl ModelClient for InProcessModelClient {
    fn generate(
        &self,
        prompt: &str,
        model_name: Option<&str>,
        capability: Option<&str>,
        max_tokens: Option<usize>,
        thinking: Option<bool>,
    ) -> anyhow::Result<String> {
        self.service
            .generate_completion(prompt, model_name, capability, max_tokens, thinking)
    }

    /// `GenerationChunk` in, plain text out — the wire carries text, so the
    /// in-process path yields the same thing rather than making every caller
    /// handle two shapes.
    fn stream<'a>(
        &'a self,
        prompt: &str,
        model_name: Option<&str>,
        capability: Option<&str>,
        cancelled: Option<CancelEvent>,
        thinking: Option<bool>,
    ) -> anyhow::Result<TextStream<'a>> {
        let chunks = self
            .service
            .stream_completion(prompt, model_name, cancelled, capability, thinking)?;
        Ok(Box::new(chunks.map(|chunk| chunk.map(|c| c.text))))
    }

    /// `encode_local`, not `encode_texts`: this process owns the embedder, and
    /// this is also the host's own serving path.  Asking the routing entry
    /// point would let a host serve `EncodeTexts` by sending one — a call it
    /// can only answer by sending another, one gRPC worker consumed per hop.
    fn encode_texts(
        &self,
        texts: &[String],
        base_dir: Option<&str>,
    ) -> anyhow::Result<Option<Array2<f32>>> {
        Ok(crate::db::embeddings::encode_local(texts, base_dir))
    }

    fn is_loaded(&self) -> anyhow::Result<bool> {
        Ok(self.service.is_loaded())
    }

    fn supports_thinking(&self, capability: Option<&str>) -> anyhow::Result<bool> {
        Ok(self.service.supports_thinking(capability))
    }

    fn model_name(&self) -> anyhow::Result<Option<String>> {
        Ok(self.service.resident_model_name())
    }

    fn model_label(&self, capability: Option<&str>) -> anyhow::Result<String> {
        Ok(self.service.response_model_label(capability))
    }

    fn count_tokens(&self, text: &str) -> anyhow::Result<usize> {
        Ok(self.service.count_tokens(text))
    }

    fn max_context_tokens(&self) -> anyhow::Result<usize> {
        Ok(self.service.get_max_context_tokens())
    }

    fn response_max_context_tokens(&self, capability: Option<&str>) -> anyhow::Result<usize> {
        Ok(self.service.get_response_max_context_tokens(capability))
    }

    fn has_capability_model(&self, capability: &str) -> anyhow::Result<bool> {
        Ok(self.service.has_capability_model(capability))
    }

    fn idle_seconds(&self) -> anyhow::Result<f64> {
        Ok(self.service.idle_seconds())
    }

    fn load_model(&self) -> anyhow::Result<Option<String>> {
        self.service.load_model()?;
        Ok(self.service.resident_model_name())
    }

    fn unload_auxiliary_models(&self, include_embedding: bool) -> anyhow::Result<()> {
        self.service.unload_auxiliary_models(include_embedding);
        Ok(())
    }

    fn unload_embedding_model(&self) -> anyhow::Result<()> {
        self.service.unload_embedding_model();
        Ok(())
    }

    /// The workspace is right here, so a child's write is one read away.
    fn refresh_project(&self, project_id: Option<&str>) -> anyhow::Result<bool> {
        self.service.refresh_project_state(project_id)
    }

    fn type_name(&self) -> String {
        "InProcessModelClient".to_string()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Test seam replacing the `EncodeTexts` RPC: receives the request and the
/// per-call timeout override (`None` = client default).  The Python tests
/// monkeypatched `client._unary`.
pub type EncodeHook = Arc<
    dyn Fn(&pb::EncodeTextsRequest, Option<f64>) -> Result<pb::EncodeTextsResponse, ModelHostError>
        + Send
        + Sync,
>;

/// Process-wide counterpart of [`RemoteModelClient::set_encode_hook`], used by
/// every client without an instance hook.  Lets a test stand in for the RPC
/// of a client it does not construct itself (the embedder's own connection);
/// the Python tests swapped `embeddings._remote_embedder` for a fake.
static GLOBAL_ENCODE_HOOK: Lazy<Mutex<Option<EncodeHook>>> = Lazy::new(|| Mutex::new(None));

/// Install (or clear) the process-wide `EncodeTexts` test seam.
pub fn set_global_encode_hook(hook: Option<EncodeHook>) {
    *GLOBAL_ENCODE_HOOK.lock() = hook;
}

/// The live transport: a gRPC client over a lazily-opened channel.
type GrpcClient = pb::model_host_client::ModelHostClient<tonic::transport::Channel>;

/// The runtime every [`RemoteModelClient`] drives its RPCs on.
///
/// The client's interface is synchronous (it is called from worker threads,
/// exactly as the Python client was), so the async transport runs here and
/// callers wait on a channel — never on `block_on`, which would refuse to run
/// on a thread that is already inside a runtime.
static CLIENT_RUNTIME: Lazy<tokio::runtime::Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("slp-model-host-client")
        .enable_all()
        .build()
        .expect("failed to start the model host client runtime")
});

/// Run `fut` on [`CLIENT_RUNTIME`] and wait for it from this (sync) thread.
fn run_on_client_runtime<F>(fut: F) -> Result<F::Output, Status>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let handle = CLIENT_RUNTIME.spawn(fut);
    futures::executor::block_on(handle)
        .map_err(|exc| Status::unknown(format!("model host call did not complete: {exc}")))
}

/// A per-call deadline from seconds; `None` for no deadline.
fn deadline(seconds: f64) -> Option<Duration> {
    if seconds.is_nan() || seconds.is_infinite() {
        return None;
    }
    Some(Duration::from_secs_f64(seconds.max(0.0)))
}

/// The device is owned by a model host — talk to it over gRPC.
///
/// One channel, used by every thread: gRPC multiplexes RPCs on it, and a
/// project agent's concurrency is already bounded by the host's own
/// `n_parallel` budget, so a pool would only move the queue without widening
/// it.
///
/// The channel is opened lazily and reconnects transparently on failure, so a
/// host restart costs the in-flight request rather than the client.
pub struct RemoteModelClient {
    /// The address this client talks to (socket path or `host:port`).
    pub address: String,
    timeout: f64,
    stub: Mutex<Option<GrpcClient>>,
    encode_hook: Mutex<Option<EncodeHook>>,
}

impl RemoteModelClient {
    /// A client for `address` with the default 600 s per-RPC deadline.
    pub fn new(address: impl Into<String>) -> Self {
        Self::with_timeout(address, DEFAULT_TIMEOUT_SECS)
    }

    /// A client for `address` with a per-RPC deadline of `timeout` seconds.
    pub fn with_timeout(address: impl Into<String>, timeout: f64) -> Self {
        Self {
            address: address.into(),
            timeout,
            stub: Mutex::new(None),
            encode_hook: Mutex::new(None),
        }
    }

    /// The client-wide per-RPC deadline, in seconds.
    pub fn timeout(&self) -> f64 {
        self.timeout
    }

    /// Install (or clear) the `EncodeTexts` test seam.
    pub fn set_encode_hook(&self, hook: Option<EncodeHook>) {
        *self.encode_hook.lock() = hook;
    }

    // -- transport ----------------------------------------------------------

    /// Open the channel if needed and return the stub.
    fn connect(&self) -> Result<GrpcClient, ModelHostError> {
        let mut stub = self.stub.lock();
        if let Some(existing) = stub.as_ref() {
            return Ok(existing.clone());
        }
        let (_host, port) = parse_address(&self.address);
        if port.is_none() && !has_unix_sockets() {
            return Err(ModelHostError::new(format!(
                "{} is a unix socket path, which this platform cannot open; the host should \
                 be listening on host:port",
                self.address
            )));
        }
        if port == Some(0) {
            return Err(ModelHostError::new(format!(
                "model host address {} is not bound: the host is not listening on a real port",
                py_repr(&self.address)
            )));
        }
        let channel = {
            // The channel's background worker is spawned onto whichever
            // runtime is current, so make it ours.
            let _enter = CLIENT_RUNTIME.enter();
            open_channel(&self.address)?
        };
        let client = GrpcClient::new(channel)
            .max_decoding_message_size(MAX_MESSAGE_BYTES)
            .max_encoding_message_size(MAX_MESSAGE_BYTES);
        *stub = Some(client.clone());
        tracing::info!(target: LOG_TARGET, "[model-host] connected to {}", self.address);
        Ok(client)
    }

    /// Close the channel and forget it, so the next call reconnects.
    fn drop_channel(&self) {
        // In-flight calls hold their own handle; the channel closes once the
        // last of them finishes.
        *self.stub.lock() = None;
    }

    /// Run one unary RPC and return its message, or a [`ModelHostError`].
    ///
    /// `timeout` overrides the client-wide default for a single call, so a
    /// latency-sensitive path can impose its own deadline (and fall back to
    /// whatever it does without the host) without reshaping every caller.
    fn unary<Req, Resp, F, Fut>(
        &self,
        request: Req,
        timeout: Option<f64>,
        call: F,
    ) -> Result<Resp, ModelHostError>
    where
        Req: Send + 'static,
        Resp: Send + 'static,
        F: FnOnce(GrpcClient, tonic::Request<Req>) -> Fut + Send + 'static,
        Fut: Future<Output = Result<tonic::Response<Resp>, Status>> + Send + 'static,
    {
        let stub = self.connect()?;
        let limit = deadline(timeout.unwrap_or(self.timeout));
        let outcome = run_on_client_runtime(async move {
            let mut request = tonic::Request::new(request);
            if let Some(limit) = limit {
                request.set_timeout(limit);
            }
            let fut = call(stub, request);
            match limit {
                Some(limit) => match tokio::time::timeout(limit, fut).await {
                    Ok(result) => result,
                    Err(_) => Err(Status::deadline_exceeded("Deadline Exceeded")),
                },
                None => fut.await,
            }
        })
        .and_then(|result| result);
        match outcome {
            Ok(response) => Ok(response.into_inner()),
            Err(status) => {
                if status.code() == Code::Unavailable {
                    // The host is gone (crashed, restarted, or never there).
                    // A dead channel caches that failure; drop it so the next
                    // call reconnects instead of re-raising.
                    self.drop_channel();
                }
                Err(rpc_error(&self.address, &status))
            }
        }
    }

    /// Embed `texts` with an optional per-call deadline (`timeout` seconds)
    /// overriding the client-wide default, so a latency-sensitive path can
    /// impose its own deadline.
    pub fn encode_texts_with_timeout(
        &self,
        texts: &[String],
        base_dir: Option<&str>,
        timeout: Option<f64>,
    ) -> Result<Option<Array2<f32>>, ModelHostError> {
        let request = pb::EncodeTextsRequest {
            texts: texts.to_vec(),
            base_dir: base_dir.unwrap_or_default().to_string(),
            project_id: pinned_project_id().unwrap_or_default(),
        };
        let hook = self
            .encode_hook
            .lock()
            .clone()
            .or_else(|| GLOBAL_ENCODE_HOOK.lock().clone());
        let reply = match hook {
            Some(hook) => hook(&request, timeout)?,
            None => self.unary(request, timeout, |mut stub, request| async move {
                stub.encode_texts(request).await
            })?,
        };
        decode_embeddings(&reply)
    }

    /// The `ProjectRequest` carrying this process's pinned project.
    fn project_request(&self) -> pb::ProjectRequest {
        pb::ProjectRequest { project_id: pinned_project_id().unwrap_or_default() }
    }
}

/// Rebuild the float32 matrix an `EncodeTextsResponse` carries; an empty
/// shape means the host has no embedder.
fn decode_embeddings(reply: &pb::EncodeTextsResponse) -> Result<Option<Array2<f32>>, ModelHostError> {
    if reply.shape.is_empty() {
        return Ok(None);
    }
    let dims: Vec<usize> = reply.shape.iter().map(|d| (*d).max(0) as usize).collect();
    let (rows, cols) = match dims.as_slice() {
        [rows, cols] => (*rows, *cols),
        [cols] => (1, *cols),
        other => {
            return Err(ModelHostError::new(format!(
                "model host returned embeddings of unexpected shape {other:?}"
            )));
        }
    };
    if reply.data.len() != rows * cols * 4 {
        return Err(ModelHostError::new(format!(
            "model host returned {} bytes for embeddings of shape {:?}",
            reply.data.len(),
            dims
        )));
    }
    let values: Vec<f32> = reply
        .data
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Array2::from_shape_vec((rows, cols), values)
        .map(Some)
        .map_err(|exc| ModelHostError::new(format!("model host returned malformed embeddings: {exc}")))
}

/// Open a lazily-connecting channel to `address`.
fn open_channel(address: &str) -> Result<tonic::transport::Channel, ModelHostError> {
    let (host, port) = parse_address(address);
    match port {
        None => open_unix_channel(&host),
        Some(port) => {
            let authority =
                if host.contains(':') { format!("[{host}]:{port}") } else { format!("{host}:{port}") };
            let endpoint = tonic::transport::Endpoint::from_shared(format!("http://{authority}"))
                .map_err(|exc| {
                    ModelHostError::new(format!("model host address {address} is not usable: {exc}"))
                })?;
            Ok(endpoint.tcp_nodelay(true).connect_lazy())
        }
    }
}

#[cfg(unix)]
fn open_unix_channel(path: &str) -> Result<tonic::transport::Channel, ModelHostError> {
    let path = path.to_string();
    // The URI is required by the API but never dialled: the connector below
    // opens the socket file instead.
    let endpoint = tonic::transport::Endpoint::from_static("http://localhost");
    Ok(endpoint.connect_with_connector_lazy(tower::service_fn(move |_: http::Uri| {
        let path = path.clone();
        async move {
            let stream = tokio::net::UnixStream::connect(path).await?;
            Ok::<_, std::io::Error>(hyper_util::rt::TokioIo::new(stream))
        }
    })))
}

#[cfg(not(unix))]
fn open_unix_channel(path: &str) -> Result<tonic::transport::Channel, ModelHostError> {
    Err(ModelHostError::new(format!(
        "{path} is a unix socket path, which this platform cannot open; the host should be \
         listening on host:port"
    )))
}

/// A streamed generation from the host, as a blocking iterator.
///
/// The RPC runs on [`CLIENT_RUNTIME`]; chunks arrive over a channel.  Dropping
/// the iterator part-way aborts the RPC, which cancels it server-side.
struct RemoteTextStream<'a> {
    client: &'a RemoteModelClient,
    receiver: tokio::sync::mpsc::Receiver<Result<String, Status>>,
    task: tokio::task::JoinHandle<()>,
    cancelled: Option<CancelEvent>,
    done: bool,
}

impl RemoteTextStream<'_> {
    fn finish(&mut self) {
        self.done = true;
        self.receiver.close();
        self.task.abort();
    }
}

impl Iterator for RemoteTextStream<'_> {
    type Item = anyhow::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match futures::executor::block_on(self.receiver.recv()) {
            None => {
                self.finish();
                None
            }
            Some(Ok(text)) => {
                if self.cancelled.as_ref().is_some_and(|event| event.load(Ordering::SeqCst)) {
                    // Advisory, exactly like the old cancel frame: the host
                    // stops between tokens, and the channel stays usable for
                    // the next request.
                    self.finish();
                    return None;
                }
                Some(Ok(text))
            }
            Some(Err(status)) => {
                self.finish();
                if status.code() == Code::Unavailable {
                    self.client.drop_channel();
                }
                Some(Err(rpc_error(&self.client.address, &status).into()))
            }
        }
    }
}

impl Drop for RemoteTextStream<'_> {
    fn drop(&mut self) {
        // Abandoned mid-way (a caller that broke out of the loop) must cancel
        // the RPC server-side, or a generation would be computed for nobody.
        // Harmless once the stream has finished.
        self.task.abort();
    }
}

impl ModelClient for RemoteModelClient {
    fn generate(
        &self,
        prompt: &str,
        model_name: Option<&str>,
        capability: Option<&str>,
        max_tokens: Option<usize>,
        thinking: Option<bool>,
    ) -> anyhow::Result<String> {
        let request = pb::GenerateRequest {
            prompt: prompt.to_string(),
            model_name: model_name.unwrap_or_default().to_string(),
            capability: capability.unwrap_or_default().to_string(),
            max_tokens: max_tokens.map(|n| i32::try_from(n).unwrap_or(i32::MAX)),
            thinking,
            project_id: pinned_project_id().unwrap_or_default(),
        };
        // No deadline: a slow answer is not a hung host (as with `stream`).
        let reply = self.unary(request, Some(f64::INFINITY), |mut stub, request| async move {
            stub.generate(request).await
        })?;
        Ok(reply.text)
    }

    /// Dropping the returned stream part-way cancels the RPC server-side, so
    /// a generation is never computed for nobody.
    fn stream<'a>(
        &'a self,
        prompt: &str,
        model_name: Option<&str>,
        capability: Option<&str>,
        cancelled: Option<CancelEvent>,
        thinking: Option<bool>,
    ) -> anyhow::Result<TextStream<'a>> {
        let request = pb::StreamRequest {
            prompt: prompt.to_string(),
            model_name: model_name.unwrap_or_default().to_string(),
            capability: capability.unwrap_or_default().to_string(),
            thinking,
            project_id: pinned_project_id().unwrap_or_default(),
        };
        let mut stub = self.connect()?;
        let (sender, receiver) = tokio::sync::mpsc::channel::<Result<String, Status>>(16);
        // No deadline: a long answer is not a hung host.
        let task = CLIENT_RUNTIME.spawn(async move {
            match stub.stream(tonic::Request::new(request)).await {
                Err(status) => {
                    let _ = sender.send(Err(status)).await;
                }
                Ok(response) => {
                    let mut chunks = response.into_inner();
                    loop {
                        match chunks.message().await {
                            Ok(Some(chunk)) => {
                                if sender.send(Ok(chunk.text)).await.is_err() {
                                    break;
                                }
                            }
                            Ok(None) => break,
                            Err(status) => {
                                let _ = sender.send(Err(status)).await;
                                break;
                            }
                        }
                    }
                }
            }
        });
        Ok(Box::new(RemoteTextStream { client: self, receiver, task, cancelled, done: false }))
    }

    fn encode_texts(
        &self,
        texts: &[String],
        base_dir: Option<&str>,
    ) -> anyhow::Result<Option<Array2<f32>>> {
        Ok(self.encode_texts_with_timeout(texts, base_dir, None)?)
    }

    fn is_loaded(&self) -> anyhow::Result<bool> {
        let reply = self.unary(self.project_request(), None, |mut stub, request| async move {
            stub.is_loaded(request).await
        })?;
        Ok(reply.value)
    }

    fn supports_thinking(&self, capability: Option<&str>) -> anyhow::Result<bool> {
        let request = pb::SupportsThinkingRequest {
            capability: capability.unwrap_or_default().to_string(),
            project_id: pinned_project_id().unwrap_or_default(),
        };
        let reply = self.unary(request, None, |mut stub, request| async move {
            stub.supports_thinking(request).await
        })?;
        Ok(reply.value)
    }

    fn model_name(&self) -> anyhow::Result<Option<String>> {
        let reply = self.unary(self.project_request(), None, |mut stub, request| async move {
            stub.model_name(request).await
        })?;
        Ok(Some(reply.value).filter(|name| !name.is_empty()))
    }

    fn model_label(&self, capability: Option<&str>) -> anyhow::Result<String> {
        let request = pb::ModelLabelRequest {
            capability: capability.unwrap_or_default().to_string(),
            project_id: pinned_project_id().unwrap_or_default(),
        };
        let reply = self.unary(request, None, |mut stub, request| async move {
            stub.model_label(request).await
        })?;
        Ok(reply.value)
    }

    fn count_tokens(&self, text: &str) -> anyhow::Result<usize> {
        let request = pb::CountTokensRequest {
            text: text.to_string(),
            project_id: pinned_project_id().unwrap_or_default(),
        };
        let reply = self.unary(request, None, |mut stub, request| async move {
            stub.count_tokens(request).await
        })?;
        Ok(reply.value.max(0) as usize)
    }

    fn max_context_tokens(&self) -> anyhow::Result<usize> {
        let reply = self.unary(self.project_request(), None, |mut stub, request| async move {
            stub.max_context_tokens(request).await
        })?;
        Ok(reply.value.max(0) as usize)
    }

    fn response_max_context_tokens(&self, capability: Option<&str>) -> anyhow::Result<usize> {
        let request = pb::ResponseMaxContextRequest {
            capability: capability.unwrap_or_default().to_string(),
            project_id: pinned_project_id().unwrap_or_default(),
        };
        let reply = self.unary(request, None, |mut stub, request| async move {
            stub.response_max_context_tokens(request).await
        })?;
        Ok(reply.value.max(0) as usize)
    }

    fn has_capability_model(&self, capability: &str) -> anyhow::Result<bool> {
        let request = pb::HasCapabilityRequest {
            capability: capability.to_string(),
            project_id: pinned_project_id().unwrap_or_default(),
        };
        let reply = self.unary(request, None, |mut stub, request| async move {
            stub.has_capability_model(request).await
        })?;
        Ok(reply.value)
    }

    fn idle_seconds(&self) -> anyhow::Result<f64> {
        let reply = self.unary(self.project_request(), None, |mut stub, request| async move {
            stub.idle_seconds(request).await
        })?;
        Ok(reply.value)
    }

    fn load_model(&self) -> anyhow::Result<Option<String>> {
        let reply = self.unary(self.project_request(), None, |mut stub, request| async move {
            stub.load_model(request).await
        })?;
        Ok(Some(reply.value).filter(|name| !name.is_empty()))
    }

    fn unload_auxiliary_models(&self, include_embedding: bool) -> anyhow::Result<()> {
        let request = pb::UnloadAuxiliaryRequest {
            include_embedding,
            project_id: pinned_project_id().unwrap_or_default(),
        };
        self.unary(request, None, |mut stub, request| async move {
            stub.unload_auxiliary_models(request).await
        })?;
        Ok(())
    }

    fn unload_embedding_model(&self) -> anyhow::Result<()> {
        self.unary(self.project_request(), None, |mut stub, request| async move {
            stub.unload_embedding_model(request).await
        })?;
        Ok(())
    }

    fn close(&self) {
        self.drop_channel();
    }

    fn type_name(&self) -> String {
        "RemoteModelClient".to_string()
    }

    fn as_remote(&self) -> Option<&RemoteModelClient> {
        Some(self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// grpcio's spelling of a status code (`StatusCode.UNAVAILABLE` minus the
/// prefix).
fn status_code_name(code: Code) -> &'static str {
    match code {
        Code::Ok => "OK",
        Code::Cancelled => "CANCELLED",
        Code::Unknown => "UNKNOWN",
        Code::InvalidArgument => "INVALID_ARGUMENT",
        Code::DeadlineExceeded => "DEADLINE_EXCEEDED",
        Code::NotFound => "NOT_FOUND",
        Code::AlreadyExists => "ALREADY_EXISTS",
        Code::PermissionDenied => "PERMISSION_DENIED",
        Code::ResourceExhausted => "RESOURCE_EXHAUSTED",
        Code::FailedPrecondition => "FAILED_PRECONDITION",
        Code::Aborted => "ABORTED",
        Code::OutOfRange => "OUT_OF_RANGE",
        Code::Unimplemented => "UNIMPLEMENTED",
        Code::Internal => "INTERNAL",
        Code::Unavailable => "UNAVAILABLE",
        Code::DataLoss => "DATA_LOSS",
        Code::Unauthenticated => "UNAUTHENTICATED",
    }
}

/// Describe a failed RPC the way a failing socket used to be described.
fn rpc_error(address: &str, status: &tonic::Status) -> ModelHostError {
    let code = status.code();
    let details = if status.message().is_empty() {
        format!("StatusCode.{}", status_code_name(code))
    } else {
        status.message().to_string()
    };
    match code {
        Code::Unavailable => {
            ModelHostError::new(format!("model host at {address} is unreachable: {details}"))
        }
        Code::Cancelled => {
            ModelHostError::new(format!("model host at {address} cancelled the call: {details}"))
        }
        _ => ModelHostError::new(format!("model host at {address} failed: {details}")),
    }
}

/// Python's `repr()` of a string, for messages that quoted an address with
/// `{address!r}`.
fn py_repr(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(value.len() + 2);
    out.push(quote);
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Split `address` into `(path, None)` or `(host, Some(port))`.
///
/// A unix socket path is the normal case.  `host:port` is what Windows gets —
/// no `AF_UNIX` there — and is also useful for debugging a host that is not on
/// the same filesystem.
pub fn parse_address(address: &str) -> (String, Option<u32>) {
    if let Some(rest) = address.strip_prefix("unix://") {
        return (rest.to_string(), None);
    }
    if let Some(rest) = address.strip_prefix("unix:") {
        return (rest.to_string(), None);
    }
    if address.contains(std::path::MAIN_SEPARATOR) || address.starts_with('.') {
        return (address.to_string(), None);
    }
    if let Some((host, port)) = address.rsplit_once(':')
        && !port.is_empty()
        && port.chars().all(|c| c.is_ascii_digit())
        && let Ok(port) = port.parse::<u32>()
    {
        let host = if host.is_empty() { "127.0.0.1" } else { host };
        return (host.to_string(), Some(port));
    }
    (address.to_string(), None)
}

/// How gRPC addresses `address`: `unix:`-prefixed or `host:port`.
pub fn grpc_target(address: &str) -> String {
    let (host, port) = parse_address(address);
    match port {
        None => format!("unix:{host}"),
        Some(port) => format!("{host}:{port}"),
    }
}

/// `os.path.abspath`: absolute and normalised, without resolving symlinks.
fn abspath(path: &str) -> String {
    let path = Path::new(path);
    let joined: PathBuf = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };
    let mut out = PathBuf::new();
    for component in joined.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out.to_string_lossy().into_owned()
}

/// Where the model host listens when nothing says otherwise.
///
/// Normally `<home>/model-host.sock` — a filesystem object, so the OS enforces
/// who may connect.  Two things take that away:
///
/// * **No AF_UNIX at all.**  Windows has none, so there is nothing to bind and
///   the host listens on loopback TCP instead.
/// * **A home too deep for `sun_path`.**  It is 104 bytes, and the macOS App
///   Sandbox puts the home under `~/Library/Containers/<bundle id>/Data`.  A
///   short name in the system temp directory, keyed by the home so two
///   workspaces never share one, usually fits; loopback is the last resort.
///
/// Loopback binds to 127.0.0.1, so nothing off-machine can reach it — but it
/// has no file permissions, so any local process may connect where a unix
/// socket would have said no.  That is the trade Windows makes today.
pub fn default_socket_path(home_dir: &str) -> String {
    if !has_unix_sockets() {
        tracing::info!(target: LOG_TARGET, "[model-host] no AF_UNIX on this platform; listening on loopback");
        return "127.0.0.1:0".to_string();
    }

    let path = Path::new(home_dir).join("model-host.sock").to_string_lossy().into_owned();
    if path.len() <= MAX_UNIX_SOCKET_PATH {
        return path;
    }

    let digest = hex::encode(Sha256::digest(abspath(home_dir).as_bytes()));
    let short = std::env::temp_dir()
        .join(format!("slp-model-host-{}.sock", &digest[..12]))
        .to_string_lossy()
        .into_owned();
    if short.len() <= MAX_UNIX_SOCKET_PATH {
        tracing::info!(
            target: LOG_TARGET,
            "[model-host] {} is too long for a unix socket; using {}", path, short
        );
        return short;
    }

    tracing::info!(target: LOG_TARGET, "[model-host] no usable unix socket path; listening on loopback instead");
    "127.0.0.1:0".to_string()
}

/// The client this process should use.
///
/// A project child gets a host address from `SLP_MODEL_HOST`; a process that
/// owns the device, or one that has not started a host yet, must stay on the
/// in-process path instead of trying to reach a placeholder `127.0.0.1:0`.
///
/// A port of zero is not a real bind and must never be treated as a usable
/// model host address; it indicates the host was not started or was asked to
/// bind on an ephemeral port that never got published.
pub fn get_model_client(
    model_service: Option<Arc<dyn LocalModelService>>,
) -> Result<Arc<dyn ModelClient>, ModelHostError> {
    let mut address = std::env::var("SLP_MODEL_HOST").unwrap_or_default().trim().to_string();
    if address.is_empty() {
        // No explicit host means there is no model-host process to reach in
        // this process.  The single-server and local-owner paths remain
        // in-process; a child without a host address should never silently
        // fall through to a remote connection against a placeholder port.
        if let Some(service) = model_service {
            return Ok(Arc::new(InProcessModelClient::new(service)));
        }
        address = default_socket_path(&AppSettings::new().home_dir());
    }

    if parse_address(&address).1 == Some(0) {
        return Err(ModelHostError::new(format!(
            "model host address {} is not bound: a real host port is required before remote \
             model access",
            py_repr(&address)
        )));
    }
    Ok(Arc::new(RemoteModelClient::new(address)))
}
