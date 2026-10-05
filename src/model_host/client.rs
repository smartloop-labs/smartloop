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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ndarray::Array2;
use parking_lot::Mutex;

use crate::api::services::model_service::ModelService;
use crate::llms::base_llm::{CancelEvent, GenerationChunk};
use crate::model_host::pb;
#[allow(unused_imports)]
use crate::workspace::context::pinned_project_id;

/// Logger name used by the Python module.
#[allow(dead_code)]
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
        let _ = project_id;
        todo!()
    }

    fn generate_completion(
        &self,
        prompt: &str,
        model_name: Option<&str>,
        capability: Option<&str>,
        max_tokens: Option<usize>,
        thinking: Option<bool>,
    ) -> anyhow::Result<String> {
        let _ = (prompt, model_name, capability, max_tokens, thinking);
        todo!()
    }

    fn stream_completion<'a>(
        &'a self,
        prompt: &str,
        model_name: Option<&str>,
        cancelled: Option<CancelEvent>,
        capability: Option<&str>,
        thinking: Option<bool>,
    ) -> anyhow::Result<Box<dyn Iterator<Item = anyhow::Result<GenerationChunk>> + 'a>> {
        let _ = (prompt, model_name, cancelled, capability, thinking);
        todo!()
    }

    fn is_loaded(&self) -> bool {
        todo!()
    }

    fn supports_thinking(&self, capability: Option<&str>) -> bool {
        let _ = capability;
        todo!()
    }

    fn resident_model_name(&self) -> Option<String> {
        todo!()
    }

    fn response_model_label(&self, capability: Option<&str>) -> String {
        let _ = capability;
        todo!()
    }

    fn count_tokens(&self, text: &str) -> usize {
        let _ = text;
        todo!()
    }

    fn get_max_context_tokens(&self) -> usize {
        todo!()
    }

    fn get_response_max_context_tokens(&self, capability: Option<&str>) -> usize {
        let _ = capability;
        todo!()
    }

    fn has_capability_model(&self, capability: &str) -> bool {
        let _ = capability;
        todo!()
    }

    fn idle_seconds(&self) -> f64 {
        todo!()
    }

    fn load_model(&self) -> anyhow::Result<()> {
        todo!()
    }

    fn unload_auxiliary_models(&self, include_embedding: bool) {
        let _ = include_embedding;
        todo!()
    }

    fn unload_embedding_model(&self) {
        todo!()
    }
}

/// The device is owned by this process — call straight through.
///
/// Every method here is the call the caller used to make directly, so
/// selecting this client changes nothing about how a request is served.
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
        let _ = (prompt, model_name, capability, max_tokens, thinking);
        todo!()
    }

    /// `GenerationChunk` in, plain text out — the wire carries text, so the
    /// in-process path yields the same thing.
    fn stream<'a>(
        &'a self,
        prompt: &str,
        model_name: Option<&str>,
        capability: Option<&str>,
        cancelled: Option<CancelEvent>,
        thinking: Option<bool>,
    ) -> anyhow::Result<TextStream<'a>> {
        let _ = (prompt, model_name, capability, cancelled, thinking);
        todo!()
    }

    /// `encode_local`, not `encode_texts`: this process owns the embedder, and
    /// this is also the host's own serving path — routing would let a host
    /// serve `EncodeTexts` by sending one.
    fn encode_texts(
        &self,
        texts: &[String],
        base_dir: Option<&str>,
    ) -> anyhow::Result<Option<Array2<f32>>> {
        let _ = (texts, base_dir);
        todo!()
    }

    fn is_loaded(&self) -> anyhow::Result<bool> {
        todo!()
    }

    fn supports_thinking(&self, capability: Option<&str>) -> anyhow::Result<bool> {
        let _ = capability;
        todo!()
    }

    fn model_name(&self) -> anyhow::Result<Option<String>> {
        todo!()
    }

    fn model_label(&self, capability: Option<&str>) -> anyhow::Result<String> {
        let _ = capability;
        todo!()
    }

    fn count_tokens(&self, text: &str) -> anyhow::Result<usize> {
        let _ = text;
        todo!()
    }

    fn max_context_tokens(&self) -> anyhow::Result<usize> {
        todo!()
    }

    fn response_max_context_tokens(&self, capability: Option<&str>) -> anyhow::Result<usize> {
        let _ = capability;
        todo!()
    }

    fn has_capability_model(&self, capability: &str) -> anyhow::Result<bool> {
        let _ = capability;
        todo!()
    }

    fn idle_seconds(&self) -> anyhow::Result<f64> {
        todo!()
    }

    fn load_model(&self) -> anyhow::Result<Option<String>> {
        todo!()
    }

    fn unload_auxiliary_models(&self, include_embedding: bool) -> anyhow::Result<()> {
        let _ = include_embedding;
        todo!()
    }

    fn unload_embedding_model(&self) -> anyhow::Result<()> {
        todo!()
    }

    /// The workspace is right here, so a child's write is one read away.
    fn refresh_project(&self, project_id: Option<&str>) -> anyhow::Result<bool> {
        let _ = project_id;
        todo!()
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

/// The live transport: a gRPC client over a lazily-opened channel.
type GrpcClient = pb::model_host_client::ModelHostClient<tonic::transport::Channel>;

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

    /// Open the channel if needed and return the stub.
    #[allow(dead_code)]
    fn connect(&self) -> Result<GrpcClient, ModelHostError> {
        todo!()
    }

    /// Close the channel and forget it, so the next call reconnects.
    #[allow(dead_code)]
    fn drop_channel(&self) {
        todo!()
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
        let _ = (texts, base_dir, timeout);
        todo!()
    }

    /// The `ProjectRequest` carrying this process's pinned project.
    #[allow(dead_code)]
    fn project_request(&self) -> pb::ProjectRequest {
        todo!()
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
        let _ = (prompt, model_name, capability, max_tokens, thinking);
        todo!()
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
        let _ = (prompt, model_name, capability, cancelled, thinking);
        todo!()
    }

    fn encode_texts(
        &self,
        texts: &[String],
        base_dir: Option<&str>,
    ) -> anyhow::Result<Option<Array2<f32>>> {
        Ok(self.encode_texts_with_timeout(texts, base_dir, None)?)
    }

    fn is_loaded(&self) -> anyhow::Result<bool> {
        todo!()
    }

    fn supports_thinking(&self, capability: Option<&str>) -> anyhow::Result<bool> {
        let _ = capability;
        todo!()
    }

    fn model_name(&self) -> anyhow::Result<Option<String>> {
        todo!()
    }

    fn model_label(&self, capability: Option<&str>) -> anyhow::Result<String> {
        let _ = capability;
        todo!()
    }

    fn count_tokens(&self, text: &str) -> anyhow::Result<usize> {
        let _ = text;
        todo!()
    }

    fn max_context_tokens(&self) -> anyhow::Result<usize> {
        todo!()
    }

    fn response_max_context_tokens(&self, capability: Option<&str>) -> anyhow::Result<usize> {
        let _ = capability;
        todo!()
    }

    fn has_capability_model(&self, capability: &str) -> anyhow::Result<bool> {
        let _ = capability;
        todo!()
    }

    fn idle_seconds(&self) -> anyhow::Result<f64> {
        todo!()
    }

    fn load_model(&self) -> anyhow::Result<Option<String>> {
        todo!()
    }

    fn unload_auxiliary_models(&self, include_embedding: bool) -> anyhow::Result<()> {
        let _ = include_embedding;
        todo!()
    }

    fn unload_embedding_model(&self) -> anyhow::Result<()> {
        todo!()
    }

    fn close(&self) {
        todo!()
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

/// Describe a failed RPC the way a failing socket used to be described.
#[allow(dead_code)]
fn rpc_error(address: &str, status: &tonic::Status) -> ModelHostError {
    let _ = (address, status);
    todo!()
}

/// Split `address` into `(path, None)` or `(host, Some(port))`.
///
/// A unix socket path is the normal case.  `host:port` is what Windows gets —
/// no `AF_UNIX` there — and is also useful for debugging a host that is not on
/// the same filesystem.
pub fn parse_address(address: &str) -> (String, Option<u32>) {
    let _ = address;
    todo!()
}

/// How gRPC addresses `address`: `unix:`-prefixed or `host:port`.
pub fn grpc_target(address: &str) -> String {
    let _ = address;
    todo!()
}

/// Where the model host listens when nothing says otherwise.
///
/// Normally `<home>/model-host.sock`.  Without AF_UNIX, or when the home is
/// too deep for `sun_path`, a short name in the system temp directory keyed by
/// the home is tried, and loopback (`127.0.0.1:0`) is the last resort.
pub fn default_socket_path(home_dir: &str) -> String {
    let _ = home_dir;
    todo!()
}

/// The client this process should use.
///
/// A project child gets a host address from `SLP_MODEL_HOST`; a process that
/// owns the device, or one that has not started a host yet, stays on the
/// in-process path.  A port of zero is never a usable model host address.
pub fn get_model_client(
    model_service: Option<Arc<dyn LocalModelService>>,
) -> Result<Arc<dyn ModelClient>, ModelHostError> {
    let _ = model_service;
    todo!()
}
