//! Port of `smartloop/model_host/host.py`.
//!
//! The process that owns the device, serving generation to project agents.
//!
//! One host for the whole workspace, never one per project.  Per-project copies
//! of the generative model were tried and reverted: weights loaded through mmap
//! share page cache between processes on CPU, but VRAM copies do not, so N
//! projects would mean N resident copies competing for the same device, each
//! with its own KV cache, serialising in the driver.  Keeping one host also
//! keeps `n_parallel` continuous batching meaningful — a single budget, shared
//! — and loads the ~417 MB BGE-M3 embedder once instead of once per child.
//!
//! The host is deliberately thin: it holds the models and answers the
//! operations in [`ModelClient`].  Everything project-shaped stays in the agent
//! that asked.
//!
//! Transport: gRPC is the only supported IPC path.  It listens on a unix socket
//! where one exists and loopback TCP where it does not (Windows); there is no
//! second framing mode or direct socket fallback.

use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use parking_lot::Mutex;
use tonic::{Request, Response, Status};

use crate::api::services::model_service::ModelService;
use crate::model_host::client::{
    default_socket_path, has_unix_sockets, mark_device_owner, parse_address,
    InProcessModelClient, LocalModelService, ModelClient,
};
use crate::model_host::pb;
use crate::model_host::pb::model_host_server::ModelHost as ModelHostService;
#[allow(unused_imports)]
use crate::workspace::context::use_project;

/// Logger name used by the Python module.
#[allow(dead_code)]
const LOG_TARGET: &str = "smartloop.model_host.host";

/// grpcio's default cap on a message is 4 MiB; an embedding batch can exceed
/// that.  This matches what the old frame ceiling bounded instead.
pub const MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

/// gRPC worker threads.  Inference serialises on the served client's own
/// semaphore (`n_parallel`), so the pool is sized for streams parked on a
/// generation, not for concurrent device work.
#[allow(dead_code)]
const WORKERS: usize = 16;

/// A running (or bound but not yet started) gRPC server.
struct ServerHandle {
    #[allow(dead_code)]
    inner: Box<dyn std::any::Any + Send>,
}

/// Serves one [`ModelClient`] to any number of project agents.
///
/// Each RPC runs on a blocking worker; the served client's inference semaphore
/// is what bounds concurrency across them, which is the point: the device has
/// one budget, and it is enforced in one place.
pub struct ModelHost {
    /// The client every RPC is answered against.
    pub client: Arc<dyn ModelClient>,
    address: Mutex<String>,
    server: Mutex<Option<ServerHandle>>,
}

impl ModelHost {
    /// A host serving `client` at `address` (socket path or `host:port`).
    ///
    /// This process holds the weights, so anything in it that would otherwise
    /// reach for a model host resolves locally instead.
    pub fn new(client: Arc<dyn ModelClient>, address: impl Into<String>) -> Self {
        mark_device_owner();
        Self {
            client,
            address: Mutex::new(address.into()),
            server: Mutex::new(None),
        }
    }

    /// The address: as configured before [`bind`](Self::bind), the address
    /// actually bound after it.
    pub fn address(&self) -> String {
        self.address.lock().clone()
    }

    // -- lifecycle ----------------------------------------------------------

    /// Create the listening socket. Returns the address actually bound.
    pub fn bind(&self) -> std::io::Result<String> {
        let _ = (&self.server, has_unix_sockets, parse_address);
        todo!()
    }

    /// Bind and start serving. Returns the bound address.
    pub fn start(&self) -> std::io::Result<String> {
        todo!()
    }

    /// The control channel is for the user's own agents; keep it private.
    #[allow(dead_code)]
    fn chmod_socket_if_any(&self) {
        todo!()
    }

    /// Serve until [`stop`](Self::stop) is called.
    pub fn serve_forever(&self) -> std::io::Result<()> {
        todo!()
    }

    /// Stop accepting and drop live RPCs.
    pub fn stop(&self) {
        todo!()
    }
}

/// Answers each RPC against the requesting project's model selection.
///
/// The host refreshes its client before every call and then binds the
/// requesting project for the whole operation — otherwise a model disabled in
/// a project kept answering for it.
pub struct ModelHostServicer {
    /// The client every RPC is answered against.
    pub client: Arc<dyn ModelClient>,
}

impl ModelHostServicer {
    /// A servicer answering against `client`.
    pub fn new(client: Arc<dyn ModelClient>) -> Self {
        Self { client }
    }

    /// Refresh `project_id`'s state (a failure is logged, not raised), then
    /// run `op` with that project bound.
    pub fn bind<R>(&self, project_id: &str, op: impl FnOnce(&dyn ModelClient) -> R) -> R {
        let _ = (project_id, op);
        todo!()
    }
}

/// Server-streaming response type for `Stream`.
pub type StreamChunkStream = Pin<Box<dyn Stream<Item = Result<pb::StreamChunk, Status>> + Send>>;

#[tonic::async_trait]
impl ModelHostService for ModelHostServicer {
    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<pb::GenerateResponse>, Status> {
        let _ = request;
        todo!()
    }

    type StreamStream = StreamChunkStream;

    /// gRPC cancellation is the cancellation mechanism: when the client drops
    /// the RPC, the served model's cancel event is raised between chunks.
    async fn stream(
        &self,
        request: Request<pb::StreamRequest>,
    ) -> Result<Response<Self::StreamStream>, Status> {
        let _ = request;
        todo!()
    }

    async fn encode_texts(
        &self,
        request: Request<pb::EncodeTextsRequest>,
    ) -> Result<Response<pb::EncodeTextsResponse>, Status> {
        let _ = request;
        todo!()
    }

    async fn is_loaded(
        &self,
        request: Request<pb::ProjectRequest>,
    ) -> Result<Response<pb::BoolReply>, Status> {
        let _ = request;
        todo!()
    }

    async fn supports_thinking(
        &self,
        request: Request<pb::SupportsThinkingRequest>,
    ) -> Result<Response<pb::BoolReply>, Status> {
        let _ = request;
        todo!()
    }

    async fn model_name(
        &self,
        request: Request<pb::ProjectRequest>,
    ) -> Result<Response<pb::StringReply>, Status> {
        let _ = request;
        todo!()
    }

    async fn model_label(
        &self,
        request: Request<pb::ModelLabelRequest>,
    ) -> Result<Response<pb::StringReply>, Status> {
        let _ = request;
        todo!()
    }

    async fn count_tokens(
        &self,
        request: Request<pb::CountTokensRequest>,
    ) -> Result<Response<pb::IntReply>, Status> {
        let _ = request;
        todo!()
    }

    async fn max_context_tokens(
        &self,
        request: Request<pb::ProjectRequest>,
    ) -> Result<Response<pb::IntReply>, Status> {
        let _ = request;
        todo!()
    }

    async fn response_max_context_tokens(
        &self,
        request: Request<pb::ResponseMaxContextRequest>,
    ) -> Result<Response<pb::IntReply>, Status> {
        let _ = request;
        todo!()
    }

    async fn has_capability_model(
        &self,
        request: Request<pb::HasCapabilityRequest>,
    ) -> Result<Response<pb::BoolReply>, Status> {
        let _ = request;
        todo!()
    }

    async fn idle_seconds(
        &self,
        request: Request<pb::ProjectRequest>,
    ) -> Result<Response<pb::FloatReply>, Status> {
        let _ = request;
        todo!()
    }

    async fn load_model(
        &self,
        request: Request<pb::ProjectRequest>,
    ) -> Result<Response<pb::StringReply>, Status> {
        let _ = request;
        todo!()
    }

    async fn unload_auxiliary_models(
        &self,
        request: Request<pb::UnloadAuxiliaryRequest>,
    ) -> Result<Response<pb::Empty>, Status> {
        let _ = request;
        todo!()
    }

    async fn unload_embedding_model(
        &self,
        request: Request<pb::ProjectRequest>,
    ) -> Result<Response<pb::Empty>, Status> {
        let _ = request;
        todo!()
    }
}

/// Fail the RPC with the device error (`INTERNAL`, `"<Type>: <message>"`),
/// without killing the server.
#[allow(dead_code)]
fn abort(op: &str, exc: &anyhow::Error) -> Status {
    let _ = (op, exc);
    todo!()
}

/// Remove a socket file no one is listening on.
///
/// A host killed with SIGKILL leaves its file behind; binding over it fails
/// with EADDRINUSE even though the address is free.  Probing first means a
/// *live* host is never unlinked out from under its clients — the restart
/// fails loudly instead.
pub fn unlink_stale_socket(path: &str) -> std::io::Result<()> {
    let _ = path;
    todo!()
}

/// Run a model host in this process, blocking until it is stopped.
///
/// Used by `slp __model-host`; the supervisor spawns exactly one of these and
/// points every project agent at it.
pub fn serve(
    address: Option<&str>,
    model_service: Option<Arc<dyn LocalModelService>>,
) -> anyhow::Result<ModelHost> {
    let _ = (address, model_service, default_socket_path);
    let _: Option<&ModelService> = None;
    let _ = InProcessModelClient::new;
    todo!()
}
