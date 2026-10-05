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
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures::Stream;
use ndarray::Array2;
use parking_lot::{Condvar, Mutex};
use tonic::{Request, Response, Status};

use crate::model_host::client::{
    default_socket_path, has_unix_sockets, mark_device_owner, parse_address,
    InProcessModelClient, LocalModelService, ModelClient, ModelHostError,
};
use crate::model_host::pb;
use crate::model_host::pb::model_host_server::{ModelHost as ModelHostService, ModelHostServer};
use crate::workspace::context::use_project;

/// Logger name used by the Python module.
const LOG_TARGET: &str = "smartloop.model_host.host";

/// grpcio's default cap on a message is 4 MiB; an embedding batch can exceed
/// that.  This matches what the old frame ceiling bounded instead.
pub const MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

/// gRPC worker threads.  Inference serialises on the served client's own
/// semaphore (`n_parallel`), so the pool is sized for streams parked on a
/// generation, not for concurrent device work.
const WORKERS: usize = 16;

/// How long [`ModelHost::stop`] waits for the server's own threads to wind
/// down.  Live RPCs are dropped at once (grace of zero); this only bounds the
/// wait for a device call already running on a worker, which cannot be
/// interrupted mid-token and is simply left to finish on its own.
const SHUTDOWN_WAIT: Duration = Duration::from_secs(2);

/// A socket bound by [`ModelHost::bind`], not yet handed to the server.
enum BoundListener {
    #[cfg(unix)]
    Unix(std::os::unix::net::UnixListener),
    Tcp(std::net::TcpListener),
}

/// A running (or bound but not yet started) gRPC server.
struct ServerHandle {
    /// The listening socket, until [`ModelHost::start`] hands it over.
    listener: Option<BoundListener>,
    /// The runtime the server runs on, once started.
    runtime: Option<tokio::runtime::Runtime>,
}

/// Signalled by [`ModelHost::stop`] so [`ModelHost::serve_forever`] returns.
#[derive(Default)]
struct Termination {
    terminated: Mutex<bool>,
    changed: Condvar,
}

impl Termination {
    fn reset(&self) {
        *self.terminated.lock() = false;
    }

    fn signal(&self) {
        *self.terminated.lock() = true;
        self.changed.notify_all();
    }

    fn wait(&self) {
        let mut terminated = self.terminated.lock();
        while !*terminated {
            self.changed.wait(&mut terminated);
        }
    }
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
    termination: Arc<Termination>,
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
            termination: Arc::new(Termination::default()),
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
        let address = self.address();
        let (path, port) = parse_address(&address);

        let (listener, bound) = match port {
            None => {
                if !has_unix_sockets() {
                    return Err(std::io::Error::other(format!(
                        "cannot listen on {path}: this platform has no AF_UNIX. \
                         Give the host a host:port address instead."
                    )));
                }
                // A socket file left by a crashed host would otherwise make
                // the bind fail even though nothing is listening.
                unlink_stale_socket(&path)?;
                let parent = std::path::Path::new(&path)
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| std::path::PathBuf::from("."));
                std::fs::create_dir_all(&parent)?;
                // The socket file is locked down once the server starts.
                (bind_unix(&path)?, path.clone())
            }
            Some(port) => {
                let listener = u16::try_from(port)
                    .ok()
                    .and_then(|p| std::net::TcpListener::bind((path.as_str(), p)).ok())
                    .ok_or_else(|| {
                        std::io::Error::other(format!("cannot listen on {path}:{port}"))
                    })?;
                let got = listener.local_addr()?.port();
                if got == 0 {
                    return Err(std::io::Error::other(format!("cannot listen on {path}:{port}")));
                }
                (BoundListener::Tcp(listener), format!("{path}:{got}"))
            }
        };

        *self.server.lock() = Some(ServerHandle { listener: Some(listener), runtime: None });
        *self.address.lock() = bound.clone();
        self.termination.reset();
        tracing::info!(target: LOG_TARGET, "[model-host] listening on {}", bound);
        Ok(bound)
    }

    /// Bind and start serving. Returns the bound address.
    pub fn start(&self) -> std::io::Result<String> {
        let bound = self.bind()?;
        self.start_server()?;
        self.chmod_socket_if_any();
        Ok(bound)
    }

    /// Hand the bound listener to a gRPC server running on the host's own
    /// runtime.
    fn start_server(&self) -> std::io::Result<()> {
        let mut guard = self.server.lock();
        let handle = guard
            .as_mut()
            .ok_or_else(|| std::io::Error::other("bind() did not create a server"))?;
        let Some(listener) = handle.listener.take() else {
            // Already started.
            return Ok(());
        };

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .thread_name("slp-model-host")
            .max_blocking_threads(WORKERS)
            .enable_all()
            .build()?;

        let service = ModelHostServer::new(ModelHostServicer::new(self.client.clone()))
            .max_decoding_message_size(MAX_MESSAGE_BYTES)
            .max_encoding_message_size(MAX_MESSAGE_BYTES);

        {
            let _enter = runtime.enter();
            match listener {
                #[cfg(unix)]
                BoundListener::Unix(listener) => {
                    listener.set_nonblocking(true)?;
                    let listener = tokio::net::UnixListener::from_std(listener)?;
                    let incoming = async_stream::stream! {
                        loop {
                            yield listener.accept().await.map(|(stream, _)| stream);
                        }
                    };
                    runtime.spawn(async move {
                        let served = tonic::transport::Server::builder()
                            .add_service(service)
                            .serve_with_incoming(incoming)
                            .await;
                        if let Err(exc) = served {
                            tracing::warn!(target: LOG_TARGET, "[model-host] server failed: {exc}");
                        }
                    });
                }
                BoundListener::Tcp(listener) => {
                    listener.set_nonblocking(true)?;
                    let listener = tokio::net::TcpListener::from_std(listener)?;
                    let incoming = async_stream::stream! {
                        loop {
                            yield listener.accept().await.map(|(stream, _)| {
                                let _ = stream.set_nodelay(true);
                                stream
                            });
                        }
                    };
                    runtime.spawn(async move {
                        let served = tonic::transport::Server::builder()
                            .add_service(service)
                            .serve_with_incoming(incoming)
                            .await;
                        if let Err(exc) = served {
                            tracing::warn!(target: LOG_TARGET, "[model-host] server failed: {exc}");
                        }
                    });
                }
            }
        }

        handle.runtime = Some(runtime);
        Ok(())
    }

    /// The control channel is for the user's own agents; keep it private.
    fn chmod_socket_if_any(&self) {
        let (path, port) = parse_address(&self.address());
        if port.is_none() {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
            }
            #[cfg(not(unix))]
            let _ = path;
        }
    }

    /// Serve until [`stop`](Self::stop) is called.
    pub fn serve_forever(&self) -> std::io::Result<()> {
        if self.server.lock().is_none() {
            self.bind()?;
        }
        self.start_server()?;
        self.chmod_socket_if_any();
        self.termination.wait();
        Ok(())
    }

    /// Stop accepting and drop live RPCs.
    pub fn stop(&self) {
        let server = self.server.lock().take();
        if let Some(runtime) = server.and_then(|handle| handle.runtime) {
            // Grace of zero: in-flight RPCs are cancelled at once, matching the
            // old behaviour of closing every live socket on shutdown.  Done on
            // a thread of its own so stopping from async code (the supervisor)
            // cannot trip the runtime's "no blocking here" guard.
            let joined = std::thread::Builder::new()
                .name("slp-model-host-stop".into())
                .spawn(move || runtime.shutdown_timeout(SHUTDOWN_WAIT))
                .map(|thread| thread.join());
            if let Err(exc) = joined {
                tracing::warn!(target: LOG_TARGET, "[model-host] could not stop cleanly: {exc}");
            }
        }
        let (path, port) = parse_address(&self.address());
        if port.is_none() {
            let _ = std::fs::remove_file(&path);
        }
        self.termination.signal();
        tracing::info!(target: LOG_TARGET, "[model-host] stopped");
    }
}

impl Drop for ModelHost {
    fn drop(&mut self) {
        // The Python host was a context manager; a host dropped while still
        // serving releases its runtime (and socket file) here.
        if self.server.lock().as_ref().is_some_and(|handle| handle.runtime.is_some()) {
            self.stop();
        }
    }
}

#[cfg(unix)]
fn bind_unix(path: &str) -> std::io::Result<BoundListener> {
    Ok(BoundListener::Unix(std::os::unix::net::UnixListener::bind(path)?))
}

#[cfg(not(unix))]
fn bind_unix(path: &str) -> std::io::Result<BoundListener> {
    Err(std::io::Error::other(format!(
        "cannot listen on {path}: this platform has no AF_UNIX. \
         Give the host a host:port address instead."
    )))
}

/// Refresh `project_id`'s state (a failure is logged, not raised), then run
/// `op` with that project bound.
fn bind_project<R>(
    client: &dyn ModelClient,
    project_id: &str,
    op: impl FnOnce(&dyn ModelClient) -> R,
) -> R {
    let pid = Some(project_id).filter(|p| !p.is_empty());
    let refreshed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        client.refresh_project(pid)
    }));
    match refreshed {
        Ok(Ok(_)) => {}
        // A project whose own file cannot be read is a reason to answer from
        // what we have, not to fail the generation.
        Ok(Err(exc)) => tracing::warn!(
            target: LOG_TARGET,
            "[model-host] could not refresh project {}: {}", project_id, exc
        ),
        Err(panic) => tracing::warn!(
            target: LOG_TARGET,
            "[model-host] could not refresh project {}: {}", project_id, panic_message(&*panic)
        ),
    }
    let _binding = use_project(pid);
    op(client)
}

/// Answers each RPC against the requesting project's model selection.
///
/// Model selection depends on per-project settings, and those are written by
/// the child pinned to the project while the host runs in the supervisor,
/// holding a copy taken when it first saw the project.  Nothing invalidates
/// that copy, so the host refreshes its client before every call and then
/// binds the requesting project for the whole operation — otherwise a model
/// disabled in a project kept answering for it, and the picker said one thing
/// while the host resolved another.
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
        bind_project(self.client.as_ref(), project_id, op)
    }

    /// Run `op` on a blocking worker with `project_id` bound, the way grpcio
    /// ran every handler on a pool thread.  A device call blocks for the whole
    /// generation, so it must never occupy an async worker.
    async fn blocking<R, F>(&self, project_id: String, op: F) -> Result<R, Status>
    where
        R: Send + 'static,
        F: FnOnce(&dyn ModelClient) -> R + Send + 'static,
    {
        let client = self.client.clone();
        tokio::task::spawn_blocking(move || bind_project(client.as_ref(), &project_id, op))
            .await
            .map_err(|exc| {
                let message = if exc.is_panic() {
                    panic_message(&*exc.into_panic())
                } else {
                    exc.to_string()
                };
                application_error(&message)
            })
    }

    /// A unary RPC whose device error is not caught by the handler: grpcio
    /// reports those as `UNKNOWN`, `Exception calling application: ...`.
    async fn unary<R, F>(&self, project_id: String, op: F) -> Result<R, Status>
    where
        R: Send + 'static,
        F: FnOnce(&dyn ModelClient) -> anyhow::Result<R> + Send + 'static,
    {
        self.blocking(project_id, op)
            .await?
            .map_err(|exc| application_error(&exc.to_string()))
    }
}

/// Server-streaming response type for `Stream`.
pub type StreamChunkStream = Pin<Box<dyn Stream<Item = Result<pb::StreamChunk, Status>> + Send>>;

#[tonic::async_trait]
impl ModelHostService for ModelHostServicer {
    // -- generation ---------------------------------------------------------

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<pb::GenerateResponse>, Status> {
        let request = request.into_inner();
        let project_id = request.project_id.clone();
        let result = self
            .blocking(project_id, move |client| {
                client.generate(
                    &request.prompt,
                    non_empty(&request.model_name),
                    non_empty(&request.capability),
                    request.max_tokens.map(|n| n.max(0) as usize),
                    request.thinking,
                )
            })
            .await?;
        match result {
            Ok(text) => Ok(Response::new(pb::GenerateResponse { text })),
            Err(exc) => Err(abort("generate", &exc)),
        }
    }

    type StreamStream = StreamChunkStream;

    /// gRPC cancellation is the cancellation mechanism.  The client aborts
    /// this RPC to stop a generation early; between chunks we notice and raise
    /// the same event the served model checks already, so the device stops
    /// being asked for tokens it would compute for nobody.
    async fn stream(
        &self,
        request: Request<pb::StreamRequest>,
    ) -> Result<Response<Self::StreamStream>, Status> {
        let request = request.into_inner();
        let client = self.client.clone();
        let cancelled = Arc::new(AtomicBool::new(false));
        // One chunk in flight at a time, as a gRPC generator handler pulls the
        // next chunk only once the last one was taken.
        let (sender, receiver) = tokio::sync::mpsc::channel::<Result<pb::StreamChunk, Status>>(1);

        tokio::task::spawn_blocking(move || {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                bind_project(client.as_ref(), &request.project_id, |client| {
                    serve_stream(client, &request, &cancelled, &sender)
                })
            }));
            if let Err(panic) = outcome {
                let _ = sender.blocking_send(Err(application_error(&panic_message(&*panic))));
            }
        });

        let chunks = async_stream::stream! {
            let mut receiver = receiver;
            while let Some(item) = receiver.recv().await {
                yield item;
            }
        };
        Ok(Response::new(Box::pin(chunks)))
    }

    // -- embedding ----------------------------------------------------------

    async fn encode_texts(
        &self,
        request: Request<pb::EncodeTextsRequest>,
    ) -> Result<Response<pb::EncodeTextsResponse>, Status> {
        let request = request.into_inner();
        let project_id = request.project_id.clone();
        let result = self
            .blocking(project_id, move |client| {
                client.encode_texts(&request.texts, non_empty(&request.base_dir))
            })
            .await?;
        match result {
            Ok(None) => Ok(Response::new(pb::EncodeTextsResponse::default())),
            Ok(Some(array)) => Ok(Response::new(encode_array(&array))),
            Err(exc) => Err(abort("encode_texts", &exc)),
        }
    }

    // -- questions about what is resident -----------------------------------

    async fn is_loaded(
        &self,
        request: Request<pb::ProjectRequest>,
    ) -> Result<Response<pb::BoolReply>, Status> {
        let request = request.into_inner();
        let value = self.unary(request.project_id, |client| client.is_loaded()).await?;
        Ok(Response::new(pb::BoolReply { value }))
    }

    async fn supports_thinking(
        &self,
        request: Request<pb::SupportsThinkingRequest>,
    ) -> Result<Response<pb::BoolReply>, Status> {
        let request = request.into_inner();
        let capability = request.capability;
        let value = self
            .unary(request.project_id, move |client| {
                client.supports_thinking(non_empty(&capability))
            })
            .await?;
        Ok(Response::new(pb::BoolReply { value }))
    }

    async fn model_name(
        &self,
        request: Request<pb::ProjectRequest>,
    ) -> Result<Response<pb::StringReply>, Status> {
        let request = request.into_inner();
        let value = self.unary(request.project_id, |client| client.model_name()).await?;
        Ok(Response::new(pb::StringReply { value: value.unwrap_or_default() }))
    }

    async fn model_label(
        &self,
        request: Request<pb::ModelLabelRequest>,
    ) -> Result<Response<pb::StringReply>, Status> {
        let request = request.into_inner();
        let capability = request.capability;
        let value = self
            .unary(request.project_id, move |client| client.model_label(non_empty(&capability)))
            .await?;
        Ok(Response::new(pb::StringReply { value }))
    }

    async fn count_tokens(
        &self,
        request: Request<pb::CountTokensRequest>,
    ) -> Result<Response<pb::IntReply>, Status> {
        let request = request.into_inner();
        let text = request.text;
        let value = self
            .unary(request.project_id, move |client| client.count_tokens(&text))
            .await?;
        Ok(Response::new(pb::IntReply { value: to_i32(value) }))
    }

    async fn max_context_tokens(
        &self,
        request: Request<pb::ProjectRequest>,
    ) -> Result<Response<pb::IntReply>, Status> {
        let request = request.into_inner();
        let value = self
            .unary(request.project_id, |client| client.max_context_tokens())
            .await?;
        Ok(Response::new(pb::IntReply { value: to_i32(value) }))
    }

    async fn response_max_context_tokens(
        &self,
        request: Request<pb::ResponseMaxContextRequest>,
    ) -> Result<Response<pb::IntReply>, Status> {
        let request = request.into_inner();
        let capability = request.capability;
        let value = self
            .unary(request.project_id, move |client| {
                client.response_max_context_tokens(non_empty(&capability))
            })
            .await?;
        Ok(Response::new(pb::IntReply { value: to_i32(value) }))
    }

    async fn has_capability_model(
        &self,
        request: Request<pb::HasCapabilityRequest>,
    ) -> Result<Response<pb::BoolReply>, Status> {
        let request = request.into_inner();
        let capability = request.capability;
        let value = self
            .unary(request.project_id, move |client| client.has_capability_model(&capability))
            .await?;
        Ok(Response::new(pb::BoolReply { value }))
    }

    async fn idle_seconds(
        &self,
        request: Request<pb::ProjectRequest>,
    ) -> Result<Response<pb::FloatReply>, Status> {
        let request = request.into_inner();
        let value = self.unary(request.project_id, |client| client.idle_seconds()).await?;
        Ok(Response::new(pb::FloatReply { value }))
    }

    // -- lifecycle ----------------------------------------------------------

    async fn load_model(
        &self,
        request: Request<pb::ProjectRequest>,
    ) -> Result<Response<pb::StringReply>, Status> {
        let request = request.into_inner();
        let value = self.unary(request.project_id, |client| client.load_model()).await?;
        Ok(Response::new(pb::StringReply { value: value.unwrap_or_default() }))
    }

    async fn unload_auxiliary_models(
        &self,
        request: Request<pb::UnloadAuxiliaryRequest>,
    ) -> Result<Response<pb::Empty>, Status> {
        let request = request.into_inner();
        let include_embedding = request.include_embedding;
        self.unary(request.project_id, move |client| {
            client.unload_auxiliary_models(include_embedding)
        })
        .await?;
        Ok(Response::new(pb::Empty {}))
    }

    async fn unload_embedding_model(
        &self,
        request: Request<pb::ProjectRequest>,
    ) -> Result<Response<pb::Empty>, Status> {
        let request = request.into_inner();
        self.unary(request.project_id, |client| client.unload_embedding_model())
            .await?;
        Ok(Response::new(pb::Empty {}))
    }
}

/// Drive one streamed generation into `sender`, on a blocking worker with the
/// project already bound.
fn serve_stream(
    client: &dyn ModelClient,
    request: &pb::StreamRequest,
    cancelled: &Arc<AtomicBool>,
    sender: &tokio::sync::mpsc::Sender<Result<pb::StreamChunk, Status>>,
) {
    let chunks = client.stream(
        &request.prompt,
        non_empty(&request.model_name),
        non_empty(&request.capability),
        Some(cancelled.clone()),
        request.thinking,
    );
    let chunks = match chunks {
        Ok(chunks) => chunks,
        Err(exc) => {
            let _ = sender.blocking_send(Err(abort("stream", &exc)));
            return;
        }
    };
    for item in chunks {
        match item {
            Ok(text) => {
                if sender.is_closed() {
                    // The client hung up mid-answer; nothing left to stream to.
                    cancelled.store(true, Ordering::SeqCst);
                    return;
                }
                if sender.blocking_send(Ok(pb::StreamChunk { text })).is_err() {
                    cancelled.store(true, Ordering::SeqCst);
                    return;
                }
            }
            Err(exc) => {
                let _ = sender.blocking_send(Err(abort("stream", &exc)));
                return;
            }
        }
    }
}

/// An optional proto3 string: empty means "not given".
fn non_empty(value: &str) -> Option<&str> {
    Some(value).filter(|v| !v.is_empty())
}

/// Saturating conversion into the wire's `int32`.
fn to_i32(value: usize) -> i32 {
    i32::try_from(value).unwrap_or(i32::MAX)
}

/// A float32 matrix as its shape plus its row-major (C-contiguous) bytes,
/// whatever its in-memory layout.
fn encode_array(array: &Array2<f32>) -> pb::EncodeTextsResponse {
    let shape = array.shape().iter().map(|d| i32::try_from(*d).unwrap_or(i32::MAX)).collect();
    // `iter()` walks in logical (row-major) order, so a strided view comes out
    // contiguous, as `np.ascontiguousarray` made it.
    let mut data = Vec::with_capacity(array.len() * 4);
    for value in array.iter() {
        data.extend_from_slice(&value.to_ne_bytes());
    }
    pb::EncodeTextsResponse { shape, data }
}

/// The text of a panic payload.
fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = panic.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "panic".to_string()
    }
}

/// What grpcio answers for a handler that raised: `UNKNOWN`,
/// `Exception calling application: <message>`.
fn application_error(message: &str) -> Status {
    Status::unknown(format!("Exception calling application: {message}"))
}

/// The Python exception class an error corresponds to, for the
/// `"<Type>: <message>"` detail the host reports.
fn error_type_name(exc: &anyhow::Error) -> &'static str {
    if exc.downcast_ref::<ModelHostError>().is_some() {
        "ModelHostError"
    } else if exc.downcast_ref::<std::io::Error>().is_some() {
        "OSError"
    } else if exc.downcast_ref::<serde_json::Error>().is_some() {
        "ValueError"
    } else {
        "RuntimeError"
    }
}

/// Fail the RPC with the device error (`INTERNAL`, `"<Type>: <message>"`),
/// without killing the server.
fn abort(op: &str, exc: &anyhow::Error) -> Status {
    tracing::warn!(target: LOG_TARGET, "[model-host] {} failed: {}", op, exc);
    Status::internal(format!("{}: {}", error_type_name(exc), exc))
}

/// Remove a socket file no one is listening on.
///
/// A host killed with SIGKILL leaves its file behind; binding over it fails
/// with EADDRINUSE even though the address is free.  Probing first means a
/// *live* host is never unlinked out from under its clients — the restart
/// fails loudly instead, which is the correct outcome.
pub fn unlink_stale_socket(path: &str) -> std::io::Result<()> {
    if !has_unix_sockets() || !std::path::Path::new(path).exists() {
        return Ok(());
    }
    if probe_unix_socket(path) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AddrInUse,
            format!("a model host is already listening on {path}"),
        ));
    }
    if std::fs::remove_file(path).is_ok() {
        tracing::info!(target: LOG_TARGET, "[model-host] removed a stale socket at {}", path);
    }
    Ok(())
}

/// Whether something accepts connections on the unix socket at `path`.
#[cfg(unix)]
fn probe_unix_socket(path: &str) -> bool {
    match std::os::unix::net::UnixStream::connect(path) {
        Ok(probe) => {
            let _ = probe.set_read_timeout(Some(Duration::from_millis(500)));
            drop(probe);
            true
        }
        Err(_) => false,
    }
}

#[cfg(not(unix))]
fn probe_unix_socket(_path: &str) -> bool {
    false
}

/// Run a model host in this process, blocking until it is stopped.
///
/// Used by `slp -m smartloop.model_host`; the supervisor spawns exactly one of
/// these and points every project agent at it.
pub fn serve(
    address: Option<&str>,
    model_service: Option<Arc<dyn LocalModelService>>,
) -> anyhow::Result<ModelHost> {
    let model_service: Arc<dyn LocalModelService> = match model_service {
        Some(service) => service,
        None => crate::api::services::get_model_service(),
    };
    let address = address
        .map(str::to_string)
        .filter(|a| !a.is_empty())
        .or_else(|| std::env::var("SLP_MODEL_HOST").ok().filter(|a| !a.is_empty()))
        .unwrap_or_else(|| default_socket_path(&crate::config::AppSettings::new().home_dir()));

    let host = ModelHost::new(Arc::new(InProcessModelClient::new(model_service)), address);
    host.bind()?;

    // Ctrl-C ends the host the way KeyboardInterrupt did: stop, which unlinks
    // the socket, rather than dying with the file left behind.
    let interrupted = host.termination.clone();
    std::thread::Builder::new()
        .name("slp-model-host-signals".into())
        .spawn(move || {
            if let Ok(runtime) =
                tokio::runtime::Builder::new_current_thread().enable_all().build()
            {
                runtime.block_on(async {
                    if tokio::signal::ctrl_c().await.is_ok() {
                        interrupted.signal();
                    }
                });
            }
        })?;

    let served = host.serve_forever();
    host.stop();
    served?;
    Ok(host)
}
