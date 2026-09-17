"""The process that owns the device, serving generation to project agents.

One host for the whole workspace, never one per project.  Per-project copies of
the generative model were tried and reverted: weights loaded through mmap share
page cache between processes on CPU, but VRAM copies do not, so N projects
would mean N resident copies competing for the same device, each with its own
KV cache, serialising in the driver.  Keeping one host also keeps
``n_parallel`` continuous batching meaningful — a single budget, shared — and
loads the ~417 MB BGE-M3 embedder once instead of once per child.

The host is deliberately thin: it holds the models and answers the operations
in :class:`~smartloop.model_host.client.ModelClient`.  Everything project-shaped
stays in the agent that asked.

Transport: gRPC (see :mod:`smartloop.model_host.model_host_pb2_grpc`), on a
unix socket where one exists and loopback TCP where it does not (Windows).
The old length-prefixed JSON framing over raw sockets was the same wire on both,
but it hand-rolled what gRPC does out of the box — framing, cancellation,
reconnection — and the hand-rolled part is exactly where a cross-platform
transport has to be trusted most.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from concurrent import futures
from contextlib import contextmanager
from typing import Callable, Optional

import grpc
import numpy as np

from smartloop.model_host import model_host_pb2 as pb2
from smartloop.model_host import model_host_pb2_grpc as pb2_grpc
from smartloop.model_host.client import (
    HAS_UNIX_SOCKETS, InProcessModelClient, ModelClient, mark_device_owner,
    parse_address,
)
from smartloop.workspace.context import use_project

logger = logging.getLogger(__name__)

#: grpcio's default cap on a message is 4 MiB; an embedding batch can exceed
#: that.  This matches what the old frame ceiling bounded instead.
MAX_MESSAGE_BYTES = 64 * 1024 * 1024

#: gRPC worker threads.  Inference serialises on the served client's own
#: semaphore (``n_parallel``), so the pool is sized for streams parked on a
#: generation, not for concurrent device work.
_WORKERS = 16


class ModelHost:
    """Serves one :class:`ModelClient` to any number of project agents.

    gRPC gives each RPC a worker thread; the served client's inference
    semaphore is what bounds concurrency across them, which is the point: the
    device has one budget, and it is enforced in one place.
    """

    def __init__(self, client: ModelClient, address: str) -> None:
        # This process holds the weights, so anything in it that would
        # otherwise reach for a model host resolves locally instead.
        mark_device_owner()
        self.client = client
        self.address = address
        self._server: Optional[grpc.Server] = None

    # -- lifecycle ----------------------------------------------------------

    def bind(self) -> str:
        """Create the listening socket. Returns the address actually bound."""
        path, port = parse_address(self.address)
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=_WORKERS),
            options=[
                ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
                ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
            ],
        )
        pb2_grpc.add_ModelHostServicer_to_server(_ModelHostServicer(self.client), server)

        if port is None:
            if not HAS_UNIX_SOCKETS:
                raise OSError(
                    f"cannot listen on {path}: this platform has no AF_UNIX. "
                    "Give the host a host:port address instead."
                )
            # A socket file left by a crashed host would otherwise make gRPC
            # fail to bind even though nothing is listening.
            _unlink_stale_socket(path)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            server.add_insecure_port(f"unix:{path}")
            # gRPC creates the socket file when the server starts; it is
            # locked down then.
            bound = path
        else:
            got = server.add_insecure_port(f"{path}:{port}")
            if got == 0:
                raise OSError(f"cannot listen on {path}:{port}")
            bound = f"{path}:{got}"

        self._server = server
        self.address = bound
        logger.info("[model-host] listening on %s", bound)
        return bound

    def start(self) -> str:
        """Bind and start serving. Returns the bound address."""
        bound = self.bind()
        if self._server is None:  # pragma: no cover - defensive
            raise RuntimeError("bind() did not create a server")
        self._server.start()
        self._chmod_socket_if_any()
        return bound

    def _chmod_socket_if_any(self) -> None:
        """The control channel is for the user's own agents; keep it private."""
        path, port = parse_address(self.address)
        if port is None:
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass

    def serve_forever(self) -> None:
        """Serve until :meth:`stop` is called."""
        if self._server is None:
            self.bind()
            self._server.start()  # type: ignore[union-attr]
            self._chmod_socket_if_any()
        self._server.wait_for_termination()  # type: ignore[union-attr]

    def stop(self) -> None:
        """Stop accepting and drop live RPCs."""
        server, self._server = self._server, None
        if server is not None:
            # Grace of zero cancels in-flight RPCs at once, matching the old
            # behaviour of closing every live socket on shutdown.
            server.stop(grace=0)
        path, port = parse_address(self.address)
        if port is None:
            try:
                os.unlink(path)
            except OSError:
                pass
        logger.info("[model-host] stopped")

    def __enter__(self) -> "ModelHost":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()


class _ModelHostServicer(pb2_grpc.ModelHostServicer):
    """Answers each RPC against the requesting project's model selection.

    Model selection depends on per-project settings, and those are written by
    the child pinned to the project while the host runs in the supervisor,
    holding a copy taken when it first saw the project.  Nothing invalidates
    that copy, so the host refreshes its client before every call and then
    binds the requesting project for the whole operation — otherwise a model
    disabled in a project kept answering for it, and the picker said one thing
    while the host resolved another.
    """

    def __init__(self, client: ModelClient) -> None:
        self.client = client

    @contextmanager
    def _bind(self, project_id: str):
        pid = project_id or None
        try:
            self.client.refresh_project(pid)
        except Exception as exc:
            # A project whose own file cannot be read is a reason to answer
            # from what we have, not to fail the generation.
            logger.warning(
                "[model-host] could not refresh project %s: %s", project_id, exc,
            )
        with use_project(pid):
            yield

    # -- generation ---------------------------------------------------------

    def Generate(self, request, context):
        with self._bind(request.project_id):
            try:
                text = self.client.generate(
                    request.prompt or "",
                    request.model_name or None,
                    capability=request.capability or None,
                    max_tokens=request.max_tokens
                    if request.HasField("max_tokens") else None,
                    thinking=request.thinking
                    if request.HasField("thinking") else None,
                )
            except Exception as exc:
                _abort(context, "generate", exc)
            return pb2.GenerateResponse(text=text)

    def Stream(self, request, context):
        # gRPC cancellation is the cancellation mechanism.  The client aborts
        # this RPC to stop a generation early; between chunks we notice and
        # raise the same event the served model checks already, so the device
        # stops being asked for tokens it would compute for nobody.
        cancelled = threading.Event()
        with self._bind(request.project_id):
            try:
                for text in self.client.stream(
                    request.prompt or "",
                    request.model_name or None,
                    capability=request.capability or None,
                    cancelled=cancelled,
                    thinking=request.thinking
                    if request.HasField("thinking") else None,
                ):
                    if not context.is_active():
                        cancelled.set()
                        return
                    yield pb2.StreamChunk(text=text)
            except grpc.RpcError:
                # The client hung up mid-answer; nothing left to stream to.
                cancelled.set()
            except Exception as exc:
                _abort(context, "stream", exc)

    # -- embedding ----------------------------------------------------------

    def EncodeTexts(self, request, context):
        try:
            with self._bind(request.project_id):
                array = self.client.encode_texts(
                    list(request.texts), base_dir=request.base_dir or None,
                )
        except Exception as exc:
            _abort(context, "encode_texts", exc)
        if array is None:
            return pb2.EncodeTextsResponse()
        array = np.ascontiguousarray(array, dtype=np.float32)
        return pb2.EncodeTextsResponse(shape=list(array.shape), data=array.tobytes())

    # -- questions about what is resident -----------------------------------

    def IsLoaded(self, request, context):
        with self._bind(request.project_id):
            return pb2.BoolReply(value=self.client.is_loaded())

    def SupportsThinking(self, request, context):
        with self._bind(request.project_id):
            return pb2.BoolReply(
                value=self.client.supports_thinking(request.capability or None),
            )

    def ModelName(self, request, context):
        with self._bind(request.project_id):
            return pb2.StringReply(value=self.client.model_name() or "")

    def ModelLabel(self, request, context):
        with self._bind(request.project_id):
            return pb2.StringReply(
                value=self.client.model_label(request.capability or None),
            )

    def CountTokens(self, request, context):
        with self._bind(request.project_id):
            return pb2.IntReply(value=self.client.count_tokens(request.text or ""))

    def MaxContextTokens(self, request, context):
        with self._bind(request.project_id):
            return pb2.IntReply(value=self.client.max_context_tokens())

    def ResponseMaxContextTokens(self, request, context):
        with self._bind(request.project_id):
            return pb2.IntReply(
                value=self.client.response_max_context_tokens(
                    request.capability or None,
                ),
            )

    def HasCapabilityModel(self, request, context):
        with self._bind(request.project_id):
            return pb2.BoolReply(
                value=self.client.has_capability_model(request.capability or ""),
            )

    def IdleSeconds(self, request, context):
        with self._bind(request.project_id):
            return pb2.FloatReply(value=float(self.client.idle_seconds()))

    # -- lifecycle ----------------------------------------------------------

    def LoadModel(self, request, context):
        with self._bind(request.project_id):
            return pb2.StringReply(value=self.client.load_model() or "")

    def UnloadAuxiliaryModels(self, request, context):
        with self._bind(request.project_id):
            self.client.unload_auxiliary_models(
                include_embedding=request.include_embedding,
            )
        return pb2.Empty()

    def UnloadEmbeddingModel(self, request, context):
        with self._bind(request.project_id):
            self.client.unload_embedding_model()
        return pb2.Empty()


def _abort(context, op: str, exc: Exception):
    """Fail the RPC with the device error, without killing the server."""
    logger.warning("[model-host] %s failed: %s", op, exc)
    context.abort(grpc.StatusCode.INTERNAL, f"{type(exc).__name__}: {exc}")


def _unlink_stale_socket(path: str) -> None:
    """Remove a socket file no one is listening on.

    A host killed with SIGKILL leaves its file behind; binding over it fails
    with EADDRINUSE even though the address is free.  Probing first means a
    *live* host is never unlinked out from under its clients — the restart
    fails loudly instead, which is the correct outcome.
    """
    if not HAS_UNIX_SOCKETS or not os.path.exists(path):
        return
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.5)
    try:
        probe.connect(path)
    except OSError:
        try:
            os.unlink(path)
            logger.info("[model-host] removed a stale socket at %s", path)
        except OSError:
            pass
    else:
        raise OSError(f"a model host is already listening on {path}")
    finally:
        probe.close()


def serve(address: Optional[str] = None, model_service=None) -> ModelHost:
    """Run a model host in this process, blocking until it is stopped.

    Used by ``python -m smartloop.model_host``; the supervisor spawns exactly
    one of these and points every project agent at it.
    """
    from smartloop.config import AppSettings
    from smartloop.model_host.client import default_socket_path

    if model_service is None:
        from smartloop.api.services import get_model_service

        model_service = get_model_service()
    address = address or os.environ.get("SLP_MODEL_HOST") or \
        default_socket_path(AppSettings().home_dir)

    host = ModelHost(InProcessModelClient(model_service), address)
    host.bind()
    try:
        host.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        host.stop()
    return host