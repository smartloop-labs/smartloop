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
"""

from __future__ import annotations

import logging
import os
import select
import socket
import threading
from typing import Callable, Optional

from smartloop.model_host import protocol
from smartloop.model_host.client import (
    HAS_UNIX_SOCKETS, InProcessModelClient, ModelClient, mark_device_owner,
    parse_address,
)
from smartloop.model_host.protocol import (
    CANCEL, CHUNK, RESULT, ConnectionClosed, ProtocolError,
)
from smartloop.workspace.context import use_project

logger = logging.getLogger(__name__)

#: How long a connection waits for the next request before checking whether the
#: host is shutting down.  Only bounds shutdown latency — an idle client is not
#: disconnected.
_ACCEPT_POLL_SECONDS = 0.5


class ModelHost:
    """Serves one :class:`ModelClient` to any number of project agents.

    Each connection gets a thread, and each request on a connection is handled
    in order.  Concurrency across projects is therefore bounded by the
    inference semaphore inside the served client — which is the point: the
    device has one budget, and it is enforced in one place.
    """

    def __init__(self, client: ModelClient, address: str) -> None:
        # This process holds the weights, so anything in it that would
        # otherwise reach for a model host resolves locally instead.
        mark_device_owner()
        self.client = client
        self.address = address
        self._server: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._connections: set[socket.socket] = set()
        self._connections_lock = threading.Lock()
        #: Cancellation events for generations currently in flight, keyed by
        #: ``(connection id, request id)``.
        self._inflight: dict[tuple[int, int], threading.Event] = {}
        self._inflight_lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------

    def bind(self) -> str:
        """Create the listening socket. Returns the address actually bound."""
        path, port = parse_address(self.address)
        if port is None and not HAS_UNIX_SOCKETS:
            raise OSError(
                f"cannot listen on {path}: this platform has no AF_UNIX. "
                "Give the host a host:port address instead."
            )
        if port is None:
            # A socket file left by a crashed host would make bind() fail with
            # EADDRINUSE even though nothing is listening.
            _unlink_stale_socket(path)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(path)
            # The socket is a control channel for the user's own agents; keep
            # it off-limits to everyone else on the machine.
            os.chmod(path, 0o600)
            bound = path
        else:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((path, port))
            bound = f"{path}:{server.getsockname()[1]}"
        server.listen(64)
        server.settimeout(_ACCEPT_POLL_SECONDS)
        self._server = server
        self.address = bound
        logger.info("[model-host] listening on %s", bound)
        return bound

    def serve_forever(self) -> None:
        """Accept connections until :meth:`stop` is called."""
        if self._server is None:
            self.bind()
        while not self._stopping.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stopping.is_set():
                    break
                raise
            with self._connections_lock:
                self._connections.add(conn)
            threading.Thread(
                target=self._serve_connection,
                args=(conn,),
                name="model-host-conn",
                daemon=True,
            ).start()

    def start(self) -> str:
        """Bind and serve on a background thread. Returns the bound address."""
        bound = self.bind()
        self._accept_thread = threading.Thread(
            target=self.serve_forever, name="model-host", daemon=True,
        )
        self._accept_thread.start()
        return bound

    def stop(self) -> None:
        """Stop accepting, drop live connections, and remove the socket file."""
        self._stopping.set()
        server, self._server = self._server, None
        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        with self._connections_lock:
            connections, self._connections = list(self._connections), set()
        for conn in connections:
            try:
                conn.close()
            except OSError:
                pass
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=5)
            self._accept_thread = None
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

    # -- connection handling ------------------------------------------------

    def _serve_connection(self, conn: socket.socket) -> None:
        conn_id = id(conn)
        try:
            while not self._stopping.is_set():
                try:
                    frame = protocol.recv_frame(conn)
                except (ConnectionClosed, OSError):
                    break
                except ProtocolError as exc:
                    logger.warning("[model-host] dropping connection: %s", exc)
                    break
                self._handle_frame(conn, conn_id, frame)
        finally:
            with self._connections_lock:
                self._connections.discard(conn)
            try:
                conn.close()
            except OSError:
                pass

    def _handle_frame(self, conn: socket.socket, conn_id: int, frame: dict) -> None:
        request_id = frame.get("id")
        if frame.get("type") == CANCEL:
            self._apply_cancel(conn_id, request_id)
            return

        op = frame.get("op")
        args = frame.get("args") or {}
        try:
            # Bind the requesting project for the whole operation: which
            # model answers depends on that project's settings, and the host's
            # own "current" project is the UI's selection, which is somebody
            # else's answer.
            with use_project(args.get("project_id")):
                if op == "stream":
                    self._handle_stream(conn, conn_id, request_id, args)
                else:
                    value = self._dispatch(op, args)
                    protocol.send_frame(
                        conn, {"id": request_id, "type": RESULT, "value": value},
                    )
        except Exception as exc:  # a failed op must not kill the connection
            logger.warning("[model-host] %s failed: %s", op, exc)
            try:
                protocol.send_frame(conn, protocol.error_payload(request_id, exc))
            except OSError:
                pass

    def _apply_cancel(self, conn_id: int, request_id) -> None:
        """Signal the generation identified by *request_id*, if it is running."""
        with self._inflight_lock:
            event = self._inflight.get((conn_id, request_id))
        if event is not None:
            event.set()

    def _poll_for_cancel(self, conn: socket.socket, conn_id: int) -> None:
        """Pick up a CANCEL that arrived while a generation is streaming.

        The connection's read loop is *inside* this generation, so a cancel
        would otherwise sit unread in the socket buffer until the answer
        finished — which is exactly when it stops being useful.  Checked
        between chunks, which is the same granularity in-process cancellation
        has: a native llama.cpp call cannot be interrupted mid-token either.
        """
        while select.select([conn], [], [], 0)[0]:
            try:
                frame = protocol.recv_frame(conn)
            except (ConnectionClosed, ProtocolError, OSError):
                # The client went away mid-answer.  Stop every generation on
                # this connection: there is nobody left to read them, and the
                # tokens would be computed for no one.
                with self._inflight_lock:
                    for event in self._inflight.values():
                        event.set()
                return
            if frame.get("type") == CANCEL:
                self._apply_cancel(conn_id, frame.get("id"))

    def _handle_stream(self, conn: socket.socket, conn_id: int,
                       request_id, args: dict) -> None:
        cancelled = threading.Event()
        key = (conn_id, request_id)
        with self._inflight_lock:
            self._inflight[key] = cancelled
        try:
            for text in self.client.stream(
                args.get("prompt") or "",
                args.get("model_name"),
                capability=args.get("capability"),
                cancelled=cancelled,
                thinking=args.get("thinking"),
            ):
                self._poll_for_cancel(conn, conn_id)
                if cancelled.is_set():
                    break
                protocol.send_frame(
                    conn, {"id": request_id, "type": CHUNK, "text": text},
                )
            # The terminal frame is sent even for a cancelled generation, so
            # the client knows the exchange is over and the connection is free.
            protocol.send_frame(conn, {"id": request_id, "type": RESULT, "value": None})
        finally:
            with self._inflight_lock:
                self._inflight.pop(key, None)

    def _dispatch(self, op: Optional[str], args: dict):
        """Run a non-streaming operation and return its JSON-ready value."""
        handler: Optional[Callable] = _OPERATIONS.get(op or "")
        if handler is None:
            raise ProtocolError(f"unknown operation {op!r}")
        return handler(self.client, args)


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


#: Non-streaming operations, mapped to how they call the served client.
#: Streaming is handled separately because it answers with many frames.
_OPERATIONS: dict[str, Callable[[ModelClient, dict], object]] = {
    "generate": lambda c, a: c.generate(
        a.get("prompt") or "", a.get("model_name"), capability=a.get("capability"),
        max_tokens=a.get("max_tokens"), thinking=a.get("thinking"),
    ),
    "supports_thinking": lambda c, a: c.supports_thinking(a.get("capability")),
    "encode_texts": lambda c, a: protocol.encode_array(
        c.encode_texts(a.get("texts") or [], base_dir=a.get("base_dir")),
    ),
    "is_loaded": lambda c, a: c.is_loaded(),
    "model_name": lambda c, a: c.model_name(),
    "model_label": lambda c, a: c.model_label(a.get("capability")),
    "count_tokens": lambda c, a: c.count_tokens(a.get("text") or ""),
    "max_context_tokens": lambda c, a: c.max_context_tokens(),
    "response_max_context_tokens": lambda c, a: c.response_max_context_tokens(
        a.get("capability"),
    ),
    "has_capability_model": lambda c, a: c.has_capability_model(
        a.get("capability") or "",
    ),
    "idle_seconds": lambda c, a: c.idle_seconds(),
    "load_model": lambda c, a: c.load_model(),
    "unload_auxiliary_models": lambda c, a: c.unload_auxiliary_models(
        include_embedding=a.get("include_embedding", True),
    ),
    "unload_embedding_model": lambda c, a: c.unload_embedding_model(),
}


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
