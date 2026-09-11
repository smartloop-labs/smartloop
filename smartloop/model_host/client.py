"""The interface a project agent uses to reach the generative model.

Two implementations, deliberately in that order:

* :class:`InProcessModelClient` — calls the :class:`ModelService` living in
  this process.  The default, and byte-for-byte the path that ran before the
  model host existed: one extra method call, no transport, no behaviour change.
* :class:`RemoteModelClient` — the same operations over a unix socket to a host
  process that owns the device.

Defining the interface first and keeping the in-process path behind it is what
makes the transport safe to build: nothing regresses while it is written, and
the default only moves once the two have been measured against each other.

Select with ``SLP_MODEL_HOST`` — a socket path (or ``host:port``); unset means
in-process.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import os
import socket
import tempfile
import threading
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

import numpy as np

from smartloop.model_host import protocol
from smartloop.workspace.context import pinned_project_id
from smartloop.model_host.protocol import (
    CANCEL, CHUNK, ERROR, REQUEST, RESULT, ConnectionClosed, ProtocolError,
)

logger = logging.getLogger(__name__)

#: Set in the process that actually holds the weights.  Without it a host
#: started with ``SLP_MODEL_HOST`` pointing at its own socket would resolve a
#: remote client and try to answer its own requests.
_device_owner = False


#: Whether this interpreter can speak AF_UNIX at all.  Feature-detected rather
#: than guessed from the platform, which is what CPython's own ``socketpair``
#: does — Windows builds simply do not define it.
HAS_UNIX_SOCKETS = hasattr(socket, "AF_UNIX")


def mark_device_owner() -> None:
    """Declare that this process owns the models, not a host elsewhere."""
    global _device_owner
    _device_owner = True


def is_device_owner() -> bool:
    """Whether the models are resident in this process."""
    return _device_owner


class ModelClient(ABC):
    """Everything a project agent needs from whoever owns the device.

    Deliberately narrow: generation, embedding, and the questions whose answers
    depend on which model is resident.  Project state — documents, skills, MCP
    servers, sessions — is on the other side of the seam and never crosses it.
    """

    # -- generation ---------------------------------------------------------

    @abstractmethod
    def generate(self, prompt: str, model_name: Optional[str] = None,
                 capability: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 thinking: Optional[bool] = None) -> str:
        """Complete *prompt* and return the whole response.

        ``max_tokens`` caps the output; ``thinking`` forces the reasoning block
        on or off for this call whatever the user's setting says.  Both belong
        to the request rather than to the loaded model: a decision call caps
        itself so a missing stop token cannot ramble, and a call whose output
        is a tool call rather than an answer reasons even when the visible
        answer does not (see ``MCPAgent``)."""

    @abstractmethod
    def stream(self, prompt: str, model_name: Optional[str] = None,
               capability: Optional[str] = None,
               cancelled: Optional[threading.Event] = None,
               thinking: Optional[bool] = None) -> Iterator[str]:
        """Complete *prompt*, yielding text as it is produced.

        *cancelled* is cooperative and observed between tokens — a native
        llama.cpp call cannot be interrupted mid-token, in-process or over a
        socket.
        """

    # -- embedding ----------------------------------------------------------

    @abstractmethod
    def encode_texts(self, texts: List[str],
                     base_dir: Optional[str] = None) -> Optional[np.ndarray]:
        """Embed *texts*, or return ``None`` when the embedder is unavailable.

        Held by the host so BGE-M3 is loaded once for every project rather than
        ~417 MB per child.
        """

    # -- questions about what is resident -----------------------------------

    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether a generative model is resident."""

    def supports_thinking(self, capability: Optional[str] = None) -> bool:
        """Whether the model serving *capability* can reason.

        Not abstract: a client that cannot answer says no, and a caller that
        wanted reasoning simply asks for it and gets whatever the model does.
        """
        return False

    @abstractmethod
    def model_name(self) -> Optional[str]:
        """Name of the resident orchestrator model, if any."""

    @abstractmethod
    def model_label(self, capability: Optional[str] = None) -> str:
        """How the model that will answer for *capability* is named to the user.

        The progress SSE names the model that is answering, and this has to be
        the model the host's own selection lands on — not an approximation of
        it — or a remote caller shows a different name than the single-process
        server would have.  A child cannot work it out for itself: it has no
        GGUF to read metadata from and cannot see which models are resident.
        """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Token count under the resident model's tokenizer."""

    @abstractmethod
    def max_context_tokens(self) -> int:
        """Context window of the resident model."""

    @abstractmethod
    def response_max_context_tokens(self, capability: Optional[str] = None) -> int:
        """Context window of the model that would answer for *capability*."""

    @abstractmethod
    def has_capability_model(self, capability: str) -> bool:
        """Whether a model declaring *capability* is available."""

    @abstractmethod
    def idle_seconds(self) -> float:
        """Seconds since the device last finished a generation."""

    # -- lifecycle ----------------------------------------------------------

    @abstractmethod
    def load_model(self) -> Optional[str]:
        """Make a generative model resident; returns its name."""

    @abstractmethod
    def unload_auxiliary_models(self, include_embedding: bool = True) -> None:
        """Free the on-demand large/vision models (and optionally the embedder)."""

    @abstractmethod
    def unload_embedding_model(self) -> None:
        """Free the embedding model."""

    def close(self) -> None:
        """Release any transport this client holds. In-process: nothing."""


    def refresh_project(self, project_id: Optional[str]) -> bool:
        """Pick up any write to *project_id*'s own state before serving it.

        Default no-op: a client that reaches the device over the wire is not
        the process holding the workspace, so there is nothing local to
        refresh.  Overridden where the device and the workspace are in the
        same process.
        """
        return False


class InProcessModelClient(ModelClient):
    """The device is owned by this process — call straight through.

    The default. Every method here is the call the caller used to make
    directly, so selecting this client changes nothing about how a request is
    served.
    """

    def __init__(self, model_service) -> None:
        self._service = model_service

    def refresh_project(self, project_id: Optional[str]) -> bool:
        """The workspace is right here, so a child's write is one read away."""
        return self._service.refresh_project_state(project_id)

    def generate(self, prompt, model_name=None, capability=None,
                 max_tokens=None, thinking=None) -> str:
        return self._service.generate_completion(
            prompt, model_name, capability=capability,
            max_tokens=max_tokens, thinking=thinking,
        )

    def stream(self, prompt, model_name=None, capability=None, cancelled=None,
               thinking=None):
        for chunk in self._service.stream_completion(
            prompt, model_name, cancelled=cancelled, capability=capability,
            thinking=thinking,
        ):
            # ``GenerationChunk`` in, plain text out — the wire carries text,
            # so the in-process path yields the same thing rather than making
            # every caller handle two shapes.
            yield getattr(chunk, "text", chunk)

    def encode_texts(self, texts, base_dir=None):
        from smartloop.db import embeddings

        return embeddings.encode_texts(texts, base_dir=base_dir)

    def is_loaded(self) -> bool:
        return self._service.is_loaded()

    def supports_thinking(self, capability=None) -> bool:
        return self._service.supports_thinking(capability)

    def model_name(self) -> Optional[str]:
        return self._service._model_name

    def model_label(self, capability=None) -> str:
        return self._service.response_model_label(capability)

    def count_tokens(self, text: str) -> int:
        return self._service.count_tokens(text)

    def max_context_tokens(self) -> int:
        return self._service.get_max_context_tokens()

    def response_max_context_tokens(self, capability=None) -> int:
        return self._service.get_response_max_context_tokens(capability)

    def has_capability_model(self, capability: str) -> bool:
        return self._service.has_capability_model(capability)

    def idle_seconds(self) -> float:
        return self._service.idle_seconds()

    def load_model(self) -> Optional[str]:
        self._service.load_model()
        return self._service._model_name

    def unload_auxiliary_models(self, include_embedding: bool = True) -> None:
        self._service.unload_auxiliary_models(include_embedding=include_embedding)

    def unload_embedding_model(self) -> None:
        self._service.unload_embedding_model()


class RemoteModelClient(ModelClient):
    """The device is owned by a model host — talk to it over a socket.

    One connection, serialized by a lock: the protocol interleaves nothing, and
    a project agent's concurrency is already bounded by the host's own
    ``n_parallel`` budget, so a connection pool would only move the queue
    without widening it.

    The connection is opened lazily and re-opened on failure, so a host restart
    costs the in-flight request rather than the client.
    """

    def __init__(self, address: str, timeout: float = 600.0) -> None:
        self.address = address
        self._timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._lock = threading.RLock()
        self._ids = itertools.count(1)

    # -- transport ----------------------------------------------------------

    def _connect(self) -> socket.socket:
        if self._sock is not None:
            return self._sock
        host, port = parse_address(self.address)
        if port is None:
            if not HAS_UNIX_SOCKETS:
                raise ModelHostError(
                    f"{self.address} is a unix socket path, which this platform "
                    "cannot open; the host should be listening on host:port"
                )
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self._timeout)
            sock.connect(host)
        else:
            sock = socket.create_connection((host, port), timeout=self._timeout)
        self._sock = sock
        logger.info("[model-host] connected to %s", self.address)
        return sock

    def _drop(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def close(self) -> None:
        with self._lock:
            self._drop()

    def _call(self, op: str, **args):
        """Send *op* and return its single result.

        The exchange is closed explicitly rather than left to the collector.
        It holds the connection lock for its whole body, and abandoning the
        generator would leave releasing that lock to whenever the object
        happens to be finalised — which works under CPython's refcounting and
        would deadlock the next thread anywhere it does not.
        """
        exchange = self._exchange(op, args, streaming=False)
        try:
            for value in exchange:
                return value
            return None
        finally:
            exchange.close()

    def _exchange(self, op: str, args: dict, streaming: bool,
                  cancelled: Optional[threading.Event] = None) -> Iterator:
        """Run one request to completion, yielding chunks then the result.

        Holds the connection lock for the whole exchange — including a
        streamed generation, which is what serializes this client's requests.
        """
        with self._lock:
            request_id = next(self._ids)
            try:
                sock = self._connect()
                # Model selection is project-scoped — per-project settings
                # override the workspace defaults — so the host has to resolve
                # it against *this* project, not against whichever one the UI
                # happens to have selected in the supervisor.
                args = {**args, "project_id": pinned_project_id()}
                protocol.send_frame(
                    sock, {"id": request_id, "type": REQUEST, "op": op, "args": args},
                )
                cancel_sent = False
                while True:
                    if (streaming and cancelled is not None
                            and cancelled.is_set() and not cancel_sent):
                        # Advisory: the host stops between tokens, and the
                        # terminal frame still arrives, so the connection stays
                        # usable for the next request.
                        protocol.send_frame(sock, {"id": request_id, "type": CANCEL})
                        cancel_sent = True
                    frame = protocol.recv_frame(sock)
                    if frame.get("id") != request_id:
                        raise ProtocolError(
                            f"host answered request {frame.get('id')}, expected {request_id}"
                        )
                    kind = frame.get("type")
                    if kind == CHUNK:
                        yield frame.get("text", "")
                    elif kind == RESULT:
                        yield frame.get("value")
                        return
                    elif kind == ERROR:
                        raise ModelHostError(frame.get("error") or "model host failed")
                    else:
                        raise ProtocolError(f"unexpected frame type {kind!r}")
            except (OSError, ConnectionClosed) as exc:
                self._drop()
                raise ModelHostError(f"model host at {self.address} is unreachable: {exc}") from exc

    # -- ModelClient --------------------------------------------------------

    def generate(self, prompt, model_name=None, capability=None,
                 max_tokens=None, thinking=None) -> str:
        return self._call("generate", prompt=prompt, model_name=model_name,
                          capability=capability, max_tokens=max_tokens,
                          thinking=thinking) or ""

    def stream(self, prompt, model_name=None, capability=None, cancelled=None,
               thinking=None):
        stream = self._exchange(
            "stream",
            {"prompt": prompt, "model_name": model_name,
             "capability": capability, "thinking": thinking},
            streaming=True,
            cancelled=cancelled,
        )
        for item in stream:
            # The terminal RESULT frame closes the stream and carries nothing.
            if item is None:
                return
            yield item

    def encode_texts(self, texts, base_dir=None):
        return protocol.decode_array(
            self._call("encode_texts", texts=list(texts), base_dir=base_dir)
        )

    def is_loaded(self) -> bool:
        return bool(self._call("is_loaded"))

    def supports_thinking(self, capability=None) -> bool:
        return bool(self._call("supports_thinking", capability=capability))

    def model_name(self) -> Optional[str]:
        return self._call("model_name")

    def model_label(self, capability=None) -> str:
        return self._call("model_label", capability=capability) or ""

    def count_tokens(self, text: str) -> int:
        return int(self._call("count_tokens", text=text) or 0)

    def max_context_tokens(self) -> int:
        return int(self._call("max_context_tokens") or 0)

    def response_max_context_tokens(self, capability=None) -> int:
        return int(self._call("response_max_context_tokens",
                              capability=capability) or 0)

    def has_capability_model(self, capability: str) -> bool:
        return bool(self._call("has_capability_model", capability=capability))

    def idle_seconds(self) -> float:
        return float(self._call("idle_seconds") or 0.0)

    def load_model(self) -> Optional[str]:
        return self._call("load_model")

    def unload_auxiliary_models(self, include_embedding: bool = True) -> None:
        self._call("unload_auxiliary_models", include_embedding=include_embedding)

    def unload_embedding_model(self) -> None:
        self._call("unload_embedding_model")


class ModelHostError(RuntimeError):
    """The model host refused, failed, or could not be reached."""


def parse_address(address: str) -> tuple[str, Optional[int]]:
    """Split *address* into ``(path, None)`` or ``(host, port)``.

    A unix socket path is the normal case.  ``host:port`` is what Windows
    gets — no ``AF_UNIX`` there — and is also useful for debugging a host that
    is not on the same filesystem.
    """
    if address.startswith("unix://"):
        return address[len("unix://"):], None
    if address.startswith("unix:"):
        return address[len("unix:"):], None
    if os.sep in address or address.startswith("."):
        return address, None
    host, sep, port = address.rpartition(":")
    if sep and port.isdigit():
        return (host or "127.0.0.1"), int(port)
    return address, None


#: Longest a unix socket path may be. ``sockaddr_un.sun_path`` is 108 bytes on
#: Linux and 104 on macOS; the smaller is used so an address that works on one
#: works on the other, and one byte is left for the terminator.
MAX_UNIX_SOCKET_PATH = 103


def default_socket_path(home_dir: str) -> str:
    """Where the model host listens when nothing says otherwise.

    Normally ``<home>/model-host.sock`` — a filesystem object, so the OS
    enforces who may connect.  Two things take that away:

    * **No AF_UNIX at all.**  Windows builds of CPython do not define it, so
      there is nothing to bind and the host listens on loopback TCP instead.
    * **A home too deep for ``sun_path``.**  It is 104 bytes, and the macOS App
      Sandbox puts the home under ``~/Library/Containers/<bundle id>/Data``.
      A short name in the system temp directory, keyed by the home so two
      workspaces never share one, usually fits; loopback is the last resort.

    Loopback binds to 127.0.0.1, so nothing off-machine can reach it — but it
    has no file permissions, so any local process may connect where a unix
    socket would have said no.  That is the trade Windows makes today.
    """
    if not HAS_UNIX_SOCKETS:
        logger.info(
            "[model-host] no AF_UNIX on this platform; listening on loopback",
        )
        return "127.0.0.1:0"

    path = os.path.join(home_dir, "model-host.sock")
    if len(path.encode("utf-8")) <= MAX_UNIX_SOCKET_PATH:
        return path

    digest = hashlib.sha256(os.path.abspath(home_dir).encode("utf-8")).hexdigest()[:12]
    short = os.path.join(tempfile.gettempdir(), f"slp-model-host-{digest}.sock")
    if len(short.encode("utf-8")) <= MAX_UNIX_SOCKET_PATH:
        logger.info(
            "[model-host] %s is too long for a unix socket; using %s", path, short,
        )
        return short

    logger.info(
        "[model-host] no usable unix socket path; listening on loopback instead",
    )
    return "127.0.0.1:0"


def get_model_client(model_service=None) -> ModelClient:
    """The client this process should use, from ``SLP_MODEL_HOST``.

    Unset means the model lives in this process and calls go straight
    through — which is the supervisor itself (it hosts the device) and the
    ``--no-supervised`` fallback.
    """
    address = (os.environ.get("SLP_MODEL_HOST") or "").strip()
    if not address or is_device_owner():
        if model_service is None:
            from smartloop.api.services import get_model_service

            model_service = get_model_service()
        return InProcessModelClient(model_service)
    return RemoteModelClient(address)
