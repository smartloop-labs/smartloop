"""The interface a project agent uses to reach the generative model.

The model is always reached through the gRPC IPC transport.  The host process
owns the device and serves the model; project agents only ever call into it
through :class:`RemoteModelClient`.

The in-process implementation remains only for the host's own local call path,
not as a user-visible fallback or a second transport mode.

Select with ``SLP_MODEL_HOST`` — a unix socket path (or ``host:port``); if it
is unset, the default host socket is used.
"""

from __future__ import annotations

import hashlib
import logging
import os
import socket
import tempfile
import threading
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

import grpc
import numpy as np

from smartloop.model_host import model_host_pb2 as pb2
from smartloop.model_host import model_host_pb2_grpc as pb2_grpc
from smartloop.workspace.context import pinned_project_id

logger = logging.getLogger(__name__)

#: Set in the process that actually holds the weights.  Without it a host
#: started with ``SLP_MODEL_HOST`` pointing at its own socket would resolve a
#: remote client and try to answer its own requests.
_device_owner = False


#: Whether this interpreter can speak AF_UNIX at all.  Feature-detected rather
#: than guessed from the platform, which is what CPython's own ``socketpair``
#: does — Windows builds simply do not define it.
HAS_UNIX_SOCKETS = hasattr(socket, "AF_UNIX")

#: grpcio's default cap on a received message is 4 MiB; an embedding batch can
#: exceed that.  Mirrors the server's own ceiling so a large payload is
#: refused on both sides at the same limit.
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


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
        llama.cpp call cannot be interrupted mid-token, in-process or over the
        wire.
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
    """The device is owned by a model host — talk to it over gRPC.

    One channel, used by every thread: gRPC multiplexes RPCs on it, and a
    project agent's concurrency is already bounded by the host's own
    ``n_parallel`` budget, so a pool would only move the queue without
    widening it.

    The channel is opened lazily and reconnects transparently on failure, so a
    host restart costs the in-flight request rather than the client.
    """

    def __init__(self, address: str, timeout: float = 600.0) -> None:
        self.address = address
        self._timeout = timeout
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[pb2_grpc.ModelHostStub] = None
        self._lock = threading.Lock()

    # -- transport ----------------------------------------------------------

    def _connect(self) -> pb2_grpc.ModelHostStub:
        if self._stub is not None:
            return self._stub
        with self._lock:
            if self._stub is None:
                host, port = parse_address(self.address)
                if port is None and not HAS_UNIX_SOCKETS:
                    raise ModelHostError(
                        f"{self.address} is a unix socket path, which this platform "
                        "cannot open; the host should be listening on host:port"
                    )
                if port == 0:
                    raise ModelHostError(
                        f"model host address {self.address!r} is not bound: "
                        "the host is not listening on a real port"
                    )
                channel = grpc.insecure_channel(
                    _grpc_target(self.address),
                    options=[
                        ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
                        ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
                    ],
                )
                self._channel = channel
                self._stub = pb2_grpc.ModelHostStub(channel)
                logger.info("[model-host] connected to %s", self.address)
            return self._stub

    def close(self) -> None:
        with self._lock:
            self._drop()

    def _drop(self) -> None:
        """Close the channel and forget it, so the next call reconnects."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def _unary(self, method: str, request):
        """Run one unary RPC and return its message, or raise :class:`ModelHostError`."""
        try:
            return getattr(self._connect(), method)(request, timeout=self._timeout)
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.UNAVAILABLE:
                # The host is gone (crashed, restarted, or never there).  A
                # dead channel caches that failure; drop it so the next call
                # reconnects instead of re-raising.
                self._drop()
            raise _rpc_error(self.address, exc) from exc

    # -- ModelClient --------------------------------------------------------

    def generate(self, prompt, model_name=None, capability=None,
                 max_tokens=None, thinking=None) -> str:
        request = pb2.GenerateRequest(prompt=prompt or "")
        if model_name:
            request.model_name = model_name
        if capability:
            request.capability = capability
        if max_tokens is not None:
            request.max_tokens = max_tokens
        if thinking is not None:
            request.thinking = thinking
        request.project_id = pinned_project_id() or ""
        reply = self._unary("Generate", request)
        return reply.text or ""

    def stream(self, prompt, model_name=None, capability=None, cancelled=None,
               thinking=None):
        request = pb2.StreamRequest(prompt=prompt or "")
        if model_name:
            request.model_name = model_name
        if capability:
            request.capability = capability
        if thinking is not None:
            request.thinking = thinking
        request.project_id = pinned_project_id() or ""

        call = None
        try:
            call = self._connect().Stream(request, timeout=None)
            for item in call:
                if cancelled is not None and cancelled.is_set():
                    # Advisory, exactly like the old cancel frame: the host
                    # stops between tokens, and the connection stays usable
                    # for the next request.
                    call.cancel()
                    return
                yield item.text
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.UNAVAILABLE:
                self._drop()
            raise _rpc_error(self.address, exc) from exc
        finally:
            # Abandoned mid-way (a caller that broke out of the loop) must
            # cancel the RPC server-side, or a generation would be computed
            # for nobody.  Harmless once the stream has finished.
            if call is not None:
                call.cancel()

    def encode_texts(self, texts, base_dir=None):
        request = pb2.EncodeTextsRequest(texts=list(texts), base_dir=base_dir or "")
        request.project_id = pinned_project_id() or ""
        reply = self._unary("EncodeTexts", request)
        if not reply.shape:
            return None
        return np.frombuffer(reply.data, dtype=np.float32).reshape(reply.shape)

    def is_loaded(self) -> bool:
        return bool(self._bool_reply("IsLoaded").value)

    def supports_thinking(self, capability=None) -> bool:
        request = pb2.SupportsThinkingRequest(capability=capability or "")
        request.project_id = pinned_project_id() or ""
        return self._unary("SupportsThinking", request).value

    def model_name(self) -> Optional[str]:
        return self._str_reply("ModelName").value or None

    def model_label(self, capability=None) -> str:
        request = pb2.ModelLabelRequest(capability=capability or "")
        request.project_id = pinned_project_id() or ""
        return self._unary("ModelLabel", request).value or ""

    def count_tokens(self, text: str) -> int:
        request = pb2.CountTokensRequest(text=text or "")
        request.project_id = pinned_project_id() or ""
        return self._unary("CountTokens", request).value

    def max_context_tokens(self) -> int:
        return self._int_reply("MaxContextTokens").value

    def response_max_context_tokens(self, capability=None) -> int:
        request = pb2.ResponseMaxContextRequest(capability=capability or "")
        request.project_id = pinned_project_id() or ""
        return self._unary("ResponseMaxContextTokens", request).value

    def has_capability_model(self, capability: str) -> bool:
        request = pb2.HasCapabilityRequest(capability=capability or "")
        request.project_id = pinned_project_id() or ""
        return self._unary("HasCapabilityModel", request).value

    def idle_seconds(self) -> float:
        return self._float_reply("IdleSeconds").value

    def load_model(self) -> Optional[str]:
        return self._str_reply("LoadModel").value or None

    def unload_auxiliary_models(self, include_embedding: bool = True) -> None:
        request = pb2.UnloadAuxiliaryRequest(include_embedding=include_embedding)
        request.project_id = pinned_project_id() or ""
        self._unary("UnloadAuxiliaryModels", request)

    def unload_embedding_model(self) -> None:
        self._unary("UnloadEmbeddingModel", self._project_request())

    # -- request/reply helpers ---------------------------------------------

    def _project_request(self) -> pb2.ProjectRequest:
        return pb2.ProjectRequest(project_id=pinned_project_id() or "")

    def _bool_reply(self, method: str) -> pb2.BoolReply:
        return self._unary(method, self._project_request())

    def _str_reply(self, method: str) -> pb2.StringReply:
        return self._unary(method, self._project_request())

    def _int_reply(self, method: str) -> pb2.IntReply:
        return self._unary(method, self._project_request())

    def _float_reply(self, method: str) -> pb2.FloatReply:
        return self._unary(method, self._project_request())


class ModelHostError(RuntimeError):
    """The model host refused, failed, or could not be reached."""


def _rpc_error(address: str, exc: grpc.RpcError) -> ModelHostError:
    """Describe a failed RPC the way a failing socket used to be described."""
    code = exc.code()
    details = exc.details() or str(code)
    if code == grpc.StatusCode.UNAVAILABLE:
        return ModelHostError(f"model host at {address} is unreachable: {details}")
    if code == grpc.StatusCode.CANCELLED:
        return ModelHostError(f"model host at {address} cancelled the call: {details}")
    return ModelHostError(f"model host at {address} failed: {details}")


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


def _grpc_target(address: str) -> str:
    """How gRPC addresses *address*: ``unix:``-prefixed or ``host:port``."""
    host, port = parse_address(address)
    if port is None:
        return f"unix:{host}"
    return f"{host}:{port}"


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
    """The client this process should use.

    A project child gets a host address from ``SLP_MODEL_HOST``; a process that
    owns the device, or one that has not started a host yet, must stay on the
    in-process path instead of trying to reach a placeholder ``127.0.0.1:0``.

    A port of zero is not a real bind and must never be treated as a usable
    model host address; it indicates the host was not started or was asked to
    bind on an ephemeral port that never got published.
    """
    from smartloop.config import AppSettings

    address = (os.environ.get("SLP_MODEL_HOST") or "").strip()
    if not address:
        # No explicit host means there is no model-host process to reach in this
        # process.  The single-server and local-owner paths remain in-process;
        # a child without a host address should never silently fall through to
        # a remote connection against a placeholder port.
        if model_service is not None:
            return InProcessModelClient(model_service)
        address = default_socket_path(AppSettings().home_dir)

    if parse_address(address)[1] == 0:
        raise ModelHostError(
            f"model host address {address!r} is not bound: "
            "a real host port is required before remote model access"
        )
    return RemoteModelClient(address)