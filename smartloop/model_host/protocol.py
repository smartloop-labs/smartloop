"""Framing and message shapes for the model host connection.

The generative model cannot be per-project.  Weights loaded through mmap share
page cache between processes on CPU, but VRAM copies do not: N projects would
mean N resident copies competing for the same device, each with its own KV
cache, serialising in the driver.  That was already tried and reverted — see
the note on ``n_parallel`` in ``smartloop/config/app_settings.py``.  So the
split runs between *project work* and *inference*: a project agent owns its
queue, index, MCP clients and sessions; one model host owns the device and
keeps a single continuous-batching budget.

This module is the wire between them.

Framing
-------
A 4-byte big-endian length followed by that many bytes of UTF-8 JSON.  Chosen
over newline delimiting because generated text contains newlines and would
otherwise need escaping on the hot path.

Conversation
------------
The client sends one :data:`REQUEST` frame carrying an ``id``, an ``op`` and
its ``args``.  The host answers with any number of :data:`CHUNK` frames (for a
streamed generation) and exactly one terminal frame — :data:`RESULT` or
:data:`ERROR` — bearing the same ``id``.  Ids are per connection.

A client that wants to stop a generation early sends :data:`CANCEL` for that
id on the same connection; the host sets the cancellation event the generation
is already watching between tokens.  It is advisory and cooperative, exactly as
in-process cancellation is: a native llama.cpp call cannot be interrupted
mid-token.

Numpy arrays (embeddings) do not survive JSON, so they travel as
:func:`encode_array` — base64 of the raw float32 buffer plus its shape — which
costs one copy and no precision.
"""

from __future__ import annotations

import base64
import json
import socket
import struct
from typing import Any, Optional

import numpy as np

# --- frame types -----------------------------------------------------------

REQUEST = "request"
CANCEL = "cancel"
CHUNK = "chunk"
RESULT = "result"
ERROR = "error"

#: Length prefix: 4 bytes, big-endian, unsigned.
_HEADER = struct.Struct(">I")

#: Refuse a frame larger than this rather than allocating whatever a peer
#: claims to be sending.  Generously above any real payload (a long prompt with
#: a document's worth of context) and far below anything that would exhaust
#: memory.
MAX_FRAME_BYTES = 64 * 1024 * 1024


class ProtocolError(RuntimeError):
    """The peer sent something that is not a well-formed frame."""


class ConnectionClosed(ProtocolError):
    """The peer went away between frames."""


def send_frame(sock: socket.socket, payload: dict) -> None:
    """Write one length-prefixed JSON frame to *sock*."""
    body = json.dumps(payload).encode("utf-8")
    if len(body) > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame of {len(body)} bytes exceeds the limit")
    sock.sendall(_HEADER.pack(len(body)) + body)


def _recv_exactly(sock: socket.socket, count: int) -> bytes:
    """Read exactly *count* bytes, or raise.

    ``recv`` is free to return short reads on a stream socket, so a frame has
    to be reassembled rather than assumed to arrive whole.
    """
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        block = sock.recv(remaining)
        if not block:
            raise ConnectionClosed("peer closed the connection mid-frame")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def recv_frame(sock: socket.socket) -> dict:
    """Read one frame from *sock*, blocking until it is complete."""
    header = _recv_exactly(sock, _HEADER.size)
    (length,) = _HEADER.unpack(header)
    if length > MAX_FRAME_BYTES:
        raise ProtocolError(f"peer announced a {length}-byte frame")
    body = _recv_exactly(sock, length)
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"frame is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("frame is not a JSON object")
    return payload


# --- array transport -------------------------------------------------------

def encode_array(array: Optional[np.ndarray]) -> Optional[dict]:
    """Pack a float32 array for JSON transport (``None`` passes through)."""
    if array is None:
        return None
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return {
        "shape": list(contiguous.shape),
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def decode_array(payload: Optional[dict]) -> Optional[np.ndarray]:
    """Unpack what :func:`encode_array` produced."""
    if payload is None:
        return None
    raw = base64.b64decode(payload["data"])
    return np.frombuffer(raw, dtype=np.float32).reshape(payload["shape"])


# --- helpers ---------------------------------------------------------------

def error_payload(request_id: Any, exc: BaseException) -> dict:
    """Describe *exc* well enough for the client to re-raise something useful."""
    return {
        "id": request_id,
        "type": ERROR,
        "error": f"{type(exc).__name__}: {exc}",
    }
