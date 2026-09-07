"""Dense embedding helpers backed by llama.cpp.

Provides a lazily-loaded singleton GGUF embedding model (see
:mod:`smartloop.db.llama_embeddings`) plus utilities for encoding texts and
serialising embeddings to/from SQLite BLOBs.  Embeddings run on the bundled
llama.cpp Metal/Vulkan backend — no PyTorch — which avoids the
``libtorch_cpu.dylib`` MPS shader-JIT segfault seen in the packaged build.

When the model cannot be loaded (missing GGUF, download failure, etc.) all
public functions gracefully return ``None`` so callers fall back to TF-IDF.
"""

import atexit
import contextlib
import logging
import os
import threading
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Cache the GGUF under its own subdir so switching models doesn't collide on disk.
_CACHE_SUBDIR = os.path.join("models", "embeddings")

# Module-level singleton — set on first successful load.  The lock serializes
# the one-time load so parallel DAG worker threads don't build the model twice
# (or race the llama.cpp init).
_model = None
_model_load_attempted = False
_load_lock = threading.Lock()


def active_embedding_identity() -> str:
    """Identity of the active embedding model, stored in the index.

    Encodes the engine + model so switching either invalidates the stored
    vectors (they live in different, non-comparable spaces) and the sqlite store
    recomputes.  Mirrors the model selection in :func:`_get_model`.
    """
    try:
        from smartloop.config.app_settings import AppSettings
        settings = AppSettings()
        return f"llama-cpp:{settings.embedding_gguf_repo}/{settings.embedding_gguf_file}"
    except Exception:
        return "llama-cpp:unknown"


#: Connection to the model host, when one owns the embedder for this process.
_remote = None
_remote_lock = threading.Lock()


def _remote_embedder():
    """The model host to embed through, or ``None`` to embed locally.

    BGE-M3 is ~417 MB.  Loading it in every per-project agent would multiply
    that by the number of open projects for no benefit — the vectors are
    identical whoever computes them — so a child hands the work to the process
    that already holds the model.  The host itself always embeds locally: it
    *is* the model, and delegating would be a call to itself.
    """
    global _remote

    from smartloop.model_host.client import is_device_owner

    address = (os.environ.get("SLP_MODEL_HOST") or "").strip()
    if not address or is_device_owner():
        return None
    if _remote is not None and _remote.address == address:
        return _remote
    with _remote_lock:
        if _remote is None or _remote.address != address:
            from smartloop.model_host.client import RemoteModelClient

            _remote = RemoteModelClient(address)
    return _remote


def _get_model(base_dir: Optional[str] = None):
    """Return the cached embedding model, loading on first call.

    The GGUF is cached under ``<base_dir>/models/embeddings/``.  Returns ``None``
    if the model cannot be loaded (callers then fall back to TF-IDF).
    """
    global _model, _model_load_attempted

    if _model is not None:
        return _model
    if _model_load_attempted:
        return None

    with _load_lock:
        # Re-check inside the lock: another thread may have finished loading
        # while we waited.
        if _model is not None:
            return _model
        if _model_load_attempted:
            return None
        _model_load_attempted = True

        from smartloop.config import AppSettings
        settings = AppSettings()

        cache_dir = None
        if base_dir:
            cache_dir = os.path.join(base_dir, _CACHE_SUBDIR)
            os.makedirs(cache_dir, exist_ok=True)

        from smartloop.db.llama_embeddings import load_embedder
        _model = load_embedder(
            cache_dir,
            settings.embedding_gguf_repo,
            settings.embedding_gguf_file,
            device=settings.embedding_device,
            base_url=settings.embedding_gguf_base_url,
        )
        return _model


def unload() -> None:
    """Release the embedding model, freeing its weights and llama.cpp context.

    Indexing is bursty — an MCP crawl embeds a few hundred chunks and then
    nothing touches the model for minutes — but the singleton held the GGUF
    resident for the life of the process. Dropping it after a burst returns
    that memory; the next search reloads it lazily.

    ``_model_load_attempted`` is cleared as well as ``_model``. It exists to
    stop retrying a load that failed, and leaving it set here would make the
    unload permanent: every later call would see the "already tried" flag and
    fall back to TF-IDF for the rest of the process.
    """
    global _model, _model_load_attempted

    with _load_lock:
        m, _model = _model, None
        _model_load_attempted = False

    if m is None:
        return
    try:
        close = getattr(m, "close", None)
        if close is not None:
            close()
        logger.info("[embeddings] model unloaded")
    except Exception as e:
        logger.warning("[embeddings] unload failed: %s", e)


# Depth of nested indexing_burst() scopes. Only the outermost unloads: an
# indexer that calls another would otherwise drop the model out from under its
# caller and force a mid-burst reload.
_burst_depth = 0
_burst_lock = threading.Lock()


@contextlib.contextmanager
def indexing_burst():
    """Release the embedding model when this indexing scope finishes.

    Indexing is bursty — a crawl or a workspace pass embeds a few hundred
    chunks and then nothing touches the model for minutes — so holding a GGUF
    and its llama.cpp context resident afterwards is pure cost. Wrapping the
    entry points keeps that decision in one place; adding an indexer means
    wrapping it, not remembering to call unload().

    Unloads in a finally: an indexing pass that raised has still loaded the
    model. Search is deliberately *not* wrapped — it queries repeatedly, and
    unloading between queries would just thrash.
    """
    global _burst_depth

    with _burst_lock:
        _burst_depth += 1
    try:
        yield
    finally:
        with _burst_lock:
            _burst_depth -= 1
            outermost = _burst_depth == 0
        if outermost:
            unload()


@atexit.register
def _shutdown_embeddings() -> None:
    """Close the GPU-offloaded embedder at interpreter exit.

    Without this, llama-cpp-python 0.3.16's Metal context aborts (SIGABRT) during
    final GC after multi-threaded use — a dirty shutdown that a supervisor may
    read as a crash.  Closing explicitly here makes exit clean.
    """
    global _model
    m, _model = _model, None
    if m is not None and hasattr(m, "close"):
        try:
            m.close()
        except Exception:
            pass


def warmup(base_dir: Optional[str] = None) -> None:
    """Pre-load the embedding model so the first search doesn't pay the cost.

    A no-op when a model host owns the embedder: it is already warm there, and
    loading a second copy here is the cost this avoids.
    """
    if _remote_embedder() is not None:
        return
    _get_model(base_dir)


def encode_texts(
    texts: List[str], base_dir: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Encode *texts* into a dense embedding matrix.

    Returns an ``(N, dim)`` numpy array, or ``None`` when the model is
    unavailable.  Routed to the model host when one owns the embedder, so every
    project agent shares a single resident copy.
    """
    remote = _remote_embedder()
    if remote is not None:
        try:
            return remote.encode_texts(texts, base_dir=base_dir)
        except Exception as exc:
            # Same contract as a failed local load: callers fall back to
            # TF-IDF rather than the child quietly loading its own model.
            logger.warning("[embeddings] model host encoding failed: %s", exc)
            return None

    model = _get_model(base_dir)
    if model is None:
        return None

    try:
        embeddings = model.encode(texts, show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float32)
    except Exception as exc:
        logger.warning(f"[embeddings] Encoding failed: {exc}")
        return None


# ------------------------------------------------------------------ BLOB I/O

def embedding_to_blob(vec: np.ndarray) -> bytes:
    """Serialise a 1-D numpy vector to raw bytes for SQLite BLOB storage."""
    return vec.astype(np.float32).tobytes()


def blob_to_embedding(blob: bytes) -> np.ndarray:
    """Deserialise a BLOB back to a 1-D float32 numpy vector."""
    return np.frombuffer(blob, dtype=np.float32)


# --------------------------------------------------------------- similarity

def cosine_distances(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine distance (``1 - similarity``) from *query_vec* to each row.

    The one place the comparison is defined, so everything that asks "is this
    text about the same thing as that text?" — chunk retrieval ranking a
    corpus against the user's question, citation verification checking a
    source against the finished answer — agrees on what a distance of 0.55
    means.  Rows are re-normalised here rather than trusted: vectors come
    both from the embedder (already L2-normalised) and from SQLite BLOBs
    written by older index versions.
    """
    matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float32))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    q = np.asarray(query_vec, dtype=np.float32).ravel()
    q = q / (np.linalg.norm(q) or 1)
    return 1.0 - ((matrix / norms) @ q)
