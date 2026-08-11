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
        )
        return _model


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
    """Pre-load the embedding model so the first search doesn't pay the cost."""
    _get_model(base_dir)


def encode_texts(
    texts: List[str], base_dir: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Encode *texts* into a dense embedding matrix.

    Returns an ``(N, dim)`` numpy array, or ``None`` when the model is
    unavailable.
    """
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
