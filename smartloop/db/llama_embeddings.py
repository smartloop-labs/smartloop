"""llama.cpp-backed dense embeddings.

Runs a GGUF embedding model on the same bundled llama.cpp Metal/Vulkan backend
used for generation, so it needs no torch — which removes the
``libtorch_cpu.dylib`` MPS shader-JIT segfault seen only in the packaged
(PyInstaller) build, and lets embeddings offload to the GPU like the generative
models do.  Configured via ``slp_embedding_gguf_repo`` / ``slp_embedding_gguf_file``
and ``slp_embedding_device``.

The public surface mirrors the single method the sqlite store uses from a
``SentenceTransformer`` — ``.encode(texts, show_progress_bar=...)`` returning a
float32 ``numpy`` array — so ``embeddings._get_model`` can return either backend
transparently.
"""
import logging
import os
import sys
import threading
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Subdir under the PyInstaller bundle where the spec places the GGUF (see
# smartloop.spec). Present only in frozen builds that bundled the model.
_BUNDLE_SUBDIR = "embeddings"

# Let the model drive pooling: BGE-M3 uses CLS, E5/GTE use MEAN, and the GGUF
# records the right one.  LLAMA_POOLING_TYPE_UNSPECIFIED == -1 tells llama.cpp to
# use the model's own pooling_type; hardcoded so importing this module never
# requires llama_cpp (kept lazy).  Forcing a specific type (e.g. MEAN) yields
# wrong vectors for CLS models like BGE-M3.
_POOLING_TYPE_UNSPECIFIED = -1

# Context window for the embedding model.  BGE-M3 trains to 8192; retrieval
# chunks are far shorter, but sizing to the model's max avoids silent truncation.
_EMBED_N_CTX = 8192

# Chunks handed to llama.cpp per embed() call.  The whole list used to go in
# one call, so indexing a large document (an MCP crawl can produce hundreds of
# chunks) allocated for every chunk at once — with n_ubatch sized to the full
# 8192-token context, that is what pushed memory over.  Slicing bounds the peak
# to a fixed cost regardless of how much is being indexed; the model is loaded
# once either way, so the only price is a few more calls.
_MAX_ENCODE_BATCH = 16

# First-run GGUF download.  The read timeout bounds the idle gap between
# received bytes (a stalled transfer is detected instead of hanging), not the
# total transfer time — a ~400MB model is allowed to take as long as it takes.
_DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024
_DOWNLOAD_CONNECT_TIMEOUT = 30.0
_DOWNLOAD_READ_TIMEOUT = 60.0


class LlamaEmbedder:
    """Adapts a llama.cpp embedding ``Llama`` to the ``.encode()`` call site."""

    def __init__(self, model, identity: str):
        self._model = model
        # A llama.cpp context is NOT thread-safe: the retrieval fan-out queries
        # the documents/feedback/workspace stores from parallel DAG worker
        # threads, all sharing this one embedder — concurrent embed() calls on a
        # single context segfault.  Serialize them (embedding is fast).
        self._lock = threading.Lock()
        # Stored in the sqlite index so switching model/engine triggers a
        # re-embed (the vector space is not comparable across models).
        self.identity = identity

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress_bar: bool = False,  # accepted for ST call-site parity
        **_: object,
    ) -> np.ndarray:
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        if not batch:
            return np.empty((0, 0), dtype=np.float32)

        # embed() returns one vector per input (pooling is enabled).  Sliced
        # so peak memory tracks _MAX_ENCODE_BATCH rather than len(batch).
        vecs: List[list] = []
        with self._lock:
            for i in range(0, len(batch), _MAX_ENCODE_BATCH):
                vecs.extend(self._model.embed(batch[i:i + _MAX_ENCODE_BATCH]))
        arr = np.asarray(vecs, dtype=np.float32)

        # L2-normalize so cosine similarity == dot product (the sqlite store
        # ranks by dot/cosine).  SentenceTransformer returns normalized vectors
        # for these retrieval models, so match that to keep scores comparable.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.clip(norms, 1e-12, None)

        return arr[0] if single else arr

    def close(self) -> None:
        """Free the llama.cpp context.

        Must run at shutdown when the model is GPU-offloaded: llama-cpp-python
        0.3.16's Metal teardown aborts (SIGABRT) if the context is still live at
        interpreter exit after multi-threaded use.  Closing explicitly avoids it.
        """
        with self._lock:
            try:
                close = getattr(self._model, "close", None)
                if close is not None:
                    close()
            except Exception:
                pass


def _n_gpu_layers_for(device: str) -> int:
    """Map the ``embedding_device`` setting to llama.cpp offload.

    ``cpu`` keeps embeddings off the GPU (0 layers) so they don't compete with
    the generative model copies for VRAM; anything else offloads fully (-1),
    which is the point of the llama-cpp backend on Apple/Metal.
    """
    return 0 if (device or "").strip().lower() == "cpu" else -1


def _bundled_gguf_path(filename: str) -> Optional[str]:
    """Return the packaged GGUF path in a frozen build, or ``None``.

    PyInstaller unpacks bundled data under ``sys._MEIPASS``; the spec places the
    embedding GGUF at ``<_MEIPASS>/embeddings/<filename>``.  When present this is
    used directly so a packaged install needs no first-run download and works
    fully offline.
    """
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        return None
    path = os.path.join(base, _BUNDLE_SUBDIR, filename)
    return path if os.path.isfile(path) else None


def _hf_cached_gguf_path(
    cache_dir: Optional[str], repo_id: str, filename: str
) -> Optional[str]:
    """An already-downloaded copy in the HuggingFace cache layout, or ``None``.

    Installs that predate the bucket download hold the GGUF under
    ``<cache_dir>/models--<org>--<repo>/snapshots/<rev>/<filename>``.  Reusing
    it keeps upgrades from re-fetching ~400MB that is already on disk.  This is
    a filesystem lookup only — no Hub call, no token.
    """
    if not cache_dir:
        return None
    repo_dir = os.path.join(cache_dir, "models--" + repo_id.replace("/", "--"))
    snapshots = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(snapshots):
        return None
    for rev in sorted(os.listdir(snapshots)):
        candidate = os.path.join(snapshots, rev, filename)
        if os.path.isfile(candidate):
            return candidate
    return None


def _cdn_gguf_path(cache_dir: Optional[str], base_url: str, filename: str) -> Optional[str]:
    """Download *filename* from our public bucket into *cache_dir*, or ``None``.

    Preferred over the HuggingFace Hub: the Hub gates GGUF downloads behind a
    user token and is slow from most regions, whereas this is an
    unauthenticated GCS object served from the same bucket as the release
    archives.  Falls through (returns ``None``) on any failure so the caller
    can still try HF.

    The file lands via a ``.part`` file renamed on success, so an interrupted
    download never leaves a truncated GGUF that llama.cpp would later fail to
    load — and the size is checked against ``Content-Length`` first.
    """
    if not cache_dir or not base_url:
        return None

    target = os.path.join(cache_dir, filename)
    if os.path.isfile(target):
        logger.info("[embeddings] Using cached GGUF %s", target)
        return target

    url = f"{base_url.rstrip('/')}/{filename}"
    partial = f"{target}.part"
    try:
        import httpx

        os.makedirs(cache_dir, exist_ok=True)
        logger.info("[embeddings] Downloading GGUF from %s", url)
        written = 0
        expected = None
        timeout = httpx.Timeout(
            connect=_DOWNLOAD_CONNECT_TIMEOUT,
            read=_DOWNLOAD_READ_TIMEOUT,
            write=_DOWNLOAD_CONNECT_TIMEOUT,
            pool=_DOWNLOAD_CONNECT_TIMEOUT,
        )
        with httpx.stream(
            "GET", url, follow_redirects=True, timeout=timeout,
        ) as response:
            response.raise_for_status()
            length = response.headers.get("content-length")
            expected = int(length) if length and length.isdigit() else None
            with open(partial, "wb") as fh:
                for chunk in response.iter_bytes(_DOWNLOAD_CHUNK_SIZE):
                    fh.write(chunk)
                    written += len(chunk)

        if expected is not None and written != expected:
            raise OSError(
                f"truncated download: got {written} of {expected} bytes"
            )

        os.replace(partial, target)
        logger.info("[embeddings] Downloaded GGUF %s (%d bytes)", filename, written)
        return target
    except Exception as exc:
        logger.warning("[embeddings] Failed to fetch GGUF from %s: %s", url, exc)
        try:
            os.remove(partial)
        except OSError:
            pass
        return None


def load_embedder(
    cache_dir: Optional[str],
    repo_id: str,
    filename: str,
    device: str = "cpu",
    base_url: str = "",
) -> Optional[LlamaEmbedder]:
    """Load the GGUF embedding model — bundled copy first, else download.

    Resolution order is bundled → an existing HuggingFace-cache copy on disk →
    our public bucket (*base_url*).  The Hub itself is deliberately not called:
    it gates GGUF downloads behind a user token and is slow from most regions,
    which is the whole reason the model is mirrored to our bucket.  Returns
    ``None`` on any failure so callers fall back to TF-IDF, matching the
    sentence-transformers path's contract.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.warning(
            "[embeddings] llama-cpp-python is not installed — "
            "cannot use the llama-cpp embedding backend"
        )
        return None

    model_path = _bundled_gguf_path(filename)
    if model_path:
        logger.info("[embeddings] Using bundled GGUF %s", filename)
    else:
        model_path = (
            _hf_cached_gguf_path(cache_dir, repo_id, filename)
            or _cdn_gguf_path(cache_dir, base_url, filename)
        )
    if not model_path:
        logger.warning(
            "[embeddings] No GGUF available for %s — falling back to TF-IDF",
            filename,
        )
        return None

    try:
        model = Llama(
            model_path=model_path,
            embedding=True,
            pooling_type=_POOLING_TYPE_UNSPECIFIED,
            n_gpu_layers=_n_gpu_layers_for(device),
            n_ctx=_EMBED_N_CTX,
            # BGE-M3 is a non-causal encoder: llama.cpp processes each sequence
            # in a single micro-batch (no KV-cache splitting), so n_ubatch must
            # be >= the longest input or ggml_abort()s in set_inputs.  Retrieval
            # chunks can exceed the 512 default, so size both to the context.
            n_batch=_EMBED_N_CTX,
            n_ubatch=_EMBED_N_CTX,
            verbose=False,
        )
    except Exception as exc:
        logger.warning(
            f"[embeddings] Failed to load GGUF embedder {filename}: {exc}"
        )
        return None

    identity = f"llama-cpp:{repo_id}/{filename}"
    logger.info(
        "[embeddings] Loaded %s (llama-cpp, n_gpu_layers=%d)",
        identity, _n_gpu_layers_for(device),
    )
    return LlamaEmbedder(model, identity)
