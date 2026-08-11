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

        # embed() returns one vector per input (pooling is enabled).
        with self._lock:
            vecs = self._model.embed(batch)
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


def load_embedder(
    cache_dir: Optional[str],
    repo_id: str,
    filename: str,
    device: str = "cpu",
) -> Optional[LlamaEmbedder]:
    """Load the GGUF embedding model — bundled copy first, else download.

    Returns ``None`` on any failure so callers fall back to TF-IDF, matching the
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
        try:
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            logger.warning(
                f"[embeddings] Failed to fetch GGUF {repo_id}/{filename}: {exc}"
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
