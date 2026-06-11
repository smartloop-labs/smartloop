"""Dense embedding helpers using sentence-transformers.

Provides a lazily-loaded singleton ``SentenceTransformer`` model and
utility functions for encoding texts and serialising embeddings to/from
SQLite BLOBs.  When the model cannot be loaded (missing dependency,
download failure, etc.) all public functions gracefully return ``None``
so callers can fall back to TF-IDF.
"""

import contextlib
import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _hf_offline():
    """Force huggingface_hub / transformers fully offline for the duration.

    ``local_files_only=True`` is only threaded into *some* of the internal
    download calls that ``SentenceTransformer.__init__`` makes — others (the
    ``/api/models`` metadata probe and the freshness HEAD for config files)
    still hit the network and, when offline, retry with backoff
    (Retry 1/5 … 5/5) for ~30s before falling back to the cache.

    Both huggingface_hub and transformers gate their network calls on an
    offline flag that is read from the env var **once at import time**, so
    setting the env var now is too late — we must patch the already-imported
    module attributes:

    * ``huggingface_hub.constants.HF_HUB_OFFLINE`` — short-circuits every Hub
      request.
    * ``transformers.utils.hub._is_offline_mode`` — without this, the tokenizer
      loader's ``_patch_mistral_regex`` calls ``model_info()`` over the network
      even for a fully-cached local model (transformers >= 4.57); flipping it
      makes transformers treat the model as local and skip that probe.

    All three are restored on exit so this only affects the wrapped
    (known-cached) load.
    """
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    saved_env = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = "1"

    patches = []  # (module, attr, previous_value)
    for module_path, attr in (
        ("huggingface_hub.constants", "HF_HUB_OFFLINE"),
        ("transformers.utils.hub", "_is_offline_mode"),
    ):
        try:
            import importlib
            mod = importlib.import_module(module_path)
            patches.append((mod, attr, getattr(mod, attr)))
            setattr(mod, attr, True)
        except Exception:
            pass

    try:
        yield
    finally:
        for mod, attr, prev in patches:
            setattr(mod, attr, prev)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

def _resolve_model_name() -> str:
    try:
        from smartloop.config.app_settings import AppSettings
        return AppSettings().embedding_model
    except Exception:
        return "sentence-transformers/all-mpnet-base-v2"


_MODEL_NAME = _resolve_model_name()
# Cache each model under its own subdir (basename) so switching models doesn't
# collide on disk.
_CACHE_SUBDIR = os.path.join("models", "embeddings", _MODEL_NAME.split("/")[-1])

# Module-level singleton — set on first successful load.
_model = None
_model_load_attempted = False


def _is_offline_error(exc: Exception) -> bool:
    """Heuristic: did this load fail because the network was unreachable?

    huggingface_hub raises ``OfflineModeIsEnabled`` /
    ``LocalEntryNotFoundError`` when it can't reach the Hub, and the
    underlying urllib3/requests errors mention DNS/connection failures.
    """
    try:
        from huggingface_hub.utils import (
            LocalEntryNotFoundError,
            OfflineModeIsEnabled,
        )
        if isinstance(exc, (LocalEntryNotFoundError, OfflineModeIsEnabled)):
            return True
    except Exception:
        pass

    text = f"{type(exc).__name__}: {exc}".lower()
    needles = (
        "maxretryerror",
        "nameresolutionerror",
        "failed to resolve",
        "connectionerror",
        "max retries exceeded",
        "temporary failure in name resolution",
        "offline",
    )
    return any(n in text for n in needles)


def _is_model_cached(cache_dir: Optional[str]) -> bool:
    """Return True if the model already exists in the local HF cache.

    Uses ``try_to_load_from_cache`` which only inspects the on-disk cache
    and never makes a network request. We probe ``modules.json`` — the
    SentenceTransformer manifest that must be present for a usable model.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        return False

    path = try_to_load_from_cache(
        repo_id=_MODEL_NAME,
        filename="modules.json",
        cache_dir=cache_dir or HF_HUB_CACHE,
    )
    return isinstance(path, str) and os.path.isfile(path)


def _get_model(base_dir: Optional[str] = None):
    """Return the cached ``SentenceTransformer`` model, loading on first call.

    The model weights are cached under ``<base_dir>/<_CACHE_SUBDIR>``
    (defaults to ``~/.slp/models/embeddings/all-mpnet-base-v2/``).
    Returns ``None`` if the model cannot be loaded.
    """
    global _model, _model_load_attempted

    if _model is not None:
        return _model

    if _model_load_attempted:
        return None

    _model_load_attempted = True

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning(
            "[embeddings] sentence-transformers is not installed — "
            "falling back to TF-IDF"
        )
        return None

    cache_dir = None
    if base_dir:
        cache_dir = os.path.join(base_dir, _CACHE_SUBDIR)
        os.makedirs(cache_dir, exist_ok=True)

    # Serialize the load and repair any leaked accelerate meta-init patch
    # so weights don't land on the 'meta' device (see torch_meta_guard).
    from smartloop.utils.torch_meta_guard import safe_model_load

    def _load(local_only: bool):
        # When forcing local-only, also flip huggingface_hub fully offline via
        # env vars — local_files_only alone doesn't suppress the freshness HEAD
        # request that SentenceTransformer makes for config files.
        offline = _hf_offline() if local_only else contextlib.nullcontext()
        with offline, safe_model_load():
            return SentenceTransformer(
                _MODEL_NAME, cache_folder=cache_dir,
                model_kwargs={"low_cpu_mem_usage": False},
                local_files_only=local_only,
            )

    # If the model is already cached on disk, load it directly with
    # local_files_only=True. This skips the freshness HEAD request to
    # huggingface.co entirely — otherwise, when offline, that request
    # retries with backoff (Retry 1/5 … 5/5) for ~30s on every load
    # before we'd ever fall back to the cache.
    if _is_model_cached(cache_dir):
        try:
            _model = _load(local_only=True)
            logger.info(f"[embeddings] Loaded {_MODEL_NAME} from local cache")
            return _model
        except Exception as exc:
            logger.warning(
                f"[embeddings] Local cache load failed for {_MODEL_NAME}, "
                f"retrying online: {exc}"
            )

    try:
        _model = _load(local_only=False)
        logger.info(f"[embeddings] Loaded {_MODEL_NAME}")
        return _model
    except Exception as exc:
        # A cached model can fail to load purely because the freshness HEAD
        # request to huggingface.co can't reach the network (offline / DNS
        # failure). Retry once forcing local files only so the on-disk cache
        # is used without phoning home.
        if _is_offline_error(exc):
            try:
                _model = _load(local_only=True)
                logger.info(
                    f"[embeddings] Loaded {_MODEL_NAME} from local cache (offline)"
                )
                return _model
            except Exception as offline_exc:
                exc = offline_exc
        logger.warning(f"[embeddings] Failed to load {_MODEL_NAME}: {exc}")
        return None


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
