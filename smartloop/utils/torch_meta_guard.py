"""Guard against the leaked accelerate "empty weights" (meta device) state.

Loading an HF model with ``device_map`` / ``low_cpu_mem_usage`` enters
accelerate's ``init_empty_weights`` context, which patches the *global*
``nn.Module.register_parameter`` / ``register_buffer`` so new parameters are
created on the ``meta`` device.  The context normally restores those methods,
but because they are class-level (shared across threads) a *concurrent* model
load — e.g. the document indexer loading the docling layout model while the
embedding model loads — can observe the patched version and build its weights
on ``meta``.  The load then fails with::

    NotImplementedError: Cannot copy out of meta tensor; no data!

Note ``torch.get_default_device()`` stays ``cpu`` in this case, so a default-
device check does not detect it.

This module provides:
  * :func:`repair_meta_init` — restore the patched methods if they're patched.
  * :func:`safe_model_load` — a context manager that serializes model loads
    (so one load's meta window can't bleed into another) and repairs first.
"""
import logging
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Serializes model *loads* across threads so accelerate's global meta patch
# from one load can't be observed by another.  Inference is unaffected.
_MODEL_LOAD_LOCK = threading.RLock()

_PATCHED_MARKERS = ("register_empty", "init_on_device")


def _recover_original(patched, method_name: str):
    """Recover the original nn.Module method from accelerate's closure."""
    for cell in (getattr(patched, "__closure__", None) or []):
        try:
            candidate = cell.cell_contents
        except ValueError:
            continue
        if callable(candidate) and getattr(candidate, "__qualname__", "").endswith(
            f"Module.{method_name}"
        ):
            return candidate
    return None


def repair_meta_init() -> bool:
    """Undo a leaked accelerate empty-init patch. Returns True if repaired."""
    try:
        import torch.nn as nn
    except Exception:
        return False

    repaired = []
    for method_name in ("register_parameter", "register_buffer"):
        fn = getattr(nn.Module, method_name, None)
        qualname = getattr(fn, "__qualname__", "")
        if any(marker in qualname for marker in _PATCHED_MARKERS):
            original = _recover_original(fn, method_name)
            if original is not None:
                setattr(nn.Module, method_name, original)
                repaired.append(method_name)

    if repaired:
        logger.warning(
            "[meta-guard] Restored leaked accelerate empty-init patch "
            f"({', '.join(repaired)}); models were being created on the "
            "'meta' device."
        )
    return bool(repaired)


@contextmanager
def safe_model_load():
    """Serialize a torch model load and repair any leaked meta-init patch.

    Wrap model construction (embedding models, docling layout models, HF
    models) so concurrent loads don't poison each other via accelerate's
    global ``init_empty_weights`` patch.
    """
    with _MODEL_LOAD_LOCK:
        repair_meta_init()
        yield
