"""Regression tests for _evict_capability_models and the default-model guard.

Pressing Esc mid-stream aborts the SSE generator, whose ``finally`` calls
``unload_auxiliary_models()`` -> ``_evict_capability_models()``.  When the
capability ranking had picked the *default* orchestrator model, its
``_cap_llm_<default>`` slot holds a separate Llama instance for the same
weights, so ``_free_llm``'s identity check against ``self._llm`` does not
cover it.  Closing it there frees the native context a live stream is still
decoding on — a use-after-free that segfaults the whole server process.
"""

from unittest.mock import MagicMock

import pytest

from smartloop.api.services.model_service import ModelService, _GPU_FAULTED


@pytest.fixture
def svc():
    """A ModelService with only the attributes the eviction path touches."""
    s = object.__new__(ModelService)
    s._llm = MagicMock(name="resident_orchestrator")
    s._model_name = "slp-orchestra-mini"
    s._large_llm = None
    s._large_model_name = None
    s._large_model_attempted = False
    # The on-demand vision model is evicted here too, so the next step's
    # capability model can fit (see test_vision_evicted_for_capability.py).
    s._vision_llm = None
    s._vision_model_name = None
    s._vision_unavailable = False
    s._cleanup_gpu_cache = MagicMock()
    s._free_llm = MagicMock()
    return s


def test_default_model_capability_slot_is_not_evicted(svc):
    """The default model's own capability slot must survive eviction."""
    default_slot = MagicMock(name="cap_llm_default")
    svc._cap_llm_slp_orchestra_mini = None  # noqa: F841 - unrelated attr
    setattr(svc, "_cap_llm_slp-orchestra-mini", default_slot)

    freed = svc._evict_capability_models()

    assert freed is False
    svc._free_llm.assert_not_called()
    assert getattr(svc, "_cap_llm_slp-orchestra-mini") is default_slot


def test_non_default_capability_models_are_still_evicted(svc):
    """Other capability models are still dropped to reclaim VRAM."""
    default_slot = MagicMock(name="cap_llm_default")
    other_slot = MagicMock(name="cap_llm_other")
    setattr(svc, "_cap_llm_slp-orchestra-mini", default_slot)
    setattr(svc, "_cap_llm_big-doc-model", other_slot)

    freed = svc._evict_capability_models()

    assert freed is True
    svc._free_llm.assert_called_once_with(other_slot)
    assert getattr(svc, "_cap_llm_big-doc-model") is None
    assert getattr(svc, "_cap_llm_slp-orchestra-mini") is default_slot


def test_keep_model_name_still_honoured(svc):
    """An explicitly kept model is preserved alongside the default."""
    keep_slot = MagicMock(name="cap_llm_keep")
    drop_slot = MagicMock(name="cap_llm_drop")
    setattr(svc, "_cap_llm_keep-me", keep_slot)
    setattr(svc, "_cap_llm_drop-me", drop_slot)

    freed = svc._evict_capability_models(keep_model_name="keep-me")

    assert freed is True
    svc._free_llm.assert_called_once_with(drop_slot)
    assert getattr(svc, "_cap_llm_keep-me") is keep_slot


def test_gpu_faulted_sentinel_is_left_in_place(svc):
    """A faulted slot is a latch, not a live model — never freed."""
    setattr(svc, "_cap_llm_faulted", _GPU_FAULTED)

    freed = svc._evict_capability_models()

    assert freed is False
    svc._free_llm.assert_not_called()
    assert getattr(svc, "_cap_llm_faulted") is _GPU_FAULTED


def test_no_default_model_name_does_not_crash(svc):
    """Before a default is resolved, eviction still works normally."""
    svc._model_name = None
    slot = MagicMock(name="cap_llm_some")
    setattr(svc, "_cap_llm_some-model", slot)

    freed = svc._evict_capability_models()

    assert freed is True
    svc._free_llm.assert_called_once_with(slot)
