"""The vision model must give up its VRAM to the next step's capability model.

A document request with an image runs OCR first, which loads the multimodal
model (e.g. qwen2-vl-2b) into ``_vision_llm``.  Extraction runs next and needs
the 'dp' model.  ``_evict_capability_models`` used to free only the
``_cap_llm_*`` slots and the large model, so the vision model stayed resident,
the 'dp' model failed its fit guard against the VRAM it held, and extraction
fell through to the bootstrap router — the SSE reported "Selected sl-mini for
document processing" on a workspace where a 'dp' model was selected.
"""

from unittest.mock import MagicMock

import pytest

from smartloop.api.services.model_service import ModelService


@pytest.fixture
def svc():
    s = object.__new__(ModelService)
    s._llm = MagicMock(name="resident_orchestrator")
    s._model_name = "sl-mini"
    s._large_llm = None
    s._large_model_name = None
    s._large_model_attempted = False
    s._vision_llm = None
    s._vision_model_name = None
    s._vision_unavailable = False
    s._cleanup_gpu_cache = MagicMock()
    s._free_llm = MagicMock()
    # unload_vision_model takes an inference slot; stub the contextmanager away.
    s._inference_slot = MagicMock()
    s._inference_slot.return_value.__enter__ = MagicMock()
    s._inference_slot.return_value.__exit__ = MagicMock(return_value=False)
    return s


def _load_vision(svc, name="qwen2-vl-2b"):
    svc._vision_llm = MagicMock(name=f"vision_{name}")
    svc._vision_model_name = name
    return svc._vision_llm


def test_vision_model_is_evicted_to_make_room(svc):
    vision = _load_vision(svc)

    freed = svc._evict_capability_models(keep_model_name="gemma4-e2b")

    assert freed is True
    svc._free_llm.assert_called_once_with(vision)
    assert svc._vision_llm is None


def test_eviction_rearms_lazy_vision_reload(svc):
    """A later page still needing OCR must be able to load it again."""
    _load_vision(svc)
    svc._vision_unavailable = True

    svc._evict_capability_models(keep_model_name="gemma4-e2b")

    assert svc._vision_unavailable is False
    assert svc._vision_model_name is None


def test_vision_model_survives_when_it_is_the_model_being_loaded(svc):
    vision = _load_vision(svc)

    freed = svc._evict_capability_models(keep_model_name="qwen2-vl-2b")

    assert freed is False
    svc._free_llm.assert_not_called()
    assert svc._vision_llm is vision


def test_no_vision_model_loaded_is_a_no_op(svc):
    freed = svc._evict_capability_models(keep_model_name="gemma4-e2b")

    assert freed is False
    svc._free_llm.assert_not_called()
