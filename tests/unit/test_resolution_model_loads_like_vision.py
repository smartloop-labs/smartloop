"""Tool resolution loads its model on the same terms as image OCR.

MCPAgent's resolve call is the one step whose judgement is worth a reasoning
block: it picks the tool *and* fills its arguments in a single pass, and it
runs with ``thinking=True`` whatever the picker's per-model setting says (the
monologue is stripped before anything reads the JSON).  Which model runs it is
capability routing's job — ``ModelCapability.THINKING`` — exactly as
``VISION`` decides which model reads an image.

The fit guard is where the two used to differ.  ``get_vision_llm`` waives the
headroom kept for the main model (``reserve_gb=0.0``), so an image step gets
its model; the thinking capability kept the reserve, so on a memory-tight
device the tool call quietly resolved on the small router model instead —
the outcome the routing exists to avoid.  There is nothing for the reserve to
protect here: the alternative to loading is not a smaller reasoning model, it
is no reasoning at all.
"""

from unittest.mock import MagicMock, patch

import pytest

from smartloop.api.services.model_service import ModelService
from smartloop.types import ModelCapability


def _ws_model(name: str):
    m = MagicMock(name=f"ws_model:{name}")
    m.model_name = name
    m.model_id = name
    return m


@pytest.fixture
def svc():
    s = object.__new__(ModelService)
    s._llm = MagicMock(name="resident_orchestrator")
    s._model_name = "sl-mini"
    s._large_llm = None
    s._large_model_name = None
    s._model = MagicMock()
    s._model.workspace.current_project = None
    return s


def _resolve(svc, capability, ranked, pinned=None):
    """Run the real resolution and report how the fit guard was called."""
    svc._model.workspace.ranked_models_for.return_value = ranked
    with patch.object(ModelService, "_ranked_candidates", return_value=ranked), \
         patch.object(ModelService, "_pinned_model", return_value=pinned), \
         patch(
             "smartloop.llms.llm_factory.LlmFactory.get_llama_cpp_instance"
         ) as factory, \
         patch.object(
             ModelService, "_find_model_ggufs", return_value=("/models/x.gguf", None)
         ), \
         patch.object(
             ModelService, "_large_model_fits", return_value=True
         ) as fits:
        llm = ModelService._get_llm_for_capability(svc, capability)
    return llm, factory, fits


class TestTheReasoningModelLoadsWithoutTheReserve:
    def test_a_selected_thinking_model_is_loaded_for_the_resolve_step(self, svc):
        """Two models selected: the one declaring THINKING runs the step."""
        llm, factory, fits = _resolve(
            svc, ModelCapability.THINKING.value, [_ws_model("qwen3.5-2b")],
        )

        assert llm is factory.return_value
        assert factory.call_args.kwargs["model_name"] == "qwen3.5-2b"

    def test_the_fit_guard_waives_the_reserve(self, svc):
        _, _, fits = _resolve(
            svc, ModelCapability.THINKING.value, [_ws_model("qwen3.5-2b")],
        )

        assert fits.call_args.kwargs["reserve_gb"] == 0.0

    def test_the_sole_selected_model_gets_the_same_treatment(self, svc):
        """One selection is pinned to every capability it can serve, and the
        resolve step is one of them."""
        _, _, fits = _resolve(
            svc, ModelCapability.THINKING.value, [], pinned=_ws_model("qwen3.5-2b"),
        )

        assert fits.call_args.kwargs["reserve_gb"] == 0.0

    def test_every_other_capability_keeps_the_reserve(self, svc):
        """Waiving it is specific to the step that has no fallback worth
        having — a skills or dp model that doesn't fit still steps down."""
        _, _, fits = _resolve(svc, "reasoning", [_ws_model("gemma4-4b")])

        assert fits.call_args.kwargs["reserve_gb"] is None

    def test_a_reasoning_model_that_still_does_not_fit_falls_back(self, svc):
        """Waived is not unconditional: no free memory is no free memory, and
        the step lands back on the router model rather than OOMing."""
        svc._model.workspace.ranked_models_for.return_value = [_ws_model("qwen3.5-2b")]
        with patch.object(
                ModelService, "_ranked_candidates",
                return_value=[_ws_model("qwen3.5-2b")]), \
             patch.object(ModelService, "_pinned_model", return_value=None), \
             patch.object(ModelService, "_evict_capability_models", return_value=False), \
             patch(
                 "smartloop.llms.llm_factory.LlmFactory.get_llama_cpp_instance"
             ) as factory, \
             patch.object(
                 ModelService, "_find_model_ggufs",
                 return_value=("/models/x.gguf", None)), \
             patch.object(ModelService, "_large_model_fits", return_value=False), \
             patch.object(ModelService, "_fit_shortfall_detail", return_value=None):
            llm = ModelService._get_llm_for_capability(
                svc, ModelCapability.THINKING.value,
            )

        assert llm is None
        assert not factory.called
