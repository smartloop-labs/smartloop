"""A capability step must reuse a resident model, not load a second copy.

The orchestrator model declares GENERAL, so a step routed to that capability
used to walk straight past the resident instance and build its own llama.cpp
context over the *same* GGUF — the weights and the KV cache of a model already
sitting in memory, duplicated.  On a shared-memory GPU (an Intel iGPU with the
embedder also offloaded) that second copy is what tips the device over its
allocation limit:

    ggml_vulkan: Failed to allocate pinned memory
    (Requested buffer size exceeds device buffer size limit)

``_get_llm_for_capability`` now returns the resident instance for any candidate
that is already loaded — the orchestrator model or the on-demand large one.
"""

from unittest.mock import MagicMock, patch

import pytest

from smartloop.api.services.model_service import ModelService


def _ws_model(name: str):
    m = MagicMock(name=f"ws_model:{name}")
    m.model_name = name
    m.model_id = name
    return m


@pytest.fixture
def svc():
    """A ModelService carrying only what the capability path reads."""
    s = object.__new__(ModelService)
    s._llm = MagicMock(name="resident_orchestrator")
    s._model_name = "sl-mini"
    s._large_llm = None
    s._large_model_name = None
    s._model = MagicMock()
    s._model.workspace.current_project = None
    return s


def _resolve(svc, capability, ranked):
    """Run the real _get_llm_for_capability over *ranked* candidates,
    returning ``(llm, loaded)`` — ``loaded`` is True if a GGUF was opened."""
    svc._model.workspace.ranked_models_for.return_value = ranked
    with patch.object(ModelService, "_ranked_candidates", return_value=ranked), \
         patch(
             "smartloop.llms.llm_factory.LlmFactory.get_llama_cpp_instance"
         ) as factory, \
         patch.object(
             ModelService, "_find_model_ggufs", return_value=("/models/x.gguf", None)
         ), \
         patch.object(ModelService, "_large_model_fits", return_value=True):
        llm = ModelService._get_llm_for_capability(svc, capability)
    return llm, factory.called


class TestResidentModelsAreReused:
    def test_the_orchestrator_model_is_not_loaded_twice(self, svc):
        """The regression: capability GENERAL resolves to the model already
        answering as the router, so it must hand back that instance."""
        llm, loaded = _resolve(svc, "general", [_ws_model("sl-mini")])

        assert llm is svc._llm
        assert not loaded

    def test_a_legacy_name_still_matches_the_resident_model(self, svc):
        """A workspace written before the rename records the same model as
        ``slp-orchestra-mini``; it is not a different model."""
        llm, loaded = _resolve(svc, "general", [_ws_model("slp-orchestra-mini")])

        assert llm is svc._llm
        assert not loaded

    def test_the_resident_large_model_is_reused_too(self, svc):
        """The on-demand large model is resident outside the capability cache,
        so it is the same trap."""
        svc._large_llm = MagicMock(name="resident_large")
        svc._large_model_name = "sl-4b"

        llm, loaded = _resolve(svc, "skills", [_ws_model("sl-4b")])

        assert llm is svc._large_llm
        assert not loaded

    def test_a_model_that_is_not_resident_is_still_loaded(self, svc):
        """Reuse must not turn into a refusal to load anything: a genuinely
        different capability model is loaded and cached as before."""
        llm, loaded = _resolve(svc, "coding", [_ws_model("coder-7b")])

        assert loaded
        assert llm is not svc._llm
        assert getattr(svc, "_cap_llm_coder-7b") is llm
