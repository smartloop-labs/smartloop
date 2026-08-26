"""A fallback says why the selected model could not run.

A capability model that passes selection can still fail to load — most often
because it does not fit in memory alongside the resident orchestrator.  The
step then answers on the bootstrap model, which used to be reported as a bare
"Selected sl-mini": indistinguishable from the user's selection being ignored.
Memory is the one cause they can act on, so it is named.
"""

from unittest.mock import MagicMock

import pytest

from smartloop.agents.orchestration.model_router import ModelRouter
from smartloop.agents.planner import ModelTier
from smartloop.api.services.model_service import ModelService, _GPU_FAULTED
from smartloop.types import ModelCapability


def _llm(name):
    llm = MagicMock()
    llm.model.metadata = {"general.name": name, "general.size_label": ""}
    return llm


def _router(skip):
    router = ModelRouter(
        orchestrator_llm=_llm("Slp Orchestra Mini"),
        agent_llm=_llm("Slp Orchestra Mini"),
        capability_llm_provider=lambda cap: None,
        capability_available=lambda cap: True,
        capability_skip_reason=lambda: skip,
    )
    router.begin_request(MagicMock())
    return router


def _messages(router):
    return [c.args[2] for c in router._progress_callback.call_args_list]


class TestTheFallbackExplainsItself:
    def test_memory_is_named(self):
        router = _router(("gemma4-e2b", "memory"))

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)

        assert _messages(router)[-1] == (
            "Selected sl-mini — not enough memory for gemma4-e2b"
        )

    def test_a_gpu_fault_is_also_a_memory_message(self):
        router = _router(("gemma4-e2b", "faulted"))

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)

        assert "ran out of memory" in _messages(router)[-1]

    def test_a_missing_download_is_named(self):
        router = _router(("gemma4-e2b", "missing"))

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)

        assert "not downloaded" in _messages(router)[-1]

    def test_the_legacy_name_is_shown_as_the_product_name(self):
        router = _router(("slp-orchestra-mini", "memory"))

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)

        assert "sl-mini" in _messages(router)[-1]
        assert "orchestra" not in _messages(router)[-1].lower()

    def test_no_recorded_reason_leaves_the_message_bare(self):
        router = _router(None)

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)

        assert _messages(router)[-1] == "Selected sl-mini"

    def test_an_unknown_cause_is_not_guessed_at(self):
        router = _router(("gemma4-e2b", "something-new"))

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)

        assert _messages(router)[-1] == "Selected sl-mini"

    def test_a_broken_reporter_does_not_break_the_event(self):
        router = ModelRouter(
            orchestrator_llm=_llm("Slp Orchestra Mini"),
            agent_llm=_llm("Slp Orchestra Mini"),
            capability_llm_provider=lambda cap: None,
            capability_available=lambda cap: True,
            capability_skip_reason=MagicMock(side_effect=RuntimeError("boom")),
        )
        router.begin_request(MagicMock())

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)

        assert _messages(router)[-1] == "Selected sl-mini"

    def test_no_reporter_wired_in_keeps_the_bare_message(self):
        router = ModelRouter(
            orchestrator_llm=_llm("Slp Orchestra Mini"),
            agent_llm=_llm("Slp Orchestra Mini"),
            capability_llm_provider=lambda cap: None,
            capability_available=lambda cap: True,
        )
        router.begin_request(MagicMock())

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)

        assert _messages(router)[-1] == "Selected sl-mini"


class TestTheReasonIsRecordedWhereItHappens:
    def _service(self, *, fits=True, gguf="/models/m.gguf", faulted=False):
        svc = object.__new__(ModelService)
        svc._model = MagicMock(workspace=MagicMock())
        svc._model.workspace.model_dir_for.return_value = "/models"
        svc._capability_skip = None
        svc._find_model_ggufs = MagicMock(return_value=(gguf, None))
        svc._large_model_fits = MagicMock(return_value=fits)
        svc._evict_capability_models = MagicMock(return_value=False)
        svc.get_project_id = MagicMock(return_value="p1")
        if faulted:
            setattr(svc, "_cap_llm_gemma4-e2b", _GPU_FAULTED)
        return svc

    def _model(self):
        m = MagicMock()
        m.model_name = "gemma4-e2b"
        m.model_id = "google/gemma4-e2b"
        return m

    def test_a_fit_failure_records_memory(self):
        svc = self._service(fits=False)

        assert svc._load_workspace_llm(self._model(), "capability 'skills'") is None
        assert svc.last_capability_skip() == ("gemma4-e2b", "memory")

    def test_a_missing_gguf_records_missing(self):
        svc = self._service(gguf=None)

        assert svc._load_workspace_llm(self._model(), "capability 'skills'") is None
        assert svc.last_capability_skip() == ("gemma4-e2b", "missing")

    def test_a_faulted_slot_records_memory(self):
        svc = self._service(faulted=True)

        assert svc._load_workspace_llm(self._model(), "capability 'skills'") is None
        assert svc.last_capability_skip() == ("gemma4-e2b", "memory")

    def test_a_resident_model_clears_a_stale_reason(self):
        svc = self._service()
        svc._capability_skip = ("gemma4-e2b", "memory")
        setattr(svc, "_cap_llm_gemma4-e2b", MagicMock(name="resident"))

        assert svc._load_workspace_llm(self._model(), "capability 'skills'") is not None
        assert svc.last_capability_skip() is None
