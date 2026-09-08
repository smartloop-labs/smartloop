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

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.REASONING)

        assert _messages(router)[-1] == (
            "Selected sl-mini — not enough memory for gemma4-e2b"
        )

    def test_a_gpu_fault_is_also_a_memory_message(self):
        router = _router(("gemma4-e2b", "faulted"))

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.REASONING)

        assert "ran out of memory" in _messages(router)[-1]

    def test_a_missing_download_is_named(self):
        router = _router(("gemma4-e2b", "missing"))

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.REASONING)

        assert "not downloaded" in _messages(router)[-1]

    def test_the_legacy_name_is_shown_as_the_product_name(self):
        router = _router(("slp-orchestra-mini", "memory"))

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.REASONING)

        assert "sl-mini" in _messages(router)[-1]
        assert "orchestra" not in _messages(router)[-1].lower()

    def test_no_recorded_reason_leaves_the_message_bare(self):
        router = _router(None)

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.REASONING)

        assert _messages(router)[-1] == "Selected sl-mini"

    def test_an_unknown_cause_is_not_guessed_at(self):
        router = _router(("gemma4-e2b", "something-new"))

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.REASONING)

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

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.REASONING)

        assert _messages(router)[-1] == "Selected sl-mini"

    def test_no_reporter_wired_in_keeps_the_bare_message(self):
        router = ModelRouter(
            orchestrator_llm=_llm("Slp Orchestra Mini"),
            agent_llm=_llm("Slp Orchestra Mini"),
            capability_llm_provider=lambda cap: None,
            capability_available=lambda cap: True,
        )
        router.begin_request(MagicMock())

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.REASONING)

        assert _messages(router)[-1] == "Selected sl-mini"


class TestTheReasonIsRecordedWhereItHappens:
    def _service(self, *, fits=True, gguf="/models/m.gguf", faulted=False):
        svc = object.__new__(ModelService)
        svc._model = MagicMock(workspace=MagicMock())
        svc._model.workspace.model_dir_for.return_value = "/models"
        svc._capability_skip = None
        svc._last_fit_shortfall = None
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

        assert svc._load_workspace_llm(self._model(), "capability 'reasoning'") is None
        assert svc.last_capability_skip() == ("gemma4-e2b", "memory", None)

    def test_a_fit_failure_carries_the_measured_shortfall(self):
        svc = self._service(fits=False)
        svc._last_fit_shortfall = (4.9, 4.4)

        assert svc._load_workspace_llm(self._model(), "capability 'reasoning'") is None
        assert svc.last_capability_skip() == (
            "gemma4-e2b", "memory", "needs ~4.9GB, 4.4GB free",
        )

    def test_a_missing_gguf_records_missing(self):
        svc = self._service(gguf=None)

        assert svc._load_workspace_llm(self._model(), "capability 'reasoning'") is None
        assert svc.last_capability_skip() == ("gemma4-e2b", "missing", None)

    def test_a_faulted_slot_records_memory(self):
        """A runtime OOM has no measurement to report — only the cause."""
        svc = self._service(faulted=True)

        assert svc._load_workspace_llm(self._model(), "capability 'reasoning'") is None
        assert svc.last_capability_skip() == ("gemma4-e2b", "memory", None)

    def test_a_resident_model_clears_a_stale_reason(self):
        svc = self._service()
        svc._capability_skip = ("gemma4-e2b", "memory")
        setattr(svc, "_cap_llm_gemma4-e2b", MagicMock(name="resident"))

        assert svc._load_workspace_llm(self._model(), "capability 'reasoning'") is not None
        assert svc.last_capability_skip() is None


# ── the figures travel to the user, not just the log ──────────────────────

class TestTheShortfallIsReported:
    """"Not enough memory" does not say how much to free.  The numbers do."""

    def test_the_figures_are_appended(self):
        router = _router(("gemma4-e2b", "memory", "needs ~4.9GB, 4.4GB free"))

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.REASONING)

        assert _messages(router)[-1] == (
            "Selected sl-mini — not enough memory for gemma4-e2b "
            "(needs ~4.9GB, 4.4GB free)"
        )

    def test_a_cause_with_no_figures_reads_as_before(self):
        router = _router(("gemma4-e2b", "missing", None))

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.REASONING)

        assert _messages(router)[-1] == (
            "Selected sl-mini — gemma4-e2b is not downloaded"
        )

    def test_a_two_element_record_is_still_accepted(self):
        router = _router(("gemma4-e2b", "memory"))

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.REASONING)

        assert _messages(router)[-1] == (
            "Selected sl-mini — not enough memory for gemma4-e2b"
        )

    def test_a_malformed_record_is_ignored(self):
        router = _router(("gemma4-e2b",))

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.REASONING)

        assert _messages(router)[-1] == "Selected sl-mini"


class TestTheShortfallIsMeasuredWhereItHappens:
    def _service(self, free=(4.4, 24.0), weights_gb=3.2):
        svc = object.__new__(ModelService)
        svc._device_free_total_gb = MagicMock(return_value=free)
        svc._last_fit_shortfall = None
        return svc, weights_gb

    def test_the_reserve_is_folded_into_the_requirement(self):
        """Quoting the footprint alone shows a number below the free memory."""
        from unittest.mock import patch

        svc, weights = self._service()
        with patch("os.path.getsize", return_value=int(weights * 1024**3)):
            assert svc._large_model_fits("/m.gguf", 0, "gemma4-e2b") is False

        # 3.2 weights + 0.7 overhead + 1.0 reserve = 4.9
        assert svc._fit_shortfall_detail() == "needs ~4.9GB, 4.4GB free"

    def test_a_successful_fit_clears_the_figures(self):
        from unittest.mock import patch

        svc, _ = self._service(free=(20.0, 24.0))
        svc._last_fit_shortfall = (4.9, 4.4)
        with patch("os.path.getsize", return_value=int(1.0 * 1024**3)):
            assert svc._large_model_fits("/m.gguf", 0, "gemma4-e2b") is True

        assert svc._fit_shortfall_detail() is None

    def test_no_measurement_yields_no_detail(self):
        svc, _ = self._service()
        assert svc._fit_shortfall_detail() is None
