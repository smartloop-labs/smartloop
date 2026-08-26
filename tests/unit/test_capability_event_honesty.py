"""Model-selection events must describe what actually happened.

"Loading model for skills…" fires *before* the provider is consulted — that is
the point of it, since the load it precedes is slow.  But on a workspace where
no selected model declares the capability there is nothing to load, so the
event promised a load that never happened and the step then fell back to the
bootstrap model, which was in turn announced as "Selected sl-mini for skills"
— naming a capability nothing had been selected for.
"""

from unittest.mock import MagicMock

import pytest

from smartloop.agents.orchestration.model_router import ModelRouter
from smartloop.agents.planner import ModelTier
from smartloop.types import ModelCapability


def _llm(name):
    llm = MagicMock()
    llm.model.metadata = {"general.name": name, "general.size_label": ""}
    return llm


def _router(*, capability_llm=None, available=True):
    router = ModelRouter(
        orchestrator_llm=_llm("Slp Orchestra Mini"),
        agent_llm=_llm("Slp Orchestra Mini"),
        capability_llm_provider=lambda cap: capability_llm,
        capability_available=lambda cap: available,
    )
    router.begin_request(MagicMock())
    return router


def _messages(router):
    return [c.args[2] for c in router._progress_callback.call_args_list]


def test_no_load_is_announced_when_nothing_provides_the_capability():
    router = _router(capability_llm=None, available=False)

    router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)

    assert not any("Loading model for skills" in m for m in _messages(router))


def test_the_fallback_model_is_still_named():
    """The user should know which model answered, just not a false capability."""
    router = _router(capability_llm=None, available=False)

    router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)

    assert _messages(router) == ["Selected sl-mini"]


def test_the_fallback_does_not_claim_the_capability():
    router = _router(capability_llm=None, available=False)

    router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)

    assert not any("for skills" in m for m in _messages(router))


def test_a_real_capability_model_is_announced_as_before():
    cap_llm = _llm("Gemma 3n E2B")
    router = _router(capability_llm=cap_llm, available=True)

    resolved = router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)

    assert resolved is cap_llm
    assert _messages(router) == [
        "Loading model for skills…",
        "Selected Gemma 3n E2B for skills",
    ]


def test_an_available_capability_that_fails_to_load_still_warns_first():
    """Availability is declared, not guaranteed — a fit failure still falls back."""
    router = _router(capability_llm=None, available=True)

    router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)

    assert _messages(router) == [
        "Loading model for skills…",
        "Selected sl-mini",
    ]


def test_no_predicate_wired_in_keeps_the_previous_behaviour():
    router = ModelRouter(
        orchestrator_llm=_llm("Slp Orchestra Mini"),
        agent_llm=_llm("Slp Orchestra Mini"),
        capability_llm_provider=lambda cap: None,
    )
    router.begin_request(MagicMock())

    router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)

    assert "Loading model for skills…" in _messages(router)


def test_a_broken_predicate_does_not_suppress_the_event():
    router = ModelRouter(
        orchestrator_llm=_llm("Slp Orchestra Mini"),
        agent_llm=_llm("Slp Orchestra Mini"),
        capability_llm_provider=lambda cap: None,
        capability_available=MagicMock(side_effect=RuntimeError("boom")),
    )
    router.begin_request(MagicMock())

    router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)

    assert "Loading model for skills…" in _messages(router)


def test_each_event_is_reported_once_per_request():
    """A fan-out resolves the model per document; the user needs one line."""
    router = _router(capability_llm=None, available=False)

    router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)
    router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)

    assert _messages(router) == ["Selected sl-mini"]


class TestHasCapabilityModel:
    def _service(self, ranked, pinned=None):
        from smartloop.api.services.model_service import ModelService

        workspace = MagicMock()
        workspace.current_project = MagicMock()
        workspace.ranked_models_for.return_value = ranked
        svc = object.__new__(ModelService)
        svc._model = MagicMock(workspace=workspace)
        svc._pinned_model = MagicMock(return_value=pinned)
        return svc

    def test_true_when_a_model_declares_it(self):
        assert self._service([MagicMock()]).has_capability_model("skills") is True

    def test_false_when_none_declares_it(self):
        assert self._service([]).has_capability_model("skills") is False

    def test_a_pinned_model_serves_every_capability(self):
        svc = self._service([], pinned=MagicMock())
        assert svc.has_capability_model("skills") is True

    def test_false_without_a_workspace(self):
        from smartloop.api.services.model_service import ModelService

        svc = object.__new__(ModelService)
        svc._model = None
        assert svc.has_capability_model("skills") is False


# ── the event names the work, not the routing tag ─────────────────────────

class TestPurposeOverridesTheCapabilityLabel:
    """Synthesis routes on the 'skills' capability so a skills-capable model
    writes the answer.  But the user is watching the final answer being
    written, and "Loading Model For Skills…" under "Synthesizing the final
    answer…" reads as a second, unrelated step.
    """

    def test_loading_names_the_purpose(self):
        router = _router(capability_llm=_llm("Gemma 3n E2B"), available=True)

        router.llm_for_tier(
            ModelTier.LARGE, ModelCapability.SKILLS, purpose="the final answer",
        )

        assert _messages(router) == [
            "Loading model for the final answer…",
            "Selected Gemma 3n E2B for the final answer",
        ]

    def test_the_capability_tag_never_reaches_the_user(self):
        router = _router(capability_llm=_llm("Gemma 3n E2B"), available=True)

        router.llm_for_tier(
            ModelTier.LARGE, ModelCapability.SKILLS, purpose="the final answer",
        )

        assert not any("skills" in m.lower() for m in _messages(router))

    def test_the_fallback_still_drops_the_capability_claim(self):
        router = _router(capability_llm=None, available=True)

        router.llm_for_tier(
            ModelTier.LARGE, ModelCapability.SKILLS, purpose="the final answer",
        )

        assert _messages(router) == [
            "Loading model for the final answer…",
            "Selected sl-mini",
        ]

    def test_two_purposes_on_one_capability_each_get_a_line(self):
        """Dedupe is per purpose, not per capability — they are separate jobs."""
        router = _router(capability_llm=_llm("Gemma 3n E2B"), available=True)

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)
        router.llm_for_tier(
            ModelTier.LARGE, ModelCapability.SKILLS, purpose="the final answer",
        )

        assert _messages(router) == [
            "Loading model for skills…",
            "Selected Gemma 3n E2B for skills",
            "Loading model for the final answer…",
            "Selected Gemma 3n E2B for the final answer",
        ]

    def test_the_same_purpose_is_still_reported_once(self):
        router = _router(capability_llm=_llm("Gemma 3n E2B"), available=True)

        for _ in range(2):
            router.llm_for_tier(
                ModelTier.LARGE, ModelCapability.SKILLS, purpose="the final answer",
            )

        assert _messages(router) == [
            "Loading model for the final answer…",
            "Selected Gemma 3n E2B for the final answer",
        ]

    def test_no_purpose_keeps_the_capability_label(self):
        router = _router(capability_llm=_llm("Gemma 3n E2B"), available=True)

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.DP)

        assert _messages(router) == [
            "Loading model for document processing…",
            "Selected Gemma 3n E2B for document processing",
        ]


def test_synthesis_reports_the_final_answer_not_skills():
    """End to end through SkillProcessor.synthesize — the reported case."""
    from smartloop.agents.orchestration.skill_processor import SkillProcessor

    router = _router(capability_llm=_llm("Gemma 3n E2B"), available=True)
    processor = SkillProcessor(router, explorer=MagicMock())

    processor.synthesize("what is this", "step output", "skill body")

    assert not any("skills" in m.lower() for m in _messages(router))
    assert "Loading model for the final answer…" in _messages(router)
