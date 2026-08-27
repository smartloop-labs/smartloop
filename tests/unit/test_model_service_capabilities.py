"""Tests for capability resolution inside ModelService.

The selection sweeps (``get_vision_llm``, ``_detect_large_model``,
``orchestrate``'s available-capability union, and the ``_ranked_candidates``
tie-break) must read the capabilities a model is *enabled* for in the current
project, not the full set its class declares.  Otherwise a capability the user
switched off still steers model choice.
"""

from unittest.mock import MagicMock

import pytest

from smartloop.api.services.model_service import ModelService
from smartloop.types import ModelCapability
from smartloop.workspace.workspace import Project, Workspace, WorkspaceModel


@pytest.fixture
def workspace() -> Workspace:
    """Two 4B models, one of which declares vision on top of general/dp."""
    return Workspace(
        id="ws-caps",
        models=[
            WorkspaceModel(
                model_name="gemma4-e4b",           # general, dp, vision (v4)
                model_id="google/gemma-4-e4b-it",
                downloaded=True,
            ),
            WorkspaceModel(
                model_name="gemma3-4b",            # general, dp (v3)
                model_id="google/gemma-3-4b-it",
                downloaded=True,
            ),
        ],
        projects=[Project(id="p1", project_dir="/tmp/p1", current=True, name="p1")],
    )


@pytest.fixture
def svc(workspace):
    """A ModelService with only the attributes capability resolution touches."""
    s = object.__new__(ModelService)
    s._model = MagicMock()
    s._model.workspace = workspace
    return s


class TestEnabledCapabilities:
    def test_reports_the_supported_set_when_nothing_is_disabled(self, svc, workspace):
        model = workspace.get_model("gemma4-e4b")

        # "thinking" is not a selection target, so it never appears in the
        # enabled set — see ``Workspace._selectable_capabilities``.
        assert svc._enabled_capabilities(model) == [
            c for c in model.effective_capabilities if c != ModelCapability.THINKING
        ]

    def test_drops_what_the_current_project_disabled(self, svc, workspace):
        workspace.set_project_model_capability(
            workspace.current_project, "gemma4-e4b", "vision", False,
        )
        model = workspace.get_model("gemma4-e4b")

        assert ModelCapability.VISION not in svc._enabled_capabilities(model)
        # …and the model still supports it, for the client to render.
        assert ModelCapability.VISION in model.effective_capabilities

    def test_falls_back_to_the_supported_set_without_a_workspace(self, workspace):
        """Nothing to resolve against — better the full set than none at all."""
        s = object.__new__(ModelService)
        s._model = None
        model = workspace.get_model("gemma4-e4b")

        assert s._enabled_capabilities(model) == model.effective_capabilities


class TestRankedCandidatesTieBreak:
    """Overlap is scored on enabled capabilities, so a switched-off one loses.

    Both models are 4B, so size can't separate them and the overlap score
    decides — ahead of the model version, which would otherwise put the newer
    ``gemma4-e4b`` first.
    """

    def _order(self, svc, workspace, ref_caps):
        return [
            m.model_name for m in svc._ranked_candidates(
                list(workspace.models), ref_caps=ref_caps,
            )
        ]

    def test_equal_overlap_falls_through_to_the_model_version(self, svc, workspace):
        assert self._order(svc, workspace, {ModelCapability.DP}) == [
            "gemma4-e4b", "gemma3-4b",
        ]

    def test_a_disabled_capability_loses_the_tie_break(self, svc, workspace):
        workspace.set_project_model_capability(
            workspace.current_project, "gemma4-e4b", "dp", False,
        )

        # gemma3-4b now has the only dp overlap, which outweighs being a
        # version behind.
        assert self._order(svc, workspace, {ModelCapability.DP}) == [
            "gemma3-4b", "gemma4-e4b",
        ]

    def test_a_plain_string_capability_scores_the_same_as_the_enum(self, svc, workspace):
        """``_get_llm_for_capability`` passes the planner's raw string through.

        Its ``ref_caps`` is ``{capability}`` — a plain string — where
        ``get_vision_llm`` passes a ``ModelCapability`` member, so both spellings
        have to score the same overlap or the tie-break would silently collapse
        to the model version for one of the two callers.
        """
        workspace.set_project_model_capability(
            workspace.current_project, "gemma4-e4b", "dp", False,
        )

        assert self._order(svc, workspace, {"dp"}) == ["gemma3-4b", "gemma4-e4b"]


class TestUnloadDisabledCapabilitySlots:
    """Only the slot that lost its capability is freed.

    ``gemma4-e2b`` declares general/skills/dp/vision, so it is the case where
    a user switches vision off but still wants the model for skills.  Its
    vision instance and its text instance are two different llama.cpp contexts
    — the vision one is loaded with the mmproj projector — so evicting one must
    not take the other with it.
    """

    @pytest.fixture
    def svc(self):
        import threading

        ws = Workspace(
            id="ws-unload",
            models=[
                WorkspaceModel(
                    model_name="gemma4-e2b",   # general, skills, dp, vision
                    model_id="google/gemma-4-e2b-it",
                    downloaded=True,
                ),
            ],
            projects=[Project(id="p1", project_dir="/tmp/p1", current=True, name="p1")],
        )
        s = object.__new__(ModelService)
        s._model = MagicMock()
        s._model.workspace = ws
        s._inference_lock = threading.RLock()
        s._inference_state_lock = threading.RLock()
        s._inference_active = 0
        s._last_inference_at = 0.0
        s._free_llm = MagicMock()
        s._cleanup_gpu_cache = MagicMock()
        # Both slots resident, backed by the same model under different names.
        s._vision_llm = MagicMock(name="vision-instance")
        s._vision_model_name = "gemma4-e2b"
        s._vision_unavailable = False
        s._large_llm = MagicMock(name="text-instance")
        s._large_model_name = "gemma4-e2b"
        s._large_model_attempted = True
        return s

    def _disable(self, svc, capability):
        ws = svc._model.workspace
        ws.set_project_model_capability(
            ws.current_project, "gemma4-e2b", capability, False,
        )

    def test_vision_off_frees_vision_and_keeps_the_text_instance(self, svc):
        """The scenario: vision off, still wanted for skills."""
        text_instance = svc._large_llm
        self._disable(svc, "vision")

        assert svc.unload_disabled_capability_slots("gemma4-e2b") is True

        assert svc._vision_llm is None
        assert svc._vision_model_name is None
        # Skills still resolves to this model, so its text context stays hot.
        assert svc._large_llm is text_instance
        svc._free_llm.assert_called_once()

    def test_skills_off_alone_frees_nothing(self, svc):
        """general/dp still make it a valid large-model pick, and vision is on."""
        vision, text = svc._vision_llm, svc._large_llm
        self._disable(svc, "skills")

        assert svc.unload_disabled_capability_slots("gemma4-e2b") is False

        assert svc._vision_llm is vision
        assert svc._large_llm is text
        svc._free_llm.assert_not_called()

    def test_the_last_text_capability_going_frees_the_large_model(self, svc):
        vision = svc._vision_llm
        for cap in ("skills", "dp", "general"):
            self._disable(svc, cap)

        assert svc.unload_disabled_capability_slots("gemma4-e2b") is True

        assert svc._large_llm is None
        assert svc._large_model_attempted is False
        # Vision is the one capability left, so that instance survives.
        assert svc._vision_llm is vision

    def test_the_last_text_capability_going_frees_the_cached_slot(self, svc):
        """``_cap_llm_<name>`` is keyed by model, not capability.

        The pinned-model path hands a sole selected model straight to
        ``_load_workspace_llm``, which returns this cache without consulting the
        project gate — so once no text capability is left it has to go, or the
        model keeps answering from cache for work it can no longer be picked for.
        """
        cached = MagicMock(name="cap-slot-instance")
        setattr(svc, "_cap_llm_gemma4-e2b", cached)
        for cap in ("skills", "dp", "general"):
            self._disable(svc, cap)

        assert svc.unload_disabled_capability_slots("gemma4-e2b") is True

        assert getattr(svc, "_cap_llm_gemma4-e2b") is None

    def test_one_text_capability_going_keeps_the_cached_slot(self, svc):
        """The slot is text-only and undifferentiated, so 'general' still owns it."""
        cached = MagicMock(name="cap-slot-instance")
        setattr(svc, "_cap_llm_gemma4-e2b", cached)
        self._disable(svc, "dp")

        assert svc.unload_disabled_capability_slots("gemma4-e2b") is False

        assert getattr(svc, "_cap_llm_gemma4-e2b") is cached

    def test_a_gpu_faulted_cached_slot_is_left_in_place(self, svc):
        """The sentinel is not an instance to free — see _evict_capability_models."""
        from smartloop.api.services.model_service import _GPU_FAULTED

        setattr(svc, "_cap_llm_gemma4-e2b", _GPU_FAULTED)
        for cap in ("skills", "dp", "general"):
            self._disable(svc, cap)

        svc.unload_disabled_capability_slots("gemma4-e2b")

        assert getattr(svc, "_cap_llm_gemma4-e2b") is _GPU_FAULTED

    def test_another_model_in_the_slots_is_untouched(self, svc):
        svc._vision_model_name = "qwen2-vl-2b"
        svc._large_model_name = "qwen2-vl-2b"
        vision, text = svc._vision_llm, svc._large_llm
        self._disable(svc, "vision")

        assert svc.unload_disabled_capability_slots("gemma4-e2b") is False

        assert svc._vision_llm is vision
        assert svc._large_llm is text

    def test_nothing_disabled_frees_nothing(self, svc):
        assert svc.unload_disabled_capability_slots("gemma4-e2b") is False
        svc._free_llm.assert_not_called()

    def test_no_workspace_is_a_no_op(self, svc):
        svc._model = None

        assert svc.unload_disabled_capability_slots("gemma4-e2b") is False
        svc._free_llm.assert_not_called()


class TestPinnedModelHonoursTheProjectGate:
    """A sole selected model is pinned only to capabilities still switched on.

    ``_pinned_model`` is the one selection path that skips capability ranking
    altogether, so without this it would be the one place a per-project toggle
    is silently ignored — and it gates six call sites, including
    ``_get_llm_for_capability`` and ``has_capability_model``.
    """

    @pytest.fixture
    def svc(self):
        """One selected specialist: vision + skills, no 'general'."""
        ws = Workspace(
            id="ws-pin",
            models=[
                WorkspaceModel(
                    model_name="gemma4-e2b",   # general, skills, dp, vision
                    model_id="google/gemma-4-e2b-it",
                    downloaded=True,
                ),
            ],
            projects=[Project(id="p1", project_dir="/tmp/p1", current=True, name="p1")],
        )
        s = object.__new__(ModelService)
        s._model = MagicMock()
        s._model.workspace = ws
        s._model_name = "sl-mini"
        return s

    def _disable(self, svc, capability):
        ws = svc._model.workspace
        ws.set_project_model_capability(
            ws.current_project, "gemma4-e2b", capability, False,
        )

    def test_a_general_capable_selection_is_pinned_to_everything(self, svc):
        assert svc._pinned_model("vision").model_name == "gemma4-e2b"

    def test_a_specialist_is_not_pinned_to_a_switched_off_capability(self, svc):
        """The gap ``dev``'s pinned-model short-circuit opened.

        With 'general' off the model is pinned per capability, so switching
        'vision' off has to drop it from the image step rather than leaving the
        pin to read the full declared set.
        """
        self._disable(svc, "general")
        assert svc._pinned_model("vision").model_name == "gemma4-e2b"

        self._disable(svc, "vision")

        assert svc._pinned_model("vision") is None

    def test_a_specialist_keeps_the_capabilities_still_on(self, svc):
        self._disable(svc, "general")
        self._disable(svc, "vision")

        assert svc._pinned_model("skills").model_name == "gemma4-e2b"
