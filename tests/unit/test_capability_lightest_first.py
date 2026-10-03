"""A routed step runs on the lightest model that declares its capability.

Every candidate can do the step, so the smallest is enough — and it loads
sooner, decodes faster and leaves the most room for the next step.  A larger
capable model is used only when the smaller ones cannot be loaded.
"""
from unittest.mock import MagicMock, patch

from smartloop.api.services.model_service import ModelService


def _ws_model(name, size, version=1, caps=("reasoning",)):
    m = MagicMock(name=f"ws_model:{name}")
    m.model_name = name
    m._rank = (size, version)
    m.effective_capabilities = list(caps)
    return m


def _svc(ranked):
    s = object.__new__(ModelService)
    s._model = MagicMock()
    s._model.workspace.current_project = None
    s._model.workspace.ranked_models_for.return_value = ranked
    return s


def _walk(ranked, loads):
    """The candidates tried, in order, and what was returned."""
    tried = []

    def load(self, ws_model, reason, reserve_gb=None):
        tried.append(ws_model.model_name)
        return loads.get(ws_model.model_name)

    svc = _svc(ranked)
    with patch.object(ModelService, "_remote_device", return_value=None), \
         patch.object(ModelService, "_pinned_model", return_value=None), \
         patch.object(ModelService, "_load_workspace_llm", load):
        llm = ModelService._get_llm_for_capability(svc, "reasoning")
    return tried, llm


class TestLightestFirst:
    def test_the_smallest_capable_model_is_used(self):
        small = MagicMock(name="small_llm")
        tried, llm = _walk(
            [_ws_model("big", 8), _ws_model("mid", 4), _ws_model("small", 2)],
            {"big": MagicMock(), "mid": MagicMock(), "small": small},
        )
        assert tried == ["small"]
        assert llm is small

    def test_a_larger_one_is_used_when_the_smaller_cannot_load(self):
        mid = MagicMock(name="mid_llm")
        tried, llm = _walk(
            [_ws_model("big", 8), _ws_model("mid", 4), _ws_model("small", 2)],
            {"mid": mid},
        )
        assert tried == ["small", "mid"]
        assert llm is mid

    def test_size_ties_go_to_broader_coverage_then_the_newer_version(self):
        ordered = ModelService._lightest_first(
            [
                _ws_model("old", 4, version=1),
                _ws_model("narrow-new", 4, version=3),
                _ws_model("broad", 4, version=2, caps=("reasoning", "general")),
                _ws_model("new", 4, version=2),
            ],
            ref_caps={"reasoning", "general"},
        )
        assert [m.model_name for m in ordered] == ["broad", "narrow-new", "new", "old"]
