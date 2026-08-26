"""Progress events name models the way the model picker names them.

The bootstrap model's GGUF carries ``general.name = "Slp Orchestra Mini"``,
which is the upstream artifact name and appears nowhere in the product.  The
"Selected <model> for document processing" SSE is built from that metadata, so
it has to be mapped back to the short name the user selected models by.
"""

from unittest.mock import MagicMock

import pytest

from smartloop.agents.orchestration.model_router import ModelRouter
from smartloop.api.services.model_service import ModelService
from smartloop.constants import display_model_label


def _llm(name, size=""):
    llm = MagicMock()
    llm.model.metadata = {"general.name": name, "general.size_label": size}
    return llm


@pytest.mark.parametrize("label_fn", [ModelRouter.model_label, ModelService._model_label])
def test_bootstrap_metadata_name_is_shown_as_the_short_name(label_fn):
    assert label_fn(_llm("Slp Orchestra Mini")) == "sl-mini"


@pytest.mark.parametrize("label_fn", [ModelRouter.model_label, ModelService._model_label])
def test_size_label_is_still_appended(label_fn):
    assert label_fn(_llm("Slp Orchestra Mini", "1B")) == "sl-mini (1B)"


@pytest.mark.parametrize("label_fn", [ModelRouter.model_label, ModelService._model_label])
def test_third_party_metadata_name_is_left_alone(label_fn):
    assert label_fn(_llm("Gemma 3n E2B", "2B")) == "Gemma 3n E2B (2B)"


@pytest.mark.parametrize("label_fn", [ModelRouter.model_label, ModelService._model_label])
def test_unreadable_metadata_falls_back_to_the_class_name(label_fn):
    llm = MagicMock()
    type(llm).model = property(lambda self: (_ for _ in ()).throw(RuntimeError()))
    assert label_fn(llm) == type(llm).__name__


def test_display_model_label_passes_through_unknown_and_empty_names():
    assert display_model_label("Qwen2-VL-2B-Instruct") == "Qwen2-VL-2B-Instruct"
    assert display_model_label("") == ""
    assert display_model_label(None) is None


def test_display_model_label_accepts_the_slug_form_too():
    assert display_model_label("slp-orchestra-mini") == "sl-mini"
