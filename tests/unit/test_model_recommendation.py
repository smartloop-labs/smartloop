"""The framework, not the client, decides which models to recommend.

Every capable model at Low memory pressure is badged; with none at Low, the
lightest model per capability that fits.
"""
from unittest.mock import patch

from smartloop.api.routers import models as router
from smartloop.api.routers.models import ModelEntry, _mark_recommended

SIZES = {"tiny": 1, "small": 2, "mid": 4, "big": 8, "huge": 30}


def _entry(name, caps, accessible=True):
    return ModelEntry(name=name, model_id=name, capabilities=list(caps),
                      is_accessible=accessible)


def _recommended(entries, total_gb):
    class _Cls:
        def __init__(self, b):
            self._b = b

        def param_size_in_b(self):
            return self._b

    fake = {n: _Cls(b) for n, b in SIZES.items()}
    with patch.object(router, "SUPPORTED_MODELS", fake):
        _mark_recommended(entries, total_gb)
    return {e.name for e in entries if e.recommended}


def test_every_low_pressure_capable_model_is_recommended():
    entries = [
        _entry("tiny", ["vision"]),
        _entry("small", ["general", "reasoning"]),
        _entry("mid", ["general"]),
    ]
    # 32GB: tiny and small are <=10% (Low); mid is 12.5% (Medium).
    assert _recommended(entries, 32) == {"tiny", "small"}


def test_without_a_low_model_the_lightest_per_capability_is_recommended():
    entries = [
        _entry("small", ["general", "dp"]),
        _entry("mid", ["general", "vision"]),
        _entry("big", ["vision", "coding"]),
        _entry("huge", ["coding"]),  # won't fit
    ]
    # 16GB: small 12.5%, mid 25%, big 50%, huge doesn't fit.
    assert _recommended(entries, 16) == {"small", "mid", "big"}


def test_locked_and_capability_less_models_are_never_recommended():
    entries = [
        _entry("tiny", ["general"], accessible=False),
        _entry("small", []),
    ]
    assert _recommended(entries, 64) == set()


def test_unknown_ram_recommends_nothing():
    entries = [_entry("tiny", ["general"])]
    assert _recommended(entries, None) == set()


def test_the_lightest_reasoning_model_is_recommended_even_above_low():
    entries = [
        _entry("tiny", ["general", "vision"]),
        _entry("big", ["general", "reasoning"]),
        _entry("huge", ["reasoning"]),  # won't fit
    ]
    # 32GB: tiny is Low; big (25%) is not, but it is the only reasoning
    # model that fits.
    assert _recommended(entries, 32) == {"tiny", "big"}


def test_a_reasoning_model_wins_a_size_tie_in_the_fallback():
    entries = [
        _entry("mid", ["general"]),
        _entry("mid-r", ["general", "reasoning"]),
    ]
    SIZES["mid-r"] = 4
    try:
        # 16GB: both are 25%, the reasoning one covers 'general' first.
        assert _recommended(entries, 16) == {"mid-r"}
    finally:
        del SIZES["mid-r"]


def test_within_a_size_only_the_reasoning_model_is_recommended():
    entries = [
        _entry("small", ["general", "reasoning"]),
        _entry("mid", ["vision"]),
        _entry("mid-r", ["general", "reasoning"]),
    ]
    SIZES["mid-r"] = 4
    try:
        # 16GB, nothing at Low.  mid would be badged for vision, but mid-r is
        # the same size and can reason, so it takes the badge from mid.
        assert _recommended(entries, 16) == {"small", "mid-r"}
    finally:
        del SIZES["mid-r"]


def test_the_same_holds_at_low_pressure():
    entries = [
        _entry("small", ["general", "vision"]),
        _entry("small-r", ["general", "reasoning"]),
        _entry("tiny", ["vision"]),
    ]
    SIZES["small-r"] = 2
    try:
        # 64GB: all three are Low; small and small-r share a size.
        assert _recommended(entries, 64) == {"small-r", "tiny"}
    finally:
        del SIZES["small-r"]
