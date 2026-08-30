"""One unknown access level must not take down the whole model picker.

The catalog is served by the platform and gains tiers without a client
release — ``internal`` arrived that way and failed ``ModelEntry`` validation,
which fails the ``list[ModelEntry]`` response model for *every* model in the
list, so /v1/models 500'd instead of dropping one entry.
"""

from smartloop.api.routers.models import AccessLevel, ModelEntry


def _entry(level):
    return ModelEntry(name="qwen3.5-2b", model_id="Qwen/Qwen3.5-2B", access_level=level)


def test_internal_is_a_known_tier():
    assert _entry("internal").access_level is AccessLevel.internal


def test_the_public_tiers_still_parse():
    assert _entry("free").access_level is AccessLevel.free
    assert _entry("pro").access_level is AccessLevel.paid


def test_the_bogus_premium_tier_degrades():
    """``premium`` never existed on the platform's ``ModelAccessLevel``, so it
    is not a member here — it takes the unknown-tier path."""
    assert _entry("premium").access_level is None


def test_a_future_tier_degrades_instead_of_failing():
    assert _entry("enterprise-preview").access_level is None


def test_an_absent_level_stays_absent():
    assert ModelEntry(name="x", model_id="y").access_level is None
