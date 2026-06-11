"""Tests for the shared keyword-coverage boost used by the embedding stores.

Dense embeddings cluster look-alike documents together, so an entity-specific
query ("Adam Hart") can pull in similar documents ("Aaron Hawkins").  The boost
reduces a document's distance by the number of distinct query keywords it
contains so the more complete match ranks first.
"""
from smartloop.db.store import (
    keyword_coverage_boost,
    _PER_TERM_BOOST,
    _MAX_TERM_BOOST,
)


def test_boost_scales_with_number_of_matched_keywords():
    """A document matching two query terms is boosted more than one."""
    full = keyword_coverage_boost(
        "invoices Adam Hart", "invoice_Adam Hart_16384.pdf", "invoice total 540",
    )
    partial = keyword_coverage_boost(
        "invoices Adam Hart", "invoice_Adam Smith_99.pdf", "invoice total 10",
    )
    assert full == 2 * _PER_TERM_BOOST
    assert partial == _PER_TERM_BOOST
    assert full > partial


def test_lookalike_with_no_matching_keywords_gets_no_boost():
    """A semantically similar but unnamed document is not boosted."""
    boost = keyword_coverage_boost(
        "invoices Adam Hart", "invoice_Aaron Hawkins_36651.pdf", "invoice total 90",
    )
    assert boost == 0.0


def test_boost_is_capped():
    """Many matching terms cannot zero out the distance."""
    boost = keyword_coverage_boost(
        "alpha beta gamma delta epsilon zeta eta",
        "alpha beta gamma delta epsilon zeta eta.pdf",
        "alpha beta gamma delta epsilon zeta eta",
    )
    assert boost == _MAX_TERM_BOOST


def test_distinct_terms_only_short_tokens_ignored():
    """Repeated terms count once; tokens <= 2 chars are ignored."""
    # "of" is too short to count; "hart" appears twice but counts once.
    boost = keyword_coverage_boost(
        "hart hart of", "Hart report.pdf", "hart hart hart",
    )
    assert boost == _PER_TERM_BOOST


def test_empty_query_returns_zero():
    assert keyword_coverage_boost("", "anything.pdf", "content") == 0.0
