//! Port of `tests/unit/test_keyword_coverage_boost.py`.
//!
//! Tests for the shared keyword-coverage boost used by the embedding stores.
//!
//! Dense embeddings cluster look-alike documents together, so an
//! entity-specific query ("Adam Hart") can pull in similar documents ("Aaron
//! Hawkins").  The boost reduces a document's distance by the number of
//! distinct query keywords it contains so the more complete match ranks first.

use smartloop::db::store::{keyword_coverage_boost, MAX_TERM_BOOST, PER_TERM_BOOST};

/// A document matching two query terms is boosted more than one.
#[test]
fn test_boost_scales_with_number_of_matched_keywords() {
    let full = keyword_coverage_boost(
        "invoices Adam Hart",
        Some("invoice_Adam Hart_16384.pdf"),
        Some("invoice total 540"),
    );
    let partial = keyword_coverage_boost(
        "invoices Adam Hart",
        Some("invoice_Adam Smith_99.pdf"),
        Some("invoice total 10"),
    );
    assert_eq!(full, 2.0 * PER_TERM_BOOST);
    assert_eq!(partial, PER_TERM_BOOST);
    assert!(full > partial);
}

/// A semantically similar but unnamed document is not boosted.
#[test]
fn test_lookalike_with_no_matching_keywords_gets_no_boost() {
    let boost = keyword_coverage_boost(
        "invoices Adam Hart",
        Some("invoice_Aaron Hawkins_36651.pdf"),
        Some("invoice total 90"),
    );
    assert_eq!(boost, 0.0);
}

/// Many matching terms cannot zero out the distance.
#[test]
fn test_boost_is_capped() {
    let boost = keyword_coverage_boost(
        "alpha beta gamma delta epsilon zeta eta",
        Some("alpha beta gamma delta epsilon zeta eta.pdf"),
        Some("alpha beta gamma delta epsilon zeta eta"),
    );
    assert_eq!(boost, MAX_TERM_BOOST);
}

/// Repeated terms count once; tokens <= 2 chars are ignored.
#[test]
fn test_distinct_terms_only_short_tokens_ignored() {
    // "of" is too short to count; "hart" appears twice but counts once.
    let boost = keyword_coverage_boost("hart hart of", Some("Hart report.pdf"), Some("hart hart hart"));
    assert_eq!(boost, PER_TERM_BOOST);
}

#[test]
fn test_empty_query_returns_zero() {
    assert_eq!(keyword_coverage_boost("", Some("anything.pdf"), Some("content")), 0.0);
}
