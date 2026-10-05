//! Port of `tests/unit/test_model_access_level.py`.
//!
//! One unknown access level must not take down the whole model picker.
//!
//! The catalog is served by the platform and gains tiers without a client
//! release — `internal` arrived that way and failed `ModelEntry` validation,
//! which fails the `list[ModelEntry]` response model for *every* model in the
//! list, so /v1/models 500'd instead of dropping one entry.

use serde_json::json;
use smartloop::api::routers::models::{AccessLevel, ModelEntry};

/// `ModelEntry(name=..., model_id=..., access_level=level)` — built through
/// validation, which is where the unknown-tier tolerance lives.
fn entry(level: &str) -> ModelEntry {
    serde_json::from_value(json!({
        "name": "qwen3.5-2b",
        "model_id": "Qwen/Qwen3.5-2B",
        "access_level": level,
    }))
    .expect("a catalog row with any access level must validate")
}

#[test]
fn test_internal_is_a_known_tier() {
    assert_eq!(entry("internal").access_level, Some(AccessLevel::Internal));
}

#[test]
fn test_the_public_tiers_still_parse() {
    assert_eq!(entry("free").access_level, Some(AccessLevel::Free));
    assert_eq!(entry("pro").access_level, Some(AccessLevel::Paid));
}

/// `premium` never existed on the platform's `ModelAccessLevel`, so it is not
/// a member here — it takes the unknown-tier path.
#[test]
fn test_the_bogus_premium_tier_degrades() {
    assert_eq!(entry("premium").access_level, None);
}

#[test]
fn test_a_future_tier_degrades_instead_of_failing() {
    assert_eq!(entry("enterprise-preview").access_level, None);
}

#[test]
fn test_an_absent_level_stays_absent() {
    assert_eq!(ModelEntry::new("x", "y").access_level, None);
    // And through validation, with the field missing entirely.
    let parsed: ModelEntry = serde_json::from_value(json!({"name": "x", "model_id": "y"})).unwrap();
    assert_eq!(parsed.access_level, None);
}
