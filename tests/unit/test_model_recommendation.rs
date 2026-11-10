//! Port of `tests/unit/test_model_recommendation.py`.
//!
//! The framework, not the client, decides which models to recommend.
//!
//! Measured against free memory: per capability, the largest model at or
//! below half of it; with none that comfortable, the lightest model per
//! capability that fits.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use serial_test::serial;

use smartloop::api::routers::models::{self as router, mark_recommended, ModelEntry};

/// Parameter sizes (billions) of the fake models.
fn sizes() -> HashMap<String, f64> {
    [("tiny", 1.0), ("small", 2.0), ("mid", 4.0), ("big", 8.0), ("huge", 30.0)]
        .into_iter()
        .map(|(n, b)| (n.to_string(), b))
        .collect()
}

fn entry(name: &str, caps: &[&str], accessible: bool) -> ModelEntry {
    let mut e = ModelEntry::new(name, name);
    e.capabilities = caps.iter().map(|c| c.to_string()).collect();
    e.is_accessible = accessible;
    e
}

fn capable(name: &str, caps: &[&str]) -> ModelEntry {
    entry(name, caps, true)
}

/// Run `mark_recommended` with model sizes patched to *sizes*, and
/// return the names it badged.
fn recommended_with(entries: &mut [ModelEntry], free_gb: Option<f64>, sizes: HashMap<String, f64>) -> BTreeSet<String> {
    /// The size override, undone on drop.
    struct Patched;
    impl Drop for Patched {
        fn drop(&mut self) {
            router::set_param_size_override(None);
        }
    }

    router::set_param_size_override(Some(Arc::new(move |name: &str| sizes.get(name).copied())));
    let patched = Patched;
    mark_recommended(entries, free_gb);
    drop(patched);
    entries.iter().filter(|e| e.recommended).map(|e| e.name.clone()).collect()
}

fn recommended(entries: &mut [ModelEntry], free_gb: Option<f64>) -> BTreeSet<String> {
    recommended_with(entries, free_gb, sizes())
}

fn set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|n| n.to_string()).collect()
}

fn sizes_plus(extra: &[(&str, f64)]) -> HashMap<String, f64> {
    let mut s = sizes();
    for (name, b) in extra {
        s.insert(name.to_string(), *b);
    }
    s
}

#[test]
#[serial]
fn test_the_larger_comfortable_model_is_recommended_over_a_smaller_one() {
    let mut entries = vec![capable("small", &["general", "vision"]), capable("mid", &["general", "vision"])];
    // 16GB free: both are comfortable (12.5%, 25%); the 4B answers better.
    assert_eq!(recommended(&mut entries, Some(16.0)), set(&["mid"]));
}

#[test]
#[serial]
fn test_the_largest_comfortable_model_per_capability_is_recommended() {
    let mut entries = vec![
        capable("tiny", &["vision"]),
        capable("small", &["general", "reasoning"]),
        capable("mid", &["general"]),
        capable("big", &["vision", "coding"]),
    ];
    // 16GB free: everything is comfortable (big is exactly 50%).  big covers
    // vision and coding, mid general, small reasoning; tiny adds nothing.
    assert_eq!(recommended(&mut entries, Some(16.0)), set(&["big", "mid", "small"]));
}

#[test]
#[serial]
fn test_less_free_memory_recommends_smaller_models() {
    let mut entries = vec![capable("small", &["general", "vision"]), capable("mid", &["general", "vision"])];
    // Same models, other apps open: 6GB free.  mid (67%) still fits but is no
    // longer comfortable; small (33%) is.
    assert_eq!(recommended(&mut entries, Some(6.0)), set(&["small"]));
}

#[test]
#[serial]
fn test_without_a_comfortable_model_the_lightest_per_capability_is_recommended() {
    let mut entries = vec![
        capable("big", &["general", "dp"]),
        capable("big-2", &["general", "vision"]),
        capable("huge", &["coding"]), // won't fit
    ];
    // 12GB free: big 67%, big-2 75%, huge doesn't fit.
    assert_eq!(
        recommended_with(&mut entries, Some(12.0), sizes_plus(&[("big-2", 9.0)])),
        set(&["big", "big-2"])
    );
}

#[test]
#[serial]
fn test_locked_and_capability_less_models_are_never_recommended() {
    let mut entries = vec![entry("tiny", &["general"], false), capable("small", &[])];
    assert_eq!(recommended(&mut entries, Some(64.0)), set(&[]));
}

#[test]
#[serial]
fn test_unknown_free_memory_recommends_nothing() {
    let mut entries = vec![capable("tiny", &["general"])];
    assert_eq!(recommended(&mut entries, None), set(&[]));
}

#[test]
#[serial]
fn test_the_lightest_reasoning_model_is_recommended_even_when_not_comfortable() {
    let mut entries = vec![
        capable("tiny", &["general", "vision"]),
        capable("big-r", &["general", "reasoning"]),
        capable("huge", &["reasoning"]), // won't fit
    ];
    // 16GB free: tiny is comfortable; big-r (62.5%) is not, but it is the
    // only reasoning model that fits.
    assert_eq!(
        recommended_with(&mut entries, Some(16.0), sizes_plus(&[("big-r", 10.0)])),
        set(&["tiny", "big-r"])
    );
}

#[test]
#[serial]
fn test_a_reasoning_model_wins_a_size_tie() {
    let mut entries = vec![capable("mid", &["general"]), capable("mid-r", &["general", "reasoning"])];
    assert_eq!(
        recommended_with(&mut entries, Some(16.0), sizes_plus(&[("mid-r", 4.0)])),
        set(&["mid-r"])
    );
}

#[test]
#[serial]
fn test_within_a_size_only_the_reasoning_model_is_recommended() {
    let mut entries = vec![
        capable("small", &["general", "reasoning"]),
        capable("mid", &["vision"]),
        capable("mid-r", &["general", "reasoning"]),
    ];
    // 16GB free.  mid would be badged for vision, but mid-r is the same size
    // and can reason, so it takes the badge from mid.
    assert_eq!(
        recommended_with(&mut entries, Some(16.0), sizes_plus(&[("mid-r", 4.0)])),
        set(&["mid-r"])
    );
}

#[test]
#[serial]
fn test_a_capability_dropped_in_a_size_tie_falls_to_another_size() {
    let mut entries = vec![
        capable("small", &["general", "vision"]),
        capable("small-r", &["general", "reasoning"]),
        capable("tiny", &["vision"]),
    ];
    // small-r outranks small at 2B, so vision falls to tiny.
    assert_eq!(
        recommended_with(&mut entries, Some(64.0), sizes_plus(&[("small-r", 2.0)])),
        set(&["small-r", "tiny"])
    );
}

#[test]
#[serial]
fn test_within_a_size_the_most_capable_model_is_recommended() {
    let mut entries = vec![capable("small", &["general"]), capable("small-plus", &["general", "dp", "vision"])];
    assert_eq!(
        recommended_with(&mut entries, Some(64.0), sizes_plus(&[("small-plus", 2.0)])),
        set(&["small-plus"])
    );
}

#[test]
#[serial]
fn test_reasoning_outranks_capability_count_within_a_size() {
    let mut entries = vec![capable("small-plus", &["general", "dp", "vision"]), capable("small-r", &["reasoning"])];
    assert_eq!(
        recommended_with(&mut entries, Some(64.0), sizes_plus(&[("small-plus", 2.0), ("small-r", 2.0)])),
        set(&["small-r"])
    );
}

#[test]
#[serial]
fn test_an_exact_tie_shares_the_badge() {
    let mut entries = vec![capable("small", &["general", "vision"]), capable("small-2", &["general", "dp"])];
    assert_eq!(
        recommended_with(&mut entries, Some(64.0), sizes_plus(&[("small-2", 2.0)])),
        set(&["small", "small-2"])
    );
}
