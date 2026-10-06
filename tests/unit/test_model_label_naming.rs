//! Port of `tests/unit/test_model_label_naming.py`.
//!
//! Progress events name models the way the model picker names them.
//!
//! The bootstrap model's GGUF carries `general.name = "Slp Orchestra Mini"`,
//! which is the upstream artifact name and appears nowhere in the product.
//! The "Selected <model> for document processing" SSE is built from that
//! metadata, so it has to be mapped back to the short name the user selected
//! models by.

use smartloop::agents::orchestration::model_router::ModelRouter;
use smartloop::api::services::model_service::ModelService;
use smartloop::constants::display_model_label;
use smartloop::llms::BaseLLM;

use crate::shared::orch_steps::MetadataLlm;

fn llm(name: &str, size: &str) -> MetadataLlm {
    MetadataLlm::named(name, size)
}

/// `@pytest.mark.parametrize("label_fn", [ModelRouter.model_label,
/// ModelService._model_label])`.
type LabelFn = fn(&dyn BaseLLM) -> String;

const ROUTER: LabelFn = ModelRouter::model_label;
const SERVICE: LabelFn = ModelService::model_label;

fn bootstrap_metadata_name_is_shown_as_the_short_name(label_fn: LabelFn) {
    assert_eq!(label_fn(&llm("Slp Orchestra Mini", "")), "sl-mini");
}

#[test]
fn test_bootstrap_metadata_name_is_shown_as_the_short_name_model_router() {
    bootstrap_metadata_name_is_shown_as_the_short_name(ROUTER);
}

#[test]
fn test_bootstrap_metadata_name_is_shown_as_the_short_name_model_service() {
    bootstrap_metadata_name_is_shown_as_the_short_name(SERVICE);
}

fn size_label_is_still_appended(label_fn: LabelFn) {
    assert_eq!(label_fn(&llm("Slp Orchestra Mini", "1B")), "sl-mini (1B)");
}

#[test]
fn test_size_label_is_still_appended_model_router() {
    size_label_is_still_appended(ROUTER);
}

#[test]
fn test_size_label_is_still_appended_model_service() {
    size_label_is_still_appended(SERVICE);
}

fn third_party_metadata_name_is_left_alone(label_fn: LabelFn) {
    assert_eq!(label_fn(&llm("Gemma 3n E2B", "2B")), "Gemma 3n E2B (2B)");
}

#[test]
fn test_third_party_metadata_name_is_left_alone_model_router() {
    third_party_metadata_name_is_left_alone(ROUTER);
}

#[test]
fn test_third_party_metadata_name_is_left_alone_model_service() {
    third_party_metadata_name_is_left_alone(SERVICE);
}

fn unreadable_metadata_falls_back_to_the_class_name(label_fn: LabelFn) {
    let llm = MetadataLlm::unreadable();
    assert_eq!(label_fn(&llm), llm.class_name());
    assert_eq!(label_fn(&llm), "MetadataLlm");
}

#[test]
fn test_unreadable_metadata_falls_back_to_the_class_name_model_router() {
    unreadable_metadata_falls_back_to_the_class_name(ROUTER);
}

#[test]
fn test_unreadable_metadata_falls_back_to_the_class_name_model_service() {
    unreadable_metadata_falls_back_to_the_class_name(SERVICE);
}

#[test]
fn test_display_model_label_passes_through_unknown_and_empty_names() {
    assert_eq!(display_model_label("Qwen2-VL-2B-Instruct"), "Qwen2-VL-2B-Instruct");
    assert_eq!(display_model_label(""), "");
    // Python's `display_model_label(None) is None`: the Rust function takes a
    // `&str`, so "no name" is modelled by the caller's `Option` and passes
    // through untouched.
    assert_eq!(None::<&str>.map(display_model_label), None);
}

#[test]
fn test_display_model_label_accepts_the_slug_form_too() {
    assert_eq!(display_model_label("slp-orchestra-mini"), "sl-mini");
}
