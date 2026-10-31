//! The capability the final answer is written on.
//!
//! The planner's `coding` pick has to reach the answer, or a coding request is
//! written by whichever model ranks best for `general`.

use smartloop::api::routers::chat::response_capability;
use smartloop::types::ModelCapability;

fn context(text: &str) -> Option<String> {
    Some(text.to_string())
}

#[test]
fn coding_and_document_work_are_answered_on_document_processing() {
    assert_eq!(response_capability(&context("TOOL RESULTS: fn main() {}"), Some("dp"), true), Some(ModelCapability::Dp));
    assert_eq!(response_capability(&None, None, true), Some(ModelCapability::Dp));
    assert_eq!(response_capability(&context(""), Some("general"), true), Some(ModelCapability::Dp));
    // Classified as document processing, with or without context.
    assert_eq!(response_capability(&context("DOCUMENT CONTEXT"), Some("dp"), false), Some(ModelCapability::Dp));
    assert_eq!(response_capability(&None, Some("dp"), false), Some(ModelCapability::Dp));
}

#[test]
fn other_requests_with_context_are_answered_on_general() {
    for classified in [Some("vision"), Some("reasoning"), Some("general"), None] {
        assert_eq!(response_capability(&context("DOCUMENT CONTEXT"), classified, false), Some(ModelCapability::General));
    }
}

#[test]
fn a_bare_chat_stays_on_the_default_model() {
    assert_eq!(response_capability(&None, Some("general"), false), None);
    assert_eq!(response_capability(&context(""), None, false), None);
}
