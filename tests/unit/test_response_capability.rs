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
fn a_coding_request_is_answered_on_document_processing() {
    assert_eq!(response_capability(&context("TOOL RESULTS: fn main() {}"), true), Some(ModelCapability::Dp));
    assert_eq!(response_capability(&None, true), Some(ModelCapability::Dp));
    assert_eq!(response_capability(&context(""), true), Some(ModelCapability::Dp));
}

#[test]
fn other_requests_with_context_are_answered_on_general() {
    assert_eq!(response_capability(&context("DOCUMENT CONTEXT"), false), Some(ModelCapability::General));
}

#[test]
fn a_bare_chat_stays_on_the_default_model() {
    assert_eq!(response_capability(&None, false), None);
    assert_eq!(response_capability(&context(""), false), None);
}
