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
fn a_coding_request_is_answered_on_coding() {
    assert_eq!(response_capability(&context("TOOL RESULTS: fn main() {}"), Some("coding")), Some(ModelCapability::Coding));
}

#[test]
fn a_coding_request_without_context_is_still_answered_on_coding() {
    assert_eq!(response_capability(&None, Some("coding")), Some(ModelCapability::Coding));
    assert_eq!(response_capability(&context(""), Some("coding")), Some(ModelCapability::Coding));
}

#[test]
fn other_picks_with_context_are_answered_on_general() {
    for classified in [Some("dp"), Some("vision"), Some("reasoning"), Some("general"), None] {
        assert_eq!(response_capability(&context("DOCUMENT CONTEXT"), classified), Some(ModelCapability::General));
    }
}

#[test]
fn a_bare_chat_stays_on_the_default_model() {
    assert_eq!(response_capability(&None, Some("dp")), None);
    assert_eq!(response_capability(&context(""), None), None);
}
