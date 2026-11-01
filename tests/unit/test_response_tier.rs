//! The tier the final answer is written on: the answer tier for programming
//! work and for anything that gathered context (the router steps up to the
//! largest model that fits), the default model for a bare chat.

use smartloop::api::routers::chat::response_tier;
use smartloop::types::ANSWER_TIER;

fn context(text: &str) -> Option<String> {
    Some(text.to_string())
}

#[test]
fn test_coding_work_answers_on_the_answer_tier_with_or_without_context() {
    assert_eq!(response_tier(&context("TOOL RESULTS: fn main() {}"), true), Some(ANSWER_TIER));
    assert_eq!(response_tier(&None, true), Some(ANSWER_TIER));
    assert_eq!(response_tier(&context(""), true), Some(ANSWER_TIER));
}

#[test]
fn test_gathered_context_answers_on_the_answer_tier() {
    assert_eq!(response_tier(&context("DOCUMENT CONTEXT"), false), Some(ANSWER_TIER));
}

#[test]
fn test_a_bare_chat_stays_on_the_default_model() {
    assert_eq!(response_tier(&None, false), None);
    assert_eq!(response_tier(&context(""), false), None);
}
