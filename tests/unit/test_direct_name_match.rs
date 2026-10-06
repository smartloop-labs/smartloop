//! Port of `tests/unit/test_direct_name_match.py`.
//!
//! Tests for MCP explorer direct (name-invocation) skill matching.
//!
//! `match_skills` unions an LLM qualifier (intent-aware) with a deterministic
//! *direct* match that fires only when the user names the skill explicitly.
//! These tests pin the direct matcher: a bare topic mention must NOT match —
//! that judgement is left to the LLM — while an explicit invocation does.

use smartloop::agents::mcp::explorer::{Explorer, SkillDict};

fn fund_skill() -> SkillDict {
    SkillDict::from([
        ("name".to_string(), "fund-info".to_string()),
        (
            "description".to_string(),
            "Use this skill when user requests to summarize the funds, like \
             fund names, ticker symbol, annual fund operating expenses"
                .to_string(),
        ),
        (
            "content".to_string(),
            "Extract fund names, ticker symbols and operating expenses.".to_string(),
        ),
    ])
}

fn names(result: &[SkillDict]) -> Vec<&str> {
    result.iter().map(|s| s["name"].as_str()).collect()
}

/// A general question that only mentions the subject must NOT match.
#[test]
fn test_topic_mention_is_not_a_direct_match() {
    let result = Explorer::direct_name_matches("what is the purpose of the funds ?", &[fund_skill()]);
    assert!(result.is_empty());
}

/// 'summarize the fund' is left to the LLM, not the direct matcher.
#[test]
fn test_summarize_request_is_not_a_direct_match() {
    let result = Explorer::direct_name_matches("summarize the fund", &[fund_skill()]);
    assert!(result.is_empty());
}

/// Naming the skill's phrase directly is a match.
#[test]
fn test_explicit_invocation_matches() {
    let result = Explorer::direct_name_matches("give me the fund info for my portfolio", &[fund_skill()]);
    assert_eq!(names(&result), vec!["fund-info"]);
}

/// The hyphenated name form is matched too.
#[test]
fn test_hyphenated_skill_name_matches() {
    let result = Explorer::direct_name_matches("run the fund-info skill", &[fund_skill()]);
    assert_eq!(names(&result), vec!["fund-info"]);
}

/// A substring that isn't a whole-word phrase must NOT match.
#[test]
fn test_partial_word_does_not_match() {
    let result = Explorer::direct_name_matches("refunding information request", &[fund_skill()]);
    assert!(result.is_empty());
}
