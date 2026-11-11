//! Port of `tests/unit/test_current_date_grounding.py`.
//!
//! Tests for how "now" reaches the model.
//!
//! Two things had to be true for a day-aware skill (e.g. "today's workout") to
//! work, and neither was:
//!
//!   * the rendered date must name the weekday — `%B %d, %Y` gave the model no
//!     way to derive it, so it hedged by enumerating every day of the week;
//!   * the skill instruction block must carry the date — it was injected only
//!     into the final chat prompt, far from the skill body that actually needs
//!     it.
//!
//! The weekday is always derived from the resolved instant (never taken from
//! the client as a separate field) so the day and date cannot contradict each
//! other.

use chrono::NaiveDate;
use serde_json::json;

use smartloop::agents::orchestration::response_builder::ResponseBuilder;
use smartloop::api::model::chat::{ChatCompletionRequest, ChatMessage, Role};
use smartloop::api::routers::chat::{format_messages_to_prompt, request_current_date};
use smartloop::constants::{resolve_current_date, CURRENT_DATE_FORMAT};

/// `date.today().strftime(fmt)` — the server's local calendar day.
fn today(fmt: &str) -> String {
    chrono::Local::now().date_naive().format(fmt).to_string()
}

// --------------------------------------------------------------------------
// resolve_current_date
// --------------------------------------------------------------------------

#[test]
fn test_format_leads_with_weekday() {
    assert!(CURRENT_DATE_FORMAT.starts_with("%A"));
}

#[test]
fn test_client_timestamp_renders_weekday() {
    // 2025-08-19 00:00:00 UTC was a Tuesday.
    assert_eq!(resolve_current_date(Some(1755561600), None), "Tuesday, August 19, 2025");
}

/// The weekday follows the client's timezone, not UTC.
///
/// The regression this guards: same instant, different calendar day. A user
/// in New York is still on Monday evening when UTC has already rolled to
/// Tuesday.
#[test]
fn test_client_timezone_shifts_the_weekday() {
    let utc = resolve_current_date(Some(1755561600), None);
    let ny = resolve_current_date(Some(1755561600), Some("America/New_York"));
    assert_eq!(utc, "Tuesday, August 19, 2025");
    assert_eq!(ny, "Monday, August 18, 2025");
}

/// The weekday always agrees with the date it is rendered beside.
#[test]
fn test_weekday_is_derived_not_client_supplied() {
    for ts in [1755561600_i64, 1700000000, 1609459200] {
        for tz in [None, Some("America/New_York"), Some("Asia/Tokyo")] {
            let rendered = resolve_current_date(Some(ts), tz);
            let (weekday, remainder) = rendered.split_once(", ").expect("weekday, date");
            let parsed = NaiveDate::parse_from_str(remainder, "%B %d, %Y").expect("a date");
            assert_eq!(parsed.format("%A").to_string(), weekday);
        }
    }
}

#[test]
fn test_unknown_timezone_falls_back_to_utc_not_crash() {
    assert_eq!(resolve_current_date(Some(1755561600), Some("Not/AZone")), "Tuesday, August 19, 2025");
}

// `@pytest.mark.parametrize("bad", ["garbage", None, "", float("nan")])`.
// The typed signature takes an optional integer timestamp, so "garbage", ""
// and NaN cannot reach it at all (the request schema rejects them); the cases
// that can are a missing value and a timestamp no calendar can render.

#[test]
fn test_invalid_client_values_fall_back_to_server_date_none() {
    assert_eq!(resolve_current_date(None, None), today(CURRENT_DATE_FORMAT));
}

#[test]
fn test_invalid_client_values_fall_back_to_server_date_out_of_range() {
    assert_eq!(resolve_current_date(Some(i64::MAX), None), today(CURRENT_DATE_FORMAT));
}

#[test]
fn test_invalid_client_values_fall_back_to_server_date_out_of_range_negative() {
    assert_eq!(resolve_current_date(Some(i64::MIN), None), today(CURRENT_DATE_FORMAT));
}

/// The fallback path must not drift from the client path's format.
#[test]
fn test_server_fallback_also_names_the_weekday() {
    let rendered = resolve_current_date(None, None);
    assert_eq!(rendered.split(',').next().unwrap(), today("%A"));
}

// --------------------------------------------------------------------------
// skill instruction block
// --------------------------------------------------------------------------

fn fmt(
    combined: &str,
    skill_instruction: Option<&str>,
    has_tool_results: bool,
    current_date: Option<&str>,
) -> String {
    ResponseBuilder::format_ready_context(
        Some(combined),
        skill_instruction,
        None,
        Some("/tmp"),
        has_tool_results,
        current_date,
    )
}

#[test]
fn test_skill_instructions_carry_the_date() {
    let out = fmt("tool output", Some("Give today's workout."), true, Some("Tuesday, August 18, 2026"));
    assert!(out.contains("[Current date: Tuesday, August 18, 2026]"), "{out}");
}

/// A long skill body must not push the date away from the instructions.
#[test]
fn test_date_sits_next_to_the_skill_body_not_after_the_data() {
    let data = "x".repeat(5000);
    let out = fmt(&data, Some("Give today's workout."), true, Some("Tuesday, August 18, 2026"));
    let date_at = out.find("[Current date:").expect("date present");
    let data_at = out.find(&data).expect("data present");
    assert!(date_at < data_at);
}

#[test]
fn test_no_date_emitted_when_unresolved() {
    let out = fmt("tool output", Some("Give today's workout."), true, None);
    assert!(!out.contains("[Current date:"));
}

/// Without a skill there is no instruction block to attach the date to.
///
/// The plain-chat path already injects the date via
/// `format_messages_to_prompt`, so emitting it here too would duplicate it.
#[test]
fn test_date_block_requires_a_skill() {
    let out = fmt("tool output", None, true, Some("Tuesday, August 18, 2026"));
    assert!(!out.contains("[Current date:"));
}

// --------------------------------------------------------------------------
// chat prompt
// --------------------------------------------------------------------------

fn user(content: &str) -> ChatMessage {
    ChatMessage { role: Role::User, content: Some(content.to_string()), attachments: None, ..Default::default() }
}

#[test]
fn test_chat_prompt_injects_weekday_bearing_date() {
    let prompt =
        format_messages_to_prompt(&[user("what day is it?")], None, None, None, Some("Tuesday, August 18, 2026"));
    assert!(prompt.contains("[Current date: Tuesday, August 18, 2026]"), "{prompt}");
}

/// With no client date the server fallback still carries a weekday.
#[test]
fn test_chat_prompt_fallback_names_the_weekday() {
    let prompt = format_messages_to_prompt(&[user("what day is it?")], None, None, None, None);
    assert!(prompt.contains(&format!("[Current date: {}]", today(CURRENT_DATE_FORMAT))), "{prompt}");
}

// --------------------------------------------------------------------------
// request plumbing
// --------------------------------------------------------------------------

/// tz + user_date from the client survive the schema and are honoured.
#[test]
fn test_client_params_reach_the_resolved_date() {
    let req: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "additional_message_params": {
            "tz": "America/New_York",
            "user_date": 1755561600,
        },
    }))
    .unwrap();
    assert_eq!(request_current_date(&req), "Monday, August 18, 2025");
}

#[test]
fn test_missing_params_fall_back_to_server_date() {
    let req: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .unwrap();
    assert_eq!(request_current_date(&req), today(CURRENT_DATE_FORMAT));
}
