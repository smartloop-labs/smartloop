//! Port of `tests/unit/test_thinking_filter.py`.
//!
//! A reasoning model's <think> block never reaches a caller.
//!
//! Qwen3.5 can think before it answers (its spec declares THINKING), and
//! everything downstream of the LLM wrapper treats the generated text as the
//! answer — the planner parses it as JSON, the chat endpoint streams it to the
//! user.  So the monologue is filtered out of both the blocking reply and the
//! token stream, including when its tags arrive split across stream chunks.

use smartloop::llms::thinking::{ThinkingFilter, strip_thinking};
use smartloop::model::spec::MODEL_SPECS;

/// Run chunks through the streaming filter, as `LlamaCpp::stream` does.
fn stream(chunks: &[&str]) -> String {
    let mut f = ThinkingFilter::new();
    let mut out: String = chunks.iter().map(|c| f.feed(c)).collect();
    out.push_str(&f.flush());
    out
}

#[test]
fn test_reasoning_block_is_removed() {
    assert_eq!(strip_thinking("<think>weigh it up</think>The answer is 4."), "The answer is 4.");
}

#[test]
fn test_text_without_reasoning_is_untouched() {
    assert_eq!(strip_thinking("The answer is 4."), "The answer is 4.");
}

#[test]
fn test_pre_filled_open_tag_leaves_only_a_close_tag() {
    // Some chat templates put the opening <think> in the generation prompt, so
    // the model only emits the closing tag.
    assert_eq!(strip_thinking("weigh it up</think>The answer is 4."), "The answer is 4.");
}

#[test]
fn test_json_reply_survives_reasoning() {
    let out = strip_thinking("<think>which tool?</think>{\"tool\": \"search\"}");
    assert_eq!(out, "{\"tool\": \"search\"}");
}

#[test]
fn test_tags_split_across_stream_chunks() {
    assert_eq!(stream(&["<thi", "nk>hmm", "m</thin", "k>Fo", "ur."]), "Four.");
}

#[test]
fn test_partial_tag_that_turns_out_to_be_text_is_released() {
    assert_eq!(stream(&["Compare a <", "b in C."]), "Compare a <b in C.");
}

#[test]
fn test_reasoning_is_withheld_chunk_by_chunk() {
    let mut f = ThinkingFilter::new();
    assert_eq!(f.feed("<think>step one"), "");
    assert_eq!(f.feed(" step two"), "");
    assert_eq!(f.feed("</think>Done."), "Done.");
    assert_eq!(f.flush(), "");
    assert!(!f.unterminated());
}

#[test]
fn test_truncated_thought_falls_back_to_the_reasoning() {
    // max_tokens cut generation off mid-thought: there is no reply to show,
    // and a blank answer is worse output than the truncated reasoning.
    let mut f = ThinkingFilter::new();
    assert_eq!(f.feed("<think>halfway through"), "");
    assert!(f.unterminated());
    assert_eq!(f.flush(), "halfway through");
}

#[test]
fn test_truncated_thought_after_a_reply_adds_nothing() {
    let mut f = ThinkingFilter::new();
    assert_eq!(f.feed("Four.<think>second thoughts"), "Four.");
    assert_eq!(f.flush(), "");
}

#[test]
fn test_gemma4_reasoning_channel_is_removed() {
    // Gemma 4 does not use <think>: its template emits reasoning on a
    // <|channel>thought…<channel|> channel (and carries a strip_thinking macro
    // that splits on exactly these markers).
    let out = strip_thinking("<|channel>thought\nweigh it up\n<channel|>The answer is 4.");
    assert_eq!(out, "The answer is 4.");
}

#[test]
fn test_gemma4_channel_split_across_stream_chunks() {
    assert_eq!(stream(&["<|chan", "nel>thought\nhmm", "<chan", "nel|>Four."]), "Four.");
}

#[test]
fn test_reasoning_is_opt_in_for_every_model() {
    // Support is declared by ModelCapability::Thinking; the default is off, so
    // reasoning costs nothing until switched on in the model picker.
    for spec in MODEL_SPECS.specs() {
        assert!(!spec.thinking_default, "{}", spec.name);
    }
}
