//! A small model asked which skill applies picks the only one listed,
//! whatever the request: the on-device router matched a calendar skill to
//! "commit and push" on every turn.  Its pick must share a word with the
//! skill; a larger model's judgement stands on intent alone.

use std::any::Any;
use std::sync::Arc;

use smartloop::agents::mcp::explorer::{Explorer, SkillDict};
use smartloop::llms::base_llm::{BaseLLM, CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream};

/// A model that always answers "1", under `label`.
struct AlwaysOne {
    label: String,
}

impl BaseLLM for AlwaysOne {
    fn project_id(&self) -> &str {
        ""
    }
    fn get_num_tokens(&self, text: &str) -> usize {
        text.len().div_ceil(4)
    }
    fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        messages
    }
    fn get_max_tokens(&self) -> usize {
        4096
    }
    fn call(&self, _prompt: &str, _stop: Option<&[String]>, _options: CallOptions) -> anyhow::Result<String> {
        Ok("1".to_string())
    }
    fn stream<'a>(&'a self, _prompt: &str, _options: CallOptions) -> GenerationStream<'a> {
        Box::new(std::iter::once(GenerationChunk { text: "1".to_string() }))
    }
    fn invoke(&self, _prompt: &str, _max_tokens: Option<usize>, _thinking: Option<bool>) -> String {
        "1".to_string()
    }
    fn model_label(&self) -> String {
        self.label.clone()
    }
    fn cancelled(&self) -> Option<CancelEvent> {
        None
    }
    fn set_cancelled(&self, _event: Option<CancelEvent>) {}
    fn llm_type(&self) -> &str {
        "always-one"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn calendar() -> SkillDict {
    SkillDict::from([
        ("name".to_string(), "my-calendar".to_string()),
        (
            "description".to_string(),
            "Show today's meetings from macOS Calendar as a markdown table (Time, Title, Calendar).".to_string(),
        ),
    ])
}

fn matched(label: &str, message: &str) -> Vec<String> {
    let llm: Arc<dyn BaseLLM> = Arc::new(AlwaysOne { label: label.to_string() });
    let explorer = Explorer::new(llm.clone(), None, None, false);
    explorer
        .llm_match_skills(message, &[calendar()], Some(llm.as_ref()))
        .iter()
        .map(|s| s["name"].clone())
        .collect()
}

#[test]
fn test_a_small_models_pick_with_nothing_in_common_is_dropped() {
    // sl-mini is the 1B on-device router.
    assert!(matched("sl-mini", "commit and push the repo changes").is_empty());
}

#[test]
fn test_a_small_models_pick_that_shares_a_word_stands() {
    assert_eq!(matched("sl-mini", "what meetings do I have today?"), ["my-calendar"]);
    assert_eq!(matched("sl-mini", "check my calendar"), ["my-calendar"]);
}

#[test]
fn test_a_larger_or_unknown_models_pick_stands_on_intent() {
    // qwen3.5-4b is 4B; an unlabelled model's size isn't known.
    assert_eq!(matched("qwen3.5-4b", "anything on my agenda?"), ["my-calendar"]);
    assert_eq!(matched("", "anything on my agenda?"), ["my-calendar"]);
}
