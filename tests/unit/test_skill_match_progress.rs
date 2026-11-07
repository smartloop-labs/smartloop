//! Port of `tests/unit/test_skill_match_progress.py`.
//!
//! The skill-match event describes the skills that can actually be matched.
//!
//! It used to fire on entry, counting `ctx.skills` — every skill on disk,
//! including the built-in tool skills (bash, file-read, file-write, grep,
//! explore) and the `__`-prefixed system skills, none of which are matchable.
//! So a project whose only real skill was one the user wrote was reported as
//! "3 available skill(s)", and a project with nothing matchable at all still
//! announced an analysis that never ran.
//!
//! Python mocked the router (recording `emit` calls) and the explorer
//! (`match_skills` returning nothing).  Here both are real: the router's
//! progress sink records what it is told, and the explorer's model — which
//! answers no match — records every prompt, so "the matcher ran" means "the
//! model was asked", and the skills it weighed are the ones in that prompt.

use std::sync::Arc;

use smartloop::agents::mcp::explorer::Explorer;
use smartloop::agents::nodes::{NodeResult, SkillMatchNode};
use smartloop::agents::orchestration::model_router::{ModelRouter, ModelRouterOptions};
use smartloop::agents::orchestration::skill_processor::SkillProcessor;
use smartloop::agents::orchestration::OrchestratorContext;

use crate::shared::agents_core::{EmitSink, FakeLlm};

fn skill(name: &str) -> String {
    format!("---\nname: {name}\ndescription: does a thing\n---\n\nBody of {name}.")
}

struct Run {
    result: NodeResult,
    messages: Vec<String>,
    /// Every prompt the matcher's model was asked.
    prompts: Vec<String>,
}

fn run(skills: Vec<String>, user_message: &str, requested_skill: Option<&str>) -> Run {
    let llm = FakeLlm::new("");
    let router = Arc::new(ModelRouter::new(llm.shared(), llm.shared(), ModelRouterOptions::default()));
    let sink = EmitSink::default();
    router.begin_request(Some(sink.callback()));
    let explorer = Arc::new(Explorer::new(llm.shared(), None, None, false));
    let processor = SkillProcessor::new(router, explorer);

    let mut ctx = OrchestratorContext::new(user_message, vec![], vec![], "/tmp");
    ctx.skills = skills;
    ctx.requested_skill = requested_skill.map(str::to_string);

    let result = processor.match_skills(&mut ctx, &SkillMatchNode::new(&[]));
    Run { result, messages: sink.messages(), prompts: llm.calls() }
}

const IMAGE_QUESTION: &str = "what is in this image";

/// bash / file-read / file-write are built-in tools, not the user's skills.
#[test]
fn test_only_matchable_skills_are_counted() {
    let run = run(
        vec![skill("bash"), skill("file-read"), skill("file-write"), skill("good-morning")],
        IMAGE_QUESTION,
        None,
    );

    assert_eq!(run.messages, vec!["Analyzing your request against 1 skill".to_string()]);
}

#[test]
fn test_several_user_skills_are_counted_and_pluralised() {
    let run = run(vec![skill("bash"), skill("good-morning"), skill("fund-summary")], IMAGE_QUESTION, None);

    assert_eq!(run.messages, vec!["Analyzing your request against 2 skills".to_string()]);
}

/// A project with only built-in tool skills runs no match at all.
#[test]
fn test_nothing_matchable_says_nothing() {
    let run = run(vec![skill("bash"), skill("file-read"), skill("file-write")], IMAGE_QUESTION, None);

    assert!(run.messages.is_empty());
    assert!(run.prompts.is_empty(), "the matcher ran");
    assert_eq!(run.result.data["matched_skills"], serde_json::json!([]));
}

#[test]
fn test_no_skills_at_all_says_nothing() {
    let run = run(vec![], IMAGE_QUESTION, None);

    assert!(run.messages.is_empty());
    assert!(run.prompts.is_empty(), "the matcher ran");
}

/// fetch-url only becomes a candidate when the message carries a URL.
#[test]
fn test_fetch_skills_excluded_without_a_url_are_not_counted() {
    let run = run(vec![skill("fetch-url"), skill("good-morning")], IMAGE_QUESTION, None);

    assert_eq!(run.messages, vec!["Analyzing your request against 1 skill".to_string()]);
}

#[test]
fn test_fetch_skill_counts_once_the_message_has_a_url() {
    let run = run(vec![skill("fetch-url"), skill("good-morning")], "summarise https://example.com/post", None);

    assert_eq!(run.messages, vec!["Analyzing your request against 2 skills".to_string()]);
}

/// The number reported is the number actually weighed.
#[test]
fn test_the_count_reaches_the_matcher_unchanged() {
    let run = run(vec![skill("bash"), skill("good-morning"), skill("fund-summary")], IMAGE_QUESTION, None);

    // The skills the matcher weighed are the ones put to its model.
    let weighed: Vec<&str> = ["bash", "good-morning", "fund-summary"]
        .into_iter()
        .filter(|name| run.prompts.iter().any(|p| p.contains(&format!("{name}:")) || p.contains(&format!("\"{name}\""))))
        .collect();
    assert_eq!(weighed, vec!["good-morning", "fund-summary"]);
    assert_eq!(run.messages, vec![format!("Analyzing your request against {} skills", weighed.len())]);
}

#[test]
fn test_an_explicitly_requested_skill_reports_that_instead() {
    let run = run(vec![skill("bash"), skill("good-morning")], IMAGE_QUESTION, Some("good-morning"));

    assert_eq!(run.messages, vec!["Fulfilling your request with the 'good-morning' skill".to_string()]);
    assert!(run.prompts.is_empty(), "the matcher ran");
}

/// A skill the user names outright is matched without asking the model.
#[test]
fn test_a_named_skill_skips_the_model() {
    let run = run(vec![skill("good-morning"), skill("fund-summary")], "run the good-morning skill", None);

    assert!(run.prompts.is_empty(), "{:?}", run.prompts);
    assert!(run.result.data.to_string().contains("good-morning"), "{}", run.result.data);
}
