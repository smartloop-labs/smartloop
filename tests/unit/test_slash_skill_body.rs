//! Port of `tests/unit/test_slash_skill_body.py`.
//!
//! A bare `/<skill>` stands in the skill's body, not its name.
//!
//! Tool arguments are extracted from the user's message, so a prompt of just
//! `/morning` used to reach its tools as "Run the 'morning' skill." — the
//! name, where the skill's instructions were what the tools needed.

use smartloop::api::routers::chat::resolve_requested_skill;
use smartloop::workspace::types::Skill;

fn skill(name: &str, content: &str) -> Skill {
    Skill {
        id: String::new(),
        name: name.to_string(),
        description: None,
        content: content.to_string(),
        system: false,
    }
}

fn morning() -> Skill {
    skill(
        "morning",
        "---\nname: morning\ndescription: brief\n---\n\nList today's calendar events.",
    )
}

#[test]
fn test_a_bare_invocation_uses_the_skill_body() {
    assert_eq!(
        resolve_requested_skill("/morning", &[morning()]),
        (Some("morning".to_string()), "List today's calendar events.".to_string()),
    );
}

#[test]
fn test_text_after_the_token_is_still_the_message() {
    assert_eq!(
        resolve_requested_skill("/morning only meetings", &[morning()]),
        (Some("morning".to_string()), "only meetings".to_string()),
    );
}

#[test]
fn test_an_empty_skill_falls_back_to_naming_it() {
    let empty = skill("morning", "");
    assert_eq!(
        resolve_requested_skill("/morning", &[empty]),
        (Some("morning".to_string()), "Run the 'morning' skill.".to_string()),
    );
}

/// A skill edited from the template keeps the placeholder block above the
/// real one; the unfenced second block must not lead the message.
#[test]
fn test_a_second_frontmatter_block_is_not_the_message() {
    let doubled = skill(
        "job-match",
        concat!(
            "---\nname: job-match\ndescription: placeholder\n---\n\n",
            "name: job-match\ndescription: Uses clera for job matches\n---\n\n",
            "# Job Match\n\nReturn a list of job matches."
        ),
    );
    let (_, message) = resolve_requested_skill("/job-match", &[doubled]);
    assert_eq!(message, "# Job Match\n\nReturn a list of job matches.");
}
