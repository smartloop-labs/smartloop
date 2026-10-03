"""A bare ``/<skill>`` stands in the skill's body, not its name.

Tool arguments are extracted from the user's message, so a prompt of just
``/morning`` used to reach its tools as "Run the 'morning' skill." — the name,
where the skill's instructions were what the tools needed.
"""
from types import SimpleNamespace

from smartloop.api.routers.chat import _resolve_requested_skill

SKILL = SimpleNamespace(
    name="morning",
    content="---\nname: morning\ndescription: brief\n---\n\nList today's calendar events.",
)


def test_a_bare_invocation_uses_the_skill_body():
    assert _resolve_requested_skill("/morning", [SKILL]) == (
        "morning", "List today's calendar events.",
    )


def test_text_after_the_token_is_still_the_message():
    assert _resolve_requested_skill("/morning only meetings", [SKILL]) == (
        "morning", "only meetings",
    )


def test_an_empty_skill_falls_back_to_naming_it():
    empty = SimpleNamespace(name="morning", content="")
    assert _resolve_requested_skill("/morning", [empty]) == (
        "morning", "Run the 'morning' skill.",
    )


def test_a_second_frontmatter_block_is_not_the_message():
    """A skill edited from the template keeps the placeholder block above the
    real one; the unfenced second block must not lead the message."""
    doubled = SimpleNamespace(name="job-match", content=(
        "---\nname: job-match\ndescription: placeholder\n---\n\n"
        "name: job-match\ndescription: Uses clera for job matches\n---\n\n"
        "# Job Match\n\nReturn a list of job matches."
    ))
    _, message = _resolve_requested_skill("/job-match", [doubled])
    assert message == "# Job Match\n\nReturn a list of job matches."
