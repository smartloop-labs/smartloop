"""The skill-match event describes the skills that can actually be matched.

It used to fire on entry, counting ``ctx.skills`` — every skill on disk,
including the built-in tool skills (bash, file-read, file-write, grep, explore)
and the ``__``-prefixed system skills, none of which are matchable.  So a
project whose only real skill was one the user wrote was reported as "3
available skill(s)", and a project with nothing matchable at all still
announced an analysis that never ran.
"""

from unittest.mock import MagicMock

import pytest

from smartloop.agents.nodes import SkillMatchNode
from smartloop.agents.orchestration.skill_processor import SkillProcessor


def _skill(name, description="does a thing"):
    return f"---\nname: {name}\ndescription: {description}\n---\n\nBody of {name}."


def _run(skills, user_message="what is in this image", requested_skill=None):
    router = MagicMock()
    explorer = MagicMock()
    explorer.match_skills.return_value = []
    processor = SkillProcessor(router, explorer)

    ctx = MagicMock()
    ctx.skills = skills
    ctx.user_message = user_message
    ctx.requested_skill = requested_skill

    result = processor.match(ctx, SkillMatchNode())
    messages = [c.args[2] for c in router.emit.call_args_list]
    return result, messages, explorer


def test_only_matchable_skills_are_counted():
    """bash / file-read / file-write are built-in tools, not the user's skills."""
    _, messages, _ = _run([
        _skill("bash"), _skill("file-read"), _skill("file-write"),
        _skill("good-morning"),
    ])

    assert messages == ["Analyzing your request against 1 skill"]


def test_several_user_skills_are_counted_and_pluralised():
    _, messages, _ = _run([
        _skill("bash"), _skill("good-morning"), _skill("fund-summary"),
    ])

    assert messages == ["Analyzing your request against 2 skills"]


def test_nothing_matchable_says_nothing():
    """A project with only built-in tool skills runs no match at all."""
    result, messages, explorer = _run([
        _skill("bash"), _skill("file-read"), _skill("file-write"),
    ])

    assert messages == []
    explorer.match_skills.assert_not_called()
    assert result.data["matched_skills"] == []


def test_no_skills_at_all_says_nothing():
    _, messages, explorer = _run([])

    assert messages == []
    explorer.match_skills.assert_not_called()


def test_fetch_skills_excluded_without_a_url_are_not_counted():
    """fetch-url only becomes a candidate when the message carries a URL."""
    _, messages, _ = _run([_skill("fetch-url"), _skill("good-morning")])

    assert messages == ["Analyzing your request against 1 skill"]


def test_fetch_skill_counts_once_the_message_has_a_url():
    _, messages, _ = _run(
        [_skill("fetch-url"), _skill("good-morning")],
        user_message="summarise https://example.com/post",
    )

    assert messages == ["Analyzing your request against 2 skills"]


def test_the_count_reaches_the_matcher_unchanged():
    """The number reported is the number actually weighed."""
    _, messages, explorer = _run([
        _skill("bash"), _skill("good-morning"), _skill("fund-summary"),
    ])

    weighed = explorer.match_skills.call_args.args[1]
    assert len(weighed) == 2
    assert messages == [f"Analyzing your request against {len(weighed)} skills"]


def test_an_explicitly_requested_skill_reports_that_instead():
    _, messages, explorer = _run(
        [_skill("bash"), _skill("good-morning")], requested_skill="good-morning",
    )

    assert messages == ["Fulfilling your request with the 'good-morning' skill"]
    explorer.match_skills.assert_not_called()
