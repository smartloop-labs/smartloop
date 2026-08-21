"""Tests for the hallucinated-tool-call guard on the final response.

A small model asked to run a prose skill ("recommend a workout routine")
answers in its tool-calling format instead of prose.  Tool execution is over
by then, so the block is unexecutable — it must never be handed to the user.
"""

import pytest

from smartloop.agents.tool_call_guard import (
    is_tool_call_only,
    may_be_tool_call_prefix,
)
from smartloop.api.routers.chat import _append_to_last_user_turn

# The exact payload observed from sl-mini on `/good-morning`.
GOOD_MORNING_LEAK = (
    '```tool_code\n'
    '{"tool_call": "recommend_workout_routine", '
    '"parameters": { "day_of_week": "Friday"} }\n'
    '```'
)


@pytest.mark.parametrize("text", [
    GOOD_MORNING_LEAK,
    # Unclosed fence — the model hit its token limit mid-call.
    '```tool_code\n{"tool_call": "x", "parameters": {}}',
    # Other execution fences the pipeline uses internally.
    '```mcp\nsmartloop/file-read {"path": "a.md"}\n```',
    # Fenced as json, but the payload is a call.
    '```json\n{"tool_name": "get_weather", "arguments": {"city": "NYC"}}\n```',
    # Bare, no fence.
    '{"tool_call": "recommend_workout_routine", "parameters": {"day": "Fri"}}',
    # OpenAI function shape.
    '{"name": "get_weather", "arguments": {"city": "NYC"}}',
    # Leading/trailing whitespace.
    '\n\n' + GOOD_MORNING_LEAK + '\n',
])
def test_leaked_tool_calls_are_detected(text):
    assert is_tool_call_only(text)


@pytest.mark.parametrize("text", [
    "Happy Friday! Here's a 10-minute lower-body routine: 3 rounds of squats…",
    "",
    "   ",
    # A JSON answer the user actually asked for — no tool-call shape.
    '```json\n{"squats": 20, "lunges": 15}\n```',
    # A name without arguments is ordinary JSON, not a call.
    '{"name": "Friday routine", "exercises": ["squats"]}',
    # Real code the answer is allowed to contain.
    '```python\nprint("hello")\n```',
    # Prose that merely mentions a tool call.
    "I would normally call recommend_workout_routine, but here is the routine…",
    # Prose answer that happens to end with a code block.
    "Here is the routine:\n\n```\n3x squats\n```",
])
def test_real_answers_are_not_flagged(text):
    assert not is_tool_call_only(text)


def test_prose_with_a_trailing_tool_block_is_not_tool_call_only():
    """A real answer is never discarded because a stray block follows it —
    that is the strip path's job, not the retry path's."""
    text = "Here is your Friday routine: 3x10 squats.\n\n```tool_code\n{}\n```"
    assert not is_tool_call_only(text)


@pytest.mark.parametrize("prefix", [
    "```tool_code\n{\"tool_call\": \"recommend_workout_rou",
    "```tool_code\n",
    "```mcp\nsmartloop/file-read",
    '{"tool_call": "recommend_wor',
    '```json\n{"tool_name": "get_weat',
])
def test_partial_tool_calls_hold_the_stream(prefix):
    assert may_be_tool_call_prefix(prefix)


@pytest.mark.parametrize("prefix", [
    "Happy Friday! Here's a 10-minute",
    "```json\n{\"squats\": 20,",
    "```python\nprint(",
    "",
])
def test_real_openings_stream_immediately(prefix):
    assert not may_be_tool_call_prefix(prefix)


def test_directive_lands_inside_the_last_user_turn():
    prompt = "<system>sys</system>\n<user>question</user>"
    out = _append_to_last_user_turn(prompt, "DIRECTIVE")
    assert out.endswith("DIRECTIVE</user>")
    assert out.count("<user>") == 1


def test_directive_falls_back_to_a_new_turn():
    """An unexpected prompt shape still gets the directive, as its own turn."""
    out = _append_to_last_user_turn("bare prompt", "DIRECTIVE")
    assert out == "bare prompt\n<user>DIRECTIVE</user>"
