"""Tests for how "now" reaches the model.

Two things had to be true for a day-aware skill (e.g. "today's workout") to
work, and neither was:

  * the rendered date must name the weekday — ``%B %d, %Y`` gave the model no
    way to derive it, so it hedged by enumerating every day of the week;
  * the skill instruction block must carry the date — it was injected only into
    the final chat prompt, far from the skill body that actually needs it.

The weekday is always derived from the resolved instant (never taken from the
client as a separate field) so the day and date cannot contradict each other.
"""

from datetime import date, datetime, timezone

import pytest

from smartloop.constants import CURRENT_DATE_FORMAT, resolve_current_date


# --------------------------------------------------------------------------
# resolve_current_date
# --------------------------------------------------------------------------

def test_format_leads_with_weekday():
    assert CURRENT_DATE_FORMAT.startswith("%A")


def test_client_timestamp_renders_weekday():
    # 2025-08-19 00:00:00 UTC was a Tuesday.
    assert resolve_current_date(1755561600) == "Tuesday, August 19, 2025"


def test_client_timezone_shifts_the_weekday():
    """The weekday follows the client's timezone, not UTC.

    The regression this guards: same instant, different calendar day. A user in
    New York is still on Monday evening when UTC has already rolled to Tuesday.
    """
    utc = resolve_current_date(1755561600)
    ny = resolve_current_date(1755561600, "America/New_York")
    assert utc == "Tuesday, August 19, 2025"
    assert ny == "Monday, August 18, 2025"


def test_weekday_is_derived_not_client_supplied():
    """The weekday always agrees with the date it is rendered beside."""
    for ts in (1755561600, 1700000000, 1609459200):
        for tz in (None, "America/New_York", "Asia/Tokyo"):
            rendered = resolve_current_date(ts, tz)
            weekday, _, remainder = rendered.partition(", ")
            parsed = datetime.strptime(remainder, "%B %d, %Y")
            assert parsed.strftime("%A") == weekday


def test_unknown_timezone_falls_back_to_utc_not_crash():
    assert resolve_current_date(1755561600, "Not/AZone") == "Tuesday, August 19, 2025"


@pytest.mark.parametrize("bad", ["garbage", None, "", float("nan")])
def test_invalid_client_values_fall_back_to_server_date(bad):
    assert resolve_current_date(bad) == date.today().strftime(CURRENT_DATE_FORMAT)


def test_server_fallback_also_names_the_weekday():
    """The fallback path must not drift from the client path's format."""
    rendered = resolve_current_date()
    assert rendered.split(",")[0] == date.today().strftime("%A")


# --------------------------------------------------------------------------
# skill instruction block
# --------------------------------------------------------------------------

@pytest.fixture
def fmt():
    from smartloop.agents.orchestrator import SLPOrchestrator
    return SLPOrchestrator._format_ready_context


def test_skill_instructions_carry_the_date(fmt):
    out = fmt(
        "tool output", "Give today's workout.", None, "/tmp",
        has_tool_results=True, current_date="Tuesday, August 18, 2026",
    )
    assert "[Current date: Tuesday, August 18, 2026]" in out


def test_date_sits_next_to_the_skill_body_not_after_the_data(fmt):
    """A long skill body must not push the date away from the instructions."""
    out = fmt(
        "x" * 5000, "Give today's workout.", None, "/tmp",
        has_tool_results=True, current_date="Tuesday, August 18, 2026",
    )
    assert out.index("[Current date:") < out.index("x" * 5000)


def test_no_date_emitted_when_unresolved(fmt):
    out = fmt(
        "tool output", "Give today's workout.", None, "/tmp",
        has_tool_results=True, current_date=None,
    )
    assert "[Current date:" not in out


def test_date_block_requires_a_skill(fmt):
    """Without a skill there is no instruction block to attach the date to.

    The plain-chat path already injects the date via format_messages_to_prompt,
    so emitting it here too would duplicate it.
    """
    out = fmt(
        "tool output", None, None, "/tmp",
        has_tool_results=True, current_date="Tuesday, August 18, 2026",
    )
    assert "[Current date:" not in out


# --------------------------------------------------------------------------
# chat prompt
# --------------------------------------------------------------------------

def test_chat_prompt_injects_weekday_bearing_date():
    from smartloop.api.model.chat import ChatMessage
    from smartloop.api.routers.chat import format_messages_to_prompt

    prompt = format_messages_to_prompt(
        [ChatMessage(role="user", content="what day is it?")],
        current_date="Tuesday, August 18, 2026",
    )
    assert "[Current date: Tuesday, August 18, 2026]" in prompt


def test_chat_prompt_fallback_names_the_weekday():
    """With no client date the server fallback still carries a weekday."""
    from smartloop.api.model.chat import ChatMessage
    from smartloop.api.routers.chat import format_messages_to_prompt

    prompt = format_messages_to_prompt(
        [ChatMessage(role="user", content="what day is it?")],
    )
    assert f"[Current date: {date.today().strftime(CURRENT_DATE_FORMAT)}]" in prompt


# --------------------------------------------------------------------------
# request plumbing
# --------------------------------------------------------------------------

def test_client_params_reach_the_resolved_date():
    """tz + user_date from the client survive the schema and are honoured."""
    from smartloop.api.model.chat import ChatCompletionRequest
    from smartloop.api.routers.chat import _request_current_date

    req = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        additional_message_params={
            "tz": "America/New_York",
            "user_date": 1755561600,
        },
    )
    assert _request_current_date(req) == "Monday, August 18, 2025"


def test_missing_params_fall_back_to_server_date():
    from smartloop.api.model.chat import ChatCompletionRequest
    from smartloop.api.routers.chat import _request_current_date

    req = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}],
    )
    assert _request_current_date(req) == date.today().strftime(CURRENT_DATE_FORMAT)
