"""The leaked-tool-call retry must still stream token by token.

Driving the real SSE generator (``stream_chunks``) with a fake model service:
the first generation answers a skill prompt with a hallucinated tool call, the
retry answers in prose.  The user must never see the call, and must see the
prose arrive progressively — a retry delivered as one blob reads as a multi-
second freeze followed by a wall of text.
"""

import json

import pytest

from smartloop.api.model import ChatCompletionRequest, ChatMessage
from smartloop.api.routers import chat as chat_router

LEAK = [
    "```tool_code\n",
    '{"tool_call": ',
    '"recommend_workout_routine"',
    ', "parameters": {"day_of_week": "Friday"}}',
    "\n```",
]

PROSE = [
    "Good morning! ", "Here is your ", "Friday lower-body ", "routine:\n\n",
    "1. Goblet squats ", "3x12\n", "2. Dumbbell lunges ", "3x10\n",
    "3. Calf raises ", "3x20\n", "Keep it light ", "and stretch after.",
]


class _Chunk:
    def __init__(self, text):
        self.text = text


class _FakeModelService:
    """Just enough ModelService surface for the streaming path."""

    def __init__(self, streams):
        self._streams = list(streams)
        self.stream_prompts = []
        self.generate_prompts = []

    # --- orchestration -------------------------------------------------
    def orchestrate(self, *args, **kwargs):
        # (context, _, tool_error, citations, skill_instruction, is_mutation)
        return ("TOOL RESULTS:\nworkout skill instructions", None, None, None, None, False)

    # --- generation ----------------------------------------------------
    def stream_completion(self, prompt, model, cancelled=None, capability=None):
        self.stream_prompts.append(prompt)
        for piece in self._streams.pop(0):
            yield _Chunk(piece)

    def generate_completion(self, prompt, model, capability=None):
        self.generate_prompts.append(prompt)
        return "simplified fallback answer"

    # --- incidental plumbing -------------------------------------------
    def set_current_project(self, project_id): pass
    def get_relevant_history(self, session_id, message): return []
    def get_project_skills(self): return []
    def get_project_id(self): return "project-1"
    def get_project_info(self): return None
    def count_tokens(self, text): return max(1, len(text or "") // 4)
    def get_response_max_context_tokens(self, capability=None): return 8192
    def get_max_context_tokens(self): return 8192
    def append_to_session(self, *args, **kwargs): pass
    def maybe_write_output(self, *args, **kwargs): return None
    def unload_auxiliary_models(self): pass
    def get_asset_images(self, attachments): return []


def _content_chunks(events):
    """The visible answer text, chunk by chunk, from raw SSE events."""
    out = []
    for event in events:
        if not event.startswith("data: "):
            continue
        payload = event[len("data: "):].strip()
        if payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except ValueError:
            continue
        for choice in obj.get("choices") or []:
            text = (choice.get("delta") or {}).get("content")
            if text:
                out.append(text)
    return out


async def _run(model_service):
    request = ChatCompletionRequest(
        model="sl-mini",
        messages=[ChatMessage(role="user", content="Run the 'good-morning' skill.")],
        stream=True,
    )
    return [event async for event in chat_router.stream_chunks(request, model_service)]


@pytest.fixture(autouse=True)
def _no_telemetry(monkeypatch):
    class _NoopClient:
        def push_token_usage(self, **kwargs): pass
    monkeypatch.setattr(chat_router, "get_api_client", lambda: _NoopClient())


@pytest.mark.asyncio
async def test_leaked_tool_call_is_replaced_by_streamed_prose():
    svc = _FakeModelService([LEAK, PROSE])
    events = await _run(svc)
    chunks = _content_chunks(events)
    answer = "".join(chunks)

    # The hallucinated call never reaches the client.
    assert "tool_call" not in answer
    assert "tool_code" not in answer
    assert "recommend_workout_routine" not in answer

    # The prose answer arrives complete...
    assert answer == "".join(PROSE)
    # ...and progressively, not as one blob.
    assert len(chunks) > 1

    # The retry reused the original prompt (skill context intact) plus the
    # no-tools directive — it did not fall back to the simplified prompt.
    assert len(svc.stream_prompts) == 2
    assert "You have NO tools" in svc.stream_prompts[1]
    assert "TOOL RESULTS" in svc.stream_prompts[1]
    assert svc.generate_prompts == []


@pytest.mark.asyncio
async def test_clean_first_response_streams_without_a_retry():
    """A normal answer is untouched: streamed once, no second generation."""
    svc = _FakeModelService([PROSE])
    events = await _run(svc)
    chunks = _content_chunks(events)

    assert "".join(chunks) == "".join(PROSE)
    assert len(chunks) > 1
    assert len(svc.stream_prompts) == 1
    assert svc.generate_prompts == []


@pytest.mark.asyncio
async def test_two_leaks_fall_back_to_the_simplified_prompt():
    """When the retry also emits a call, the simplified non-streaming retry
    (existing behaviour) takes over rather than shipping the JSON."""
    svc = _FakeModelService([LEAK, LEAK])
    events = await _run(svc)
    answer = "".join(_content_chunks(events))

    assert "tool_call" not in answer
    assert answer == "simplified fallback answer"
    assert len(svc.generate_prompts) == 1
