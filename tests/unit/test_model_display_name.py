"""The public model name shown in SSE streams.

The internal id ``slp-orchestra-mini`` keys the on-disk model directory, the
workspace records and the HuggingFace repo, so it can't be renamed.  Anything
a user reads — the chat completion chunks and the /v1/init progress events —
shows the short public name ``sl-mini`` instead.
"""
import ast
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from smartloop.constants import display_model_name


# --------------------------------------------------------------------------
# The mapping itself
# --------------------------------------------------------------------------

def test_orchestrator_id_maps_to_public_name():
    assert display_model_name("slp-orchestra-mini") == "sl-mini"


def test_unknown_model_passes_through():
    """Workspace models the user added keep whatever name they were given."""
    assert display_model_name("qwen3-8b") == "qwen3-8b"


@pytest.mark.parametrize("empty", [None, ""])
def test_empty_name_passes_through(empty):
    assert display_model_name(empty) == empty


# --------------------------------------------------------------------------
# Chat SSE chunks
# --------------------------------------------------------------------------

def _chat_source() -> str:
    from smartloop.api.routers import chat as chat_router
    return Path(chat_router.__file__).read_text(encoding="utf-8")


def test_every_sse_chunk_reports_the_display_name():
    """No ``ChatCompletionChunk`` may echo the raw requested id.

    ``stream_model`` is resolved once at the top of ``stream_chunks``; a new
    chunk that passes ``request.model`` straight through would leak
    ``slp-orchestra-mini`` back to the client mid-stream.
    """
    chunks = [
        node for node in ast.walk(ast.parse(_chat_source()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ChatCompletionChunk"
    ]
    assert chunks, "expected ChatCompletionChunk constructions in chat.py"

    for call in chunks:
        model_kw = next((k for k in call.keywords if k.arg == "model"), None)
        assert model_kw is not None, f"chunk at line {call.lineno} has no model="
        assert isinstance(model_kw.value, ast.Name) and model_kw.value.id == "stream_model", (
            f"chunk at line {call.lineno} must use stream_model, not the raw request id"
        )


def test_token_usage_still_reports_the_internal_id():
    """Telemetry keys off the real model id — the alias is display-only."""
    assert "model=request.model," in _chat_source()


# --------------------------------------------------------------------------
# /v1/init SSE progress events
# --------------------------------------------------------------------------

async def test_init_load_stream_shows_public_name(monkeypatch):
    """``Loading orchestrator model ...`` names sl-mini, not the internal id."""
    from smartloop.api.routers import init as init_router

    monkeypatch.setattr(init_router.api_settings, "default_model", "slp-orchestra-mini")

    model_service = MagicMock()
    model_service._model_name = "slp-orchestra-mini"
    model_service.load_model = MagicMock(return_value=None)

    messages = []
    async for event in init_router._load_orchestrator_and_stream(model_service):
        for line in event.splitlines():
            if line.startswith("data: "):
                messages.append(json.loads(line[len("data: "):]).get("message", ""))

    joined = "\n".join(messages)
    assert "sl-mini" in joined
    assert "slp-orchestra-mini" not in joined
