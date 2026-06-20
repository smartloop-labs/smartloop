"""Tests for the OpenAI-compatible vision endpoint (/vision/v1).

This is the API a scanned PDF's per-page OCR calls: an image is posted to
``/vision/v1/chat/completions`` and the vision-capable model transcribes it.
The endpoint is backed by ``ModelService.get_vision_llm()``, which only returns
a model when a usable multimodal (main + mmproj) GGUF is present.
"""
import threading
from types import SimpleNamespace

import pytest

from smartloop.api.main import app
from smartloop.api.services import get_model_service


class _FakeVisionService:
    """Minimal stand-in for ModelService used by the vision router."""

    def __init__(self, llm):
        self._llm = llm
        self._model = None
        self._inference_lock = threading.Semaphore(1)
        self.completion_calls = []

    def get_vision_llm(self):
        return self._llm


def _fake_llm(captured_calls):
    def _create_chat_completion(**kwargs):
        captured_calls.append(kwargs)
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Transcribed: revenue +20%"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
        }

    return SimpleNamespace(model=SimpleNamespace(create_chat_completion=_create_chat_completion))


@pytest.fixture
def override_service():
    """Install a fake ModelService and clean up afterwards."""
    calls = []

    def _install(llm):
        svc = _FakeVisionService(llm)
        svc.completion_calls = calls
        app.dependency_overrides[get_model_service] = lambda: svc
        return svc, calls

    yield _install
    app.dependency_overrides.pop(get_model_service, None)


_IMAGE_REQUEST = {
    "model": "vision",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this page to markdown."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANS"},
                },
            ],
        }
    ],
    "max_tokens": 256,
}


def test_chat_completion_transcribes_image(client, override_service):
    captured = []
    _, _ = override_service(_fake_llm(captured))

    resp = client.post("/vision/v1/chat/completions", json=_IMAGE_REQUEST)

    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Transcribed: revenue +20%"
    # OpenAI-shaped fields are backfilled.
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")

    # The image actually reached the model via create_chat_completion.
    assert len(captured) == 1
    parts = captured[0]["messages"][0]["content"]
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert image_parts and image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert captured[0]["max_tokens"] == 256


def test_chat_completion_503_when_no_vision_model(client, override_service):
    override_service(None)  # get_vision_llm() -> None (e.g. mmproj missing)

    resp = client.post("/vision/v1/chat/completions", json=_IMAGE_REQUEST)

    assert resp.status_code == 503
    assert "vision" in resp.json()["detail"].lower()


def test_health_reports_availability(client, override_service):
    override_service(_fake_llm([]))
    resp = client.get("/vision/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"available": True}


def test_health_false_when_unavailable(client, override_service):
    override_service(None)
    resp = client.get("/vision/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"available": False}
