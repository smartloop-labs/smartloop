"""Validate that scanned (image-only) PDFs are OCR'd through the vision API.

The document parser extracts text with MarkItDown for normal PDFs, but a
scanned / image-only PDF has no extractable text.  In that case each sparse
page must be rendered to an image and transcribed by the vision-capable model
via the local ``/vision/v1/chat/completions`` endpoint (an OpenAI-compatible
client call).  These tests assert that path is exercised — and that it is NOT
exercised for text PDFs or when no vision model is available.
"""
import os
import tempfile
from types import SimpleNamespace

import pytest

fitz = pytest.importorskip("fitz")  # PyMuPDF
pytest.importorskip("markitdown")

from smartloop.parsers import document_parser as dp
from smartloop.parsers.parser_factory import ParserFactory


# --- fakes for the OpenAI-compatible vision client ---------------------------

class _FakeCompletions:
    """Records create() calls and returns a canned transcription."""

    def __init__(self, content):
        self._content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


class _FakeVisionClient:
    def __init__(self, content="# Page\nTranscribed by vision model."):
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


# --- helpers -----------------------------------------------------------------

def _scanned_pdf(pages=3) -> str:
    """An image-only PDF: filled rectangles, no text layer."""
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.draw_rect(fitz.Rect(40, 40, 520, 760), fill=(0.1, 0.1, 0.1))
    path = os.path.join(tempfile.mkdtemp(), "scanned.pdf")
    doc.save(path)
    doc.close()
    return path


def _text_pdf(pages=2) -> str:
    """A text PDF with plenty of extractable text per page."""
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            f"Page {i} of the quarterly report. " * 12,  # well over the threshold
        )
    path = os.path.join(tempfile.mkdtemp(), "textual.pdf")
    doc.save(path)
    doc.close()
    return path


def _parse(path, logger):
    parser = ParserFactory(None).get_parser("application/pdf", path, logger)
    return parser, parser.parse()


# --- tests -------------------------------------------------------------------

def test_scanned_pdf_is_transcribed_via_vision_api(monkeypatch, caplog):
    """Each sparse page is sent to the vision chat-completions API as an image."""
    fake = _FakeVisionClient(content="Vision OCR: revenue grew 20%.")
    monkeypatch.setattr(dp, "_vision_client", lambda: (fake, "vision"))
    monkeypatch.setattr(dp, "_vision_available", lambda base_url, logger: True)

    import logging
    path = _scanned_pdf(pages=3)
    _, text = _parse(path, logging.getLogger("test"))

    create_calls = fake.chat.completions.calls
    # One vision API call per scanned page.
    assert len(create_calls) == 3, f"expected 3 OCR calls, got {len(create_calls)}"

    # Every call must carry the page image as a base64 data URI (i.e. it really
    # went through the vision API, not text generation).
    for call in create_calls:
        parts = call["messages"][0]["content"]
        image_parts = [p for p in parts if p.get("type") == "image_url"]
        assert image_parts, "vision API call missing image content"
        assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert call["model"] == "vision"

    # The transcription is included in the parser output.
    assert "Vision OCR: revenue grew 20%." in text


def test_scanned_pdf_skips_ocr_when_vision_unavailable(monkeypatch):
    """No mmproj / endpoint down → fast fallback, zero vision API calls."""
    fake = _FakeVisionClient()
    monkeypatch.setattr(dp, "_vision_client", lambda: (fake, "vision"))
    monkeypatch.setattr(dp, "_vision_available", lambda base_url, logger: False)

    import logging
    path = _scanned_pdf(pages=4)
    _, text = _parse(path, logging.getLogger("test"))

    assert fake.chat.completions.calls == [], "must not call vision API when unavailable"
    # Scanned pages have no text, so the result is empty — but it returns fast.
    assert text == ""


def test_text_pdf_does_not_use_vision_api(monkeypatch):
    """Text-rich PDFs use the fast extraction path with no vision API calls."""
    fake = _FakeVisionClient()
    monkeypatch.setattr(dp, "_vision_client", lambda: (fake, "vision"))
    # If the parser wrongly probed availability we'd still catch zero create calls,
    # but make availability True to prove the fast path skips OCR on its own.
    monkeypatch.setattr(dp, "_vision_available", lambda base_url, logger: True)

    import logging
    path = _text_pdf(pages=2)
    _, text = _parse(path, logging.getLogger("test"))

    assert fake.chat.completions.calls == [], "text PDF must not hit the vision API"
    assert "quarterly report" in text.lower()
