"""Validate that scanned PDFs and images are OCR'd through the vision model.

The document parser extracts text with MarkItDown for normal docs.  Images and
scanned / image-only PDF pages carry no extractable text, so they are sent to
the vision-capable model (selected by ``ModelService.get_vision_llm()``) through
an in-process client that speaks MarkItDown's OpenAI ``llm_client`` interface:

* images go through MarkItDown's image converter, and
* scanned PDF pages are rendered with PyMuPDF and transcribed page-by-page.

These tests assert those paths are exercised — and that they are NOT exercised
for text PDFs or when no vision model is available.
"""
import logging
import os
import tempfile
from types import SimpleNamespace

import pytest

fitz = pytest.importorskip("fitz")  # PyMuPDF
pytest.importorskip("markitdown")

from smartloop.parsers import document_parser as dp
from smartloop.parsers.parser_factory import ParserFactory


# --- fakes for the in-process vision client ----------------------------------

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
    """OpenAI-``llm_client``-shaped fake (what MarkItDown / _ocr_page call)."""

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


def _png() -> str:
    """A small standalone PNG image (no text layer — only the vision model
    can describe it)."""
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.draw_rect(fitz.Rect(20, 20, 180, 180), fill=(0.2, 0.4, 0.6))
    pix = page.get_pixmap()
    path = os.path.join(tempfile.mkdtemp(), "image.png")
    pix.save(path)
    doc.close()
    return path


def _parse(mime, path):
    parser = ParserFactory(None).get_parser(mime, path, logging.getLogger("test"))
    return parser, parser.parse()


# --- tests -------------------------------------------------------------------

def test_scanned_pdf_is_transcribed_via_vision_model(monkeypatch):
    """Each sparse page is sent to the vision model as an image."""
    fake = _FakeVisionClient(content="Vision OCR: revenue grew 20%.")
    monkeypatch.setattr(dp, "_vision_client", lambda: (fake, "vision"))
    monkeypatch.setattr(dp, "_vision_available", lambda logger: True)

    path = _scanned_pdf(pages=3)
    _, text = _parse("application/pdf", path)

    create_calls = fake.chat.completions.calls
    # One vision call per scanned page.
    assert len(create_calls) == 3, f"expected 3 OCR calls, got {len(create_calls)}"

    # Every call must carry the page image as a base64 data URI.
    for call in create_calls:
        parts = call["messages"][0]["content"]
        image_parts = [p for p in parts if p.get("type") == "image_url"]
        assert image_parts, "vision call missing image content"
        assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")

    assert "Vision OCR: revenue grew 20%." in text


def test_scanned_pdf_skips_ocr_when_vision_unavailable(monkeypatch):
    """No vision model → fast fallback, zero vision calls."""
    fake = _FakeVisionClient()
    monkeypatch.setattr(dp, "_vision_client", lambda: (fake, "vision"))
    monkeypatch.setattr(dp, "_vision_available", lambda logger: False)

    path = _scanned_pdf(pages=4)
    _, text = _parse("application/pdf", path)

    assert fake.chat.completions.calls == [], "must not call vision when unavailable"
    # Scanned pages have no text, so the result is empty — but it returns fast.
    assert text == ""


def test_text_pdf_does_not_use_vision_model(monkeypatch):
    """Text-rich PDFs use the fast extraction path with no vision calls."""
    fake = _FakeVisionClient()
    monkeypatch.setattr(dp, "_vision_client", lambda: (fake, "vision"))
    monkeypatch.setattr(dp, "_vision_available", lambda logger: True)

    path = _text_pdf(pages=2)
    _, text = _parse("application/pdf", path)

    assert fake.chat.completions.calls == [], "text PDF must not invoke the vision model"
    assert "quarterly report" in text.lower()


def test_image_is_described_via_vision_model(monkeypatch):
    """A standalone image is OCR'd/described through MarkItDown's image
    converter, backed by the in-process vision client."""
    fake = _FakeVisionClient(content="A blue square. Text: HELLO WORLD")
    monkeypatch.setattr(dp, "_vision_client", lambda: (fake, "vision"))

    path = _png()
    _, text = _parse("image/png", path)

    # The image reached the vision model as an image_url part...
    calls = fake.chat.completions.calls
    assert calls, "image must be sent to the vision model"
    parts = calls[0]["messages"][0]["content"]
    assert any(p.get("type") == "image_url" for p in parts)
    # ...and its description is in the parser output.
    assert "A blue square. Text: HELLO WORLD" in text
