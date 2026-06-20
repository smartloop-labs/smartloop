"""Validate that scanned (image-only) PDFs are OCR'd through the vision model.

The document parser extracts text with MarkItDown for normal PDFs, but a
scanned / image-only PDF has no extractable text.  In that case each sparse
page must be rendered to an image and transcribed by the vision-capable model
selected by ``ModelService.get_vision_llm()`` — called **in-process** via
``create_chat_completion`` with an ``image_url`` data URI.  These tests assert
that path is exercised — and that it is NOT exercised for text PDFs or when no
vision model is available.
"""
import contextlib
import os
import tempfile
from types import SimpleNamespace

import pytest

fitz = pytest.importorskip("fitz")  # PyMuPDF
pytest.importorskip("markitdown")

from smartloop.parsers import document_parser as dp
from smartloop.parsers.parser_factory import ParserFactory


# --- fakes for the in-process vision model -----------------------------------

class _FakeVisionLLM:
    """Records create_chat_completion() calls; returns a canned transcription."""

    def __init__(self, content):
        self._content = content
        self.calls = []
        # The parser reaches the completion via ``vision_llm.model``.
        self.model = SimpleNamespace(create_chat_completion=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeModelService:
    """Minimal stand-in: just the inference semaphore the parser locks on."""

    _inference_lock = contextlib.nullcontext()


def _patch_vision(monkeypatch, llm):
    """Make ``_vision_llm()`` return (llm, fake_service), or (None, None)."""
    service = _FakeModelService()
    monkeypatch.setattr(
        dp, "_vision_llm",
        lambda: (llm, service) if llm is not None else (None, None),
    )


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

def test_scanned_pdf_is_transcribed_via_vision_model(monkeypatch):
    """Each sparse page is sent to the vision model as an image."""
    llm = _FakeVisionLLM(content="Vision OCR: revenue grew 20%.")
    _patch_vision(monkeypatch, llm)

    import logging
    path = _scanned_pdf(pages=3)
    _, text = _parse(path, logging.getLogger("test"))

    # One vision completion per scanned page.
    assert len(llm.calls) == 3, f"expected 3 OCR calls, got {len(llm.calls)}"

    # Every call must carry the page image as a base64 data URI (i.e. it really
    # went through the multimodal model, not text generation).
    for call in llm.calls:
        parts = call["messages"][0]["content"]
        image_parts = [p for p in parts if p.get("type") == "image_url"]
        assert image_parts, "vision call missing image content"
        assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")

    # The transcription is included in the parser output.
    assert "Vision OCR: revenue grew 20%." in text


def test_scanned_pdf_skips_ocr_when_vision_unavailable(monkeypatch):
    """No vision model → fast fallback, zero vision calls."""
    _patch_vision(monkeypatch, None)

    import logging
    path = _scanned_pdf(pages=4)
    _, text = _parse(path, logging.getLogger("test"))

    # Scanned pages have no text, so the result is empty — but it returns fast.
    assert text == ""


def test_text_pdf_does_not_use_vision_model(monkeypatch):
    """Text-rich PDFs use the fast extraction path with no vision calls."""
    llm = _FakeVisionLLM(content="should not be called")
    _patch_vision(monkeypatch, llm)

    import logging
    path = _text_pdf(pages=2)
    _, text = _parse(path, logging.getLogger("test"))

    assert llm.calls == [], "text PDF must not invoke the vision model"
    assert "quarterly report" in text.lower()
