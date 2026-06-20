"""Document → Markdown conversion via MarkItDown.

MarkItDown handles fast, CPU-light text extraction for office formats (PDF,
DOCX, PPTX, XLSX, HTML).  OCR / image understanding is delegated to the
vision-capable model served at the local ``/vision/v1`` endpoint (selected by
the ``"vision"`` capability, like the orchestrator).  For scanned / image-only
PDF pages — where text extraction yields little or nothing — the page is
rendered to an image and transcribed by the vision model so extraction is
maximized.
"""

import base64
import logging
from pathlib import Path

from markitdown import MarkItDown

from .base_parser import BaseParser
from smartloop.config.app_settings import AppSettings

_OCR_PROMPT = (
    "Transcribe ALL text and tables from this document page into clean GitHub-"
    "flavored Markdown. Preserve headings, lists and table structure. Output "
    "only the transcription, with no commentary."
)


def _vision_client():
    """Return (OpenAI client, model_name) for the local vision endpoint.

    Returns ``(None, None)`` when vision is disabled or the OpenAI client can't
    be constructed — callers then fall back to plain text extraction.
    """
    settings = AppSettings()
    if not settings.vision_enabled:
        return None, None
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=settings.vision_base_url, api_key="local",
            timeout=600.0, max_retries=1,
        )
        return client, "vision"
    except Exception:
        return None, None


def _vision_available(base_url: str, logger: logging.Logger) -> bool:
    """One-shot probe of the /vision/v1/health endpoint.

    Returns False when no multimodal model is usable (e.g. mmproj GGUF missing),
    so the caller skips the per-page OCR loop instead of attempting — and
    failing — on every page.  The health call triggers the (lazy) model load,
    which can take a while the first time; allow a generous timeout.
    """
    try:
        import httpx
        resp = httpx.get(f"{base_url}/health", timeout=600.0)
        resp.raise_for_status()
        return bool(resp.json().get("available"))
    except Exception as e:
        logger.warning(f"Vision endpoint unavailable, skipping OCR: {e}")
        return False


class DocumentParser(BaseParser):
    """Convert a document to markdown using MarkItDown (+ vision OCR for PDFs)."""

    def parse(self) -> str:
        logger = self.logger or logging.getLogger(__name__)
        if Path(self.file_path).suffix.lower() == ".pdf":
            return self._parse_pdf(logger)
        return self._parse_with_markitdown(logger)

    def _parse_with_markitdown(self, logger: logging.Logger) -> str:
        """Convert any supported document to markdown via MarkItDown."""
        client, vision_model = _vision_client()
        # Standalone-image description is handled by MarkItDown when given an
        # llm_client; harmless to pass for all types.
        md = (MarkItDown(llm_client=client, llm_model=vision_model)
              if client is not None else MarkItDown())
        try:
            result = md.convert(self.file_path)
            return (result.text_content or "").strip()
        except Exception as e:
            logger.warning(f"MarkItDown conversion failed for {self.file_path}: {e}")
            return ""

    def _parse_pdf(self, logger: logging.Logger) -> str:
        """Parse a PDF, OCR'ing scanned/sparse pages via the vision model.

        PyMuPDF probes per-page text *first* (milliseconds).  Text-rich PDFs go
        through MarkItDown for the best layout.  Scanned / image-only PDFs skip
        MarkItDown entirely — its pdfminer image pass can take a minute and
        yields nothing — and are transcribed per page by the vision model
        instead (or returned text-only when no vision model is available).
        """
        try:
            import fitz  # PyMuPDF
        except Exception as e:
            logger.warning(f"PyMuPDF unavailable, using MarkItDown only: {e}")
            return self._parse_with_markitdown(logger)

        try:
            doc = fitz.open(self.file_path)
        except Exception as e:
            logger.warning(f"Could not open PDF, using MarkItDown only: {e}")
            return self._parse_with_markitdown(logger)

        name = Path(self.file_path).name
        min_chars = AppSettings().vision_ocr_min_chars_per_page
        try:
            page_texts = [page.get_text("text").strip() for page in doc]
            sparse = [i for i, t in enumerate(page_texts) if len(t) < min_chars]

            if not sparse:
                # Text-rich PDF — MarkItDown gives richer markdown (tables).
                return self._parse_with_markitdown(logger)

            client, vision_model = _vision_client()
            if client is None or not _vision_available(
                    AppSettings().vision_base_url, logger):
                logger.warning(
                    f"{len(sparse)}/{len(page_texts)} scanned page(s) in {name} but "
                    f"vision OCR is unavailable — returning text-only extraction"
                )
                return "\n\n".join(t for t in page_texts if t).strip()

            logger.info(
                f"Vision OCR: {len(sparse)}/{len(page_texts)} scanned page(s) in {name}"
            )
            parts = []
            for i, page in enumerate(doc):
                if i in sparse:
                    ocr = self._ocr_page(page, client, vision_model, logger)
                    parts.append(ocr or page_texts[i])
                else:
                    parts.append(page_texts[i])
            return "\n\n".join(p for p in parts if p).strip()
        finally:
            doc.close()

    @staticmethod
    def _ocr_page(page, client, model: str, logger: logging.Logger) -> str:
        """Render a PDF page to PNG and transcribe it with the vision model."""
        try:
            pix = page.get_pixmap(dpi=150)
            b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _OCR_PROMPT},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }],
                max_tokens=2048,
                temperature=0.0,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning(f"Vision OCR failed for page {page.number}: {e}")
            return ""
