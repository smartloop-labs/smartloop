"""Document → Markdown conversion via MarkItDown.

MarkItDown handles fast, CPU-light text extraction for office formats (PDF,
DOCX, PPTX, XLSX, HTML).  For scanned / image-only PDF pages — where text
extraction yields little or nothing — the page is rendered to an image and
transcribed by the vision-capable model selected by
``ModelService.get_vision_llm()`` (the same ``"vision"`` capability the
orchestrator uses), called **in-process** — no HTTP, no OpenAI client.
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


def _vision_llm():
    """Return (vision_llm, model_service) for OCR, or (None, None).

    Resolves the multimodal model the same way the orchestrator does — via
    ``ModelService.get_vision_llm()`` — and runs in-process.  Returns
    ``(None, None)`` when vision is disabled or no usable vision model exists
    (e.g. mmproj GGUF missing), so callers fall back to plain text extraction.
    The model is loaded lazily/cached by ``get_vision_llm()`` on first use.
    """
    if not AppSettings().vision_enabled:
        return None, None
    try:
        from smartloop.api.services import get_model_service
        model_service = get_model_service()
        llm = model_service.get_vision_llm()
        return (llm, model_service) if llm is not None else (None, None)
    except Exception:
        return None, None


class DocumentParser(BaseParser):
    """Convert a document to markdown using MarkItDown (+ vision OCR for PDFs)."""

    def parse(self) -> str:
        logger = self.logger or logging.getLogger(__name__)
        if Path(self.file_path).suffix.lower() == ".pdf":
            return self._parse_pdf(logger)
        return self._parse_with_markitdown(logger)

    def _parse_with_markitdown(self, logger: logging.Logger) -> str:
        """Convert any supported document to markdown via MarkItDown."""
        try:
            result = MarkItDown().convert(self.file_path)
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

            vision_llm, model_service = _vision_llm()
            if vision_llm is None:
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
                    ocr = self._ocr_page(page, vision_llm, model_service, logger)
                    parts.append(ocr or page_texts[i])
                else:
                    parts.append(page_texts[i])
            return "\n\n".join(p for p in parts if p).strip()
        finally:
            doc.close()

    @staticmethod
    def _ocr_page(page, vision_llm, model_service, logger: logging.Logger) -> str:
        """Render a PDF page to PNG and transcribe it with the vision model.

        Calls the multimodal LLM directly (``create_chat_completion`` with an
        ``image_url`` data URI — how llama.cpp's vision handler takes images),
        serialized on the ModelService inference semaphore.
        """
        try:
            pix = page.get_pixmap(dpi=150)
            b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
            with model_service._inference_lock:
                resp = vision_llm.model.create_chat_completion(
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
            content = ((resp.get("choices") or [{}])[0]
                       .get("message", {}).get("content") or "")
            return content.strip()
        except Exception as e:
            logger.warning(f"Vision OCR failed for page {page.number}: {e}")
            return ""
