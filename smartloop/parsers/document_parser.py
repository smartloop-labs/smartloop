"""Document & image → Markdown conversion via MarkItDown.

MarkItDown handles fast, CPU-light text extraction for office formats (PDF,
DOCX, PPTX, XLSX, HTML).  Image understanding / OCR is delegated to a
vision-capable model selected by ``ModelService.get_vision_llm()`` (the same
``"vision"`` capability the orchestrator uses), reached through a small
in-process client that speaks MarkItDown's OpenAI ``llm_client`` interface — no
HTTP, no separate endpoint:

* Standalone images (PNG/JPEG) and images embedded in office docs are
  described/transcribed by MarkItDown's image converter via that client.
* Scanned / image-only PDF pages — which MarkItDown only runs text extraction
  over — are rendered to images with PyMuPDF and transcribed page-by-page
  through the same client.
"""

import base64
import logging
from pathlib import Path
from types import SimpleNamespace

from markitdown import MarkItDown

from .base_parser import BaseParser
from smartloop.config.app_settings import AppSettings

_OCR_PROMPT = (
    "Transcribe ALL text and tables from this document page into clean GitHub-"
    "flavored Markdown. Preserve headings, lists and table structure. Output "
    "only the transcription, with no commentary."
)
_IMAGE_PROMPT = (
    "Describe this image. Transcribe any text, tables or diagrams it contains "
    "into clean GitHub-flavored Markdown."
)

# Page-render DPI and output budget for PDF OCR.  A full page image consumes a
# large share of the vision model's (small) context — at 150 DPI a letter page
# is ~2800 image tokens, which alone overflows a 2048 ctx.  96 DPI keeps a page
# near ~1150 tokens so it plus the transcription fit within vision_n_ctx (4096)
# with room to spare, and shrinks the patch grid fed to the native clip encoder
# (a smaller grid avoids the slice-encode segfault seen on large pages), while
# staying legible for OCR.
_OCR_DPI = 96
_OCR_MAX_TOKENS = 1536
# Cap the longest edge of rendered page images so the CLIP patch grid stays
# small (~10x14 patches for a letter page).  A 26x37 grid (962 patches at
# 96 DPI uncapped) dominates the vision-encode time; 560 px keeps it under
# ~200 patches while remaining legible for OCR.
_OCR_MAX_PIXELS = 560


class _VisionChatCompletions:
    """``chat.completions`` over the in-process vision LLM (OpenAI-shaped).

    This is the interface MarkItDown's image converter calls
    (``client.chat.completions.create(model=, messages=[...])``), so the same
    object backs both MarkItDown image OCR and the PDF page-OCR loop.  The
    vision model is resolved lazily on first ``create()`` so parsing a
    text-only office doc never forces the (heavy) vision model to load.
    """

    def __init__(self, model_service):
        self._model_service = model_service

    def create(self, *, messages, model=None, max_tokens=2048, temperature=0.0, **_):
        import time
        llm = self._model_service.get_vision_llm()
        if llm is None:
            raise RuntimeError("No vision-capable model available")
        n_gpu = llm.model.model_params.n_gpu_layers
        logging.getLogger(__name__).info(
            "[vision-ocr] create_chat_completion n_gpu_layers=%s", n_gpu,
        )
        t0 = time.monotonic()
        with self._model_service._inference_lock:
            result = llm.model.create_chat_completion(
                messages=messages, max_tokens=max_tokens,
                temperature=temperature, stream=False,
            )
        elapsed = time.monotonic() - t0
        logging.getLogger(__name__).info(
            "[vision-ocr] completed in %.1fs", elapsed,
        )
        content = ((result.get("choices") or [{}])[0]
                   .get("message", {}).get("content") or "")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _VisionChatClient:
    """Minimal OpenAI-client stand-in backed by the in-process vision model."""

    def __init__(self, model_service):
        self.chat = SimpleNamespace(completions=_VisionChatCompletions(model_service))


def _vision_client():
    """Return (vision client, model_name) for OCR, or (None, None).

    The client speaks MarkItDown's OpenAI ``llm_client`` interface but runs the
    local multimodal model in-process.  Returns ``(None, None)`` when vision is
    disabled or the ModelService singleton can't be reached.  The client is
    lazy: no vision model is loaded until an image is actually transcribed.
    """
    if not AppSettings().vision_enabled:
        return None, None
    try:
        from smartloop.api.services import get_model_service
        return _VisionChatClient(get_model_service()), "vision"
    except Exception:
        return None, None


def _vision_available(logger: logging.Logger) -> bool:
    """True when a multimodal vision model is loaded/loadable for OCR.

    Triggers the (lazy, cached) model load via ``get_vision_llm()`` so the PDF
    path can skip the per-page OCR loop and fall back to text-only extraction
    when no usable vision model exists (e.g. mmproj GGUF missing).
    """
    try:
        from smartloop.api.services import get_model_service
        return get_model_service().get_vision_llm() is not None
    except Exception as e:
        logger.warning(f"Vision model unavailable, skipping OCR: {e}")
        return False


class DocumentParser(BaseParser):
    """Convert a document or image to markdown via MarkItDown (+ PDF vision OCR)."""

    def parse(self) -> str:
        logger = self.logger or logging.getLogger(__name__)
        if Path(self.file_path).suffix.lower() == ".pdf":
            return self._parse_pdf(logger)
        return self._parse_with_markitdown(logger)

    def _parse_with_markitdown(self, logger: logging.Logger) -> str:
        """Convert a document/image to markdown via MarkItDown.

        When a vision client is available it is handed to MarkItDown as the
        ``llm_client`` so images (standalone files and images embedded in office
        docs) are described/transcribed by the vision model.  The client is
        lazy, so a text-only document never loads the vision model.
        """
        client, vision_model = _vision_client()
        md = (MarkItDown(llm_client=client, llm_model=vision_model,
                         llm_prompt=_IMAGE_PROMPT)
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
        MarkItDown entirely — its pdfminer pass yields nothing on images — and
        are transcribed per page by the vision model instead (or returned
        text-only when no vision model is available).
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
            if client is None or not _vision_available(logger):
                logger.warning(
                    f"{len(sparse)}/{len(page_texts)} scanned page(s) in {name} but "
                    f"vision OCR is unavailable — returning text-only extraction"
                )
                return "\n\n".join(t for t in page_texts if t).strip()

            logger.info(
                f"Vision OCR: {len(sparse)}/{len(page_texts)} scanned page(s) in {name}"
            )

            sparse_set = set(sparse)
            parts = []
            for i in range(len(page_texts)):
                if i in sparse_set:
                    ocr = self._ocr_page(doc[i], client, vision_model, logger)
                    parts.append(ocr or page_texts[i])
                else:
                    parts.append(page_texts[i])
            return "\n\n".join(p for p in parts if p).strip()
        finally:
            doc.close()

    @staticmethod
    def _ocr_page(page, client, model: str, logger: logging.Logger) -> str:
        """Render a single PDF page to a capped-size PNG and transcribe it.

        The render DPI is reduced so the longest edge stays at most
        ``_OCR_MAX_PIXELS`` px.  This keeps the CLIP patch grid small
        (e.g. ~15x20 instead of 26x37), which is the dominant cost in
        the vision encode step.
        """
        try:
            rect = page.rect
            long_edge_pts = max(rect.width, rect.height)
            base_px = long_edge_pts * _OCR_DPI / 72.0
            if base_px > _OCR_MAX_PIXELS:
                dpi = int(_OCR_DPI * _OCR_MAX_PIXELS / base_px)
            else:
                dpi = _OCR_DPI
            pix = page.get_pixmap(dpi=dpi)
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
                max_tokens=_OCR_MAX_TOKENS,
                temperature=0.0,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning(f"Vision OCR failed for page {page.number}: {e}")
            return ""
