"""URL fetch agent — fetches a URL and returns its content as markdown.

Replaces the fetch MCP server.  Given a URL it performs the HTTP GET itself
and converts the response to markdown based on content type:

* HTML → BeautifulSoup (strip non-content tags) + markdownify.
* PDF  → MarkItDown via the shared ``PdfParser`` (same converter used for
  uploaded documents), so scanned PDFs still get vision OCR.
* anything else → returned as raw text.

It runs as an ordinary plan step (``StepType.FETCH``) resolved by the
orchestrator, not as an MCP tool.
"""

import logging
import os
import re
import sys
import tempfile

import httpx
import markdownify
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_STRIP_TAGS = ["script", "style", "noscript", "svg", "iframe", "link", "meta"]
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SmartloopFetch/1.0)"}
_URL_RE = re.compile(r'https?://[^\s<>\'")\]]+')

_JS_PHRASES = [
    "enable javascript", "javascript is required",
    "javascript is needed", "requires javascript",
    "turn on javascript", "javascript must be enabled",
    "please enable javascript",
]


class FetchAgent:
    """Fetch a URL and return its content converted to markdown."""

    def fetch(
        self,
        url: str,
        max_length: int = sys.maxsize,
        start_index: int = 0,
        raw: bool = False,
    ) -> str:
        """Fetch ``url`` and return its contents as markdown.

        Args:
            url: URL to fetch.
            max_length: Maximum number of characters to return.
            start_index: Start content from this character index.
            raw: Return raw content without markdown conversion.
        """
        try:
            with httpx.Client(follow_redirects=True, timeout=30) as client:
                response = client.get(url, headers=_HEADERS)
                response.raise_for_status()
        except Exception as e:
            logger.warning(f"[fetch] Fetch error for {url}: {e}")
            return ""

        content_type = response.headers.get("content-type", "").lower()

        if not raw and self._is_pdf(url, content_type):
            text = self._pdf_to_markdown(response.content)
        elif not raw and "html" in content_type:
            text = self._html_to_markdown(response.text)
            if self.is_javascript_page(response.text):
                logger.info(f"[fetch] JavaScript page detected: {url}")
                return (
                    f"The page at {url} requires JavaScript to render its content. "
                    "JavaScript-rendered pages cannot be processed."
                )
        else:
            text = response.text

        logger.info(f"[fetch] Fetched {len(text)} chars from {url}")
        return text[start_index : start_index + max_length]

    @staticmethod
    def _is_pdf(url: str, content_type: str) -> bool:
        return "application/pdf" in content_type or url.lower().split("?")[0].endswith(".pdf")

    @staticmethod
    def _html_to_markdown(html: str) -> str:
        """Convert HTML to markdown, stripping non-content elements first."""
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup.find_all(_STRIP_TAGS):
                tag.decompose()
            cleaned = str(soup)
            md = markdownify.markdownify(cleaned, strip=["img"])
            md = re.sub(r"\n{3,}", "\n\n", md)
            return md.strip()
        except Exception:
            return html

    @staticmethod
    def _pdf_to_markdown(data: bytes) -> str:
        """Convert PDF bytes to markdown via the shared PdfParser (MarkItDown)."""
        # Imported lazily so a plain HTML fetch never pulls in the PDF stack.
        from smartloop.parsers.pdf_parser import PdfParser

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            return PdfParser(llm=None, file_path=tmp_path, logger=logger).parse()
        except Exception as e:
            logger.warning(f"[fetch] PDF conversion failed: {e}")
            return ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def is_javascript_page(content: str) -> bool:
        """True when the page needs JavaScript to render its content."""
        if not content:
            return False
        soup = BeautifulSoup(content, "html.parser")
        noscript = soup.find("noscript")
        if noscript:
            noscript_text = noscript.get_text(strip=True).lower()
            if any(phrase in noscript_text for phrase in _JS_PHRASES):
                return True
        body = soup.find("body") or soup
        visible_text = body.get_text(separator=" ", strip=True)
        scripts = soup.find_all("script")
        if len(scripts) > 0 and len(visible_text) < 200:
            return True
        root_div = soup.find("div", id=re.compile(r"^(root|app|__next)$"))
        if root_div and not root_div.get_text(strip=True):
            return True
        return False
