"""FastMCP-based fetch server – runs as a stdio subprocess."""

import logging
import sys

import httpx
import markdownify
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("fetch")

_STRIP_TAGS = ["script", "style", "noscript", "svg", "iframe", "link", "meta"]


def _html_to_markdown(html: str) -> str:
    """Convert HTML to markdown, stripping non-content elements first."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(_STRIP_TAGS):
            tag.decompose()
        # Also remove JSON-LD and other non-visible script-like blocks
        for tag in soup.find_all("script"):
            tag.decompose()
        cleaned = str(soup)
        md = markdownify.markdownify(cleaned, strip=["img"])
        # Collapse excessive blank lines
        import re
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md.strip()
    except Exception:
        return html


@mcp.tool()
async def fetch(
    url: str,
    max_length: int = sys.maxsize,
    start_index: int = 0,
    raw: bool = False,
) -> str:
    """Fetches a URL from the internet and extracts its contents as markdown.

    Args:
        url: URL to fetch.
        max_length: Maximum number of characters to return.
        start_index: Start content from this character index.
        raw: Get raw content without markdown conversion.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SmartloopFetch/1.0)",
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    text = response.text

    if not raw and "html" in content_type:
        text = _html_to_markdown(text)

    return text[start_index : start_index + max_length]


if __name__ == "__main__":
    mcp.run(transport="stdio")
