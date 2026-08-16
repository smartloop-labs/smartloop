import json
import re
from logging import Logger
from typing import Any, Dict, List, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter
from smartloop.parsers.parser_factory import BaseLLM
from smartloop.workspace.types import Document

max_tokens = 512  # max tokens per chunk — set lower than model limit to allow room for prompt


def json_to_table(text: str) -> Optional[str]:
    """Convert a JSON array-of-objects document into a markdown table.

    Handles a top-level array or an object holding one under a common key
    (``rows``, ``data``, ``items``, ``results``, ``players``, ``values``).
    Each object becomes a table row; columns are the union of keys, with
    nested objects flattened using dot-prefixed keys (e.g. a player's
    ``"values": {"epa": 0.3, ...}`` becomes ``values.epa``).  Floats are
    rounded to 4 decimals and integral floats shown without the decimal
    so dense stats stay compact.

    Returns ``None`` when the text isn't JSON or has no tabular structure,
    so callers fall back to text splitting / LLM extraction.  Deterministic
    and context-window-independent — a large data file becomes a table
    without ever loading it into a model (whose window overflows for big
    documents, silently killing extraction).
    """
    _ROW_CAP = 500
    try:
        data = json.loads(text)
    except Exception:
        return None

    rows = data if isinstance(data, list) else None
    if isinstance(data, dict):
        for key in ("rows", "data", "items", "results", "players", "values"):
            candidate = data.get(key)
            if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
                rows = candidate
                break
    if not isinstance(rows, list):
        return None

    flat_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        flat: Dict[str, Any] = {}

        def _flatten(obj: Dict[str, Any], prefix: str = "") -> None:
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _flatten(v, key)
                else:
                    flat[key] = v

        _flatten(row)
        flat_rows.append(flat)
    if not flat_rows:
        return None

    total_rows = len(flat_rows)
    truncated = total_rows > _ROW_CAP
    flat_rows = flat_rows[:_ROW_CAP]
    columns = list(dict.fromkeys(k for r in flat_rows for k in r))

    def _cell(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, float):
            return str(int(v)) if v.is_integer() else f"{v:.4f}"
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        return str(v)

    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for r in flat_rows:
        lines.append("| " + " | ".join(_cell(r.get(c)) for c in columns) + " |")
    if truncated:
        lines.append(f"\n_({_ROW_CAP} of {total_rows} rows shown)_")
    return "\n".join(lines)


def is_markdown_table(text: str) -> bool:
    """True when the text contains a complete markdown table.

    Detection is the header separator row (e.g. ``|---|---|``) — a line of
    only dashes/colons/pipes/spaces.  Content that is already a table needs
    no LLM extraction: it's the structured representation a data query wants,
    and feeding it through the extractor risks a context-window overflow.
    """
    return bool(re.search(r'(?m)^\|\s*[-:|]+\s*\|$', text))

class DocumentSplitter:
    """Utility class for splitting documents into chunks."""

    def __init__(self, llm: BaseLLM,  logger: Logger):
        self.llm = llm
        self.logger = logger

    def split_document(self, document: Document) -> List[str]:
        """Utility function to generate text chunks from a document without creating a full dataset."""
        with open(document.path, 'r', encoding='utf-8') as f:
            document_text = f.read()
        return self._split_document_text(document_text)

    def split_text(self, text: str) -> List[str]:
        """Split already-converted markdown text into chunks."""
        return self._split_document_text(text)

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a given text using the LLM's tokenizer."""
        return self.llm.get_num_tokens(text)

    def _split_oversized(self, text: str, limit: int) -> List[str]:
        """Split text that exceeds *limit* tokens by paragraphs, then by lines."""
        paragraphs = re.split(r'\n{2,}', text)
        chunks: List[str] = []
        current: List[str] = []
        current_count = 0

        for para in paragraphs:
            para_count = self._count_tokens(para)
            if current and current_count + para_count > limit:
                chunks.append("\n\n".join(current))
                current = []
                current_count = 0

            if para_count > limit:
                # paragraph itself exceeds limit — split by lines
                lines = para.split("\n")
                line_buf: List[str] = []
                line_count = 0
                for line in lines:
                    lc = self._count_tokens(line)
                    if line_buf and line_count + lc > limit:
                        chunks.append("\n".join(line_buf))
                        line_buf = []
                        line_count = 0
                    line_buf.append(line)
                    line_count += lc
                if line_buf:
                    current.append("\n".join(line_buf))
                    current_count += line_count
            else:
                current.append(para)
                current_count += para_count

        if current:
            chunks.append("\n\n".join(current))

        return [c.strip() for c in chunks if c.strip()]


    def _split_document_text(self, document_text: str) -> List[str]:
        """Split document text into chunks for vector indexing.

        Indexes the entire document as a single chunk when it fits within
        the token budget.  Only splits into smaller pieces when the full
        text exceeds ``max_tokens``, using paragraph boundaries to keep
        related content together.

        Args:
            document_text: The raw document text.

        Returns:
            List of text chunks.
        """
        # Strip image placeholders
        cleaned = re.sub(r'<!--\s*image\s*-->', '', document_text, flags=re.IGNORECASE).strip()
        if not cleaned:
            return []

        # Keep as single chunk when it fits
        if self._count_tokens(cleaned) <= max_tokens:
            return [cleaned]

        # Otherwise split by paragraphs/lines to stay within budget
        return self._split_oversized(cleaned, max_tokens)
