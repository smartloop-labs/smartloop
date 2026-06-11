"""Lightweight metadata extraction for Chroma — no LLM required.

Parses the document filename with simple string/regex rules to produce
a flat dict of metadata stored alongside each chunk in Chroma.

The vector search itself handles entity-based queries (e.g. "Aaron Bergman")
by semantic similarity.  The metadata is only used for cheap pre-filtering
when an exact attribute match is possible (e.g. document_type=invoice).
"""

import re
from pathlib import Path

_DOC_TYPE_KEYWORDS = {
    "invoice", "receipt", "contract", "agreement",
    "report", "statement", "order", "quote", "proposal",
    "letter", "memo", "summary", "plan",
}

_AMOUNT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)")
_DATE_RE = re.compile(
    r"(\d{4}[-_]\d{2}[-_]\d{2}|\d{2}[-_]\d{2}[-_]\d{4}|\d{4})"
)


def extract_document_metadata(filename: str, content: str = "", llm=None) -> dict:
    """Return a flat metadata dict parsed from the filename.

    No LLM is used — extraction is pure string/regex so it adds zero
    latency to the indexing pipeline.

    Detected fields (all optional):
      document_type  — invoice, contract, report, etc.
      date           — YYYY-MM-DD or YYYY found in filename
      amount         — first numeric value found in filename
    """
    stem = Path(filename).stem
    # Normalise separators to spaces
    normalised = re.sub(r"[_\-]+", " ", stem)
    parts = normalised.split()

    meta: dict = {}

    # document_type
    for part in parts:
        if part.lower() in _DOC_TYPE_KEYWORDS:
            meta["document_type"] = part.lower()
            break

    # date
    date_match = _DATE_RE.search(stem)
    if date_match:
        meta["date"] = date_match.group(1).replace("_", "-")

    # amount (first standalone number that looks like a currency value)
    amount_match = _AMOUNT_RE.search(stem)
    if amount_match:
        val = amount_match.group(1).replace(",", "")
        try:
            meta["amount"] = float(val) if "." in val else int(val)
        except ValueError:
            pass

    return meta
