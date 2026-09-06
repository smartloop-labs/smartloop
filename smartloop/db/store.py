from abc import ABC, abstractmethod
from typing import List, Optional
import logging
import os
import re

from pydantic import BaseModel

# Keyword-coverage boost shared by all dense-embedding stores.  Dense
# embeddings cluster similar documents together (e.g. all invoices), so a
# query naming a specific entity ("Adam Hart") pulls in look-alikes
# ("Aaron Hawkins").  Rewarding documents by the NUMBER of distinct query
# keywords they contain pushes the more complete match to the top.
_PER_TERM_BOOST = 0.15   # distance reduction per distinct query term matched
_MAX_TERM_BOOST = 0.60   # cap so a long query can't zero out the distance
_TERM_RE = re.compile(r"\w+")


def keyword_coverage_boost(query_text: str, name: str, content: str) -> float:
    """Return the distance reduction for a document based on how many
    distinct query keywords appear in its name + content.

    Subtract the returned value from a cosine distance to boost documents
    that match more of the query (clamped to ``_MAX_TERM_BOOST``).
    """
    query_tokens = {t.lower() for t in _TERM_RE.findall(query_text or "") if len(t) > 2}
    if not query_tokens:
        return 0.0
    haystack = f"{(name or '').replace('_', ' ')} {content or ''}".lower()
    doc_words = set(_TERM_RE.findall(haystack))
    matched = len(query_tokens & doc_words)
    if not matched:
        return 0.0
    return min(_MAX_TERM_BOOST, _PER_TERM_BOOST * matched)


class ChunkResult(BaseModel):
    """A single vector search result with text and source metadata.

    All citation data (document_name, page_number, chunk_index, etc.)
    is stored in the ``metadata`` dict — no separate citation field needed.
    """
    text: str
    document_id: str
    document_name: Optional[str] = None
    distance: float = 0.0
    metadata: dict = {}
    source_content: Optional[str] = None
    # This chunk's embedding, as the raw BLOB the index stores.  Carried out of
    # the store so callers that need to compare the chunk against something
    # else — citation verification checking it against the finished answer —
    # can do it without paying to encode text that was already encoded at
    # index time.  ``None`` from the TF-IDF fallback path, which has no
    # vectors.  Kept as bytes rather than an array so ``ChunkResult`` stays a
    # plain pydantic model; ``blob_to_embedding`` decodes it.
    embedding: Optional[bytes] = None


class VectorStore(ABC):
    """Abstract base for vector stores."""

    def __init__(
        self,
        project_dir: str,
        collection_name: str = "documents",
        logger: logging.Logger = None,
        base_dir: str = None,
    ):
        self.project_dir = project_dir
        self.collection_name = collection_name
        self.logger = logger or logging.getLogger(__name__)
        self.base_dir = base_dir

    @abstractmethod
    def add_chunks(self, chunks: List[dict]):
        """Add document chunks with 'text', 'document_id', and optional 'chunk_index'."""
        pass

    @abstractmethod
    def query(self, query_text: str, n_results: int = 5) -> List[str]:
        """Semantic search, returns top matching texts."""
        pass

    def query_with_metadata(
        self, query_text: str, n_results: int = 5, max_distance: float = 0.8,
        where: Optional[dict] = None,
    ) -> List[ChunkResult]:
        """Semantic search returning structured results with source metadata.

        Default implementation delegates to query() with no metadata.
        Subclasses should override for richer results.
        """
        texts = self.query(query_text, n_results)
        return [ChunkResult(text=t, document_id="", distance=0.0) for t in texts]

    def get_by_document_ids(self, document_ids: List[str]) -> List["ChunkResult"]:
        """Fetch all chunks for the given document IDs (no semantic query)."""
        return []

    def count(self) -> int:
        """Return the number of stored chunks in this collection.

        Default implementation returns 0; subclasses backed by a real store
        should override.
        """
        return 0

    @abstractmethod
    def delete_document(self, document_id: str):
        """Remove all chunks for a document."""
        pass
