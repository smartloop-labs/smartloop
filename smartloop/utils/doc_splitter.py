import re
from logging import Logger
from typing import List

from langchain_text_splitters import MarkdownHeaderTextSplitter
from smartloop.parsers.parser_factory import BaseLLM
from smartloop.workspace import Document

max_tokens = 512  # max tokens per chunk — set lower than model limit to allow room for prompt

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
