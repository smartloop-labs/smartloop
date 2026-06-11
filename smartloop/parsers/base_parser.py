from abc import ABC, abstractmethod
import uuid
import math
import re
import os
import logging


from typing import Optional

from .document_chunk import DocumentChunk
from smartloop.llms import BaseLLM

class BaseParser(ABC):
    MAX_WORDS_PER_CHUNK = int(os.getenv("SLP_PARSER_MAX_WORDS_PER_CHUNK", 1024))

    def __init__(self, llm: Optional[BaseLLM], file_path: str, logger: logging.Logger):
        self.llm = llm
        self.file_path = file_path
        self.logger = logger

    @abstractmethod
    def parse(self) -> str:
        """Parse the document and return a DocumentChunk"""
        pass

    def calculate_number_of_chunks(self, total_items, max_items_per_chunk=MAX_WORDS_PER_CHUNK) -> int:
        # Int arithmetic to get the minimum number of chunks to hold a paragraph
        return math.ceil(total_items / max_items_per_chunk)

    def chunkify_paragraph(self, words: list) -> list[str]:
        chunk_count = self.calculate_number_of_chunks(len(words))

        chunks = []
        for i in range(chunk_count):
            start = i * self.MAX_WORDS_PER_CHUNK
            end = start + self.MAX_WORDS_PER_CHUNK
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)

        return chunks

    def sanitize(self, paragraph: str) -> str:
        # closure
        def convert_to_lower(match_obj):
            if match_obj.group() is not None:
                return match_obj.group().lower()

        paragraph = re.sub(r'\s+|(\.)', ' ', paragraph)
        paragraph = re.sub(r'\\t+', '', paragraph)
        paragraph = re.sub(r'(-)', ' ', paragraph)
        paragraph = re.sub(r'^\s+|\s+$', '', paragraph)
        paragraph = re.sub(r'[A-Z]', convert_to_lower, paragraph)

        return paragraph

    def create_chunk_from_document(self, document_id: str, document: str) -> DocumentChunk:
        return  DocumentChunk(
                    document_id=document_id,
                    chunk_id=0,
                    paragraph_id=str(uuid.uuid4()),
                    text=document,
                    total_chunks=1
                )

    def process_paragraphs(self, paragraphs: list[str], document_id: str = "") -> list[DocumentChunk]:
        """Strip down the paragraphs and break them into DocumentChunks"""
        document_chunks = []
        _paragraphs = []
        token_count = 0

        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            _paragraph = self.sanitize(paragraph)
            max_tokens = self.llm.get_max_tokens()

            if token_count + len(_paragraph)  < max_tokens:
                _paragraphs.append(_paragraph)
                token_count += self.llm.get_num_tokens(_paragraph)
            else:
                document_chunks.append(self.create_chunk_from_document(document_id, "\n".join(_paragraphs)))
                token_count = 0
                _paragraphs = [_paragraph]

        if len(_paragraphs) > 0:
               document_chunks.append(self.create_chunk_from_document(document_id, "\n".join(_paragraphs)))

        return document_chunks

    def normalize_whitespace(self, text_items: list[str]) -> str:
        """Strip leading/trailing space & multiple newlines also convert remaining newlines to spaces"""
        results = []

        for item in text_items:
            item = re.sub(r'\s+', ' ', item)

            # Replace remaining newlines with spaces
            item = item.replace('\n', ' ')
            item = item.strip()

            if item:
                results.append(item)

        return results
