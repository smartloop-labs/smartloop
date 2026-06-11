"""Process DocumentChunks from Lance into Pydantic DocumentChunks & ContextItems"""
import logging
import json
from typing import List, Any

from smartloop.llms.base_llm import BaseLLM

from .document_chunk import DocumentChunk
from .context_item import ContextItem

class DocumentChunkProcessor:
    """Process DocumentChunks from Lance into Pydantic DocumentChunks & ContextItems"""
    document_chunk_models: List[Any] = []
    llm: BaseLLM

    def __init__(self, llm: BaseLLM, document_chunk_models: List[Any], logger: logging.Logger):
        print("initializing DocumentChunkProcessor...")
        self.logger = logger
        self.llm = llm
        self.document_chunk_models = document_chunk_models or []

        self.logger.info("DocumentChunkProcessor: initialized with the following chunks:")
        for chunk in self.document_chunk_models:
            self.logger.info("Chunk document_id: %s -- distance: %s",chunk.document_id, chunk.distance)
            self.logger.debug("DOCUMENT CHUNK TEXT: %s", chunk.text)


    def get_document_chunks(self) -> List[DocumentChunk]:
        """Converts all LanceDB models into Pydantic DocumentChunks"""
        self.logger.info("Getting get_document_chunks()")
        results = []

        for model in self.document_chunk_models:
            results.append(DocumentChunk(
                document_id = model.document_id,
                paragraph_id = model.paragraph_id,
                chunk_id = model.chunk_id,
                total_chunks = model.total_chunks,
                text = model.text,
                vector = model.vector,
                distance = model.distance
            ))

        return results


    def get_chunks_as_dicts(self) -> List[dict]:
        """Return all DocumentChunks into a List of dicts"""
        self.logger.info("Getting chunks as DICT for models")
        return [ chunk.model_dump() for chunk in self.get_document_chunks() ]

    def get_context_items(self) -> List[ContextItem]:
        """Combine all DocumentChunks into ContextItems that keep paragraphs contiguous"""
        self.logger.info("Getting get_combined_chunks()")
        results = []

        for chunk in self.document_chunk_models:
            self.logger.debug("Chunk document_id: %s -- paragraph_id: %s -- chunk_id: %s", chunk.document_id, chunk.paragraph_id, chunk.chunk_id)
            self.logger.debug("Chunk text: %s", chunk.text)

            results.append(ContextItem(
                document_id=chunk.document_id,
                paragraph_id=chunk.paragraph_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                token_count=self.llm.get_num_tokens(chunk.text)
            ))

        return results

