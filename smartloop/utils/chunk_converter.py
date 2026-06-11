"""Utility to convert chunk arrays into DocumentChunk objects"""
import uuid
import json
from typing import List, Dict, Any

from smartloop.parsers.document_chunk import DocumentChunk


def convert_chunks_to_document_chunks(chunks: List[Dict[str, Any]], document_id: str) -> List[DocumentChunk]:
    """
    Convert an array of chunk dictionaries into DocumentChunk objects

    Args:
        chunks: List of chunk dictionaries from chunking agent
        document_id: The document ID for all chunks

    Returns:
        List[DocumentChunk]: List of DocumentChunk objects
    """
    document_chunks = []

    for i, chunk_data in enumerate(chunks):
        # Extract content - handle different field names
        text = chunk_data.get('content', chunk_data.get('text', ''))

        # Handle title if present
        title = chunk_data.get('title', '')
        if title and title not in text:
            text = f"{title}\n\n{text}" if text else title

        # Create metadata from additional fields
        metadata = None
        chunk_id = str(uuid.uuid4())
        metadata_fields = dict(document=text, chunk_id=chunk_id)

        if 'title' in chunk_data:
            metadata_fields['title'] = chunk_data['title']
        if 'token_count' in chunk_data:
            metadata_fields['token_count'] = chunk_data['token_count']

        if metadata_fields:
            metadata = json.dumps(metadata_fields)

        # Create DocumentChunk
        document_chunk = DocumentChunk(
            document_id=document_id,
            paragraph_id=chunk_data.get('paragraph_id', str(uuid.uuid4())),
            chunk_id=chunk_id,
            total_chunks=len(chunks),
            text=text,
            metadata=metadata
        )

        document_chunks.append(document_chunk)

    return document_chunks
