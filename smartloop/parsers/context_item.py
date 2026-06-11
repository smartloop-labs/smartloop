"""Validation model"""
from pydantic import BaseModel

class ContextItem(BaseModel):
    """Object combining the text of all DocumentChunks for a paragraph"""
    document_id: str
    chunk_id: str
    paragraph_id: str
    text: str
    token_count: int
