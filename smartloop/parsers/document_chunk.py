import uuid

from pydantic import BaseModel, Field
from typing import Optional

# Stores one "chunk" of a paragraph's text up to 500 words based on current config
# If there are more words then the paragraph text will flow into multiple chunks
# all tied back to the paragraph_id
class DocumentChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # All chunks have a unique id
    document_id: str
    paragraph_id: str
    chunk_id: str
    total_chunks: int
    metadata: Optional[str] = None  # contains summary and keyword
    text: str
    distance: Optional[float] = 0.0
    score: Optional[float] = 0.0
    vector: Optional[list[float]] = None

    def to_dict(self):
        return self.model_dump()
