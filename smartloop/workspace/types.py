"""Lightweight workspace types shared between the workspace and model layers.

Kept in a separate module so the model layer (``smartloop.model.generic``, etc.)
can import these without triggering the circular dependency that arises when
``smartloop.workspace`` (which imports ``supported_models``) is imported during
model-class loading.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict


class DocumentStatus(Enum):
    PENDING = "pending"
    CONVERTING = "converting"
    INDEXING = "indexing"
    FAILED = "failed"
    INDEXED = "indexed"
    DELETED = "deleted"
    PROCESSED = "processed"


class Document(BaseModel):
    model_config = ConfigDict(json_schema_extra={"exclude_none": True})

    id: str
    path: str
    name: Optional[str] = None  # original filename — set once, never overwritten by indexer
    summary: Optional[str] = None
    status: DocumentStatus = DocumentStatus.PENDING

    def to_info(self) -> "DocumentInfo":
        """Return a serialization-ready view of this document."""
        return DocumentInfo(
            id=self.id,
            path=self.path,
            status=self.status.value,
            summary=self.summary,
            filename=self.name or (Path(self.path).name if self.path else None),
        )


class DocumentInfo(BaseModel):
    """Serialization-ready view of a Document, suitable for API responses and SSE events."""

    id: Optional[str] = None
    path: Optional[str] = None
    status: str = "pending"
    summary: Optional[str] = None
    filename: Optional[str] = None


class Skill(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    content: str
    system: bool = False
