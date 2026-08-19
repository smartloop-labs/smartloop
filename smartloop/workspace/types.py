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


def relative_document_path(
    path: Optional[str], project_dir: Optional[str] = None
) -> Optional[str]:
    """Render *path* relative to the project, else to the framework home.

    Tried in order:

    * relative to ``project_dir`` — ``documents/invoice.pdf``, the form that
      survives the whole framework being copied to another machine, user, or
      sandbox container;
    * relative to ``AppSettings.home_dir`` — for documents outside the project
      but still inside the framework;
    * the bare filename — so a path from somewhere else entirely still tells
      the client what the file is without disclosing where it lives.

    Always returns POSIX separators so a path rendered on Windows and one
    rendered on macOS compare equal.
    """
    if not path:
        return path

    def _rel(base: Optional[str]) -> Optional[str]:
        if not base:
            return None
        try:
            return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()
        except (ValueError, OSError):
            return None

    resolved = _rel(project_dir)
    if resolved is not None:
        return resolved

    try:
        from smartloop.config.app_settings import AppSettings
        resolved = _rel(AppSettings().home_dir)
    except Exception:
        resolved = None
    if resolved is not None:
        return resolved

    return Path(path).name


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

    def to_info(self, project_dir: Optional[str] = None) -> "DocumentInfo":
        """Return a serialization-ready view of this document.

        ``path`` is emitted **relative**, never absolute.  An absolute path
        pins a client to one machine's layout — it breaks when ``<home_dir>``
        is copied elsewhere or relocated into the macOS App Sandbox container,
        and it discloses the real home directory to every API consumer.
        """
        return DocumentInfo(
            id=self.id,
            path=relative_document_path(self.path, project_dir),
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
