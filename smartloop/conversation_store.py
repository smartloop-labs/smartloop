"""Persistent JSONL conversation store.

Each session is stored as ``<home_dir>/conversations/<session_id>.jsonl``
where every line is a JSON-encoded ``{"role": ..., "content": ...}`` message.
Serialization and deserialization are intentionally trivial so the file
can be inspected or replayed with any JSON tool.
"""

import json
import os
from typing import List, Optional
from pathlib import Path

from pydantic import BaseModel


class MessageInput(BaseModel):
    role: str
    content: str
    message_id: Optional[str] = None
    rating: Optional[str] = None
    citations: list | None = None

class ConversationStore:
    """Persist chat sessions as JSONL files under ``<home_dir>/conversations/``."""

    def __init__(self, home_dir: str) -> None:
        self._dir = Path(os.path.join(home_dir, "conversations"))
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return Path(os.path.join(self._dir, f"{session_id}.jsonl"))

    def load(self, session_id: str) -> List[MessageInput]:
        """Return all messages for *session_id* loaded from disk.

        Returns an empty list when no file exists yet.
        """
        path = self._path(session_id)
        if not path.exists():
            return []
        messages: List[MessageInput] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        messages.append(MessageInput(**json.loads(line)))
                    except json.JSONDecodeError:
                        pass
        return messages

    def append(self, session_id: str, message: MessageInput) -> None:
        """Append a single message to *session_id*'s JSONL file."""
        with self._path(session_id).open("a", encoding="utf-8") as fh:
            fh.write(message.model_dump_json() + "\n")

    def save(self, session_id: str, messages: List[MessageInput]) -> None:
        """Rewrite the entire session file with *messages*."""
        with self._path(session_id).open("w", encoding="utf-8") as fh:
            for msg in messages:
                fh.write(msg.model_dump_json() + "\n")

    def exists(self, session_id: str) -> bool:
        """Return True when a conversation file exists for *session_id*."""
        return self._path(session_id).exists()
