"""Persistent JSONL feedback store.

Each project stores feedback as ``<project_dir>/feedback.jsonl``
where every line is a JSON-encoded :class:`FeedbackEntry`.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel


class ConversationTurn(BaseModel):
    role: str
    content: str


class FeedbackEntry(BaseModel):
    feedback_id: str
    session_id: str
    message_id: str
    rating: Literal["up", "down"]
    question: str
    answer: str
    conversation: List[ConversationTurn] = []
    created_at: float

    @staticmethod
    def create(
        session_id: str,
        message_id: str,
        rating: Literal["up", "down"],
        question: str,
        answer: str,
        conversation: Optional[List[ConversationTurn]] = None,
    ) -> "FeedbackEntry":
        return FeedbackEntry(
            feedback_id=uuid4().hex[:12],
            session_id=session_id,
            message_id=message_id,
            rating=rating,
            question=question,
            answer=answer,
            conversation=conversation or [],
            created_at=time.time(),
        )


class FeedbackStore:
    def __init__(self, project_dir: str) -> None:
        self._path = Path(os.path.join(project_dir, "feedback.jsonl"))

    def append(self, entry: FeedbackEntry) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")

    def load(self) -> List[FeedbackEntry]:
        if not self._path.exists():
            return []
        entries: List[FeedbackEntry] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(FeedbackEntry(**json.loads(line)))
                    except (json.JSONDecodeError, ValueError):
                        pass
        return entries

    def find(self, session_id: str, message_id: str) -> Optional[FeedbackEntry]:
        for entry in self.load():
            if entry.session_id == session_id and entry.message_id == message_id:
                return entry
        return None

    def remove(self, feedback_id: str) -> None:
        entries = [e for e in self.load() if e.feedback_id != feedback_id]
        with self._path.open("w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(entry.model_dump_json() + "\n")
