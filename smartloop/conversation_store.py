"""Persistent JSONL conversation store.

Each session is stored as ``<home_dir>/conversations/<session_id>.jsonl``
where every line is a JSON-encoded ``{"role": ..., "content": ...}`` message.
Serialization and deserialization are intentionally trivial so the file
can be inspected or replayed with any JSON tool.
"""

import json
import os
import uuid
from typing import List, Optional
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

    # ------------------------------------------------------------------ #
    # Relevance-based history retrieval                                    #
    # ------------------------------------------------------------------ #

    def query_relevant(
        self,
        session_id: str,
        user_query: str,
        max_turns: int = 4,
        history: Optional[List[MessageInput]] = None,
    ) -> List[MessageInput]:
        """Return the most relevant history turns for *user_query*.

        Uses scikit-learn's ``TfidfVectorizer`` to compute cosine
        similarity between the query and each history turn.  This
        handles multilingual text correctly — IDF naturally
        down-weights frequent terms in any language without needing
        a hardcoded stop-word list.

        Always includes the most recent turn for conversational
        continuity.  Returns turns in chronological order.

        Parameters
        ----------
        session_id:
            Session identifier (used to load history from disk when
            *history* is not supplied).
        user_query:
            The current user message to score against.
        max_turns:
            Maximum number of history messages to return.
        history:
            Pre-loaded history list.  When ``None``, history is loaded
            from the session file.
        """
        if history is None:
            history = self.load(session_id)

        if not history or max_turns <= 0:
            return []

        # If history fits within budget, return as-is
        if len(history) <= max_turns:
            return list(history)

        # Build corpus: each turn's content is a document
        corpus = [msg.content or "" for msg in history]

        try:
            vectorizer = TfidfVectorizer(
                # Unicode-aware tokenization — works across languages
                token_pattern=r"(?u)\b\w\w+\b",
                sublinear_tf=True,
                max_features=5000,
            )
            tfidf_matrix = vectorizer.fit_transform(corpus)
            query_vec = vectorizer.transform([user_query])
            scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        except ValueError:
            # All turns are empty or vectorizer found no features
            return list(history[-max_turns:])

        # Always reserve a slot for the most recent turn
        last_idx = len(history) - 1
        selected_indices = {last_idx}

        # Zero out the last index so it doesn't compete for remaining slots
        scores[last_idx] = -1.0

        # Pick top (max_turns - 1) by score
        top_indices = np.argsort(scores)[::-1]
        for idx in top_indices:
            if len(selected_indices) >= max_turns:
                break
            if scores[idx] > 0:
                selected_indices.add(int(idx))

        # If we didn't fill up, pad with most recent turns
        if len(selected_indices) < max_turns:
            for i in range(last_idx - 1, -1, -1):
                if i not in selected_indices:
                    selected_indices.add(i)
                if len(selected_indices) >= max_turns:
                    break

        # Return in chronological order
        return [history[i] for i in sorted(selected_indices)]
