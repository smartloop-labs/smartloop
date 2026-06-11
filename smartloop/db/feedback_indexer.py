"""Vector index for thumbed-up Q&A pairs.

Stores feedback in a dedicated ChromaDB collection so positively-rated
conversation exchanges can be retrieved during exploration.
"""

import logging
from typing import List, Optional

from smartloop.db.store import ChunkResult
from smartloop.db.store_factory import StoreFactory
from smartloop.feedback_store import FeedbackEntry

logger = logging.getLogger(__name__)

FEEDBACK_COLLECTION = "feedback_pairs"


class FeedbackIndexer:
    def __init__(
        self,
        project_dir: str,
        logger: Optional[logging.Logger] = None,
        base_dir: Optional[str] = None,
    ):
        self._logger = logger or logging.getLogger(__name__)
        self._store = StoreFactory.create_store(
            project_dir=project_dir,
            collection_name=FEEDBACK_COLLECTION,
            logger=self._logger,
            base_dir=base_dir,
        )

    def index_feedback(self, entry: FeedbackEntry) -> None:
        text = f"Q: {entry.question}\nA: {entry.answer}"
        self._store.add_chunks([{
            "text": text,
            "document_id": entry.feedback_id,
            "chunk_index": 0,
            "document_name": f"feedback/{entry.session_id}/{entry.message_id}",
        }])
        self._logger.info(f"[feedback-index] Indexed {entry.feedback_id}")

    def remove_feedback(self, feedback_id: str) -> None:
        try:
            self._store.delete_document(feedback_id)
            self._logger.info(f"[feedback-index] Removed {feedback_id}")
        except Exception as e:
            self._logger.warning(f"[feedback-index] Remove failed: {e}")

    def search(
        self,
        query: str,
        n_results: int = 5,
        max_distance: float = 0.65,
    ) -> List[ChunkResult]:
        try:
            return self._store.query_with_metadata(
                query, n_results=n_results, max_distance=max_distance
            )
        except Exception as e:
            self._logger.warning(f"[feedback-index] Search failed: {e}")
            return []
