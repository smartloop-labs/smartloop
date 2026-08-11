"""Factory to create VectorStore instances"""
import logging
from typing import Optional

from smartloop.db.store import VectorStore
from smartloop.db.sqlite_store import IndexStore


class StoreType:
    SQLite = "sqlite"


def _resolve_default_store_type() -> str:
    """Return the active backend.

    SQLite is the only backend; kept as a function so existing call sites
    (e.g. the legacy chroma→sqlite migration guard) stay valid.
    """
    return StoreType.SQLite


class StoreFactory:
    """Factory to create instances of VectorStore implementations"""

    @staticmethod
    def create_store(
        project_dir: str,
        collection_name: str = "documents",
        logger: Optional[logging.Logger] = None,
        base_dir: str = None,
        store_type: Optional[str] = None,
    ) -> VectorStore:
        """Create a vector store instance (SQLite-backed dense embeddings).

        Args:
            project_dir: Project directory path
            collection_name: Name of the collection/index
            logger: Optional logger instance
            base_dir: Base directory for model cache and configs
            store_type: Ignored; retained for call-site compatibility.

        Returns:
            VectorStore instance
        """
        return IndexStore(
            project_dir=project_dir,
            collection_name=collection_name,
            logger=logger,
            base_dir=base_dir,
        )
