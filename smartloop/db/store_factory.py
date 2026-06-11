"""Factory to create VectorStore instances"""
import logging
import os
from typing import Optional

from smartloop.db.store import VectorStore
from smartloop.db.sqlite_store import IndexStore

class StoreType:
    Chroma = "chroma"
    SQLite = "sqlite"


def _resolve_default_store_type() -> str:
    """Read the configured backend (``slp_vector_store_backend``).

    Falls back to SQLite if settings can't be loaded for any reason so
    retrieval always has a working backend.
    """
    try:
        from smartloop.config.app_settings import AppSettings
        return AppSettings().vector_store_backend
    except Exception:
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
        """Create a vector store instance.

        Args:
            project_dir: Project directory path
            collection_name: Name of the collection/index
            logger: Optional logger instance
            base_dir: Base directory for model cache and configs
            store_type: Backend to use ("sqlite" or "chroma").  When None
                (the default), the backend is read from the
                ``slp_vector_store_backend`` setting so every call site
                stays consistent without passing it explicitly.

        Returns:
            VectorStore instance

        Raises:
            ValueError: If store_type is not supported
        """
        if store_type is None:
            store_type = _resolve_default_store_type()

        if store_type == StoreType.SQLite:
            return IndexStore(
                project_dir=project_dir,
                collection_name=collection_name,
                logger=logger,
                base_dir=base_dir,
            )
        elif store_type == StoreType.Chroma:
            try:
                from smartloop.db.chroma_store import ChromaStore
            except ImportError as exc:
                raise ImportError(
                    "The 'chroma' vector store backend requires the "
                    "chromadb package. Install it with: pip install chromadb"
                ) from exc
            return ChromaStore(
                project_dir=project_dir,
                collection_name=collection_name,
                logger=logger,
                base_dir=base_dir,
            )
        else:
            raise ValueError(f"Unsupported store type: {store_type}")
