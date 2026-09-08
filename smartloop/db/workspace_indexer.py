"""
WorkspaceIndexer — indexes local project files into a dedicated ChromaDB
collection ("workspace_files") for language-agnostic semantic search.

Files are split into overlapping chunks (≤ CHUNK_SIZE chars) so they fit
within the embedding model's token window.  A SHA-256 manifest persisted
alongside the database tracks which files are up-to-date so only changed
or new files are re-indexed on subsequent runs.
"""
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from smartloop.db.store import ChunkResult
from smartloop.db.store_factory import StoreFactory
from smartloop.utils import json_to_table
from smartloop.workspace.workspace import PROJECT_STATE_FILE

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ constants

WORKSPACE_COLLECTION = "workspace_files"

# Characters per chunk (~500 tokens for English text, fits the 512-token
# embedding model window). Overlap avoids losing context at boundaries.
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

INDEX_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".kt", ".scala",
    ".yaml", ".yml", ".toml", ".json", ".cfg", ".ini",
    ".md", ".txt", ".rst",
    ".html", ".css", ".sql", ".proto", ".graphql",
    ".sh",
})

SKIP_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", "venv", ".venv",
    "dist", "build", ".tox", ".mypy_cache", ".ruff_cache",
    ".eggs", ".pytest_cache", ".out", "skills",
})

_MANIFEST_FILENAME = ".workspace_manifest.json"

# Directories holding the framework's own state, skipped when they sit at the
# root of the scan.  The scan root *is* ``project_dir`` (see IndexerTask), so
# the vector store's own files are always inside it.
SKIP_ROOT_DIRS = frozenset({"db"})

# Files holding the framework's own state, skipped at the root of the scan for
# the same reason as SKIP_ROOT_DIRS — and for the reason hidden files are
# skipped below, which ``project.json`` shares without being hidden.
#
# It is the project's own record, rewritten on every turn (a new session, a
# document, an MCP registration), and it grows with use: by a few hundred
# conversations it was a megabyte, 567 chunks.  Indexing it meant every turn
# changed its hash, so every scan deleted and re-embedded all of it — a
# minute of GPU spent re-learning the app's own bookkeeping, competing with
# the reply the user was waiting for, and never answering a question anyone
# would ask of a workspace.
SKIP_ROOT_FILES = frozenset({PROJECT_STATE_FILE})


# ------------------------------------------------------------------ helpers

def _file_hash(path: str) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split *text* into overlapping chunks of at most *size* characters."""
    if len(text) <= size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def _stable_id(rel_path: str) -> str:
    """Return a stable, short document ID derived from a relative file path."""
    return hashlib.md5(rel_path.encode()).hexdigest()


# ------------------------------------------------------------------ indexer

class WorkspaceIndexer:
    """Indexes workspace files into a dedicated ChromaDB collection.

    The indexer is *incremental*: on each ``index()`` call it reads the
    on-disk manifest, hashes every candidate file, and only re-indexes
    files whose hash has changed.  Deleted files are removed from the
    collection.

    Usage::

        indexer = WorkspaceIndexer(project_dir="/path/to/.slp/my-project")
        indexer.index(work_dir="/path/to/workspace")  # incremental
        results = indexer.search("authentication logic")  # List[ChunkResult]
    """

    def __init__(
        self,
        project_dir: str,
        logger: Optional[logging.Logger] = None,
        base_dir: Optional[str] = None,
    ):
        self._project_dir = project_dir
        self._logger = logger or logging.getLogger(__name__)
        self._store = StoreFactory.create_store(
            project_dir=project_dir,
            collection_name=WORKSPACE_COLLECTION,
            logger=self._logger,
            base_dir=base_dir,
        )
        self._manifest_path = os.path.join(project_dir, _MANIFEST_FILENAME)
        self._manifest: Dict[str, str] = self._load_manifest()
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- public

    @property
    def file_count(self) -> int:
        return len(self._manifest)

    def index(self, work_dir: str) -> Dict[str, int]:
        """Walk *work_dir* and incrementally index changed/new files.

        Returns a stats dict: ``{"indexed": N, "skipped": N, "removed": N}``.
        Thread-safe — concurrent calls are serialised.
        """
        from smartloop.db.embeddings import indexing_burst

        with self._lock, indexing_burst():
            return self._index(work_dir)

    def search(
        self,
        query: str,
        n_results: int = 10,
        max_distance: float = 0.65,
    ) -> List[ChunkResult]:
        """Semantic search over indexed workspace files.

        Args:
            query: Natural-language search query.
            n_results: Maximum number of chunks to return.
            max_distance: Cosine distance threshold (lower = stricter).

        Returns:
            List of :class:`~smartloop.db.store.ChunkResult` ordered by
            relevance (closest distance first).
        """
        try:
            return self._store.query_with_metadata(
                query, n_results=n_results, max_distance=max_distance
            )
        except Exception as e:
            self._logger.warning(f"[workspace-index] Search failed: {e}")
            return []

    # ---------------------------------------------------------------- internals

    def _load_manifest(self) -> Dict[str, str]:
        if os.path.isfile(self._manifest_path):
            try:
                with open(self._manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self._logger.warning(f"[workspace-index] Could not load manifest: {e}")
        return {}

    def _save_manifest(self) -> None:
        try:
            with open(self._manifest_path, "w", encoding="utf-8") as f:
                json.dump(self._manifest, f, indent=2)
        except Exception as e:
            self._logger.warning(f"[workspace-index] Could not save manifest: {e}")

    def _index(self, work_dir: str) -> Dict[str, int]:
        work_dir = os.path.realpath(work_dir)
        stats = {"indexed": 0, "skipped": 0, "removed": 0}

        # Collect all current candidate files
        current_files: Dict[str, str] = {}  # rel_path -> abs_path
        for root, dirs, files in os.walk(work_dir, topdown=True):
            at_root = os.path.realpath(root) == work_dir
            dirs[:] = sorted(
                d for d in dirs
                if d not in SKIP_DIRS and not d.endswith(".egg-info")
                and not (at_root and d in SKIP_ROOT_DIRS)
            )
            for fname in files:
                # Hidden files are state, not content — and one of them is this
                # indexer's own manifest, written at the end of every run.
                # Indexing it made every run dirty the input to the next one: a
                # file re-embedded every scan, forever, for nothing.
                if fname.startswith("."):
                    continue
                if at_root and fname in SKIP_ROOT_FILES:
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in INDEX_EXTENSIONS:
                    continue
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, work_dir)
                current_files[rel_path] = abs_path

        new_manifest: Dict[str, str] = {}

        # Index new / changed files
        for rel_path, abs_path in current_files.items():
            try:
                file_hash = _file_hash(abs_path)
            except Exception as e:
                self._logger.debug(f"[workspace-index] Cannot hash {rel_path}: {e}")
                continue

            new_manifest[rel_path] = file_hash

            if self._manifest.get(rel_path) == file_hash:
                stats["skipped"] += 1
                continue

            try:
                text = Path(abs_path).read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                self._logger.debug(f"[workspace-index] Cannot read {rel_path}: {e}")
                continue

            if not text.strip():
                stats["skipped"] += 1
                continue

            # Structured JSON documents (an array of objects) are converted to
            # a markdown table BEFORE chunking so the workspace index holds
            # clean table rows instead of fragmented JSON text — matching the
            # main document indexer's behavior (see indexer_task).
            table = json_to_table(text)
            if table is not None:
                self._logger.info(
                    f"[workspace-index] {rel_path}: converted JSON → table "
                    f"({len(table)} chars)"
                )
                text = table

            doc_id = _stable_id(rel_path)
            chunks = _chunk_text(text)

            # Remove stale chunks before re-adding
            if rel_path in self._manifest:
                try:
                    self._store.delete_document(doc_id)
                except Exception:
                    pass

            chunk_dicts: List[Dict[str, Any]] = [
                {
                    "text": chunk,
                    "document_id": doc_id,
                    "chunk_index": i,
                    "document_name": rel_path,
                }
                for i, chunk in enumerate(chunks)
            ]

            try:
                self._store.add_chunks(chunk_dicts)
                stats["indexed"] += 1
            except Exception as e:
                self._logger.warning(f"[workspace-index] Failed to index {rel_path}: {e}")

        # Remove deleted files from the index
        for rel_path in self._manifest:
            if rel_path not in current_files:
                try:
                    self._store.delete_document(_stable_id(rel_path))
                    stats["removed"] += 1
                except Exception:
                    pass

        self._manifest = new_manifest
        self._save_manifest()

        self._logger.info(
            f"[workspace-index] Complete — "
            f"indexed={stats['indexed']}, skipped={stats['skipped']}, "
            f"removed={stats['removed']}, total={len(new_manifest)} files"
        )
        return stats
