"""One-time migration from ChromaDB (chroma.sqlite3) to IndexStore (store.db).

Reads collections and their documents directly from the ChromaDB SQLite
file — no ChromaDB Python dependency required — and inserts them into the
new IndexStore ``source`` table.

After a successful migration the ``chroma.sqlite3`` file is renamed to
``chroma.sqlite3.bak`` so the migration is not repeated.
"""

import logging
import os
import sqlite3
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def migrate_chroma_to_index_store(project_dir: str) -> bool:
    """Migrate data from ChromaDB to IndexStore for a single project.

    Returns True if migration was performed, False if skipped (no chroma db
    found or already migrated).
    """
    db_dir = os.path.join(project_dir, "db")
    chroma_path = os.path.join(db_dir, "chroma.sqlite3")

    if not os.path.exists(chroma_path):
        return False

    try:
        collections = _read_chroma_collections(chroma_path)
        if not collections:
            _archive_chroma(chroma_path)
            return True

        from smartloop.db.sqlite_store import IndexStore

        for collection_name, chunks in collections.items():
            if not chunks:
                continue
            store = IndexStore(
                project_dir=project_dir,
                collection_name=collection_name,
            )
            store.add_chunks(chunks)
            logger.info(
                f"[migrate] Migrated {len(chunks)} chunks from ChromaDB "
                f"collection '{collection_name}'"
            )

        _archive_chroma(chroma_path)
        logger.info(f"[migrate] ChromaDB migration complete for {project_dir}")
        return True

    except Exception as e:
        logger.warning(f"[migrate] ChromaDB migration failed for {project_dir}: {e}")
        return False


def _archive_chroma(chroma_path: str) -> None:
    """Rename chroma.sqlite3 → chroma.sqlite3.bak."""
    bak = chroma_path + ".bak"
    try:
        os.rename(chroma_path, bak)
    except OSError as e:
        logger.warning(f"[migrate] Could not rename {chroma_path}: {e}")


def _read_chroma_collections(chroma_path: str) -> Dict[str, List[dict]]:
    """Read all collections and their documents from a ChromaDB SQLite file.

    Returns a dict mapping collection name → list of chunk dicts suitable
    for ``IndexStore.add_chunks()``.
    """
    conn = sqlite3.connect(chroma_path)
    conn.row_factory = sqlite3.Row
    result: Dict[str, List[dict]] = {}

    try:
        # Map collection id → name
        coll_rows = conn.execute("SELECT id, name FROM collections").fetchall()
        coll_map = {row["id"]: row["name"] for row in coll_rows}

        # Map collection id → metadata segment id
        seg_rows = conn.execute(
            "SELECT id, collection FROM segments WHERE scope = 'METADATA'"
        ).fetchall()
        seg_to_coll: Dict[str, str] = {}
        for row in seg_rows:
            seg_to_coll[row["id"]] = row["collection"]

        # Read all embeddings grouped by segment
        emb_rows = conn.execute(
            "SELECT id, segment_id, embedding_id FROM embeddings"
        ).fetchall()

        # Group embedding internal ids by collection name
        emb_by_coll: Dict[str, List[tuple]] = {}
        for row in emb_rows:
            coll_id = seg_to_coll.get(row["segment_id"])
            if coll_id is None:
                continue
            coll_name = coll_map.get(coll_id)
            if coll_name is None:
                continue
            emb_by_coll.setdefault(coll_name, []).append(
                (row["id"], row["embedding_id"])
            )

        # For each collection, read metadata for all embeddings
        for coll_name, embeddings in emb_by_coll.items():
            chunks = []
            for internal_id, embedding_id in embeddings:
                meta_rows = conn.execute(
                    "SELECT key, string_value, int_value, float_value "
                    "FROM embedding_metadata WHERE id = ?",
                    (internal_id,),
                ).fetchall()

                chunk = _meta_rows_to_chunk(meta_rows, embedding_id)
                if chunk:
                    chunks.append(chunk)

            result[coll_name] = chunks

    finally:
        conn.close()

    return result


def _meta_rows_to_chunk(
    meta_rows: list, embedding_id: str
) -> Optional[dict]:
    """Convert ChromaDB metadata rows into a chunk dict for IndexStore."""
    text = None
    document_id = ""
    document_name = None
    source_content = None
    chunk_index = 0
    extra: dict = {}

    for row in meta_rows:
        key = row["key"]
        if key == "chroma:document":
            text = row["string_value"]
        elif key == "document_id":
            document_id = row["string_value"] or ""
        elif key == "document_name":
            document_name = row["string_value"]
        elif key == "source_content":
            source_content = row["string_value"]
        elif key == "chunk_index":
            chunk_index = row["int_value"] or 0
        else:
            val = row["string_value"] or row["int_value"] or row["float_value"]
            if val is not None:
                extra[key] = val

    if text is None:
        return None

    return {
        "text": text,
        "document_id": document_id,
        "document_name": document_name,
        "chunk_index": chunk_index,
        "source_content": source_content or "",
        **extra,
    }
