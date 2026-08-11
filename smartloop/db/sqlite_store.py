"""SQLite vector store with dense embeddings and TF-IDF fallback.

Uses sentence-transformers (all-mpnet-base-v2) for dense embedding
similarity search.  Falls back to scikit-learn TF-IDF when the
embedding model is unavailable.  All source types (document, feedback,
workspace, conversation) share a single ``indexes`` table differentiated
by ``type`` in ``<project_dir>/db/store.db``.
"""

import json
import logging
import os
import sqlite3
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from smartloop.db.embeddings import (
    blob_to_embedding,
    embedding_to_blob,
    encode_texts,
)
from smartloop.db.store import ChunkResult, VectorStore, keyword_coverage_boost

_TABLE = "indexes"

# Multiplier for the TF-IDF coordination boost: a document matching all
# distinct query terms gets its similarity scaled by (1 + this), one matching
# half by (1 + this/2), etc.  Rewards keyword coverage over raw term frequency.
_TFIDF_COORD_BOOST = 1.0

# Map legacy collection names to short type values
_TYPE_MAP = {
    "documents": "document",
    "feedback_pairs": "feedback",
    "workspace_files": "workspace",
}


def _resolve_type(collection_name: str) -> str:
    return _TYPE_MAP.get(collection_name, collection_name)


def _get_db_path(project_dir: str) -> str:
    db_dir = os.path.join(project_dir, "db")
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "store.db")


def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _upgrade_indexes_schema(conn: sqlite3.Connection) -> None:
    """Apply incremental column renames to an existing table."""
    cols = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({_TABLE})").fetchall()
    }
    if not cols:
        return
    if "document_id" in cols and "ref" not in cols:
        conn.execute(f"ALTER TABLE {_TABLE} RENAME COLUMN document_id TO ref")
    if "embedding" not in cols:
        try:
            conn.execute(f"ALTER TABLE {_TABLE} ADD COLUMN embedding BLOB")
        except sqlite3.OperationalError:
            pass  # Column already exists (race or stale PRAGMA cache)
    # Clean up stale indices from a previous iteration that had a project column
    conn.execute("DROP INDEX IF EXISTS idx_indexes_project_type")
    conn.execute("DROP INDEX IF EXISTS idx_indexes_project_type_ref")


def _ensure_schema(db_path: str) -> None:
    """Create the indexes table and indices if they don't exist yet.

    Also migrates data from older table layouts (per-collection tables,
    the intermediate ``chunks`` table, or the ``source`` table) if present.
    """
    with _get_conn(db_path) as conn:
        _upgrade_indexes_schema(conn)

        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_TABLE} (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                ref TEXT NOT NULL,
                name TEXT,
                text TEXT NOT NULL,
                content TEXT,
                metadata TEXT,
                embedding BLOB
            )
        """)
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_indexes_type "
            f"ON {_TABLE}(type)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_indexes_type_ref "
            f"ON {_TABLE}(type, ref)"
        )

        _migrate_legacy_tables(conn)


def _ensure_embedding_model(db_path: str, model_name: str) -> None:
    """Invalidate stored embeddings when the embedding model changes.

    Embeddings from different models live in different vector spaces, so mixing
    them (or querying old vectors with a new model) produces nonsense.  A marker
    file records which model produced the stored embeddings; if it differs from
    the configured model, the ``embedding`` column is cleared so the lazy
    backfill recomputes every vector with the new model.
    """
    marker = os.path.join(os.path.dirname(db_path), "embedding_model.txt")
    previous = None
    if os.path.exists(marker):
        try:
            with open(marker) as fh:
                previous = fh.read().strip()
        except Exception:
            previous = None

    if previous == model_name:
        return

    try:
        with _get_conn(db_path) as conn:
            cur = conn.execute(
                f"UPDATE {_TABLE} SET embedding = NULL WHERE embedding IS NOT NULL"
            )
            cleared = cur.rowcount
        if cleared:
            logging.getLogger(__name__).info(
                f"[sqlite] Embedding model changed "
                f"({previous or 'unknown'} → {model_name}); cleared {cleared} "
                f"stored embedding(s) for re-embedding."
            )
    except Exception as exc:
        logging.getLogger(__name__).warning(
            f"[sqlite] Could not clear stale embeddings: {exc}"
        )

    try:
        with open(marker, "w") as fh:
            fh.write(model_name)
    except Exception:
        pass


# Tables from the very first per-collection layout
_LEGACY_COLLECTION_TABLES = ("documents", "feedback_pairs", "workspace_files")


def _migrate_legacy_tables(conn: sqlite3.Connection) -> None:
    """Move rows from older table layouts into the unified indexes table."""
    existing = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    # v1: per-collection tables (documents, feedback_pairs, workspace_files)
    for table in _LEGACY_COLLECTION_TABLES:
        if table not in existing:
            continue
        count = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
        if count == 0:
            conn.execute(f"DROP TABLE [{table}]")
            continue
        type_val = _resolve_type(table)
        conn.execute(
            f"INSERT OR IGNORE INTO {_TABLE} "
            f"(id, type, ref, name, text, content, metadata) "
            f"SELECT id, ?, document_id, document_name, text, "
            f"source_content, metadata FROM [{table}]",
            (type_val,),
        )
        conn.execute(f"DROP TABLE [{table}]")

    # v2: unified chunks table with collection column
    if "chunks" in existing:
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if count == 0:
            conn.execute("DROP TABLE chunks")
        else:
            for old_coll, new_type in _TYPE_MAP.items():
                conn.execute(
                    f"INSERT OR IGNORE INTO {_TABLE} "
                    f"(id, type, ref, name, text, content, metadata) "
                    f"SELECT id, ?, document_id, document_name, text, "
                    f"source_content, metadata FROM chunks "
                    f"WHERE collection = ?",
                    (new_type, old_coll),
                )
            placeholders = ",".join("?" for _ in _TYPE_MAP)
            conn.execute(
                f"INSERT OR IGNORE INTO {_TABLE} "
                f"(id, type, ref, name, text, content, metadata) "
                f"SELECT id, collection, document_id, document_name, text, "
                f"source_content, metadata FROM chunks "
                f"WHERE collection NOT IN ({placeholders})",
                list(_TYPE_MAP.keys()),
            )
            conn.execute("DROP TABLE chunks")

    # v3: source table (same schema, just renamed to indexes)
    if "source" in existing:
        count = conn.execute("SELECT COUNT(*) FROM source").fetchone()[0]
        if count > 0:
            conn.execute(
                f"INSERT OR IGNORE INTO {_TABLE} "
                f"(id, type, ref, name, text, content, metadata) "
                f"SELECT id, type, "
                f"CASE WHEN ref IS NOT NULL THEN ref ELSE document_id END, "
                f"name, text, content, metadata "
                f"FROM source"
            )
        conn.execute("DROP TABLE source")


class IndexStore(VectorStore):
    """Vector store backed by SQLite with TF-IDF similarity search."""

    def __init__(
        self,
        project_dir: str,
        collection_name: str = "documents",
        logger: logging.Logger = None,
        base_dir: str = None,
    ):
        super().__init__(project_dir, collection_name, logger, base_dir)
        self._db_path = _get_db_path(self.project_dir)
        self._type = _resolve_type(self.collection_name)
        _ensure_schema(self._db_path)
        # Clear stored embeddings if the configured embedding model changed
        # (e.g. switching to the multilingual model) so they're recomputed.
        try:
            from smartloop.db.embeddings import active_embedding_identity
            _ensure_embedding_model(self._db_path, active_embedding_identity())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # add_chunks
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[dict]):
        if not chunks:
            return

        texts = []
        row_data = []
        _RESERVED = {"text", "chunk_index", "document_id", "document_name", "source_content"}
        for i, chunk in enumerate(chunks):
            chunk_id = f"{chunk['document_id']}_{chunk.get('chunk_index', i)}"
            meta = {}
            for k, v in chunk.items():
                if k not in _RESERVED and isinstance(v, (str, int, float)):
                    meta[k] = v
            content = chunk.get("source_content", "")
            if isinstance(content, str) and len(content) > 4000:
                content = content[:4000]

            texts.append(chunk["text"])
            row_data.append((
                chunk_id,
                self._type,
                chunk["document_id"],
                chunk.get("document_name"),
                chunk["text"],
                content,
                json.dumps(meta) if meta else None,
            ))

        # Batch-encode all chunk texts into dense embeddings
        embeddings = encode_texts(texts, self.base_dir)

        rows = []
        for i, rd in enumerate(row_data):
            if embeddings is not None:
                blob = embedding_to_blob(embeddings[i])
            else:
                blob = None
            rows.append(rd + (blob,))

        with _get_conn(self._db_path) as conn:
            conn.executemany(
                f"INSERT OR REPLACE INTO {_TABLE} "
                "(id, type, ref, name, text, content, metadata, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        self.logger.info(f"Indexed {len(rows)} chunks into SQLite store")

    # ------------------------------------------------------------------
    # query / query_with_metadata
    # ------------------------------------------------------------------

    def query(
        self, query_text: str, n_results: int = None, max_distance: float = 0.8
    ) -> List[str]:
        results = self.query_with_metadata(query_text, n_results, max_distance)
        return [r.text for r in results]

    def query_with_metadata(
        self,
        query_text: str,
        n_results: int = None,
        max_distance: float = 0.8,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> List[ChunkResult]:
        rows = self._fetch_rows(where, where_document)
        if not rows:
            return []

        # Always use dense embeddings (semantic recall — handles paraphrases
        # and different languages); fall back to TF-IDF only when the embedding
        # model is unavailable.  The keyword-coverage boost in the dense path
        # plus the downstream per-document relevance filter suppress the
        # clustering noise dense search can otherwise introduce.
        query_vecs = encode_texts([query_text], self.base_dir)
        if query_vecs is not None:
            return self._query_by_embedding(
                rows, query_vecs[0], n_results, max_distance,
                query_text=query_text,
            )
        return self._query_by_tfidf(rows, query_text, n_results, max_distance)

    # ------------------------------------------------------------------
    # Embedding-based search
    # ------------------------------------------------------------------

    def _query_by_embedding(
        self,
        rows: list,
        query_vec: np.ndarray,
        n_results: Optional[int],
        max_distance: float,
        query_text: str = "",
    ) -> List[ChunkResult]:
        """Rank rows by cosine similarity against dense embeddings."""
        # Separate rows with / without precomputed embeddings
        has_emb = []
        missing_emb = []
        for i, row in enumerate(rows):
            if row["embedding"] is not None:
                has_emb.append((i, row))
            else:
                missing_emb.append((i, row))

        # Lazy backfill: compute embeddings for rows that lack them
        if missing_emb:
            texts = [row["text"] for _, row in missing_emb]
            new_vecs = encode_texts(texts, self.base_dir)
            if new_vecs is not None:
                updates = []
                for j, (idx, row) in enumerate(missing_emb):
                    blob = embedding_to_blob(new_vecs[j])
                    updates.append((blob, row["id"]))
                    has_emb.append((idx, row))
                # Persist backfilled embeddings
                try:
                    with _get_conn(self._db_path) as conn:
                        conn.executemany(
                            f"UPDATE {_TABLE} SET embedding = ? WHERE id = ?",
                            updates,
                        )
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        f"[sqlite] Backfill persist failed: {exc}"
                    )
                # Use freshly computed vectors for ranking
                emb_matrix = []
                emb_rows = []
                # Rebuild in original order
                vec_by_idx = {idx: new_vecs[j] for j, (idx, _) in enumerate(missing_emb)}
                for idx, row in sorted(has_emb, key=lambda x: x[0]):
                    if idx in vec_by_idx:
                        emb_matrix.append(vec_by_idx[idx])
                    else:
                        emb_matrix.append(blob_to_embedding(row["embedding"]))
                    emb_rows.append(row)
            else:
                # Could not encode missing rows — rank only rows with embeddings
                emb_matrix = [blob_to_embedding(row["embedding"]) for _, row in has_emb]
                emb_rows = [row for _, row in has_emb]
        else:
            emb_matrix = [blob_to_embedding(row["embedding"]) for _, row in has_emb]
            emb_rows = [row for _, row in has_emb]

        if not emb_matrix:
            return self._query_by_tfidf(rows, "", n_results, max_distance)

        corpus_matrix = np.stack(emb_matrix)
        # Normalise for cosine similarity
        norms = np.linalg.norm(corpus_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        corpus_matrix = corpus_matrix / norms

        q = query_vec / (np.linalg.norm(query_vec) or 1)
        similarities = corpus_matrix @ q
        distances = 1.0 - similarities

        # Boost documents by the NUMBER of distinct query keywords they
        # contain so the document the user actually named ("Adam Hart") ranks
        # above embedding look-alikes ("Aaron Hawkins").  See
        # store.keyword_coverage_boost.
        if query_text:
            for i, row in enumerate(emb_rows):
                boost = keyword_coverage_boost(
                    query_text, row["name"], row["content"] or row["text"],
                )
                if boost:
                    distances[i] = max(0.0, distances[i] - boost)

        indices = np.argsort(distances)

        results: List[ChunkResult] = []
        limit = n_results if n_results else len(emb_rows)
        for idx in indices:
            if len(results) >= limit:
                break
            dist = float(distances[idx])
            if dist > max_distance:
                continue
            row = emb_rows[idx]
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            meta["document_id"] = row["ref"]
            if row["name"]:
                meta["document_name"] = row["name"]
            results.append(
                ChunkResult(
                    text=row["text"],
                    document_id=row["ref"],
                    document_name=row["name"],
                    distance=dist,
                    metadata=meta,
                    source_content=row["content"],
                )
            )
        return results

    # ------------------------------------------------------------------
    # TF-IDF fallback search
    # ------------------------------------------------------------------

    def _query_by_tfidf(
        self,
        rows: list,
        query_text: str,
        n_results: Optional[int],
        max_distance: float,
    ) -> List[ChunkResult]:
        """Rank rows using TF-IDF cosine similarity (fallback)."""
        normalized_query = query_text.replace("_", " ")
        corpus = [row["content"] or row["text"] or "" for row in rows]

        try:
            vectorizer = TfidfVectorizer(
                token_pattern=r"(?u)\b\w\w+\b",
                sublinear_tf=True,
                max_features=5000,
            )
            tfidf_matrix = vectorizer.fit_transform(corpus + [normalized_query])
            query_vec = tfidf_matrix[-1]
            corpus_matrix = tfidf_matrix[:-1]
            similarities = cosine_similarity(query_vec, corpus_matrix).flatten()
        except ValueError:
            return []

        # Coordination boost: reward documents that contain more of the
        # distinct query terms.  Cosine alone can rank a document repeating one
        # common term above one that matches more of the query, so scale each
        # document's score by its query-term coverage.  The query vector's
        # nonzero columns are exactly the in-vocabulary query terms.
        query_term_cols = query_vec.indices
        if len(query_term_cols):
            present = corpus_matrix[:, query_term_cols] > 0
            matched = np.asarray(present.sum(axis=1)).flatten()
            coverage = matched / len(query_term_cols)
            similarities = similarities * (1.0 + _TFIDF_COORD_BOOST * coverage)

        max_sim = float(similarities.max()) if len(similarities) else 0.0
        if max_sim > 0:
            distances = 1.0 - (similarities / max_sim)
        else:
            distances = np.ones_like(similarities)
        indices = np.argsort(distances)

        results: List[ChunkResult] = []
        limit = n_results if n_results else len(rows)
        for idx in indices:
            if len(results) >= limit:
                break
            dist = float(distances[idx])
            if dist > max_distance:
                continue
            row = rows[idx]
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            meta["document_id"] = row["ref"]
            if row["name"]:
                meta["document_name"] = row["name"]
            results.append(
                ChunkResult(
                    text=row["text"],
                    document_id=row["ref"],
                    document_name=row["name"],
                    distance=dist,
                    metadata=meta,
                    source_content=row["content"],
                )
            )
        return results

    # ------------------------------------------------------------------
    # get_by_document_ids / delete_document
    # ------------------------------------------------------------------

    def get_by_document_ids(self, document_ids: List[str]) -> List[ChunkResult]:
        if not document_ids:
            return []
        placeholders = ",".join("?" for _ in document_ids)
        with _get_conn(self._db_path) as conn:
            cursor = conn.execute(
                f"SELECT * FROM {_TABLE} "
                f"WHERE type = ? AND ref IN ({placeholders})",
                [self._type] + document_ids,
            )
            rows = cursor.fetchall()

        results: List[ChunkResult] = []
        for row in rows:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            meta["document_id"] = row["ref"]
            if row["name"]:
                meta["document_name"] = row["name"]
            results.append(
                ChunkResult(
                    text=row["text"],
                    document_id=row["ref"],
                    document_name=row["name"],
                    distance=0.0,
                    metadata=meta,
                    source_content=row["content"],
                )
            )
        return results

    def delete_document(self, document_id: str):
        with _get_conn(self._db_path) as conn:
            conn.execute(
                f"DELETE FROM {_TABLE} WHERE type = ? AND ref = ?",
                (self._type, document_id),
            )
        self.logger.info(f"Deleted chunks for document: {document_id}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_rows(
        self,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> list:
        """Fetch rows from the indexes table for this type."""
        clauses: List[str] = ["type = ?"]
        params: list = [self._type]

        if where:
            for key, condition in where.items():
                if isinstance(condition, dict):
                    for op, val in condition.items():
                        if op == "$eq":
                            clauses.append(
                                f"json_extract(metadata, '$.{key}') = ?"
                            )
                            params.append(val)
                        elif op == "$in":
                            ph = ",".join("?" for _ in val)
                            clauses.append(
                                f"json_extract(metadata, '$.{key}') IN ({ph})"
                            )
                            params.extend(val)
                else:
                    clauses.append(f"json_extract(metadata, '$.{key}') = ?")
                    params.append(condition)

        if where_document:
            contains = where_document.get("$contains")
            if contains:
                clauses.append("text LIKE ?")
                params.append(f"%{contains}%")

        sql = f"SELECT * FROM {_TABLE} WHERE " + " AND ".join(clauses)

        with _get_conn(self._db_path) as conn:
            cursor = conn.execute(sql, params)
            return cursor.fetchall()
