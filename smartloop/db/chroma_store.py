import logging
import os
import threading
from typing import List

from smartloop.db.store import ChunkResult, VectorStore, keyword_coverage_boost

from smartloop.config.app_settings import AppSettings
settings = AppSettings()

EMBEDDING_MODEL_NAME = settings.embedding_model
EMBEDDING_DIR_NAME = "embeddings"

# A ChromaDB PersistentClient must be shared per path: creating multiple
# clients for the same directory (e.g. one per collection — documents,
# feedback, workspace — or per request) corrupts chromadb 1.x's shared system
# and fails with "Could not connect to tenant default_tenant" or
# "'RustBindingsAPI' object has no attribute 'bindings'".  Cache one client per
# path and serialize creation so concurrent DAG threads don't race.
_CLIENTS: dict = {}
_CLIENTS_LOCK = threading.Lock()


def _get_client(persist_dir: str):
    """Return the shared PersistentClient for *persist_dir*, creating once."""
    import chromadb
    from chromadb.config import Settings

    client = _CLIENTS.get(persist_dir)
    if client is not None:
        return client
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(persist_dir)
        if client is None:
            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            _CLIENTS[persist_dir] = client
        return client


def _get_embedding_cache_dir(base_dir: str = None) -> str:
    """Return the path where the embedding model is cached locally.

    Layout: ``<base_dir>/models/embeddings/all-mpnet-base-v2/``
    """
    if base_dir is None:
        base_dir = settings.home_dir
    path = os.path.join(
        base_dir, "models", EMBEDDING_DIR_NAME, EMBEDDING_MODEL_NAME.split("/")[-1]
    )
    os.makedirs(path, exist_ok=True)
    return path


def _load_embedding_function(cache_dir: str):
    """Load sentence-transformers model from *cache_dir*, downloading once if needed.

    Loads on the device from ``slp_embedding_device`` (CPU by default so the
    GPU stays free for the generative model copies).  Falls back to CPU if the
    requested device fails — e.g. a meta-tensor / out-of-memory error when the
    GPU is already saturated by the LLM workers.
    """
    import contextlib

    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    from smartloop.db.embeddings import _hf_offline, _is_model_cached
    from smartloop.utils.torch_meta_guard import safe_model_load

    device = (settings.embedding_device or "cpu").strip().lower()
    if device == "auto":
        device = None  # let sentence-transformers pick

    # SentenceTransformerEmbeddingFunction has no local_files_only knob, so when
    # the model is already cached force huggingface_hub fully offline to skip the
    # freshness HEAD request (which retries for ~30s when there's no network).
    # A factory so each (re)build gets a fresh, single-use context manager.
    cached = _is_model_cached(cache_dir)
    offline = _hf_offline if cached else contextlib.nullcontext

    # low_cpu_mem_usage=False forces the weights to load directly onto the
    # device instead of through the 'meta' empty-init path that accelerate /
    # transformers use when another model load has switched on low-memory
    # mode.  Without this the load fails with "Cannot copy out of meta tensor;
    # no data!" even though the model is fully downloaded.
    model_kwargs = {"low_cpu_mem_usage": False}

    def _build(dev):
        # safe_model_load serializes the load and repairs any leaked accelerate
        # meta-init patch so the embedding weights don't land on the meta device.
        with offline(), safe_model_load():
            return SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL_NAME,
                cache_folder=cache_dir,
                device=dev,
                model_kwargs=model_kwargs,
            )

    try:
        return _build(device)
    except Exception as exc:
        if device == "cpu":
            raise
        logging.getLogger(__name__).warning(
            f"[chroma] Embedding model failed to load on device={device!r} "
            f"({exc}); falling back to CPU."
        )
        return _build("cpu")


def warmup_embeddings(base_dir: str = None) -> None:
    """Pre-download / cache the embedding model so first vectorization is fast."""
    cache_dir = _get_embedding_cache_dir(base_dir)
    _load_embedding_function(cache_dir)


class ChromaStore(VectorStore):
    """Vector wrapper for indexing and querying document chunks."""

    def __init__(
        self,
        project_dir: str,
        collection_name: str = "documents",
        logger: logging.Logger = None,
        base_dir: str = None,
    ):
        if base_dir is None:
            base_dir = settings.home_dir
        super().__init__(project_dir, collection_name, logger, base_dir)
        self.persist_dir = os.path.join(self.project_dir, "db")
        os.makedirs(self.persist_dir, exist_ok=True)

        # Load the embedding model (cached in .slp/models/embeddings/)
        cache_dir = _get_embedding_cache_dir(self.base_dir)
        self._embedding_fn = _load_embedding_function(cache_dir)

        os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

        # Shared per-path client (see _get_client) — never create more than one
        # PersistentClient for the same directory.
        self._client = _get_client(self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_fn,
        )

    def add_chunks(self, chunks: List[dict]):
        """Add document chunks to the vector store.

        Each chunk dict should have:
          - text: str
          - document_id: str
          - chunk_index: int (optional)
          - document_name: str (optional) — source filename for citations
        """
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        _RESERVED = {"text", "chunk_index"}
        for i, chunk in enumerate(chunks):
            chunk_id = f"{chunk['document_id']}_{chunk.get('chunk_index', i)}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            meta = {"document_id": chunk["document_id"]}
            if chunk.get("document_name"):
                meta["document_name"] = chunk["document_name"]
            # Pass through any extra metadata fields (person, document_type, etc.)
            # Chroma only accepts str/int/float values in metadata
            for k, v in chunk.items():
                if k not in _RESERVED and k not in meta and isinstance(v, (str, int, float)):
                    meta[k] = v
            # Truncate source_content to stay within Chroma metadata value limits
            if "source_content" in meta and isinstance(meta["source_content"], str):
                meta["source_content"] = meta["source_content"][:4000]
            metadatas.append(meta)

        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        self.logger.info(f"Indexed {len(ids)} chunks into vector store")

    def query(self, query_text: str, n_results: int = None, max_distance: float = 0.8) -> List[str]:
        """Semantic search, returns top matching texts.

        Only results with cosine distance <= max_distance are returned.
        This prevents irrelevant chunks from being injected into the prompt
        when the query has nothing to do with the indexed documents.
        """
        results = self.query_with_metadata(query_text, n_results, max_distance)
        return [r.text for r in results]

    def query_with_metadata(
        self,
        query_text: str,
        n_results: int = None,
        max_distance: float = 0.8,
        where: dict = None,
        where_document: dict = None,
    ) -> List[ChunkResult]:
        """Semantic search returning structured results with source metadata.

        ``where`` is an optional Chroma metadata filter, e.g.::

            {"person": {"$eq": "Aaron Bergman"}}
            {"document_type": {"$eq": "invoice"}}

        ``where_document`` is an optional Chroma full-text filter on chunk
        content, e.g.::

            {"$contains": "Adam"}

        When provided, only chunks whose text contains the given substring
        are returned. This is useful for scoping results to a specific
        entity mentioned in the query.
        """
        if self._collection.count() == 0:
            return []

        effective_n = min(n_results, self._collection.count()) if n_results else self._collection.count()

        query_kwargs = dict(
            query_texts=[query_text],
            n_results=effective_n,
            include=["documents", "distances", "metadatas"],
        )
        if where:
            query_kwargs["where"] = where
        if where_document:
            query_kwargs["where_document"] = where_document

        results = self._collection.query(**query_kwargs)

        documents = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []

        # Keyword-coverage boost: dense embeddings cluster look-alike documents
        # together, so reward documents that match more of the distinct query
        # keywords (e.g. "Adam Hart") by reducing their distance.  This mirrors
        # the sqlite embedding path; see store.keyword_coverage_boost.
        chunk_results = []
        for i, (doc, dist) in enumerate(zip(documents, distances)):
            meta = metadatas[i] if i < len(metadatas) else {}
            boost = keyword_coverage_boost(
                query_text, meta.get("document_name"),
                meta.get("source_content") or doc,
            )
            boosted = max(0.0, dist - boost)
            if boosted > max_distance:
                continue
            chunk_results.append(ChunkResult(
                text=doc,
                document_id=meta.get("document_id", ""),
                document_name=meta.get("document_name"),
                distance=boosted,
                metadata=meta,
                source_content=meta.get("source_content"),
            ))

        # Re-sort by the boosted distance so keyword-rich matches rank first.
        chunk_results.sort(key=lambda r: r.distance)
        return chunk_results

    def get_by_document_ids(self, document_ids: List[str]) -> List[ChunkResult]:
        """Fetch all chunks for the given document IDs without a semantic query.

        Used after a discovery search to pull every chunk from matched documents
        so the full document content is available for the LLM.
        """
        if not document_ids:
            return []
        try:
            where = (
                {"document_id": {"$eq": document_ids[0]}}
                if len(document_ids) == 1
                else {"document_id": {"$in": document_ids}}
            )
            result = self._collection.get(
                where=where,
                include=["documents", "metadatas"],
            )
            docs = result.get("documents") or []
            metas = result.get("metadatas") or []
            chunks = []
            for doc, meta in zip(docs, metas):
                chunks.append(ChunkResult(
                    text=doc,
                    document_id=meta.get("document_id", ""),
                    document_name=meta.get("document_name"),
                    distance=0.0,
                    metadata=meta,
                    source_content=meta.get("source_content"),
                ))
            return chunks
        except Exception as e:
            self.logger.warning(f"get_by_document_ids failed: {e}")
            return []

    def delete_document(self, document_id: str):
        """Remove all chunks for a document."""
        self._collection.delete(where={"document_id": document_id})
        self.logger.info(f"Deleted chunks for document: {document_id}")
