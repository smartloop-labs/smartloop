"""
Indexer Task — indexes documents into the vector store via a persistent event loop.

Each enqueued document is dispatched to a worker thread sized to ``os.cpu_count()``
so uploads can be processed in parallel.

Runs a single background thread that blocks on an internal queue.  When a
document is uploaded, its ID is pushed onto the queue and the loop wakes
immediately to process it.

Documents always stay in ~/smartloop/<project-name>/documents/ (the user-visible
folder).  Conversion to markdown happens in-memory; doc.path is never updated
to an internal sources/ path.

A JSONL manifest at <project_dir>/documents.jsonl tracks every indexed file.
The scan loop uses it to detect deletions: files in the manifest but missing
from the folder are removed from the vector store and workspace config.
"""

import hashlib
import logging
import mimetypes
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional

from smartloop.db import StoreFactory
from smartloop.db.workspace_indexer import WorkspaceIndexer
from smartloop.utils import DocumentSplitter, normalize_path, json_to_table
from smartloop.parsers.parser_factory import BaseLLM, ParserFactory
from smartloop.tasks.base_task import BaseTask, TaskResult, TaskStatus
from smartloop.tasks.documents_manifest import DocumentsManifest
from smartloop.workspace import Document, DocumentStatus, Workspace


class IndexerTask(BaseTask):
    """Background task that processes documents from a queue in a persistent event loop."""

    def __init__(
        self,
        workspace: Workspace,
        logger: logging.Logger,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(name="indexer", logger=logger, auto=True, **kwargs)
        self.workspace = workspace
        self.progress_callback = progress_callback
        self._loop_thread: Optional[threading.Thread] = None
        self._queue: queue.Queue[str] = queue.Queue()
        self._max_workers: int = max(1, os.cpu_count() or 1)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._workspace_lock = threading.Lock()
        self._inflight_lock = threading.Lock()
        self._inflight: set = set()
        self._last_scan: float = 0.0
        self._SCAN_INTERVAL: float = 5.0
        # Workspace (project working-directory) indexing runs on a slower
        # cadence than the documents folder — the repo is larger and its
        # index is only consulted opportunistically during chat.
        self._last_workspace_scan: float = 0.0
        self._WORKSPACE_SCAN_INTERVAL: float = 30.0
        self._workspace_indexer: Optional[WorkspaceIndexer] = None
        self._workspace_indexer_dir: Optional[str] = None
        self._workspace_scan_inflight: bool = False
        # Wall-clock time and stats of the last completed workspace pass,
        # surfaced via workspace_status for the /health endpoint.
        self._last_workspace_scan_at: Optional[float] = None
        self._last_workspace_stats: Optional[dict] = None
        # True once a document has been processed since the queue last went
        # idle; used to release the on-demand vision model exactly once when
        # the queue drains (so only the orchestrator stays resident).
        self._processed_since_idle: bool = False

    # ------------------------------------------------------------------
    # Public queue API
    # ------------------------------------------------------------------

    def enqueue(self, documents: List[Document]) -> int:
        """Push one or more documents onto the indexer queue."""
        count = 0
        for doc in documents:
            self._queue.put(doc.id)
            count += 1
        if count:
            self.logger.info(f"[Indexer] {count} document(s) enqueued for indexing")
        return count

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    def start_loop(self) -> None:
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._stop_event.clear()
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="index-worker",
        )
        self._loop_thread = threading.Thread(
            target=self._event_loop,
            name="index-loop",
            daemon=True,
        )
        self._loop_thread.start()
        self.logger.info(f"[Indexer] Event loop started ({self._max_workers} worker(s))")

    def _event_loop(self) -> None:
        self._set_status(TaskStatus.PENDING, "Waiting for documents")

        while not self._stop_event.is_set():
            try:
                now = time.monotonic()
                if now - self._last_scan >= self._SCAN_INTERVAL:
                    self._scan_documents()
                    self._last_scan = now

                # Incrementally index the project's working directory into
                # the workspace collection.  Run it on the executor so a
                # large first-time scan never blocks document processing.
                if (
                    not self._workspace_scan_inflight
                    and self._executor is not None
                    and now - self._last_workspace_scan >= self._WORKSPACE_SCAN_INTERVAL
                ):
                    self._last_workspace_scan = now
                    self._workspace_scan_inflight = True
                    self._executor.submit(self._run_workspace_scan)

                doc_id = self._queue.get(timeout=1)
                with self._inflight_lock:
                    if doc_id in self._inflight:
                        continue
                    self._inflight.add(doc_id)
                self._set_status(TaskStatus.RUNNING, "Processing document")
                self._processed_since_idle = True
                self._executor.submit(self._worker_run, doc_id)
            except queue.Empty:
                with self._inflight_lock:
                    idle = not self._inflight
                if idle:
                    self._set_status(TaskStatus.PENDING, "Waiting for documents")
                    # Stage boundary: all queued documents are indexed, so the
                    # on-demand vision/OCR model is no longer needed — free its
                    # VRAM (only the orchestrator stays resident). Done once per
                    # drain to avoid reload churn while idle.
                    if self._processed_since_idle:
                        self._processed_since_idle = False
                        self._release_on_demand_models()
            except Exception as e:
                self.logger.error(f"[Indexer] Event loop error: {e}")

        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._set_status(TaskStatus.COMPLETED, "Event loop stopped")

    def _worker_run(self, doc_id: str) -> None:
        try:
            self._process_document_by_id(doc_id)
        except Exception as e:
            self.logger.error(f"[Indexer] Worker error for {doc_id[:8]}: {e}")
        finally:
            with self._inflight_lock:
                self._inflight.discard(doc_id)

    # ------------------------------------------------------------------
    # Single-document processing
    # ------------------------------------------------------------------

    def _process_document_by_id(self, doc_id: str) -> None:
        self._refresh_workspace()
        project = self.workspace.current_project
        if not project:
            self.logger.warning("[Indexer] No active project — skipping document")
            return

        doc = next((d for d in project.documents if d.id == doc_id), None)
        if doc is None:
            self.logger.warning(f"[Indexer] Document {doc_id[:8]} not found in project")
            return

        if doc.status not in (DocumentStatus.PENDING, DocumentStatus.CONVERTING, DocumentStatus.INDEXING):
            return

        self._process_single_document(doc, project)

    def _process_single_document(self, doc: Document, project) -> None:
        """Convert and index a document.

        doc.path always points to the original file in the user-visible
        ~/smartloop/<project-name>/documents/ folder and is never changed.
        Conversion to markdown is done in-memory so no intermediate file
        is written and the original is never moved.
        """
        doc_name = doc.name or Path(doc.path).name
        self.logger.info(f"[Indexer] Processing {doc_name}")

        if self.progress_callback:
            self.progress_callback({
                "stage": "indexing",
                "filename": doc_name,
                "current": 1,
                "total": 1 + self._queue.qsize(),
                "message": f"Indexing {doc_name}...",
            })

        manifest = self._get_manifest(project)

        try:
            doc_path = Path(doc.path)
            if not doc_path.exists():
                self.logger.warning(f"[Indexer] Source file missing, skipping: {doc.path}")
                return

            # Ensure document is in the manifest so the scan loop can
            # detect deletions later (API-uploaded docs skip the scan
            # that normally creates manifest entries).
            manifest.upsert(
                doc_id=doc.id,
                filename=doc_name,
                path=str(doc_path),
                status=DocumentStatus.CONVERTING.value,
                file_hash=doc.id,
            )

            with self._workspace_lock:
                doc.status = DocumentStatus.CONVERTING
                self.workspace.save()

            # --- Convert to plain text / markdown in-memory ---
            mime_type, _ = mimetypes.guess_type(str(doc_path))
            if doc_path.suffix.lower() in ('.txt', '.md'):
                text_content = doc_path.read_text(encoding="utf-8", errors="replace")
            else:
                self.logger.info(f"[Indexer] Converting {doc_name} to markdown")
                try:
                    llm = self._ensure_model_loaded()
                    parser = ParserFactory(llm).get_parser(mime_type, str(doc_path), self.logger)
                    text_content = parser.parse()
                except ValueError as ve:
                    self.logger.warning(f"[Indexer] Failed to convert {doc_name}: {ve}")
                    with self._workspace_lock:
                        doc.status = DocumentStatus.FAILED
                        self.workspace.save()
                    manifest.update_status(doc.id, DocumentStatus.FAILED.value)
                    return

            with self._workspace_lock:
                doc.status = DocumentStatus.INDEXING
                self.workspace.save()
            manifest.update_status(doc.id, DocumentStatus.INDEXING.value)

            store = StoreFactory.create_store(
                project.project_dir,
                logger=self.logger,
                base_dir=self.workspace.workspace_dir,
            )

            llm = self._ensure_model_loaded()

            # Structured JSON documents (an array of objects) are converted to
            # a markdown table BEFORE chunking, so the vector index holds clean
            # table rows instead of fragmented JSON text — and the query-time
            # extractor can use the normalized table directly (no overflow).
            table = json_to_table(text_content)
            if table is not None:
                self.logger.info(
                    f"[Indexer] {doc_name}: converted JSON → table "
                    f"({len(table)} chars)"
                )
                text_content = table

            # Split the in-memory text — doc.path stays on the original file
            text_chunks = DocumentSplitter(llm=llm, logger=self.logger).split_text(text_content)

            total_chunks = len(text_chunks)

            chunk_dicts = []
            for idx, chunk in enumerate(text_chunks):
                page_number = self._estimate_page_number(
                    chunk, idx, total_chunks, text_content
                )

                chunk_dicts.append({
                    "text": self._compact_for_index(chunk),
                    "document_id": doc.id,
                    "document_name": doc_name,
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "page_number": page_number,
                    "source_content": chunk,
                })

            store.add_chunks(chunk_dicts)

            with self._workspace_lock:
                doc.status = DocumentStatus.INDEXED
                self.workspace.save()

            manifest.update_status(doc.id, DocumentStatus.INDEXED.value)
            self.logger.info(f"[Indexer] Indexed {doc_name}")

        except Exception as e:
            self.logger.error(f"[Indexer] Failed to process {doc_name}: {e}")

    # ------------------------------------------------------------------
    # Index text preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _compact_for_index(text: str) -> str:
        """Strip formatting and compress text for embedding search.

        The indexed text is used only for semantic similarity matching.
        The full original lives in ``source_content`` metadata for LLM
        analysis.  Stripping markdown, punctuation noise, and excess
        whitespace produces tighter embeddings.
        """
        import re as _re
        # Remove markdown formatting (headers, bold, italic, links, images)
        s = _re.sub(r'#{1,6}\s*', '', text)
        s = _re.sub(r'\*{1,3}|_{1,3}', '', s)
        s = _re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', s)
        s = _re.sub(r'!\[([^\]]*)\]\([^)]*\)', '', s)
        # Remove HTML tags
        s = _re.sub(r'<[^>]+>', '', s)
        # Remove bullet/list markers
        s = _re.sub(r'^\s*[-*+]\s+', '', s, flags=_re.MULTILINE)
        s = _re.sub(r'^\s*\d+\.\s+', '', s, flags=_re.MULTILINE)
        # Collapse whitespace
        s = _re.sub(r'\s+', ' ', s).strip()
        return s

    # ------------------------------------------------------------------
    # Page estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_page_number(
        chunk: str, chunk_idx: int, total_chunks: int, full_text: str
    ) -> int:
        """Estimate the page number for a chunk.

        Uses explicit page-break markers (``---``, ``\\f``, or markdown
        page headings) when present in the full text.  Falls back to a
        proportional estimate based on chunk position.
        """
        import re as _re

        # Look for form-feed characters (PDF page breaks in raw text)
        page_breaks = [m.start() for m in _re.finditer(r'\f', full_text)]
        if page_breaks:
            # Find where this chunk starts in the full text
            chunk_start = full_text.find(chunk[:80])
            if chunk_start >= 0:
                page = 1
                for bp in page_breaks:
                    if bp < chunk_start:
                        page += 1
                    else:
                        break
                return page

        # Fallback: proportional estimate (assume ~1 page per 3000 chars)
        chars_per_page = 3000
        total_pages = max(1, len(full_text) // chars_per_page)
        chunk_start = full_text.find(chunk[:80])
        if chunk_start >= 0:
            return min(total_pages, max(1, (chunk_start // chars_per_page) + 1))

        # Last resort: proportional from chunk index
        return max(1, int((chunk_idx / max(total_chunks, 1)) * total_pages) + 1)

    # ------------------------------------------------------------------
    # Folder scan — new files and deletions
    # ------------------------------------------------------------------

    def _scan_documents(self) -> None:
        """Scan the project's documents folder, register new files, and remove deleted ones.

        Uses the JSONL manifest as the reference of what has been indexed so
        that deletions from the folder are reflected in the vector store and
        workspace config.
        """
        self._refresh_workspace()
        project = self.workspace.current_project
        if not project:
            return

        base = normalize_path(project.visible_dir)
        inbox_dir = Path(base) / "documents"
        if not inbox_dir.is_dir():
            return

        manifest = self._get_manifest(project)
        manifest_entries = manifest.entries()  # {doc_id: entry}

        # Current files on disk (skip hidden/system files)
        disk_files: dict[str, Path] = {}  # filename → path
        for fp in inbox_dir.iterdir():
            if fp.is_file() and not fp.name.startswith("."):
                disk_files[fp.name] = fp

        # --- Detect deletions (manifest) ---
        for doc_id, entry in list(manifest_entries.items()):
            if entry["filename"] not in disk_files:
                self.logger.info(f"[Indexer] File removed from folder: {entry['filename']} — cleaning up")
                self._remove_document(doc_id, project)
                manifest.remove(doc_id)

        # --- Detect deletions (workspace docs missing from disk) ---
        # Catches documents added via the API that were indexed before the
        # manifest upsert fix, so they have no manifest entry.
        for doc in list(project.documents):
            if doc.status != DocumentStatus.INDEXED:
                continue
            if doc.path and not Path(doc.path).exists():
                self.logger.info(f"[Indexer] File missing from disk: {doc.name or doc.path} — cleaning up")
                self._remove_document(doc.id, project)
                manifest.remove(doc.id)

        # --- Detect new files ---
        existing_hashes = {doc.id for doc in project.documents}
        existing_paths = {doc.path for doc in project.documents}

        new_docs: list[Document] = []
        for filename, file_path in disk_files.items():
            if str(file_path) in existing_paths:
                continue
            try:
                with open(file_path, "rb") as fh:
                    doc_hash = hashlib.md5(fh.read()).hexdigest()
                if doc_hash in existing_hashes:
                    continue
                doc = Document(
                    id=doc_hash,
                    path=str(file_path),
                    name=filename,
                    summary=None,
                    status=DocumentStatus.PENDING,
                )
                new_docs.append(doc)
                manifest.upsert(
                    doc_id=doc_hash,
                    filename=filename,
                    path=str(file_path),
                    status=DocumentStatus.PENDING.value,
                    file_hash=doc_hash,
                )
            except Exception as e:
                self.logger.warning(f"[Indexer] Could not register {filename}: {e}")

        if new_docs:
            with self._workspace_lock:
                project.documents.extend(new_docs)
                self.workspace.save()
            self.logger.info(f"[Indexer] Picked up {len(new_docs)} new file(s)")
            self.enqueue(new_docs)

    def _remove_document(self, doc_id: str, project) -> None:
        """Remove a document from the vector store and workspace config."""
        try:
            store = StoreFactory.create_store(
                project.project_dir,
                logger=self.logger,
                base_dir=self.workspace.workspace_dir,
            )
            store.delete_document(doc_id)
        except Exception as e:
            self.logger.warning(f"[Indexer] Could not remove chunks for {doc_id[:8]}: {e}")

        with self._workspace_lock:
            project.documents = [d for d in project.documents if d.id != doc_id]
            self.workspace.save()

    # ------------------------------------------------------------------
    # Workspace (project working-directory) indexing
    # ------------------------------------------------------------------

    def request_workspace_scan(self) -> None:
        """Ask the event loop to run a workspace index pass on its next tick.

        Resetting the timer makes the loop fire a scan within ~1s (the queue
        poll interval) instead of waiting out the full scan interval.  The
        scan still runs on the executor, so callers are never blocked.  Used
        on project switch and at the start of a chat request so freshly-edited
        files are picked up promptly.
        """
        self._last_workspace_scan = 0.0

    def _run_workspace_scan(self) -> None:
        """Executor entry point for the background workspace index pass."""
        try:
            self._scan_workspace_files()
        except Exception as e:
            self.logger.warning(f"[Indexer] Workspace scan failed: {e}")
        finally:
            self._workspace_scan_inflight = False

    def _scan_workspace_files(self) -> None:
        """Incrementally index the project's working-directory files.

        Mirrors ``_scan_documents`` but targets the code/workspace files in
        the project's visible directory, indexing them into the dedicated
        ``workspace_files`` collection.  Indexing here keeps it off the
        request path — ``ModelService.search_workspace`` only queries the
        store.  The underlying :class:`WorkspaceIndexer` is hash-incremental,
        so repeat passes only re-embed changed files.
        """
        self._refresh_workspace()
        project = self.workspace.current_project
        if not project:
            return

        work_dir = project.visible_dir
        if not work_dir or not os.path.isdir(work_dir):
            return

        # Recreate the indexer when the active project changes so the store
        # and manifest always point at the current project's directory.
        if (
            self._workspace_indexer is None
            or self._workspace_indexer_dir != project.project_dir
        ):
            self._workspace_indexer = WorkspaceIndexer(
                project_dir=project.project_dir,
                logger=self.logger,
                base_dir=self.workspace.workspace_dir,
            )
            self._workspace_indexer_dir = project.project_dir

        stats = self._workspace_indexer.index(work_dir)
        self._last_workspace_stats = stats
        self._last_workspace_scan_at = time.time()
        if stats.get("indexed") or stats.get("removed"):
            self.logger.info(
                f"[Indexer] Workspace index — indexed={stats['indexed']}, "
                f"skipped={stats['skipped']}, removed={stats['removed']}"
            )

    @property
    def workspace_status(self) -> dict:
        """Snapshot of background workspace-file indexing for /health.

        Reports whether a scan is in flight, how many files are currently
        indexed, when the last pass completed (epoch seconds), the last
        pass's stats, and the scan cadence.
        """
        indexer = self._workspace_indexer
        return {
            "scanning": self._workspace_scan_inflight,
            "files_indexed": indexer.file_count if indexer else 0,
            "last_scan_at": self._last_workspace_scan_at,
            "last_stats": self._last_workspace_stats,
            "scan_interval_seconds": self._WORKSPACE_SCAN_INTERVAL,
        }

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def should_run(self) -> bool:
        return not self._queue.empty()

    def execute(self, **kwargs) -> TaskResult:
        """Batch fallback for TaskQueue.submit() compatibility."""
        try:
            self._refresh_workspace()
            project = self.workspace.current_project
            if not project:
                return TaskResult(success=False, error="No active project")

            queued_ids = self._drain_queue()
            if not queued_ids:
                return TaskResult(success=True, data={"processed": 0})

            doc_by_id = {doc.id: doc for doc in project.documents}
            processed = 0
            for doc_id in queued_ids:
                if self._stop_event.is_set():
                    break
                doc = doc_by_id.get(doc_id)
                if doc and doc.status in (DocumentStatus.PENDING, DocumentStatus.CONVERTING, DocumentStatus.INDEXING):
                    self._process_single_document(doc, project)
                    if doc.status == DocumentStatus.INDEXED:
                        processed += 1

            return TaskResult(success=True, data={"processed": processed})

        except Exception as e:
            self.logger.error(f"[Indexer] Processing failed: {e}")
            return TaskResult(success=False, error=str(e))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_manifest(self, project) -> DocumentsManifest:
        manifest_path = Path(project.project_dir) / "documents.jsonl"
        return DocumentsManifest(manifest_path)

    def _refresh_workspace(self) -> None:
        current = Workspace.get_instance()
        if current is not None and current is not self.workspace:
            self.logger.info("[Indexer] Workspace singleton changed — refreshing reference")
            self.workspace = current

    def _drain_queue(self) -> set:
        ids: set = set()
        while True:
            try:
                ids.add(self._queue.get_nowait())
            except queue.Empty:
                break
        return ids

    def _ensure_model_loaded(self) -> BaseLLM:
        from smartloop.api.services import get_model_service
        model_service = get_model_service()
        llm = model_service.get_llm()
        if llm is None:
            llm = model_service.load_model()
        return llm

    def _release_on_demand_models(self) -> None:
        """Free on-demand models when indexing goes idle (orchestrator stays)."""
        try:
            from smartloop.api.services import get_model_service
            get_model_service().unload_vision_model()
        except Exception as e:
            self.logger.warning(f"[Indexer] Failed to release vision model: {e}")

    def start_watcher(self) -> None:
        """Alias for start_loop()."""
        self.start_loop()
