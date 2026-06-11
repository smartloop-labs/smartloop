"""JSONL manifest that tracks the indexing state of a project's documents folder.

Each line in the JSONL file is one document entry:
    {"doc_id": "...", "filename": "...", "path": "...", "status": "...", "hash": "..."}

The manifest is the source of truth for the indexer watcher:
- files in the folder but not in the manifest  → new, enqueue for indexing
- files in the manifest but not in the folder  → deleted, remove from store + config
"""

import json
import threading
from pathlib import Path
from typing import Optional


class DocumentsManifest:
    """Read/write a per-project JSONL manifest for the documents inbox folder."""

    def __init__(self, manifest_path: Path):
        self._path = manifest_path
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def entries(self) -> dict[str, dict]:
        """Return all entries keyed by ``doc_id``."""
        result: dict[str, dict] = {}
        if not self._path.exists():
            return result
        with self._lock:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    result[entry["doc_id"]] = entry
                except (json.JSONDecodeError, KeyError):
                    continue
        return result

    def upsert(self, doc_id: str, filename: str, path: str, status: str, file_hash: str) -> None:
        """Add or update the entry for *doc_id*."""
        entry = {
            "doc_id": doc_id,
            "filename": filename,
            "path": path,
            "status": status,
            "hash": file_hash,
        }
        with self._lock:
            current = self._load_raw()
            current[doc_id] = entry
            self._write_raw(current)

    def update_status(self, doc_id: str, status: str) -> None:
        """Update just the status field of an existing entry."""
        with self._lock:
            current = self._load_raw()
            if doc_id in current:
                current[doc_id]["status"] = status
                self._write_raw(current)

    def remove(self, doc_id: str) -> None:
        """Remove an entry from the manifest."""
        with self._lock:
            current = self._load_raw()
            if doc_id in current:
                del current[doc_id]
                self._write_raw(current)

    def filenames(self) -> set[str]:
        """Return the set of all tracked filenames."""
        return {e["filename"] for e in self.entries().values()}

    def paths(self) -> set[str]:
        """Return the set of all tracked file paths."""
        return {e["path"] for e in self.entries().values()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_raw(self) -> dict[str, dict]:
        """Read the JSONL file and return a dict keyed by doc_id (no lock)."""
        result: dict[str, dict] = {}
        if not self._path.exists():
            return result
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                result[entry["doc_id"]] = entry
            except (json.JSONDecodeError, KeyError):
                continue
        return result

    def _write_raw(self, entries: dict[str, dict]) -> None:
        """Rewrite the JSONL file from the in-memory dict (no lock)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(e, ensure_ascii=False) for e in entries.values()]
        self._path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
