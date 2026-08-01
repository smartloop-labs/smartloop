from abc import abstractmethod
from pathlib import Path
import logging
import threading
from typing import Callable, Optional

from smartloop.config import ModelConfig
from smartloop.workspace import Workspace, Project


class DownloadProgress:
    """Progress info emitted during model download."""
    def __init__(self, filename: str, downloaded: int, total: int, status: str = "downloading"):
        self.filename = filename
        self.downloaded = downloaded
        self.total = total
        self.status = status

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "downloaded": self.downloaded,
            "total": self.total,
            "status": self.status,
        }


# Type alias for progress callbacks
ProgressCallback = Callable[[DownloadProgress], None]


class AggregateProgress:
    """Roll up per-file progress into a single 0->100% download bar.

    A multi-file model (e.g. a vision model with a main GGUF plus an mmproj /
    CLIP projector) downloads its files in parallel, and each file reports its
    own cumulative ``downloaded``/``total``. Forwarding those raw per-file
    events makes the client's percentage jump between files. This wrapper sums
    the per-file byte counts against the known grand total and emits one
    aggregate ``DownloadProgress`` under a single stable ``label``.

    Thread-safe: the parallel download workers all call the same instance.
    """

    def __init__(self, total_bytes: int, callback: Optional[ProgressCallback],
                 label: str):
        self._total = total_bytes
        self._callback = callback
        self._label = label
        self._lock = threading.Lock()
        self._per_file: dict[str, int] = {}

    def __call__(self, progress: DownloadProgress) -> None:
        if not self._callback:
            return
        with self._lock:
            # Per-file counts only ever move forward — ignore any regression
            # from a resumed/restarted file so the rolled-up bar never dips.
            prev = self._per_file.get(progress.filename, 0)
            self._per_file[progress.filename] = max(prev, progress.downloaded)
            downloaded = sum(self._per_file.values())
        if self._total > 0:
            downloaded = min(downloaded, self._total)
        self._callback(DownloadProgress(
            filename=self._label,
            downloaded=downloaded,
            total=self._total,
            status="downloading",
        ))

    def finish(self) -> None:
        """Emit a terminal 100% completed event for the whole download."""
        if not self._callback:
            return
        self._callback(DownloadProgress(
            filename=self._label,
            downloaded=self._total,
            total=self._total,
            status="completed",
        ))


class SLPDownloader:
    def __init__(self, workspace: Workspace, logger: logging.Logger):
        self.workspace = workspace
        self.logger = logger

    @abstractmethod
    def download(self, model_id: str, progress_callback: Optional[ProgressCallback] = None) -> Path:
        pass
