from abc import abstractmethod
from pathlib import Path
import logging
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


class SLPDownloader:
    def __init__(self, workspace: Workspace, logger: logging.Logger):
        self.workspace = workspace
        self.logger = logger

    @abstractmethod
    def download(self, model_id: str, progress_callback: Optional[ProgressCallback] = None) -> Path:
        pass
