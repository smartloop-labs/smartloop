"""Tasks — isolated units of work that run in background threads."""

from .base_task import BaseTask, TaskStatus, TaskResult
from .task_queue import TaskQueue
from .downloader_task import DownloaderTask
from .indexer_task import IndexerTask
from .mcp_task import MCPTask
from .documents_manifest import DocumentsManifest

# Training/build tasks live in smartloop.trainer (optional dependency).
try:
    from smartloop_trainer.build_task import BuildTask
    from smartloop_trainer.train_task import TrainTask
except ImportError:
    BuildTask = None
    TrainTask = None

__all__ = [
    "BaseTask",
    "TaskStatus",
    "TaskResult",
    "TaskQueue",
    "DownloaderTask",
    "BuildTask",
    "TrainTask",
    "IndexerTask",
    "MCPTask",
    "DocumentsManifest",
]
