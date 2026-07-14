"""
Task queue — manages submission, scheduling, and lifecycle of Tasks.

All tasks are registered at startup and executed in isolated threads.
Failed tasks are automatically retried.  The queue provides a unified
status view for the API layer.
"""

import logging
import threading
from typing import Callable, Dict, List, Optional

from smartloop.tasks.base_task import BaseTask, TaskStatus, TaskResult


class TaskQueue:
    """Central registry and scheduler for Tasks.

    Usage::

        queue = TaskQueue(logger)
        queue.register(downloader_task)
        queue.register(build_task)
        queue.start_auto_tasks()       # kicks off tasks with auto=True
        queue.submit("downloader")     # manually trigger by name
        queue.shutdown()               # graceful stop
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._tasks: Dict[str, BaseTask] = {}
        self._lock = threading.Lock()
        self._status_callbacks: List[Callable] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, task: BaseTask) -> None:
        """Register a task.  Overwrites any existing task with the same name."""
        with self._lock:
            task.on_status_change = self._on_task_status_change
            self._tasks[task.name] = task
            self.logger.info(f"[TaskQueue] Registered task: {task.name} (auto={task.auto})")

    def unregister(self, name: str) -> None:
        with self._lock:
            task = self._tasks.pop(name, None)
            if task:
                task.stop()

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(self, name: str, **kwargs) -> bool:
        """Submit a task for execution by name.

        Returns True if the task was found and submitted.
        """
        task = self._tasks.get(name)
        if task is None:
            self.logger.warning(f"[TaskQueue] Unknown task: {name}")
            return False
        task.submit(**kwargs)
        return True

    def start_auto_tasks(self) -> None:
        """Start all tasks that have ``auto=True`` and pass their ``should_run`` check."""
        for task in self._tasks.values():
            if task.auto and task.should_run():
                self.logger.info(f"[TaskQueue] Auto-starting task: {task.name}")
                task.submit()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self, name: Optional[str] = None) -> dict:
        """Return status dict for one or all tasks."""
        if name:
            task = self._tasks.get(name)
            if task is None:
                return {"error": f"Unknown task: {name}"}
            return self._task_status_dict(task)

        return {
            name: self._task_status_dict(task)
            for name, task in self._tasks.items()
        }

    def get_task(self, name: str) -> Optional[BaseTask]:
        """Return the registered task by name, or None."""
        return self._tasks.get(name)

    def on_status_change(self, callback: Callable) -> None:
        """Register a global status-change listener."""
        self._status_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def wait_for(self, name: str, timeout: Optional[float] = None) -> bool:
        """Block until a specific task completes.  Returns True if completed."""
        task = self._tasks.get(name)
        if task is None:
            return True
        return task.wait(timeout=timeout)

    def shutdown(self) -> None:
        """Stop all tasks gracefully."""
        self.logger.info("[TaskQueue] Shutting down all tasks...")
        for task in self._tasks.values():
            task.stop()
        # Give threads a moment to wind down
        for task in self._tasks.values():
            task.wait(timeout=5)
        self.logger.info("[TaskQueue] All tasks stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _task_status_dict(task: BaseTask) -> dict:
        status = {
            "name": task.name,
            "status": task.status.value,
            "auto": task.auto,
            "last_error": task.last_error,
        }
        # Expose queue depth for tasks that support it (e.g. IndexerTask)
        if hasattr(task, "queue_size"):
            status["queued"] = task.queue_size
        # Expose background workspace-indexing state (IndexerTask)
        if hasattr(task, "workspace_status"):
            status["workspace"] = task.workspace_status
        return status

    def _on_task_status_change(
        self, task: BaseTask, status: TaskStatus, message: Optional[str]
    ) -> None:
        """Relay individual task status changes to global listeners."""
        for cb in self._status_callbacks:
            try:
                cb(task.name, status, message)
            except Exception:
                pass
