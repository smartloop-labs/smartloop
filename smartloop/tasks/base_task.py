"""
Base Task for long-running background work.

Each task runs in its own isolated thread with retry logic
and status tracking. Tasks can use agents to fulfill their work.
"""

import logging
import threading
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Optional


class TaskStatus(str, Enum):
    """Status of a task in the task queue."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class TaskResult:
    """Result of a task execution."""

    def __init__(self, success: bool, data: Any = None, error: Optional[str] = None):
        self.success = success
        self.data = data
        self.error = error

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
        }


class BaseTask(ABC):
    """Base class for long-running isolated tasks.

    Each task:
    - Runs in its own isolated daemon thread
    - Has configurable auto-start behaviour
    - Supports retry with exponential backoff on failure
    - Emits status updates via an optional callback
    - Can use agents to fulfill its work
    """

    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 2  # seconds; doubles each attempt

    def __init__(
        self,
        name: str,
        logger: logging.Logger,
        auto: bool = False,
        max_retries: int = MAX_RETRIES,
        on_status_change: Optional[Callable[["BaseTask", TaskStatus, Optional[str]], None]] = None,
    ):
        self.name = name
        self.logger = logger
        self.auto = auto
        self.max_retries = max_retries
        self.on_status_change = on_status_change

        self._status = TaskStatus.PENDING
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._retry_count = 0
        self._last_error: Optional[str] = None
        self._result: Optional[TaskResult] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def status(self) -> TaskStatus:
        return self._status

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def result(self) -> Optional[TaskResult]:
        return self._result

    @property
    def is_running(self) -> bool:
        return self._status == TaskStatus.RUNNING

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abstractmethod
    def execute(self, **kwargs) -> TaskResult:
        """Execute the task's work.  Must be implemented by subclasses.

        Returns a ``TaskResult``.  Raising an exception is treated as failure
        and triggers a retry (up to ``max_retries``).
        """
        ...

    def should_run(self) -> bool:
        """Override to add pre-condition checks (e.g. config changed).

        Called before ``execute`` on auto-triggered tasks.  Return ``False``
        to skip execution.
        """
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, **kwargs) -> None:
        """Submit the task for execution in its own thread."""
        with self._lock:
            if self._status == TaskStatus.RUNNING:
                self.logger.warning(f"[{self.name}] Already running — ignoring submit")
                return
            self._retry_count = 0
            self._stop_event.clear()
            self._start_thread(**kwargs)

    def stop(self) -> None:
        """Signal the task to stop gracefully."""
        self._stop_event.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the task thread finishes.  Returns True if completed."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            return not self._thread.is_alive()
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_status(self, status: TaskStatus, message: Optional[str] = None) -> None:
        self._status = status
        if self.on_status_change:
            try:
                self.on_status_change(self, status, message)
            except Exception:
                pass

    def _start_thread(self, **kwargs) -> None:
        self._thread = threading.Thread(
            target=self._run_with_retries,
            kwargs=kwargs,
            name=f"task-{self.name}",
            daemon=True,
        )
        self._thread.start()

    # Exceptions that should fail immediately without retrying
    NON_RETRYABLE_ERRORS = (PermissionError,)

    def _run_with_retries(self, **kwargs) -> None:
        """Execute with retry + exponential backoff."""
        while self._retry_count <= self.max_retries:
            if self._stop_event.is_set():
                self._set_status(TaskStatus.FAILED, "Stopped by user")
                return

            try:
                self._set_status(TaskStatus.RUNNING, f"Attempt {self._retry_count + 1}")
                self.logger.info(
                    f"[{self.name}] Starting (attempt {self._retry_count + 1}/{self.max_retries + 1})"
                )
                result = self.execute(**kwargs)
                self._result = result

                if result.success:
                    self._set_status(TaskStatus.COMPLETED, "Success")
                    self.logger.info(f"[{self.name}] Completed successfully")
                    return
                else:
                    raise RuntimeError(result.error or "Task returned failure")

            except Exception as exc:
                self._last_error = str(exc)
                self.logger.error(
                    f"[{self.name}] Failed (attempt {self._retry_count + 1}): {exc}"
                )
                self.logger.debug(traceback.format_exc())

                # Non-retryable errors (e.g. 403 Forbidden) fail immediately
                if isinstance(exc, self.NON_RETRYABLE_ERRORS) or isinstance(exc.__cause__, self.NON_RETRYABLE_ERRORS):
                    self._result = TaskResult(success=False, error=self._last_error)
                    self._set_status(TaskStatus.FAILED, self._last_error)
                    self.logger.error(f"[{self.name}] Non-retryable error — failing immediately")
                    return

                self._retry_count += 1
                if self._retry_count > self.max_retries:
                    self._result = TaskResult(success=False, error=self._last_error)
                    self._set_status(TaskStatus.FAILED, self._last_error)
                    self.logger.error(
                        f"[{self.name}] Exhausted all {self.max_retries + 1} attempts"
                    )
                    return

                backoff = self.RETRY_BACKOFF_BASE ** self._retry_count
                self._set_status(TaskStatus.RETRYING, f"Retrying in {backoff}s")
                self.logger.info(f"[{self.name}] Retrying in {backoff}s...")

                # Interruptible sleep
                if self._stop_event.wait(timeout=backoff):
                    self._set_status(TaskStatus.FAILED, "Stopped during backoff")
                    return
