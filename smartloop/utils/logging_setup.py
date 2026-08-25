"""Centralized logging setup for the API server and its spawn workers.

All threads (TaskQueue jobs, IndexerTask workers, DAG executor, hybrid
executor I/O pool) and all spawn worker processes (the LLM
``ProcessPoolExecutor``) emit through a single ``QueueHandler`` on the
root logger.  A ``QueueListener`` thread in the parent process owns the
real handlers (rotating file + optional console) and writes records in
arrival order so concurrent producers never tear a line.

Used by:
- ``smartloop.api.main.run_server`` / ``start_server_foreground`` — installs
  the queue + listener.
- ``smartloop.agents.hybrid_executor.LLMExecutor`` — pulls the shared queue
  via :func:`get_log_queue` and passes it to the spawn pool initializer.
- ``smartloop.agents.llm_worker.init_worker`` — each spawned worker calls
  :func:`worker_setup_logging` so its log records flow back to the parent
  listener.
"""

from __future__ import annotations

import gc
import logging
import logging.handlers
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Optional

# Process-wide singleton — set by make_log_queue() and read by
# the hybrid executor when it spawns the worker pool.
_LOG_QUEUE: Optional["multiprocessing.queues.Queue"] = None

DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5


def make_log_queue() -> "multiprocessing.queues.Queue":
    """Create the process-wide log queue. Idempotent — returns the existing
    queue if one was already created.

    A ``multiprocessing.Queue`` is used so log records from spawn worker
    processes (the LLM ``ProcessPoolExecutor``) reach the parent listener.
    It relies on a POSIX named semaphore (``sem_open``), which the macOS App
    Sandbox (Mac App Store build) denies with ``PermissionError``/``OSError``.
    When that happens we fall back to a plain thread-safe ``queue.Queue``:
    under the sandbox the LLM pool isn't spawned anyway (Apple Silicon caps at
    one worker and reuses the in-process model — see hybrid_executor), so every
    producer is a thread in this process and the thread queue is sufficient.
    ``QueueHandler``/``QueueListener`` treat both queues identically."""
    global _LOG_QUEUE
    if _LOG_QUEUE is None:
        try:
            _LOG_QUEUE = multiprocessing.Queue(-1)
        except (PermissionError, OSError):
            import queue as _queue
            _LOG_QUEUE = _queue.Queue(-1)
    return _LOG_QUEUE


def get_log_queue() -> Optional["multiprocessing.queues.Queue"]:
    """Return the shared log queue, or ``None`` if logging hasn't been set up
    yet (e.g. running outside the server, in unit tests, or before startup)."""
    return _LOG_QUEUE


def _resolve_level(level) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return logging.getLevelName(level.upper())
    return logging.INFO


def setup_root_logging(
    queue: "multiprocessing.queues.Queue",
    *,
    level="DEBUG",
) -> None:
    """Replace root handlers with a single ``QueueHandler``.

    Root level is set permissively (DEBUG by default) so per-handler levels
    on the listener side decide what actually gets emitted.
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(logging.handlers.QueueHandler(queue))
    root.setLevel(_resolve_level(level))


def start_listener(
    queue: "multiprocessing.queues.Queue",
    *,
    home_dir: str,
    file_level="INFO",
    console_level="INFO",
    force_console: Optional[bool] = None,
) -> logging.handlers.QueueListener:
    """Start the parent-process listener that owns the real handlers.

    Always adds a ``RotatingFileHandler`` writing to ``<home_dir>/server.log``.
    Adds a stdout ``StreamHandler`` when running interactively (TTY and not a
    PyInstaller frozen build), unless ``force_console`` overrides the
    auto-detection.
    """
    log_path = Path(home_dir) / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(DEFAULT_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        str(log_path),
        maxBytes=DEFAULT_MAX_BYTES,
        backupCount=DEFAULT_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(_resolve_level(file_level))
    file_handler.setFormatter(formatter)

    handlers: list[logging.Handler] = [file_handler]

    if force_console is None:
        console = sys.stdout.isatty() and not getattr(sys, "frozen", False)
    else:
        console = force_console

    if console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(_resolve_level(console_level))
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

    listener = logging.handlers.QueueListener(
        queue, *handlers, respect_handler_level=True
    )
    listener.start()
    return listener


def shutdown_logging(listener: Optional[logging.handlers.QueueListener] = None) -> None:
    """Stop the listener and release the shared log queue's OS resources.

    A ``multiprocessing.Queue`` owns three POSIX semaphores that are unlinked
    by finalizers which only run on a *clean* interpreter exit.  A server that
    is SIGKILLed — ``slp agent stop`` escalates to SIGKILL when SIGTERM hasn't
    landed within 10s — never runs them, and the resource tracker then prints
    ``resource_tracker: There appear to be 3 leaked semaphore objects to clean
    up at shutdown``.  Releasing the queue during the lifespan shutdown means
    the semaphores are already gone by the time that window opens.

    Safe to call more than once, and safe when the fallback ``queue.Queue``
    (macOS App Sandbox) is in use — that one owns no OS resources.
    """
    global _LOG_QUEUE

    if listener is not None:
        try:
            listener.stop()
        except Exception:
            pass
        # ``stop()`` joins the listener thread but keeps ``listener.queue``
        # set, and the caller still holds the listener itself — that single
        # strong reference is enough to keep the Queue (and therefore its
        # semaphores) alive past this function.  Drop it explicitly.
        listener.queue = None

    # Detach the QueueHandler first: a record emitted after the queue closes
    # would raise inside logging.  Root falls back to logging.lastResort,
    # which still puts WARNING+ on stderr.
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.handlers.QueueHandler):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
            # Handler.close() only deregisters the handler from logging's
            # internal tables; ``h.queue`` stays bound, so clear it too.
            h.queue = None

    queue, _LOG_QUEUE = _LOG_QUEUE, None
    # queue.Queue has no close(); only the multiprocessing one holds semaphores.
    if queue is not None and hasattr(queue, "close"):
        try:
            queue.close()
            queue.join_thread()
        except Exception:
            pass

    # close() only tears down the pipe.  Each of the three semaphores is
    # unlinked by a ``util.Finalize`` that fires when its ``SemLock`` is
    # collected — so waiting for the *queue* to become unreachable makes
    # cleanup hostage to every reference anywhere in the process (a caller's
    # local, an ``LLMExecutor`` that outlived teardown).  Detaching the locks
    # makes them unreachable on their own, so the standard finalizers run
    # here, under our control, whoever else still holds the queue.  Safe
    # because the queue is closed: ``put``/``get`` check ``_closed`` and
    # raise ValueError before they ever reach these locks.
    locks, lock = [], None  # bound up front: the loop may not run at all
    for attr in ("_rlock", "_wlock", "_sem"):
        lock = getattr(queue, attr, None)
        if lock is None:
            continue
        locks.append(lock)
        try:
            setattr(queue, attr, None)
        except Exception:
            locks.pop()

    del queue, locks, lock
    gc.collect()


def worker_setup_logging(queue: Optional["multiprocessing.queues.Queue"]) -> None:
    """Configure the root logger inside a spawn worker process.

    When ``queue`` is provided, all worker log records flow back to the
    parent's :class:`QueueListener`.  When ``None`` (tests, ad-hoc usage),
    fall back to a stderr handler so the worker is still observable.
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)

    if queue is not None:
        root.addHandler(logging.handlers.QueueHandler(queue))
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
