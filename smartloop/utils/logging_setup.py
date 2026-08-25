"""Centralized logging setup for the API server.

All threads (TaskQueue jobs, IndexerTask workers, DAG executor, hybrid
executor I/O pool) emit through a single ``QueueHandler`` on the root
logger.  A ``QueueListener`` thread owns the real handlers (rotating file
+ optional console) and writes records in arrival order so concurrent
producers never tear a line.

The queue is a plain ``queue.Queue``.  It used to be a
``multiprocessing.Queue`` so that the LLM ``ProcessPoolExecutor``'s spawn
workers could write to it from another process, at a cost of three POSIX
named semaphores per startup — semaphores that had to be unlinked before
the process died or the resource tracker reported them leaked, and that
``sem_open`` refuses outright under the macOS App Sandbox.  That worker
pool is gone (see :mod:`smartloop.agents.hybrid_executor`), so every
producer is a thread in this process and none of that applies.

Used by ``smartloop.api.main.run_server`` / ``start_server_foreground``,
which install the queue + listener.
"""

from __future__ import annotations

import logging
import logging.handlers
import queue as _queue
import sys
from pathlib import Path
from typing import Optional

# Process-wide singleton — set by make_log_queue().
_LOG_QUEUE: Optional[_queue.Queue] = None

DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5


def make_log_queue() -> _queue.Queue:
    """Create the process-wide log queue. Idempotent — returns the existing
    queue if one was already created."""
    global _LOG_QUEUE
    if _LOG_QUEUE is None:
        _LOG_QUEUE = _queue.Queue(-1)
    return _LOG_QUEUE


def get_log_queue() -> Optional[_queue.Queue]:
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
    queue: _queue.Queue,
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
    queue: _queue.Queue,
    *,
    home_dir: str,
    file_level="INFO",
    console_level="INFO",
    force_console: Optional[bool] = None,
) -> logging.handlers.QueueListener:
    """Start the listener that owns the real handlers.

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
    """Stop the listener and detach the shared queue.

    The queue owns no OS resources, so this is only about not leaving a
    ``QueueHandler`` pointing at a listener that no longer drains it — a
    record emitted after this would otherwise pile up unread.  Root falls
    back to ``logging.lastResort``, which still puts WARNING+ on stderr.

    Safe to call more than once.
    """
    global _LOG_QUEUE

    if listener is not None:
        try:
            listener.stop()
        except Exception:
            pass
        # stop() joins the listener thread but leaves listener.queue bound.
        listener.queue = None

    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.handlers.QueueHandler):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
            # Handler.close() only deregisters the handler from logging's
            # internal tables; h.queue stays bound, so clear it too.
            h.queue = None

    _LOG_QUEUE = None
