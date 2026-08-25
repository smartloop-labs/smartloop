"""Centralized logging setup for the API server and its spawn workers.

All threads (TaskQueue jobs, IndexerTask workers, DAG executor, hybrid
executor I/O pool) and all spawn worker processes (the LLM
``ProcessPoolExecutor``) emit through a single ``QueueHandler`` on the
root logger.  A ``QueueListener`` thread in the parent process owns the
real handlers (rotating file + optional console) and writes records in
arrival order so concurrent producers never tear a line.

The shared queue is a plain thread queue: in-process threads are the only
producers in every configuration we actually ship.  A worker *process*
can't write to it, so the pool — on the rare path where one is really
spawned — gets its own ``multiprocessing.Queue`` plus a forwarder thread
draining it into the shared queue (:func:`make_worker_log_bridge`).  That
confines POSIX semaphores to the lifetime of the pool that needs them
instead of paying for them on every startup.

Used by:
- ``smartloop.api.main.run_server`` / ``start_server_foreground`` — installs
  the queue + listener.
- ``smartloop.agents.hybrid_executor.LLMExecutor`` — builds a bridge via
  :func:`make_worker_log_bridge` and passes its mp queue to the spawn pool
  initializer, in the same branch that creates the pool.
- ``smartloop.agents.llm_worker.init_worker`` — each spawned worker calls
  :func:`worker_setup_logging` so its log records flow back through the
  bridge to the parent listener.
"""

from __future__ import annotations

import gc
import logging
import logging.handlers
import multiprocessing
import os
import queue as _queue
import sys
import threading
from pathlib import Path
from typing import Optional

# Process-wide singleton — set by make_log_queue() and read by
# the hybrid executor when it bridges the worker pool's records in.
_LOG_QUEUE: Optional[_queue.Queue] = None

DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5


def make_log_queue() -> _queue.Queue:
    """Create the process-wide log queue. Idempotent — returns the existing
    queue if one was already created.

    A plain thread-safe ``queue.Queue``, because in-process threads are the
    only producers in every configuration we ship.  This used to be a
    ``multiprocessing.Queue`` so that spawn workers could write to it
    directly, which cost three POSIX named semaphores (``_rlock``, ``_wlock``,
    ``_sem``) on every startup — for a pool that is skipped whenever GPU
    offload is available or ``n_llm_workers <= 1`` (the default), i.e. always,
    on any GPU machine.  Those semaphores then had to be unlinked before the
    process died or the resource tracker reported them as leaked, and
    ``sem_open`` is denied outright under the macOS App Sandbox.

    A worker *process* still can't write here, so the pool gets its own
    ``multiprocessing.Queue`` from :func:`make_worker_log_bridge`, created in
    the same branch that creates the pool and torn down with it.  Semaphores
    now exist only while there are real worker processes to feed them.
    ``QueueHandler``/``QueueListener`` treat both queue types identically."""
    global _LOG_QUEUE
    if _LOG_QUEUE is None:
        _LOG_QUEUE = _queue.Queue(-1)
    return _LOG_QUEUE


def get_log_queue() -> Optional[_queue.Queue]:
    """Return the shared log queue, or ``None`` if logging hasn't been set up
    yet (e.g. running outside the server, in unit tests, or before startup)."""
    return _LOG_QUEUE


class WorkerLogBridge:
    """Carries log records from spawn worker processes to the shared queue.

    Owns a ``multiprocessing.Queue`` — the only kind a worker process can
    write to — and a daemon thread draining it into *target*, so worker
    records still reach the single parent listener in arrival order.  Its
    lifetime is the worker pool's: :meth:`close` releases the queue's three
    POSIX semaphores, so nothing survives to be reported as leaked.
    """

    def __init__(self, target: _queue.Queue):
        self._target = target
        self.queue = multiprocessing.Queue(-1)
        self._thread = threading.Thread(
            target=self._drain, name="log-bridge", daemon=True
        )
        try:
            self._thread.start()
        except BaseException:
            # A bridge with no forwarder is just three orphaned semaphores.
            _release_mp_queue(self.queue)
            raise

    def _drain(self) -> None:
        while True:
            try:
                record = self.queue.get()
            except (EOFError, OSError, ValueError):
                return  # queue closed under us — nothing left to forward
            if record is None:
                return  # sentinel from close()
            try:
                self._target.put_nowait(record)
            except Exception:
                pass  # listener already torn down; dropping beats raising

    def close(self) -> None:
        """Stop the forwarder and release the mp queue's semaphores."""
        try:
            self.queue.put_nowait(None)
        except Exception:
            pass
        self._thread.join(timeout=2)
        _release_mp_queue(self.queue)


def make_worker_log_bridge(
    target: Optional[_queue.Queue] = None,
) -> Optional[WorkerLogBridge]:
    """Build a :class:`WorkerLogBridge` feeding *target* (default: the shared
    queue).

    Returns ``None`` when there is no target — logging was never set up, so
    there is nothing to forward into and the caller should hand its workers
    ``None`` so they fall back to their own stderr (see
    :func:`worker_setup_logging`).  Also returns ``None`` if the mp queue
    can't be created at all: ``sem_open`` is denied under the macOS App
    Sandbox, and stderr-only workers beat a failed pool.
    """
    if target is None:
        target = _LOG_QUEUE
    if target is None:
        return None
    try:
        return WorkerLogBridge(target)
    except (PermissionError, OSError):
        return None


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


def _release_mp_queue(queue) -> None:
    """Release a ``multiprocessing.Queue``'s pipe and POSIX semaphores.

    No-op for a plain ``queue.Queue``, which owns no OS resources.

    Each of the three semaphores is unlinked by a ``util.Finalize`` that fires
    when its ``SemLock`` is garbage collected — so waiting for the *queue* to
    become unreachable makes cleanup hostage to every reference anywhere in
    the process.  Detaching the locks makes them unreachable on their own, so
    the standard finalizers run here, under our control, whoever else still
    holds the queue.  Safe because the queue is closed first: ``put``/``get``
    check ``_closed`` and raise ValueError before they reach these locks.

    Without this the tracker prints ``resource_tracker: There appear to be 3
    leaked semaphore objects to clean up at shutdown`` whenever the process
    dies before its exit finalizers run — which ``slp agent stop`` guarantees,
    escalating to SIGKILL when SIGTERM hasn't landed within 10s.
    """
    if queue is None or not hasattr(queue, "close"):
        return

    try:
        queue.close()
    except Exception:
        pass

    # ``Queue._start_thread`` passes ``_wlock`` and ``_sem`` straight into the
    # feeder thread's ``args``, so those two stay reachable — and un-unlinked —
    # for as long as it runs, however thoroughly we detach them below.
    # ``close()`` has just queued the sentinel that stops it; wait for it to
    # notice.  Bounded, because a feeder wedged in ``send_bytes`` must not hang
    # the shutdown: skipping the join leaks the two semaphores this call exists
    # to reclaim, which is still better than never exiting.
    feeder = getattr(queue, "_thread", None)
    if feeder is not None:
        feeder.join(timeout=2)
        if feeder.is_alive():
            try:
                queue.cancel_join_thread()  # don't join it again at exit
            except Exception:
                pass

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


def shutdown_logging(listener: Optional[logging.handlers.QueueListener] = None) -> None:
    """Stop the listener and release the shared log queue.

    The shared queue is a plain thread queue (see :func:`make_log_queue`), so
    there are no OS resources to reclaim here — worker-process semaphores live
    on the :class:`WorkerLogBridge`, which the pool closes with itself.  The
    queue is still dropped from the listener and the root handler so it can be
    collected, and :func:`_release_mp_queue` covers the case where a caller
    installed a ``multiprocessing.Queue`` of their own.

    Safe to call more than once.
    """
    global _LOG_QUEUE

    if listener is not None:
        try:
            listener.stop()
        except Exception:
            pass
        # ``stop()`` joins the listener thread but keeps ``listener.queue``
        # set, and the caller still holds the listener itself.
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
    _release_mp_queue(queue)


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
