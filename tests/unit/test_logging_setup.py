"""
Centralized logging plumbing:

* A thread inside the parent process emits through the root QueueHandler,
  the listener picks it up, and the record reaches the attached handler.
* A spawn worker process configured via ``worker_setup_logging`` emits
  through the same queue and its records land in the same handler — which
  is the whole point of routing through ``multiprocessing.Queue``.
"""
import io
import logging
import logging.handlers
import multiprocessing
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

import smartloop.utils.logging_setup as logging_setup
from smartloop.utils.logging_setup import (
    DEFAULT_FORMAT,
    get_log_queue,
    make_log_queue,
    setup_root_logging,
    shutdown_logging,
    start_listener,
    worker_setup_logging,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _drain_listener(listener) -> None:
    """Block briefly until the listener's queue is drained, then stop it."""
    # QueueListener processes records in a background thread; give it a
    # moment to flush, then stop.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not listener.queue.empty():
        time.sleep(0.01)
    listener.stop()


def _make_capturing_listener(queue):
    """Start a listener that writes formatted records into a StringIO buffer."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    listener = logging.handlers.QueueListener(
        queue, handler, respect_handler_level=True
    )
    listener.start()
    return listener, buf


# --------------------------------------------------------------------------
# Thread → parent listener
# --------------------------------------------------------------------------

def test_thread_log_records_reach_listener_handler():
    """A worker thread logging via the root logger lands in the listener sink."""
    queue = multiprocessing.Queue(-1)
    setup_root_logging(queue, level="DEBUG")
    listener, buf = _make_capturing_listener(queue)

    try:
        def emit():
            logging.getLogger("thread.test").info("hello from thread")

        t = threading.Thread(target=emit, name="probe-thread")
        t.start()
        t.join(timeout=2)
    finally:
        _drain_listener(listener)

    assert "hello from thread" in buf.getvalue()
    assert "thread.test" in buf.getvalue()


# --------------------------------------------------------------------------
# Listener mode selection
# --------------------------------------------------------------------------

def test_start_listener_creates_rotating_file_handler(tmp_path):
    """``start_listener`` should always attach a rotating file handler to <home>/server.log."""
    queue = multiprocessing.Queue(-1)
    setup_root_logging(queue, level="DEBUG")
    listener = start_listener(
        queue,
        home_dir=str(tmp_path),
        file_level="DEBUG",
        force_console=False,
    )
    try:
        logging.getLogger("file.test").info("file probe")
        # Listener thread is async — give it a moment to flush before reading.
        time.sleep(0.2)
        # Force the handler to flush its buffer to disk.
        for h in listener.handlers:
            h.flush()
    finally:
        _drain_listener(listener)

    log_path = Path(tmp_path) / "server.log"
    assert log_path.exists()
    assert "file probe" in log_path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Spawn-process → parent listener
# --------------------------------------------------------------------------

def _spawn_emit(queue):
    """Body of the spawn child: configure via worker_setup_logging, emit, exit."""
    worker_setup_logging(queue)
    logging.getLogger("spawn.test").info("hello from spawn")


def test_spawn_process_records_reach_parent_listener():
    """Records emitted from a spawn child must arrive at the parent listener."""
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue(-1)
    listener, buf = _make_capturing_listener(queue)

    try:
        proc = ctx.Process(target=_spawn_emit, args=(queue,), name="probe-worker")
        proc.start()
        proc.join(timeout=10)
        assert proc.exitcode == 0
    finally:
        _drain_listener(listener)

    assert "hello from spawn" in buf.getvalue()
    assert "spawn.test" in buf.getvalue()


# --------------------------------------------------------------------------
# api_settings.log_level controls whether the smartloop gag is lifted
# --------------------------------------------------------------------------

def _run_install_with_log_level(monkeypatch, tmp_path, level):
    """Set api_settings.log_level, install centralized logging, return listener."""
    from smartloop.api import main as api_main

    # main.py clamps smartloop to WARNING; restore that baseline before
    # _install_centralized_logging so each case starts from the same state.
    logging.getLogger("smartloop").setLevel(logging.WARNING)

    monkeypatch.setattr(api_main.api_settings, "log_level", level)
    # home_dir is a computed property backed by slp_home — set the field.
    monkeypatch.setattr(api_main._app_settings, "slp_home", str(tmp_path))

    # Fresh queue per call — the singleton in logging_setup persists across
    # tests otherwise.
    import smartloop.utils.logging_setup as ls
    monkeypatch.setattr(ls, "_LOG_QUEUE", None)

    api_main._install_centralized_logging()
    return api_main.app.state.log_listener


def test_log_level_info_keeps_smartloop_at_warning(monkeypatch, tmp_path):
    """Shipped electron build (log_level=INFO) keeps main.py's smartloop gag."""
    listener = _run_install_with_log_level(monkeypatch, tmp_path, "INFO")
    try:
        assert logging.getLogger("smartloop").level == logging.WARNING
    finally:
        listener.stop()


def test_log_level_debug_lifts_smartloop_gag(monkeypatch, tmp_path):
    """Dev mode (log_level=DEBUG) lifts the gag so orchestrator logs reach the file."""
    listener = _run_install_with_log_level(monkeypatch, tmp_path, "DEBUG")
    try:
        assert logging.getLogger("smartloop").level == logging.DEBUG
    finally:
        listener.stop()


# --------------------------------------------------------------------------
# Shutdown releases the queue's OS resources
# --------------------------------------------------------------------------

def test_shutdown_logging_closes_queue_and_detaches_handler(monkeypatch):
    """``shutdown_logging`` drops the QueueHandler and closes the mp queue."""
    monkeypatch.setattr(logging_setup, "_LOG_QUEUE", None)
    queue = make_log_queue()
    setup_root_logging(queue, level="DEBUG")
    listener, _ = _make_capturing_listener(queue)

    shutdown_logging(listener)

    root = logging.getLogger()
    assert not [h for h in root.handlers
                if isinstance(h, logging.handlers.QueueHandler)]
    assert get_log_queue() is None
    # A closed multiprocessing.Queue refuses further writes.
    with pytest.raises(ValueError):
        queue.put("after close")


def test_shutdown_logging_is_idempotent(monkeypatch):
    """Called twice (lifespan shutdown + a retry path) it must not raise."""
    monkeypatch.setattr(logging_setup, "_LOG_QUEUE", None)
    queue = make_log_queue()
    setup_root_logging(queue, level="DEBUG")
    shutdown_logging()
    shutdown_logging()
    assert get_log_queue() is None


def test_shutdown_logging_prevents_leaked_semaphores():
    """No ``leaked semaphore objects`` warning when the process dies abruptly.

    A ``multiprocessing.Queue`` holds three POSIX semaphores that are unlinked
    by exit finalizers.  ``slp agent stop`` escalates to SIGKILL, which skips
    them — ``os._exit`` reproduces that here.  ``shutdown_logging`` must have
    already released them, so the resource tracker finds nothing to warn about.
    """
    prog = textwrap.dedent(
        """
        import os
        from smartloop.utils.logging_setup import (
            get_log_queue, make_log_queue, setup_root_logging, shutdown_logging,
        )
        make_log_queue()
        setup_root_logging(get_log_queue())
        shutdown_logging()
        os._exit(0)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", prog], capture_output=True, text=True, timeout=120,
    )
    assert "leaked semaphore" not in proc.stderr, proc.stderr
