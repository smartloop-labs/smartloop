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
import threading
import time
from pathlib import Path

from smartloop.utils.logging_setup import (
    DEFAULT_FORMAT,
    make_log_queue,
    setup_root_logging,
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
