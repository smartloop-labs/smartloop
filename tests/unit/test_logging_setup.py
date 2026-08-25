"""
Centralized logging plumbing:

* A thread emits through the root QueueHandler, the listener picks it up, and
  the record reaches the attached handler.
* ``start_listener`` always writes to ``<home>/server.log``.
* The queue owns no OS resources, so a process that dies abruptly leaves
  nothing behind for the resource tracker to report.

The spawn-worker path is gone with the LLM worker pool (see
``smartloop.agents.hybrid_executor``), so there is no ``multiprocessing.Queue``
and no cross-process forwarding left to cover.
"""
import io
import logging
import logging.handlers
import multiprocessing.queues
import queue as _queue
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
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_log_queue_singleton():
    """Keep ``logging_setup._LOG_QUEUE`` from leaking between tests.

    It is a process-wide singleton, so a queue created here would otherwise be
    handed to every later caller of ``make_log_queue`` in the same session.
    """
    import smartloop.utils.logging_setup as ls

    ls._LOG_QUEUE = None
    try:
        yield
    finally:
        ls._LOG_QUEUE = None


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
    queue = _queue.Queue(-1)
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
    queue = _queue.Queue(-1)
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


def test_shutdown_logging_is_idempotent(monkeypatch):
    """Called twice (lifespan shutdown + a retry path) it must not raise."""
    monkeypatch.setattr(logging_setup, "_LOG_QUEUE", None)
    queue = make_log_queue()
    setup_root_logging(queue, level="DEBUG")
    shutdown_logging()
    shutdown_logging()
    assert get_log_queue() is None


def _tracker_stderr(prog, tmp_path, *argv):
    """Run *prog* in a child and return the stderr its resource tracker wrote.

    The leaked-semaphore warning comes from the resource-tracker *process*,
    not the child, so stderr must go to a file both inherit — a
    ``capture_output`` pipe closes before the tracker gets to write. The
    tracker writes as soon as it sees EOF on the child's pipe; poll anyway so
    a slow write can't turn a real leak into a passing test.
    """
    err_path = tmp_path / "child-stderr.txt"
    with err_path.open("wb") as err:
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(prog), *argv],
            stdout=subprocess.DEVNULL, stderr=err, timeout=120,
        )

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        stderr = err_path.read_text()
        if "leaked semaphore" in stderr:
            break
        time.sleep(0.1)
    return stderr


def test_make_log_queue_owns_no_os_resources():
    """The shared queue must be a thread queue — no semaphores, no sem_open.

    A multiprocessing.Queue here cost three POSIX semaphores on every startup,
    which had to be unlinked before the process died or the resource tracker
    reported them leaked — and sem_open is denied under the macOS App Sandbox.
    Nothing writes to this queue from another process any more.
    """
    assert isinstance(make_log_queue(), _queue.Queue)
    assert not isinstance(make_log_queue(), multiprocessing.queues.Queue)
    assert make_log_queue() is make_log_queue()  # idempotent singleton


def test_server_logging_setup_leaks_no_semaphores(tmp_path):
    """A server that dies abruptly must leave nothing for the tracker to warn about.

    ``os._exit`` stands in for the SIGKILL ``slp agent stop`` escalates to,
    which skips every exit finalizer.  Nothing is torn down here on purpose:
    the arrangement must hold no OS resources in the first place.

    The warning would come from the resource-tracker process, not the child, so
    stderr goes to a file both inherit — a ``capture_output`` pipe closes too
    early to catch it — and is polled after the child exits.
    """
    prog = textwrap.dedent(
        """
        import os, sys
        from smartloop.utils.logging_setup import (
            make_log_queue, setup_root_logging, start_listener,
        )
        queue = make_log_queue()
        setup_root_logging(queue, level="DEBUG")
        listener = start_listener(queue, home_dir=sys.argv[1], force_console=False)
        os._exit(0)
        """
    )
    err_path = tmp_path / "child-stderr.txt"
    with err_path.open("wb") as err:
        subprocess.run(
            [sys.executable, "-c", prog, str(tmp_path)],
            stdout=subprocess.DEVNULL, stderr=err, timeout=120,
        )

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        stderr = err_path.read_text()
        if "leaked semaphore" in stderr:
            break
        time.sleep(0.1)

    assert "leaked semaphore" not in stderr, stderr
