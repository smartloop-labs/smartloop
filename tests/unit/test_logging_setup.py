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

from conftest import _release_queue

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
        leaked, ls._LOG_QUEUE = ls._LOG_QUEUE, None
        if leaked is not None:
            _release_queue(leaked)


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
        _release_queue(queue)

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

    A multiprocessing.Queue here costs three POSIX semaphores on every
    startup, for a worker pool that is skipped whenever GPU offload is
    available or n_llm_workers <= 1 (the default).  It also can't be created
    at all under the macOS App Sandbox, which denies sem_open.
    """
    import queue as _queue

    assert isinstance(make_log_queue(), _queue.Queue)
    assert not isinstance(make_log_queue(), multiprocessing.queues.Queue)
    assert make_log_queue() is make_log_queue()  # idempotent singleton


def test_server_logging_setup_leaks_no_semaphores(tmp_path):
    """A server that dies abruptly mid-run must leave no semaphores behind.

    ``os._exit`` stands in for the SIGKILL ``slp agent stop`` escalates to,
    which skips every exit finalizer.  Nothing is torn down here on purpose:
    the default arrangement must hold no OS resources in the first place.
    """
    prog = """
        import os, sys
        from smartloop.utils.logging_setup import (
            make_log_queue, setup_root_logging, start_listener,
        )
        queue = make_log_queue()
        setup_root_logging(queue, level="DEBUG")
        listener = start_listener(queue, home_dir=sys.argv[1], force_console=False)
        os._exit(0)
        """
    stderr = _tracker_stderr(prog, tmp_path, str(tmp_path))
    assert "leaked semaphore" not in stderr, stderr


def test_worker_log_bridge_releases_semaphores(tmp_path):
    """``WorkerLogBridge.close()`` must unlink its queue's three semaphores.

    The bridge is the one place an mp queue still exists.  Its semaphores are
    unlinked by finalizers that only fire on a clean exit, so ``close()`` has
    to release them itself — the child holds the bridge and the shared queue
    throughout, proving the release doesn't depend on dropping references.
    """
    prog = """
        import os, sys
        from smartloop.utils.logging_setup import (
            make_log_queue, make_worker_log_bridge, setup_root_logging,
        )
        setup_root_logging(make_log_queue(), level="DEBUG")
        bridge = make_worker_log_bridge()
        assert bridge is not None
        bridge.close()
        os._exit(0)
        """
    stderr = _tracker_stderr(prog, tmp_path)
    assert "leaked semaphore" not in stderr, stderr


def test_worker_log_bridge_forwards_spawn_records(monkeypatch):
    """A spawn child's records must cross the bridge into the shared queue.

    This is the whole point of the bridge: worker processes can't write to a
    thread queue, so records have to arrive by this path or be lost.
    """
    monkeypatch.setattr(logging_setup, "_LOG_QUEUE", None)
    shared = make_log_queue()
    listener, buf = _make_capturing_listener(shared)
    bridge = logging_setup.make_worker_log_bridge()
    assert bridge is not None

    try:
        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=_spawn_emit, args=(bridge.queue,), name="probe-worker"
        )
        proc.start()
        proc.join(timeout=30)
        assert proc.exitcode == 0

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and "hello from spawn" not in buf.getvalue():
            time.sleep(0.05)
    finally:
        bridge.close()
        _drain_listener(listener)

    assert "hello from spawn" in buf.getvalue()
    assert "spawn.test" in buf.getvalue()


def test_bridge_returns_none_without_a_target(monkeypatch):
    """No shared queue means nothing to forward into — and no mp queue built."""
    monkeypatch.setattr(logging_setup, "_LOG_QUEUE", None)
    assert logging_setup.make_worker_log_bridge() is None
