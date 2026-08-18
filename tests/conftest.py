"""Shared test fixtures."""

import logging

import pytest

from smartloop.auth import credential_store


@pytest.fixture(autouse=True)
def _isolate_credential_store(tmp_path, monkeypatch):
    """Redirect credential_store to a temp directory so we never touch real credentials."""
    monkeypatch.setattr(credential_store, "_slp_home", lambda: str(tmp_path))


def _release_queue(queue) -> None:
    """Tear down a ``multiprocessing.Queue`` without blocking at exit.

    ``cancel_join_thread`` is the load-bearing call: it drops the exit-time
    join even when the feeder thread is wedged mid-flush. A plain
    ``queue.Queue`` has neither method and needs no cleanup.
    """
    cancel = getattr(queue, "cancel_join_thread", None)
    close = getattr(queue, "close", None)
    if cancel is None or close is None:
        return
    try:
        cancel()
        close()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _restore_root_logging():
    """Snapshot and restore the root logger around every test.

    A test that points the root logger at a ``multiprocessing.Queue`` and
    leaves it installed turns every later test's log call into a write nobody
    reads. Once the pipe buffer fills (~64KB) the queue's feeder thread blocks
    in ``send_bytes`` forever, and interpreter shutdown deadlocks joining it —
    the suite goes green and then never exits. Removing test-added handlers
    before releasing their queues keeps such a leak local to the test that
    made it, and in the right order: no handler survives pointing at a queue
    we've already closed.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        yield
    finally:
        leaked = [h for h in root.handlers if h not in saved_handlers]
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        for handler in leaked:
            _release_queue(getattr(handler, "queue", None))
