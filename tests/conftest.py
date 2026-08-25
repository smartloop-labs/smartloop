"""Shared test fixtures."""

import logging

import pytest

from smartloop.auth import credential_store


@pytest.fixture(autouse=True)
def _isolate_credential_store(tmp_path, monkeypatch):
    """Redirect credential_store to a temp directory so we never touch real credentials."""
    monkeypatch.setattr(credential_store, "_slp_home", lambda: str(tmp_path))


@pytest.fixture(autouse=True)
def _restore_root_logging():
    """Snapshot and restore the root logger around every test.

    A test that points the root logger at a queue and leaves it installed
    turns every later test's log call into a write nobody reads. Keep such a
    leak local to the test that made it.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        yield
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
