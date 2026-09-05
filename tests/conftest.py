"""Shared test fixtures."""

import logging

import pytest

from smartloop.auth import credential_store


@pytest.fixture(autouse=True)
def _isolate_credential_store(tmp_path, monkeypatch):
    """Redirect credential_store to a temp directory so we never touch real credentials."""
    monkeypatch.setattr(credential_store, "_slp_home", lambda: str(tmp_path))


@pytest.fixture(autouse=True)
def _reset_refresh_cooldown():
    """Forget any failed-refresh cooldown left by another test.

    ``token_agent`` suppresses a redemption for a window after one fails, so a
    test that exercises a failing refresh would otherwise make the *next* test's
    legitimate refresh return False without calling anything — a failure that
    only appears in full-suite runs and disappears in isolation.
    """
    from smartloop.platform import token_agent

    token_agent._last_failure_at = 0.0
    yield
    token_agent._last_failure_at = 0.0


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
