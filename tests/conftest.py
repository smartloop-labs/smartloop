"""Shared test fixtures."""

import pytest

from smartloop.auth import credential_store


@pytest.fixture(autouse=True)
def _isolate_credential_store(tmp_path, monkeypatch):
    """Redirect credential_store to a temp directory so we never touch real credentials."""
    monkeypatch.setattr(credential_store, "_slp_home", lambda: str(tmp_path))
