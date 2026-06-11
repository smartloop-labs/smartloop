"""
Tests for config-driven vector store backend selection.

The vector store backend is chosen via the ``slp_vector_store_backend``
setting and resolved centrally in ``StoreFactory`` so every call site
stays consistent without passing ``store_type`` explicitly.
"""
import logging
import tempfile

import pytest
from pydantic import ValidationError

from smartloop.config.app_settings import AppSettings
from smartloop.db.store_factory import StoreFactory, StoreType, _resolve_default_store_type
from smartloop.db.sqlite_store import IndexStore


@pytest.fixture
def project_dir():
    return tempfile.mkdtemp()


class TestSettings:
    def test_default_backend_is_sqlite(self, monkeypatch):
        monkeypatch.delenv("SLP_VECTOR_STORE_BACKEND", raising=False)
        assert AppSettings().vector_store_backend == "sqlite"

    def test_backend_is_normalized(self, monkeypatch):
        monkeypatch.setenv("SLP_VECTOR_STORE_BACKEND", "  Chroma  ")
        assert AppSettings().vector_store_backend == "chroma"

    def test_invalid_backend_rejected(self, monkeypatch):
        monkeypatch.setenv("SLP_VECTOR_STORE_BACKEND", "weaviate")
        with pytest.raises(ValidationError):
            AppSettings()


class TestFactoryResolution:
    def test_default_resolves_to_configured_backend(self, monkeypatch):
        monkeypatch.delenv("SLP_VECTOR_STORE_BACKEND", raising=False)
        assert _resolve_default_store_type() == "sqlite"
        monkeypatch.setenv("SLP_VECTOR_STORE_BACKEND", "chroma")
        assert _resolve_default_store_type() == "chroma"

    def test_factory_builds_configured_sqlite(self, project_dir, monkeypatch):
        monkeypatch.setenv("SLP_VECTOR_STORE_BACKEND", "sqlite")
        store = StoreFactory.create_store(project_dir, logger=logging.getLogger("t"))
        assert isinstance(store, IndexStore)

    def test_explicit_sqlite_overrides_config(self, project_dir, monkeypatch):
        # Even when config says chroma, an explicit store_type wins.
        monkeypatch.setenv("SLP_VECTOR_STORE_BACKEND", "chroma")
        store = StoreFactory.create_store(
            project_dir, logger=logging.getLogger("t"), store_type=StoreType.SQLite,
        )
        assert isinstance(store, IndexStore)
