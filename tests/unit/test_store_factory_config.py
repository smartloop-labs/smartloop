"""
Tests for the vector store factory.

SQLite (dense embeddings, TF-IDF fallback) is the only backend; the factory
always builds an ``IndexStore`` regardless of the ``store_type`` argument, which
is retained only for call-site compatibility.
"""
import logging
import tempfile

import pytest

from smartloop.db.store_factory import StoreFactory, StoreType, _resolve_default_store_type
from smartloop.db.sqlite_store import IndexStore


@pytest.fixture
def project_dir():
    return tempfile.mkdtemp()


class TestFactoryResolution:
    def test_default_store_type_is_sqlite(self):
        assert _resolve_default_store_type() == StoreType.SQLite

    def test_factory_builds_sqlite_by_default(self, project_dir):
        store = StoreFactory.create_store(project_dir, logger=logging.getLogger("t"))
        assert isinstance(store, IndexStore)

    def test_store_type_argument_is_ignored(self, project_dir):
        # The legacy store_type arg is accepted but always yields an IndexStore.
        store = StoreFactory.create_store(
            project_dir, logger=logging.getLogger("t"), store_type="anything",
        )
        assert isinstance(store, IndexStore)
