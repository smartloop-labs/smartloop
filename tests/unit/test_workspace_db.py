"""Tests for smartloop.platform.workspace_db.WorkspaceDB."""

import os
import time

import pytest

from smartloop.platform.workspace_db import WorkspaceDB


@pytest.fixture
def db(tmp_path):
    """Create a WorkspaceDB backed by a temp file."""
    return WorkspaceDB(str(tmp_path / "workspace.db"))


_USER = {
    "email": "chris@example.com",
    "name": "Chris",
    "is_premium": True,
    "is_paid": True,
    "number_of_tokens_processed": 42,
    "default_temperature": 0.7,
    "default_memory": True,
}

_MODELS = [
    {
        "model_id": "google/gemma-3-1b-it",
        "name": "gemma3-1b",
        "size": "1B",
        "description": "Gemma 3 1B",
        "access_level": "public",
        "is_accessible": True,
    },
    {
        "model_id": "google/gemma-3-4b-it",
        "name": "gemma3-4b",
        "size": "4B",
        "description": "Gemma 3 4B",
        "access_level": "paid",
        "is_accessible": False,
    },
]


class TestWorkspaceDBCreation:
    def test_creates_db_file(self, tmp_path):
        db_path = str(tmp_path / "sub" / "workspace.db")
        WorkspaceDB(db_path)
        assert os.path.exists(db_path)

    def test_empty_on_creation(self, db):
        assert db.get_user() is None
        assert db.list_models() == []
        assert db.user_fetched_at() is None
        assert db.models_fetched_at() is None


class TestUserPersistence:
    def test_save_and_get(self, db):
        db.save_user(_USER)
        result = db.get_user()
        assert result["email"] == "chris@example.com"
        assert result["is_paid"] is True
        assert result["number_of_tokens_processed"] == 42
        assert result["default_temperature"] == 0.7

    def test_upsert_overwrites(self, db):
        db.save_user(_USER)
        updated = {**_USER, "is_paid": False, "default_temperature": 0.4}
        db.save_user(updated)
        result = db.get_user()
        assert result["is_paid"] is False
        assert result["default_temperature"] == 0.4

    def test_fetched_at_set(self, db):
        assert db.user_fetched_at() is None
        db.save_user(_USER)
        ts = db.user_fetched_at()
        assert ts is not None
        assert ts > 0

    def test_defaults_for_missing_fields(self, db):
        db.save_user({"email": "min@example.com"})
        result = db.get_user()
        assert result["email"] == "min@example.com"
        assert result["is_paid"] is False
        assert result["default_temperature"] == 0.3
        assert result["default_memory"] is True


class TestModelPersistence:
    def test_save_and_list(self, db):
        db.save_models(_MODELS)
        result = db.list_models()
        assert len(result) == 2
        assert result[0]["model_id"] == "google/gemma-3-1b-it"
        assert result[0]["is_accessible"] is True
        assert result[1]["is_accessible"] is False

    def test_save_replaces_all(self, db):
        db.save_models(_MODELS)
        db.save_models([_MODELS[0]])
        result = db.list_models()
        assert len(result) == 1

    def test_fetched_at_set(self, db):
        assert db.models_fetched_at() is None
        db.save_models(_MODELS)
        ts = db.models_fetched_at()
        assert ts is not None
        assert ts > 0

    def test_empty_save(self, db):
        db.save_models(_MODELS)
        db.save_models([])
        assert db.list_models() == []
        assert db.models_fetched_at() is None


class TestWorkspacePersistence:
    def test_empty_on_creation(self, db):
        assert db.get_workspace() is None

    def test_save_and_get(self, db):
        db.save_workspace("ws-uuid-123")
        result = db.get_workspace()
        assert result["id"] == "ws-uuid-123"
        assert result["created_at"] > 0
        assert result["updated_at"] == result["created_at"]

    def test_second_save_bumps_updated_at_only(self, db):
        db.save_workspace("ws-uuid-123")
        first = db.get_workspace()
        time.sleep(0.01)
        db.save_workspace("ws-uuid-123")
        second = db.get_workspace()
        assert second["id"] == first["id"]
        assert second["created_at"] == first["created_at"]
        assert second["updated_at"] > first["updated_at"]
