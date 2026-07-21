"""Tests for smartloop.platform.service.PlatformService."""

import time

import pytest

from smartloop.platform.service import PlatformService, USER_TTL, RELEASES_TTL
from smartloop.platform.workspace_db import WorkspaceDB


_USER = {
    "email": "chris@example.com",
    "name": "Chris",
    "is_premium": False,
    "is_paid": True,
    "number_of_tokens_processed": 10,
    "default_temperature": 0.3,
    "default_memory": True,
}

_MODELS = [
    {"model_id": "google/gemma-3-1b-it", "name": "gemma3-1b", "size": "1B",
     "description": "Gemma 3 1B", "access_level": "public", "is_accessible": True},
]

_RELEASE = {"version": "1.0.15", "name": "v", "notes": "", "channel": "stable",
            "artifacts": [], "min_version": None}
# The cloud client returns the full LatestReleaseResponse wrapper.
_RELEASE_WRAPPER = {"update_available": None, "update_required": False, "release": _RELEASE}


class _FakeClient:
    """Minimal stand-in for SLPPlatformClient."""

    def __init__(self, *, user=None, models=None, release_wrapper=None, fail=False):
        self._user = user
        self._models = models
        self._release_wrapper = release_wrapper
        self._fail = fail
        self.call_count = {"user": 0, "models": 0, "update": 0, "release": 0}

    def get_latest_release(self, channel="stable"):
        self.call_count["release"] += 1
        if self._fail:
            raise ConnectionError("offline")
        return self._release_wrapper

    def get_current_user(self):
        self.call_count["user"] += 1
        if self._fail:
            raise ConnectionError("offline")
        return self._user

    def update_current_user(self, payload):
        self.call_count["update"] += 1
        if self._fail:
            raise ConnectionError("offline")
        merged = {**self._user, **payload}
        self._user = merged
        return merged

    def list_models(self):
        self.call_count["models"] += 1
        if self._fail:
            raise ConnectionError("offline")
        return self._models


@pytest.fixture
def db(tmp_path):
    return WorkspaceDB(str(tmp_path / "workspace.db"))


class TestGetCurrentUser:
    def test_fetches_from_cloud_on_first_call(self, db):
        client = _FakeClient(user=_USER)
        svc = PlatformService(client, db)

        result = svc.get_current_user()
        assert result["email"] == "chris@example.com"
        assert client.call_count["user"] == 1

    def test_returns_cached_within_ttl(self, db):
        client = _FakeClient(user=_USER)
        svc = PlatformService(client, db)

        svc.get_current_user()
        svc.get_current_user()
        assert client.call_count["user"] == 1  # only one cloud call

    def test_falls_back_to_cache_when_offline(self, db):
        # Seed the cache
        db.save_user(_USER)

        client = _FakeClient(fail=True)
        svc = PlatformService(client, db)

        result = svc.get_current_user()
        assert result["email"] == "chris@example.com"

    def test_raises_when_offline_and_no_cache(self, db):
        client = _FakeClient(fail=True)
        svc = PlatformService(client, db)

        with pytest.raises(ConnectionError):
            svc.get_current_user()

    def test_refetches_after_ttl(self, db, monkeypatch):
        client = _FakeClient(user=_USER)
        svc = PlatformService(client, db)

        svc.get_current_user()
        assert client.call_count["user"] == 1

        # Simulate time passing beyond TTL
        original_time = time.time
        monkeypatch.setattr(time, "time", lambda: original_time() + USER_TTL + 1)

        svc.get_current_user()
        assert client.call_count["user"] == 2


class TestUpdateCurrentUser:
    def test_updates_cloud_and_cache(self, db):
        client = _FakeClient(user=_USER)
        svc = PlatformService(client, db)

        result = svc.update_current_user({"default_temperature": 0.9})
        assert result["default_temperature"] == 0.9

        # Cache should be updated
        cached = db.get_user()
        assert cached["default_temperature"] == 0.9

    def test_raises_when_offline(self, db):
        client = _FakeClient(fail=True)
        svc = PlatformService(client, db)

        with pytest.raises(ConnectionError):
            svc.update_current_user({"default_temperature": 0.9})


class TestNoDB:
    """When no workspace exists yet, the service runs as a pure proxy."""

    def test_get_user_proxies_to_cloud(self):
        client = _FakeClient(user=_USER)
        svc = PlatformService(client, db=None)
        assert svc.get_current_user() == _USER
        # Hits cloud every time — no cache.
        svc.get_current_user()
        assert client.call_count["user"] == 2

    def test_list_models_proxies_to_cloud(self):
        client = _FakeClient(models=_MODELS)
        svc = PlatformService(client, db=None)
        assert svc.list_models() == _MODELS
        svc.list_models()
        assert client.call_count["models"] == 2

    def test_get_latest_release_proxies_to_cloud(self):
        client = _FakeClient(release_wrapper=_RELEASE_WRAPPER)
        svc = PlatformService(client, db=None)
        # returns the LatestReleaseResponse envelope
        assert svc.get_latest_release("stable") == _RELEASE_WRAPPER
        svc.get_latest_release("stable")
        assert client.call_count["release"] == 2

    def test_update_proxies_without_caching(self):
        client = _FakeClient(user=_USER)
        svc = PlatformService(client, db=None)
        result = svc.update_current_user({"default_temperature": 0.5})
        assert result["default_temperature"] == 0.5


class TestListModels:
    def test_fetches_from_cloud_on_first_call(self, db):
        client = _FakeClient(models=_MODELS)
        svc = PlatformService(client, db)

        result = svc.list_models()
        assert len(result) == 1
        assert result[0]["model_id"] == "google/gemma-3-1b-it"
        assert client.call_count["models"] == 1

    def test_returns_cached_within_ttl(self, db):
        client = _FakeClient(models=_MODELS)
        svc = PlatformService(client, db)

        svc.list_models()
        svc.list_models()
        assert client.call_count["models"] == 1

    def test_falls_back_to_cache_when_offline(self, db):
        db.save_models(_MODELS)

        client = _FakeClient(fail=True)
        svc = PlatformService(client, db)

        result = svc.list_models()
        assert len(result) == 1

    def test_raises_when_offline_and_no_cache(self, db):
        client = _FakeClient(fail=True)
        svc = PlatformService(client, db)

        with pytest.raises(ConnectionError):
            svc.list_models()


class TestGetLatestRelease:
    def test_returns_envelope_and_caches_the_release_field(self, db):
        client = _FakeClient(release_wrapper=_RELEASE_WRAPPER)
        svc = PlatformService(client, db)

        result = svc.get_latest_release("stable")
        assert result == _RELEASE_WRAPPER
        assert client.call_count["release"] == 1
        # only the inner release is cached (client-agnostic)
        assert db.get_release("stable") == _RELEASE

    def test_returns_cached_within_ttl(self, db):
        client = _FakeClient(release_wrapper=_RELEASE_WRAPPER)
        svc = PlatformService(client, db)

        svc.get_latest_release("stable")
        svc.get_latest_release("stable")
        assert client.call_count["release"] == 1

    def test_force_bypasses_ttl(self, db):
        client = _FakeClient(release_wrapper=_RELEASE_WRAPPER)
        svc = PlatformService(client, db)

        svc.get_latest_release("stable")
        svc.get_latest_release("stable", force=True)
        assert client.call_count["release"] == 2

    def test_falls_back_to_cache_when_offline(self, db):
        db.save_release("stable", _RELEASE)

        client = _FakeClient(fail=True)
        svc = PlatformService(client, db)

        assert svc.get_latest_release("stable")["release"] == _RELEASE

    def test_refetches_after_ttl(self, db, monkeypatch):
        client = _FakeClient(release_wrapper=_RELEASE_WRAPPER)
        svc = PlatformService(client, db)

        svc.get_latest_release("stable")
        assert client.call_count["release"] == 1

        original_time = time.time
        monkeypatch.setattr(time, "time", lambda: original_time() + RELEASES_TTL + 1)

        svc.get_latest_release("stable")
        assert client.call_count["release"] == 2

    def test_raises_when_offline_and_never_fetched(self, db):
        client = _FakeClient(fail=True)
        svc = PlatformService(client, db)

        with pytest.raises(ConnectionError):
            svc.get_latest_release("beta")


# A channel that exists but has nothing published yet (e.g. an unreleased
# `beta`): the cloud returns a wrapper whose `release` is None. `None` is a
# real cached value here, not a cache miss — these tests guard the sentinel fix.
_EMPTY_WRAPPER = {"update_available": None, "update_required": False, "release": None}


class TestGetLatestReleaseEmptyChannel:
    def test_empty_release_is_cached_within_ttl(self, db):
        client = _FakeClient(release_wrapper=_EMPTY_WRAPPER)
        svc = PlatformService(client, db)

        first = svc.get_latest_release("beta")
        second = svc.get_latest_release("beta")

        assert first["release"] is None
        assert second["release"] is None
        # The TTL fast-path must apply even though the release is None — no
        # redundant second cloud call for an empty channel.
        assert client.call_count["release"] == 1

    def test_empty_release_served_offline_after_being_fetched(self, db):
        client = _FakeClient(release_wrapper=_EMPTY_WRAPPER)
        svc = PlatformService(client, db)

        # First poll learns "nothing published" and records a timestamp.
        svc.get_latest_release("beta")

        # Now go offline and force past the TTL: we should serve the known-empty
        # state rather than raising, because we HAVE fetched before.
        client._fail = True
        result = svc.get_latest_release("beta", force=True)
        assert result["release"] is None

    def test_refetches_empty_channel_after_ttl(self, db, monkeypatch):
        client = _FakeClient(release_wrapper=_EMPTY_WRAPPER)
        svc = PlatformService(client, db)

        svc.get_latest_release("beta")
        assert client.call_count["release"] == 1

        original_time = time.time
        monkeypatch.setattr(time, "time", lambda: original_time() + RELEASES_TTL + 1)

        svc.get_latest_release("beta")
        assert client.call_count["release"] == 2
