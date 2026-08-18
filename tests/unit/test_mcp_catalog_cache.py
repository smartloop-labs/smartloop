"""The curated MCP catalog is cached in workspace_db.

Without a cache the Connections list 502s and a curated install silently
degrades to LLM enrichment whenever the platform API is unreachable or the
token has gone stale — neither of which means the user is offline. They can
still reach a third-party MCP server and complete OAuth.
"""

from unittest.mock import MagicMock

import pytest

from smartloop.platform.service import PlatformService
from smartloop.platform.workspace_db import WorkspaceDB


_ENTRY = {
    "slug": "linear",
    "display_name": "Linear",
    "description": "Issue tracking for software teams",
    "hosted_by": "external",
    "remote_config": {"server_url": "https://mcp.linear.app/mcp", "auth_type": "oauth"},
    "tags": ["issues", "planning"],
    "category": "Productivity",
}


@pytest.fixture
def db(tmp_path):
    return WorkspaceDB(str(tmp_path / "workspace.db"))


class TestWorkspaceDBRoundTrip:
    def test_nested_config_and_tags_survive(self, db):
        """Entries are stored as JSON blobs precisely so nesting survives."""
        db.save_mcp_catalog([_ENTRY])
        got = db.list_mcp_catalog()
        assert len(got) == 1
        assert got[0]["remote_config"]["server_url"] == "https://mcp.linear.app/mcp"
        assert got[0]["tags"] == ["issues", "planning"]

    def test_empty_before_first_fetch(self, db):
        assert db.list_mcp_catalog() == []
        assert db.mcp_catalog_fetched_at() is None

    def test_timestamp_set_after_save(self, db):
        db.save_mcp_catalog([_ENTRY])
        assert db.mcp_catalog_fetched_at() is not None

    def test_save_replaces_rather_than_appends(self, db):
        db.save_mcp_catalog([_ENTRY])
        db.save_mcp_catalog([{**_ENTRY, "slug": "notion", "display_name": "Notion"}])
        assert [e["slug"] for e in db.list_mcp_catalog()] == ["notion"]

    def test_entries_without_a_slug_are_skipped(self, db):
        db.save_mcp_catalog([_ENTRY, {"display_name": "Broken"}])
        assert [e["slug"] for e in db.list_mcp_catalog()] == ["linear"]


class TestServeStaleWhenPlatformIsDown:
    def test_cache_serves_when_client_cannot_connect(self, db):
        client = MagicMock()
        client.list_mcp_catalog.return_value = [_ENTRY]
        svc = PlatformService(client, db)

        assert svc.list_mcp_catalog() == [_ENTRY]          # populates cache

        client.list_mcp_catalog.side_effect = ConnectionError("platform down")
        assert svc.list_mcp_catalog(force=True) == [_ENTRY]  # still served

    def test_fresh_cache_avoids_a_network_call(self, db):
        client = MagicMock()
        client.list_mcp_catalog.return_value = [_ENTRY]
        svc = PlatformService(client, db)

        svc.list_mcp_catalog()
        svc.list_mcp_catalog()
        assert client.list_mcp_catalog.call_count == 1

    def test_no_db_falls_back_to_the_client(self):
        client = MagicMock()
        client.list_mcp_catalog.return_value = [_ENTRY]
        assert PlatformService(client, None).list_mcp_catalog() == [_ENTRY]


class TestResolveCatalogEntry:
    """Resolution must never raise — a failed lookup falls back to the
    client-supplied URL and lets enrichment run, so an install is never blocked
    by the catalog."""

    def _patch_service(self, monkeypatch, **kwargs):
        import smartloop.api.routers.mcp_catalog as mod
        svc = MagicMock()
        svc.list_mcp_catalog.configure_mock(**kwargs)
        monkeypatch.setattr(mod, "get_platform_service", lambda: svc)
        return mod

    def test_resolves_known_slug(self, monkeypatch):
        mod = self._patch_service(monkeypatch, return_value=[_ENTRY])
        entry = mod.resolve_catalog_entry("linear")
        assert entry is not None
        assert entry.description == "Issue tracking for software teams"
        assert entry.tags == ["issues", "planning"]
        assert entry.server_type == "remote"

    def test_unknown_slug_returns_none(self, monkeypatch):
        mod = self._patch_service(monkeypatch, return_value=[_ENTRY])
        assert mod.resolve_catalog_entry("nope") is None

    def test_lookup_failure_returns_none_instead_of_raising(self, monkeypatch):
        mod = self._patch_service(monkeypatch, side_effect=ConnectionError("down"))
        assert mod.resolve_catalog_entry("linear") is None

    def test_empty_slug_short_circuits(self, monkeypatch):
        mod = self._patch_service(monkeypatch, return_value=[_ENTRY])
        assert mod.resolve_catalog_entry("") is None

    def test_local_entry_derives_local_server_type(self, monkeypatch):
        """Dispatch keys off the resolved type, so this is what lets a future
        local listing work with no client release."""
        local = {
            "slug": "weather",
            "display_name": "Weather",
            "hosted_by": "local",
            "local_config": {"command": "uvx", "args": ["weather-mcp"]},
            "tags": ["weather"],
        }
        mod = self._patch_service(monkeypatch, return_value=[local])
        entry = mod.resolve_catalog_entry("weather")
        assert entry.server_type == "local"
        assert entry.local_config.command == "uvx"
