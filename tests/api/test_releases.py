"""API tests for the /v1/releases router."""

import pytest


_RELEASE = {
    # shaped like the cloud ReleaseRead, including fields the local ReleaseInfo drops
    "id": "abc123",
    "version": "1.0.15",
    "name": "Release 1.0.15",
    "notes": "shiny",
    "channel": "stable",
    "status": "published",
    "published_at": "2026-07-01T00:00:00Z",
    "artifacts": [
        {"platform": "darwin", "arch": "arm64", "kind": "zip",
         "url": "https://cdn/x.zip", "sha512": "abc", "size": 123},
    ],
    "min_version": "1.0.10",
    "created_at": "2026-06-01T00:00:00Z",
    "updated_at": "2026-06-02T00:00:00Z",
}


def _wrap(release):
    """Wrap an inner release dict in the cloud LatestReleaseResponse envelope
    that PlatformService.get_latest_release actually returns."""
    return {"update_available": None, "update_required": False, "release": release}


class TestLatestRelease:
    def test_no_current_version_returns_release_and_feed_url(self, client, platform_service):
        platform_service.get_latest_release.return_value = _wrap(_RELEASE)

        resp = client.get("/v1/releases/latest")
        assert resp.status_code == 200
        body = resp.json()
        assert body["update_available"] is None
        assert body["update_required"] is False
        assert body["channel"] == "stable"
        assert body["feed_url"].endswith("/v1/releases/feed/stable")
        assert body["release"]["version"] == "1.0.15"
        # extra cloud fields are dropped by the local response model
        assert "id" not in body["release"]
        assert "status" not in body["release"]

    def test_older_client_gets_update_available_and_required(self, client, platform_service):
        platform_service.get_latest_release.return_value = _wrap(_RELEASE)

        body = client.get("/v1/releases/latest?current_version=0.0.1").json()
        assert body["update_available"] is True
        assert body["update_required"] is True  # min_version 1.0.10 > 0.0.1

    def test_current_client_gets_no_update(self, client, platform_service):
        platform_service.get_latest_release.return_value = _wrap(_RELEASE)

        body = client.get("/v1/releases/latest?current_version=1.0.15").json()
        assert body["update_available"] is False
        assert body["update_required"] is False

    def test_beta_channel_switch(self, client, platform_service):
        platform_service.get_latest_release.return_value = _wrap({**_RELEASE, "channel": "beta"})

        body = client.get("/v1/releases/latest?channel=beta").json()
        assert body["channel"] == "beta"
        assert body["feed_url"].endswith("/v1/releases/feed/beta")
        platform_service.get_latest_release.assert_called_with("beta", force=False)

    def test_unknown_channel_rejected(self, client, platform_service):
        resp = client.get("/v1/releases/latest?channel=nope")
        assert resp.status_code == 400

    def test_default_does_not_force(self, client, platform_service):
        platform_service.get_latest_release.return_value = _wrap(_RELEASE)

        client.get("/v1/releases/latest")
        platform_service.get_latest_release.assert_called_with("stable", force=False)

    def test_force_bypasses_cache(self, client, platform_service):
        # A "check for updates" click sends force=true, which re-polls the cloud
        # immediately instead of waiting out the local cache TTL.
        platform_service.get_latest_release.return_value = _wrap(_RELEASE)

        client.get("/v1/releases/latest?force=true&channel=beta")
        platform_service.get_latest_release.assert_called_with("beta", force=True)

    def test_no_published_release(self, client, platform_service):
        # Cloud API's null-release response: wrapper present, inner release is None
        platform_service.get_latest_release.return_value = _wrap(None)

        resp = client.get("/v1/releases/latest")
        assert resp.status_code == 200
        body = resp.json()
        assert body["release"] is None
        assert body["update_available"] is None
        assert body["feed_url"].endswith("/v1/releases/feed/stable")

    def test_wrapper_missing_release_key_returns_no_release(self, client, platform_service):
        # Defensive: if the cloud response ever comes back without a `release` key,
        # treat it as "nothing published" rather than 500ing.
        platform_service.get_latest_release.return_value = {
            "update_available": None,
            "update_required": False,
        }

        resp = client.get("/v1/releases/latest")
        assert resp.status_code == 200
        assert resp.json()["release"] is None

    def test_platform_unavailable_returns_502(self, client, platform_service):
        platform_service.get_latest_release.side_effect = ConnectionError("down")

        resp = client.get("/v1/releases/latest")
        assert resp.status_code == 502
