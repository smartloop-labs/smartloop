"""Free-tier models must be downloadable without an account.

The platform serves FREE artifacts to anonymous callers and rejects unentitled
ones with 403, so it owns the decision. The local side must not second-guess it
by refusing to make the request.
"""

import logging

from smartloop.model.downloader import artifact_downloader as ad


def _downloader(tmp_path, asked):
    class FakeSource:
        def get_artifacts(self, model_name):
            asked.append(model_name)
            return {}  # no GGUFs -> download() returns early

    d = ad.SLPArtifactDownloader.__new__(ad.SLPArtifactDownloader)
    d.workspace = type("WS", (), {"models_dir": str(tmp_path)})()
    d.logger = logging.getLogger("test")
    d._model_source = FakeSource()
    return d


class TestDownloaderNeedsNoCredentials:
    def test_download_proceeds_while_signed_out(self, monkeypatch, tmp_path):
        """No credentials must not stop the artifact request being made."""
        from smartloop.auth import credential_store

        monkeypatch.setattr(credential_store, "auth_headers", lambda: {})
        assert not credential_store.has_credentials()

        asked = []
        _downloader(tmp_path, asked).download("google/gemma-4-e2b-it")
        assert asked, "download() refused to ask the platform for artifacts"

    def test_downloader_holds_no_credential_gate(self):
        """The blanket check is gone; entitlement lives at the router + platform."""
        assert not hasattr(ad, "platform_auth")


class TestArtifactFetchTypesFailuresApart:
    """A refusal and a rate limit are different problems, so they are different types.

    ``_run_with_retries`` re-raises ``PermissionError`` immediately and backs off
    on everything else, so typing a 429 as a permission problem would both tell
    the user to upgrade and disable the retry that would have fixed it. The
    download route is rate-limited per IP, which anonymous downloads now reach.
    """

    def _client(self, status, monkeypatch):
        import httpx

        from smartloop.platform import client as platform_client

        resp = httpx.Response(
            status,
            request=httpx.Request("GET", "https://api.example/v1/models/x/download"),
            json={},
        )
        monkeypatch.setattr(platform_client.httpx, "get", lambda *a, **k: resp)
        return platform_client.SLPPlatformClient("https://api.example", lambda: {})

    def test_forbidden_is_a_permission_error(self, monkeypatch):
        import pytest

        c = self._client(403, monkeypatch)
        with pytest.raises(PermissionError) as exc:
            c.get_model_artifacts("gemma3-12b")
        assert exc.value.status_code == 403
        assert "plan" in str(exc.value)

    def test_rate_limit_is_transient_not_a_plan_problem(self, monkeypatch):
        import pytest

        c = self._client(429, monkeypatch)
        with pytest.raises(ConnectionError) as exc:
            c.get_model_artifacts("gemma3-12b")
        assert not isinstance(exc.value, PermissionError)
        assert exc.value.status_code == 429
        assert "upgrade" not in str(exc.value).lower()

    def test_server_error_is_transient(self, monkeypatch):
        import pytest

        c = self._client(503, monkeypatch)
        with pytest.raises(ConnectionError) as exc:
            c.get_model_artifacts("gemma3-12b")
        assert not isinstance(exc.value, PermissionError)
        assert exc.value.status_code == 503


class TestRefusedDownloadIsNotLeftOnDisk:
    def test_no_model_dir_created_when_entitlement_is_refused(self, tmp_path):
        """A 403 must not leave an empty model directory for the workspace to find."""
        import pytest

        class RefusingSource:
            def get_artifacts(self, model_name):
                raise PermissionError("not on your plan")

        d = ad.SLPArtifactDownloader.__new__(ad.SLPArtifactDownloader)
        d.workspace = type("WS", (), {"models_dir": str(tmp_path)})()
        d.logger = logging.getLogger("test")
        d._model_source = RefusingSource()

        with pytest.raises(PermissionError):
            d.download("google/gemma-3-12b-it")

        assert list(tmp_path.iterdir()) == []
