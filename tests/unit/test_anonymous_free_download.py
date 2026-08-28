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
