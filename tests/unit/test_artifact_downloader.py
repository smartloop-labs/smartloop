"""Tests for smartloop.model.downloader.artifact_downloader."""

import hashlib
import logging
import os

import pytest

from smartloop.auth import credential_store
from smartloop.model.downloader.artifact_downloader import SLPArtifactDownloader
from smartloop.platform import ModelAccessError


class TestComputeLocalMd5:
    def test_correct_hash(self, tmp_path):
        f = tmp_path / "test.bin"
        content = b"hello world"
        f.write_bytes(content)
        expected = hashlib.md5(content).hexdigest()
        assert SLPArtifactDownloader._compute_local_md5(str(f)) == expected

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        expected = hashlib.md5(b"").hexdigest()
        assert SLPArtifactDownloader._compute_local_md5(str(f)) == expected


class TestParseRemoteMd5:
    def test_valid_etag(self):
        headers = {"ETag": '"d41d8cd98f00b204e9800998ecf8427e"'}
        assert SLPArtifactDownloader._parse_remote_md5(headers) == "d41d8cd98f00b204e9800998ecf8427e"

    def test_unquoted_etag(self):
        headers = {"ETag": "d41d8cd98f00b204e9800998ecf8427e"}
        assert SLPArtifactDownloader._parse_remote_md5(headers) == "d41d8cd98f00b204e9800998ecf8427e"

    def test_non_md5_etag_returns_none(self):
        # Composite GCS objects have longer ETags
        headers = {"ETag": '"abc123-5"'}
        assert SLPArtifactDownloader._parse_remote_md5(headers) is None

    def test_missing_etag(self):
        assert SLPArtifactDownloader._parse_remote_md5({}) is None


class TestResolveModelName:
    def test_known_model_roundtrip(self):
        # _resolve_model_id and _resolve_model_name should be inverses
        model_id = SLPArtifactDownloader._resolve_model_id("sl-mini")
        if model_id != "sl-mini":  # only if the model is in SUPPORTED_MODELS
            name = SLPArtifactDownloader._resolve_model_name(model_id)
            assert name == "sl-mini"

    def test_legacy_name_resolves_to_same_model_id(self):
        # An existing workspace still records the pre-rename name.
        assert (SLPArtifactDownloader._resolve_model_id("slp-orchestra-mini")
                == SLPArtifactDownloader._resolve_model_id("sl-mini"))

    def test_reverse_lookup_returns_canonical_name(self):
        model_id = SLPArtifactDownloader._resolve_model_id("sl-mini")
        assert SLPArtifactDownloader._resolve_model_name(model_id) == "sl-mini"

    def test_unknown_model_passthrough(self):
        assert SLPArtifactDownloader._resolve_model_name("unknown/model") == "unknown/model"
        assert SLPArtifactDownloader._resolve_model_id("nonexistent") == "nonexistent"


class TestDownloadIsNotAuthGated:
    """Free models download anonymously, so the downloader must not refuse to
    try before it has asked the platform.

    The platform is the only authority on access: it serves the free tier to an
    anonymous caller and answers 403 (``ModelAccessError``) for anything the
    caller's plan doesn't grant. A local credential check would have denied the
    free download the platform would have served.
    """

    def _downloader(self, tmp_path, source):
        downloader = SLPArtifactDownloader.__new__(SLPArtifactDownloader)
        downloader.workspace = type("WS", (), {"models_dir": str(tmp_path)})()
        downloader.logger = logging.getLogger("test")
        downloader._model_source = source
        return downloader

    def test_reaches_the_platform_with_no_credentials(self, tmp_path):
        asked = []

        class FakeSource:
            def get_artifacts(self, model_name):
                asked.append(model_name)
                return {}  # no GGUFs -> download() returns early

        # No credentials are set (conftest isolates the store to a temp dir).
        assert credential_store.has_credentials() is False

        downloader = self._downloader(tmp_path, FakeSource())
        downloader.download("google/gemma-3-4b-it")

        assert asked == ["gemma3-4b"]

    def test_plan_denial_propagates(self, tmp_path):
        class DenyingSource:
            def get_artifacts(self, model_name):
                raise ModelAccessError(f"Model '{model_name}' is not available on your current plan.")

        downloader = self._downloader(tmp_path, DenyingSource())
        with pytest.raises(ModelAccessError, match="not available on your current plan"):
            downloader.download("google/gemma-3-4b-it")
