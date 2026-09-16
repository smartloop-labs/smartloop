"""Tests for smartloop.model.downloader.artifact_downloader."""

import hashlib
import logging
import os

import pytest

from smartloop.auth import credential_store
import httpx

from smartloop.model.downloader.artifact_downloader import (
    SLPArtifactDownloader,
    StaleDownloadURLError,
    _aggregate_error,
)
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


class TestResolveArtifactsIsSeparableFromTheTransfer:
    """Entitlement is decided by asking for the signed URLs, so that call is
    split out: /v1/init runs it before opening its progress stream, then hands
    the result to download() rather than paying for the rate-limited route
    twice."""

    def _downloader(self, tmp_path, source):
        downloader = SLPArtifactDownloader.__new__(SLPArtifactDownloader)
        downloader.workspace = type("WS", (), {"models_dir": str(tmp_path)})()
        downloader.logger = logging.getLogger("test")
        downloader._model_source = source
        return downloader

    def test_resolve_selects_the_gguf_artifacts(self, tmp_path):
        class FakeSource:
            def get_artifacts(self, model_name):
                return {
                    "model.gguf": "https://signed/model.gguf",
                    "README.md": "https://signed/README.md",
                }

        resolved = self._downloader(tmp_path, FakeSource()).resolve_artifacts(
            "google/gemma-3-4b-it")

        assert resolved == {"model.gguf": "https://signed/model.gguf"}

    def test_passed_artifacts_skip_the_second_platform_call(self, tmp_path):
        calls = []

        class CountingSource:
            def get_artifacts(self, model_name):
                calls.append(model_name)
                return {}

        downloader = self._downloader(tmp_path, CountingSource())
        downloader.download("google/gemma-3-4b-it", artifacts={})

        assert calls == [], "download() re-minted URLs the caller already had"

    def test_refusal_leaves_no_model_directory_behind(self, tmp_path):
        class DenyingSource:
            def get_artifacts(self, model_name):
                raise ModelAccessError("nope")

        downloader = self._downloader(tmp_path, DenyingSource())
        with pytest.raises(ModelAccessError):
            downloader.download("google/gemma-3-4b-it")

        assert not (tmp_path / "google/gemma-3-4b-it").exists()


class TestAggregateErrorKeepsTheFailureKind:
    """Per-file failures collapse into one exception that still says what broke.

    The router maps whatever escapes onto an HTTP status, so flattening
    everything to ``RuntimeError`` — as this once did — told the client "500"
    for a dropped socket and a corrupt file alike. They need different answers:
    one is worth retrying, the other never will be.
    """

    def _status_error(self, code):
        return httpx.HTTPStatusError(
            "boom", request=httpx.Request("GET", "http://x"),
            response=httpx.Response(code),
        )

    @pytest.mark.parametrize("exc", [
        httpx.ConnectError("dropped"),
        httpx.ReadTimeout("stalled"),
        StaleDownloadURLError("url expired"),
    ])
    def test_transient_failures_aggregate_to_connection_error(self, exc):
        result = _aggregate_error([("model.gguf", exc)])
        assert isinstance(result, ConnectionError)
        assert "model.gguf" in str(result)

    @pytest.mark.parametrize("code", [401, 403])
    def test_stale_signed_url_is_transient_not_a_verdict(self, code):
        """GCS answers 403 ExpiredToken past X-Goog-Expires.

        Entitlement was settled before the transfer started, so this is about
        the URL, not the account — and it must not surface as a plan refusal.
        """
        result = _aggregate_error([("model.gguf", self._status_error(code))])
        assert isinstance(result, ConnectionError)
        assert not isinstance(result, PermissionError)

    @pytest.mark.parametrize("exc", [
        RuntimeError("model.gguf: MD5 verification failed after 3 attempts"),
        ValueError("no artifacts"),
    ])
    def test_permanent_failures_stay_runtime_errors(self, exc):
        result = _aggregate_error([("model.gguf", exc)])
        assert isinstance(result, RuntimeError)
        assert not isinstance(result, ConnectionError)

    def test_one_permanent_failure_makes_the_batch_permanent(self):
        """Don't promise a retry will help when part of the batch is unfixable."""
        result = _aggregate_error([
            ("a.gguf", httpx.ConnectError("dropped")),
            ("b.gguf", RuntimeError("MD5 verification failed")),
        ])
        assert isinstance(result, RuntimeError)
        assert not isinstance(result, ConnectionError)

    def test_message_names_every_failed_file(self):
        result = _aggregate_error([
            ("a.gguf", httpx.ConnectError("x")),
            ("b.gguf", httpx.ConnectError("y")),
        ])
        assert "Failed to download 2 file(s): a.gguf, b.gguf" == str(result)
