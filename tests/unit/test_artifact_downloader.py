"""Tests for smartloop.model.downloader.artifact_downloader."""

import hashlib
import os

import pytest

from smartloop.model.downloader.artifact_downloader import SLPArtifactDownloader


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
        model_id = SLPArtifactDownloader._resolve_model_id("slp-orchestra-mini")
        if model_id != "slp-orchestra-mini":  # only if the model is in SUPPORTED_MODELS
            name = SLPArtifactDownloader._resolve_model_name(model_id)
            assert name == "slp-orchestra-mini"

    def test_unknown_model_passthrough(self):
        assert SLPArtifactDownloader._resolve_model_name("unknown/model") == "unknown/model"
        assert SLPArtifactDownloader._resolve_model_id("nonexistent") == "nonexistent"
