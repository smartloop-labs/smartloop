"""The embedding GGUF is fetched from our own bucket, not HuggingFace.

The Hub gates GGUF downloads behind a user token and is slow from most
regions, so first run resolves in order: bundled copy → an already-downloaded
HF-cache copy on disk → our public bucket.  The Hub itself is never called.
"""
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from smartloop.db.llama_embeddings import (
    _cdn_gguf_path,
    _hf_cached_gguf_path,
    load_embedder,
)

BASE_URL = "https://dl.smartloop.ai/embeddings"
FILENAME = "bge-m3-Q4_K_M.gguf"
REPO = "gpustack/bge-m3-GGUF"


class _FakeResponse:
    def __init__(self, chunks, content_length=None, status_error=None):
        self._chunks = chunks
        self._status_error = status_error
        self.headers = {}
        if content_length is not None:
            self.headers["content-length"] = str(content_length)

    def raise_for_status(self):
        if self._status_error:
            raise self._status_error

    def iter_bytes(self, _size):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _stream(response):
    """Patch target for ``httpx.stream``."""
    return lambda *args, **kwargs: response


class TestCdnDownload:
    def test_downloads_and_caches_the_gguf(self, tmp_path):
        response = _FakeResponse([b"gguf", b"body"], content_length=8)
        with patch("httpx.stream", _stream(response)):
            path = _cdn_gguf_path(str(tmp_path), BASE_URL, FILENAME)

        assert path == str(tmp_path / FILENAME)
        assert open(path, "rb").read() == b"ggufbody"

    def test_cached_file_short_circuits_the_network(self, tmp_path):
        (tmp_path / FILENAME).write_bytes(b"already here")

        def _explode(*_a, **_k):
            raise AssertionError("network hit despite a cached GGUF")

        with patch("httpx.stream", _explode):
            assert _cdn_gguf_path(str(tmp_path), BASE_URL, FILENAME) == str(
                tmp_path / FILENAME
            )

    def test_truncated_download_is_rejected_and_leaves_nothing_behind(self, tmp_path):
        # Server promised 100 bytes, delivered 8 — a resumed-but-incomplete
        # GGUF would fail deep inside llama.cpp instead of here.
        response = _FakeResponse([b"gguf", b"body"], content_length=100)
        with patch("httpx.stream", _stream(response)):
            assert _cdn_gguf_path(str(tmp_path), BASE_URL, FILENAME) is None

        assert os.listdir(tmp_path) == []

    def test_failed_request_falls_through_without_a_partial_file(self, tmp_path):
        def _fail(*_a, **_k):
            raise OSError("connection reset")

        with patch("httpx.stream", _fail):
            assert _cdn_gguf_path(str(tmp_path), BASE_URL, FILENAME) is None

        assert os.listdir(tmp_path) == []

    @pytest.mark.parametrize("cache_dir,base_url", [
        (None, BASE_URL),   # no workspace cache to write into
        ("/tmp", ""),       # explicitly opted out of the bucket
    ])
    def test_skipped_without_a_cache_dir_or_base_url(self, cache_dir, base_url):
        assert _cdn_gguf_path(cache_dir, base_url, FILENAME) is None


class TestHFCacheReuse:
    def test_finds_a_gguf_already_in_the_hf_cache_layout(self, tmp_path):
        snapshot = tmp_path / "models--gpustack--bge-m3-GGUF" / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        (snapshot / FILENAME).write_bytes(b"x")

        assert _hf_cached_gguf_path(str(tmp_path), REPO, FILENAME) == str(
            snapshot / FILENAME
        )

    def test_absent_cache_returns_none(self, tmp_path):
        assert _hf_cached_gguf_path(str(tmp_path), REPO, FILENAME) is None
        assert _hf_cached_gguf_path(None, REPO, FILENAME) is None


@pytest.fixture
def fake_llama_cpp():
    """Stub out ``llama_cpp`` so load_embedder gets past its import."""
    module = types.ModuleType("llama_cpp")
    module.Llama = MagicMock(return_value=MagicMock())
    with patch.dict(sys.modules, {"llama_cpp": module}):
        yield module


class TestResolutionOrder:
    def test_bucket_supplies_the_gguf_without_touching_the_hub(self, tmp_path, fake_llama_cpp):
        response = _FakeResponse([b"gguf"], content_length=4)
        hf = MagicMock()
        with patch("httpx.stream", _stream(response)), \
             patch.dict(sys.modules, {"huggingface_hub": MagicMock(hf_hub_download=hf)}):
            embedder = load_embedder(
                str(tmp_path), REPO, FILENAME, device="cpu", base_url=BASE_URL,
            )

        assert embedder is not None
        hf.assert_not_called()
        assert fake_llama_cpp.Llama.call_args.kwargs["model_path"] == str(
            tmp_path / FILENAME
        )

    def test_bucket_outage_falls_back_to_tfidf_not_the_hub(self, tmp_path, fake_llama_cpp):
        """The Hub is never called — it is what the token requirement came from.

        With no GGUF reachable the loader returns None and retrieval degrades
        to TF-IDF, rather than reintroducing a token-gated download.
        """
        def _fail(*_a, **_k):
            raise OSError("bucket unreachable")

        hub = MagicMock()
        with patch("httpx.stream", _fail), \
             patch.dict(sys.modules, {"huggingface_hub": hub}):
            embedder = load_embedder(
                str(tmp_path), REPO, FILENAME, device="cpu", base_url=BASE_URL,
            )

        assert embedder is None
        hub.hf_hub_download.assert_not_called()
