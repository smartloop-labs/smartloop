"""Tests for retry/resume behavior in the artifact downloader."""

import hashlib
import logging
import threading

import httpx
import pytest

import smartloop.model.downloader.artifact_downloader as ad
from smartloop.model.downloader.artifact_downloader import (
    MAX_DOWNLOAD_ATTEMPTS,
    SLPArtifactDownloader,
    _backoff_delay,
    _is_retryable_error,
)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Make backoff instant so retry tests don't actually wait."""
    monkeypatch.setattr(ad.time, "sleep", lambda _s: None)


def _downloader():
    """A downloader instance without running __init__ (no network/workspace)."""
    dl = SLPArtifactDownloader.__new__(SLPArtifactDownloader)
    dl.logger = logging.getLogger("test-downloader")
    return dl


class _FakeResp:
    """A stand-in for an httpx streaming response used as a context manager."""

    def __init__(self, status_code, chunks, raise_at=None, exc=None):
        self.status_code = status_code
        self._chunks = chunks
        self._raise_at = raise_at  # index at which to raise `exc` instead of yielding
        self._exc = exc

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error", request=None, response=httpx.Response(self.status_code)
            )

    def iter_bytes(self, chunk_size=None):
        for i, chunk in enumerate(self._chunks):
            if self._raise_at is not None and i == self._raise_at:
                raise self._exc
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _ScriptedStream:
    """Replaces httpx.stream; returns scripted responses and records headers."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.header_calls = []
        self.kwargs_calls = []

    def __call__(self, method, url, headers=None, **kwargs):
        self.header_calls.append(headers or {})
        self.kwargs_calls.append(kwargs)
        return self._responses.pop(0)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

class TestRetryClassification:
    def test_transport_errors_are_retryable(self):
        assert _is_retryable_error(httpx.ReadError("boom"))
        assert _is_retryable_error(httpx.ConnectError("boom"))
        assert _is_retryable_error(httpx.RemoteProtocolError("boom"))
        assert _is_retryable_error(httpx.ReadTimeout("boom"))

    def test_transient_statuses_are_retryable(self):
        for code in (408, 429, 500, 502, 503, 504):
            err = httpx.HTTPStatusError("e", request=None, response=httpx.Response(code))
            assert _is_retryable_error(err), code

    def test_client_errors_are_not_retryable(self):
        for code in (400, 403, 404):
            err = httpx.HTTPStatusError("e", request=None, response=httpx.Response(code))
            assert not _is_retryable_error(err), code
        assert not _is_retryable_error(ValueError("nope"))

    def test_backoff_grows_and_is_capped_and_jittered(self):
        # Jitter keeps each delay within (0.5, 1.0] of the exponential base.
        for attempt in range(1, 8):
            base = min(ad._RETRY_BASE_DELAY * (2 ** (attempt - 1)), ad._RETRY_MAX_DELAY)
            d = _backoff_delay(attempt)
            assert 0.5 * base <= d <= base
        assert _backoff_delay(20) <= ad._RETRY_MAX_DELAY


class TestRunWithRetries:
    def test_returns_on_first_success(self):
        dl = _downloader()
        calls = []
        result = dl._run_with_retries("op", lambda n: calls.append(n) or "ok")
        assert result == "ok"
        assert calls == [1]

    def test_retries_transient_then_succeeds(self):
        dl = _downloader()
        calls = []

        def attempt(n):
            calls.append(n)
            if n < 3:
                raise httpx.ReadError("drop")
            return "done"

        assert dl._run_with_retries("op", attempt) == "done"
        assert calls == [1, 2, 3]

    def test_non_retryable_raises_immediately(self):
        dl = _downloader()
        calls = []

        def attempt(n):
            calls.append(n)
            raise ValueError("fatal")

        with pytest.raises(ValueError):
            dl._run_with_retries("op", attempt)
        assert calls == [1]

    def test_permission_error_not_retried(self):
        dl = _downloader()
        calls = []

        def attempt(n):
            calls.append(n)
            raise PermissionError("401")

        with pytest.raises(PermissionError):
            dl._run_with_retries("op", attempt)
        assert calls == [1]

    def test_exhausts_attempts_then_raises(self):
        dl = _downloader()
        calls = []

        def attempt(n):
            calls.append(n)
            raise httpx.ConnectError("down")

        with pytest.raises(httpx.ConnectError):
            dl._run_with_retries("op", attempt)
        assert calls == list(range(1, MAX_DOWNLOAD_ATTEMPTS + 1))


# --------------------------------------------------------------------------- #
# Simple (single-stream) download
# --------------------------------------------------------------------------- #

class TestSimpleResume:
    def test_resume_after_midstream_drop(self, tmp_path, monkeypatch):
        dest = tmp_path / "file.bin"
        # First attempt: 200, writes b"ABCD" then the stream drops.
        first = _FakeResp(200, [b"ABCD", b"NEVER"], raise_at=1, exc=httpx.ReadError("drop"))
        # Second attempt: 206 (Range honored), delivers the remainder.
        second = _FakeResp(206, [b"EFGH"])
        stream = _ScriptedStream(first, second)
        monkeypatch.setattr(ad.httpx, "stream", stream)

        dl = _downloader()
        dl._download_file_simple("file.bin", "http://x/file", str(dest), size=8,
                                 progress_callback=None, supports_range=True)

        assert dest.read_bytes() == b"ABCDEFGH"
        # The retry asked to resume from byte 4.
        assert stream.header_calls[0] == {}
        assert stream.header_calls[1] == {"Range": "bytes=4-"}

    def test_server_ignores_range_restarts_clean(self, tmp_path, monkeypatch):
        dest = tmp_path / "file.bin"
        first = _FakeResp(200, [b"ABCD", b"NEVER"], raise_at=1, exc=httpx.ReadError("drop"))
        # Server ignores Range on retry (200, full body) -> must overwrite, not append.
        second = _FakeResp(200, [b"ABCDEFGH"])
        stream = _ScriptedStream(first, second)
        monkeypatch.setattr(ad.httpx, "stream", stream)

        dl = _downloader()
        dl._download_file_simple("file.bin", "http://x/file", str(dest), size=8,
                                 progress_callback=None, supports_range=True)

        assert dest.read_bytes() == b"ABCDEFGH"

    def test_401_raises_without_retry(self, tmp_path, monkeypatch):
        """A 401 on the transfer is a stale URL, not a credentials problem.

        Entitlement was settled before the transfer began, so this cannot mean
        "you are not allowed" — the signed URL went past its expiry. It is a
        ``ConnectionError`` so the caller offers a retry rather than a sign-in
        prompt, and it still short-circuits the retry loop, because re-fetching
        the same dead URL eight times helps nobody.
        """
        dest = tmp_path / "file.bin"
        stream = _ScriptedStream(_FakeResp(401, []))
        monkeypatch.setattr(ad.httpx, "stream", stream)

        dl = _downloader()
        with pytest.raises(ad.StaleDownloadURLError) as exc:
            dl._download_file_simple("file.bin", "http://x/file", str(dest), size=8,
                                     progress_callback=None, supports_range=True)
        assert isinstance(exc.value, ConnectionError)
        assert not isinstance(exc.value, PermissionError)
        assert len(stream.header_calls) == 1

    def test_recovers_from_midstream_read_timeout(self, tmp_path, monkeypatch):
        """A stalled stream (Wi-Fi drop) surfaces as ReadTimeout and is retried."""
        dest = tmp_path / "file.bin"
        first = _FakeResp(200, [b"ABCD", b"NEVER"], raise_at=1,
                          exc=httpx.ReadTimeout("stalled"))
        second = _FakeResp(206, [b"EFGH"])
        stream = _ScriptedStream(first, second)
        monkeypatch.setattr(ad.httpx, "stream", stream)

        dl = _downloader()
        dl._download_file_simple("file.bin", "http://x/file", str(dest), size=8,
                                 progress_callback=None, supports_range=True)
        assert dest.read_bytes() == b"ABCDEFGH"

    def test_uses_short_read_timeout(self, tmp_path, monkeypatch):
        """Streaming uses a bounded read timeout so stalls are detected quickly."""
        dest = tmp_path / "file.bin"
        stream = _ScriptedStream(_FakeResp(200, [b"ABCDEFGH"]))
        monkeypatch.setattr(ad.httpx, "stream", stream)

        dl = _downloader()
        dl._download_file_simple("file.bin", "http://x/file", str(dest), size=8,
                                 progress_callback=None, supports_range=False)
        timeout = stream.kwargs_calls[0]["timeout"]
        assert timeout is ad._STREAM_TIMEOUT
        assert timeout.read is not None and timeout.read <= 120


# --------------------------------------------------------------------------- #
# Chunked (parallel Range) download
# --------------------------------------------------------------------------- #

class TestChunkResume:
    def test_chunk_resumes_and_counts_once(self, tmp_path, monkeypatch):
        dest = tmp_path / "big.bin"
        dest.write_bytes(b"\0" * 8)  # pre-allocated, as the chunked path does

        first = _FakeResp(206, [b"ABCD", b"NEVER"], raise_at=1, exc=httpx.ReadError("drop"))
        second = _FakeResp(206, [b"EFGH"])
        stream = _ScriptedStream(first, second)
        monkeypatch.setattr(ad.httpx, "stream", stream)

        counter = [0]
        lock = threading.Lock()
        dl = _downloader()
        dl._download_chunk("http://x/file", str(dest), start=0, end=7,
                           flat_name="big.bin", total_size=8,
                           downloaded_counter=counter, lock=lock,
                           progress_callback=lambda p: None)

        assert dest.read_bytes() == b"ABCDEFGH"
        # Progress counted only the 8 real bytes, not the dropped retry bytes.
        assert counter[0] == 8
        # Retry resumed from the correct offset within the chunk.
        assert stream.header_calls[0] == {"Range": "bytes=0-7"}
        assert stream.header_calls[1] == {"Range": "bytes=4-7"}


# --------------------------------------------------------------------------- #
# MD5 verification + atomic (.part) publish, via the _download_file wrapper
# --------------------------------------------------------------------------- #

class TestVerifyAndAtomicPublish:
    def _run(self, tmp_path, monkeypatch, *responses, remote_md5):
        stream = _ScriptedStream(*responses)
        monkeypatch.setattr(ad.httpx, "stream", stream)
        dl = _downloader()
        # size below CHUNKED_THRESHOLD + supports_range=False -> simple path,
        # one stream call per verification attempt.
        dl._download_file("model.gguf", "http://x/f", str(tmp_path), 8, 0,
                          None, False, remote_md5)
        return stream

    def test_verified_download_published_atomically(self, tmp_path, monkeypatch):
        data = b"ABCDEFGH"
        stream = self._run(tmp_path, monkeypatch, _FakeResp(200, [data]),
                           remote_md5=hashlib.md5(data).hexdigest())
        assert (tmp_path / "model.gguf").read_bytes() == data
        assert not (tmp_path / "model.gguf.part").exists()
        assert len(stream.header_calls) == 1  # no re-download needed

    def test_mismatch_redownloads_then_succeeds(self, tmp_path, monkeypatch):
        good = b"ABCDEFGH"
        stream = self._run(
            tmp_path, monkeypatch,
            _FakeResp(200, [b"XXXXXXXX"]),  # corrupt first
            _FakeResp(200, [good]),          # clean on re-download
            remote_md5=hashlib.md5(good).hexdigest(),
        )
        assert (tmp_path / "model.gguf").read_bytes() == good
        assert not (tmp_path / "model.gguf.part").exists()
        assert len(stream.header_calls) == 2

    def test_persistent_mismatch_raises_and_leaves_nothing(self, tmp_path, monkeypatch):
        corrupt = [_FakeResp(200, [b"XXXXXXXX"]) for _ in range(ad._MAX_VERIFY_ATTEMPTS)]
        with pytest.raises(RuntimeError, match="MD5 verification failed"):
            self._run(tmp_path, monkeypatch, *corrupt,
                      remote_md5=hashlib.md5(b"ABCDEFGH").hexdigest())
        # Neither the final file nor the partial sidecar survive a hard failure.
        assert not (tmp_path / "model.gguf").exists()
        assert not (tmp_path / "model.gguf.part").exists()

    def test_no_remote_md5_skips_verify_but_still_atomic(self, tmp_path, monkeypatch):
        data = b"ABCDEFGH"
        stream = self._run(tmp_path, monkeypatch, _FakeResp(200, [data]), remote_md5=None)
        assert (tmp_path / "model.gguf").read_bytes() == data
        assert not (tmp_path / "model.gguf.part").exists()
        assert len(stream.header_calls) == 1  # verification skipped, no re-download


class TestChunkedFailureKeepsItsKind:
    """The chunked path is the one production actually takes.

    CHUNKED_THRESHOLD is 100 MB and every shipped model is larger, so a real
    dropped connection always arrives through here. It used to re-raise every
    chunk failure as a bare RuntimeError, which threw away the classification
    the outer layers were careful to preserve — the SSE frame said 500 for a
    dead link no matter what the layers above it did.
    """

    def _chunked(self, tmp_path, monkeypatch, exc):
        monkeypatch.setattr(ad, "CHUNK_WORKERS", 1)
        monkeypatch.setattr(ad.SLPArtifactDownloader, "_download_chunk",
                            lambda *a, **k: (_ for _ in ()).throw(exc))
        dl = _downloader()
        with pytest.raises(Exception) as caught:
            dl._download_file_chunked("file.bin", "http://x/file",
                                      str(tmp_path / "file.bin"), size=1024,
                                      progress_callback=None)
        return caught.value

    def test_dropped_connection_stays_a_connection_error(self, tmp_path, monkeypatch):
        err = self._chunked(tmp_path, monkeypatch, httpx.ConnectError("dropped"))
        assert isinstance(err, ConnectionError)
        assert "Chunk download failed" in str(err)

    def test_stale_signed_url_stays_a_connection_error(self, tmp_path, monkeypatch):
        err = self._chunked(tmp_path, monkeypatch, ad.StaleDownloadURLError("expired"))
        assert isinstance(err, ConnectionError)

    def test_permanent_failure_stays_a_runtime_error(self, tmp_path, monkeypatch):
        err = self._chunked(tmp_path, monkeypatch, ValueError("corrupt"))
        assert isinstance(err, RuntimeError)
        assert not isinstance(err, ConnectionError)
