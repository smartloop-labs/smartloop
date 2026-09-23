"""A download stops when asked, and says so in a way nothing mistakes.

Two properties are under test throughout.  The transfer must notice the flag
promptly — at the next chunk boundary, not at the end of the file and not after
a 30s backoff.  And the cancel must survive every layer that classifies
failures on its way out: retried, aggregated with a sibling's dropped socket,
or rewrapped by the chunked path, it would reach the client as a 502 for a
button the user pressed.
"""

import hashlib
import logging
import threading

import httpx
import pytest

import smartloop.model.downloader.artifact_downloader as ad
from smartloop.model.downloader.artifact_downloader import (
    MAX_DOWNLOAD_ATTEMPTS,
    SLPArtifactDownloader,
    _aggregate_error,
    _is_retryable_error,
    _is_transient_error,
)
from smartloop.model.downloader.cancellation import CancelToken, DownloadCancelled


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Backoff is instant, but a cancelled token still reports cancelled.

    ``CancelToken.wait`` *is* the backoff now, so leaving it real makes any
    test that exhausts the retries — the HEAD probe's fallback, for one — wait
    out the full ~90s window.  Returning the token's actual state keeps the
    substitution honest.
    """
    monkeypatch.setattr(CancelToken, "wait", lambda self, _timeout: self.cancelled)


def _downloader(cancel=None):
    dl = SLPArtifactDownloader.__new__(SLPArtifactDownloader)
    dl.logger = logging.getLogger("test-downloader")
    dl.cancel_token = cancel or CancelToken()
    return dl


class _CancellingResp:
    """A streaming response that trips *token* partway through the body."""

    def __init__(self, chunks, token, cancel_at):
        self.status_code = 200
        self._chunks = chunks
        self._token = token
        self._cancel_at = cancel_at
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_bytes(self, chunk_size=None):
        for i, chunk in enumerate(self._chunks):
            if i == self._cancel_at:
                self._token.cancel()
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.closed = True
        return False


class _Stream:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = 0

    def __call__(self, method, url, headers=None, **kwargs):
        self.calls += 1
        return self._responses.pop(0)


# --------------------------------------------------------------------------- #
# Classification — a cancel must never look like a network problem
# --------------------------------------------------------------------------- #

class TestClassification:
    def test_a_cancel_is_not_retryable(self):
        assert not _is_retryable_error(DownloadCancelled("stop"))

    def test_a_cancel_is_not_transient(self):
        """Transient would have ``status_for`` answer 502 for a button press."""
        assert not _is_transient_error(DownloadCancelled("stop"))

    def test_it_carries_its_own_status(self):
        assert DownloadCancelled("stop").status_code == 499

    def test_it_is_not_a_connection_error(self):
        # _is_transient_error ends in isinstance(exc, ConnectionError), so this
        # inheritance choice is what keeps the classification above true.
        assert not isinstance(DownloadCancelled("stop"), ConnectionError)
        assert not isinstance(DownloadCancelled("stop"), OSError)

    def test_aggregate_surfaces_the_cancel_over_a_transient_sibling(self):
        cancel = DownloadCancelled("stop")
        result = _aggregate_error([
            ("weights.gguf", httpx.ReadError("dropped")),
            ("mmproj.gguf", cancel),
        ])
        assert result is cancel

    def test_aggregate_surfaces_the_cancel_whichever_order_it_lands_in(self):
        cancel = DownloadCancelled("stop")
        result = _aggregate_error([
            ("mmproj.gguf", cancel),
            ("weights.gguf", httpx.ReadError("dropped")),
        ])
        assert result is cancel

    def test_without_a_cancel_the_old_aggregation_stands(self):
        result = _aggregate_error([("weights.gguf", httpx.ReadError("dropped"))])
        assert isinstance(result, ConnectionError)
        assert "weights.gguf" in str(result)


# --------------------------------------------------------------------------- #
# The retry loop
# --------------------------------------------------------------------------- #

class TestTheRetryLoop:
    def test_a_cancel_is_never_retried(self):
        dl = _downloader()
        attempts = []

        def attempt(n):
            attempts.append(n)
            raise DownloadCancelled("stop")

        with pytest.raises(DownloadCancelled):
            dl._run_with_retries("thing", attempt)
        assert attempts == [1]

    def test_a_token_already_set_stops_before_the_first_attempt(self):
        token = CancelToken()
        token.cancel()
        dl = _downloader(token)
        attempts = []

        with pytest.raises(DownloadCancelled):
            dl._run_with_retries("thing", lambda n: attempts.append(n))
        assert attempts == []

    def test_a_cancel_during_backoff_ends_it_immediately(self):
        """Backoff reaches 30s, and a stop request must not wait it out."""
        token = CancelToken()
        dl = _downloader(token)
        attempts = []

        def attempt(n):
            attempts.append(n)
            if n == 1:
                token.cancel()               # arrives while the first backoff runs
            raise httpx.ReadError("dropped")

        with pytest.raises(DownloadCancelled):
            dl._run_with_retries("thing", attempt)
        # One real attempt, then the backoff returned cancelled instead of
        # sleeping and re-entering.
        assert attempts == [1]
        assert len(attempts) < MAX_DOWNLOAD_ATTEMPTS

    def test_the_cancel_keeps_the_transient_failure_as_its_cause(self):
        token = CancelToken()
        dl = _downloader(token)

        def attempt(n):
            token.cancel()
            raise httpx.ReadError("dropped")

        with pytest.raises(DownloadCancelled) as excinfo:
            dl._run_with_retries("thing", attempt)
        assert isinstance(excinfo.value.__cause__, httpx.ReadError)


# --------------------------------------------------------------------------- #
# The byte loops
# --------------------------------------------------------------------------- #

class TestTheSimplePath:
    def test_it_stops_at_the_next_chunk(self, tmp_path, monkeypatch):
        token = CancelToken()
        dl = _downloader(token)
        resp = _CancellingResp([b"a" * 8, b"b" * 8, b"c" * 8], token, cancel_at=1)
        monkeypatch.setattr(ad.httpx, "stream", _Stream(resp))

        dest = str(tmp_path / "weights.gguf.part")
        with pytest.raises(DownloadCancelled):
            dl._download_file_simple("weights.gguf", "https://x/f", dest, 24)

        # Checked before the write, so the chunk in hand when the flag went up
        # is dropped rather than flushed — the file stops at the last one that
        # was already committed.
        assert open(dest, "rb").read() == b"a" * 8

    def test_the_socket_is_closed_on_the_way_out(self, tmp_path, monkeypatch):
        """Raising inside the ``with`` is what drops the connection.

        Returning instead would leave it to the 60s read timeout.
        """
        token = CancelToken()
        dl = _downloader(token)
        resp = _CancellingResp([b"a" * 8, b"b" * 8], token, cancel_at=0)
        monkeypatch.setattr(ad.httpx, "stream", _Stream(resp))

        with pytest.raises(DownloadCancelled):
            dl._download_file_simple("w.gguf", "https://x/f",
                                     str(tmp_path / "w.part"), 16)
        assert resp.closed

    def test_it_does_not_reconnect_after_a_cancel(self, tmp_path, monkeypatch):
        token = CancelToken()
        dl = _downloader(token)
        stream = _Stream(_CancellingResp([b"a" * 8, b"b" * 8], token, cancel_at=0))
        monkeypatch.setattr(ad.httpx, "stream", stream)

        with pytest.raises(DownloadCancelled):
            dl._download_file_simple("w.gguf", "https://x/f",
                                     str(tmp_path / "w.part"), 16)
        assert stream.calls == 1


class TestTheChunkedPath:
    def test_a_range_stops_at_the_next_read(self, tmp_path, monkeypatch):
        token = CancelToken()
        dl = _downloader(token)
        dest = tmp_path / "w.part"
        dest.write_bytes(b"\0" * 24)

        resp = _CancellingResp([b"a" * 8, b"b" * 8, b"c" * 8], token, cancel_at=1)
        monkeypatch.setattr(ad.httpx, "stream", _Stream(resp))

        with pytest.raises(DownloadCancelled):
            dl._download_chunk("https://x/f", str(dest), 0, 23, "w.gguf", 24,
                               [0], threading.Lock())
        assert dest.read_bytes() == b"a" * 8 + b"\0" * 16

    def test_a_queued_range_never_opens_a_socket(self, tmp_path, monkeypatch):
        """Every model worth downloading takes this path, so its pool is where
        a cancel usually lands — the workers not yet started must exit free."""
        token = CancelToken()
        token.cancel()
        dl = _downloader(token)
        stream = _Stream()
        monkeypatch.setattr(ad.httpx, "stream", stream)

        with pytest.raises(DownloadCancelled):
            dl._download_chunk("https://x/f", str(tmp_path / "w.part"), 0, 7,
                               "w.gguf", 8, [0], threading.Lock())
        assert stream.calls == 0

    def test_the_pool_wrapper_does_not_rewrap_a_cancel(self, tmp_path, monkeypatch):
        """``_classified`` would turn it into a ConnectionError and lose 499."""
        token = CancelToken()
        token.cancel()
        dl = _downloader(token)
        monkeypatch.setattr(ad.httpx, "stream", _Stream())

        with pytest.raises(DownloadCancelled):
            dl._download_file_chunked("w.gguf", "https://x/f",
                                      str(tmp_path / "w.part"), 64)


class TestTheFileLevel:
    def test_a_queued_file_never_opens_a_socket(self, tmp_path, monkeypatch):
        token = CancelToken()
        token.cancel()
        dl = _downloader(token)
        stream = _Stream()
        monkeypatch.setattr(ad.httpx, "stream", stream)

        with pytest.raises(DownloadCancelled):
            dl._download_file("a/weights.gguf", "https://x/f", str(tmp_path), 8, 0)
        assert stream.calls == 0

    def test_the_part_is_not_published_on_a_cancel(self, tmp_path, monkeypatch):
        token = CancelToken()
        dl = _downloader(token)
        resp = _CancellingResp([b"a" * 8, b"b" * 8], token, cancel_at=1)
        monkeypatch.setattr(ad.httpx, "stream", _Stream(resp))

        with pytest.raises(DownloadCancelled):
            dl._download_file("a/weights.gguf", "https://x/f", str(tmp_path), 16, 0)

        assert not (tmp_path / "weights.gguf").exists()
        assert (tmp_path / "weights.gguf.part").exists()

    def test_the_digest_is_interruptible(self, tmp_path):
        """A 4GB digest is seconds of work with no network in it."""
        big = tmp_path / "w.gguf"
        big.write_bytes(b"x" * (ad.CHUNK_SIZE * 2 + 16))
        token = CancelToken()
        token.cancel()

        with pytest.raises(DownloadCancelled):
            SLPArtifactDownloader._compute_local_md5(str(big), token)

    def test_the_digest_is_unchanged_without_a_token(self, tmp_path):
        f = tmp_path / "w.gguf"
        f.write_bytes(b"hello")
        assert (SLPArtifactDownloader._compute_local_md5(str(f))
                == hashlib.md5(b"hello").hexdigest())


class TestTheHeadProbe:
    def test_it_does_not_swallow_a_cancel(self, monkeypatch):
        """Swallowed, the probe answers "no size, no digest, no Range" and the
        file goes on to download single-stream and unverified."""
        token = CancelToken()
        token.cancel()
        dl = _downloader(token)
        monkeypatch.setattr(ad.httpx, "head",
                            lambda *a, **k: pytest.fail("probe should not run"))

        with pytest.raises(DownloadCancelled):
            dl._get_remote_file_info("https://x/f")

    def test_an_ordinary_probe_failure_still_falls_back(self, monkeypatch):
        dl = _downloader()

        def boom(*a, **k):
            raise httpx.ConnectError("no dns")

        monkeypatch.setattr(ad.httpx, "head", boom)
        assert dl._get_remote_file_info("https://x/f") == (0, None, False)
