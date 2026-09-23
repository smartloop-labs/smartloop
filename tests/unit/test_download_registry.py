"""The registry that lets a second request find a running download."""

import threading
import time

import pytest

from smartloop.model.downloader import cancellation
from smartloop.model.downloader.cancellation import (
    CancelToken,
    DownloadCancelled,
    DownloadInProgress,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    cancellation.reset_for_tests()
    yield
    cancellation.reset_for_tests()


class TestTheToken:
    def test_check_is_silent_until_cancelled(self):
        token = CancelToken()
        token.check()                       # no raise
        token.cancel()
        with pytest.raises(DownloadCancelled):
            token.check()

    def test_the_message_names_what_was_stopped(self):
        token = CancelToken()
        token.cancel()
        with pytest.raises(DownloadCancelled, match="model.gguf"):
            token.check("model.gguf")

    def test_cancelling_twice_is_harmless(self):
        token = CancelToken()
        token.cancel()
        token.cancel()
        assert token.cancelled

    def test_wait_returns_false_when_the_time_runs_out(self):
        assert CancelToken().wait(0.01) is False

    def test_wait_returns_early_when_cancelled(self):
        token = CancelToken()
        threading.Timer(0.02, token.cancel).start()
        started = time.monotonic()
        assert token.wait(5.0) is True
        # The point of the substitution: a cancel does not wait out the backoff.
        assert time.monotonic() - started < 1.0


class TestClaimingAModel:
    def test_begin_hands_out_a_token_and_records_it(self):
        token = cancellation.begin("org/repo")
        assert cancellation.find("org/repo") is token
        assert cancellation.active_keys() == ["org/repo"]

    def test_a_second_download_of_the_same_model_is_refused(self):
        cancellation.begin("org/repo")
        with pytest.raises(DownloadInProgress, match="org/repo"):
            cancellation.begin("org/repo", wait=0)

    def test_a_different_model_is_not_refused(self):
        cancellation.begin("org/one")
        cancellation.begin("org/two")
        assert cancellation.active_keys() == ["org/one", "org/two"]

    def test_begin_waits_out_a_departing_entry(self):
        """Cancel-then-immediately-re-enable must not read as a duplicate."""
        first = cancellation.begin("org/repo")
        threading.Timer(0.05, lambda: cancellation.end("org/repo", first)).start()

        second = cancellation.begin("org/repo", wait=5.0)
        assert second is not first
        assert cancellation.find("org/repo") is second

    def test_the_refusal_carries_a_409(self):
        cancellation.begin("org/repo")
        with pytest.raises(DownloadInProgress) as excinfo:
            cancellation.begin("org/repo", wait=0)
        assert excinfo.value.status_code == 409


class TestReleasing:
    def test_end_releases_the_key(self):
        token = cancellation.begin("org/repo")
        cancellation.end("org/repo", token)
        assert cancellation.find("org/repo") is None
        assert cancellation.active_keys() == []

    def test_end_sets_done(self):
        token = cancellation.begin("org/repo")
        assert not token.done.is_set()
        cancellation.end("org/repo", token)
        assert token.done.is_set()

    def test_end_only_pops_its_own_token(self):
        """A slow wind-down must not unregister the download that replaced it.

        A cancelled transfer can still be inside a 60s socket read while its
        replacement has already claimed the key; an unconditional pop there
        would leave the live download with no token for the next cancel.
        """
        stale = cancellation.begin("org/repo")
        cancellation.end("org/repo", stale)
        live = cancellation.begin("org/repo")

        cancellation.end("org/repo", stale)          # the late release lands

        assert cancellation.find("org/repo") is live

    def test_a_late_release_still_marks_its_own_token_done(self):
        stale = cancellation.begin("org/repo")
        cancellation.end("org/repo", stale)
        cancellation.begin("org/repo")
        cancellation.end("org/repo", stale)
        assert stale.done.is_set()


class TestRequestingACancel:
    def test_it_sets_the_flag_and_returns_the_token(self):
        token = cancellation.begin("org/repo")
        assert cancellation.request_cancel("org/repo") is token
        assert token.cancelled

    def test_nothing_running_answers_none(self):
        assert cancellation.request_cancel("org/repo") is None

    def test_a_second_cancel_is_idempotent(self):
        cancellation.begin("org/repo")
        first = cancellation.request_cancel("org/repo")
        second = cancellation.request_cancel("org/repo")
        assert first is second
        assert first.cancelled


class TestConcurrency:
    def test_only_one_of_many_racing_claims_wins(self):
        winners = []
        barrier = threading.Barrier(8)

        def claim():
            barrier.wait()
            try:
                winners.append(cancellation.begin("org/repo", wait=0))
            except DownloadInProgress:
                pass

        threads = [threading.Thread(target=claim) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(winners) == 1
        assert cancellation.find("org/repo") is winners[0]
