"""The access token is renewed before it dies, and on demand when it already has.

A 401 used to delete the access token and leave the refresh token untouched,
with nothing to redeem it until the background loop's next tick — up to 45
minutes of being logged out while a valid refresh token sat on disk.  The loop
also slept *first*, so nothing refreshed at all during the first interval after
start.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from smartloop.auth import credential_store
from smartloop.platform import token_agent
from smartloop.platform.token_agent import (
    REFRESH_INTERVAL_SECONDS, _jwt_exp, _needs_refresh, refresh_access_token_now,
)


def _jwt(exp_offset_seconds):
    """A JWT-shaped token whose exp is *offset* seconds from now."""
    import base64, json

    payload = base64.urlsafe_b64encode(
        json.dumps({"exp": int(time.time() + exp_offset_seconds)}).encode()
    ).decode().rstrip("=")
    return f"header.{payload}.signature"


@pytest.fixture(autouse=True)
def _clean_credentials():
    for clear in (credential_store.clear_access_token,
                  credential_store.clear_refresh_token,
                  credential_store.clear_token):
        clear()
    yield
    for clear in (credential_store.clear_access_token,
                  credential_store.clear_refresh_token,
                  credential_store.clear_token):
        clear()


class TestReadingTheExpiry:
    def test_a_jwt_exp_is_read(self):
        assert _jwt_exp(_jwt(600)) == pytest.approx(time.time() + 600, abs=5)

    @pytest.mark.parametrize("token", ["", "opaque-token", "a.b", "a.b.c.d", None])
    def test_anything_that_is_not_a_jwt_reads_as_unknown(self, token):
        assert _jwt_exp(token) is None

    def test_undecodable_claims_read_as_unknown(self):
        assert _jwt_exp("header.!!!not-base64!!!.sig") is None


class TestDecidingWhetherToRefresh:
    def test_no_refresh_token_means_nothing_to_do(self):
        credential_store.set_access_token(_jwt(-60))
        assert _needs_refresh() is False

    def test_a_missing_access_token_is_refreshed(self):
        """The state a 401 used to leave behind — and never recover from."""
        credential_store.set_refresh_token("rt")
        assert _needs_refresh() is True

    def test_a_token_expiring_before_the_next_tick_is_refreshed(self):
        credential_store.set_refresh_token("rt")
        credential_store.set_access_token(_jwt(REFRESH_INTERVAL_SECONDS - 60))
        assert _needs_refresh() is True

    def test_an_already_expired_token_is_refreshed(self):
        credential_store.set_refresh_token("rt")
        credential_store.set_access_token(_jwt(-60))
        assert _needs_refresh() is True

    def test_a_token_that_outlives_the_next_tick_is_left_alone(self):
        credential_store.set_refresh_token("rt")
        credential_store.set_access_token(_jwt(REFRESH_INTERVAL_SECONDS + 600))
        assert _needs_refresh() is False

    def test_an_opaque_token_is_refreshed_on_the_interval(self):
        """No expiry to read — renew on schedule, as it always did."""
        credential_store.set_refresh_token("rt")
        credential_store.set_access_token("opaque")
        assert _needs_refresh() is True


class TestRedeemingTheRefreshToken:
    def _client(self, **kwargs):
        client = MagicMock()
        client.refresh_access_token = MagicMock(**kwargs)
        return client

    def test_a_new_access_token_is_stored(self):
        credential_store.set_refresh_token("old-rt")
        client = self._client(return_value={
            "access_token": "fresh", "refresh_token": "new-rt",
        })

        assert refresh_access_token_now(client) is True
        assert credential_store.get_access_token() == "fresh"
        assert credential_store.get_refresh_token() == "new-rt"

    def test_rotation_is_optional(self):
        credential_store.set_refresh_token("old-rt")
        client = self._client(return_value={"access_token": "fresh"})

        assert refresh_access_token_now(client) is True
        assert credential_store.get_refresh_token() == "old-rt"

    def test_a_rejected_refresh_token_clears_both(self):
        credential_store.set_access_token("stale")
        credential_store.set_refresh_token("dead-rt")
        client = self._client(side_effect=PermissionError("401"))

        assert refresh_access_token_now(client) is False
        assert credential_store.get_access_token() is None
        assert credential_store.get_refresh_token() is None

    def test_an_outage_keeps_the_credentials_for_a_retry(self):
        """A network failure is not evidence the token is bad."""
        credential_store.set_access_token("stale")
        credential_store.set_refresh_token("rt")
        client = self._client(side_effect=ConnectionError("unreachable"))

        assert refresh_access_token_now(client) is False
        assert credential_store.get_refresh_token() == "rt"
        assert credential_store.get_access_token() == "stale"

    def test_a_reply_with_no_token_changes_nothing(self):
        credential_store.set_refresh_token("rt")
        client = self._client(return_value={})

        assert refresh_access_token_now(client) is False
        assert credential_store.get_access_token() is None
        assert credential_store.get_refresh_token() == "rt"

    def test_nothing_to_redeem_is_not_an_error(self):
        client = self._client()

        assert refresh_access_token_now(client) is False
        client.refresh_access_token.assert_not_called()

    @pytest.mark.parametrize("status", [429, 500, 502, 503])
    def test_a_platform_outage_is_not_a_rejection(self, status):
        """Only a 401 may clear the credentials.

        ``_raise_for_status`` used to give every non-2xx the endpoint's error
        class, so a 500 from the refresh endpoint arrived as ``PermissionError``
        and landed in the branch below that clears *both* tokens.  A bad minute
        on the platform logged the user out with no way back but a re-login —
        the exact opposite of what the docstring promised.
        """
        credential_store.set_access_token("stale")
        credential_store.set_refresh_token("rt")
        exc = ConnectionError(f"API error ({status})")
        exc.status_code = status
        client = self._client(side_effect=exc)

        assert refresh_access_token_now(client) is False
        assert credential_store.get_access_token() == "stale"
        assert credential_store.get_refresh_token() == "rt"


class TestConcurrentCallersRedeemOnce:
    """Two callers reach the refresh agent: the background loop and a request
    thread whose call just came back 401.  With single-use rotating refresh
    tokens, the second redemption of the same token is *refused* — which takes
    the 401 branch and clears both credentials, logging the user out on what was
    a successful refresh.

    Two, because these run in the workspace owner.  A pinned agent child is a
    third caller the lock below could not order, and it declines instead — see
    ``test_token_refresh_is_workspace_work``.
    """

    def test_the_refresh_token_is_spent_once(self):
        credential_store.set_refresh_token("old-rt")
        started = threading.Barrier(2)
        calls = []

        def redeem(refresh_token):
            calls.append(refresh_token)
            time.sleep(0.05)  # hold the lock long enough for the other to queue
            return {"access_token": f"fresh-{len(calls)}", "refresh_token": "rotated"}

        client = MagicMock()
        client.refresh_access_token = MagicMock(side_effect=redeem)

        def call():
            started.wait()
            return refresh_access_token_now(client)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = [f.result() for f in [pool.submit(call), pool.submit(call)]]

        assert results == [True, True]
        assert calls == ["old-rt"], f"refresh token redeemed {len(calls)} times"
        assert credential_store.get_access_token() == "fresh-1"
        assert credential_store.get_refresh_token() == "rotated"

    def test_a_failed_redemption_holds_off_the_queue_behind_it(self):
        """The lock alone only collapses callers when the refresh *succeeds*.

        On failure the stored token is unchanged, so every waiter falls through
        and pays the full timeout in turn. With the auth endpoints served from
        the threadpool, ten concurrent 401s against a degraded platform meant
        ten serialized redemptions holding ten threadpool slots.
        """
        credential_store.set_refresh_token("rt")
        client = MagicMock()
        client.refresh_access_token = MagicMock(
            side_effect=ConnectionError("platform unreachable"))

        assert refresh_access_token_now(client) is False
        assert refresh_access_token_now(client) is False
        assert refresh_access_token_now(client) is False

        assert client.refresh_access_token.call_count == 1
        # Nothing was thrown away — the credentials are still there to retry with.
        assert credential_store.get_refresh_token() == "rt"

    def test_the_hold_off_lifts_once_one_succeeds(self):
        credential_store.set_refresh_token("rt")
        client = MagicMock()
        client.refresh_access_token = MagicMock(side_effect=ConnectionError("down"))
        assert refresh_access_token_now(client) is False

        token_agent._last_failure_at = 0.0  # window elapsed
        client.refresh_access_token = MagicMock(return_value={"access_token": "fresh"})
        assert refresh_access_token_now(client) is True
        assert credential_store.get_access_token() == "fresh"

    def test_a_caller_that_genuinely_needs_one_still_gets_one(self):
        """The short-circuit is "someone else just refreshed", not "ever refreshed"."""
        credential_store.set_refresh_token("rt")
        client = MagicMock()
        client.refresh_access_token = MagicMock(
            side_effect=[{"access_token": "first"}, {"access_token": "second"}],
        )

        assert refresh_access_token_now(client) is True
        assert refresh_access_token_now(client) is True
        assert credential_store.get_access_token() == "second"


class TestTheLoopRunsBeforeItSleeps:
    @pytest.mark.asyncio
    async def test_the_first_check_happens_immediately(self):
        """Sleeping first left the first whole interval unprotected."""
        calls = []

        async def _stop_after_first(_seconds):
            raise _Done()

        class _Done(Exception):
            pass

        with patch.object(token_agent, "_needs_refresh", return_value=True), \
             patch.object(token_agent, "refresh_access_token_now",
                          side_effect=lambda c=None: calls.append("refreshed")), \
             patch.object(token_agent.asyncio, "sleep", _stop_after_first), \
             patch.object(token_agent, "SLPPlatformClient", MagicMock()):
            with pytest.raises(_Done):
                await token_agent.run_token_refresh_loop()

        assert calls == ["refreshed"]

    @pytest.mark.asyncio
    async def test_a_failing_cycle_does_not_kill_the_loop(self):
        cycles = []

        class _Done(Exception):
            pass

        async def _count(_seconds):
            cycles.append(1)
            if len(cycles) >= 2:
                raise _Done()

        with patch.object(token_agent, "_needs_refresh",
                          side_effect=RuntimeError("credential store unreadable")), \
             patch.object(token_agent.asyncio, "sleep", _count), \
             patch.object(token_agent, "SLPPlatformClient", MagicMock()):
            with pytest.raises(_Done):
                await token_agent.run_token_refresh_loop()

        assert len(cycles) == 2


class TestTheLoopDoesNotBlockTheEventLoop:
    """Redemption is synchronous; the loop that drives it must not be.

    ``refresh_access_token_now`` is a blocking httpx POST with a 10s budget,
    taken behind a lock a request thread may already hold.  Awaited inline it
    stalls the whole event loop — and under supervision that loop is the
    supervisor's, which proxies every project's chat stream through it byte by
    byte, so one expired token would pause generation everywhere at once.
    """

    async def test_other_work_runs_while_a_redemption_is_in_flight(self):
        """Ordering, not elapsed time.

        A blocking call does not make this hang — the redemption still returns
        eventually and everything queued behind it still runs.  What gives it
        away is *when*: inline, the refresh finishes before any other task gets
        a turn.  So the assertion is that the other task ran first, not that
        the test was quick.
        """
        import asyncio

        credential_store.set_refresh_token("rt")
        credential_store.set_access_token(_jwt(-10))   # expired: force a refresh

        redeeming = threading.Event()
        release = threading.Event()
        order = []

        def _slow_refresh(client):
            redeeming.set()
            release.wait(timeout=5)
            order.append("refresh-done")
            return True

        async def _other_work():
            for _ in range(3):
                await asyncio.sleep(0)
                order.append("tick")

        with patch.object(token_agent, "refresh_access_token_now", _slow_refresh):
            loop_task = asyncio.create_task(token_agent.run_token_refresh_loop())
            other = asyncio.create_task(_other_work())

            await asyncio.to_thread(redeeming.wait, 5)
            await other

            assert order == ["tick", "tick", "tick"], (
                "the redemption ran to completion before the event loop got a "
                f"turn — it is blocking the loop (order was {order})"
            )

            release.set()
            loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                pass
