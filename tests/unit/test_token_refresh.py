"""The access token is renewed before it dies, and on demand when it already has.

A 401 used to delete the access token and leave the refresh token untouched,
with nothing to redeem it until the background loop's next tick — up to 45
minutes of being logged out while a valid refresh token sat on disk.  The loop
also slept *first*, so nothing refreshed at all during the first interval after
start.
"""

import time
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
