"""A refreshed MCP token has to reach the workspace.

``MCPToolCaller.refresh_access_token`` stored the new credentials on the
in-memory server object and stopped there.  Only the content crawl wrote them
back, through its own ``on_server_updated`` hook — so whether a connection
survived a restart depended on which code path happened to refresh it last.
With a provider that rotates refresh tokens, the workspace kept one the provider
had already retired, and the connection could never be renewed again.
"""

from unittest.mock import MagicMock, patch

import pytest

from smartloop.agents.mcp.mcp_tool_caller import MCPToolCaller


def _server(**creds):
    from smartloop.workspace.workspace import (
        MCPAuthType, MCPOAuthCredentials, MCPServer, MCPServerType,
        RemoteMCPServerConfig,
    )

    return MCPServer(
        id="s1",
        name="monday",
        server_type=MCPServerType.REMOTE,
        remote_config=RemoteMCPServerConfig(
            server_url="https://mcp.monday.com/mcp",
            auth_type=MCPAuthType.OAUTH,
            token_endpoint="https://auth.monday.com/oauth_ms/oauth/token",
        ),
        oauth_credentials=MCPOAuthCredentials(
            client_id="cid", access_token="expired", refresh_token="old-rt", **creds,
        ),
    )


def _reply(payload):
    """Patch httpx.Client so the refresh grant returns *payload*."""
    resp = MagicMock()
    resp.json = MagicMock(return_value=payload)
    resp.raise_for_status = MagicMock()
    http = MagicMock()
    http.post = MagicMock(return_value=resp)
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=http)
    ctx.__exit__ = MagicMock(return_value=False)
    return patch("httpx.Client", return_value=ctx)


def _reply_from(grant):
    """Like :func:`_reply`, but *grant* is called with the posted form."""
    def _post(url, data=None, **kwargs):
        resp = MagicMock()
        resp.json = MagicMock(return_value=grant(data))
        resp.raise_for_status = MagicMock()
        return resp

    http = MagicMock()
    http.post = MagicMock(side_effect=_post)
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=http)
    ctx.__exit__ = MagicMock(return_value=False)
    return patch("httpx.Client", return_value=ctx)


class TestTheRefreshIsWrittenBack:
    def test_rotated_credentials_are_persisted(self):
        server = _server()
        saves = []
        caller = MCPToolCaller(None, None, persist=lambda: saves.append(1))

        with _reply({"access_token": "fresh", "refresh_token": "rotated"}):
            assert caller.refresh_access_token(server) == "fresh"

        assert server.oauth_credentials.access_token == "fresh"
        assert server.oauth_credentials.refresh_token == "rotated"
        assert saves == [1], "the rotated refresh token never reached the workspace"

    def test_a_failed_write_does_not_sink_the_call(self):
        """The token in memory is good — the call this refresh was for proceeds."""
        server = _server()

        def boom():
            raise OSError("read-only file system")

        caller = MCPToolCaller(None, None, persist=boom)
        with _reply({"access_token": "fresh"}):
            assert caller.refresh_access_token(server) == "fresh"

    def test_nothing_is_persisted_when_the_refresh_fails(self):
        server = _server()
        saves = []
        caller = MCPToolCaller(None, None, persist=lambda: saves.append(1))

        with _reply({"token_type": "Bearer"}):  # no access_token
            assert caller.refresh_access_token(server) is None

        assert saves == []
        assert server.oauth_credentials.access_token == "expired"

    def test_a_caller_without_a_hook_still_refreshes(self):
        """The content crawl persists through its own ``on_server_updated``."""
        server = _server()
        caller = MCPToolCaller(None, None)

        with _reply({"access_token": "fresh"}):
            assert caller.refresh_access_token(server) == "fresh"

    @pytest.mark.parametrize("missing", ["refresh_token", "token_endpoint"])
    def test_nothing_to_refresh_with_is_not_an_error(self, missing):
        server = _server()
        if missing == "refresh_token":
            server.oauth_credentials.refresh_token = None
        else:
            server.remote_config.token_endpoint = None

        caller = MCPToolCaller(None, None, persist=lambda: pytest.fail("no refresh happened"))
        assert caller.refresh_access_token(server) is None


class TestOneRedemptionPerRotation:
    """Concurrent 401s must not spend the same refresh token twice.

    Parallel step groups and tool fan-out share one caller, so several calls can
    come back 401 together.  Each used to redeem independently — and a provider
    that issues single-use refresh tokens invalidates the grant on the second
    attempt, killing the connection the refresh was meant to save.
    """

    def test_the_loser_of_the_race_reuses_the_winners_token(self):
        server = _server()
        caller = MCPToolCaller(None, None)

        # The state after another thread has already refreshed: the access token
        # on the server is no longer the one this caller's request failed with.
        server.oauth_credentials.access_token = "fresh"

        with patch("httpx.Client") as http:
            token = caller.refresh_access_token(server, stale_token="expired")

        assert token == "fresh"
        http.assert_not_called()
        assert server.oauth_credentials.refresh_token == "old-rt"

    def test_concurrent_callers_redeem_exactly_once(self):
        import threading
        import time

        server = _server()
        caller = MCPToolCaller(None, None)
        grants = []
        barrier = threading.Barrier(4)

        def _grant(payload):
            grants.append(payload["refresh_token"])
            n = len(grants)
            # The round trip is what makes the window real: without the lock
            # every thread clears the stale check while the first is still
            # waiting on the network, and they all redeem.
            time.sleep(0.05)
            return {"access_token": f"fresh-{n}", "refresh_token": "new-rt"}

        with _reply_from(_grant):
            results = []
            lock = threading.Lock()

            def _go():
                barrier.wait()
                token = caller.refresh_access_token(server, stale_token="expired")
                with lock:
                    results.append(token)

            threads = [threading.Thread(target=_go) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(grants) == 1, f"redeemed {len(grants)} times: {grants}"
        assert set(results) == {"fresh-1"}

    def test_no_stale_token_still_refreshes(self):
        """Callers that cannot say what they failed with are not blocked."""
        server = _server()
        caller = MCPToolCaller(None, None)
        server.oauth_credentials.access_token = "something-else"

        with _reply({"access_token": "fresh", "refresh_token": "new-rt"}):
            assert caller.refresh_access_token(server) == "fresh"
