"""A refresh token belongs to the connection, not to whoever is holding it.

Redemption was serialized on a lock held per ``MCPToolCaller``, and callers are
built freely: one per orchestrator, a fresh one for every provisioning probe,
another for each read-only call that is not passed one.  So the lock covered a
single orchestrator's fan-out and nothing else — a probe and a chat turn held
different locks over the same token and could redeem it in parallel, which with
a single-use grant invalidates it between them.  That is the failure the
serialization was added to prevent.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from smartloop.agents.mcp import mcp_tool_caller as mod
from smartloop.agents.mcp.mcp_tool_caller import MCPToolCaller, reset_refresh_state


@pytest.fixture(autouse=True)
def _clean():
    reset_refresh_state()
    yield
    reset_refresh_state()


def _server(server_id="s1"):
    return SimpleNamespace(
        id=server_id, name="notion", status="ready", last_error=None,
        remote_config=SimpleNamespace(
            server_url="https://mcp.notion.com/mcp", token=None,
            token_endpoint="https://auth.notion.com/token", resource=None,
        ),
        oauth_credentials=SimpleNamespace(
            access_token="stale", refresh_token="rt-1",
            client_id="c", client_secret="s", expires_at=None,
        ),
    )


class _Exchange:
    """A token endpoint that is slow enough for a second caller to arrive."""

    def __init__(self, delay=0.05):
        self.calls = 0
        self._delay = delay
        self._lock = threading.Lock()

    def __call__(self, *a, **kw):
        with self._lock:
            self.calls += 1
            n = self.calls
        time.sleep(self._delay)
        return SimpleNamespace(
            access_token=f"fresh-{n}", refresh_token=f"rt-{n + 1}",
            expires_in=3600, token_type="Bearer",
        )


def _run_both(server_a, server_b):
    """Two separately-constructed callers refresh concurrently."""
    exchange = _Exchange()
    results = {}

    def go(key, server):
        caller = MCPToolCaller(None, None)        # a fresh caller, as in prod
        results[key] = caller.refresh_access_token(server, stale_token="stale")

    with patch("smartloop.mcp.oauth.MCPOAuthService") as svc:
        svc.return_value.__enter__.return_value.refresh_access_token = exchange
        threads = [
            threading.Thread(target=go, args=("a", server_a)),
            threading.Thread(target=go, args=("b", server_b)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    return exchange, results


class TestOneConnectionIsRedeemedOnce:

    def test_two_callers_spend_the_refresh_token_once(self):
        server = _server()

        exchange, results = _run_both(server, server)

        assert exchange.calls == 1
        # The loser reuses what the winner minted rather than redeeming again.
        assert results["a"] == results["b"] == "fresh-1"

    def test_the_connection_keeps_the_rotation(self):
        server = _server()

        _run_both(server, server)

        assert server.oauth_credentials.access_token == "fresh-1"
        assert server.oauth_credentials.refresh_token == "rt-2"


class TestConnectionsDoNotBlockEachOther:

    def test_two_servers_refresh_concurrently(self):
        """Per-server locks: a slow token endpoint stalls its own connection."""
        exchange, results = _run_both(_server("s1"), _server("s2"))

        assert exchange.calls == 2
        assert sorted(results.values()) == ["fresh-1", "fresh-2"]


class TestTheCooldownIsSharedToo:

    def test_a_refusal_recorded_by_one_caller_holds_back_another(self):
        server = _server()
        mod._refresh_failures[server.id] = (
            "rt-1", time.monotonic() + mod._REFRESH_FAILURE_COOLDOWN_S,
        )

        with patch("smartloop.mcp.oauth.MCPOAuthService") as svc:
            exchange = _Exchange(delay=0)
            svc.return_value.__enter__.return_value.refresh_access_token = exchange
            token = MCPToolCaller(None, None).refresh_access_token(
                server, stale_token="stale",
            )

        assert token is None
        assert exchange.calls == 0, "the refused token must not be re-offered"

    def test_a_reconnection_is_not_held_back(self):
        """The cooldown is keyed on the token that failed, not the server."""
        server = _server()
        mod._refresh_failures[server.id] = (
            "rt-old", time.monotonic() + mod._REFRESH_FAILURE_COOLDOWN_S,
        )

        with patch("smartloop.mcp.oauth.MCPOAuthService") as svc:
            exchange = _Exchange(delay=0)
            svc.return_value.__enter__.return_value.refresh_access_token = exchange
            token = MCPToolCaller(None, None).refresh_access_token(
                server, stale_token="stale",
            )

        assert token == "fresh-1"
