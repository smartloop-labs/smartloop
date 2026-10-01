"""Re-checking a connection has to actually re-check it.

``POST /v1/projects/{id}/mcp/{server_id}/refresh`` is the only recovery the UI
offers.  It cleared ``status`` and ``last_error`` up front and queued a
provisioning pass — and that pass only discovers when a server has no tools.  A
connection whose authorization died keeps every tool discovery ever found, so
the pass skipped discovery and set it READY again, having made no network call
at all.  The user pressed the one button there was and was told it was fixed.

The other half: discovery runs on the stored credentials, so it ages out like
any call.  Renewing has to be tried before a connection is called broken, or
fixing the false green just produces a false red.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from smartloop.api.services.model_service import ModelService
from smartloop.mcp.client import (
    MCPAuthExpiredError, MCPConnectResult, MCPConnectionError,
)
from smartloop.workspace.workspace import MCPServerStatus


_REFRESH = (
    "smartloop.agents.mcp.mcp_tool_caller.MCPToolCaller.refresh_access_token"
)


def _tool(name="search"):
    return SimpleNamespace(name=name, description="", input_schema={})


def _server(status=MCPServerStatus.FAILED, tools=None, token="stale"):
    """A remote connection that failed on auth but kept its catalog."""
    return SimpleNamespace(
        id="s1", name="notion",
        server_type=SimpleNamespace(value="remote"),
        status=status,
        last_error="This connection's authorization has expired. "
                   "Reconnect it to keep using it.",
        tools=tools if tools is not None else [_tool()],
        description=None, summary=None, server_info=None, account=None,
        catalog_slug=None, enabled=True, system=False,
        remote_config=SimpleNamespace(
            server_url="https://mcp.notion.com/mcp", token=None,
            token_endpoint="https://t", resource=None,
        ),
        oauth_credentials=SimpleNamespace(
            access_token=token, refresh_token="rt",
            client_id="c", client_secret="s", expires_at=None,
        ),
    )


class TestARetryDropsTheCatalogItCannotTrust:

    def _svc(self, server):
        svc = MagicMock()
        svc._model = MagicMock()
        project = SimpleNamespace(id="p1", mcp_servers=[server])
        svc._get_project.return_value = project
        svc._model.workspace.current_project = project
        svc._mcp_server_to_dict = lambda s: {"id": s.id}
        return svc, project

    def test_the_cached_tools_are_cleared(self):
        server = _server()
        svc, _ = self._svc(server)

        ModelService.retry_mcp_server(svc, "s1", project_id="p1")

        assert server.tools == []
        assert server.status == MCPServerStatus.PENDING

    def test_the_pass_is_queued(self):
        server = _server()
        svc, project = self._svc(server)

        ModelService.retry_mcp_server(svc, "s1", project_id="p1")

        svc._queue_mcp_provisioning.assert_called_once_with(project, server)

    def test_provisioning_then_discovers_rather_than_trusting_the_cache(self):
        """The two halves together: after a retry, the pass must go to the wire."""
        server = _server()
        svc, project = self._svc(server)
        ModelService.retry_mcp_server(svc, "s1", project_id="p1")

        svc._discover_mcp_tools = MagicMock()
        svc._enrich_mcp_server = MagicMock()
        svc._probe_mcp_account = MagicMock()
        ModelService._provision_mcp_server(svc, "p1", "s1")

        svc._discover_mcp_tools.assert_called_once_with(server)

    def test_without_a_retry_a_known_catalog_is_still_not_rediscovered(self):
        """The guard earns its keep on the registration path — leave it there."""
        server = _server(status=MCPServerStatus.PENDING)
        svc, _ = self._svc(server)
        svc._discover_mcp_tools = MagicMock()
        svc._enrich_mcp_server = MagicMock()
        svc._probe_mcp_account = MagicMock()

        ModelService._provision_mcp_server(svc, "p1", "s1")

        svc._discover_mcp_tools.assert_not_called()


class TestDiscoveryRenewsBeforeGivingUp:

    def _svc(self, server):
        svc = MagicMock()
        svc._model = MagicMock()
        svc._model.workspace.projects = [SimpleNamespace(mcp_servers=[server])]
        svc._persist_mcp_server = MagicMock()
        svc._friendly_discovery_error = lambda e: e
        return svc

    def test_an_expired_token_is_renewed_and_discovery_retried(self):
        server = _server(tools=[])
        svc = self._svc(server)
        refused = MCPConnectResult(success=False, error="401", auth_expired=True)
        accepted = MCPConnectResult(success=True, tools=[_tool("search")])

        with patch("smartloop.mcp.MCPClient") as client_cls, \
             patch(_REFRESH, return_value="fresh") as refresh:
            client = client_cls.return_value.__enter__.return_value
            client.connect_and_discover_tools.side_effect = [refused, accepted]
            ModelService._discover_mcp_tools(svc, server)

        refresh.assert_called_once()
        assert [t.name for t in server.tools] == ["search"]
        # The second attempt must carry the renewed credential, not the stale one.
        assert client.connect_and_discover_tools.call_args.kwargs[
            "access_token"
        ] == "fresh"

    def test_a_retired_grant_keeps_its_own_diagnosis(self):
        """Not overwritten by whatever the wire said."""
        server = _server(tools=[])
        svc = self._svc(server)
        refused = MCPConnectResult(success=False, error="401", auth_expired=True)

        def _die(srv, stale_token=None):
            srv.status = MCPServerStatus.FAILED
            return None

        with patch("smartloop.mcp.MCPClient") as client_cls, \
             patch(_REFRESH, side_effect=_die):
            client = client_cls.return_value.__enter__.return_value
            client.connect_and_discover_tools.return_value = refused
            with pytest.raises(MCPAuthExpiredError):
                ModelService._discover_mcp_tools(svc, server)

    def test_the_provisioning_pass_preserves_that_message(self):
        server = _server(tools=[])
        svc = MagicMock()
        svc._model = MagicMock()
        project = SimpleNamespace(id="p1", mcp_servers=[server])
        svc._get_project.return_value = project
        svc._discover_mcp_tools.side_effect = MCPAuthExpiredError("x")

        result = ModelService._provision_mcp_server(svc, "p1", "s1")

        assert result["status"] == "failed"
        assert "Reconnect it" in server.last_error
        assert server.status == MCPServerStatus.FAILED


class TestTheConnectResultCarriesTheAuthFailure:
    """Decided from the status, as the tool-call path already does."""

    def _result_for(self, exc):
        from smartloop.mcp.client import MCPClient
        client = MCPClient.__new__(MCPClient)
        with patch.object(MCPClient, "_send_request", side_effect=exc):
            return client.connect_and_discover_tools("https://mcp.example.com/mcp")

    def test_a_401_sets_auth_expired(self):
        result = self._result_for(MCPConnectionError("nope", status_code=401))
        assert result.success is False
        assert result.auth_expired is True

    def test_a_403_does_not(self):
        """Insufficient permission is not something a new token fixes."""
        result = self._result_for(MCPConnectionError("nope", status_code=403))
        assert result.auth_expired is False

    def test_a_transport_error_does_not(self):
        result = self._result_for(MCPConnectionError("connection refused"))
        assert result.auth_expired is False
