"""MCP registration must not run on the event loop.

``register_mcp_server`` is ``async def``, but everything under it blocks: a
``subprocess.Popen`` with a 2s startup poll, up to three launch attempts
through the uvx/mcp<2 retry ladder, then local inference and an embedding pass.
Left on the loop thread that stalls every other request — and every open SSE
chat stream — for the better part of a minute.
"""

import asyncio
import threading

import pytest
from fastapi import HTTPException
from unittest.mock import MagicMock

from smartloop.api.model.mcp import MCPRegisterRequest
from smartloop.api.routers.mcp_servers import register_mcp_server
from smartloop.mcp.client import MCPDiscoveryError

_RESULT = {
    "id": "srv-1", "name": "demo", "enabled": True, "tools": [],
    "server_type": "local", "args": [],
}


def _register(svc, request):
    return asyncio.run(register_mcp_server("proj-1", request, None, svc))


def _local_request():
    return MCPRegisterRequest(server_type="local", command="uvx", args=["demo-mcp"])


class TestRegistrationRunsOffTheLoop:
    def test_blocking_service_call_runs_on_a_worker_thread(self):
        seen = []
        svc = MagicMock()
        svc.add_local_mcp_server.side_effect = (
            lambda **kw: (seen.append(threading.current_thread()), _RESULT)[1]
        )

        _register(svc, _local_request())

        assert seen, "add_local_mcp_server was never called"
        assert seen[0] is not threading.main_thread()

    def test_discovery_failure_still_maps_to_422(self):
        """The exception has to survive the executor hop, or a bad package
        would surface as a 500 instead of the actionable 422."""
        svc = MagicMock()
        svc.add_local_mcp_server.side_effect = MCPDiscoveryError("no such package")

        with pytest.raises(HTTPException) as exc:
            _register(svc, _local_request())
        assert exc.value.status_code == 422
        assert "no such package" in str(exc.value.detail)

    def test_value_error_still_maps_to_400(self):
        svc = MagicMock()
        svc.add_local_mcp_server.side_effect = ValueError("bad args")

        with pytest.raises(HTTPException) as exc:
            _register(svc, _local_request())
        assert exc.value.status_code == 400
