"""What a client polls to know a connection is ready to use.

Registering an MCP server returns before it is usable: tool discovery runs on
the MCP task.  Until it finishes the connection can answer nothing — so the API
has to say so, otherwise the UI shows a Notion connection that looks connected
and knows nothing.

The block is still called ``indexing`` on the wire, because that is the field
clients read; what it reports is provisioning, which is all that happens in the
background now that content crawling is gone.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from smartloop.api.services.model_service import ModelService
from smartloop.workspace import MCPServer, MCPServerStatus, MCPToolDefinition


def _server(**fields):
    fields.setdefault("id", "srv1")
    fields.setdefault("name", "notion")
    fields.setdefault("catalog_slug", "notion")
    fields.setdefault("tools", [
        MCPToolDefinition(name="search", description="d", input_schema={}),
    ])
    return MCPServer(**fields)


class TestProvisioningState:
    def test_a_server_still_provisioning_is_in_progress(self):
        state = ModelService._mcp_provisioning_state(
            _server(status=MCPServerStatus.PENDING),
        )
        assert state["state"] == "provisioning"
        assert state["in_progress"] is True

    def test_a_discovered_server_is_ready(self):
        state = ModelService._mcp_provisioning_state(_server())
        assert state["state"] == "ready"
        assert state["in_progress"] is False
        assert state["error"] is None

    def test_a_failed_provisioning_carries_its_reason(self):
        state = ModelService._mcp_provisioning_state(
            _server(status=MCPServerStatus.FAILED, last_error="401 from server"),
        )
        assert state["state"] == "failed"
        assert state["in_progress"] is False
        assert state["error"] == "401 from server"

    def test_the_tool_count_is_reported(self):
        """The tools discovery found are the catalog a request is resolved
        against, so "ready with 0 tools" is a state worth being able to see."""
        assert ModelService._mcp_provisioning_state(_server())["tools"] == 1
        assert ModelService._mcp_provisioning_state(_server(tools=[]))["tools"] == 0


class TestServerStatusEndpointShape:
    def _svc(self, *servers):
        project = SimpleNamespace(
            id="p1", name="proj", project_dir="/tmp/p1", mcp_servers=list(servers),
        )
        svc = MagicMock()
        svc._get_project = MagicMock(return_value=project)
        svc._model.workspace.current_project = project
        svc._mcp_server_to_dict = ModelService._mcp_server_to_dict
        svc._mcp_provisioning_state = ModelService._mcp_provisioning_state
        return svc

    def test_the_server_dict_carries_its_indexing_block(self):
        svc = self._svc(_server(status=MCPServerStatus.PENDING))
        result = ModelService.mcp_server_status(svc, "srv1", project_id="p1")
        assert result["id"] == "srv1"
        assert result["indexing"]["state"] == "provisioning"
        assert result["indexing"]["in_progress"] is True

    def test_an_unknown_server_is_a_not_found(self):
        svc = self._svc(_server())
        try:
            ModelService.mcp_server_status(svc, "nope", project_id="p1")
        except ValueError as e:
            assert "nope" in str(e)
        else:
            raise AssertionError("expected a ValueError the router maps to 404")

    def test_the_list_carries_the_same_block(self):
        # Otherwise watching a freshly registered connection costs one request
        # per server on every poll.
        svc = self._svc(_server())
        servers = ModelService.list_mcp_servers_with_status(svc, "p1")
        assert [s["indexing"]["state"] for s in servers] == ["ready"]

    def test_system_servers_stay_out_of_the_list(self):
        svc = self._svc(_server(), _server(id="sys", name="smartloop", system=True))
        servers = ModelService.list_mcp_servers_with_status(svc, "p1")
        assert [s["id"] for s in servers] == ["srv1"]
