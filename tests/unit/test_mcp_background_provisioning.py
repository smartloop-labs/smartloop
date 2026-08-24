"""Registering an MCP server must not block on discovery, the LLM or indexing.

Registration used to launch the server process, call the LLM twice for
description/tags/summary and embed the tool vectors, all in front of the
client's request.  It now persists the connection and hands the rest to the MCP
task; these tests pin that the slow work has actually left the request path and
that the background pass records a state the client can act on.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from smartloop.api.services.model_service import ModelService
from smartloop.mcp.client import MCPConnectResult, MCPDiscoveryError
from smartloop.tasks.mcp_task import MCPTask
from smartloop.workspace import MCPServerStatus, MCPToolDefinition

_TOOLS = [MCPToolDefinition(name="search", description="d", input_schema={"type": "object"})]


def _project():
    return SimpleNamespace(id="p1", name="proj", project_dir="/tmp/p1", mcp_servers=[])


def _svc(project):
    svc = MagicMock()
    svc._get_project = MagicMock(return_value=project)
    svc._model = MagicMock()
    svc._model.workspace.projects = [project]
    svc._model.workspace.current_project = project
    svc._friendly_discovery_error = ModelService._friendly_discovery_error
    # Static/real helpers the code under test calls on self; MagicMock would
    # otherwise stub them out and hide what the methods actually return.
    svc._mcp_server_to_dict = ModelService._mcp_server_to_dict
    return svc


class TestRegistrationIsNonBlocking:
    def test_local_registration_does_not_launch_the_server(self):
        project = _project()
        svc = _svc(project)
        with patch("smartloop.api.services.model_service.LocalMCPClient") as client_cls:
            ModelService.add_local_mcp_server(
                svc, command="uvx", args=["--from", "pkg", "pkg"], project_id="p1",
            )
        client_cls.assert_not_called()

    def test_local_registration_makes_no_llm_call_and_no_index_write(self):
        project = _project()
        svc = _svc(project)
        with patch("smartloop.api.services.model_service.LocalMCPClient"):
            ModelService.add_local_mcp_server(
                svc, command="uvx", args=["--from", "pkg", "pkg"], project_id="p1",
            )
        svc._enrich_mcp_server.assert_not_called()
        svc._index_mcp_tools.assert_not_called()

    def test_local_registration_persists_pending_and_queues_the_work(self):
        project = _project()
        svc = _svc(project)
        with patch("smartloop.api.services.model_service.LocalMCPClient"):
            result = ModelService.add_local_mcp_server(
                svc, command="uvx", args=["--from", "pkg", "pkg"], project_id="p1",
            )
        assert len(project.mcp_servers) == 1
        assert project.mcp_servers[0].status == MCPServerStatus.PENDING
        assert result["status"] == "pending"
        svc._queue_mcp_provisioning.assert_called_once()

    def test_registration_never_loads_a_model(self):
        """The whole point: no inference, and no model load, on this path.

        ``_enrich_mcp_server`` calls ``load_model()`` when nothing is resident,
        so registering a connection used to be able to pull a multi-GB model
        into memory before the API could answer.
        """
        project = _project()
        svc = _svc(project)
        with patch("smartloop.api.services.model_service.LocalMCPClient"):
            ModelService.add_local_mcp_server(
                svc, command="uvx", args=["--from", "pkg", "pkg"], project_id="p1",
            )
        ModelService.add_remote_mcp_server(
            svc, name="notion", server_url="https://mcp.notion.com/mcp",
            project_id="p1",
        )
        svc.load_model.assert_not_called()
        svc._get_large_llm.assert_not_called()
        svc.get_llm.assert_not_called()
        svc._enrich_mcp_server.assert_not_called()

    def test_remote_registration_defers_enrichment_and_indexing(self):
        project = _project()
        svc = _svc(project)
        result = ModelService.add_remote_mcp_server(
            svc, name="notion", server_url="https://mcp.notion.com/mcp",
            tools=[{"name": "search", "description": "d"}], project_id="p1",
        )
        svc._enrich_mcp_server.assert_not_called()
        svc._index_mcp_tools.assert_not_called()
        svc._queue_mcp_provisioning.assert_called_once()
        assert result["status"] == "pending"

    def test_registration_response_carries_the_state_clients_poll(self):
        project = _project()
        svc = _svc(project)
        ModelService.add_remote_mcp_server(
            svc, name="notion", server_url="https://mcp.notion.com/mcp",
            project_id="p1",
        )
        result = ModelService._mcp_server_to_dict(project.mcp_servers[0])
        assert result["status"] == "pending"
        assert result["last_error"] is None
        assert "last_crawl_at" in result


class TestProvisioningPass:
    def test_successful_pass_discovers_enriches_indexes_and_crawls(self):
        project = _project()
        svc = _svc(project)
        ModelService.add_remote_mcp_server(
            svc, name="notion", server_url="https://mcp.notion.com/mcp",
            project_id="p1",
        )
        svc._discover_mcp_tools.side_effect = lambda s: setattr(s, "tools", _TOOLS)
        svc._crawl_mcp_server.return_value = {"indexed": 3}

        result = ModelService._provision_mcp_server(svc, "p1", project.mcp_servers[0].id)

        server = project.mcp_servers[0]
        assert server.status == MCPServerStatus.READY
        assert server.last_error is None
        svc._enrich_mcp_server.assert_called_once()
        svc._index_mcp_tools.assert_called_once()
        svc._crawl_mcp_server.assert_called_once()
        assert result["status"] == "ready"

    def test_failed_discovery_is_recorded_on_the_server_not_raised(self):
        project = _project()
        svc = _svc(project)
        ModelService.add_local_mcp_server(
            svc, command="uvx", args=["--from", "broken", "broken"], project_id="p1",
        )
        svc._discover_mcp_tools.side_effect = MCPDiscoveryError("Package not found.")

        result = ModelService._provision_mcp_server(svc, "p1", project.mcp_servers[0].id)

        server = project.mcp_servers[0]
        assert server.status == MCPServerStatus.FAILED
        assert server.last_error == "Package not found."
        assert result["status"] == "failed"
        # A failed discovery must not leave half-provisioned state behind.
        svc._index_mcp_tools.assert_not_called()
        svc._crawl_mcp_server.assert_not_called()

    def test_discovery_is_skipped_when_registration_supplied_the_tools(self):
        project = _project()
        svc = _svc(project)
        ModelService.add_remote_mcp_server(
            svc, name="notion", server_url="https://mcp.notion.com/mcp",
            tools=[{"name": "search", "description": "d"}], project_id="p1",
        )
        ModelService._provision_mcp_server(svc, "p1", project.mcp_servers[0].id)
        svc._discover_mcp_tools.assert_not_called()
        svc._index_mcp_tools.assert_called_once()

    def test_a_deleted_server_is_a_no_op(self):
        project = _project()
        svc = _svc(project)
        result = ModelService._provision_mcp_server(svc, "p1", "gone")
        assert result["status"] == "skipped"

    def test_retry_resets_the_server_to_pending_and_requeues(self):
        project = _project()
        svc = _svc(project)
        ModelService.add_local_mcp_server(
            svc, command="uvx", args=["--from", "pkg", "pkg"], project_id="p1",
        )
        server = project.mcp_servers[0]
        server.status = MCPServerStatus.FAILED
        server.last_error = "boom"

        result = ModelService.retry_mcp_server(svc, server.id, project_id="p1")

        assert server.status == MCPServerStatus.PENDING
        assert server.last_error is None
        assert result["status"] == "pending"
        assert svc._queue_mcp_provisioning.call_count == 2


class TestRemoteDiscovery:
    def test_remote_tools_are_discovered_in_the_background_pass(self):
        server = SimpleNamespace(
            id="s1", name="notion", tools=[], description=None, server_info=None,
            server_type=MagicMock(), oauth_credentials=None,
            remote_config=SimpleNamespace(
                server_url="https://mcp.notion.com/mcp", token="tok",
            ),
        )
        svc = MagicMock()
        svc._friendly_discovery_error = ModelService._friendly_discovery_error
        client = MagicMock()
        client.connect_and_discover_tools.return_value = MCPConnectResult(
            success=True, tools=_TOOLS,
        )
        with patch("smartloop.mcp.MCPClient") as client_cls:
            client_cls.return_value.__enter__.return_value = client
            ModelService._discover_mcp_tools(svc, server)

        assert [t.name for t in server.tools] == ["search"]
        assert client.connect_and_discover_tools.call_args.kwargs["access_token"] == "tok"

    def test_remote_discovery_failure_raises_a_readable_reason(self):
        server = SimpleNamespace(
            id="s1", name="notion", tools=[], description=None, server_info=None,
            server_type=MagicMock(), oauth_credentials=None,
            remote_config=SimpleNamespace(server_url="https://x.test/mcp", token=None),
        )
        svc = MagicMock()
        svc._friendly_discovery_error = ModelService._friendly_discovery_error
        client = MagicMock()
        client.connect_and_discover_tools.return_value = MCPConnectResult(
            success=False, error="Connection refused",
        )
        with patch("smartloop.mcp.MCPClient") as client_cls:
            client_cls.return_value.__enter__.return_value = client
            with pytest.raises(MCPDiscoveryError):
                ModelService._discover_mcp_tools(svc, server)


class TestMCPTaskSchedule:
    def test_first_tick_runs_a_crawl(self):
        """The start cycle: whatever was indexed last session gets refreshed."""
        done = threading.Event()
        task = MCPTask(
            logger=MagicMock(),
            crawler=lambda: (done.set(), {"servers": {}})[1],
            crawl_interval=3600,
        )
        try:
            task.start_loop()
            assert done.wait(timeout=5)
        finally:
            task.stop()

    def test_interval_gates_later_crawls(self):
        task = MCPTask(logger=MagicMock(), crawler=lambda: {}, crawl_interval=3600)
        task._crawl_requested = False
        task._last_crawl_cycle = 100.0
        assert task._crawl_due(200.0) is False
        assert task._crawl_due(3701.0) is True

    def test_zero_interval_disables_the_periodic_crawl(self):
        task = MCPTask(logger=MagicMock(), crawler=lambda: {}, crawl_interval=0)
        task._crawl_requested = False
        assert task._crawl_due(1_000_000.0) is False
        # An explicit request still runs — "disabled" means no cadence, not
        # that a client can no longer ask for a refresh.
        task._crawl_requested = True
        assert task._crawl_due(1_000_000.0) is True

    def test_enqueued_server_is_provisioned_off_the_caller_thread(self):
        seen = []
        done = threading.Event()

        def provisioner(project_id, server_id):
            seen.append((project_id, server_id))
            done.set()
            return {}

        task = MCPTask(logger=MagicMock(), provisioner=provisioner, crawl_interval=3600)
        try:
            task.enqueue("p1", "s1")
            assert done.wait(timeout=5)
            assert seen == [("p1", "s1")]
        finally:
            task.stop()

    def test_pending_servers_from_a_previous_run_are_adopted(self):
        done = threading.Event()
        seen = []

        def provisioner(project_id, server_id):
            seen.append(server_id)
            done.set()
            return {}

        task = MCPTask(
            logger=MagicMock(),
            provisioner=provisioner,
            pending_provider=lambda: [("p1", "left-pending")],
            crawl_interval=3600,
        )
        try:
            task.start_loop()
            assert done.wait(timeout=5)
            assert seen == ["left-pending"]
        finally:
            task.stop()

    def test_status_reports_the_cycle_for_health(self):
        task = MCPTask(logger=MagicMock(), crawler=lambda: {}, crawl_interval=900)
        status = task.mcp_status
        assert status["crawl_interval_seconds"] == 900
        assert status["queued"] == 0
        assert status["crawling"] is False
