"""Registering an MCP server must not block on discovery or the LLM.

Registration used to launch the server process and call the LLM twice for
description/tags/summary, all in front of the client's request.  It now persists the connection and hands the rest to the MCP
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

    def test_local_registration_makes_no_llm_call(self):
        project = _project()
        svc = _svc(project)
        with patch("smartloop.api.services.model_service.LocalMCPClient"):
            ModelService.add_local_mcp_server(
                svc, command="uvx", args=["--from", "pkg", "pkg"], project_id="p1",
            )
        svc._enrich_mcp_server.assert_not_called()

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

    def test_remote_registration_defers_enrichment(self):
        project = _project()
        svc = _svc(project)
        result = ModelService.add_remote_mcp_server(
            svc, name="notion", server_url="https://mcp.notion.com/mcp",
            tools=[{"name": "search", "description": "d"}], project_id="p1",
        )
        svc._enrich_mcp_server.assert_not_called()
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
        assert result["tools"] == []


class TestProvisioningPass:
    def test_successful_pass_discovers_and_enriches(self):
        project = _project()
        svc = _svc(project)
        ModelService.add_remote_mcp_server(
            svc, name="notion", server_url="https://mcp.notion.com/mcp",
            project_id="p1",
        )
        svc._discover_mcp_tools.side_effect = lambda s: setattr(s, "tools", _TOOLS)

        result = ModelService._provision_mcp_server(svc, "p1", project.mcp_servers[0].id)

        server = project.mcp_servers[0]
        assert server.status == MCPServerStatus.READY
        assert server.last_error is None
        svc._enrich_mcp_server.assert_called_once()
        # Discovery is the whole catalog: the tools it recorded are what
        # MCPAgent resolves a step against, so there is nothing to index and
        # nothing to crawl.
        assert server.tools == _TOOLS
        assert result["status"] == "ready"
        assert result["tools"] == 1

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
        svc._enrich_mcp_server.assert_not_called()

    def test_discovery_is_skipped_when_registration_supplied_the_tools(self):
        project = _project()
        svc = _svc(project)
        ModelService.add_remote_mcp_server(
            svc, name="notion", server_url="https://mcp.notion.com/mcp",
            tools=[{"name": "search", "description": "d"}], project_id="p1",
        )
        ModelService._provision_mcp_server(svc, "p1", project.mcp_servers[0].id)
        svc._discover_mcp_tools.assert_not_called()
        svc._enrich_mcp_server.assert_called_once()

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

    def test_re_indexing_a_healthy_server_still_requeues_it(self):
        # The "re-index" button in the connections list: nothing is wrong with
        # the server, the user just wants its content walked again.
        project = _project()
        svc = _svc(project)
        ModelService.add_remote_mcp_server(
            svc, name="notion", server_url="https://mcp.notion.com/mcp",
            project_id="p1",
        )
        server = project.mcp_servers[0]
        server.status = MCPServerStatus.READY

        ModelService.retry_mcp_server(svc, server.id, project_id="p1")

        # Pending is what the card polls to show the pass as running.
        assert server.status == MCPServerStatus.PENDING
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


class TestMCPTaskQueue:
    """The task owns provisioning and nothing else — the periodic content
    crawl it used to run is gone, so a session starts without indexing."""

    def test_enqueued_server_is_provisioned_off_the_caller_thread(self):
        seen = []
        done = threading.Event()

        def provisioner(project_id, server_id):
            seen.append((project_id, server_id))
            done.set()
            return {}

        task = MCPTask(logger=MagicMock(), provisioner=provisioner)
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
        )
        try:
            task.start_loop()
            assert done.wait(timeout=5)
            assert seen == ["left-pending"]
        finally:
            task.stop()

    def test_status_reports_the_queue_for_health(self):
        status = MCPTask(logger=MagicMock()).mcp_status
        assert status == {"queued": 0, "provisioning": 0}


class TestIdleSignal:
    """What tells background indexing to stay out of the user's way."""

    def _svc(self):
        svc = ModelService.__new__(ModelService)
        svc._inference_lock = threading.Semaphore(1)
        svc._inference_active = 0
        svc._inference_state_lock = threading.Lock()
        svc._last_inference_at = 0.0
        return svc

    def test_a_fresh_session_is_idle(self):
        assert ModelService.idle_seconds(self._svc()) == float("inf")

    def test_holding_an_inference_slot_reads_as_busy(self):
        svc = self._svc()
        with ModelService._inference_slot(svc):
            assert ModelService.idle_seconds(svc) == 0.0

    def test_idle_time_starts_when_the_answer_finishes(self):
        svc = self._svc()
        with ModelService._inference_slot(svc):
            pass
        idle = ModelService.idle_seconds(svc)
        assert 0.0 <= idle < 5.0

    def test_a_raising_inference_still_releases_the_slot(self):
        # Otherwise one failed generation leaves the app "busy" forever and the
        # content crawl never runs again.
        svc = self._svc()
        try:
            with ModelService._inference_slot(svc):
                raise RuntimeError("generation failed")
        except RuntimeError:
            pass
        assert ModelService.idle_seconds(svc) != 0.0
