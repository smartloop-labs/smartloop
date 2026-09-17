"""The payoff: one project's problem stays one project's problem.

Separate processes contain a *crash*. A ceiling contains a leak, which would
otherwise still take the machine down with every project on it. And a log per
project makes "which of these lines is the project I am debugging" stop being a
question.
"""

import os
import sys
import time

import httpx
import pytest

from smartloop.supervisor.pool import ProjectAgentPool, process_rss
from smartloop.supervisor.registry import AgentRegistry, process_alive
from smartloop.supervisor.service import (
    Supervisor, model_host_enabled, supervision_enabled,
)

from .test_supervisor_pool import STUB_AGENT


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "ws").mkdir()
    stub = tmp_path / "stub_agent.py"
    stub.write_text(STUB_AGENT)

    def project_dir_for(project_id):
        return str(tmp_path / "ws" / project_id)

    return {"home": str(tmp_path), "stub": str(stub),
            "project_dir_for": project_dir_for}


def _pool(workspace, **kwargs) -> ProjectAgentPool:
    return ProjectAgentPool(
        registry=AgentRegistry(workspace["home"]),
        project_dir_for=workspace["project_dir_for"],
        idle_ttl=0,
        start_timeout=30,
        command_for=lambda project_id: [
            sys.executable, workspace["stub"],
            workspace["project_dir_for"](project_id),
        ],
        **kwargs,
    )


class TestMemoryCeilings:
    def test_no_ceiling_by_default(self, workspace):
        pool = _pool(workspace)
        try:
            pool.ensure("alpha")
            assert pool.sweep()["over_limit"] == []
        finally:
            pool.stop_all()

    def test_a_child_over_the_ceiling_is_stopped(self, workspace):
        # One byte: every real process is over it, which is the point — the
        # policy is under test, not the arithmetic of a particular workload.
        pool = _pool(workspace, memory_limit_bytes=1)
        try:
            record = pool.ensure("alpha")

            assert pool.sweep()["over_limit"] == ["alpha"]

            for _ in range(100):
                if not process_alive(record.pid):
                    break
                time.sleep(0.05)
            assert pool.registry.get("alpha") is None
        finally:
            pool.stop_all()

    def test_a_child_under_the_ceiling_is_left_alone(self, workspace):
        pool = _pool(workspace, memory_limit_bytes=64 * 1024 ** 3)  # 64 GB
        try:
            pool.ensure("alpha")
            assert pool.sweep()["over_limit"] == []
            assert pool.registry.get("alpha") is not None
        finally:
            pool.stop_all()

    def test_stopping_one_over_the_ceiling_leaves_the_others_serving(self, workspace):
        """The difference from a single process going OOM."""
        pool = _pool(workspace)
        try:
            alpha = pool.ensure("alpha")
            beta = pool.ensure("beta")

            # Put only alpha over the line, by pretending beta is weightless.
            real_rss = process_rss
            alpha_pid = alpha.pid

            def fake_rss(pid):
                return 10 ** 12 if pid == alpha_pid else 1

            import smartloop.supervisor.pool as pool_module
            pool_module.process_rss = fake_rss
            pool.memory_limit_bytes = 1024
            try:
                assert pool.sweep()["over_limit"] == ["alpha"]
            finally:
                pool_module.process_rss = real_rss

            assert process_alive(beta.pid)
            assert httpx.get(f"{beta.base_url}/health", timeout=5).status_code == 200
        finally:
            pool.stop_all()

    def test_the_project_comes_back_on_its_next_request(self, workspace):
        pool = _pool(workspace, memory_limit_bytes=1)
        try:
            first = pool.ensure("alpha")
            pool.sweep()
            pool.clear_backoff("alpha")

            second = pool.ensure("alpha")

            assert second.pid != first.pid
            assert httpx.get(f"{second.base_url}/health", timeout=5).status_code == 200
        finally:
            pool.stop_all()

    def test_rss_of_a_process_that_is_gone_reads_as_none(self):
        assert process_rss(4194305) is None
        assert process_rss(None) is None

    def test_rss_of_this_process_is_a_real_number(self):
        assert process_rss(os.getpid()) > 0


class TestTheSettings:
    def test_the_ceiling_comes_from_settings(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SLP_AGENT_MEMORY_LIMIT_MB", "512")
        monkeypatch.setenv("SLP_HOME", str(tmp_path))

        supervisor = Supervisor(home_dir=str(tmp_path))

        assert supervisor.memory_limit_bytes == 512 * 1024 * 1024

    def test_the_idle_ttl_comes_from_settings(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SLP_AGENT_IDLE_TTL_S", "60")

        supervisor = Supervisor(home_dir=str(tmp_path))

        assert supervisor.idle_ttl == 60.0

    def test_explicit_values_win_over_settings(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SLP_AGENT_MEMORY_LIMIT_MB", "512")

        supervisor = Supervisor(home_dir=str(tmp_path), idle_ttl=5,
                                memory_limit_mb=16)

        assert supervisor.idle_ttl == 5
        assert supervisor.memory_limit_bytes == 16 * 1024 * 1024

    def test_no_ceiling_is_reported_as_none(self, tmp_path):
        supervisor = Supervisor(home_dir=str(tmp_path), memory_limit_mb=0)
        assert supervisor.status()["limits"]["memory_limit_bytes"] is None


class TestTheSwitches:
    @pytest.fixture(autouse=True)
    def _unpinned(self):
        """These assert the environment's effect; a pin overrides it entirely."""
        from smartloop.workspace.context import pin_project

        pin_project(None)
        yield
        pin_project(None)

    def test_supervision_is_on_by_default(self, monkeypatch):
        monkeypatch.delenv("SLP_SUPERVISOR", raising=False)
        assert supervision_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE"])
    def test_supervision_can_be_turned_off(self, monkeypatch, value):
        monkeypatch.setenv("SLP_SUPERVISOR", value)
        assert supervision_enabled() is False

    def test_a_meaningless_value_leaves_it_on(self, monkeypatch):
        """Only an explicit opt-out counts; anything else keeps the default."""
        monkeypatch.setenv("SLP_SUPERVISOR", "maybe")
        assert supervision_enabled() is True

    def test_a_pinned_process_never_supervises(self, monkeypatch):
        """A child inherits the environment; the pin is what stops a grandchild."""
        from smartloop.workspace.context import pin_project

        monkeypatch.setenv("SLP_SUPERVISOR", "1")
        pin_project("alpha")

        assert supervision_enabled() is False

    def test_the_model_host_is_on_by_default_when_supervising(self, monkeypatch):
        monkeypatch.delenv("SLP_MODEL_HOST_DISABLE", raising=False)
        assert model_host_enabled() is True

    def test_the_model_host_can_never_disable_the_grpc_path(self, monkeypatch):
        monkeypatch.setenv("SLP_MODEL_HOST_DISABLE", "1")
        assert model_host_enabled() is True


class TestStatusReporting:
    def test_it_reports_every_agent(self, workspace, tmp_path):
        pool = _pool(workspace)
        supervisor = Supervisor(home_dir=workspace["home"])
        supervisor.pool = pool
        supervisor.registry = pool.registry
        try:
            pool.ensure("alpha")
            pool.ensure("beta")

            status = supervisor.status()

            assert {a["project_id"] for a in status["agents"]} == {"alpha", "beta"}
            for agent in status["agents"]:
                assert agent["alive"] is True
                assert agent["rss_bytes"] > 0
                assert agent["port"] > 0
        finally:
            pool.stop_all()

    def test_an_empty_workspace_reports_no_agents(self, tmp_path):
        assert Supervisor(home_dir=str(tmp_path)).status()["agents"] == []


class TestPerProjectLogs:
    def test_a_child_writes_its_own_log(self, tmp_path):
        """One shared server.log would interleave every project's trace."""
        from smartloop.api.project_agent import (
            AGENT_LOG_FILE, _install_project_logging,
        )
        from smartloop.utils.logging_setup import shutdown_logging
        import logging

        project_dir = tmp_path / "alpha"
        project_dir.mkdir()

        listener = _install_project_logging(str(project_dir), "info")
        try:
            logging.getLogger("smartloop.test").warning("a line for alpha only")
            time.sleep(0.3)
        finally:
            shutdown_logging(listener)

        log = project_dir / AGENT_LOG_FILE
        assert log.exists()
        assert "a line for alpha only" in log.read_text()

    def test_two_projects_write_to_two_files(self, tmp_path):
        from smartloop.api.project_agent import (
            AGENT_LOG_FILE, _install_project_logging,
        )
        from smartloop.utils.logging_setup import shutdown_logging
        import logging

        for project_id in ("alpha", "beta"):
            project_dir = tmp_path / project_id
            project_dir.mkdir()
            listener = _install_project_logging(str(project_dir), "info")
            try:
                logging.getLogger("smartloop.test").warning("line for %s", project_id)
                time.sleep(0.3)
            finally:
                shutdown_logging(listener)

        alpha = (tmp_path / "alpha" / AGENT_LOG_FILE).read_text()
        beta = (tmp_path / "beta" / AGENT_LOG_FILE).read_text()
        assert "line for alpha" in alpha and "line for beta" not in alpha
        assert "line for beta" in beta and "line for alpha" not in beta

    def test_a_log_that_cannot_be_opened_is_not_fatal(self, tmp_path):
        """A project must still be served when its log file cannot be written."""
        from smartloop.api.project_agent import _install_project_logging

        # A path whose parent is a file, so opening the log fails.
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("")

        assert _install_project_logging(str(blocker / "nested"), "info") is None
