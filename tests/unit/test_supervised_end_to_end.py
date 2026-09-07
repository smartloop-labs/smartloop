"""A supervisor, real project agent processes, and the client contract intact.

Everything here runs for real: the supervisor binds a port, spawns
``smartloop.api.project_agent`` as separate OS processes, and requests are
answered by the child that owns the project. What it does not do is load a
model — ``slp_api_preload_model=0`` keeps the device out of it, which is the
one thing about a live agent a test cannot have.

This is the phase's claim under test: same port, same responses, one process
per project.
"""

import json
import os
import sys
import threading
import time

import httpx
import pytest

from smartloop.supervisor.pool import ProjectAgentPool
from smartloop.supervisor.registry import AgentRegistry, process_alive


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A workspace with two projects and no models."""
    workspace_id = "ws-e2e"
    workspace_dir = tmp_path / workspace_id
    projects = []
    for project_id in ("alpha", "beta"):
        project_dir = workspace_dir / project_id
        project_dir.mkdir(parents=True)
        projects.append({
            "id": project_id,
            "name": project_id.title(),
            "project_dir": str(project_dir),
            "current": project_id == "alpha",
            "system": False,
        })
    (tmp_path / "config.json").write_text(json.dumps({
        "id": "",
        "workspace": {
            "id": workspace_id,
            "workspace_dir": str(workspace_dir),
            "models_dir": str(workspace_dir / "models"),
            "models": [],
            "projects": projects,
        },
    }))
    return tmp_path


@pytest.fixture
def child_env(home):
    """Environment for a real project agent: this workspace, no model load."""
    env = dict(os.environ)
    env["SLP_HOME"] = str(home)
    env["SLP_API_PRELOAD_MODEL"] = "0"
    env["SLP_LOG_LEVEL"] = "warning"
    env.pop("SLP_SUPERVISOR", None)
    env.pop("SLP_MODEL_HOST", None)
    return env


@pytest.fixture
def pool(home, child_env):
    def project_dir_for(project_id):
        directory = home / "ws-e2e" / project_id
        return str(directory) if directory.exists() else None

    class _Pool(ProjectAgentPool):
        def _environment(self, project_id):
            env = dict(child_env)
            env["SLP_PROJECT_ID"] = project_id
            return env

    made = _Pool(
        registry=AgentRegistry(str(home)),
        project_dir_for=project_dir_for,
        idle_ttl=0,
        start_timeout=120,
        command_for=lambda project_id: [
            sys.executable, "-m", "smartloop.api.project_agent",
            "--project", project_id, "--log-level", "warning",
        ],
    )
    yield made
    made.stop_reaper()
    made.stop_all()


class TestRealProjectAgents:
    def test_an_agent_starts_and_answers(self, pool):
        record = pool.ensure("alpha")

        response = httpx.get(f"{record.base_url}/health", timeout=30)

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_each_project_gets_its_own_process(self, pool):
        alpha = pool.ensure("alpha")
        beta = pool.ensure("beta")

        assert alpha.pid != beta.pid
        assert process_alive(alpha.pid) and process_alive(beta.pid)

    def test_an_agent_serves_the_project_it_was_spawned_for(self, pool):
        """The pointer is gone: the child *is* the project, fixed at spawn."""
        alpha = pool.ensure("alpha")
        beta = pool.ensure("beta")

        alpha_docs = httpx.get(
            f"{alpha.base_url}/v1/projects/alpha/documents", timeout=30,
        )
        beta_docs = httpx.get(
            f"{beta.base_url}/v1/projects/beta/documents", timeout=30,
        )

        assert alpha_docs.status_code == 200
        assert beta_docs.status_code == 200

    def test_the_unselected_project_gets_an_agent_of_its_own(self, pool):
        """``alpha`` is the selected project; ``beta`` still gets served.

        A single server had to move the selection to answer for beta, which is
        what made two concurrent projects race. A child is pinned to beta at
        spawn and never touches the flag — which still reads ``alpha``, because
        the selection is the UI's and nothing here changed it.
        """
        beta = pool.ensure("beta")

        assert httpx.get(f"{beta.base_url}/health", timeout=30).status_code == 200

        projects = httpx.get(f"{beta.base_url}/v1/projects", timeout=30).json()
        selected = next(
            (p for p in projects.get("projects", []) if p.get("current")), None,
        )
        assert selected is not None and selected["id"] == "alpha"

    def test_killing_one_agent_leaves_the_other_serving(self, pool):
        """The blast radius: one project's crash is one project's problem."""
        alpha = pool.ensure("alpha")
        beta = pool.ensure("beta")

        os.kill(alpha.pid, 9)
        for _ in range(200):
            if pool._running("alpha") is None:
                break
            time.sleep(0.05)

        assert httpx.get(f"{beta.base_url}/health", timeout=30).status_code == 200

    def test_a_crashed_project_comes_back_on_the_next_request(self, pool):
        first = pool.ensure("alpha")
        os.kill(first.pid, 9)
        for _ in range(200):
            if pool._running("alpha") is None:
                break
            time.sleep(0.05)
        pool.sweep()
        pool.clear_backoff("alpha")

        second = pool.ensure("alpha")

        assert second.pid != first.pid
        assert httpx.get(f"{second.base_url}/health", timeout=30).status_code == 200

    def test_the_agent_publishes_its_pid_and_port(self, pool, home):
        from smartloop.supervisor.registry import read_project_agent

        record = pool.ensure("alpha")

        published = read_project_agent(str(home / "ws-e2e" / "alpha"))
        assert published == (record.pid, record.port)

    def test_the_agent_binds_only_loopback(self, pool):
        record = pool.ensure("alpha")
        assert record.base_url.startswith("http://127.0.0.1:")


class TestTheClientContractIsUnchanged:
    """Same port, same responses — the client should not be able to tell."""

    @pytest.fixture
    def supervisor(self, pool, home, monkeypatch):
        """A supervisor app on a real port, routing to the real children."""
        import uvicorn
        from fastapi import FastAPI

        from smartloop.supervisor.routing import ProjectRoutingMiddleware

        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"status": "healthy", "served_by": "supervisor"}

        @app.get("/v1/projects/{project_id}/documents")
        async def documents(project_id: str):  # pragma: no cover - never reached
            return {"served_by": "supervisor"}

        app.add_middleware(ProjectRoutingMiddleware, pool_provider=lambda: pool)

        config = uvicorn.Config(app, host="127.0.0.1", port=0,
                                log_level="warning", log_config=None)
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        for _ in range(300):
            if server.started and server.servers:
                break
            time.sleep(0.05)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}"
        server.should_exit = True
        thread.join(timeout=10)

    def test_a_project_request_is_answered_by_its_child(self, supervisor, pool):
        response = httpx.get(
            f"{supervisor}/v1/projects/alpha/documents", timeout=60,
        )

        assert response.status_code == 200
        assert response.json().get("served_by") != "supervisor"
        assert pool.registry.get("alpha") is not None

    def test_the_first_request_is_what_starts_the_project(self, supervisor, pool):
        assert pool.registry.get("beta") is None

        httpx.get(f"{supervisor}/v1/projects/beta/documents", timeout=60)

        assert pool.registry.get("beta") is not None

    def test_workspace_level_requests_never_start_anything(self, supervisor, pool):
        httpx.get(f"{supervisor}/health", timeout=30)

        assert pool.registry.all() == []
