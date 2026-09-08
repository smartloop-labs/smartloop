"""Agents that are running but in nobody's registry get taken back.

The registry is how a supervisor finds the children an earlier one left, so a
child missing from it is invisible: nothing routes to it, nothing stops it, and
``slp agent status`` does not list it.  It still holds the project's port, its
task queue and its MCP children.  A project directory describes itself
(``agent.pid``, ``agent.port``) even when the registry does not, and that is
what this reads.
"""

import os
import subprocess
import sys
import textwrap
import time
import types

import httpx
import pytest

from smartloop.supervisor.registry import AgentRegistry, process_alive
from smartloop.supervisor.service import Supervisor

STRAY_AGENT = textwrap.dedent('''
    import http.server, os, socket, sys, threading

    project_dir = sys.argv[1]
    answer_health = sys.argv[2] == "healthy"

    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if answer_health else 503)
            self.send_header("content-length", "0")
            self.end_headers()

        def log_message(self, *a):
            pass

    if answer_health:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()

    os.makedirs(project_dir, exist_ok=True)
    with open(os.path.join(project_dir, "agent.pid"), "w") as fh:
        fh.write(str(os.getpid()))
    with open(os.path.join(project_dir, "agent.port"), "w") as fh:
        fh.write(str(port))

    threading.Event().wait()
''')


@pytest.fixture
def stray(tmp_path):
    """A live agent-shaped process beside a project directory."""
    script = tmp_path / "stray_agent.py"
    script.write_text(STRAY_AGENT)
    project_dir = tmp_path / "alpha"
    spawned = []

    def spawn(healthy: bool = True):
        process = subprocess.Popen(
            [sys.executable, str(script), str(project_dir),
             "healthy" if healthy else "wedged"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        spawned.append(process)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if (project_dir / "agent.port").exists():
                return process
            time.sleep(0.05)
        raise AssertionError("the stray never published a port")

    yield spawn, str(project_dir)
    for process in spawned:
        try:
            process.kill()
            process.wait(timeout=5)
        except Exception:
            pass


@pytest.fixture
def supervisor(tmp_path, monkeypatch, stray):
    """A supervisor over a workspace with one project, and an empty registry."""
    _, project_dir = stray
    monkeypatch.setenv("SLP_HOME", str(tmp_path))

    from smartloop.workspace import Workspace

    project = types.SimpleNamespace(id="alpha", project_dir=project_dir)
    monkeypatch.setattr(
        Workspace, "load",
        classmethod(lambda cls: types.SimpleNamespace(projects=[project])),
    )
    made = Supervisor(home_dir=str(tmp_path))
    made.registry = AgentRegistry(str(tmp_path))
    return made


class TestReclaimingStrays:
    def test_a_healthy_stray_is_adopted_rather_than_killed(self, supervisor, stray):
        """It is a warm process already serving the right project."""
        spawn, _ = stray
        process = spawn(healthy=True)

        reclaimed = supervisor._reclaim_strays()

        assert reclaimed == ["alpha"]
        assert supervisor.registry.get("alpha").pid == process.pid
        assert process_alive(process.pid)

    def test_an_adopted_stray_is_routable(self, supervisor, stray):
        spawn, _ = stray
        spawn(healthy=True)

        supervisor._reclaim_strays()

        record = supervisor.registry.get("alpha")
        assert httpx.get(f"{record.base_url}/health", timeout=5).status_code == 200

    def test_a_stray_that_does_not_answer_is_stopped(self, supervisor, stray):
        """Nothing is listening, so there is nothing to adopt."""
        spawn, _ = stray
        process = spawn(healthy=False)

        reclaimed = supervisor._reclaim_strays()

        assert reclaimed == []
        assert not process_alive(process.pid)

    def test_an_agent_the_registry_already_knows_is_left_alone(
        self, supervisor, stray,
    ):
        from smartloop.supervisor.registry import AgentRecord, read_project_agent

        spawn, project_dir = stray
        process = spawn(healthy=True)
        _, port = read_project_agent(project_dir)
        supervisor.registry.add(AgentRecord("alpha", process.pid, port))

        assert supervisor._reclaim_strays() == []
        assert process_alive(process.pid)

    def test_a_pid_file_outliving_its_process_is_ignored(self, supervisor, stray):
        """The normal result of a crash — not a stray, just a stale file."""
        spawn, project_dir = stray
        process = spawn(healthy=True)
        process.kill()
        process.wait(timeout=5)

        assert supervisor._reclaim_strays() == []
        assert supervisor.registry.get("alpha") is None

    def test_a_project_with_no_agent_files_is_ignored(self, supervisor):
        assert supervisor._reclaim_strays() == []

    def test_no_workspace_is_not_an_error(self, supervisor, monkeypatch):
        from smartloop.workspace import Workspace

        monkeypatch.setattr(Workspace, "load", classmethod(lambda cls: None))

        assert supervisor._reclaim_strays() == []
