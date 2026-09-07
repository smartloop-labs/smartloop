"""Starting, watching and reaping a process per project.

Real child processes: the parts worth testing here — publishing a port, going
healthy, dying, being restarted with backoff, being stopped when idle — are
exactly the parts a mock would assume away. The children stand in for a project
agent without loading a model, which is the only thing about a real one that
makes it unusable in a test.
"""

import os
import sys
import textwrap
import time

import httpx
import pytest

from smartloop.supervisor.pool import (
    _BACKOFF_MAX, _BACKOFF_START, ChildStartError, ProjectAgentPool,
)
from smartloop.supervisor.registry import AgentRegistry, process_alive

#: A stand-in project agent: publishes its port beside the project the way the
#: real one does, then answers /health until it is stopped.
STUB_AGENT = textwrap.dedent('''
    import http.server, os, socket, sys, threading

    project_dir = sys.argv[1]

    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"status": "healthy", "path": "' + self.path.encode() + b'"}'
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length)
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

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
def workspace(tmp_path):
    """A home with project directories, and the lookup the pool is given."""
    (tmp_path / "ws").mkdir()
    stub = tmp_path / "stub_agent.py"
    stub.write_text(STUB_AGENT)

    def project_dir_for(project_id):
        return str(tmp_path / "ws" / project_id) if project_id != "unknown" else None

    return {
        "home": str(tmp_path),
        "stub": str(stub),
        "project_dir_for": project_dir_for,
    }


@pytest.fixture
def pool(workspace):
    made = ProjectAgentPool(
        registry=AgentRegistry(workspace["home"]),
        project_dir_for=workspace["project_dir_for"],
        idle_ttl=0,  # off unless a test asks for it
        start_timeout=30,
        command_for=lambda project_id: [
            sys.executable, workspace["stub"],
            workspace["project_dir_for"](project_id) or "",
        ],
    )
    yield made
    made.stop_reaper()
    made.stop_all()


class TestStartingOnDemand:
    def test_the_first_request_starts_the_project(self, pool):
        record = pool.ensure("alpha")

        assert record.project_id == "alpha"
        assert process_alive(record.pid)
        assert httpx.get(f"{record.base_url}/health", timeout=5).status_code == 200

    def test_a_second_request_reuses_the_running_agent(self, pool):
        first = pool.ensure("alpha")
        second = pool.ensure("alpha")

        assert (second.pid, second.port) == (first.pid, first.port)

    def test_two_projects_get_two_processes(self, pool):
        alpha = pool.ensure("alpha")
        beta = pool.ensure("beta")

        assert alpha.pid != beta.pid
        assert alpha.port != beta.port

    def test_concurrent_cold_starts_produce_one_child(self, pool):
        """Two requests arriving at once on a cold project is the normal case."""
        import threading

        start = threading.Barrier(4, timeout=20)
        seen: list = []
        errors: list = []

        def request():
            try:
                start.wait()
                seen.append(pool.ensure("alpha").pid)
            except Exception as exc:  # pragma: no cover - failure detail
                errors.append(exc)

        threads = [threading.Thread(target=request) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=45)

        assert not errors
        assert len(set(seen)) == 1

    def test_the_child_is_told_which_project_it_is(self, pool):
        env = pool._environment("alpha")
        assert env["SLP_PROJECT_ID"] == "alpha"

    def test_a_child_is_told_explicitly_not_to_supervise(self, pool, monkeypatch):
        """Unsetting would not do: absent means "on" now, so it must say "0"."""
        monkeypatch.setenv("SLP_SUPERVISOR", "1")
        assert pool._environment("alpha")["SLP_SUPERVISOR"] == "0"

    def test_a_child_is_told_not_to_supervise_even_when_nothing_was_set(
        self, pool, monkeypatch,
    ):
        monkeypatch.delenv("SLP_SUPERVISOR", raising=False)
        assert pool._environment("alpha")["SLP_SUPERVISOR"] == "0"

    def test_the_model_host_is_passed_down(self, workspace):
        pool = ProjectAgentPool(
            registry=AgentRegistry(workspace["home"]),
            project_dir_for=workspace["project_dir_for"],
            model_host_address="/tmp/model-host.sock",
        )
        assert pool._environment("alpha")["SLP_MODEL_HOST"] == "/tmp/model-host.sock"

    def test_a_child_with_a_host_does_not_load_its_own_weights(self, workspace):
        """N copies of the model in one device is what the host exists to avoid."""
        pool = ProjectAgentPool(
            registry=AgentRegistry(workspace["home"]),
            project_dir_for=workspace["project_dir_for"],
            model_host_address="/tmp/model-host.sock",
        )
        assert pool._environment("alpha")["SLP_API_PRELOAD_MODEL"] == "0"

    def test_a_child_without_a_host_loads_its_own(self, pool, monkeypatch):
        """No host — the --no-supervised style fallback — means local weights."""
        monkeypatch.delenv("SLP_API_PRELOAD_MODEL", raising=False)
        env = pool._environment("alpha")
        assert "SLP_MODEL_HOST" not in env
        assert "SLP_API_PRELOAD_MODEL" not in env

    def test_the_default_command_re_execs_this_interpreter(self, pool):
        command = pool._default_command("alpha")
        assert command[:3] == [sys.executable, "-m", "smartloop.api.project_agent"]
        assert command[-2:] == ["--project", "alpha"]


class TestTheRegistryTracksThem:
    def test_a_started_agent_is_recorded(self, pool):
        record = pool.ensure("alpha")
        assert pool.registry.get("alpha").pid == record.pid

    def test_the_project_directory_records_it_too(self, pool, workspace):
        from smartloop.supervisor.registry import read_project_agent

        record = pool.ensure("alpha")
        published = read_project_agent(workspace["project_dir_for"]("alpha"))
        assert published == (record.pid, record.port)


class TestFailingToStart:
    def test_a_project_with_no_directory_cannot_be_started(self, pool):
        with pytest.raises(ChildStartError, match="no directory"):
            pool.ensure("unknown")

    def test_a_child_that_exits_immediately_is_reported(self, workspace):
        pool = ProjectAgentPool(
            registry=AgentRegistry(workspace["home"]),
            project_dir_for=workspace["project_dir_for"],
            start_timeout=10,
            command_for=lambda _p: [sys.executable, "-c", "raise SystemExit(3)"],
        )
        with pytest.raises(ChildStartError, match="exited with code 3"):
            pool.ensure("alpha")

    def test_a_child_that_never_publishes_a_port_times_out(self, workspace):
        pool = ProjectAgentPool(
            registry=AgentRegistry(workspace["home"]),
            project_dir_for=workspace["project_dir_for"],
            start_timeout=1,
            command_for=lambda _p: [
                sys.executable, "-c", "import threading; threading.Event().wait()",
            ],
        )
        with pytest.raises(ChildStartError, match="did not publish a port"):
            pool.ensure("alpha")

    def test_an_unlaunchable_command_is_reported(self, workspace):
        pool = ProjectAgentPool(
            registry=AgentRegistry(workspace["home"]),
            project_dir_for=workspace["project_dir_for"],
            command_for=lambda _p: ["/definitely/not/a/binary"],
        )
        with pytest.raises(ChildStartError, match="could not spawn"):
            pool.ensure("alpha")


class TestStopping:
    def test_stopping_ends_the_process(self, pool):
        record = pool.ensure("alpha")

        assert pool.stop("alpha") is True

        for _ in range(100):
            if not process_alive(record.pid):
                break
            time.sleep(0.05)
        assert not process_alive(record.pid)
        assert pool.registry.get("alpha") is None

    def test_stopping_something_that_is_not_running_is_not_an_error(self, pool):
        assert pool.stop("alpha") is False

    def test_stopping_one_project_leaves_the_others_alone(self, pool):
        alpha = pool.ensure("alpha")
        beta = pool.ensure("beta")

        pool.stop("alpha")

        assert not process_alive(alpha.pid) or pool.registry.get("alpha") is None
        assert process_alive(beta.pid)
        assert httpx.get(f"{beta.base_url}/health", timeout=5).status_code == 200

    def test_stop_all_ends_every_child(self, pool):
        records = [pool.ensure("alpha"), pool.ensure("beta")]

        pool.stop_all()

        for record in records:
            for _ in range(100):
                if not process_alive(record.pid):
                    break
                time.sleep(0.05)
            assert not process_alive(record.pid)


class TestTheSweep:
    def test_an_idle_agent_is_stopped(self, pool):
        pool.idle_ttl = 60
        record = pool.ensure("alpha")
        record.last_used_at -= 120

        assert pool.sweep()["idled"] == ["alpha"]
        assert pool.registry.get("alpha") is None

    def test_a_busy_agent_is_left_alone(self, pool):
        pool.idle_ttl = 60
        pool.ensure("alpha")

        assert pool.sweep()["idled"] == []
        assert pool.registry.get("alpha") is not None

    def test_idle_stop_can_be_switched_off(self, pool):
        pool.idle_ttl = 0
        record = pool.ensure("alpha")
        record.last_used_at -= 100000

        assert pool.sweep()["idled"] == []

    def test_a_crashed_agent_is_noticed(self, pool):
        record = pool.ensure("alpha")
        os.kill(record.pid, 9)
        for _ in range(100):
            if not process_alive(record.pid):
                break
            time.sleep(0.05)

        assert pool.sweep()["crashed"] == ["alpha"]
        assert pool.registry.get("alpha") is None

    def test_a_crash_does_not_disturb_another_project(self, pool):
        """The blast radius the split exists to shrink."""
        alpha = pool.ensure("alpha")
        beta = pool.ensure("beta")
        os.kill(alpha.pid, 9)
        for _ in range(100):
            if not process_alive(alpha.pid):
                break
            time.sleep(0.05)

        pool.sweep()

        assert process_alive(beta.pid)
        assert httpx.get(f"{beta.base_url}/health", timeout=5).status_code == 200

    def test_a_crashed_project_starts_again_on_the_next_request(self, pool):
        first = pool.ensure("alpha")
        os.kill(first.pid, 9)
        for _ in range(100):
            if not process_alive(first.pid):
                break
            time.sleep(0.05)
        pool.sweep()
        pool.clear_backoff("alpha")  # skip the wait; the delay is tested below

        second = pool.ensure("alpha")

        assert second.pid != first.pid
        assert process_alive(second.pid)


class TestZombiesAreNotLiveAgents:
    """A child we spawned and never waited on stays signalable after it dies."""

    def test_a_killed_child_is_replaced_rather_than_reused(self, pool):
        first = pool.ensure("alpha")
        os.kill(first.pid, 9)
        for _ in range(100):
            try:
                os.waitpid(first.pid, os.WNOHANG)
            except ChildProcessError:
                pass
            if not process_alive(first.pid):
                break
            time.sleep(0.05)

        # The pid may still answer signal 0 as a zombie; the pool must decide
        # from the handle it owns, not from the pid, or it would forward this
        # project's requests to a port nothing is listening on.
        second = pool.ensure("alpha")

        assert second.pid != first.pid
        assert httpx.get(f"{second.base_url}/health", timeout=5).status_code == 200

    def test_running_reports_nothing_for_a_dead_child(self, pool):
        record = pool.ensure("alpha")
        os.kill(record.pid, 9)
        for _ in range(100):
            if pool._running("alpha") is None:
                break
            time.sleep(0.05)

        assert pool._running("alpha") is None


class TestRestartBackoff:
    def test_the_first_crash_sets_a_delay(self, pool):
        pool.note_exit("alpha", uptime=1.0)
        assert pool._backoff["alpha"] == _BACKOFF_START

    def test_repeated_crashes_back_off_further(self, pool):
        for _ in range(4):
            pool.note_exit("alpha", uptime=1.0)
        assert pool._backoff["alpha"] > _BACKOFF_START

    def test_the_delay_is_capped(self, pool):
        for _ in range(50):
            pool.note_exit("alpha", uptime=1.0)
        assert pool._backoff["alpha"] == _BACKOFF_MAX

    def test_a_child_that_ran_a_while_resets_the_backoff(self, pool):
        """A crash after two healthy minutes is not a crash loop."""
        for _ in range(5):
            pool.note_exit("alpha", uptime=1.0)
        pool.note_exit("alpha", uptime=300.0)
        assert pool._backoff["alpha"] == _BACKOFF_START

    def test_one_project_s_backoff_is_its_own(self, pool):
        pool.note_exit("alpha", uptime=1.0)
        assert "beta" not in pool._backoff
