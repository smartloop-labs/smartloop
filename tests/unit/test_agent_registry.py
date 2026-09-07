"""What is running, for a workspace that is now several processes.

``server.pid`` and ``server.port`` described a workspace when a workspace was
one process. With a process per project there is a set of them, and the
registry is what can answer which project is on which port — for routing, for
``slp agent status``, and for a supervisor adopting what an earlier one left.
"""

import json
import os

import pytest

from smartloop.supervisor.registry import (
    AGENT_PID_FILE, AGENT_PORT_FILE, AgentRecord, AgentRegistry,
    process_alive, read_project_agent,
)


@pytest.fixture
def registry(tmp_path) -> AgentRegistry:
    return AgentRegistry(str(tmp_path))


def _record(project_id="alpha", pid=None, port=41000) -> AgentRecord:
    return AgentRecord(project_id, pid if pid is not None else os.getpid(), port)


class TestLiveness:
    def test_this_process_is_alive(self):
        assert process_alive(os.getpid()) is True

    def test_a_pid_that_is_gone_is_not(self):
        # 2^22 is above the default pid_max on Linux and macOS, so nothing
        # holds it.
        assert process_alive(4194305) is False

    @pytest.mark.parametrize("pid", [None, 0, -1])
    def test_a_missing_pid_is_not_alive(self, pid):
        assert process_alive(pid) is False


class TestRecords:
    def test_it_knows_where_to_forward(self):
        assert _record(port=41234).base_url == "http://127.0.0.1:41234"

    def test_a_fresh_record_is_not_idle(self):
        assert _record().idle_seconds < 1.0

    def test_touching_resets_the_idle_clock(self):
        record = _record()
        record.last_used_at -= 500
        assert record.idle_seconds > 400
        record.touch()
        assert record.idle_seconds < 1.0

    def test_it_round_trips(self):
        original = _record()
        restored = AgentRecord.from_dict("alpha", original.to_dict())
        assert restored.pid == original.pid
        assert restored.port == original.port

    def test_a_malformed_record_is_rejected_not_raised(self):
        assert AgentRecord.from_dict("alpha", {"pid": "not-a-pid"}) is None
        assert AgentRecord.from_dict("alpha", {}) is None


class TestTheRegistry:
    def test_an_added_agent_can_be_found(self, registry):
        registry.add(_record())
        assert registry.get("alpha").port == 41000

    def test_an_unknown_project_has_none(self, registry):
        assert registry.get("nope") is None

    def test_live_excludes_the_dead(self, registry):
        registry.add(_record("alpha"))
        registry.add(_record("beta", pid=4194305))
        assert [r.project_id for r in registry.live()] == ["alpha"]

    def test_reaping_drops_the_dead_and_keeps_the_rest(self, registry):
        registry.add(_record("alpha"))
        registry.add(_record("beta", pid=4194305))

        assert registry.reap() == ["beta"]
        assert [r.project_id for r in registry.all()] == ["alpha"]

    def test_removing_forgets_it(self, registry):
        registry.add(_record())
        registry.remove("alpha")
        assert registry.get("alpha") is None


class TestPersistence:
    def test_it_survives_a_restart(self, registry, tmp_path):
        registry.set_supervisor(os.getpid(), 38540)
        registry.set_model_host(os.getpid(), "/tmp/model-host.sock")
        registry.add(_record())

        reloaded = AgentRegistry(str(tmp_path)).load()

        assert reloaded.supervisor == {"pid": os.getpid(), "port": 38540}
        assert reloaded.model_host["address"] == "/tmp/model-host.sock"
        assert reloaded.get("alpha").port == 41000

    def test_a_new_supervisor_does_not_adopt_dead_children(self, registry, tmp_path):
        """A pid file outliving its process is the normal result of a crash."""
        registry.add(_record("alpha"))
        registry.add(_record("beta", pid=4194305))

        reloaded = AgentRegistry(str(tmp_path)).load()

        assert reloaded.get("alpha") is not None
        assert reloaded.get("beta") is None

    def test_no_registry_file_is_not_an_error(self, tmp_path):
        assert AgentRegistry(str(tmp_path / "fresh")).load().all() == []

    def test_a_corrupt_registry_is_not_an_error(self, tmp_path):
        (tmp_path / "agents.json").write_text("{not json")
        assert AgentRegistry(str(tmp_path)).load().all() == []

    def test_the_file_is_readable_json(self, registry, tmp_path):
        registry.add(_record())
        saved = json.loads((tmp_path / "agents.json").read_text())
        assert saved["version"] == 1
        assert saved["agents"]["alpha"]["port"] == 41000


class TestTheProjectsOwnFiles:
    """A project directory should still tell its own story."""

    def test_pid_and_port_land_beside_the_project(self, registry, tmp_path):
        project_dir = tmp_path / "ws" / "alpha"
        registry.add(_record(), str(project_dir))

        assert (project_dir / AGENT_PID_FILE).read_text() == str(os.getpid())
        assert (project_dir / AGENT_PORT_FILE).read_text() == "41000"
        assert read_project_agent(str(project_dir)) == (os.getpid(), 41000)

    def test_they_are_cleared_when_the_agent_goes(self, registry, tmp_path):
        project_dir = tmp_path / "ws" / "alpha"
        registry.add(_record(), str(project_dir))
        registry.remove("alpha", str(project_dir))

        assert not (project_dir / AGENT_PID_FILE).exists()
        assert read_project_agent(str(project_dir)) is None

    def test_a_project_with_no_files_reads_as_none(self, tmp_path):
        assert read_project_agent(str(tmp_path)) is None

    def test_garbage_in_the_files_reads_as_none(self, tmp_path):
        (tmp_path / AGENT_PID_FILE).write_text("not-a-pid")
        (tmp_path / AGENT_PORT_FILE).write_text("nope")
        assert read_project_agent(str(tmp_path)) is None
