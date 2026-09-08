"""A project agent must not outlive the supervisor that spawned it.

Children are spawned detached, so the OS does not link them to the parent.
Before the parent watch, a supervisor that was killed rather than stopped left
its agents running — reparented to init, serving projects nothing routed to,
until the machine rebooted.  These tests use real processes for the parent,
because the two things that make the watch correct (a pid really being gone,
and a *recycled* pid not counting as the parent) are properties of the process
table, not of a mock.
"""

import json
import os
import subprocess
import sys
import threading
import time

import pytest

from smartloop.supervisor.parent_watch import (
    PARENT_PID_ENV, PARENT_STARTED_AT_ENV, ParentWatch, parent_environment,
    parent_is_alive, process_started_at, start_parent_watch,
)
from smartloop.supervisor.registry import AgentRegistry


@pytest.fixture
def sacrificial():
    """A real process we can kill, standing in for a supervisor."""
    spawned = []

    def spawn():
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(300)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        spawned.append(process)
        return process

    yield spawn
    for process in spawned:
        try:
            process.kill()
            process.wait(timeout=5)
        except Exception:
            pass


def _wait_for(predicate, timeout=10.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class TestParentIdentity:
    def test_the_environment_carries_the_pid_and_the_start_time(self):
        env = parent_environment()

        assert env[PARENT_PID_ENV] == str(os.getpid())
        assert float(env[PARENT_STARTED_AT_ENV]) == pytest.approx(
            process_started_at(os.getpid())
        )

    def test_a_live_parent_reads_as_alive(self, sacrificial):
        process = sacrificial()

        assert parent_is_alive(process.pid, process_started_at(process.pid))

    def test_a_dead_parent_reads_as_gone(self, sacrificial):
        process = sacrificial()
        started_at = process_started_at(process.pid)
        process.kill()
        process.wait(timeout=5)

        assert not parent_is_alive(process.pid, started_at)

    def test_a_recycled_pid_is_not_the_parent(self):
        """The check that keeps an orphan from adopting a stranger.

        Pids are reused, so on a busy machine the number a dead supervisor
        held can belong to something else within minutes.  Matching on the pid
        alone would read that as a living parent and the child would stay up
        for good — behind a process that has never heard of it.
        """
        alive_but_unrelated = os.getpid()
        long_ago = process_started_at(alive_but_unrelated) - 3600

        assert not parent_is_alive(alive_but_unrelated, long_ago)

    def test_an_unknown_start_time_does_not_condemn_a_live_parent(self):
        """Being unable to tell is not evidence of an orphan."""
        assert parent_is_alive(os.getpid(), None)


class TestTheWatch:
    def test_it_does_nothing_while_the_parent_is_alive(self, sacrificial):
        process = sacrificial()
        fired = threading.Event()
        watch = ParentWatch(
            pid=process.pid, started_at=process_started_at(process.pid),
            on_orphaned=fired.set, poll_interval=0.05, grace=0.1,
        ).start()

        assert not fired.wait(1.0)
        watch.stop()

    def test_it_fires_once_the_parent_is_gone(self, sacrificial):
        process = sacrificial()
        fired = threading.Event()
        watch = ParentWatch(
            pid=process.pid, started_at=process_started_at(process.pid),
            on_orphaned=fired.set, poll_interval=0.05, grace=0.2,
        ).start()

        process.kill()
        process.wait(timeout=5)

        assert fired.wait(10.0)
        watch.stop()

    def test_it_waits_out_the_grace_period_first(self, sacrificial):
        """A supervisor restart is a handover, not an orphaning."""
        process = sacrificial()
        fired = threading.Event()
        watch = ParentWatch(
            pid=process.pid, started_at=process_started_at(process.pid),
            on_orphaned=fired.set, poll_interval=0.05, grace=1.5,
        ).start()

        process.kill()
        process.wait(timeout=5)

        assert not fired.wait(0.7)
        assert fired.wait(10.0)
        watch.stop()

    def test_stopping_the_watch_stops_the_shutdown(self, sacrificial):
        """An ordinary shutdown must not be reported as an orphaning."""
        process = sacrificial()
        fired = threading.Event()
        watch = ParentWatch(
            pid=process.pid, started_at=process_started_at(process.pid),
            on_orphaned=fired.set, poll_interval=0.05, grace=0.1,
        ).start()

        watch.stop()
        process.kill()
        process.wait(timeout=5)

        assert not fired.wait(1.0)

    def test_a_failing_callback_does_not_take_the_thread_down_silently(
        self, sacrificial
    ):
        process = sacrificial()
        called = threading.Event()

        def boom():
            called.set()
            raise RuntimeError("shutdown failed")

        watch = ParentWatch(
            pid=process.pid, started_at=process_started_at(process.pid),
            on_orphaned=boom, poll_interval=0.05, grace=0.1,
        ).start()
        process.kill()
        process.wait(timeout=5)

        assert called.wait(10.0)
        watch.stop()


class TestHandover:
    """A replacement supervisor adopts the children, so the watch re-targets."""

    def test_a_live_successor_in_the_registry_keeps_the_child_up(
        self, tmp_path, sacrificial
    ):
        parent = sacrificial()
        successor = sacrificial()
        registry = AgentRegistry(str(tmp_path))
        registry.set_supervisor(successor.pid, 38540)

        fired = threading.Event()
        watch = ParentWatch(
            pid=parent.pid, started_at=process_started_at(parent.pid),
            home_dir=str(tmp_path), on_orphaned=fired.set,
            poll_interval=0.05, grace=0.2,
        ).start()
        parent.kill()
        parent.wait(timeout=5)

        assert not fired.wait(1.5)
        assert watch.pid == successor.pid, "the watch should re-target"
        watch.stop()

    def test_a_dead_successor_is_not_a_successor(self, tmp_path, sacrificial):
        parent = sacrificial()
        successor = sacrificial()
        AgentRegistry(str(tmp_path)).set_supervisor(successor.pid, 38540)
        for process in (parent, successor):
            process.kill()
            process.wait(timeout=5)

        fired = threading.Event()
        watch = ParentWatch(
            pid=parent.pid, started_at=process_started_at(parent.pid),
            home_dir=str(tmp_path), on_orphaned=fired.set,
            poll_interval=0.05, grace=0.2,
        ).start()

        assert fired.wait(10.0)
        watch.stop()

    def test_a_recycled_successor_pid_is_not_a_successor(
        self, tmp_path, sacrificial
    ):
        """The registry records a start time for exactly this case."""
        parent = sacrificial()
        registry = AgentRegistry(str(tmp_path))
        registry.set_supervisor(os.getpid(), 38540)
        # Rewrite the recorded start time as if the pid had been reused.
        path = tmp_path / "agents.json"
        data = json.loads(path.read_text())
        data["supervisor"]["started_at"] = process_started_at(os.getpid()) - 3600
        path.write_text(json.dumps(data))

        fired = threading.Event()
        watch = ParentWatch(
            pid=parent.pid, started_at=process_started_at(parent.pid),
            home_dir=str(tmp_path), on_orphaned=fired.set,
            poll_interval=0.05, grace=0.2,
        ).start()
        parent.kill()
        parent.wait(timeout=5)

        assert fired.wait(10.0)
        watch.stop()

    def test_an_unreadable_registry_is_not_a_successor(self, tmp_path, sacrificial):
        parent = sacrificial()
        (tmp_path / "agents.json").write_text("{ not json")

        fired = threading.Event()
        watch = ParentWatch(
            pid=parent.pid, started_at=process_started_at(parent.pid),
            home_dir=str(tmp_path), on_orphaned=fired.set,
            poll_interval=0.05, grace=0.2,
        ).start()
        parent.kill()
        parent.wait(timeout=5)

        assert fired.wait(10.0)
        watch.stop()


class TestOptingOut:
    def test_a_process_with_no_parent_is_not_watched(self, monkeypatch):
        """The shared server, and a hand-run agent, have nothing to watch."""
        monkeypatch.delenv(PARENT_PID_ENV, raising=False)

        assert start_parent_watch(lambda: None) is None

    def test_a_nonsense_parent_pid_is_not_watched(self, monkeypatch):
        monkeypatch.setenv(PARENT_PID_ENV, "not-a-pid")

        assert start_parent_watch(lambda: None) is None

    def test_a_process_is_not_its_own_parent(self, monkeypatch):
        monkeypatch.setenv(PARENT_PID_ENV, str(os.getpid()))

        assert start_parent_watch(lambda: None) is None

    def test_the_interval_and_grace_are_configurable(self, monkeypatch, sacrificial):
        process = sacrificial()
        monkeypatch.setenv(PARENT_PID_ENV, str(process.pid))
        monkeypatch.setenv(PARENT_STARTED_AT_ENV,
                           repr(process_started_at(process.pid)))
        monkeypatch.setenv("SLP_PARENT_WATCH_INTERVAL_S", "0.25")
        monkeypatch.setenv("SLP_PARENT_WATCH_GRACE_S", "2.5")

        watch = start_parent_watch(lambda: None)

        assert (watch.poll_interval, watch.grace) == (0.25, 2.5)
        watch.stop()
