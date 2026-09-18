"""The fork-free detached spawn used for per-project agents.

``subprocess.Popen(..., start_new_session=True, close_fds=True)`` forks on
macOS/Linux (CPython's ``posix_spawn`` fallback excludes both flags), and a
``fork()`` taken while the supervisor's gRPC server is mid-RPC is exactly the
accident this module exists to avoid.  The behaviours below are the ones that
must survive the swap to ``os.posix_spawn``: the child lands in its own
session, its stdin is ``/dev/null``, no unrelated descriptors are inherited,
and the handle the pool relies on still polls, waits, reaps and signals.
"""

import json
import os
import subprocess
import sys
import time

import pytest

from smartloop.supervisor.spawn import (
    DetachedProcess,
    popen_detached_devnull,
    spawn_detached,
)

_SLEEP = [sys.executable, "-c", "import time; time.sleep(30)"]


class TestDetachedProcessFacade:
    def test_wait_reaps_and_poll_returns_the_exit_code(self):
        proc = DetachedProcess(
            spawn_detached([sys.executable, "-c", "raise SystemExit(7)"])
        )

        assert proc.wait(timeout=10) == 7
        assert proc.poll() == 7
        assert proc.returncode == 7

    def test_a_wait_timeout_raises_like_popen(self):
        proc = DetachedProcess(spawn_detached(_SLEEP))
        try:
            started = time.monotonic()
            with pytest.raises(subprocess.TimeoutExpired):
                proc.wait(timeout=0.2)
            assert time.monotonic() - started < 10
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)

    def test_terminate_signals_the_child(self):
        proc = DetachedProcess(spawn_detached(_SLEEP))
        try:
            proc.terminate()
            # No handler installed: SIGTERM kills it, reported as -15.
            assert proc.wait(timeout=5) == -15
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)


class TestSpawnedChild:
    def test_the_child_gets_its_own_session_and_devnull_stdin(self, tmp_path):
        probe_path = tmp_path / "probe.json"
        probe = (
            "import os\n"
            "open(os.environ['PROBE'], 'w').write(__import__('json').dumps({\n"
            "  'sid': os.getsid(0), 'pgid': os.getpgid(0), 'pid': os.getpid(),\n"
            "  'stdin_devnull': os.fstat(0).st_rdev == "
            "os.fstat(os.open(os.devnull, os.O_RDWR)).st_rdev,\n"
            "  'fds': sorted(int(f) for f in os.listdir('/dev/fd') if f.isdigit()),\n"
            "}))\n"
        )
        env = dict(os.environ)
        env["PROBE"] = str(probe_path)
        pid = spawn_detached([sys.executable, "-c", probe], env=env)
        DetachedProcess(pid).wait(timeout=10)

        data = json.loads(probe_path.read_text())
        assert data["sid"] == pid and data["pgid"] == pid
        assert data["stdin_devnull"] is True
        # A fork would have copied this pytest process's whole table: logging
        # handlers, sockets, the works.  In a spawned child nothing of the
        # parent's survives — 0, 1, 2 (devnull, stdout, stderr) plus at most
        # the probe's own transient descriptors.
        assert set(data["fds"]) >= {0, 1, 2}
        assert len(data["fds"]) <= 6


class TestPopenFallback:
    def test_it_spawns_a_real_child(self):
        proc = popen_detached_devnull([sys.executable, "-c", "pass"])
        assert proc.pid > 0
        assert proc.wait(timeout=10) == 0

    def test_it_spawns_dozens_while_a_grpc_server_is_live(self):
        """The scenario this exists for: forking with gRPC threads active."""
        import grpc
        from concurrent.futures import ThreadPoolExecutor

        server = grpc.server(ThreadPoolExecutor(max_workers=2))
        server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            for _ in range(10):
                proc = popen_detached_devnull(
                    [sys.executable, "-c", "import time; time.sleep(0.05)"]
                )
                assert proc.wait(timeout=10) == 0
        finally:
            server.stop(0)

    def test_it_falls_back_to_subprocess_when_posix_spawn_is_unusable(
        self, monkeypatch,
    ):
        import smartloop.supervisor.spawn as spawn_mod

        def no_posix_spawn(*_a, **_k):
            raise AttributeError("no posix_spawn")

        monkeypatch.setattr(spawn_mod, "spawn_detached", no_posix_spawn)

        proc = popen_detached_devnull([sys.executable, "-c", "pass"])

        assert isinstance(proc, subprocess.Popen)
        assert proc.wait(timeout=10) == 0