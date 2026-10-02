"""Spawning, watching and reaping the per-project agent processes.

The isolation payoff lives here.  When one process served every project, a
runaway index job, a wedged MCP server or an OOM took down all of them,
because there was one task queue, one process and one ``server.pid``.  A child
per project turns each of those into one project's problem: the supervisor
notices, restarts that child with backoff, and the others never knew.

Lifecycle:

* **lazy start** — a child is spawned on the first request that names its
  project, so an install with ten projects does not pay for ten processes to
  answer a question about one;
* **idle stop** — a child that has served nothing for
  :data:`DEFAULT_IDLE_TTL_SECONDS` is stopped, returning its baseline RSS;
* **restart on crash** — with exponential backoff, and the backoff reset once a
  child has been up long enough to count as healthy.  The policy is lifted
  from ``start_server_foreground``, which has always done exactly this for the
  single process; here it is applied per child.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, Optional

import httpx

from smartloop.supervisor.registry import AgentRecord, AgentRegistry
from smartloop.supervisor.spawn import DetachedProcess, popen_detached_devnull

logger = logging.getLogger(__name__)

#: How long a child may sit unused before it is stopped.
DEFAULT_IDLE_TTL_SECONDS = 900.0

#: How long to wait for a freshly spawned child to answer /health.
DEFAULT_START_TIMEOUT_SECONDS = 120.0

#: Restart backoff, matching the single-process policy it is lifted from.
_BACKOFF_START = 1.0
_BACKOFF_MAX = 60.0
_BACKOFF_RESET_AFTER = 120.0

#: Per-child resident-size ceiling, in bytes. 0 (the default) means no ceiling.
#: A ceiling is the other half of the isolation: a crash is contained by having
#: separate processes, but a project that leaks or over-indexes would still
#: take the machine down with it. Stopping that one child returns the memory
#: and leaves every other project serving; the next request starts it again.
DEFAULT_MEMORY_LIMIT_BYTES = 0


class ChildStartError(RuntimeError):
    """A project agent could not be started, or never became healthy."""


def _is_own_pid(launcher_pid: int, candidate_pid: int) -> bool:
    """Whether *candidate_pid* is the process we spawned, or one it launched.

    Some Python installs on Windows point ``sys.executable`` at a ``python.exe``
    that is itself a launcher: it starts the real interpreter as a child and
    simply waits on it, so ``Popen.pid`` names the launcher while the project
    agent's own ``os.getpid()`` — the one it publishes — names the grandchild.
    Direct equality is the common case; walking descendants covers that one,
    without which a launcher-style interpreter can never pass this check and
    every cold start times out and retries forever.
    """
    if launcher_pid == candidate_pid:
        return True
    try:
        import psutil
        return any(
            child.pid == candidate_pid
            for child in psutil.Process(launcher_pid).children(recursive=True)
        )
    except Exception:
        return False


def _live_children(pid: int) -> list:
    """Snapshot of *pid*'s live children, before anything might kill it.

    Has to be taken before termination, not after: once a process has exited,
    ``psutil`` can no longer be asked what its children were, and the whole
    point of this is to find a real project agent process still running
    under a launcher-style interpreter (see :func:`_is_own_pid`) after that
    launcher is gone.
    """
    try:
        import psutil
        return psutil.Process(pid).children(recursive=True)
    except Exception:
        return []


def _kill_all(procs) -> None:
    """Best-effort kill of every process in *procs*, waited on to completion.

    ``Popen.terminate()``/``kill()`` reach only the process we hold the handle
    for. On Windows nothing propagates that to a child a launcher-style
    interpreter started, so without this the real project agent process is
    orphaned rather than stopped whenever the launcher and the worker are
    different pids. The wait matters as much as the kill: without it, a
    caller checking liveness right after ``stop()`` returns can still catch
    the process before Windows has reaped it.
    """
    for proc in procs:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


class ProjectAgentPool:
    """The set of running project agents, and the policy for keeping them.

    Every method that spawns or stops takes the pool lock, so two concurrent
    first-requests for the same cold project start one child rather than two.
    Requests to *different* projects do not contend for anything but the brief
    bookkeeping.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        project_dir_for: Callable[[str], Optional[str]],
        idle_ttl: float = DEFAULT_IDLE_TTL_SECONDS,
        start_timeout: float = DEFAULT_START_TIMEOUT_SECONDS,
        model_host_address: Optional[str] = None,
        command_for: Optional[Callable[[str], list]] = None,
        memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES,
    ) -> None:
        self.registry = registry
        self.project_dir_for = project_dir_for
        self.idle_ttl = idle_ttl
        self.start_timeout = start_timeout
        self.model_host_address = model_host_address
        self.memory_limit_bytes = memory_limit_bytes
        # How a child is launched.  Overridable so the lifecycle can be
        # exercised against a process that does not load a model.
        self._command_for = command_for or self._default_command
        self._processes: Dict[str, subprocess.Popen | DetachedProcess] = {}
        self._backoff: Dict[str, float] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._reaper: Optional[threading.Thread] = None

    # -- locking ------------------------------------------------------------

    def _project_lock(self, project_id: str) -> threading.Lock:
        """One lock per project, so cold starts never serialise across projects."""
        with self._lock:
            lock = self._locks.get(project_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[project_id] = lock
            return lock

    # -- starting -----------------------------------------------------------

    def _running(self, project_id: str) -> Optional[AgentRecord]:
        """The agent for *project_id* if it is genuinely serving, else ``None``.

        A child we spawned and have not waited on stays a zombie after it
        dies — ``os.kill(pid, 0)`` still succeeds, so the pid alone would say
        "alive" forever and the supervisor would keep forwarding requests to a
        port nothing is listening on.  Polling the handle we own settles it and
        reaps the zombie at the same time.  A record adopted from an earlier
        supervisor has no handle here, and for that one the pid is all there is.
        """
        record = self.registry.get(project_id)
        if record is None:
            return None
        with self._lock:
            process = self._processes.get(project_id)
        if process is not None:
            # Both signals have to agree, because for a moment after a kill
            # they disagree: the process can already be gone from the process
            # table while ``waitpid`` has not yet reported it exited.  Trusting
            # the handle alone leaves a window in which the supervisor forwards
            # a request to a port with nothing behind it.
            if process.poll() is not None:
                return None
        return record if record.alive else None

    def ensure(self, project_id: str) -> AgentRecord:
        """Return a live agent for *project_id*, starting one if needed."""
        record = self._running(project_id)
        if record is not None:
            record.touch()
            return record

        with self._project_lock(project_id):
            # Re-check: another request may have started it while we waited,
            # which is the normal case when a cold project gets two requests
            # at once.
            record = self._running(project_id)
            if record is not None:
                record.touch()
                return record
            return self._start(project_id)

    def _output_path(self, project_id: str) -> Optional[str]:
        """Where a Windows child's raw stdout/stderr go: a crash traceback
        printed before its logging is up, beside its ``agent.log``.  A file of
        its own, because ``agent.log`` rotates and Windows will not rename a
        file another process holds open."""
        project_dir = self.project_dir_for(project_id)
        if not project_dir or not os.path.isdir(project_dir):
            return None
        return os.path.join(project_dir, "agent.out.log")

    def _start(self, project_id: str) -> AgentRecord:
        """Spawn a child and wait for it to answer. Caller holds the lock."""
        self._await_backoff(project_id)

        command = self._command_for(project_id)
        env = self._environment(project_id)
        logger.info("[supervisor] starting project agent for %s", project_id)
        try:
            # Detach from our console/process group so a Ctrl-C in the terminal
            # reaches the supervisor only — it then stops the children itself,
            # in order, after the model host they use.  Detachment and the
            # close_fds hygiene come from spawn.py, which uses posix_spawn so
            # no fork() is taken: a fork() while the host's gRPC server is
            # mid-RPC is exactly the class of crash this pool is built to
            # contain.  Windows falls back to the creation-flag Popen.
            process = popen_detached_devnull(
                command, env, output=self._output_path(project_id),
            )
        except OSError as exc:
            raise ChildStartError(
                f"could not spawn an agent for project {project_id}: {exc}"
            ) from exc

        # Own the handle from the moment it exists.  Recording it only after
        # the child was healthy left every failure in between — a start
        # timeout, an exception, a KeyboardInterrupt through this frame —
        # holding the one reference to a detached process, which is exactly
        # how an agent ends up reparented to init and running for hours.
        with self._lock:
            self._processes[project_id] = process
        try:
            project_dir = self.project_dir_for(project_id)
            real_pid, port = self._await_port(project_id, process, project_dir)
            record = AgentRecord(project_id, real_pid, port)
            self._await_health(project_id, process, record)
        except BaseException:
            with self._lock:
                self._processes.pop(project_id, None)
            self._terminate(process, timeout=5.0)
            raise

        self.registry.add(record, project_dir)
        logger.info(
            "[supervisor] project %s ready on port %d (pid %d)",
            project_id, port, process.pid,
        )
        return record

    def _default_command(self, project_id: str) -> list[str]:
        """How to launch a child.

        ``sys.executable -m <module>`` works for a frozen build too: ``main.py``
        honours ``-m`` the way ``python -m`` does, precisely so a PyInstaller
        binary can re-exec itself.
        """
        return [
            sys.executable, "-m", "smartloop.api.project_agent",
            "--project", project_id,
        ]

    def _environment(self, project_id: str) -> dict:
        from smartloop.supervisor.parent_watch import parent_environment

        env = dict(os.environ)
        env["SLP_PROJECT_ID"] = project_id
        # Detaching the child (below) is what keeps Ctrl-C off it; this is what
        # stops that detachment from outliving us.  A child that finds no
        # supervisor behind this pid — and no replacement in the registry —
        # shuts itself down rather than serving a project nobody can reach.
        env.update(parent_environment())
        if self.model_host_address:
            env["SLP_MODEL_HOST"] = self.model_host_address
            # The whole point of the host is that the weights are loaded once.
            # A child inherits ``preload_model`` from the supervisor, which is
            # on by default — leaving it would have every child pull its own
            # copy into the same device, which is precisely what per-project
            # model copies were reverted for.
            env["SLP_API_PRELOAD_MODEL"] = "0"
        # A child must never try to supervise: it serves one project directly.
        # Set explicitly rather than unset — supervision is the default now, so
        # an absent variable would mean "on" and the child would spawn a
        # grandchild for the project it already is.
        env["SLP_SUPERVISOR"] = "0"
        return env

    def _await_port(self, project_id: str, process: subprocess.Popen,
                    project_dir: Optional[str]) -> tuple[int, int]:
        """Wait for the child to publish the port it bound.

        The child writes ``agent.port`` beside its project as soon as it has
        one, before uvicorn finishes starting, so this resolves well before the
        model is loaded.

        Returns ``(pid, port)`` rather than just the port: on some Windows
        Python installs, the interpreter named by ``sys.executable`` is itself
        a launcher that starts the real interpreter as a child and waits on
        it, so ``process.pid`` names the launcher while the project agent's
        own ``os.getpid()`` — the pid it publishes — names the grandchild.
        The caller needs that real pid to track and stop the right process.
        """
        from smartloop.supervisor.registry import read_project_agent

        if not project_dir:
            raise ChildStartError(
                f"project {project_id} has no directory to read its port from"
            )
        deadline = time.monotonic() + self.start_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise ChildStartError(
                    f"agent for project {project_id} exited with code "
                    f"{process.returncode} before binding a port"
                )
            published = read_project_agent(project_dir)
            if published and _is_own_pid(process.pid, published[0]):
                return published
            time.sleep(0.05)
        self._kill(process)
        raise ChildStartError(
            f"agent for project {project_id} did not publish a port in "
            f"{self.start_timeout:.0f}s"
        )

    def _await_health(self, project_id: str, process: subprocess.Popen,
                      record: AgentRecord) -> None:
        """Wait until the child answers ``/health``."""
        deadline = time.monotonic() + self.start_timeout
        url = f"{record.base_url}/health"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise ChildStartError(
                    f"agent for project {project_id} exited with code "
                    f"{process.returncode} during startup"
                )
            try:
                if httpx.get(url, timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        self._kill(process)
        raise ChildStartError(
            f"agent for project {project_id} was not healthy within "
            f"{self.start_timeout:.0f}s"
        )

    # -- backoff ------------------------------------------------------------

    def _await_backoff(self, project_id: str) -> None:
        """Hold off restarting a project that just failed.

        Without this, a child that crashes on startup — a corrupt index, an MCP
        server that kills the process — would be respawned as fast as requests
        arrive.
        """
        with self._lock:
            delay = self._backoff.get(project_id, 0.0)
        if delay:
            logger.warning(
                "[supervisor] project %s: waiting %.0fs before restarting",
                project_id, delay,
            )
            time.sleep(delay)

    def note_exit(self, project_id: str, uptime: float) -> None:
        """Record that a child died, and lengthen or reset its backoff."""
        with self._lock:
            if uptime >= _BACKOFF_RESET_AFTER:
                self._backoff[project_id] = _BACKOFF_START
            else:
                current = self._backoff.get(project_id, 0.0)
                self._backoff[project_id] = min(
                    max(current * 2, _BACKOFF_START), _BACKOFF_MAX,
                )

    def clear_backoff(self, project_id: str) -> None:
        with self._lock:
            self._backoff.pop(project_id, None)

    # -- stopping -----------------------------------------------------------

    def stop(self, project_id: str, timeout: float = 10.0) -> bool:
        """Stop one project's agent. Returns whether there was one to stop."""
        with self._project_lock(project_id):
            record = self.registry.get(project_id)
            with self._lock:
                process = self._processes.pop(project_id, None)
            self.registry.remove(project_id, self.project_dir_for(project_id))
            if process is not None:
                self._terminate(process, timeout)
                return True
            if record is not None and record.alive:
                # Adopted from a previous supervisor: we have a pid but no
                # handle, so signal it directly.
                terminate_pid(record.pid, timeout)
                return True
            return False

    def stop_all(self, timeout: float = 10.0) -> None:
        """Stop every child. Called when the supervisor itself shuts down."""
        self._stopping.set()
        for project_id in [r.project_id for r in self.registry.all()]:
            try:
                self.stop(project_id, timeout)
            except Exception:
                logger.exception("[supervisor] failed to stop project %s", project_id)

        # Anything we spawned but never registered — a child that was still
        # starting when the shutdown arrived.  Driven off the handles rather
        # than the registry, because the registry only learns about a child
        # once it is healthy, and a half-started one is just as detached.
        with self._lock:
            stragglers = list(self._processes.items())
            self._processes.clear()
        for project_id, process in stragglers:
            logger.info("[supervisor] stopping unregistered agent for %s (pid %d)",
                        project_id, process.pid)
            try:
                self._terminate(process, timeout)
            except Exception:
                logger.exception("[supervisor] failed to stop pid %d", process.pid)

    def _terminate(self, process: subprocess.Popen, timeout: float) -> None:
        descendants = _live_children(process.pid)
        if process.poll() is not None:
            _kill_all(descendants)
            return
        try:
            process.terminate()
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("[supervisor] pid %d ignored SIGTERM; killing it",
                           process.pid)
            self._kill(process)
        except OSError:
            pass
        _kill_all(descendants)

    @staticmethod
    def _kill(process: subprocess.Popen) -> None:
        descendants = _live_children(process.pid)
        try:
            process.kill()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        _kill_all(descendants)

    # -- the reaper ---------------------------------------------------------

    def start_reaper(self, interval: float = 30.0) -> None:
        """Run the idle-stop and crash-cleanup sweep in the background."""
        if self._reaper is not None:
            return
        self._stopping.clear()
        self._reaper = threading.Thread(
            target=self._reap_loop, args=(interval,),
            name="supervisor-reaper", daemon=True,
        )
        self._reaper.start()

    def stop_reaper(self) -> None:
        self._stopping.set()
        reaper, self._reaper = self._reaper, None
        if reaper is not None:
            reaper.join(timeout=5)

    def _reap_loop(self, interval: float) -> None:
        while not self._stopping.wait(interval):
            try:
                self.sweep()
            except Exception:
                logger.exception("[supervisor] reaper pass failed")

    def sweep(self) -> dict:
        """One pass: forget the dead, stop the idle. Returns what it did."""
        crashed: list[str] = []

        # Children we spawned: poll the handle, which both settles whether the
        # process is really gone and reaps it if it is.
        with self._lock:
            owned = list(self._processes.items())
        for project_id, process in owned:
            if process.poll() is None:
                continue
            uptime = max(0.0, time.time() - _process_started_at(process))
            logger.warning(
                "[supervisor] project %s: agent exited with code %s",
                project_id, process.returncode,
            )
            with self._lock:
                self._processes.pop(project_id, None)
            self.registry.remove(project_id, self.project_dir_for(project_id))
            self.note_exit(project_id, uptime)
            crashed.append(project_id)

        # Records adopted from an earlier supervisor, whose processes are not
        # ours to wait on: the pid is all we have.
        for project_id in self.registry.reap():
            logger.warning("[supervisor] project %s: agent is gone", project_id)
            self.note_exit(project_id, 0.0)
            crashed.append(project_id)

        idled: list[str] = []
        if self.idle_ttl > 0:
            for record in self.registry.all():
                if self._running(record.project_id) is None:
                    continue
                if record.idle_seconds >= self.idle_ttl:
                    logger.info(
                        "[supervisor] project %s: idle for %.0fs, stopping it",
                        record.project_id, record.idle_seconds,
                    )
                    if self.stop(record.project_id):
                        idled.append(record.project_id)

        return {"crashed": crashed, "idled": idled,
                "over_limit": self._enforce_memory_limit()}

    def _enforce_memory_limit(self) -> list[str]:
        """Stop any child that has grown past the ceiling. Returns the ids.

        Stopped rather than killed outright: SIGTERM lets it close its SQLite
        handles and llama.cpp contexts cleanly, and the next request for that
        project starts a fresh one. The rest of the workspace never notices,
        which is the whole difference from a single process going OOM.
        """
        if self.memory_limit_bytes <= 0:
            return []

        stopped: list[str] = []
        for record in self.registry.all():
            if self._running(record.project_id) is None:
                continue
            rss = process_rss(record.pid)
            if rss is None or rss <= self.memory_limit_bytes:
                continue
            logger.warning(
                "[supervisor] project %s: %.0f MB exceeds the %.0f MB ceiling, "
                "stopping it", record.project_id,
                rss / (1024 * 1024), self.memory_limit_bytes / (1024 * 1024),
            )
            if self.stop(record.project_id):
                stopped.append(record.project_id)
        return stopped


def process_rss(pid: Optional[int]) -> Optional[int]:
    """Resident size of *pid* in bytes, or ``None`` when it cannot be read."""
    if not pid:
        return None
    try:
        import psutil

        return psutil.Process(pid).memory_info().rss
    except Exception:
        return None


def _process_started_at(process: subprocess.Popen) -> float:
    """Best-effort start time, for deciding whether a crash resets the backoff."""
    try:
        import psutil

        return psutil.Process(process.pid).create_time()
    except Exception:
        return time.time()


def terminate_pid(pid: int, timeout: float) -> None:
    """Terminate a process we did not spawn and therefore cannot ``wait`` on.

    Through psutil, which spells "ask it to stop, then insist" portably.
    ``signal.SIGKILL`` does not exist on Windows at all, and ``os.kill`` there
    terminates on any signal — so the POSIX two-step written literally would
    be an AttributeError on one platform and an immediate kill on the other.
    """
    try:
        import psutil

        proc = psutil.Process(pid)
    except Exception:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
