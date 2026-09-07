"""Where the supervisor records the processes it is responsible for.

One ``server.pid`` and one ``server.port`` described a workspace when a
workspace was one process.  With a process per project there is a set of them,
and something has to be able to answer "what is running, on which port, for
which project" — for the supervisor's own routing, for ``slp agent status``,
and for a restart that has to find the children an earlier supervisor left
behind.

Written in two places, deliberately:

* ``~/.smartloop/agents.json`` — the supervisor's own view, the file it reads
  on startup to adopt or reap what a previous run left;
* ``<project_dir>/agent.pid`` and ``<project_dir>/agent.port`` — beside the
  project they belong to, so a project directory is still self-describing and
  a human debugging one project does not have to parse the whole registry.

The registry is advisory.  A pid in it is a claim, not a guarantee — the
process may have been killed — so every read that matters is checked against
the live process (:func:`process_alive`).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

REGISTRY_FILE = "agents.json"
REGISTRY_VERSION = 1

#: Per-project bookkeeping, beside the project it describes.
AGENT_PID_FILE = "agent.pid"
AGENT_PORT_FILE = "agent.port"


def process_alive(pid: Optional[int]) -> bool:
    """Whether *pid* names a live process.

    Asked through psutil rather than the usual ``os.kill(pid, 0)``.  The
    signal-0 trick is POSIX: on Windows ``os.kill`` maps every signal other
    than ``CTRL_C_EVENT``/``CTRL_BREAK_EVENT`` onto ``TerminateProcess``, so
    the conventional liveness check would *kill the process it is asking
    about* — and this is called from ``slp agent status``.

    A zombie does not count as alive.  On POSIX a child nobody has waited on
    still answers signal 0 after it dies, and treating that as running would
    have the supervisor keep forwarding requests to a port with nothing
    behind it.  A process we may not inspect does count: ``AccessDenied``
    means it is there, just not ours.
    """
    if not pid or pid <= 0:
        return False
    try:
        import psutil

        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except ImportError:  # pragma: no cover - psutil is a hard dependency
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    except Exception as exc:
        if exc.__class__.__name__ == "AccessDenied":
            return True
        return False


class AgentRecord:
    """One running project agent, as the supervisor knows it."""

    __slots__ = ("project_id", "pid", "port", "started_at", "last_used_at")

    def __init__(self, project_id: str, pid: int, port: int,
                 started_at: Optional[float] = None,
                 last_used_at: Optional[float] = None) -> None:
        self.project_id = project_id
        self.pid = pid
        self.port = port
        self.started_at = started_at if started_at is not None else time.time()
        self.last_used_at = last_used_at if last_used_at is not None else self.started_at

    @property
    def base_url(self) -> str:
        """Where the supervisor forwards this project's requests."""
        return f"http://127.0.0.1:{self.port}"

    @property
    def alive(self) -> bool:
        return process_alive(self.pid)

    @property
    def idle_seconds(self) -> float:
        return max(0.0, time.time() - self.last_used_at)

    def touch(self) -> None:
        """Record that this agent just served a request."""
        self.last_used_at = time.time()

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "port": self.port,
            "started_at": self.started_at,
            "last_used_at": self.last_used_at,
        }

    @classmethod
    def from_dict(cls, project_id: str, data: dict) -> Optional["AgentRecord"]:
        try:
            return cls(
                project_id=project_id,
                pid=int(data["pid"]),
                port=int(data["port"]),
                started_at=data.get("started_at"),
                last_used_at=data.get("last_used_at"),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"AgentRecord(project_id={self.project_id!r}, pid={self.pid}, "
                f"port={self.port})")


class AgentRegistry:
    """The set of processes this workspace is running, persisted to disk.

    In-memory state is authoritative for the running supervisor; the file is
    how a *new* supervisor discovers what an old one left behind, and how
    ``slp agent status`` reports without talking to anything.
    """

    def __init__(self, home_dir: str) -> None:
        self.home_dir = home_dir
        self.path = os.path.join(home_dir, REGISTRY_FILE)
        self._agents: Dict[str, AgentRecord] = {}
        self._supervisor: dict = {}
        self._model_host: dict = {}
        self._lock = threading.RLock()

    # -- accessors ----------------------------------------------------------

    def get(self, project_id: str) -> Optional[AgentRecord]:
        with self._lock:
            return self._agents.get(project_id)

    def all(self) -> list[AgentRecord]:
        with self._lock:
            return list(self._agents.values())

    def live(self) -> list[AgentRecord]:
        """Records whose process is actually still there."""
        return [record for record in self.all() if record.alive]

    @property
    def supervisor(self) -> dict:
        with self._lock:
            return dict(self._supervisor)

    @property
    def model_host(self) -> dict:
        with self._lock:
            return dict(self._model_host)

    # -- mutation -----------------------------------------------------------

    def add(self, record: AgentRecord, project_dir: Optional[str] = None) -> AgentRecord:
        with self._lock:
            self._agents[record.project_id] = record
            self.save()
        if project_dir:
            _write_project_files(project_dir, record)
        return record

    def remove(self, project_id: str, project_dir: Optional[str] = None) -> None:
        with self._lock:
            self._agents.pop(project_id, None)
            self.save()
        if project_dir:
            _clear_project_files(project_dir)

    def set_supervisor(self, pid: int, port: int) -> None:
        with self._lock:
            self._supervisor = {"pid": pid, "port": port}
            self.save()

    def set_model_host(self, pid: Optional[int], address: Optional[str]) -> None:
        with self._lock:
            self._model_host = {"pid": pid, "address": address}
            self.save()

    def reap(self) -> list[str]:
        """Drop records whose process is gone. Returns the ids dropped."""
        with self._lock:
            dead = [pid for pid, record in self._agents.items() if not record.alive]
            for project_id in dead:
                self._agents.pop(project_id, None)
            if dead:
                self.save()
        return dead

    # -- persistence --------------------------------------------------------

    def save(self) -> None:
        """Write the registry, atomically.

        Best-effort: the registry is a convenience, and a workspace on a
        read-only or full filesystem should still serve requests rather than
        fail on bookkeeping.
        """
        payload = {
            "version": REGISTRY_VERSION,
            "supervisor": self._supervisor,
            "model_host": self._model_host,
            "agents": {pid: record.to_dict() for pid, record in self._agents.items()},
        }
        try:
            os.makedirs(self.home_dir, exist_ok=True)
            tmp = f"{self.path}.tmp"
            with open(tmp, "w") as fh:
                json.dump(payload, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except OSError as exc:
            logger.warning("[supervisor] could not write %s: %s", self.path, exc)

    def load(self) -> "AgentRegistry":
        """Read what a previous supervisor left, keeping only live processes.

        Dead records are dropped rather than reported: a pid file outliving its
        process is the normal outcome of a crash, and adopting one would have
        the supervisor route requests into nothing.
        """
        try:
            with open(self.path, "r") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return self

        with self._lock:
            self._supervisor = data.get("supervisor") or {}
            self._model_host = data.get("model_host") or {}
            self._agents = {}
            for project_id, raw in (data.get("agents") or {}).items():
                record = AgentRecord.from_dict(project_id, raw or {})
                if record is not None and record.alive:
                    self._agents[project_id] = record
                elif record is not None:
                    logger.info(
                        "[supervisor] project %s: pid %s is gone, dropping it",
                        project_id, record.pid,
                    )
        return self


def _write_project_files(project_dir: str, record: AgentRecord) -> None:
    """Record pid/port beside the project, so its directory tells its own story."""
    try:
        os.makedirs(project_dir, exist_ok=True)
        with open(os.path.join(project_dir, AGENT_PID_FILE), "w") as fh:
            fh.write(str(record.pid))
        with open(os.path.join(project_dir, AGENT_PORT_FILE), "w") as fh:
            fh.write(str(record.port))
    except OSError as exc:
        logger.warning("[supervisor] could not write agent files in %s: %s",
                       project_dir, exc)


def _clear_project_files(project_dir: str) -> None:
    for name in (AGENT_PID_FILE, AGENT_PORT_FILE):
        try:
            os.unlink(os.path.join(project_dir, name))
        except OSError:
            pass


def read_project_agent(project_dir: str) -> Optional[tuple[int, int]]:
    """``(pid, port)`` recorded beside a project, or ``None``.

    Used by tooling that has a project directory and no registry — and as the
    cross-check that the registry and the project agree.
    """
    try:
        pid = int(open(os.path.join(project_dir, AGENT_PID_FILE)).read().strip())
        port = int(open(os.path.join(project_dir, AGENT_PORT_FILE)).read().strip())
    except (OSError, ValueError):
        return None
    return pid, port
