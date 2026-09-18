"""The supervisor: one port, one workspace registry, a process per project.

What it owns (Tier 1 in the RFC): the client-facing port, the workspace record,
the model catalog and credentials, release polling and token refresh — and the
children.  What it does not own is any project's state; that belongs to the
child fixed to that project, which is the whole point of the split.

This is the architecture, not a mode.  ``SLP_SUPERVISOR=0`` (or ``slp agent
start --no-supervised``) still falls back to the single shared server, but that
is a migration escape hatch to be removed once this has run in the field — not
a second topology to keep working indefinitely.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from smartloop.config import AppSettings
from smartloop.supervisor.pool import ProjectAgentPool
from smartloop.supervisor.registry import AgentRegistry

logger = logging.getLogger(__name__)


def _settings_now() -> AppSettings:
    """Read the settings at the moment they are needed, not at import.

    A module-level ``AppSettings()`` freezes the environment as it was when
    this module was first imported, which makes the agent limits impossible to
    set from a launcher — and, more to the point, is the same import-time
    pinning the workspace paths had to shed for a child to be pointed at its
    own home.
    """
    return AppSettings()


#: Values of ``SLP_SUPERVISOR`` that turn supervision *off*.  Supervision is
#: the default, so the environment variable is an opt-out.
_DISABLED_VALUES = ("0", "false", "no", "off")


def supervision_enabled() -> bool:
    """Whether this process should supervise rather than serve projects itself.

    On by default: a project's runaway index job, wedged MCP server or OOM
    should cost that project and no other, and that is only true when each one
    has its own process.  ``SLP_SUPERVISOR=0`` (or ``slp agent start
    --no-supervised``) falls back to the single shared server.

    A process pinned to a project is never a supervisor, whatever the
    environment says.  Children inherit the environment, so with supervision
    defaulted on this is what stops a child from trying to supervise itself and
    spawning a grandchild for the project it already is.
    """
    from smartloop.workspace.context import pinned_project_id

    if pinned_project_id() is not None:
        return False
    return (os.environ.get("SLP_SUPERVISOR", "").strip().lower()
            not in _DISABLED_VALUES)


def model_host_enabled() -> bool:
    """Whether the supervisor should run a model host for its children.

    The gRPC IPC path is the only supported transport; any opt-out is ignored so
    the shared model host remains the default and the only routing choice.
    """
    return True


class Supervisor:
    """Owns the registry, the model host, and the pool of project agents."""

    def __init__(self, home_dir: Optional[str] = None,
                 idle_ttl: Optional[float] = None,
                 memory_limit_mb: Optional[int] = None) -> None:
        settings = _settings_now()
        self.home_dir = home_dir or settings.home_dir
        self.registry = AgentRegistry(self.home_dir).load()
        self.idle_ttl = (settings.agent_idle_ttl_s if idle_ttl is None
                         else idle_ttl)
        self.memory_limit_bytes = int(
            (settings.agent_memory_limit_mb if memory_limit_mb is None
             else memory_limit_mb) * 1024 * 1024
        )
        self.model_host = None
        self._model_host_thread: Optional[threading.Thread] = None
        self.pool: Optional[ProjectAgentPool] = None

    # -- project lookup -----------------------------------------------------

    def project_dir_for(self, project_id: str) -> Optional[str]:
        """Where a project's state lives, read fresh from the workspace record.

        Deliberately not cached: a project created while the supervisor is
        running must be routable without a restart, and the workspace record is
        cheap to read now that it no longer carries every project's documents.
        """
        from smartloop.workspace import Workspace

        workspace = Workspace.load()
        if workspace is None:
            return None
        project = next(
            (p for p in workspace.projects if p.id == project_id), None,
        )
        return project.project_dir if project else None

    # -- lifecycle ----------------------------------------------------------

    def start(self, port: int) -> None:
        """Bring up the model host and the pool, and record ourselves."""
        from smartloop.workspace.context import delegate_projects

        self.registry.set_supervisor(os.getpid(), port)

        # From here each project's own file belongs to the child pinned to it.
        # Declared before the pool exists rather than after, so a save on the
        # way up cannot republish this process's snapshot over a project a
        # reclaimed agent is already serving.
        delegate_projects(True)

        address = self._start_model_host()
        self.pool = ProjectAgentPool(
            registry=self.registry,
            project_dir_for=self.project_dir_for,
            idle_ttl=self.idle_ttl,
            model_host_address=address,
            memory_limit_bytes=self.memory_limit_bytes,
        )
        self.pool.start_reaper()
        adopted = [r.project_id for r in self.registry.live()]
        if adopted:
            logger.info("[supervisor] adopted running agents for %s",
                        ", ".join(adopted))
        self._reclaim_strays()
        logger.info("[supervisor] ready on port %d", port)

    def _reclaim_strays(self) -> list[str]:
        """Take back agents that are running but in nobody's registry.

        The registry is how a new supervisor finds the children an old one
        left, so anything missing from it is invisible: a stray keeps its
        project's task queue and MCP children alive, holds the port recorded
        beside the project, and — before the parent watch — never exited.  The
        registry goes missing more easily than it sounds: a workspace whose
        home was wiped, an agent from a test run against a temporary home, a
        crash between spawning a child and recording it.

        A project directory still describes itself (``agent.pid``,
        ``agent.port``), so that is what is read here.  A stray that answers
        ``/health`` is adopted rather than killed — it is a warm process
        serving the right project, and once it is in the registry this
        supervisor is the successor its parent watch will find.  One that does
        not answer is stopped, and its project's next request starts a fresh
        one.
        """
        from smartloop.supervisor.pool import terminate_pid
        from smartloop.supervisor.registry import (
            AgentRecord, process_alive, read_project_agent,
        )
        from smartloop.workspace import Workspace

        workspace = Workspace.load()
        if workspace is None:
            return []

        reclaimed: list[str] = []
        for project in workspace.projects:
            published = read_project_agent(project.project_dir)
            if published is None:
                continue
            pid, port = published
            if not process_alive(pid):
                continue
            known = self.registry.get(project.id)
            if known is not None and known.pid == pid:
                continue  # already adopted above

            record = AgentRecord(project.id, pid, port)
            if _answers_health(record.base_url):
                logger.info(
                    "[supervisor] reclaiming stray agent for %s (pid %d, port %d)",
                    project.id, pid, port,
                )
                self.registry.add(record, project.project_dir)
                reclaimed.append(project.id)
            else:
                logger.warning(
                    "[supervisor] stray agent for %s (pid %d) is not answering; "
                    "stopping it", project.id, pid,
                )
                terminate_pid(pid, timeout=10.0)
        return reclaimed

    def _start_model_host(self) -> Optional[str]:
        """Run the model host in this process, on a background thread.

        In-process rather than a fourth process: the supervisor is already
        long-lived and does no project work, so it is the natural home for the
        device, and one less process is one less thing to supervise.  Children
        reach it over the socket exactly as they would a separate host, so
        splitting it out later changes nothing on their side.
        """
        from smartloop.api.services import get_model_service
        from smartloop.model_host.client import (
            InProcessModelClient, default_socket_path,
        )
        from smartloop.model_host.host import ModelHost

        address = (os.environ.get("SLP_MODEL_HOST") or "").strip() or \
            default_socket_path(self.home_dir)
        try:
            host = ModelHost(InProcessModelClient(get_model_service()), address)
            bound = host.start()
        except OSError as exc:
            # Losing the host is not fatal: children fall back to loading their
            # own models, which is slower and heavier but still correct.
            logger.warning("[supervisor] could not start the model host: %s", exc)
            self.registry.set_model_host(None, None)
            return None

        self.model_host = host
        self.registry.set_model_host(os.getpid(), bound)
        return bound

    def stop(self) -> None:
        """Stop the children, then the host. Children first: they use it."""
        from smartloop.workspace.context import delegate_projects

        delegate_projects(False)
        if self.pool is not None:
            self.pool.stop_reaper()
            self.pool.stop_all()
            self.pool = None
        if self.model_host is not None:
            self.model_host.stop()
            self.model_host = None
        self.registry.set_model_host(None, None)
        logger.info("[supervisor] stopped")

    def release(self, project_id: str) -> bool:
        """Stop the agent pinned to *project_id* and forget its backoff.

        For a project that is going away — the only lifecycle event that has to
        reach the pool.  Everything else the pool stops (idle, over the RSS
        ceiling, crashed) is a project that will be started again on its next
        request; this one will not, and leaving the child up would have it
        serving a project whose directory is about to be removed underneath it.

        Returns whether there was an agent to stop.
        """
        if self.pool is None:
            return False
        stopped = self.pool.stop(project_id)
        self.pool.clear_backoff(project_id)
        return stopped

    # -- reporting ----------------------------------------------------------

    def status(self) -> dict:
        """What ``slp agent status`` reports: the whole tree, not one process."""
        import psutil

        agents = []
        for record in self.registry.all():
            entry = {
                "project_id": record.project_id,
                "pid": record.pid,
                "port": record.port,
                "alive": record.alive,
                "idle_seconds": round(record.idle_seconds, 1),
                "rss_bytes": _rss(record.pid, psutil),
            }
            agents.append(entry)

        return {
            "supervisor": {
                **self.registry.supervisor,
                "rss_bytes": _rss(self.registry.supervisor.get("pid"), psutil),
            },
            "model_host": self.registry.model_host,
            "agents": agents,
            "limits": {
                "idle_ttl_seconds": self.idle_ttl,
                "memory_limit_bytes": self.memory_limit_bytes or None,
            },
        }


def _answers_health(base_url: str, timeout: float = 5.0) -> bool:
    """Whether something is serving the agent contract at *base_url*."""
    import httpx

    try:
        return httpx.get(f"{base_url}/health", timeout=timeout).status_code == 200
    except Exception:
        return False


def _rss(pid: Optional[int], psutil) -> Optional[int]:
    """Resident size of *pid*, or ``None`` when it cannot be read.

    The number the RFC wants measured before deciding whether children are
    lazily started or pre-warmed: a child carries FastAPI, the SQLite index and
    the LangChain import graph even with no model in it.
    """
    if not pid:
        return None
    try:
        return psutil.Process(pid).memory_info().rss
    except Exception:
        return None


#: The process-wide supervisor, when this process is one.
_supervisor: Optional[Supervisor] = None
_supervisor_lock = threading.Lock()


def get_supervisor() -> Optional[Supervisor]:
    """The running supervisor, or ``None`` when it was explicitly disabled."""
    return _supervisor


def release_project(project_id: str) -> bool:
    """Stop *project_id*'s agent, if this process is supervising one.

    A no-op in a child and under ``--no-supervised``, where there is no pool and
    the process serving the request is the only one there is.
    """
    supervisor = get_supervisor()
    if supervisor is None:
        return False
    return supervisor.release(project_id)


def start_supervisor(port: int, idle_ttl: Optional[float] = None) -> Supervisor:
    """Create and start the process-wide supervisor."""
    global _supervisor
    with _supervisor_lock:
        if _supervisor is None:
            _supervisor = Supervisor(idle_ttl=idle_ttl)
            _supervisor.start(port)
        return _supervisor


def stop_supervisor() -> None:
    """Stop and forget the process-wide supervisor."""
    global _supervisor
    with _supervisor_lock:
        if _supervisor is not None:
            _supervisor.stop()
            _supervisor = None


def get_pool() -> Optional[ProjectAgentPool]:
    """The pool routing should use, or ``None`` when not supervising."""
    supervisor = _supervisor
    return supervisor.pool if supervisor is not None else None
