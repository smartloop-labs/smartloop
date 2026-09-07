"""Request-scoped project binding.

The project a piece of work belongs to used to be global process state: a
``current`` flag on one :class:`~smartloop.workspace.workspace.Project`, flipped
per request by ``ModelService.set_current_project``.  Two chats on different
projects therefore raced — the second request moved the flag while the first
was still resolving documents, skills and MCP servers against it.

Here the binding is *scoped* instead of *mutated*.  A request enters
:func:`use_project` for its whole lifetime, and every downstream reader resolves
against the context rather than the flag (see
:py:attr:`Workspace.current_project`).  Two requests on different projects hold
different bindings at the same time, so neither can move the other's.

Three layers resolve, in order:

1. the :data:`_active_project_id` context variable — the request's own binding;
2. the *process pin* (:func:`pin_project`) — a per-project agent child fixes its
   project once at spawn from ``SLP_PROJECT_ID`` and never varies it;
3. ``None`` — the caller falls back to the ``current`` flag, which after this
   change means only "the project the UI has selected".

Threads
-------
A context variable does not cross a thread boundary on its own, and the chat
path hands work to executors several times per request (orchestration, the DAG
executor, citation verification).  :func:`bind` captures the binding at the
submitting side and re-establishes it in the worker, and
:class:`ProjectBoundExecutor` applies it to every ``submit``.  Binding is
copied rather than entered, so the same binding can be live in any number of
worker threads at once — which ``contextvars.Context.run`` would refuse.
"""

from __future__ import annotations

import functools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Callable, Iterator, Optional

if TYPE_CHECKING:  # pragma: no cover
    from smartloop.workspace.workspace import Project, Workspace

logger = logging.getLogger(__name__)

#: The project this unit of work belongs to. ``None`` means "unbound" — fall
#: through to the process pin, then to the ``current`` flag.
_active_project_id: ContextVar[Optional[str]] = ContextVar(
    "smartloop_active_project_id", default=None
)

#: Process-wide pin, set once at startup by a per-project agent child.  Not a
#: context variable: it is a property of the process, not of a request, and it
#: must survive threads that were never bound (background tasks, the task
#: queue) without every one of them having to opt in.
_pinned_project_id: Optional[str] = None
_pin_lock = threading.Lock()


def pin_project(project_id: Optional[str]) -> None:
    """Fix this process to *project_id* for its whole lifetime.

    Used by a per-project agent child, which is spawned for exactly one project
    and has no reason to ever look at another.  The supervisor process leaves
    this unset, so its own reads still resolve through the ``current`` flag.
    """
    global _pinned_project_id
    with _pin_lock:
        _pinned_project_id = project_id
    if project_id:
        logger.info("[project-context] process pinned to project %s", project_id)


def pinned_project_id() -> Optional[str]:
    """The project this process is pinned to, or ``None`` when unpinned."""
    return _pinned_project_id


def active_project_id() -> Optional[str]:
    """The project bound to the current request or process, if any.

    Returns ``None`` when nothing is bound — callers then fall back to the
    ``current`` flag, which is what a supervisor process and the CLI want.
    """
    return _active_project_id.get() or _pinned_project_id


@contextmanager
def use_project(project_id: Optional[str]) -> Iterator[Optional[str]]:
    """Bind *project_id* for the duration of the block.

    A falsy *project_id* is a no-op binding: the caller did not name a project,
    so resolution should keep falling through rather than pin to nothing.
    """
    if not project_id:
        yield None
        return
    token = _active_project_id.set(project_id)
    try:
        yield project_id
    finally:
        _active_project_id.reset(token)


def set_active_project(project_id: Optional[str]):
    """Bind *project_id* and return the token to reset with.

    The imperative form of :func:`use_project`, for the streaming chat path,
    where the binding has to outlive the function that establishes it (the
    generator is resumed by the event loop long after the setup call returns).
    """
    return _active_project_id.set(project_id)


def reset_active_project(token) -> None:
    """Undo a :func:`set_active_project`, tolerating an already-unwound token.

    A streaming response can be abandoned by the client mid-generation, which
    unwinds the generator from a different task than the one that set the
    binding; resetting a token from the wrong context raises rather than
    corrupting anything, and there is nothing to repair when it does.
    """
    if token is None:
        return
    try:
        _active_project_id.reset(token)
    except ValueError:
        pass


def bind(fn: Callable) -> Callable:
    """Wrap *fn* so it runs under the caller's project binding.

    Capture happens now, on the submitting thread; the binding is applied
    inside the worker.  Use for any hop into a thread that will resolve a
    project — ``run_in_executor``, ``Thread(target=...)``, a pool ``submit``.
    """
    project_id = active_project_id()

    @functools.wraps(fn)
    def _bound(*args, **kwargs):
        token = _active_project_id.set(project_id)
        try:
            return fn(*args, **kwargs)
        finally:
            _active_project_id.reset(token)

    return _bound


class ProjectBoundExecutor(ThreadPoolExecutor):
    """A thread pool that carries the submitter's project binding into workers.

    Drop-in for :class:`~concurrent.futures.ThreadPoolExecutor` anywhere the
    submitted work resolves a project — the orchestrator's DAG and I/O pools,
    the MCP explorer's fan-out.  Without it those workers would read whichever
    project happened to be bound to the pool thread from an earlier request.
    """

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
        return super().submit(bind(fn), *args, **kwargs)


class ProjectContext:
    """An explicit ``(workspace, project)`` pair, resolved once.

    Passed where a callee needs the project as a value rather than as ambient
    state — a background task owns one for its whole run, and a per-project
    child builds one at startup.  Resolution is late (``project`` is looked up
    on each access) so a reload of the workspace is picked up rather than
    pinning a stale object.
    """

    __slots__ = ("workspace", "project_id")

    def __init__(self, workspace: "Workspace", project_id: str) -> None:
        self.workspace = workspace
        self.project_id = project_id

    @classmethod
    def resolve(cls, workspace: "Workspace") -> Optional["ProjectContext"]:
        """Build a context from whatever is bound, or ``None`` if nothing is."""
        project_id = active_project_id()
        if not project_id or workspace is None:
            return None
        return cls(workspace, project_id)

    @property
    def project(self) -> Optional["Project"]:
        return next(
            (p for p in (self.workspace.projects or []) if p.id == self.project_id),
            None,
        )

    @property
    def project_dir(self) -> Optional[str]:
        project = self.project
        return project.project_dir if project else None

    @contextmanager
    def bound(self) -> Iterator["ProjectContext"]:
        """Make this context the ambient binding for the duration of the block."""
        with use_project(self.project_id):
            yield self

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ProjectContext(project_id={self.project_id!r})"
