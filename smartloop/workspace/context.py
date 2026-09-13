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

Ownership
---------
Binding says which project a piece of work is *about*; :func:`delegate_projects`
and :func:`own_project` say whose state a process may *write*.  The supervisor
delegates every project's own file to the child pinned to it, and claims one
back only for the operations that straddle a child's lifetime — create, import,
delete.  :meth:`~smartloop.workspace.workspace.Workspace.save` reads both.

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


#: Whether this process has handed every project's own state to someone else.
#: Set by the supervisor, which spawns a child per project: from that moment the
#: files under each ``project_dir`` belong to those children, and the supervisor
#: keeps only ``config.json``.  Not a context variable for the same reason the
#: pin is not — it is a property of the process.
_projects_delegated: bool = False


def delegate_projects(delegated: bool = True) -> None:
    """Declare that per-project state is owned by other processes.

    Called by the supervisor when its pool comes up, and cleared when it goes
    down.  :meth:`~smartloop.workspace.workspace.Workspace.save` reads this to
    decide whether it may write every project's file or only the workspace
    record — see :func:`own_project` for the exception.
    """
    global _projects_delegated
    with _pin_lock:
        _projects_delegated = delegated
    logger.info("[project-context] per-project state delegated=%s", delegated)


def projects_delegated() -> bool:
    """Whether another process owns each project's own state."""
    return _projects_delegated


#: The project whose file this unit of work is entitled to write, in a process
#: that has otherwise delegated them.
_owned_project_id: ContextVar[Optional[str]] = ContextVar(
    "smartloop_owned_project_id", default=None
)


@contextmanager
def own_project(project_id: Optional[str]) -> Iterator[Optional[str]]:
    """Claim *project_id*'s own state for the duration of the block.

    The supervisor delegates project files to its children, but three
    operations happen either side of a child's existence and must still write
    one: creating a project, importing one, and deleting one.  Claiming it says
    which file this save is allowed to touch, so the claim stays as narrow as
    the operation instead of re-opening the whole workspace to writes.
    """
    if not project_id:
        yield None
        return
    token = _owned_project_id.set(project_id)
    try:
        yield project_id
    finally:
        _owned_project_id.reset(token)


def owned_project_id() -> Optional[str]:
    """The project claimed by :func:`own_project`, if any."""
    return _owned_project_id.get()


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


@contextmanager
def unbind_project() -> Iterator[None]:
    """Resolve as if no project were named, for the duration of the block.

    :func:`use_project` deliberately treats a falsy id as a no-op — a caller
    that named no project should keep falling through rather than pin to
    nothing — so this is the explicit form for the one case that means it:
    work that has just removed the project it is bound to.

    Deleting a project arrives on a path that names it, so the request stays
    bound to it after the removal.  Anything resolving that binding then warns
    about an absence the request was asked to create, and
    :meth:`~smartloop.workspace.workspace.Workspace.save` does resolve it —
    ``current_project`` is a computed field, so every ``model_dump`` reaches
    it.  The warning is the same sentence a genuinely stale binding produces,
    which is reason enough not to emit it where nothing is wrong.

    The process pin is left alone: a pinned child never reaches an operation
    like this, because removing a project is the workspace owner's work.
    """
    token = _active_project_id.set(None)
    try:
        yield
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
