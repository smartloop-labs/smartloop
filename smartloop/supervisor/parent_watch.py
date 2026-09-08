"""Tie a project agent's lifetime to the supervisor that spawned it.

A child is deliberately detached from the supervisor's process group (see
``_detached_process_kwargs`` in :mod:`smartloop.supervisor.pool`) so a Ctrl-C
in the terminal reaches the supervisor alone, which then stops the children
itself, in order, after the model host they talk to.  That detachment is also
what makes an orphan possible: nothing links a child to its parent afterwards,
so a supervisor that is ``SIGKILL``-ed, OOM-killed, or lost with its terminal
leaves children running.  Each one keeps a project's task queue, its MCP stdio
grandchildren and its share of the device, answers on a port recorded in a
registry nobody reads any more, and never exits — a uvicorn server with no one
left to signal it runs until the machine reboots.

The link is put back here, from the child's side: the supervisor passes its pid
and start time in the environment (:func:`parent_environment`) and the child
watches them (:func:`start_parent_watch`).

Watched rather than inherited because inheritance is not portable — the
"die with my parent" primitive is ``PR_SET_PDEATHSIG`` on Linux, ``kqueue`` on
BSD, job objects on Windows, and this repo ships all three — and because a
poll can do something inheritance cannot: survive a *planned* handover.
Adopting the children an earlier supervisor left is a feature, so a child whose
parent has vanished looks in the registry for a live replacement before giving
up, and re-targets onto it when there is one.  Only a child with no parent and
no successor after :data:`DEFAULT_GRACE_SECONDS` shuts itself down.

Start time is checked alongside the pid because pids are recycled: on a busy
machine the number a dead supervisor used can belong to something unrelated
within minutes, and a child that mistakes a stranger for its parent is an
orphan again — just one that looks supervised.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)

#: The supervisor's pid, and the start time that disambiguates a recycled one.
PARENT_PID_ENV = "SLP_PARENT_PID"
PARENT_STARTED_AT_ENV = "SLP_PARENT_STARTED_AT"

#: How often the child checks that it still has a parent.  Cheap — one
#: ``/proc`` read (or the Windows equivalent) per tick — so the interval is set
#: by how long an orphan may linger, not by what the check costs.
DEFAULT_POLL_SECONDS = 5.0

#: How long a child keeps serving after its parent disappears, waiting for a
#: replacement supervisor to come up and adopt it.  Long enough to cover a
#: restart, short enough that an orphan is measured in seconds.
DEFAULT_GRACE_SECONDS = 30.0

#: Two ``create_time`` readings of the same process can differ in the last
#: decimal across platforms; a second of slack is far below pid-reuse latency.
_START_TIME_TOLERANCE = 1.0


def parent_environment() -> dict:
    """The environment a spawning supervisor adds so its child can watch it."""
    pid = os.getpid()
    env = {PARENT_PID_ENV: str(pid)}
    started_at = process_started_at(pid)
    if started_at is not None:
        env[PARENT_STARTED_AT_ENV] = repr(started_at)
    return env


def process_started_at(pid: Optional[int]) -> Optional[float]:
    """When *pid* started, or ``None`` when that cannot be read."""
    if not pid or pid <= 0:
        return None
    try:
        import psutil

        return psutil.Process(pid).create_time()
    except Exception:
        return None


def parent_is_alive(pid: Optional[int], started_at: Optional[float]) -> bool:
    """Whether *pid* is still the process that started at *started_at*.

    Liveness itself is :func:`~smartloop.supervisor.registry.process_alive`,
    which spells the question portably — ``os.kill(pid, 0)`` is POSIX-only, and
    on Windows would terminate the process it is asking about.  The start time
    is the part that matters here: without it a recycled pid reads as a living
    parent and the child never notices it was orphaned.
    """
    from smartloop.supervisor.registry import process_alive

    if not process_alive(pid):
        return False
    if started_at is None:
        return True
    observed = process_started_at(pid)
    if observed is None:
        # The process is there but we may not inspect it — being unable to
        # tell is not evidence of an orphan, so keep serving.
        return True
    return abs(observed - started_at) <= _START_TIME_TOLERANCE


def _configured(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("[parent-watch] %s=%r is not a number; using %.0fs",
                       name, raw, default)
        return default
    return value if value > 0 else default


class ParentWatch:
    """Polls for the supervisor, and calls back when there is not one.

    Runs on a daemon thread: the watch must never be the reason a child that
    *is* supervised stays up, and the callback it fires — an orderly uvicorn
    shutdown — has to be able to join it.
    """

    def __init__(
        self,
        pid: int,
        started_at: Optional[float] = None,
        home_dir: Optional[str] = None,
        on_orphaned: Optional[Callable[[], None]] = None,
        poll_interval: float = DEFAULT_POLL_SECONDS,
        grace: float = DEFAULT_GRACE_SECONDS,
    ) -> None:
        self.pid = pid
        self.started_at = started_at
        self.home_dir = home_dir
        self.on_orphaned = on_orphaned
        self.poll_interval = poll_interval
        self.grace = grace
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> "ParentWatch":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._run, name="parent-watch", daemon=True,
        )
        self._thread.start()
        logger.info(
            "[parent-watch] watching supervisor pid %d (every %.0fs, %.0fs grace)",
            self.pid, self.poll_interval, self.grace,
        )
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Stop watching.  Called on an ordinary shutdown, which is not an orphaning."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    # -- the loop -----------------------------------------------------------

    def _run(self) -> None:
        missing_since: Optional[float] = None
        while not self._stop.wait(self.poll_interval):
            if parent_is_alive(self.pid, self.started_at):
                missing_since = None
                continue

            successor = self._successor()
            if successor is not None:
                pid, started_at = successor
                logger.info(
                    "[parent-watch] supervisor pid %d is gone; adopted by pid %d",
                    self.pid, pid,
                )
                self.pid, self.started_at = pid, started_at
                missing_since = None
                continue

            now = time.monotonic()
            if missing_since is None:
                missing_since = now
                logger.warning(
                    "[parent-watch] supervisor pid %d is gone; waiting %.0fs for "
                    "a replacement before shutting down", self.pid, self.grace,
                )
                continue
            if now - missing_since >= self.grace:
                logger.warning(
                    "[parent-watch] no supervisor after %.0fs; this agent is an "
                    "orphan, shutting it down", self.grace,
                )
                self._fire()
                return

    def _successor(self) -> Optional[Tuple[int, Optional[float]]]:
        """A live supervisor that is not the one we were watching, if any.

        Read from the registry rather than passed in, because a replacement
        supervisor is by definition a process that did not exist when this
        child was spawned.  It records itself before it adopts anybody, so a
        pid here means the handover is underway even if this child has not been
        routed to yet.
        """
        if not self.home_dir:
            return None
        try:
            from smartloop.supervisor.registry import AgentRegistry

            supervisor = AgentRegistry(self.home_dir).load().supervisor
        except Exception:  # a registry we cannot read is not a successor
            return None
        pid = supervisor.get("pid")
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return None
        if pid == self.pid or pid == os.getpid():
            return None
        started_at = supervisor.get("started_at")
        started_at = float(started_at) if isinstance(started_at, (int, float)) else None
        if not parent_is_alive(pid, started_at):
            return None
        return pid, started_at

    def _fire(self) -> None:
        if self.on_orphaned is None:
            return
        try:
            self.on_orphaned()
        except Exception:
            logger.exception("[parent-watch] orphan shutdown failed")


def start_parent_watch(
    on_orphaned: Callable[[], None],
    home_dir: Optional[str] = None,
) -> Optional[ParentWatch]:
    """Watch the parent named in the environment, if this process has one.

    Returns ``None`` when :data:`PARENT_PID_ENV` is unset — the single shared
    server and a hand-run ``python -m smartloop.api.project_agent`` were not
    spawned by a supervisor and have nothing to be orphaned from.
    """
    raw = (os.environ.get(PARENT_PID_ENV) or "").strip()
    if not raw:
        return None
    try:
        pid = int(raw)
    except ValueError:
        logger.warning("[parent-watch] %s=%r is not a pid; not watching",
                       PARENT_PID_ENV, raw)
        return None
    if pid <= 0 or pid == os.getpid():
        return None

    raw_started = (os.environ.get(PARENT_STARTED_AT_ENV) or "").strip()
    try:
        started_at = float(raw_started) if raw_started else None
    except ValueError:
        started_at = None

    return ParentWatch(
        pid=pid,
        started_at=started_at,
        home_dir=home_dir,
        on_orphaned=on_orphaned,
        poll_interval=_configured("SLP_PARENT_WATCH_INTERVAL_S",
                                  DEFAULT_POLL_SECONDS),
        grace=_configured("SLP_PARENT_WATCH_GRACE_S", DEFAULT_GRACE_SECONDS),
    ).start()
