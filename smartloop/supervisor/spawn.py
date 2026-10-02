"""Fork-free process spawning for detached children.

``subprocess.Popen(..., start_new_session=True, close_fds=True)`` — the shape
the process pool has always used to put a child in its own session — cannot go
through ``posix_spawn`` in CPython: the fallback conditions in
``Popen._execute_child`` reject ``posix_spawn`` whenever ``start_new_session``
*or* ``close_fds`` is requested, so the child is forked instead.

A ``fork()`` taken while the supervisor's gRPC server is mid-RPC is not safe:
grpc-python registers atfork handlers to quiesce the runtime, and when another
thread is inside a call at fork time those handlers cannot take the lock, so
grpc logs ``Other threads are currently calling into gRPC, skipping fork()
handlers`` and forks anyway.  That ``fork()`` is not on the child's side of the
transition — the child ``exec()``s microseconds later, so a fine performance
detail for the child, but it *is* a real risk in the parent.  This module
spawns the same detached child through ``os.posix_spawn``, which never forks
the parent at all: the kernel places the child image directly.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time

__all__ = ["spawn_detached", "DetachedProcess", "popen_detached_devnull"]


def _fd_dir() -> str:
    """A directory enumerating this process's open file descriptors."""
    for candidate in ("/dev/fd", "/proc/self/fd"):
        if os.path.isdir(candidate):
            return candidate
    raise OSError("no /dev/fd or /proc/self/fd to enumerate open file descriptors")


def _still_open(fd: int) -> bool:
    """Whether *fd* is open at this instant."""
    try:
        os.fstat(fd)
        return True
    except OSError:
        return False


def spawn_detached(command: list, env: dict | None = None) -> int:
    """``posix_spawn`` a child in a new session, inheriting ``stdout``/``stderr``.

    Comparable to ``subprocess.Popen(command, env=env, stdin=DEVNULL,
    start_new_session=True, close_fds=True)`` but without ever forking: the
    child gets its own session, its stdin is ``/dev/null``, and every other
    inherited descriptor is closed in the new image.

    Returns the child pid.  Raises on failure (the caller falls back to the
    ``subprocess`` path rather than losing the ability to spawn).
    """
    if os.name == "nt" or not hasattr(os, "posix_spawn"):
        raise AttributeError("os.posix_spawn is not available on this platform")

    command = list(command)
    if not command:
        raise ValueError("command must name a program to spawn")
    if env is None:
        env = dict(os.environ)

    devnull = os.open(os.devnull, os.O_RDWR)
    dirfd = None
    try:
        # Enumerate the inherited descriptors.  The enumeration keeps the
        # directory fd it iterates open, and releasing the iterator also
        # releases a transient copy of that fd for every open descriptor —
        # so the snapshot always contains descriptors that are already closed
        # by the time the spawn runs.  One of them is left out of the file
        # actions (an EBADF is not the failure we want); the rest are
        # re-verified open a moment before the spawn, closing only what is
        # genuinely inherited, so a stale entry cannot abort the spawn.
        dirfd = os.open(_fd_dir(), os.O_RDONLY)
        with os.scandir(dirfd) as entries:
            candidates = [
                int(entry.name) for entry in entries if entry.name.isdigit()
            ]
        close_fds = [
            fd for fd in candidates
            if fd > 2 and _still_open(fd)
        ]

        # The file-actions surface differs across CPython builds: some expose
        # the ``PosixSpawnFileActions`` builder, others a flat sequence of
        # ``(POSIX_SPAWN_*)`` tuples.  Both describe the same child operation.
        try:  # tuple-sequence API (e.g. CPython 3.8+ on macOS)
            actions = [(os.POSIX_SPAWN_DUP2, devnull, 0)]
            actions += [(os.POSIX_SPAWN_CLOSE, fd) for fd in close_fds]
            kwargs = {"file_actions": actions, "setsid": True}
        except AttributeError:  # older builder API
            builder = getattr(os, "PosixSpawnFileActions", None)
            if builder is None:
                raise AttributeError(
                    "no usable posix_spawn file_actions API on this platform"
                )
            actions = builder()
            actions.adddup2(devnull, 0)
            for fd in close_fds:
                actions.addclose(fd)
            kwargs = {"file_actions": actions, "setsid": True}

        spawn = os.posix_spawnp if hasattr(os, "posix_spawnp") else os.posix_spawn
        try:
            return spawn(command[0], command, env, **kwargs)
        except TypeError:
            kwargs.pop("setsid", None)
            return spawn(command[0], command, env, **kwargs)
    finally:
        if dirfd is not None:
            os.close(dirfd)
        os.close(devnull)


class DetachedProcess:
    """The ``subprocess.Popen`` surface callers rely on, for a spawned pid.

    ``os.posix_spawn`` hands back a raw pid rather than a handle object, and
    nothing here needs a pipe or a handle that keeps the process tied to its
    parent: the pool polls for exit, signals by pid, and waits.
    """

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        """The exit code once the child has exited, else ``None``."""
        if self.returncode is not None:
            return self.returncode
        try:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
        except (ChildProcessError, InterruptedError):
            return self.returncode
        if pid:
            self.returncode = os.waitstatus_to_exitcode(status)
        return self.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        """Block until the child exits (or *timeout* expires) and reap it."""
        if self.returncode is None:
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                result = self.poll()
                if result is not None:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(self.pid, timeout)
                time.sleep(0.05)
        return self.returncode

    def terminate(self) -> None:
        os.kill(self.pid, signal.SIGTERM)

    def kill(self) -> None:
        os.kill(self.pid, signal.SIGKILL)


def popen_detached_devnull(command: list, env: dict | None = None,
                           output: str | None = None):
    """Spawn *command* detached, falling back to a forking ``Popen``.

    Returns a ``DetachedProcess`` when ``posix_spawn`` is available and the
    spawn succeeds, and a ``subprocess.Popen`` (same authors: stdin is
    ``/dev/null``, own session via ``start_new_session``) otherwise — so a
    platform or runtime that cannot use ``posix_spawn`` still gets its child.

    On Windows the child does not inherit our stdout/stderr: they are whatever
    launched the supervisor gave it (a log file, the desktop app's pipes, a
    console), and passing some of those through ``CreateProcess``'s handle list
    fails with ``[WinError 50] The request is not supported``.  Its output goes
    to *output* instead, appended, or nowhere when that is ``None``.  It runs
    without a console window too: a supervisor started without a console would
    otherwise give every child a visible one of its own.
    """
    if os.name != "nt":
        try:
            return DetachedProcess(spawn_detached(command, env))
        except (AttributeError, OSError, ValueError):
            pass
        return subprocess.Popen(
            command, env=env, stdin=subprocess.DEVNULL, stdout=None, stderr=None,
            start_new_session=True,
        )

    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    if output is None:
        return subprocess.Popen(
            command, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
    # The child holds its own copy of the handle; ours can close at once.
    with open(output, "ab") as sink:
        return subprocess.Popen(
            command, env=env, stdin=subprocess.DEVNULL,
            stdout=sink, stderr=subprocess.STDOUT,
            creationflags=flags,
        )
