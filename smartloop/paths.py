"""Where the user's home actually is, once a store's sandbox has moved it.

``$HOME`` is what every tilde expands against, and both stores we ship
through rewrite it to a private per-app directory:

* the macOS App Sandbox (Mac App Store) points it at
  ``~/Library/Containers/<bundle-id>/Data``
* snapd, under strict confinement, points it at
  ``~/snap/<snap-name>/<revision>``

Neither is a directory the user has ever put a file in, so anything they
wrote a ``~`` into — a project directory, an MCP server's ``command``, a
document handed to a parser — expanded to a path that was never going to
exist, and the framework home itself landed inside the container instead of
at ``~/.smartloop``.

What the sandboxes rewrite is the environment variable, not the passwd
database, so the record behind ``getpwuid`` still names the real home; snapd
2.46 and later also states it outright in ``SNAP_REAL_HOME``.  That is the
whole of the trick here.

Resolving the path is not the same as being allowed to read it: under the App
Sandbox the user has to pick the directory (see the ``files.user-selected``
entitlement in ``packaging/macos``), and a strict snap reaches the real home
only through the ``home`` interface, which covers non-hidden files — a
``personal-files`` plug is what grants ``~/.smartloop``.  This module answers
*where*, and the caller still has to cope with a denial.
"""

from __future__ import annotations

import os
import re

#: How macOS spells the App Sandbox container it rewrites ``$HOME`` to.
_SANDBOX_HOME_MARKER = "/Library/Containers/"

#: How snapd spells ``$SNAP_USER_DATA``, which it rewrites ``$HOME`` to:
#: ``<real home>/snap/<snap-name>/<revision>``.
_SNAP_HOME_RE = re.compile(r"^(?P<home>.+)/snap/[^/]+/[^/]+/?$")


def _unconfined_parent(home: str) -> str | None:
    """The real home *home* is a sandbox directory inside, or ``None``.

    ``None`` means ``$HOME`` is not a confinement directory at all and so is
    to be taken at its word — a service unit or a test that sets it means it,
    and only the two spellings below are ever overridden.
    """
    index = home.find(_SANDBOX_HOME_MARKER)
    if index > 0:
        return home[:index]
    # A plain ``~/snap/...`` is a perfectly ordinary directory to have; it is
    # only the rewritten home when we are in fact running inside a snap, which
    # snapd says by setting ``SNAP_NAME``.
    if os.environ.get("SNAP_NAME"):
        match = _SNAP_HOME_RE.match(home)
        if match:
            return match.group("home")
    return None


def real_home() -> str:
    """The user's home, reading past a store sandbox's rewrite of ``$HOME``."""
    home = os.environ.get("HOME") or ""
    parent = _unconfined_parent(home)
    if parent is None:
        # Not confined, so ``$HOME`` is the answer and is taken as given.
        return os.path.expanduser("~")

    # snapd >= 2.46 states the real home outright.  Only consulted once
    # ``$HOME`` has shown we are the confined process: a snap that launches
    # something else — a terminal emulator starting a shell, say — leaks its
    # whole ``SNAP_*`` block into a child that is not confined at all, and
    # that child's ``$HOME`` means exactly what it says.
    snap_home = os.environ.get("SNAP_REAL_HOME")
    if snap_home:
        return snap_home

    # Neither sandbox rewrites the passwd record.
    try:
        import pwd
        return pwd.getpwuid(os.getuid()).pw_dir or parent
    except Exception:
        # No passwd record to read (or no ``pwd`` module at all, on Windows).
        # The directory the confinement path sits in is the next best answer.
        return parent


def confined_home() -> str | None:
    """The sandbox directory ``$HOME`` was rewritten to, or ``None``.

    The counterpart to :func:`real_home`: where the framework *used* to put
    its state on a store build, and so where an install made before the home
    was resolved properly still has it.  ``None`` off a store build, which is
    every developer machine and the Developer ID build.
    """
    home = os.environ.get("HOME") or ""
    return home if _unconfined_parent(home) is not None else None


def expand_home(path: str) -> str:
    """``~``-expand *path* against :func:`real_home`.

    ``~otheruser/...`` names somebody else's home and is left to
    ``expanduser``, which is the only thing that knows how to resolve it.
    """
    if not path.startswith("~"):
        return path
    if path == "~":
        return real_home()
    if path.startswith("~/"):
        return os.path.join(real_home(), path[2:])
    return os.path.expanduser(path)
