"""MCP connections a file describes, and how to launch them.

A definition is the whole of what a ``.toml`` file says about one server: what
it is called, and how to reach it — a URL for a remote one, a command line for
a local one.  Dropping such a file into ``~/.smartloop/mcp.d/`` installs the
connection (see :mod:`smartloop.mcp.definition_loader` and
``POST /projects/{id}/mcp/definition``), which is what lets a server be added
without a code change or a trip through the UI.

Nothing here indexes anything.  A connection's *tools* are enumerated once at
registration and that list is the catalog a step is resolved against (see
:class:`~smartloop.agents.mcp.mcp_agent.MCPAgent`); its *content* is read by
calling those tools when a question needs them.  This module used to also
carry crawl plans — declarative walks that vectorized a server's pages, issues
and cards ahead of time — and they are gone: a walk went stale between
questions, only covered the listings someone had written a plan for, and
answered from a snapshot when the tool that would give the current answer was
one call away.  A file still carrying those fields installs anyway — see
:data:`RETIRED_FIELDS` — because the connection it names is unaffected by the
indexing it can no longer ask for.
"""

from __future__ import annotations

import re
import shlex
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from smartloop.paths import expand_home, real_home  # noqa: F401  (re-exported)


def normalize_name(name: str) -> str:
    """Lowercase *name* with separators stripped: ``notion-search`` → ``notionsearch``.

    The one place tool-name spelling is normalised.  Anything comparing tool or
    argument names goes through it, so ``notion-search``/``notion_search``/
    ``NotionSearch`` are one name everywhere rather than in each caller's own
    approximation of the rule.
    """
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


# ----------------------------------------------------------------- launchers
#
# A local server is started by a command line, and which command that is
# belongs to the server: a Python one runs under `uvx`, a Node one under `npx`,
# and plenty of others are a module, a script or an absolute path.  Nothing
# here assumes one of them.

def split_command_line(text: str) -> Tuple[Optional[str], List[str]]:
    """``"npx -y a-mcp"`` → ``("npx", ["-y", "a-mcp"])``.

    Quoting is honoured (``shlex``), so an argument with a space in it survives
    as one argument.  An unclosed quote is a plain error naming the field, which
    is better than the silent half-command a naive split would produce.
    """
    parts = shlex.split((text or "").strip())
    if not parts:
        return None, []
    return parts[0], parts[1:]


def launcher_for(package: str) -> Tuple[str, List[str]]:
    """``(command, args)`` to run a bare package name.

    A scoped npm name (``@scope/pkg``) can only be an npm package, so it runs
    under ``npx -y``.  Anything else falls to ``uvx``, which is what typing a
    bare name into "Add your own" does — a file and that field have to install
    the same server rather than two subtly different ones.  Every other
    launcher is named explicitly, either as a full command line in ``package``
    or as ``command`` + ``args``.
    """
    if package.startswith("@"):
        return "npx", ["-y", package]
    return "uvx", [package]


# ------------------------------------------------------------- home directory
#
# A definition writes a local server's path the way anyone would — `command =
# "~/.local/bin/craft"` — and `$HOME` is what expands the tilde.  Under the
# store sandboxes that env var is not the user's home, so the tilde is
# expanded against :func:`smartloop.paths.real_home` instead; re-exported here
# because this is where `launch` reaches for it.

# --------------------------------------------------------------- retired keys
#
# Crawling is gone, but the files that configured it are still on people's
# disks, and a file that says `index_every = "15m"` beside a url or a package
# name still names a working connection.  So the retired keys are dropped
# rather than rejected: such a file installs, with a logged line naming the
# fields that did nothing.  Everything *else* unknown stays an error —
# `packag = ` has to name the typo rather than quietly install a server with
# no command.

#: Per-entry fields the old indexing format used, accepted and ignored.
RETIRED_FIELDS = ("crawl", "index", "index_every", "step", "extend")


def retired_fields(raw: Mapping[str, Any]) -> Tuple[str, ...]:
    """Which retired indexing fields *raw* declares, in the order it lists them."""
    return tuple(key for key in raw if key in RETIRED_FIELDS)


class MCPDefinition(BaseModel):
    """One MCP server, as a file describes it.

    Two parts: **identity** (``name``, plus the ``slug``/``match`` that tie the
    file to a server already installed) and **connection** (``url`` for a
    remote server, or ``package``/``command`` + ``args`` for one to run
    locally).  The connection is optional — a file that only names a server
    still matches an installed one, which is how a definition dropped in after
    the fact attaches to a connection added through the UI.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    # A URL or a package name — the same two things the "Add your own" field
    # takes, because they are the two ways anyone actually names a server.
    # Which one is present is what makes a connection remote or local; there is
    # no third thing to declare.
    url: Optional[str] = None
    # A command line, not just a PyPI name: `package = "yahoo-finance-mcp"`
    # runs under uvx, `package = "npx -y @foo/bar-mcp"` runs under npx, and
    # `package = "python -m my.server"` runs as written.  See `launch`.
    package: Optional[str] = None
    # The explicit form, used exactly as given — for a server whose launcher
    # takes flags, or one that is a module or an absolute path.
    #   command = "npx"
    #   args = ["-y", "some-mcp"]
    command: Optional[str] = None
    args: Tuple[str, ...] = ()
    env: Dict[str, str] = Field(default_factory=dict)
    cwd: Optional[str] = None
    # For the remote servers that take a bearer token; OAuth is negotiated at
    # install time and needs nothing here.
    token: Optional[str] = None
    # Defaults to `name`: a file usually describes one server and names it once.
    slug: Optional[str] = None
    # Extra substrings matched against an installed server's name, package and
    # URL.  `name` and `slug` are always tried, so this is for the aliases.
    match: Tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _drop_retired(cls, data: Any) -> Any:
        """Ignore the indexing fields a file written for the old format carries.

        See :data:`RETIRED_FIELDS`.  Dropped in the model, so every way of
        building a definition — a drop-in file, an upload, a test — treats
        them alike.
        """
        if isinstance(data, dict) and retired_fields(data):
            return {k: v for k, v in data.items() if k not in RETIRED_FIELDS}
        return data

    @field_validator("args", mode="before")
    @classmethod
    def _coerce_args(cls, value: Any) -> Any:
        """Accept ``args = "-y some-mcp"`` as well as a list.

        A command line is how everyone writes these down (and how every other
        MCP client shows them), so a string is split the way a shell would
        rather than rejected — or, worse, iterated one character per argument.
        """
        if isinstance(value, str):
            return shlex.split(value.strip())
        return value

    @model_validator(mode="after")
    def _check_connection(self) -> "MCPDefinition":
        """A server is named one way, not two."""
        if self.args and not (self.command or self.package):
            raise ValueError(
                f"'{self.name}': args without a command — add `command = ` "
                f"(the program to run) or use `package = ` for the whole "
                f"command line"
            )
        if self.url and (self.package or self.command):
            raise ValueError(
                f"'{self.name}': a url and a package name are alternatives — "
                f"a server is reached over HTTP or run as a process, not both"
            )
        if self.package and self.command:
            raise ValueError(
                f"'{self.name}': package is shorthand for command + args, so "
                f"setting both says the same thing twice"
            )
        return self

    @property
    def kind(self) -> Optional[str]:
        """``"local"``, ``"remote"``, or None when the file declares no connection."""
        if self.url:
            return "remote"
        if self.package or self.command:
            return "local"
        return None

    @property
    def launch(self) -> Tuple[Optional[str], List[str]]:
        """``(command, args)`` for a local server.

        ``command`` + ``args`` are taken as written; a ``command`` that carries
        its own arguments (``command = "npx -y a-mcp"``) is split, with any
        ``args`` appended.  ``package`` is the shorthand and it is a command
        line too: a bare name gets a launcher from :func:`launcher_for`, and
        anything with a space in it is already a command and is used as one.

        The launcher is never assumed beyond that single bare-name case.  A
        file describing a Node server had no way to say so while ``package``
        meant ``uvx <package>``, which left every npm-published MCP server
        unreachable from a definition file — the thing the field exists for.
        """
        if self.command:
            command, embedded = split_command_line(self.command)
            return expand_home(command) if command else None, [*embedded, *self.args]
        if self.package:
            command, args = split_command_line(self.package)
            if command is None:
                return None, []
            if args:
                return command, args
            return launcher_for(command)
        return None, list(self.args)

    @property
    def key(self) -> str:
        """The slug this definition is registered and looked up under."""
        return (self.slug or self.name).strip().lower()

    @property
    def match_terms(self) -> Tuple[str, ...]:
        """Every substring that ties this definition to an installed server."""
        return tuple(dict.fromkeys((self.key, self.name.lower(), *self.match)))


def _server_haystack(server: Any) -> str:
    """Lowercased name + package + url of an installed server, for matching."""
    parts = [
        str(getattr(server, "name", "") or ""),
        str(getattr(server, "pkg", "") or ""),
    ]
    remote = getattr(server, "remote_config", None)
    if remote is not None:
        parts.append(str(getattr(remote, "server_url", "") or ""))
    config = getattr(server, "config", None)
    if config is not None:
        parts.append(str(getattr(config, "command", "") or ""))
        parts.extend(str(a) for a in (getattr(config, "args", None) or []))
    return " ".join(parts).lower()


def definition_for(server: Any) -> Optional[MCPDefinition]:
    """The dropped-in definition that describes *server*, or None.

    Matched on the catalog slug first — provenance the backend established
    itself — then on the substrings a definition declares.  Used to find the
    file to delete when a connection is removed, so a re-added server does not
    silently inherit an old file.
    """
    from smartloop.mcp.definition_loader import definitions

    try:
        found = definitions()
    except Exception:
        return None
    if not found:
        return None

    slug = (getattr(server, "catalog_slug", None) or "").strip().lower()
    if slug and slug in found:
        return found[slug]

    haystack = _server_haystack(server)
    for definition in found.values():
        if any(term and term in haystack for term in definition.match_terms):
            return definition
    return None
