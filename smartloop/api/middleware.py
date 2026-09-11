"""ASGI middleware binding each request to the project it addresses.

Most routers already take a ``project_id`` and pass it down explicitly, but the
service layer still falls back to "the current project" in plenty of places
(and did so, until now, by reading a process-wide flag that any concurrent
request could move).  Binding the project for the request's whole lifetime
makes that fallback resolve to the project the caller actually asked about.

This is deliberately a *pure* ASGI middleware rather than a
``BaseHTTPMiddleware``: the latter runs the downstream app in a separate anyio
task, and a streaming response then outlives the middleware's own frame.  Here
``await self.app(...)`` runs in the caller's task, so the binding covers the
response body as it streams, not just the handler that returned it.

The chat endpoint is the exception — it carries ``project_id`` in the request
body, which cannot be read here without consuming the stream, so it binds
itself (``chat._bind_request_project``).
"""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import parse_qs

from smartloop.workspace.context import use_project

#: ``/v1/projects/<id>/...`` — the id segment as the routers declare it.
_PATH_PROJECT_RE = re.compile(r"/projects/([^/]+)")

#: Verbs on the project collection.  They sit in the same slot in the path as
#: a project id and are emphatically not one, so the pattern above has to be
#: told about them: reading ``import`` as a project bound a request to a
#: project called "import", and the first thing that resolved the binding
#: warned about a workspace member that could never exist.
#:
#: Listed rather than pattern-matched so adding one is a decision, not an
#: accident.  The supervisor's routing reads the same set to decide what never
#: to forward (see :func:`smartloop.supervisor.routing.project_to_proxy`) —
#: one list, because a path that names no project must neither bind one nor be
#: proxied to one.
_COLLECTION_ACTIONS = frozenset({"/v1/projects/import"})


def _project_from_query(scope: dict) -> Optional[str]:
    """The ``project_id`` in a request's query string, if there is one.

    Split out because the supervisor's routing asks a narrower question about
    the path — which project a request should be *forwarded* to, which is not
    every project a request is about — but the same one about the query.  See
    :func:`smartloop.supervisor.routing.project_to_proxy`.
    """
    raw_query = scope.get("query_string") or b""
    if not raw_query:
        return None
    try:
        values = parse_qs(raw_query.decode("latin-1")).get("project_id")
    except Exception:
        return None
    return values[0] if values else None


def _project_from_scope(scope: dict) -> Optional[str]:
    """The project this request addresses, from its path or query string."""
    path = scope.get("path") or ""
    if path in _COLLECTION_ACTIONS:
        return None
    match = _PATH_PROJECT_RE.search(path)
    if match:
        return match.group(1) or None
    return _project_from_query(scope)


class ProjectScopeMiddleware:
    """Bind the addressed project for the duration of each HTTP request."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        with use_project(_project_from_scope(scope)):
            await self.app(scope, receive, send)
