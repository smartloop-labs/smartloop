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


def _project_from_scope(scope: dict) -> Optional[str]:
    """The project this request addresses, from its path or query string."""
    match = _PATH_PROJECT_RE.search(scope.get("path") or "")
    if match:
        return match.group(1) or None

    raw_query = scope.get("query_string") or b""
    if not raw_query:
        return None
    try:
        values = parse_qs(raw_query.decode("latin-1")).get("project_id")
    except Exception:
        return None
    return values[0] if values else None


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
