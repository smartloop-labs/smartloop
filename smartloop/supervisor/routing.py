"""Getting a request from :38540 to the child that owns its project.

The RFC left the granularity open: proxy the request bodies, or redirect the
client to the child's port?  This proxies, for two reasons.  The client
contract is supposed to be unchanged, and a redirect is not unchanged — the
desktop app and CLI would have to follow to a second origin, which moves CORS,
auth and the "everything is on :38540" promise onto every client.  And one
origin means one auth surface: the child's loopback port never has to be safe
to expose, because nothing outside the supervisor ever addresses it.

The cost a redirect would have avoided is streaming a token-by-token SSE
response through two hops.  That is answered by never buffering: the response
is forwarded as raw bytes as they arrive (:func:`proxy`), so a chunk crosses
the supervisor and reaches the client without waiting for the next one.

Which requests are project-scoped:

* a project id in the path (``/v1/projects/<id>``, with or without a
  sub-resource after it) — read from the ASGI scope, no body needed;
* a ``project_id`` in the query string — same;
* a ``project_id`` in a JSON body — chat, essentially.  The body has to be read
  to forward it anyway, so reading it to route costs nothing extra.

Everything else is workspace-level — models, releases, auth, users, the project
list — and the supervisor answers it from its own app.

Two of those need saying out loud, because a path that merely *contains* a
project-shaped segment used to be enough to forward it:

* **a delete** is the one operation on a project that its own agent cannot
  carry out.  Removing a project means removing it from a list that lives only
  in the workspace record, which a pinned ``Workspace.save()`` never writes —
  and could not write here even in principle, since the project it is pinned to
  is the one leaving the list.  Forwarded, it deleted its own directory and then
  persisted nothing, leaving the record behind and the agent running.  Every
  *other* write to a project record leaves the project in existence, so the
  child's own file is still the right place for it and it still goes there;
* **``/v1/projects/import``** names a verb, not a project.  Read as an id it had
  the supervisor try to start an agent for a project called "import".
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Awaitable, Callable, Optional

import httpx

from smartloop.api.middleware import _COLLECTION_ACTIONS, _project_from_query

logger = logging.getLogger(__name__)

#: Hop-by-hop headers, which describe one connection and must not be forwarded
#: onto another (RFC 9110 §7.6.1).  ``content-length`` goes too: the body is
#: re-sent, and a streamed response has none.
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length",
    "host",
})

#: Paths whose body may name a project.  Kept explicit so the supervisor never
#: reads a body it has no reason to.
#:
#: ``/v1/feedback`` is here because its POST carries ``project_id`` in the body
#: while its GET and DELETE carry it in the query string — so without this the
#: three halves of one endpoint were answered by two different processes, and
#: the POST was the one that ended up writing a project's feedback index from
#: outside the child that owns it.
_BODY_ROUTED_PATHS = ("/v1/chat/completions", "/v1/feedback")

#: ``/v1/projects/<id>`` — the project record itself, with or without a
#: sub-resource after it.
_PROJECT_PATH_RE = re.compile(r"/projects/([^/]+)(?P<rest>/.+)?$")

# ``_COLLECTION_ACTIONS`` — the verbs on the project collection, which occupy
# the same slot in the path as a project id and are emphatically not one — is
# imported from the binding middleware rather than repeated here.  Reading
# ``import`` as a project had the supervisor try to cold-start an agent for
# it; the copy left behind in the middleware then went on binding requests to
# the same project that cannot exist.  One list means the next verb added
# cannot be fixed in one layer and missed in the other.

#: Methods on a project record that its own agent cannot serve.  Only the
#: delete: it is the one that takes the project out of the workspace record.
_WORKSPACE_ONLY_ITEM_METHODS = frozenset({"DELETE"})


async def read_body(receive: Callable[[], Awaitable[dict]]) -> bytes:
    """Drain the request body from the ASGI ``receive`` channel."""
    chunks: list[bytes] = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            break
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


def replay_body(body: bytes) -> Callable[[], Awaitable[dict]]:
    """An ASGI ``receive`` that serves *body* once, for a request already read.

    Needed when routing decides the supervisor should handle a request after
    all: its body has been consumed, and the downstream app has to see it.
    """
    sent = False

    async def receive() -> dict:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


def project_from_body(body: bytes) -> Optional[str]:
    """The ``project_id`` in a JSON body, if there is one."""
    if not body:
        return None
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    project_id = payload.get("project_id")
    return project_id if isinstance(project_id, str) and project_id else None


def routes_by_body(scope: dict) -> bool:
    """Whether this request's project can only be found in its body."""
    return (scope.get("path") or "") in _BODY_ROUTED_PATHS


def project_to_proxy(scope: dict) -> Optional[str]:
    """The project this request should be *forwarded* to, or ``None``.

    Narrower than :func:`~smartloop.api.middleware._project_from_scope`, which
    answers a different question — which project a request is *about*, for
    binding — and still binds the ones answered here.
    """
    path = scope.get("path") or ""
    if path in _COLLECTION_ACTIONS:
        return None

    match = _PROJECT_PATH_RE.search(path)
    if match:
        if (match.group("rest") is None
                and scope.get("method") in _WORKSPACE_ONLY_ITEM_METHODS):
            return None
        return match.group(1) or None
    return _project_from_query(scope)


def forwardable_headers(raw_headers: list[tuple[bytes, bytes]]) -> dict:
    """Request headers minus the ones that describe *this* connection."""
    return {
        name.decode("latin-1"): value.decode("latin-1")
        for name, value in raw_headers
        if name.decode("latin-1").lower() not in _HOP_BY_HOP
    }


def response_headers(headers) -> list[tuple[bytes, bytes]]:
    """Response headers to pass back, minus hop-by-hop ones."""
    return [
        (name.encode("latin-1"), value.encode("latin-1"))
        for name, value in headers.multi_items()
        if name.lower() not in _HOP_BY_HOP
    ]


async def proxy(scope: dict, send: Callable[[dict], Awaitable[None]],
                body: bytes, base_url: str,
                client: httpx.AsyncClient) -> None:
    """Forward one request to *base_url* and stream the answer back.

    Raw bytes, unbuffered, in both directions: an SSE response reaches the
    client chunk by chunk exactly as the child produced it, which is what keeps
    the second hop from turning a streaming answer into a pause and then a
    wall of text.
    """
    url = httpx.URL(base_url + scope.get("path", ""))
    query = scope.get("query_string") or b""
    if query:
        url = url.copy_with(query=query)

    request = client.build_request(
        scope.get("method", "GET"),
        url,
        headers=forwardable_headers(scope.get("headers") or []),
        content=body,
    )

    upstream = await client.send(request, stream=True)
    try:
        await send({
            "type": "http.response.start",
            "status": upstream.status_code,
            "headers": response_headers(upstream.headers),
        })
        async for chunk in upstream.aiter_raw():
            await send({
                "type": "http.response.body", "body": chunk, "more_body": True,
            })
        await send({"type": "http.response.body", "body": b"", "more_body": False})
    finally:
        await upstream.aclose()


class ProjectRoutingMiddleware:
    """Send each project-scoped request to that project's own agent.

    Sits outermost on the supervisor's app.  A request that names no project
    falls through to the app itself, so every workspace-level endpoint —
    models, releases, auth, the project list — is served here as before and the
    client cannot tell which requests were forwarded.

    Inert when there is no pool — which is a child serving its own project,
    and the ``--no-supervised`` fallback while that still exists.
    """

    def __init__(self, app, pool_provider: Callable[[], object]) -> None:
        self.app = app
        self._pool_provider = pool_provider
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            # No total timeout: a long orchestration legitimately streams for
            # minutes, and cutting it off at the proxy would look to the client
            # like the model stopped mid-answer.
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(None, connect=10.0),
                follow_redirects=False,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __call__(self, scope, receive, send) -> None:
        pool = self._pool_provider()
        if scope.get("type") != "http" or pool is None:
            await self.app(scope, receive, send)
            return

        project_id = project_to_proxy(scope)
        body = b""
        if project_id is None and routes_by_body(scope):
            body = await read_body(receive)
            project_id = project_from_body(body)
            receive = replay_body(body)

        if not project_id:
            await self.app(scope, receive, send)
            return

        try:
            # ``ensure`` is synchronous and, for a cold project, can block for
            # minutes waiting on a child's port and health check. Run it off
            # the event loop thread — otherwise that wait freezes the whole
            # supervisor, including every other project's already-running
            # requests, for as long as the cold start takes.
            record = await asyncio.to_thread(pool.ensure, project_id)
        except Exception as exc:
            # A project that cannot be started is the supervisor's problem to
            # report, not a reason to answer from the wrong process.
            logger.exception("[supervisor] could not start project %s", project_id)
            await _error(send, 503, f"project agent unavailable: {exc}")
            return

        if not body and scope.get("method") not in ("GET", "HEAD", "DELETE"):
            body = await read_body(receive)

        try:
            await proxy(scope, send, body, record.base_url, self.client)
        except httpx.HTTPError as exc:
            logger.warning("[supervisor] project %s: proxy failed: %s",
                           project_id, exc)
            await _error(send, 502, f"project agent did not answer: {exc}")


async def _error(send: Callable[[dict], Awaitable[None]], status: int,
                 detail: str) -> None:
    """Answer with the same JSON shape the API's own errors use."""
    payload = json.dumps({"detail": detail}).encode()
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ],
    })
    await send({"type": "http.response.body", "body": payload, "more_body": False})
