"""MCP server management router"""

import asyncio
import functools
import logging
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from html import escape

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from smartloop.api.model import (
    MCPCrawlRequest,
    MCPDefinitionRequest,
    MCPServerResponse,
    DeleteMCPServerResponse,
    MCPToolResponse,
    MCPRegisterRequest,
    MCPRegisterResponse,
    MCPOAuthCallbackRequest,
)
from smartloop.api.routers.mcp_catalog import resolve_catalog_entry
from smartloop.api.services import ModelService, get_model_service
from smartloop.mcp.client import MCPDiscoveryError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["mcp"])

# Registration spawns a subprocess, waits on the MCP handshake, may run local
# inference and an embedding pass — all synchronous.  A single worker keeps
# that off the event loop without letting two registrations race the workspace
# record or the lazily-loaded model caches.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-register")

# In-memory store for pending MCP OAuth sessions (keyed by state).  Entries are
# popped by the callback, but an abandoned flow (user closes the browser tab)
# never gets one — and the session holds a client_secret and code_verifier, so
# it must not live for the life of the process.  Hence the TTL sweep below.
_OAUTH_SESSION_TTL_S = 600.0
# Completed callback responses are cached only to absorb the browser's
# duplicate GET of the redirect, which arrives within seconds.
_OAUTH_RESPONSE_TTL_S = 120.0
_OAUTH_RESPONSE_MAX = 32

_pending_oauth_sessions: dict[str, dict] = {}
_completed_oauth_responses: "OrderedDict[str, tuple[float, HTMLResponse]]" = OrderedDict()


def _purge_oauth_state(now: float | None = None) -> None:
    """Drop expired pending OAuth sessions and cached callback responses.

    Piggybacked on request handling rather than run as a background task: the
    dicts only grow when a registration happens, so sweeping on the same events
    keeps them bounded without any extra machinery.
    """
    now = time.time() if now is None else now
    for state, session in list(_pending_oauth_sessions.items()):
        created = session.get("created_at")
        if created is None:
            # Written by a path that predates the timestamp (or seeded by a
            # test).  Stamp it here so it still expires one TTL from now,
            # rather than reading as infinitely old and vanishing instantly.
            session["created_at"] = now
            continue
        if now - created > _OAUTH_SESSION_TTL_S:
            logger.info("Evicting expired OAuth session state=%s", state)
            _pending_oauth_sessions.pop(state, None)
    for state in [st for st, (exp, _) in _completed_oauth_responses.items() if now > exp]:
        _completed_oauth_responses.pop(state, None)
    while len(_completed_oauth_responses) > _OAUTH_RESPONSE_MAX:
        _completed_oauth_responses.popitem(last=False)


async def _offload(fn, *args, **kwargs):
    """Run a blocking call on ``_executor`` and await it.

    Every MCP service/network call below is synchronous — subprocess spawns
    with a 2s startup poll, HTTP handshakes, local inference, an embedding
    pass.  Left on the event loop they stall every other request (and every
    open SSE stream) for the duration.  Exceptions propagate through the await
    unchanged, so the callers' HTTPException mapping still works.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, functools.partial(fn, *args, **kwargs))


def _exchange_oauth_code(**kwargs):
    """Blocking OAuth token exchange, for use via _offload."""
    from smartloop.mcp import MCPOAuthService
    with MCPOAuthService() as oauth:
        return oauth.exchange_code_for_tokens(**kwargs)


def _discover_remote_tools(server_url: str, **kwargs):
    """Blocking remote MCP handshake + tool discovery, for use via _offload."""
    from smartloop.mcp import MCPClient
    with MCPClient() as client:
        return client.connect_and_discover_tools(server_url, **kwargs)


def _build_server_info_dict(si) -> dict | None:
    """Build a server_info dict from an MCPServerInfo, including description and icons."""
    if not si:
        return None
    d = {
        "name": si.name,
        "title": si.title,
        "version": si.version,
    }
    if si.description:
        d["description"] = si.description
    if si.icons:
        d["icons"] = si.icons
    return d


def _build_server_response(result: dict) -> MCPServerResponse:
    """Build an MCPServerResponse from a _mcp_server_to_dict result."""
    return MCPServerResponse(
        id=result["id"],
        name=result["name"],
        description=result.get("description"),
        tags=result.get("tags", []),
        catalog_slug=result.get("catalog_slug"),
        command=result.get("command"),
        args=result.get("args", []),
        cwd=result.get("cwd"),
        server_type=result.get("server_type", "local"),
        server_url=result.get("server_url"),
        enabled=result["enabled"],
        system=result.get("system", False),
        tools=[MCPToolResponse(**t) for t in result["tools"]],
        server_info=result.get("server_info"),
        status=result.get("status", "ready"),
        last_error=result.get("last_error"),
        crawl=result.get("crawl"),
        last_crawl_at=result.get("last_crawl_at"),
        account=result.get("account"),
    )


def _oauth_success_page(
    server_name: str,
    tool_count: int,
    server_id: str,
    return_url: str | None = None,
) -> HTMLResponse:
    """Render the "connected" page shown in the browser after OAuth.

    It reports what was connected and stops there.  It used to navigate to the
    app's ``smartloop://`` deep link automatically, which made the last step of
    every install feel slow: the browser hands the URL to the OS, which decides
    whether to prompt, then wakes the app — all while the user watches a
    "Redirecting..." page that may never go away if no handler is registered.
    The app already learns about the server from the API, so the browser has
    nothing left to deliver: the page states the tool count and lets the user
    close the window.  When a return URL was supplied it is offered as a link,
    so going back to the app stays one click away without being forced.
    """
    tools_label = "tool" if tool_count == 1 else "tools"
    link = ""
    if return_url:
        from urllib.parse import urlencode
        separator = "&" if "?" in return_url else "?"
        href = f"{return_url}{separator}{urlencode({'server_id': server_id})}"
        link = f'<p><a href="{escape(href, quote=True)}">Return to Smartloop</a></p>'

    return HTMLResponse(
        "<html><head><title>MCP server connected</title></head>"
        '<body style="font-family:system-ui,sans-serif;max-width:34rem;margin:4rem auto">'
        "<h2>MCP server connected</h2>"
        f"<p><strong>{escape(server_name)}</strong> registered with "
        f"{tool_count} {tools_label}.</p>"
        "<p>Indexing and metadata finish in the background — you can close "
        "this window.</p>"
        f"{link}"
        '<p><button onclick="window.close()">Close window</button></p>'
        "</body></html>"
    )


async def _register_local_mcp_server(
    request: MCPRegisterRequest,
    model_service: ModelService,
    project_id: str,
    curated=None,
) -> MCPRegisterResponse:
    """Register a local (stdio) MCP server from a client-supplied command.

    Standard MCP stdio config: the caller provides ``command`` (e.g. ``uvx``,
    ``npx``) plus ``args``; the command must be installed on the host PATH.
    When *curated* is a resolved catalog entry, its config and metadata win over
    the request.

    Returns as soon as the connection is persisted, with ``status="pending"``:
    launching the process, enumerating tools, generating metadata and indexing
    all happen on the MCP background task.  Clients follow the server's
    ``status``/``last_error`` (via ``GET /projects/{id}/mcp``) instead of
    waiting here, and can re-run a failed pass with ``.../mcp/{id}/refresh``.
    """
    local_cfg = getattr(curated, "local_config", None) if curated else None
    command = local_cfg.command if local_cfg else request.command
    args = list(local_cfg.args) if local_cfg else request.args
    env = dict(local_cfg.env) if local_cfg and local_cfg.env else (request.env or {})

    if not command:
        raise HTTPException(status_code=400, detail="'command' is required for local servers")
    if not args:
        raise HTTPException(
            status_code=400,
            detail="'args' is required for local servers (e.g. args=['yahoo-finance-mcp'] for uvx, or args=['-m', 'module'] for python)",
        )
    try:
        result = await _offload(
            model_service.add_local_mcp_server,
            command=command,
            args=args,
            name=(curated.display_name if curated else request.name),
            env=env,
            cwd=request.cwd,
            project_id=project_id,
            description=(curated.description if curated else None),
            tags=(list(curated.tags) if curated else None),
            catalog_slug=(curated.slug if curated else None) or request.catalog_slug,
        )
    except MCPDiscoveryError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    server_resp = _build_server_response(result)
    return MCPRegisterResponse(auth_type="none", server=server_resp)


async def _try_register_smartloop_mcp_server(
    server_url: str,
    model_service: ModelService,
    project_id: str,
) -> MCPRegisterResponse | None:
    """Try to register a Smartloop-hosted MCP server by probing its /info endpoint.

    Returns None if the URL doesn't expose a Smartloop /info endpoint,
    allowing the caller to fall through to the standard MCP handshake.
    """
    import httpx

    info_url = server_url.rstrip("/") + "/info"

    def _probe():
        resp = httpx.get(info_url, timeout=10.0)
        resp.raise_for_status()
        return resp.json()

    try:
        info = await _offload(_probe)
        if "pkg" not in info:
            return None
    except Exception:
        return None

    result = await _offload(
        model_service.add_remote_mcp_server,
        name=info.get("name", "smartloop-fetch"),
        server_url=server_url,
        auth_type="none",
        tools=info.get("tools", []),
        project_id=project_id,
        server_info=info.get("server_info"),
        description=info.get("description"),
        pkg=info.get("pkg"),
    )
    server_resp = _build_server_response(result)
    return MCPRegisterResponse(auth_type="none", server=server_resp)


async def _register_remote_mcp_server(
    request: MCPRegisterRequest,
    model_service: ModelService,
    http_request: Request | None,
    project_id: str,
    curated=None,
) -> MCPRegisterResponse:
    """Register a remote MCP server. Tries direct connection first, falls back to OAuth.

    When *curated* is a resolved catalog entry, its ``server_url`` and metadata
    win over the request — the catalog is authoritative for servers we list, so
    a stale client can't pin an outdated URL under a trusted name.
    """
    from smartloop.mcp import MCPOAuthService
    from smartloop.mcp.oauth import MCPOAuthError
    from urllib.parse import urlparse, parse_qs

    curated_url = (
        curated.remote_config.server_url
        if curated and curated.remote_config
        else None
    )
    effective_url = curated_url or request.server_url
    if not effective_url:
        raise HTTPException(status_code=400, detail="'server_url' is required for remote servers")

    curated_description = curated.description if curated else None
    curated_tags = list(curated.tags) if curated else None
    # Provenance is the client's claim about where it installed from, and it
    # survives a lookup that missed.  `resolve_catalog_entry` never raises — a
    # stale cache, an expired platform token and an outage all read as "not
    # curated" — and dropping the slug there costs more than the metadata does:
    # the connection then pays for LLM enrichment it doesn't need and has to
    # find its crawl plan by matching substrings against its own name.
    curated_slug = (curated.slug if curated else None) or request.catalog_slug

    server_url = effective_url.strip()

    # Probe for a Smartloop-hosted /info endpoint before attempting MCP handshake.
    # Any server that exposes /info with a "pkg" field is treated as a Smartloop
    # MCP server and registered directly — no SSE/OAuth negotiation needed.
    smartloop_result = await _try_register_smartloop_mcp_server(server_url, model_service, project_id)
    if smartloop_result is not None:
        return smartloop_result

    parsed = urlparse(server_url)
    query_params = parse_qs(parsed.query)
    token = request.token or query_params.get("token", [None])[0]
    clean_url = (
        f"{parsed.scheme}://{parsed.netloc}{parsed.path}" if token else server_url
    )

    # Try direct connection (no auth or token)
    connect_result = None
    try:
        connect_result = await _offload(
            _discover_remote_tools,
            clean_url,
            access_token=token,
            token_type="Bearer",
        )

        if connect_result.success:
            if not connect_result.tools:
                raise HTTPException(
                    status_code=400,
                    detail="MCP server has no tools available",
                )
            si = connect_result.server_info
            server_info_dict = _build_server_info_dict(si)
            result = await _offload(
                model_service.add_remote_mcp_server,
                name=(curated.display_name if curated else None)
                or request.name
                or (si.name if si else parsed.netloc),
                server_url=clean_url,
                auth_type="token" if token else "none",
                token=token,
                tools=[
                    {
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema,
                    }
                    for t in connect_result.tools
                ],
                project_id=project_id,
                server_info=server_info_dict,
                description=curated_description or (si.description if si else None),
                tags=curated_tags,
                catalog_slug=curated_slug,
            )
            server_resp = _build_server_response(result)
            return MCPRegisterResponse(auth_type="none", server=server_resp)
    except Exception:
        logger.debug("Direct connection failed for %s, trying OAuth", clean_url)
    if connect_result is not None and not connect_result.success:
        # A 401 here is the normal prelude to OAuth, but the detail also carries
        # the WWW-Authenticate header and body — the only clue when the failure
        # is a dead endpoint rather than a missing token.
        logger.info(
            "Direct connect to %s returned no usable session, falling back to OAuth: %s",
            clean_url,
            connect_result.error,
        )

    # Fall back to OAuth
    try:
        base_url = str(http_request.base_url).rstrip("/") if http_request else "http://localhost:38540"
        callback_url = request.callback_url or f"{base_url}/v1/mcp/oauth/callback"

        def _oauth_discover():
            """Blocking OAuth discovery + dynamic client registration.

            Every call in here is a synchronous HTTP round-trip to a third
            party, so the whole sequence runs on the executor rather than the
            event loop.  Returns everything the caller needs after the
            MCPOAuthService context has closed.
            """
            with MCPOAuthService() as oauth:
                oauth_metadata = None
                prm = None
                try:
                    prm = oauth.discover_protected_resource_metadata(
                        clean_url,
                        metadata_url=getattr(connect_result, "resource_metadata_url", None),
                    )
                    if prm.authorization_servers:
                        oauth_metadata = oauth.discover_oauth_metadata(
                            prm.authorization_servers[0]
                        )
                except MCPOAuthError:
                    pass
                if not oauth_metadata:
                    oauth_metadata = oauth.discover_oauth_metadata(clean_url)

                if not oauth_metadata.registration_endpoint:
                    raise HTTPException(
                        status_code=400,
                        detail="Server requires authentication but does not support OAuth dynamic client registration",
                    )

                registration = oauth.register_client(
                    oauth_metadata.registration_endpoint, callback_url
                )

                code_verifier, code_challenge = MCPOAuthService.generate_pkce()
                state = MCPOAuthService.generate_state()

                scopes = oauth_metadata.scopes_supported or (
                    prm.scopes_supported if prm else None
                )
                scope = " ".join(scopes) if scopes else None

                # RFC 8707 resource indicator, required by MCP auth (rev 2025-06-18).
                # Prefer the identifier the resource server publishes for itself; fall
                # back to the canonical server URL when it publishes no PRM. Servers
                # that predate the requirement ignore the extra parameter (RFC 6749
                # §3.1 and §3.2 both mandate ignoring unrecognized ones).
                resource = (prm.resource if prm else None) or oauth.canonical_resource(
                    clean_url
                )

                auth_url = oauth.build_authorization_url(
                    authorization_endpoint=oauth_metadata.authorization_endpoint,
                    client_id=registration.client_id,
                    redirect_uri=callback_url,
                    state=state,
                    code_challenge=code_challenge,
                    scope=scope,
                    resource=resource,
                )

            return (oauth_metadata, registration, code_verifier,
                    state, resource, auth_url)

        (oauth_metadata, registration, code_verifier,
         state, resource, auth_url) = await _offload(_oauth_discover)

        # Store session for the GET callback to use
        _purge_oauth_state()
        _pending_oauth_sessions[state] = {
            "created_at": time.time(),
            "project_id": project_id,
            "server_url": clean_url,
            "state": state,
            "code_verifier": code_verifier,
            "client_id": registration.client_id,
            "client_secret": registration.client_secret,
            "token_endpoint": oauth_metadata.token_endpoint,
            "authorization_endpoint": oauth_metadata.authorization_endpoint,
            "resource": resource,
            "callback_url": callback_url,
            "name": (curated.display_name if curated else None) or request.name,
            "return_url": request.return_url,
            # Curated metadata has to survive the redirect — the record is only
            # written in the callback, long after this request has returned.
            "description": curated_description,
            "tags": curated_tags,
            "catalog_slug": curated_slug,
        }

        return MCPRegisterResponse(
            auth_type="oauth",
            auth_url=auth_url,
            server_url=clean_url,
            state=state,
            code_verifier=code_verifier,
            client_id=registration.client_id,
            client_secret=registration.client_secret,
            token_endpoint=oauth_metadata.token_endpoint,
            authorization_endpoint=oauth_metadata.authorization_endpoint,
            resource=resource,
        )
    except HTTPException:
        raise
    except MCPOAuthError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to connect to MCP server and OAuth discovery failed: {e}",
        )
    except Exception as e:
        logger.exception("Error registering MCP server")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/mcp")
async def list_mcp_servers_for_project(
    project_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """List MCP servers for a specific project.

    Each server carries an ``indexing`` block — provisioning and content-crawl
    state — so a client can show "still indexing" without a request per server.
    """
    try:
        return {"servers": model_service.list_mcp_servers_with_status(project_id)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error listing MCP servers for project")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/mcp/tools/count")
async def count_indexed_mcp_tools_for_project(
    project_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """Diagnostic: how many MCP tool vectors are indexed for the project.

    ``tool_vectors`` is the count in the ``mcp_tools`` vector collection;
    ``server_tools`` is the total tools across registered servers. They should
    match once every server has been indexed — a mismatch means some server's
    tools were not vectorized.
    """
    try:
        return model_service.count_indexed_mcp_tools(project_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error counting indexed MCP tools")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/mcp/content")
async def mcp_content_status_for_project(
    project_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """Crawled MCP content state for the project.

    Reports the number of indexed content chunks and, per server, the crawl
    plan that matched, how many records/documents it produced and when it last
    ran — the content counterpart to ``mcp/tools/count``.
    """
    try:
        return model_service.mcp_content_status(project_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error reading MCP content status")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/mcp/{server_id}")
async def get_mcp_server(
    project_id: str,
    server_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """One MCP server, with whether its content is still being indexed.

    Registration returns before the server is usable: tool discovery and the
    content crawl both run on the MCP task.  ``indexing.state`` is what a
    client polls — ``provisioning`` → ``queued`` → ``indexing`` → ``indexed``,
    or ``failed`` with ``indexing.error``, or ``disabled`` for a server whose
    content isn't crawled.  ``indexing.in_progress`` is the single flag to
    drive a spinner off.

    Declared after the literal ``mcp/tools/count`` and ``mcp/content`` routes:
    a path parameter here would otherwise swallow them.
    """
    try:
        return model_service.mcp_server_status(server_id, project_id=project_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error reading MCP server status")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/mcp/content/refresh")
async def refresh_mcp_content_for_project(
    project_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """Re-crawl every document-type server in the project now.

    The crawl also runs at startup and on the periodic cycle; this is the
    manual "fetch it again" for when the source just changed.
    """
    try:
        return model_service.crawl_mcp_content(project_id=project_id, force=True)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error refreshing MCP content")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/mcp/{server_id}/refresh", response_model=MCPServerResponse)
async def refresh_mcp_server(
    project_id: str,
    server_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """Re-run background provisioning for one server (retry a failed discovery).

    Returns the server with ``status="pending"``; the pass itself runs on the
    MCP task.
    """
    try:
        result = model_service.retry_mcp_server(server_id, project_id=project_id)
        return _build_server_response(result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error refreshing MCP server")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/mcp/{server_id}/crawl")
async def set_mcp_server_crawl(
    project_id: str,
    server_id: str,
    request: MCPCrawlRequest,
    model_service: ModelService = Depends(get_model_service),
):
    """Opt one server's content in or out of the crawl cycle.

    Document-type servers (Notion, Linear, Trello, …) crawl by default; this
    turns it on for a server we don't recognise as one, or off for a server
    whose content shouldn't be indexed.  Disabling also drops what was already
    indexed for it.
    """
    try:
        return model_service.set_mcp_crawl_enabled(
            server_id, request.enabled, project_id=project_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error updating MCP crawl setting")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/projects/{project_id}/mcp/{server_id}", response_model=DeleteMCPServerResponse)
async def delete_mcp_server_from_project(
    project_id: str,
    server_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """Delete an MCP server by ID from a specific project."""
    try:
        result = model_service.delete_mcp_server(server_id, project_id=project_id)
        return DeleteMCPServerResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error deleting MCP server from project")
        raise HTTPException(status_code=500, detail=str(e))


# ---- Register endpoint (unified) ----


@router.post("/projects/{project_id}/mcp/register", response_model=MCPRegisterResponse)
async def register_mcp_server(
    project_id: str,
    request: MCPRegisterRequest,
    http_request: Request = None,
    model_service: ModelService = Depends(get_model_service),
):
    """Unified MCP server registration (project-scoped).
    For local servers, pass server_type="local" with command (e.g. "uvx", "npx") and args.
    For remote servers, pass server_url (server_type defaults to "remote").
    For Smartloop-hosted MCP servers, pass server_type="smartloop" and server_url.
    """
    # Resolve curated metadata server-side. Dispatch on the *resolved* entry's
    # type rather than the caller's claim, so a future local catalog listing
    # works with no client release. A miss leaves curated=None and the install
    # proceeds on the client-supplied values, enrichment and all.
    curated = resolve_catalog_entry(request.catalog_slug) if request.catalog_slug else None
    server_type = curated.server_type if curated else request.server_type

    if server_type == "local":
        return await _register_local_mcp_server(request, model_service, project_id, curated)
    return await _register_remote_mcp_server(
        request, model_service, http_request, project_id, curated,
    )


@router.post("/projects/{project_id}/mcp/definition", response_model=MCPRegisterResponse)
async def register_mcp_from_definition(
    project_id: str,
    request: MCPDefinitionRequest,
    http_request: Request = None,
    model_service: ModelService = Depends(get_model_service),
):
    """Install a connection from a dropped-in TOML file.

    The file is the whole definition: what the server is called, how to reach
    it, and — optionally — how to index its content.  Both halves are kept: the
    file is saved under ``~/.smartloop/mcp.d/`` so its crawl plan is there for
    every later refresh, and the connection itself goes through the ordinary
    registration path, so a remote server needing OAuth returns an ``auth_url``
    exactly as it would if the URL had been typed in.
    """
    from smartloop.mcp.plan_loader import (
        USER_PLANS_DIR, parse_definitions, reload_plans,
    )

    try:
        found = parse_definitions(request.toml, request.filename or "upload")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Could not read this connection file: {e}",
        )
    if not found:
        raise HTTPException(
            status_code=400,
            detail="No [[mcp]] entry in this file — it needs at least a name "
                   "and either a local command or a remote url.",
        )
    if len(found) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"This file describes {len(found)} connections "
                   f"({', '.join(sorted(found))}); drop one file per connection.",
        )

    definition = next(iter(found.values()))
    if not definition.kind:
        raise HTTPException(
            status_code=400,
            detail=f"'{definition.name}' has no connection: add a url for a "
                   f"remote server, or a package name for one to run locally.",
        )

    # Persist before registering: the plan has to be on disk by the time the
    # provisioning pass runs its first crawl, and that is queued by the
    # registration below.
    try:
        from smartloop.config.app_settings import AppSettings
        drop_in_dir = os.path.join(AppSettings().home_dir, USER_PLANS_DIR)
        os.makedirs(drop_in_dir, exist_ok=True)
        # Named after the connection, not after whatever the file was called on
        # the way in — that is what makes re-dropping an edited file replace
        # the old plan instead of stacking a second copy beside it.
        target = os.path.join(drop_in_dir, f"{definition.key}.toml")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(request.toml)
        reload_plans()
        logger.info("[mcp-definition] Saved '%s' to %s", definition.key, target)
    except Exception as e:
        # The connection is still worth installing without its plan; the crawl
        # falls back to a derived one.
        logger.warning("[mcp-definition] Could not save the plan file: %s", e)

    command, args = definition.launch
    register = MCPRegisterRequest(
        name=definition.name,
        server_type=definition.kind,
        command=command,
        args=args,
        env=dict(definition.env),
        cwd=definition.cwd,
        server_url=definition.url,
        token=definition.token,
        return_url="smartloop://mcp-authorization",
    )
    return await register_mcp_server(
        project_id=project_id, request=register,
        http_request=http_request, model_service=model_service,
    )


@router.get("/mcp/oauth/callback", response_class=HTMLResponse)
async def mcp_oauth_callback(
    code: str = None,
    state: str = None,
    error: str = None,
    model_service: ModelService = Depends(get_model_service),
):
    """OAuth callback endpoint. The OAuth provider redirects here with the authorization code.
    Exchanges the code for tokens, discovers tools, and registers the MCP server.
    """
    from smartloop.mcp.oauth import MCPOAuthError

    if error:
        return HTMLResponse(
            f"<html><body><h2>Authorization failed</h2><p>{escape(error)}</p>"
            "<p>You can close this window.</p></body></html>",
            status_code=400,
        )

    if not code or not state:
        logger.warning("OAuth callback missing parameters: code=%s, state=%s", bool(code), bool(state))
        return HTMLResponse(
            "<html><body><h2>Missing parameters</h2>"
            "<p>No authorization code or state received.</p></body></html>",
            status_code=400,
        )

    logger.debug("OAuth callback received: state=%s", state)

    # Return cached response for duplicate browser requests
    _purge_oauth_state()
    cached = _completed_oauth_responses.get(state)
    if cached is not None:
        logger.info("Returning cached OAuth response for state=%s", state)
        return cached[1]

    session = _pending_oauth_sessions.pop(state, None)
    if not session:
        logger.warning("OAuth session not found for state=%s", state)
        return HTMLResponse(
            "<html><body><h2>Session expired</h2>"
            "<p>OAuth session not found. Please try again.</p></body></html>",
            status_code=400,
        )

    try:
        # Exchange code for tokens
        token_response = await _offload(
            _exchange_oauth_code,
            token_endpoint=session["token_endpoint"],
            code=code,
            redirect_uri=session["callback_url"],
            client_id=session["client_id"],
            code_verifier=session["code_verifier"],
            client_secret=session.get("client_secret"),
            resource=session.get("resource"),
        )

        expires_at = (
            time.time() + token_response.expires_in
            if token_response.expires_in
            else None
        )

        # Discover tools
        connect_result = await _offload(
            _discover_remote_tools,
            session["server_url"],
            access_token=token_response.access_token,
            token_type=token_response.token_type,
        )

        tools = connect_result.tools if connect_result.success else []
        if not tools:
            # Two very different failures reach this point — the handshake was
            # refused, or it succeeded and the server offered nothing. Collapsing
            # both into "no tools available" hides the reason (a dead transport,
            # a rejected token, a grant with no product scopes), so name it.
            reason = connect_result.error or (
                "The server completed the MCP handshake but exposes no tools. "
                "This usually means the authorization granted no scopes."
            )
            logger.error(
                "Tool discovery failed after OAuth for %s: %s",
                session["server_url"],
                reason,
            )
            return HTMLResponse(
                "<html><body><h2>Registration failed</h2>"
                f"<p>Could not list tools on {escape(session['server_url'])}.</p>"
                f"<pre>{escape(reason)}</pre></body></html>",
                status_code=400,
            )
        si = connect_result.server_info if connect_result.success else None
        server_name = (
            session.get("name")
            or (si.name if si else None)
            or "Remote MCP Server"
        )
        server_info_dict = _build_server_info_dict(si)

        # Register the server
        result = await _offload(
            model_service.add_remote_mcp_server,
            name=server_name,
            server_url=session["server_url"],
            auth_type="oauth",
            tools=[
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ],
            token_endpoint=session["token_endpoint"],
            authorization_endpoint=session["authorization_endpoint"],
            resource=session.get("resource"),
            client_id=session["client_id"],
            client_secret=session.get("client_secret"),
            access_token=token_response.access_token,
            refresh_token=token_response.refresh_token,
            expires_at=expires_at,
            token_type=token_response.token_type,
            project_id=session["project_id"],
            server_info=server_info_dict,
            # Curated copy stashed at register time wins over whatever the
            # vendor advertises in its handshake.
            description=session.get("description") or (si.description if si else None),
            tags=session.get("tags"),
            catalog_slug=session.get("catalog_slug"),
        )

        server_id = result["id"]
        tool_count = len(result.get("tools", []))
        response = _oauth_success_page(
            server_name=result["name"],
            tool_count=tool_count,
            server_id=server_id,
            return_url=session.get("return_url"),
        )

        # Cache response for duplicate browser requests
        _completed_oauth_responses[state] = (
            time.time() + _OAUTH_RESPONSE_TTL_S, response,
        )
        return response
    except MCPOAuthError as e:
        logger.exception("MCP OAuth callback error")
        return HTMLResponse(
            f"<html><body><h2>Authorization failed</h2><p>{e}</p>"
            f"<p>You can close this window and try again.</p></body></html>",
            status_code=400,
        )
    except Exception as e:
        logger.exception("MCP OAuth callback error")
        return HTMLResponse(
            f"<html><body><h2>Error</h2><p>{e}</p>"
            f"<p>You can close this window and try again.</p></body></html>",
            status_code=500,
        )


# ---- OAuth flow endpoints ----


@router.post("/projects/{project_id}/mcp/oauth/callback", response_model=MCPServerResponse)
async def complete_mcp_oauth(
    project_id: str,
    request: MCPOAuthCallbackRequest,
    model_service: ModelService = Depends(get_model_service),
):
    """Exchange OAuth authorization code for tokens and register the remote MCP server."""
    from smartloop.mcp.oauth import MCPOAuthError

    try:
        callback_url = request.callback_url or "http://localhost:9876/callback"

        # Exchange code for tokens
        token_response = await _offload(
            _exchange_oauth_code,
            token_endpoint=request.token_endpoint,
            code=request.code,
            redirect_uri=callback_url,
            client_id=request.client_id,
            code_verifier=request.code_verifier,
            client_secret=request.client_secret,
            resource=request.resource,
        )

        expires_at = (
            time.time() + token_response.expires_in
            if token_response.expires_in
            else None
        )

        # Discover tools
        connect_result = await _offload(
            _discover_remote_tools,
            request.server_url,
            access_token=token_response.access_token,
            token_type=token_response.token_type,
        )

        tools = connect_result.tools if connect_result.success else []
        si = connect_result.server_info if connect_result.success else None
        server_info_dict = _build_server_info_dict(si)

        # The curated copy stashed at register time, for a client that posts the
        # grant back instead of letting the browser redirect land on the GET
        # callback.  Both completions write the same record, so both take their
        # name, description, tags and provenance from the same place — read
        # rather than popped, since the redirect may still be in flight.
        session = _pending_oauth_sessions.get(request.state) or {}

        # Save the remote MCP server
        result = await _offload(
            model_service.add_remote_mcp_server,
            name=session.get("name") or (si.name if si else "Remote MCP Server"),
            server_url=request.server_url,
            auth_type="oauth",
            tools=[
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ],
            token_endpoint=request.token_endpoint,
            authorization_endpoint=request.authorization_endpoint,
            resource=request.resource,
            client_id=request.client_id,
            client_secret=request.client_secret,
            access_token=token_response.access_token,
            refresh_token=token_response.refresh_token,
            expires_at=expires_at,
            token_type=token_response.token_type,
            project_id=project_id,
            server_info=server_info_dict,
            description=session.get("description") or (si.description if si else None),
            tags=session.get("tags"),
            catalog_slug=session.get("catalog_slug"),
        )

        return _build_server_response(result)
    except MCPOAuthError as e:
        raise HTTPException(status_code=400, detail=f"OAuth error: {e}")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error completing MCP OAuth")
        raise HTTPException(status_code=500, detail=str(e))
