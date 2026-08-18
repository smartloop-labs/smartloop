"""MCP server management router"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from smartloop.api.model import (
    MCPServerResponse,
    DeleteMCPServerResponse,
    MCPToolResponse,
    MCPRegisterRequest,
    MCPRegisterResponse,
    MCPOAuthCallbackRequest,
)
from smartloop.api.services import ModelService, get_model_service
from smartloop.mcp.client import MCPDiscoveryError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["mcp"])

# In-memory store for pending MCP OAuth sessions (keyed by state)
_pending_oauth_sessions: dict[str, dict] = {}
# Cache of completed OAuth callback responses (keyed by state) to handle duplicate browser requests
_completed_oauth_responses: dict[str, HTMLResponse] = {}


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
        command=result.get("command"),
        args=result.get("args", []),
        cwd=result.get("cwd"),
        server_type=result.get("server_type", "local"),
        server_url=result.get("server_url"),
        enabled=result["enabled"],
        system=result.get("system", False),
        tools=[MCPToolResponse(**t) for t in result["tools"]],
        server_info=result.get("server_info"),
    )


async def _register_local_mcp_server(
    request: MCPRegisterRequest,
    model_service: ModelService,
    project_id: str,
) -> MCPRegisterResponse:
    """Register a local (stdio) MCP server from a client-supplied command.

    Standard MCP stdio config: the caller provides ``command`` (e.g. ``uvx``,
    ``npx``) plus ``args``; the command must be installed on the host PATH.
    """
    if not request.command:
        raise HTTPException(status_code=400, detail="'command' is required for local servers")
    if not request.args:
        raise HTTPException(
            status_code=400,
            detail="'args' is required for local servers (e.g. args=['yahoo-finance-mcp'] for uvx, or args=['-m', 'module'] for python)",
        )
    try:
        result = model_service.add_local_mcp_server(
            command=request.command,
            args=request.args,
            name=request.name,
            env=request.env or {},
            cwd=request.cwd,
            project_id=project_id,
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
    try:
        resp = httpx.get(info_url, timeout=10.0)
        resp.raise_for_status()
        info = resp.json()
        if "pkg" not in info:
            return None
    except Exception:
        return None

    result = model_service.add_remote_mcp_server(
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
) -> MCPRegisterResponse:
    """Register a remote MCP server. Tries direct connection first, falls back to OAuth."""
    from smartloop.mcp import MCPClient, MCPOAuthService
    from smartloop.mcp.oauth import MCPOAuthError
    from urllib.parse import urlparse, parse_qs

    if not request.server_url:
        raise HTTPException(status_code=400, detail="'server_url' is required for remote servers")

    server_url = request.server_url.strip()

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
        with MCPClient() as client:
            connect_result = client.connect_and_discover_tools(
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
            result = model_service.add_remote_mcp_server(
                name=request.name
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
                description=si.description if si else None,
            )
            server_resp = _build_server_response(result)
            return MCPRegisterResponse(auth_type="none", server=server_resp)
    except Exception:
        logger.debug("Direct connection failed for %s, trying OAuth", clean_url)

    # Fall back to OAuth
    try:
        base_url = str(http_request.base_url).rstrip("/") if http_request else "http://localhost:38540"
        callback_url = request.callback_url or f"{base_url}/v1/mcp/oauth/callback"

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

            auth_url = oauth.build_authorization_url(
                authorization_endpoint=oauth_metadata.authorization_endpoint,
                client_id=registration.client_id,
                redirect_uri=callback_url,
                state=state,
                code_challenge=code_challenge,
                scope=scope,
            )

        # Store session for the GET callback to use
        _pending_oauth_sessions[state] = {
            "project_id": project_id,
            "server_url": clean_url,
            "state": state,
            "code_verifier": code_verifier,
            "client_id": registration.client_id,
            "client_secret": registration.client_secret,
            "token_endpoint": oauth_metadata.token_endpoint,
            "authorization_endpoint": oauth_metadata.authorization_endpoint,
            "callback_url": callback_url,
            "name": request.name,
            "return_url": request.return_url,
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
    """List MCP servers for a specific project."""
    try:
        project = model_service._get_project(project_id)
        servers = [
            model_service._mcp_server_to_dict(s)
            for s in project.mcp_servers
            if not s.system
        ]
        return {"servers": servers}
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
    if request.server_type == "local":
        return await _register_local_mcp_server(request, model_service, project_id)
    return await _register_remote_mcp_server(request, model_service, http_request, project_id)


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
    from smartloop.mcp import MCPClient, MCPOAuthService
    from smartloop.mcp.oauth import MCPOAuthError

    if error:
        return HTMLResponse(
            f"<html><body><h2>Authorization failed</h2><p>{error}</p>"
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
    if state in _completed_oauth_responses:
        logger.info("Returning cached OAuth response for state=%s", state)
        return _completed_oauth_responses[state]

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
        with MCPOAuthService() as oauth:
            token_response = oauth.exchange_code_for_tokens(
                token_endpoint=session["token_endpoint"],
                code=code,
                redirect_uri=session["callback_url"],
                client_id=session["client_id"],
                code_verifier=session["code_verifier"],
                client_secret=session.get("client_secret"),
            )

        expires_at = (
            time.time() + token_response.expires_in
            if token_response.expires_in
            else None
        )

        # Discover tools
        with MCPClient() as client:
            connect_result = client.connect_and_discover_tools(
                session["server_url"],
                access_token=token_response.access_token,
                token_type=token_response.token_type,
            )

        tools = connect_result.tools if connect_result.success else []
        if not tools:
            return HTMLResponse(
                "<html><body><h2>Registration failed</h2>"
                "<p>MCP server has no tools available.</p></body></html>",
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
        result = model_service.add_remote_mcp_server(
            name=server_name,
            server_url=session["server_url"],
            auth_type="oauth",
            tools=[
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ],
            token_endpoint=session["token_endpoint"],
            authorization_endpoint=session["authorization_endpoint"],
            client_id=session["client_id"],
            client_secret=session.get("client_secret"),
            access_token=token_response.access_token,
            refresh_token=token_response.refresh_token,
            expires_at=expires_at,
            token_type=token_response.token_type,
            project_id=session["project_id"],
            server_info=server_info_dict,
            description=si.description if si else None,
        )

        server_id = result["id"]
        tool_count = len(result.get("tools", []))
        return_url = session.get("return_url")

        if return_url:
            from urllib.parse import urlencode
            separator = "&" if "?" in return_url else "?"
            redirect_url = f"{return_url}{separator}{urlencode({'server_id': server_id})}"
            response = HTMLResponse(
                f"<html><body>"
                f"<h2>MCP server connected</h2>"
                f"<p><strong>{result['name']}</strong> registered with {tool_count} tools.</p>"
                f"<p>Redirecting...</p>"
                f"<script>window.location.href = '{redirect_url}';</script>"
                f"</body></html>"
            )
        else:
            response = HTMLResponse(
                f"<html><body>"
                f"<h2>MCP server connected</h2>"
                f"<p><strong>{result['name']}</strong> registered with {tool_count} tools.</p>"
                f"<p>You can close this window.</p>"
                f"</body></html>"
            )

        # Cache response for duplicate browser requests
        _completed_oauth_responses[state] = response
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
    from smartloop.mcp import MCPClient, MCPOAuthService
    from smartloop.mcp.oauth import MCPOAuthError

    try:
        callback_url = request.callback_url or "http://localhost:9876/callback"

        # Exchange code for tokens
        with MCPOAuthService() as oauth:
            token_response = oauth.exchange_code_for_tokens(
                token_endpoint=request.token_endpoint,
                code=request.code,
                redirect_uri=callback_url,
                client_id=request.client_id,
                code_verifier=request.code_verifier,
                client_secret=request.client_secret,
            )

        expires_at = (
            time.time() + token_response.expires_in
            if token_response.expires_in
            else None
        )

        # Discover tools
        with MCPClient() as client:
            connect_result = client.connect_and_discover_tools(
                request.server_url,
                access_token=token_response.access_token,
                token_type=token_response.token_type,
            )

        tools = connect_result.tools if connect_result.success else []
        si = connect_result.server_info if connect_result.success else None
        server_info_dict = _build_server_info_dict(si)

        # Save the remote MCP server
        result = model_service.add_remote_mcp_server(
            name=si.name if si else "Remote MCP Server",
            server_url=request.server_url,
            auth_type="oauth",
            tools=[
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ],
            token_endpoint=request.token_endpoint,
            authorization_endpoint=request.authorization_endpoint,
            client_id=request.client_id,
            client_secret=request.client_secret,
            access_token=token_response.access_token,
            refresh_token=token_response.refresh_token,
            expires_at=expires_at,
            token_type=token_response.token_type,
            project_id=project_id,
            server_info=server_info_dict,
            description=si.description if si else None,
        )

        return _build_server_response(result)
    except MCPOAuthError as e:
        raise HTTPException(status_code=400, detail=f"OAuth error: {e}")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error completing MCP OAuth")
        raise HTTPException(status_code=500, detail=str(e))
