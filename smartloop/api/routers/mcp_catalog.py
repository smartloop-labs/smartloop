"""MCP server catalog router.

Fetches the curated MCP server catalog from the Smartloop platform API
and exposes it to local clients (TUI, Electron) for one-click install.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, model_validator

from smartloop.platform import PlatformService, get_platform_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["mcp-catalog"])


class RemoteConfig(BaseModel):
    server_url: str
    auth_type: str | None = None
    oauth_metadata_url: str | None = None
    authorization_endpoint: str | None = None
    token_endpoint: str | None = None
    token: str | None = None


class LocalConfig(BaseModel):
    command: str
    args: list[str] = []
    env: dict[str, str] = {}


class MCPCatalogEntry(BaseModel):
    slug: str
    display_name: str
    description: str | None = None
    # The platform catalog does not store server_type; derive it below so clients
    # still get a populated "local"/"remote" value.
    server_type: str | None = None
    hosted_by: str  # "external", "internal", or "local"
    remote_config: RemoteConfig | None = None
    local_config: LocalConfig | None = None
    icon_url: str | None = None
    category: str | None = None
    tags: list[str] = []
    sort_order: int = 0
    catalog_version: int = 1

    @model_validator(mode="after")
    def _default_server_type(self):
        if not self.server_type:
            self.server_type = (
                "local"
                if self.local_config is not None or self.hosted_by == "local"
                else "remote"
            )
        return self


def _to_entries(raw: list) -> list[MCPCatalogEntry]:
    return [MCPCatalogEntry(**e) if isinstance(e, dict) else e for e in raw]


@router.get("/mcp-catalog", response_model=list[MCPCatalogEntry])
async def list_mcp_catalog(
    svc: PlatformService = Depends(get_platform_service),
):
    """List curated MCP server catalog entries.

    Clients (TUI, Electron) use this to populate the one-click install catalog.
    Served through the cache-through platform service, so the list survives a
    platform outage or an expired token instead of leaving the panel empty.
    """
    try:
        entries = svc.list_mcp_catalog()
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except ConnectionError:
        raise HTTPException(status_code=502, detail="Platform API unavailable")

    return _to_entries(entries)


def resolve_catalog_entry(slug: str) -> MCPCatalogEntry | None:
    """Look up one curated entry by slug, or ``None`` if it can't be resolved.

    Never raises: the caller falls back to the client-supplied server_url and
    lets enrichment run, so an install is never blocked by a catalog lookup.
    The platform exposes a list endpoint only, so this is fetch-and-find over
    the cached list.
    """
    if not slug:
        return None
    try:
        entries = get_platform_service().list_mcp_catalog()
    except Exception as e:
        logger.warning("Catalog lookup failed for slug '%s': %s", slug, e)
        return None
    for entry in _to_entries(entries):
        if entry.slug == slug:
            return entry
    logger.warning("No catalog entry for slug '%s'", slug)
    return None
