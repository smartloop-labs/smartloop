"""MCP server presets router.

Fetches the curated MCP server preset catalog from the Smartloop platform API
and exposes it to local clients (TUI, Electron) for one-click install.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from smartloop.config.app_settings import AppSettings
from smartloop.platform.client import SLPPlatformClient

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["mcp-presets"])


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


class MCPPresetEntry(BaseModel):
    slug: str
    display_name: str
    description: str | None = None
    server_type: str
    remote_config: RemoteConfig | None = None
    local_config: LocalConfig | None = None
    icon_url: str | None = None
    category: str | None = None
    tags: list[str] = []
    sort_order: int = 0
    preset_version: int = 1


@router.get("/mcp-presets", response_model=list[MCPPresetEntry])
async def list_mcp_presets():
    """List curated MCP server presets from the platform API.

    Clients (TUI, Electron) use this to populate the one-click install catalog.
    """
    settings = AppSettings()
    client = SLPPlatformClient(settings.slp_api_base)

    try:
        presets = client.list_mcp_presets()
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except ConnectionError:
        raise HTTPException(status_code=502, detail="Platform API unavailable")

    return [MCPPresetEntry(**p) if isinstance(p, dict) else p for p in presets]
