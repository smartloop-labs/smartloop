"""Pydantic models for MCP server API endpoints."""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel


class MCPToolResponse(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Optional[dict] = None


class AddMCPServerRequest(BaseModel):
    name: Optional[str] = None       # optional display name; derived from args if omitted
    command: str                     # stdio command, e.g. "uvx", "npx" (must be on PATH)
    args: Optional[List[str]] = []
    description: Optional[str] = None
    env: Dict[str, str] = {}
    cwd: Optional[str] = None


class MCPServerResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    tags: List[str] = []
    command: Optional[str] = None
    args: List[str] = []
    cwd: Optional[str] = None
    server_type: str = "local"
    server_url: Optional[str] = None
    enabled: bool = True
    system: bool = False
    tools: List[MCPToolResponse] = []
    server_info: Optional[Dict] = None


class MCPServerListResponse(BaseModel):
    servers: List[MCPServerResponse]


class DeleteMCPServerResponse(BaseModel):
    id: str
    deleted: bool
    message: str


class AddRemoteMCPServerRequest(BaseModel):
    name: Optional[str] = None
    server_url: str
    auth_type: str = "none"
    tools: List[Dict] = []
    token: Optional[str] = None
    token_endpoint: Optional[str] = None
    authorization_endpoint: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[float] = None
    token_type: str = "Bearer"


class MCPRegisterRequest(BaseModel):
    server_type: str = "remote"       # "local" or "remote"
    name: Optional[str] = None        # display name (optional for local, derived from args)
    command: Optional[str] = None     # stdio command for local server, e.g. "uvx", "npx"
    args: Optional[List[str]] =  []    # command args for local server
    # Local server fields
    env: Dict[str, str] = {}
    cwd: Optional[str] = None
    # Remote server fields
    server_url: Optional[str] = None
    token: Optional[str] = None
    callback_url: Optional[str] = None
    return_url: Optional[str] = None   # URL to redirect to after successful registration


class MCPRegisterResponse(BaseModel):
    """Unified response for MCP server registration.
    If auth_type is 'oauth', auth_url and oauth session fields are returned.
    Otherwise, the server is registered directly and server fields are populated.
    """
    auth_type: str  # 'none', 'token', 'oauth'
    # Populated when server is registered directly (no OAuth)
    server: Optional[MCPServerResponse] = None
    # Populated when OAuth is required
    auth_url: Optional[str] = None
    state: Optional[str] = None
    code_verifier: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    token_endpoint: Optional[str] = None
    authorization_endpoint: Optional[str] = None
    server_url: Optional[str] = None


class MCPOAuthInitiateRequest(BaseModel):
    server_url: str
    callback_url: Optional[str] = None


class MCPOAuthInitiateResponse(BaseModel):
    auth_url: str
    server_url: str
    state: str
    code_verifier: str
    client_id: str
    client_secret: Optional[str] = None
    token_endpoint: str
    authorization_endpoint: str


class MCPOAuthCallbackRequest(BaseModel):
    server_url: str
    code: str
    state: str
    code_verifier: str
    client_id: str
    client_secret: Optional[str] = None
    token_endpoint: str
    authorization_endpoint: str
    callback_url: Optional[str] = None


class MCPToolCallRequest(BaseModel):
    tool_name: str
    arguments: Optional[Dict[str, Union[str, int, float, bool, list, dict]]] = None


class MCPToolCallResponse(BaseModel):
    success: bool
    content: Optional[List[Dict]] = None
    is_error: bool = False
    message: Optional[str] = None
