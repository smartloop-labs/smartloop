"""Sync MCP client service for connecting to remote MCP servers and discovering tools."""
import json
import logging
import re
from typing import Optional, List
from dataclasses import dataclass, field

import httpx

from smartloop.workspace import MCPToolDefinition

logger = logging.getLogger(__name__)


class MCPClientError(Exception):
    pass


class MCPConnectionError(MCPClientError):
    def __init__(self, message: str, www_authenticate: Optional[str] = None,
                 status_code: Optional[int] = None):
        super().__init__(message)
        self.www_authenticate = www_authenticate
        self.status_code = status_code


class MCPProtocolError(MCPClientError):
    pass


class MCPLaunchError(MCPClientError):
    """Raised when the MCP subprocess fails to start or exits before the handshake."""
    pass


class MCPDiscoveryError(MCPClientError):
    """Raised when an MCP server is reachable-ish but its tools cannot be discovered.

    Covers launcher-not-found, handshake failures, and ``tools/list`` errors —
    any condition that leaves us unable to enumerate the server's tools.
    """
    pass


@dataclass
class MCPServerInfo:
    name: Optional[str] = None
    title: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None
    icons: Optional[List[dict]] = None


@dataclass
class MCPConnectResult:
    success: bool
    tools: List[MCPToolDefinition] = field(default_factory=list)
    server_info: Optional[MCPServerInfo] = None
    error: Optional[str] = None
    resource_hint: Optional[str] = None  # resource URI from 401 WWW-Authenticate
    resource_metadata_url: Optional[str] = None  # PRM metadata URL from 401 WWW-Authenticate


@dataclass
class MCPToolCallResult:
    success: bool
    content: Optional[List[dict]] = None
    is_error: bool = False
    error: Optional[str] = None
    #: The server rejected the token (HTTP 401), so refreshing it may help.
    #: Callers used to ask ``"401" in result.error`` instead — a substring test
    #: against a message that embeds the server's own response body, so any tool
    #: returning a document mentioning 401 spent a refresh token on nothing.
    auth_expired: bool = False


class MCPClient:
    """Sync MCP client using httpx for connecting to remote MCP servers."""

    PROTOCOL_VERSION = "2024-11-05"
    CLIENT_INFO = {
        "name": "smartloop-cli",
        "version": "1.0.0"
    }

    def __init__(self):
        self._client: Optional[httpx.Client] = None
        self._request_id = 0
        self._session_id: Optional[str] = None

    def __enter__(self):
        self._client = httpx.Client(timeout=60.0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            self._client.close()
            self._client = None
        self._session_id = None

    def _next_request_id(self) -> str:
        self._request_id += 1
        return str(self._request_id)

    def _get_headers(self, access_token: Optional[str], token_type: str = "Bearer") -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    @staticmethod
    def _parse_resource_from_www_authenticate(www_authenticate: str) -> Optional[str]:
        """Parse the resource URI from a WWW-Authenticate header (RFC 6750 / RFC 8707)."""
        if not www_authenticate:
            return None
        # Match resource="..." or resource=<non-whitespace>
        match = re.search(r'resource="([^"]+)"', www_authenticate)
        if match:
            return match.group(1)
        match = re.search(r'resource=(\S+)', www_authenticate)
        if match:
            return match.group(1).rstrip(',')
        return None

    @staticmethod
    def _parse_resource_metadata_from_www_authenticate(www_authenticate: str) -> Optional[str]:
        """Parse the PRM metadata URL from a WWW-Authenticate header (RFC 9728 §5.1).

        The server advertises the Protected Resource Metadata location via
        ``resource_metadata="<url>"``. This is distinct from the RFC 8707
        ``resource=`` field parsed by :meth:`_parse_resource_from_www_authenticate`.
        """
        if not www_authenticate:
            return None
        # Match resource_metadata="..." or resource_metadata=<non-whitespace>.
        # The literal ``_metadata`` suffix keeps this from matching the plain ``resource=`` field.
        match = re.search(r'resource_metadata="([^"]+)"', www_authenticate)
        if match:
            return match.group(1)
        match = re.search(r'resource_metadata=(\S+)', www_authenticate)
        if match:
            return match.group(1).rstrip(',')
        return None

    def _parse_sse_response(self, response_text: str) -> dict:
        data_lines = []
        for line in response_text.split('\n'):
            line = line.strip()
            if not line or line.startswith(':'):
                continue
            if line.startswith('data:'):
                data_content = line[5:].strip()
                if data_content:
                    data_lines.append(data_content)

        if data_lines:
            for data_content in data_lines:
                try:
                    result = json.loads(data_content)
                    if isinstance(result, dict) and "jsonrpc" in result:
                        return result
                except json.JSONDecodeError:
                    continue

            try:
                combined = ''.join(data_lines)
                result = json.loads(combined)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        raise MCPProtocolError("No valid JSON found in SSE response")

    def _send_request(
        self,
        server_url: str,
        method: str,
        params: Optional[dict] = None,
        access_token: Optional[str] = None,
        token_type: str = "Bearer"
    ) -> dict:
        request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": method,
        }
        if params:
            request["params"] = params

        headers = self._get_headers(access_token, token_type)

        try:
            response = self._client.post(
                server_url,
                json=request,
                headers=headers,
            )

            if response.status_code == 401:
                www_auth = response.headers.get("www-authenticate", "")
                # Capture response details for debugging
                try:
                    body = response.text[:500]
                except Exception:
                    body = ""
                auth_sent = headers.get("Authorization", "")
                auth_preview = f"{auth_sent[:30]}..." if len(auth_sent) > 30 else auth_sent
                detail = (
                    f"Authentication failed (401). "
                    f"WWW-Authenticate: {www_auth or '(none)'}. "
                    f"Response: {body or '(empty)'}. "
                    f"Authorization sent: {auth_preview}"
                )
                raise MCPConnectionError(detail, www_authenticate=www_auth, status_code=401)
            if response.status_code == 403:
                raise MCPConnectionError(
                    "Authorization failed: Insufficient permissions", status_code=403)

            response.raise_for_status()

            session_id = response.headers.get("mcp-session-id")
            if session_id:
                self._session_id = session_id

            content_type = response.headers.get("content-type", "")

            if "text/event-stream" in content_type:
                result = self._parse_sse_response(response.text)
            else:
                try:
                    result = response.json()
                except json.JSONDecodeError:
                    result = self._parse_sse_response(response.text)

            if "error" in result:
                error = result["error"]
                raise MCPProtocolError(f"MCP error {error.get('code')}: {error.get('message')}")

            return result.get("result", {})

        except httpx.HTTPStatusError as e:
            raise MCPConnectionError(f"HTTP error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise MCPConnectionError(f"Connection failed: {str(e)}") from e

    def _send_notification(
        self,
        server_url: str,
        method: str,
        params: Optional[dict] = None,
        access_token: Optional[str] = None,
        token_type: str = "Bearer"
    ) -> None:
        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            notification["params"] = params

        headers = self._get_headers(access_token, token_type)

        try:
            self._client.post(
                server_url,
                json=notification,
                headers=headers,
            )
        except Exception:
            pass

    def connect_and_discover_tools(
        self,
        server_url: str,
        access_token: Optional[str] = None,
        token_type: str = "Bearer"
    ) -> MCPConnectResult:
        try:
            init_result = self._send_request(
                server_url,
                "initialize",
                {
                    "protocolVersion": self.PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": self.CLIENT_INFO
                },
                access_token,
                token_type
            )

            raw_server_info = init_result.get("serverInfo", {})
            server_info = None
            if raw_server_info:
                server_info = MCPServerInfo(
                    name=raw_server_info.get("name"),
                    title=raw_server_info.get("title"),
                    version=raw_server_info.get("version"),
                    description=raw_server_info.get("description"),
                    icons=raw_server_info.get("icons"),
                )

            self._send_notification(
                server_url,
                "notifications/initialized",
                {},
                access_token,
                token_type
            )

            tools_result = self._send_request(
                server_url,
                "tools/list",
                {},
                access_token,
                token_type
            )

            raw_tools = tools_result.get("tools", [])
            tools = []
            for tool_data in raw_tools:
                tool = MCPToolDefinition(
                    name=tool_data.get("name", ""),
                    description=tool_data.get("description"),
                    input_schema=tool_data.get("inputSchema")
                )
                tools.append(tool)

            return MCPConnectResult(
                success=True,
                tools=tools,
                server_info=server_info
            )

        except MCPConnectionError as e:
            resource_hint = self._parse_resource_from_www_authenticate(
                e.www_authenticate
            ) if e.www_authenticate else None
            resource_metadata_url = self._parse_resource_metadata_from_www_authenticate(
                e.www_authenticate
            ) if e.www_authenticate else None
            return MCPConnectResult(
                success=False,
                error=str(e),
                resource_hint=resource_hint,
                resource_metadata_url=resource_metadata_url,
            )
        except MCPProtocolError as e:
            return MCPConnectResult(success=False, error=str(e))
        except Exception as e:
            return MCPConnectResult(success=False, error=f"Unexpected error: {str(e)}")

    def _ensure_session(
        self,
        server_url: str,
        access_token: Optional[str] = None,
        token_type: str = "Bearer",
    ) -> None:
        """Perform the MCP initialize handshake if no session exists yet."""
        if self._session_id is not None:
            return

        self._send_request(
            server_url,
            "initialize",
            {
                "protocolVersion": self.PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": self.CLIENT_INFO,
            },
            access_token,
            token_type,
        )
        self._send_notification(
            server_url,
            "notifications/initialized",
            {},
            access_token,
            token_type,
        )

    def call_tool(
        self,
        server_url: str,
        tool_name: str,
        arguments: Optional[dict] = None,
        access_token: Optional[str] = None,
        token_type: str = "Bearer"
    ) -> MCPToolCallResult:
        try:
            self._ensure_session(server_url, access_token, token_type)

            result = self._send_request(
                server_url,
                "tools/call",
                {
                    "name": tool_name,
                    "arguments": arguments or {}
                },
                access_token,
                token_type
            )

            content = result.get("content", [])
            is_error = result.get("isError", False)

            return MCPToolCallResult(
                success=True,
                content=content,
                is_error=is_error
            )

        except (MCPConnectionError, MCPProtocolError) as e:
            logger.error("[mcp-remote] Tool call '%s' on %s failed: %s", tool_name, server_url, e)
            return MCPToolCallResult(
                success=False, error=str(e),
                auth_expired=getattr(e, "status_code", None) == 401,
            )
        except Exception as e:
            logger.error("[mcp-remote] Tool call '%s' on %s unexpected error: %s", tool_name, server_url, e, exc_info=True)
            return MCPToolCallResult(success=False, error=f"Unexpected error: {str(e)}")
