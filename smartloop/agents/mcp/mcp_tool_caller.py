import httpx
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from smartloop.mcp import MCPClient
from smartloop.mcp.local_client import LocalMCPClient
from smartloop.workspace.workspace import MCPServerType

from .argument_extractor import MCPArgumentExtractor
from .result_mapper import MCPResultMapper

if TYPE_CHECKING:
    from smartloop.workspace.workspace import MCPServer

logger = logging.getLogger(__name__)


class MCPToolCaller:
    """Handles MCP server communication: tool calls, server lookup, auth, and prerequisites."""

    def __init__(self, arg_extractor: MCPArgumentExtractor, result_mapper: MCPResultMapper,
                 persist: Optional[Callable[[], None]] = None):
        self.arg_extractor = arg_extractor
        self.result_mapper = result_mapper
        #: Called after a refresh replaces a server's credentials, to write them
        #: back to the workspace.  Optional because one caller — the content
        #: crawl — already persists through its own ``on_server_updated`` hook.
        self._persist = persist

    def set_persist(self, persist: Optional[Callable[[], None]]) -> None:
        """Install the callback that writes refreshed credentials back to disk."""
        self._persist = persist

    def call_tool(
        self,
        server: "MCPServer",
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        access_token: Optional[str] = None,
        cwd: Optional[str] = None,
    ):
        if server.server_type == MCPServerType.LOCAL and server.config:
            with LocalMCPClient(
                command=server.config.command,
                args=server.config.args,
                env=server.config.env,
                cwd=cwd or server.config.cwd,
            ) as client:
                result = client.call_tool(tool_name, arguments)
                if not result.success:
                    logger.error(
                        "[tool_caller] Local tool '%s' on server '%s' failed: %s",
                        tool_name, server.name, result.error,
                    )
                return result
        else:
            server_url = self.get_server_url(server)
            with MCPClient() as client:
                result = client.call_tool(
                    server_url=server_url,
                    tool_name=tool_name,
                    arguments=arguments,
                    access_token=access_token,
                )
                if not result.success:
                    logger.error(
                        "[tool_caller] Remote tool '%s' on server '%s' (%s) failed: %s",
                        tool_name, server.name, server_url, result.error,
                    )
                return result

    def find_server(
        self,
        mcp_servers: List["MCPServer"],
        server_id: str,
        server_name: str,
    ) -> Optional["MCPServer"]:
        for s in mcp_servers:
            if s.id == server_id:
                return s
        for s in mcp_servers:
            if s.name == server_name:
                return s
        for s in mcp_servers:
            for t in s.tools:
                if t.name == server_id or t.name == server_name:
                    logger.info(
                        f"[find_server] Resolved '{server_id}/{server_name}' "
                        f"via tool-name match on server '{s.name}' (id={s.id})"
                    )
                    return s
        return None

    def find_tool_schema(self, server: "MCPServer", tool_name: str) -> Optional[Dict[str, Any]]:
        for tool in server.tools:
            if tool.name == tool_name:
                return {
                    'name': tool.name,
                    'description': tool.description,
                    'input_schema': tool.input_schema,
                }
        return None

    def find_tool_by_keyword(self, server: "MCPServer", instruction: str) -> Optional[str]:
        if not instruction:
            return None
        instr_lower = instruction.lower()
        best_tool = None
        best_score = 0
        for tool in server.tools:
            score = 0
            t_name = (tool.name or "").lower()
            t_desc = (tool.description or "").lower()
            for word in re.findall(r"\w{3,}", instr_lower):
                if word in t_name:
                    score += 3
                if word in t_desc:
                    score += 1
            if score > best_score:
                best_score = score
                best_tool = tool.name
        return best_tool if best_score > 0 else None

    @staticmethod
    def get_server_url(server: "MCPServer") -> Optional[str]:
        if server.remote_config:
            return server.remote_config.server_url
        return None

    @staticmethod
    def get_access_token(server: "MCPServer") -> Optional[str]:
        if server.oauth_credentials and server.oauth_credentials.access_token:
            return server.oauth_credentials.access_token
        # A server registered from a URL carrying its own credential
        # (``?token=...``) keeps it on the remote config, not in OAuth
        # credentials there are none of.  Discovery already falls back here, so
        # without the same fallback a static-token connection lists its tools
        # and then 401s on every call to them.
        if server.remote_config and server.remote_config.token:
            return server.remote_config.token
        return None

    def refresh_access_token(self, server: "MCPServer") -> Optional[str]:
        """Redeem *server*'s refresh token, storing the result on the server.

        The new credentials are written back through :attr:`_persist`.  Leaving
        them in memory was a slow-acting break: providers that rotate refresh
        tokens hand back a new one and retire the old, so a refresh during a chat
        turn left the workspace holding a refresh token the provider had already
        invalidated.  The connection kept working until the process restarted and
        then could never be renewed again — only the content crawl persisted, so
        which of the two paths happened to run last decided whether a connection
        survived.
        """
        creds = server.oauth_credentials
        if not creds or not creds.refresh_token:
            logger.warning(f"No refresh token available for server '{server.name}'")
            return None

        remote = server.remote_config
        if not remote or not remote.token_endpoint:
            logger.warning(f"No token endpoint configured for server '{server.name}'")
            return None

        try:
            payload = {
                "grant_type": "refresh_token",
                "refresh_token": creds.refresh_token,
                "client_id": creds.client_id,
            }
            if creds.client_secret:
                payload["client_secret"] = creds.client_secret
            # RFC 8707: a refresh must carry the same resource indicator as the
            # grant it descends from, or strict servers (monday.com) reject it.
            if remote.resource:
                payload["resource"] = remote.resource

            with httpx.Client(timeout=30.0) as http:
                resp = http.post(remote.token_endpoint, data=payload)
                resp.raise_for_status()
                data = resp.json()

            new_access = data.get("access_token")
            if not new_access:
                logger.error("Token refresh response missing access_token")
                return None

            creds.access_token = new_access
            if data.get("refresh_token"):
                creds.refresh_token = data["refresh_token"]
            if data.get("expires_in"):
                creds.expires_at = time.time() + data["expires_in"]

            if self._persist is not None:
                try:
                    self._persist()
                except Exception as e:
                    # The token in memory is good, so the call this refresh was
                    # for should still go ahead; it is the *next* process that
                    # pays for a failed write.
                    logger.warning(
                        f"Refreshed '{server.name}' but could not persist credentials: {e}"
                    )

            logger.info(f"Refreshed access token for server '{server.name}'")
            return new_access

        except Exception as e:
            logger.error(f"Token refresh failed for server '{server.name}': {e}")
            return None

    def validate_argument_values(
        self,
        arguments: Dict[str, Any],
        properties: Dict[str, Any],
    ) -> List[str]:
        issues: List[str] = []

        for param_name, value in arguments.items():
            if value is None:
                continue
            param_info = properties.get(param_name, {})

            if self.arg_extractor._is_url_parameter(param_name, param_info):
                if isinstance(value, str) and not re.match(r'^https?://', value.strip()):
                    issues.append(
                        f"'{param_name}' expects a URL but got plain text: "
                        f"{value[:80]}..."
                    )
                continue

            if self.arg_extractor._is_id_parameter(param_name, param_info):
                if isinstance(value, str) and not self.arg_extractor._is_valid_id_value(value):
                    issues.append(
                        f"'{param_name}' expects an identifier but got: "
                        f"{value[:80]}..."
                    )
                continue

            param_type = param_info.get('type', '')
            if param_type in ('integer', 'number') and isinstance(value, str):
                try:
                    float(value)
                except ValueError:
                    issues.append(
                        f"'{param_name}' expects a {param_type} but got: "
                        f"{value[:80]}"
                    )
                continue

            enum_values = param_info.get('enum')
            if enum_values and value not in enum_values:
                issues.append(
                    f"'{param_name}' expects one of {enum_values} but got: "
                    f"{value}"
                )

        return issues

    def execute_prerequisites(
        self,
        dep_analysis,
        server: "MCPServer",
        user_message: str,
        input_schema: Dict[str, Any],
        previous_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        resolved_ids = {}
        access_token = self.get_access_token(server)
        properties = input_schema.get('properties', {})

        prior_corpus = ""
        if previous_results:
            prior_corpus = "\n".join(
                str(r.get("result", "")) for r in previous_results
            )

        for dep in dep_analysis.dependencies:
            if not dep.prerequisite_tool:
                continue

            if prior_corpus:
                resolved = self.result_mapper.get_resolved_arguments(
                    required_params=[dep.param_name],
                    properties=properties,
                    user_input=prior_corpus,
                )
                if resolved.get(dep.param_name):
                    logger.info(
                        f"[prereq] Resolved '{dep.param_name}' from previous results "
                        f"(skipping API call to '{dep.prerequisite_tool}')"
                    )
                    resolved_ids.update(resolved)
                    continue

            logger.info(f"[prereq] Calling prerequisite tool: {dep.prerequisite_tool}")
            try:
                result = self.call_tool(
                    server=server,
                    tool_name=dep.prerequisite_tool,
                    arguments={},
                    access_token=access_token,
                )

                if result.success and result.content:
                    self.result_mapper.process_list_result(
                        tool_name=dep.prerequisite_tool,
                        result=result.content,
                    )
                    resolved = self.result_mapper.get_resolved_arguments(
                        required_params=[dep.param_name],
                        properties=properties,
                        user_input=user_message,
                    )
                    resolved_ids.update(resolved)
                    logger.info(f"[prereq] Resolved from API call: {resolved}")
                else:
                    logger.warning(
                        f"[prereq] Prerequisite tool '{dep.prerequisite_tool}' failed: {result.error}"
                    )

            except Exception as e:
                logger.error(f"[prereq] Failed to execute prerequisite '{dep.prerequisite_tool}': {e}")

        return resolved_ids
