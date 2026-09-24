import logging
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from smartloop.mcp import MCPClient
from smartloop.mcp.local_client import LocalMCPClient
from smartloop.workspace.workspace import MCPServerStatus, MCPServerType

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
        #: back to the workspace.  The hook is the *only* thing that persists
        #: them: the content crawl, which used to write them through its own
        #: ``on_server_updated``, was removed with indexing in ``abc232b``.
        self._persist = persist
        #: Serializes redemptions.  Parallel step groups and tool fan-out share
        #: one caller, so several calls can 401 together and each redeem the
        #: same refresh token — and a provider that issues them single-use
        #: invalidates the grant on the second attempt.
        self._refresh_lock = threading.Lock()

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

    def call_with_refresh(
        self,
        server: "MCPServer",
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        cwd: Optional[str] = None,
    ):
        """Call a tool, renewing an expired token once and trying again.

        Returns ``(result, refreshed)`` — ``refreshed`` says the server's
        credentials changed, so a caller that owns the write-back knows there is
        something to write.  (:attr:`_persist` already handles the callers that
        do not.)

        Every MCP call path had its own copy of this, and they had drifted: two
        guarded on a server URL, one reported the rotation, one retried with the
        wrong tool name, and the collect path — the main one since indexing was
        replaced by live tool calls — had no retry at all, so an aged-out
        connection looked to retrieval like an app with nothing in it.
        """
        access_token = self.get_access_token(server)
        result = self.call_tool(
            server=server, tool_name=tool_name,
            arguments=arguments, access_token=access_token, cwd=cwd,
        )
        if result.success or not result.auth_expired:
            return result, False

        new_token = self.refresh_access_token(server, stale_token=access_token)
        if not new_token:
            return result, False

        result = self.call_tool(
            server=server, tool_name=tool_name,
            arguments=arguments, access_token=new_token, cwd=cwd,
        )
        return result, True

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

    def refresh_access_token(
        self, server: "MCPServer", stale_token: Optional[str] = None,
    ) -> Optional[str]:
        """Redeem *server*'s refresh token, storing the result on the server.

        The new credentials are written back through :attr:`_persist`.  Leaving
        them in memory was a slow-acting break: providers that rotate refresh
        tokens hand back a new one and retire the old, so a refresh during a chat
        turn left the workspace holding a refresh token the provider had already
        invalidated.  The connection kept working until the process restarted and
        then could never be renewed again.

        *stale_token* is the access token the caller's request failed with.
        Given it, a redemption that lost the race to another thread returns that
        thread's token instead of spending the refresh token a second time —
        which is the thing that kills a single-use grant outright.
        """
        with self._refresh_lock:
            return self._refresh_locked(server, stale_token)

    def _refresh_locked(
        self, server: "MCPServer", stale_token: Optional[str],
    ) -> Optional[str]:
        creds = server.oauth_credentials
        if stale_token and creds and creds.access_token not in (None, stale_token):
            # Another caller rotated it while this one waited for the lock.  Its
            # result is good and already persisted; redeeming again would not
            # improve on it and, single-use, would invalidate it.
            logger.info(
                f"Reusing the token another caller just refreshed for '{server.name}'"
            )
            return creds.access_token

        if not creds or not creds.refresh_token:
            # Nothing to redeem.  Callers reach this only after a 401, so for a
            # pasted credential that 401 is the verdict on the credential
            # itself rather than a sign its token aged out.
            self.record_credential_rejected(server)
            logger.warning(f"No refresh token available for server '{server.name}'")
            return None

        remote = server.remote_config
        if not remote or not remote.token_endpoint:
            logger.warning(f"No token endpoint configured for server '{server.name}'")
            return None

        from smartloop.mcp.oauth import MCPOAuthService, TokenExchangeError

        try:
            with MCPOAuthService() as oauth:
                # RFC 8707: a refresh must carry the same resource indicator as
                # the grant it descends from, or strict servers (monday.com)
                # reject it.
                tokens = oauth.refresh_access_token(
                    token_endpoint=remote.token_endpoint,
                    refresh_token=creds.refresh_token,
                    client_id=creds.client_id,
                    client_secret=creds.client_secret,
                    resource=remote.resource,
                )
        except TokenExchangeError as e:
            logger.error(f"Token refresh failed for server '{server.name}': {e}")
            if e.grant_is_dead:
                # The server has retired this grant.  Re-minting it cannot work
                # again, so this is the one refusal that should cost the user
                # their connection — a blip or a 5xx must not.
                self.record_grant_expired(server)
            return None
        except Exception as e:
            logger.error(f"Token refresh failed for server '{server.name}': {e}")
            return None

        creds.access_token = tokens.access_token
        if tokens.refresh_token:
            creds.refresh_token = tokens.refresh_token
        if tokens.expires_in:
            creds.expires_at = time.time() + tokens.expires_in

        self._write_back(server, "Refreshed")
        logger.info(f"Refreshed access token for server '{server.name}'")
        return tokens.access_token

    def _write_back(self, server: "MCPServer", what: str) -> None:
        """Persist a change to *server*, logging rather than raising on failure.

        The state in memory is good, so whatever this change was made for should
        still go ahead; it is the *next* process that pays for a failed write.
        """
        if self._persist is None:
            return
        try:
            self._persist()
        except Exception as e:
            logger.warning(
                f"{what} '{server.name}' but could not persist the change: {e}"
            )

    def record_credential_rejected(self, server: "MCPServer") -> bool:
        """Fail a connection whose pasted credential the server refused.

        Only for a credential that was pasted rather than granted.  An OAuth
        connection has a refresh to fall back on and a failed refresh can be
        transient, but a token the server has stopped accepting never starts
        working again on its own.  Left unreported it looks exactly like a tool
        with no data: ``status: ready``, an empty answer, nothing in the log.

        Returns whether the server's state changed.
        """
        remote = server.remote_config
        if not remote or not remote.token:
            return False
        return self._fail(
            server,
            "The access token for this connection was rejected. "
            "Enter a new token to reconnect.",
        )

    def record_grant_expired(self, server: "MCPServer") -> bool:
        """Fail a connection whose OAuth grant the server has retired.

        Reached only for ``invalid_grant``/``invalid_client`` — a refusal that
        re-minting the same grant cannot recover from.  Until this existed a
        dead connection reported ``ready`` with its full tool list forever,
        while every call 401'd and was swallowed.
        """
        return self._fail(
            server,
            "This connection's authorization has expired. Reconnect it to "
            "keep using it.",
        )

    def _fail(self, server: "MCPServer", message: str) -> bool:
        if server.status == MCPServerStatus.FAILED and server.last_error == message:
            return False
        server.status = MCPServerStatus.FAILED
        server.last_error = message
        logger.warning(
            f"Server '{server.name}' can no longer authenticate: {message}"
        )
        self._write_back(server, "Failed")
        return True

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
                result, _refreshed = self.call_with_refresh(
                    server=server,
                    tool_name=dep.prerequisite_tool,
                    arguments={},
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
