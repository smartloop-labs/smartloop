"""Local MCP client for subprocess-based (stdio) MCP servers.

MCP over stdio uses newline-delimited JSON-RPC 2.0:
  - Send a JSON object + \\n to the process stdin
  - Read a JSON object + \\n from the process stdout
"""
import json
import logging
import os
import select
import shutil
import subprocess
import sys
import time
from typing import Optional

from smartloop.workspace import MCPToolDefinition
from .client import (
    MCPConnectResult,
    MCPServerInfo,
    MCPToolCallResult,
    MCPProtocolError,
)

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = "2024-11-05"
_CLIENT_INFO = {"name": "smartloop-cli", "version": "1.0.0"}


class LocalMCPClient:
    """Communicates with a local MCP server via stdin/stdout JSON-RPC."""

    def __init__(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.command = command
        self.args = args
        self.env = env or {}
        self.cwd = str(os.path.expanduser(cwd)) if cwd else None
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._initialized = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def _resolve_command(self, env: dict[str, str]) -> str:
        """Resolve the command to a full path.

        Built-in Smartloop servers run as FastMCP modules under the current
        runtime. Everything else (``uvx``, ``npx``, an absolute path, …) is the
        standard MCP stdio config supplied by the caller and is run as-is,
        resolved against the system ``PATH``.
        """
        # Built-in MCP servers are launched as `<interpreter> -m smartloop.<...>`.
        # The interpreter path is captured at registration time and persisted
        # per project, so it goes stale across dev/binary runs and version
        # upgrades (e.g. a saved `.../1.0.23/bin/slp` that argparse rejects, or a
        # venv python from another checkout). Always run our own modules under
        # the *current* runtime executable. When frozen, sys.executable is the
        # `slp` binary, which honours `-m <module>` (see main.py).
        if len(self.args) >= 2 and self.args[0] == "-m" and self.args[1].startswith("smartloop."):
            return sys.executable

        if os.path.isabs(self.command):
            return self.command

        # Only apply PATH workaround inside a frozen (PyInstaller) bundle
        if not getattr(sys, "frozen", False):
            return self.command

        # Try resolving with the current env PATH first
        found = shutil.which(self.command, path=env.get("PATH"))
        if found:
            return found

        # Augment PATH with the Python executable's directory (covers
        # entry-point scripts installed alongside python/pip)
        python_bin_dir = os.path.dirname(sys.executable)
        augmented_path = python_bin_dir + os.pathsep + env.get("PATH", "")
        found = shutil.which(self.command, path=augmented_path)
        if found:
            return found

        return self.command

    def __enter__(self) -> "LocalMCPClient":
        merged_env = {**os.environ, **self.env}
        resolved_command = self._resolve_command(merged_env)
        self._process = subprocess.Popen(
            [resolved_command, *self.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=merged_env,
            cwd=self.cwd,
            text=True,
            bufsize=1,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._process:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Low-level wire protocol
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send(self, message: dict) -> None:
        assert self._process is not None, "StdioMCPClient not started"
        line = json.dumps(message) + "\n"
        self._process.stdin.write(line)
        self._process.stdin.flush()

    def _recv(self, expected_id: int) -> dict:
        """Read lines from stdout until we get a response matching expected_id."""
        assert self._process is not None, "StdioMCPClient not started"
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPProtocolError(
                    f"MCP tool call timed out after {self.timeout}s"
                )
            readable, _, _ = select.select(
                [self._process.stdout], [], [], remaining
            )
            if not readable:
                raise MCPProtocolError(
                    f"MCP tool call timed out after {self.timeout}s"
                )
            line = self._process.stdout.readline()
            if not line:
                stderr = ""
                try:
                    stderr = self._process.stderr.read(500)
                except Exception:
                    pass
                raise MCPProtocolError(
                    f"MCP process closed its stdout unexpectedly. stderr: {stderr}"
                )
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Non-JSON line from MCP stdio process: %r", line)
                continue
            # Accept any message that has the right id, or has no id (notification)
            msg_id = msg.get("id")
            if msg_id is None:
                # server-sent notification — ignore
                continue
            if msg_id == expected_id:
                return msg

    def _request(self, method: str, params: dict | None = None) -> dict:
        req_id = self._next_id()
        req = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            req["params"] = params
        self._send(req)
        resp = self._recv(req_id)
        if "error" in resp:
            err = resp["error"]
            raise MCPProtocolError(f"MCP error {err.get('code')}: {err.get('message')}")
        return resp.get("result", {})

    def _notify(self, method: str, params: dict | None = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)

    # ------------------------------------------------------------------
    # MCP handshake
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._request(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": _CLIENT_INFO,
            },
        )
        self._notify("notifications/initialized", {})
        self._initialized = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect_and_discover_tools(self) -> MCPConnectResult:
        """Perform the MCP handshake and return all advertised tools."""
        try:
            init_result = self._request(
                "initialize",
                {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": _CLIENT_INFO,
                },
            )
            raw_info = init_result.get("serverInfo", {})
            server_info = (
                MCPServerInfo(
                    name=raw_info.get("name"),
                    title=raw_info.get("title"),
                    version=raw_info.get("version"),
                    description=raw_info.get("description"),
                    icons=raw_info.get("icons"),
                )
                if raw_info
                else None
            )
            self._notify("notifications/initialized", {})
            self._initialized = True

            tools_result = self._request("tools/list", {})
            tools = [
                MCPToolDefinition(
                    name=t.get("name", ""),
                    description=t.get("description"),
                    input_schema=t.get("inputSchema"),
                )
                for t in tools_result.get("tools", [])
            ]
            return MCPConnectResult(success=True, tools=tools, server_info=server_info)

        except MCPProtocolError as e:
            return MCPConnectResult(success=False, error=str(e))
        except Exception as e:
            return MCPConnectResult(success=False, error=f"Unexpected error: {e}")

    def call_tool(
        self,
        tool_name: str,
        arguments: dict | None = None,
    ) -> MCPToolCallResult:
        """Call a single tool and return its result."""
        try:
            self._ensure_initialized()
            result = self._request(
                "tools/call",
                {"name": tool_name, "arguments": arguments or {}},
            )
            return MCPToolCallResult(
                success=True,
                content=result.get("content", []),
                is_error=result.get("isError", False),
            )
        except MCPProtocolError as e:
            return MCPToolCallResult(success=False, error=str(e))
        except Exception as e:
            return MCPToolCallResult(success=False, error=f"Unexpected error: {e}")
