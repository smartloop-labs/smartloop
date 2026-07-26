"""Local MCP client for subprocess-based (stdio) MCP servers.

MCP over stdio uses newline-delimited JSON-RPC 2.0:
  - Send a JSON object + \\n to the process stdin
  - Read a JSON object + \\n from the process stdout
"""
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Optional

from smartloop.workspace import MCPToolDefinition
from .client import (
    MCPConnectResult,
    MCPServerInfo,
    MCPToolCallResult,
    MCPLaunchError,
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
        # Background readers drain stdout/stderr into these. select() cannot be
        # used on pipe handles on Windows (WinError 10038 "not a socket"), so we
        # read on daemon threads and apply timeouts via Queue.get instead.
        self._stdout_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._stderr_chunks: list[str] = []
        self._stderr_lock = threading.Lock()
        self._reader_stdout: Optional[threading.Thread] = None
        self._reader_stderr: Optional[threading.Thread] = None

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

    @staticmethod
    def _startup_hint(stderr: str) -> str:
        """Return an actionable hint for known launcher failures, else ''.

        ``uvx`` is only a thin trampoline that execs the real ``uv`` binary,
        which it locates solely as a sibling next to itself (it does not search
        PATH). A host with ``uvx`` but no adjacent ``uv`` fails with
        "Could not find the `uv` binary". Point the user at the fix.
        """
        if stderr and "Could not find the `uv` binary" in stderr:
            return (
                "\nHint: `uvx` requires the `uv` binary installed alongside it. "
                "Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` "
                "(Windows: `powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"`)."
            )
        return ""

    def _is_builtin_module(self) -> bool:
        return (
            len(self.args) >= 2
            and self.args[0] == "-m"
            and self.args[1].startswith("smartloop.")
        )

    def __enter__(self) -> "LocalMCPClient":
        if not self.args:
            raise MCPLaunchError(
                f"Cannot launch MCP server: command '{self.command}' has no arguments. "
                "Provide at least the module or package name (e.g. args=['-m', 'smartloop.api.mcp.system_server'] "
                "or args=['yahoo-finance-mcp'])."
            )

        merged_env = {**os.environ, **self.env}
        resolved_command = self._resolve_command(merged_env)

        if self._is_builtin_module():
            import smartloop as _pkg
            pkg_root = os.path.dirname(os.path.dirname(_pkg.__file__))
            existing = merged_env.get("PYTHONPATH", "")
            if pkg_root not in existing.split(os.pathsep):
                merged_env["PYTHONPATH"] = (
                    pkg_root + os.pathsep + existing if existing else pkg_root
                )

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

        # For external processes (uvx, npx, …) give the launcher a moment to
        # either start the MCP server or fail visibly.  Built-in modules start
        # in-process and don't need this guard.
        if not self._is_builtin_module():
            # Poll up to 2 s in 50 ms steps waiting for the process to either
            # become an MCP server (still alive → break) or exit immediately.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                time.sleep(0.05)
                code = self._process.poll()
                if code is None:
                    # Still running — assume the MCP server is ready.
                    break
                # Process exited — read all available stderr and raise now so
                # the caller (model_service's hint-retry logic) can inspect it.
                # No reader threads are running yet, so a direct blocking read is
                # safe here (the pipe is at EOF).
                try:
                    stderr = self._process.stderr.read()
                except Exception:
                    stderr = ""
                raise MCPLaunchError(
                    f"MCP server process exited during startup (exit code {code})."
                    + self._startup_hint(stderr)
                    + (f"\nstderr: {stderr.strip()}" if stderr and stderr.strip() else "")
                )

        self._start_readers()
        return self

    # ------------------------------------------------------------------
    # Background pipe readers (portable replacement for select())
    # ------------------------------------------------------------------

    def _start_readers(self) -> None:
        self._reader_stdout = threading.Thread(
            target=self._pump_stdout, daemon=True
        )
        self._reader_stderr = threading.Thread(
            target=self._pump_stderr, daemon=True
        )
        self._reader_stdout.start()
        self._reader_stderr.start()

    def _pump_stdout(self) -> None:
        stdout = self._process.stdout
        try:
            while True:
                line = stdout.readline()
                if line == "":  # EOF
                    break
                self._stdout_queue.put(line)
        except Exception:
            pass
        finally:
            self._stdout_queue.put(None)  # signal EOF to _recv

    def _pump_stderr(self) -> None:
        stderr = self._process.stderr
        try:
            while True:
                line = stderr.readline()
                if line == "":  # EOF
                    break
                with self._stderr_lock:
                    self._stderr_chunks.append(line)
        except Exception:
            pass

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

    def _read_stderr_nonblocking(self, max_bytes: int = 500) -> str:
        """Return up to *max_bytes* of stderr collected by the reader thread.

        Gives the pump a brief moment to flush any pending diagnostic output so
        error messages can carry a useful stderr tail.
        """
        try:
            if self._reader_stderr is not None:
                self._reader_stderr.join(timeout=0.5)
            with self._stderr_lock:
                data = "".join(self._stderr_chunks)
            return data.strip()[:max_bytes]
        except Exception as e:
            logger.debug("Error reading stderr: %s", e, exc_info=True)
        return ""

    def _send(self, message: dict) -> None:
        assert self._process is not None, "StdioMCPClient not started"
        # Check if the process has already exited before trying to write.
        if self._process.poll() is not None:
            stderr = self._read_stderr_nonblocking()
            raise MCPLaunchError(
                f"MCP process exited (code {self._process.returncode}) before message could be sent."
                + (f" stderr: {stderr}" if stderr else "")
            )
        line = json.dumps(message) + "\n"
        try:
            self._process.stdin.write(line)
            self._process.stdin.flush()
        except BrokenPipeError:
            stderr = self._read_stderr_nonblocking()
            raise MCPLaunchError(
                "MCP process closed its stdin unexpectedly (broken pipe)."
                + (f" stderr: {stderr}" if stderr else "")
            )

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
            try:
                line = self._stdout_queue.get(timeout=remaining)
            except queue.Empty:
                raise MCPProtocolError(
                    f"MCP tool call timed out after {self.timeout}s"
                )
            if line is None:  # stdout reader hit EOF
                exited = self._process.poll() is not None
                if exited:
                    stderr = self._read_stderr_nonblocking()
                    raise MCPLaunchError(
                        f"MCP process exited (code {self._process.returncode})."
                        + (f" stderr: {stderr}" if stderr else "")
                    )
                stderr = self._read_stderr_nonblocking()
                raise MCPProtocolError(
                    "MCP process closed its stdout unexpectedly."
                    + (f" stderr: {stderr}" if stderr else "")
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

        except MCPLaunchError as e:
            raise e
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
            logger.error("[mcp-local] Tool call '%s' failed: %s", tool_name, e)
            return MCPToolCallResult(success=False, error=str(e))
        except Exception as e:
            logger.error("[mcp-local] Tool call '%s' unexpected error: %s", tool_name, e, exc_info=True)
            return MCPToolCallResult(success=False, error=f"Unexpected error: {e}")
