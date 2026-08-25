"""LocalMCPClient must reap its child and bound its buffers.

Two problems, both on the MCP registration path:

* ``__enter__`` raises ``MCPLaunchError`` *after* ``Popen`` when the launcher
  exits immediately.  Python does not call ``__exit__`` when ``__enter__``
  raises, so the child and its three pipe FDs stayed alive for as long as the
  exception's traceback held that frame — which is the whole of
  ``add_local_mcp_server``'s retry ladder, up to two more spawns on top.
* ``_stderr_chunks`` retained the entire stderr stream even though only the
  last few hundred bytes are ever read, so a chatty launcher (uv dependency
  resolution, tracebacks) piled up unboundedly.
"""

from unittest.mock import MagicMock, patch

import pytest

from smartloop.api.services.model_service import ModelService
from smartloop.mcp.client import MCPLaunchError
from smartloop.mcp.local_client import LocalMCPClient, _STDERR_MAX_LINES

_MCP2_TAIL = "ModuleNotFoundError: No module named 'mcp.server.fastmcp'"


def _dead_process(exit_code=1, stderr_text="boom\n" + _MCP2_TAIL):
    proc = MagicMock()
    proc.poll.return_value = exit_code
    proc.stderr.read.return_value = stderr_text
    return proc


class TestFailedEnterCleansUp:
    def test_child_is_reaped_and_pipes_closed(self):
        proc = _dead_process()
        with patch("smartloop.mcp.local_client.subprocess.Popen", return_value=proc), \
             patch.object(LocalMCPClient, "_resolve_command", return_value="uvx"):
            with pytest.raises(MCPLaunchError):
                with LocalMCPClient(command="uvx", args=["demo-mcp"]):
                    pass

        proc.terminate.assert_called_once()
        proc.stdin.close.assert_called_once()
        proc.stdout.close.assert_called_once()
        proc.stderr.close.assert_called_once()

    def test_launch_error_still_carries_the_stderr_tail(self):
        """Cleanup must not cost us the diagnostic — the retry ladder classifies
        the failure from the last traceback line."""
        proc = _dead_process()
        with patch("smartloop.mcp.local_client.subprocess.Popen", return_value=proc), \
             patch.object(LocalMCPClient, "_resolve_command", return_value="uvx"):
            with pytest.raises(MCPLaunchError) as exc:
                with LocalMCPClient(command="uvx", args=["demo-mcp"]):
                    pass

        assert _MCP2_TAIL in str(exc.value)
        assert ModelService._mcp2_incompat_error(str(exc.value)) is True


class TestStderrBufferIsBounded:
    def test_buffer_caps_and_keeps_the_tail(self):
        client = LocalMCPClient(command="uvx", args=["demo-mcp"])
        for i in range(_STDERR_MAX_LINES * 10):
            client._stderr_chunks.append(f"noise line {i}\n")
        client._stderr_chunks.append(_MCP2_TAIL)

        assert len(client._stderr_chunks) <= _STDERR_MAX_LINES
        # The classifiers match the *last* line, so the tail must survive.
        assert client._read_stderr_nonblocking().endswith(_MCP2_TAIL)
        assert ModelService._mcp2_incompat_error(
            client._read_stderr_nonblocking()
        ) is True


class TestExitIsIdempotent:
    def test_exit_can_run_twice(self):
        """__enter__'s failure path calls __exit__, and the `with` statement may
        call it again — neither must raise."""
        proc = _dead_process(exit_code=None)
        client = LocalMCPClient(command="uvx", args=["demo-mcp"])
        client._process = proc
        client.__exit__(None, None, None)
        client.__exit__(None, None, None)
        assert client._process is None
