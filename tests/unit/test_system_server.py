"""Tests for the smartloop system-tools MCP server (bash/file-read/
file-write/grep), invoked the same way the orchestrator invokes it — a real
stdio subprocess via LocalMCPClient, with ``cwd`` set to the work dir."""

import os
import sys
from pathlib import Path

import pytest

from smartloop.mcp.local_client import LocalMCPClient


def _client(cwd):
    return LocalMCPClient(
        command=sys.executable,
        args=["-m", "smartloop.api.mcp.system_server"],
        cwd=str(cwd),
    )


def _text(result):
    return " ".join(
        b.get("text", "") if isinstance(b, dict) else str(b)
        for b in (result.content or [])
    )


def test_file_write_then_read_relative_path(tmp_path):
    """A relative path resolves against the subprocess cwd (the work dir)."""
    with _client(tmp_path) as c:
        w = c.call_tool("file-write", {"path": "out/report.md", "content": "hello world"})
        assert w.success
        assert (tmp_path / "out" / "report.md").read_text() == "hello world"

        r = c.call_tool("file-read", {"path": "out/report.md"})
        assert r.success
        assert "hello world" in _text(r)


def test_file_write_strips_code_fence(tmp_path):
    """A fenced ```` ``` ```` wrapper the model adds is stripped before write."""
    fenced = "```markdown\n# Title\n\nbody\n```"
    with _client(tmp_path) as c:
        w = c.call_tool("file-write", {"path": "doc.md", "content": fenced})
        assert w.success
    written = (tmp_path / "doc.md").read_text()
    assert written == "# Title\n\nbody\n"
    assert "```" not in written


def test_grep_finds_matches(tmp_path):
    (tmp_path / "a.py").write_text("alpha = 1\nNEEDLE here\n")
    (tmp_path / "b.txt").write_text("no match\n")
    with _client(tmp_path) as c:
        r = c.call_tool("grep", {"pattern": "NEEDLE"})
        assert r.success
        out = _text(r)
        assert "a.py" in out
        assert "NEEDLE" in out


def test_grep_glob_filter(tmp_path):
    (tmp_path / "a.py").write_text("TARGET\n")
    (tmp_path / "b.txt").write_text("TARGET\n")
    with _client(tmp_path) as c:
        r = c.call_tool("grep", {"pattern": "TARGET", "glob": "*.py"})
        assert r.success
        out = _text(r)
        assert "a.py" in out
        assert "b.txt" not in out


def test_grep_no_matches_is_error(tmp_path):
    """A miss raises so the orchestrator skips the step and the workspace
    search fallback fires, instead of 'No matches' suppressing it."""
    (tmp_path / "a.py").write_text("nothing\n")
    with _client(tmp_path) as c:
        r = c.call_tool("grep", {"pattern": "ZZZ_NOPE"})
        assert r.is_error
        assert "no matches" in _text(r).lower()


def test_grep_bad_regex_is_error(tmp_path):
    """A real grep failure (exit code 2) is an error, not a clean miss."""
    (tmp_path / "a.py").write_text("content\n")
    with _client(tmp_path) as c:
        r = c.call_tool("grep", {"pattern": "["})
        assert r.is_error
        assert "no matches" not in _text(r).lower()


def test_file_read_missing_path_is_error(tmp_path):
    with _client(tmp_path) as c:
        r = c.call_tool("file-read", {"path": "does/not/exist.md"})
        assert r.is_error
        assert "not found" in _text(r)


def test_file_write_empty_path_is_error(tmp_path):
    with _client(tmp_path) as c:
        r = c.call_tool("file-write", {"path": "", "content": "hello"})
        assert r.is_error


def test_bash_nonzero_exit_is_normal_output(tmp_path):
    """A command that ran to completion is a real result even when it
    fails — its exit code and output belong in the answer."""
    with _client(tmp_path) as c:
        r = c.call_tool("bash", {"command": "echo boom >&2; exit 3"})
        assert r.success
        assert not r.is_error
        out = _text(r)
        assert "exit code 3" in out
        assert "boom" in out


def test_bash_timeout_is_error(tmp_path):
    with _client(tmp_path) as c:
        r = c.call_tool("bash", {"command": "sleep 5", "timeout": 1})
        assert r.is_error
        assert "timed out" in _text(r)


def test_bash_cwd_relative_subdir_is_honoured(tmp_path):
    """A relative cwd resolves against the work dir and is used as-is."""
    (tmp_path / "sub").mkdir()
    with _client(tmp_path) as c:
        r = c.call_tool("bash", {"command": "pwd", "cwd": "sub"})
        assert r.success
        assert os.path.realpath(tmp_path / "sub") in _text(r)


@pytest.mark.parametrize("escape", ["..", "../..", "/", "~", str(Path.home())])
def test_bash_cwd_outside_work_dir_is_clamped(tmp_path, escape):
    """A cwd pointing outside the project dir — traversal, absolute, or the
    user's home — is dropped in favour of the work dir itself."""
    with _client(tmp_path) as c:
        r = c.call_tool("bash", {"command": "pwd", "cwd": escape})
        assert r.success
        assert _text(r).strip().endswith(os.path.realpath(tmp_path))


def test_bash_cwd_missing_dir_falls_back_to_work_dir(tmp_path):
    """An in-bounds path that does not exist still runs, from the work dir,
    rather than failing to spawn."""
    with _client(tmp_path) as c:
        r = c.call_tool("bash", {"command": "pwd", "cwd": "no/such/dir"})
        assert r.success
        assert _text(r).strip().endswith(os.path.realpath(tmp_path))
