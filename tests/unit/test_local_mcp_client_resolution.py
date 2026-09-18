"""Which program a local MCP server's command actually resolves to.

A definition can name a server the app ships with — a copy next to the ``slp``
binary (the app bundle when packaged), or one a project keeps under the
framework home.  Resolution has to find it from a bare name or a path relative
to those anchors, and inside the macOS App Sandbox it has to *prefer* it: only
the bundle and the container may be exec'd, so the user's own install at the
same basename is refused with EPERM however the path was written.

Outside the sandbox an absolute command still wins outright — the app does not
silently redirect a path the user configured.
"""

import os

import pytest

from smartloop.mcp import local_client as local_client_module
from smartloop.mcp.local_client import LocalMCPClient

_CONTAINER_HOME = "/Users/tester/Library/Containers/ai.smartloop.studio/Data"


def _exe(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def slp_dir(tmp_path):
    return tmp_path / "slp-bin"


@pytest.fixture
def home(tmp_path):
    return tmp_path / "slp-home"


@pytest.fixture(autouse=True)
def _spl_on_this_machine(slp_dir, monkeypatch):
    # Resolution reads the running interpreter's directory as the slp binary's;
    # pin it at a temp dir so the test never consults a real install.
    monkeypatch.setattr(
        local_client_module.sys, "executable", str(slp_dir / "slp"),
    )


def _sandboxed(monkeypatch, home):
    monkeypatch.setenv("HOME", _CONTAINER_HOME)
    monkeypatch.setenv("SLP_HOME", str(home))


def _developer(monkeypatch, home):
    monkeypatch.setenv("HOME", "/Users/tester")
    monkeypatch.setenv("SLP_HOME", str(home))


def test_the_sandbox_prefers_the_copy_next_to_the_slp_binary(
    monkeypatch, tmp_path, slp_dir, home,
):
    _sandboxed(monkeypatch, home)
    bundled = _exe(slp_dir / "craft")
    installed = _exe(tmp_path / "user" / ".local" / "bin" / "craft")
    client = LocalMCPClient(command=str(installed), args=["mcp"])
    assert client._resolve_command({"PATH": ""}) == str(bundled)


def test_the_sandbox_falls_back_to_a_copy_under_the_framework_home(
    monkeypatch, tmp_path, home,
):
    _sandboxed(monkeypatch, home)
    in_home = _exe(home / "craft")
    installed = _exe(tmp_path / "user" / ".local" / "bin" / "craft")
    client = LocalMCPClient(command=str(installed), args=["mcp"])
    assert client._resolve_command({"PATH": ""}) == str(in_home)


def test_the_sandbox_keeps_the_configured_path_without_a_bundled_copy(
    monkeypatch, tmp_path, home,
):
    # Nothing to redirect to: the caller keeps the path, the spawn fails with
    # EPERM, and the discovery message explains the sandbox.  Silently running
    # *some other* `craft` would be worse than the honest failure.
    _sandboxed(monkeypatch, home)
    installed = _exe(tmp_path / "user" / ".local" / "bin" / "craft")
    client = LocalMCPClient(command=str(installed), args=["mcp"])
    assert client._resolve_command({"PATH": ""}) == str(installed)


def test_the_developer_build_keeps_an_absolute_command(
    monkeypatch, tmp_path, slp_dir, home,
):
    # A bundled copy next to slp must not shadow the path a user configured
    # when the app is not sandboxed.
    _developer(monkeypatch, home)
    _exe(slp_dir / "craft")
    installed = _exe(tmp_path / "user" / ".local" / "bin" / "craft")
    client = LocalMCPClient(command=str(installed), args=["mcp"])
    assert client._resolve_command({"PATH": ""}) == str(installed)


def test_a_bundled_copy_is_found_from_a_bare_name(
    monkeypatch, slp_dir, home,
):
    _developer(monkeypatch, home)
    bundled = _exe(slp_dir / "craft")
    client = LocalMCPClient(command="craft", args=["mcp"])
    assert client._resolve_command({"PATH": ""}) == str(bundled)


def test_a_path_relative_to_the_framework_home_is_resolved(
    monkeypatch, home,
):
    _developer(monkeypatch, home)
    in_home = _exe(home / "bin" / "craft")
    client = LocalMCPClient(command="bin/craft", args=["mcp"])
    assert client._resolve_command({"PATH": ""}) == str(in_home)


def test_the_path_wins_over_the_anchors_for_a_bare_name(
    monkeypatch, tmp_path, slp_dir, home,
):
    # Anchors are the fallback, not an override: a `craft` the user put on
    # PATH is still the one that runs in a build that can launch it.
    _developer(monkeypatch, home)
    _exe(slp_dir / "craft")
    on_path = _exe(tmp_path / "path-bin" / "craft")
    client = LocalMCPClient(command="craft", args=["mcp"])
    assert client._resolve_command({"PATH": str(on_path.parent)}) == str(on_path)
