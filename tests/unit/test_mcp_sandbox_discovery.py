"""The discovery error a sandboxed build reports for a local MCP server.

Inside the macOS App Sandbox the tilde is resolved correctly (the path in the
error is the right one) but the sandbox refuses to exec any program outside the
app bundle with EPERM — "[Errno 1] Operation not permitted: '/path'".  That is
not a misconfiguration a retry ladder can walk back, so the friendly mapping
must say what it is instead of handing an OS error to the user to chase.  The
Developer ID build (no sandbox in the environment) keeps the raw first line.
"""

import pytest

from smartloop.api.services.model_service import ModelService

_CONTAINER_HOME = (
    "/Users/tester/Library/Containers/ai.smartloop.studio/Data"
)
_EPERM = (
    "[Errno 1] Operation not permitted: "
    "'/Users/mehfuzh/.local/bin/craft'"
)


@pytest.fixture(autouse=True)
def _clean_sandbox_env(monkeypatch):
    monkeypatch.delenv("SLP_HOME", raising=False)


def test_an_eperm_under_the_sandbox_names_the_build_limitation(monkeypatch):
    monkeypatch.setenv("HOME", _CONTAINER_HOME)
    msg = ModelService._friendly_discovery_error(_EPERM)
    assert "App Sandbox" in msg
    assert "build from smartloop.ai" in msg
    assert "/Users/mehfuzh/.local/bin/craft" in msg


def test_an_eperm_under_the_sandbox_named_by_slp_home(monkeypatch):
    monkeypatch.setenv("HOME", "/Users/tester")
    monkeypatch.setenv(
        "SLP_HOME",
        _CONTAINER_HOME + "/Library/Application Support/smartloop-studio/slp",
    )
    msg = ModelService._friendly_discovery_error(_EPERM)
    assert "App Sandbox" in msg


def test_the_same_eperm_outside_the_sandbox_stays_raw(monkeypatch):
    # A Developer ID build has a real HOME, no container anywhere: whatever
    # denial this is (odd permissions, a NOEXEC mount, …) is a genuine upload
    # problem, not a sandbox one, so the mapping must not dress it up.
    monkeypatch.setenv("HOME", "/Users/tester")
    assert ModelService._friendly_discovery_error(_EPERM) == _EPERM


def test_unrelated_discovery_errors_are_unchanged_under_the_sandbox(monkeypatch):
    monkeypatch.setenv("HOME", _CONTAINER_HOME)
    assert (
        ModelService._friendly_discovery_error(
            "ModuleNotFoundError: No module named 'mcp.server.fastmcp'"
        )
        == "This package isn't compatible with the current MCP SDK — its "
        "dependencies are out of date. Try another package, or ask its "
        "maintainer to update it."
    )
    assert ModelService._friendly_discovery_error("") == (
        "Couldn't start this connection."
    )