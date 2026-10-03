"""An agent started on port 0 reports the port it actually bound.

The launcher asks for a free port with ``SLP_API_PORT=0``.  The settings kept
saying 0 after the bind, and the MCP OAuth redirect is built from them, so a
provider sent the browser back to ``localhost:0`` and the sign-in never
returned.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from smartloop.api import main
from smartloop.api.routers import mcp_servers


@pytest.fixture(autouse=True)
def _restore_port(monkeypatch):
    monkeypatch.setenv("SLP_API_PORT", "0")
    monkeypatch.setattr(main.api_settings, "api_port", 0)


def test_the_redirect_names_the_bound_port():
    with patch.object(main, "get_supervisor", return_value=None):
        main._adopt_bound_port(52575)
    assert mcp_servers._default_callback_url("p1") == (
        "http://localhost:52575/v1/projects/p1/mcp/oauth/callback"
    )


def test_children_inherit_the_bound_port():
    """A child builds the redirect from the environment it was spawned with."""
    with patch.object(main, "get_supervisor", return_value=None):
        main._adopt_bound_port(52575)
    assert main.os.environ["SLP_API_PORT"] == "52575"
    assert main.api_settings.api_port == 52575


def test_the_supervisor_record_names_the_bound_port():
    """The supervisor starts during lifespan startup, before the bind."""
    supervisor = SimpleNamespace(registry=MagicMock())
    with patch.object(main, "get_supervisor", return_value=supervisor):
        main._adopt_bound_port(52575)
    supervisor.registry.set_supervisor.assert_called_once_with(main.os.getpid(), 52575)
