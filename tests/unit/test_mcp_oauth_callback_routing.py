"""The OAuth redirect has to come back to the process that can save it.

Registration is project-routed, so the pending session — client_secret,
code_verifier, the curated metadata — is in the child's memory, and only the
child's ``save()`` writes that project's ``project.json``.  The redirect target
was built from ``request.base_url`` and carried no project, which made it the
one MCP route the supervisor answered itself: it looked up a session it did not
have, and on the path where it did have one it would have written nothing.
"""

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from smartloop.api.routers import mcp_servers as mod
from smartloop.supervisor.routing import project_to_proxy


class TestTheRedirectTargetIsAddressedToTheSupervisor:

    def test_it_names_the_project(self):
        url = mod._default_callback_url("p1")
        assert url.endswith("/v1/projects/p1/mcp/oauth/callback")

    def test_it_uses_the_public_port_not_the_childs(self):
        """A child serves on an ephemeral port that is reaped when idle."""
        with patch.object(mod, "AppSettings", lambda: SimpleNamespace(api_port=38540)):
            assert mod._default_callback_url("p1") == (
                "http://localhost:38540/v1/projects/p1/mcp/oauth/callback"
            )

    def test_the_supervisor_forwards_it_to_the_owning_child(self):
        scope = {"path": mod._default_callback_url("p1").split("localhost:38540")[1],
                 "method": "GET"}
        assert project_to_proxy(scope) == "p1"

    def test_the_unscoped_path_still_stops_at_the_supervisor(self):
        """Which is why nothing generates it any more."""
        assert project_to_proxy(
            {"path": "/v1/mcp/oauth/callback", "method": "GET"},
        ) is None


class TestWhoMayFinishTheFlow:
    """``_can_persist_project`` mirrors what ``Workspace.save()`` will do."""

    def _ctx(self, pinned=None, delegated=False, owned=None):
        return patch.multiple(
            "smartloop.workspace.context",
            pinned_project_id=lambda: pinned,
            projects_delegated=lambda: delegated,
            owned_project_id=lambda: owned,
        )

    def test_the_child_pinned_to_the_project_may(self):
        with self._ctx(pinned="p1"):
            assert mod._can_persist_project("p1") is True

    def test_a_child_pinned_to_another_project_may_not(self):
        with self._ctx(pinned="p2"):
            assert mod._can_persist_project("p1") is False

    def test_a_delegating_supervisor_may_not(self):
        """It writes the workspace record and no project file at all."""
        with self._ctx(delegated=True, owned=None):
            assert mod._can_persist_project("p1") is False

    def test_a_supervisor_that_claimed_the_project_may(self):
        with self._ctx(delegated=True, owned="p1"):
            assert mod._can_persist_project("p1") is True

    def test_an_unsupervised_server_may(self):
        with self._ctx():
            assert mod._can_persist_project("p1") is True

    def test_a_session_with_no_project_may_not(self):
        with self._ctx():
            assert mod._can_persist_project(None) is False


@pytest.mark.anyio
class TestTheCallbackRefusesRatherThanSpendTheCode:
    """An authorization code is single-use: failing after the exchange leaves
    the user with nothing to retry."""

    def setup_method(self):
        mod._pending_oauth_sessions.clear()
        mod._completed_oauth_responses.clear()

    async def _call(self, routed_project_id, session_project_id, delegated=False):
        mod._pending_oauth_sessions["st"] = {
            "created_at": time.time(), "project_id": session_project_id,
            "server_url": "https://mcp.notion.com/mcp", "state": "st",
            "code_verifier": "v", "client_id": "c", "client_secret": "s",
            "token_endpoint": "https://t", "authorization_endpoint": "https://a",
            "resource": None, "callback_url": "https://cb", "name": "notion",
            "return_url": None, "description": None, "tags": None,
            "catalog_slug": None,
        }
        with patch.multiple(
            "smartloop.workspace.context",
            pinned_project_id=lambda: None,
            projects_delegated=lambda: delegated,
            owned_project_id=lambda: None,
        ), patch.object(mod, "_exchange_oauth_code") as exchange:
            response = await mod._complete_oauth_redirect(
                code="abc", state="st", error=None, model_service=None,
                routed_project_id=routed_project_id,
            )
        return response, exchange

    async def test_a_delegating_supervisor_refuses_before_exchanging(self):
        response, exchange = await self._call(None, "p1", delegated=True)
        assert response.status_code == 500
        exchange.assert_not_called()

    async def test_a_mismatched_route_refuses_before_exchanging(self):
        """The redirect named one project; the session belongs to another."""
        response, exchange = await self._call("p2", "p1")
        assert response.status_code == 500
        exchange.assert_not_called()
