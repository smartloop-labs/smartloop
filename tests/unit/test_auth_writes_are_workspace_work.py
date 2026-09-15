"""Discarding the shared access token belongs to the workspace owner.

``_resolve_user`` drops a dead access token so later requests fall back to the
developer token.  That write is workspace-wide, and routing does not keep it in
the supervisor as reliably as it looks: a project is resolved from a
``project_id`` query parameter on *any* path, so ``/v1/auth/status?project_id=X``
is proxied to an agent child.  A child acting on its own 401 would delete the
credential for every project — and it cannot tell a dead token from one the
owner has already replaced, because it never saw the replacement.
"""

from unittest.mock import MagicMock, patch

import pytest

from smartloop.api.routers import auth as auth_router
from smartloop.auth import credential_store
from smartloop.workspace.context import pin_project


@pytest.fixture(autouse=True)
def _signed_in():
    credential_store.set_access_token("at")
    credential_store.set_refresh_token("rt")
    yield
    for clear in (credential_store.clear_access_token,
                  credential_store.clear_refresh_token):
        clear()


def _refusing_service(status):
    """A platform service whose ``get_current_user`` refuses with *status*."""
    exc = PermissionError(f"HTTP {status}")
    exc.status_code = status
    service = MagicMock()
    service.get_current_user.side_effect = exc
    return service


class TestClearingTheAccessTokenIsOwnerWork:
    def test_a_pinned_process_does_not_clear_the_access_token(self):
        pin_project("alpha")

        with patch.object(auth_router, "get_platform_service",
                          return_value=_refusing_service(401)):
            assert auth_router._resolve_user() is None

        assert credential_store.get_access_token() == "at"

    def test_the_workspace_owner_still_clears_a_dead_access_token(self):
        with patch.object(auth_router, "get_platform_service",
                          return_value=_refusing_service(401)):
            assert auth_router._resolve_user() is None

        assert credential_store.get_access_token() is None
        assert credential_store.get_refresh_token() == "rt", (
            "only the access token is local to auth status"
        )

    @pytest.mark.parametrize("pinned", [None, "alpha"])
    @pytest.mark.parametrize("status", [403, 404])
    def test_a_non_401_leaves_the_credential_alone_in_either_process(self, pinned, status):
        """A refusal that is not about the token is not a reason to drop it.

        ``get_current_user`` raises ``PermissionError`` for every non-transient
        refusal, so an API deploy that moved a route (404) or a tenant policy
        (403) used to log the desktop client out.
        """
        pin_project(pinned)

        with patch.object(auth_router, "get_platform_service",
                          return_value=_refusing_service(status)):
            assert auth_router._resolve_user() is None

        assert credential_store.get_access_token() == "at"
