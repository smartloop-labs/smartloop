"""Redeeming the refresh token belongs to the process that owns the workspace.

Under supervision the server is not one process but N+1: a supervisor plus an
agent child per project, all running the same app against one credential file.
Everything that stops two *threads* redeeming the same single-use refresh token
— ``_refresh_lock``, the failed-refresh cooldown — is process-local, so none of
it orders two *processes*.

The consequence is not a wasted request.  The second redemption of a rotated
token is refused, ``_redeem`` reads that refusal as "the refresh token is dead",
and clears both credentials: every project logged out, by the machinery that
exists to keep people logged in.  So a pinned process declines, and the
workspace owner is the only one that ever spends the token.
"""

from unittest.mock import MagicMock

import pytest

from smartloop.auth import credential_store
from smartloop.platform.client import SLPPlatformClient
from smartloop.platform.token_agent import refresh_access_token_now
from smartloop.workspace.context import pin_project


@pytest.fixture(autouse=True)
def _clean_credentials():
    for clear in (credential_store.clear_access_token,
                  credential_store.clear_refresh_token,
                  credential_store.clear_token):
        clear()
    yield
    for clear in (credential_store.clear_access_token,
                  credential_store.clear_refresh_token,
                  credential_store.clear_token):
        clear()


def _rotating_client():
    """A client whose refresh endpoint hands back a rotated pair."""
    client = MagicMock(spec=SLPPlatformClient)
    client.refresh_access_token.return_value = {
        "access_token": "new-at",
        "refresh_token": "new-rt",
    }
    return client


class TestOnlyTheWorkspaceOwnerRedeems:
    def test_a_pinned_child_does_not_redeem(self):
        credential_store.set_refresh_token("rt")
        pin_project("alpha")
        client = _rotating_client()

        assert refresh_access_token_now(client) is False
        client.refresh_access_token.assert_not_called()

    def test_a_pinned_child_does_not_clear_on_a_dead_refresh_token(self):
        """The regression.  A child must not turn its own 401 into a logout.

        Without the gate this is what a child racing the supervisor looks like:
        it presents a token the supervisor already spent, the platform refuses
        it, and ``_redeem`` deletes both credentials for every project.
        """
        credential_store.set_access_token("at")
        credential_store.set_refresh_token("rt")
        pin_project("alpha")

        client = MagicMock(spec=SLPPlatformClient)
        client.refresh_access_token.side_effect = PermissionError("HTTP 401")

        assert refresh_access_token_now(client) is False
        client.refresh_access_token.assert_not_called()
        assert credential_store.get_access_token() == "at"
        assert credential_store.get_refresh_token() == "rt"

    def test_a_pinned_child_reads_no_credentials_at_all(self):
        """The gate is the first statement, ahead of any credential I/O.

        White-box on purpose: a child answering 401s during a chat turn should
        not be decrypting the credential file to discover it has nothing to do.
        """
        credential_store.set_refresh_token("rt")
        pin_project("alpha")

        with pytest.MonkeyPatch.context() as mp:
            def _boom():
                raise AssertionError("a pinned process read the credential store")

            mp.setattr(credential_store, "get_access_token", _boom)
            assert refresh_access_token_now(_rotating_client()) is False

    def test_the_workspace_owner_still_redeems(self):
        credential_store.set_access_token("old-at")
        credential_store.set_refresh_token("old-rt")
        client = _rotating_client()

        assert refresh_access_token_now(client) is True
        client.refresh_access_token.assert_called_once_with("old-rt")
        assert credential_store.get_access_token() == "new-at"
        assert credential_store.get_refresh_token() == "new-rt"

    def test_the_default_client_hook_is_gated_too(self):
        """The gate covers the path ``_authed`` actually takes.

        ``_authed`` calls ``refresh_fn``, which defaults to
        ``_refresh_credentials`` — so gating anywhere shallower than
        ``refresh_access_token_now`` would leave this reachable.
        """
        credential_store.set_refresh_token("rt")
        pin_project("alpha")

        client = SLPPlatformClient("http://127.0.0.1:1")
        client.refresh_access_token = MagicMock()

        assert client._refresh_credentials() is False
        client.refresh_access_token.assert_not_called()
