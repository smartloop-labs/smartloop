"""Which account a connection is linked as.

"Connected" told a user nothing they couldn't already see. The question they
actually have, looking at two Notion cards, is whose Notion each one is — so a
connection carries an account label, probed once when it is connected.

Labelling a card with the *wrong* account is worse than leaving it blank, so
most of what is pinned here is the probe declining to guess.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from smartloop.agents.mcp.mcp_tool_caller import MCPToolCaller
from smartloop.mcp.account import (
    extract_account,
    probe_account,
    resolve_account_tool,
)


def _tool(name):
    return SimpleNamespace(name=name, description=name, input_schema={})


def _server(tools=(), name="notion"):
    return SimpleNamespace(id="s1", name=name, tools=list(tools))


class TestToolResolution:
    def test_the_identity_tool_is_found_under_any_spelling(self):
        assert resolve_account_tool(["notion-get-self"]) == "notion-get-self"
        assert resolve_account_tool(["API-get-self"]) == "API-get-self"
        assert resolve_account_tool(["get_viewer", "list_issues"]) == "get_viewer"

    def test_a_member_listing_is_not_an_identity_tool(self):
        # A loose matcher would pair `get_me` with `get_members`; naming a
        # connection after whoever tops a member list is worse than not naming
        # it at all, so this match is exact.
        assert resolve_account_tool(["get_members", "get_boards"]) is None

    def test_a_server_with_no_identity_tool_resolves_to_nothing(self):
        assert resolve_account_tool(["search", "fetch"]) is None


class TestExtraction:
    def test_notion_oauth_yields_the_person_behind_the_bot(self):
        # Notion authenticates as a bot whose own `name` is the integration;
        # the human is several levels down.
        payload = json.dumps({
            "object": "user", "id": "u1", "name": "Smartloop",
            "bot": {
                "owner": {"type": "user", "user": {
                    "name": "Mehfuz", "person": {"email": "mehfuz@smartloop.ai"},
                }},
                "workspace_name": "Smartloop HQ",
            },
        })
        assert extract_account(payload) == "mehfuz@smartloop.ai"

    def test_an_integration_token_falls_back_to_the_workspace(self):
        # No person attached — "Smartloop" is the integration's name, which
        # would label every such connection identically.
        payload = json.dumps({
            "object": "user", "name": "Smartloop",
            "bot": {"workspace_name": "Smartloop HQ"},
        })
        assert extract_account(payload) == "Smartloop HQ"

    def test_email_wins_over_a_display_name_at_the_top_level(self):
        payload = json.dumps({
            "id": "u1", "name": "Mehfuz", "displayName": "mehfuz",
            "email": "mehfuz@smartloop.ai",
        })
        assert extract_account(payload) == "mehfuz@smartloop.ai"

    def test_a_name_is_used_when_there_is_no_email(self):
        assert extract_account(json.dumps({"id": "u1", "name": "Mehfuz"})) == "Mehfuz"

    def test_a_directory_of_users_names_nobody(self):
        # `get_users` returning everyone is not an answer to "who am I".
        payload = json.dumps({"results": [
            {"name": "Ada", "email": "ada@example.com"},
            {"name": "Grace", "email": "grace@example.com"},
        ]})
        assert extract_account(payload) is None

    def test_a_single_wrapped_record_still_resolves(self):
        payload = json.dumps({"data": {"viewer": {"email": "mehfuz@smartloop.ai"}}})
        assert extract_account(payload) == "mehfuz@smartloop.ai"

    def test_a_paragraph_is_not_a_label(self):
        long_text = "You are authenticated as " + "x" * 200
        assert extract_account(long_text) is None

    def test_empty_and_unparseable_payloads_yield_nothing(self):
        assert extract_account(None) is None
        assert extract_account("") is None
        assert extract_account(json.dumps({"unrelated": {"count": 3}})) is None


class TestProbe:
    def _caller(self, payload, success=True):
        # A real caller with the transport stubbed: the refresh-and-retry that
        # every MCP path shares lives on the caller, so a MagicMock standing in
        # for the whole class would not exercise the dispatch under test.
        caller = MCPToolCaller(None, None)
        caller.get_access_token = MagicMock(return_value="tok")
        caller.call_tool = MagicMock(return_value=SimpleNamespace(
            success=success, is_error=False, error=None, auth_expired=False,
            content=[{"type": "text", "text": payload}],
        ))
        return caller

    def test_the_identity_tool_is_called_with_no_arguments(self):
        caller = self._caller(json.dumps({"email": "mehfuz@smartloop.ai"}))
        server = _server(tools=[_tool("notion-get-self"), _tool("notion-search")])

        assert probe_account(server, tool_caller=caller) == (
            "mehfuz@smartloop.ai", False,
        )
        assert caller.call_tool.call_args.kwargs["tool_name"] == "notion-get-self"
        assert caller.call_tool.call_args.kwargs["arguments"] == {}

    def test_no_identity_tool_means_no_call_at_all(self):
        caller = self._caller("{}")
        probe_account(_server(tools=[_tool("search")]), tool_caller=caller)
        caller.call_tool.assert_not_called()

    def test_a_failed_call_leaves_the_connection_unlabelled(self):
        caller = self._caller("", success=False)
        assert probe_account(
            _server(tools=[_tool("get_self")]), tool_caller=caller,
        ) == (None, False)

    def test_a_raising_transport_does_not_break_provisioning(self):
        caller = MCPToolCaller(None, None)
        caller.get_access_token = MagicMock(side_effect=RuntimeError("server went away"))
        assert probe_account(
            _server(tools=[_tool("get_self")]), tool_caller=caller,
        ) == (None, False)


def _svc_holding(*servers):
    """A service whose workspace really contains *servers*.

    ``_persist_mcp_server`` re-finds the server in the live workspace before
    saving, so a bare MagicMock would not exercise the path under test.
    """
    from smartloop.api.services.model_service import ModelService

    svc = MagicMock()
    svc._model = MagicMock()
    svc._model.workspace.projects = [SimpleNamespace(mcp_servers=list(servers))]
    # The tests drive the unbound method with a mock self, so the collaborator
    # under test has to be the real one rather than another mock.
    svc._persist_mcp_server = lambda srv: ModelService._persist_mcp_server(svc, srv)
    return svc


class TestProvisioningRecordsTheAccount:
    def _svc(self, server):
        return _svc_holding(server)

    def test_a_probed_account_is_persisted(self):
        from smartloop.api.services.model_service import ModelService

        server = SimpleNamespace(id="s1", name="notion", account=None, tools=[])
        svc = self._svc(server)
        with patch("smartloop.mcp.account.probe_account",
                   return_value=("mehfuz@smartloop.ai", False)):
            ModelService._probe_mcp_account(svc, server)
        assert server.account == "mehfuz@smartloop.ai"
        svc._model.save.assert_called_once()

    def test_an_unchanged_account_is_not_re_saved(self):
        from smartloop.api.services.model_service import ModelService

        server = SimpleNamespace(id="s1", name="notion", account="mehfuz@smartloop.ai", tools=[])
        svc = self._svc(server)
        with patch("smartloop.mcp.account.probe_account",
                   return_value=("mehfuz@smartloop.ai", False)):
            ModelService._probe_mcp_account(svc, server)
        svc._model.save.assert_not_called()

    def test_a_probe_that_finds_nothing_keeps_the_previous_label(self):
        # A transient failure on re-index must not blank a card that was
        # correctly labelled a moment ago.
        from smartloop.api.services.model_service import ModelService

        server = SimpleNamespace(id="s1", name="notion", account="mehfuz@smartloop.ai", tools=[])
        svc = self._svc(server)
        with patch("smartloop.mcp.account.probe_account", return_value=(None, False)):
            ModelService._probe_mcp_account(svc, server)
        assert server.account == "mehfuz@smartloop.ai"


class TestARotatedTokenIsWrittenBack:
    """The probe is where a refresh token most often gets spent.

    It runs on every connect and re-index, refreshes on a 401 through the shared
    read-only path — and used to drop the flag saying it had.  Against a
    provider that rotates refresh tokens single-use, that left the retired one
    on disk: the connection worked until the next restart and then could never
    be renewed, with nothing reporting it.
    """

    def _svc(self, server=None):
        return _svc_holding(*( [server] if server is not None else [] ))

    def test_a_rotation_is_saved_even_when_the_label_is_unchanged(self):
        from smartloop.api.services.model_service import ModelService

        server = SimpleNamespace(id="s1", name="notion", account="mehfuz@smartloop.ai", tools=[])
        svc = self._svc(server)
        with patch("smartloop.mcp.account.probe_account",
                   return_value=("mehfuz@smartloop.ai", True)):
            ModelService._probe_mcp_account(svc, server)
        svc._model.save.assert_called_once()

    def test_a_rotation_is_saved_even_when_there_is_no_label(self):
        from smartloop.api.services.model_service import ModelService

        server = SimpleNamespace(id="s1", name="notion", account=None, tools=[])
        svc = self._svc(server)
        with patch("smartloop.mcp.account.probe_account", return_value=(None, True)):
            ModelService._probe_mcp_account(svc, server)
        svc._model.save.assert_called_once()

    def test_the_probe_caller_carries_the_persistence_hook(self):
        """Belt: the hook writes the rotation without waiting for the flag."""
        from smartloop.api.services.model_service import ModelService

        server = SimpleNamespace(id="s1", name="notion", account=None, tools=[])
        svc = self._svc(server)
        seen = {}
        with patch("smartloop.mcp.account.probe_account") as probe:
            probe.return_value = (None, False)
            ModelService._probe_mcp_account(svc, server)
            seen["caller"] = probe.call_args.kwargs["tool_caller"]

        # It persists through the re-attaching save, not through a bare one.
        seen["caller"]._persist()
        svc._model.save.assert_called_once()

    def test_the_flag_reaches_the_caller_from_the_read_only_path(self):
        """Brace: probe_account forwards what call_read_only reported."""
        with patch("smartloop.mcp.account.call_read_only",
                   return_value=("{}", True, None)):
            assert probe_account(
                _server(tools=[_tool("get_self")]), tool_caller=MagicMock(),
            ) == (None, True)


class TestTheWriteLandsOnTheLiveWorkspace:
    """A reload mid-pass must not send the rotation to an orphan.

    ``Workspace.reload()`` rebinds ``projects`` wholesale — whatever its
    docstring says about keeping references valid — and a chat request opens
    with one.  So a provisioning pass that began earlier is mutating an object
    the workspace no longer holds, and a plain ``save()`` serializes the list
    that replaced it: the rotated refresh token goes to the orphan while the
    refresh token the provider already retired stays on disk.
    """

    def _pair(self):
        """The object the pass holds, and the one the workspace holds now."""
        held = SimpleNamespace(
            id="s1", name="notion", account=None, tools=[],
            oauth_credentials=SimpleNamespace(refresh_token="old-rt"),
            status="ready", last_error=None,
        )
        live = SimpleNamespace(
            id="s1", name="notion", account=None, tools=[],
            oauth_credentials=SimpleNamespace(refresh_token="old-rt"),
            status="ready", last_error=None,
        )
        return held, live

    def test_a_rotation_is_carried_onto_the_replacement(self):
        from smartloop.api.services.model_service import ModelService

        held, live = self._pair()
        svc = _svc_holding(live)          # the reload already happened
        held.oauth_credentials = SimpleNamespace(refresh_token="rotated-rt")

        ModelService._persist_mcp_server(svc, held)

        assert live.oauth_credentials.refresh_token == "rotated-rt"
        svc._model.save.assert_called_once()

    def test_a_failed_connection_is_carried_too(self):
        from smartloop.api.services.model_service import ModelService

        held, live = self._pair()
        svc = _svc_holding(live)
        held.status = "failed"
        held.last_error = "This connection's authorization has expired."

        ModelService._persist_mcp_server(svc, held)

        assert live.status == "failed"
        assert "expired" in live.last_error

    def test_a_server_deleted_mid_pass_is_left_deleted(self):
        """Re-finding by id must not resurrect what the user removed."""
        from smartloop.api.services.model_service import ModelService

        held, _ = self._pair()
        svc = _svc_holding()              # the workspace no longer has it

        ModelService._persist_mcp_server(svc, held)

        assert svc._model.workspace.projects[0].mcp_servers == []
        svc._model.save.assert_called_once()

    def test_the_ordinary_case_writes_through_untouched(self):
        """No reload happened: the held object *is* the live one."""
        from smartloop.api.services.model_service import ModelService

        held, _ = self._pair()
        svc = _svc_holding(held)
        held.oauth_credentials = SimpleNamespace(refresh_token="rotated-rt")

        ModelService._persist_mcp_server(svc, held)

        assert svc._model.workspace.projects[0].mcp_servers[0] is held
        svc._model.save.assert_called_once()
