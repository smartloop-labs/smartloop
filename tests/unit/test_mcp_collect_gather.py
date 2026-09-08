"""Every connected server is asked, and a listing is followed to its bodies.

Collect puts a question to documents, workspace files, feedback and the web at
once; a connected server is a source like any of them.  It used to resolve one
tool across the whole workspace, so a question spanning two servers came back
with half its context — and the one call it did make returned a list of titles,
which is how "outline the pitch doc" got answered from a search hit's headline
instead of the page.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from smartloop.agents.mcp.explorer import Explorer
from smartloop.agents.mcp.mcp_agent import ResolvedCall, clear_resolution_cache


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_resolution_cache()
    yield
    clear_resolution_cache()


def _server(name, tool="search", server_id=None):
    return SimpleNamespace(
        id=server_id or name, name=name, enabled=True, system=False,
        catalog_slug=None, pkg=None, remote_config=None, config=None,
        tools=[SimpleNamespace(name=tool, description="", input_schema={})],
    )


def _explorer(*, results, resolved):
    """An Explorer whose tool calls return *results* keyed by (server, tool)."""
    ex = Explorer.__new__(Explorer)
    ex.arg_extractor = MagicMock()
    ex.mcp_agent = MagicMock()
    ex.mcp_agent.resolve.side_effect = lambda q, servers, llm=None: resolved(servers[0])

    def call_tool(server, tool_name, arguments, access_token=None, **kwargs):
        text = results.get((server.name, tool_name))
        return SimpleNamespace(
            success=text is not None,
            content=[{"type": "text", "text": text}] if text else [],
            is_error=False, error=None,
        )

    ex.tool_caller = MagicMock()
    ex.tool_caller.call_tool.side_effect = call_tool
    ex.tool_caller.get_access_token.return_value = None
    return ex


class TestEveryServerIsAsked:
    def test_two_servers_both_contribute(self):
        servers = [_server("Notion", "notion-search"), _server("Linear", "list_issues")]
        ex = _explorer(
            results={
                ("Notion", "notion-search"): "the pitch page",
                ("Linear", "list_issues"): "ENG-1 open",
            },
            resolved=lambda s: ResolvedCall(s.id, s.tools[0].name, {"query": "x"}),
        )

        out = ex._run_mcp("what is on my plate", servers, "/tmp")

        assert "MCP TOOL (Notion.notion-search)" in out
        assert "the pitch page" in out
        assert "MCP TOOL (Linear.list_issues)" in out
        assert "ENG-1 open" in out

    def test_one_server_failing_does_not_lose_the_others(self):
        servers = [_server("Broken", "search"), _server("Linear", "list_issues")]
        ex = _explorer(
            results={("Linear", "list_issues"): "ENG-1 open"},
            resolved=lambda s: ResolvedCall(s.id, s.tools[0].name, {}),
        )
        ex.tool_caller.get_access_token.side_effect = (
            lambda s: (_ for _ in ()).throw(RuntimeError("token gone"))
            if s.name == "Broken" else None
        )

        out = ex._run_mcp("what is on my plate", servers, "/tmp")

        assert "ENG-1 open" in out
        assert "Broken" not in out

    def test_a_write_tool_is_left_to_a_planned_step(self):
        servers = [_server("Notion", "notion-create-pages")]
        ex = _explorer(
            results={("Notion", "notion-create-pages"): "created"},
            resolved=lambda s: ResolvedCall(s.id, s.tools[0].name, {}),
        )

        assert ex._run_mcp("write it down", servers, "/tmp") == ""
        ex.tool_caller.call_tool.assert_not_called()

    def test_a_server_that_resolves_to_nothing_is_skipped(self):
        servers = [_server("Notion", "notion-search"), _server("Linear", "list_issues")]
        ex = _explorer(
            results={("Linear", "list_issues"): "ENG-1 open"},
            resolved=lambda s: (
                None if s.name == "Notion"
                else ResolvedCall(s.id, s.tools[0].name, {})
            ),
        )

        out = ex._run_mcp("my issues", servers, "/tmp")
        assert "ENG-1 open" in out and "Notion" not in out

    def test_the_number_of_servers_asked_is_bounded(self):
        servers = [_server(f"srv{i}", "search") for i in range(8)]
        ex = _explorer(
            results={(f"srv{i}", "search"): f"hit {i}" for i in range(8)},
            resolved=lambda s: ResolvedCall(s.id, "search", {}),
        )

        out = ex._run_mcp("anything", servers, "/tmp")

        assert out.count("MCP TOOL") == Explorer._MAX_SERVERS_PER_QUESTION

    def test_disabled_and_system_servers_are_not_asked(self):
        disabled = _server("Off", "search")
        disabled.enabled = False
        system = _server("smartloop", "bash")
        system.system = True
        ex = _explorer(
            results={("Off", "search"): "x", ("smartloop", "bash"): "x"},
            resolved=lambda s: ResolvedCall(s.id, s.tools[0].name, {}),
        )

        assert ex._run_mcp("anything", [disabled, system], "/tmp") == ""


class TestTheListingIsFollowedToItsBodies:
    def test_a_search_hit_becomes_the_page_behind_it(self, monkeypatch):
        servers = [_server("Notion", "notion-search")]
        ex = _explorer(
            results={
                ("Notion", "notion-search"): '{"results": [{"id": "abc"}]}',
                ("Notion", "notion-fetch"): "# Pitch\\nThe actual page text.",
            },
            resolved=lambda s: ResolvedCall(s.id, "notion-search", {"query": "pitch"}),
        )
        monkeypatch.setattr(
            "smartloop.mcp.call_plans.follow_up_calls",
            lambda server, tool, payload: [("notion-fetch", {"id": "abc"})],
        )

        out = ex._run_mcp("outline the pitch", servers, "/tmp")

        assert "The actual page text." in out

    def test_no_plan_means_the_listing_is_the_answer(self, monkeypatch):
        servers = [_server("Weather", "get_forecast")]
        ex = _explorer(
            results={("Weather", "get_forecast"): "18C and clear"},
            resolved=lambda s: ResolvedCall(s.id, "get_forecast", {}),
        )
        monkeypatch.setattr(
            "smartloop.mcp.call_plans.follow_up_calls",
            lambda server, tool, payload: [],
        )

        out = ex._run_mcp("weather?", servers, "/tmp")

        assert "18C and clear" in out
        assert ex.tool_caller.call_tool.call_count == 1

    def test_a_failed_follow_up_keeps_the_listing(self, monkeypatch):
        servers = [_server("Notion", "notion-search")]
        ex = _explorer(
            results={("Notion", "notion-search"): '{"results": [{"id": "abc"}]}'},
            resolved=lambda s: ResolvedCall(s.id, "notion-search", {"query": "pitch"}),
        )
        monkeypatch.setattr(
            "smartloop.mcp.call_plans.follow_up_calls",
            lambda server, tool, payload: [("notion-fetch", {"id": "abc"})],
        )

        out = ex._run_mcp("outline the pitch", servers, "/tmp")

        assert '"id": "abc"' in out
