"""A tool the account cannot call must not answer for the document.

A paywalled MCP tool does not fail.  Notion's ``notion-ai-search`` on a
non-Business plan returns HTTP 200, ``success=True``, and a body reading "This
tool requires a Business plan or higher" with an upsell link.  Every layer
above treated that as retrieved content, so the model was handed an
advertisement in place of the founders agreement and wrote its answer from it —
"the provided data does not contain a Founders Agreement", followed by an
invented template.  Discovery cannot warn about this: it lists what a server
*has*, not what the plan is entitled to.
"""

from unittest.mock import MagicMock

import pytest

from smartloop.agents.mcp import mcp_agent
from smartloop.agents.mcp.explorer import Explorer
from smartloop.agents.mcp.mcp_agent import (
    MCPAgent, forget_unavailable_tools, is_entitlement_notice,
    mark_tool_unavailable, unavailable_tools,
)

# The real payload, from the agent log.
NOTION_UPSELL = (
    "This tool requires a Business plan or higher. Learn how to access the "
    "full Notion MCP, including a free trial: "
    "https://app.notion.com/notion-mcp?source=mcp_tool_upsell&tool=ai_search"
    "&product=business&mcpRequestId=5c38eb6a-b115-4a7e-bd3b-2de99746f35c"
)


@pytest.fixture(autouse=True)
def _clean_registry():
    forget_unavailable_tools()
    yield
    forget_unavailable_tools()


def _explorer():
    exp = Explorer.__new__(Explorer)
    exp.mcp_agent = MagicMock()
    exp.tool_caller = MagicMock()
    exp.arg_extractor = MagicMock()
    exp._fetch_record_details = MagicMock(return_value="")
    return exp


def _server(name="Notion", server_id="notion-1"):
    s = MagicMock()
    s.name = name
    s.id = server_id
    return s


def _pick(tool_name, **arguments):
    p = MagicMock()
    p.tool_name = tool_name
    p.arguments = arguments
    return p


class TestRecognisingTheNotice:
    def test_the_real_notion_upsell_is_recognised(self):
        assert is_entitlement_notice(NOTION_UPSELL)

    @pytest.mark.parametrize("text", [
        "This tool requires a Business plan or higher.",
        "That requires a subscription.",
        "AI search is not available on your current plan.",
        "Please upgrade your plan to continue.",
    ])
    def test_other_phrasings_are_recognised(self, text):
        assert is_entitlement_notice(text)

    def test_a_long_document_mentioning_plans_is_not_a_notice(self):
        # The failure mode to avoid: discarding a real page about pricing,
        # which is exactly the content someone would ask for.
        page = (
            "Pricing page. The Business plan requires a seat minimum of five. "
        ) * 60
        assert not is_entitlement_notice(page)

    def test_ordinary_content_is_not_a_notice(self):
        assert not is_entitlement_notice(
            "The founders agreement was signed on 1 March by both parties."
        )

    def test_nothing_is_not_a_notice(self):
        assert not is_entitlement_notice("")
        assert not is_entitlement_notice(None)


class TestTheSameTurnFallsToAToolThatWorks:
    def test_a_paywalled_pick_is_retried_and_the_second_tool_answers(self):
        exp = _explorer()
        exp.mcp_agent.resolve.side_effect = [
            _pick("notion-ai-search", query="founders agreement"),
            _pick("notion-search", query="founders agreement"),
        ]
        exp._call_mcp_tool = MagicMock(side_effect=[NOTION_UPSELL, "the real page text"])

        out = exp._run_mcp_server("outline the founders agreement", _server())

        assert "the real page text" in out
        assert "Business plan" not in out
        assert exp.mcp_agent.resolve.call_count == 2

    def test_the_upsell_never_becomes_context(self):
        # Even with no working alternative, an advertisement must not reach the
        # model as though it were the document.
        exp = _explorer()
        exp.mcp_agent.resolve.return_value = _pick("notion-ai-search", query="x")
        exp._call_mcp_tool = MagicMock(return_value=NOTION_UPSELL)

        out = exp._run_mcp_server("outline the founders agreement", _server())

        assert out == ""

    def test_the_retry_is_bounded(self):
        exp = _explorer()
        exp.mcp_agent.resolve.return_value = _pick("notion-ai-search", query="x")
        exp._call_mcp_tool = MagicMock(return_value=NOTION_UPSELL)

        exp._run_mcp_server("q", _server())

        assert exp.mcp_agent.resolve.call_count == 2

    def test_a_working_tool_is_called_once(self):
        exp = _explorer()
        exp.mcp_agent.resolve.return_value = _pick("notion-search", query="x")
        exp._call_mcp_tool = MagicMock(return_value="real records")

        out = exp._run_mcp_server("q", _server())

        assert "real records" in out
        assert exp.mcp_agent.resolve.call_count == 1

    def test_the_user_is_told_the_tool_needs_a_higher_plan(self):
        exp = _explorer()
        exp.mcp_agent.resolve.side_effect = [
            _pick("notion-ai-search", query="x"), _pick("notion-search", query="x"),
        ]
        exp._call_mcp_tool = MagicMock(side_effect=[NOTION_UPSELL, "real text"])

        seen = []
        exp._run_mcp_server("q", _server(), progress=lambda *a: seen.append(a))

        assert any("higher plan" in m for _, _, m in seen), [m for _, _, m in seen]


class TestTheToolIsNotOfferedAgain:
    def test_a_paywalled_tool_is_recorded(self):
        exp = _explorer()
        exp.mcp_agent.resolve.side_effect = [
            _pick("notion-ai-search", query="x"), _pick("notion-search", query="x"),
        ]
        exp._call_mcp_tool = MagicMock(side_effect=[NOTION_UPSELL, "real text"])

        exp._run_mcp_server("q", _server(server_id="notion-1"))

        assert ("notion-1", "notion-ai-search") in unavailable_tools()

    def test_the_catalogue_stops_offering_it(self):
        # Without this the resolver keeps picking the same paywalled tool every
        # turn — it is still in discovery and still the best-sounding name.
        agent = MCPAgent.__new__(MCPAgent)
        agent._logger = MagicMock()
        server = MagicMock()
        server.id = "notion-1"
        server.name = "Notion"
        server.system = False
        server.enabled = True
        server.tools = [
            {"name": "notion-ai-search", "description": "AI search", "input_schema": {}},
            {"name": "notion-search", "description": "Search pages", "input_schema": {}},
        ]

        before = {e.tool_name for e in agent._catalog([server])}
        assert before == {"notion-ai-search", "notion-search"}

        mark_tool_unavailable("notion-1", "notion-ai-search")

        after = {e.tool_name for e in agent._catalog([server])}
        assert after == {"notion-search"}

    def test_it_is_scoped_to_the_server_that_declined(self):
        mark_tool_unavailable("notion-1", "notion-ai-search")
        assert ("notion-2", "notion-ai-search") not in unavailable_tools()

    def test_the_registry_can_be_cleared_for_reauthentication(self):
        mark_tool_unavailable("notion-1", "notion-ai-search")
        assert unavailable_tools()
        mcp_agent.forget_unavailable_tools()
        assert not unavailable_tools()
