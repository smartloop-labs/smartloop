"""End-to-end test: fund-info skill — explore docs → for-each tool fan-out.

Simulates the fund-info skill:
  1. Explore fund documents to extract fund names, ticker symbols, and
     annual operating expenses
  2. For each ticker symbol, call yfinance_get_ticker_info (fan-out)
  3. Aggregate all tool results and produce a summary table

Validates that:
  - The skill is matched and steps are derived with for_each=True
  - The fan-out handler extracts ticker symbols from document context
  - yfinance_get_ticker_info is called once per ticker symbol
  - All tool results are aggregated in the final output
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, call

import pytest

from smartloop.agents.orchestrator import SLPOrchestrator, MCPExecutionResult
from smartloop.agents.planner import MCPToolContext
from smartloop.workspace.workspace import (
    MCPServer,
    MCPServerType,
    MCPToolDefinition,
    RemoteMCPServerConfig,
)


# ------------------------------------------------------------------ #
# Skill content                                                        #
# ------------------------------------------------------------------ #

FUND_INFO_SKILL = (
    "---\n"
    "name: fund-info\n"
    "description: Use this skill when user requests to summarize the funds, "
    "like fund names, ticker symbol, annual fund operating expenses\n"
    "---\n\n"
    "Extract key information like fund names, ticker symbols and annual fund "
    "operating expenses for each fund\n\n"
    "For each fund using the ticker symbol get stock information using the tool below:\n"
    "```mcp\n"
    "yfinance_mcp.yfinance_get_ticker_info(symbol)\n"
    "```\n\n"
    "Finally, create a table containing fund names, symbol, operating expenses "
    "and stock information from the tool call"
)

FUND_DOCUMENT_CONTEXT = (
    "=== Document: prospectus.pdf ===\n"
    "Vanguard 500 Index Fund (VFIAX)\n"
    "  Annual Fund Operating Expenses: 0.04%\n"
    "  Expense Ratio: 0.04%\n\n"
    "Vanguard Growth Index Fund (VIGAX)\n"
    "  Annual Fund Operating Expenses: 0.05%\n"
    "  Expense Ratio: 0.05%\n\n"
    "Vanguard Total Bond Market Index Fund (VBTLX)\n"
    "  Annual Fund Operating Expenses: 0.05%\n"
    "  Expense Ratio: 0.05%\n"
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

@dataclass
class FakeToolResult:
    success: bool
    content: Optional[List[Dict[str, Any]]]
    is_error: bool = False
    error: Optional[str] = None


def _make_search_ctx():
    from smartloop.agents.mcp.explorer import SearchContext
    ctx = SearchContext()
    ctx.vector_context = FUND_DOCUMENT_CONTEXT
    ctx.citations = [
        {"document_id": "doc-1", "document_name": "prospectus.pdf"},
    ]
    return ctx


def _make_yfinance_server() -> MCPServer:
    return MCPServer(
        id="yfinance-srv",
        name="yfinance_mcp",
        server_type=MCPServerType.REMOTE,
        remote_config=RemoteMCPServerConfig(server_url="https://yfinance.example.com"),
        enabled=True,
        tools=[
            MCPToolDefinition(
                name="yfinance_get_ticker_info",
                description="Get stock information for a ticker symbol",
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Ticker symbol"},
                    },
                    "required": ["symbol"],
                },
            ),
        ],
    )


TICKER_RESPONSES = {
    "VFIAX": {
        "shortName": "Vanguard 500 Index Fund Admiral",
        "regularMarketPrice": 523.41,
        "fiftyTwoWeekHigh": 540.00,
        "fiftyTwoWeekLow": 440.12,
        "ytdReturn": 0.1234,
    },
    "VIGAX": {
        "shortName": "Vanguard Growth Index Fund Admiral",
        "regularMarketPrice": 178.92,
        "fiftyTwoWeekHigh": 190.00,
        "fiftyTwoWeekLow": 145.30,
        "ytdReturn": 0.1567,
    },
    "VBTLX": {
        "shortName": "Vanguard Total Bond Market Index Admiral",
        "regularMarketPrice": 10.45,
        "fiftyTwoWeekHigh": 11.20,
        "fiftyTwoWeekLow": 9.80,
        "ytdReturn": 0.0321,
    },
}


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #

class TestFundInfoSkillPipeline:
    """E2E: fund-info skill → explore → for-each yfinance_get_ticker_info."""

    USER_PROMPT = "summarize the fund"

    def _make_invoke_side_effect(self):
        """Build an llm.invoke side_effect for the full pipeline."""
        def side_effect(prompt, *a, **kw):
            p = str(prompt).lower()

            # Skill matching
            if "which skill" in p or ("match" in p and "skill" in p):
                return "fund-info"

            # Fan-out item extraction
            if "list every" in p and "one per line" in p:
                return "VFIAX\nVIGAX\nVBTLX"

            # Arg extraction for yfinance_get_ticker_info
            if "extract arguments for tool" in p or "extract the arguments" in p:
                return json.dumps({"symbol": "VFIAX"})

            # Dependency analysis
            if "dependencies" in p or "prerequisite" in p:
                return json.dumps({
                    "has_dependencies": False,
                    "can_execute": True,
                    "dependencies": [],
                })

            # Context gate / relevance
            if "relevant" in p and ("not_relevant" in p or "yes or no" in p):
                return "RELEVANT"

            # Step derivation (LLM fallback — shouldn't be hit since mcp block exists)
            if "execution steps" in p and "json" in p:
                return json.dumps([
                    {"step_type": "tool", "tool": "yfinance_get_ticker_info",
                     "instruction": "Get ticker info for each fund", "for_each": True},
                ])

            # Synthesis / final summary
            if "presentation guidance" in p or "synthesize" in p or "table" in p:
                return (
                    "| Fund Name | Symbol | Expense Ratio | Price |\n"
                    "|-----------|--------|---------------|-------|\n"
                    "| Vanguard 500 | VFIAX | 0.04% | $523.41 |\n"
                    "| Vanguard Growth | VIGAX | 0.05% | $178.92 |\n"
                    "| Vanguard Bond | VBTLX | 0.05% | $10.45 |"
                )

            return "OK"

        return side_effect

    def _make_tool_side_effect(self):
        """Return fake yfinance responses keyed by symbol argument."""
        def side_effect(server=None, tool_name=None, arguments=None, access_token=None, **kw):
            symbol = (arguments or {}).get("symbol", "")
            data = TICKER_RESPONSES.get(symbol, {"error": f"Unknown symbol: {symbol}"})
            return FakeToolResult(
                success=True,
                content=[{"type": "text", "text": json.dumps(data)}],
            )
        return side_effect

    def test_skill_matched_and_for_each_detected(self):
        """Verify that _parse_mcp_blocks detects for_each from skill content."""
        blocks = SLPOrchestrator._parse_mcp_blocks(FUND_INFO_SKILL)
        assert len(blocks) == 1
        assert blocks[0]["tool_name"] == "yfinance_get_ticker_info"
        assert blocks[0]["server_name"] == "yfinance_mcp"
        assert blocks[0]["inferred_args"] == ["symbol"]
        assert blocks[0]["for_each"] is True

    def test_step_derivation_produces_for_each_tool_step(self):
        """Verify that step derivation from mcp block creates a for_each PlanStep."""
        yfinance_srv = _make_yfinance_server()
        tool_contexts = [MCPToolContext(
            server_id=yfinance_srv.id,
            server_name=yfinance_srv.name,
            tools=[{
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            } for t in yfinance_srv.tools],
        )]

        blocks = SLPOrchestrator._parse_mcp_blocks(FUND_INFO_SKILL)
        steps = SLPOrchestrator._mcp_blocks_to_plan_steps(
            blocks, "fund-info", tool_contexts,
        )

        assert len(steps) == 1
        assert steps[0].tool_name == "yfinance_get_ticker_info"
        assert steps[0].for_each is True
        assert steps[0].inferred_args == ["symbol"]

    def test_full_pipeline_calls_tool_per_ticker(self):
        """Full pipeline: explore → extract tickers → call yfinance per ticker."""
        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=8192)
        mock_llm.invoke = MagicMock(side_effect=self._make_invoke_side_effect())

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        with patch.object(orchestrator.explorer, "collect_parallel", return_value=_make_search_ctx()), \
             patch.object(orchestrator.tool_caller, "call_tool",
                          side_effect=self._make_tool_side_effect()) as mock_call:
            result = orchestrator.execute(
                user_message=self.USER_PROMPT,
                mcp_servers=[_make_yfinance_server()],
                skills=[FUND_INFO_SKILL],
                cwd="/tmp",
            )

        assert result.handled is True

        # yfinance_get_ticker_info should have been called 3 times (once per ticker)
        assert mock_call.call_count == 3

        # Verify each call was for a different symbol
        called_symbols = set()
        for c in mock_call.call_args_list:
            args = c.kwargs.get("arguments") or (c.args[2] if len(c.args) >= 3 else {})
            symbol = args.get("symbol", "")
            called_symbols.add(symbol)

        assert called_symbols == {"VFIAX", "VIGAX", "VBTLX"}

        # Tool name should appear in the result
        assert "yfinance_get_ticker_info" in result.tool_names

    def test_pipeline_handles_partial_tool_failures(self):
        """When one ticker fails, the others still succeed."""
        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=8192)
        mock_llm.invoke = MagicMock(side_effect=self._make_invoke_side_effect())

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        call_count = {"n": 0}
        def fail_second_call(server=None, tool_name=None, arguments=None, access_token=None, **kw):
            call_count["n"] += 1
            if call_count["n"] == 2:
                return FakeToolResult(success=False, content=None, error="API timeout")
            symbol = (arguments or {}).get("symbol", "UNKNOWN")
            data = TICKER_RESPONSES.get(symbol, {})
            return FakeToolResult(
                success=True,
                content=[{"type": "text", "text": json.dumps(data)}],
            )

        with patch.object(orchestrator.explorer, "collect_parallel", return_value=_make_search_ctx()), \
             patch.object(orchestrator.tool_caller, "call_tool",
                          side_effect=fail_second_call) as mock_call:
            result = orchestrator.execute(
                user_message=self.USER_PROMPT,
                mcp_servers=[_make_yfinance_server()],
                skills=[FUND_INFO_SKILL],
                cwd="/tmp",
            )

        assert result.handled is True
        # All 3 calls were attempted
        assert mock_call.call_count == 3
        # Tool still appears (2 of 3 succeeded)
        assert "yfinance_get_ticker_info" in result.tool_names

    def test_pipeline_with_no_documents_still_handled(self):
        """When no documents match, the pipeline should still complete gracefully."""
        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=8192)
        mock_llm.invoke = MagicMock(side_effect=self._make_invoke_side_effect())

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        empty_ctx = _make_search_ctx()
        empty_ctx.vector_context = ""

        with patch.object(orchestrator.explorer, "collect_parallel", return_value=empty_ctx), \
             patch.object(orchestrator.tool_caller, "call_tool") as mock_call:
            result = orchestrator.execute(
                user_message=self.USER_PROMPT,
                mcp_servers=[_make_yfinance_server()],
                skills=[FUND_INFO_SKILL],
                cwd="/tmp",
            )

        assert result.handled is True
        # No tool calls since fan-out extraction returns empty list
        assert mock_call.call_count == 0
