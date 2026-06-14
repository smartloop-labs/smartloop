"""
Tests for tool fan-out (iterative tool calling) in SLPOrchestrator.

When a skill says "for each X, call tool Y", the pipeline extracts a
list of items from upstream context and calls the tool once per item,
aggregating all results into a single NodeResult.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from smartloop.agents.orchestrator import SLPOrchestrator
from smartloop.agents.mcp.explorer import SearchContext
from smartloop.agents.nodes import (
    NodeResult, LLMDecisionNode, ToolCallNode, TransformNode,
)
from smartloop.agents.planner import PlanStep, StepType


def _make_orchestrator():
    llm = MagicMock()
    llm.get_num_tokens = MagicMock(return_value=100)
    llm.get_max_tokens = MagicMock(return_value=8192)
    llm.invoke = MagicMock(return_value="OK")
    return SLPOrchestrator(orchestrator_llm=llm)


def _ctx(orch, user_message="summarize fund info"):
    return orch._build_context(
        user_message=user_message,
        mcp_servers=[],
        skills=None,
        cwd=None,
        document_context=None,
        has_attachments=False,
        project_dir=None,
        model_service=None,
        conversation_history=None,
    )


# ------------------------------------------------------------------ #
# _parse_mcp_blocks: for_each detection                               #
# ------------------------------------------------------------------ #

class TestParseMcpBlocksForEach:
    def test_for_each_detected_before_block(self):
        content = (
            "For each fund using the ticker symbol get stock info:\n"
            "```mcp\nyfinance.get_ticker_info(symbol)\n```"
        )
        blocks = SLPOrchestrator._parse_mcp_blocks(content)
        assert len(blocks) == 1
        assert blocks[0]["for_each"] is True

    def test_for_every_detected(self):
        content = (
            "For every item in the list, look it up:\n"
            "```mcp\napi.lookup_item(item_id)\n```"
        )
        blocks = SLPOrchestrator._parse_mcp_blocks(content)
        assert len(blocks) == 1
        assert blocks[0]["for_each"] is True

    def test_for_all_detected(self):
        content = (
            "For all entries, validate:\n"
            "```mcp\nvalidator.check(entry)\n```"
        )
        blocks = SLPOrchestrator._parse_mcp_blocks(content)
        assert len(blocks) == 1
        assert blocks[0]["for_each"] is True

    def test_no_for_each_when_absent(self):
        content = (
            "Get stock info:\n"
            "```mcp\nyfinance.get_ticker_info(symbol)\n```"
        )
        blocks = SLPOrchestrator._parse_mcp_blocks(content)
        assert len(blocks) == 1
        assert blocks[0]["for_each"] is False

    def test_for_each_outside_lookback_window(self):
        content = (
            "For each fund do something.\n"
            + "x" * 400
            + "\nNow call:\n"
            "```mcp\nyfinance.get_ticker_info(symbol)\n```"
        )
        blocks = SLPOrchestrator._parse_mcp_blocks(content)
        assert len(blocks) == 1
        assert blocks[0]["for_each"] is False

    def test_multiple_blocks_independent_detection(self):
        content = (
            "Get the summary:\n"
            "```mcp\napi.get_summary(id)\n```\n\n"
            "For each item, get details:\n"
            "```mcp\napi.get_details(item_id)\n```"
        )
        blocks = SLPOrchestrator._parse_mcp_blocks(content)
        assert len(blocks) == 2
        assert blocks[0]["for_each"] is False
        assert blocks[1]["for_each"] is True


# ------------------------------------------------------------------ #
# _mcp_blocks_to_plan_steps: for_each propagation                     #
# ------------------------------------------------------------------ #

class TestMcpBlocksToPlanStepsForEach:
    def _tool_contexts(self):
        from smartloop.agents.planner import MCPToolContext
        return [MCPToolContext(
            server_id="yfinance-srv",
            server_name="yfinance",
            tools=[{
                "name": "get_ticker_info",
                "description": "Get ticker info",
                "input_schema": {
                    "type": "object",
                    "properties": {"symbol": {"type": "string"}},
                    "required": ["symbol"],
                },
            }],
        )]

    def test_for_each_propagated_to_plan_step(self):
        blocks = [{
            "server_name": "yfinance",
            "tool_name": "get_ticker_info",
            "inferred_args": ["symbol"],
            "default_args": {},
            "for_each": True,
        }]
        steps = SLPOrchestrator._mcp_blocks_to_plan_steps(
            blocks, "fund-info", self._tool_contexts(),
        )
        assert len(steps) == 1
        assert steps[0].for_each is True

    def test_for_each_absent_remains_none(self):
        blocks = [{
            "server_name": "yfinance",
            "tool_name": "get_ticker_info",
            "inferred_args": ["symbol"],
            "default_args": {},
            "for_each": False,
        }]
        steps = SLPOrchestrator._mcp_blocks_to_plan_steps(
            blocks, "fund-info", self._tool_contexts(),
        )
        assert len(steps) == 1
        assert steps[0].for_each is None


# ------------------------------------------------------------------ #
# _build_step_nodes: fan-out node creation                            #
# ------------------------------------------------------------------ #

class TestBuildStepNodesForEach:
    def test_for_each_step_creates_single_llm_node(self):
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        search_ctx = SearchContext()
        parent = LLMDecisionNode()

        step = PlanStep(
            step=1, step_type=StepType.TOOL,
            tool_name="get_ticker_info",
            server_id="yfinance-srv", server_name="yfinance",
            instruction="Get ticker info",
            for_each=True, inferred_args=["symbol"],
        )
        nodes = orch._build_step_nodes(ctx, [step], search_ctx, parent=parent)
        node_types = [type(n).__name__ for n, _ in nodes]
        assert node_types == ["LLMDecisionNode", "TransformNode"]
        assert "ToolCallNode" not in node_types

    def test_normal_step_creates_resolve_execute_pair(self):
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        search_ctx = SearchContext()
        parent = LLMDecisionNode()

        step = PlanStep(
            step=1, step_type=StepType.TOOL,
            tool_name="get_ticker_info",
            server_id="yfinance-srv", server_name="yfinance",
            instruction="Get ticker info",
        )
        nodes = orch._build_step_nodes(ctx, [step], search_ctx, parent=parent)
        node_types = [type(n).__name__ for n, _ in nodes]
        assert "LLMDecisionNode" in node_types
        assert "ToolCallNode" in node_types


# ------------------------------------------------------------------ #
# _extract_fanout_items                                                #
# ------------------------------------------------------------------ #

class TestParseFanoutResponse:
    def test_one_per_line(self):
        items = SLPOrchestrator._parse_fanout_response("AAPL\nMSFT\nGOOGL")
        assert items == ["AAPL", "MSFT", "GOOGL"]

    def test_json_array(self):
        items = SLPOrchestrator._parse_fanout_response('["AAPL", "MSFT"]')
        assert items == ["AAPL", "MSFT"]

    def test_strips_bullets_and_numbers(self):
        items = SLPOrchestrator._parse_fanout_response("1. AAPL\n2. MSFT\n- GOOGL\n• VYM")
        assert items == ["AAPL", "MSFT", "GOOGL", "VYM"]

    def test_skips_preamble_lines(self):
        items = SLPOrchestrator._parse_fanout_response(
            "Here are the symbols:\nAAPL\nMSFT"
        )
        assert items == ["AAPL", "MSFT"]

    def test_empty_response(self):
        assert SLPOrchestrator._parse_fanout_response("") == []

    def test_single_item_json_falls_through_to_lines(self):
        items = SLPOrchestrator._parse_fanout_response('["AAPL"]')
        assert items == ["AAPL"]


class TestExtractFanoutItems:
    def test_extracts_from_one_per_line_response(self):
        orch = _make_orchestrator()
        orch.agent_llm = MagicMock()
        orch.agent_llm.invoke = MagicMock(return_value="AAPL\nMSFT\nGOOGL")
        ctx = _ctx(orch)

        step = PlanStep(
            step=1, step_type=StepType.TOOL,
            tool_name="get_ticker_info",
            instruction="Get ticker info",
            inferred_args=["symbol"],
        )
        search_ctx = SearchContext()
        search_ctx.vector_context = "Fund A ticker AAPL, Fund B ticker MSFT, Fund C ticker GOOGL"

        items = orch._extract_fanout_items(ctx, step, [], search_ctx)
        assert items == ["AAPL", "MSFT", "GOOGL"]

    def test_extracts_from_json_array(self):
        orch = _make_orchestrator()
        orch.agent_llm = MagicMock()
        orch.agent_llm.invoke = MagicMock(return_value='["AAPL", "MSFT"]')
        ctx = _ctx(orch)

        step = PlanStep(
            step=1, step_type=StepType.TOOL,
            tool_name="get_ticker_info",
            instruction="Get ticker info",
            inferred_args=["symbol"],
        )
        search_ctx = SearchContext()
        search_ctx.vector_context = "AAPL and MSFT funds"

        items = orch._extract_fanout_items(ctx, step, [], search_ctx)
        assert items == ["AAPL", "MSFT"]

    def test_empty_context_returns_empty(self):
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        step = PlanStep(
            step=1, step_type=StepType.TOOL,
            tool_name="get_ticker_info",
            instruction="Get ticker info",
        )
        search_ctx = SearchContext()
        items = orch._extract_fanout_items(ctx, step, [], search_ctx)
        assert items == []

    def test_caps_at_max_items(self):
        orch = _make_orchestrator()
        orch.agent_llm = MagicMock()
        orch.agent_llm.invoke = MagicMock(
            return_value="\n".join(f"TICK{i}" for i in range(30))
        )
        ctx = _ctx(orch)

        step = PlanStep(
            step=1, step_type=StepType.TOOL,
            tool_name="get_ticker_info",
            instruction="Get ticker info",
            inferred_args=["symbol"],
        )
        search_ctx = SearchContext()
        search_ctx.vector_context = "lots of tickers"

        items = orch._extract_fanout_items(ctx, step, [], search_ctx)
        assert len(items) == orch._FANOUT_MAX_ITEMS

    def test_llm_failure_returns_empty(self):
        orch = _make_orchestrator()
        orch.agent_llm = MagicMock()
        orch.agent_llm.invoke = MagicMock(side_effect=RuntimeError("boom"))
        ctx = _ctx(orch)

        step = PlanStep(
            step=1, step_type=StepType.TOOL,
            tool_name="get_ticker_info",
            instruction="Get ticker info",
            inferred_args=["symbol"],
        )
        search_ctx = SearchContext()
        search_ctx.vector_context = "some context"

        items = orch._extract_fanout_items(ctx, step, [], search_ctx)
        assert items == []


# ------------------------------------------------------------------ #
# _make_fanout_handler: full execution                                 #
# ------------------------------------------------------------------ #

class TestFanoutHandler:
    def test_calls_tool_per_item_and_aggregates(self):
        orch = _make_orchestrator()
        orch.agent_llm = MagicMock()
        ctx = _ctx(orch)
        search_ctx = SearchContext()
        search_ctx.vector_context = "Fund A AAPL, Fund B MSFT"

        step = PlanStep(
            step=1, step_type=StepType.TOOL,
            tool_name="get_ticker_info",
            server_id="yfinance-srv", server_name="yfinance",
            instruction="Get ticker info",
            inferred_args=["symbol"],
            for_each=True,
        )

        handler = orch._make_fanout_handler(ctx, step, search_ctx)
        mock_server = MagicMock()

        with patch.object(orch, '_extract_fanout_items', return_value=["AAPL", "MSFT"]), \
             patch.object(orch, '_find_fanout_server', return_value=mock_server), \
             patch.object(orch, '_call_tool') as mock_call:

            mock_call.return_value = ("Price: $150", '{"price": 150}')

            result = handler({})

            assert result.success
            assert mock_call.call_count == 2
            assert "AAPL" in result.data["content"]
            assert "MSFT" in result.data["content"]

    def test_empty_items_returns_empty_result(self):
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        search_ctx = SearchContext()

        step = PlanStep(
            step=1, step_type=StepType.TOOL,
            tool_name="get_ticker_info",
            instruction="Get ticker info",
            for_each=True,
        )

        handler = orch._make_fanout_handler(ctx, step, search_ctx)

        with patch.object(orch, '_extract_fanout_items', return_value=[]):
            result = handler({})
            assert result.success
            assert result.data["content"] == ""

    def test_partial_tool_failures_handled(self):
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        search_ctx = SearchContext()

        step = PlanStep(
            step=1, step_type=StepType.TOOL,
            tool_name="get_ticker_info",
            server_id="yfinance-srv", server_name="yfinance",
            instruction="Get ticker info",
            inferred_args=["symbol"],
            for_each=True,
        )

        handler = orch._make_fanout_handler(ctx, step, search_ctx)
        mock_server = MagicMock()

        call_count = {"n": 0}
        def fail_second(server, tool_name, arguments, cwd=None):
            call_count["n"] += 1
            if call_count["n"] == 2:
                return None
            return ("Price: $150", '{"price": 150}')

        with patch.object(orch, '_extract_fanout_items', return_value=["AAPL", "BAD", "MSFT"]), \
             patch.object(orch, '_find_fanout_server', return_value=mock_server), \
             patch.object(orch, '_call_tool', side_effect=fail_second) as mock_call:

            result = handler({})
            assert result.success
            assert mock_call.call_count == 3
            assert "AAPL" in result.data["content"]
            assert "MSFT" in result.data["content"]

    def test_single_inferred_arg_prefilled_in_arguments(self):
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        search_ctx = SearchContext()

        step = PlanStep(
            step=1, step_type=StepType.TOOL,
            tool_name="get_ticker_info",
            server_id="yfinance-srv", server_name="yfinance",
            instruction="Get ticker info",
            inferred_args=["symbol"],
            for_each=True,
        )

        handler = orch._make_fanout_handler(ctx, step, search_ctx)
        mock_server = MagicMock()

        with patch.object(orch, '_extract_fanout_items', return_value=["AAPL"]), \
             patch.object(orch, '_find_fanout_server', return_value=mock_server), \
             patch.object(orch, '_call_tool') as mock_call:

            mock_call.return_value = ("Price: $150", '{"price": 150}')
            result = handler({})

            call_args = mock_call.call_args
            arguments = call_args[0][2] if len(call_args[0]) >= 3 else call_args[1].get("arguments", {})
            assert arguments == {"symbol": "AAPL"}

    def test_dict_items_merged_into_arguments(self):
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        search_ctx = SearchContext()

        step = PlanStep(
            step=1, step_type=StepType.TOOL,
            tool_name="get_ticker_info",
            server_id="yfinance-srv", server_name="yfinance",
            instruction="Get ticker info",
            inferred_args=["symbol", "exchange"],
            for_each=True,
        )

        handler = orch._make_fanout_handler(ctx, step, search_ctx)
        mock_server = MagicMock()

        with patch.object(orch, '_extract_fanout_items',
                          return_value=[{"symbol": "AAPL", "exchange": "NASDAQ"}]), \
             patch.object(orch, '_find_fanout_server', return_value=mock_server), \
             patch.object(orch, '_call_tool') as mock_call:

            mock_call.return_value = ("Info", "raw")
            result = handler({})

            call_args = mock_call.call_args
            arguments = call_args[0][2] if len(call_args[0]) >= 3 else call_args[1].get("arguments", {})
            assert arguments["symbol"] == "AAPL"
            assert arguments["exchange"] == "NASDAQ"

    def test_server_not_found_returns_empty(self):
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        search_ctx = SearchContext()
        search_ctx.vector_context = "some context"

        step = PlanStep(
            step=1, step_type=StepType.TOOL,
            tool_name="get_ticker_info",
            instruction="Get ticker info",
            inferred_args=["symbol"],
            for_each=True,
        )

        handler = orch._make_fanout_handler(ctx, step, search_ctx)

        with patch.object(orch, '_extract_fanout_items', return_value=["AAPL"]), \
             patch.object(orch, '_find_fanout_server', return_value=None):

            result = handler({})
            assert result.success
            assert result.data["content"] == ""
