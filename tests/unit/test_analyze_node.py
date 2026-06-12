"""Unit tests for the ANALYZE node in the skill pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from smartloop.agents.nodes import NodeResult
from smartloop.agents.planner import (
    PlanStep,
    StepType,
    SYSTEM_TOOLS_SERVER_ID,
    SYSTEM_TOOLS_SERVER_NAME,
)
from smartloop.agents.mcp.explorer import SearchContext
from smartloop.agents.mcp.server_selector import EXPLORE_TOOL


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke = MagicMock(return_value="")
    llm.get_max_tokens = MagicMock(return_value=4096)
    llm.get_num_tokens = MagicMock(return_value=100)
    return llm


@pytest.fixture
def orchestrator(mock_llm):
    with patch("smartloop.agents.orchestrator.LLMExecutor"), \
         patch("smartloop.agents.orchestrator.DAGExecutor"), \
         patch("smartloop.agents.orchestrator.AppSettings", create=True), \
         patch("smartloop.config.app_settings.AppSettings"):
        from smartloop.agents.orchestrator import SLPOrchestrator
        orch = SLPOrchestrator(orchestrator_llm=mock_llm)
    return orch


@pytest.fixture
def analyze_step():
    return PlanStep(
        step=2,
        step_type=StepType.ANALYZE,
        tool_name=None,
        server_id="",
        server_name="",
        instruction="Assess relevance of search results to user request.",
    )


@pytest.fixture
def ctx():
    """Minimal _OrchestratorContext-like object."""
    ctx = MagicMock()
    ctx.user_message = "proof read my document"
    return ctx


@pytest.fixture
def search_ctx():
    return SearchContext(
        vector_context="",
        workspace_context="",
        feedback_context="",
        skill_tool_results=[],
        citations=[],
        synthesized=False,
    )


def _make_deps(content: str) -> dict:
    """Build a deps dict simulating upstream explore output."""
    return {
        "skill_1_explore": NodeResult(
            node_id="skill_1_explore",
            success=True,
            data={
                "tool_name": EXPLORE_TOOL,
                "content": content,
                "raw": content,
            },
        )
    }


class TestAnalyzeHandler:
    """Tests for the ANALYZE branch in _make_skill_step_handler."""

    def test_analyze_invokes_llm_with_explore_context(
        self, orchestrator, mock_llm, analyze_step, ctx, search_ctx
    ):
        """ANALYZE handler should call the LLM with the explore output."""
        mock_llm.invoke.return_value = "The document contains a summary of Q3 results."

        handler = orchestrator._make_step_handler(
            ctx, analyze_step, search_ctx=search_ctx,
        )
        deps = _make_deps("Q3 financial results: revenue up 15%.")
        result = handler(deps)

        mock_llm.invoke.assert_called_once()
        prompt_arg = mock_llm.invoke.call_args[0][0]
        assert "Q3 financial results" in prompt_arg
        assert "proof read" in prompt_arg
        assert result.success is True
        assert result.data["content"] != ""

    def test_analyze_irrelevant_context_passes_through_raw(
        self, orchestrator, mock_llm, analyze_step, ctx, search_ctx
    ):
        """When the LLM deems context irrelevant, handler passes through raw explore data."""
        mock_llm.invoke.return_value = "NOT_RELEVANT"

        handler = orchestrator._make_step_handler(
            ctx, analyze_step, search_ctx=search_ctx,
        )
        deps = _make_deps("Unrelated cooking recipe data.")
        result = handler(deps)

        assert result.success is True
        # Raw explore content is passed through so downstream steps can use it
        assert result.data["content"] == "Unrelated cooking recipe data."
        assert not hasattr(orchestrator, '_skill_report') or orchestrator._skill_report is None or orchestrator._skill_report == ""

    def test_analyze_relevant_context_stores_skill_report(
        self, orchestrator, mock_llm, analyze_step, ctx, search_ctx
    ):
        """Relevant analysis should be stored in _skill_report."""
        analysis = "The document has several grammatical errors in paragraphs 2 and 5."
        mock_llm.invoke.return_value = analysis

        handler = orchestrator._make_step_handler(
            ctx, analyze_step, search_ctx=search_ctx,
        )
        deps = _make_deps("Draft document with typos and grammar issues.")
        result = handler(deps)

        assert orchestrator._skill_report == analysis
        assert result.data["content"] == analysis
        assert result.data["raw"] == analysis

    def test_analyze_empty_deps_returns_empty(
        self, orchestrator, mock_llm, analyze_step, ctx, search_ctx
    ):
        """When explore produced no content, ANALYZE should skip the LLM call."""
        handler = orchestrator._make_step_handler(
            ctx, analyze_step, search_ctx=search_ctx,
        )
        deps = _make_deps("")
        result = handler(deps)

        mock_llm.invoke.assert_not_called()
        assert result.data["content"] == ""

    def test_analyze_llm_failure_returns_empty(
        self, orchestrator, mock_llm, analyze_step, ctx, search_ctx
    ):
        """If the LLM raises an exception, ANALYZE passes through raw explore data."""
        mock_llm.invoke.side_effect = RuntimeError("model error")

        handler = orchestrator._make_step_handler(
            ctx, analyze_step, search_ctx=search_ctx,
        )
        deps = _make_deps("Some explore content.")
        result = handler(deps)

        assert result.success is True
        # Raw explore content is passed through on failure
        assert result.data["content"] == "Some explore content."


class TestFallbackSkillStepsIncludesAnalyze:
    """Verify that _fallback_skill_steps includes ANALYZE after EXPLORE."""

    def test_fallback_steps_contain_analyze(self, orchestrator):
        steps = orchestrator._fallback_skill_steps(
            matched_skills=[],
            mcp_tool_contexts=[],
        )
        step_types = [s.step_type for s in steps]
        assert StepType.EXPLORE in step_types
        assert StepType.ANALYZE in step_types
        explore_idx = step_types.index(StepType.EXPLORE)
        analyze_idx = step_types.index(StepType.ANALYZE)
        assert analyze_idx == explore_idx + 1
