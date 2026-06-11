"""Unit tests for the context_gate relevance filter in the classify pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from smartloop.agents.nodes import NodeResult
from smartloop.agents.mcp.explorer import SearchContext


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
def ctx():
    from smartloop.agents.orchestrator import _OrchestratorContext
    return _OrchestratorContext(
        user_message="proof read the following text: hello world",
        mcp_servers=[],
        available_servers=[],
        effective_cwd="/tmp",
        document_context=None,
        project_dir=None,
        model_service=MagicMock(),
        conversation_history=[],
        has_attachments=False,
    )


class TestContextGateIrrelevantCleared:
    """When LLM says NOT_RELEVANT, feedback/workspace context should be cleared."""

    def test_clears_irrelevant_feedback(self, orchestrator, ctx, mock_llm):
        mock_llm.invoke.return_value = "NOT_RELEVANT"
        search_ctx = SearchContext(
            feedback_context="What is the capital of France? Paris.",
            workspace_context="Some old workspace data",
        )

        handler = orchestrator._handle_context_gate(ctx, search_ctx)
        result = handler({})

        assert result.success is True
        assert search_ctx.feedback_context == ""
        assert search_ctx.workspace_context == ""


class TestContextGateRelevantPreserved:
    """When LLM says RELEVANT, feedback/workspace context should be kept."""

    def test_preserves_relevant_feedback(self, orchestrator, ctx, mock_llm):
        mock_llm.invoke.return_value = "RELEVANT"
        search_ctx = SearchContext(
            feedback_context="Proofread notes: check grammar carefully.",
            workspace_context="Style guide for proofreading",
        )

        handler = orchestrator._handle_context_gate(ctx, search_ctx)
        result = handler({})

        assert result.success is True
        assert search_ctx.feedback_context == "Proofread notes: check grammar carefully."
        assert search_ctx.workspace_context == "Style guide for proofreading"


class TestContextGateEmptySkipsLLM:
    """When there's no feedback/workspace context, LLM should not be called."""

    def test_skips_llm_when_empty(self, orchestrator, ctx, mock_llm):
        search_ctx = SearchContext()

        handler = orchestrator._handle_context_gate(ctx, search_ctx)
        result = handler({})

        assert result.success is True
        mock_llm.invoke.assert_not_called()

    def test_skips_llm_when_only_vector_context(self, orchestrator, ctx, mock_llm):
        search_ctx = SearchContext(vector_context="some vector data")

        handler = orchestrator._handle_context_gate(ctx, search_ctx)
        result = handler({})

        assert result.success is True
        mock_llm.invoke.assert_not_called()


class TestContextGateLLMFailure:
    """When the LLM call fails, context should be preserved (safe default)."""

    def test_preserves_context_on_error(self, orchestrator, ctx, mock_llm):
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")
        search_ctx = SearchContext(
            feedback_context="Some feedback",
        )

        handler = orchestrator._handle_context_gate(ctx, search_ctx)
        result = handler({})

        assert result.success is True
        assert search_ctx.feedback_context == "Some feedback"
