"""Crawled MCP content has to reach the prompt to be worth crawling.

Covers the retrieval half: the explore fan-out searches the ``mcp_documents``
collection, the result counts as indexed context, and the relevance gate can
drop it like any other stored source.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from smartloop.agents.mcp.explorer import Explorer, SearchContext
from smartloop.db.store import ChunkResult


def _chunk(name, text, distance=0.2):
    return ChunkResult(
        text=text, document_id=name, document_name=name, distance=distance,
    )


class TestExplorerSearch:
    def test_crawled_records_are_returned_as_labelled_context(self):
        svc = MagicMock()
        svc.search_mcp_documents.return_value = [
            _chunk("notion/pages/Roadmap", "Q3 goals: ship the crawler"),
            _chunk("linear/issues/DEV-1", "Fix login redirect"),
        ]
        text = Explorer._search_mcp_documents("what is on the roadmap", svc)
        assert "notion/pages/Roadmap" in text
        assert "Q3 goals: ship the crawler" in text
        assert "linear/issues/DEV-1" in text

    def test_closest_chunk_wins_per_record(self):
        svc = MagicMock()
        svc.search_mcp_documents.return_value = [
            _chunk("notion/pages/Roadmap", "far chunk", distance=0.6),
            _chunk("notion/pages/Roadmap", "near chunk", distance=0.1),
        ]
        text = Explorer._search_mcp_documents("roadmap", svc)
        assert "near chunk" in text
        assert "far chunk" not in text

    def test_no_results_is_empty_not_an_error(self):
        svc = MagicMock()
        svc.search_mcp_documents.return_value = []
        assert Explorer._search_mcp_documents("anything", svc) == ""

    def test_search_failure_is_swallowed(self):
        svc = MagicMock()
        svc.search_mcp_documents.side_effect = RuntimeError("store gone")
        assert Explorer._search_mcp_documents("anything", svc) == ""

    def test_history_is_passed_through_for_follow_ups(self):
        svc = MagicMock()
        svc.search_mcp_documents.return_value = []
        history = [{"role": "user", "content": "what about the roadmap?"}]
        Explorer._search_mcp_documents("and the next one?", svc, history)
        assert svc.search_mcp_documents.call_args.kwargs["conversation_history"] is history


class TestSearchContext:
    def test_crawled_content_counts_as_context(self):
        ctx = SearchContext(mcp_document_context="notion page text")
        assert ctx.has_context is True

    def test_field_is_separate_from_the_live_tool_result(self):
        # A live tool call suppresses the web search; stored crawl content must
        # not, so the two cannot share a field.
        ctx = SearchContext(mcp_document_context="stored")
        assert ctx.mcp_context == ""


@pytest.fixture
def orchestrator():
    llm = MagicMock()
    llm.invoke = MagicMock(return_value="")
    llm.get_max_tokens = MagicMock(return_value=4096)
    llm.get_num_tokens = MagicMock(return_value=100)
    with patch("smartloop.agents.orchestrator.LLMExecutor"), \
         patch("smartloop.agents.orchestrator.DAGExecutor"), \
         patch("smartloop.config.app_settings.AppSettings"):
        from smartloop.agents.orchestrator import SLPOrchestrator
        return SLPOrchestrator(orchestrator_llm=llm)


@pytest.fixture
def ctx():
    from smartloop.agents.orchestrator import _OrchestratorContext
    return _OrchestratorContext(
        user_message="what is on the roadmap?",
        mcp_servers=[],
        available_servers=[],
        effective_cwd="/tmp",
        document_context=None,
        project_dir=None,
        model_service=MagicMock(),
        conversation_history=[],
        has_attachments=False,
    )


class TestRelevanceGate:
    """Crawled MCP content is judged before it reaches the answer context."""

    def test_irrelevant_crawled_content_is_cleared(self, orchestrator, ctx):
        orchestrator.agent_llm.invoke = MagicMock(return_value="NOT_RELEVANT")
        search_ctx = SearchContext(mcp_document_context="unrelated trello cards")
        orchestrator._judge.make_handler(ctx, search_ctx)({})
        assert search_ctx.mcp_document_context == ""

    def test_relevant_crawled_content_is_kept(self, orchestrator, ctx):
        orchestrator.agent_llm.invoke = MagicMock(return_value="RELEVANT")
        search_ctx = SearchContext(mcp_document_context="Q3 roadmap: ship it")
        orchestrator._judge.make_handler(ctx, search_ctx)({})
        assert search_ctx.mcp_document_context == "Q3 roadmap: ship it"
