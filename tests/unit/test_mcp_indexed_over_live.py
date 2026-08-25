"""Crawled MCP content answers instead of calling the server again.

A crawl already pulled this server's content, so asking the server the same
question returns what is already in hand — at the cost of a network round-trip
that can also fail. The live call is made only when nothing was indexed.

The trade is freshness for latency, which is only honest if the model can tell
how old the snapshot is; hence the age travelling with the text.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from smartloop.agents.mcp.explorer import Explorer, _crawl_age


class TestCrawlAge:
    def _chunk(self, age_seconds):
        return SimpleNamespace(metadata={"crawled_at": time.time() - age_seconds})

    def test_recent_content_reads_as_just_now(self):
        assert _crawl_age(self._chunk(10)) == " (indexed just now)"

    def test_minutes_hours_and_days(self):
        assert _crawl_age(self._chunk(300)) == " (indexed 5m ago)"
        assert _crawl_age(self._chunk(7200)) == " (indexed 2h ago)"
        assert _crawl_age(self._chunk(200_000)) == " (indexed 2d ago)"

    def test_content_without_a_timestamp_claims_nothing(self):
        # Better to say nothing than to guess an age.
        assert _crawl_age(SimpleNamespace(metadata={})) == ""
        assert _crawl_age(SimpleNamespace(metadata={"crawled_at": "nonsense"})) == ""
        assert _crawl_age(SimpleNamespace(metadata=None)) == ""


class TestIndexedContentCarriesItsAge:
    def test_the_age_reaches_the_prompt(self):
        # Without this the model states a fifteen-minute-old quote as the
        # current price, with nothing in the prompt to suggest otherwise.
        svc = MagicMock()
        svc.search_mcp_documents.return_value = [
            SimpleNamespace(
                text="AAPL 310.34",
                document_id="d1",
                document_name="yahoo-finance/stock-info/AAPL",
                distance=0.1,
                metadata={"crawled_at": time.time() - 900},
            ),
        ]
        text = Explorer._search_mcp_documents("apple price", svc)
        assert "yahoo-finance/stock-info/AAPL (indexed 15m ago)" in text
        assert "AAPL 310.34" in text


class TestLiveCallIsSkipped:
    """The ordering that makes the rule work: the cheap local search resolves
    before the expensive round-trip is even submitted."""

    def _explorer(self):
        ex = Explorer.__new__(Explorer)
        ex.web_search_agent = MagicMock()
        ex.web_search_agent.is_enabled.return_value = False
        return ex

    def _collect(self, doc_context):
        ex = self._explorer()
        with patch.object(Explorer, "_search_mcp_documents", return_value=doc_context), \
             patch.object(Explorer, "_run_mcp", return_value="LIVE") as run_mcp, \
             patch.object(Explorer, "_search_vector", return_value=None), \
             patch.object(Explorer, "_search_feedback", return_value=""), \
             patch.object(Explorer, "_search_workspace", return_value=""):
            ctx = ex.collect_parallel(
                "what is AAPL trading at",
                model_service=MagicMock(),
                mcp_servers=[SimpleNamespace(id="s1", name="yahoo-finance")],
                project_dir="/tmp/p1",
            )
        return ctx, run_mcp

    def test_indexed_content_replaces_the_live_call(self):
        ctx, run_mcp = self._collect("--- yahoo-finance/stock-info/AAPL ---\n310.34")
        run_mcp.assert_not_called()
        assert ctx.mcp_context == ""
        assert "310.34" in ctx.mcp_document_context

    def test_nothing_indexed_still_calls_the_server(self):
        # The crawl covers what it reached; a question it doesn't answer must
        # still reach the server.
        ctx, run_mcp = self._collect("")
        run_mcp.assert_called_once()
        assert ctx.mcp_context == "LIVE"
