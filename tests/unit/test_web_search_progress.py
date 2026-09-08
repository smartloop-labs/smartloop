"""Web search reports its progress on the SSE stream.

Since web search moved into ``Explorer.collect_parallel`` it ran with no
progress events at all — only the legacy ``WEB_SEARCH`` plan step emitted one.
It is also the longest leg of a request (a provider round-trip plus one serial
fetch per result), so the client sat on the previous step's status line for the
whole search.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from smartloop.agents.mcp.explorer import Explorer
from smartloop.agents.web_search_agent import WebSearchAgent


# ── the agent reports each page as it is fetched ──────────────────────────

@pytest.fixture
def agent():
    fetch = type("F", (), {
        "fetch": lambda self, url, max_length=0: f"content of {url} " * 40,
    })()
    # llm=None → no rerank, so the endpoint order is kept and the assertions
    # below can name the sources they expect.
    return WebSearchAgent(fetch_agent=fetch, top_k=2, num_candidates=2, llm=None)


def _events(sink):
    return [c.args[2] for c in sink.call_args_list]


def test_each_fetched_page_is_reported(agent):
    sink = MagicMock()
    links = ["https://example.com/a", "https://other.org/b"]

    with patch.object(WebSearchAgent, "is_enabled", return_value=True), \
         patch.object(WebSearchAgent, "_fetch_links", return_value=links):
        agent.search("canyon bikes", progress=sink)

    messages = _events(sink)
    assert "Reading example.com" in messages
    assert "Reading other.org" in messages


def test_a_summary_is_reported_when_the_search_finishes(agent):
    sink = MagicMock()
    with patch.object(WebSearchAgent, "is_enabled", return_value=True), \
         patch.object(WebSearchAgent, "_fetch_links",
                      return_value=["https://example.com/a", "https://other.org/b"]):
        agent.search("canyon bikes", progress=sink)

    assert _events(sink)[-1] == "Read 2 web sources"


def test_the_summary_is_singular_for_one_source(agent):
    sink = MagicMock()
    agent._top_k = 1
    with patch.object(WebSearchAgent, "is_enabled", return_value=True), \
         patch.object(WebSearchAgent, "_fetch_links",
                      return_value=["https://example.com/a"]):
        agent.search("canyon bikes", progress=sink)

    assert _events(sink)[-1] == "Read 1 web source"


def test_events_use_the_web_search_step_name(agent):
    sink = MagicMock()
    with patch.object(WebSearchAgent, "is_enabled", return_value=True), \
         patch.object(WebSearchAgent, "_fetch_links",
                      return_value=["https://example.com/a"]):
        agent.search("canyon bikes", progress=sink)

    assert all(c.args[0] == "web_search" for c in sink.call_args_list)
    assert all(c.args[1] == "processing" for c in sink.call_args_list)


def test_a_failing_sink_never_breaks_the_search(agent):
    """Progress is cosmetic; the answer is not."""
    sink = MagicMock(side_effect=RuntimeError("client went away"))
    with patch.object(WebSearchAgent, "is_enabled", return_value=True), \
         patch.object(WebSearchAgent, "_fetch_links",
                      return_value=["https://example.com/a"]):
        content = agent.search("canyon bikes", progress=sink)

    assert "example.com" in content


def test_search_without_a_sink_still_works(agent):
    with patch.object(WebSearchAgent, "is_enabled", return_value=True), \
         patch.object(WebSearchAgent, "_fetch_links",
                      return_value=["https://example.com/a"]):
        assert agent.search("canyon bikes")


# ── collect announces the search before it starts ─────────────────────────

def _collect(*, enabled, needed=True, progress=None):
    ex = Explorer.__new__(Explorer)
    ex.web_search_agent = MagicMock()
    ex.web_search_agent.is_enabled.return_value = enabled
    ex.web_search_agent.last_sources = ["https://example.com/a"]
    ex.web_search_agent.search.return_value = "WEB"

    with patch.object(Explorer, "_search_vector", return_value=None), \
         patch.object(Explorer, "_search_feedback", return_value=""), \
         patch.object(Explorer, "_search_workspace", return_value=""), \
         patch.object(Explorer, "_needs_web_search", return_value=needed):
        ctx = ex.collect_parallel(
            "what did they announce today",
            model_service=MagicMock(),
            progress=progress,
        )
    return ex, ctx


def test_collect_announces_the_search_before_running_it():
    sink = MagicMock()
    _collect(enabled=True, progress=sink)

    assert _events(sink) == ["Searching the web for the latest information"]


def test_collect_forwards_the_sink_to_the_agent():
    """Per-page events have to come from the agent, mid-fetch."""
    sink = MagicMock()
    ex, _ = _collect(enabled=True, progress=sink)

    assert ex.web_search_agent.search.call_args.kwargs["progress"] is sink


def test_a_skipped_search_stays_silent():
    sink = MagicMock()
    _collect(enabled=False, progress=sink)
    sink.assert_not_called()


def test_a_search_not_needed_stays_silent():
    """A follow-up that only reworks the answer on screen searches nothing."""
    sink = MagicMock()
    _collect(enabled=True, needed=False, progress=sink)
    sink.assert_not_called()


def test_collect_without_a_sink_still_searches():
    _, ctx = _collect(enabled=True, progress=None)
    assert ctx.web_context == "WEB"
