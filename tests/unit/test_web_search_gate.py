"""Web search must not be cancelled by an unrelated local retrieval hit.

Vector/feedback retrieval always returns its nearest neighbours, related or
not, so gating the search on "did anything match locally?" let one indexed
document or one stored feedback Q&A silence it for every query.  Which of the
collected sources actually answers the question is decided downstream.
"""

from unittest.mock import MagicMock

import pytest

from smartloop.agents.mcp.explorer import Explorer


def _explorer(monkeypatch, searched, enabled=True):
    ex = Explorer(llm=MagicMock())
    monkeypatch.setattr(ex.web_search_agent, "is_enabled", lambda: enabled)
    monkeypatch.setattr(ex.web_search_agent, "last_sources", ["http://x"])
    monkeypatch.setattr(
        ex, "_search_web",
        lambda q: searched.append(q) or "X2D II: 100MP medium format.",
    )
    monkeypatch.setattr(Explorer, "_search_vector", staticmethod(lambda *a: None))
    monkeypatch.setattr(Explorer, "_search_workspace", staticmethod(lambda *a: ""))
    monkeypatch.setattr(Explorer, "_search_feedback", staticmethod(lambda *a: ""))
    return ex


@pytest.fixture
def searched():
    return []


class TestWebSearchRunsAlongsideLocalContext:
    def test_unrelated_feedback_does_not_suppress_web_search(self, monkeypatch, searched):
        ex = _explorer(monkeypatch, searched)
        monkeypatch.setattr(
            Explorer, "_search_feedback",
            staticmethod(lambda *a: "Q: cat breeds?\nA: Siberian cats shed little."),
        )

        ctx = ex.collect_parallel("what is the Hasselblad X2D II", model_service=MagicMock())

        assert searched == ["what is the Hasselblad X2D II"]
        assert ctx.web_context
        assert ctx.web_sources == ["http://x"]
        assert ctx.feedback_context  # local hit is still collected alongside

    def test_indexed_document_does_not_suppress_web_search(self, monkeypatch, searched):
        ex = _explorer(monkeypatch, searched)
        monkeypatch.setattr(
            Explorer, "_search_vector",
            staticmethod(lambda *a: ("Siberian cats have a triple coat.", [], [])),
        )

        ctx = ex.collect_parallel("what is the Hasselblad X2D II", model_service=MagicMock())

        assert searched == ["what is the Hasselblad X2D II"]
        assert ctx.web_context
        assert ctx.vector_context

    def test_workspace_hit_does_not_suppress_web_search(self, monkeypatch, searched):
        ex = _explorer(monkeypatch, searched)
        monkeypatch.setattr(
            Explorer, "_search_workspace", staticmethod(lambda *a: "--- notes.md ---\nold notes"),
        )

        ctx = ex.collect_parallel("latest camera releases", model_service=MagicMock(), cwd="/tmp")

        assert searched == ["latest camera releases"]
        assert ctx.web_context


class TestWebSearchStillGated:
    def test_mcp_tool_result_suppresses_web_search(self, monkeypatch, searched):
        ex = _explorer(monkeypatch, searched)
        monkeypatch.setattr(Explorer, "_run_mcp", lambda self, *a: "MCP TOOL (list_issues):\n#1 open")

        ctx = ex.collect_parallel(
            "list my issues", model_service=MagicMock(),
            mcp_servers=[MagicMock()], project_dir="/tmp/p",
        )

        assert searched == []
        assert ctx.web_context == ""
        assert ctx.mcp_context

    def test_disabled_project_never_searches(self, monkeypatch, searched):
        ex = _explorer(monkeypatch, searched, enabled=False)

        ctx = ex.collect_parallel("what is the Hasselblad X2D II", model_service=MagicMock())

        assert searched == []
        assert ctx.web_context == ""


class TestFeedbackRelevanceFloor:
    """Feedback retrieval applies the same quality floor as documents.

    It was the one path with no floor — recall to a fixed 0.65 distance and
    return everything — so unrelated stored Q&A landed in the prompt and the
    model answered from it.
    """

    def _service(self, indexer):
        from smartloop.api.services.model_service import ModelService

        svc = ModelService.__new__(ModelService)
        svc._model = MagicMock()
        svc._feedback_indexer = indexer
        return svc

    def _chunk(self, distance):
        from smartloop.db.store import ChunkResult

        return ChunkResult(text=f"Q: q?\nA: a. ({distance})", document_id="f1",
                           document_name="feedback/s/m", distance=distance,
                           metadata={}, source_content="")

    def test_weak_best_match_returns_nothing(self):
        indexer = MagicMock()
        indexer.search.return_value = [self._chunk(0.62)]

        assert self._service(indexer).search_feedback("what is the X2D II") == []

    def test_strong_match_kept_and_far_neighbours_trimmed(self):
        indexer = MagicMock()
        indexer.search.return_value = [self._chunk(0.20), self._chunk(0.28), self._chunk(0.50)]

        kept = self._service(indexer).search_feedback("refund window")

        assert [c.distance for c in kept] == [0.20, 0.28]

    def test_caps_at_requested_n_results(self):
        indexer = MagicMock()
        indexer.search.return_value = [self._chunk(0.10) for _ in range(10)]

        assert len(self._service(indexer).search_feedback("refund window", n_results=3)) == 3
