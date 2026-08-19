"""
Regression tests for ModelService vector-search grouping.

`_vector_search_grouped` powers both get_document_context_with_citations
(concatenated string) and get_document_groups (per-document fan-out). Its
exceptions are swallowed and turned into None, so a NameError there fails
silently at runtime — these tests exercise the real code path with a
mocked store to catch that class of bug.
"""
from unittest.mock import MagicMock, patch

from smartloop.api.services.model_service import ModelService
from smartloop.db.store import ChunkResult


def _service_with_store(store):
    """Build a bare ModelService wired to a fake project + store."""
    svc = ModelService.__new__(ModelService)
    svc._doc_store = None
    svc._doc_store_project_id = None

    project = MagicMock()
    project.id = "proj-1"
    project.project_dir = "/tmp/proj-1"
    project.documents = []

    workspace = MagicMock()
    workspace.current_project = project
    workspace.workspace_dir = "/tmp"

    model = MagicMock()
    model.workspace = workspace
    svc._model = model
    return svc


def _chunks():
    return [
        ChunkResult(text="t1", document_id="d1", document_name="alpha.pdf",
                    distance=0.10, metadata={"page_number": 1}, source_content="Alpha body"),
        ChunkResult(text="t2", document_id="d2", document_name="beta.pdf",
                    distance=0.20, metadata={"page_number": 2}, source_content="Beta body"),
    ]


def _fake_store():
    chunks = _chunks()
    store = MagicMock()
    store.query_with_metadata.return_value = chunks
    store.get_by_document_ids.side_effect = (
        lambda ids: [c for c in chunks if c.document_id in ids]
    )
    return store


def test_get_document_groups_returns_documents_and_citations():
    store = _fake_store()
    svc = _service_with_store(store)
    with patch("smartloop.api.services.model_service.StoreFactory.create_store",
               return_value=store):
        result = svc.get_document_groups(query="alpha beta")

    assert result is not None, "vector search returned None (swallowed exception?)"
    documents, citations = result
    assert {d["document_name"] for d in documents} == {"alpha.pdf", "beta.pdf"}
    assert all("content" in d for d in documents)
    # Citations are built (this is the regression: all_citations must exist).
    assert {c["document_name"] for c in citations} == {"alpha.pdf", "beta.pdf"}
    assert any(c.get("page_numbers") for c in citations)


def test_get_document_context_with_citations_tuple_shape():
    store = _fake_store()
    svc = _service_with_store(store)
    with patch("smartloop.api.services.model_service.StoreFactory.create_store",
               return_value=store):
        result = svc.get_document_context_with_citations(query="alpha beta")

    assert result is not None
    context, citations = result
    assert "=== Document: alpha.pdf ===" in context
    assert "Alpha body" in context
    assert len(citations) == 2


def _weak_match_store():
    """Store where every chunk is only weakly related (distance > 0.55)."""
    chunks = [
        ChunkResult(text="t1", document_id="d1", document_name="unrelated.pdf",
                    distance=0.72, metadata={"page_number": 1},
                    source_content="Unrelated body"),
    ]
    store = MagicMock()
    store.query_with_metadata.return_value = chunks
    store.get_by_document_ids.side_effect = (
        lambda ids: [c for c in chunks if c.document_id in ids]
    )
    return store


def test_weak_best_match_returns_none():
    """Best-match distance above the quality floor → no documents surfaced,
    so purely conversational prompts don't get routed through EXPLORE."""
    store = _weak_match_store()
    svc = _service_with_store(store)
    with patch("smartloop.api.services.model_service.StoreFactory.create_store",
               return_value=store):
        assert svc.get_document_groups(query="places to visit in madrid") is None
        assert svc.get_document_context_with_citations(
            query="places to visit in madrid"
        ) is None


def test_citations_carry_the_document_text_as_a_private_extract():
    """Each citation ships the text its document contributed to the prompt.

    Without it the router has no evidence to check the finished answer
    against, so every retrieved document gets credited whether or not the
    answer drew on it.
    """
    store = _fake_store()
    svc = _service_with_store(store)
    with patch("smartloop.api.services.model_service.StoreFactory.create_store",
               return_value=store):
        _, citations = svc.get_document_groups(query="alpha beta")

    by_name = {c["document_name"]: c for c in citations}
    assert by_name["alpha.pdf"]["_extract"] == "Alpha body"
    assert by_name["beta.pdf"]["_extract"] == "Beta body"
