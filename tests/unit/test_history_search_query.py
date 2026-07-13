"""Conversation history always grounds the search query for a session.

``_build_history_search_query`` no longer applies a semantic gate: whenever the
session has prior turns it consults history via an LLM rewrite. The rewrite
prompt is responsible for returning a self-contained message UNCHANGED and for
not dragging in entities the latest message does not reference.
"""
from unittest.mock import MagicMock

from smartloop.api.services.model_service import ModelService


def _svc(rewrite_return="REWRITTEN"):
    """A minimal stand-in exposing just what the method touches."""
    svc = MagicMock()
    svc.generate_completion = MagicMock(return_value=rewrite_return)
    return svc


_HISTORY = [
    {"role": "user", "content": "tell me about Adam Hart's invoices"},
    {"role": "assistant", "content": "Adam Hart purchased chairs and a printer."},
]


def test_self_contained_query_still_consults_history():
    """A self-contained query is passed through the rewrite; the LLM returns it
    unchanged (its prompt says to), so the output matches the input."""
    query = "what items did Aaron Bergman purchase"
    svc = _svc(rewrite_return=query)
    out = ModelService._build_history_search_query(svc, query, _HISTORY)
    assert out == query
    svc.generate_completion.assert_called_once()


def test_referential_query_is_resolved_via_history():
    """A query that references prior turns ('those') is rewritten."""
    svc = _svc(rewrite_return="Adam Hart invoice items")
    out = ModelService._build_history_search_query(svc, "what about those", _HISTORY)
    assert out == "Adam Hart invoice items"
    svc.generate_completion.assert_called_once()


def test_definite_article_followup_is_resolved_via_history():
    """A multi-word follow-up with no pronoun ('the place') — previously skipped
    by the word-count gate — now gets grounded against history."""
    svc = _svc(rewrite_return="key highlights of the report")
    out = ModelService._build_history_search_query(
        svc, "what are the key highlights of the place", _HISTORY
    )
    assert out == "key highlights of the report"
    svc.generate_completion.assert_called_once()


def test_single_turn_history_returns_query_unchanged():
    svc = _svc()
    out = ModelService._build_history_search_query(
        svc, "hello", [{"role": "user", "content": "hello"}]
    )
    assert out == "hello"
    svc.generate_completion.assert_not_called()
