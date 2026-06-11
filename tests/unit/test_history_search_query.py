"""Conversation history must provide context for resolving the search query,
not bias it toward earlier topics.

``_build_history_search_query`` should leave a self-contained query untouched
(no rewrite) and only consult history when the latest message references prior
turns (pronouns / ellipsis).
"""
from unittest.mock import MagicMock

from smartloop.api.services.model_service import ModelService


def _svc(rewrite_return="REWRITTEN"):
    """A minimal stand-in exposing just what the method touches."""
    svc = MagicMock()
    svc._REFERENCE_WORDS = ModelService._REFERENCE_WORDS
    svc.generate_completion = MagicMock(return_value=rewrite_return)
    return svc


_HISTORY = [
    {"role": "user", "content": "tell me about Adam Hart's invoices"},
    {"role": "assistant", "content": "Adam Hart purchased chairs and a printer."},
]


def test_self_contained_query_used_verbatim_no_bias():
    """A complete query about a different entity is NOT rewritten through the
    history (which would drag 'Adam Hart' into an 'Aaron Bergman' query)."""
    svc = _svc()
    query = "what items did Aaron Bergman purchase"
    out = ModelService._build_history_search_query(svc, query, _HISTORY)
    assert out == query
    svc.generate_completion.assert_not_called()


def test_referential_query_is_resolved_via_history():
    """A query that references prior turns ('those') is rewritten."""
    svc = _svc(rewrite_return="Adam Hart invoice items")
    out = ModelService._build_history_search_query(svc, "what about those", _HISTORY)
    assert out == "Adam Hart invoice items"
    svc.generate_completion.assert_called_once()


def test_single_turn_history_returns_query_unchanged():
    svc = _svc()
    out = ModelService._build_history_search_query(
        svc, "hello", [{"role": "user", "content": "hello"}]
    )
    assert out == "hello"
    svc.generate_completion.assert_not_called()
