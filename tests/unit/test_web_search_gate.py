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
    def test_mcp_tool_result_no_longer_suppresses_web_search(self, monkeypatch, searched):
        """A live tool result used to cancel the search outright.

        "Deliberately invoked" only meant the router picked the tool, not that
        its output answered the question — so a query spanning both ("how does
        our roadmap compare to what they announced") lost half its context.
        Both are collected now; which one actually answers is synthesis's call,
        and the explore assembly puts the local source first.
        """
        ex = _explorer(monkeypatch, searched)
        monkeypatch.setattr(Explorer, "_run_mcp", lambda self, *a: "MCP TOOL (list_issues):\n#1 open")

        ctx = ex.collect_parallel(
            "list my issues", model_service=MagicMock(),
            mcp_servers=[MagicMock()], project_dir="/tmp/p",
        )

        assert searched == ["list my issues"]
        assert ctx.web_context
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


class TestFollowUpGate:
    """A follow-up that reworks the previous answer must not search again.

    "Show me latest local events" → "now create a table".  The second message
    carries none of the subject, so a fresh search returns pages about tables
    and drags them into the context and the citation list.
    """

    def _explorer_with_verdict(self, monkeypatch, searched, verdict):
        llm = MagicMock()
        llm.invoke = MagicMock(return_value=verdict)
        ex = _explorer(monkeypatch, searched)
        ex.llm = llm
        return ex, llm

    HISTORY = [
        {"role": "user", "content": "Show me latest local events"},
        {"role": "assistant", "content": "Jazz in the Park on Friday, Food Truck Rally Saturday."},
    ]

    def test_reformat_follow_up_does_not_search(self, monkeypatch, searched):
        ex, _ = self._explorer_with_verdict(monkeypatch, searched, "REUSE")

        ctx = ex.collect_parallel(
            "now create a table", model_service=MagicMock(),
            conversation_history=self.HISTORY,
        )

        assert searched == []
        assert not ctx.web_context
        assert ctx.web_sources == []

    def test_new_information_follow_up_still_searches(self, monkeypatch, searched):
        ex, _ = self._explorer_with_verdict(monkeypatch, searched, "SEARCH")

        ctx = ex.collect_parallel(
            "what about next weekend?", model_service=MagicMock(),
            conversation_history=self.HISTORY,
        )

        assert searched == ["what about next weekend?"]
        assert ctx.web_context

    def test_first_message_never_consults_the_gate(self, monkeypatch, searched):
        ex, llm = self._explorer_with_verdict(monkeypatch, searched, "REUSE")

        ex.collect_parallel("Show me latest local events", model_service=MagicMock())

        assert searched == ["Show me latest local events"]
        llm.invoke.assert_not_called()

    def test_gate_sees_the_transcript_and_the_latest_message(self, monkeypatch, searched):
        ex, llm = self._explorer_with_verdict(monkeypatch, searched, "REUSE")

        ex.collect_parallel(
            "now create a table", model_service=MagicMock(),
            conversation_history=self.HISTORY,
        )

        prompt = llm.invoke.call_args.args[0]
        assert "Show me latest local events" in prompt
        assert "Jazz in the Park" in prompt
        assert "Latest message: now create a table" in prompt

    def test_llm_failure_falls_open_to_searching(self, monkeypatch, searched):
        ex, llm = self._explorer_with_verdict(monkeypatch, searched, "REUSE")
        llm.invoke.side_effect = RuntimeError("model unavailable")

        ctx = ex.collect_parallel(
            "now create a table", model_service=MagicMock(),
            conversation_history=self.HISTORY,
        )

        assert searched == ["now create a table"]
        assert ctx.web_context

    @pytest.mark.parametrize("verdict", ["SEARCH", "search", " SEARCH\n", "unclear", ""])
    def test_anything_but_a_clear_reuse_searches(self, monkeypatch, searched, verdict):
        ex, _ = self._explorer_with_verdict(monkeypatch, searched, verdict)

        ex.collect_parallel(
            "now create a table", model_service=MagicMock(),
            conversation_history=self.HISTORY,
        )

        assert searched == ["now create a table"]

    def test_empty_history_entries_are_treated_as_no_history(self, monkeypatch, searched):
        ex, llm = self._explorer_with_verdict(monkeypatch, searched, "REUSE")

        ex.collect_parallel(
            "now create a table", model_service=MagicMock(),
            conversation_history=[{"role": "user", "content": ""}],
        )

        assert searched == ["now create a table"]
        llm.invoke.assert_not_called()


class TestHostedStatusCache:
    """The status check sits in front of every query, so it must be cheap and
    must never suppress a search it cannot verify."""

    def setup_method(self):
        from smartloop.search import providers

        providers.reset_status_cache()

    def teardown_method(self):
        from smartloop.search import providers

        providers.reset_status_cache()

    @staticmethod
    def _signed_in(monkeypatch, has_credentials=True):
        from smartloop.platform import auth as platform_auth

        monkeypatch.setattr(platform_auth, "has_credentials", lambda: has_credentials)

    @staticmethod
    def _client(monkeypatch, status, calls=None):
        """Point providers at a stub platform client whose status call is *status*.

        *status* is a zero-arg callable standing in for
        ``SLPPlatformClient.web_search_status`` — return a bool or raise.
        """
        from smartloop.search import providers

        class _Stub:
            def web_search_status(self, *a, **k):
                if calls is not None:
                    calls.append(1)
                return status()

        monkeypatch.setattr(providers, "_platform_client", lambda: _Stub())

    def test_result_is_cached_across_calls(self, monkeypatch):
        from smartloop.search import providers

        calls = []
        self._signed_in(monkeypatch)
        self._client(monkeypatch, lambda: True, calls)

        assert providers.hosted_search_available() is True
        assert providers.hosted_search_available() is True
        assert len(calls) == 1, "second call should come from the cache"

    def test_signed_out_short_circuits_without_a_request(self, monkeypatch):
        from smartloop.search import providers

        self._signed_in(monkeypatch, has_credentials=False)

        def _boom():
            raise AssertionError("must not call the endpoint without a token")

        monkeypatch.setattr(providers, "_platform_client", _boom)

        assert providers.hosted_search_available() is False

    def test_a_failed_check_fails_open(self, monkeypatch):
        """A health check that cannot complete says nothing about whether
        search works, and the search itself already skips on a 403."""
        from smartloop.search import providers

        self._signed_in(monkeypatch)
        self._client(monkeypatch, _raiser(ConnectionError("network down")))

        assert providers.hosted_search_available() is True

    def test_forbidden_reports_unavailable(self, monkeypatch):
        """Only an explicit 401/403 — surfaced by the client as PermissionError —
        counts as a real "search is off"."""
        from smartloop.search import providers

        self._signed_in(monkeypatch)
        self._client(monkeypatch, _raiser(PermissionError("HTTP 403")))

        assert providers.hosted_search_available() is False


def _raiser(exc):
    def _raise():
        raise exc
    return _raise


class TestHostedSearchProvider:
    """The search call itself goes through SLPPlatformClient; what stays here
    is the translation from the client's errors to the agent's skip signals."""

    @staticmethod
    def _client(monkeypatch, search):
        from smartloop.search import providers

        class _Stub:
            def web_search(self, query, count, *a, **k):
                return search(query, count)

        monkeypatch.setattr(providers, "_platform_client", lambda: _Stub())

    def test_returns_links(self, monkeypatch):
        from smartloop.platform import auth as platform_auth
        from smartloop.search import providers

        monkeypatch.setattr(platform_auth, "has_credentials", lambda: True)
        seen = {}

        def _search(query, count):
            seen.update(query=query, count=count)
            return ["http://a", "http://b"]

        self._client(monkeypatch, _search)

        assert providers.HostedSearchProvider().search("bikes", 5) == ["http://a", "http://b"]
        assert seen == {"query": "bikes", "count": 5}

    def test_count_is_clamped(self, monkeypatch):
        from smartloop.platform import auth as platform_auth
        from smartloop.search import providers

        monkeypatch.setattr(platform_auth, "has_credentials", lambda: True)
        counts = []
        self._client(monkeypatch, lambda q, c: counts.append(c) or [])

        providers.HostedSearchProvider().search("bikes", 99)
        providers.HostedSearchProvider().search("bikes", 0)
        assert counts == [20, 1]

    def test_no_credentials_raises(self, monkeypatch):
        from smartloop.platform import auth as platform_auth
        from smartloop.search import providers

        monkeypatch.setattr(platform_auth, "has_credentials", lambda: False)

        def _boom():
            raise AssertionError("must not call the endpoint without credentials")

        monkeypatch.setattr(providers, "_platform_client", _boom)

        with pytest.raises(providers.NoSearchCredentials):
            providers.HostedSearchProvider().search("bikes", 5)

    def test_permission_error_becomes_search_auth_error(self, monkeypatch):
        from smartloop.platform import auth as platform_auth
        from smartloop.search import providers

        monkeypatch.setattr(platform_auth, "has_credentials", lambda: True)
        self._client(monkeypatch, lambda q, c: (_ for _ in ()).throw(
            PermissionError("HTTP 403")))

        with pytest.raises(providers.SearchAuthError):
            providers.HostedSearchProvider().search("bikes", 5)

    def test_transient_error_is_not_an_auth_error(self, monkeypatch):
        """A 500 or a dropped connection must stay distinguishable from "off" —
        the agent skips either way, but only 401/403 means search is disabled."""
        from smartloop.platform import auth as platform_auth
        from smartloop.search import providers

        monkeypatch.setattr(platform_auth, "has_credentials", lambda: True)
        self._client(monkeypatch, lambda q, c: (_ for _ in ()).throw(
            ConnectionError("boom")))

        with pytest.raises(ConnectionError):
            providers.HostedSearchProvider().search("bikes", 5)
