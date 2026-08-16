from smartloop.agents.web_search_agent import normalize_search_query


class TestNormalizeSearchQuery:
    def test_strips_give_me_filler(self):
        assert normalize_search_query("give me a list latest canyon bikes") == "canyon bikes"

    def test_strips_please_and_question_mark(self):
        assert normalize_search_query("please tell me the weather") == "the weather"

    def test_strips_leading_filler_and_trailing_politeness(self):
        assert normalize_search_query("can you show me the new macbooks?") == "new macbooks"

    def test_keeps_meaningful_subject(self):
        assert normalize_search_query("what is the latest iphone release date") == "iphone release date"

    def test_no_filler_leaves_message_intact(self):
        assert normalize_search_query("latest canyon bikes") == "canyon bikes"

    def test_empty_falls_back_to_raw(self):
        assert normalize_search_query("") == ""

    def test_only_filler_falls_back_to_raw(self):
        assert normalize_search_query("give me") == "give me"

    def test_collapses_whitespace(self):
        assert normalize_search_query("  tell me   about  canyon   bikes ") == "canyon bikes"

    def test_drops_weak_stopwords_so_endpoint_doesnt_latch_onto_them(self):
        assert normalize_search_query("the current stock prices for apple") == "stock prices apple"

    def test_keeps_single_subject_with_stopword(self):
        assert normalize_search_query("electric current") == "electric current"


class TestRerank:
    def _agent(self, llm=None):
        from smartloop.agents.web_search_agent import WebSearchAgent

        fetch = type("F", (), {"fetch": lambda self, url, max_length=0: url})()
        return WebSearchAgent(fetch_agent=fetch, llm=llm)

    def test_no_llm_keeps_endpoint_order(self):
        agent = self._agent()
        candidates = [("a.com", "aaa"), ("b.com", "bbb"), ("c.com", "ccc")]
        assert agent._rerank("query", candidates) == candidates

    def test_single_candidate_untouched(self):
        agent = self._agent()
        assert agent._rerank("query", [("a.com", "aaa")]) == [("a.com", "aaa")]

    def test_reranks_by_model_ranking(self):
        class FakeLLM:
            def invoke(self, prompt, max_tokens=0):
                assert "Query: canyon bikes" in prompt
                return "3,1,2"

        agent = self._agent(FakeLLM())
        candidates = [("a.com", "aaa"), ("b.com", "bbb"), ("c.com", "ccc")]
        ranked = agent._rerank("canyon bikes", candidates)
        assert [u for u, _ in ranked] == ["c.com", "a.com", "b.com"]

    def test_out_of_range_and_duplicate_indices_are_guarded(self):
        class FakeLLM:
            def invoke(self, prompt, max_tokens=0):
                return "5,1,1,2"

        agent = self._agent(FakeLLM())
        candidates = [("a.com", "aaa"), ("b.com", "bbb"), ("c.com", "ccc")]
        ranked = agent._rerank("query", candidates)
        assert [u for u, _ in ranked] == ["a.com", "b.com", "c.com"]

    def test_rerank_failure_falls_back_to_endpoint_order(self):
        class BoomLLM:
            def invoke(self, prompt, max_tokens=0):
                raise RuntimeError("model down")

        agent = self._agent(BoomLLM())
        candidates = [("a.com", "aaa"), ("b.com", "bbb"), ("c.com", "ccc")]
        assert agent._rerank("query", candidates) == candidates

    def test_garbled_response_appends_unranked_at_end(self):
        class GarbledLLM:
            def invoke(self, prompt, max_tokens=0):
                return "3 blah"

        agent = self._agent(GarbledLLM())
        candidates = [("a.com", "aaa"), ("b.com", "bbb"), ("c.com", "ccc")]
        ranked = agent._rerank("query", candidates)
        assert [u for u, _ in ranked] == ["c.com", "a.com", "b.com"]

