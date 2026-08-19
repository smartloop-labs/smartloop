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



class TestEnablement:
    """The per-project ``web_search_enabled`` flag must actually stop searches."""

    def _write_config(self, tmp_path, enabled, monkeypatch):
        import json

        from smartloop.workspace.workspace import Workspace

        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({
            "id": "m",
            "workspace": {
                "id": "ws",
                "workspace_dir": str(tmp_path),
                "models_dir": str(tmp_path),
                "models": [],
                "projects": [
                    {"id": "other", "project_dir": str(tmp_path), "current": False,
                     "name": "other", "web_search_enabled": True},
                    {"id": "p1", "project_dir": str(tmp_path), "current": True,
                     "name": "p1", "web_search_enabled": enabled},
                ],
            },
        }))
        monkeypatch.setattr(Workspace, "config_file", str(cfg))
        return Workspace

    def test_disabled_project_skips_search(self, tmp_path, monkeypatch):
        from smartloop.agents.web_search_agent import WebSearchAgent
        import smartloop.search.providers as providers

        self._write_config(tmp_path, False, monkeypatch)
        calls = []
        monkeypatch.setattr(providers, "get_search_provider",
                            lambda: type("P", (), {"search": lambda self, q, count: calls.append(q) or ["http://x"]})())

        assert WebSearchAgent.is_enabled() is False
        assert WebSearchAgent().search("latest canyon bikes") == ""
        assert calls == []

    def test_enabled_project_reports_enabled(self, tmp_path, monkeypatch):
        from smartloop.agents.web_search_agent import WebSearchAgent

        self._write_config(tmp_path, True, monkeypatch)
        assert WebSearchAgent.is_enabled() is True

    def test_flag_absent_from_disk_defaults_to_enabled(self, tmp_path, monkeypatch):
        """A project written before the flag existed (or freshly created) is on."""
        import json

        from smartloop.agents.web_search_agent import WebSearchAgent
        from smartloop.workspace.workspace import Workspace

        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({
            "id": "m",
            "workspace": {
                "id": "ws", "workspace_dir": str(tmp_path), "models_dir": str(tmp_path),
                "models": [],
                # No web_search_enabled key at all.
                "projects": [{"id": "p1", "project_dir": str(tmp_path),
                              "current": True, "name": "p1"}],
            },
        }))
        monkeypatch.setattr(Workspace, "config_file", str(cfg))
        assert WebSearchAgent.is_enabled() is True

    def test_new_project_is_enabled_by_default(self):
        from smartloop.workspace.workspace import Project

        project = Project(id="p1", project_dir="/tmp/p1", current=True, name="p1")
        assert project.web_search_enabled is True

    def test_disk_state_wins_over_stale_cached_workspace(self, tmp_path, monkeypatch):
        from smartloop.agents.web_search_agent import WebSearchAgent

        Workspace = self._write_config(tmp_path, False, monkeypatch)
        # A worker process can hold a singleton loaded before the user toggled
        # web search off; the on-disk flag is the source of truth.
        monkeypatch.setattr(Workspace, "_instance", Workspace.model_validate({
            "id": "ws", "workspace_dir": str(tmp_path), "models_dir": str(tmp_path),
            "models": [],
            "projects": [{"id": "p1", "project_dir": str(tmp_path), "current": True,
                          "name": "p1", "web_search_enabled": True}],
        }))
        assert WebSearchAgent.is_enabled() is False


class TestSourceExtracts:
    """The kept pages' text is recorded so a link can be verified later."""

    def _agent(self, pages):
        from smartloop.agents.web_search_agent import WebSearchAgent

        fetch = type("F", (), {
            "fetch": lambda self, url, max_length=0: pages.get(url, ""),
        })()
        return WebSearchAgent(fetch_agent=fetch, top_k=2, num_candidates=2)

    def test_fetch_top_records_text_per_kept_url(self):
        # Pages must clear the usability floor (>= 200 chars of real text).
        pages = {
            "a.com": "Canyon Aeroad CFR weighs 6.8kg. " + "Aero road details. " * 12,
            "b.com": "Canyon Ultimate CF SLX starts at 4299 euros. " + "Spec notes. " * 16,
        }
        agent = self._agent(pages)
        agent._fetch_top("canyon bikes", ["a.com", "b.com"], 2, 2)

        assert agent.last_sources == ["a.com", "b.com"]
        assert agent.last_source_extracts == {
            url: text.strip() for url, text in pages.items()
        }

    def test_unfetchable_links_leave_no_extracts(self):
        agent = self._agent({})
        out = agent._fetch_top("canyon bikes", ["a.com"], 1, 1)

        assert "a.com" in out
        assert agent.last_source_extracts == {}
