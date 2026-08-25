"""
Tests for the document map-reduce fan-out in SLPOrchestrator.

When several documents match the search, the orchestrator extends the DAG
at runtime with one node per document/batch (parallel extraction) plus a
single reduce node that merges them into one synthesized document.  The
downstream skill/classify pipeline then hangs off that reduce node.
"""
from unittest.mock import MagicMock, patch

from smartloop.agents.orchestrator import SLPOrchestrator
from smartloop.agents.mcp.explorer import SearchContext
from smartloop.agents.nodes import (
    NodeResult, LLMDecisionNode, TransformNode,
    SkillMatchNode, CollectNode, RouteNode,
)


def _make_orchestrator():
    llm = MagicMock()
    llm.get_num_tokens = MagicMock(return_value=100)
    llm.get_max_tokens = MagicMock(return_value=8192)
    llm.invoke = MagicMock(return_value="OK")
    return SLPOrchestrator(orchestrator_llm=llm)


def test_frame_web_results_instructs_model_not_to_disclaim():
    orch = _make_orchestrator()
    framed = orch._frame_web_results("## [1] https://canyon.com\n\nSome bikes")
    assert "live web search was just performed" in framed
    assert "real-time search results" in framed
    assert "do NOT say you lack" in framed.lower() or "Do NOT say you lack" in framed
    assert "WEB SEARCH RESULTS:\n## [1] https://canyon.com" in framed


def _docs(n, size=100):
    return [
        {"document_name": f"doc{i}.pdf", "content": "X" * size + f" body {i}"}
        for i in range(n)
    ]


def _ctx(orch, user_message="summarize the documents"):
    return orch._build_context(
        user_message=user_message,
        mcp_servers=[],
        skills=None,
        cwd=None,
        document_context=None,
        has_attachments=False,
        project_dir=None,
        model_service=None,
        conversation_history=None,
    )


# ------------------------------------------------------------------ #
# _batch_documents packing                                            #
# ------------------------------------------------------------------ #

class TestBatchDocuments:
    def test_each_document_gets_its_own_batch(self):
        orch = _make_orchestrator()
        batches = orch._batch_documents(_docs(5, size=50))
        assert len(batches) == 5
        assert all(len(b) == 1 for b in batches)

    def test_all_documents_processed_regardless_of_count(self):
        orch = _make_orchestrator()
        docs = _docs(50, size=100)
        batches = orch._batch_documents(docs)
        assert len(batches) == 50
        assert all(len(b) == 1 for b in batches)
        names = [b[0]["document_name"] for b in batches]
        assert names == [d["document_name"] for d in docs]


# ------------------------------------------------------------------ #
# Route-level wiring                                                  #
# ------------------------------------------------------------------ #

class TestRouteFanout:
    def _route(self, orch, sctx, matched_skills=None):
        ctx = _ctx(orch)
        captured = {}

        skill_match_node = SkillMatchNode()
        collect_node = CollectNode(depends_on=[skill_match_node])
        route_node = RouteNode(depends_on=[skill_match_node, collect_node])

        from smartloop.agents.mcp.explorer import EnrichedContext
        ctx.enriched = EnrichedContext()

        def fake_pipeline(c, ms, sc, ec, parent=None):
            captured["parent"] = parent
            return []

        orch._build_unified_pipeline = fake_pipeline

        deps = {
            skill_match_node.node_id: NodeResult(
                node_id=skill_match_node.node_id,
                data={"matched_skills": matched_skills},
            ),
            collect_node.node_id: NodeResult(
                node_id=collect_node.node_id,
                data={"search_ctx": sctx},
            ),
        }
        res = orch._handle_route(
            ctx, deps, route_node, skill_match_node, collect_node,
        )
        nodes = {type(n).__name__: n for n, _ in res.data["_extend_dag"]}
        return nodes, captured

    def test_fanout_injected_for_multiple_documents(self):
        orch = _make_orchestrator()
        sctx = SearchContext(vector_context="raw", documents=_docs(3))
        nodes, captured = self._route(orch, sctx)

        batch_nodes = [n for n, _ in [
            (n, None) for n in nodes.values() if isinstance(n, LLMDecisionNode)
        ]]
        reduce_nodes = [n for n in nodes.values() if isinstance(n, TransformNode)]

        assert len(batch_nodes) >= 1
        assert len(reduce_nodes) >= 1
        assert isinstance(captured["parent"], TransformNode)

    def test_single_document_uses_one_batch_node(self):
        # DOC_FANOUT_THRESHOLD = 1: even a single document goes through the
        # same map-reduce structure so every retrieved document is processed
        # by its own DAG node.  With one doc that's one batch + one reduce.
        orch = _make_orchestrator()
        sctx = SearchContext(vector_context="raw", documents=_docs(1))
        nodes, captured = self._route(orch, sctx)

        batch_nodes = [n for n in nodes.values() if isinstance(n, LLMDecisionNode)]
        reduce_nodes = [n for n in nodes.values() if isinstance(n, TransformNode)]

        assert len(batch_nodes) == 1
        assert len(reduce_nodes) >= 1
        assert isinstance(captured["parent"], TransformNode)

    def test_no_fanout_when_no_documents(self):
        orch = _make_orchestrator()
        sctx = SearchContext(vector_context="raw")
        nodes, captured = self._route(orch, sctx)

        assert not any(isinstance(n, TransformNode) and n != captured.get("parent") for n in nodes.values())
        assert isinstance(captured["parent"], RouteNode)


# ------------------------------------------------------------------ #
# Batch + reduce handler behavior                                     #
# ------------------------------------------------------------------ #

class TestFanoutHandlers:
    def test_reduce_merges_extracts_and_replaces_context(self):
        orch = _make_orchestrator()
        orch.agent_llm.invoke = MagicMock(side_effect=lambda p, *a, **k: "EXTRACT::" + p[-1])

        sctx = SearchContext(vector_context="ORIGINAL RAW CONCAT", documents=_docs(3))
        ctx = _ctx(orch)

        parent = RouteNode()
        nodes_and_handlers, reduce_node = orch._build_doc_fanout_nodes(ctx, sctx, parent)
        assert isinstance(reduce_node, TransformNode)

        results = {}
        reduce_handler = None
        for node, handler in nodes_and_handlers:
            if isinstance(node, TransformNode):
                reduce_handler = handler
                continue
            results[node.node_id] = handler({})

        assert all(r.data["content"].startswith("EXTRACT::") for r in results.values())

        reduce_result = reduce_handler(results)
        merged = reduce_result.data["content"]
        assert merged.count("EXTRACT::") == len(results)
        assert sctx.vector_context == merged
        assert sctx.vector_context != "ORIGINAL RAW CONCAT"

    def test_reduce_restricts_citations_to_surviving_documents(self):
        orch = _make_orchestrator()
        def extract(prompt, *a, **k):
            return "Relevant extract." if "body 0" in prompt else "NONE"
        orch.agent_llm.invoke = MagicMock(side_effect=extract)

        sctx = SearchContext(
            documents=_docs(2, size=5000),
            citations=[
                {"document_id": "0", "document_name": "doc0.pdf"},
                {"document_id": "1", "document_name": "doc1.pdf"},
            ],
        )
        ctx = _ctx(orch)
        parent = RouteNode()
        nodes_and_handlers, _ = orch._build_doc_fanout_nodes(ctx, sctx, parent)

        results = {}
        reduce_handler = None
        for node, handler in nodes_and_handlers:
            if isinstance(node, TransformNode):
                reduce_handler = handler
                continue
            results[node.node_id] = handler({})

        reduce_handler(results)
        cited = {c["document_name"] for c in sctx.citations}
        assert cited == {"doc0.pdf"}
        assert sctx.cited_documents == ["doc0.pdf"]

    def test_extraction_uses_the_in_process_dp_model(self):
        """Every batch extracts on the shared in-process model.

        Extraction used to be dispatched to an LLM worker *process* when a
        pool existed, with ``parallel_inference=True`` to opt the node out of
        the DAG executor's LLM lock.  The pool is gone (same-device model
        copies serialize in the driver anyway), so there is one inference path
        and every LLMDecisionNode is serialized.
        """
        orch = _make_orchestrator()
        dp_llm = MagicMock()
        dp_llm.get_max_tokens = MagicMock(return_value=8192)
        dp_llm.invoke = MagicMock(return_value="DP EXTRACT")
        orch._llm_for_tier = MagicMock(return_value=dp_llm)

        sctx = SearchContext(documents=_docs(3))
        ctx = _ctx(orch)
        parent = RouteNode()
        nodes_and_handlers, _ = orch._build_doc_fanout_nodes(ctx, sctx, parent)

        batch_nodes = [n for n, _ in nodes_and_handlers if isinstance(n, LLMDecisionNode)]
        assert batch_nodes
        # The opt-out flag is gone, so the executor's lock always applies.
        assert not any(hasattr(n, "parallel_inference") for n in batch_nodes)

        for node, handler in nodes_and_handlers:
            if isinstance(node, LLMDecisionNode):
                assert handler({}).data["content"] == "DP EXTRACT"
        assert dp_llm.invoke.call_count == len(batch_nodes)

    def test_batch_drops_document_when_llm_returns_none(self):
        orch = _make_orchestrator()
        orch.agent_llm.invoke = MagicMock(return_value="NONE")

        sctx = SearchContext(documents=_docs(2))
        ctx = _ctx(orch)
        parent = RouteNode()
        nodes_and_handlers, _ = orch._build_doc_fanout_nodes(ctx, sctx, parent)

        batch_node, batch_handler = nodes_and_handlers[0]
        result = batch_handler({})
        assert result.data["content"] == ""
        assert "body" not in result.data["content"]

    def test_batch_falls_back_to_raw_when_llm_call_fails(self):
        orch = _make_orchestrator()
        orch.agent_llm.invoke = MagicMock(side_effect=RuntimeError("boom"))

        sctx = SearchContext(documents=_docs(2))
        ctx = _ctx(orch)
        parent = RouteNode()
        nodes_and_handlers, _ = orch._build_doc_fanout_nodes(ctx, sctx, parent)

        batch_node, batch_handler = nodes_and_handlers[0]
        result = batch_handler({})
        assert "body" in result.data["content"]

    def test_batch_falls_back_to_raw_when_llm_returns_placeholder(self):
        # llama.cpp's empty-output fallback string must never be surfaced as
        # extracted document content — treat it like a failure and keep a raw
        # excerpt so the document isn't silently replaced by placeholder text.
        orch = _make_orchestrator()
        orch.agent_llm.invoke = MagicMock(return_value="No response generated")

        sctx = SearchContext(documents=_docs(2))
        ctx = _ctx(orch)
        parent = RouteNode()
        nodes_and_handlers, _ = orch._build_doc_fanout_nodes(ctx, sctx, parent)

        batch_node, batch_handler = nodes_and_handlers[0]
        result = batch_handler({})
        assert "No response generated" not in result.data["content"]
        assert "body 0" in result.data["content"]

    def test_batch_skips_llm_when_document_is_json_table(self):
        import json as _json

        orch = _make_orchestrator()
        orch.agent_llm.invoke = MagicMock(side_effect=AssertionError("LLM must not run for JSON docs"))

        doc = {
            "document_name": "stats.json",
            "content": _json.dumps({
                "rows": [
                    {"name": "Drake Maye", "team": "NE", "values": {"epa": 0.296484, "td": 31}},
                    {"name": "Jordan Love", "team": "GB", "values": {"epa": 0.240092, "td": 23}},
                ]
            }),
        }
        sctx = SearchContext(documents=[doc])
        ctx = _ctx(orch)
        parent = RouteNode()
        nodes_and_handlers, _ = orch._build_doc_fanout_nodes(ctx, sctx, parent)

        batch_node, batch_handler = nodes_and_handlers[0]
        result = batch_handler({})
        content = result.data["content"]
        assert "| name | team | values.epa | values.td |" in content
        assert "| Drake Maye | NE | 0.2965 | 31 |" in content
        assert "| Jordan Love | GB | 0.2401 | 23 |" in content

    def test_batch_uses_existing_markdown_table_directly(self):
        # Content already normalized to a table at index time must be used
        # as-is — no LLM extraction (which could overflow the model window).
        orch = _make_orchestrator()
        orch.agent_llm.invoke = MagicMock(side_effect=AssertionError("LLM must not run for tables"))

        doc = {
            "document_name": "stats.txt",
            "content": (
                "| name | team | td |\n"
                "|---|---|---|\n"
                "| Drake Maye | NE | 31 |\n"
                "| Jordan Love | GB | 23 |\n"
            ),
        }
        sctx = SearchContext(documents=[doc])
        ctx = _ctx(orch)
        parent = RouteNode()
        nodes_and_handlers, _ = orch._build_doc_fanout_nodes(ctx, sctx, parent)

        batch_node, batch_handler = nodes_and_handlers[0]
        result = batch_handler({})
        assert "| Drake Maye | NE | 31 |" in result.data["content"]
        assert "| Jordan Love | GB | 23 |" in result.data["content"]


# ------------------------------------------------------------------ #
# _json_to_table                                                      #
# ------------------------------------------------------------------ #

class TestJsonToTable:
    def _orch(self):
        return _make_orchestrator()

    def test_top_level_array(self):
        table = self._orch()._json_to_table(
            '[{"name": "A", "v": 1}, {"name": "B", "v": 2}]'
        )
        assert "| name | v |" in table
        assert "| A | 1 |" in table
        assert "| B | 2 |" in table

    def test_rows_container(self):
        table = self._orch()._json_to_table('{"rows": [{"name": "A", "v": 1}]}')
        assert "| name | v |" in table
        assert "| A | 1 |" in table

    def test_nested_values_flattened_with_dotted_keys(self):
        table = self._orch()._json_to_table(
            '[{"name": "A", "values": {"epa": 0.296484, "td": 31}}]'
        )
        assert "| name | values.epa | values.td |" in table
        assert "| A | 0.2965 | 31 |" in table

    def test_float_compaction(self):
        table = self._orch()._json_to_table('[{"v": 4394.0, "epa": 0.296484}]')
        assert "| 4394 | 0.2965 |" in table

    def test_returns_none_for_non_json(self):
        assert self._orch()._json_to_table("just plain text") is None

    def test_returns_none_for_json_without_rows(self):
        assert self._orch()._json_to_table('{"category": "EPA/play"}') is None


# ------------------------------------------------------------------ #
# Full execute() integration                                         #
# ------------------------------------------------------------------ #

class TestFanoutIntegration:
    def test_execute_triggers_fanout_extraction(self):
        extraction_calls = {"n": 0}

        def side_effect(prompt, *a, **k):
            p = str(prompt).lower()
            if "extract the information relevant" in p:
                extraction_calls["n"] += 1
                return "Extracted fact from document."
            if "classify" in p or "intent" in p:
                return "conversation"
            return "OK"

        llm = MagicMock()
        llm.get_num_tokens = MagicMock(return_value=100)
        llm.get_max_tokens = MagicMock(return_value=8192)
        llm.invoke = MagicMock(side_effect=side_effect)

        orch = SLPOrchestrator(orchestrator_llm=llm)
        sctx = SearchContext(
            vector_context="raw concat",
            documents=_docs(3),
            citations=[{"document_id": f"d{i}", "document_name": f"doc{i}.pdf"} for i in range(3)],
        )

        with patch.object(orch.explorer, "collect_parallel", return_value=sctx):
            result = orch.execute(
                user_message="summarize all the documents",
                mcp_servers=[],
                skills=None,
                cwd="/tmp",
            )

        assert extraction_calls["n"] >= 1
        assert "Extracted fact from document." in sctx.vector_context
        assert result.handled is True
