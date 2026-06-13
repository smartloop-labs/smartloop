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
    EnrichNode, SkillMatchNode, CollectNode, RouteNode,
)


def _make_orchestrator():
    llm = MagicMock()
    llm.get_num_tokens = MagicMock(return_value=100)
    llm.get_max_tokens = MagicMock(return_value=8192)
    llm.invoke = MagicMock(return_value="OK")
    return SLPOrchestrator(orchestrator_llm=llm)


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

        enrich_node = EnrichNode()
        skill_match_node = SkillMatchNode(depends_on=[enrich_node])
        collect_node = CollectNode(depends_on=[skill_match_node, enrich_node])
        route_node = RouteNode(depends_on=[skill_match_node, collect_node, enrich_node])

        from smartloop.agents.mcp.explorer import EnrichedContext
        enrich_ctx = EnrichedContext()

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
            enrich_node.node_id: NodeResult(
                node_id=enrich_node.node_id,
                data={"enrich_ctx": enrich_ctx},
            ),
        }
        res = orch._handle_route(
            ctx, deps, route_node, skill_match_node, enrich_node, collect_node,
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

    def test_no_fanout_for_single_document(self):
        orch = _make_orchestrator()
        sctx = SearchContext(vector_context="raw", documents=_docs(1))
        nodes, captured = self._route(orch, sctx)

        assert not any(isinstance(n, LLMDecisionNode) for n in nodes.values())
        assert isinstance(captured["parent"], RouteNode)

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

    def test_uses_worker_pool_when_available(self):
        from concurrent.futures import Future

        class FakePool:
            has_llm_pool = True

            def __init__(self):
                self.calls = 0

            def submit_llm(self, prompt):
                self.calls += 1
                f = Future()
                f.set_result("POOL EXTRACT")
                return f

        orch = _make_orchestrator()
        orch.agent_llm.invoke = MagicMock(side_effect=AssertionError("used in-thread model"))
        pool = FakePool()
        orch._executor = pool

        sctx = SearchContext(documents=_docs(3))
        ctx = _ctx(orch)
        parent = RouteNode()
        nodes_and_handlers, _ = orch._build_doc_fanout_nodes(ctx, sctx, parent)

        batch_nodes = [n for n, _ in nodes_and_handlers if isinstance(n, LLMDecisionNode)]
        assert all(n.parallel_inference for n in batch_nodes)

        for node, handler in nodes_and_handlers:
            if isinstance(node, LLMDecisionNode):
                result = handler({})
                assert result.data["content"] == "POOL EXTRACT"
        assert pool.calls == len(batch_nodes)

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
