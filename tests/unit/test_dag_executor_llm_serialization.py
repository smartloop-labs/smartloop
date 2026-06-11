"""
DAGExecutor must serialize LLMDecisionNode instances (a single local
llama.cpp model is not thread-safe) while still running I/O / transform
nodes in parallel.
"""
import threading
import time

from smartloop.agents.dag_executor import DAGExecutor
from smartloop.agents.nodes import (
    DAG, LLMDecisionNode, SearchNode, TransformNode, NodeResult,
)


def _concurrency_probe():
    """Return (handler_factory, peak_getter) tracking max simultaneous calls."""
    state = {"active": 0, "peak": 0}
    lock = threading.Lock()

    def make_handler():
        def handler(deps):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.02)
            with lock:
                state["active"] -= 1
            return NodeResult(node_id="x", success=True)
        return handler

    return make_handler, lambda: state["peak"]


def test_llm_decision_nodes_are_serialized():
    dag = DAG()
    root = TransformNode(node_id="root")
    dag.add_node(root)

    llm_nodes = []
    for i in range(4):
        n = LLMDecisionNode(node_id=f"llm_{i}", depends_on=[root])
        dag.add_node(n)
        llm_nodes.append(n)

    make_handler, peak = _concurrency_probe()
    handlers = {root.node_id: lambda deps: NodeResult(node_id="root", success=True)}
    for n in llm_nodes:
        handlers[n.node_id] = make_handler()

    DAGExecutor(max_workers=6).execute(dag, handlers)
    assert peak() == 1, f"LLM_DECISION nodes ran concurrently (peak={peak()})"


def test_io_nodes_still_run_in_parallel():
    dag = DAG()
    root = TransformNode(node_id="root")
    dag.add_node(root)

    io_nodes = []
    for i in range(4):
        n = SearchNode(node_id=f"io_{i}", depends_on=[root])
        dag.add_node(n)
        io_nodes.append(n)

    make_handler, peak = _concurrency_probe()
    handlers = {root.node_id: lambda deps: NodeResult(node_id="root", success=True)}
    for n in io_nodes:
        handlers[n.node_id] = make_handler()

    DAGExecutor(max_workers=6).execute(dag, handlers)
    assert peak() > 1, "SEARCH nodes were unexpectedly serialized"
