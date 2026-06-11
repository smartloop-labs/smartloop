"""
DAG-based parallel executor.

Walks an DAG, scheduling nodes whose dependencies are satisfied
in parallel via a shared ThreadPoolExecutor.
"""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict

from smartloop.agents.nodes import DAG, LLMDecisionNode, NodeResult

logger = logging.getLogger(__name__)


class DAGExecutor:
    """Execute an :class:`DAG` with maximum parallelism.

    Nodes whose ``depends_on`` list is fully satisfied are submitted
    concurrently.  The executor loops until every node has a result.

    :class:`LLMDecisionNode` instances are serialized against a shared
    lock: a single local llama.cpp model instance is not thread-safe, so
    concurrent inference would crash the process.  All other node types
    run fully in parallel.
    """

    def __init__(self, max_workers: int = 6):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="dag")
        self._llm_lock = threading.Lock()

    def _wrap_handler(self, node, handler: Callable) -> Callable:
        """Serialize shared-model inference nodes; pass others through."""
        if not isinstance(node, LLMDecisionNode) or node.parallel_inference:
            return handler

        def locked(deps):
            with self._llm_lock:
                return handler(deps)

        return locked

    def execute(
        self,
        dag: DAG,
        handlers: Dict[str, Callable],
    ) -> Dict[str, NodeResult]:
        """Walk the DAG — nodes with satisfied deps run in parallel.

        Args:
            dag: The execution graph to process.
            handlers: Mapping of ``node_id`` → callable.  Each handler
                receives a dict of ``{dep_node_id: NodeResult}`` for its
                declared dependencies and must return a :class:`NodeResult`.

        Returns:
            The completed ``dag.results`` dict.
        """
        while not dag.is_complete():
            ready = dag.ready_nodes()
            if not ready:
                remaining = set(dag.nodes.keys()) - set(dag.results.keys())
                logger.error(
                    f"[dag] Deadlock — no ready nodes. Remaining: {remaining}"
                )
                for node_id in remaining:
                    dag.results[node_id] = NodeResult(
                        node_id=node_id, success=False,
                        error="Deadlock: unresolvable dependencies",
                    )
                break

            futures = {}
            submit_times = {}
            for node in ready:
                handler = handlers.get(node.node_id)
                if handler is None:
                    dag.results[node.node_id] = NodeResult(
                        node_id=node.node_id, success=False,
                        error=f"No handler registered for '{node.node_id}'",
                    )
                    continue
                dep_results = {d: dag.results[d] for d in node.depends_on}
                future = self._pool.submit(self._wrap_handler(node, handler), dep_results)
                futures[future] = node
                submit_times[future] = time.monotonic()

            for future in as_completed(futures):
                node = futures[future]
                t0 = submit_times[future]
                try:
                    result = future.result()
                    result.duration_ms = (time.monotonic() - t0) * 1000
                    dag.results[node.node_id] = result
                    logger.info(
                        f"[dag] {node.node_id} completed in "
                        f"{result.duration_ms:.0f}ms "
                        f"(success={result.success})"
                    )
                    if isinstance(result.data, dict):
                        for new_node, new_handler in result.data.get("_extend_dag", []):
                            dag.add_node(new_node)
                            handlers[new_node.node_id] = new_handler
                            logger.info(
                                f"[dag] added node '{new_node.node_id}' "
                                f"(depends_on={new_node.depends_on})"
                            )
                except Exception as exc:
                    elapsed = (time.monotonic() - t0) * 1000
                    dag.results[node.node_id] = NodeResult(
                        node_id=node.node_id,
                        success=False,
                        error=str(exc),
                        duration_ms=elapsed,
                    )
                    logger.warning(
                        f"[dag] {node.node_id} failed after {elapsed:.0f}ms: {exc}"
                    )

        return dag.results

    def shutdown(self, wait: bool = False) -> None:
        """Shut down the underlying thread pool."""
        self._pool.shutdown(wait=wait)
