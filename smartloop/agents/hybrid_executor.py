"""LLMExecutor — thread pool for the orchestrator's I/O-bound work.

Dispatches search, MCP calls, and other I/O to a thread pool.  LLM
inference runs on the shared in-process model, serialized by
:class:`~smartloop.agents.dag_executor.DAGExecutor`'s lock, because a
single local llama.cpp instance is not thread-safe.

This used to also own a ``ProcessPoolExecutor`` of spawn workers, each
holding its own copy of the model, for parallel inference.  That path was
unreachable: worker copies cap at one per GPU (same-device copies
serialize in the driver, and a Vulkan/Metal backend isn't enumerable), and
``n_llm_workers`` defaulted to ``n_parallel`` = 1 — so ``effective_workers``
was always <= 1 and the pool was always skipped in favour of reusing the
already-resident shared model.  Concurrency comes from llama.cpp
``n_parallel`` continuous batching within that one instance instead.
"""
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class LLMExecutor:
    """Thread pool for the orchestrator's I/O-bound work.

    Parameters
    ----------
    max_io_workers:
        Size of the thread pool used for search, MCP calls, and other
        I/O-bound work.
    """

    def __init__(self, max_io_workers: int = 6):
        self._io_pool = ThreadPoolExecutor(
            max_workers=max_io_workers, thread_name_prefix="io"
        )

    # -- Public dispatch API ------------------------------------------------

    def submit_io(self, fn, *args, **kwargs):
        """Submit I/O-bound work (search, MCP calls) to the thread pool."""
        return self._io_pool.submit(fn, *args, **kwargs)

    # -- Lifecycle ----------------------------------------------------------

    def shutdown(self, wait: bool = False) -> None:
        """Shut down the thread pool."""
        self._io_pool.shutdown(wait=wait)
