//! LLMExecutor — thread pool for the orchestrator's I/O-bound work.
//!
//! Port of `smartloop/agents/hybrid_executor.py`.
//!
//! Dispatches search, MCP calls, and other I/O to a thread pool.  LLM
//! inference runs on the shared in-process model, serialized by
//! [`DAGExecutor`](crate::agents::dag_executor::DAGExecutor)'s lock, because a
//! single local llama.cpp instance is not thread-safe.
//!
//! This used to also own a process pool of spawn workers, each holding its own
//! copy of the model, for parallel inference.  That path was unreachable:
//! worker copies cap at one per GPU (same-device copies serialize in the
//! driver, and a Vulkan/Metal backend isn't enumerable), and `n_llm_workers`
//! defaulted to `n_parallel` = 1 — so the pool was always skipped in favour of
//! reusing the already-resident shared model.  Concurrency comes from
//! llama.cpp `n_parallel` continuous batching within that one instance
//! instead.

use crate::workspace::context::{PoolFuture, ProjectBoundExecutor};

/// Thread pool for the orchestrator's I/O-bound work.
pub struct LLMExecutor {
    io_pool: ProjectBoundExecutor,
}

impl Default for LLMExecutor {
    fn default() -> Self {
        Self::new(6)
    }
}

impl LLMExecutor {
    /// Build a pool of `max_io_workers` threads (Python default 6).
    /// Project-bound: the searches and MCP calls dispatched here resolve
    /// against the requesting project (see [`crate::workspace::context`]).
    pub fn new(max_io_workers: usize) -> Self {
        Self { io_pool: ProjectBoundExecutor::new(max_io_workers, Some("io")) }
    }

    /// The underlying I/O pool (Python `_io_pool`), handed to the explorer's
    /// parallel collect.
    pub fn io_pool(&self) -> &ProjectBoundExecutor {
        &self.io_pool
    }

    // -- Public dispatch API ------------------------------------------------

    /// Submit I/O-bound work (search, MCP calls) to the thread pool.
    pub fn submit_io<F, R>(&self, f: F) -> PoolFuture<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.io_pool.submit(f)
    }

    // -- Lifecycle ----------------------------------------------------------

    /// Shut down the thread pool.
    pub fn shutdown(&self, wait: bool) {
        self.io_pool.shutdown(wait);
    }
}
