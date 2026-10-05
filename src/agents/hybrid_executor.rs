//! LLMExecutor — thread pool for the orchestrator's I/O-bound work.
//!
//! Port of `smartloop/agents/hybrid_executor.py`.
//!
//! Dispatches search, MCP calls, and other I/O to a thread pool.  LLM
//! inference runs on the shared in-process model, serialized by
//! [`DAGExecutor`](crate::agents::dag_executor::DAGExecutor)'s lock, because a
//! single local llama.cpp instance is not thread-safe.  Concurrency comes from
//! llama.cpp `n_parallel` continuous batching within that one instance.

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
    /// Project-bound: the work dispatched here resolves against the
    /// requesting project.
    pub fn new(max_io_workers: usize) -> Self {
        let _ = max_io_workers;
        todo!()
    }

    /// Submit I/O-bound work (search, MCP calls) to the thread pool.
    pub fn submit_io<F, R>(&self, f: F) -> PoolFuture<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let _ = (&self.io_pool, f);
        todo!()
    }

    /// Shut down the thread pool.
    pub fn shutdown(&self, wait: bool) {
        let _ = wait;
        todo!()
    }
}
