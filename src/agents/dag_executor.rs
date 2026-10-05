//! DAG-based parallel executor.
//!
//! Port of `smartloop/agents/dag_executor.py`.
//!
//! Walks a [`DAG`], scheduling nodes whose dependencies are satisfied in
//! parallel via a shared project-bound thread pool.

use std::collections::HashMap;
use std::sync::Arc;

use indexmap::IndexMap;
use parking_lot::Mutex;

use crate::agents::nodes::{AgentNode, NodeHandler, NodeResult, DAG};
use crate::workspace::context::ProjectBoundExecutor;

/// Execute a [`DAG`] with maximum parallelism.
///
/// Nodes whose `depends_on` list is fully satisfied are submitted
/// concurrently.  The executor loops until every node has a result.
///
/// LLM-decision nodes are serialized against a shared lock: a single local
/// llama.cpp model instance is not thread-safe, so concurrent inference would
/// crash the process.  All other node types run fully in parallel.
pub struct DAGExecutor {
    pool: ProjectBoundExecutor,
    llm_lock: Arc<Mutex<()>>,
}

impl Default for DAGExecutor {
    fn default() -> Self {
        Self::new(6)
    }
}

impl DAGExecutor {
    /// Build an executor with `max_workers` threads (Python default 6).
    pub fn new(max_workers: usize) -> Self {
        let _ = max_workers;
        todo!()
    }

    /// Serialize shared-model inference nodes; pass others through.
    fn wrap_handler(&self, node: &AgentNode, handler: NodeHandler) -> NodeHandler {
        let _ = (node, handler, &self.llm_lock, &self.pool);
        todo!()
    }

    /// Walk the DAG — nodes with satisfied deps run in parallel.
    ///
    /// `handlers` maps `node_id` → handler; each receives the results of its
    /// declared dependencies.  Results carrying `extend_dag` entries add those
    /// nodes (and handlers) to the graph.  Returns the completed
    /// `dag.results`.
    pub fn execute(
        &self,
        dag: &mut DAG,
        handlers: &mut HashMap<String, NodeHandler>,
    ) -> IndexMap<String, NodeResult> {
        let _ = (dag, handlers);
        let _ = Self::wrap_handler;
        todo!()
    }

    /// Shut down the underlying thread pool.
    pub fn shutdown(&self, wait: bool) {
        let _ = wait;
        todo!()
    }
}
