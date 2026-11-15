//! DAG-based parallel executor.
//!
//! Port of `smartloop/agents/dag_executor.py`.
//!
//! Walks a [`DAG`], scheduling nodes whose dependencies are satisfied in
//! parallel via a shared project-bound thread pool.

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use indexmap::IndexMap;
use parking_lot::Mutex;
use serde_json::json;

use crate::agents::nodes::{AgentNode, NodeHandler, NodeResult, DAG};
use crate::utils::timing;
use crate::workspace::context::{PoolFuture, ProjectBoundExecutor};

/// Python logger name.
const LOG_TARGET: &str = "smartloop.agents.dag_executor";

/// Python `repr()` of a list of strings: `['a', 'b']`.
fn py_list_repr<'a>(items: impl IntoIterator<Item = &'a String>) -> String {
    let parts: Vec<String> = items.into_iter().map(|s| format!("'{}'", s)).collect();
    format!("[{}]", parts.join(", "))
}

/// Python `repr()` of a set of strings: `{'a', 'b'}`.
fn py_set_repr<'a>(items: impl IntoIterator<Item = &'a String>) -> String {
    let parts: Vec<String> = items.into_iter().map(|s| format!("'{}'", s)).collect();
    if parts.is_empty() {
        return "set()".to_string();
    }
    format!("{{{}}}", parts.join(", "))
}

/// Python `str(bool)`.
fn py_bool(b: bool) -> &'static str {
    if b { "True" } else { "False" }
}

/// The message of a panic payload (Python `str(exc)`).
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "handler panicked".to_string()
    }
}

/// What a submitted handler produced: its own result, or the panic it raised.
type HandlerOutcome = std::thread::Result<anyhow::Result<NodeResult>>;

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
        // Project-bound: nodes resolve documents, skills and MCP servers
        // against the project of the request that scheduled them, and a plain
        // pool would leave a worker on whichever project it last ran.
        Self {
            pool: ProjectBoundExecutor::new(max_workers, Some("dag")),
            llm_lock: Arc::new(Mutex::new(())),
        }
    }

    /// Serialize shared-model inference nodes; pass others through.
    fn wrap_handler(&self, node: &AgentNode, handler: NodeHandler) -> NodeHandler {
        if !node.is_llm_decision() {
            return handler;
        }
        let lock = Arc::clone(&self.llm_lock);
        Arc::new(move |deps| {
            let _guard = lock.lock();
            handler(deps)
        })
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
        while !dag.is_complete() {
            let ready = dag.ready_nodes();
            if ready.is_empty() {
                let remaining: Vec<String> =
                    dag.nodes.keys().filter(|k| !dag.results.contains_key(*k)).cloned().collect();
                tracing::error!(
                    target: LOG_TARGET,
                    "[dag] Deadlock — no ready nodes. Remaining: {}",
                    py_set_repr(remaining.iter())
                );
                for node_id in remaining {
                    dag.results.insert(
                        node_id.clone(),
                        NodeResult::failed(node_id, "Deadlock: unresolvable dependencies"),
                    );
                }
                break;
            }

            // Submit every ready node; completions are reported on `done_rx`
            // in the order they finish (Python `as_completed`).
            let (done_tx, done_rx) = mpsc::channel::<usize>();
            let mut submitted: Vec<(AgentNode, Instant, Option<PoolFuture<HandlerOutcome>>)> = Vec::new();
            for node in ready {
                let Some(handler) = handlers.get(&node.node_id).cloned() else {
                    dag.results.insert(
                        node.node_id.clone(),
                        NodeResult::failed(
                            node.node_id.clone(),
                            format!("No handler registered for '{}'", node.node_id),
                        ),
                    );
                    continue;
                };
                let dep_results: HashMap<String, NodeResult> = node
                    .depends_on
                    .iter()
                    .filter_map(|d| dag.results.get(d).map(|r| (d.clone(), r.clone())))
                    .collect();
                let wrapped = self.wrap_handler(&node, handler);
                let index = submitted.len();
                let tx = done_tx.clone();
                let node_id = node.node_id.clone();
                let queued = Instant::now();
                let future = self.pool.submit(move || {
                    let queue_ms = timing::ms(queued.elapsed());
                    let _t = timing::phase_with("dag.node", json!({"node": node_id, "queue_ms": queue_ms}));
                    let _caller = timing::caller(&format!("dag.{node_id}"));
                    let outcome: HandlerOutcome =
                        std::panic::catch_unwind(AssertUnwindSafe(|| wrapped(dep_results)));
                    let _ = tx.send(index);
                    outcome
                });
                submitted.push((node, Instant::now(), Some(future)));
            }
            drop(done_tx);

            let mut pending = submitted.len();
            while pending > 0 {
                let index = match done_rx.recv() {
                    Ok(index) => index,
                    // Every job is gone without reporting (the pool was shut
                    // down under us): collect what is left below.
                    Err(_) => break,
                };
                if let Some(future) = submitted[index].2.take() {
                    pending -= 1;
                    let (node, t0, _) = &submitted[index];
                    self.record(dag, handlers, node, *t0, future);
                }
            }
            for (node, t0, future) in submitted.iter_mut() {
                if let Some(future) = future.take() {
                    self.record(dag, handlers, node, *t0, future);
                }
            }
        }

        dag.results.clone()
    }

    /// Record one finished node's result and graft any nodes it adds.
    fn record(
        &self,
        dag: &mut DAG,
        handlers: &mut HashMap<String, NodeHandler>,
        node: &AgentNode,
        t0: Instant,
        future: PoolFuture<HandlerOutcome>,
    ) {
        let outcome: Result<NodeResult, String> = match future.result() {
            Ok(Ok(Ok(result))) => Ok(result),
            Ok(Ok(Err(e))) => Err(e.to_string()),
            Ok(Err(panic)) | Err(panic) => Err(panic_message(panic.as_ref())),
        };
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        match outcome {
            Ok(mut result) => {
                result.duration_ms = elapsed;
                tracing::info!(
                    target: LOG_TARGET,
                    "[dag] {} completed in {:.0}ms (success={})",
                    node.node_id,
                    result.duration_ms,
                    py_bool(result.success)
                );
                let extensions = result.extend_dag.clone();
                dag.results.insert(node.node_id.clone(), result);
                for ext in extensions {
                    let new_id = ext.node.node_id.clone();
                    let depends_on = py_list_repr(ext.node.depends_on.iter());
                    dag.add_node(ext.node);
                    handlers.insert(new_id.clone(), ext.handler);
                    tracing::info!(
                        target: LOG_TARGET,
                        "[dag] added node '{}' (depends_on={})",
                        new_id,
                        depends_on
                    );
                }
            }
            Err(message) => {
                dag.results.insert(
                    node.node_id.clone(),
                    NodeResult {
                        node_id: node.node_id.clone(),
                        success: false,
                        error: Some(message.clone()),
                        duration_ms: elapsed,
                        ..Default::default()
                    },
                );
                tracing::warn!(
                    target: LOG_TARGET,
                    "[dag] {} failed after {:.0}ms: {}",
                    node.node_id,
                    elapsed,
                    message
                );
            }
        }
    }

    /// Shut down the underlying thread pool.
    pub fn shutdown(&self, wait: bool) {
        self.pool.shutdown(wait);
    }
}
