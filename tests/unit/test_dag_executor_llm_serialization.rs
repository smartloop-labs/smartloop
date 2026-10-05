//! Port of `tests/unit/test_dag_executor_llm_serialization.py`.
//!
//! DAGExecutor must serialize LLMDecisionNode instances (a single local
//! llama.cpp model is not thread-safe) while still running I/O / transform
//! nodes in parallel.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use smartloop::agents::dag_executor::DAGExecutor;
use smartloop::agents::nodes::{LLMDecisionNode, NodeHandler, NodeResult, SearchNode, TransformNode, DAG};

/// `(handler_factory, peak_getter)` tracking max simultaneous calls.
fn concurrency_probe() -> (impl Fn() -> NodeHandler, impl Fn() -> usize) {
    // (active, peak)
    let state = Arc::new(Mutex::new((0usize, 0usize)));
    let for_handlers = Arc::clone(&state);
    let make_handler = move || -> NodeHandler {
        let state = Arc::clone(&for_handlers);
        Arc::new(move |_deps| {
            {
                let mut s = state.lock();
                s.0 += 1;
                s.1 = s.1.max(s.0);
            }
            std::thread::sleep(Duration::from_millis(20));
            {
                let mut s = state.lock();
                s.0 -= 1;
            }
            Ok(NodeResult::ok("x"))
        })
    };
    let peak = move || state.lock().1;
    (make_handler, peak)
}

fn root_handler() -> NodeHandler {
    Arc::new(|_deps| Ok(NodeResult::ok("root")))
}

#[test]
fn test_llm_decision_nodes_are_serialized() {
    let mut dag = DAG::new();
    let root = TransformNode::with_id("root", &[]);
    dag.add_node(root.clone());

    let mut llm_nodes = Vec::new();
    for i in 0..4 {
        let n = LLMDecisionNode::with_id(format!("llm_{i}"), &[&root]);
        dag.add_node(n.clone());
        llm_nodes.push(n);
    }

    let (make_handler, peak) = concurrency_probe();
    let mut handlers: HashMap<String, NodeHandler> = HashMap::new();
    handlers.insert(root.node_id.clone(), root_handler());
    for n in &llm_nodes {
        handlers.insert(n.node_id.clone(), make_handler());
    }

    DAGExecutor::new(6).execute(&mut dag, &mut handlers);
    assert_eq!(peak(), 1, "LLM_DECISION nodes ran concurrently (peak={})", peak());
}

#[test]
fn test_io_nodes_still_run_in_parallel() {
    let mut dag = DAG::new();
    let root = TransformNode::with_id("root", &[]);
    dag.add_node(root.clone());

    let mut io_nodes = Vec::new();
    for i in 0..4 {
        let n = SearchNode::with_id(format!("io_{i}"), &[&root]);
        dag.add_node(n.clone());
        io_nodes.push(n);
    }

    let (make_handler, peak) = concurrency_probe();
    let mut handlers: HashMap<String, NodeHandler> = HashMap::new();
    handlers.insert(root.node_id.clone(), root_handler());
    for n in &io_nodes {
        handlers.insert(n.node_id.clone(), make_handler());
    }

    DAGExecutor::new(6).execute(&mut dag, &mut handlers);
    assert!(peak() > 1, "SEARCH nodes were unexpectedly serialized");
}
