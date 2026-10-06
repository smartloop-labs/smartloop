//! Typed node and DAG models for the parallel execution engine.
//!
//! Port of `smartloop/agents/nodes.py`.
//!
//! Nodes represent units of work (search, tool call, LLM decision, transform).
//! A [`DAG`] tracks dependencies and results so the
//! [`DAGExecutor`](crate::agents::dag_executor::DAGExecutor) can schedule
//! independent nodes in parallel.
//!
//! Python modelled the node categories as a class hierarchy and the executor
//! dispatched on `isinstance`.  Here every node is an [`AgentNode`] carrying a
//! [`NodeCategory`] (the category base class) and a [`NodeKind`] (the concrete
//! fixed-pipeline class, if any).  The Python class names survive as unit
//! structs ([`LLMDecisionNode`], [`SearchNode`], ...) whose `new`/`with_id`
//! constructors build an [`AgentNode`] of that category.

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Well-known node ids of the fixed pipeline.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AgentNodeName {
    #[serde(rename = "enrich")]
    Enrich,
    #[serde(rename = "skill_match")]
    SkillMatch,
    #[serde(rename = "collect_parallel")]
    CollectParallel,
    #[serde(rename = "route")]
    Route,
    #[serde(rename = "context_gate")]
    ContextGate,
    #[serde(rename = "reason")]
    Reason,
}

impl AgentNodeName {
    /// The string value (Python `str` enum value), used as a node id.
    pub fn as_str(&self) -> &'static str {
        match self {
            AgentNodeName::Enrich => "enrich",
            AgentNodeName::SkillMatch => "skill_match",
            AgentNodeName::CollectParallel => "collect_parallel",
            AgentNodeName::Route => "route",
            AgentNodeName::ContextGate => "context_gate",
            AgentNodeName::Reason => "reason",
        }
    }
}

impl fmt::Display for AgentNodeName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<AgentNodeName> for String {
    fn from(n: AgentNodeName) -> Self {
        n.as_str().to_string()
    }
}

/// What kind of LLM decision a node makes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DecisionType {
    #[serde(rename = "aggregation")]
    Aggregation,
    #[serde(rename = "classification")]
    Classification,
}

/// Handler for one node: receives `{dep_node_id: NodeResult}` for its declared
/// dependencies and returns its [`NodeResult`].  An `Err` is recorded as a
/// failed result (Python: an exception raised by the handler).
pub type NodeHandler =
    Arc<dyn Fn(HashMap<String, NodeResult>) -> anyhow::Result<NodeResult> + Send + Sync>;

/// One node (and its handler) a result asks the executor to add to the DAG
/// (Python: an entry of `data["_extend_dag"]`).
#[derive(Clone)]
pub struct DagExtension {
    pub node: AgentNode,
    pub handler: NodeHandler,
}

impl DagExtension {
    /// Pair a node with its handler.
    pub fn new(node: AgentNode, handler: NodeHandler) -> Self {
        Self { node, handler }
    }
}

impl fmt::Debug for DagExtension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DagExtension").field("node", &self.node).finish_non_exhaustive()
    }
}

impl PartialEq for DagExtension {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
    }
}

/// Typed, non-serializable payload a node can hand to its dependants
/// (Python stored arbitrary objects such as a `SearchContext` in `data`).
pub type NodePayload = Arc<dyn Any + Send + Sync>;

/// The outcome of running one node.
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct NodeResult {
    pub node_id: String,
    #[serde(default = "default_true")]
    pub success: bool,
    /// JSON-shaped result data (Python `data: Any`).
    #[serde(default)]
    pub data: Value,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub duration_ms: f64,
    /// Typed objects that do not fit in JSON (`search_ctx`, ...).
    #[serde(skip)]
    pub payload: Option<NodePayload>,
    /// Nodes to add to the DAG once this one completes (`_extend_dag`).
    #[serde(skip)]
    pub extend_dag: Vec<DagExtension>,
}

fn default_true() -> bool {
    true
}

impl NodeResult {
    /// A successful result with no data.
    pub fn ok(node_id: impl Into<String>) -> Self {
        Self { node_id: node_id.into(), success: true, ..Default::default() }
    }

    /// A successful result carrying `data`.
    pub fn with_data(node_id: impl Into<String>, data: Value) -> Self {
        Self { node_id: node_id.into(), success: true, data, ..Default::default() }
    }

    /// A failed result with `error`.
    pub fn failed(node_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            success: false,
            error: Some(error.into()),
            ..Default::default()
        }
    }

    /// Attach a typed payload.
    pub fn with_payload<T: Any + Send + Sync>(mut self, payload: T) -> Self {
        self.payload = Some(Arc::new(payload));
        self
    }

    /// Downcast the payload to `T`.
    pub fn payload_as<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.payload.as_ref().and_then(|p| p.downcast_ref::<T>())
    }
}

impl fmt::Debug for NodeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeResult")
            .field("node_id", &self.node_id)
            .field("success", &self.success)
            .field("data", &self.data)
            .field("error", &self.error)
            .field("duration_ms", &self.duration_ms)
            .field("payload", &self.payload.is_some())
            .field("extend_dag", &self.extend_dag)
            .finish()
    }
}

impl PartialEq for NodeResult {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id
            && self.success == other.success
            && self.data == other.data
            && self.error == other.error
            && self.duration_ms == other.duration_ms
            && self.extend_dag == other.extend_dag
    }
}

/// The category base class of a node; the executor dispatches on it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub enum NodeCategory {
    /// Plain `AgentNode`.
    #[default]
    Base,
    /// LLM-bound node — serialized by the executor.
    LLMDecision,
    /// CPU-bound transform / merge node.
    Transform,
    /// I/O-bound search node (vector, workspace, feedback).
    Search,
    /// I/O-bound tool execution node (MCP calls, URL fetch).
    ToolCall,
}

/// The concrete fixed-pipeline node class, if any.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub enum NodeKind {
    /// A plain node of its category.
    #[default]
    Generic,
    Enrich,
    SkillMatch,
    Collect,
    Route,
}

/// A unit of work in the DAG.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentNode {
    /// Unique id (a fresh UUID4 by default).
    pub node_id: String,
    /// Ids of the nodes this one waits on.
    #[serde(default)]
    pub depends_on: Vec<String>,
    #[serde(default)]
    pub decision_type: Option<DecisionType>,
    /// Category base class (Python class hierarchy).
    #[serde(skip)]
    pub category: NodeCategory,
    /// Concrete pipeline class.
    #[serde(skip)]
    pub kind: NodeKind,
}

impl Default for AgentNode {
    fn default() -> Self {
        Self::new(NodeCategory::Base, NodeKind::Generic, None, &[])
    }
}

impl AgentNode {
    /// Build a node.  `node_id` defaults to a fresh UUID4; `depends_on` is
    /// normalized to the dependencies' node ids.
    pub fn new(
        category: NodeCategory,
        kind: NodeKind,
        node_id: Option<String>,
        depends_on: &[&AgentNode],
    ) -> Self {
        Self {
            node_id: node_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            depends_on: depends_on.iter().map(|d| d.node_id.clone()).collect(),
            decision_type: None,
            category,
            kind,
        }
    }

    /// Set the decision type (builder style).
    pub fn with_decision_type(mut self, decision_type: DecisionType) -> Self {
        self.decision_type = Some(decision_type);
        self
    }

    /// `isinstance(node, LLMDecisionNode)`.
    pub fn is_llm_decision(&self) -> bool {
        self.category == NodeCategory::LLMDecision
    }

    /// `isinstance(node, TransformNode)`.
    pub fn is_transform(&self) -> bool {
        self.category == NodeCategory::Transform
    }

    /// `isinstance(node, SearchNode)`.
    pub fn is_search(&self) -> bool {
        self.category == NodeCategory::Search
    }

    /// `isinstance(node, ToolCallNode)`.
    pub fn is_tool_call(&self) -> bool {
        self.category == NodeCategory::ToolCall
    }
}

macro_rules! node_class {
    ($(#[$doc:meta])* $name:ident, $cat:expr, $kind:expr) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $name;

        impl $name {
            /// Build a node of this class with a fresh UUID id.
            pub fn new(depends_on: &[&AgentNode]) -> AgentNode {
                AgentNode::new($cat, $kind, None, depends_on)
            }

            /// Build a node of this class with an explicit id.
            pub fn with_id(node_id: impl Into<String>, depends_on: &[&AgentNode]) -> AgentNode {
                AgentNode::new($cat, $kind, Some(node_id.into()), depends_on)
            }
        }
    };
}

// ── Category base classes (executor dispatches on these) ──

node_class!(
    /// LLM-bound node — serialized by the executor.
    LLMDecisionNode, NodeCategory::LLMDecision, NodeKind::Generic
);
node_class!(
    /// CPU-bound transform / merge node.
    TransformNode, NodeCategory::Transform, NodeKind::Generic
);
node_class!(
    /// I/O-bound search node (vector, workspace, feedback).
    SearchNode, NodeCategory::Search, NodeKind::Generic
);
node_class!(
    /// I/O-bound tool execution node (MCP calls, URL fetch).
    ToolCallNode, NodeCategory::ToolCall, NodeKind::Generic
);

// ── Fixed pipeline nodes ──

node_class!(
    /// Enrichment transform.
    EnrichNode, NodeCategory::Transform, NodeKind::Enrich
);
node_class!(
    /// Skill matching (LLM decision).
    SkillMatchNode, NodeCategory::LLMDecision, NodeKind::SkillMatch
);
node_class!(
    /// Parallel source collection (search).
    CollectNode, NodeCategory::Search, NodeKind::Collect
);
node_class!(
    /// Routing (LLM decision).
    RouteNode, NodeCategory::LLMDecision, NodeKind::Route
);

// ── Graph ──

/// Execution graph: nodes plus the results gathered so far.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DAG {
    #[serde(default)]
    pub nodes: IndexMap<String, AgentNode>,
    #[serde(default)]
    pub results: IndexMap<String, NodeResult>,
}

impl DAG {
    /// Empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add (or replace) a node keyed by its id.
    pub fn add_node(&mut self, node: AgentNode) {
        self.nodes.insert(node.node_id.clone(), node);
    }

    /// Return nodes whose dependencies are all satisfied.
    pub fn ready_nodes(&self) -> Vec<AgentNode> {
        let done: HashSet<&String> = self.results.keys().collect();
        self.nodes
            .values()
            .filter(|n| !done.contains(&n.node_id) && n.depends_on.iter().all(|d| done.contains(d)))
            .cloned()
            .collect()
    }

    /// True when every node has a result.
    pub fn is_complete(&self) -> bool {
        self.nodes.keys().all(|k| self.results.contains_key(k))
    }
}
