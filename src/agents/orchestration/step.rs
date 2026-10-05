//! Port of `smartloop/agents/orchestration/step.py`.
//!
//! The Step abstraction — one stage that contributes nodes to the DAG.
//!
//! A step decides whether it applies to the request at hand, then builds the
//! node(s) and handler(s) that carry it out.  Building and running are
//! separate on purpose: the DAG is assembled up front from every step that
//! applies, and only then walked, so the executor can see the whole graph and
//! run independent branches in parallel.
//!
//! Not to be confused with [`crate::agents::planner::PlanStep`], which is one
//! *planned action* inside a request ("fetch this URL", "read that file").  A
//! `Step` here is a stage of the orchestration itself — evaluating retrieved
//! context, reducing documents — and a single Step may build many DAG nodes.

use crate::agents::mcp::explorer::SearchContext;
use crate::agents::nodes::{AgentNode, NodeHandler};
use crate::agents::orchestration::context::{OrchestratorContext, SharedOrchestratorContext};

/// One DAG node and the handler that runs it.
pub type NodePair = (AgentNode, NodeHandler);

/// What a step contributed to the DAG.
///
/// `parent` is the node downstream steps must depend on — usually the last
/// node this step built, so the graph stays ordered.  A step that fans out
/// returns its join/reduce node here rather than the fan.
#[derive(Clone)]
pub struct StepNodes {
    pub pairs: Vec<NodePair>,
    pub parent: AgentNode,
}

impl StepNodes {
    /// Nodes plus the parent downstream steps depend on.
    pub fn new(pairs: Vec<NodePair>, parent: AgentNode) -> Self {
        Self { pairs, parent }
    }

    /// Nothing contributed — the parent passes straight through.
    pub fn none(parent: AgentNode) -> Self {
        Self { pairs: Vec::new(), parent }
    }
}

impl std::fmt::Debug for StepNodes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StepNodes")
            .field("pairs", &self.pairs.iter().map(|(n, _)| n).collect::<Vec<_>>())
            .field("parent", &self.parent)
            .finish()
    }
}

/// A stage of the orchestration DAG.
pub trait Step: Send + Sync {
    /// Short name used in logs and progress events.
    fn name(&self) -> &'static str {
        "step"
    }

    /// Reset per-request state.
    ///
    /// Steps are long-lived and shared across requests; anything scoped to a
    /// single request (a memo, a dedupe set) is cleared here.  Steps that hold
    /// no such state need not override this.
    fn begin_request(&self) {}

    /// True when this step has work to do for this request.
    ///
    /// A step that does not apply is left out of the DAG entirely, rather
    /// than built and short-circuited at run time — an unbuilt node costs
    /// nothing and keeps the graph honest about what actually ran.
    fn applies_to(&self, _ctx: &OrchestratorContext, _search_ctx: Option<&SearchContext>) -> bool {
        true
    }

    /// Build the node(s) and handler(s) this step contributes.
    fn build(
        &self,
        ctx: &SharedOrchestratorContext,
        search_ctx: Option<&SearchContext>,
        parent: &AgentNode,
    ) -> StepNodes;
}
