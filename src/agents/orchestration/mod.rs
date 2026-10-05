//! Port of `smartloop/agents/orchestration/__init__.py`.
//!
//! Orchestration components.
//!
//! [`crate::agents::orchestrator::SLPOrchestrator`] is a coordinator: it
//! builds the request context, walks the DAG, and hands the accumulated
//! results to the response builder.  Everything it coordinates lives here,
//! one concern per module.
//!
//! Two kinds of thing live in this package.  A **Step** ([`step`])
//! contributes node(s) to the DAG and knows whether it applies to a given
//! request; everything else is a **service** a step or handler calls into.
//! The dependency graph is acyclic and shallow — components depend downward,
//! and nothing imports the orchestrator back:
//!
//! ```text
//!     ModelRouter, ResponseBuilder          (no dependencies)
//!       <- JudgeStep*, DocumentFanoutStep*, SkillProcessor, StepExecutor
//!         <- ToolFanout
//!           <- PipelineBuilder
//!             <- SLPOrchestrator
//!
//!     * a Step; the rest are services.
//! ```

pub mod context;
pub mod document_fanout;
pub mod judge_step;
pub mod model_router;
pub mod pipeline_builder;
pub mod replies;
pub mod response_builder;
pub mod skill_processor;
pub mod step;
pub mod step_executor;
pub mod tool_fanout;

pub use context::{MCPExecutionResult, OrchestratorContext, SharedOrchestratorContext};
pub use step::{NodePair, Step, StepNodes};
