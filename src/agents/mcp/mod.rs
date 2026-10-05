//! Port of `smartloop/agents/mcp/__init__.py`.
//!
//! Re-exports `MCPServerSummarizer` plus the names the Python package exposed
//! lazily via `__getattr__` (`MCPToolOrchestrator` = `SLPOrchestrator`,
//! `MCPExecutionResult`, `StepType`, `PlanStep`, `ExecutionPlan`).

pub mod argument_extractor;
pub mod dependency_analyzer;
pub mod explorer;
pub mod file_ops;
pub mod mcp_agent;
pub mod mcp_tool_caller;
pub mod result_mapper;
pub mod server_selector;
pub mod server_summarizer;
pub mod servers;

pub use server_summarizer::MCPServerSummarizer;

pub use crate::agents::orchestration::context::MCPExecutionResult;
/// Backwards-compatible alias for the orchestrator.
pub use crate::agents::orchestrator::SLPOrchestrator as MCPToolOrchestrator;
pub use crate::agents::planner::{ExecutionPlan, PlanStep, StepType};
