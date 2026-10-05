//! Port of `smartloop/agents/__init__.py`.

pub mod agent;
pub mod dag_executor;
pub mod fetch_agent;
pub mod hybrid_executor;
pub mod mcp;
pub mod nodes;
pub mod orchestration;
pub mod orchestrator;
pub mod planner;
pub mod prompt_agent;
pub mod sanitize;
pub mod summarizer_agent;
pub mod web_search_agent;

pub use prompt_agent::{DocumentInput, PromptAgent, PromptResponse};
pub use summarizer_agent::{SummarizerAgent, SummarizerResponse};
pub use planner::Planner;
pub use orchestrator::SLPOrchestrator;
/// Python alias `MCPToolOrchestrator` → [`SLPOrchestrator`].
pub use orchestrator::SLPOrchestrator as MCPToolOrchestrator;
pub use orchestration::context::MCPExecutionResult;
pub use mcp::explorer::{EnrichedContext, Explorer};
