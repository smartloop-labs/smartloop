from .prompt_agent import PromptAgent, PromptResponse, DocumentInput
from .summarizer_agent import SummarizerAgent, SummarizerResponse


def __getattr__(name):
    if name == "Planner":
        from .planner import Planner
        return Planner
    if name == "MCPToolOrchestrator":
        from .orchestrator import SLPOrchestrator
        return SLPOrchestrator
    if name == "MCPExecutionResult":
        from .orchestrator import MCPExecutionResult
        return MCPExecutionResult
    if name == "Explorer":
        from .mcp.explore import Explorer
        return Explorer
    if name == "EnrichedContext":
        from .mcp.explore import EnrichedContext
        return EnrichedContext
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")