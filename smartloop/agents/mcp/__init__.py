from .server_summarizer import MCPServerSummarizer


def __getattr__(name):
    if name in ("MCPToolOrchestrator", "MCPExecutionResult"):
        from smartloop.agents.orchestrator import SLPOrchestrator, MCPExecutionResult
        return {"MCPToolOrchestrator": SLPOrchestrator, "MCPExecutionResult": MCPExecutionResult}[name]
    if name in ("StepType", "PlanStep", "ExecutionPlan"):
        from smartloop.agents.planner import StepType, PlanStep, ExecutionPlan
        return {"StepType": StepType, "PlanStep": PlanStep, "ExecutionPlan": ExecutionPlan}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
