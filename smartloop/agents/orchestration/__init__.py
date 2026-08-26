"""Orchestration components.

:class:`~smartloop.agents.orchestrator.SLPOrchestrator` is a coordinator: it
builds the request context, walks the DAG, and hands the accumulated results
to the response builder.  Everything it coordinates lives here, one concern
per module.

Two kinds of thing live in this package.  A **Step** (:mod:`step`) contributes
node(s) to the DAG and knows whether it applies to a given request; everything
else is a **service** a step or handler calls into.  The dependency graph is
acyclic and shallow — components depend downward, and nothing imports the
orchestrator back:

    ModelRouter, ResponseBuilder          (no dependencies)
      ← JudgeStep*, DocumentFanoutStep*, SkillProcessor, StepExecutor
        ← ToolFanout
          ← PipelineBuilder
            ← SLPOrchestrator

    * a Step; the rest are services.
"""
from smartloop.agents.orchestration.context import (
    _OrchestratorContext, MCPExecutionResult,
)
from smartloop.agents.orchestration.step import Step, StepNodes

__all__ = ["_OrchestratorContext", "MCPExecutionResult", "Step", "StepNodes"]
