"""The Step abstraction — one stage that contributes nodes to the DAG.

A step decides whether it applies to the request at hand, then builds the
node(s) and handler(s) that carry it out.  Building and running are separate
on purpose: the DAG is assembled up front from every step that applies, and
only then walked, so the executor can see the whole graph and run independent
branches in parallel.

Not to be confused with :class:`~smartloop.agents.planner.PlanStep`, which is
one *planned action* inside a request ("fetch this URL", "read that file").  A
``Step`` here is a stage of the orchestration itself — evaluating retrieved
context, reducing documents — and a single Step may build many DAG nodes.
"""
from abc import ABC, abstractmethod
from typing import Callable, ClassVar, List, Optional, Tuple

from pydantic import BaseModel, Field

from smartloop.agents.nodes import AgentNode
from smartloop.agents.mcp.explorer import SearchContext
from smartloop.agents.orchestration.context import _OrchestratorContext

# One DAG node and the handler that runs it.
NodePair = Tuple[AgentNode, Callable]


class StepNodes(BaseModel):
    """What a step contributed to the DAG.

    ``parent`` is the node downstream steps must depend on — usually the last
    node this step built, so the graph stays ordered.  A step that fans out
    returns its join/reduce node here rather than the fan.
    """

    model_config = {"arbitrary_types_allowed": True}

    pairs: List[NodePair] = Field(default_factory=list)
    parent: AgentNode

    @classmethod
    def none(cls, parent: AgentNode) -> "StepNodes":
        """Nothing contributed — the parent passes straight through."""
        return cls(parent=parent)


class Step(ABC):
    """A stage of the orchestration DAG."""

    #: Short name used in logs and progress events.
    name: ClassVar[str] = "step"

    def begin_request(self) -> None:
        """Reset per-request state.

        Steps are long-lived and shared across requests; anything scoped to a
        single request (a memo, a dedupe set) is cleared here.  Steps that hold
        no such state need not override this.
        """

    def applies_to(
        self,
        ctx: _OrchestratorContext,
        search_ctx: Optional[SearchContext],
    ) -> bool:
        """True when this step has work to do for this request.

        A step that does not apply is left out of the DAG entirely, rather than
        built and short-circuited at run time — an unbuilt node costs nothing
        and keeps the graph honest about what actually ran.
        """
        return True

    @abstractmethod
    def build(
        self,
        ctx: _OrchestratorContext,
        search_ctx: Optional[SearchContext],
        parent: AgentNode,
    ) -> StepNodes:
        """Build the node(s) and handler(s) this step contributes."""
