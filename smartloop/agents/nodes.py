"""
Typed node and DAG models for the parallel execution engine.

Nodes represent units of work (search, tool call, LLM decision, transform).
An DAG tracks dependencies and results so the DAGExecutor can
schedule independent nodes in parallel.
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid

from pydantic import BaseModel, Field, field_validator


class AgentNodeName(str, Enum):
    ENRICH = "enrich"
    SKILL_MATCH = "skill_match"
    COLLECT_PARALLEL = "collect_parallel"
    ROUTE = "route"
    CONTEXT_GATE = "context_gate"
    REASON = "reason"


class DecisionType(str, Enum):
    AGGREGATION = "aggregation"
    CLASSIFICATION = "classification"


class NodeResult(BaseModel):
    node_id: str
    success: bool = True
    data: Any = None
    error: Optional[str] = None
    duration_ms: float = 0


class AgentNode(BaseModel):
    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    depends_on: List[str] = Field(default_factory=list)
    decision_type: Optional[DecisionType] = None

    @field_validator("depends_on", mode="before")
    @classmethod
    def _normalize_deps(cls, v: Any) -> List[str]:
        return [dep.node_id for dep in v]


# ── Category base classes (executor dispatches on these) ──

class LLMDecisionNode(AgentNode):
    """LLM-bound node — serialized by the executor."""
    pass


class TransformNode(AgentNode):
    """CPU-bound transform / merge node."""
    pass


class SearchNode(AgentNode):
    """I/O-bound search node (vector, workspace, feedback)."""
    pass


class ToolCallNode(AgentNode):
    """I/O-bound tool execution node (MCP calls, URL fetch)."""
    pass


# ── Fixed pipeline nodes ──

class EnrichNode(TransformNode):
    pass


class SkillMatchNode(LLMDecisionNode):
    pass


class CollectNode(SearchNode):
    pass


class RouteNode(LLMDecisionNode):
    pass


# ── Graph ──

class DAG(BaseModel):
    nodes: Dict[str, AgentNode] = Field(default_factory=dict)
    results: Dict[str, NodeResult] = Field(default_factory=dict)

    def add_node(self, node: AgentNode) -> None:
        self.nodes[node.node_id] = node

    def ready_nodes(self) -> List[AgentNode]:
        """Return nodes whose dependencies are all satisfied."""
        done = set(self.results.keys())
        return [
            n for n in self.nodes.values()
            if n.node_id not in done
            and all(d in done for d in n.depends_on)
        ]

    def is_complete(self) -> bool:
        """True when every node has a result."""
        return set(self.results.keys()) >= set(self.nodes.keys())
