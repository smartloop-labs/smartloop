"""Per-request context and result types shared across orchestration components.

These live in their own module so every component can import them without
importing the orchestrator itself — the orchestrator depends on all of the
components, so any component importing it back would be a cycle.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class _OrchestratorContext(BaseModel):
    """Shared parameters passed to DAG handler methods."""
    model_config = {"arbitrary_types_allowed": True}

    user_message: str
    mcp_servers: list
    available_servers: list
    effective_cwd: str
    skills: list = Field(default_factory=list)
    # Skill the user explicitly asked for (e.g. the API resolved a leading
    # ``/fund-info`` slash-command to a skill name). When set, the request is
    # fulfilled with this skill directly, skipping the LLM skill-match guess.
    requested_skill: Optional[str] = None
    document_context: Optional[str] = None
    project_dir: Optional[str] = None
    model_service: Any = None
    conversation_history: Optional[list] = None
    has_attachments: bool = False
    # Built once during context construction:
    enriched: Optional[Any] = None
    # Populated by handlers as they run:
    mcp_tool_contexts: list = Field(default_factory=list)
    parsed_skills: list = Field(default_factory=list)
    # Stripped skill body text (set by route handler when skills match).
    # ``skill_context`` is the joined form (used downstream for instruction
    # enrichment); ``skill_bodies`` keeps each matched skill separate so the
    # planner can plan across multiple matched skills.
    skill_context: Optional[str] = None
    skill_bodies: List[str] = Field(default_factory=list)
    available_capabilities: List[str] = Field(default_factory=list)

    # ── Per-request scratch state ──
    #
    # Written by one component and read by another (an ANALYZE step writes
    # ``skill_report``, the response builder reads it), so it belongs on the
    # object already threaded through every handler rather than on whichever
    # component happens to run first.

    # The client's wall-clock date ("Month DD, YYYY"), used to ground
    # date-aware prompts in the user's clock instead of the server's.
    current_date: Optional[str] = None
    # An ANALYZE step's findings, surfaced ahead of raw context in the answer.
    skill_report: Optional[str] = None
    # Citations from the documents the explore step actually drew on.
    explore_citations: Optional[List[Dict[str, Any]]] = None


class MCPExecutionResult(BaseModel):
    """Result of MCP tool orchestration."""
    handled: bool = Field(description="Whether the message was handled by an MCP tool")
    content: Optional[str] = Field(default=None, description="Formatted result content")
    skill_instruction: Optional[str] = Field(default=None, description="Matched skill instruction to follow")
    tool_name: Optional[str] = Field(default=None, description="Name of the tool that was executed")
    server_name: Optional[str] = Field(default=None, description="Name of the MCP server used")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    tool_names: List[str] = Field(default_factory=list, description="All tools executed in the chain")
    citations: Optional[List[Dict[str, Any]]] = Field(default=None, description="Source documents cited")
    mutation: bool = Field(default=False, description="Whether this was a mutation (write) operation — skip LLM, return content directly")
