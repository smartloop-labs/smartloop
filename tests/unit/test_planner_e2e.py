"""End-to-end tests for the classify → plan → execute pipeline.

Each test exercises ``MCPToolOrchestrator.execute()`` with realistic
scenarios.  The LLM and external MCP calls are mocked; the built-in
action handlers (file-read, file-write, explore) run against real temp
files so the full dispatch path is verified.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from smartloop.agents.orchestrator import SLPOrchestrator, MCPExecutionResult
from smartloop.agents.mcp.explorer import SearchContext
from smartloop.agents.planner import StepType, PlanStep, ExecutionPlan
from smartloop.workspace.workspace import (
    MCPServer,
    MCPServerType,
    MCPToolDefinition,
    RemoteMCPServerConfig,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

RESUME_PDF_TEXT = (
    "# John Doe — Software Engineer\n\n"
    "## Experience\n"
    "- 5 years Python, FastAPI, Django\n"
    "- 3 years React, TypeScript\n"
    "- Cloud: AWS (EC2, Lambda, S3)\n"
    "- ML: scikit-learn, PyTorch basics\n\n"
    "## Education\n"
    "B.S. Computer Science, MIT 2018\n"
)

JOB_DESCRIPTION_HTML = (
    "# Senior Backend Engineer — Anthropic\n\n"
    "## Requirements\n"
    "- 5+ years backend development (Python preferred)\n"
    "- Experience with large-scale distributed systems\n"
    "- Strong understanding of ML/AI concepts\n"
    "- Experience with cloud infrastructure (AWS/GCP)\n"
    "- Excellent communication skills\n\n"
    "## Nice to Have\n"
    "- Rust or Go experience\n"
    "- Contributions to open-source projects\n"
)

GENERATED_REQUIREMENTS = (
    "# Tailored Requirements for John Doe\n\n"
    "## Strong Matches\n"
    "- 5 years Python experience (meets 5+ years requirement)\n"
    "- AWS cloud infrastructure experience\n"
    "- ML basics with scikit-learn and PyTorch\n\n"
    "## Gaps to Address\n"
    "- No distributed systems experience mentioned\n"
    "- No Rust or Go experience\n"
    "- Should highlight open-source contributions if any\n"
)


def _make_fetch_server() -> MCPServer:
    """Create a mock MCP server with a fetch tool."""
    return MCPServer(
        id="fetch-srv",
        name="Fetch MCP Server",
        server_type=MCPServerType.REMOTE,
        remote_config=RemoteMCPServerConfig(server_url="https://mcp.example.com"),
        enabled=True,
        tools=[
            MCPToolDefinition(
                name="fetch",
                description="Fetch a URL and return its content as markdown",
                input_schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                    },
                    "required": ["url"],
                },
            ),
        ],
    )


@dataclass
class FakeToolResult:
    """Mimics the result returned by MCPClient.call_tool()."""
    success: bool
    content: Optional[List[Dict[str, Any]]]
    is_error: bool = False
    error: Optional[str] = None


def _make_llm_side_effect(read_args: dict, write_args: dict, generated_text: str, plan_json: str = "[]"):
    """Build a side_effect function for ``llm.invoke()`` that returns
    different responses depending on which prompt is being served.

    In the new classify → plan flow, the LLM is called for:
    - Classification (Tier 2 fallback, if deterministic detection finds nothing)
    - Multi-step planning (only for MULTI_STEP intent)
    - Arg extraction for built-in tools
    - Conversational / summarize steps
    """
    def side_effect(prompt, *args, **kwargs):
        prompt_str = str(prompt)

        # Classification prompt (contains "Classify the user's intent")
        if "classify the user" in prompt_str.lower():
            return "MULTI_STEP"

        # Planner prompt (contains "task planner" or "JSON array")
        if "task planner" in prompt_str.lower() or "json array" in prompt_str.lower():
            return plan_json

        # Arg extraction prompts (contain "Reply with ONLY a JSON object")
        if "reply with only a json object" in prompt_str.lower():
            if "file-write" in prompt_str.lower() or '"content"' in prompt_str.lower():
                return json.dumps(write_args)
            return json.dumps(read_args)

        # Conversational / summarize / enrich — return generated text
        return generated_text

    return side_effect


# ------------------------------------------------------------------ #
# E2E Tests                                                            #
# ------------------------------------------------------------------ #

class TestClassifyPlanExecutePipeline:
    """Full pipeline: classify → plan → execute → return results."""

    @pytest.fixture
    def tmp_workspace(self, tmp_path):
        resume = tmp_path / "resume.txt"
        resume.write_text(RESUME_PDF_TEXT)
        return tmp_path

    @pytest.fixture
    def fetch_server(self):
        return _make_fetch_server()

    def test_multi_step_read_fetch_write(self, tmp_workspace, fetch_server):
        """File path + URL + write → MULTI_STEP → read + fetch + write steps."""
        resume_path = str(tmp_workspace / "resume.txt")
        output_path = str(tmp_workspace / "requirements.md")

        plan_response = json.dumps([
            {
                "step": 1,
                "step_type": "action",
                "action_kind": "read",
                "tool_name": "file-read",
                "server_id": "__system_tools__",
                "server_name": "smartloop",
                "instruction": f"Read the resume file at {resume_path}",
            },
            {
                "step": 2,
                "step_type": "tool",
                "tool_name": "fetch",
                "server_id": "fetch-srv",
                "server_name": "Fetch MCP Server",
                "instruction": "Fetch https://job-boards.greenhouse.io/anthropic/jobs/123 and extract the job description.",
            },
            {
                "step": 3,
                "step_type": "action",
                "action_kind": "write",
                "tool_name": "file-write",
                "server_id": "__system_tools__",
                "server_name": "smartloop",
                "instruction": f"Write the tailored requirements to {output_path}",
            },
        ])

        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=4096)
        mock_llm.invoke = MagicMock(side_effect=_make_llm_side_effect(
            plan_json=plan_response,
            read_args={"path": resume_path, "query": ""},
            write_args={"path": output_path, "content": GENERATED_REQUIREMENTS},
            generated_text=GENERATED_REQUIREMENTS,
        ))

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        fake_fetch_result = FakeToolResult(
            success=True,
            content=[{"type": "text", "text": JOB_DESCRIPTION_HTML}],
        )

        with patch.object(orchestrator.tool_caller, "call_tool", return_value=fake_fetch_result):
            result = orchestrator.execute(
                user_message=(
                    f"Read my resume at {resume_path}, "
                    "look at the job description at https://job-boards.greenhouse.io/anthropic/jobs/123, "
                    f"and write out the tailored requirements to {output_path}"
                ),
                mcp_servers=[fetch_server],
                cwd=str(tmp_workspace),
            )

        assert result.handled is True
        assert result.content is not None
        assert len(result.tool_names) >= 2

        # Verify the output file was written
        assert os.path.isfile(output_path)
        written = open(output_path).read()
        assert "Tailored Requirements" in written

    def test_single_tool_call_fetch(self, tmp_workspace, fetch_server):
        """URL only → EXPLORE (explore handles fetch internally)."""
        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=4096)
        mock_llm.invoke = MagicMock(return_value="Here is a summary of the job posting with key details extracted.")

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        fake_fetch_result = FakeToolResult(
            success=True,
            content=[{"type": "text", "text": JOB_DESCRIPTION_HTML}],
        )

        job_ctx = SearchContext()
        job_ctx.vector_context = JOB_DESCRIPTION_HTML

        with patch.object(orchestrator.explorer, "collect_parallel", return_value=job_ctx), \
             patch.object(orchestrator.tool_caller, "call_tool", return_value=fake_fetch_result):
            result = orchestrator.execute(
                user_message="Fetch https://example.com/job and summarize",
                mcp_servers=[fetch_server],
                cwd=str(tmp_workspace),
            )

        assert result.handled is True
        assert "explore" in result.tool_names

    def test_single_action_read(self, tmp_workspace):
        """File path → ACTION(read) → single read step (deterministic)."""
        resume_path = str(tmp_workspace / "resume.txt")

        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=4096)
        mock_llm.invoke = MagicMock(side_effect=_make_llm_side_effect(
            read_args={"path": resume_path, "query": ""},
            write_args={},
            generated_text="Key skills: Python, FastAPI.",
        ))

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        result = orchestrator.execute(
            user_message=f"Read {resume_path} and summarize the key skills",
            mcp_servers=[],
            skills=["Be concise"],
            cwd=str(tmp_workspace),
        )

        assert result.handled is True
        assert "file-read" in result.tool_names

    def test_fetch_failure_returns_partial(self, tmp_workspace, fetch_server):
        """When fetch fails, pipeline returns partial result (read step)."""
        resume_path = str(tmp_workspace / "resume.txt")

        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=4096)
        mock_llm.invoke = MagicMock(side_effect=_make_llm_side_effect(
            read_args={"path": resume_path, "query": ""},
            write_args={},
            generated_text="fallback",
        ))

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        fake_fetch_result = FakeToolResult(
            success=False, content=None, error="Connection timeout",
        )

        with patch.object(orchestrator.tool_caller, "call_tool", return_value=fake_fetch_result):
            result = orchestrator.execute(
                user_message=f"Read {resume_path} and fetch https://example.com/job",
                mcp_servers=[fetch_server],
                cwd=str(tmp_workspace),
            )

        # Should return partial result (just the file-read step)
        assert result.handled is True
        assert "file-read" in result.tool_names
        assert "John Doe" in result.content

    def test_no_servers_no_skills_returns_unhandled(self):
        """With no MCP servers and no skills, execute() returns unhandled."""
        mock_llm = MagicMock()
        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        result = orchestrator.execute(
            user_message="Read my resume and fetch a job description",
            mcp_servers=[],
        )
        assert result.handled is False

    def test_conversational_returns_unhandled(self, fetch_server):
        """A pure greeting → CONVERSATION → handled=False."""
        mock_llm = MagicMock()
        mock_llm.get_max_tokens = MagicMock(return_value=4096)
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.invoke = MagicMock(return_value="CONVERSATION")

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        result = orchestrator.execute(
            user_message="Hello, how are you?",
            mcp_servers=[fetch_server],
        )
        assert result.handled is False

    def test_document_context_with_tool(self, tmp_workspace, fetch_server):
        """Document context is passed through and available in results."""
        doc_context = "Internal policy: all candidates must have 3+ years Python."

        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=4096)
        mock_llm.invoke = MagicMock(return_value="Summary based on context.")

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        fake_fetch_result = FakeToolResult(
            success=True,
            content=[{"type": "text", "text": JOB_DESCRIPTION_HTML}],
        )

        doc_ctx = SearchContext()
        doc_ctx.vector_context = doc_context

        with patch.object(orchestrator.explorer, "collect_parallel", return_value=doc_ctx), \
             patch.object(orchestrator.tool_caller, "call_tool", return_value=fake_fetch_result):
            result = orchestrator.execute(
                user_message="Check the job at https://example.com/job",
                mcp_servers=[fetch_server],
                cwd=str(tmp_workspace),
                document_context=doc_context,
            )

        assert result.handled is True
        assert "explore" in result.tool_names
