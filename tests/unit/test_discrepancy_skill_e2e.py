"""End-to-end test: multi-step skill — explore docs → conditional tool call.

Simulates the discrepancy-check skill:
  1. Explore invoice, receipt, and order documents to build a report
  2. If discrepancies found → call save_issue on Linear MCP

Validates that:
  - The skill pipeline (explore → condition → tool) executes in the correct
    order through the LLMExecutor.
  - The condition LLM call correctly gates the save_issue step.
  - Tool arguments (title, description, team) are extracted from the report.
  - When no discrepancies exist, save_issue is skipped.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from smartloop.agents.orchestrator import SLPOrchestrator, MCPExecutionResult
from smartloop.agents.planner import MCPToolContext
from smartloop.workspace.workspace import (
    MCPServer,
    MCPServerType,
    MCPToolDefinition,
    RemoteMCPServerConfig,
)


# ------------------------------------------------------------------ #
# Skill content                                                        #
# ------------------------------------------------------------------ #

EXPLORE_RAW_CONTENT = (
    "=== Document: invoice.pdf ===\n"
    "Invoice #INV-2026-042, Vendor: Acme Corp, "
    "Item: Widget Pro qty 10 @ $50.00, Total: $540.00\n\n"
    "=== Document: receipt.pdf ===\n"
    "Receipt #REC-2026-042, Paid: $540.00\n\n"
    "=== Document: order.pdf ===\n"
    "PO #PO-2026-042, Item: Widget Pro qty 10 @ $45.00, Total: $495.00"
)


def _make_search_ctx(content=EXPLORE_RAW_CONTENT, citations=None):
    """Build a SearchContext with pre-populated vector content."""
    from smartloop.agents.mcp.explorer import SearchContext
    ctx = SearchContext()
    ctx.vector_context = content
    ctx.citations = citations or [
        {"document_id": "doc-1", "document_name": "invoice.pdf"},
        {"document_id": "doc-2", "document_name": "receipt.pdf"},
        {"document_id": "doc-3", "document_name": "order.pdf"},
    ]
    return ctx


DISCREPANCY_SKILL = (
    "---\n"
    "name: discrepancy-check\n"
    "description: The skill is triggered when user requests to check for invoice discrepancy\n"
    "tools: [save_issue]\n"
    "---\n\n"
    "When user requests to find discrepancies in the invoice, do a three way check "
    "on receipt, invoice and order and create a report based on that.\n\n"
    "If discrepancies found then create an issue on Linear. Use the title, description "
    "from the report then make a tool call call save_issue on Linear MCP to save it, "
    "use team=DEV, assignee=me\n\n"
    "Execute the tool and execute it with the following arguments:\n"
    "- title\n"
    "- description\n"
    "- team\n\n"
    "Confirm that the tool returned successfully."
)

# ------------------------------------------------------------------ #
# Document data                                                        #
# ------------------------------------------------------------------ #

INVOICE_TEXT = (
    "Invoice #INV-2026-042\n"
    "Vendor: Acme Corp\n"
    "Date: 2026-05-01\n"
    "Item: Widget Pro (qty 10) @ $50.00 each\n"
    "Subtotal: $500.00\n"
    "Tax: $40.00\n"
    "Total: $540.00\n"
)

RECEIPT_TEXT = (
    "Receipt #REC-2026-042\n"
    "Vendor: Acme Corp\n"
    "Date: 2026-05-01\n"
    "Paid: $540.00\n"
    "Payment method: Wire transfer\n"
)

ORDER_TEXT = (
    "Purchase Order #PO-2026-042\n"
    "Vendor: Acme Corp\n"
    "Item: Widget Pro (qty 10) @ $45.00 each\n"  # <-- discrepancy: $45 vs $50
    "Approved total: $495.00\n"  # <-- discrepancy: $495 vs $540
)

DISCREPANCY_REPORT = (
    "## Three-Way Match Report\n\n"
    "### Documents Compared\n"
    "- Invoice #INV-2026-042\n"
    "- Receipt #REC-2026-042\n"
    "- Purchase Order #PO-2026-042\n\n"
    "### Discrepancies Found\n"
    "1. **Unit price mismatch:** Invoice shows $50.00/unit, PO shows $45.00/unit\n"
    "2. **Total mismatch:** Invoice total $540.00 vs PO approved total $495.00\n"
    "3. **Overcharge:** $45.00 above approved amount\n\n"
    "### Recommendation\n"
    "Flag for review — vendor invoiced above the approved PO rate."
)

NO_DISCREPANCY_REPORT = (
    "## Three-Way Match Report\n\n"
    "### Documents Compared\n"
    "- Invoice #INV-2026-042\n"
    "- Receipt #REC-2026-042\n"
    "- Purchase Order #PO-2026-042\n\n"
    "### Result\n"
    "All documents match. No discrepancies found.\n"
    "Invoice amount, receipt, and PO are consistent."
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

@dataclass
class FakeToolResult:
    success: bool
    content: Optional[List[Dict[str, Any]]]
    is_error: bool = False
    error: Optional[str] = None


def _make_linear_server() -> MCPServer:
    return MCPServer(
        id="linear-srv",
        name="Linear MCP",
        server_type=MCPServerType.REMOTE,
        remote_config=RemoteMCPServerConfig(server_url="https://linear.example.com"),
        enabled=True,
        tools=[
            MCPToolDefinition(
                name="save_issue",
                description="Create an issue in Linear",
                input_schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Issue title"},
                        "description": {"type": "string", "description": "Issue description"},
                        "team": {"type": "string", "description": "Team name to assign the issue to"},
                    },
                    "required": ["title", "description", "team"],
                },
            ),
        ],
    )


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #

class TestDiscrepancySkillPipeline:
    """E2E: discrepancy-check skill → explore → condition → save_issue."""

    USER_PROMPT = "Check the invoice for discrepancies against receipt and order"

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create document files in a temp workspace."""
        docs = tmp_path / "documents"
        docs.mkdir()
        (docs / "invoice.txt").write_text(INVOICE_TEXT)
        (docs / "receipt.txt").write_text(RECEIPT_TEXT)
        (docs / "order.txt").write_text(ORDER_TEXT)
        skills = tmp_path / "skills"
        skills.mkdir()
        skill_dir = skills / "discrepancy-check"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(DISCREPANCY_SKILL)
        return tmp_path

    @pytest.fixture
    def linear_server(self):
        return _make_linear_server()

    # Default step derivation response for the discrepancy skill
    DERIVED_STEPS_WITH_TOOL = json.dumps([
        {"step": 1, "step_type": "explore", "tool": "explore",
         "instruction": "Gather invoice, receipt, and order documents"},
        {"step": 2, "step_type": "analyze", "tool": "analyze",
         "instruction": "Produce a structured analysis report. Include specific "
         "findings, discrepancies, or results. Be factual and precise."},
        {"step": 3, "step_type": "tool", "tool": "save_issue",
         "server": "Linear MCP",
         "condition": "llm:Were discrepancies found in the report?",
         "instruction": "Create issue on Linear with report details",
         "default_args": {"team": "DEV", "assignee": "me"}},
    ])

    DERIVED_STEPS_NO_TOOL = json.dumps([
        {"step": 1, "step_type": "explore", "tool": "explore",
         "instruction": "Gather invoice, receipt, and order documents"},
        {"step": 2, "step_type": "analyze", "tool": "analyze",
         "instruction": "Produce a structured analysis report."},
    ])

    def _make_invoke_side_effect(self, report: str, *, expect_save: bool = True):
        """Build an llm.invoke side_effect for the full pipeline."""
        call_count = {"n": 0}

        def side_effect(prompt, *a, **kw):
            call_count["n"] += 1
            p = str(prompt).lower()

            # Step derivation: LLM produces step plan from skill content
            if "list the execution steps" in p and "json array" in p:
                if expect_save:
                    return self.DERIVED_STEPS_WITH_TOOL
                return self.DERIVED_STEPS_NO_TOOL

            # Arg extraction for save_issue (must be checked before generic matches)
            if "extract arguments for tool" in p or "extract the arguments" in p or "return only a valid json" in p:
                return json.dumps({
                    "title": "Invoice discrepancy: unit price mismatch",
                    "description": (
                        "Three-way match found discrepancies:\n"
                        "- Invoice #INV-2026-042 shows $50.00/unit vs PO $45.00/unit\n"
                        "- Total overcharge: $45.00"
                    ),
                    "team": "DEV",
                })

            # ANALYZE node: relevance assessment of explore context
            if "relevance assessment" in p:
                return report

            # Condition evaluation: YES/NO based on report content
            if "yes or no" in p:
                if "no discrepancies" in p:
                    return "NO"
                return "YES" if expect_save else "NO"

            # Skill matching: return the discrepancy skill as matched
            if "which skill" in p or ("match" in p and "skill" in p):
                return "discrepancy-check"

            # Classify: EXPLORE intent (skill pipeline handles it)
            if "classify" in p or "intent" in p:
                return "explore"

            # Explore: pick files
            if "which files" in p or "most relevant" in p:
                return "invoice.txt\nreceipt.txt\norder.txt"

            # Explore: summarize/enrich — return the report
            if "produce a structured analysis report" in p:
                return report

            # Explore: summarize/enrich — return the report
            if "provide" in p or "structured" in p or "based on" in p or "file tree" in p:
                return report

            # Dependency analysis
            if "dependencies" in p or "prerequisite" in p:
                return json.dumps({
                    "has_dependencies": False,
                    "can_execute": True,
                    "dependencies": [],
                })

            return "OK"

        return side_effect

    def test_skill_explore_then_save_issue(self, workspace, linear_server):
        """Full pipeline: explore docs → find discrepancies → save_issue on Linear."""
        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=8192)
        mock_llm.invoke = MagicMock(
            side_effect=self._make_invoke_side_effect(DISCREPANCY_REPORT, expect_save=True)
        )

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        fake_save = FakeToolResult(
            success=True,
            content=[{"type": "text", "text": '{"status": "created", "title": "Invoice discrepancy"}'}],
        )

        with patch.object(orchestrator.explorer, "collect_parallel", return_value=_make_search_ctx()), \
             patch.object(orchestrator.tool_caller, "call_tool", return_value=fake_save) as mock_call:
            result = orchestrator.execute(
                user_message=self.USER_PROMPT,
                mcp_servers=[linear_server],
                skills=[DISCREPANCY_SKILL],
                cwd=str(workspace),
            )

        assert result.handled is True

        # save_issue was called
        assert mock_call.call_count >= 1
        call_args = mock_call.call_args
        assert call_args is not None
        # Verify tool name
        assert call_args.kwargs.get("tool_name") == "save_issue" or (
            len(call_args.args) >= 2 and call_args.args[1] == "save_issue"
        )
        # Verify arguments contain the required fields
        arguments = call_args.kwargs.get("arguments") or (
            call_args.args[2] if len(call_args.args) >= 3 else {}
        )
        assert "title" in arguments
        assert "description" in arguments
        assert arguments.get("team") == "DEV"

        # save_issue appears in tool_names
        assert "save_issue" in result.tool_names

    def test_no_discrepancies_skips_save_issue(self, workspace, linear_server):
        """When no discrepancies are found, save_issue should be skipped."""
        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=8192)
        mock_llm.invoke = MagicMock(
            side_effect=self._make_invoke_side_effect(NO_DISCREPANCY_REPORT, expect_save=False)
        )

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        fake_save = FakeToolResult(
            success=True,
            content=[{"type": "text", "text": '{"status": "created", "title": "No discrepancy"}'}],
        )

        with patch.object(orchestrator.explorer, "collect_parallel", return_value=_make_search_ctx()), \
             patch.object(orchestrator.tool_caller, "call_tool", return_value=fake_save) as mock_call:
            result = orchestrator.execute(
                user_message=self.USER_PROMPT,
                mcp_servers=[linear_server],
                skills=[DISCREPANCY_SKILL],
                cwd=str(workspace),
            )

        assert result.handled is True

        # save_issue should NOT have been called
        assert mock_call.call_count == 0, (
            f"save_issue was called {mock_call.call_count} time(s) "
            f"but should have been skipped (no discrepancies)"
        )
        assert "save_issue" not in result.tool_names

    def test_explore_results_passed_to_condition(self, workspace, linear_server):
        """Verify explore results are available to the condition evaluator."""
        condition_prompts = []

        def capture_invoke(prompt, *a, **kw):
            p = str(prompt)
            p_lower = p.lower()

            # Step derivation
            if "list the execution steps" in p_lower and "json array" in p_lower:
                return self.DERIVED_STEPS_WITH_TOOL
            # Arg extraction (check before generic keyword matches)
            if "extract the arguments" in p_lower or "return only a valid json" in p_lower:
                return json.dumps({
                    "title": "Discrepancy found",
                    "description": "Price mismatch",
                    "team": "DEV",
                })
            # ANALYZE node: relevance assessment
            if "relevance assessment" in p_lower:
                return DISCREPANCY_REPORT
            # Condition evaluation
            if "yes or no" in p_lower:
                condition_prompts.append(p)
                return "YES"
            # Skill matching
            if "which skill" in p_lower or ("match" in p_lower and "skill" in p_lower):
                return "discrepancy-check"
            if "classify" in p_lower or "intent" in p_lower:
                return "explore"
            if "which files" in p_lower or "most relevant" in p_lower:
                return "invoice.txt"
            if "produce a structured analysis report" in p_lower:
                return DISCREPANCY_REPORT
            if "provide" in p_lower or "structured" in p_lower or "based on" in p_lower or "file tree" in p_lower:
                return DISCREPANCY_REPORT
            if "dependencies" in p_lower or "prerequisite" in p_lower:
                return json.dumps({
                    "has_dependencies": False,
                    "can_execute": True,
                    "dependencies": [],
                })
            return "OK"

        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=8192)
        mock_llm.invoke = MagicMock(side_effect=capture_invoke)

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)
        fake_save = FakeToolResult(
            success=True,
            content=[{"type": "text", "text": "created"}],
        )

        with patch.object(orchestrator.explorer, "collect_parallel", return_value=_make_search_ctx()), \
             patch.object(orchestrator.tool_caller, "call_tool", return_value=fake_save):
            orchestrator.execute(
                user_message=self.USER_PROMPT,
                mcp_servers=[linear_server],
                skills=[DISCREPANCY_SKILL],
                cwd=str(workspace),
            )

        # The condition evaluator should have received the explore results
        assert len(condition_prompts) >= 1, (
            "Condition evaluation LLM call was never made — "
            "explore results were not passed to _evaluate_condition"
        )
        # The explore report content should appear in the condition prompt
        cond_text = condition_prompts[0].lower()
        assert "discrepanc" in cond_text or "mismatch" in cond_text, (
            "Condition prompt does not contain explore results (discrepancy report)"
        )

    def test_hybrid_executor_uses_io_pool_for_skill_steps(self, workspace, linear_server):
        """Verify the skill pipeline dispatches explore through _io_pool,
        not the LLM process pool."""
        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=8192)
        mock_llm.invoke = MagicMock(
            side_effect=self._make_invoke_side_effect(DISCREPANCY_REPORT, expect_save=True)
        )

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        # The executor should be a LLMExecutor
        from smartloop.agents.hybrid_executor import LLMExecutor
        assert isinstance(orchestrator._executor, LLMExecutor)

        # _io_pool should be a ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor
        assert isinstance(orchestrator._executor._io_pool, ThreadPoolExecutor)

        # With n_llm_workers defaulting to n_parallel (2) but no model_path,
        # the llm_pool should be None (no GGUF path → no process pool)
        assert orchestrator._executor._llm_pool is None

    def test_synthesis_step_produces_report_in_output(self, workspace, linear_server):
        """Verify the synthesis (ANALYZE) step runs between explore and
        condition check, and its output appears in the final result."""
        synthesis_prompts = []

        def side_effect(prompt, *a, **kw):
            p = str(prompt).lower()

            # Step derivation
            if "list the execution steps" in p and "json array" in p:
                return self.DERIVED_STEPS_WITH_TOOL

            # Capture synthesis prompts (ANALYZE step or final synthesis)
            if "produce a structured analysis report" in p:
                synthesis_prompts.append(str(prompt))
                return DISCREPANCY_REPORT

            # ANALYZE node: relevance assessment
            if "relevance assessment" in p:
                synthesis_prompts.append(str(prompt))
                return DISCREPANCY_REPORT

            # Final synthesis (merge step outputs into coherent document)
            if "data synthesis assistant" in p and "merge them into a single" in p:
                synthesis_prompts.append(str(prompt))
                return DISCREPANCY_REPORT

            # Arg extraction
            if "extract arguments for tool" in p or "extract the arguments" in p or "return only a valid json" in p:
                return json.dumps({
                    "title": "Unit price mismatch",
                    "description": "Invoice $50 vs PO $45",
                    "team": "DEV",
                })
            if "yes or no" in p:
                return "YES"
            if "which skill" in p or ("match" in p and "skill" in p):
                return "discrepancy-check"
            if "which files" in p or "most relevant" in p:
                return "invoice.txt"
            if "provide" in p or "structured" in p or "based on" in p or "file tree" in p:
                return "Raw document data: Invoice total $540, PO total $495"
            if "dependencies" in p:
                return json.dumps({
                    "has_dependencies": False,
                    "can_execute": True,
                    "dependencies": [],
                })
            return "OK"

        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=8192)
        mock_llm.invoke = MagicMock(side_effect=side_effect)

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)
        fake_save = FakeToolResult(
            success=True,
            content=[{"type": "text", "text": '{"status": "created", "title": "Unit price mismatch"}'}],
        )

        with patch.object(orchestrator.explorer, "collect_parallel", return_value=_make_search_ctx()), \
             patch.object(orchestrator.tool_caller, "call_tool", return_value=fake_save):
            result = orchestrator.execute(
                user_message=self.USER_PROMPT,
                mcp_servers=[linear_server],
                skills=[DISCREPANCY_SKILL],
                cwd=str(workspace),
            )

        assert result.handled is True

        # Synthesis step was called
        assert len(synthesis_prompts) >= 1, (
            "Synthesis LLM call was never made — explore results were not "
            "synthesized into a report before condition evaluation"
        )

        # The synthesized report should appear in the final output
        # (synthesis report is included as document context for the LLM)
        assert result.content is not None, "Result content is None"
        assert "Discrepancies Found" in result.content or "mismatch" in result.content.lower() or "save issue" in result.content.lower(), (
            f"Expected report or success message in result.content, got: {result.content[:200]}"
        )

        # save_issue was triggered (condition passed on synthesized report)
        assert "save_issue" in result.tool_names

    def test_tool_call_failure_still_returns_report(self, workspace, linear_server):
        """When the MCP tool call fails, the synthesized report should still
        appear in the final output (not raw document chunks)."""
        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=8192)
        mock_llm.invoke = MagicMock(
            side_effect=self._make_invoke_side_effect(DISCREPANCY_REPORT, expect_save=True)
        )

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        # Tool call fails (server error)
        fail_result = FakeToolResult(
            success=False, content=None, is_error=True,
            error="500 Internal Server Error",
        )

        with patch.object(orchestrator.explorer, "collect_parallel", return_value=_make_search_ctx()), \
             patch.object(orchestrator.tool_caller, "call_tool", return_value=fail_result):
            result = orchestrator.execute(
                user_message=self.USER_PROMPT,
                mcp_servers=[linear_server],
                skills=[DISCREPANCY_SKILL],
                cwd=str(workspace),
            )

        assert result.handled is True
        assert result.content is not None

        # The synthesized report should be in the output — not raw doc chunks
        assert "Discrepancies Found" in result.content or "mismatch" in result.content.lower(), (
            f"Expected synthesized report in output when tool fails, got: {result.content[:300]}"
        )


class TestSkillStepDerivation:
    """Verify _derive_skill_steps parses skill content into ordered steps."""

    def test_derive_steps_from_natural_language(self):
        """LLM-derived steps should include explore, analyze and tool steps."""
        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=8192)
        mock_llm.invoke = MagicMock(return_value=json.dumps([
            {"step": 1, "step_type": "explore", "tool": "explore",
             "instruction": "Gather documents"},
            {"step": 2, "step_type": "analyze", "tool": "analyze",
             "instruction": "Create three-way match report"},
            {"step": 3, "step_type": "tool", "tool": "save_issue",
             "server": "Linear MCP",
             "condition": "llm:Were discrepancies found?",
             "instruction": "Create issue on Linear"},
        ]))

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)
        matched = [{"name": "discrepancy-check", "content": DISCREPANCY_SKILL,
                     "skill_map": [], "description": ""}]

        linear = _make_linear_server()
        mcp_ctx = [MCPToolContext(
            server_id=linear.id, server_name=linear.name,
            tools=[{"name": t.name, "description": t.description,
                    "input_schema": t.input_schema} for t in linear.tools],
        )]

        steps = orchestrator._derive_skill_steps(matched, mcp_ctx, "check discrepancy")

        # explore → analyze → tool.  ANALYZE is a real step (relevance
        # assessment of the explore output); the terminal SYNTHESIZE node
        # produces the final report separately.
        assert len(steps) == 3
        assert steps[0].step_type.value == "explore"
        assert steps[1].step_type.value == "analyze"
        tool_step = steps[2]
        assert tool_step.step_type.value == "tool"
        assert tool_step.tool_name == "save_issue"
        assert tool_step.condition is not None
        assert "discrepanc" in tool_step.condition.lower()

    def test_explicit_skill_map_takes_priority(self):
        """When skill has ## skill_map, it should be used instead of LLM derivation."""
        skill_with_map = (
            DISCREPANCY_SKILL + "\n\n"
            "## skill_map\n"
            "- step: 1\n"
            "  step_type: explore\n"
            "  tool: explore\n"
            "  instruction: Gather documents\n"
            "- step: 2\n"
            "  step_type: tool\n"
            "  tool: save_issue\n"
            "  server: Linear MCP\n"
            "  condition: llm:Were discrepancies found?\n"
            "  instruction: Create Linear issue\n"
        )

        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=8192)
        # LLM should NOT be called for step derivation
        mock_llm.invoke = MagicMock(return_value="should not be called")

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)
        skill_map = orchestrator._parse_skill_map(skill_with_map)
        matched = [{"name": "discrepancy-check", "content": skill_with_map,
                     "skill_map": skill_map, "description": ""}]

        linear = _make_linear_server()
        mcp_ctx = [MCPToolContext(
            server_id=linear.id, server_name=linear.name,
            tools=[{"name": t.name, "description": t.description,
                    "input_schema": t.input_schema} for t in linear.tools],
        )]

        steps = orchestrator._derive_skill_steps(matched, mcp_ctx, "check discrepancy")

        # Should have explore (prepended) + analyze (prepended) + tool from skill_map
        tool_steps = [s for s in steps if s.step_type.value == "tool"]
        assert len(tool_steps) >= 1
        assert tool_steps[0].tool_name == "save_issue"

    def test_fallback_when_llm_fails(self):
        """When LLM derivation fails, fallback to default steps."""
        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=8192)
        mock_llm.invoke = MagicMock(side_effect=RuntimeError("LLM unavailable"))

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)
        matched = [{"name": "discrepancy-check", "content": DISCREPANCY_SKILL,
                     "skill_map": [], "description": ""}]

        linear = _make_linear_server()
        mcp_ctx = [MCPToolContext(
            server_id=linear.id, server_name=linear.name,
            tools=[{"name": t.name, "description": t.description,
                    "input_schema": t.input_schema} for t in linear.tools],
        )]

        steps = orchestrator._derive_skill_steps(matched, mcp_ctx, "check discrepancy")

        # Should fall back to default: explore → save_issue
        # (analyze is no longer a separate step — synthesis happens in format_result)
        assert len(steps) >= 2
        assert steps[0].step_type.value == "explore"
        tool_steps = [s for s in steps if s.step_type.value == "tool"]
        assert len(tool_steps) >= 1


class TestCitationFormatting:
    """Verify _format_ready_context includes source document names for citations."""

    def test_citations_include_document_names_in_context(self):
        """With multiple citations, the formatted context should list all
        source document names and instruct the LLM to attribute facts to each
        document evenly rather than funnelling the answer through one."""
        citations = [
            {"document_id": "doc-1", "document_name": "invoice_Adam Hart_16384.pdf"},
            {"document_id": "doc-2", "document_name": "invoice_Aaron Bergman_36258.pdf"},
        ]
        combined = (
            "=== Document: invoice_Adam Hart_16384.pdf ===\n"
            "Invoice total: $540.00\n\n"
            "=== Document: invoice_Aaron Bergman_36258.pdf ===\n"
            "Invoice total: $120.00"
        )
        result = SLPOrchestrator._format_ready_context(
            combined, skill_instruction=None, citations=citations, cwd="/tmp",
        )

        # Must contain DOCUMENT CONTEXT label (not TOOL CONTEXT)
        assert "DOCUMENT CONTEXT" in result

        # Must list source document names explicitly
        assert "invoice_Adam Hart_16384.pdf" in result
        assert "invoice_Aaron Bergman_36258.pdf" in result

        # Multi-document responses must not be biased toward a single source
        assert "MULTIPLE documents" in result
        assert "ALL relevant" in result
        # The filename is only an identifier, not a description of the contents
        assert "may not describe the contents" in result

    def test_single_citation_instructs_reference_by_name(self):
        """With a single document, the LLM must reference it by name."""
        citations = [
            {"document_id": "doc-1", "document_name": "invoice_Adam Hart_16384.pdf"},
        ]
        result = SLPOrchestrator._format_ready_context(
            "Invoice total: $540.00", skill_instruction=None,
            citations=citations, cwd="/tmp",
        )
        assert "DOCUMENT CONTEXT" in result
        assert "MUST reference" in result
        assert "may not describe the contents" in result

    def test_no_citations_uses_tool_results(self):
        """Without citations, tool output is labelled as TOOL RESULTS."""
        result = SLPOrchestrator._format_ready_context(
            "Some tool output", skill_instruction=None, citations=None, cwd="/tmp",
            has_tool_results=True,
        )
        assert "TOOL RESULTS" in result
        assert "DOCUMENT CONTEXT" not in result

    def test_no_citations_no_tool_results_passes_through(self):
        """Without citations or tool results flag, context passes through plain."""
        result = SLPOrchestrator._format_ready_context(
            "Some tool output", skill_instruction=None, citations=None, cwd="/tmp",
        )
        assert "Some tool output" in result
        assert "DOCUMENT CONTEXT" not in result

    def test_empty_citations_list_uses_tool_results(self):
        """An empty citations list with tool results flag should use TOOL RESULTS."""
        result = SLPOrchestrator._format_ready_context(
            "Some tool output", skill_instruction=None, citations=[], cwd="/tmp",
            has_tool_results=True,
        )
        assert "TOOL RESULTS" in result
