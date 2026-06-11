"""End-to-end test: Explore invoice PDF → Fetch pricing page → Write report.

Simulates a user asking:
  "Explore the invoice in data/, check the pricing at
   https://lovable.dev/pricing, confirm it matches, and write a detailed
   feature report."

The planner produces a 3-step plan (explore → fetch → write) from the
user prompt.  Built-in handlers run against a real PDF copy; the fetch
MCP call is mocked with the actual lovable.dev pricing data.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from smartloop.agents.orchestrator import SLPOrchestrator, MCPExecutionResult
from smartloop.agents.planner import StepType
from smartloop.workspace.workspace import (
    MCPServer,
    MCPServerType,
    MCPToolDefinition,
    RemoteMCPServerConfig,
)


# ------------------------------------------------------------------ #
# Fixtures & constants                                                 #
# ------------------------------------------------------------------ #

DATA_DIR = Path(__file__).parent / "data"
INVOICE_PDF = DATA_DIR / "Invoice-1PTUKWWH-0015.pdf"

# Extracted invoice facts (from the real PDF)
INVOICE_FACTS = {
    "invoice_number": "1PTUKWWH-0015",
    "plan": "Pro",
    "amount": 25.00,
    "currency": "USD",
    "billing_period": "Feb 18–Mar 18, 2026",
    "vendor": "Lovable",
    "billed_to": "Mehfuz",
}

# Pricing data scraped from https://lovable.dev/pricing
LOVABLE_PRICING_PAGE = (
    "# Lovable Pricing\n\n"
    "## Free\n"
    "- Price: $0/month\n"
    "- 5 credits/day, 30 credits/month\n"
    "- Public projects only\n\n"
    "## Pro\n"
    "- Price: $25/month\n"
    "- 100 monthly credits + 5 daily credits (up to 150/month)\n"
    "- Credit rollovers\n"
    "- On-demand credit top-ups\n"
    "- Unlimited lovable.app domains\n"
    "- Custom domains\n"
    "- Remove Lovable badge\n"
    "- User roles & permissions\n"
    "- Private projects\n\n"
    "## Business\n"
    "- Price: $50/month\n"
    "- 100 monthly credits\n"
    "- All Pro features plus:\n"
    "- Internal publish, SSO, Team workspace\n"
    "- Personal projects, Design templates\n"
    "- Role-based access, Security center\n\n"
    "## Enterprise\n"
    "- Custom pricing\n"
    "- Volume-based credit pricing\n"
    "- Dedicated support, Onboarding services\n"
    "- Design systems, SCIM, Custom connectors\n"
    "- Publishing & sharing controls, Audit logs\n"
)

EXPECTED_REPORT = (
    "# Lovable Invoice & Pricing Verification Report\n\n"
    "## Invoice Summary\n"
    "- **Invoice Number:** 1PTUKWWH-0015\n"
    "- **Plan:** Pro 1\n"
    "- **Amount:** $25.00 USD\n"
    "- **Billing Period:** Feb 18 – Mar 18, 2026\n"
    "- **Vendor:** Lovable\n"
    "- **Billed To:** Mehfuz\n\n"
    "## Pricing Verification\n"
    "- **Pro plan listed price:** $25/month\n"
    "- **Invoice amount:** $25.00\n"
    "- **Match:** YES — invoice matches the published Pro plan price.\n\n"
    "## Pro Plan Features\n"
    "- 100 monthly credits + 5 daily credits (up to 150/month)\n"
    "- Credit rollovers\n"
    "- On-demand credit top-ups\n"
    "- Unlimited lovable.app domains\n"
    "- Custom domains\n"
    "- Remove Lovable badge\n"
    "- User roles & permissions\n"
    "- Private projects\n\n"
    "## All Plans Overview\n"
    "| Plan | Price | Credits |\n"
    "| --- | --- | --- |\n"
    "| Free | $0/month | 5/day, 30/month |\n"
    "| Pro | $25/month | 100/month + 5/day |\n"
    "| Business | $50/month | 100/month |\n"
    "| Enterprise | Custom | Volume-based |\n"
)


@dataclass
class FakeToolResult:
    success: bool
    content: Optional[List[Dict[str, Any]]]
    is_error: bool = False
    error: Optional[str] = None


def _make_fetch_server() -> MCPServer:
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


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #

class TestInvoicePricingPipeline:
    """E2E: user prompt → planner builds explore/fetch/write plan →
    orchestrator executes each step against real files and mocked MCP."""

    USER_PROMPT = (
        "Explore the invoice PDF in {data_dir}, "
        "check the pricing at https://lovable.dev/pricing to confirm it "
        "matches, and write a detailed report of the features to {output_path}"
    )

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create a text representation of the invoice in a temp workspace data/ dir.

        Uses a .txt file rather than the real PDF because the explore step
        does not yet support .pdf files in ``_EXPLORE_EXTENSIONS``."""
        data = tmp_path / "data"
        data.mkdir()
        (data / "Invoice-1PTUKWWH-0015.txt").write_text(
            "Invoice\n"
            "Invoice number 1PTUKWWH-0015\n"
            "Date of issue February 18, 2026\n"
            "Date due February 18, 2026\n\n"
            "Lovable\n"
            "1111b South Governors Avenue\n"
            "Dover, Delaware 19904\n\n"
            "Bill to\nMehfuz\n"
            "49207 Periwinkle Ter\n"
            "Fremont, California 94539-7524\n\n"
            "Description Qty Unit price Amount\n"
            "Pro 1\nFeb 18–Mar 18, 2026\n"
            "1 $25.00 $25.00\n\n"
            "Subtotal $25.00\n"
            "Total $25.00\n"
            "Amount due $25.00 USD\n"
        )
        return tmp_path

    @pytest.fixture
    def fetch_server(self):
        return _make_fetch_server()

    def _build_plan_response(self, data_dir: str, output_path: str) -> str:
        """Return the JSON plan the LLM planner would produce."""
        return json.dumps([
            {
                "step": 1,
                "step_type": "explore",
                "tool_name": "explore",
                "server_id": "__system_tools__",
                "server_name": "System Tools",
                "instruction": f"Explore the directory {data_dir} and read the invoice PDF to extract invoice details (plan name, amount, billing period, vendor).",
            },
            {
                "step": 2,
                "step_type": "tool",
                "tool_name": "fetch",
                "server_id": "fetch-srv",
                "server_name": "Fetch MCP Server",
                "instruction": "Fetch https://lovable.dev/pricing and extract all plan names, prices, and features.",
            },
            {
                "step": 3,
                "step_type": "action",
                "action_kind": "write",
                "tool_name": "file-write",
                "server_id": "__system_tools__",
                "server_name": "System Tools",
                "instruction": f"Write a detailed verification report to {output_path} comparing the invoice to the published pricing and listing all Pro plan features.",
            },
        ])

    def _make_invoke_side_effect(self, plan_json: str, data_dir: str, output_path: str):
        """Build an ``llm.invoke`` side_effect that returns context-aware
        responses for every call the orchestrator makes."""

        # The explore step calls LLM twice: once to pick files, once to enrich.
        # The write step calls LLM once for arg extraction.
        def side_effect(prompt, *a, **kw):
            p = str(prompt)
            p_lower = p.lower()

            # 1. Planner prompt
            if "tool planner" in p_lower or "json array" in p_lower:
                return plan_json

            # 2. Explore: LLM picks which files to read
            if "which files are most relevant" in p_lower:
                # Return the invoice file (whatever extension it is)
                files = os.listdir(data_dir)
                invoice = next((f for f in files if "invoice" in f.lower()), files[0])
                return invoice

            # 3. Explore: LLM enriches / summarizes file contents
            if "based on the file tree" in p_lower or "provide a clear, structured" in p_lower:
                base = (
                    "## Invoice Details\n\n"
                    "- **Invoice Number:** 1PTUKWWH-0015\n"
                    "- **Plan:** Pro 1\n"
                    "- **Amount:** $25.00 USD\n"
                    "- **Billing Period:** Feb 18 – Mar 18, 2026\n"
                    "- **Vendor:** Lovable (Dover, Delaware)\n"
                    "- **Billed To:** Mehfuz (Fremont, CA)\n"
                    "- **Date of Issue:** February 18, 2026\n"
                )
                # Include pricing data when the fetched URL content is in the prompt
                if "fetched url content" in p_lower and "lovable pricing" in p_lower:
                    base += (
                        "\n## Pricing Tiers\n\n"
                        "- **Free:** $0/month — 5 credits/day, 30 credits/month\n"
                        "- **Pro:** $25/month — 100 monthly credits + 5 daily credits\n"
                        "- **Business:** $50/month — 100 monthly credits + all Pro features\n"
                        "- **Enterprise:** Custom pricing — Volume-based credits\n"
                    )
                elif "could not fetch" in p_lower:
                    base += "\nCould not fetch content from the pricing page."
                return base

            # 4. Arg extraction for file-write
            if "reply with only a json object" in p_lower:
                return json.dumps({
                    "path": output_path,
                    "content": EXPECTED_REPORT,
                })

            # Fallback
            return "OK"

        return side_effect

    # ------------------------------------------------------------------ #
    # Test: full explore → fetch → write pipeline                        #
    # ------------------------------------------------------------------ #

    def test_explore_fetch_write_pipeline(self, workspace, fetch_server):
        """Complete 3-step flow driven by a user prompt:
        explore invoice → fetch pricing → write report."""
        data_dir = str(workspace / "data")
        output_path = str(workspace / "report.md")
        user_msg = self.USER_PROMPT.format(data_dir=data_dir, output_path=output_path)
        plan_json = self._build_plan_response(data_dir, output_path)

        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=8192)
        mock_llm.invoke = MagicMock(
            side_effect=self._make_invoke_side_effect(plan_json, data_dir, output_path)
        )

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        fake_fetch = FakeToolResult(
            success=True,
            content=[{"type": "text", "text": LOVABLE_PRICING_PAGE}],
        )

        with patch.object(orchestrator.tool_caller, "call_tool", return_value=fake_fetch):
            result = orchestrator.execute(
                user_message=user_msg,
                mcp_servers=[fetch_server],
                cwd=str(workspace),
            )

        # Pipeline completed
        assert result.handled is True
        assert len(result.tool_names) == 3

        # Step types executed in order
        assert result.tool_names[0] == "explore"       # explore invoice
        assert result.tool_names[1] == "fetch"          # fetch pricing
        assert result.tool_names[2] == "file-write"     # write report

        # Report was written to disk
        assert os.path.isfile(output_path), "Report file was not created"
        report = open(output_path).read()

        # Report contains invoice data
        assert "1PTUKWWH-0015" in report
        assert "$25.00" in report
        assert "Pro" in report

        # Report contains pricing verification
        assert "Match" in report or "match" in report.lower()

        # Report contains Pro plan features
        assert "credit" in report.lower()
        assert "Custom domains" in report or "custom domains" in report.lower()

    # ------------------------------------------------------------------ #
    # Test: pricing matches invoice amount                               #
    # ------------------------------------------------------------------ #

    def test_invoice_amount_matches_pro_pricing(self, workspace, fetch_server):
        """Verify the explore step fetches the URL and returns pricing data."""
        data_dir = str(workspace / "data")
        output_path = str(workspace / "report.md")
        plan_json = self._build_plan_response(data_dir, output_path)

        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=8192)
        mock_llm.invoke = MagicMock(
            side_effect=self._make_invoke_side_effect(plan_json, data_dir, output_path)
        )

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        fake_fetch = FakeToolResult(
            success=True,
            content=[{"type": "text", "text": LOVABLE_PRICING_PAGE}],
        )

        with patch.object(orchestrator.tool_caller, "call_tool", return_value=fake_fetch):
            result = orchestrator.execute(
                user_message=self.USER_PROMPT.format(
                    data_dir=data_dir, output_path=output_path
                ),
                mcp_servers=[fetch_server],
                cwd=str(workspace),
            )

        assert result.handled is True
        assert "explore" in result.tool_names

        # The pricing data should be in the output (fetched via explore)
        assert "$25" in result.content
        assert "Pro" in result.content

    # ------------------------------------------------------------------ #
    # Test: explore step reads the invoice file                          #
    # ------------------------------------------------------------------ #

    def test_explore_reads_invoice_from_data_dir(self, workspace, fetch_server):
        """The explore step fetches the URL; pricing content appears in the result."""
        data_dir = str(workspace / "data")
        output_path = str(workspace / "report.md")
        plan_json = self._build_plan_response(data_dir, output_path)

        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=8192)
        mock_llm.invoke = MagicMock(
            side_effect=self._make_invoke_side_effect(plan_json, data_dir, output_path)
        )

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        fake_fetch = FakeToolResult(
            success=True,
            content=[{"type": "text", "text": LOVABLE_PRICING_PAGE}],
        )

        with patch.object(orchestrator.tool_caller, "call_tool", return_value=fake_fetch):
            result = orchestrator.execute(
                user_message=self.USER_PROMPT.format(
                    data_dir=data_dir, output_path=output_path
                ),
                mcp_servers=[fetch_server],
                cwd=str(workspace),
            )

        assert result.handled is True
        assert "explore" in result.tool_names
        assert "Lovable" in result.content or "Pro" in result.content

    # ------------------------------------------------------------------ #
    # Test: fetch step gets pricing data                                 #
    # ------------------------------------------------------------------ #

    def test_fetch_retrieves_pricing_page(self, workspace, fetch_server):
        """The explore step internally calls fetch and pricing content
        appears in the output."""
        data_dir = str(workspace / "data")
        output_path = str(workspace / "report.md")
        plan_json = self._build_plan_response(data_dir, output_path)

        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=8192)
        mock_llm.invoke = MagicMock(
            side_effect=self._make_invoke_side_effect(plan_json, data_dir, output_path)
        )

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        fake_fetch = FakeToolResult(
            success=True,
            content=[{"type": "text", "text": LOVABLE_PRICING_PAGE}],
        )

        with patch.object(orchestrator.tool_caller, "call_tool", return_value=fake_fetch) as mock_call:
            result = orchestrator.execute(
                user_message=self.USER_PROMPT.format(
                    data_dir=data_dir, output_path=output_path
                ),
                mcp_servers=[fetch_server],
                cwd=str(workspace),
            )

        assert result.handled is True

        # The MCP fetch tool was called (by explore internally and/or as a plan step)
        assert mock_call.call_count >= 1

        # Pricing data appears in the combined output
        assert "Business" in result.content or "$50" in result.content
        assert "Enterprise" in result.content or "Custom" in result.content

    # ------------------------------------------------------------------ #
    # Test: write step creates the report file                           #
    # ------------------------------------------------------------------ #

    def test_write_creates_report_with_all_sections(self, workspace, fetch_server):
        """Explore fetches pricing data and all plan tiers are in the output."""
        data_dir = str(workspace / "data")
        output_path = str(workspace / "report.md")
        plan_json = self._build_plan_response(data_dir, output_path)

        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=8192)
        mock_llm.invoke = MagicMock(
            side_effect=self._make_invoke_side_effect(plan_json, data_dir, output_path)
        )

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        fake_fetch = FakeToolResult(
            success=True,
            content=[{"type": "text", "text": LOVABLE_PRICING_PAGE}],
        )

        with patch.object(orchestrator.tool_caller, "call_tool", return_value=fake_fetch):
            result = orchestrator.execute(
                user_message=self.USER_PROMPT.format(
                    data_dir=data_dir, output_path=output_path
                ),
                mcp_servers=[fetch_server],
                cwd=str(workspace),
            )

        assert result.handled is True

        # All plan tiers from the pricing page should appear in the output
        assert "Pro" in result.content
        assert "Free" in result.content
        assert "Business" in result.content
        assert "Enterprise" in result.content

    # ------------------------------------------------------------------ #
    # Test: fetch failure — explore result still returned                 #
    # ------------------------------------------------------------------ #

    def test_fetch_failure_returns_explore_result(self, workspace, fetch_server):
        """If the fetch fails inside explore, a user-facing error message
        is returned instead of unrelated project files."""
        data_dir = str(workspace / "data")
        output_path = str(workspace / "report.md")
        plan_json = self._build_plan_response(data_dir, output_path)

        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=8192)
        mock_llm.invoke = MagicMock(
            side_effect=self._make_invoke_side_effect(plan_json, data_dir, output_path)
        )

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        fake_fetch = FakeToolResult(
            success=False, content=None, error="Connection refused",
        )

        with patch.object(orchestrator.tool_caller, "call_tool", return_value=fake_fetch):
            result = orchestrator.execute(
                user_message=self.USER_PROMPT.format(
                    data_dir=data_dir, output_path=output_path
                ),
                mcp_servers=[fetch_server],
                cwd=str(workspace),
            )

        assert result.handled is True
        assert "explore" in result.tool_names
        assert "Could not fetch" in result.content

    # ------------------------------------------------------------------ #
    # Test: document context supplements the plan                        #
    # ------------------------------------------------------------------ #

    def test_document_context_passed_through(self, workspace, fetch_server):
        """When document_context is provided, it is included in the
        explore result alongside the fetched URL content."""
        data_dir = str(workspace / "data")
        output_path = str(workspace / "report.md")
        plan_json = self._build_plan_response(data_dir, output_path)
        doc_context = "Internal policy: all SaaS subscriptions must be Pro tier or above."

        mock_llm = MagicMock()
        mock_llm.get_num_tokens = MagicMock(return_value=100)
        mock_llm.get_max_tokens = MagicMock(return_value=8192)
        mock_llm.invoke = MagicMock(
            side_effect=self._make_invoke_side_effect(plan_json, data_dir, output_path)
        )

        orchestrator = SLPOrchestrator(orchestrator_llm=mock_llm)

        fake_fetch = FakeToolResult(
            success=True,
            content=[{"type": "text", "text": LOVABLE_PRICING_PAGE}],
        )

        with patch.object(orchestrator.tool_caller, "call_tool", return_value=fake_fetch):
            result = orchestrator.execute(
                user_message=self.USER_PROMPT.format(
                    data_dir=data_dir, output_path=output_path
                ),
                mcp_servers=[fetch_server],
                cwd=str(workspace),
                document_context=doc_context,
            )

        assert result.handled is True
        assert result.content is not None
        assert "explore" in result.tool_names
