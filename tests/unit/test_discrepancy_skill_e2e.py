"""Citation formatting for the final ready-context.

These tests pin how ``_format_ready_context`` labels and attributes
context: document citations produce a DOCUMENT CONTEXT block listing the
source names, while plain tool output is labelled TOOL RESULTS.

(The deterministic skill-pipeline e2e tests that previously lived here
were removed when skill execution was unified onto the single
classify → plan → execute path.)
"""

from smartloop.agents.orchestrator import SLPOrchestrator


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
