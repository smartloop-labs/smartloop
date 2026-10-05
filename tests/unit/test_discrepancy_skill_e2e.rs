//! Port of `tests/unit/test_discrepancy_skill_e2e.py`.
//!
//! Citation formatting for the final ready-context.
//!
//! These tests pin how `ResponseBuilder::format_ready_context` labels and
//! attributes context: document citations produce a DOCUMENT CONTEXT block
//! listing the source names, while plain tool output is labelled TOOL RESULTS.
//!
//! (The deterministic skill-pipeline e2e tests that previously lived here
//! were removed when skill execution was unified onto the single
//! classify → plan → execute path.)

use serde_json::{json, Value};

use smartloop::agents::orchestration::response_builder::ResponseBuilder;

/// `ResponseBuilder.format_ready_context(combined, skill_instruction=None,
/// citations=..., cwd="/tmp", has_tool_results=...)`.
fn format(combined: &str, citations: Option<&[Value]>, has_tool_results: bool) -> String {
    ResponseBuilder::format_ready_context(Some(combined), None, citations, Some("/tmp"), has_tool_results, None)
}

mod citation_formatting {
    //! Verify `format_ready_context` includes source document names for
    //! citations.

    use super::*;

    /// With multiple citations, the formatted context should list all source
    /// document names and instruct the LLM to attribute facts to each
    /// document evenly rather than funnelling the answer through one.
    #[test]
    fn test_citations_include_document_names_in_context() {
        let citations = vec![
            json!({"document_id": "doc-1", "document_name": "invoice_Adam Hart_16384.pdf"}),
            json!({"document_id": "doc-2", "document_name": "invoice_Aaron Bergman_36258.pdf"}),
        ];
        let combined = "=== Document: invoice_Adam Hart_16384.pdf ===\n\
                        Invoice total: $540.00\n\n\
                        === Document: invoice_Aaron Bergman_36258.pdf ===\n\
                        Invoice total: $120.00";
        let result = format(combined, Some(&citations), false);

        // Must contain DOCUMENT CONTEXT label (not TOOL CONTEXT)
        assert!(result.contains("DOCUMENT CONTEXT"));

        // Must list source document names explicitly
        assert!(result.contains("invoice_Adam Hart_16384.pdf"));
        assert!(result.contains("invoice_Aaron Bergman_36258.pdf"));

        // Multi-document responses must not be biased toward a single source
        assert!(result.contains("MULTIPLE documents"));
        assert!(result.contains("ALL relevant"));
        // The filename is only an identifier, not a description of the contents
        assert!(result.contains("may not describe the contents"));
    }

    /// With a single document, the LLM must reference it by name.
    #[test]
    fn test_single_citation_instructs_reference_by_name() {
        let citations = vec![json!({"document_id": "doc-1", "document_name": "invoice_Adam Hart_16384.pdf"})];
        let result = format("Invoice total: $540.00", Some(&citations), false);
        assert!(result.contains("DOCUMENT CONTEXT"));
        assert!(result.contains("MUST reference"));
        assert!(result.contains("may not describe the contents"));
    }

    /// Without citations, tool output is labelled as TOOL RESULTS.
    #[test]
    fn test_no_citations_uses_tool_results() {
        let result = format("Some tool output", None, true);
        assert!(result.contains("TOOL RESULTS"));
        assert!(!result.contains("DOCUMENT CONTEXT"));
    }

    /// Without citations or tool results flag, context passes through plain.
    #[test]
    fn test_no_citations_no_tool_results_passes_through() {
        let result = format("Some tool output", None, false);
        assert!(result.contains("Some tool output"));
        assert!(!result.contains("DOCUMENT CONTEXT"));
    }

    /// An empty citations list with tool results flag should use TOOL RESULTS.
    #[test]
    fn test_empty_citations_list_uses_tool_results() {
        let result = format("Some tool output", Some(&[]), true);
        assert!(result.contains("TOOL RESULTS"));
    }
}
