//! Port of `tests/unit/test_context_precedence.py`.
//!
//! Local sources lead the explore context, and none of them cancel another.
//!
//! Three failure modes this pins down:
//!   * web results ordered ahead of the user's own connected-server data;
//!   * a live MCP tool result suppressing the web search entirely, so a query
//!     spanning both came back with half its context;
//!   * the live MCP call itself being skipped, which is what a locally indexed
//!     copy of the server's content used to do — a question is now put to every
//!     source, the servers included.
//!
//! Like the Python original (which used `inspect.getsource`), these read the
//! Rust sources of `pipeline_builder` and `explorer` and check the shape of
//! the code that assembles and gathers the explore context.

use regex::Regex;

const PIPELINE_BUILDER_SRC: &str = include_str!("../../src/agents/orchestration/pipeline_builder.rs");
const EXPLORER_SRC: &str = include_str!("../../src/agents/mcp/explorer.rs");

/// `src[start..start + len]`, clipped to the end and to a char boundary.
fn window(src: &str, start: usize, len: usize) -> &str {
    let mut end = (start + len).min(src.len());
    while !src.is_char_boundary(end) {
        end -= 1;
    }
    &src[start..end]
}

/// The source of `fn <name>` up to the next method at the same indent.
fn method_source<'a>(src: &'a str, name: &str) -> &'a str {
    let marker = format!("fn {name}(");
    let start = src
        .find(&marker)
        .unwrap_or_else(|| panic!("`{marker}` not found in source"));
    let rest = &src[start + marker.len()..];
    let next = [rest.find("\n    pub fn "), rest.find("\n    fn "), rest.find("\n    pub(crate) fn ")]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(rest.len());
    &src[start..start + marker.len() + next]
}

mod test_explore_ordering {
    use super::*;

    #[test]
    fn test_web_is_appended_after_every_local_source() {
        let src = PIPELINE_BUILDER_SRC;
        // The assembly branch (not the node-construction match arm that
        // also names `StepType::Explore`).
        let start = src
            .find("if step.step_type == StepType::Explore {")
            .expect("pipeline builder never handles StepType::Explore");
        // Bound the block at the join, so the window cannot silently clip a
        // source and make the ordering assertion vacuous.
        let join = Regex::new(r#"\w+\.join\("\\n\\n---\\n\\n"\)"#).unwrap();
        let end = join
            .find_at(src, start)
            .map(|m| m.start())
            .expect("explore assembly never joins its parts with \"\\n\\n---\\n\\n\"");
        let block = &src[start..end];

        let pos = |fragment: &str| -> usize {
            block
                .find(fragment)
                .unwrap_or_else(|| panic!("missing from explore assembly: {fragment}"))
        };

        // The Rust code reads a snapshot of the shared search context taken
        // under its lock (`let sc = search_ctx.lock().clone();`), so the
        // fields are read through `sc.` rather than `search_ctx.`.
        assert!(
            block.contains("let sc = search_ctx.lock()"),
            "explore assembly must read the shared search context"
        );
        let web = pos("sc.web_context");
        for local in [
            "sc.vector_context",
            "sc.workspace_context",
            "sc.mcp_context",
            "sc.feedback_context",
        ] {
            assert!(pos(local) < web, "{local} must be appended before web results");
        }
    }
}

mod test_sources_combine {
    use super::*;

    fn web_gate_block() -> &'static str {
        let src = EXPLORER_SRC;
        let start = src
            .find("web_search_agent.is_enabled()")
            .expect("explorer never checks web_search_agent.is_enabled()");
        window(src, start, 500)
    }

    #[test]
    fn test_a_live_mcp_result_no_longer_cancels_web_search() {
        let block = web_gate_block();
        assert!(
            !block.contains("Skipping web search"),
            "a live MCP tool result must not suppress the web search"
        );
    }

    /// Combining sources must not lose the "now make it a table" check — that
    /// one is about the *query*, not about competing sources.
    #[test]
    fn test_the_follow_up_gate_is_still_applied() {
        let block = web_gate_block();
        assert!(block.contains("needs_web_search"));
    }

    /// Nothing else matching may skip it: the connected servers are one of the
    /// sources a question is put to, not a fallback for when the local ones
    /// come back empty.
    ///
    /// A matched skill is the one exception, and it is not a source coming back
    /// with something: the skill's plan makes its tool calls as steps.
    #[test]
    fn test_the_live_mcp_call_is_submitted_whenever_a_server_is_connected() {
        let src = method_source(EXPLORER_SRC, "collect_parallel");
        let start = src.find("f_mcp").expect("collect_parallel never submits the MCP leg (f_mcp)");
        // The whole `let f_mcp = ...;` statement (the Rust spelling is wordier
        // than Python's three lines, so bound it at the blank line ending it
        // rather than at a fixed width).
        let end = src[start..].find("\n\n").map_or(src.len(), |i| start + i);
        let block = window(src, start, (end - start).min(1200));

        // The gate on the MCP leg is "servers connected and no matched skill"
        // (Python: `if mcp_servers and not matched_skills:`).  In Rust either
        // an `if` over both, or a `match` on the non-empty server list with an
        // `if !skill_matched` guard, where `skill_matched` is derived from a
        // non-empty `matched_skills`.
        let if_gate = Regex::new(r"(?s)\bif\b[^{]*mcp_servers[^{]*!\s*\w*matched\w*[^{]*\{").unwrap();
        let match_gate = Regex::new(
            r"(?s)\bmatch\s+mcp_servers\.filter\(\|\w+\|\s*!\w+\.is_empty\(\)\)\s*\{\s*Some\(\w+\)\s+if\s+!skill_matched\s*=>",
        )
        .unwrap();
        assert!(
            if_gate.is_match(block) || match_gate.is_match(block),
            "MCP leg must be gated on mcp_servers and !matched_skills:\n{block}"
        );
        if !block.contains("matched_skills") {
            let derived = Regex::new(r"let skill_matched = matched_skills\.is_some_and\(\|\w+\| !\w+\.is_empty\(\)\)").unwrap();
            assert!(derived.is_match(src), "skill_matched must mean \"a skill matched\"");
        }
        // `run_mcp_with_deferred` is `run_mcp` that also hands on the write
        // tools it resolved and left for a planned step.
        assert!(block.contains("run_mcp(") || block.contains("run_mcp_with_deferred("));
        // Beyond that, the only condition on it is "is there a server at all".
        assert!(!src.contains("mcp_document_context"));
    }
}
