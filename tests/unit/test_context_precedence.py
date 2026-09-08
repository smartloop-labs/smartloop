"""Local sources lead the explore context, and none of them cancel another.

Three failure modes this pins down:
  * web results ordered ahead of the user's own connected-server data;
  * a live MCP tool result suppressing the web search entirely, so a query
    spanning both came back with half its context;
  * the live MCP call itself being skipped, which is what a locally indexed
    copy of the server's content used to do — a question is now put to every
    source, the servers included.
"""
import inspect

from smartloop.agents.mcp import explorer
from smartloop.agents.orchestration import pipeline_builder


class TestExploreOrdering:
    def test_web_is_appended_after_every_local_source(self):
        src = inspect.getsource(pipeline_builder.PipelineBuilder)
        start = src.index("if step.step_type == StepType.EXPLORE:")
        # Bound the block at the join, so the window cannot silently clip a
        # source and make the ordering assertion vacuous.
        end = src.index('content = "\\n\\n---\\n\\n".join(parts)', start)
        block = src[start:end]

        def pos(fragment: str) -> int:
            assert fragment in block, f"missing from explore assembly: {fragment}"
            return block.index(fragment)

        web = pos("search_ctx.web_context")
        for local in (
            "search_ctx.vector_context",
            "search_ctx.workspace_context",
            "search_ctx.mcp_context",
            "search_ctx.feedback_context",
        ):
            assert pos(local) < web, f"{local} must be appended before web results"


class TestSourcesCombine:
    def test_a_live_mcp_result_no_longer_cancels_web_search(self):
        src = inspect.getsource(explorer)
        start = src.index("if self.web_search_agent.is_enabled():")
        block = src[start:start + 500]

        assert "Skipping web search" not in block, (
            "a live MCP tool result must not suppress the web search"
        )

    def test_the_follow_up_gate_is_still_applied(self):
        """Combining sources must not lose the "now make it a table" check —
        that one is about the *query*, not about competing sources."""
        src = inspect.getsource(explorer)
        start = src.index("if self.web_search_agent.is_enabled():")
        block = src[start:start + 500]

        assert "_needs_web_search" in block

    def test_the_live_mcp_call_is_submitted_whenever_a_server_is_connected(self):
        """Nothing else matching may skip it: the connected servers are one of
        the sources a question is put to, not a fallback for when the local
        ones come back empty."""
        src = inspect.getsource(explorer.Explorer.collect_parallel)
        start = src.index("f_mcp = None")
        block = src[start:start + 300]

        assert "if mcp_servers:" in block
        assert "self._run_mcp" in block
        # The only condition on it is "is there a server at all".
        assert "mcp_document_context" not in src
