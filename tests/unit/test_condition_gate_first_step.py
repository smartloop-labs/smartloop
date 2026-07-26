"""A first tool step must not be skipped by a skill-extracted condition.

Regression: a lone fetch step matched via the fetch-url skill set
``ctx.skill_context``; the resolver then read the skill's applicability
clause ("when the user requests a URL, fetch it") as a runtime condition and
skipped the only step, so the answer was synthesized from no fetched content.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke = MagicMock(return_value="YES")
    llm.get_max_tokens = MagicMock(return_value=4096)
    llm.get_num_tokens = MagicMock(return_value=100)
    return llm


@pytest.fixture
def orchestrator(mock_llm):
    with patch("smartloop.agents.orchestrator.LLMExecutor"), \
         patch("smartloop.agents.orchestrator.DAGExecutor"), \
         patch("smartloop.agents.orchestrator.AppSettings", create=True), \
         patch("smartloop.config.app_settings.AppSettings"):
        from smartloop.agents.orchestrator import SLPOrchestrator
        orch = SLPOrchestrator(orchestrator_llm=mock_llm)
    return orch


def _ctx(skill_context):
    from smartloop.agents.orchestrator import _OrchestratorContext
    ctx = _OrchestratorContext(
        user_message="summarize https://example.com/post",
        mcp_servers=[],
        available_servers=[],
        effective_cwd="/tmp",
        document_context=None,
        project_dir=None,
        model_service=MagicMock(),
        conversation_history=[],
        has_attachments=False,
    )
    ctx.skill_context = skill_context
    return ctx


def _fetch_step():
    from smartloop.agents.planner import PlanStep, StepType
    return PlanStep(
        step=1, step_type=StepType.TOOL, tool_name="fetch",
        server_id="", server_name="fetch",
        instruction="Fetch the URL and extract its content.",
    )


class TestExtractorHardening:
    def test_bare_yes_is_not_a_condition(self, orchestrator):
        """A reply with no '?' is malformed → no condition."""
        assert orchestrator._extract_tool_condition_from_text(
            "when user requests a url use fetch tool", "fetch"
        ) is None

    def test_real_question_is_a_condition(self, orchestrator, mock_llm):
        mock_llm.invoke.return_value = "Were discrepancies found in the results?"
        cond = orchestrator._extract_tool_condition_from_text(
            "after analysis, if discrepancies found call fetch", "fetch"
        )
        assert cond == "llm:Were discrepancies found in the results?"


class TestFirstStepNotGated:
    def test_first_step_runs_despite_skill_condition(self, orchestrator):
        """has_upstream=False → the step resolves instead of being skipped."""
        ctx = _ctx("when the user requests a url use fetch tool to fetch it")
        step = _fetch_step()
        # Isolate the condition gate from real arg resolution. The resolver
        # returns a (server, tool_name, arguments) tuple on success.
        server = MagicMock(id="srv")
        orchestrator._resolve_tool_args = MagicMock(
            return_value=(server, "fetch", {"url": "x"})
        )
        orchestrator.tool_caller.find_tool_schema = MagicMock(return_value=None)
        # A spy on the extractor proves it is never consulted for a first step.
        orchestrator._extract_tool_condition_from_text = MagicMock(return_value="llm:x?")

        handler = orchestrator._make_resolve_handler(
            ctx, step, MagicMock(), has_upstream=False
        )
        result = handler({})

        assert not orchestrator._extract_tool_condition_from_text.called
        assert orchestrator._resolve_tool_args.called
        assert result.data.get("skip") is False
        assert result.data.get("arguments") == {"url": "x"}

    def test_downstream_step_still_gated_when_no_prior_results(self, orchestrator):
        """has_upstream=True with an extracted condition and no prior results
        → the step is skipped (existing behavior preserved)."""
        ctx = _ctx("after analysis, if discrepancies found call fetch")
        orchestrator.agent_llm.invoke = MagicMock(
            return_value="Were discrepancies found?"
        )
        step = _fetch_step()

        handler = orchestrator._make_resolve_handler(
            ctx, step, MagicMock(), has_upstream=True
        )
        result = handler({})

        assert result.data.get("skip") is True
