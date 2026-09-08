"""Tests for :class:`JudgeStep` — relevance evaluation of context.

Every piece of retrieved context and every step output is judged against the
user's request before it is appended to the answer context.  Without that
check an unrelated document reaches the prompt fenced as DOCUMENT CONTEXT, and
the model answers that it has no context for the question instead of answering
the question.
"""
import logging
from unittest.mock import MagicMock, patch

import pytest

from smartloop.agents.mcp.explorer import SearchContext
from smartloop.agents.nodes import LLMDecisionNode, NodeResult
from smartloop.agents.orchestration.judge_step import JudgeStep
from smartloop.agents.planner import PlanStep, StepType


def _make_orchestrator():
    llm = MagicMock()
    llm.get_num_tokens = MagicMock(return_value=100)
    llm.get_max_tokens = MagicMock(return_value=8192)
    llm.invoke = MagicMock(return_value="RELEVANT")
    with patch("smartloop.agents.orchestrator.DAGExecutor"):
        from smartloop.agents.orchestrator import SLPOrchestrator
        return SLPOrchestrator(orchestrator_llm=llm)


def _ctx(orch, user_message="what are things to do in madrid"):
    from smartloop.agents.orchestrator import _OrchestratorContext
    return _OrchestratorContext(
        user_message=user_message,
        mcp_servers=[],
        available_servers=[],
        effective_cwd="/tmp",
        document_context=None,
        project_dir=None,
        model_service=MagicMock(),
        conversation_history=[],
        has_attachments=False,
    )


def _verdicts(orch, mapping, default="RELEVANT"):
    """Route the judge's reply off whichever fixture text is in the prompt."""
    def invoke(prompt, **kwargs):
        for needle, verdict in mapping.items():
            if needle in prompt:
                return verdict
        return default
    orch.agent_llm.invoke = MagicMock(side_effect=invoke)


def _judge():
    """A judge wired to a mock router, with no orchestrator around it."""
    from smartloop.agents.orchestration.model_router import ModelRouter
    llm = MagicMock()
    llm.invoke = MagicMock(return_value="RELEVANT")
    return JudgeStep(ModelRouter(orchestrator_llm=llm, agent_llm=llm)), llm


# ------------------------------------------------------------------ #
# JudgeStep.serves_intent                                            #
# ------------------------------------------------------------------ #

class TestServesIntent:
    def test_relevant_content_is_kept(self):
        judge, llm = _judge()
        llm.invoke = MagicMock(return_value="RELEVANT")
        assert judge.serves_intent(
            "things to do in madrid", "documents", "Retiro Park and the Prado",
        ) is True

    def test_irrelevant_content_is_rejected(self):
        judge, llm = _judge()
        llm.invoke = MagicMock(return_value="NOT_RELEVANT")
        assert judge.serves_intent(
            "things to do in madrid", "documents", "Q3 invoice for Adam Hart",
        ) is False

    def test_empty_content_is_rejected_without_an_llm_call(self):
        judge, llm = _judge()
        assert judge.serves_intent("madrid", "documents", "   ") is False
        llm.invoke.assert_not_called()

    def test_llm_failure_keeps_the_context(self):
        judge, llm = _judge()
        llm.invoke = MagicMock(side_effect=RuntimeError("no model"))
        assert judge.serves_intent("madrid", "documents", "text") is True

    def test_unparseable_verdict_keeps_the_context(self):
        judge, llm = _judge()
        llm.invoke = MagicMock(return_value="hmm, maybe?")
        assert judge.serves_intent("madrid", "documents", "text") is True

    def test_an_echoed_instruction_is_not_a_rejection(self):
        """The reply that broke monday.com retrieval.

        The judge's own prompt says "Reply RELEVANT or NOT_RELEVANT", and a
        small model quoting it back was read as a rejection by the old
        substring check — so a 0.111-distance match for "list my boards on
        monday.com" was discarded and answered with "I don't have access to
        your Monday.com boards".
        """
        judge, llm = _judge()
        llm.invoke = MagicMock(return_value="Reply RELEVANT or NOT_RELEVANT")
        assert judge.serves_intent(
            "list my boards on monday.com", "MCP content", "boards: Tickets",
        ) is True

    def test_a_prose_answer_that_means_relevant_is_kept(self):
        judge, llm = _judge()
        llm.invoke = MagicMock(
            return_value="The context is RELEVANT, not NOT_RELEVANT.",
        )
        assert judge.serves_intent("madrid", "documents", "Retiro Park") is True

    def test_a_rejection_that_explains_itself_still_rejects(self):
        """A well-formed answer leads with the verdict and then reasons, and the
        reasoning naming the other label must not turn the verdict into a
        keep."""
        judge, llm = _judge()
        llm.invoke = MagicMock(
            return_value="NOT_RELEVANT — this is about Trello, not relevant "
                         "to the monday.com question",
        )
        assert judge.serves_intent(
            "list my boards on monday.com", "MCP content", "name: Pay Christy",
        ) is False

    def test_a_decorated_verdict_still_counts(self):
        judge, llm = _judge()
        llm.invoke = MagicMock(return_value="**NOT_RELEVANT**")
        assert judge.serves_intent("madrid", "documents", "invoice") is False

    def test_a_dropped_source_logs_what_the_model_replied(self, caplog):
        """Without the reply in the log, a misjudging model and a misread
        verdict produce the same line."""
        judge, llm = _judge()
        llm.invoke = MagicMock(return_value="NOT_RELEVANT, wrong subject")
        with caplog.at_level(logging.INFO):
            judge.serves_intent("madrid", "documents", "invoice")
        assert "wrong subject" in caplog.text

    def test_a_verdictless_reply_is_logged_as_kept(self, caplog):
        judge, llm = _judge()
        llm.invoke = MagicMock(return_value="MONDAY.COM")
        with caplog.at_level(logging.INFO):
            assert judge.serves_intent("madrid", "documents", "text") is True
        assert "no verdict" in caplog.text
        assert "MONDAY.COM" in caplog.text

    def test_verdict_is_memoized_per_request(self):
        judge, llm = _judge()
        llm.invoke = MagicMock(return_value="NOT_RELEVANT")
        for _ in range(3):
            judge.serves_intent("madrid", "documents", "invoice")
        assert llm.invoke.call_count == 1

    def test_begin_request_clears_the_memo(self):
        """Verdicts are scoped to one request — a new one re-judges."""
        judge, llm = _judge()
        llm.invoke = MagicMock(return_value="NOT_RELEVANT")
        judge.serves_intent("madrid", "documents", "invoice")
        judge.begin_request()
        judge.serves_intent("madrid", "documents", "invoice")
        assert llm.invoke.call_count == 2


# ------------------------------------------------------------------ #
# JudgeStep.evaluate_sources                                         #
# ------------------------------------------------------------------ #

class TestEvaluateSources:
    def test_each_source_is_judged_on_its_own(self):
        """One irrelevant source must not take the relevant ones with it."""
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        _verdicts(orch, {"OLD INVOICE": "NOT_RELEVANT"})
        search_ctx = SearchContext(
            workspace_context="OLD INVOICE for Adam Hart",
            feedback_context="Madrid trip notes: visit the Prado",
        )

        dropped = orch._judge.evaluate_sources(ctx, search_ctx)

        assert dropped == ["workspace files"]
        assert search_ctx.workspace_context == ""
        assert search_ctx.feedback_context == "Madrid trip notes: visit the Prado"

    def test_all_irrelevant_sources_are_cleared(self):
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        orch.agent_llm.invoke = MagicMock(return_value="NOT_RELEVANT")
        search_ctx = SearchContext(
            workspace_context="unrelated",
            feedback_context="unrelated",
            mcp_context="unrelated",
        )

        dropped = orch._judge.evaluate_sources(ctx, search_ctx)

        assert dropped == ["workspace files", "feedback"]
        # The MCP result is exempt — it is the user's connected-app data.
        assert search_ctx.mcp_context == "unrelated"

    def test_a_live_tool_result_is_never_judged(self):
        """A tool call is deliberate and its result is definite — not recall.

        The judge runs on the small decision model and fails toward dropping,
        so gating tool output means one bad verdict discards the data the
        request went and fetched.  A wrong tool is a routing problem; binning
        its result does not produce the right one.
        """
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        orch.agent_llm.invoke = MagicMock(return_value="NOT_RELEVANT")
        search_ctx = SearchContext(
            mcp_context="sprint backlog: move model hosting to R2",
            web_context="The Hasselblad X2D II has a 100MP sensor.",
        )

        dropped = orch._judge.evaluate_sources(ctx, search_ctx)

        assert dropped == []
        assert search_ctx.mcp_context == "sprint backlog: move model hosting to R2"
        # Web results were fetched for this query — never judged, never dropped.
        assert search_ctx.web_context == (
            "The Hasselblad X2D II has a 100MP sensor."
        )

    def test_mcp_context_alone_does_not_start_the_judge(self):
        """Nothing judgeable left, so the judge stays off the critical path."""
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        search_ctx = SearchContext(mcp_context="open tickets = 4")

        assert orch._judge.applies_to(ctx, search_ctx) is False

    def test_an_empty_search_context_never_reaches_the_llm(self):
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        orch.agent_llm.invoke = MagicMock()

        assert orch._judge.evaluate_sources(ctx, SearchContext()) == []
        orch.agent_llm.invoke.assert_not_called()

    def test_web_context_is_never_judged(self):
        """Web results were fetched for this query — on-topic by construction."""
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        orch.agent_llm.invoke = MagicMock(return_value="NOT_RELEVANT")
        search_ctx = SearchContext(web_context="live page text")

        assert orch._judge.evaluate_sources(ctx, search_ctx) == []
        assert search_ctx.web_context == "live page text"
        orch.agent_llm.invoke.assert_not_called()

    def test_document_context_is_left_to_the_fanout_judge(self):
        """Documents get a per-document judge in the fan-out — not re-judged."""
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        orch.agent_llm.invoke = MagicMock(return_value="NOT_RELEVANT")
        search_ctx = SearchContext(vector_context="some document text")

        assert orch._judge.evaluate_sources(ctx, search_ctx) == []
        assert search_ctx.vector_context == "some document text"
        orch.agent_llm.invoke.assert_not_called()


# ------------------------------------------------------------------ #
# Eval-step placement in the DAG                                      #
# ------------------------------------------------------------------ #

class TestEvalStepPlacement:
    def test_eval_step_runs_on_the_planned_step_path(self):
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        search_ctx = SearchContext(workspace_context="some workspace file")
        step = PlanStep(
            step=1, step_type=StepType.EXPLORE, instruction="explore",
        )

        nodes = orch._pipeline.build_step_nodes(
            ctx, [step], search_ctx, parent=LLMDecisionNode(),
        )

        # eval → explore → reason, with the eval step ahead of the step
        # that reads these sources off search_ctx.
        assert [type(n).__name__ for n, _ in nodes] == [
            "LLMDecisionNode", "SearchNode", "TransformNode",
        ]

    def test_no_eval_step_when_there_is_nothing_to_judge(self):
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        step = PlanStep(
            step=1, step_type=StepType.EXPLORE, instruction="explore",
        )

        nodes = orch._pipeline.build_step_nodes(
            ctx, [step], SearchContext(), parent=LLMDecisionNode(),
        )

        assert [type(n).__name__ for n, _ in nodes] == [
            "SearchNode", "TransformNode",
        ]

    def test_no_eval_step_when_a_skill_supplies_the_context(self):
        """A skill's own context is deliberate — the judge must not filter it."""
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        ctx.skill_context = "Follow this procedure"
        search_ctx = SearchContext(workspace_context="some workspace file")

        assert orch._judge.applies_to(ctx, search_ctx) is False


# ------------------------------------------------------------------ #
# Per-step judging in the reason handler                              #
# ------------------------------------------------------------------ #

def _dep(node_id, tool_name, content, is_tool=False):
    return NodeResult(node_id=node_id, success=True, data={
        "tool_name": tool_name,
        "content": content,
        "raw": content,
        "is_tool_result": is_tool,
    })


class TestReasonHandlerJudging:
    def test_step_output_that_misses_the_intent_is_not_appended(self):
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        _verdicts(orch, {"OLD INVOICE": "NOT_RELEVANT"})

        handler = orch._pipeline.make_reason_handler(ctx, SearchContext(), False)
        result = handler({
            "a": _dep("a", "explore", "Madrid: the Prado, Retiro Park"),
            "b": _dep("b", "file-read", "OLD INVOICE for Adam Hart"),
        })

        contents = [e["content"] for e in result.data["accumulated"]]
        assert contents == ["Madrid: the Prado, Retiro Park"]

    def test_relevant_step_outputs_are_all_kept(self):
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        orch.agent_llm.invoke = MagicMock(return_value="RELEVANT")

        handler = orch._pipeline.make_reason_handler(ctx, SearchContext(), False)
        result = handler({
            "a": _dep("a", "explore", "Madrid: the Prado"),
            "b": _dep("b", "web_search", "Madrid: Retiro Park"),
        })

        assert len(result.data["accumulated"]) == 2

    def test_rejecting_every_output_keeps_them_all(self):
        """An all-reject verdict is far likelier to be a misfiring judge."""
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        orch.agent_llm.invoke = MagicMock(return_value="NOT_RELEVANT")

        handler = orch._pipeline.make_reason_handler(ctx, SearchContext(), False)
        result = handler({
            "a": _dep("a", "explore", "some content"),
            "b": _dep("b", "file-read", "other content"),
        })

        assert len(result.data["accumulated"]) == 2

    def test_tool_results_survive_the_accumulated_split(self):
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        orch.agent_llm.invoke = MagicMock(return_value="RELEVANT")

        handler = orch._pipeline.make_reason_handler(ctx, SearchContext(), False)
        result = handler({
            "a": _dep("a", "explore", "Madrid: the Prado"),
            "b": _dep("b", "get_weather", "18C and sunny", is_tool=True),
        })

        assert [e["content"] for e in result.data["tool_results"]] == ["18C and sunny"]
        assert all("_is_tool" not in e for e in result.data["accumulated"])

    def test_a_tool_step_output_is_never_judged(self):
        """The other half of the tool exemption: a TOOL step in the plan.

        Same reasoning as the live explore call — the plan resolved a tool,
        the server ran it, and the result is what the request went to get.
        """
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        orch.agent_llm.invoke = MagicMock(return_value="NOT_RELEVANT")

        handler = orch._pipeline.make_reason_handler(ctx, SearchContext(), False)
        result = handler({
            "a": _dep("a", "list_boards", "boards: Tickets, Roadmap", is_tool=True),
        })

        assert [e["content"] for e in result.data["tool_results"]] == [
            "boards: Tickets, Roadmap",
        ]

    def test_a_rejected_retrieval_step_still_drops_alongside_a_tool_result(self):
        """The exemption is for tool output only — retrieval is still judged."""
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        _verdicts(orch, {"OLD INVOICE": "NOT_RELEVANT"})

        handler = orch._pipeline.make_reason_handler(ctx, SearchContext(), False)
        result = handler({
            "a": _dep("a", "file-read", "OLD INVOICE for Adam Hart"),
            "b": _dep("b", "get_weather", "18C and sunny", is_tool=True),
        })

        contents = [e["content"] for e in result.data["accumulated"]]
        assert "OLD INVOICE for Adam Hart" not in contents
        assert [e["content"] for e in result.data["tool_results"]] == ["18C and sunny"]

    def test_skill_context_bypasses_the_judge(self):
        """A skill's own steps are its procedure — not candidates to filter."""
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        ctx.skill_context = "Follow this procedure"
        orch.agent_llm.invoke = MagicMock(return_value="NOT_RELEVANT")

        handler = orch._pipeline.make_reason_handler(ctx, SearchContext(), False)
        result = handler({"a": _dep("a", "explore", "step output")})

        assert len(result.data["accumulated"]) == 1


# ------------------------------------------------------------------ #
# Document fan-out: all documents judged irrelevant                   #
# ------------------------------------------------------------------ #

class TestDocReduceAllIrrelevant:
    def test_clears_document_context_when_every_document_is_dropped(self):
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        search_ctx = SearchContext(
            vector_context="RAW: Q3 invoice for Adam Hart",
            citations=[{"document_name": "invoice.pdf"}],
        )

        handler = orch._doc_fanout.make_reduce_handler(ctx, search_ctx, ["batch_0"])
        # An empty extract is the batch handler's "NONE" — the document was
        # judged irrelevant to the request.
        handler({"batch_0": NodeResult(
            node_id="batch_0", success=True,
            data={"doc_names": ["invoice.pdf"], "content": ""},
        )})

        assert search_ctx.vector_context == ""
        assert search_ctx.citations is None
        assert search_ctx.cited_documents is None

    def test_keeps_the_merged_extract_when_a_document_survives(self):
        orch = _make_orchestrator()
        ctx = _ctx(orch)
        search_ctx = SearchContext(vector_context="RAW")

        handler = orch._doc_fanout.make_reduce_handler(ctx, search_ctx, ["batch_0"])
        handler({"batch_0": NodeResult(
            node_id="batch_0", success=True,
            data={"doc_names": ["madrid.pdf"], "content": "Visit the Prado"},
        )})

        assert "Visit the Prado" in search_ctx.vector_context
        assert search_ctx.cited_documents == ["madrid.pdf"]
