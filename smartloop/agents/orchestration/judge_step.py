"""JudgeStep — LLM-as-judge relevance evaluation of context before it is used.

Retrieval is recall-oriented — a vector store returns its nearest neighbours
whether or not any of them is actually about the question — so without a check
an unrelated document reaches the prompt fenced as DOCUMENT CONTEXT under
instructions to answer from it.  The model then correctly reports that it
cannot ("I don't have context") instead of falling through to what it knows.

This module is that check.  It runs at two points: as a step in the DAG,
judging each retrieval source before any later step reads it, and inline in
the reason handler, judging each step's output before it is appended.
"""
import logging
from typing import Callable, Dict, List, Optional

from smartloop.agents.nodes import AgentNode, LLMDecisionNode, NodeResult
from smartloop.agents.mcp.explorer import SearchContext
from smartloop.agents.orchestration.context import _OrchestratorContext
from smartloop.agents.orchestration.model_router import (
    DECISION_MAX_TOKENS, ModelRouter,
)
from smartloop.agents.orchestration.step import Step, StepNodes

logger = logging.getLogger(__name__)

JUDGE_STEP_NODE = "judge_step"


class JudgeStep(Step):
    """Judges whether a piece of context serves the user's request.

    One decision-model call per distinct piece of context, memoized for the
    life of a request.  Every verdict fails open: an LLM error, an empty reply
    or an unparseable answer all keep the context, because dropping real
    context is the worse failure and this runs on the small model.
    """

    # How much of a piece of context the judge reads.  The opening of a
    # retrieved chunk is what decides its subject; feeding the whole thing
    # would blow the decision model's window for no extra signal.
    JUDGE_SAMPLE_CHARS = 2000

    # Retrieval sources judged individually before they reach the answer
    # context, as ``(SearchContext attribute, human label)``.
    #
    # ``mcp_context`` is a live tool result gathered during explore, and is
    # judged like the rest.  That the router chose to invoke a tool says only
    # that it looked applicable, not that its output answered the question —
    # and this source is placed AHEAD of web results in the final context, so
    # an off-topic tool result does not merely sit alongside a good web answer,
    # it precedes it.
    #
    # ``vector_context`` is deliberately absent: retrieved documents already
    # get their own per-document judge in the fan-out, where each one is
    # extracted against the request and dropped on a ``NONE`` reply (see
    # :class:`~smartloop.agents.orchestration.document_fanout.DocumentFanoutStep`).
    # Judging the merged extract again here would spend a second LLM call to
    # re-decide what the fan-out just decided with far more context.
    #
    # ``web_context`` is absent for a different reason: it was fetched for
    # THIS query moments ago, so it is on-topic by construction.
    _JUDGED_SOURCES = (
        ("workspace_context", "workspace files"),
        ("feedback_context", "feedback"),
        ("mcp_context", "connected tools"),
        ("mcp_document_context", "MCP content"),
    )

    name = "judge"

    def __init__(self, models: ModelRouter):
        self._models = models
        # Verdicts memoized for the life of a request: the same piece of
        # context is judged at more than one point in the DAG (the eval step,
        # then again as a step output), so this keeps each piece to one call.
        self._judge_cache: Dict[tuple, bool] = {}

    def begin_request(self) -> None:
        """Clear memoized verdicts — they are scoped to one user request."""
        self._judge_cache = {}

    def serves_intent(
        self, user_message: str, label: str, content: str,
    ) -> bool:
        """True when ``content`` is worth appending to the answer context.

        Retrieval is recall-oriented — a vector store returns its nearest
        neighbours whether or not any of them is actually about the question —
        so an unrelated document reaches the prompt fenced as DOCUMENT CONTEXT
        under instructions to answer from it.  The model then correctly reports
        that it cannot ("I don't have context"), instead of falling through to
        what it knows.  This judge is the check that keeps such a chunk out.

        Fails open: an LLM error, an empty reply or an unparseable verdict all
        keep the context.  Dropping real context is the worse failure, and the
        judge runs on the small decision model.

        Verdicts are memoized per request, so a piece of context judged at the
        retrieval gate is not re-judged when it resurfaces as a step output.
        """
        text = (content or "").strip()
        if not text:
            return False

        key = (label, text)
        cached = self._judge_cache.get(key)
        if cached is not None:
            return cached

        prompt = f"""\
<system>
You are a relevance judge in a retrieval pipeline. One piece of retrieved
context is shown below. Decide whether it serves the user's request.

Reply with exactly one word:
RELEVANT — it contains information that helps answer the request.
NOT_RELEVANT — it is about a different subject and would not help.

Judge the subject matter, not the wording. Context that merely shares a few
words with the request but is about something else is NOT_RELEVANT.
</system>
<user>
User request: {user_message}

Retrieved context ({label}):
{text[:self.JUDGE_SAMPLE_CHARS]}

Does this context serve the user's request? Reply RELEVANT or NOT_RELEVANT.
</user>"""

        serves = True
        try:
            reply = str(self._models.agent_llm.invoke(
                prompt, max_tokens=DECISION_MAX_TOKENS,
            )).strip().upper()
            if "NOT_RELEVANT" in reply or "NOT RELEVANT" in reply:
                serves = False
        except Exception:
            logger.debug(
                "[judge] %s: LLM call failed — keeping context",
                label, exc_info=True,
            )

        if not serves:
            logger.info(
                "[judge] %s (%d chars) does not serve the request — dropped",
                label, len(text),
            )
        self._judge_cache[key] = serves
        return serves

    def evaluate_sources(
        self,
        ctx: _OrchestratorContext,
        search_ctx: SearchContext,
    ) -> List[str]:
        """Judge each retrieval source on its own, clearing the ones that
        don't serve the request.  Returns the labels dropped.

        Judged per source rather than over the concatenation: one verdict for
        the whole blob meant a single relevant source kept three irrelevant
        ones alive, and a single irrelevant one could sink the rest.
        """
        dropped: List[str] = []
        for attr, label in self._JUDGED_SOURCES:
            text = getattr(search_ctx, attr, "") or ""
            if not text:
                continue
            if self.serves_intent(ctx.user_message, label, text):
                continue
            setattr(search_ctx, attr, "")
            dropped.append(label)

        if dropped:
            self._models.emit(
                "filter", "processing",
                f"Set aside context that does not match the request: "
                f"{', '.join(dropped)}",
            )
        return dropped

    def has_judgeable_sources(self, search_ctx: SearchContext) -> bool:
        """True when at least one judged retrieval source has content."""
        return any(
            getattr(search_ctx, attr, "") for attr, _ in self._JUDGED_SOURCES
        )

    def applies_to(
        self,
        ctx: _OrchestratorContext,
        search_ctx: Optional[SearchContext],
    ) -> bool:
        """True when there is retrieved context worth judging.

        False when a matched skill supplies its own context deliberately and
        must not have it filtered out from under it, or when no judged source
        has content — keeping the judge off the critical path of requests it
        could not affect.
        """
        if ctx.skill_context:
            return False
        return self.has_judgeable_sources(search_ctx)

    def build(
        self,
        ctx: _OrchestratorContext,
        search_ctx: Optional[SearchContext],
        parent: AgentNode,
    ) -> StepNodes:
        """Build the single node that judges every retrieval source."""
        node = LLMDecisionNode(depends_on=[parent])
        return StepNodes(
            pairs=[(node, self.make_handler(ctx, search_ctx))], parent=node,
        )

    def make_handler(
        self,
        ctx: _OrchestratorContext,
        search_ctx: SearchContext,
    ) -> Callable:
        """Create the handler that judges and clears irrelevant sources."""
        def handler(deps: Dict) -> NodeResult:
            self.evaluate_sources(ctx, search_ctx)
            return NodeResult(node_id=JUDGE_STEP_NODE, success=True, data={})

        return handler
