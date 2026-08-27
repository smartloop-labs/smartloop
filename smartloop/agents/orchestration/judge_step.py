"""JudgeStep — LLM-as-judge relevance evaluation of context before it is used.

Retrieval is recall-oriented — a vector store returns its nearest neighbours
whether or not any of them is actually about the question — so without a check
an unrelated document reaches the prompt fenced as DOCUMENT CONTEXT under
instructions to answer from it.  The model then correctly reports that it
cannot ("I don't have context") instead of falling through to what it knows.

This module is that check.  It runs at two points: as a step in the DAG,
judging each retrieval source before any later step reads it, and inline in
the reason handler, judging each step's output before it is appended.

It never gates the user's connected-app data.  Tool results — a live MCP call
during explore, a tool step's output in the plan — are exempt at both points:
they were fetched on purpose rather than returned by a similarity search, so
there is no recall noise to filter, and a wrong verdict throws away the very
data the request went to get.  Crawled MCP content is exempt too, because it
already passed the retrieval quality floor and does not need a second, worse
gate.  See ``_JUDGED_SOURCES`` below.
"""
import logging
import re
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

# A verdict token, matched whole: ``RELEVANT`` inside ``NOT_RELEVANT`` is not a
# separate word, and the alternation puts the negative first so the space-
# separated spelling is read as one token rather than as a bare ``RELEVANT``.
_VERDICT_RE = re.compile(r"\b(NOT[_ ]RELEVANT|RELEVANT)\b")

# Leading decoration a small model puts in front of its answer: markdown
# emphasis, a bullet, a "Verdict:" colon.  Stripped so a verdict that opens the
# reply is still recognised as opening it.
_VERDICT_PREFIX = "*_#>-—: \t\n\"'`"

# How much of an undecided or rejecting reply is logged.  Long enough to show
# the model's own words, short enough not to dump a paragraph per request.
_REPLY_LOG_CHARS = 200


def parse_verdict(reply: str) -> Optional[bool]:
    """``True``/``False`` for a decisive verdict, ``None`` when there isn't one.

    The check used to be ``"NOT_RELEVANT" in reply``, which reads any reply
    *mentioning* the word as a rejection — including the two things a small
    model most often says instead of answering: it quotes the instruction back
    ("Reply RELEVANT or NOT_RELEVANT"), or it answers in prose ("the context is
    RELEVANT, not NOT_RELEVANT").  Both dropped the context, which is how a
    0.111-distance match for "list my boards on monday.com" ended up discarded
    and answered with "I don't have access to your Monday.com boards".

    So: whole tokens, and a reply that names *both* labels without leading with
    one is treated as no verdict at all.  A verdict that opens the reply is
    still decisive when the reasoning after it names the other label, which is
    the normal shape of a well-formed answer ("NOT_RELEVANT — this is about
    Trello, not relevant to the question").
    """
    text = (reply or "").strip().upper().lstrip(_VERDICT_PREFIX).strip()
    matches = list(_VERDICT_RE.finditer(text))
    if not matches:
        return None
    labels = {m.group(1).replace(" ", "_") for m in matches}
    leading = matches[0].group(1).replace(" ", "_") if matches[0].start() == 0 else None
    if leading:
        return leading != "NOT_RELEVANT"
    if len(labels) > 1:
        return None
    return "NOT_RELEVANT" not in labels


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
    # What belongs here is *retrieval*: content a similarity search handed us
    # because it was nearest, not because it was asked for.  That is the whole
    # premise of the judge — recall returns neighbours whether or not any of
    # them is about the question.
    #
    # Both MCP sources are deliberately absent — the user's connected-app data
    # is never gated by this judge.
    #
    # ``mcp_context`` is a live tool result, and a tool call is not recall:
    # something decided to invoke it, the server ran it, and what came back is
    # the answer to that call.  Judging it put a small decision model between
    # the user and data they explicitly went and fetched — and every verdict
    # here fails toward dropping, so the failure mode was running the
    # Monday.com tool, getting the boards, and answering "I don't have access
    # to your boards".  A tool that should not have been called is a routing
    # problem, and gating its output cannot fix it: the call already happened,
    # and binning the result only loses the one piece of context that was
    # certainly about the request.
    #
    # ``mcp_document_context`` is crawled MCP content, and it is already
    # relevance-gated where the gate belongs — in retrieval.
    # ``search_mcp_documents`` runs it through ``_select_relevant_chunks``,
    # the same quality floor (best-match distance <= 0.55) as document and
    # workspace search, and returns nothing at all when the best match misses.
    # Content that reaches this attribute has therefore already cleared a
    # threshold; a second opinion from the small model is the *worse* of the
    # two gates and only ever subtracts.  Identical reasoning to
    # ``vector_context`` below.
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
        reply = ""
        try:
            reply = str(self._models.agent_llm.invoke(
                prompt, max_tokens=DECISION_MAX_TOKENS,
            )).strip()
            verdict = parse_verdict(reply)
            if verdict is None:
                # No verdict in the reply.  Kept — but said out loud, because a
                # judge that cannot answer its own prompt is a thing to know
                # about rather than a silent pass.
                logger.info(
                    "[judge] %s: no verdict in reply (%s) — keeping context",
                    label, reply[:_REPLY_LOG_CHARS] or "empty",
                )
            else:
                serves = verdict
        except Exception:
            logger.debug(
                "[judge] %s: LLM call failed — keeping context",
                label, exc_info=True,
            )

        if not serves:
            # The reply travels with the decision: a misjudging model and a
            # misread verdict produce the same log line without it, which is
            # the position the last investigation of this ended in.
            logger.info(
                "[judge] %s (%d chars) does not serve the request — dropped "
                "(reply: %s)",
                label, len(text), reply[:_REPLY_LOG_CHARS],
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
                "filter", "processing", self._set_aside_message(dropped),
            )
        return dropped

    @staticmethod
    def _set_aside_message(dropped: List[str]) -> str:
        """Say what was put aside, in the voice the user is reading in.

        "Set aside context that does not match the request: MCP content" reads
        like a filter's own log line — a colon-delimited list of internal
        source names.  The person watching is waiting on an answer, not
        auditing the pipeline, so it names the sources in a sentence and says
        why, which is also the part that keeps a dropped source from looking
        lost.
        """
        if len(dropped) == 1:
            items, pronoun = dropped[0], "it"
        else:
            items = f"{', '.join(dropped[:-1])} and {dropped[-1]}"
            pronoun = "they"
        return f"Set aside {items} — {pronoun} didn't seem relevant to your question"

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
