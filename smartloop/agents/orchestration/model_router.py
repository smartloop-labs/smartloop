"""Model selection and progress reporting.

Which LLM runs a given node, and telling the client what is happening while it
runs.  The two belong together: resolving a capability model is exactly the
moment the client needs to be told a model is loading, and the answer to
"which model" is the answer to "what do we report".

This is the leaf of the orchestration graph — it depends on nothing else.
"""
import logging
from typing import Callable, Optional

from smartloop.constants import display_model_label
from smartloop.llms import BaseLLM
from smartloop.types import ModelCapability
from smartloop.agents.planner import ModelTier

logger = logging.getLogger(__name__)

# Output cap for decision/classification LLM calls (yes/no, relevance, a short
# condition question).  Their answer is a handful of tokens, but without a cap
# the model may fail to emit a stop token and ramble up to the full context
# window — and with several matched skills those runaways stack into a
# multi-minute stall.  Content-generating calls (extract/analyze/synthesize)
# are intentionally NOT capped: they stay bounded by the model's context window
# (see llama_cpp._get_available_tokens) so capping would only risk truncating a
# legitimate answer.
DECISION_MAX_TOKENS = 64

# Cap for short free-text extractions (e.g. a 1-3 sentence formatting hint).
SHORT_TEXT_MAX_TOKENS = 256


class ModelRouter:
    """Resolves the LLM for a node and reports progress to the client.

    ``progress_callback(step, status, message)`` — supplied per request by the
    streaming API layer — is invoked as each step runs so the client can
    surface live activity.  It fires from DAG worker threads, so it must be
    thread-safe; a failure in it must never break orchestration.
    """

    def __init__(
        self,
        orchestrator_llm: BaseLLM,
        agent_llm: BaseLLM,
        large_llm: Optional[BaseLLM] = None,
        large_llm_provider: Optional[Callable[[], Optional[BaseLLM]]] = None,
        capability_llm_provider: Optional[Callable[[str], Optional[BaseLLM]]] = None,
        capability_available: Optional[Callable[[str], bool]] = None,
        capability_skip_reason: Optional[Callable[[], Optional[tuple]]] = None,
    ):
        self.orchestrator_llm = orchestrator_llm
        self.agent_llm = agent_llm
        self.large_llm = large_llm
        self._large_llm_provider = large_llm_provider
        self._capability_llm_provider = capability_llm_provider
        # Answers "does any selected model declare this capability?" without
        # loading anything, so a load is only announced when one can happen.
        self._capability_available = capability_available
        # Answers "why did the last capability model not load?" as
        # ``(model_name, cause)``, so a fallback can say what went wrong.
        self._capability_skip_reason = capability_skip_reason
        # Per-request state, reset by begin_request().
        self._progress_callback: Optional[Callable[[str, str, str], None]] = None
        self._emitted_model_events: set = set()

    @property
    def has_large_provider(self) -> bool:
        """True when a generic large model can still be loaded on demand."""
        return self._large_llm_provider is not None

    def begin_request(
        self,
        progress_callback: Optional[Callable[[str, str, str], None]] = None,
    ) -> None:
        """Reset per-request state and install this request's progress sink.

        The model-selection dedupe set has to be cleared per request: a
        capability's model is resolved once per DAG node (once per document in
        a fan-out, say), but the user only needs to be told which model was
        picked once.
        """
        self._progress_callback = progress_callback
        self._emitted_model_events = set()

    def llm_for_tier(
        self,
        tier: ModelTier,
        capability: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> BaseLLM:
        """Resolve the model for a node, reporting the choice to the client.

        ``capability`` routes; ``purpose`` — when given — is what the user is
        told the model is for.  The two are not the same thing.  Synthesis
        routes on the ``skills`` capability because a skill's answer should be
        written by a skills-capable model, but the user is watching the final
        answer being written, and "Loading model for skills…" names an internal
        routing tag rather than the work in progress.
        """
        if tier == ModelTier.LARGE:
            if capability and self._capability_llm_provider is not None:
                # Don't cache the capability LLM here — the provider
                # (model_service) is the authoritative cache and may evict a
                # prior step's model to fit this one. Holding our own strong
                # reference would pin that VRAM and defeat the eager release,
                # so a subsequent capability (e.g. skills after dp) can't fit.
                try:
                    # Only announce the load when there is something to load.
                    # This event fires *before* the provider is consulted (its
                    # whole point is to precede a slow load), so on a workspace
                    # where no selected model declares the capability it used to
                    # promise a load that never happened — "Loading model for
                    # skills…" on a request with no skills-capable model, then a
                    # fallback to the bootstrap model.
                    if self._capability_model_exists(capability):
                        self.emit_model_loading(capability, purpose)
                    llm = self._capability_llm_provider(capability)
                    if llm is not None:
                        logger.info(
                            "[models] capability '%s' → %s",
                            capability, self.model_label(llm),
                        )
                        self.emit_model_selected(capability, llm, purpose)
                        return llm
                except Exception as e:
                    logger.warning("[models] capability model load failed (%s): %s", capability, e)
            # No capability-specific model available — fall through to the
            # generic large model so synthesis / analysis still gets the
            # best available model rather than the small orchestrator.
            if self.large_llm is None and self._large_llm_provider is not None:
                try:
                    self.large_llm = self._large_llm_provider()
                except Exception as e:
                    logger.warning("[models] large model load failed: %s", e)
                    self.large_llm = None
            fallback = self.large_llm or self.agent_llm
            if capability:
                logger.warning(
                    "[models] capability '%s' unresolved — falling back to %s",
                    capability, self.model_label(fallback),
                )
                self.emit_model_fallback(capability, fallback, purpose)
            return fallback
        # A capability tag routes at every tier, not only LARGE.  A SMALL step
        # is a decision node — picking an MCP tool, extracting its arguments,
        # judging relevance — so it routes *best-effort*: decision_llm takes the
        # capability model only when it is already resident or comfortably fits,
        # and otherwise lands back on agent_llm without forcing a load.
        #
        # Dropping the tag here is what kept every MCP tool call on the
        # bootstrap router: pipeline_builder tags a non-skill TOOL step SMALL
        # and still sets its capability, and that capability was read by
        # nothing.  It only surfaced with two or more models selected — a
        # single selection is pinned into agent_llm upstream, so the default
        # happened to be the user's model.
        return self.decision_llm(capability)

    def _capability_model_exists(self, capability: Optional[str]) -> bool:
        """True when some selected model declares *capability*.

        Unknown (no predicate wired in) counts as True, so a caller that does
        not supply one keeps the previous behaviour.
        """
        if not capability or self._capability_available is None:
            return True
        try:
            return bool(self._capability_available(capability))
        except Exception:
            logger.debug(
                "[models] capability availability check failed for %s",
                capability, exc_info=True,
            )
            return True

    def decision_llm(self, capability: Optional[str]) -> BaseLLM:
        """Resolve a capability model for a low-value *decision* node
        (skill matching, multi-step planning, MCP tool selection).

        Unlike :meth:`_llm_for_tier`, this never forces a generic large-model
        load: it uses the capability model only when it is already resident or
        comfortably fits (the provider's own headroom guard decides), and
        otherwise falls straight back to the small orchestrator model. This
        keeps the better intent judgement "when available" without adding VRAM
        pressure — and risking an OOM — on memory-tight devices.
        """
        if capability and self._capability_llm_provider is not None:
            # No local caching — see _llm_for_tier: the provider owns the
            # capability cache and may evict to fit, so pinning a reference here
            # would prevent that VRAM from being reclaimed for the next step.
            try:
                llm = self._capability_llm_provider(capability)
                if llm is not None:
                    logger.info(
                        "[models] decision capability '%s' → %s",
                        capability, self.model_label(llm),
                    )
                    self.emit_model_selected(capability, llm)
                    return llm
            except Exception as e:
                logger.warning(
                    "[models] decision capability load failed (%s): %s",
                    capability, e,
                )
        return self.agent_llm

    @staticmethod
    def model_label(llm: BaseLLM) -> str:
        """Name this model the way the user's model picker names it.

        The GGUF's ``general.name`` is the upstream author's label, which for
        our own model is ``Slp Orchestra Mini`` — a name that appears nowhere
        in the product.  This label goes out on the progress SSE, so it is
        mapped back to the short name (``sl-mini``) before it is shown.
        """
        # A handle to a model owned by another process reports its own label —
        # there is no local GGUF here to read metadata from, and falling
        # through would put "RemoteLLM" on the user's progress event.
        # Must be a real string: ``getattr`` on a test double answers every
        # attribute with a truthy mock, which would shadow the metadata path
        # for every locally-loaded model.
        explicit = getattr(llm, "model_label", None)
        if isinstance(explicit, str) and explicit:
            return explicit
        try:
            meta = llm.model.metadata
            name = display_model_label(meta.get("general.name", ""))
            size = meta.get("general.size_label", "")
            if name:
                return f"{name} ({size})" if size else name
        except Exception:
            pass
        return type(llm).__name__

    def emit(self, step: str, status: str, message: str) -> None:
        """Report an orchestration step to the optional progress callback.

        The callback is supplied by the streaming API layer to surface live
        activity as SSE ``chat.status`` events.  It is invoked from DAG worker
        threads, so it must be thread-safe; any failure here must never break
        orchestration.
        """
        cb = self._progress_callback
        if cb is None:
            return
        try:
            cb(step, status, message)
        except Exception:
            logger.debug("[progress] callback failed", exc_info=True)

    # Client-facing labels for capabilities. The enum values (e.g. "dp") are
    # internal terms; map them to human-readable text before surfacing in SSEs.
    _CAPABILITY_LABELS = {
        ModelCapability.GENERAL.value: "general",
        ModelCapability.SKILLS.value: "skills",
        ModelCapability.DP.value: "document processing",
        ModelCapability.VISION.value: "vision",
        ModelCapability.CODING.value: "coding",
        ModelCapability.THINKING.value: "thinking",
    }

    def _capability_label(self, capability: str) -> str:
        return self._CAPABILITY_LABELS.get(capability.lower(), capability)

    def _model_event_seen(self, key: str) -> bool:
        """True if this model SSE already fired this request (and record it).

        Collapses the per-node emits of a fan-out (one model resolution per
        document) into a single client message. Racy across the parallel batch
        threads, but a rare duplicate emit is harmless.
        """
        if key in self._emitted_model_events:
            return True
        self._emitted_model_events.add(key)
        return False

    def _event_label(
        self, capability: Optional[str], purpose: Optional[str],
    ) -> tuple:
        """``(label, key)`` for a model event — what to say, and what to dedupe
        on.  ``purpose`` wins over the capability tag on both counts, so two
        different jobs routed through the same capability each get their line.
        """
        if purpose:
            return purpose, purpose.lower()
        if capability:
            return self._capability_label(capability), capability.lower()
        return "", ""

    def emit_model_loading(
        self, capability: Optional[str], purpose: Optional[str] = None,
    ) -> None:
        """Surface an *immediate* "loading" SSE before the (potentially slow)
        capability model load, so the user gets feedback right away instead of
        only after the model is resident. Emitted once per capability/request."""
        label, key = self._event_label(capability, purpose)
        if not key or self._model_event_seen(f"loading:{key}"):
            return
        self.emit("model", "processing", f"Loading model for {label}…")

    def emit_model_selected(
        self,
        capability: Optional[str],
        llm: BaseLLM,
        purpose: Optional[str] = None,
    ) -> None:
        """Surface the selected capability model as an SSE. Emitted once per
        capability/request (a fan-out resolves the model once per document)."""
        label, key = self._event_label(capability, purpose)
        if not key or self._model_event_seen(f"selected:{key}"):
            return
        self.emit(
            "model", "processing",
            f"Selected {self.model_label(llm)} for {label}",
        )

    # Why a capability model was skipped, phrased for the user.  ``memory`` is
    # the one they can act on — close something, or deselect a model — so it
    # says so plainly instead of leaving the fallback looking like their
    # selection was ignored.
    _SKIP_REASONS = {
        "memory": "not enough memory for {model}",
        "missing": "{model} is not downloaded",
        "faulted": "{model} ran out of memory",
        "error": "{model} could not be loaded",
    }

    def _skip_explanation(self) -> str:
        """Trailing clause explaining why no capability model was used, or ``""``."""
        if self._capability_skip_reason is None:
            return ""
        try:
            skip = self._capability_skip_reason()
        except Exception:
            logger.debug("[models] skip-reason lookup failed", exc_info=True)
            return ""
        if not isinstance(skip, (tuple, list)) or len(skip) < 2:
            return ""
        model_name, cause = skip[0], skip[1]
        # ``detail`` is optional and carries the figures behind the cause —
        # "needs ~4.9GB, 4.4GB free" — which is the part the user can act on.
        detail = skip[2] if len(skip) > 2 else None
        template = self._SKIP_REASONS.get(cause)
        if not template or not model_name:
            return ""
        from smartloop.constants import display_model_name
        explanation = template.format(model=display_model_name(model_name))
        return f"{explanation} ({detail})" if detail else explanation

    def emit_model_fallback(
        self,
        capability: Optional[str],
        llm: BaseLLM,
        purpose: Optional[str] = None,
    ) -> None:
        """Name the model that will run the step when the capability could not
        be resolved, and say why when the reason is known.

        Deliberately does not name the capability: no capability model was
        found, so "Selected sl-mini for skills" reads as though one had been
        chosen for that work.  Shares the selection dedupe key, so a later real
        resolution for the same capability does not report twice.
        """
        _label, key = self._event_label(capability, purpose)
        if not key or self._model_event_seen(f"selected:{key}"):
            return
        explanation = self._skip_explanation()
        message = f"Selected {self.model_label(llm)}"
        if explanation:
            message = f"{message} — {explanation}"
        self.emit("model", "processing", message)

    # Thin forwarding for test compatibility (tests mock _call_tool_for_server)