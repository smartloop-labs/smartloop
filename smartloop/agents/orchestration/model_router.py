"""Model selection and progress reporting.

Which LLM runs a given node, and telling the client what is happening while it
runs.  The two belong together: resolving a capability model is exactly the
moment the client needs to be told a model is loading, and the answer to
"which model" is the answer to "what do we report".

This is the leaf of the orchestration graph — it depends on nothing else.
"""
import logging
from typing import Callable, Optional

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
    ):
        self.orchestrator_llm = orchestrator_llm
        self.agent_llm = agent_llm
        self.large_llm = large_llm
        self._large_llm_provider = large_llm_provider
        self._capability_llm_provider = capability_llm_provider
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

    def llm_for_tier(self, tier: ModelTier, capability: Optional[str] = None) -> BaseLLM:
        if tier == ModelTier.LARGE:
            if capability and self._capability_llm_provider is not None:
                # Don't cache the capability LLM here — the provider
                # (model_service) is the authoritative cache and may evict a
                # prior step's model to fit this one. Holding our own strong
                # reference would pin that VRAM and defeat the eager release,
                # so a subsequent capability (e.g. skills after dp) can't fit.
                try:
                    self.emit_model_loading(capability)
                    llm = self._capability_llm_provider(capability)
                    if llm is not None:
                        logger.info(
                            "[models] capability '%s' → %s",
                            capability, self.model_label(llm),
                        )
                        self.emit_model_selected(capability, llm)
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
                self.emit_model_selected(capability, fallback)
            return fallback
        return self.agent_llm

    def decision_llm(self, capability: str) -> BaseLLM:
        """Resolve a capability model for a low-value *decision* node
        (skill matching, multi-step planning).

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
        try:
            meta = llm.model.metadata
            name = meta.get("general.name", "")
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

    def emit_model_loading(self, capability: Optional[str]) -> None:
        """Surface an *immediate* "loading" SSE before the (potentially slow)
        capability model load, so the user gets feedback right away instead of
        only after the model is resident. Emitted once per capability/request."""
        if not capability or self._model_event_seen(f"loading:{capability.lower()}"):
            return
        self.emit(
            "model", "processing",
            f"Loading model for {self._capability_label(capability)}…",
        )

    def emit_model_selected(self, capability: Optional[str], llm: BaseLLM) -> None:
        """Surface the selected capability model as an SSE. Emitted once per
        capability/request (a fan-out resolves the model once per document)."""
        if not capability or self._model_event_seen(f"selected:{capability.lower()}"):
            return
        self.emit(
            "model", "processing",
            f"Selected {self.model_label(llm)} for {self._capability_label(capability)}",
        )

    # Thin forwarding for test compatibility (tests mock _call_tool_for_server)