"""A :class:`BaseLLM` whose weights live in the model host, not this process.

Routing the *response* through a model client was only half the job.  The
orchestrator does far more inference than the final answer — classification,
skill matching, planning, every DAG decision node — and it takes ``BaseLLM``
objects directly rather than going through the service.  Worse, it does not
fail loudly without one: ``orchestrate`` logs "BYPASSED — model/llm not
loaded" and falls back to raw generation, so a child with no weights would
quietly lose skills, tools and planning while still answering.

So a child hands the orchestrator one of these instead.  It satisfies the same
interface the local llama.cpp LLM does, and every call crosses to the host.

Capability selection stays on the host side: the ``capability`` this instance
carries is sent with each request, and the host picks the model the same way it
would in-process.  A child has no business choosing — it cannot see the device,
so it cannot know what fits.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Iterator, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk

from smartloop.llms.base_llm import BaseLLM

logger = logging.getLogger(__name__)

#: Used when the host cannot say how big its context is.  Matches the local
#: LLM's own fallback, so prompt sizing degrades the same way in both.
DEFAULT_MAX_CONTEXT_TOKENS = 8192


class RemoteLLM(BaseLLM):
    """The host's model, addressed as if it were loaded here.

    Deliberately thin: no prompt fitting, no context lock, no chat template.
    All three belong to whoever owns the weights, and doing them twice — once
    on each side of the socket — is how a prompt ends up truncated to a window
    that is not the one it will actually be run in.
    """

    def __init__(self, client, project_id: str = "",
                 capability: Optional[str] = None,
                 logger_: Optional[logging.Logger] = None) -> None:
        # ``model`` is the client: BaseLLM stores it, and it is what this LLM
        # is a handle to.  There is no local tokenizer — token counting is one
        # more question for the host.
        super().__init__(
            model=client,
            tokenizer=None,
            logger=logger_ or logger,
            project_id=project_id,
        )
        self._cancelled: Optional[threading.Event] = None
        self._capability = capability

    # -- identity -----------------------------------------------------------

    @property
    def _llm_type(self) -> str:
        return "smartloop-remote"

    @property
    def client(self):
        return self.model

    @property
    def capability(self) -> Optional[str]:
        return self._capability

    # -- generation ---------------------------------------------------------

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self.client.generate(prompt, capability=self._capability)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        # ``_cancelled`` is set by the caller between calls, exactly as it is
        # on the local LLM; here it travels to the host as a cancel frame.
        for text in self.client.stream(
            prompt, capability=self._capability, cancelled=self._cancelled,
        ):
            if self._cancelled is not None and self._cancelled.is_set():
                break
            yield GenerationChunk(text=text)

    def invoke(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate once, matching the local LLM's forgiving contract.

        Errors are returned rather than raised for the same reason they are
        locally: the orchestrator calls this from DAG nodes that treat a bad
        answer as a dead branch, and an exception there would take down a whole
        plan over one step.
        """
        try:
            response = (self._call(prompt) or "").strip()
            if response.endswith("<|eot_id|>"):
                response = response[:-10].strip()
            return response or "No response generated"
        except Exception as exc:
            self.logger.error(f"Remote LLM invocation failed: {exc}")
            return f"Error generating response: {str(exc)}"

    # -- naming -------------------------------------------------------------

    @property
    def model_label(self) -> str:
        """What the host calls the model behind this handle.

        Read by both ``model_label`` helpers, which otherwise fall through to
        the class name — and this label goes out on the progress SSE, so the
        fallthrough is a user-visible "RemoteLLM" where the model's name
        belongs.  Resolved per call rather than cached: the host is free to
        load a different model for a capability between requests, and a stale
        name here would be worse than the round trip.
        """
        try:
            return self.client.model_label(self._capability) or ""
        except Exception as exc:
            self.logger.warning(f"Remote model label failed: {exc}")
            return ""

    # -- sizing -------------------------------------------------------------

    def get_num_tokens(self, text: str) -> int:
        """Count with the host's tokenizer — the only one that matches."""
        try:
            return int(self.client.count_tokens(text))
        except Exception as exc:
            self.logger.warning(f"Remote token count failed: {exc}")
            return 0

    def get_max_tokens(self) -> int:
        try:
            return int(
                self.client.response_max_context_tokens(self._capability)
            ) or DEFAULT_MAX_CONTEXT_TOKENS
        except Exception as exc:
            self.logger.warning(f"Remote context size failed: {exc}")
            return DEFAULT_MAX_CONTEXT_TOKENS

    # -- template -----------------------------------------------------------

    def _apply_chat_template(self, messages: list) -> list:
        """Left to the host, which owns the model that defines the template."""
        return messages

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RemoteLLM(capability={self._capability!r})"
