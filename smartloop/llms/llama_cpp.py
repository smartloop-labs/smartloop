"""Llama Cpp wrapper class"""
import contextlib
import threading
import time
import re
import logging

from typing import Any, Iterator, Dict, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk

from llama_cpp import Llama

from smartloop.constants import OPENING_TOKENS

from .base_llm import BaseLLM
from .thinking import ThinkingFilter, strip_thinking

# Attribute the per-context lock is stashed under on the ``Llama`` object, and
# the guard that makes creating it race-free.
_LOCK_ATTR = "_slp_context_lock"
_LOCK_REGISTRY_LOCK = threading.Lock()

# Output budget below which a reasoning model is told not to think.  Decision
# calls cap output hard (8-256 tokens for a yes/no or a short label); a <think>
# block would consume the whole budget and return a truncated thought where the
# caller expected "yes".  Module-level, not a class attribute: pydantic turns a
# leading-underscore class attribute into a private attr that reads back as a
# descriptor off the class.
THINKING_MIN_BUDGET = 512

class LlamaCpp(BaseLLM[Llama]):
    """Llama wrapper"""
    temperature: float = 0.3
    def __init__(self, model: Llama, tokenizer: Any, logger: logging.Logger, project_id: str, is_training: bool = False, model_name: Optional[str] = None):
        super().__init__(model=model, tokenizer=tokenizer, logger=logger, project_id=project_id, is_training=is_training)
        # The name this model was *selected* under, which is what the picker
        # shows and therefore what a status event has to say.
        self._slp_model_name = model_name

    @property
    def model_label(self) -> str:
        """The model's name as the user chose it, or ``""``.

        A GGUF's ``general.name`` is the upstream author's — for a model the
        user built or merged it is whatever the build was called
        ("Build Merged"), a name that appears nowhere in the product.  The
        selected name is the one in the picker, so it leads; empty here means
        the caller falls back to the metadata (see
        ``ModelRouter.model_label``).
        """
        if not self._slp_model_name:
            return ""
        try:
            from smartloop.constants import display_model_name
            return display_model_name(self._slp_model_name)
        except Exception:
            return self._slp_model_name
    
    def _context_lock(self) -> "threading.RLock":
        """The lock serializing generation on this llama.cpp context.

        A ``Llama`` context is NOT thread-safe — one KV cache, one token
        buffer, one sampler — so two concurrent ``create_chat_completion``
        calls corrupt the heap and segfault the process.  Callers reach the
        model from many threads: the DAG's node pool, the collect fan-out in
        :meth:`Explorer.collect_parallel`, MCP tool selection and argument
        extraction.  Their own locks — ``DAGExecutor._llm_lock`` and
        ``ModelService._inference_lock`` — are separate objects guarding
        different subsets of those callers, so neither excludes the other, and
        several call sites take no lock at all.  Serializing here is what makes
        the model safe whoever calls it, mirroring
        :class:`~smartloop.db.llama_embeddings.LlamaEmbedder`.

        Held only around generation.  ``tokenize``/``detokenize`` run on the
        vocab (``Llama._model``) rather than the context and need no guard.

        The lock lives on the ``Llama`` object, not on this wrapper, so two
        wrappers over one context still share one lock.  It is reentrant:
        generation tokenizes through :meth:`_fit_prompt` and
        :meth:`get_num_tokens` while already holding it.
        """
        lock = getattr(self.model, _LOCK_ATTR, None)
        if lock is None:
            with _LOCK_REGISTRY_LOCK:
                lock = getattr(self.model, _LOCK_ATTR, None)
                if lock is None:
                    lock = threading.RLock()
                    setattr(self.model, _LOCK_ATTR, lock)
        return lock

    def __del__(self):
        """Clean up model resources, handling llama-cpp-python cleanup issues"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                # Try to close the model properly
                if hasattr(self.model, 'close'):
                    self.model.close()
        except AttributeError:
            # Ignore AttributeError during cleanup (known issue in llama-cpp-python 0.3.16)
            pass
        except Exception:
            # Ignore other cleanup errors
            pass

    def get_max_tokens(self) -> int:
        # Return the effective context size: the smaller of the runtime n_ctx
        # and the model's trained context length (architectural max).
        n_ctx = self.model.n_ctx() if callable(self.model.n_ctx) else self.model.n_ctx
        n_ctx_train = self._get_n_ctx_train()
        if n_ctx_train and n_ctx_train > 0:
            return min(n_ctx, n_ctx_train)
        return n_ctx

    def _get_n_ctx_train(self) -> int:
        """Read the model's trained context length from GGUF metadata."""
        try:
            metadata = self.model.metadata
            # Key follows pattern: {architecture}.context_length
            for key, value in metadata.items():
                if key.endswith(".context_length"):
                    return int(value)
        except Exception:
            pass
        return 0

    def get_num_tokens(self, text: str) -> int:
        """Use Llama.cpp's tokenizer to tokenize the string and return the number of tokens"""
        return len(self.model.tokenize(text.encode('utf-8')))

    # Overhead reserved for chat-template formatting when sizing prompt/output.
    _CTX_OVERHEAD = 64
    # Output tokens to keep free when a prompt has to be truncated to fit.
    _MIN_OUTPUT_RESERVE = 512

    def _get_available_tokens(self, prompt: str) -> int:
        """Output tokens that fit after the prompt, never exceeding n_ctx.

        ``prompt + output`` must stay within ``n_ctx`` or llama.cpp rejects the
        request.  Returns the true remaining budget (floored at 1) — it must
        NOT force a fixed minimum, which is what previously drove
        ``prompt + 1024 > n_ctx`` overflows.
        """
        n_ctx = self.get_max_tokens()
        prompt_tokens = self.get_num_tokens(prompt)
        available = n_ctx - prompt_tokens - self._CTX_OVERHEAD
        return max(1, available)

    def _fit_prompt(self, prompt: str) -> str:
        """Truncate an over-long prompt so it fits n_ctx with room for output.

        A prompt that alone exceeds ``n_ctx`` makes llama.cpp reject the request
        ("Requested tokens exceed context window") — no ``max_tokens`` value can
        save it.  Keep the head (system/instructions) and tail (the user
        question, usually last) and drop the middle bulk (retrieved context /
        web results), inserting a marker so the model knows content was elided.
        """
        try:
            n_ctx = self.get_max_tokens()
            budget = n_ctx - self._MIN_OUTPUT_RESERVE - self._CTX_OVERHEAD
            if budget <= 0:
                return prompt
            tokens = self.model.tokenize(prompt.encode("utf-8"))
            if len(tokens) <= budget:
                return prompt

            marker = self.model.tokenize(b"\n\n...[context truncated to fit]...\n\n")
            keep = budget - len(marker)
            if keep <= 0:
                kept = tokens[:budget]
            else:
                head = keep * 2 // 3
                tail = keep - head
                kept = tokens[:head] + marker + (tokens[-tail:] if tail > 0 else [])
            truncated = self.model.detokenize(kept).decode("utf-8", errors="ignore")
            self.logger.warning(
                "[llm] prompt %d tok exceeds budget %d (n_ctx=%d) — truncated context to fit",
                len(tokens), budget, n_ctx,
            )
            return truncated
        except Exception as e:
            self.logger.warning("[llm] prompt-fit guard failed: %s", e)
            return prompt

    @property
    def supports_thinking(self) -> bool:
        """True when this model's chat template can reason.

        The flag is bound onto the ``Llama`` at load and only for a model whose
        template reads ``enable_thinking`` (see ``LlmFactory._bind_thinking``),
        so its presence is the answer.  Its *value* is the user's setting for
        visible answers, which a caller can override per call.
        """
        return hasattr(self.model, "_slp_thinking")

    @staticmethod
    def _rate(
        prompt_tokens: int,
        generated: int,
        start_time: float,
        first_token_at: Optional[float],
        opening_at: Optional[float] = None,
    ) -> str:
        """`` | 512 prompt, ttft 1.8s, 12.4 tok/s`` for a generation.

        "Slow" is not actionable and neither is a wall-clock total: a long
        prompt on a fast model and a short prompt on a model that fell back to
        CPU take the same ten seconds and want opposite fixes.  Prefill (time
        to the first token, which the prompt pays for) and decode rate (what
        the weights and the offload decide) separate them, and both are free to
        measure here.

        The average hides a ramp, though, and a ramp is what a reader notices:
        an answer that opens slowly and then runs is reported as the same
        10 tok/s as one that was even throughout.  So when a generation is long
        enough to have two halves, *opening_at* — the moment the first
        ``OPENING_TOKENS`` were out — splits the decode rate into the opening
        the reader waits on and the rest, and something loading onto the device
        alongside the first tokens shows up as the difference between them.
        """
        if not generated or first_token_at is None:
            return ""
        ttft = first_token_at - start_time
        now = time.time()
        decode = max(1e-6, now - first_token_at)
        rate = f"{generated / decode:.1f} tok/s"
        if opening_at is not None and generated > OPENING_TOKENS:
            opening = max(1e-6, opening_at - first_token_at)
            rest = max(1e-6, now - opening_at)
            rate = (
                f"{rate} (first {OPENING_TOKENS}: "
                f"{OPENING_TOKENS / opening:.1f}, then "
                f"{(generated - OPENING_TOKENS) / rest:.1f})"
            )
        return f" | {prompt_tokens} prompt, ttft {ttft:.2f}s, {rate}"

    @contextlib.contextmanager
    def _thinking_override(self, thinking: Optional[bool]):
        """Force the reasoning block on or off for one call.

        Used by callers whose output is not the visible answer — resolving an
        MCP tool call, for one — where the monologue is stripped before
        anything reads the text and the judgement is worth the tokens.  Safe on
        the shared ``Llama``: generation on it is serialized by the context
        lock this runs inside.
        """
        model = self.model
        if thinking is None or not hasattr(model, "_slp_thinking"):
            yield
            return
        previous = model._slp_thinking
        model._slp_thinking = bool(thinking)
        try:
            yield
        finally:
            model._slp_thinking = previous

    @contextlib.contextmanager
    def _thinking_for_budget(self, max_tokens: Optional[int]):
        """Suppress the reasoning block for a call too small to hold one.

        ``max_tokens`` is the call's *effective* output budget, so this covers
        both ways it gets small: a decision call that caps its own output (8-256
        tokens for a yes/no or a short label), and a long prompt that leaves
        little context headroom behind it.  Either way a <think> block would
        spend the budget on itself and the caller would get a truncated thought
        where it expected an answer.

        Only applies to a model that was loaded with reasoning on (see
        ``LlmFactory._bind_thinking``); everything else keeps its default.
        Flipping the flag on the shared ``Llama`` is safe because the context
        lock this runs under already serializes generation on it.
        """
        model = self.model
        previous = getattr(model, "_slp_thinking", None)
        suppress = (
            previous
            and max_tokens is not None
            and max_tokens < THINKING_MIN_BUDGET
        )
        if not suppress:
            yield
            return
        model._slp_thinking = False
        try:
            yield
        finally:
            model._slp_thinking = previous

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input."""
        with self._context_lock():
            start_time = time.time()
            prompt = self._fit_prompt(prompt)
            prompt_tokens = self.get_num_tokens(prompt)
            self.logger.info(f"Generating response | prompt_tokens={prompt_tokens}")

            try:
                messages = self.create_messages_from_prompt(prompt)
                requested_max = kwargs.get("max_tokens")
                available_tokens = (
                    requested_max if requested_max is not None
                    else self._get_available_tokens(prompt)
                )
                # Stream internally so generation can be interrupted between
                # tokens when the request is cancelled. stream=False is a single
                # blocking call that cannot be halted once started.
                # The guard only has to span the call itself: the chat handler
                # renders the template (where enable_thinking is read) eagerly
                # and returns an already-prompted token generator.
                with self._thinking_override(kwargs.get("thinking")), \
                        self._thinking_for_budget(available_tokens):
                    stream = self.model.create_chat_completion(
                        messages=messages,
                        max_tokens=available_tokens,
                        temperature=self.temperature,
                        repeat_penalty=1.2,
                        frequency_penalty=0.1,
                        presence_penalty=0.1,
                        stream=True
                    )

                parts: List[str] = []
                chunk_count = 0
                cancelled = False
                first_token_at: Optional[float] = None
                opening_at: Optional[float] = None
                for chunk in stream:
                    # Check for cancellation between chunks.
                    if self._cancelled is not None and self._cancelled.is_set():
                        cancelled = True
                        break

                    text = ""
                    if isinstance(chunk, dict) and chunk.get('choices'):
                        choice = chunk['choices'][0]
                        if 'delta' in choice and 'content' in choice['delta']:
                            text = choice['delta']['content'] or ""
                        elif 'message' in choice and 'content' in choice['message']:
                            text = choice['message']['content'] or ""
                        elif 'text' in choice:
                            text = choice['text'] or ""

                    if text:
                        if first_token_at is None:
                            first_token_at = time.time()
                        parts.append(text)
                        chunk_count += 1
                        if chunk_count == OPENING_TOKENS:
                            opening_at = time.time()

                elapsed = time.time() - start_time
                self.logger.info(
                    f"{'Cancelled' if cancelled else 'Completed'} | "
                    f"chunks={chunk_count} elapsed={elapsed:.3f}s"
                    f"{self._rate(prompt_tokens, chunk_count, start_time, first_token_at, opening_at)}"
                )
                # A reasoning model's <think> block is not the answer: every
                # caller from here on treats this string as one (JSON planning,
                # skill matching, the chat reply), so drop it.  A no-op for
                # models that don't think.
                text = strip_thinking("".join(parts))
                return re.sub(r'[\t]+', "", text.strip())
            except Exception as ex:
                self.logger.error(f"Generation failed: {ex}")
            finally:
                elapsed = time.time() - start_time
                self.logger.info(f"Total elapsed={elapsed:.3f}s")

            return ""

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM on the given prompt.

        Returns:
            An iterator of GenerationChunks.
        """
        with self._context_lock():

            # Use create_chat_completion with stream=True to get streaming response
            # This automatically applies the model's chat template
            prompt = self._fit_prompt(prompt)
            messages = self.create_messages_from_prompt(prompt)
            available_tokens = self._get_available_tokens(prompt)

            with self._thinking_override(kwargs.get("thinking")):
                stream = self.model.create_chat_completion(
                    messages=messages,
                    max_tokens=available_tokens,
                    temperature=self.temperature,
                    repeat_penalty=1.1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stream=True
                )

            chunk_count = 0
            start_time = time.time()
            first_token_at: Optional[float] = None
            opening_at: Optional[float] = None
            # Reasoning is withheld from the stream rather than shown and then
            # retracted; the tags can split across chunks, so the filter is
            # stateful and spans the whole stream.
            thinking = ThinkingFilter()
            for chunk in stream:
                # Check for cancellation between chunks
                if self._cancelled is not None and self._cancelled.is_set():
                    break

                chunk_count += 1
                if first_token_at is None:
                    first_token_at = time.time()
                elif chunk_count == OPENING_TOKENS:
                    opening_at = time.time()
                # Extract text from llama_cpp CreateChatCompletionStreamResponse format
                text = ""

                if isinstance(chunk, dict) and 'choices' in chunk:
                    if chunk['choices'] and len(chunk['choices']) > 0:
                        choice = chunk['choices'][0]
                        # For chat completion streaming, text is in delta.content
                        if 'delta' in choice and 'content' in choice['delta']:
                            text = choice['delta']['content']
                        elif 'message' in choice and 'content' in choice['message']:
                            text = choice['message']['content']
                        # Fallback to text field for compatibility
                        elif 'text' in choice:
                            text = choice['text']
                else:
                    text = str(chunk)

                # Only yield non-empty text chunks
                visible = thinking.feed(text) if text else ""
                if visible:
                    # Create GenerationChunk from the text
                    generation_chunk = GenerationChunk(text=visible)

                    # Notify run manager if available
                    if run_manager:
                        run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)

                    yield generation_chunk

            # Release whatever the filter held back — a trailing partial tag
            # that turned out to be ordinary text, or (when generation was cut
            # off mid-thought) the reasoning itself, which beats a blank reply.
            if thinking.unterminated:
                self.logger.warning(
                    "Generation ended inside a <think> block — the reply was "
                    "never reached (raise max_tokens or disable thinking)"
                )
            tail = thinking.flush()
            if tail:
                generation_chunk = GenerationChunk(text=tail)
                if run_manager:
                    run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)
                yield generation_chunk

            if chunk_count == 0:
                self.logger.warning("No chunks received from stream")
            else:
                self.logger.info(
                    "Streamed | chunks=%d elapsed=%.3fs%s",
                    chunk_count, time.time() - start_time,
                    self._rate(
                        self.get_num_tokens(prompt), chunk_count,
                        start_time, first_token_at, opening_at,
                    ),
                )

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.name,
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "smartloop"

    def invoke(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        thinking: Optional[bool] = None,
    ) -> str:
        """
        Invoke the LLM with the given prompt

        Args:
            prompt: The user query string
            max_tokens: Optional cap on generated tokens. Use a small value
                for short classification-style calls (e.g. skill matching) to
                avoid runaway greedy generation when the model fails to emit
                a stop token.
            thinking: Force the reasoning block on (True) or off (False) for
                this call, whatever the user's setting says. For a call whose
                output is not the visible answer — the monologue is stripped
                either way — and where the judgement is worth the tokens. None
                leaves the setting alone.

        Returns:
            Generated response text
        """

        try:
            kwargs = {}
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if thinking is not None:
                kwargs["thinking"] = thinking
            response = self._call(prompt, **kwargs)

            cleaned_response = response.strip()
            if cleaned_response.endswith('<|eot_id|>'):
                cleaned_response = cleaned_response[:-10].strip()

            return cleaned_response if cleaned_response else "No response generated"

        except Exception as e:
            self.logger.error(f"LLM invocation failed: {e}")
            return f"Error generating response: {str(e)}"

    def _apply_chat_template(self, messages: list) -> list:
        """
        Apply llama.cpp chat template - returns messages directly for chat completion API.

        llama.cpp handles chat template formatting internally via create_chat_completion,
        so we just return the standard message format.
        """
        return messages
