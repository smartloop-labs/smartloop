"""OpenVINO GenAI wrapper class — runs OpenVINO IR models on Intel GPU/NPU"""
import time
import re
import queue
import threading
import logging

from typing import Any, Iterator, Dict, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk

from .base_llm import BaseLLM

_STREAM_DONE = object()


class OpenVinoGenAI(BaseLLM[Any]):
    """Wrapper around ``openvino_genai.LLMPipeline``.

    Drop-in alternative to :class:`LlamaCpp` for Intel machines without an
    NVIDIA GPU.  The ggml-openvino llama.cpp backend cannot decode several
    architectures (Gemma-3 included) on any OpenVINO device, while GenAI runs
    INT4 OpenVINO IR exports natively on the iGPU/NPU.  Model artifacts are
    IR directories (openvino_model.xml + tokenizer), not GGUF files — see
    LlmFactory._find_openvino_ir_dir for the discovery convention.
    """
    temperature: float = 0.3

    def __init__(self, model: Any, tokenizer: Any, logger: logging.Logger,
                 project_id: str, max_context: int = 8192,
                 is_training: bool = False):
        super().__init__(model=model, tokenizer=tokenizer, logger=logger,
                         project_id=project_id, is_training=is_training)
        self._max_context = max_context

    def get_max_tokens(self) -> int:
        return self._max_context

    def get_num_tokens(self, text: str) -> int:
        """Tokenize with the pipeline's own tokenizer and count tokens."""
        try:
            return int(self.tokenizer.encode(text).input_ids.shape[1])
        except Exception:
            # Rough fallback so context budgeting still works.
            return max(1, len(text) // 4)

    def _get_available_tokens(self, prompt: str) -> int:
        """Calculate available output tokens after accounting for prompt size."""
        available = self.get_max_tokens() - self.get_num_tokens(prompt) - 64
        return max(1024, available)

    def _render_prompt(self, messages: List[dict]) -> str:
        """Apply the model's own chat template to standard messages."""
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

    def _generation_config(self, max_tokens: int):
        import openvino_genai as ov_genai
        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = max_tokens
        # Mirror the LlamaCpp sampling parameters.
        cfg.do_sample = self.temperature > 0
        cfg.temperature = self.temperature
        cfg.repetition_penalty = 1.2
        cfg.frequency_penalty = 0.1
        cfg.presence_penalty = 0.1
        return cfg

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input."""
        start_time = time.time()
        try:
            messages = self.create_messages_from_prompt(prompt)
            rendered = self._render_prompt(messages)
            cfg = self._generation_config(self._get_available_tokens(rendered))
            self.logger.info(
                f"Generating response | prompt_tokens={self.get_num_tokens(rendered)}"
            )
            output = str(self.model.generate(rendered, cfg))
            elapsed = time.time() - start_time
            self.logger.info(
                f"Completed | completion_tokens={self.get_num_tokens(output)} "
                f"elapsed={elapsed:.3f}s"
            )
            return re.sub(r'[\t]+', "", output.strip())
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

        ``LLMPipeline.generate`` is blocking and reports tokens through a
        callback, so generation runs in a worker thread that feeds a queue
        this iterator drains.
        """
        messages = self.create_messages_from_prompt(prompt)
        rendered = self._render_prompt(messages)
        cfg = self._generation_config(self._get_available_tokens(rendered))

        chunks: queue.Queue = queue.Queue()

        def _streamer(subword: str) -> bool:
            chunks.put(subword)
            # Returning True stops generation.
            return self._cancelled is not None and self._cancelled.is_set()

        def _generate():
            try:
                self.model.generate(rendered, cfg, _streamer)
            except Exception as ex:
                self.logger.error(f"Streaming generation failed: {ex}")
            finally:
                chunks.put(_STREAM_DONE)

        worker = threading.Thread(target=_generate, daemon=True)
        worker.start()

        chunk_count = 0
        while True:
            item = chunks.get()
            if item is _STREAM_DONE:
                break
            if not item:
                continue
            chunk_count += 1
            generation_chunk = GenerationChunk(text=item)
            if run_manager:
                run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)
            yield generation_chunk

        worker.join(timeout=5)
        if chunk_count == 0:
            self.logger.warning("No chunks received from stream")

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": self.name,
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "smartloop"

    def invoke(self, prompt: str) -> str:
        """
        Invoke the LLM with the given prompt

        Args:
            prompt: The user query string

        Returns:
            Generated response text
        """
        try:
            response = self._call(prompt)
            cleaned_response = response.strip()
            return cleaned_response if cleaned_response else "No response generated"
        except Exception as e:
            self.logger.error(f"LLM invocation failed: {e}")
            return f"Error generating response: {str(e)}"

    def _apply_chat_template(self, messages: list) -> list:
        """Messages are rendered with the pipeline tokenizer's chat template
        in _render_prompt; the standard format passes through unchanged."""
        return messages
