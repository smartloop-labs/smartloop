"""Llama Cpp wrapper class"""
import time
import re
import logging

from typing import Any, Iterator, Dict, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk

from llama_cpp import Llama

from .base_llm import BaseLLM

class LlamaCpp(BaseLLM[Llama]):
    """Llama wrapper"""
    temperature: float = 0.3
    def __init__(self, model: Llama, tokenizer: Any, logger: logging.Logger, project_id: str, is_training: bool = False):
        super().__init__(model=model, tokenizer=tokenizer, logger=logger, project_id=project_id, is_training=is_training)
    
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

    def _get_available_tokens(self, prompt: str) -> int:
        """Calculate available output tokens after accounting for prompt size."""
        n_ctx = self.get_max_tokens()
        prompt_tokens = self.get_num_tokens(prompt)
        # Reserve some overhead for chat template formatting
        available = n_ctx - prompt_tokens - 64
        return max(1024, available)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input."""
        start_time = time.time()
        print('Generating Text response', start_time)

        try:
            messages = self.create_messages_from_prompt(prompt)
            available_tokens = self._get_available_tokens(prompt)
            output = self.model.create_chat_completion(
                messages=messages,
                max_tokens=available_tokens,
                temperature=self.temperature,
                repeat_penalty=1.2,  # Reduce repetition (correct parameter name)
                frequency_penalty=0.1,   # Reduce frequency of repeated tokens
                presence_penalty=0.1,    # Reduce presence of repeated tokens
                stream=False
            )
            print('Completed', (time.time() - start_time) / 60)
            return re.sub(r'[\t]+', "", output['choices'][0]['message']['content'].strip())
        except Exception as ex:
            print(ex)
        finally:
            print('Total Time taken', (time.time() - start_time) / 60)

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

        # Use create_chat_completion with stream=True to get streaming response
        # This automatically applies the model's chat template
        messages = self.create_messages_from_prompt(prompt)
        available_tokens = self._get_available_tokens(prompt)

        stream = self.model.create_chat_completion(
            messages=messages,
            max_tokens=available_tokens,
            temperature=self.temperature,
            repeat_penalty=1.2,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stream=True
        )

        chunk_count = 0
        for chunk in stream:
            # Check for cancellation between chunks
            if self._cancelled is not None and self._cancelled.is_set():
                break

            chunk_count += 1
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
            if text:
                # Create GenerationChunk from the text
                generation_chunk = GenerationChunk(text=text)

                # Notify run manager if available
                if run_manager:
                    run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)

                yield generation_chunk

        # Debug: print if no chunks were processed
        if chunk_count == 0:
            print("Warning: No chunks received from stream")

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
            if cleaned_response.endswith('<|eot_id|>'):
                cleaned_response = cleaned_response[:-10].strip()

            return cleaned_response if cleaned_response else "No response generated"

        except Exception as e:
            print(f"Error during LLM invocation: {e}")
            return f"Error generating response: {str(e)}"

    def _apply_chat_template(self, messages: list) -> list:
        """
        Apply llama.cpp chat template - returns messages directly for chat completion API.

        llama.cpp handles chat template formatting internally via create_chat_completion,
        so we just return the standard message format.
        """
        return messages
