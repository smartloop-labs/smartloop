"""Wrapped class for pretrained LLM"""
import time
import re
import logging
import uuid
import torch
from typing import Any, Iterator, Dict, List, Optional, Callable

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk
from transformers import StoppingCriteria

from smartloop.utils.device_utils import get_device_config

from .base_llm import BaseLLM


class _CancelledStoppingCriteria(StoppingCriteria):
    """Stops HF generate() when a threading.Event is set."""

    def __init__(self, cancelled):
        self._cancelled = cancelled

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self._cancelled is not None and self._cancelled.is_set()

class PreTrainedLLM(BaseLLM[Any]):
    temperature: float = 0.3
    pending_process: bool = False

    """PretrainedLLM wrapped class"""  
    def __init__(self, model: Any, tokenizer: Any, logger: logging.Logger, project_id: str, is_training: bool = False):
        # Initialize with all required fields for Pydantic validation
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            logger=logger,
            project_id=project_id,
            is_training=is_training,
        )
    
    @property
    def do_sample(self) -> bool:
        """Determine whether to sample based on temperature."""
        return self.temperature > 0.0


    def _apply_chat_template(self, messages: list) -> Any:
        """
        Apply the tokenizer's chat template to format messages.

        Uses the model's built-in tokenizer.apply_chat_template method for
        LLM-agnostic formatting, with fallback to Llama format if needed.
        """
        try:
            # Get device from model parameters (more reliable than model.device)
            device = next(self.model.parameters()).device
            
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # Handle different return types (dict, BatchEncoding, or Tensor)
            if torch.is_tensor(formatted_prompt):
                # If it's a tensor, wrap it
                formatted_prompt = {
                    'input_ids': formatted_prompt,
                    'attention_mask': torch.ones_like(formatted_prompt)
                }
            elif hasattr(formatted_prompt, 'data'):
                # If it's a BatchEncoding, extract the data dict
                formatted_prompt = dict(formatted_prompt.data)
            elif not isinstance(formatted_prompt, dict):
                # Convert dict-like objects to plain dict
                formatted_prompt = dict(formatted_prompt)
            
            # Move tensors to device
            formatted_prompt = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                               for k, v in formatted_prompt.items()}
           
            return formatted_prompt

        except Exception as e:
            raise ValueError(f"Chat template formatting failed: {e}")
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input using direct model calls."""
        
        start_time = time.time()
        self.pending_process = True
        
        # Skip adapter checks during training
        if not self.is_training:
            # Verify model type and adapters (debug logging)
            model_type_name = type(self.model).__name__
            is_peft = 'Peft' in model_type_name
            
            if hasattr(self.logger, 'info'):
                self.logger.info(f"[info] Inference - Model type: {model_type_name} (PEFT: {is_peft})")
                
            if hasattr(self.model, 'active_adapters'):
                active = self.model.active_adapters
                if hasattr(self.logger, 'info'):
                    self.logger.info(f"[info] Active adapters: {active}")
        
        try:
            # Convert the input prompt to a list of messages, by adding a system message
            messages = self.create_messages_from_prompt(prompt)
            
            inputs = self._apply_chat_template(messages)
            prompt_length = inputs['input_ids'].shape[-1]

            # Ensure tokenizer has pad_token_id
            pad_token_id = self.tokenizer.eos_token_id

            if pad_token_id is None:
                pad_token_id = self.tokenizer.pad

            pad_token_id = getattr(self.tokenizer, 'pad_token_id', None) or pad_token_id
            
            # Optimized generation parameters for maximum GPU utilization
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=2048,  # Increased from 512 for longer generations
                repetition_penalty=1.2,
                do_sample=False,
                pad_token_id=pad_token_id,
                # GPU optimization parameters
                use_cache=True,  # Enable KV cache for faster generation
                num_beams=1,  # Greedy decoding is fastest for inference
                early_stopping=False,  # Let model generate full response
                # Memory optimization
                output_attentions=False,  # Don't return attention weights to save memory
                output_hidden_states=False,  # Don't return hidden states to save memory
            )
  
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
            
            return re.sub(r'[\t]+', "", generated_text)
            
        except Exception as ex:
            error_msg = f"PreTrainedLLM generation error: {ex}"
            if hasattr(self, 'logger') and self.logger:
                try:
                    self.logger.error(error_msg)
                    # Log additional context for debugging
                    self.logger.error(f"Temperature: {self.temperature}, do_sample: {self.do_sample}")
                    self.logger.error(f"Prompt length: {prompt_length if 'prompt_length' in locals() else 'unknown'}")
                except Exception:
                    print(error_msg)
            else:
                print(error_msg)
        finally:
            self.pending_process = False

        return "no response"

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM on the given prompt using TextIteratorStreamer.

        Returns:
            An iterator of GenerationChunks.
        """
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        try:
            # Convert the input prompt to a list of messages
            messages = self.create_messages_from_prompt(prompt)
            
            inputs = self._apply_chat_template(messages)
            prompt_length = inputs['input_ids'].shape[-1]
      
            # Create streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Prepare generation kwargs with GPU optimization
            generation_kwargs = dict(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=2048,  # Increased from 512 for longer generations
                repetition_penalty=1.2,
                do_sample=False,
                streamer=streamer,
                # GPU optimization parameters
                use_cache=True,  # Enable KV cache for faster generation
                num_beams=1,  # Greedy decoding is fastest for inference
                # Memory optimization
                output_attentions=False,
                output_hidden_states=False,
            )

            # Inject stopping criteria so model.generate() stops when cancelled
            if self._cancelled is not None:
                generation_kwargs["stopping_criteria"] = [
                    _CancelledStoppingCriteria(self._cancelled),
                ]
            
            # Start generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream the generated text
            chunk_count = 0
            for text in streamer:
                # Check for cancellation between chunks
                if self._cancelled is not None and self._cancelled.is_set():
                    break

                chunk_count += 1
                if text:
                    # Create GenerationChunk from the text
                    generation_chunk = GenerationChunk(text=text)
                    
                    # Notify run manager if available
                    if run_manager:
                        run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)
                    
                    yield generation_chunk
            
            # Wait for generation to complete
            thread.join()
            
            # Debug: print if no chunks were processed
            if chunk_count == 0:
                self.logger.warning("No chunks received from stream")
                
        except Exception as ex:
            error_msg = f"PreTrainedLLM streaming error: {ex}"
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)

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
            prompt: The PromptTemplate object containing the user query

        Returns:
            Generated response text
        """

        try:
            # Use _call method which already exists and handles the pipeline
            response = self._call(prompt)
            return response if response else "No response generated"

        except Exception as e:
            self.logger.error(f"Error during LLM invocation: {e}")
            return f"Error generating response: {str(e)}"

    def get_num_tokens(self, text: str) -> int:
        return len(self.tokenizer.tokenize(text))
    
    def batch_invoke(self, prompts: List[str], progress_callback: Optional[Callable[[int], None]] = None) -> List[str]:
        """
        Process a batch of prompts in parallel on GPU.
        
        Args:
            prompts: List of prompts to process (should be pre-batched for optimal token usage)
            progress_callback: Optional callback function called with 1 after each individual response completes
        
        Returns:
            List of generated responses
        """
        if not prompts:
            return []
        
        batch = prompts
        results = []
        
        try:
            # Convert all prompts to messages
            batch_messages = [self.create_messages_from_prompt(prompt) for prompt in batch]
            
            # Apply chat template to all messages
            batch_inputs = []
            for idx, messages in enumerate(batch_messages):
                try:
                    inputs = self._apply_chat_template(messages)
                    batch_inputs.append(inputs)
                except Exception as template_error:
                    self.logger.error(f"Error applying chat template for prompt {idx}: {template_error}")
                    self.logger.error(f"Messages: {messages}")
                    raise
            
            # Stack inputs for batch processing
            # Pad sequences to same length
            max_length = max(inp['input_ids'].shape[-1] for inp in batch_inputs)
            
            padded_input_ids = []
            attention_masks = []
            prompt_lengths = []
            
            for inp in batch_inputs:
                input_ids = inp['input_ids']
                prompt_lengths.append(input_ids.shape[-1])
                
                # Pad to max_length
                pad_length = max_length - input_ids.shape[-1]
                if pad_length > 0:
                    pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                    padding = torch.full((1, pad_length), pad_token_id, 
                                       dtype=input_ids.dtype, device=input_ids.device)
                    input_ids = torch.cat([input_ids, padding], dim=1)
                    
                    # Update attention mask
                    attention_mask = inp.get('attention_mask')
                    if attention_mask is not None:
                        mask_padding = torch.zeros((1, pad_length), 
                                                  dtype=attention_mask.dtype, 
                                                  device=attention_mask.device)
                        attention_mask = torch.cat([attention_mask, mask_padding], dim=1)
                    else:
                        # Create attention mask if not present
                        attention_mask = torch.ones_like(input_ids)
                        attention_mask[:, -pad_length:] = 0
                else:
                    attention_mask = inp.get('attention_mask')
                    if attention_mask is None:
                        attention_mask = torch.ones_like(input_ids)
                
                padded_input_ids.append(input_ids)
                attention_masks.append(attention_mask)
            
            # Stack into batch tensors
            batch_input_ids = torch.cat(padded_input_ids, dim=0)
            batch_attention_masks = torch.cat(attention_masks, dim=0)
            
            # Generate for entire batch
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            
            outputs = self.model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks,
                max_new_tokens=1024,  # Reduced from 2048 for faster generation
                repetition_penalty=1.2,
                do_sample=False,
                pad_token_id=pad_token_id,
                # GPU optimization parameters
                use_cache=True,
                early_stopping=False,
                output_attentions=False,
                output_hidden_states=False,
                # Maximize GPU throughput
                num_beams=1,  # Greedy decoding (fastest)
            )
            
            # Decode each output, skipping the prompt
            for j, output in enumerate(outputs):
                generated_text = self.tokenizer.decode(
                    output[prompt_lengths[j]:], 
                    skip_special_tokens=True
                )
                results.append(re.sub(r'[\t]+', "", generated_text))
                
                # Call progress callback for each individual response
                if progress_callback:
                    progress_callback(1)
            
        except Exception as ex:
            error_msg = f"Batch inference error: {ex}"
            self.logger.error(error_msg)
            # Add error placeholders for failed batch items
            for _ in batch:
                results.append(f"Error: {str(ex)}")
                # Call progress callback for each failed item
                if progress_callback:
                    progress_callback(1)
        
        return results
