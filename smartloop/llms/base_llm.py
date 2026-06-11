"""Base Model for all LLM classes"""
import threading
from typing import Generic, Optional, Any
from abc import abstractmethod
import logging
from smartloop.constants import SYSTEM_PROMPT

import re
from typing import List, Generic

from langchain_core.language_models.llms import LLM

from typing import TypeVar

T = TypeVar('T', bound=Any)

class BaseLLM(LLM, Generic[T]):
    project_id: str
    model: T
    tokenizer: Any
    logger: logging.Logger
    is_training: bool = False

    """LLM base class"""
    def __init__(self, model: T, tokenizer: Any, logger: logging.Logger, project_id: str, is_training: bool = False):
        super().__init__(model=model, tokenizer=tokenizer, logger=logger, project_id=project_id, is_training=is_training)
        self._cancelled: Optional[threading.Event] = None
 
    def get_num_tokens(self, text: str) -> int:
        raise ValueError("LLM specific")

    def format_prompt(self, prompt: str, system_message: Optional[str] = None) -> List[dict]:
        """
        Format the prompt for the LLM in a standardized way.

        This method creates a standard message format that can be used across
        different LLM implementations, similar to llama.cpp's approach.

        Args:
            prompt: The user's input prompt
            system_message: Optional system message (defaults to helpful assistant)

        Returns:
            Formatted prompt - can be string or messages list depending on LLM type
        """
     
        # Create standard message format
        messages = [
            {"role": "system", "content": system_message or SYSTEM_PROMPT},
            {"role": "user", "content": f"{prompt}"}
         ]

        # Default implementation returns messages - subclasses can override
        return messages

        
    def create_messages_from_prompt(self, prompt: str) -> list:
        """Convert a prompt string to the proper message format using tag-based separation"""
        messages = []

        # Find all <system>, <user> and <assistant> tags with their content
        system_pattern = r'<system>(.*?)</system>'
        user_pattern = r'<user>(.*?)</user>'
        assistant_pattern = r'<assistant>(.*?)</assistant>'

        find_matches = lambda prompt, pattern: [(m.start(), m.end(), m.group(1).strip()) for m in re.finditer(pattern, prompt, re.DOTALL)]

        # Find all matches and their positions
        system_matches = find_matches(prompt, system_pattern)
        user_matches = find_matches(prompt, user_pattern)
        assistant_matches = find_matches(prompt, assistant_pattern)

        # Combine and sort by position
        all_matches = []
        all_matches.extend([(pos, 'system', content) for pos, _, content in system_matches])
        all_matches.extend([(pos, 'user', content) for pos, _, content in user_matches])
        all_matches.extend([(pos, 'assistant', content) for pos, _, content in assistant_matches])
        all_matches.sort(key=lambda x: x[0])

        # Create messages from tags
        for _, role, content in all_matches:
            if content:  # Only add non-empty content
                messages.append({
                    "role": role,
                    "content": content
                })

        # If no tags found, process the raw prompt as a user message
        if not messages:
           return self.format_prompt(f"user_question: {prompt}")

        # Merge consecutive messages with the same role so the model
        # always sees alternating user/assistant turns.
        merged: list[dict] = [messages[0]]
        for msg in messages[1:]:
            if msg["role"] == merged[-1]["role"]:
                merged[-1] = {
                    "role": msg["role"],
                    "content": merged[-1]["content"] + "\n" + msg["content"],
                }
            else:
                merged.append(msg)

        return merged


    @abstractmethod
    def _apply_chat_template(self, messages: list) -> any:
        """
        Apply the LLM-specific chat template to the standard message format.
        This is where each LLM implementation handles its specific formatting.

        Args:
            messages: List of {"role": ..., "content": ...} messages

        Returns:
            LLM-specific formatted prompt
        """
        raise NotImplementedError("LLM subclasses must implement _apply_chat_template method")

    def get_max_tokens(self) -> int:
        # default max
        return self.model.max_tokens
