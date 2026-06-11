from logging import Logger
import re
import json
from typing import Generic, TypeVar

from pydantic import BaseModel
from smartloop.llms.base_llm import BaseLLM
from langchain_core.output_parsers import PydanticOutputParser
from typing import Union, List
from langchain_core.prompts import PromptTemplate

from abc import ABC, abstractmethod

# Define generic type for output model
T = TypeVar('T', bound=BaseModel)

class Agent(ABC, Generic[T]):
    """
    An agent is a class that can be used to create a new agent.
    """

    def __init__(self, llm: BaseLLM, prompt: PromptTemplate, return_type: type[T], logger: Logger):
        self.llm = llm
        self.prompt = prompt
        self.return_type = return_type
        self.logger = logger
        
    def _clean_response(self, text: str) -> str:
        """Clean LLM response by removing markdown fences and non-ASCII characters"""
        # Remove non-ASCII characters first
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        
        # Remove markdown code fences more aggressively
        text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```\s*', '', text)
        text = re.sub(r'^```', '', text)
        text = re.sub(r'```$', '', text)
    
        # Remove any remaining backticks
        text = text.replace('`', '')
        # Clean up whitespace and newlines
        text = text.replace('\n', '')
        text = text.strip()

        return text

    @abstractmethod
    def invoke(self, input: Union[str, List[str]]) -> Union[T, List[T]]:
        results = []

        # Process inputs sequentially for robustness
        if isinstance(input, str):
            responses = [self.llm.invoke(input)]
        else:
            # Process list of inputs one at a time
            responses = [self.llm.invoke(prompt) for prompt in input]

        for i, resp in enumerate(responses):
            parser = PydanticOutputParser[self.return_type](pydantic_object=self.return_type)
            try:
                raw_response = resp
                cleaned_response = self._clean_response(raw_response)
                if not cleaned_response:
                    raise ValueError("Empty response after cleaning")
                results.append(parser.parse(cleaned_response))
            except ValueError as e:
                raise ValueError(f"Failed to parse response: {e}")

        return results
