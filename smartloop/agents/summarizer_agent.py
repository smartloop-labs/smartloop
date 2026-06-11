import json
import re

from langchain_core.prompts import PromptTemplate

from smartloop.agents.agent import Agent
from smartloop.llms.base_llm import BaseLLM
from logging import Logger
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from typing import List

from json_repair import repair_json

class SummarizerResponse(BaseModel):
    text: str
    keywords: List[str]


class SummarizerAgent(Agent[SummarizerResponse]):
    """
    Summarizes a given input document
    """

    def __init__(self, llm: BaseLLM, logger: Logger):
        self.parser = PydanticOutputParser[SummarizerResponse](pydantic_object=SummarizerResponse)
        prompt = PromptTemplate(
            template=("""
<system>You are an expert at document summarizer. You are given a input document and you need to summarize it.</system>
<user>
Summarize the input document, provide all important information, and return the summary in the "text" field.
{document}

Return your response in this exact JSON format:
{{
    "text": "summary"
    "keywords": ["keyword1", "keyword2", "keyword3"]
}}</user>"""),
            input_variables=['document']
        )
        super().__init__(llm=llm, return_type=SummarizerResponse, prompt=prompt, logger=logger)

    def invoke(self, document: str) -> SummarizerResponse:
        try:
            formatted_prompt = self.prompt.format(document=document)
            response = self.llm.invoke(formatted_prompt)
            return self.parser.parse(repair_json(response))
        except Exception as e:
            self.logger.error(f"Error generating prompts: {e}")
            return SummarizerResponse(text="", keywords=[])
