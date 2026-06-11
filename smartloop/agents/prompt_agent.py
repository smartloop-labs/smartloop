import json
import re

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from smartloop.agents.agent import Agent
from smartloop.llms.base_llm import BaseLLM
from logging import Logger
from typing import List, Optional
from pydantic import BaseModel
from json_repair import repair_json

class DocumentInput(BaseModel):
    text: str
    id: str


class Prompt(BaseModel):
    prompt: str
    text: str

class PromptResponse(BaseModel):
    id: str
    prompts: List[Prompt]

class PromptAgent(Agent[PromptResponse]):
    """
    Generates prompts for a given document
    """

    def __init__(self, llm: BaseLLM, logger: Logger):
        self.parser = PydanticOutputParser(pydantic_object=PromptResponse)
        prompt = PromptTemplate(
            template=("""You are an expert prompt generator. Analyze the input document and generate 5-7 questions with their answers.
Input Document:
{document}

Output Format: Return ONLY valid JSON in this exact structure:

{{
  "id": "document_id_here",
  "prompts": [
    {{"prompt": "question 1", "text": "answer 1"}},
    {{"prompt": "question 2", "text": "answer 2"}},
    {{"prompt": "question 3", "text": "answer 3"}}
  ]
}}

Important:
- Generate 5-7 **question-answer** pairs based on the document
- Each **question** should be answerable from the document content.
- **answer** should be in the same language as the question.
- **answer** should be as detailed as possible, it should include keywords and important information.
- Return ONLY the JSON object, no other text
- Make sure the JSON is valid and properly formatted
"""),
            input_variables=["document"],
        )
        super().__init__(llm=llm, return_type=PromptResponse, prompt=prompt, logger=logger)

    
    def _parse_response(self, response: str, doc_id: str = "") -> PromptResponse:
        """Parse LLM response into GeneratePromptResponse object"""
        import json
        
        try:
            # Extract JSON from response if it contains extra text
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            else:
                json_text = response
            
            # Try to repair malformed JSON
            cleaned_response = repair_json(json_text)
            if not cleaned_response:
                self.logger.error(f"JSON repair failed for response: {response[:200]}")
                raise ValueError(f"Error parsing response")
            
            # Parse the JSON manually first to check structure
            try:
                data = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error: {e}. Response: {cleaned_response[:200]}")
                raise ValueError(f"Invalid JSON: {e}")
            
            # Ensure proper structure
            if not isinstance(data, dict):
                self.logger.error(f"Response is not a dict: {type(data)}")
                return PromptResponse(id=doc_id, prompts=[])
            
            # Set id if missing
            if "id" not in data:
                data["id"] = doc_id
            
            # Ensure prompts is a list
            if "prompts" not in data:
                data["prompts"] = []
            elif not isinstance(data["prompts"], list):
                self.logger.error(f"prompts field is not a list: {type(data['prompts'])}")
                data["prompts"] = []
            
            # Create response object directly from validated data
            return PromptResponse(
                id=data.get("id", doc_id),
                prompts=[
                    Prompt(prompt=p.get("prompt", ""), text=p.get("text", ""))
                    for p in data.get("prompts", [])
                    if isinstance(p, dict) and "prompt" in p and "text" in p
                ]
            )
        except Exception as e:
            # Return empty if all parsing attempts fail
            self.logger.error(f"Could not parse response for doc {doc_id}: {e}. Response snippet: {response[:200]}")
            return PromptResponse(id=doc_id, prompts=[])

    def invoke(self, document: DocumentInput) -> PromptResponse:
        try:
            formatted_prompt = self.prompt.format(document=document.model_dump_json())
            response = self.llm.invoke(formatted_prompt)
            return self._parse_response(response, doc_id=document.id)
        except Exception as e:
            self.logger.error(f"Error generating prompts: {e}")
            return PromptResponse(id=document.id, prompts=[])
