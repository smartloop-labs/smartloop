//! Generates question/answer prompts for a document.
//!
//! Port of `smartloop/agents/prompt_agent.py`.

use serde::{Deserialize, Serialize};

use crate::agents::agent::{Agent, PromptTemplate, SharedLLM};

/// A document handed to the prompt generator.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DocumentInput {
    pub text: String,
    pub id: String,
}

/// One generated question (`prompt`) and its answer (`text`).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Prompt {
    pub prompt: String,
    pub text: String,
}

/// The prompts generated for one document.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct PromptResponse {
    pub id: String,
    pub prompts: Vec<Prompt>,
}

/// The prompt-generation template (verbatim from the Python source).
pub const PROMPT_AGENT_TEMPLATE: &str = r#"You are an expert prompt generator. Analyze the input document and generate 5-7 questions with their answers.
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
"#;

/// Generates prompts for a given document.
#[derive(Clone)]
pub struct PromptAgent {
    pub base: Agent<PromptResponse>,
}

impl PromptAgent {
    /// Build the agent over `llm`.
    pub fn new(llm: SharedLLM) -> Self {
        Self { base: Agent::new(llm, PromptTemplate::new(PROMPT_AGENT_TEMPLATE, &["document"])) }
    }

    /// Parse an LLM response into a [`PromptResponse`]; never fails — any
    /// parse problem yields an empty response for `doc_id`.
    pub fn parse_response(&self, response: &str, doc_id: &str) -> PromptResponse {
        let _ = (response, doc_id);
        todo!()
    }

    /// Generate prompts for `document`; errors yield an empty response.
    pub fn invoke(&self, document: &DocumentInput) -> PromptResponse {
        let _ = document;
        todo!()
    }
}
