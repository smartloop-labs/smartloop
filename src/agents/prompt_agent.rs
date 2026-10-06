//! Generates question/answer prompts for a document.
//!
//! Port of `smartloop/agents/prompt_agent.py`.

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agents::agent::{repair_json, Agent, PromptTemplate, SharedLLM};

/// Logger name used for this agent's messages.
const LOG_TARGET: &str = "smartloop.agents.prompt_agent";

/// The outermost `{...}` span of a response (greedy, across lines).
static JSON_OBJECT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)\{.*\}").unwrap());

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

/// The first `n` characters of `s` (Python `s[:n]`).
fn head(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Python's `type(x)` for a JSON value, for log lines.
fn py_type(v: &Value) -> &'static str {
    match v {
        Value::Null => "<class 'NoneType'>",
        Value::Bool(_) => "<class 'bool'>",
        Value::Number(n) if n.is_f64() => "<class 'float'>",
        Value::Number(_) => "<class 'int'>",
        Value::String(_) => "<class 'str'>",
        Value::Array(_) => "<class 'list'>",
        Value::Object(_) => "<class 'dict'>",
    }
}

/// A pydantic `str` field: anything but a string fails validation.
fn str_field(model: &str, name: &str, value: Option<&Value>, default: &str) -> anyhow::Result<String> {
    match value {
        Some(Value::String(s)) => Ok(s.clone()),
        None => Ok(default.to_string()),
        Some(other) => anyhow::bail!(
            "1 validation error for {}\n{}\n  Input should be a valid string [type=string_type, input_value={}]",
            model,
            name,
            other
        ),
    }
}

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
        match Self::try_parse_response(response, doc_id) {
            Ok(parsed) => parsed,
            Err(e) => {
                // Return empty if all parsing attempts fail
                tracing::error!(
                    target: LOG_TARGET,
                    "Could not parse response for doc {}: {}. Response snippet: {}",
                    doc_id,
                    e,
                    head(response, 200)
                );
                PromptResponse { id: doc_id.to_string(), prompts: Vec::new() }
            }
        }
    }

    fn try_parse_response(response: &str, doc_id: &str) -> anyhow::Result<PromptResponse> {
        // Extract JSON from response if it contains extra text
        let json_text = match JSON_OBJECT_RE.find(response) {
            Some(m) => m.as_str(),
            None => response,
        };

        // Try to repair malformed JSON
        let cleaned_response = repair_json(json_text);
        if cleaned_response.is_empty() {
            tracing::error!(target: LOG_TARGET, "JSON repair failed for response: {}", head(response, 200));
            anyhow::bail!("Error parsing response");
        }

        // Parse the JSON manually first to check structure
        let data: Value = match serde_json::from_str(&cleaned_response) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!(
                    target: LOG_TARGET,
                    "JSON decode error: {}. Response: {}",
                    e,
                    head(&cleaned_response, 200)
                );
                anyhow::bail!("Invalid JSON: {}", e);
            }
        };

        // Ensure proper structure
        let mut data = match data {
            Value::Object(map) => map,
            other => {
                tracing::error!(target: LOG_TARGET, "Response is not a dict: {}", py_type(&other));
                return Ok(PromptResponse { id: doc_id.to_string(), prompts: Vec::new() });
            }
        };

        // Set id if missing
        if !data.contains_key("id") {
            data.insert("id".to_string(), Value::String(doc_id.to_string()));
        }

        // Ensure prompts is a list
        let prompts_ok = match data.get("prompts") {
            None => false,
            Some(Value::Array(_)) => true,
            Some(other) => {
                tracing::error!(target: LOG_TARGET, "prompts field is not a list: {}", py_type(other));
                false
            }
        };
        if !prompts_ok {
            data.insert("prompts".to_string(), Value::Array(Vec::new()));
        }

        // Create response object directly from validated data
        let id = str_field("PromptResponse", "id", data.get("id"), doc_id)?;
        let mut prompts = Vec::new();
        if let Some(Value::Array(items)) = data.get("prompts") {
            for p in items {
                let Value::Object(p) = p else { continue };
                if !(p.contains_key("prompt") && p.contains_key("text")) {
                    continue;
                }
                prompts.push(Prompt {
                    prompt: str_field("Prompt", "prompt", p.get("prompt"), "")?,
                    text: str_field("Prompt", "text", p.get("text"), "")?,
                });
            }
        }
        Ok(PromptResponse { id, prompts })
    }

    /// Generate prompts for `document`; errors yield an empty response.
    pub fn invoke(&self, document: &DocumentInput) -> PromptResponse {
        let attempt = || -> anyhow::Result<PromptResponse> {
            // pydantic `model_dump_json()`: compact JSON in field order.
            let document_json = serde_json::to_string(document)?;
            let formatted_prompt = self.base.prompt.format(&[("document", &document_json)])?;
            let response = self.base.llm.invoke(&formatted_prompt, None, None);
            Ok(self.parse_response(&response, &document.id))
        };
        match attempt() {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(target: LOG_TARGET, "Error generating prompts: {}", e);
                PromptResponse { id: document.id.clone(), prompts: Vec::new() }
            }
        }
    }
}
