//! Summarizes a document.
//!
//! Port of `smartloop/agents/summarizer_agent.py`.

use serde::{Deserialize, Serialize};

use crate::agents::agent::{Agent, PromptTemplate, SharedLLM};

/// A summary and its keywords.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct SummarizerResponse {
    pub text: String,
    pub keywords: Vec<String>,
}

/// The summarizer template (verbatim from the Python source).
pub const SUMMARIZER_TEMPLATE: &str = r#"
<system>You are an expert at document summarizer. You are given a input document and you need to summarize it.</system>
<user>
Summarize the input document, provide all important information, and return the summary in the "text" field.
{document}

Return your response in this exact JSON format:
{{
    "text": "summary"
    "keywords": ["keyword1", "keyword2", "keyword3"]
}}</user>"#;

/// Summarizes a given input document.
#[derive(Clone)]
pub struct SummarizerAgent {
    pub base: Agent<SummarizerResponse>,
}

impl SummarizerAgent {
    /// Build the agent over `llm`.
    pub fn new(llm: SharedLLM) -> Self {
        Self { base: Agent::new(llm, PromptTemplate::new(SUMMARIZER_TEMPLATE, &["document"])) }
    }

    /// Summarize `document`; errors yield an empty summary.
    pub fn invoke(&self, document: &str) -> SummarizerResponse {
        let _ = document;
        todo!()
    }
}
