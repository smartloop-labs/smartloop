//! Summarizes a document.
//!
//! Port of `smartloop/agents/summarizer_agent.py`.

use serde::{Deserialize, Serialize};

use crate::agents::agent::{parse_output, repair_json, Agent, PromptTemplate, SharedLLM};

/// Logger name used for this agent's messages.
const LOG_TARGET: &str = "smartloop.agents.summarizer_agent";

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
        let attempt = || -> anyhow::Result<SummarizerResponse> {
            let formatted_prompt = self.base.prompt.format(&[("document", document)])?;
            let response = self.base.llm.invoke(&formatted_prompt, None, None);
            parse_output::<SummarizerResponse>(&repair_json(&response))
        };
        match attempt() {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(target: LOG_TARGET, "Error generating prompts: {}", e);
                SummarizerResponse { text: String::new(), keywords: Vec::new() }
            }
        }
    }
}
