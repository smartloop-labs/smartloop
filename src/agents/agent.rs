//! Generic LLM agent base.
//!
//! Port of `smartloop/agents/agent.py`.  The LangChain pieces it used
//! (`PromptTemplate`, `PydanticOutputParser`) and the `json_repair` package
//! have no Rust equivalent and are hand-ported here.

use std::marker::PhantomData;
use std::sync::Arc;

use serde::de::DeserializeOwned;

use crate::llms::base_llm::BaseLLM;

/// Shared handle to an LLM.
pub type SharedLLM = Arc<dyn BaseLLM + Send + Sync>;

/// LangChain `PromptTemplate` (f-string format): `{name}` placeholders are
/// substituted, `{{` / `}}` render as literal braces.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PromptTemplate {
    pub template: String,
    pub input_variables: Vec<String>,
}

impl PromptTemplate {
    /// Build a template.
    pub fn new(template: impl Into<String>, input_variables: &[&str]) -> Self {
        Self {
            template: template.into(),
            input_variables: input_variables.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Render the template with `vars` (`name -> value`).  Fails when a
    /// declared input variable is missing (LangChain raises `KeyError`).
    pub fn format(&self, vars: &[(&str, &str)]) -> anyhow::Result<String> {
        let _ = vars;
        todo!()
    }
}

/// LangChain `PydanticOutputParser.parse`: parse `text` as JSON into `T`.
pub fn parse_output<T: DeserializeOwned>(text: &str) -> anyhow::Result<T> {
    let _ = text;
    todo!()
}

/// Hand-port of `json_repair.repair_json`: best-effort repair of malformed
/// JSON (missing commas/quotes/brackets, trailing commas, single quotes,
/// surrounding prose).  Returns the repaired JSON text, or `""` when nothing
/// JSON-like could be recovered.
pub fn repair_json(text: &str) -> String {
    let _ = text;
    todo!()
}

/// Input to [`Agent::invoke`]: a single prompt or a list of prompts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentInput {
    Single(String),
    Many(Vec<String>),
}

impl From<&str> for AgentInput {
    fn from(s: &str) -> Self {
        AgentInput::Single(s.to_string())
    }
}

impl From<String> for AgentInput {
    fn from(s: String) -> Self {
        AgentInput::Single(s)
    }
}

impl From<Vec<String>> for AgentInput {
    fn from(v: Vec<String>) -> Self {
        AgentInput::Many(v)
    }
}

/// An agent: an LLM, a prompt template, and the structured type `T` its
/// responses parse into.
pub struct Agent<T> {
    pub llm: SharedLLM,
    pub prompt: PromptTemplate,
    return_type: PhantomData<fn() -> T>,
}

impl<T> Clone for Agent<T> {
    fn clone(&self) -> Self {
        Self { llm: self.llm.clone(), prompt: self.prompt.clone(), return_type: PhantomData }
    }
}

impl<T: DeserializeOwned> Agent<T> {
    /// Build an agent over `llm` with `prompt`.
    pub fn new(llm: SharedLLM, prompt: PromptTemplate) -> Self {
        Self { llm, prompt, return_type: PhantomData }
    }

    /// Clean LLM response by removing markdown fences and non-ASCII characters.
    pub fn clean_response(&self, text: &str) -> String {
        let _ = text;
        todo!()
    }

    /// Run each prompt through the LLM sequentially and parse every response
    /// into `T`.  Fails with `"Failed to parse response: ..."` when a response
    /// is empty after cleaning or does not parse.
    pub fn invoke(&self, input: AgentInput) -> anyhow::Result<Vec<T>> {
        let _ = input;
        todo!()
    }
}
