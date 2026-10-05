//! Port of `smartloop/parsers/text_parser.py`.

use crate::parsers::base_parser::{BaseParser, SharedLLM};

/// Plain-text parser (Python used langchain's `TextLoader`, hand-ported here).
pub struct TextParser {
    pub llm: Option<SharedLLM>,
    pub file_path: String,
}

impl TextParser {
    pub fn new(llm: Option<SharedLLM>, file_path: impl Into<String>) -> Self {
        Self { llm, file_path: file_path.into() }
    }

    /// Read the whole file as UTF-8 text (langchain `TextLoader` semantics:
    /// errors as `Error loading {path}`).
    pub fn load_text(&self) -> anyhow::Result<String> {
        todo!()
    }

    /// Split on blank lines, normalize whitespace and drop empty paragraphs.
    pub fn split_by_paragraph(&self, text: &str) -> Vec<String> {
        let _ = text;
        todo!()
    }
}

impl BaseParser for TextParser {
    fn llm(&self) -> Option<&SharedLLM> {
        self.llm.as_ref()
    }
    fn file_path(&self) -> &str {
        &self.file_path
    }
    fn type_name(&self) -> &'static str {
        "TextParser"
    }
    fn parse(&self) -> anyhow::Result<String> {
        self.load_text()
    }
}
