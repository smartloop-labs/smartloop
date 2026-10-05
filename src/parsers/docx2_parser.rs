//! Port of `smartloop/parsers/docx2_parser.py`.
//!
//! A parser for .docx files; a thin wrapper over [`DocumentParser`].

use crate::parsers::base_parser::{BaseParser, SharedLLM};
use crate::parsers::document_parser::DocumentParser;

/// `.docx` parser (Python `class Docx2Parser(DocumentParser): pass`).
pub struct Docx2Parser {
    pub inner: DocumentParser,
}

impl Docx2Parser {
    pub fn new(llm: Option<SharedLLM>, file_path: impl Into<String>) -> Self {
        Self { inner: DocumentParser::new(llm, file_path) }
    }
}

impl std::ops::Deref for Docx2Parser {
    type Target = DocumentParser;
    fn deref(&self) -> &DocumentParser {
        &self.inner
    }
}

impl BaseParser for Docx2Parser {
    fn llm(&self) -> Option<&SharedLLM> {
        self.inner.llm()
    }
    fn file_path(&self) -> &str {
        self.inner.file_path()
    }
    fn type_name(&self) -> &'static str {
        "Docx2Parser"
    }
    fn parse(&self) -> anyhow::Result<String> {
        self.inner.parse()
    }
}
