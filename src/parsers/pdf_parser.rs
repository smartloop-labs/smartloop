//! Port of `smartloop/parsers/pdf_parser.py`.
//!
//! A parser for PDF files; a thin wrapper over [`DocumentParser`].

use crate::parsers::base_parser::{BaseParser, SharedLLM};
use crate::parsers::document_parser::DocumentParser;

/// PDF parser (Python `class PdfParser(DocumentParser): pass`).
pub struct PdfParser {
    pub inner: DocumentParser,
}

impl PdfParser {
    pub fn new(llm: Option<SharedLLM>, file_path: impl Into<String>) -> Self {
        Self { inner: DocumentParser::new(llm, file_path) }
    }
}

impl std::ops::Deref for PdfParser {
    type Target = DocumentParser;
    fn deref(&self) -> &DocumentParser {
        &self.inner
    }
}

impl BaseParser for PdfParser {
    fn llm(&self) -> Option<&SharedLLM> {
        self.inner.llm()
    }
    fn file_path(&self) -> &str {
        self.inner.file_path()
    }
    fn type_name(&self) -> &'static str {
        "PdfParser"
    }
    fn parse(&self) -> anyhow::Result<String> {
        self.inner.parse()
    }
}
