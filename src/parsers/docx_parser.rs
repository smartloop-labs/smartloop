//! Port of `smartloop/parsers/docx_parser.py`.
//!
//! A parser for processing .docx files in XML format (python-docx paragraph
//! walking hand-ported over `zip` + `quick-xml`).

use crate::parsers::base_parser::{BaseParser, SharedLLM};

/// Paragraph alignment (python-docx `WD_PARAGRAPH_ALIGNMENT`), from `w:jc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParagraphAlignment {
    Left,
    Center,
    Right,
    Justify,
    Distribute,
    Other,
}

/// One `w:p` paragraph: its text, resolved style name and explicit alignment.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DocxParagraph {
    pub text: String,
    /// Style display name (e.g. "Normal", "Heading 1", "List Paragraph").
    pub style_name: String,
    /// Explicit paragraph alignment; `None` when inherited.
    pub alignment: Option<ParagraphAlignment>,
}

/// A parser for processing .docx files in XML format.
pub struct DocxParser {
    pub llm: Option<SharedLLM>,
    pub file_path: String,
}

impl DocxParser {
    pub fn new(llm: Option<SharedLLM>, file_path: impl Into<String>) -> Self {
        Self { llm, file_path: file_path.into() }
    }

    /// Determines if a paragraph is a list item (style starts with "List" or
    /// explicit left alignment).
    pub fn is_list_item(&self, paragraph: &DocxParagraph) -> bool {
        let _ = paragraph;
        todo!()
    }

    /// Read every paragraph of the document body.
    pub fn read_paragraphs(&self) -> anyhow::Result<Vec<DocxParagraph>> {
        todo!()
    }

    /// Group paragraphs into heading-led text blocks.
    pub fn load_paragraphs(&self) -> anyhow::Result<Vec<String>> {
        todo!()
    }
}

impl BaseParser for DocxParser {
    fn llm(&self) -> Option<&SharedLLM> {
        self.llm.as_ref()
    }
    fn file_path(&self) -> &str {
        &self.file_path
    }
    fn type_name(&self) -> &'static str {
        "DocxParser"
    }
    fn parse(&self) -> anyhow::Result<String> {
        Ok(self.load_paragraphs()?.join("\n\n"))
    }
}
