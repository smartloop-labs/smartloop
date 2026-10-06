//! Port of `smartloop/parsers/text_parser.py`.

use once_cell::sync::Lazy;
use regex::Regex;

use crate::parsers::base_parser::{read_text_universal, BaseParser, SharedLLM};

/// `[\n][\n]+` — paragraph separator (two or more newlines).
static PARAGRAPH_SPLIT: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\n][\n]+").expect("valid regex"));
/// `\s+` — any whitespace run.
static WHITESPACE_RUN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").expect("valid regex"));

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
        // TextLoader opens the file in text mode (universal newlines) and
        // returns a single document holding the whole content.
        read_text_universal(&self.file_path).map_err(|e| e.context(format!("Error loading {}", self.file_path)))
    }

    /// Split on blank lines, normalize whitespace and drop empty paragraphs.
    pub fn split_by_paragraph(&self, text: &str) -> Vec<String> {
        let mut processed_paragraphs = Vec::new();
        for para in PARAGRAPH_SPLIT.split(text) {
            // Normalize whitespace: replace multiple spaces with a single space
            let para = WHITESPACE_RUN.replace_all(para, " ");
            // Replace remaining newlines with spaces
            let para = para.replace('\n', " ");
            let para = para.trim();
            if !para.is_empty() {
                processed_paragraphs.push(para.to_string());
            }
        }
        processed_paragraphs
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
