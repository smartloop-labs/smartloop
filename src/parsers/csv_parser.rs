//! Port of `smartloop/parsers/csv_parser.py`.
//!
//! A parser for processing .csv files into a Markdown table.

use crate::parsers::base_parser::{BaseParser, SharedLLM};
#[allow(unused_imports)]
use crate::paths::expand_home;

/// A parser for processing .csv files.
pub struct CsvParser {
    pub llm: Option<SharedLLM>,
    pub file_path: String,
}

impl CsvParser {
    pub fn new(llm: Option<SharedLLM>, file_path: impl Into<String>) -> Self {
        Self { llm, file_path: file_path.into() }
    }

    /// Validates the CSV file structure: not empty and every row has the same
    /// number of columns as the header.
    pub fn is_valid_csv(&self) -> anyhow::Result<bool> {
        todo!()
    }

    /// Converts the CSV file content to a Markdown table.
    pub fn convert_to_markdown(&self) -> anyhow::Result<String> {
        todo!()
    }
}

impl BaseParser for CsvParser {
    fn llm(&self) -> Option<&SharedLLM> {
        self.llm.as_ref()
    }
    fn file_path(&self) -> &str {
        &self.file_path
    }
    fn type_name(&self) -> &'static str {
        "CsvParser"
    }
    fn parse(&self) -> anyhow::Result<String> {
        todo!()
    }
}
