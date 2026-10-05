//! Port of `smartloop/agents/mcp/file_ops.py`.
//!
//! Binary document parsing for the explorer.

use std::sync::Arc;

#[allow(unused_imports)]
use crate::llms::BaseLLM;
#[allow(unused_imports)]
use crate::parsers::ParserFactory;

/// Logger name used by the Python module.
pub const LOGGER: &str = "smartloop.agents.mcp.file_ops";

/// Binary document parsing for the explorer.
///
/// File read/write/grep/bash now run through the smartloop system-tools MCP
/// server (see `crate::api::mcp::system_server`); the only direct helper
/// still needed is binary parsing, which requires an LLM and so cannot live
/// in the stdio MCP subprocess.
pub struct FileOperations {
    pub llm: Arc<dyn BaseLLM>,
}

impl FileOperations {
    /// Extensions routed to [`FileOperations::parse_binary_file`].
    pub const BINARY_EXTENSIONS: &'static [&'static str] = &[".pdf", ".docx", ".pptx", ".csv"];

    /// Wrap the LLM used by the parsers (vision/OCR, summaries).
    pub fn new(llm: Arc<dyn BaseLLM>) -> Self {
        Self { llm }
    }

    /// Parse a binary document to text by its guessed mime type.
    ///
    /// Returns `None` when the mime type cannot be guessed, no parser handles
    /// it (`[file-read] No parser for mime type: ...`), or the parser fails.
    pub fn parse_binary_file(&self, file_path: &str) -> Option<String> {
        let _ = file_path;
        todo!()
    }
}
