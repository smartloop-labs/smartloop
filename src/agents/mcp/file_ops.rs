//! Port of `smartloop/agents/mcp/file_ops.py`.
//!
//! Binary document parsing for the explorer.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use crate::llms::BaseLLM;
use crate::parsers::base_parser::{ParserError, SharedLLM};
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
        // `mimetypes.guess_type`: by extension only, `None` when unknown.
        let mime_type = mime_guess::from_path(file_path).first()?.essence_str().to_string();

        let llm: SharedLLM = self.llm.clone();
        // Python wrapped the whole parse in `except Exception`, so a parser
        // that blows up (a panic here) is a failed parse, not a crash.
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let factory = ParserFactory::new(Some(llm));
            match factory.get_parser(&mime_type, file_path) {
                // A `ValueError` raised by the parse itself was caught by the
                // same `except ValueError` as the factory's.
                Ok(parser) => parser.parse().map_err(|e| match e.downcast_ref::<ParserError>() {
                    Some(ParserError::Value(_)) => ParseFailure::NoParser,
                    _ => ParseFailure::Failed(e.to_string()),
                }),
                Err(ParserError::Value(_)) => Err(ParseFailure::NoParser),
                Err(e) => Err(ParseFailure::Failed(e.to_string())),
            }
        }));

        match outcome {
            Ok(Ok(text)) => Some(text),
            Ok(Err(ParseFailure::NoParser)) => {
                tracing::warn!(target: LOGGER, "[file-read] No parser for mime type: {}", mime_type);
                None
            }
            Ok(Err(ParseFailure::Failed(e))) => {
                tracing::error!(target: LOGGER, "[file-read] Parser failed for {}: {}", file_path, e);
                None
            }
            Err(panic) => {
                tracing::error!(
                    target: LOGGER,
                    "[file-read] Parser failed for {}: {}",
                    file_path,
                    panic_message(panic.as_ref())
                );
                None
            }
        }
    }
}

/// Why a parse produced no text.
enum ParseFailure {
    /// `ValueError` from the factory: nothing handles this mime type.
    NoParser,
    /// Any other error while building or running the parser.
    Failed(String),
}

/// The message carried by a panic payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown error".to_string()
    }
}
