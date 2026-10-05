//! Port of `smartloop/parsers/parser_factory.py`.
//!
//! Creates a parser object based on the requested file's mimetype.

use std::collections::HashMap;

use once_cell::sync::Lazy;

use crate::parsers::base_parser::{BaseParser, ParserError, SharedLLM};
#[allow(unused_imports)]
use crate::parsers::csv_parser::CsvParser;
#[allow(unused_imports)]
use crate::parsers::document_parser::DocumentParser;
#[allow(unused_imports)]
use crate::parsers::docx2_parser::Docx2Parser;
#[allow(unused_imports)]
use crate::parsers::json_parser::{looks_like_json, JsonParser};
#[allow(unused_imports)]
use crate::parsers::pdf_parser::PdfParser;
#[allow(unused_imports)]
use crate::parsers::text_parser::TextParser;

/// Supported document types: short name -> mime type (Python
/// `ParserFactory.SUPPORTED_DOCUMENT_TYPE`, with `mimetypes.types_map` values).
pub static SUPPORTED_DOCUMENT_TYPE: Lazy<Vec<(&'static str, &'static str)>> = Lazy::new(|| {
    vec![
        ("pdf", "application/pdf"),
        ("txt", "text/plain"),
        ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
        ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("html", "text/html"),
        ("csv", "text/csv"),
        ("json", "application/json"),
        ("png", "image/png"),
        ("jpeg", "image/jpeg"),
    ]
});

/// Which concrete parser handles a mime type (Python stored the classes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParserKind {
    Pdf,
    Text,
    Docx2,
    Document,
    Json,
    Csv,
}

/// Create an instance of a file type specific parser object based on mimetype.
pub struct ParserFactory {
    pub llm: Option<SharedLLM>,
    /// mime type -> parser kind.
    pub parsers: HashMap<String, ParserKind>,
}

impl ParserFactory {
    /// Look up a supported mime type by short name (e.g. "pdf").
    pub fn supported_document_type(name: &str) -> Option<&'static str> {
        SUPPORTED_DOCUMENT_TYPE.iter().find(|(k, _)| *k == name).map(|(_, v)| *v)
    }

    /// PDF, DOCX, PPTX, XLSX, HTML and images go through DocumentParser
    /// (PDF/DOCX via thin subclasses); TXT and CSV use lightweight parsers;
    /// JSON (and JSON-bodied TXT) is rendered by JsonParser.
    pub fn new(llm: Option<SharedLLM>) -> Self {
        let _ = llm;
        todo!()
    }

    /// Return a mime-type appropriate instance of a parser. A `.txt` whose body
    /// looks like JSON is routed to JsonParser. Errors with
    /// `No parser available for mime_type: '{mime_type}'`.
    pub fn get_parser(&self, mime_type: &str, document_path: &str) -> Result<Box<dyn BaseParser>, ParserError> {
        let _ = (mime_type, document_path);
        todo!()
    }
}
