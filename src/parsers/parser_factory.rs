//! Port of `smartloop/parsers/parser_factory.py`.
//!
//! Creates a parser object based on the requested file's mimetype.

use std::collections::HashMap;

use once_cell::sync::Lazy;

use crate::parsers::base_parser::{BaseParser, ParserError, SharedLLM};
use crate::parsers::csv_parser::CsvParser;
use crate::parsers::document_parser::DocumentParser;
use crate::parsers::docx2_parser::Docx2Parser;
use crate::parsers::json_parser::{looks_like_json, JsonParser};
use crate::parsers::pdf_parser::PdfParser;
use crate::parsers::text_parser::TextParser;

/// Logger name used by the Python module.
const LOGGER: &str = "smartloop.parsers.parser_factory";

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
        // PDF, DOCX, PPTX, XLSX, HTML and images are all converted to markdown
        // via MarkItDown (DocumentParser); PDF/DOCX keep thin subclasses for
        // readability. Images (PNG/JPEG) are OCR'd/described by the vision
        // model through MarkItDown's image converter. TXT and CSV use their own
        // lightweight parsers; JSON (and JSON-bodied TXT) is rendered to Markdown
        // by JsonParser since MarkItDown returns JSON verbatim.
        let table = [
            ("pdf", ParserKind::Pdf),
            ("txt", ParserKind::Text),
            ("docx", ParserKind::Docx2),
            ("pptx", ParserKind::Document),
            ("xlsx", ParserKind::Document),
            ("html", ParserKind::Document),
            ("json", ParserKind::Json),
            ("png", ParserKind::Document),
            ("jpeg", ParserKind::Document),
            ("csv", ParserKind::Csv),
        ];
        let parsers = table
            .iter()
            .filter_map(|(name, kind)| Self::supported_document_type(name).map(|mime| (mime.to_string(), *kind)))
            .collect();
        Self { llm, parsers }
    }

    /// Return a mime-type appropriate instance of a parser. A `.txt` whose body
    /// looks like JSON is routed to JsonParser. Errors with
    /// `No parser available for mime_type: '{mime_type}'`.
    pub fn get_parser(&self, mime_type: &str, document_path: &str) -> Result<Box<dyn BaseParser>, ParserError> {
        let mut kind = self.parsers.get(mime_type).copied();

        // A .txt whose body is actually JSON is routed to JsonParser so it gets
        // structured into Markdown instead of stored as a raw blob.
        if kind == Some(ParserKind::Text) && looks_like_json(document_path) {
            kind = Some(ParserKind::Json);
        }

        let Some(kind) = kind else {
            return Err(ParserError::Value(format!("No parser available for mime_type: '{mime_type}'")));
        };

        let llm = self.llm.clone();
        let parser: Box<dyn BaseParser> = match kind {
            ParserKind::Pdf => Box::new(PdfParser::new(llm, document_path)),
            ParserKind::Text => Box::new(TextParser::new(llm, document_path)),
            ParserKind::Docx2 => Box::new(Docx2Parser::new(llm, document_path)),
            ParserKind::Document => Box::new(DocumentParser::new(llm, document_path)),
            ParserKind::Json => Box::new(JsonParser::new(llm, document_path)),
            ParserKind::Csv => Box::new(CsvParser::new(llm, document_path)),
        };
        tracing::info!(target: LOGGER, "Created a {} for file type {}", parser.type_name(), mime_type);

        Ok(parser)
    }
}
