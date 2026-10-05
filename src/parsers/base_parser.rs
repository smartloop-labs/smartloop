//! Port of `smartloop/parsers/base_parser.py`.
//!
//! Shared parser behaviour: chunking, sanitizing and paragraph processing.
//! Python's abstract `BaseParser` class becomes the [`BaseParser`] trait; the
//! concrete parsers implement the accessors and `parse`, and inherit the
//! helper methods as default trait methods.
//!
//! Python passed a `logging.Logger` into every parser; in Rust logging goes
//! through `tracing` with `target: "smartloop.parsers.<module>"`, so the
//! logger argument is dropped.

use std::sync::Arc;

use once_cell::sync::Lazy;
use regex::Regex;

use crate::llms::base_llm::BaseLLM;
use crate::parsers::document_chunk::DocumentChunk;

/// Shared handle to the LLM used for token counting (Python `Optional[BaseLLM]`).
pub type SharedLLM = Arc<dyn BaseLLM + Send + Sync>;

/// Maximum words per chunk; `SLP_PARSER_MAX_WORDS_PER_CHUNK` env var, default 1024.
pub static MAX_WORDS_PER_CHUNK: Lazy<usize> = Lazy::new(|| {
    std::env::var("SLP_PARSER_MAX_WORDS_PER_CHUNK")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(1024)
});

/// Errors raised by parsers that callers distinguish (Python `FileNotFoundError`
/// and `ValueError`). `parse()` returns `anyhow::Result`, wrapping these so
/// callers can `downcast_ref::<ParserError>()`.
#[derive(Debug, thiserror::Error)]
pub enum ParserError {
    /// Python `FileNotFoundError(f"File not found: {filepath}")`.
    #[error("File not found: {0}")]
    FileNotFound(String),
    /// Python `ValueError(msg)`.
    #[error("{0}")]
    Value(String),
}

/// `\s+|(\.)` — whitespace runs and periods (each replaced by a space).
static SANITIZE_WS_OR_DOT: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+|(\.)").expect("valid regex"));
/// `\\t+` — a literal backslash followed by `t`s (escaped tabs).
static SANITIZE_ESCAPED_TABS: Lazy<Regex> = Lazy::new(|| Regex::new(r"\\t+").expect("valid regex"));
/// `^\s+|\s+$` — leading/trailing whitespace.
static SANITIZE_EDGES: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\s+|\s+$").expect("valid regex"));
/// `\s+` — any whitespace run.
static WHITESPACE_RUN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").expect("valid regex"));

/// Abstract document parser (Python `BaseParser(ABC)`).
pub trait BaseParser: Send + Sync {
    /// The LLM used for token counting, if any.
    fn llm(&self) -> Option<&SharedLLM>;

    /// Path of the document being parsed.
    fn file_path(&self) -> &str;

    /// Concrete type name (Python `type(parser).__name__`), used in logs.
    fn type_name(&self) -> &'static str;

    /// Parse the document and return its text (usually Markdown).
    fn parse(&self) -> anyhow::Result<String>;

    /// Minimum number of chunks of `max_items_per_chunk` (default
    /// [`MAX_WORDS_PER_CHUNK`]) needed to hold `total_items`.
    fn calculate_number_of_chunks(&self, total_items: usize, max_items_per_chunk: Option<usize>) -> usize {
        // Int arithmetic to get the minimum number of chunks to hold a paragraph
        let max_items = max_items_per_chunk.unwrap_or(*MAX_WORDS_PER_CHUNK);
        total_items.div_ceil(max_items)
    }

    /// Split a list of words into space-joined chunks of `MAX_WORDS_PER_CHUNK` words.
    fn chunkify_paragraph(&self, words: &[String]) -> Vec<String> {
        let chunk_count = self.calculate_number_of_chunks(words.len(), None);
        let max_words = *MAX_WORDS_PER_CHUNK;

        let mut chunks = Vec::with_capacity(chunk_count);
        for i in 0..chunk_count {
            let start = (i * max_words).min(words.len());
            let end = (start + max_words).min(words.len());
            chunks.push(words[start..end].join(" "));
        }
        chunks
    }

    /// Collapse whitespace/periods/dashes to spaces, trim and lowercase.
    fn sanitize(&self, paragraph: &str) -> String {
        let paragraph = SANITIZE_WS_OR_DOT.replace_all(paragraph, " ");
        let paragraph = SANITIZE_ESCAPED_TABS.replace_all(&paragraph, "");
        let paragraph = paragraph.replace('-', " ");
        let paragraph = SANITIZE_EDGES.replace_all(&paragraph, "");
        // Only ASCII capitals are lowered (Python `re.sub(r'[A-Z]', lower, ...)`).
        paragraph.chars().map(|c| c.to_ascii_lowercase()).collect()
    }

    /// Wrap a whole document's text in a single `DocumentChunk`.
    fn create_chunk_from_document(&self, document_id: &str, document: &str) -> DocumentChunk {
        DocumentChunk {
            document_id: document_id.to_string(),
            chunk_id: "0".to_string(),
            paragraph_id: uuid::Uuid::new_v4().to_string(),
            text: document.to_string(),
            total_chunks: 1,
            ..DocumentChunk::default()
        }
    }

    /// Strip down the paragraphs and break them into DocumentChunks.
    ///
    /// Requires an LLM for token counting: Python dereferenced `self.llm`
    /// unconditionally and raised `AttributeError` on `None`.
    fn process_paragraphs(&self, paragraphs: &[String], document_id: &str) -> Vec<DocumentChunk> {
        let mut document_chunks = Vec::new();
        let mut current: Vec<String> = Vec::new();
        let mut token_count: usize = 0;

        for paragraph in paragraphs {
            if paragraph.trim().is_empty() {
                continue;
            }

            let sanitized = self.sanitize(paragraph);
            let llm = self
                .llm()
                .expect("'NoneType' object has no attribute 'get_max_tokens': process_paragraphs requires an LLM");
            let max_tokens = llm.get_max_tokens();

            // Python compares the running token count plus the paragraph's
            // *character* length against the budget, then adds its token count.
            if token_count + sanitized.chars().count() < max_tokens {
                token_count += llm.get_num_tokens(&sanitized);
                current.push(sanitized);
            } else {
                document_chunks.push(self.create_chunk_from_document(document_id, &current.join("\n")));
                token_count = 0;
                current = vec![sanitized];
            }
        }

        if !current.is_empty() {
            document_chunks.push(self.create_chunk_from_document(document_id, &current.join("\n")));
        }

        document_chunks
    }

    /// Strip leading/trailing space & multiple newlines, also convert remaining
    /// newlines to spaces; drops empty items.
    fn normalize_whitespace(&self, text_items: &[String]) -> Vec<String> {
        let mut results = Vec::new();
        for item in text_items {
            let item = WHITESPACE_RUN.replace_all(item, " ");
            // Replace remaining newlines with spaces
            let item = item.replace('\n', " ");
            let item = item.trim();
            if !item.is_empty() {
                results.push(item.to_string());
            }
        }
        results
    }
}

/// Read a text file the way Python's `open(path, encoding="utf-8").read()`
/// does: strict UTF-8 decoding with universal-newline translation (`\r\n` and
/// lone `\r` become `\n`).
pub fn read_text_universal(path: &str) -> anyhow::Result<String> {
    let bytes = std::fs::read(path)?;
    let text = String::from_utf8(bytes)
        .map_err(|e| anyhow::anyhow!("'utf-8' codec can't decode file {path}: {e}"))?;
    Ok(translate_newlines(&text))
}

/// Universal-newline translation: `\r\n` and lone `\r` become `\n`.
pub fn translate_newlines(text: &str) -> String {
    if !text.contains('\r') {
        return text.to_string();
    }
    text.replace("\r\n", "\n").replace('\r', "\n")
}
