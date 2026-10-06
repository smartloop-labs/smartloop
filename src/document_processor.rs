//! DocumentProcessor — parses pending documents and stores their chunks.
//!
//! Port of `smartloop/document_processor.py`.  The Python class took
//! duck-typed collections (`projects`, `documents`, `gridfs`, `vector_db`);
//! here they are small traits so any backing store can be plugged in.
//! Documents and projects are free-form JSON objects, as in Python.  The
//! Python `logger` argument is replaced by `tracing` under the target
//! `smartloop.document_processor`.

use std::path::PathBuf;
use std::sync::Arc;

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::llms::BaseLLM;
use crate::parsers::base_parser::ParserError;
use crate::parsers::document_chunk::DocumentChunk;
use crate::parsers::parser_factory::ParserFactory;

const LOG: &str = "smartloop.document_processor";

/// Raised when a document is missing (or its parser rejects it).
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct DocumentMissingError(pub String);

/// Chunk output (`{"chunks": [...]}`).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ChunkOutput {
    /// List of document chunks
    pub chunks: Vec<Map<String, Value>>,
}

/// A document summary.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Summary {
    pub id: String,
    pub text: String,
    pub num_of_tokens: i64,
}

/// Document key flagging that the token limit was reached.
pub const IS_TOKEN_LIMIT_REACHED: &str = "is_token_limit_reached";
/// Document key holding the document's token count.
pub const NUM_OF_TOKENS: &str = "num_of_tokens";

/// A free-form document / project record.
pub type Record = Map<String, Value>;

/// Project lookup (`projects.get(project_id)`).
pub trait ProjectRepository: Send + Sync {
    fn get(&self, project_id: &str) -> Option<Record>;
}

/// Document records (`documents.*`).
pub trait DocumentRepository: Send + Sync {
    fn get_all_pending(&self, project_id: &str) -> Vec<Record>;
    fn update(&self, document: &Record);
    fn find_by_project(&self, project_id: &str) -> Vec<Record>;
}

/// Blob store (`gridfs.get(file_id).read()`).
pub trait FileStore: Send + Sync {
    fn get(&self, file_id: &Value) -> anyhow::Result<Vec<u8>>;
}

/// Vector store operations used here.
pub trait ChunkVectorStore: Send + Sync {
    fn open_project(&self, project_id: &str);
    fn delete_all_chunks_by_document_id(&self, document_id: &str);
    fn upsert_document_chunks(&self, chunks: Vec<DocumentChunk>);
}

/// Markdown header line: `^(#{1,6})\s+(.+)$`.
static HEADER_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(#{1,6})\s+(.+)$").expect("valid regex"));

/// A run of whitespace (the separator candidates of the sentence split).
static WHITESPACE_RUN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").expect("valid regex"));

/// Python `str.isspace()` for one character.
fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// Python `str.strip()`.
fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}

/// Python `str(value)` for a JSON value.
fn py_str(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => "None".to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(true)) => "True".to_string(),
        Some(Value::Bool(false)) => "False".to_string(),
        Some(other) => other.to_string(),
    }
}

/// Python truthiness of a JSON value.
fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// `tempfile.tempdir`: the process temp directory.
fn temp_dir() -> PathBuf {
    std::env::temp_dir()
}

/// `re.split(r'(?<=[.!?])\s+', text)`: split at whitespace runs that follow
/// sentence-ending punctuation.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut cursor = 0;
    for m in WHITESPACE_RUN.find_iter(text) {
        let preceded = text[..m.start()].chars().next_back().is_some_and(|c| matches!(c, '.' | '!' | '?'));
        if preceded {
            parts.push(&text[cursor..m.start()]);
            cursor = m.end();
        }
    }
    parts.push(&text[cursor..]);
    parts
}

/// Collects pending documents, parses them and stores them in vector storage.
pub struct DocumentProcessor {
    pub llm: Arc<dyn BaseLLM + Send + Sync>,
    pub projects: Arc<dyn ProjectRepository>,
    pub documents: Arc<dyn DocumentRepository>,
    pub gridfs: Arc<dyn FileStore>,
    pub vector_db: Arc<dyn ChunkVectorStore>,
    pub section_token_size: usize,
}

impl DocumentProcessor {
    /// Build a processor (`section_token_size` defaults to 512 in Python).
    pub fn new(
        llm: Arc<dyn BaseLLM + Send + Sync>,
        projects: Arc<dyn ProjectRepository>,
        documents: Arc<dyn DocumentRepository>,
        gridfs: Arc<dyn FileStore>,
        vector_db: Arc<dyn ChunkVectorStore>,
        section_token_size: usize,
    ) -> Self {
        DocumentProcessor { llm, projects, documents, gridfs, vector_db, section_token_size }
    }

    /// Find all currently pending documents and process each one.
    pub fn process_pending_documents(&self, project_id: &str) {
        tracing::info!(target: LOG, "processing pending docs...");

        let pending_documents = self.documents.get_all_pending(project_id);

        tracing::info!(target: LOG, "Found pending documents: {:?}", pending_documents);

        for mut document in pending_documents {
            futures::executor::block_on(self.process_document(&mut document));
        }
    }

    /// Process an individual document (errors are logged, never raised; the
    /// document is always marked not pending and updated).
    pub async fn process_document(&self, document: &mut Record) {
        let project_id = document.get("project_id").cloned();
        // A DocumentMissingError and any other failure are both logged (with
        // their chain) and swallowed.
        if let Err(error) = self.process_document_inner(document, project_id.as_ref()) {
            tracing::error!(target: LOG, "{error:?}");
        }
        document.insert("is_pending".to_string(), Value::Bool(false));
        self.documents.update(document);
    }

    fn process_document_inner(&self, document: &mut Record, project_id: Option<&Value>) -> anyhow::Result<()> {
        let deleted = truthy(document.get("is_deleted"));
        // process if not deleted
        if !deleted {
            let project = self.projects.get(&py_str(project_id));

            if let Some(project) = project {
                let file_id = document.get("file_id").cloned().unwrap_or(Value::Null);
                let dat = self.gridfs.get(&file_id)?;

                // Extract project ID from document path (first part before the first slash)
                let path = document
                    .get("path")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow::anyhow!("document has no path"))?
                    .to_string();
                let project_id_from_path = path.split('/').next().unwrap_or("");
                let document_dir = temp_dir().join(project_id_from_path);

                if !document_dir.is_dir() {
                    std::fs::create_dir_all(&document_dir)?;
                }

                let filepath = temp_dir().join(&path);
                std::fs::write(&filepath, &dat)?;

                self.process_document_as_chunks(&project, document)?;

                document.insert("is_pending".to_string(), Value::Bool(false));
                self.documents.update(document);
            }
        } else {
            // If marked deleted then the project was deleted so delete all
            // chunks for this document.
            let document_id = py_str(document.get("_id"));
            // init the collection
            self.vector_db.open_project(&py_str(project_id));
            self.vector_db.delete_all_chunks_by_document_id(&document_id);
        }
        Ok(())
    }

    /// Sum of `num_of_tokens` over the project's other documents.
    pub fn count_processed_tokens_for_project(&self, project_id: &str, document_id: &Value) -> i64 {
        self.documents
            .find_by_project(project_id)
            .iter()
            .filter(|doc| doc.get("_id").unwrap_or(&Value::Null) != document_id)
            .map(|doc| match doc.get(NUM_OF_TOKENS) {
                Some(Value::Number(n)) => n.as_i64().unwrap_or_else(|| n.as_f64().unwrap_or(0.0) as i64),
                _ => 0,
            })
            .sum()
    }

    /// Split document text into sections ensuring each section doesn't exceed
    /// `target_tokens_per_section`.  First tries to split by markdown headers
    /// if they exist, then by paragraphs, then by sentences, then by words.
    pub fn get_sections(&self, text: &str, target_tokens_per_section: usize) -> Vec<String> {
        let mut sections = Vec::new();

        // First, try to split by markdown headers if they exist
        let lines: Vec<&str> = text.split('\n').collect();
        let has_headers = lines.iter().any(|line| HEADER_PATTERN.is_match(line));

        if has_headers {
            // Process markdown sections
            let mut markdown_sections: Vec<String> = Vec::new();
            let mut current_section: Vec<&str> = Vec::new();
            let mut current_header_level = 0usize;

            for line in &lines {
                if let Some(header_match) = HEADER_PATTERN.captures(line) {
                    let header_level = header_match[1].len();

                    if !current_section.is_empty()
                        && (header_level <= current_header_level || current_header_level == 0)
                    {
                        let section_content = py_strip(&current_section.join("\n")).to_string();
                        if !section_content.is_empty() {
                            markdown_sections.push(section_content);
                        }
                        current_section = Vec::new();
                    }

                    current_section.push(line);
                    current_header_level = header_level;
                } else {
                    current_section.push(line);
                }
            }

            if !current_section.is_empty() {
                let section_content = py_strip(&current_section.join("\n")).to_string();
                if !section_content.is_empty() {
                    markdown_sections.push(section_content);
                }
            }

            // Check if any markdown section exceeds token limit
            for section in &markdown_sections {
                sections.extend(self.split_large_section(section, target_tokens_per_section));
            }
        } else {
            // No markdown headers, split by other methods
            sections.extend(self.split_large_section(text, target_tokens_per_section));
        }

        tracing::debug!(
            target: LOG,
            "Split document into {} sections, each under {} tokens",
            sections.len(),
            target_tokens_per_section
        );
        sections
    }

    /// Split a large text section into smaller chunks that don't exceed
    /// `max_tokens`, using paragraph boundaries, then sentence boundaries,
    /// then word boundaries.
    fn split_large_section(&self, text: &str, max_tokens: usize) -> Vec<String> {
        let mut sections = Vec::new();

        // If the entire section is within limit, return as-is
        if self.llm.get_num_tokens(text) <= max_tokens {
            let stripped = py_strip(text);
            return if stripped.is_empty() { Vec::new() } else { vec![stripped.to_string()] };
        }

        // Try splitting by paragraphs first
        let paragraphs: Vec<&str> = text.split("\n\n").map(py_strip).filter(|p| !p.is_empty()).collect();

        if paragraphs.len() > 1 {
            let mut current_chunk = String::new();
            let mut current_tokens = 0usize;

            for paragraph in paragraphs {
                let paragraph_tokens = self.llm.get_num_tokens(paragraph);

                // If single paragraph exceeds limit, split it further
                if paragraph_tokens > max_tokens {
                    // Save current chunk if it has content
                    if !current_chunk.is_empty() {
                        sections.push(py_strip(&current_chunk).to_string());
                        current_chunk = String::new();
                        current_tokens = 0;
                    }

                    // Split the large paragraph
                    sections.extend(self.split_by_sentences(paragraph, max_tokens));
                    continue;
                }

                // If adding this paragraph would exceed limit, save current chunk
                if current_tokens + paragraph_tokens > max_tokens && !current_chunk.is_empty() {
                    sections.push(py_strip(&current_chunk).to_string());
                    current_chunk = paragraph.to_string();
                    current_tokens = paragraph_tokens;
                } else {
                    // Add paragraph to current chunk
                    if !current_chunk.is_empty() {
                        current_chunk.push_str("\n\n");
                        current_chunk.push_str(paragraph);
                    } else {
                        current_chunk = paragraph.to_string();
                    }
                    current_tokens += paragraph_tokens;
                }
            }

            // Add the last chunk
            if !current_chunk.is_empty() {
                sections.push(py_strip(&current_chunk).to_string());
            }
        } else {
            // Single paragraph or no paragraph breaks, split by sentences
            sections.extend(self.split_by_sentences(text, max_tokens));
        }

        sections
    }

    /// Split text by sentences, keeping within token limits.
    fn split_by_sentences(&self, text: &str, max_tokens: usize) -> Vec<String> {
        // Simple sentence splitting.
        let sentences = split_sentences(text);
        let mut sections = Vec::new();
        let mut current_chunk = String::new();
        let mut current_tokens = 0usize;

        for sentence in sentences {
            let sentence = py_strip(sentence);
            if sentence.is_empty() {
                continue;
            }

            let sentence_tokens = self.llm.get_num_tokens(sentence);

            // If single sentence exceeds limit, split by characters
            if sentence_tokens > max_tokens {
                // Save current chunk if it has content
                if !current_chunk.is_empty() {
                    sections.push(py_strip(&current_chunk).to_string());
                    current_chunk = String::new();
                    current_tokens = 0;
                }

                // Split by characters as last resort
                sections.extend(self.split_by_characters(sentence, max_tokens));
                continue;
            }

            // If adding this sentence would exceed limit, save current chunk
            if current_tokens + sentence_tokens > max_tokens && !current_chunk.is_empty() {
                sections.push(py_strip(&current_chunk).to_string());
                current_chunk = sentence.to_string();
                current_tokens = sentence_tokens;
            } else {
                // Add sentence to current chunk
                if !current_chunk.is_empty() {
                    current_chunk.push(' ');
                    current_chunk.push_str(sentence);
                } else {
                    current_chunk = sentence.to_string();
                }
                current_tokens += sentence_tokens;
            }
        }

        // Add the last chunk
        if !current_chunk.is_empty() {
            sections.push(py_strip(&current_chunk).to_string());
        }

        sections
    }

    /// Split text by words as a last resort, keeping within token limits.
    fn split_by_characters(&self, text: &str, max_tokens: usize) -> Vec<String> {
        let mut sections = Vec::new();
        let words = text.split(is_py_space).filter(|w| !w.is_empty());
        let mut current_chunk = String::new();
        let mut current_tokens = 0usize;

        for word in words {
            let word_tokens = self.llm.get_num_tokens(word);

            if current_tokens + word_tokens > max_tokens && !current_chunk.is_empty() {
                sections.push(py_strip(&current_chunk).to_string());
                current_chunk = word.to_string();
                current_tokens = word_tokens;
            } else {
                if !current_chunk.is_empty() {
                    current_chunk.push(' ');
                    current_chunk.push_str(word);
                } else {
                    current_chunk = word.to_string();
                }
                current_tokens += word_tokens;
            }
        }

        if !current_chunk.is_empty() {
            sections.push(py_strip(&current_chunk).to_string());
        }

        sections
    }

    /// Process a large document by breaking it into sections and turning each
    /// section into a chunk.  Maximizes token usage per section by combining
    /// smaller sections together.
    pub fn process_large_document_in_sections(
        &self,
        document_text: &str,
        document_id: &str,
    ) -> Vec<DocumentChunk> {
        let mut all_chunks = Vec::new();
        let max_tokens = self.llm.get_max_tokens();
        // Use configured section token size as the target tokens per section
        let target_tokens_per_section =
            if max_tokens > self.section_token_size { self.section_token_size } else { max_tokens };

        tracing::info!(target: LOG, "Processing large document in sections for document_id: {document_id}");
        tracing::info!(target: LOG, "Target tokens per section: {target_tokens_per_section}");

        // Split document into sections with configured token limit
        let sections = self.get_sections(document_text, target_tokens_per_section);
        tracing::info!(target: LOG, "Document split into {} sections", sections.len());

        for (section_idx, section) in sections.iter().enumerate() {
            if py_strip(section).is_empty() {
                continue;
            }
            let section_tokens = self.llm.get_num_tokens(section);
            tracing::info!(
                target: LOG,
                "Processing section {}/{} with {} tokens",
                section_idx + 1,
                sections.len(),
                section_tokens
            );
            let chunk = DocumentChunk {
                document_id: document_id.to_string(),
                chunk_id: uuid::Uuid::new_v4().to_string(),
                paragraph_id: format!("{document_id}_{section_idx}"),
                text: section.clone(),
                metadata: Some(format!("{{\"token_count\": {section_tokens}}}")),
                total_chunks: sections.len() as i64,
                ..Default::default()
            };
            all_chunks.push(chunk);
            tracing::info!(
                target: LOG,
                "Section {} converted to chunk with {} tokens",
                section_idx + 1,
                section_tokens
            );
        }

        tracing::info!(target: LOG, "Total chunks generated from all sections: {}", all_chunks.len());
        all_chunks
    }

    /// Create a doc specific parser and parse the document into chunks, then
    /// insert the chunks into the vector DB (which creates the embeddings in
    /// the process); records summary/token fields on `document`.
    ///
    /// A parser `ValueError` surfaces as [`DocumentMissingError`].  The
    /// document is always marked not pending/processing and updated.
    pub fn process_document_as_chunks(
        &self,
        project: &Record,
        document: &mut Record,
    ) -> anyhow::Result<()> {
        let result = self.chunk_document(project, document);
        document.insert("pending".to_string(), Value::Bool(false));
        document.insert("processing".to_string(), Value::Bool(false));
        self.documents.update(document);
        result.map_err(|error| {
            let is_value_error = matches!(error.downcast_ref::<ParserError>(), Some(ParserError::Value(_)));
            if is_value_error {
                DocumentMissingError(error.to_string()).into()
            } else {
                error
            }
        })
    }

    fn chunk_document(&self, project: &Record, document: &mut Record) -> anyhow::Result<()> {
        let path = py_str(document.get("path"));
        let document_path = temp_dir().join(&path);
        let document_path_str = document_path.to_string_lossy().into_owned();
        let mime_type = mime_guess::from_path(&document_path).first_raw().unwrap_or("");
        let document_id = py_str(document.get("_id"));
        let project_id = py_str(project.get("_id"));

        // Parse the document into chunks
        let parser = ParserFactory::new(Some(self.llm.clone())).get_parser(mime_type, &document_path_str)?;
        let document_text = parser.parse()?;

        let num_of_tokens = self.llm.get_num_tokens(&document_text);

        let document_chunks = self.process_large_document_in_sections(&document_text, &document_id);

        self.vector_db.open_project(&project_id);
        self.vector_db.upsert_document_chunks(document_chunks);

        // Build a quick-view summary from the first paragraph of the document
        let first_para = document_text
            .split("\n\n")
            .map(py_strip)
            .find(|p| !p.is_empty())
            .unwrap_or_else(|| py_strip(&document_text));
        let summary = if first_para.is_empty() {
            Value::Null
        } else {
            Value::String(first_para.chars().take(300).collect())
        };
        document.insert("summary".to_string(), summary);
        document.insert(IS_TOKEN_LIMIT_REACHED.to_string(), Value::Bool(false));
        document.insert(NUM_OF_TOKENS.to_string(), Value::from(num_of_tokens as u64));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::split_sentences;

    #[test]
    fn sentence_split_follows_punctuation_only() {
        assert_eq!(split_sentences("One. Two!  Three? four five"), vec!["One.", "Two!", "Three?", "four five"]);
        assert_eq!(split_sentences("no break here"), vec!["no break here"]);
        assert_eq!(split_sentences("End."), vec!["End."]);
        assert_eq!(split_sentences("End. "), vec!["End.", ""]);
    }
}
