//! Hand-port of the parts of Microsoft's `markitdown` package (0.1.6) that
//! smartloop's [`DocumentParser`](crate::parsers::document_parser::DocumentParser)
//! relies on.
//!
//! `MarkItDown.convert(path)` builds a list of stream-info guesses (from the
//! file extension and from the file content), then tries every registered
//! converter that accepts a guess, in priority order, until one succeeds. The
//! converters ported here are the ones reachable through `ParserFactory`'s
//! mime types (PDF, DOCX, PPTX, XLSX, HTML, PNG/JPEG) plus the generic
//! plain-text / CSV / ZIP / XLS fallbacks:
//!
//! * [`html`] — `HtmlConverter` + `_CustomMarkdownify` (markdownify 1.2).
//! * [`docx`] — `DocxConverter`: OMML math pre-processing ([`omml`]) and a
//!   port of `mammoth`'s DOCX → HTML conversion ([`mammoth`]), then
//!   `HtmlConverter`.
//! * [`pptx`] — `PptxConverter` over a port of the `python-pptx` reader.
//! * [`xlsx`] — `XlsxConverter` / `XlsConverter`: `pandas.read_excel` +
//!   `DataFrame.to_html` emulated over `calamine`, then `HtmlConverter`.
//! * [`pdf`] — `PdfConverter` (MuPDF stands in for pdfminer/pdfplumber).
//! * image, plain text, CSV and ZIP converters live in this module.
//!
//! The URL-, feed-, audio-, notebook-, Outlook- and EPUB-specific converters
//! of MarkItDown never accept the local office/HTML/image files smartloop
//! routes here and are not ported.
//!
//! MarkItDown's content sniffing uses the `magika` ML model; it is replaced
//! by a deterministic signature sniff ([`sniff_content`]) that yields the same
//! labels for the formats above.

pub mod dingbats;
pub mod docx;
pub mod html;
pub mod mammoth;
pub mod omml;
pub mod ooxml;
pub mod pdf;
pub mod pptx;
pub mod xlsx;

use std::path::Path;
use std::sync::Arc;

use base64::Engine as _;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Value};

use crate::parsers::csv_parser::py_csv_reader;
use crate::parsers::document_parser::{VisionClient, VisionRequest};

/// The result of converting a document to Markdown.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DocumentConverterResult {
    /// Optional title of the document.
    pub title: Option<String>,
    /// The converted Markdown text (Python `markdown` / `text_content`).
    pub text_content: String,
}

impl DocumentConverterResult {
    /// A result holding only Markdown text.
    pub fn markdown(text: impl Into<String>) -> Self {
        Self { title: None, text_content: text.into() }
    }
}

/// What is known (or guessed) about a stream (Python `StreamInfo`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StreamInfo {
    pub mimetype: Option<String>,
    /// Extension including the leading dot, e.g. `.docx`.
    pub extension: Option<String>,
    pub charset: Option<String>,
    pub filename: Option<String>,
    pub local_path: Option<String>,
    pub url: Option<String>,
}

impl StreamInfo {
    fn mimetype_lower(&self) -> String {
        self.mimetype.as_deref().unwrap_or("").to_lowercase()
    }

    fn extension_lower(&self) -> String {
        self.extension.as_deref().unwrap_or("").to_lowercase()
    }

    /// True when the extension is one of `extensions` or the mimetype starts
    /// with one of `mime_prefixes` (the `accepts()` shape of most converters).
    fn matches(&self, extensions: &[&str], mime_prefixes: &[&str]) -> bool {
        let extension = self.extension_lower();
        if extensions.contains(&extension.as_str()) {
            return true;
        }
        let mimetype = self.mimetype_lower();
        mime_prefixes.iter().any(|p| mimetype.starts_with(p))
    }
}

/// Options every converter receives (Python's forwarded `**kwargs`).
#[derive(Clone, Default)]
pub struct ConvertOptions {
    pub llm_client: Option<Arc<dyn VisionClient>>,
    pub llm_model: Option<String>,
    pub llm_prompt: Option<String>,
    pub exiftool_path: Option<String>,
}

/// A MarkItDown document converter.
pub trait DocumentConverter: Send + Sync {
    /// Python class name, used in failure messages.
    fn name(&self) -> &'static str;
    /// Whether this converter handles a stream described by `info`.
    fn accepts(&self, info: &StreamInfo) -> bool;
    /// Convert the stream's bytes to Markdown.
    fn convert(&self, data: &[u8], info: &StreamInfo, options: &ConvertOptions) -> anyhow::Result<DocumentConverterResult>;
}

/// Converter priority: specific formats are tried before generic ones.
const PRIORITY_SPECIFIC_FILE_FORMAT: f64 = 0.0;
const PRIORITY_GENERIC_FILE_FORMAT: f64 = 10.0;

/// A failed conversion attempt (converter name, error message, error type).
struct FailedConversionAttempt {
    converter: &'static str,
    error: anyhow::Error,
}

/// Raised when a suitable converter was found but every attempt failed.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct FileConversionException(pub String);

/// Raised when no converter accepted the stream.
#[derive(Debug, thiserror::Error)]
#[error("Could not convert stream to Markdown. No converter attempted a conversion, suggesting that the filetype is simply not supported.")]
pub struct UnsupportedFormatException;

/// Hand-port of `markitdown.MarkItDown` (built-in converters, no plugins).
#[derive(Default, Clone)]
pub struct MarkItDown {
    pub llm_client: Option<Arc<dyn VisionClient>>,
    pub llm_model: Option<String>,
    pub llm_prompt: Option<String>,
    /// `exiftool_path`; resolved from `EXIFTOOL_PATH` / well-known locations
    /// when not given.
    pub exiftool_path: Option<String>,
}

impl MarkItDown {
    /// `MarkItDown()` with no LLM client.
    pub fn new() -> Self {
        Self::default()
    }

    /// `MarkItDown(llm_client=, llm_model=, llm_prompt=)`.
    pub fn with_llm(client: Arc<dyn VisionClient>, model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self { llm_client: Some(client), llm_model: Some(model.into()), llm_prompt: Some(prompt.into()), exiftool_path: None }
    }

    /// The registered converters, sorted by priority (stable; within one
    /// priority the most recently registered converter comes first).
    fn converters() -> Vec<Box<dyn DocumentConverter>> {
        // Registration order of `enable_builtins`; each registration is
        // inserted at the front of the list.
        let registrations: Vec<(Box<dyn DocumentConverter>, f64)> = vec![
            (Box::new(PlainTextConverter), PRIORITY_GENERIC_FILE_FORMAT),
            (Box::new(ZipConverter), PRIORITY_GENERIC_FILE_FORMAT),
            (Box::new(html::HtmlConverter), PRIORITY_GENERIC_FILE_FORMAT),
            (Box::new(docx::DocxConverter), PRIORITY_SPECIFIC_FILE_FORMAT),
            (Box::new(xlsx::XlsxConverter), PRIORITY_SPECIFIC_FILE_FORMAT),
            (Box::new(xlsx::XlsConverter), PRIORITY_SPECIFIC_FILE_FORMAT),
            (Box::new(pptx::PptxConverter), PRIORITY_SPECIFIC_FILE_FORMAT),
            (Box::new(ImageConverter), PRIORITY_SPECIFIC_FILE_FORMAT),
            (Box::new(pdf::PdfConverter), PRIORITY_SPECIFIC_FILE_FORMAT),
            (Box::new(CsvConverter), PRIORITY_SPECIFIC_FILE_FORMAT),
        ];
        let mut list: Vec<(Box<dyn DocumentConverter>, f64)> = Vec::new();
        for registration in registrations {
            list.insert(0, registration);
        }
        list.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        list.into_iter().map(|(c, _)| c).collect()
    }

    /// The options forwarded to converters (`_convert`'s `_kwargs`).
    fn options(&self) -> ConvertOptions {
        ConvertOptions {
            llm_client: self.llm_client.clone(),
            llm_model: self.llm_model.clone(),
            llm_prompt: self.llm_prompt.clone(),
            exiftool_path: self.exiftool_path.clone().or_else(resolve_exiftool_path),
        }
    }

    /// Convert a local file to Markdown (`MarkItDown.convert_local`).
    pub fn convert(&self, path: &str) -> anyhow::Result<DocumentConverterResult> {
        let data = std::fs::read(path)?;
        // os.path.splitext: the extension keeps its dot; dotfiles have none.
        let extension = Path::new(path).extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
        let base_guess = StreamInfo {
            local_path: Some(path.to_string()),
            extension: Some(extension),
            filename: Path::new(path).file_name().map(|f| f.to_string_lossy().to_string()),
            ..StreamInfo::default()
        };
        let guesses = get_stream_info_guesses(&data, &base_guess);
        self.convert_bytes(&data, guesses)
    }

    /// A MarkItDown carrying the options a converter was called with
    /// (nested conversions reuse the parent's settings).
    fn from_options(options: &ConvertOptions) -> Self {
        Self {
            llm_client: options.llm_client.clone(),
            llm_model: options.llm_model.clone(),
            llm_prompt: options.llm_prompt.clone(),
            exiftool_path: options.exiftool_path.clone(),
        }
    }

    /// Convert in-memory bytes described by `stream_info`
    /// (`MarkItDown.convert_stream`).
    pub fn convert_stream(&self, data: &[u8], stream_info: &StreamInfo) -> anyhow::Result<DocumentConverterResult> {
        let guesses = get_stream_info_guesses(data, stream_info);
        self.convert_bytes(data, guesses)
    }

    /// `MarkItDown._convert`: try every accepting converter for each guess.
    fn convert_bytes(&self, data: &[u8], guesses: Vec<StreamInfo>) -> anyhow::Result<DocumentConverterResult> {
        let converters = Self::converters();
        let options = self.options();
        let mut failed_attempts: Vec<FailedConversionAttempt> = Vec::new();

        for stream_info in guesses.into_iter().chain(std::iter::once(StreamInfo::default())) {
            for converter in &converters {
                if !converter.accepts(&stream_info) {
                    continue;
                }
                match converter.convert(data, &stream_info, &options) {
                    Ok(mut res) => {
                        // Normalize the content
                        res.text_content = normalize_text_content(&res.text_content);
                        return Ok(res);
                    }
                    Err(error) => failed_attempts.push(FailedConversionAttempt { converter: converter.name(), error }),
                }
            }
        }

        // If we got this far without success, report any exceptions
        if !failed_attempts.is_empty() {
            let mut message = format!("File conversion failed after {} attempts:\n", failed_attempts.len());
            for attempt in &failed_attempts {
                message.push_str(&format!(" - {} threw an error with message: {}\n", attempt.converter, attempt.error));
            }
            return Err(FileConversionException(message).into());
        }

        // Nothing can handle it!
        Err(UnsupportedFormatException.into())
    }
}

static RE_LINE_SPLIT: Lazy<Regex> = Lazy::new(|| Regex::new(r"\r?\n").expect("valid regex"));
static RE_THREE_NEWLINES: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n{3,}").expect("valid regex"));

/// Strip trailing whitespace from every line and collapse 3+ newlines.
pub fn normalize_text_content(text: &str) -> String {
    let joined = RE_LINE_SPLIT.split(text).map(|line| line.trim_end()).collect::<Vec<_>>().join("\n");
    RE_THREE_NEWLINES.replace_all(&joined, "\n\n").into_owned()
}

/// `mimetypes.guess_type("placeholder" + extension, strict=False)[0]`.
pub fn guess_mimetype_for_extension(extension: &str) -> Option<String> {
    let ext = extension.trim_start_matches('.').to_lowercase();
    if ext.is_empty() {
        return None;
    }
    // Python's table differs from mime_guess for a few common types.
    let python = match ext.as_str() {
        "txt" | "text" => Some("text/plain"),
        "md" | "markdown" => Some("text/markdown"),
        "json" => Some("application/json"),
        "csv" => Some("text/csv"),
        "htm" | "html" => Some("text/html"),
        "xhtml" => Some("application/xhtml+xml"),
        "jpg" | "jpeg" | "jpe" => Some("image/jpeg"),
        "png" => Some("image/png"),
        "pdf" => Some("application/pdf"),
        "xls" => Some("application/vnd.ms-excel"),
        _ => None,
    };
    python.map(str::to_string).or_else(|| mime_guess::from_ext(&ext).first_raw().map(str::to_string))
}

/// A content-based guess (what `magika` would report).
#[derive(Debug, Clone, PartialEq)]
pub struct ContentGuess {
    pub mime_type: String,
    /// Known extensions for the label, without dots; the first is preferred.
    pub extensions: Vec<&'static str>,
    pub is_text: bool,
}

/// Identify a stream from its leading bytes (stand-in for `magika`).
pub fn sniff_content(data: &[u8]) -> Option<ContentGuess> {
    let guess = |mime: &str, extensions: Vec<&'static str>, is_text: bool| {
        Some(ContentGuess { mime_type: mime.to_string(), extensions, is_text })
    };
    if data.is_empty() {
        return None;
    }
    if data.starts_with(b"%PDF") {
        return guess("application/pdf", vec!["pdf"], false);
    }
    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        return guess("image/png", vec!["png"], false);
    }
    if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return guess("image/jpeg", vec!["jpg", "jpeg"], false);
    }
    if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        return guess("image/gif", vec!["gif"], false);
    }
    if data.starts_with(b"PK\x03\x04") {
        if let Ok(archive) = zip::ZipArchive::new(std::io::Cursor::new(data)) {
            let names: Vec<String> = archive.file_names().map(|n| n.to_string()).collect();
            let has = |n: &str| names.iter().any(|x| x == n);
            if has("word/document.xml") {
                return guess(
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    vec!["docx", "docm"],
                    false,
                );
            }
            if has("ppt/presentation.xml") {
                return guess(
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    vec!["pptx", "pptm"],
                    false,
                );
            }
            if has("xl/workbook.xml") {
                return guess(
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    vec!["xlsx", "xlsm"],
                    false,
                );
            }
            if has("META-INF/container.xml") && has("mimetype") {
                return guess("application/epub+zip", vec!["epub"], false);
            }
        }
        return guess("application/zip", vec!["zip"], false);
    }
    if data.starts_with(&[0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1]) {
        // An OLE compound file; without parsing its directory we cannot tell
        // .doc from .xls from .msg, so report it as unknown.
        return None;
    }

    // Text formats.
    let text = std::str::from_utf8(data).ok()?;
    if text.contains('\0') {
        return None;
    }
    let head: String = text.trim_start_matches('\u{feff}').trim_start().chars().take(1024).collect::<String>().to_lowercase();
    if head.starts_with("<!doctype html")
        || head.starts_with("<html")
        || head.contains("<html")
        || head.contains("<body")
        || head.contains("<head")
        || head.contains("<table")
        || head.contains("<div")
        || head.contains("<p>")
    {
        return guess("text/html", vec!["html", "htm", "xhtml", "xht"], true);
    }
    let trimmed = text.trim_start_matches('\u{feff}').trim();
    if (trimmed.starts_with('{') || trimmed.starts_with('[')) && serde_json::from_str::<Value>(trimmed).is_ok() {
        return guess("application/json", vec!["json"], true);
    }
    guess("text/plain", vec!["txt"], true)
}

/// `MarkItDown._get_stream_info_guesses`.
pub fn get_stream_info_guesses(data: &[u8], base_guess: &StreamInfo) -> Vec<StreamInfo> {
    let mut guesses = Vec::new();

    // Enhance the base guess with information based on the extension or mimetype
    let mut enhanced_guess = base_guess.clone();
    if base_guess.mimetype.is_none()
        && let Some(ext) = base_guess.extension.as_deref()
        && let Some(m) = guess_mimetype_for_extension(ext)
    {
        enhanced_guess.mimetype = Some(m);
    }

    match sniff_content(data) {
        Some(result) => {
            // If it's text, also guess the charset
            let charset = if result.is_text { Some("utf-8".to_string()) } else { None };
            let guessed_extension = result.extensions.first().map(|e| format!(".{e}"));

            // Determine if the guess is compatible with the base guess
            let mut compatible = true;
            if base_guess.mimetype.as_ref().is_some_and(|m| *m != result.mime_type) {
                compatible = false;
            }
            if let Some(ext) = base_guess.extension.as_deref()
                && !result.extensions.contains(&ext.trim_start_matches('.'))
            {
                compatible = false;
            }
            if base_guess.charset.is_some() && normalize_charset(base_guess.charset.as_deref()) != charset {
                compatible = false;
            }

            if compatible {
                // Add the compatible base guess
                guesses.push(StreamInfo {
                    mimetype: base_guess.mimetype.clone().or(Some(result.mime_type.clone())),
                    extension: base_guess.extension.clone().filter(|e| !e.is_empty()).or(guessed_extension),
                    charset: base_guess.charset.clone().or(charset),
                    filename: base_guess.filename.clone(),
                    local_path: base_guess.local_path.clone(),
                    url: base_guess.url.clone(),
                });
            } else {
                // The guess was incompatible with the base guess, so add both guesses
                guesses.push(enhanced_guess);
                guesses.push(StreamInfo {
                    mimetype: Some(result.mime_type.clone()),
                    extension: guessed_extension,
                    charset,
                    filename: base_guess.filename.clone(),
                    local_path: base_guess.local_path.clone(),
                    url: base_guess.url.clone(),
                });
            }
        }
        None => guesses.push(enhanced_guess),
    }

    guesses
}

/// `codecs.lookup(charset).name` for the charsets this module produces.
fn normalize_charset(charset: Option<&str>) -> Option<String> {
    let c = charset?;
    let lower = c.to_lowercase().replace('_', "-");
    Some(match lower.as_str() {
        "utf8" | "utf-8" | "u8" => "utf-8".to_string(),
        "latin-1" | "latin1" | "iso-8859-1" | "iso8859-1" => "iso8859-1".to_string(),
        "cp1252" | "windows-1252" => "cp1252".to_string(),
        "ascii" | "us-ascii" => "ascii".to_string(),
        _ => c.to_string(),
    })
}

/// Decode text the way the plain-text / CSV converters do: with the guessed
/// charset when known, otherwise with the best detected encoding (UTF-8,
/// falling back to Windows-1252).
pub fn decode_text(data: &[u8], charset: Option<&str>) -> anyhow::Result<String> {
    match normalize_charset(charset).as_deref() {
        Some("utf-8") => Ok(std::str::from_utf8(data)
            .map_err(|e| anyhow::anyhow!("'utf-8' codec can't decode bytes: {e}"))?
            .to_string()),
        Some("ascii") => {
            if let Some(pos) = data.iter().position(|b| *b >= 0x80) {
                anyhow::bail!("'ascii' codec can't decode byte 0x{:02x} in position {pos}", data[pos]);
            }
            Ok(String::from_utf8_lossy(data).into_owned())
        }
        Some("iso8859-1") => Ok(data.iter().map(|b| *b as char).collect()),
        Some("cp1252") => Ok(decode_cp1252(data)),
        Some(other) => anyhow::bail!("unknown encoding: {other}"),
        None => {
            // charset_normalizer: a BOM'd or valid UTF-8 stream decodes as
            // UTF-8 (BOM removed); anything else as the closest single-byte
            // encoding.
            let body = data.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(data);
            match std::str::from_utf8(body) {
                Ok(t) => Ok(t.to_string()),
                Err(_) => Ok(decode_cp1252(body)),
            }
        }
    }
}

/// Windows-1252 decoding (undefined bytes map to the C1 control points).
fn decode_cp1252(data: &[u8]) -> String {
    const HIGH: [char; 32] = [
        '\u{20AC}', '\u{81}', '\u{201A}', '\u{0192}', '\u{201E}', '\u{2026}', '\u{2020}', '\u{2021}', '\u{02C6}',
        '\u{2030}', '\u{0160}', '\u{2039}', '\u{0152}', '\u{8D}', '\u{017D}', '\u{8F}', '\u{90}', '\u{2018}',
        '\u{2019}', '\u{201C}', '\u{201D}', '\u{2022}', '\u{2013}', '\u{2014}', '\u{02DC}', '\u{2122}', '\u{0161}',
        '\u{203A}', '\u{0153}', '\u{9D}', '\u{017E}', '\u{0178}',
    ];
    data.iter()
        .map(|&b| if (0x80..0xA0).contains(&b) { HIGH[(b - 0x80) as usize] } else { b as char })
        .collect()
}

// -- ZipConverter ---------------------------------------------------------------------

/// `os.path.splitext(path)[1]` (POSIX): the extension of the last path
/// component, ignoring leading dots.
fn py_splitext_ext(path: &str) -> String {
    let base_start = path.rfind('/').map(|i| i + 1).unwrap_or(0);
    let base = &path[base_start..];
    let first_non_dot = base.find(|c| c != '.').unwrap_or(base.len());
    match base.rfind('.') {
        Some(dot) if dot > first_non_dot => base[dot..].to_string(),
        _ => String::new(),
    }
}

/// Converts ZIP files to Markdown by converting each contained file with the
/// parent MarkItDown's converters.
pub struct ZipConverter;

impl DocumentConverter for ZipConverter {
    fn name(&self) -> &'static str {
        "ZipConverter"
    }

    fn accepts(&self, info: &StreamInfo) -> bool {
        info.matches(&[".zip"], &["application/zip"])
    }

    fn convert(&self, data: &[u8], info: &StreamInfo, options: &ConvertOptions) -> anyhow::Result<DocumentConverterResult> {
        use std::io::Read;

        let file_path = info
            .url
            .clone()
            .or_else(|| info.local_path.clone())
            .or_else(|| info.filename.clone())
            .unwrap_or_else(|| "None".to_string());
        let mut md_content = format!("Content from the zip file `{file_path}`:\n\n");
        let markitdown = MarkItDown::from_options(options);

        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(data))?;
        for index in 0..archive.len() {
            let (name, bytes) = {
                let mut entry = archive.by_index(index)?;
                let mut bytes = Vec::new();
                entry.read_to_end(&mut bytes)?;
                (entry.name().to_string(), bytes)
            };
            let z_file_stream_info = StreamInfo {
                extension: Some(py_splitext_ext(&name)),
                filename: Some(name.rsplit('/').next().unwrap_or(&name).to_string()),
                ..StreamInfo::default()
            };
            match markitdown.convert_stream(&bytes, &z_file_stream_info) {
                Ok(result) => {
                    md_content.push_str(&format!("## File: {name}\n\n"));
                    md_content.push_str(&result.text_content);
                    md_content.push_str("\n\n");
                }
                Err(e) if e.is::<UnsupportedFormatException>() || e.is::<FileConversionException>() => {}
                Err(e) => return Err(e),
            }
        }

        Ok(DocumentConverterResult::markdown(md_content.trim()))
    }
}

// -- PlainTextConverter -----------------------------------------------------------

/// Anything with content type text/plain.
pub struct PlainTextConverter;

impl DocumentConverter for PlainTextConverter {
    fn name(&self) -> &'static str {
        "PlainTextConverter"
    }

    fn accepts(&self, info: &StreamInfo) -> bool {
        // If we have a charset, we can safely assume it's text
        if info.charset.is_some() {
            return true;
        }
        info.matches(
            &[".txt", ".text", ".md", ".markdown", ".json", ".jsonl"],
            &["text/", "application/json", "application/markdown"],
        )
    }

    fn convert(&self, data: &[u8], info: &StreamInfo, _options: &ConvertOptions) -> anyhow::Result<DocumentConverterResult> {
        Ok(DocumentConverterResult::markdown(decode_text(data, info.charset.as_deref())?))
    }
}

// -- CsvConverter -------------------------------------------------------------------

/// Converts CSV files to Markdown tables.
pub struct CsvConverter;

impl DocumentConverter for CsvConverter {
    fn name(&self) -> &'static str {
        "CsvConverter"
    }

    fn accepts(&self, info: &StreamInfo) -> bool {
        info.matches(&[".csv"], &["text/csv", "application/csv"])
    }

    fn convert(&self, data: &[u8], info: &StreamInfo, _options: &ConvertOptions) -> anyhow::Result<DocumentConverterResult> {
        // Read the file content
        let content = decode_text(data, info.charset.as_deref())?;

        // Parse CSV content (io.StringIO does no newline translation).
        let rows = py_csv_reader(&content)?;
        if rows.is_empty() {
            return Ok(DocumentConverterResult::markdown(""));
        }

        let header = &rows[0];
        let mut markdown_table = vec![
            format!("| {} |", header.join(" | ")),
            format!("| {} |", vec!["---"; header.len()].join(" | ")),
        ];
        for row in &rows[1..] {
            // Make sure row has the same number of columns as header
            let mut row = row.clone();
            while row.len() < header.len() {
                row.push(String::new());
            }
            // Truncate if row has more columns than header
            row.truncate(header.len());
            markdown_table.push(format!("| {} |", row.join(" | ")));
        }

        Ok(DocumentConverterResult::markdown(markdown_table.join("\n")))
    }
}

// -- ImageConverter -----------------------------------------------------------------

/// Converts images to markdown via extraction of metadata (if `exiftool` is
/// installed), and description via a multimodal LLM (if an llm_client is
/// configured).
pub struct ImageConverter;

/// Metadata fields reported by the image converter, in output order.
const IMAGE_METADATA_FIELDS: &[&str] = &[
    "ImageSize",
    "Title",
    "Caption",
    "Description",
    "Keywords",
    "Artist",
    "Author",
    "DateTimeOriginal",
    "CreateDate",
    "GPSPosition",
];

impl DocumentConverter for ImageConverter {
    fn name(&self) -> &'static str {
        "ImageConverter"
    }

    fn accepts(&self, info: &StreamInfo) -> bool {
        info.matches(&[".jpg", ".jpeg", ".png"], &["image/jpeg", "image/png"])
    }

    fn convert(&self, data: &[u8], info: &StreamInfo, options: &ConvertOptions) -> anyhow::Result<DocumentConverterResult> {
        let mut md_content = String::new();

        // Add metadata
        let metadata = exiftool_metadata(data, options.exiftool_path.as_deref())?;
        if let Some(map) = metadata.as_object() {
            for field in IMAGE_METADATA_FIELDS {
                if let Some(value) = map.get(*field) {
                    md_content.push_str(&format!("{field}: {}\n", py_str_value(value)));
                }
            }
        }

        // Try describing the image with the LLM
        if let (Some(client), Some(model)) = (options.llm_client.as_ref(), options.llm_model.as_ref())
            && let Some(description) = llm_caption(data, info, client.as_ref(), model, options.llm_prompt.as_deref())?
        {
            md_content.push_str(&format!("\n# Description:\n{}\n", description.trim()));
        }

        Ok(DocumentConverterResult::markdown(md_content))
    }
}

/// Python `str()` of a JSON value from exiftool's output.
fn py_str_value(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                n.to_string()
            } else {
                n.as_f64().map(crate::parsers::json_parser::py_float_repr).unwrap_or_else(|| n.to_string())
            }
        }
        other => crate::parsers::json_parser::py_repr_json(other),
    }
}

/// `llm_caption` / `ImageConverter._get_llm_description`: send the image as a
/// data URI to the vision client. `Ok(None)` when the image can't be encoded.
pub fn llm_caption(
    data: &[u8],
    info: &StreamInfo,
    client: &dyn VisionClient,
    model: &str,
    prompt: Option<&str>,
) -> anyhow::Result<Option<String>> {
    let prompt = match prompt {
        Some(p) if !p.trim().is_empty() => p.to_string(),
        _ => "Write a detailed caption for this image.".to_string(),
    };

    // Get the content type
    let content_type = info
        .mimetype
        .clone()
        .filter(|m| !m.is_empty())
        .or_else(|| guess_mimetype_for_extension(info.extension.as_deref().unwrap_or("")))
        .unwrap_or_else(|| "application/octet-stream".to_string());

    // Convert to base64 and prepare the data-uri
    let base64_image = base64::engine::general_purpose::STANDARD.encode(data);
    let data_uri = format!("data:{content_type};base64,{base64_image}");

    let messages = vec![json!({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ],
    })];

    let response = client.create(VisionRequest::new(Some(model.to_string()), messages))?;
    Ok(Some(response))
}

/// Run `exiftool -json -` over the image bytes (after checking the version is
/// not vulnerable to CVE-2021-22204). Returns `{}` when no exiftool is set.
pub fn exiftool_metadata(data: &[u8], exiftool_path: Option<&str>) -> anyhow::Result<Value> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    // Nothing to do
    let Some(exiftool_path) = exiftool_path.filter(|p| !p.is_empty()) else {
        return Ok(json!({}));
    };

    // Verify exiftool version
    let verify_failed = || anyhow::anyhow!("Failed to verify ExifTool version.");
    let output = Command::new(exiftool_path).arg("-ver").output()?;
    if !output.status.success() {
        return Err(verify_failed());
    }
    let version_output = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let version: Vec<i64> =
        version_output.split('.').map(|p| p.parse::<i64>()).collect::<Result<_, _>>().map_err(|_| verify_failed())?;
    if version < vec![12, 24] {
        anyhow::bail!(
            "ExifTool version {version_output} is vulnerable to CVE-2021-22204. Please upgrade to version 12.24 or later."
        );
    }

    // Run exiftool
    let mut child = Command::new(exiftool_path)
        .args(["-json", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        // exiftool may stop reading early; a broken pipe is not an error.
        let _ = stdin.write_all(data);
    }
    let output = child.wait_with_output()?;
    let parsed: Value = serde_json::from_slice(&output.stdout)?;
    parsed
        .as_array()
        .and_then(|a| a.first().cloned())
        .ok_or_else(|| anyhow::anyhow!("list index out of range"))
}

/// `EXIFTOOL_PATH`, else an `exiftool` on PATH inside a well-known directory.
fn resolve_exiftool_path() -> Option<String> {
    if let Ok(path) = std::env::var("EXIFTOOL_PATH") {
        return Some(path);
    }
    let candidate = which("exiftool")?;
    let candidate = std::path::absolute(&candidate).unwrap_or(candidate);
    let dir = candidate.parent()?.to_string_lossy().to_string();
    let well_known = [
        "/usr/bin",
        "/usr/local/bin",
        "/opt",
        "/opt/bin",
        "/opt/local/bin",
        "/opt/homebrew/bin",
        "C:\\Windows\\System32",
        "C:\\Program Files",
        "C:\\Program Files (x86)",
    ];
    if well_known.contains(&dir.as_str()) { Some(candidate.to_string_lossy().to_string()) } else { None }
}

/// `shutil.which`: first executable named `name` on PATH.
fn which(name: &str) -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if is_executable(&candidate) {
            return Some(candidate);
        }
        if cfg!(windows) {
            for ext in ["exe", "bat", "cmd"] {
                let with_ext = candidate.with_extension(ext);
                if with_ext.is_file() {
                    return Some(with_ext);
                }
            }
        }
    }
    None
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    path.metadata().map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0).unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_lines() {
        assert_eq!(normalize_text_content("a  \r\nb\n\n\n\nc\t"), "a\nb\n\nc");
    }

    #[test]
    fn converter_order_matches_markitdown() {
        let names: Vec<&str> = MarkItDown::converters().iter().map(|c| c.name()).collect();
        assert_eq!(
            names,
            vec![
                "CsvConverter",
                "PdfConverter",
                "ImageConverter",
                "PptxConverter",
                "XlsConverter",
                "XlsxConverter",
                "DocxConverter",
                "HtmlConverter",
                "ZipConverter",
                "PlainTextConverter",
            ]
        );
    }

    #[test]
    fn guesses_for_text_and_png() {
        let base = StreamInfo { extension: Some(".txt".into()), ..Default::default() };
        let guesses = get_stream_info_guesses(b"hello", &base);
        assert_eq!(guesses.len(), 1);
        assert_eq!(guesses[0].mimetype.as_deref(), Some("text/plain"));
        assert_eq!(guesses[0].charset.as_deref(), Some("utf-8"));

        let base = StreamInfo { extension: Some(".html".into()), ..Default::default() };
        let guesses = get_stream_info_guesses(b"<html><body>x</body></html>", &base);
        assert_eq!(guesses[0].mimetype.as_deref(), Some("text/html"));

        // Incompatible extension: both guesses are tried.
        let base = StreamInfo { extension: Some(".png".into()), ..Default::default() };
        let guesses = get_stream_info_guesses(b"plain words", &base);
        assert_eq!(guesses.len(), 2);
        assert_eq!(guesses[0].mimetype.as_deref(), Some("image/png"));
        assert_eq!(guesses[1].mimetype.as_deref(), Some("text/plain"));
    }

    #[test]
    fn csv_converter_pads_rows() {
        let res = CsvConverter.convert(b"a,b\n1\n2,3,4\n", &StreamInfo::default(), &ConvertOptions::default()).unwrap();
        assert_eq!(res.text_content, "| a | b |\n| --- | --- |\n| 1 |  |\n| 2 | 3 |");
    }
}
