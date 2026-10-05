//! URL fetch agent — fetches a URL and returns its content as markdown.
//!
//! Port of `smartloop/agents/fetch_agent.py`.
//!
//! Replaces the fetch MCP server.  Given a URL it performs the HTTP GET itself
//! and converts the response to markdown based on content type:
//!
//! * HTML → strip non-content tags + html→markdown (`htmd`).
//! * PDF  → the shared [`PdfParser`](crate::parsers::pdf_parser::PdfParser)
//!   (same converter used for uploaded documents).
//! * anything else → returned as raw text.
//!
//! It runs as an ordinary plan step (`StepType::Fetch`) resolved by the
//! orchestrator, not as an MCP tool.

#[allow(unused_imports)]
use crate::parsers::pdf_parser::PdfParser;

/// Tags removed before HTML → markdown conversion.
pub const STRIP_TAGS: &[&str] = &["script", "style", "noscript", "svg", "iframe", "link", "meta"];

/// User-Agent sent with every fetch.
pub const USER_AGENT: &str = "Mozilla/5.0 (compatible; Smartloop/1.0)";

/// Phrases in a `<noscript>` block that mark a JavaScript-only page.
pub const JS_PHRASES: &[&str] = &[
    "enable javascript",
    "javascript is required",
    "javascript is needed",
    "requires javascript",
    "turn on javascript",
    "javascript must be enabled",
    "please enable javascript",
];

/// Anything that can fetch a URL as text (the `fetch_agent` seam the web
/// search agent and orchestrator accept, so tests can inject a fake).
pub trait UrlFetcher: Send + Sync {
    /// Fetch `url` and return its contents (see [`FetchAgent::fetch`]).
    fn fetch(&self, url: &str, max_length: usize, start_index: usize, raw: bool) -> String;
}

/// Fetch a URL and return its content converted to markdown.
#[derive(Debug, Clone, Copy, Default)]
pub struct FetchAgent;

impl FetchAgent {
    /// Construct the agent.
    pub fn new() -> Self {
        Self
    }

    /// Fetch `url` and return its contents as markdown.
    ///
    /// * `max_length`: maximum number of characters to return (`usize::MAX`
    ///   for Python's `sys.maxsize` default).
    /// * `start_index`: start content from this character index.
    /// * `raw`: return raw content without markdown conversion.
    ///
    /// Returns `""` on any fetch error.
    pub fn fetch(&self, url: &str, max_length: usize, start_index: usize, raw: bool) -> String {
        let _ = (url, max_length, start_index, raw);
        todo!()
    }

    /// True when the response is a PDF (by content type or URL suffix).
    pub fn is_pdf(url: &str, content_type: &str) -> bool {
        let _ = (url, content_type);
        todo!()
    }

    /// Convert HTML to markdown, stripping non-content elements first.
    pub fn html_to_markdown(html: &str) -> String {
        let _ = html;
        todo!()
    }

    /// Convert PDF bytes to markdown via the shared `PdfParser`.
    pub fn pdf_to_markdown(data: &[u8]) -> String {
        let _ = data;
        todo!()
    }

    /// True when the page needs JavaScript to render its content.
    pub fn is_javascript_page(content: &str) -> bool {
        let _ = content;
        todo!()
    }
}

impl UrlFetcher for FetchAgent {
    fn fetch(&self, url: &str, max_length: usize, start_index: usize, raw: bool) -> String {
        FetchAgent::fetch(self, url, max_length, start_index, raw)
    }
}
