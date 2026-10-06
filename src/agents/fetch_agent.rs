//! URL fetch agent — fetches a URL and returns its content as markdown.
//!
//! Port of `smartloop/agents/fetch_agent.py`.
//!
//! Replaces the fetch MCP server.  Given a URL it performs the HTTP GET itself
//! and converts the response to markdown based on content type:
//!
//! * HTML → strip non-content tags + html→markdown (`htmd`, in place of
//!   BeautifulSoup + markdownify).
//! * PDF  → the shared [`PdfParser`] (same converter used for uploaded
//!   documents), so scanned PDFs still get vision OCR.
//! * anything else → returned as raw text.
//!
//! It runs as an ordinary plan step (`StepType::Fetch`) resolved by the
//! orchestrator, not as an MCP tool.

use std::io::Write;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use markup5ever_rcdom::{Node, NodeData};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use regex::Regex;

use crate::parsers::base_parser::BaseParser;
use crate::parsers::pdf_parser::PdfParser;

/// Python logger name.
const LOG_TARGET: &str = "smartloop.agents.fetch_agent";

/// Tags removed before HTML → markdown conversion.
pub const STRIP_TAGS: &[&str] = &["script", "style", "noscript", "svg", "iframe", "link", "meta"];

/// User-Agent sent with every fetch.
pub const USER_AGENT: &str = "Mozilla/5.0 (compatible; Smartloop/1.0)";

/// Request timeout in seconds.
pub const FETCH_TIMEOUT_SECS: u64 = 30;

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

/// Replacement for every [`FetchAgent::fetch`] call in the process:
/// `(url, max_length, start_index, raw) -> content`.
pub type FetchOverride = Arc<dyn Fn(&str, usize, usize, bool) -> String + Send + Sync>;

/// Process-wide [`FetchOverride`] (the Python tests'
/// `patch("smartloop.agents.fetch_agent.FetchAgent.fetch", ...)`, which
/// patches the class and so every instance — the explorer's and the
/// pipeline's alike).
static FETCH_OVERRIDE: Lazy<RwLock<Option<FetchOverride>>> = Lazy::new(|| RwLock::new(None));

/// Replace (or, with `None`, restore) what [`FetchAgent::fetch`] does for
/// every instance in the process.
pub fn set_fetch_override(f: Option<FetchOverride>) {
    *FETCH_OVERRIDE.write() = f;
}

static EXCESS_NEWLINES_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n{3,}").unwrap());
static BODY_TAG_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)<body[\s>/]").unwrap());
static ROOT_ID_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(root|app|__next)$").unwrap());

/// Elements whose text BeautifulSoup does not count as text (`Script`,
/// `Stylesheet` and `TemplateString` strings are skipped by `get_text`).
const NON_TEXT_TAGS: &[&str] = &["script", "style", "template"];

/// The local name of an element node, or `None`.
fn tag_name(node: &Node) -> Option<String> {
    match &node.data {
        NodeData::Element { name, .. } => Some(name.local.to_string()),
        _ => None,
    }
}

/// The value of attribute `attr` on an element node.
fn attr_value(node: &Node, attr: &str) -> Option<String> {
    match &node.data {
        NodeData::Element { attrs, .. } => attrs
            .borrow()
            .iter()
            .find(|a| &*a.name.local == attr)
            .map(|a| a.value.to_string()),
        _ => None,
    }
}

/// The first element (document order) satisfying `pred`.
fn find_first(node: &Rc<Node>, pred: &dyn Fn(&Node) -> bool) -> Option<Rc<Node>> {
    for child in node.children.borrow().iter() {
        if pred(child) {
            return Some(Rc::clone(child));
        }
        if let Some(found) = find_first(child, pred) {
            return Some(found);
        }
    }
    None
}

/// Count elements named `tag` under `node`.
fn count_tags(node: &Rc<Node>, tag: &str) -> usize {
    node.children
        .borrow()
        .iter()
        .map(|c| usize::from(tag_name(c).as_deref() == Some(tag)) + count_tags(c, tag))
        .sum()
}

/// BeautifulSoup's `_all_strings(strip=True)`: every text string under
/// `node`, stripped, empty ones dropped.
fn collect_strings(node: &Rc<Node>, out: &mut Vec<String>) {
    for child in node.children.borrow().iter() {
        match &child.data {
            NodeData::Text { contents } => {
                let text = contents.borrow();
                let stripped = text.trim();
                if !stripped.is_empty() {
                    out.push(stripped.to_string());
                }
            }
            NodeData::Element { name, .. } => {
                if NON_TEXT_TAGS.contains(&&*name.local) {
                    continue;
                }
                collect_strings(child, out);
            }
            _ => {}
        }
    }
}

/// BeautifulSoup `get_text(separator=sep, strip=True)`.
fn get_text(node: &Rc<Node>, separator: &str) -> String {
    let mut parts = Vec::new();
    collect_strings(node, &mut parts);
    parts.join(separator)
}

/// Parse HTML the way BeautifulSoup's `html.parser` sees it: `<noscript>`
/// content is markup, not raw text.
fn parse_html(html: &str) -> std::io::Result<Rc<Node>> {
    htmd::HtmlToMarkdown::builder().scripting_enabled(false).build().html_to_tree(html)
}

/// The fetched response, read according to what the caller needs.
struct FetchedResponse {
    content_type: String,
    body: FetchedBody,
}

enum FetchedBody {
    Bytes(Vec<u8>),
    Text(String),
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
        let override_fn = FETCH_OVERRIDE.read().clone();
        if let Some(f) = override_fn {
            return f(url, max_length, start_index, raw);
        }

        let response = match Self::http_get(url, raw) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(target: LOG_TARGET, "[fetch] Fetch error for {}: {}", url, e);
                return String::new();
            }
        };

        let content_type = response.content_type;
        let text = match response.body {
            FetchedBody::Bytes(data) => Self::pdf_to_markdown(&data),
            FetchedBody::Text(body) => {
                if !raw && content_type.contains("html") {
                    let md = Self::html_to_markdown(&body);
                    if Self::is_javascript_page(&body) {
                        return format!(
                            "The page at {} requires JavaScript to render its content. \
                             JavaScript-rendered pages cannot be processed.",
                            url
                        );
                    }
                    md
                } else {
                    body
                }
            }
        };

        tracing::info!(target: LOG_TARGET, "[fetch] Fetched {} chars from {}", text.chars().count(), url);
        text.chars().skip(start_index).take(max_length).collect()
    }

    /// GET `url` (redirects followed, 30s timeout) and read the body: bytes
    /// for a PDF (unless `raw`), decoded text otherwise.  Fails on a
    /// transport error or a non-2xx status (`raise_for_status`).
    fn http_get(url: &str, raw: bool) -> anyhow::Result<FetchedResponse> {
        let url = url.to_string();
        let run = move || -> anyhow::Result<FetchedResponse> {
            let client = reqwest::blocking::Client::builder()
                .redirect(reqwest::redirect::Policy::limited(20))
                .timeout(Duration::from_secs(FETCH_TIMEOUT_SECS))
                .build()?;
            let response =
                client.get(&url).header(reqwest::header::USER_AGENT, USER_AGENT).send()?.error_for_status()?;
            let content_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_lowercase();
            let body = if !raw && Self::is_pdf(&url, &content_type) {
                FetchedBody::Bytes(response.bytes()?.to_vec())
            } else {
                FetchedBody::Text(response.text()?)
            };
            Ok(FetchedResponse { content_type, body })
        };
        // The blocking client must not run on an async runtime thread.
        if tokio::runtime::Handle::try_current().is_ok() {
            std::thread::spawn(run).join().map_err(|_| anyhow::anyhow!("fetch thread panicked"))?
        } else {
            run()
        }
    }

    /// True when the response is a PDF (by content type or URL suffix).
    pub fn is_pdf(url: &str, content_type: &str) -> bool {
        content_type.contains("application/pdf")
            || url.to_lowercase().split('?').next().unwrap_or("").ends_with(".pdf")
    }

    /// Convert HTML to markdown, stripping non-content elements first.
    /// Falls back to the raw HTML when conversion fails.
    pub fn html_to_markdown(html: &str) -> String {
        let options = htmd::options::Options {
            heading_style: htmd::options::HeadingStyle::Setex,
            ul_bullet_spacing: 1,
            ol_number_spacing: 1,
            ..Default::default()
        };
        let mut skip: Vec<&str> = STRIP_TAGS.to_vec();
        // markdownify(strip=["img"]): images contribute nothing.
        skip.push("img");
        let converter = htmd::HtmlToMarkdown::builder().options(options).skip_tags(skip).build();
        match converter.convert(html) {
            Ok(md) => EXCESS_NEWLINES_RE.replace_all(&md, "\n\n").trim().to_string(),
            Err(_) => html.to_string(),
        }
    }

    /// Convert PDF bytes to markdown via the shared `PdfParser`.
    pub fn pdf_to_markdown(data: &[u8]) -> String {
        let attempt = || -> anyhow::Result<String> {
            let mut tmp = tempfile::Builder::new().suffix(".pdf").tempfile()?;
            tmp.write_all(data)?;
            tmp.flush()?;
            let path = tmp.path().to_string_lossy().to_string();
            // The temp file is removed when `tmp` drops, after parsing.
            PdfParser::new(None, path).parse()
        };
        match attempt() {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!(target: LOG_TARGET, "[fetch] PDF conversion failed: {}", e);
                String::new()
            }
        }
    }

    /// True when the page needs JavaScript to render its content.
    pub fn is_javascript_page(content: &str) -> bool {
        if content.is_empty() {
            return false;
        }
        let Ok(doc) = parse_html(content) else {
            return false;
        };

        if let Some(noscript) = find_first(&doc, &|n| tag_name(n).as_deref() == Some("noscript")) {
            let noscript_text = get_text(&noscript, "").to_lowercase();
            if JS_PHRASES.iter().any(|phrase| noscript_text.contains(phrase)) {
                return true;
            }
        }

        // `soup.find("body") or soup`: html.parser only has a body element when
        // the markup spells one out.
        let body = if BODY_TAG_RE.is_match(content) {
            find_first(&doc, &|n| tag_name(n).as_deref() == Some("body")).unwrap_or_else(|| Rc::clone(&doc))
        } else {
            Rc::clone(&doc)
        };
        let visible_text = get_text(&body, " ");
        let scripts = count_tags(&doc, "script");
        if scripts > 0 && visible_text.chars().count() < 200 {
            return true;
        }

        let root_div = find_first(&doc, &|n| {
            tag_name(n).as_deref() == Some("div")
                && attr_value(n, "id").map(|id| ROOT_ID_RE.is_match(&id)).unwrap_or(false)
        });
        if let Some(root_div) = root_div {
            if get_text(&root_div, "").is_empty() {
                return true;
            }
        }
        false
    }
}

impl UrlFetcher for FetchAgent {
    fn fetch(&self, url: &str, max_length: usize, start_index: usize, raw: bool) -> String {
        FetchAgent::fetch(self, url, max_length, start_index, raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noscript_phrase_marks_a_javascript_page() {
        let html = "<html><body><noscript>Please enable JavaScript to continue.</noscript><p>x</p></body></html>";
        assert!(FetchAgent::is_javascript_page(html));
        // `get_text(strip=True)` joins the strings with nothing, so a phrase
        // split across tags does not read as one (as in BeautifulSoup).
        let split = format!(
            "<html><body><noscript>Please <b>enable</b> JavaScript</noscript><p>{}</p></body></html>",
            "Real text. ".repeat(30)
        );
        assert!(!FetchAgent::is_javascript_page(&split));
    }

    #[test]
    fn scripts_with_little_text_mark_a_javascript_page() {
        let html = "<html><head><script>var a=1;</script></head><body><p>Loading</p></body></html>";
        assert!(FetchAgent::is_javascript_page(html));
    }

    #[test]
    fn empty_root_div_marks_a_javascript_page() {
        let html = "<html><body><div id=\"root\"></div></body></html>";
        assert!(FetchAgent::is_javascript_page(html));
    }

    #[test]
    fn content_page_is_not_a_javascript_page() {
        let text = "Real article text. ".repeat(20);
        let html = format!("<html><head><script>x</script></head><body><p>{text}</p></body></html>");
        assert!(!FetchAgent::is_javascript_page(&html));
        assert!(!FetchAgent::is_javascript_page(""));
    }

    #[test]
    fn html_to_markdown_strips_non_content() {
        let md = FetchAgent::html_to_markdown(
            "<h1>Hi</h1><script>evil()</script><style>.a{}</style><p>a <img src=x alt=y> b</p>\n\n\n\n<p>c</p>",
        );
        assert!(md.contains("Hi"));
        assert!(!md.contains("evil"));
        assert!(!md.contains(".a{}"));
        assert!(!md.contains("\n\n\n"));
    }

    #[test]
    fn pdf_detection() {
        assert!(FetchAgent::is_pdf("https://x.com/a.PDF?x=1", ""));
        assert!(FetchAgent::is_pdf("https://x.com/a", "application/pdf"));
        assert!(!FetchAgent::is_pdf("https://x.com/a.html", "text/html"));
    }
}
