//! HTML → Markdown: hand-port of MarkItDown's `HtmlConverter` and
//! `_CustomMarkdownify` (itself a subclass of `markdownify` 1.2's
//! `MarkdownConverter`).
//!
//! HTML is parsed with html5ever (through `htmd`'s tree builder) into a small
//! arena DOM that offers the BeautifulSoup navigation markdownify relies on
//! (siblings, parents, `find_all`), then rendered by a faithful port of
//! markdownify's recursive conversion.

use std::collections::HashSet;
use std::rc::Rc;

use markup5ever_rcdom::{Node, NodeData};
use once_cell::sync::Lazy;
use regex::Regex;

use super::DocumentConverterResult;

/// Kind of an arena node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HtmlNodeKind {
    Document,
    Element,
    Text,
    /// Comments, doctypes and processing instructions (always ignored).
    Other,
}

/// One arena node.
#[derive(Debug, Clone)]
pub struct HtmlNode {
    pub kind: HtmlNodeKind,
    /// Lower-case tag name for elements; empty otherwise.
    pub name: String,
    pub attrs: Vec<(String, String)>,
    /// Text content for text nodes.
    pub text: String,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
}

/// An arena DOM with BeautifulSoup-style navigation.
#[derive(Debug, Clone)]
pub struct HtmlDom {
    pub nodes: Vec<HtmlNode>,
    pub root: usize,
}

impl HtmlDom {
    /// Parse an HTML document.
    pub fn parse(html: &str) -> Self {
        let tree = htmd::HtmlToMarkdown::new()
            .html_to_tree(html)
            .unwrap_or_else(|_| Node::new(NodeData::Document));
        let mut dom = HtmlDom { nodes: Vec::new(), root: 0 };
        dom.root = dom.build(&tree, None);
        dom
    }

    fn build(&mut self, node: &Rc<Node>, parent: Option<usize>) -> usize {
        let (kind, name, attrs, text) = match &node.data {
            NodeData::Document => (HtmlNodeKind::Document, "[document]".to_string(), Vec::new(), String::new()),
            NodeData::Element { name, attrs, .. } => (
                HtmlNodeKind::Element,
                name.local.to_string().to_lowercase(),
                attrs
                    .borrow()
                    .iter()
                    .map(|a| (a.name.local.to_string(), a.value.to_string()))
                    .collect(),
                String::new(),
            ),
            NodeData::Text { contents } => (HtmlNodeKind::Text, String::new(), Vec::new(), contents.borrow().to_string()),
            _ => (HtmlNodeKind::Other, String::new(), Vec::new(), String::new()),
        };
        let id = self.nodes.len();
        self.nodes.push(HtmlNode { kind, name, attrs, text, parent, children: Vec::new() });
        let children: Vec<Rc<Node>> = node.children.borrow().iter().cloned().collect();
        for child in &children {
            let child_id = self.build(child, Some(id));
            self.nodes[id].children.push(child_id);
        }
        id
    }

    pub fn node(&self, id: usize) -> &HtmlNode {
        &self.nodes[id]
    }

    fn is_tag(&self, id: usize) -> bool {
        self.nodes[id].kind == HtmlNodeKind::Element
    }

    fn name(&self, id: usize) -> &str {
        &self.nodes[id].name
    }

    /// Attribute value of an element.
    pub fn attr(&self, id: usize, name: &str) -> Option<&str> {
        self.nodes[id].attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    /// Remove a node from its parent (BeautifulSoup `extract()`).
    pub fn detach(&mut self, id: usize) {
        if let Some(parent) = self.nodes[id].parent.take() {
            self.nodes[parent].children.retain(|c| *c != id);
        }
    }

    /// Every descendant element whose tag is in `tags`, in document order.
    pub fn find_all(&self, id: usize, tags: &[&str]) -> Vec<usize> {
        let mut out = Vec::new();
        self.walk_descendants(id, &mut |d| {
            if self.is_tag(d) && tags.contains(&self.name(d)) {
                out.push(d);
            }
        });
        out
    }

    /// First descendant element named `tag`.
    pub fn find(&self, id: usize, tag: &str) -> Option<usize> {
        self.find_all(id, &[tag]).into_iter().next()
    }

    fn walk_descendants(&self, id: usize, f: &mut dyn FnMut(usize)) {
        for &child in &self.nodes[id].children {
            f(child);
            self.walk_descendants(child, f);
        }
    }

    fn sibling_index(&self, id: usize) -> Option<(usize, usize)> {
        let parent = self.nodes[id].parent?;
        let idx = self.nodes[parent].children.iter().position(|c| *c == id)?;
        Some((parent, idx))
    }

    /// Previous sibling node of any kind (`previous_sibling`).
    pub fn previous_sibling(&self, id: usize) -> Option<usize> {
        let (parent, idx) = self.sibling_index(id)?;
        if idx == 0 { None } else { Some(self.nodes[parent].children[idx - 1]) }
    }

    /// Next sibling node of any kind (`next_sibling`).
    pub fn next_sibling(&self, id: usize) -> Option<usize> {
        let (parent, idx) = self.sibling_index(id)?;
        self.nodes[parent].children.get(idx + 1).copied()
    }

    /// Previous sibling *element* (`find_previous_sibling()`).
    fn previous_tag_sibling(&self, id: usize) -> Option<usize> {
        let (parent, idx) = self.sibling_index(id)?;
        self.nodes[parent].children[..idx].iter().rev().copied().find(|c| self.is_tag(*c))
    }

    /// Previous sibling elements named `tag` (`find_previous_siblings(tag)`).
    fn previous_siblings_named(&self, id: usize, tag: &str) -> usize {
        let Some((parent, idx)) = self.sibling_index(id) else {
            return 0;
        };
        self.nodes[parent].children[..idx].iter().filter(|c| self.is_tag(**c) && self.name(**c) == tag).count()
    }

    /// Nearest ancestor element named `tag` (`find_parent(tag)`).
    pub fn find_parent(&self, id: usize, tag: &str) -> Option<usize> {
        let mut current = self.nodes[id].parent;
        while let Some(p) = current {
            if self.is_tag(p) && self.name(p) == tag {
                return Some(p);
            }
            current = self.nodes[p].parent;
        }
        None
    }

    /// BeautifulSoup `get_text(separator, strip=True)`.
    pub fn get_text_stripped(&self, id: usize, separator: &str) -> String {
        let mut parts = Vec::new();
        self.walk_descendants(id, &mut |d| {
            if self.nodes[d].kind == HtmlNodeKind::Text {
                let t = self.nodes[d].text.trim();
                if !t.is_empty() {
                    parts.push(t.to_string());
                }
            }
        });
        parts.join(separator)
    }

    /// BeautifulSoup `.string`: the text of a node with exactly one
    /// (recursively single) child string, else `None`.
    pub fn string(&self, id: usize) -> Option<String> {
        let node = &self.nodes[id];
        match node.kind {
            HtmlNodeKind::Text => Some(node.text.clone()),
            HtmlNodeKind::Element | HtmlNodeKind::Document => {
                if node.children.len() == 1 { self.string(node.children[0]) } else { None }
            }
            HtmlNodeKind::Other => None,
        }
    }
}

// -- markdownify ---------------------------------------------------------------

static RE_LINE_WITH_CONTENT: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?m)^(.*)").expect("valid regex"));
static RE_WHITESPACE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\t ]+").expect("valid regex"));
static RE_ALL_WHITESPACE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\t \r\n]+").expect("valid regex"));
static RE_NEWLINE_WHITESPACE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[\t \r\n]*[\r\n][\t \r\n]*").expect("valid regex"));
static RE_HTML_HEADING: Lazy<Regex> = Lazy::new(|| Regex::new(r"^h(\d+)").expect("valid regex"));
static RE_PRE_LSTRIP: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[ \n]*\n").expect("valid regex"));
static RE_PRE_RSTRIP: Lazy<Regex> = Lazy::new(|| Regex::new(r"[ \n]*$").expect("valid regex"));
static RE_BACKTICK_RUNS: Lazy<Regex> = Lazy::new(|| Regex::new(r"`+").expect("valid regex"));

/// Nesting depth past which conversion gives up (Python's `RecursionError`).
const MAX_DEPTH: usize = 480;

/// Raised when the DOM is too deeply nested to convert recursively.
#[derive(Debug)]
struct RecursionLimit;

/// Python `str.strip()` with no arguments (Unicode whitespace).
fn py_strip(s: &str) -> &str {
    s.trim()
}

fn strip_chars<'a>(s: &'a str, chars: &[char]) -> &'a str {
    s.trim_matches(|c| chars.contains(&c))
}

/// markdownify `chomp`: move a leading/trailing space of inline text outside
/// the markup.
fn chomp(text: &str) -> (&'static str, &'static str, String) {
    let prefix = if text.starts_with(' ') { " " } else { "" };
    let suffix = if text.ends_with(' ') { " " } else { "" };
    (prefix, suffix, py_strip(text).to_string())
}

fn is_heading(name: &str) -> bool {
    RE_HTML_HEADING.is_match(name)
}

/// The `_CustomMarkdownify` options MarkItDown uses (markdownify defaults
/// plus ATX headings and truncated data URIs).
struct Markdownify<'a> {
    dom: &'a HtmlDom,
}

impl<'a> Markdownify<'a> {
    fn should_remove_whitespace_inside(&self, id: Option<usize>) -> bool {
        let Some(id) = id else { return false };
        if !self.dom.is_tag(id) {
            return false;
        }
        let name = self.dom.name(id);
        if is_heading(name) {
            return true;
        }
        matches!(
            name,
            "p" | "blockquote"
                | "article"
                | "div"
                | "section"
                | "ol"
                | "ul"
                | "li"
                | "dl"
                | "dt"
                | "dd"
                | "table"
                | "thead"
                | "tbody"
                | "tfoot"
                | "tr"
                | "td"
                | "th"
        )
    }

    fn should_remove_whitespace_outside(&self, id: Option<usize>) -> bool {
        self.should_remove_whitespace_inside(id) || id.is_some_and(|i| self.dom.is_tag(i) && self.dom.name(i) == "pre")
    }

    fn is_block_content_element(&self, id: usize) -> bool {
        let node = self.dom.node(id);
        match node.kind {
            HtmlNodeKind::Element | HtmlNodeKind::Document => true,
            HtmlNodeKind::Text => !py_strip(&node.text).is_empty(),
            HtmlNodeKind::Other => false,
        }
    }

    fn next_block_content_sibling(&self, id: usize) -> Option<usize> {
        let mut current = self.dom.next_sibling(id);
        while let Some(c) = current {
            if self.is_block_content_element(c) {
                return Some(c);
            }
            current = self.dom.next_sibling(c);
        }
        None
    }

    fn convert_root(&self, id: usize) -> Result<String, RecursionLimit> {
        self.process_tag(id, &HashSet::new(), 0)
    }

    fn process_element(&self, id: usize, parent_tags: &HashSet<String>, depth: usize) -> Result<String, RecursionLimit> {
        if self.dom.node(id).kind == HtmlNodeKind::Text {
            Ok(self.process_text(id, parent_tags))
        } else {
            self.process_tag(id, parent_tags, depth)
        }
    }

    fn can_ignore(&self, id: usize, should_remove_inside: bool) -> bool {
        let node = self.dom.node(id);
        match node.kind {
            // Tags are always processed.
            HtmlNodeKind::Element | HtmlNodeKind::Document => false,
            // Comment and Doctype elements are always ignored.
            HtmlNodeKind::Other => true,
            HtmlNodeKind::Text => {
                let prev = self.dom.previous_sibling(id);
                let next = self.dom.next_sibling(id);
                if !py_strip(&node.text).is_empty() {
                    // Non-whitespace text nodes are always processed.
                    false
                } else if should_remove_inside && (prev.is_none() || next.is_none()) {
                    // Inside block elements (excluding <pre>), ignore adjacent whitespace elements.
                    true
                } else {
                    // Outside block elements (including <pre>), ignore adjacent whitespace elements.
                    self.should_remove_whitespace_outside(prev) || self.should_remove_whitespace_outside(next)
                }
            }
        }
    }

    fn process_tag(&self, id: usize, parent_tags: &HashSet<String>, depth: usize) -> Result<String, RecursionLimit> {
        if depth > MAX_DEPTH {
            return Err(RecursionLimit);
        }
        let name = self.dom.name(id).to_string();

        // Collect child elements to process, ignoring whitespace-only text elements
        // adjacent to the inner/outer boundaries of block elements.
        let should_remove_inside = self.should_remove_whitespace_inside(Some(id));
        let children_to_convert: Vec<usize> = self
            .dom
            .node(id)
            .children
            .iter()
            .copied()
            .filter(|c| !self.can_ignore(*c, should_remove_inside))
            .collect();

        // Create a copy of this tag's parent context, then update it to include
        // this tag to propagate down into the children.
        let mut parent_tags_for_children = parent_tags.clone();
        parent_tags_for_children.insert(name.clone());

        // if this tag is a heading or table cell, add an '_inline' parent pseudo-tag
        if is_heading(&name) || name == "td" || name == "th" {
            parent_tags_for_children.insert("_inline".to_string());
        }

        // if this tag is a preformatted element, add a '_noformat' parent pseudo-tag
        if matches!(name.as_str(), "pre" | "code" | "kbd" | "samp") {
            parent_tags_for_children.insert("_noformat".to_string());
        }

        // Convert the children elements into a list of result strings.
        let mut child_strings = Vec::with_capacity(children_to_convert.len());
        for child in children_to_convert {
            let s = self.process_element(child, &parent_tags_for_children, depth + 1)?;
            // Remove empty string values.
            if !s.is_empty() {
                child_strings.push(s);
            }
        }

        // Collapse newlines at child element boundaries, if needed.
        let text = if name == "pre" || self.dom.find_parent(id, "pre").is_some() {
            // Inside <pre> blocks, do not collapse newlines.
            child_strings.concat()
        } else {
            let mut updated: Vec<String> = vec![String::new()]; // so the first lookback works
            for child_string in child_strings {
                // Separate the leading/trailing newlines from the content.
                let (mut leading, content, trailing) = extract_newlines(&child_string);

                // If the last child had trailing newlines and this child has leading
                // newlines, use the larger newline count, limited to 2.
                if !updated.last().map(|s| s.is_empty()).unwrap_or(true) && !leading.is_empty() {
                    let prev_trailing = updated.pop().unwrap_or_default();
                    let num_newlines = 2.min(prev_trailing.len().max(leading.len()));
                    leading = "\n".repeat(num_newlines);
                }

                updated.push(leading);
                updated.push(content);
                updated.push(trailing);
            }
            updated.concat()
        };

        // apply this tag's final conversion function
        Ok(self.convert(id, &name, text, parent_tags))
    }

    fn escape(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        // escape_misc is off; asterisks and underscores are escaped.
        text.replace('*', r"\*").replace('_', r"\_")
    }

    fn process_text(&self, id: usize, parent_tags: &HashSet<String>) -> String {
        let mut text = self.dom.node(id).text.clone();

        // normalize whitespace if we're not inside a preformatted element
        if !parent_tags.contains("pre") {
            text = RE_NEWLINE_WHITESPACE.replace_all(&text, "\n").into_owned();
            text = RE_WHITESPACE.replace_all(&text, " ").into_owned();
        }

        // escape special characters if we're not inside a preformatted or code element
        if !parent_tags.contains("_noformat") {
            text = self.escape(&text);
        }

        // remove leading whitespace at the start or just after a block-level
        // element; remove trailing whitespace at the end or just before a
        // block-level element.
        let prev = self.dom.previous_sibling(id);
        let next = self.dom.next_sibling(id);
        let parent = self.dom.node(id).parent;
        if self.should_remove_whitespace_outside(prev) || (self.should_remove_whitespace_inside(parent) && prev.is_none()) {
            text = text.trim_start_matches([' ', '\t', '\r', '\n']).to_string();
        }
        if self.should_remove_whitespace_outside(next) || (self.should_remove_whitespace_inside(parent) && next.is_none()) {
            text = text.trim_end().to_string();
        }

        text
    }

    fn convert(&self, id: usize, name: &str, text: String, parent_tags: &HashSet<String>) -> String {
        let inline = parent_tags.contains("_inline");
        let noformat = parent_tags.contains("_noformat");
        match name {
            "a" => self.convert_a(id, &text),
            "b" | "strong" => inline_conversion("**", &text, noformat),
            "em" | "i" => inline_conversion("*", &text, noformat),
            "del" | "s" => inline_conversion("~~", &text, noformat),
            // sub_symbol / sup_symbol default to "" (markup stripped).
            "sub" | "sup" => inline_conversion("", &text, noformat),
            "blockquote" => {
                let text = strip_chars(&text, &[' ', '\t', '\r', '\n']);
                if inline {
                    return format!(" {text} ");
                }
                if text.is_empty() {
                    return "\n".to_string();
                }
                let indented = RE_LINE_WITH_CONTENT.replace_all(text, |caps: &regex::Captures| {
                    let line = &caps[1];
                    if line.is_empty() { ">".to_string() } else { format!("> {line}") }
                });
                format!("\n{indented}\n\n")
            }
            "br" => {
                if inline {
                    " ".to_string()
                } else {
                    "  \n".to_string()
                }
            }
            "code" | "kbd" | "samp" => {
                if noformat {
                    return text;
                }
                let (prefix, suffix, text) = chomp(&text);
                if text.is_empty() {
                    return String::new();
                }
                // Delimit the code span with one more backtick than the longest run inside.
                let max_backticks = RE_BACKTICK_RUNS.find_iter(&text).map(|m| m.len()).max().unwrap_or(0);
                let delimiter = "`".repeat(max_backticks + 1);
                let text = if max_backticks > 0 { format!(" {text} ") } else { text };
                format!("{prefix}{delimiter}{text}{delimiter}{suffix}")
            }
            "div" | "article" | "section" | "dl" => {
                if inline {
                    return format!(" {} ", py_strip(&text));
                }
                let text = py_strip(&text);
                if text.is_empty() { String::new() } else { format!("\n\n{text}\n\n") }
            }
            "dd" => {
                let text = py_strip(&text);
                if inline {
                    return format!(" {text} ");
                }
                if text.is_empty() {
                    return "\n".to_string();
                }
                // indent definition content lines by four spaces
                let indented = RE_LINE_WITH_CONTENT
                    .replace_all(text, |caps: &regex::Captures| {
                        let line = &caps[1];
                        if line.is_empty() { String::new() } else { format!("    {line}") }
                    })
                    .into_owned();
                // insert definition marker into first-line indent whitespace
                let rest: String = indented.chars().skip(1).collect();
                format!(":{rest}\n")
            }
            "dt" => {
                let text = py_strip(&text);
                let text = RE_ALL_WHITESPACE.replace_all(text, " ");
                if inline {
                    return format!(" {text} ");
                }
                if text.is_empty() {
                    return "\n".to_string();
                }
                format!("\n\n{text}\n")
            }
            "hr" => "\n\n---\n\n".to_string(),
            "img" => self.convert_img(id),
            "video" => self.convert_video(id, &text, inline),
            "ul" | "ol" => {
                // Converting a list to inline is undefined; ignore inline parents.
                let before_paragraph = self
                    .next_block_content_sibling(id)
                    .is_some_and(|s| !(self.dom.is_tag(s) && matches!(self.dom.name(s), "ul" | "ol")));
                if parent_tags.contains("li") {
                    // remove trailing newline if we're in a nested list
                    return format!("\n{}", text.trim_end());
                }
                format!("\n\n{}{}", text, if before_paragraph { "\n" } else { "" })
            }
            "li" => self.convert_li(id, &text),
            "p" => {
                if inline {
                    return format!(" {} ", strip_chars(&text, &[' ', '\t', '\r', '\n']));
                }
                let text = strip_chars(&text, &[' ', '\t', '\r', '\n']);
                if text.is_empty() { String::new() } else { format!("\n\n{text}\n\n") }
            }
            "pre" => {
                if text.is_empty() {
                    return String::new();
                }
                // strip_pre = STRIP: remove all leading/trailing newlines
                let text = RE_PRE_LSTRIP.replace(&text, "");
                let text = RE_PRE_RSTRIP.replace(&text, "");
                format!("\n\n```\n{text}\n```\n\n")
            }
            "q" => format!("\"{text}\""),
            "script" | "style" => String::new(),
            "table" => format!("\n\n{}\n\n", py_strip(&text)),
            "caption" => format!("{}\n\n", py_strip(&text)),
            "figcaption" => format!("\n\n{}\n\n", py_strip(&text)),
            "td" | "th" => {
                let colspan = self.colspan(id);
                format!(" {}{}", py_strip(&text).replace('\n', " "), " |".repeat(colspan))
            }
            "tr" => self.convert_tr(id, &text),
            "input" => {
                // Convert checkboxes to Markdown [x]/[ ] syntax.
                if self.dom.attr(id, "type") == Some("checkbox") {
                    if self.dom.attr(id, "checked").is_some() { "[x] ".to_string() } else { "[ ] ".to_string() }
                } else {
                    String::new()
                }
            }
            "[document]" => strip_chars(&text, &['\n']).to_string(),
            n if is_heading(n) => {
                if inline {
                    return text;
                }
                // Markdown does not support heading depths of n > 6
                let level: usize = RE_HTML_HEADING
                    .captures(n)
                    .and_then(|c| c[1].parse().ok())
                    .unwrap_or(1)
                    .clamp(1, 6);
                let text = py_strip(&text);
                let text = RE_ALL_WHITESPACE.replace_all(text, " ");
                format!("\n\n{} {}\n\n", "#".repeat(level), text)
            }
            _ => text,
        }
    }

    fn colspan(&self, id: usize) -> usize {
        match self.dom.attr(id, "colspan") {
            Some(v) if !v.is_empty() && v.chars().all(|c| c.is_ascii_digit()) => {
                v.parse::<usize>().unwrap_or(1).clamp(1, 1000)
            }
            _ => 1,
        }
    }

    /// `_CustomMarkdownify.convert_a`: removes JavaScript links and escapes URIs.
    fn convert_a(&self, id: usize, text: &str) -> String {
        let (prefix, suffix, text) = chomp(text);
        if text.is_empty() {
            return String::new();
        }

        if self.dom.find_parent(id, "pre").is_some() {
            return text;
        }

        let mut href = self.dom.attr(id, "href").map(str::to_string);
        let title = self.dom.attr(id, "title").map(str::to_string);

        // Escape URIs and skip non-http or file schemes
        if let Some(h) = href.as_ref().filter(|h| !h.is_empty()) {
            match urlparse(h) {
                Ok(parsed) => {
                    if !parsed.scheme.is_empty() && !matches!(parsed.scheme.as_str(), "http" | "https" | "file") {
                        return format!("{prefix}{text}{suffix}");
                    }
                    let path = url_quote(&url_unquote(&parsed.path));
                    href = Some(parsed.with_path(path).unparse());
                }
                Err(()) => return format!("{prefix}{text}{suffix}"),
            }
        }

        // For the replacement see #29: text nodes underscores are escaped
        let title_empty = title.as_deref().unwrap_or("").is_empty();
        if href.as_deref().is_some_and(|h| text.replace(r"\_", "_") == h) && title_empty {
            // Shortcut syntax (autolinks on, default_title off)
            return format!("<{}>", href.unwrap_or_default());
        }
        let title_part = match title.as_deref() {
            Some(t) if !t.is_empty() => format!(" \"{}\"", t.replace('"', "\\\"")),
            _ => String::new(),
        };
        match href.filter(|h| !h.is_empty()) {
            Some(h) => format!("{prefix}[{text}]({h}{title_part}){suffix}"),
            None => text,
        }
    }

    /// `_CustomMarkdownify.convert_img`: same as usual, but removes data URIs.
    fn convert_img(&self, id: usize) -> String {
        let alt = self.dom.attr(id, "alt").unwrap_or("");
        let src = self
            .dom
            .attr(id, "src")
            .filter(|s| !s.is_empty())
            .or_else(|| self.dom.attr(id, "data-src"))
            .unwrap_or("");
        let title = self.dom.attr(id, "title").unwrap_or("");
        let title_part = if title.is_empty() { String::new() } else { format!(" \"{}\"", title.replace('"', "\\\"")) };
        // Remove all line breaks from alt
        let alt = alt.replace('\n', " ");

        // Remove dataURIs
        let src = if src.starts_with("data:") {
            format!("{}...", src.split(',').next().unwrap_or(""))
        } else {
            src.to_string()
        };

        format!("![{alt}]({src}{title_part})")
    }

    fn convert_video(&self, id: usize, text: &str, inline: bool) -> String {
        if inline {
            return text.to_string();
        }
        let mut src = self.dom.attr(id, "src").unwrap_or("").to_string();
        if src.is_empty() {
            if let Some(source) = self
                .dom
                .find_all(id, &["source"])
                .into_iter()
                .find(|s| self.dom.attr(*s, "src").is_some())
            {
                src = self.dom.attr(source, "src").unwrap_or("").to_string();
            }
        }
        let poster = self.dom.attr(id, "poster").unwrap_or("");
        match (src.is_empty(), poster.is_empty()) {
            (false, false) => format!("[![{text}]({poster})]({src})"),
            (false, true) => format!("[{text}]({src})"),
            (true, false) => format!("![{text}]({poster})"),
            (true, true) => text.to_string(),
        }
    }

    fn convert_li(&self, id: usize, text: &str) -> String {
        let text = py_strip(text);
        if text.is_empty() {
            return "\n".to_string();
        }

        // determine list item bullet character to use
        let parent = self.dom.node(id).parent;
        let bullet = if let Some(p) = parent.filter(|p| self.dom.is_tag(*p) && self.dom.name(*p) == "ol") {
            let start = match self.dom.attr(p, "start") {
                Some(s) if !s.is_empty() && s.chars().all(|c| c.is_numeric()) => s.parse::<i64>().unwrap_or(1),
                _ => 1,
            };
            format!("{}.", start + self.dom.previous_siblings_named(id, "li") as i64)
        } else {
            let mut depth: i64 = -1;
            let mut current = Some(id);
            while let Some(c) = current {
                if self.dom.is_tag(c) && self.dom.name(c) == "ul" {
                    depth += 1;
                }
                current = self.dom.node(c).parent;
            }
            let bullets = ['*', '+', '-'];
            bullets[depth.rem_euclid(bullets.len() as i64) as usize].to_string()
        };
        let bullet = format!("{bullet} ");
        let bullet_width = bullet.chars().count();
        let bullet_indent = " ".repeat(bullet_width);

        // indent content lines by bullet width
        let indented = RE_LINE_WITH_CONTENT
            .replace_all(text, |caps: &regex::Captures| {
                let line = &caps[1];
                if line.is_empty() { String::new() } else { format!("{bullet_indent}{line}") }
            })
            .into_owned();

        // insert bullet into first-line indent whitespace
        let rest: String = indented.chars().skip(bullet_width).collect();
        format!("{bullet}{rest}\n")
    }

    fn convert_tr(&self, id: usize, text: &str) -> String {
        let cells = self.dom.find_all(id, &["td", "th"]);
        let is_first_row = self.dom.previous_tag_sibling(id).is_none();
        let parent = self.dom.node(id).parent;
        let parent_name = parent.map(|p| self.dom.name(p).to_string()).unwrap_or_default();
        let is_headrow = cells.iter().all(|c| self.dom.name(*c) == "th")
            || (parent_name == "thead"
                // avoid multiple tr in thead
                && parent.is_some_and(|p| self.dom.find_all(p, &["tr"]).len() == 1));
        let is_head_row_missing = (is_first_row && parent_name != "tbody")
            || (is_first_row
                && parent_name == "tbody"
                && parent
                    .and_then(|p| self.dom.node(p).parent)
                    .is_none_or(|gp| self.dom.find_all(gp, &["thead"]).is_empty()));

        let full_colspan: usize = cells.iter().map(|c| self.colspan(*c)).sum();

        let mut overline = String::new();
        let mut underline = String::new();
        // table_infer_header is off.
        if is_headrow && is_first_row {
            // first row and is headline: print headline underline
            underline.push_str(&format!("| {} |\n", vec!["---"; full_colspan].join(" | ")));
        } else if is_head_row_missing
            || (is_first_row
                && (parent_name == "table"
                    || (parent_name == "tbody" && parent.and_then(|p| self.dom.previous_tag_sibling(p)).is_none())))
        {
            // headline is missing (header inference disabled), or first row, not
            // headline, at the beginning of a table: print empty headline above
            overline.push_str(&format!("| {} |\n", vec![""; full_colspan].join(" | ")));
            overline.push_str(&format!("| {} |\n", vec!["---"; full_colspan].join(" | ")));
        }
        format!("{overline}|{text}\n{underline}")
    }
}

/// markdownify `abstract_inline_conversion` for simple inline tags.
fn inline_conversion(markup: &str, text: &str, noformat: bool) -> String {
    if noformat {
        return text.to_string();
    }
    let (prefix, suffix, text) = chomp(text);
    if text.is_empty() {
        return String::new();
    }
    format!("{prefix}{markup}{text}{markup}{suffix}")
}

/// `re_extract_newlines`: split off leading and trailing newlines.
fn extract_newlines(s: &str) -> (String, String, String) {
    let trimmed_start = s.trim_start_matches('\n');
    let leading = s.len() - trimmed_start.len();
    let content = trimmed_start.trim_end_matches('\n');
    let trailing = trimmed_start.len() - content.len();
    ("\n".repeat(leading), content.to_string(), "\n".repeat(trailing))
}

// -- urllib.parse (the subset convert_a needs) ----------------------------------

/// Schemes whose URLs carry `;params` on the last path segment.
const USES_PARAMS: &[&str] = &[
    "", "ftp", "hdl", "prospero", "http", "imap", "https", "shttp", "rtsp", "rtspu", "sip", "sips", "mms", "sftp",
    "tel",
];
/// Schemes that use a `//netloc` part.
const USES_NETLOC: &[&str] = &[
    "", "ftp", "http", "gopher", "nntp", "telnet", "imap", "wais", "file", "mms", "https", "shttp", "snews",
    "prospero", "rtsp", "rtspu", "rsync", "svn", "svn+ssh", "sftp", "nfs", "git", "git+ssh", "ws", "wss", "itms-services",
];

struct ParsedUrl {
    scheme: String,
    netloc: String,
    path: String,
    params: String,
    query: Option<String>,
    fragment: Option<String>,
}

impl ParsedUrl {
    fn with_path(mut self, path: String) -> Self {
        self.path = path;
        self
    }

    /// `urlunparse`.
    fn unparse(&self) -> String {
        let mut url = if self.params.is_empty() { self.path.clone() } else { format!("{};{}", self.path, self.params) };
        if !self.netloc.is_empty() || (!self.scheme.is_empty() && USES_NETLOC.contains(&self.scheme.as_str()) && !url.starts_with("//")) {
            if !url.is_empty() && !url.starts_with('/') {
                url = format!("/{url}");
            }
            url = format!("//{}{}", self.netloc, url);
        }
        if !self.scheme.is_empty() {
            url = format!("{}:{}", self.scheme, url);
        }
        if let Some(q) = self.query.as_ref().filter(|q| !q.is_empty()) {
            url = format!("{url}?{q}");
        }
        if let Some(f) = self.fragment.as_ref().filter(|f| !f.is_empty()) {
            url = format!("{url}#{f}");
        }
        url
    }
}

/// `urllib.parse.urlparse`; `Err` stands for its `ValueError` (bad IPv6 netloc).
fn urlparse(url: &str) -> Result<ParsedUrl, ()> {
    let mut rest = url.trim_matches(|c: char| c <= ' ').to_string();
    rest.retain(|c| c != '\t' && c != '\r' && c != '\n');

    let mut scheme = String::new();
    if let Some(colon) = rest.find(':') {
        let candidate = &rest[..colon];
        if !candidate.is_empty()
            && candidate.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
            && candidate.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
        {
            scheme = candidate.to_ascii_lowercase();
            rest = rest[colon + 1..].to_string();
        }
    }

    let mut netloc = String::new();
    if let Some(after) = rest.strip_prefix("//") {
        let end = after.find(['/', '?', '#']).unwrap_or(after.len());
        netloc = after[..end].to_string();
        rest = after[end..].to_string();
        if netloc.contains('[') != netloc.contains(']') {
            return Err(());
        }
    }

    let mut fragment = None;
    if let Some(hash) = rest.find('#') {
        fragment = Some(rest[hash + 1..].to_string());
        rest.truncate(hash);
    }
    let mut query = None;
    if let Some(q) = rest.find('?') {
        query = Some(rest[q + 1..].to_string());
        rest.truncate(q);
    }

    let mut path = rest;
    let mut params = String::new();
    if USES_PARAMS.contains(&scheme.as_str()) && path.contains(';') {
        let segment_start = path.rfind('/').map(|i| i + 1).unwrap_or(0);
        if let Some(semi) = path[segment_start..].find(';') {
            let idx = segment_start + semi;
            params = path[idx + 1..].to_string();
            path.truncate(idx);
        }
    }

    Ok(ParsedUrl { scheme, netloc, path, params, query, fragment })
}

/// `urllib.parse.unquote` (UTF-8, invalid sequences replaced).
fn url_unquote(s: &str) -> String {
    if !s.contains('%') {
        return s.to_string();
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() + 0 && i + 2 <= bytes.len() - 1 + 0 {
            let hex = &s[i + 1..i + 3];
            if let Ok(b) = u8::from_str_radix(hex, 16)
                && hex.chars().all(|c| c.is_ascii_hexdigit())
            {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// `urllib.parse.quote(s)` with the default `safe="/"`.
fn url_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-' | b'~' | b'/') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

// -- HtmlConverter -------------------------------------------------------------

/// Convert an HTML string to Markdown (MarkItDown `HtmlConverter.convert`).
pub fn convert_html(html: &str) -> DocumentConverterResult {
    let mut dom = HtmlDom::parse(html);

    // Remove javascript and style blocks
    let scripts = dom.find_all(dom.root, &["script", "style"]);
    for id in scripts {
        dom.detach(id);
    }

    // Print only the main content
    let target = dom.find(dom.root, "body").unwrap_or(dom.root);
    let converter = Markdownify { dom: &dom };
    let webpage_text = match converter.convert_root(target) {
        Ok(text) => text,
        Err(RecursionLimit) => {
            // Large or deeply-nested HTML can exceed the recursion limit during
            // markdownify's recursive DOM traversal. Fall back to an iterative
            // get_text() so the caller still gets usable plain-text content.
            tracing::warn!(
                target: "smartloop.parsers.document_parser",
                "HTML document is too deeply nested for markdown conversion (RecursionError). Falling back to plain-text extraction."
            );
            dom.get_text_stripped(target, "\n")
        }
    };

    let title = dom.find(dom.root, "title").and_then(|t| dom.string(t));

    DocumentConverterResult { title, text_content: py_strip(&webpage_text).to_string() }
}

/// Minimal HTML text escaping (Python `html.escape(s, quote=True)`).
pub fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected strings were produced by MarkItDown's `_CustomMarkdownify`
    // (markdownify 1.2.2) on the same input.

    #[test]
    fn pandas_style_table() {
        let html = "<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th>a_b</th>\n      <th>b</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <td>1</td>\n      <td>x*y</td>\n    </tr>\n  </tbody>\n</table>";
        assert_eq!(convert_html(html).text_content, "| a\\_b | b |\n| --- | --- |\n| 1 | x\\*y |");
    }

    #[test]
    fn mixed_document() {
        let html = "<table><tr><td>a</td><td>b</td></tr>\n<tr><td>c</td></tr></table><ul><li>x<ul><li>y</li></ul></li><li>z</li></ul><p>para <b>bold </b>t</p><ol start=3><li>one</li><li>two</li></ol><a href=\"http://ex.com/a b\">ex</a> <a href=\"javascript:x\">js</a><img src=\"data:image/png;base64,AAAA\" alt=\"pic\"><h2>Head</h2><blockquote>q1<br>q2</blockquote><pre>code\n  x</pre>";
        assert_eq!(
            convert_html(html).text_content,
            "|  |  |\n| --- | --- |\n| a | b |\n| c |\n\n* x\n  + y\n* z\n\npara **bold** t\n\n3. one\n4. two\n\n[ex](http://ex.com/a%20b) js![pic](data:image/png;base64...)\n\n## Head\n\n> q1  \n> q2\n\n```\ncode\n  x\n```"
        );
    }

    #[test]
    fn title_and_scripts() {
        let html = "<html><head><title>T</title><style>p{}</style></head><body><script>x()</script><p>Hello <em>w</em></p></body></html>";
        let result = convert_html(html);
        assert_eq!(result.title.as_deref(), Some("T"));
        assert_eq!(result.text_content, "Hello *w*");
    }
}
