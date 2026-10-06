//! Port of `smartloop/parsers/docx_parser.py`.
//!
//! A parser for processing .docx files in XML format (python-docx paragraph
//! walking hand-ported over `zip` + `quick-xml`).

use std::collections::HashMap;

use crate::parsers::base_parser::{BaseParser, SharedLLM};
use crate::parsers::markitdown::ooxml::{OoxmlPackage, XmlElement, XmlNode};

/// Logger name used by the Python module.
const LOGGER: &str = "smartloop.parsers.docx_parser";

/// Paragraph alignment (python-docx `WD_PARAGRAPH_ALIGNMENT`), from `w:jc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParagraphAlignment {
    Left,
    Center,
    Right,
    Justify,
    Distribute,
    Other,
}

impl ParagraphAlignment {
    fn from_xml(value: &str) -> Self {
        match value {
            "left" => ParagraphAlignment::Left,
            "center" => ParagraphAlignment::Center,
            "right" => ParagraphAlignment::Right,
            "both" => ParagraphAlignment::Justify,
            "distribute" => ParagraphAlignment::Distribute,
            _ => ParagraphAlignment::Other,
        }
    }
}

/// One `w:p` paragraph: its text, resolved style name and explicit alignment.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DocxParagraph {
    pub text: String,
    /// Style display name (e.g. "Normal", "Heading 1", "List Paragraph").
    pub style_name: String,
    /// Explicit paragraph alignment; `None` when inherited.
    pub alignment: Option<ParagraphAlignment>,
}

/// A parser for processing .docx files in XML format.
pub struct DocxParser {
    pub llm: Option<SharedLLM>,
    pub file_path: String,
}

impl DocxParser {
    pub fn new(llm: Option<SharedLLM>, file_path: impl Into<String>) -> Self {
        Self { llm, file_path: file_path.into() }
    }

    /// Determines if a paragraph is a list item (style starts with "List" or
    /// explicit left alignment).
    pub fn is_list_item(&self, paragraph: &DocxParagraph) -> bool {
        paragraph.style_name.starts_with("List") || paragraph.alignment == Some(ParagraphAlignment::Left)
    }

    /// Read every paragraph of the document body (python-docx `doc.paragraphs`:
    /// the body's direct `w:p` children).
    pub fn read_paragraphs(&self) -> anyhow::Result<Vec<DocxParagraph>> {
        let bytes = std::fs::read(&self.file_path)?;
        let package = OoxmlPackage::from_bytes(&bytes)?;

        let document_part = main_document_part(&package);
        let document = package.read_xml(&document_part)?;
        let styles = ParagraphStyles::load(&package, &document_part);

        let Some(body) = document.child("w:body") else {
            return Ok(Vec::new());
        };

        Ok(body
            .children_named("w:p")
            .map(|p| {
                let ppr = p.child("w:pPr");
                let style_id = ppr.and_then(|ppr| ppr.child("w:pStyle")).and_then(|s| s.attr("w:val"));
                let alignment = ppr
                    .and_then(|ppr| ppr.child("w:jc"))
                    .and_then(|jc| jc.attr("w:val"))
                    .map(ParagraphAlignment::from_xml);
                DocxParagraph { text: paragraph_text(p), style_name: styles.name_for(style_id), alignment }
            })
            .collect())
    }

    /// Group paragraphs into heading-led text blocks.
    pub fn load_paragraphs(&self) -> anyhow::Result<Vec<String>> {
        let mut text_blocks: Vec<String> = Vec::new();
        let mut current_block: Option<String> = None;
        let mut carry_over_heading: Option<String> = None;

        for paragraph in self.read_paragraphs()? {
            let text = paragraph.text.trim().to_string();
            if text.is_empty() {
                continue; // Skip empty paragraphs
            }

            // Check if the paragraph is a heading
            if paragraph.style_name.starts_with("Heading") {
                if let (Some(block), Some(heading)) = (current_block.as_ref(), carry_over_heading.as_ref()) {
                    // Merge carry-over heading with the current block
                    current_block = Some(format!("{} {}", heading, block.trim()));
                    carry_over_heading = None;
                }

                // Handle deferred heading
                if let Some(block) = current_block.as_ref()
                    && !block.is_empty()
                    && block.split_whitespace().count() <= 2
                {
                    carry_over_heading = Some(block.trim().to_string());
                    current_block = None;
                }

                // Start a new block for the heading
                if let Some(block) = current_block.as_ref()
                    && !block.is_empty()
                {
                    text_blocks.push(block.trim().to_string());
                }
                tracing::info!(target: LOGGER, "Started new text block with heading: {}", text);
                current_block = Some(text);
            }
            // Check if the paragraph is a list item
            else if self.is_list_item(&paragraph) {
                append_text(&mut current_block, &mut carry_over_heading, &text);
                tracing::info!(target: LOGGER, "Appended list item to current block: {}", text);
            }
            // Append to the current block for regular paragraphs
            else {
                append_text(&mut current_block, &mut carry_over_heading, &text);
                tracing::info!(target: LOGGER, "Appended text to current block: {}", text);
            }
        }

        // Finalize any remaining blocks
        if let (Some(heading), Some(block)) = (carry_over_heading.as_ref(), current_block.as_ref())
            && !heading.is_empty()
            && !block.is_empty()
        {
            current_block = Some(format!("{} {}", heading, block.trim()));
        }
        if let Some(block) = current_block
            && !block.is_empty()
        {
            text_blocks.push(block.trim().to_string());
        }

        Ok(text_blocks)
    }
}

/// Shared branch of list items and regular paragraphs: merge a carried-over
/// heading, else append to (or start) the current block.
fn append_text(current_block: &mut Option<String>, carry_over_heading: &mut Option<String>, text: &str) {
    if let Some(heading) = carry_over_heading.take().filter(|h| !h.is_empty()) {
        // Merge the carry-over heading with the paragraph
        *current_block = Some(format!("{heading} {text}"));
    } else if let Some(block) = current_block.as_mut().filter(|b| !b.is_empty()) {
        block.push(' ');
        block.push_str(text);
    } else {
        *current_block = Some(text.to_string());
    }
}

/// Paragraph styles of `styles.xml`: id -> UI name, plus the default style.
struct ParagraphStyles {
    names: HashMap<String, String>,
    default_name: Option<String>,
}

impl ParagraphStyles {
    fn load(package: &OoxmlPackage, document_part: &str) -> Self {
        let mut names = HashMap::new();
        let mut default_name = None;

        let styles_part = package
            .relationships(document_part)
            .into_values()
            .find(|r| r.rel_type.ends_with("/styles"))
            .map(|r| r.resolved)
            .unwrap_or_else(|| "word/styles.xml".to_string());

        if let Ok(root) = package.read_xml(&styles_part) {
            for style in root.children_named("w:style") {
                if style.attr("w:type").unwrap_or("paragraph") != "paragraph" {
                    continue;
                }
                let name = style
                    .child("w:name")
                    .and_then(|n| n.attr("w:val"))
                    .map(internal_to_ui_style_name)
                    .unwrap_or_default();
                let is_default = matches!(style.attr("w:default"), Some("1") | Some("true") | Some("on"));
                if is_default && default_name.is_none() {
                    default_name = Some(name.clone());
                }
                if let Some(id) = style.attr("w:styleId") {
                    names.entry(id.to_string()).or_insert(name);
                }
            }
        }

        Self { names, default_name }
    }

    /// python-docx `paragraph.style.name`: the style with this id, else the
    /// document's default paragraph style.
    fn name_for(&self, style_id: Option<&str>) -> String {
        style_id
            .and_then(|id| self.names.get(id).cloned())
            .or_else(|| self.default_name.clone())
            .unwrap_or_default()
    }
}

/// python-docx `BabelFish.internal2ui`: special-case style names.
fn internal_to_ui_style_name(name: &str) -> String {
    match name {
        "caption" => "Caption".to_string(),
        "footer" => "Footer".to_string(),
        "header" => "Header".to_string(),
        n if n.len() == 9 && n.starts_with("heading ") && n.as_bytes()[8].is_ascii_digit() && n.as_bytes()[8] != b'0' => {
            format!("Heading {}", &n[8..])
        }
        other => other.to_string(),
    }
}

/// The main document part named by the package relationships.
pub(crate) fn main_document_part(package: &OoxmlPackage) -> String {
    package
        .relationships("")
        .into_values()
        .find(|r| r.rel_type.ends_with("/officeDocument"))
        .map(|r| r.resolved)
        .unwrap_or_else(|| "word/document.xml".to_string())
}

/// python-docx `CT_P.text`: text of the `w:r` and `w:hyperlink` children.
fn paragraph_text(p: &XmlElement) -> String {
    let mut out = String::new();
    for child in p.elements() {
        match child.name.as_str() {
            "w:r" => out.push_str(&run_text(child)),
            "w:hyperlink" => {
                for r in child.children_named("w:r") {
                    out.push_str(&run_text(r));
                }
            }
            _ => {}
        }
    }
    out
}

/// python-docx `CT_R.text`: `w:br | w:cr | w:noBreakHyphen | w:ptab | w:t | w:tab`.
fn run_text(r: &XmlElement) -> String {
    let mut out = String::new();
    for child in r.elements() {
        match child.name.as_str() {
            "w:t" => {
                for node in &child.children {
                    if let XmlNode::Text(t) = node {
                        out.push_str(t);
                    }
                }
            }
            "w:br" => {
                // A line break is "\n"; column and page breaks produce "".
                if child.attr("w:type").unwrap_or("textWrapping") == "textWrapping" {
                    out.push('\n');
                }
            }
            "w:cr" => out.push('\n'),
            "w:noBreakHyphen" => out.push('-'),
            "w:ptab" | "w:tab" => out.push('\t'),
            _ => {}
        }
    }
    out
}

impl BaseParser for DocxParser {
    fn llm(&self) -> Option<&SharedLLM> {
        self.llm.as_ref()
    }
    fn file_path(&self) -> &str {
        &self.file_path
    }
    fn type_name(&self) -> &'static str {
        "DocxParser"
    }
    fn parse(&self) -> anyhow::Result<String> {
        Ok(self.load_paragraphs()?.join("\n\n"))
    }
}
