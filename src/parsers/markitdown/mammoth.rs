//! Hand-port of `mammoth` 1.11 (`mammoth.convert_to_html`), the DOCX → HTML
//! converter MarkItDown's `DocxConverter` runs before its HTML → Markdown
//! step.
//!
//! The port keeps mammoth's pipeline: the package is read into a document
//! model (`docx/body_xml.py`, `styles_xml.py`, `numbering_xml.py`,
//! `notes_xml.py`, `comments_xml.py`), converted to an HTML node tree through
//! the style map (`conversion.py`, the default style map of `options.py` plus
//! any style map embedded at `mammoth/style-map`), then empty nodes are
//! stripped, collapsible nodes merged and the tree written as HTML
//! (`html/__init__.py`, `writers/html.py`). Warnings (mammoth's `messages`)
//! are not reported since MarkItDown only uses the converted value; errors
//! mammoth raises (a missing required attribute, a dangling relationship id,
//! a missing embedded image) are returned as errors.

use std::collections::{BTreeMap, HashMap};

use base64::Engine as _;
use indexmap::IndexMap;
use once_cell::sync::Lazy;
use regex::Regex;

use super::dingbats;
use super::ooxml::{parse_xml, OoxmlPackage, XmlElement, XmlNode};

// =============================================================================
// XML helpers (`find_child_or_null` chains over optional elements)
// =============================================================================

fn child<'a>(el: Option<&'a XmlElement>, name: &str) -> Option<&'a XmlElement> {
    el.and_then(|e| e.child(name))
}

fn attr<'a>(el: Option<&'a XmlElement>, name: &str) -> Option<&'a str> {
    el.and_then(|e| e.attr(name))
}

/// `element.attributes[name]`: a missing attribute raises `KeyError`.
fn required_attr<'a>(el: &'a XmlElement, name: &str) -> anyhow::Result<&'a str> {
    el.attr(name).ok_or_else(|| anyhow::anyhow!("'{name}'"))
}

/// `_inner_text`: the concatenated text of every text node below `node`.
fn inner_text(el: &XmlElement) -> String {
    el.inner_text()
}

/// `office_xml.read`: parse a part, replacing every `mc:AlternateContent`
/// with the children of its `mc:Fallback`.
pub fn read_office_xml(bytes: &[u8]) -> anyhow::Result<XmlElement> {
    let mut root = parse_xml(bytes)?;
    collapse_alternate_content(&mut root);
    Ok(root)
}

fn collapse_alternate_content(el: &mut XmlElement) {
    let children = std::mem::take(&mut el.children);
    for node in children {
        match node {
            XmlNode::Element(e) if e.name == "mc:AlternateContent" => {
                // The fallback's children are taken as-is (not collapsed again).
                if let Some(fallback) = e.child("mc:Fallback") {
                    el.children.extend(fallback.children.iter().cloned());
                }
            }
            XmlNode::Element(mut e) => {
                collapse_alternate_content(&mut e);
                el.children.push(XmlNode::Element(e));
            }
            text => el.children.push(text),
        }
    }
}

// =============================================================================
// Document model (`documents.py`)
// =============================================================================

/// A numbering level (`_NumberingLevel`): the level index as a string and
/// whether the list is ordered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumberingLevel {
    pub level_index: String,
    pub is_ordered: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerticalAlignment {
    Baseline,
    Superscript,
    Subscript,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakType {
    Line,
    Page,
    Column,
}

/// Where an image's bytes come from.
#[derive(Debug, Clone, PartialEq)]
pub enum ImageSource {
    /// A zip entry of the package.
    Embedded(String),
    /// An external file (`r:link`), which mammoth refuses to open.
    Linked(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Paragraph {
    pub children: Vec<DocElement>,
    pub style_id: Option<String>,
    pub style_name: Option<String>,
    pub numbering: Option<NumberingLevel>,
    pub alignment: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    pub children: Vec<DocElement>,
    pub style_id: Option<String>,
    pub style_name: Option<String>,
    pub is_bold: bool,
    pub is_italic: bool,
    pub is_underline: bool,
    pub is_strikethrough: bool,
    pub is_all_caps: bool,
    pub is_small_caps: bool,
    pub vertical_alignment: VerticalAlignment,
    pub highlight: Option<String>,
}

/// A document element.
#[derive(Debug, Clone, PartialEq)]
pub enum DocElement {
    Paragraph(Paragraph),
    Run(Run),
    Text(String),
    Hyperlink { children: Vec<DocElement>, href: Option<String>, anchor: Option<String>, target_frame: Option<String> },
    Checkbox { checked: bool },
    Table { children: Vec<DocElement>, style_id: Option<String>, style_name: Option<String> },
    TableRow { children: Vec<DocElement>, is_header: bool },
    TableCell { children: Vec<DocElement>, colspan: i64, rowspan: i64 },
    TableCellUnmerged { children: Vec<DocElement>, colspan: i64, rowspan: i64, vmerge: bool },
    Break(BreakType),
    Tab,
    Image { alt_text: Option<String>, content_type: Option<String>, source: ImageSource },
    Bookmark { name: Option<String> },
    NoteReference { note_type: String, note_id: String },
    CommentReference { comment_id: String },
}

impl DocElement {
    fn children_mut(&mut self) -> Option<&mut Vec<DocElement>> {
        match self {
            DocElement::Paragraph(p) => Some(&mut p.children),
            DocElement::Run(r) => Some(&mut r.children),
            DocElement::Hyperlink { children, .. }
            | DocElement::Table { children, .. }
            | DocElement::TableRow { children, .. }
            | DocElement::TableCell { children, .. }
            | DocElement::TableCellUnmerged { children, .. } => Some(children),
            _ => None,
        }
    }
}

/// `transforms.element_of_type(...)`: transform every matching element,
/// children first (depth-first, document order).
fn transform_elements(element: DocElement, f: &mut dyn FnMut(DocElement) -> DocElement) -> DocElement {
    let mut element = element;
    if let Some(children) = element.children_mut() {
        let taken = std::mem::take(children);
        *children = taken.into_iter().map(|c| transform_elements(c, f)).collect();
    }
    f(element)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Note {
    pub note_type: String,
    pub note_id: String,
    pub body: Vec<DocElement>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Comment {
    pub comment_id: String,
    pub body: Vec<DocElement>,
    pub author_name: Option<String>,
    pub author_initials: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    pub children: Vec<DocElement>,
    pub notes: Vec<Note>,
    pub comments: Vec<Comment>,
}

// =============================================================================
// Styles, numbering, relationships, content types
// =============================================================================

#[derive(Debug, Clone, Default)]
struct Styles {
    paragraph: HashMap<String, Option<String>>,
    character: HashMap<String, Option<String>>,
    table: HashMap<String, Option<String>>,
    /// numbering style id -> numId
    numbering: HashMap<String, Option<String>>,
}

impl Styles {
    fn read(element: &XmlElement) -> anyhow::Result<Self> {
        let mut styles = Styles::default();
        for style_element in element.children_named("w:style") {
            let element_type = required_attr(style_element, "w:type")?;
            let style_id = required_attr(style_element, "w:styleId")?.to_string();
            let set = match element_type {
                "paragraph" => &mut styles.paragraph,
                "character" => &mut styles.character,
                "table" => &mut styles.table,
                "numbering" => &mut styles.numbering,
                _ => continue,
            };
            // Per 17.7.4.17 style (Style Definition) of ECMA-376 4th edition
            // Part 1, the first definition of a style id wins; later ones are
            // reassigned, so there is no point holding onto them.
            if set.contains_key(&style_id) {
                continue;
            }
            let value = if element_type == "numbering" {
                attr(
                    child(child(child(Some(style_element), "w:pPr"), "w:numPr"), "w:numId"),
                    "w:val",
                )
                .map(str::to_string)
            } else {
                attr(child(Some(style_element), "w:name"), "w:val").map(str::to_string)
            };
            set.insert(style_id, value);
        }
        Ok(styles)
    }
}

#[derive(Debug, Clone)]
struct AbstractNumLevel {
    level_index: String,
    is_ordered: bool,
    paragraph_style_id: Option<String>,
}

#[derive(Debug, Clone)]
struct AbstractNum {
    /// Levels by index, in dict insertion order.
    levels: IndexMap<String, AbstractNumLevel>,
    num_style_link: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct Numbering {
    abstract_nums: IndexMap<Option<String>, AbstractNum>,
    /// numId -> abstractNumId
    nums: HashMap<Option<String>, String>,
    levels_by_paragraph_style_id: HashMap<String, NumberingLevel>,
    styles: Styles,
}

impl Numbering {
    fn read(element: &XmlElement, styles: &Styles) -> anyhow::Result<Self> {
        let mut abstract_nums: IndexMap<Option<String>, AbstractNum> = IndexMap::new();
        for abstract_num in element.children_named("w:abstractNum") {
            let id = abstract_num.attr("w:abstractNumId").map(str::to_string);
            let mut levels: IndexMap<String, AbstractNumLevel> = IndexMap::new();
            // Some malformed documents define numbering levels without an
            // index, and reference the numbering using a w:numPr element
            // without a w:ilvl child. To handle such cases, we assume a level
            // of 0 as a fallback.
            let mut level_without_index: Option<AbstractNumLevel> = None;
            for level_element in abstract_num.children_named("w:lvl") {
                let num_fmt = attr(child(Some(level_element), "w:numFmt"), "w:val");
                let level = AbstractNumLevel {
                    level_index: level_element.attr("w:ilvl").unwrap_or("0").to_string(),
                    is_ordered: num_fmt != Some("bullet"),
                    paragraph_style_id: attr(child(Some(level_element), "w:pStyle"), "w:val").map(str::to_string),
                };
                if level_element.attr("w:ilvl").is_none() {
                    level_without_index = Some(level);
                } else {
                    levels.insert(level.level_index.clone(), level);
                }
            }
            if let Some(level) = level_without_index
                && !levels.contains_key(&level.level_index)
            {
                levels.insert(level.level_index.clone(), level);
            }
            let num_style_link = attr(child(Some(abstract_num), "w:numStyleLink"), "w:val").map(str::to_string);
            abstract_nums.insert(id, AbstractNum { levels, num_style_link });
        }

        // `_levels_by_paragraph_style_id`, built in dict order (last wins).
        let mut levels_by_paragraph_style_id = HashMap::new();
        for abstract_num in abstract_nums.values() {
            for level in abstract_num.levels.values() {
                if let Some(style_id) = &level.paragraph_style_id {
                    levels_by_paragraph_style_id.insert(
                        style_id.clone(),
                        NumberingLevel { level_index: level.level_index.clone(), is_ordered: level.is_ordered },
                    );
                }
            }
        }

        let mut nums = HashMap::new();
        for num in element.children_named("w:num") {
            let num_id = num.attr("w:numId").map(str::to_string);
            let abstract_num_id = required_attr_opt(child(Some(num), "w:abstractNumId"), "w:val")?;
            nums.insert(num_id, abstract_num_id.to_string());
        }

        Ok(Numbering { abstract_nums, nums, levels_by_paragraph_style_id, styles: styles.clone() })
    }

    fn find_level(&self, num_id: Option<&str>, level: &str, depth: usize) -> anyhow::Result<Option<NumberingLevel>> {
        if depth > 900 {
            anyhow::bail!("maximum recursion depth exceeded");
        }
        let Some(abstract_num_id) = self.nums.get(&num_id.map(str::to_string)) else {
            return Ok(None);
        };
        let Some(abstract_num) = self.abstract_nums.get(&Some(abstract_num_id.clone())) else {
            return Ok(None);
        };
        match &abstract_num.num_style_link {
            None => Ok(abstract_num
                .levels
                .get(level)
                .map(|l| NumberingLevel { level_index: l.level_index.clone(), is_ordered: l.is_ordered })),
            Some(link) => {
                let Some(style_num_id) = self.styles.numbering.get(link) else {
                    anyhow::bail!("'NoneType' object has no attribute 'num_id'");
                };
                self.find_level(style_num_id.as_deref(), level, depth + 1)
            }
        }
    }

    fn find_level_by_paragraph_style_id(&self, style_id: &str) -> Option<NumberingLevel> {
        self.levels_by_paragraph_style_id.get(style_id).cloned()
    }
}

/// `find_child_or_null(...).attributes[name]`: `KeyError` when the element or
/// the attribute is missing.
fn required_attr_opt<'a>(el: Option<&'a XmlElement>, name: &str) -> anyhow::Result<&'a str> {
    attr(el, name).ok_or_else(|| anyhow::anyhow!("'{name}'"))
}

#[derive(Debug, Clone, Default)]
struct Relationships {
    targets_by_id: HashMap<String, String>,
    targets_by_type: Vec<(String, String)>,
}

impl Relationships {
    fn read(element: &XmlElement) -> anyhow::Result<Self> {
        let mut rels = Relationships::default();
        for rel in element.children_named("relationships:Relationship") {
            let id = required_attr(rel, "Id")?.to_string();
            let target = required_attr(rel, "Target")?.to_string();
            let rel_type = required_attr(rel, "Type")?.to_string();
            rels.targets_by_id.insert(id, target.clone());
            rels.targets_by_type.push((rel_type, target));
        }
        Ok(rels)
    }

    fn find_target_by_relationship_id(&self, key: &str) -> anyhow::Result<String> {
        self.targets_by_id.get(key).cloned().ok_or_else(|| anyhow::anyhow!("'{key}'"))
    }

    fn find_targets_by_type(&self, relationship_type: &str) -> Vec<String> {
        self.targets_by_type.iter().filter(|(t, _)| t == relationship_type).map(|(_, target)| target.clone()).collect()
    }
}

#[derive(Debug, Clone, Default)]
struct ContentTypes {
    extension_defaults: HashMap<String, String>,
    overrides: HashMap<String, String>,
}

impl ContentTypes {
    fn read(element: &XmlElement) -> anyhow::Result<Self> {
        let mut types = ContentTypes::default();
        for default in element.children_named("content-types:Default") {
            types
                .extension_defaults
                .insert(required_attr(default, "Extension")?.to_string(), required_attr(default, "ContentType")?.to_string());
        }
        for override_ in element.children_named("content-types:Override") {
            let part_name = required_attr(override_, "PartName")?.trim_start_matches('/').to_string();
            types.overrides.insert(part_name, required_attr(override_, "ContentType")?.to_string());
        }
        Ok(types)
    }

    fn find_content_type(&self, path: &str) -> Option<String> {
        if let Some(ct) = self.overrides.get(path) {
            return Some(ct.clone());
        }
        let extension = path.rsplit_once('.').map(|(_, e)| e).unwrap_or(path);
        if let Some(ct) = self.extension_defaults.get(extension) {
            return Some(ct.clone());
        }
        let image_type = match extension.to_lowercase().as_str() {
            "png" => "png",
            "gif" => "gif",
            "jpeg" | "jpg" => "jpeg",
            "tif" | "tiff" => "tiff",
            "bmp" => "bmp",
            _ => return None,
        };
        Some(format!("image/{image_type}"))
    }
}

// =============================================================================
// Package access
// =============================================================================

/// A DOCX package as mammoth reads it: zip entries addressed by exact name,
/// with optional pre-parsed replacements for some parts (MarkItDown's math
/// pre-processing).
pub struct DocxSource<'a> {
    pub package: &'a OoxmlPackage,
    pub parsed_overrides: HashMap<String, XmlElement>,
}

impl DocxSource<'_> {
    fn exists(&self, name: &str) -> bool {
        self.package.read_exact(name).is_some()
    }

    fn read_entry(&self, name: &str) -> anyhow::Result<XmlElement> {
        if let Some(parsed) = self.parsed_overrides.get(name) {
            let mut parsed = parsed.clone();
            collapse_alternate_content(&mut parsed);
            return Ok(parsed);
        }
        let bytes = self
            .package
            .read_exact(name)
            .ok_or_else(|| anyhow::anyhow!("There is no item named '{name}' in the archive"))?;
        read_office_xml(bytes)
    }

    fn try_read_entry(&self, name: &str) -> anyhow::Result<Option<XmlElement>> {
        if self.exists(name) { Ok(Some(self.read_entry(name)?)) } else { Ok(None) }
    }

    fn read_relationships(&self, name: &str) -> anyhow::Result<Relationships> {
        match self.try_read_entry(name)? {
            Some(root) => Relationships::read(&root),
            None => Ok(Relationships::default()),
        }
    }
}

/// `zips.split_path`.
fn split_path(path: &str) -> (&str, &str) {
    match path.rsplit_once('/') {
        Some((dir, base)) => (dir, base),
        None => ("", path),
    }
}

/// `zips.join_path`: join non-empty segments; an absolute segment resets.
fn join_path(parts: &[&str]) -> String {
    let mut relevant: Vec<&str> = Vec::new();
    for path in parts.iter().filter(|p| !p.is_empty()) {
        if path.starts_with('/') {
            relevant = vec![path];
        } else {
            relevant.push(path);
        }
    }
    relevant.join("/")
}

fn relationships_path_for(name: &str) -> String {
    let (dirname, basename) = split_path(name);
    join_path(&[dirname, "_rels", &format!("{basename}.rels")])
}

struct PartPaths {
    main_document: String,
    comments: String,
    endnotes: String,
    footnotes: String,
    numbering: String,
    styles: String,
}

fn find_part_path(
    source: &DocxSource<'_>,
    relationships: &Relationships,
    relationship_type: &str,
    base_path: &str,
    fallback_path: &str,
) -> String {
    relationships
        .find_targets_by_type(relationship_type)
        .iter()
        .map(|target| join_path(&[base_path, target]).trim_start_matches('/').to_string())
        .find(|target| source.exists(target))
        .unwrap_or_else(|| fallback_path.to_string())
}

fn find_part_paths(source: &DocxSource<'_>) -> anyhow::Result<PartPaths> {
    let package_relationships = source.read_relationships("_rels/.rels")?;
    let document_filename = find_part_path(
        source,
        &package_relationships,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument",
        "",
        "word/document.xml",
    );
    if !source.exists(&document_filename) {
        anyhow::bail!("Could not find main document part. Are you sure this is a valid .docx file?");
    }

    let document_relationships = source.read_relationships(&relationships_path_for(&document_filename))?;
    let base_path = split_path(&document_filename).0.to_string();
    let find = |name: &str| {
        find_part_path(
            source,
            &document_relationships,
            &format!("http://schemas.openxmlformats.org/officeDocument/2006/relationships/{name}"),
            &base_path,
            &format!("word/{name}.xml"),
        )
    };

    Ok(PartPaths {
        comments: find("comments"),
        endnotes: find("endnotes"),
        footnotes: find("footnotes"),
        numbering: find("numbering"),
        styles: find("styles"),
        main_document: document_filename,
    })
}

// =============================================================================
// Body reader (`docx/body_xml.py`)
// =============================================================================

/// `_ReadResult` without messages: converted elements plus "extra" elements
/// hoisted out of `w:pict` to follow the enclosing paragraph.
#[derive(Debug, Default)]
struct ReadResult {
    elements: Vec<DocElement>,
    extra: Vec<DocElement>,
}

impl ReadResult {
    fn success(element: DocElement) -> Self {
        Self { elements: vec![element], extra: Vec::new() }
    }

    fn empty() -> Self {
        Self::default()
    }

    fn concat(results: Vec<ReadResult>) -> Self {
        let mut out = ReadResult::default();
        for r in results {
            out.elements.extend(r.elements);
            out.extra.extend(r.extra);
        }
        out
    }

    fn into_extra(self) -> Self {
        let mut extra = self.extra;
        extra.extend(self.elements);
        Self { elements: Vec::new(), extra }
    }

    fn append_extra(self) -> Self {
        let mut elements = self.elements;
        elements.extend(self.extra);
        Self { elements, extra: Vec::new() }
    }
}

/// A complex field on the field stack.
#[derive(Debug, Clone)]
enum ComplexField {
    Begin { fld_char: XmlElement },
    Hyperlink { href: Option<String>, anchor: Option<String> },
    Checkbox { checked: bool },
    /// An unrecognised field instruction (Python `None`).
    Unknown,
}

static RE_EXTERNAL_LINK: Lazy<Regex> = Lazy::new(|| Regex::new(r#"^\s*HYPERLINK "(.*)""#).expect("valid regex"));
static RE_INTERNAL_LINK: Lazy<Regex> = Lazy::new(|| Regex::new(r#"^\s*HYPERLINK\s+\\l\s+"(.*)""#).expect("valid regex"));
static RE_FORMCHECKBOX: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\s*FORMCHECKBOX\s*").expect("valid regex"));
static RE_SYMBOL_F0: Lazy<Regex> = Lazy::new(|| Regex::new(r"^F0..").expect("valid regex"));

/// Elements ignored without a warning.
const IGNORED_ELEMENTS: &[&str] = &[
    "office-word:wrap",
    "v:shadow",
    "v:shapetype",
    "w:annotationRef",
    "w:bookmarkEnd",
    "w:sectPr",
    "w:proofErr",
    "w:lastRenderedPageBreak",
    "w:commentRangeStart",
    "w:commentRangeEnd",
    "w:del",
    "w:footnoteRef",
    "w:endnoteRef",
    "w:pPr",
    "w:rPr",
    "w:tblPr",
    "w:tblGrid",
    "w:trPr",
    "w:tcPr",
];

struct BodyReader<'a> {
    numbering: &'a Numbering,
    content_types: &'a ContentTypes,
    relationships: Relationships,
    styles: &'a Styles,
    current_instr_text: Vec<String>,
    complex_field_stack: Vec<ComplexField>,
    /// When a paragraph is marked as deleted, its contents should be combined
    /// with the following paragraph. See 17.13.5.15 del (Deleted Paragraph)
    /// of ECMA-376 4th edition Part 1.
    deleted_paragraph_contents: Vec<XmlNode>,
}

fn read_boolean_attribute_value(value: Option<&str>) -> bool {
    !matches!(value, Some("false") | Some("0"))
}

fn read_boolean_element(element: Option<&XmlElement>) -> bool {
    match element {
        None => false,
        Some(e) => read_boolean_attribute_value(e.attr("w:val")),
    }
}

fn read_underline_element(element: Option<&XmlElement>) -> bool {
    match element {
        None => false,
        Some(e) => !matches!(e.attr("w:val"), None | Some("false") | Some("0") | Some("none")),
    }
}

fn read_highlight_value(value: Option<&str>) -> Option<String> {
    match value {
        None | Some("") | Some("none") => None,
        Some(v) => Some(v.to_string()),
    }
}

/// `uris.uri_to_zip_entry_name`.
fn uri_to_zip_entry_name(base: &str, uri: &str) -> String {
    match uri.strip_prefix('/') {
        Some(rest) => rest.to_string(),
        None => format!("{base}/{uri}"),
    }
}

/// `uris.replace_fragment`.
fn replace_fragment(uri: &str, fragment: &str) -> String {
    let base = match uri.find('#') {
        Some(i) => &uri[..i],
        None => uri,
    };
    format!("{base}#{fragment}")
}

/// Python `int(value, 16)` for the strings that occur in `w:char`.
fn parse_hex(value: &str) -> anyhow::Result<u32> {
    let trimmed = value.trim();
    let digits = trimmed.strip_prefix("0x").or_else(|| trimmed.strip_prefix("0X")).unwrap_or(trimmed);
    u32::from_str_radix(&digits.replace('_', ""), 16)
        .map_err(|_| anyhow::anyhow!("invalid literal for int() with base 16: '{value}'"))
}

impl<'a> BodyReader<'a> {
    fn new(
        numbering: &'a Numbering,
        content_types: &'a ContentTypes,
        relationships: Relationships,
        styles: &'a Styles,
    ) -> Self {
        Self {
            numbering,
            content_types,
            relationships,
            styles,
            current_instr_text: Vec::new(),
            complex_field_stack: Vec::new(),
            deleted_paragraph_contents: Vec::new(),
        }
    }

    /// `_BodyReader.read_all`: the converted elements (extras are dropped).
    fn read_all(&mut self, nodes: &[XmlNode]) -> anyhow::Result<Vec<DocElement>> {
        Ok(self.read_xml_elements(nodes)?.elements)
    }

    fn read_xml_elements(&mut self, nodes: &[XmlNode]) -> anyhow::Result<ReadResult> {
        let mut results = Vec::new();
        for node in nodes {
            if let XmlNode::Element(element) = node {
                results.push(self.read(element)?);
            }
        }
        Ok(ReadResult::concat(results))
    }

    fn read_child_elements(&mut self, element: Option<&XmlElement>) -> anyhow::Result<ReadResult> {
        match element {
            Some(e) => self.read_xml_elements(&e.children),
            None => Ok(ReadResult::empty()),
        }
    }

    fn read(&mut self, element: &XmlElement) -> anyhow::Result<ReadResult> {
        match element.name.as_str() {
            "w:t" => Ok(ReadResult::success(DocElement::Text(inner_text(element)))),
            "w:r" => self.run(element),
            "w:p" => self.paragraph(element),
            "w:fldChar" => self.read_fld_char(element),
            "w:instrText" => {
                self.current_instr_text.push(inner_text(element));
                Ok(ReadResult::empty())
            }
            "w:tab" => Ok(ReadResult::success(DocElement::Tab)),
            "w:noBreakHyphen" => Ok(ReadResult::success(DocElement::Text("\u{2011}".to_string()))),
            "w:softHyphen" => Ok(ReadResult::success(DocElement::Text("\u{00ad}".to_string()))),
            "w:sym" => self.symbol(element),
            "w:tbl" => self.table(element),
            "w:tr" => self.table_row(element),
            "w:tc" => self.table_cell(element),
            "w:ins" | "w:object" | "w:smartTag" | "w:drawing" | "v:group" | "v:rect" | "v:roundrect" | "v:shape"
            | "v:textbox" | "w:txbxContent" => self.read_child_elements(Some(element)),
            "w:pict" => Ok(self.read_child_elements(Some(element))?.into_extra()),
            "w:hyperlink" => self.hyperlink(element),
            "w:bookmarkStart" => {
                let name = element.attr("w:name");
                if name == Some("_GoBack") {
                    Ok(ReadResult::empty())
                } else {
                    Ok(ReadResult::success(DocElement::Bookmark { name: name.map(str::to_string) }))
                }
            }
            "w:br" => Ok(match element.attr("w:type") {
                None | Some("") | Some("textWrapping") => ReadResult::success(DocElement::Break(BreakType::Line)),
                Some("page") => ReadResult::success(DocElement::Break(BreakType::Page)),
                Some("column") => ReadResult::success(DocElement::Break(BreakType::Column)),
                // Unsupported break type: ignored with a warning.
                Some(_) => ReadResult::empty(),
            }),
            "wp:inline" | "wp:anchor" => self.inline(element),
            "v:imagedata" => match element.attr("r:id") {
                // A v:imagedata element without a relationship ID is ignored.
                None => Ok(ReadResult::empty()),
                Some(relationship_id) => {
                    // mammoth reads `o:title`, but its parser does not map the
                    // `o` namespace, so the lookup never finds the attribute.
                    let source = self.find_embedded_image(relationship_id)?;
                    Ok(self.read_image(source, None))
                }
            },
            "w:footnoteReference" => Ok(ReadResult::success(DocElement::NoteReference {
                note_type: "footnote".to_string(),
                note_id: required_attr(element, "w:id")?.to_string(),
            })),
            "w:endnoteReference" => Ok(ReadResult::success(DocElement::NoteReference {
                note_type: "endnote".to_string(),
                note_id: required_attr(element, "w:id")?.to_string(),
            })),
            "w:commentReference" => Ok(ReadResult::success(DocElement::CommentReference {
                comment_id: required_attr(element, "w:id")?.to_string(),
            })),
            "mc:AlternateContent" => self.read_child_elements(element.child("mc:Fallback")),
            "w:sdt" => self.read_sdt(element),
            // Unrecognised elements are ignored (with a warning unless listed
            // in IGNORED_ELEMENTS).
            name => {
                if !IGNORED_ELEMENTS.contains(&name) {
                    tracing::trace!(target: "mammoth", "An unrecognised element was ignored: {}", name);
                }
                Ok(ReadResult::empty())
            }
        }
    }

    /// `_read_style`: (style id, style name) from `<properties>/<tag w:val>`.
    fn read_style(
        &self,
        properties: Option<&XmlElement>,
        style_tag_name: &str,
        find: impl Fn(&Styles, &str) -> Option<Option<String>>,
    ) -> (Option<String>, Option<String>) {
        let style_id = attr(child(properties, style_tag_name), "w:val").map(str::to_string);
        let style_name = match &style_id {
            None => None,
            // An undefined style is reported as a warning only.
            Some(id) => find(self.styles, id).flatten(),
        };
        (style_id, style_name)
    }

    fn current_hyperlink(&self) -> Option<(Option<String>, Option<String>)> {
        self.complex_field_stack.iter().rev().find_map(|field| match field {
            ComplexField::Hyperlink { href, anchor } => Some((href.clone(), anchor.clone())),
            _ => None,
        })
    }

    fn run(&mut self, element: &XmlElement) -> anyhow::Result<ReadResult> {
        let properties = element.child("w:rPr");
        let vertical_alignment = attr(child(properties, "w:vertAlign"), "w:val");
        let is_bold = read_boolean_element(child(properties, "w:b"));
        let is_italic = read_boolean_element(child(properties, "w:i"));
        let is_underline = read_underline_element(child(properties, "w:u"));
        let is_strikethrough = read_boolean_element(child(properties, "w:strike"));
        let is_all_caps = read_boolean_element(child(properties, "w:caps"));
        let is_small_caps = read_boolean_element(child(properties, "w:smallCaps"));
        let highlight = read_highlight_value(attr(child(properties, "w:highlight"), "w:val"));
        let (style_id, style_name) = self.read_style(properties, "w:rStyle", |s, id| s.character.get(id).cloned());

        let children = self.read_xml_elements(&element.children)?;
        // The complex-field hyperlink is looked up after the run's children
        // were read, so a field separator inside this run already applies.
        let ReadResult { elements, extra } = children;
        let elements = match self.current_hyperlink() {
            None => elements,
            Some((href, anchor)) => {
                vec![DocElement::Hyperlink { children: elements, href, anchor, target_frame: None }]
            }
        };

        let vertical_alignment = match vertical_alignment {
            Some("superscript") => VerticalAlignment::Superscript,
            Some("subscript") => VerticalAlignment::Subscript,
            _ => VerticalAlignment::Baseline,
        };
        Ok(ReadResult {
            elements: vec![DocElement::Run(Run {
                children: elements,
                style_id,
                style_name,
                is_bold,
                is_italic,
                is_underline,
                is_strikethrough,
                is_all_caps,
                is_small_caps,
                vertical_alignment,
                highlight,
            })],
            extra,
        })
    }

    fn paragraph(&mut self, element: &XmlElement) -> anyhow::Result<ReadResult> {
        let properties = element.child("w:pPr");
        let is_deleted = child(child(properties, "w:rPr"), "w:del").is_some();

        if is_deleted {
            self.deleted_paragraph_contents.extend(element.children.iter().cloned());
            return Ok(ReadResult::empty());
        }

        let alignment = attr(child(properties, "w:jc"), "w:val").map(str::to_string);
        let mut children_xml: Vec<XmlNode> = std::mem::take(&mut self.deleted_paragraph_contents);
        children_xml.extend(element.children.iter().cloned());

        let (style_id, style_name) = self.read_style(properties, "w:pStyle", |s, id| s.paragraph.get(id).cloned());
        let children = self.read_xml_elements(&children_xml)?;
        let numbering = self.read_numbering_properties(style_id.as_deref(), child(properties, "w:numPr"))?;

        Ok(ReadResult {
            elements: vec![DocElement::Paragraph(Paragraph {
                children: children.elements,
                style_id,
                style_name,
                numbering,
                alignment,
            })],
            extra: children.extra,
        }
        .append_extra())
    }

    fn read_numbering_properties(
        &self,
        paragraph_style_id: Option<&str>,
        element: Option<&XmlElement>,
    ) -> anyhow::Result<Option<NumberingLevel>> {
        let num_id = attr(child(element, "w:numId"), "w:val");
        let level_index = attr(child(element, "w:ilvl"), "w:val");
        if let (Some(num_id), Some(level_index)) = (num_id, level_index) {
            return self.numbering.find_level(Some(num_id), level_index, 0);
        }

        if let Some(style_id) = paragraph_style_id
            && let Some(level) = self.numbering.find_level_by_paragraph_style_id(style_id)
        {
            return Ok(Some(level));
        }

        // Some malformed documents define numbering levels without an index,
        // and reference the numbering using a w:numPr element without a
        // w:ilvl child. To handle such cases, we assume a level of 0 as a
        // fallback.
        if let Some(num_id) = num_id {
            return self.numbering.find_level(Some(num_id), "0", 0);
        }

        Ok(None)
    }

    fn read_fld_char(&mut self, element: &XmlElement) -> anyhow::Result<ReadResult> {
        match element.attr("w:fldCharType") {
            Some("begin") => {
                self.complex_field_stack.push(ComplexField::Begin { fld_char: element.clone() });
                self.current_instr_text.clear();
            }
            Some("end") => {
                let Some(mut complex_field) = self.complex_field_stack.pop() else {
                    anyhow::bail!("pop from empty list");
                };
                if matches!(complex_field, ComplexField::Begin { .. }) {
                    complex_field = self.parse_current_instr_text(&complex_field);
                }
                if let ComplexField::Checkbox { checked } = complex_field {
                    return Ok(ReadResult::success(DocElement::Checkbox { checked }));
                }
            }
            Some("separate") => {
                let Some(complex_field_separate) = self.complex_field_stack.pop() else {
                    anyhow::bail!("pop from empty list");
                };
                let complex_field = self.parse_current_instr_text(&complex_field_separate);
                self.complex_field_stack.push(complex_field);
            }
            _ => {}
        }
        Ok(ReadResult::empty())
    }

    fn parse_current_instr_text(&self, complex_field: &ComplexField) -> ComplexField {
        let instr_text = self.current_instr_text.concat();
        let fld_char = match complex_field {
            ComplexField::Begin { fld_char } => Some(fld_char),
            _ => None,
        };
        parse_instr_text(&instr_text, fld_char)
    }

    fn symbol(&mut self, element: &XmlElement) -> anyhow::Result<ReadResult> {
        // See 17.3.3.30 sym (Symbol Character) of ECMA-376 4th edition Part 1
        let font = element.attr("w:font");
        let Some(char_value) = element.attr("w:char") else {
            anyhow::bail!("int() can't convert non-string with explicit base");
        };
        let code = parse_hex(char_value)?;
        let mut unicode_code_point = font.and_then(|f| dingbats::lookup(f, code));

        if unicode_code_point.is_none() && RE_SYMBOL_F0.is_match(char_value) {
            let code = parse_hex(&char_value[2..])?;
            unicode_code_point = font.and_then(|f| dingbats::lookup(f, code));
        }

        match unicode_code_point.and_then(char::from_u32) {
            // A w:sym element with an unsupported character is ignored.
            None => Ok(ReadResult::empty()),
            Some(c) => Ok(ReadResult::success(DocElement::Text(c.to_string()))),
        }
    }

    fn table(&mut self, element: &XmlElement) -> anyhow::Result<ReadResult> {
        let properties = element.child("w:tblPr");
        let (style_id, style_name) = self.read_style(properties, "w:tblStyle", |s, id| s.table.get(id).cloned());
        let children = self.read_xml_elements(&element.children)?;
        let rows = calculate_row_spans(children.elements);
        Ok(ReadResult {
            elements: vec![DocElement::Table { children: rows, style_id, style_name }],
            extra: children.extra,
        })
    }

    fn table_row(&mut self, element: &XmlElement) -> anyhow::Result<ReadResult> {
        let properties = element.child("w:trPr");

        // See 17.13.5.12 del (Deleted Table Row) of ECMA-376 4th edition Part 1
        if child(properties, "w:del").is_some() {
            return Ok(ReadResult::empty());
        }

        let is_header = child(properties, "w:tblHeader").is_some();
        let children = self.read_xml_elements(&element.children)?;
        Ok(ReadResult {
            elements: vec![DocElement::TableRow { children: children.elements, is_header }],
            extra: children.extra,
        })
    }

    fn table_cell(&mut self, element: &XmlElement) -> anyhow::Result<ReadResult> {
        let properties = element.child("w:tcPr");
        let colspan = match attr(child(properties, "w:gridSpan"), "w:val") {
            None => 1,
            Some(v) => v
                .trim()
                .parse::<i64>()
                .map_err(|_| anyhow::anyhow!("invalid literal for int() with base 10: '{v}'"))?,
        };
        let vmerge = match child(properties, "w:vMerge") {
            None => false,
            Some(v) => matches!(v.attr("w:val"), None | Some("") | Some("continue")),
        };
        let children = self.read_xml_elements(&element.children)?;
        Ok(ReadResult {
            elements: vec![DocElement::TableCellUnmerged { children: children.elements, colspan, rowspan: 1, vmerge }],
            extra: children.extra,
        })
    }

    fn hyperlink(&mut self, element: &XmlElement) -> anyhow::Result<ReadResult> {
        let relationship_id = element.attr("r:id");
        let anchor = element.attr("w:anchor");
        let target_frame = element.attr("w:tgtFrame").filter(|t| !t.is_empty()).map(str::to_string);
        let children = self.read_xml_elements(&element.children)?;

        let (href, anchor) = if let Some(relationship_id) = relationship_id {
            let mut href = self.relationships.find_target_by_relationship_id(relationship_id)?;
            if let Some(anchor) = anchor {
                href = replace_fragment(&href, anchor);
            }
            (Some(href), None)
        } else if let Some(anchor) = anchor {
            (None, Some(anchor.to_string()))
        } else {
            return Ok(children);
        };

        Ok(ReadResult {
            elements: vec![DocElement::Hyperlink { children: children.elements, href, anchor, target_frame }],
            extra: children.extra,
        })
    }

    fn inline(&mut self, element: &XmlElement) -> anyhow::Result<ReadResult> {
        let properties = element.child("wp:docPr");
        let descr = attr(properties, "descr");
        let alt_text = if descr.unwrap_or("").trim().is_empty() {
            attr(properties, "title").map(str::to_string)
        } else {
            descr.map(str::to_string)
        };

        let mut results = Vec::new();
        for graphic in element.children_named("a:graphic") {
            for graphic_data in graphic.children_named("a:graphicData") {
                for pic in graphic_data.children_named("pic:pic") {
                    for blip_fill in pic.children_named("pic:blipFill") {
                        for blip in blip_fill.children_named("a:blip") {
                            results.push(self.read_blip(blip, alt_text.clone())?);
                        }
                    }
                }
            }
        }
        Ok(ReadResult::concat(results))
    }

    fn read_blip(&self, element: &XmlElement, alt_text: Option<String>) -> anyhow::Result<ReadResult> {
        let source = if let Some(embed) = element.attr("r:embed") {
            self.find_embedded_image(embed)?
        } else if let Some(link) = element.attr("r:link") {
            ImageSource::Linked(self.relationships.find_target_by_relationship_id(link)?)
        } else {
            // Could not find image file for a:blip element.
            return Ok(ReadResult::empty());
        };
        Ok(self.read_image(source, alt_text))
    }

    fn find_embedded_image(&self, relationship_id: &str) -> anyhow::Result<ImageSource> {
        let target = self.relationships.find_target_by_relationship_id(relationship_id)?;
        Ok(ImageSource::Embedded(uri_to_zip_entry_name("word", &target)))
    }

    fn read_image(&self, source: ImageSource, alt_text: Option<String>) -> ReadResult {
        let image_path = match &source {
            ImageSource::Embedded(p) | ImageSource::Linked(p) => p.clone(),
        };
        let content_type = self.content_types.find_content_type(&image_path);
        ReadResult::success(DocElement::Image { alt_text, content_type, source })
    }

    fn read_sdt(&mut self, element: &XmlElement) -> anyhow::Result<ReadResult> {
        let content = self.read_child_elements(element.child("w:sdtContent"))?;

        // From the WordML standard: a CT_SdtCheckbox element specifies that
        // the parent structured document tag is a checkbox when displayed in
        // the document. The parent structured document tag contents MUST
        // contain a single character and optionally an additional character
        // in a deleted run.
        let Some(checkbox) = child(element.child("w:sdtPr"), "wordml:checkbox") else {
            return Ok(content);
        };

        let checked_element = checkbox.child("wordml:checked");
        let is_checked = checked_element.is_some_and(|c| read_boolean_attribute_value(c.attr("wordml:val")));
        let document_checkbox = DocElement::Checkbox { checked: is_checked };

        let mut has_checkbox = false;
        let replaced: Vec<DocElement> = content
            .elements
            .into_iter()
            .map(|e| {
                transform_elements(e, &mut |el| match el {
                    DocElement::Text(ref value) if !value.is_empty() && !has_checkbox => {
                        has_checkbox = true;
                        document_checkbox.clone()
                    }
                    other => other,
                })
            })
            .collect();

        let elements = if has_checkbox { replaced } else { vec![document_checkbox] };
        Ok(ReadResult { elements, extra: content.extra })
    }
}

fn parse_instr_text(instr_text: &str, fld_char: Option<&XmlElement>) -> ComplexField {
    if let Some(caps) = RE_EXTERNAL_LINK.captures(instr_text) {
        return ComplexField::Hyperlink { href: Some(caps[1].to_string()), anchor: None };
    }

    if let Some(caps) = RE_INTERNAL_LINK.captures(instr_text) {
        return ComplexField::Hyperlink { href: None, anchor: Some(caps[1].to_string()) };
    }

    if RE_FORMCHECKBOX.is_match(instr_text) {
        let checkbox_element = child(child(fld_char, "w:ffData"), "w:checkBox");
        let checked = match child(checkbox_element, "w:checked") {
            None => read_boolean_element(child(checkbox_element, "w:default")),
            Some(checked_element) => read_boolean_element(Some(checked_element)),
        };
        return ComplexField::Checkbox { checked };
    }

    ComplexField::Unknown
}

/// Convert unmerged cells to cells (deeply), used when the table structure is
/// unexpected.
fn remove_unmerged_table_cells(rows: Vec<DocElement>) -> Vec<DocElement> {
    rows.into_iter()
        .map(|row| {
            transform_elements(row, &mut |el| match el {
                DocElement::TableCellUnmerged { children, colspan, rowspan, .. } => {
                    DocElement::TableCell { children, colspan, rowspan }
                }
                other => other,
            })
        })
        .collect()
}

/// Merge vertically merged cells into row spans.
fn calculate_row_spans(rows: Vec<DocElement>) -> Vec<DocElement> {
    // Unexpected non-row element in table: cell merging may be incorrect.
    if rows.iter().any(|row| !matches!(row, DocElement::TableRow { .. })) {
        return remove_unmerged_table_cells(rows);
    }
    // Unexpected non-cell element in table row: cell merging may be incorrect.
    let unexpected_non_cells = rows.iter().any(|row| match row {
        DocElement::TableRow { children, .. } => {
            children.iter().any(|cell| !matches!(cell, DocElement::TableCellUnmerged { .. }))
        }
        _ => false,
    });
    if unexpected_non_cells {
        return remove_unmerged_table_cells(rows);
    }

    // Copy the cells out so row spans can be updated across rows.
    struct Cell {
        children: Vec<DocElement>,
        colspan: i64,
        rowspan: i64,
        vmerge: bool,
    }
    let mut grid: Vec<(bool, Vec<Cell>)> = rows
        .into_iter()
        .map(|row| match row {
            DocElement::TableRow { children, is_header } => (
                is_header,
                children
                    .into_iter()
                    .map(|cell| match cell {
                        DocElement::TableCellUnmerged { children, colspan, rowspan, vmerge } => {
                            Cell { children, colspan, rowspan, vmerge }
                        }
                        _ => unreachable!("checked above"),
                    })
                    .collect(),
            ),
            _ => unreachable!("checked above"),
        })
        .collect();

    let mut columns: HashMap<i64, (usize, usize)> = HashMap::new();
    for row_index in 0..grid.len() {
        let mut cell_index: i64 = 0;
        for cell_position in 0..grid[row_index].1.len() {
            let (vmerge, colspan) = {
                let cell = &grid[row_index].1[cell_position];
                (cell.vmerge, cell.colspan)
            };
            match columns.get(&cell_index).copied() {
                Some((r, c)) if vmerge => grid[r].1[c].rowspan += 1,
                _ => {
                    columns.insert(cell_index, (row_index, cell_position));
                    grid[row_index].1[cell_position].vmerge = false;
                }
            }
            cell_index += colspan;
        }
    }

    grid.into_iter()
        .map(|(is_header, cells)| DocElement::TableRow {
            is_header,
            children: cells
                .into_iter()
                .filter(|cell| !cell.vmerge)
                .map(|cell| DocElement::TableCell { children: cell.children, colspan: cell.colspan, rowspan: cell.rowspan })
                .collect(),
        })
        .collect()
}

// =============================================================================
// Reading the package (`docx/__init__.py`)
// =============================================================================

fn read_notes(
    source: &DocxSource<'_>,
    path: &str,
    note_type: &str,
    numbering: &Numbering,
    content_types: &ContentTypes,
    styles: &Styles,
) -> anyhow::Result<Vec<Note>> {
    let Some(root) = source.try_read_entry(path)? else {
        return Ok(Vec::new());
    };
    let relationships = source.read_relationships(&relationships_path_for(path))?;
    let mut reader = BodyReader::new(numbering, content_types, relationships, styles);
    let mut notes = Vec::new();
    for element in root.children_named(&format!("w:{note_type}")) {
        if matches!(element.attr("w:type"), Some("continuationSeparator") | Some("separator")) {
            continue;
        }
        let body = reader.read_all(&element.children)?;
        notes.push(Note { note_type: note_type.to_string(), note_id: required_attr(element, "w:id")?.to_string(), body });
    }
    Ok(notes)
}

fn read_comments(
    source: &DocxSource<'_>,
    path: &str,
    numbering: &Numbering,
    content_types: &ContentTypes,
    styles: &Styles,
) -> anyhow::Result<Vec<Comment>> {
    let Some(root) = source.try_read_entry(path)? else {
        return Ok(Vec::new());
    };
    let relationships = source.read_relationships(&relationships_path_for(path))?;
    let mut reader = BodyReader::new(numbering, content_types, relationships, styles);
    let mut comments = Vec::new();
    for element in root.children_named("w:comment") {
        let read_optional_attribute = |name: &str| {
            let value = element.attr(name).unwrap_or("").trim().to_string();
            if value.is_empty() { None } else { Some(value) }
        };
        let body = reader.read_all(&element.children)?;
        comments.push(Comment {
            comment_id: required_attr(element, "w:id")?.to_string(),
            body,
            author_name: read_optional_attribute("w:author"),
            author_initials: read_optional_attribute("w:initials"),
        });
    }
    Ok(comments)
}

/// `docx.read`: read the package into a [`Document`].
pub fn read_document(source: &DocxSource<'_>) -> anyhow::Result<Document> {
    let part_paths = find_part_paths(source)?;

    let content_types = match source.try_read_entry("[Content_Types].xml")? {
        Some(root) => ContentTypes::read(&root)?,
        None => ContentTypes::default(),
    };
    let styles = match source.try_read_entry(&part_paths.styles)? {
        Some(root) => Styles::read(&root)?,
        None => Styles::default(),
    };
    let numbering = match source.try_read_entry(&part_paths.numbering)? {
        Some(root) => Numbering::read(&root, &styles)?,
        None => Numbering::default(),
    };

    let mut notes = read_notes(source, &part_paths.footnotes, "footnote", &numbering, &content_types, &styles)?;
    notes.extend(read_notes(source, &part_paths.endnotes, "endnote", &numbering, &content_types, &styles)?);
    let comments = read_comments(source, &part_paths.comments, &numbering, &content_types, &styles)?;

    let root = source.read_entry(&part_paths.main_document)?;
    let relationships = source.read_relationships(&relationships_path_for(&part_paths.main_document))?;
    let mut reader = BodyReader::new(&numbering, &content_types, relationships, &styles);
    let Some(body) = root.child("w:body") else {
        anyhow::bail!("Could not find the body element: are you sure this is a docx file?");
    };
    let children = reader.read_all(&body.children)?;
    Ok(Document { children, notes, comments })
}

// =============================================================================
// Style map (`styles/parser`, `document_matchers.py`, `html_paths.py`)
// =============================================================================

#[derive(Debug, Clone, PartialEq)]
enum StringMatcher {
    EqualTo(String),
    StartsWith(String),
}

impl StringMatcher {
    fn matches(&self, other: &str) -> bool {
        match self {
            StringMatcher::EqualTo(value) => value.to_uppercase() == other.to_uppercase(),
            StringMatcher::StartsWith(value) => other.to_uppercase().starts_with(&value.to_uppercase()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum DocumentMatcher {
    Paragraph { style_id: Option<String>, style_name: Option<StringMatcher>, numbering: Option<NumberingLevel> },
    Run { style_id: Option<String>, style_name: Option<StringMatcher> },
    Table { style_id: Option<String>, style_name: Option<StringMatcher> },
    Bold,
    Italic,
    Underline,
    Strikethrough,
    AllCaps,
    SmallCaps,
    Highlight { color: Option<String> },
    CommentReference,
    Break(BreakType),
}

/// An HTML tag of an HTML path (`html.Tag`).
#[derive(Debug, Clone, PartialEq)]
pub struct HtmlTag {
    pub tag_names: Vec<String>,
    pub attributes: BTreeMap<String, String>,
    pub collapsible: bool,
    pub separator: Option<String>,
}

impl HtmlTag {
    fn new(names: &[&str], attributes: BTreeMap<String, String>, collapsible: bool) -> Self {
        Self { tag_names: names.iter().map(|n| n.to_string()).collect(), attributes, collapsible, separator: None }
    }

    fn tag_name(&self) -> &str {
        &self.tag_names[0]
    }
}

/// `html_paths.path(...)` or `html_paths.ignore`.
#[derive(Debug, Clone, PartialEq)]
enum HtmlPath {
    Elements(Vec<HtmlTag>),
    Ignore,
}

impl HtmlPath {
    fn empty() -> Self {
        HtmlPath::Elements(Vec::new())
    }

    /// A single non-fresh (collapsible) element.
    fn element(name: &str) -> Self {
        HtmlPath::Elements(vec![HtmlTag::new(&[name], BTreeMap::new(), true)])
    }

    /// A single fresh element.
    fn fresh_element(name: &str) -> Self {
        HtmlPath::Elements(vec![HtmlTag::new(&[name], BTreeMap::new(), false)])
    }

    fn is_ignore(&self) -> bool {
        matches!(self, HtmlPath::Ignore)
    }

    /// Wrap already generated nodes (callers skip generation for `Ignore`).
    fn wrap_nodes(&self, nodes: Vec<HtmlNode>) -> Vec<HtmlNode> {
        match self {
            HtmlPath::Ignore => Vec::new(),
            HtmlPath::Elements(elements) => {
                let mut nodes = nodes;
                for tag in elements.iter().rev() {
                    nodes = vec![HtmlNode::Element(HtmlElement { tag: tag.clone(), children: nodes })];
                }
                nodes
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Style {
    document_matcher: DocumentMatcher,
    html_path: HtmlPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenType {
    Identifier,
    Symbol,
    Whitespace,
    String,
    UnterminatedString,
    Integer,
    Unknown,
    End,
}

#[derive(Debug, Clone)]
struct Token {
    token_type: TokenType,
    value: String,
}

static TOKEN_RULES: Lazy<Vec<(TokenType, Regex)>> = Lazy::new(|| {
    let string_prefix = r"'(?:\\.|[^'])*";
    let identifier_character = r"(?:[a-zA-Z\-_]|\\.)";
    let rules = vec![
        (TokenType::Identifier, format!("{identifier_character}(?:{identifier_character}|[0-9])*")),
        (TokenType::Symbol, r":|>|=>|\^=|=|\(|\)|\[|\]|\||!|\.".to_string()),
        (TokenType::Whitespace, r"\s+".to_string()),
        (TokenType::String, format!("{string_prefix}'")),
        (TokenType::UnterminatedString, string_prefix.to_string()),
        (TokenType::Integer, "([0-9]+)".to_string()),
        (TokenType::Unknown, ".".to_string()),
    ];
    rules
        .into_iter()
        .map(|(t, r)| (t, Regex::new(&format!("^(?:{r})")).expect("valid regex")))
        .collect()
});

fn tokenise(value: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut index = 0;
    while index < value.len() {
        let rest = &value[index..];
        let mut matched = false;
        for (token_type, regex) in TOKEN_RULES.iter() {
            if let Some(m) = regex.find(rest)
                && !m.as_str().is_empty()
            {
                tokens.push(Token { token_type: *token_type, value: m.as_str().to_string() });
                index += m.end();
                matched = true;
                break;
            }
        }
        if !matched {
            // A lone char the "." rule could not take (a newline): unknown.
            let c = rest.chars().next().expect("non-empty");
            tokens.push(Token { token_type: TokenType::Unknown, value: c.to_string() });
            index += c.len_utf8();
        }
    }
    tokens.push(Token { token_type: TokenType::End, value: String::new() });
    tokens
}

/// Raised by the style-mapping parser (`LineParseError`).
#[derive(Debug)]
struct LineParseError;

struct TokenIterator {
    tokens: Vec<Token>,
    index: usize,
}

type ParseResult<T> = Result<T, LineParseError>;

impl TokenIterator {
    fn token(&self) -> &Token {
        &self.tokens[self.index.min(self.tokens.len() - 1)]
    }

    fn peek_token_type(&self) -> TokenType {
        self.token().token_type
    }

    fn next_value(&mut self, token_type: Option<TokenType>) -> ParseResult<String> {
        let token = self.token().clone();
        if token_type.is_none_or(|t| t == token.token_type) {
            self.index += 1;
            Ok(token.value)
        } else {
            Err(LineParseError)
        }
    }

    fn is_next(&self, token_type: TokenType, token_value: Option<&str>) -> bool {
        let token = self.token();
        token.token_type == token_type && token_value.is_none_or(|v| token.value == v)
    }

    fn skip(&mut self, token_type: TokenType, token_value: Option<&str>) -> ParseResult<()> {
        if self.is_next(token_type, token_value) {
            self.index += 1;
            Ok(())
        } else {
            Err(LineParseError)
        }
    }

    fn try_skip(&mut self, token_type: TokenType, token_value: Option<&str>) -> bool {
        if self.is_next(token_type, token_value) {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn try_skip_many(&mut self, tokens: &[(TokenType, Option<&str>)]) -> bool {
        let start = self.index;
        for (token_type, token_value) in tokens {
            if !self.is_next(*token_type, *token_value) {
                self.index = start;
                return false;
            }
            self.index += 1;
        }
        true
    }
}

static RE_ESCAPE_SEQUENCE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\\(.)").expect("valid regex"));

fn decode_escape_sequences(value: &str) -> String {
    RE_ESCAPE_SEQUENCE
        .replace_all(value, |caps: &regex::Captures| match &caps[1] {
            "n" => "\n".to_string(),
            "r" => "\r".to_string(),
            "t" => "\t".to_string(),
            other => other.to_string(),
        })
        .into_owned()
}

fn parse_identifier(tokens: &mut TokenIterator) -> ParseResult<String> {
    Ok(decode_escape_sequences(&tokens.next_value(Some(TokenType::Identifier))?))
}

fn parse_string(tokens: &mut TokenIterator) -> ParseResult<String> {
    let value = tokens.next_value(Some(TokenType::String))?;
    let inner: String = {
        let chars: Vec<char> = value.chars().collect();
        chars[1..chars.len().saturating_sub(1).max(1)].iter().collect()
    };
    Ok(decode_escape_sequences(&inner))
}

fn try_parse_class_name(tokens: &mut TokenIterator) -> ParseResult<Option<String>> {
    if tokens.try_skip(TokenType::Symbol, Some(".")) { Ok(Some(parse_identifier(tokens)?)) } else { Ok(None) }
}

fn parse_style_name(tokens: &mut TokenIterator) -> ParseResult<Option<StringMatcher>> {
    if tokens.try_skip(TokenType::Symbol, Some("[")) {
        tokens.skip(TokenType::Identifier, Some("style-name"))?;
        let matcher = if tokens.try_skip(TokenType::Symbol, Some("=")) {
            StringMatcher::EqualTo(parse_string(tokens)?)
        } else if tokens.try_skip(TokenType::Symbol, Some("^=")) {
            StringMatcher::StartsWith(parse_string(tokens)?)
        } else {
            return Err(LineParseError);
        };
        tokens.skip(TokenType::Symbol, Some("]"))?;
        Ok(Some(matcher))
    } else {
        Ok(None)
    }
}

fn parse_numbering(tokens: &mut TokenIterator) -> ParseResult<Option<NumberingLevel>> {
    if tokens.try_skip(TokenType::Symbol, Some(":")) {
        let is_ordered = match tokens.next_value(Some(TokenType::Identifier))?.as_str() {
            "ordered-list" => true,
            "unordered-list" => false,
            _ => return Err(LineParseError),
        };
        tokens.skip(TokenType::Symbol, Some("("))?;
        let level: i64 = tokens.next_value(Some(TokenType::Integer))?.parse::<i64>().map_err(|_| LineParseError)? - 1;
        tokens.skip(TokenType::Symbol, Some(")"))?;
        Ok(Some(NumberingLevel { level_index: level.to_string(), is_ordered }))
    } else {
        Ok(None)
    }
}

fn parse_document_matcher(tokens: &mut TokenIterator) -> ParseResult<DocumentMatcher> {
    if tokens.try_skip(TokenType::Identifier, Some("p")) {
        let style_id = try_parse_class_name(tokens)?;
        let style_name = parse_style_name(tokens)?;
        let numbering = parse_numbering(tokens)?;
        Ok(DocumentMatcher::Paragraph { style_id, style_name, numbering })
    } else if tokens.try_skip(TokenType::Identifier, Some("r")) {
        let style_id = try_parse_class_name(tokens)?;
        let style_name = parse_style_name(tokens)?;
        Ok(DocumentMatcher::Run { style_id, style_name })
    } else if tokens.try_skip(TokenType::Identifier, Some("table")) {
        let style_id = try_parse_class_name(tokens)?;
        let style_name = parse_style_name(tokens)?;
        Ok(DocumentMatcher::Table { style_id, style_name })
    } else if tokens.try_skip(TokenType::Identifier, Some("b")) {
        Ok(DocumentMatcher::Bold)
    } else if tokens.try_skip(TokenType::Identifier, Some("i")) {
        Ok(DocumentMatcher::Italic)
    } else if tokens.try_skip(TokenType::Identifier, Some("u")) {
        Ok(DocumentMatcher::Underline)
    } else if tokens.try_skip(TokenType::Identifier, Some("strike")) {
        Ok(DocumentMatcher::Strikethrough)
    } else if tokens.try_skip(TokenType::Identifier, Some("all-caps")) {
        Ok(DocumentMatcher::AllCaps)
    } else if tokens.try_skip(TokenType::Identifier, Some("small-caps")) {
        Ok(DocumentMatcher::SmallCaps)
    } else if tokens.try_skip(TokenType::Identifier, Some("highlight")) {
        let color = if tokens.try_skip(TokenType::Symbol, Some("[")) {
            tokens.skip(TokenType::Identifier, Some("color"))?;
            tokens.skip(TokenType::Symbol, Some("="))?;
            let color = parse_string(tokens)?;
            tokens.skip(TokenType::Symbol, Some("]"))?;
            Some(color)
        } else {
            None
        };
        Ok(DocumentMatcher::Highlight { color })
    } else if tokens.try_skip(TokenType::Identifier, Some("comment-reference")) {
        Ok(DocumentMatcher::CommentReference)
    } else if tokens.try_skip(TokenType::Identifier, Some("br")) {
        tokens.skip(TokenType::Symbol, Some("["))?;
        tokens.skip(TokenType::Identifier, Some("type"))?;
        tokens.skip(TokenType::Symbol, Some("="))?;
        let type_name = parse_string(tokens)?;
        tokens.skip(TokenType::Symbol, Some("]"))?;
        match type_name.as_str() {
            "line" => Ok(DocumentMatcher::Break(BreakType::Line)),
            "page" => Ok(DocumentMatcher::Break(BreakType::Page)),
            "column" => Ok(DocumentMatcher::Break(BreakType::Column)),
            _ => Err(LineParseError),
        }
    } else {
        Err(LineParseError)
    }
}

fn parse_html_element(tokens: &mut TokenIterator) -> ParseResult<HtmlTag> {
    let mut tag_names = vec![parse_identifier(tokens)?];
    while tokens.try_skip(TokenType::Symbol, Some("|")) {
        tag_names.push(parse_identifier(tokens)?);
    }

    // Attributes and class names, in order.
    let mut attributes_list: Vec<(String, String, bool)> = Vec::new();
    loop {
        if tokens.is_next(TokenType::Symbol, Some("[")) {
            tokens.skip(TokenType::Symbol, Some("["))?;
            let name = parse_identifier(tokens)?;
            tokens.skip(TokenType::Symbol, Some("="))?;
            let value = parse_string(tokens)?;
            tokens.skip(TokenType::Symbol, Some("]"))?;
            attributes_list.push((name, value, false));
        } else if tokens.is_next(TokenType::Symbol, Some(".")) {
            tokens.skip(TokenType::Symbol, Some("."))?;
            let class_name = parse_identifier(tokens)?;
            attributes_list.push(("class".to_string(), class_name, true));
        } else {
            break;
        }
    }

    let is_fresh = tokens.try_skip_many(&[(TokenType::Symbol, Some(":")), (TokenType::Identifier, Some("fresh"))]);
    let separator = if tokens.try_skip_many(&[(TokenType::Symbol, Some(":")), (TokenType::Identifier, Some("separator"))]) {
        tokens.skip(TokenType::Symbol, Some("("))?;
        let value = parse_string(tokens)?;
        tokens.skip(TokenType::Symbol, Some(")"))?;
        Some(value)
    } else {
        None
    };

    let mut attributes: BTreeMap<String, String> = BTreeMap::new();
    for (name, value, append) in attributes_list {
        match attributes.get_mut(&name) {
            Some(existing) if append && !existing.is_empty() => {
                existing.push(' ');
                existing.push_str(&value);
            }
            _ => {
                attributes.insert(name, value);
            }
        }
    }

    Ok(HtmlTag { tag_names, attributes, collapsible: !is_fresh, separator })
}

fn parse_html_path(tokens: &mut TokenIterator) -> ParseResult<HtmlPath> {
    if tokens.try_skip(TokenType::Symbol, Some("!")) {
        return Ok(HtmlPath::Ignore);
    }
    let mut elements = Vec::new();
    if tokens.peek_token_type() == TokenType::Identifier {
        elements.push(parse_html_element(tokens)?);
        while tokens.try_skip_many(&[(TokenType::Whitespace, None), (TokenType::Symbol, Some(">"))]) {
            tokens.skip(TokenType::Whitespace, None)?;
            elements.push(parse_html_element(tokens)?);
        }
    }
    Ok(HtmlPath::Elements(elements))
}

fn parse_style_mapping(tokens: &mut TokenIterator) -> ParseResult<Style> {
    let document_matcher = parse_document_matcher(tokens)?;
    tokens.skip(TokenType::Whitespace, None)?;
    tokens.skip(TokenType::Symbol, Some("=>"))?;
    tokens.try_skip(TokenType::Whitespace, None);
    let html_path = parse_html_path(tokens)?;
    tokens.skip(TokenType::End, None)?;
    Ok(Style { document_matcher, html_path })
}

/// `_read_style_map`: one mapping per non-comment line; lines that do not
/// parse are ignored (with a warning).
fn read_style_map(style_text: &str) -> Vec<Style> {
    style_text
        .split('\n')
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter_map(|line| {
            let mut tokens = TokenIterator { tokens: tokenise(line), index: 0 };
            parse_style_mapping(&mut tokens).ok()
        })
        .collect()
}

const DEFAULT_STYLE_MAP: &str = "
p.Heading1 => h1:fresh
p.Heading2 => h2:fresh
p.Heading3 => h3:fresh
p.Heading4 => h4:fresh
p.Heading5 => h5:fresh
p.Heading6 => h6:fresh
p[style-name='Heading 1'] => h1:fresh
p[style-name='Heading 2'] => h2:fresh
p[style-name='Heading 3'] => h3:fresh
p[style-name='Heading 4'] => h4:fresh
p[style-name='Heading 5'] => h5:fresh
p[style-name='Heading 6'] => h6:fresh
p[style-name='heading 1'] => h1:fresh
p[style-name='heading 2'] => h2:fresh
p[style-name='heading 3'] => h3:fresh
p[style-name='heading 4'] => h4:fresh
p[style-name='heading 5'] => h5:fresh
p[style-name='heading 6'] => h6:fresh

# Apple Pages
p.Heading => h1:fresh
p[style-name='Heading'] => h1:fresh

r[style-name='Strong'] => strong

p[style-name='footnote text'] => p:fresh
r[style-name='footnote reference'] =>
p[style-name='endnote text'] => p:fresh
r[style-name='endnote reference'] =>
p[style-name='annotation text'] => p:fresh
r[style-name='annotation reference'] =>

# LibreOffice
p[style-name='Footnote'] => p:fresh
r[style-name='Footnote anchor'] =>
p[style-name='Endnote'] => p:fresh
r[style-name='Endnote anchor'] =>

p:unordered-list(1) => ul > li:fresh
p:unordered-list(2) => ul|ol > li > ul > li:fresh
p:unordered-list(3) => ul|ol > li > ul|ol > li > ul > li:fresh
p:unordered-list(4) => ul|ol > li > ul|ol > li > ul|ol > li > ul > li:fresh
p:unordered-list(5) => ul|ol > li > ul|ol > li > ul|ol > li > ul|ol > li > ul > li:fresh
p:ordered-list(1) => ol > li:fresh
p:ordered-list(2) => ul|ol > li > ol > li:fresh
p:ordered-list(3) => ul|ol > li > ul|ol > li > ol > li:fresh
p:ordered-list(4) => ul|ol > li > ul|ol > li > ul|ol > li > ol > li:fresh
p:ordered-list(5) => ul|ol > li > ul|ol > li > ul|ol > li > ul|ol > li > ol > li:fresh

r[style-name='Hyperlink'] =>

p[style-name='Normal'] => p:fresh

# Apple Pages
p.Body => p:fresh
p[style-name='Body'] => p:fresh
";

static DEFAULT_STYLES: Lazy<Vec<Style>> = Lazy::new(|| read_style_map(DEFAULT_STYLE_MAP));

// =============================================================================
// HTML nodes and writer (`html/`, `writers/html.py`)
// =============================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct HtmlElement {
    pub tag: HtmlTag,
    pub children: Vec<HtmlNode>,
}

impl HtmlElement {
    fn is_void(&self) -> bool {
        self.children.is_empty() && matches!(self.tag.tag_name(), "br" | "hr" | "img" | "input")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum HtmlNode {
    Element(HtmlElement),
    Text(String),
    ForceWrite,
}

fn html_element(name: &str, attributes: BTreeMap<String, String>, children: Vec<HtmlNode>) -> HtmlNode {
    HtmlNode::Element(HtmlElement { tag: HtmlTag::new(&[name], attributes, false), children })
}

fn collapsible_element(name: &str, attributes: BTreeMap<String, String>, children: Vec<HtmlNode>) -> HtmlNode {
    HtmlNode::Element(HtmlElement { tag: HtmlTag::new(&[name], attributes, true), children })
}

fn attrs(pairs: &[(&str, String)]) -> BTreeMap<String, String> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

fn strip_empty(nodes: Vec<HtmlNode>) -> Vec<HtmlNode> {
    nodes
        .into_iter()
        .flat_map(|node| match node {
            HtmlNode::Text(value) => {
                if value.is_empty() {
                    vec![]
                } else {
                    vec![HtmlNode::Text(value)]
                }
            }
            HtmlNode::Element(element) => {
                let was_void = element.is_void();
                let children = strip_empty(element.children);
                if children.is_empty() && !was_void {
                    vec![]
                } else {
                    vec![HtmlNode::Element(HtmlElement { tag: element.tag, children })]
                }
            }
            HtmlNode::ForceWrite => vec![HtmlNode::ForceWrite],
        })
        .collect()
}

fn collapse(nodes: Vec<HtmlNode>) -> Vec<HtmlNode> {
    let mut collapsed = Vec::new();
    for node in nodes {
        collapsing_add(&mut collapsed, node);
    }
    collapsed
}

fn collapse_node(node: HtmlNode) -> HtmlNode {
    match node {
        HtmlNode::Element(element) => {
            HtmlNode::Element(HtmlElement { tag: element.tag, children: collapse(element.children) })
        }
        other => other,
    }
}

fn collapsing_add(collapsed: &mut Vec<HtmlNode>, node: HtmlNode) {
    let collapsed_node = collapse_node(node);
    if let Some(node) = try_collapse(collapsed, collapsed_node) {
        collapsed.push(node);
    }
}

/// Merge `node` into the last collapsed node when possible; returns the node
/// back when it could not be merged.
fn try_collapse(collapsed: &mut [HtmlNode], node: HtmlNode) -> Option<HtmlNode> {
    let Some(HtmlNode::Element(last)) = collapsed.last_mut() else {
        return Some(node);
    };
    let HtmlNode::Element(element) = node else {
        return Some(node);
    };
    if !element.tag.collapsible {
        return Some(HtmlNode::Element(element));
    }
    let is_match = element.tag.tag_names.iter().any(|n| n == last.tag.tag_name())
        && last.tag.attributes == element.tag.attributes;
    if !is_match {
        return Some(HtmlNode::Element(element));
    }
    if let Some(separator) = element.tag.separator.as_ref().filter(|s| !s.is_empty()) {
        last.children.push(HtmlNode::Text(separator.clone()));
    }
    for child in element.children {
        collapsing_add(&mut last.children, child);
    }
    None
}

/// `xml.sax.saxutils.escape(text, {'"': "&quot;"})`.
fn escape_html(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn write_html(nodes: &[HtmlNode], out: &mut String) {
    for node in nodes {
        match node {
            HtmlNode::Text(value) => out.push_str(&escape_html(value)),
            HtmlNode::ForceWrite => {}
            HtmlNode::Element(element) => {
                let attribute_string: String = element
                    .tag
                    .attributes
                    .iter()
                    .map(|(k, v)| format!(" {k}=\"{}\"", escape_html(v)))
                    .collect();
                let name = element.tag.tag_name();
                if element.is_void() {
                    out.push_str(&format!("<{name}{attribute_string} />"));
                } else {
                    out.push_str(&format!("<{name}{attribute_string}>"));
                    write_html(&element.children, out);
                    out.push_str(&format!("</{name}>"));
                }
            }
        }
    }
}

// =============================================================================
// Conversion (`conversion.py`)
// =============================================================================

/// What a style matcher is compared against (`_document_matcher_matches`).
enum MatchTarget<'e> {
    Paragraph(&'e Paragraph),
    Run(&'e Run),
    Table { style_id: Option<&'e str>, style_name: Option<&'e str> },
    Property(&'static str),
    Highlight(&'e str),
    Break(BreakType),
    CommentReference,
}

fn matcher_matches(matcher: &DocumentMatcher, target: &MatchTarget<'_>) -> bool {
    fn style_matches(
        style_id: &Option<String>,
        style_name: &Option<StringMatcher>,
        element_style_id: Option<&str>,
        element_style_name: Option<&str>,
    ) -> bool {
        style_id.as_deref().is_none_or(|id| Some(id) == element_style_id)
            && style_name
                .as_ref()
                .is_none_or(|matcher| element_style_name.is_some_and(|name| matcher.matches(name)))
    }

    match (matcher, target) {
        (DocumentMatcher::Underline, MatchTarget::Property("underline"))
        | (DocumentMatcher::Strikethrough, MatchTarget::Property("strikethrough"))
        | (DocumentMatcher::AllCaps, MatchTarget::Property("all_caps"))
        | (DocumentMatcher::SmallCaps, MatchTarget::Property("small_caps"))
        | (DocumentMatcher::Bold, MatchTarget::Property("bold"))
        | (DocumentMatcher::Italic, MatchTarget::Property("italic"))
        | (DocumentMatcher::CommentReference, MatchTarget::CommentReference) => true,
        (DocumentMatcher::Highlight { color }, MatchTarget::Highlight(element_color)) => {
            color.as_deref().is_none_or(|c| c == *element_color)
        }
        (DocumentMatcher::Break(break_type), MatchTarget::Break(element_break)) => break_type == element_break,
        (DocumentMatcher::Paragraph { style_id, style_name, numbering }, MatchTarget::Paragraph(p)) => {
            style_matches(style_id, style_name, p.style_id.as_deref(), p.style_name.as_deref())
                && numbering.as_ref().is_none_or(|n| Some(n) == p.numbering.as_ref())
        }
        (DocumentMatcher::Run { style_id, style_name }, MatchTarget::Run(r)) => {
            style_matches(style_id, style_name, r.style_id.as_deref(), r.style_name.as_deref())
        }
        (DocumentMatcher::Table { style_id, style_name }, MatchTarget::Table { style_id: id, style_name: name }) => {
            style_matches(style_id, style_name, *id, *name)
        }
        _ => false,
    }
}

const UP_ARROW: &str = "\u{2191}";

struct DocumentConverter<'a> {
    style_map: &'a [Style],
    id_prefix: String,
    ignore_empty_paragraphs: bool,
    note_references: Vec<(String, String)>,
    referenced_comments: Vec<(String, Comment)>,
    comments: HashMap<String, Comment>,
    package: &'a OoxmlPackage,
}

impl<'a> DocumentConverter<'a> {
    fn find_style(&self, target: &MatchTarget<'_>) -> Option<&'a Style> {
        self.style_map.iter().find(|style| matcher_matches(&style.document_matcher, target))
    }

    fn find_html_path(&self, target: &MatchTarget<'_>, default: HtmlPath) -> HtmlPath {
        match self.find_style(target) {
            Some(style) => style.html_path.clone(),
            None => default,
        }
    }

    fn find_style_for_run_property(&self, element_type: &'static str, default: Option<&str>) -> HtmlPath {
        match self.find_style(&MatchTarget::Property(element_type)) {
            Some(style) => style.html_path.clone(),
            None => match default {
                Some(name) => HtmlPath::element(name),
                None => HtmlPath::empty(),
            },
        }
    }

    fn html_id(&self, suffix: &str) -> String {
        format!("{}{}", self.id_prefix, suffix)
    }

    fn referent_html_id(&self, reference_type: &str, reference_id: &str) -> String {
        self.html_id(&format!("{reference_type}-{reference_id}"))
    }

    fn reference_html_id(&self, reference_type: &str, reference_id: &str) -> String {
        self.html_id(&format!("{reference_type}-ref-{reference_id}"))
    }

    fn visit_all(&mut self, elements: &[DocElement], is_table_header: bool) -> anyhow::Result<Vec<HtmlNode>> {
        let mut nodes = Vec::new();
        for element in elements {
            nodes.extend(self.visit(element, is_table_header)?);
        }
        Ok(nodes)
    }

    fn visit_document(&mut self, document: &Document) -> anyhow::Result<Vec<HtmlNode>> {
        let mut nodes = self.visit_all(&document.children, false)?;

        let mut notes = Vec::new();
        for (note_type, note_id) in self.note_references.clone() {
            let note = document
                .notes
                .iter()
                .find(|n| n.note_type == note_type && n.note_id == note_id)
                .ok_or_else(|| anyhow::anyhow!("('{note_type}', '{note_id}')"))?;
            notes.push(note.clone());
        }
        let mut note_nodes = Vec::new();
        for note in &notes {
            note_nodes.extend(self.visit_note(note)?);
        }
        let notes_list = html_element("ol", BTreeMap::new(), note_nodes);

        let mut comment_nodes = Vec::new();
        for (label, comment) in self.referenced_comments.clone() {
            comment_nodes.extend(self.visit_comment(&label, &comment)?);
        }
        let comments = html_element("dl", BTreeMap::new(), comment_nodes);

        nodes.push(notes_list);
        nodes.push(comments);
        Ok(nodes)
    }

    fn visit(&mut self, element: &DocElement, is_table_header: bool) -> anyhow::Result<Vec<HtmlNode>> {
        match element {
            DocElement::Paragraph(paragraph) => {
                let default = HtmlPath::fresh_element("p");
                let html_path = self.find_html_path(&MatchTarget::Paragraph(paragraph), default);
                if html_path.is_ignore() {
                    return Ok(Vec::new());
                }
                let content = self.visit_all(&paragraph.children, is_table_header)?;
                let children = if self.ignore_empty_paragraphs {
                    content
                } else {
                    let mut c = vec![HtmlNode::ForceWrite];
                    c.extend(content);
                    c
                };
                Ok(html_path.wrap_nodes(children))
            }
            DocElement::Run(run) => self.visit_run(run, is_table_header),
            DocElement::Text(value) => Ok(vec![HtmlNode::Text(value.clone())]),
            DocElement::Hyperlink { children, href, anchor, target_frame } => {
                let href = match anchor {
                    None => href.clone().unwrap_or_else(|| "None".to_string()),
                    Some(anchor) => format!("#{}", self.html_id(anchor)),
                };
                let mut attributes = attrs(&[("href", href)]);
                if let Some(target_frame) = target_frame {
                    attributes.insert("target".to_string(), target_frame.clone());
                }
                let nodes = self.visit_all(children, is_table_header)?;
                Ok(vec![collapsible_element("a", attributes, nodes)])
            }
            DocElement::Checkbox { checked } => {
                let mut attributes = attrs(&[("type", "checkbox".to_string())]);
                if *checked {
                    attributes.insert("checked".to_string(), "checked".to_string());
                }
                Ok(vec![html_element("input", attributes, Vec::new())])
            }
            DocElement::Bookmark { name } => {
                let id = self.html_id(name.as_deref().unwrap_or("None"));
                Ok(vec![collapsible_element("a", attrs(&[("id", id)]), vec![HtmlNode::ForceWrite])])
            }
            DocElement::Tab => Ok(vec![HtmlNode::Text("\t".to_string())]),
            DocElement::Table { children, style_id, style_name } => {
                let target = MatchTarget::Table { style_id: style_id.as_deref(), style_name: style_name.as_deref() };
                let html_path = self.find_html_path(&target, HtmlPath::fresh_element("table"));
                if html_path.is_ignore() {
                    return Ok(Vec::new());
                }
                let nodes = self.convert_table_children(children)?;
                Ok(html_path.wrap_nodes(nodes))
            }
            DocElement::TableRow { children, .. } => {
                let mut nodes = vec![HtmlNode::ForceWrite];
                nodes.extend(self.visit_all(children, is_table_header)?);
                Ok(vec![html_element("tr", BTreeMap::new(), nodes)])
            }
            DocElement::TableCell { children, colspan, rowspan }
            | DocElement::TableCellUnmerged { children, colspan, rowspan, .. } => {
                let tag_name = if is_table_header { "th" } else { "td" };
                let mut attributes = BTreeMap::new();
                if *colspan != 1 {
                    attributes.insert("colspan".to_string(), colspan.to_string());
                }
                if *rowspan != 1 {
                    attributes.insert("rowspan".to_string(), rowspan.to_string());
                }
                let mut nodes = vec![HtmlNode::ForceWrite];
                nodes.extend(self.visit_all(children, is_table_header)?);
                Ok(vec![html_element(tag_name, attributes, nodes)])
            }
            DocElement::Break(break_type) => {
                let html_path = match self.find_style(&MatchTarget::Break(*break_type)) {
                    Some(style) => style.html_path.clone(),
                    None if *break_type == BreakType::Line => HtmlPath::fresh_element("br"),
                    None => HtmlPath::empty(),
                };
                Ok(html_path.wrap_nodes(Vec::new()))
            }
            DocElement::Image { alt_text, content_type, source } => self.visit_image(alt_text, content_type, source),
            DocElement::NoteReference { note_type, note_id } => {
                self.note_references.push((note_type.clone(), note_id.clone()));
                let note_number = self.note_references.len();
                let a = html_element(
                    "a",
                    attrs(&[
                        ("href", format!("#{}", self.referent_html_id(note_type, note_id))),
                        ("id", self.reference_html_id(note_type, note_id)),
                    ]),
                    vec![HtmlNode::Text(format!("[{note_number}]"))],
                );
                Ok(vec![html_element("sup", BTreeMap::new(), vec![a])])
            }
            DocElement::CommentReference { comment_id } => {
                let html_path = self.find_html_path(&MatchTarget::CommentReference, HtmlPath::Ignore);
                if html_path.is_ignore() {
                    return Ok(Vec::new());
                }
                let comment = self.comments.get(comment_id).cloned().ok_or_else(|| anyhow::anyhow!("'{comment_id}'"))?;
                let count = self.referenced_comments.len() + 1;
                let label = format!("[{}{}]", comment.author_initials.clone().unwrap_or_default(), count);
                self.referenced_comments.push((label.clone(), comment));
                let nodes = vec![html_element(
                    "a",
                    attrs(&[
                        ("href", format!("#{}", self.referent_html_id("comment", comment_id))),
                        ("id", self.reference_html_id("comment", comment_id)),
                    ]),
                    vec![HtmlNode::Text(label)],
                )];
                Ok(html_path.wrap_nodes(nodes))
            }
        }
    }

    fn visit_run(&mut self, run: &Run, is_table_header: bool) -> anyhow::Result<Vec<HtmlNode>> {
        let mut paths: Vec<HtmlPath> = Vec::new();
        if let Some(color) = &run.highlight
            && let Some(style) = self.find_style(&MatchTarget::Highlight(color))
        {
            paths.push(style.html_path.clone());
        }
        if run.is_small_caps {
            paths.push(self.find_style_for_run_property("small_caps", None));
        }
        if run.is_all_caps {
            paths.push(self.find_style_for_run_property("all_caps", None));
        }
        if run.is_strikethrough {
            paths.push(self.find_style_for_run_property("strikethrough", Some("s")));
        }
        if run.is_underline {
            paths.push(self.find_style_for_run_property("underline", None));
        }
        if run.vertical_alignment == VerticalAlignment::Subscript {
            paths.push(HtmlPath::element("sub"));
        }
        if run.vertical_alignment == VerticalAlignment::Superscript {
            paths.push(HtmlPath::element("sup"));
        }
        if run.is_italic {
            paths.push(self.find_style_for_run_property("italic", Some("em")));
        }
        if run.is_bold {
            paths.push(self.find_style_for_run_property("bold", Some("strong")));
        }
        paths.push(self.find_html_path(&MatchTarget::Run(run), HtmlPath::empty()));

        // Each path wraps the lazily generated nodes of the paths before it;
        // an ignored path never generates its inner nodes.
        let outermost_ignore = paths.iter().rposition(HtmlPath::is_ignore);
        let (mut nodes, start) = match outermost_ignore {
            Some(index) => (Vec::new(), index + 1),
            None => (self.visit_all(&run.children, is_table_header)?, 0),
        };
        for path in &paths[start..] {
            nodes = path.wrap_nodes(nodes);
        }
        Ok(nodes)
    }

    fn convert_table_children(&mut self, children: &[DocElement]) -> anyhow::Result<Vec<HtmlNode>> {
        let body_index = children
            .iter()
            .position(|child| !matches!(child, DocElement::TableRow { is_header: true, .. }))
            .unwrap_or(children.len());

        let mut nodes = vec![HtmlNode::ForceWrite];
        if body_index == 0 {
            nodes.extend(self.visit_all(children, false)?);
        } else {
            let head_rows = self.visit_all(&children[..body_index], true)?;
            let body_rows = self.visit_all(&children[body_index..], false)?;
            nodes.push(html_element("thead", BTreeMap::new(), head_rows));
            nodes.push(html_element("tbody", BTreeMap::new(), body_rows));
        }
        Ok(nodes)
    }

    /// `images.data_uri`: an `<img>` with the image inlined as a data URI.
    /// Linked (external) images are refused, which mammoth reports as a
    /// warning and skips.
    fn visit_image(
        &mut self,
        alt_text: &Option<String>,
        content_type: &Option<String>,
        source: &ImageSource,
    ) -> anyhow::Result<Vec<HtmlNode>> {
        let path = match source {
            ImageSource::Linked(_) => return Ok(Vec::new()),
            ImageSource::Embedded(path) => path,
        };
        let bytes = self
            .package
            .read_exact(path)
            .ok_or_else(|| anyhow::anyhow!("There is no item named '{path}' in the archive"))?;
        let encoded_src = base64::engine::general_purpose::STANDARD.encode(bytes);
        let mut attributes = BTreeMap::new();
        if let Some(alt) = alt_text.as_ref().filter(|a| !a.is_empty()) {
            attributes.insert("alt".to_string(), alt.clone());
        }
        attributes.insert(
            "src".to_string(),
            format!("data:{};base64,{}", content_type.as_deref().unwrap_or("None"), encoded_src),
        );
        Ok(vec![html_element("img", attributes, Vec::new())])
    }

    fn visit_note(&mut self, note: &Note) -> anyhow::Result<Vec<HtmlNode>> {
        let mut note_body = self.visit_all(&note.body, false)?;
        note_body.push(collapsible_element(
            "p",
            BTreeMap::new(),
            vec![
                HtmlNode::Text(" ".to_string()),
                html_element(
                    "a",
                    attrs(&[("href", format!("#{}", self.reference_html_id(&note.note_type, &note.note_id)))]),
                    vec![HtmlNode::Text(UP_ARROW.to_string())],
                ),
            ],
        ));
        Ok(vec![html_element(
            "li",
            attrs(&[("id", self.referent_html_id(&note.note_type, &note.note_id))]),
            note_body,
        )])
    }

    fn visit_comment(&mut self, label: &str, comment: &Comment) -> anyhow::Result<Vec<HtmlNode>> {
        let mut body = self.visit_all(&comment.body, false)?;
        body.push(collapsible_element(
            "p",
            BTreeMap::new(),
            vec![
                HtmlNode::Text(" ".to_string()),
                html_element(
                    "a",
                    attrs(&[("href", format!("#{}", self.reference_html_id("comment", &comment.comment_id)))]),
                    vec![HtmlNode::Text(UP_ARROW.to_string())],
                ),
            ],
        ));
        Ok(vec![
            html_element(
                "dt",
                attrs(&[("id", self.referent_html_id("comment", &comment.comment_id))]),
                vec![HtmlNode::Text(format!("Comment {label}"))],
            ),
            html_element("dd", BTreeMap::new(), body),
        ])
    }
}

/// `mammoth.convert_to_html(fileobj).value` with the default options
/// (embedded style map included, default style map appended, empty
/// paragraphs ignored, images inlined as data URIs).
pub fn convert_to_html(source: &DocxSource<'_>) -> anyhow::Result<String> {
    // The embedded style map comes first so it takes precedence.
    let mut style_map: Vec<Style> = Vec::new();
    if let Some(bytes) = source.package.read_exact("mammoth/style-map") {
        let text = std::str::from_utf8(bytes).map_err(|e| anyhow::anyhow!("'utf-8' codec can't decode bytes: {e}"))?;
        style_map.extend(read_style_map(text));
    }
    style_map.extend(DEFAULT_STYLES.iter().cloned());

    let document = read_document(source)?;

    let comments = document.comments.iter().map(|c| (c.comment_id.clone(), c.clone())).collect();
    let mut converter = DocumentConverter {
        style_map: &style_map,
        id_prefix: String::new(),
        ignore_empty_paragraphs: true,
        note_references: Vec::new(),
        referenced_comments: Vec::new(),
        comments,
        package: source.package,
    };
    let nodes = converter.visit_document(&document)?;

    let mut out = String::new();
    write_html(&collapse(strip_empty(nodes)), &mut out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_style_map_parses_every_line() {
        let expected = DEFAULT_STYLE_MAP
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .count();
        assert_eq!(DEFAULT_STYLES.len(), expected);
        let list = DEFAULT_STYLES
            .iter()
            .find(|s| matches!(&s.document_matcher, DocumentMatcher::Paragraph { numbering: Some(n), .. } if n.level_index == "1" && !n.is_ordered))
            .unwrap();
        match &list.html_path {
            HtmlPath::Elements(elements) => {
                let names: Vec<Vec<String>> = elements.iter().map(|e| e.tag_names.clone()).collect();
                assert_eq!(names, vec![vec!["ul".to_string(), "ol".to_string()], vec!["li".to_string()], vec!["ul".to_string()], vec!["li".to_string()]]);
                assert!(!elements[3].collapsible);
                assert!(elements[0].collapsible);
            }
            HtmlPath::Ignore => panic!("unexpected ignore"),
        }
    }

    #[test]
    fn collapses_adjacent_list_items() {
        let ul = |text: &str| {
            HtmlNode::Element(HtmlElement {
                tag: HtmlTag::new(&["ul"], BTreeMap::new(), true),
                children: vec![HtmlNode::Element(HtmlElement {
                    tag: HtmlTag::new(&["li"], BTreeMap::new(), false),
                    children: vec![HtmlNode::Text(text.to_string())],
                })],
            })
        };
        let mut out = String::new();
        write_html(&collapse(strip_empty(vec![ul("a"), ul("b")])), &mut out);
        assert_eq!(out, "<ul><li>a</li><li>b</li></ul>");
    }
}
