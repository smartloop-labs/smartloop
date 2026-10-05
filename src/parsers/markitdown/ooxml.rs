//! Office Open XML plumbing shared by the DOCX / PPTX converters (the parts of
//! `python-docx`, `python-pptx` and mammoth's `docx` reader that load a
//! package): zip parts, a small namespace-normalized XML tree, part
//! relationships and `[Content_Types].xml`.
//!
//! Element and attribute names are normalized to the conventional prefixes
//! (`w:`, `a:`, `p:`, `r:`, `c:`, ...) by namespace URI, for both the
//! transitional and strict OOXML namespaces, so lookups do not depend on the
//! prefixes a producer happened to choose.

use std::collections::HashMap;
use std::io::{Cursor, Read};

use quick_xml::events::Event;
use quick_xml::name::ResolveResult;
use quick_xml::reader::NsReader;

/// Namespace URI -> canonical prefix.
const NAMESPACES: &[(&str, &str)] = &[
    // Transitional format
    ("http://schemas.openxmlformats.org/wordprocessingml/2006/main", "w"),
    ("http://schemas.openxmlformats.org/officeDocument/2006/relationships", "r"),
    ("http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing", "wp"),
    ("http://schemas.openxmlformats.org/drawingml/2006/main", "a"),
    ("http://schemas.openxmlformats.org/drawingml/2006/picture", "pic"),
    ("http://schemas.openxmlformats.org/presentationml/2006/main", "p"),
    ("http://schemas.openxmlformats.org/drawingml/2006/chart", "c"),
    ("http://schemas.openxmlformats.org/drawingml/2006/table", "a"),
    ("http://schemas.openxmlformats.org/officeDocument/2006/math", "m"),
    // Strict format
    ("http://purl.oclc.org/ooxml/wordprocessingml/main", "w"),
    ("http://purl.oclc.org/ooxml/officeDocument/relationships", "r"),
    ("http://purl.oclc.org/ooxml/drawingml/wordprocessingDrawing", "wp"),
    ("http://purl.oclc.org/ooxml/drawingml/main", "a"),
    ("http://purl.oclc.org/ooxml/drawingml/picture", "pic"),
    ("http://purl.oclc.org/ooxml/presentationml/main", "p"),
    ("http://purl.oclc.org/ooxml/drawingml/chart", "c"),
    ("http://purl.oclc.org/ooxml/officeDocument/math", "m"),
    // Common
    ("http://schemas.openxmlformats.org/package/2006/content-types", "content-types"),
    ("http://schemas.openxmlformats.org/package/2006/relationships", "relationships"),
    ("http://schemas.openxmlformats.org/markup-compatibility/2006", "mc"),
    ("urn:schemas-microsoft-com:vml", "v"),
    ("urn:schemas-microsoft-com:office:office", "o"),
    ("urn:schemas-microsoft-com:office:word", "office-word"),
    ("http://schemas.microsoft.com/office/word/2010/wordml", "wordml"),
    ("http://www.w3.org/XML/1998/namespace", "xml"),
];

fn canonical_prefix(uri: &[u8]) -> Option<&'static str> {
    NAMESPACES.iter().find(|(u, _)| u.as_bytes() == uri).map(|(_, p)| *p)
}

/// A node of a parsed XML part.
#[derive(Debug, Clone, PartialEq)]
pub enum XmlNode {
    Element(XmlElement),
    Text(String),
}

/// An element with normalized (`prefix:local`) name and attribute names.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct XmlElement {
    pub name: String,
    pub attrs: Vec<(String, String)>,
    pub children: Vec<XmlNode>,
}

impl XmlElement {
    /// Value of the attribute `name` (e.g. `"w:val"`).
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    /// Element children, in document order.
    pub fn elements(&self) -> impl Iterator<Item = &XmlElement> {
        self.children.iter().filter_map(|c| match c {
            XmlNode::Element(e) => Some(e),
            XmlNode::Text(_) => None,
        })
    }

    /// First element child named `name`.
    pub fn child(&self, name: &str) -> Option<&XmlElement> {
        self.elements().find(|e| e.name == name)
    }

    /// Element children named `name`.
    pub fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a XmlElement> + 'a {
        self.elements().filter(move |e| e.name == name)
    }

    /// Follow a chain of first-children (`find_child_or_null` chains).
    pub fn path(&self, names: &[&str]) -> Option<&XmlElement> {
        let mut current = self;
        for name in names {
            current = current.child(name)?;
        }
        Some(current)
    }

    /// Every descendant element named `name`, depth-first in document order
    /// (XPath `.//name`).
    pub fn descendants<'a>(&'a self, name: &str) -> Vec<&'a XmlElement> {
        let mut out = Vec::new();
        fn walk<'a>(el: &'a XmlElement, name: &str, out: &mut Vec<&'a XmlElement>) {
            for child in el.elements() {
                if child.name == name {
                    out.push(child);
                }
                walk(child, name, out);
            }
        }
        walk(self, name, &mut out);
        out
    }

    /// First descendant element named `name`.
    pub fn descendant(&self, name: &str) -> Option<&XmlElement> {
        for child in self.elements() {
            if child.name == name {
                return Some(child);
            }
            if let Some(found) = child.descendant(name) {
                return Some(found);
            }
        }
        None
    }

    /// Concatenated text of every text node below this element.
    pub fn inner_text(&self) -> String {
        let mut out = String::new();
        fn walk(el: &XmlElement, out: &mut String) {
            for child in &el.children {
                match child {
                    XmlNode::Text(t) => out.push_str(t),
                    XmlNode::Element(e) => walk(e, out),
                }
            }
        }
        walk(self, &mut out);
        out
    }
}

/// Normalized `prefix:local` for an element or attribute name.
fn normalized_name(resolved: &ResolveResult<'_>, local: &[u8], raw: &[u8]) -> String {
    let local = String::from_utf8_lossy(local);
    match resolved {
        ResolveResult::Bound(ns) => match canonical_prefix(ns.as_ref()) {
            Some(prefix) => format!("{prefix}:{local}"),
            None => String::from_utf8_lossy(raw).into_owned(),
        },
        ResolveResult::Unbound => local.into_owned(),
        ResolveResult::Unknown(_) => String::from_utf8_lossy(raw).into_owned(),
    }
}

/// Parse an XML part into its root element.
pub fn parse_xml(bytes: &[u8]) -> anyhow::Result<XmlElement> {
    let mut reader = NsReader::from_reader(Cursor::new(bytes));
    let mut buf = Vec::new();
    // Stack of open elements; the bottom is a synthetic document node.
    let mut stack: Vec<XmlElement> = vec![XmlElement::default()];

    loop {
        let (resolved, event) = reader.read_resolved_event_into(&mut buf)?;
        match event {
            Event::Start(ref start) | Event::Empty(ref start) => {
                let qname = start.name();
                let name = normalized_name(&resolved, qname.local_name().as_ref(), qname.as_ref());
                let mut attrs = Vec::new();
                for attr in start.attributes().with_checks(false).flatten() {
                    let key = attr.key;
                    if key.as_ref() == b"xmlns" || key.as_ref().starts_with(b"xmlns:") {
                        continue;
                    }
                    let (attr_ns, attr_local) = reader.resolver().resolve_attribute(key);
                    let attr_name = normalized_name(&attr_ns, attr_local.as_ref(), key.as_ref());
                    let value = attr
                        .unescape_value()
                        .map(|v| v.into_owned())
                        .unwrap_or_else(|_| String::from_utf8_lossy(&attr.value).into_owned());
                    attrs.push((attr_name, value));
                }
                let element = XmlElement { name, attrs, children: Vec::new() };
                if matches!(event, Event::Empty(_)) {
                    if let Some(parent) = stack.last_mut() {
                        parent.children.push(XmlNode::Element(element));
                    }
                } else {
                    stack.push(element);
                }
            }
            Event::End(_) => {
                if stack.len() > 1 {
                    let element = stack.pop().expect("non-empty stack");
                    if let Some(parent) = stack.last_mut() {
                        parent.children.push(XmlNode::Element(element));
                    }
                }
            }
            Event::Text(text) => {
                let decoded = text.xml10_content().map(|t| t.into_owned()).unwrap_or_default();
                push_text(&mut stack, &decoded);
            }
            Event::CData(data) => {
                let decoded = String::from_utf8_lossy(&data.into_inner()).into_owned();
                push_text(&mut stack, &decoded);
            }
            Event::GeneralRef(reference) => {
                let resolved_text = match reference.resolve_char_ref() {
                    Ok(Some(ch)) => ch.to_string(),
                    _ => {
                        let name = reference.decode().map(|n| n.into_owned()).unwrap_or_default();
                        match name.as_str() {
                            "amp" => "&".to_string(),
                            "lt" => "<".to_string(),
                            "gt" => ">".to_string(),
                            "quot" => "\"".to_string(),
                            "apos" => "'".to_string(),
                            other => format!("&{other};"),
                        }
                    }
                };
                push_text(&mut stack, &resolved_text);
            }
            Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }

    // Close anything left open by a truncated document.
    while stack.len() > 1 {
        let element = stack.pop().expect("non-empty stack");
        if let Some(parent) = stack.last_mut() {
            parent.children.push(XmlNode::Element(element));
        }
    }

    let document = stack.pop().unwrap_or_default();
    document
        .children
        .into_iter()
        .find_map(|c| match c {
            XmlNode::Element(e) => Some(e),
            XmlNode::Text(_) => None,
        })
        .ok_or_else(|| anyhow::anyhow!("XML part has no root element"))
}

fn push_text(stack: &mut [XmlElement], text: &str) {
    if text.is_empty() {
        return;
    }
    if let Some(parent) = stack.last_mut() {
        if let Some(XmlNode::Text(prev)) = parent.children.last_mut() {
            prev.push_str(text);
        } else {
            parent.children.push(XmlNode::Text(text.to_string()));
        }
    }
}

/// An OOXML zip package, fully read into memory.
pub struct OoxmlPackage {
    files: HashMap<String, Vec<u8>>,
}

impl OoxmlPackage {
    /// Open a package from its bytes.
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let mut archive = zip::ZipArchive::new(Cursor::new(bytes))?;
        let mut files = HashMap::new();
        for index in 0..archive.len() {
            let mut entry = archive.by_index(index)?;
            if entry.is_dir() {
                continue;
            }
            let name = entry.name().to_string();
            let mut data = Vec::with_capacity(entry.size() as usize);
            entry.read_to_end(&mut data)?;
            files.insert(name, data);
        }
        Ok(Self { files })
    }

    /// Raw bytes of a part (zip entry names have no leading slash).
    pub fn read(&self, name: &str) -> Option<&[u8]> {
        let name = name.trim_start_matches('/');
        self.files.get(name).map(|v| v.as_slice()).or_else(|| {
            // Zip entry names are case-sensitive but OOXML part names are not.
            self.files
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(name))
                .map(|(_, v)| v.as_slice())
        })
    }

    /// True when the part exists.
    pub fn exists(&self, name: &str) -> bool {
        self.read(name).is_some()
    }

    /// Parse a part as XML.
    pub fn read_xml(&self, name: &str) -> anyhow::Result<XmlElement> {
        let bytes = self.read(name).ok_or_else(|| anyhow::anyhow!("There is no item named '{name}' in the archive"))?;
        parse_xml(bytes)
    }

    /// Relationships of `part` (from `<dir>/_rels/<name>.rels`), keyed by id.
    /// Targets of internal relationships are resolved to zip entry names.
    pub fn relationships(&self, part: &str) -> HashMap<String, Relationship> {
        let mut rels = HashMap::new();
        let Ok(root) = self.read_xml(&rels_path_for(part)) else {
            return rels;
        };
        let base_dir = part_dir(part);
        for rel in root.children_named("relationships:Relationship") {
            let (Some(id), Some(target)) = (rel.attr("Id"), rel.attr("Target")) else {
                continue;
            };
            let external = rel.attr("TargetMode") == Some("External");
            let resolved = if external { target.to_string() } else { resolve_part_target(&base_dir, target) };
            rels.insert(
                id.to_string(),
                Relationship {
                    target: target.to_string(),
                    resolved,
                    rel_type: rel.attr("Type").unwrap_or_default().to_string(),
                    external,
                },
            );
        }
        rels
    }

    /// `[Content_Types].xml` of this package.
    pub fn content_types(&self) -> ContentTypes {
        let mut types = ContentTypes::default();
        if let Ok(root) = self.read_xml("[Content_Types].xml") {
            for el in root.elements() {
                match el.name.as_str() {
                    "content-types:Default" => {
                        if let (Some(ext), Some(ct)) = (el.attr("Extension"), el.attr("ContentType")) {
                            types.defaults.insert(ext.to_ascii_lowercase(), ct.to_string());
                        }
                    }
                    "content-types:Override" => {
                        if let (Some(part), Some(ct)) = (el.attr("PartName"), el.attr("ContentType")) {
                            types.overrides.insert(part.trim_start_matches('/').to_ascii_lowercase(), ct.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }
        types
    }
}

/// One package relationship.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Relationship {
    /// The raw `Target` attribute.
    pub target: String,
    /// Zip entry name for internal targets; the raw target for external ones.
    pub resolved: String,
    pub rel_type: String,
    pub external: bool,
}

/// `[Content_Types].xml`: per-extension defaults and per-part overrides.
#[derive(Debug, Clone, Default)]
pub struct ContentTypes {
    defaults: HashMap<String, String>,
    overrides: HashMap<String, String>,
}

impl ContentTypes {
    /// Content type of a part, falling back to well-known image extensions
    /// (mammoth's `content_types.find_content_type`).
    pub fn find(&self, part: &str) -> Option<String> {
        let key = part.trim_start_matches('/').to_ascii_lowercase();
        if let Some(ct) = self.overrides.get(&key) {
            return Some(ct.clone());
        }
        let ext = key.rsplit_once('.').map(|(_, e)| e.to_string()).unwrap_or_default();
        if let Some(ct) = self.defaults.get(&ext) {
            return Some(ct.clone());
        }
        let fallback = match ext.as_str() {
            "png" => "png",
            "gif" => "gif",
            "jpeg" | "jpg" => "jpeg",
            "tif" | "tiff" => "tiff",
            "bmp" => "bmp",
            _ => return None,
        };
        Some(format!("image/{fallback}"))
    }
}

/// `word/document.xml` -> `word/_rels/document.xml.rels`.
pub fn rels_path_for(part: &str) -> String {
    let part = part.trim_start_matches('/');
    match part.rsplit_once('/') {
        Some((dir, name)) => format!("{dir}/_rels/{name}.rels"),
        None => format!("_rels/{part}.rels"),
    }
}

/// Directory of a part (`word/document.xml` -> `word`).
pub fn part_dir(part: &str) -> String {
    part.trim_start_matches('/').rsplit_once('/').map(|(d, _)| d.to_string()).unwrap_or_default()
}

/// Resolve a relationship target against the source part's directory into a
/// zip entry name (handles `..`, `.` and absolute `/` targets).
pub fn resolve_part_target(base_dir: &str, target: &str) -> String {
    let target = target.split('#').next().unwrap_or(target);
    let mut segments: Vec<&str> = if target.starts_with('/') {
        Vec::new()
    } else {
        base_dir.split('/').filter(|s| !s.is_empty()).collect()
    };
    for segment in target.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                segments.pop();
            }
            s => segments.push(s),
        }
    }
    segments.join("/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_with_canonical_prefixes() {
        let xml = br#"<?xml version="1.0"?><x:document xmlns:x="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><x:body><x:p><x:r><x:t xml:space="preserve"> a &amp; b&#33;</x:t></x:r></x:p></x:body></x:document>"#;
        let root = parse_xml(xml).unwrap();
        assert_eq!(root.name, "w:document");
        let t = root.descendant("w:t").unwrap();
        assert_eq!(t.inner_text(), " a & b!");
        assert_eq!(t.attr("xml:space"), Some("preserve"));
    }

    #[test]
    fn resolves_targets() {
        assert_eq!(resolve_part_target("ppt/slides", "../media/image1.png"), "ppt/media/image1.png");
        assert_eq!(resolve_part_target("word", "media/a.png"), "word/media/a.png");
        assert_eq!(resolve_part_target("word", "/word/media/a.png"), "word/media/a.png");
        assert_eq!(rels_path_for("word/document.xml"), "word/_rels/document.xml.rels");
    }
}
