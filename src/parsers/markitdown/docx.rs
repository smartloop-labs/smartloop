//! DOCX → Markdown: hand-port of MarkItDown's `DocxConverter` and
//! `converter_utils/docx/pre_process.py`.
//!
//! The DOCX is pre-processed (OMML equations in the document, footnotes and
//! endnotes are replaced by runs holding their LaTeX), converted to HTML by
//! the [`mammoth`](super::mammoth) port, and the HTML converted to Markdown
//! by the [`html`](super::html) converter.

use std::collections::HashMap;

use super::html::convert_html;
use super::mammoth::{self, DocxSource};
use super::omml::omath_to_latex;
use super::ooxml::{parse_xml, OoxmlPackage, XmlElement, XmlNode};
use super::{ConvertOptions, DocumentConverter, DocumentConverterResult, StreamInfo};

/// Converts DOCX files to Markdown. Style information (e.g. headings) and
/// tables are preserved where possible.
pub struct DocxConverter;

impl DocumentConverter for DocxConverter {
    fn name(&self) -> &'static str {
        "DocxConverter"
    }

    fn accepts(&self, info: &StreamInfo) -> bool {
        info.matches(&[".docx"], &["application/vnd.openxmlformats-officedocument.wordprocessingml.document"])
    }

    fn convert(&self, data: &[u8], _info: &StreamInfo, _options: &ConvertOptions) -> anyhow::Result<DocumentConverterResult> {
        let package = OoxmlPackage::from_bytes(data)?;
        let source = pre_process_docx(&package);
        let html = mammoth::convert_to_html(&source)?;
        Ok(convert_html(&html))
    }
}

/// The parts whose equations are pre-processed.
const PRE_PROCESS_ENABLE_FILES: &[&str] = &["word/document.xml", "word/footnotes.xml", "word/endnotes.xml"];

/// Pre-process a DOCX package: OMML equations in the main document,
/// footnotes and endnotes are converted to LaTeX. A part whose processing
/// fails is used unchanged.
pub fn pre_process_docx(package: &OoxmlPackage) -> DocxSource<'_> {
    let mut parsed_overrides = HashMap::new();
    for name in PRE_PROCESS_ENABLE_FILES {
        let Some(content) = package.read_exact(name) else { continue };
        // If there is an error in processing the content, the original
        // content is kept.
        if let Ok(updated) = pre_process_math(content) {
            parsed_overrides.insert(name.to_string(), updated);
        }
    }
    DocxSource { package, parsed_overrides }
}

/// Convert OMML elements of an XML part to their LaTeX equivalents:
/// `oMathPara` blocks become paragraphs of `$$...$$` runs, remaining inline
/// `oMath` elements become `$...$` runs.
pub fn pre_process_math(content: &[u8]) -> anyhow::Result<XmlElement> {
    std::str::from_utf8(content).map_err(|e| anyhow::anyhow!("'utf-8' codec can't decode bytes: {e}"))?;
    let mut root = parse_xml(content)?;
    replace_equations(&mut root, "oMathPara")?;
    replace_equations(&mut root, "oMath")?;
    Ok(root)
}

fn local_name(name: &str) -> &str {
    name.rsplit(':').next().unwrap_or(name)
}

/// Replace every element with local name `tag` (outermost first, in document
/// order) by its LaTeX replacement.
fn replace_equations(el: &mut XmlElement, tag: &str) -> anyhow::Result<()> {
    for node in el.children.iter_mut() {
        if let XmlNode::Element(child) = node {
            if local_name(&child.name) == tag {
                *child = replacement_for(child)?;
            } else {
                replace_equations(child, tag)?;
            }
        }
    }
    Ok(())
}

fn replacement_for(tag: &XmlElement) -> anyhow::Result<XmlElement> {
    match local_name(&tag.name) {
        "oMathPara" => {
            // A new paragraph holding each `oMath` as a block equation.
            let mut p_tag = XmlElement { name: "w:p".to_string(), attrs: Vec::new(), children: Vec::new() };
            for child_tag in descendants_with_local_name(tag, "oMath") {
                p_tag.children.push(XmlNode::Element(omath_tag_replacement(child_tag, true)?));
            }
            Ok(p_tag)
        }
        // An inline equation.
        "oMath" => omath_tag_replacement(tag, false),
        other => anyhow::bail!("Not supported tag: {other}"),
    }
}

fn descendants_with_local_name<'a>(el: &'a XmlElement, name: &str) -> Vec<&'a XmlElement> {
    let mut out = Vec::new();
    fn walk<'a>(el: &'a XmlElement, name: &str, out: &mut Vec<&'a XmlElement>) {
        for child in el.elements() {
            if local_name(&child.name) == name {
                out.push(child);
            }
            walk(child, name, out);
        }
    }
    walk(el, name, &mut out);
    out
}

/// A `w:r` holding the equation's LaTeX in a `w:t`, wrapped in `$$` (block)
/// or `$` (inline).
fn omath_tag_replacement(tag: &XmlElement, block: bool) -> anyhow::Result<XmlElement> {
    let latex = omath_to_latex(tag)?;
    let text = if block { format!("$${latex}$$") } else { format!("${latex}$") };
    let t_tag = XmlElement { name: "w:t".to_string(), attrs: Vec::new(), children: vec![XmlNode::Text(text)] };
    Ok(XmlElement { name: "w:r".to_string(), attrs: Vec::new(), children: vec![XmlNode::Element(t_tag)] })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_inline_and_block_math() {
        let xml = br#"<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><w:body><w:p><w:r><w:t>Area </w:t></w:r><m:oMath><m:r><m:t>x</m:t></m:r></m:oMath></w:p><w:p><m:oMathPara><m:oMath><m:r><m:t>y</m:t></m:r></m:oMath></m:oMathPara></w:p></w:body></w:document>"#;
        let root = pre_process_math(xml).unwrap();
        let texts: Vec<String> = root.descendants("w:t").iter().map(|t| t.inner_text()).collect();
        assert_eq!(texts, vec!["Area ", "$x$", "$$y$$"]);
    }
}
