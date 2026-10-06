//! PPTX → Markdown: hand-port of MarkItDown's `PptxConverter` over a port of
//! the parts of `python-pptx` it reads (slides, shape trees, placeholders
//! with inherited positions, text frames, tables, charts, pictures and notes).

use std::collections::HashMap;

use once_cell::sync::Lazy;
use regex::Regex;

use super::html::{convert_html, html_escape};
use super::ooxml::{OoxmlPackage, Relationship, XmlElement, XmlNode};
use super::{llm_caption, ConvertOptions, DocumentConverter, DocumentConverterResult, StreamInfo};

const GRAPHIC_DATA_URI_CHART: &str = "http://schemas.openxmlformats.org/drawingml/2006/chart";
const GRAPHIC_DATA_URI_TABLE: &str = "http://schemas.openxmlformats.org/drawingml/2006/table";
const GRAPHIC_DATA_URI_OLEOBJ: &str = "http://schemas.openxmlformats.org/presentationml/2006/ole";

/// Elements of a shape tree that are shapes (`CT_GroupShape._shape_tags`).
const SHAPE_TAGS: &[&str] = &["p:sp", "p:grpSp", "p:graphicFrame", "p:cxnSp", "p:pic", "p:contentPart"];

static RE_ALT_SPECIAL: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\r\n\[\]]").expect("valid regex"));
static RE_WHITESPACE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").expect("valid regex"));
static RE_NON_WORD: Lazy<Regex> = Lazy::new(|| Regex::new(r"\W").expect("valid regex"));

/// Converts PPTX files to Markdown. Supports heading, tables and images with
/// alt text.
pub struct PptxConverter;

impl DocumentConverter for PptxConverter {
    fn name(&self) -> &'static str {
        "PptxConverter"
    }

    fn accepts(&self, info: &StreamInfo) -> bool {
        info.matches(&[".pptx"], &["application/vnd.openxmlformats-officedocument.presentationml"])
    }

    fn convert(&self, data: &[u8], _info: &StreamInfo, options: &ConvertOptions) -> anyhow::Result<DocumentConverterResult> {
        let package = OoxmlPackage::from_bytes(data)?;
        let presentation = Presentation::open(&package)?;
        let mut md_content = String::new();

        for (index, slide_part) in presentation.slide_parts()?.iter().enumerate() {
            let slide_num = index + 1;
            md_content.push_str(&format!("\n\n<!-- Slide number: {slide_num} -->\n"));

            let slide = SlideContext::load(&package, slide_part)?;
            let title = slide.title();

            let shapes: Vec<Shape<'_>> = shape_elements(slide.sp_tree()?)
                .into_iter()
                .map(|el| Shape { el, slide_level: true })
                .collect();
            for shape in sort_shapes(&slide, shapes)? {
                get_shape_content(&slide, shape, title, options, &mut md_content)?;
            }

            md_content = md_content.trim().to_string();

            if let Some(notes_part) = slide.related_part("/notesSlide") {
                md_content.push_str("\n\n### Notes:\n");
                if let Some(text) = notes_text(&package, &notes_part)? {
                    md_content.push_str(&text);
                }
                md_content = md_content.trim().to_string();
            }
        }

        Ok(DocumentConverterResult::markdown(md_content.trim()))
    }
}

// -- package navigation ---------------------------------------------------------

struct Presentation<'a> {
    package: &'a OoxmlPackage,
    part: String,
}

/// The first relationship of `rels` whose type ends with `type_suffix`.
fn related(rels: &HashMap<String, Relationship>, type_suffix: &str) -> Option<String> {
    let mut matches: Vec<&Relationship> =
        rels.values().filter(|r| !r.external && r.rel_type.ends_with(type_suffix)).collect();
    matches.sort_by(|a, b| a.resolved.cmp(&b.resolved));
    matches.first().map(|r| r.resolved.clone())
}

impl<'a> Presentation<'a> {
    fn open(package: &'a OoxmlPackage) -> anyhow::Result<Self> {
        let rels = package.relationships("");
        let part = related(&rels, "/officeDocument")
            .ok_or_else(|| anyhow::anyhow!("no relationship of type 'officeDocument' in collection"))?;
        Ok(Self { package, part })
    }

    /// Slide parts in `p:sldIdLst` order.
    fn slide_parts(&self) -> anyhow::Result<Vec<String>> {
        let root = self.package.read_xml(&self.part)?;
        let rels = self.package.relationships(&self.part);
        let mut parts = Vec::new();
        if let Some(list) = root.child("p:sldIdLst") {
            for sld_id in list.children_named("p:sldId") {
                let r_id = sld_id.attr("r:id").ok_or_else(|| anyhow::anyhow!("'r:id'"))?;
                let rel = rels.get(r_id).ok_or_else(|| anyhow::anyhow!("'{r_id}'"))?;
                parts.push(rel.resolved.clone());
            }
        }
        Ok(parts)
    }
}

/// A slide part with its relationships and parsed XML.
struct SlideContext<'a> {
    package: &'a OoxmlPackage,
    root: XmlElement,
    rels: HashMap<String, Relationship>,
}

impl<'a> SlideContext<'a> {
    fn load(package: &'a OoxmlPackage, part: &str) -> anyhow::Result<Self> {
        Ok(Self {
            package,
            root: package.read_xml(part)?,
            rels: package.relationships(part),
        })
    }

    fn sp_tree(&self) -> anyhow::Result<&XmlElement> {
        self.root.path(&["p:cSld", "p:spTree"]).ok_or_else(|| anyhow::anyhow!("slide has no p:spTree"))
    }

    fn related_part(&self, type_suffix: &str) -> Option<String> {
        related(&self.rels, type_suffix)
    }

    /// `slide.shapes.title`: the first placeholder with `idx == 0`.
    fn title(&self) -> Option<&XmlElement> {
        let tree = self.sp_tree().ok()?;
        shape_elements(tree).into_iter().find(|el| ph(el).is_some_and(|p| ph_idx(p) == 0))
    }

    /// The bytes and part name of the part related by `r_id`.
    fn related_blob(&self, r_id: &str) -> anyhow::Result<(String, &'a [u8])> {
        let rel = self.rels.get(r_id).filter(|r| !r.external).ok_or_else(|| anyhow::anyhow!("'{r_id}'"))?;
        let blob = self
            .package
            .read(&rel.resolved)
            .ok_or_else(|| anyhow::anyhow!("There is no item named '{}' in the archive", rel.resolved))?;
        Ok((rel.resolved.clone(), blob))
    }
}

/// Shape elements of a shape tree, in document order.
fn shape_elements(tree: &XmlElement) -> Vec<&XmlElement> {
    tree.elements().filter(|e| SHAPE_TAGS.contains(&e.name.as_str())).collect()
}

/// `./*[1]/p:nvPr/p:ph`.
fn ph(el: &XmlElement) -> Option<&XmlElement> {
    el.elements().next()?.child("p:nvPr")?.child("p:ph")
}

fn ph_idx(ph: &XmlElement) -> i64 {
    ph.attr("idx").and_then(|v| v.trim().parse().ok()).unwrap_or(0)
}

fn ph_type(ph: &XmlElement) -> &str {
    ph.attr("type").unwrap_or("obj")
}

/// `cNvPr.name` of a shape.
fn shape_name(el: &XmlElement) -> String {
    el.elements()
        .next()
        .and_then(|nv| nv.child("p:cNvPr"))
        .and_then(|c| c.attr("name"))
        .unwrap_or("")
        .to_string()
}

/// The directly-applied `(x, y)` offset of a shape element, in EMU.
fn own_offset(el: &XmlElement) -> anyhow::Result<(Option<i64>, Option<i64>)> {
    let xfrm = match el.name.as_str() {
        "p:sp" | "p:cxnSp" | "p:pic" => el.path(&["p:spPr", "a:xfrm"]),
        "p:grpSp" => el.path(&["p:grpSpPr", "a:xfrm"]),
        "p:graphicFrame" => el.child("p:xfrm"),
        _ => None,
    };
    let off = xfrm.and_then(|x| x.child("a:off"));
    let parse = |name: &str| -> anyhow::Result<Option<i64>> {
        match off.and_then(|o| o.attr(name)) {
            None => Ok(None),
            Some(v) => v
                .trim()
                .parse::<i64>()
                .map(Some)
                .map_err(|_| anyhow::anyhow!("invalid literal for int() with base 10: '{v}'")),
        }
    };
    Ok((parse("x")?, parse("y")?))
}

/// A shape plus whether it sits directly on the slide (placeholders there
/// inherit positions from the layout; shapes inside groups do not).
#[derive(Clone, Copy)]
struct Shape<'s> {
    el: &'s XmlElement,
    slide_level: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShapeType {
    Placeholder,
    Picture,
    Media,
    Freeform,
    AutoShape,
    TextBox,
    Chart,
    Table,
    OleObject,
    Group,
    Line,
}

fn graphic_data_uri(el: &XmlElement) -> anyhow::Result<&str> {
    el.path(&["a:graphic", "a:graphicData"])
        .and_then(|g| g.attr("uri"))
        .ok_or_else(|| anyhow::anyhow!("graphicFrame has no a:graphic/a:graphicData/@uri"))
}

impl<'s> Shape<'s> {
    fn is_slide_placeholder(&self) -> bool {
        self.slide_level && matches!(self.el.name.as_str(), "p:sp" | "p:pic") && ph(self.el).is_some()
    }

    /// `shape.shape_type`; `None` for a graphic frame holding something else
    /// (e.g. SmartArt).
    fn shape_type(&self) -> anyhow::Result<Option<ShapeType>> {
        let el = self.el;
        Ok(Some(match el.name.as_str() {
            "p:pic" => {
                if self.slide_level && ph(el).is_some() {
                    ShapeType::Placeholder
                } else if el.path(&["p:nvPicPr", "p:nvPr", "a:videoFile"]).is_some() {
                    ShapeType::Media
                } else {
                    ShapeType::Picture
                }
            }
            "p:sp" => {
                if ph(el).is_some() {
                    ShapeType::Placeholder
                } else {
                    let sp_pr = el.child("p:spPr").ok_or_else(|| anyhow::anyhow!("required p:spPr element missing"))?;
                    let tx_box = matches!(
                        el.path(&["p:nvSpPr", "p:cNvSpPr"]).and_then(|c| c.attr("txBox")),
                        Some("1") | Some("true")
                    );
                    if sp_pr.child("a:custGeom").is_some() {
                        ShapeType::Freeform
                    } else if sp_pr.child("a:prstGeom").is_some() && !tx_box {
                        ShapeType::AutoShape
                    } else if tx_box {
                        ShapeType::TextBox
                    } else {
                        anyhow::bail!("Shape instance of unrecognized shape type");
                    }
                }
            }
            "p:graphicFrame" => match graphic_data_uri(el)? {
                GRAPHIC_DATA_URI_CHART => ShapeType::Chart,
                GRAPHIC_DATA_URI_TABLE => ShapeType::Table,
                GRAPHIC_DATA_URI_OLEOBJ => ShapeType::OleObject,
                _ => return Ok(None),
            },
            "p:grpSp" => ShapeType::Group,
            "p:cxnSp" => ShapeType::Line,
            _ => anyhow::bail!("BaseShape does not implement `.shape_type`"),
        }))
    }

    fn has_chart(&self) -> anyhow::Result<bool> {
        Ok(self.el.name == "p:graphicFrame" && graphic_data_uri(self.el)? == GRAPHIC_DATA_URI_CHART)
    }

    fn has_text_frame(&self) -> bool {
        self.el.name == "p:sp"
    }

    /// The effective `(left, top)`: directly applied, else inherited from the
    /// layout placeholder (for slide placeholders).
    fn position(&self, slide: &SlideContext<'_>) -> anyhow::Result<(Option<i64>, Option<i64>)> {
        let (mut x, mut y) = own_offset(self.el)?;
        if self.is_slide_placeholder() && (x.is_none() || y.is_none()) {
            let idx = ph(self.el).map(ph_idx).unwrap_or(0);
            let (bx, by) = layout_placeholder_position(slide, idx)?;
            x = x.or(bx);
            y = y.or(by);
        }
        Ok((x, y))
    }
}

/// Position of the slide layout placeholder with `idx` (inheriting from the
/// master placeholder of the matching type when not set on the layout).
fn layout_placeholder_position(slide: &SlideContext<'_>, idx: i64) -> anyhow::Result<(Option<i64>, Option<i64>)> {
    let Some(layout_part) = slide.related_part("/slideLayout") else {
        return Ok((None, None));
    };
    let layout = SlideContext::load(slide.package, &layout_part)?;
    let tree = layout.sp_tree()?;
    let Some(layout_ph) = shape_elements(tree).into_iter().find(|el| ph(el).is_some_and(|p| ph_idx(p) == idx)) else {
        return Ok((None, None));
    };
    let (mut x, mut y) = own_offset(layout_ph)?;
    // Only `p:sp` layout placeholders inherit from the master.
    if layout_ph.name == "p:sp" && (x.is_none() || y.is_none()) {
        let layout_type = ph(layout_ph).map(ph_type).unwrap_or("obj");
        let base_ph_type = match layout_type {
            "body" | "chart" | "clipArt" | "dgm" | "media" | "obj" | "pic" | "subTitle" | "tbl" => "body",
            "ctrTitle" | "title" => "title",
            "dt" => "dt",
            "ftr" => "ftr",
            "sldNum" => "sldNum",
            other => anyhow::bail!("{other}"),
        };
        if let Some(master_part) = layout.related_part("/slideMaster") {
            let master = SlideContext::load(slide.package, &master_part)?;
            let master_tree = master.sp_tree()?;
            let master_ph = shape_elements(master_tree)
                .into_iter()
                .find(|el| el.name == "p:sp" && ph(el).is_some_and(|p| ph_type(p) == base_ph_type));
            if let Some(master_ph) = master_ph {
                let (mx, my) = own_offset(master_ph)?;
                x = x.or(mx);
                y = y.or(my);
            }
        }
    }
    Ok((x, y))
}

/// Sort shapes by `(top, left)`, treating a missing or zero coordinate as
/// negative infinity (stable).
fn sort_shapes<'s>(slide: &SlideContext<'_>, shapes: Vec<Shape<'s>>) -> anyhow::Result<Vec<Shape<'s>>> {
    let mut keyed = Vec::with_capacity(shapes.len());
    for shape in shapes {
        let (left, top) = shape.position(slide)?;
        let key = |v: Option<i64>| match v {
            None | Some(0) => f64::NEG_INFINITY,
            Some(v) => v as f64,
        };
        keyed.push(((key(top), key(left)), shape));
    }
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(keyed.into_iter().map(|(_, s)| s).collect())
}

/// Text of the leading text nodes of an element (lxml `.text`).
fn element_text(el: &XmlElement) -> String {
    el.children
        .iter()
        .map_while(|c| match c {
            XmlNode::Text(t) => Some(t.as_str()),
            XmlNode::Element(_) => None,
        })
        .collect()
}

/// `TextFrame.text`: paragraphs joined by "\n"; within a paragraph runs and
/// fields contribute their text and line breaks a vertical tab.
fn text_frame_text(tx_body: Option<&XmlElement>) -> String {
    let Some(tx_body) = tx_body else { return String::new() };
    tx_body
        .children_named("a:p")
        .map(|p| {
            p.elements()
                .map(|e| match e.name.as_str() {
                    "a:r" | "a:fld" => e.child("a:t").map(element_text).unwrap_or_default(),
                    "a:br" => "\u{b}".to_string(),
                    _ => String::new(),
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The text of the notes slide's body placeholder, if it has one.
fn notes_text(package: &OoxmlPackage, notes_part: &str) -> anyhow::Result<Option<String>> {
    let notes = SlideContext::load(package, notes_part)?;
    let tree = notes.sp_tree()?;
    let body = shape_elements(tree).into_iter().find(|el| ph(el).is_some_and(|p| ph_type(p) == "body"));
    Ok(body.map(|el| text_frame_text(el.child("p:txBody"))))
}

/// PIL image format of an image blob (the formats python-pptx maps).
fn pil_format(blob: &[u8]) -> Option<&'static str> {
    if blob.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("PNG")
    } else if blob.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("JPEG")
    } else if blob.starts_with(b"GIF87a") || blob.starts_with(b"GIF89a") {
        Some("GIF")
    } else if blob.starts_with(b"BM") {
        Some("BMP")
    } else if blob.starts_with(b"II*\0") || blob.starts_with(b"MM\0*") {
        Some("TIFF")
    } else if blob.starts_with(&[0xD7, 0xCD, 0xC6, 0x9A, 0x00, 0x00]) || blob.starts_with(&[0x01, 0x00, 0x00, 0x00]) {
        Some("WMF")
    } else if blob.len() >= 12 && &blob[..4] == b"RIFF" && &blob[8..12] == b"WEBP" {
        Some("WEBP")
    } else {
        None
    }
}

/// `pptx.image.Image.content_type`.
fn image_content_type(blob: &[u8]) -> anyhow::Result<&'static str> {
    match pil_format(blob) {
        Some("BMP") => Ok("image/bmp"),
        Some("GIF") => Ok("image/gif"),
        Some("JPEG") => Ok("image/jpeg"),
        Some("PNG") => Ok("image/png"),
        Some("TIFF") => Ok("image/tiff"),
        Some("WMF") => Ok("image/x-wmf"),
        Some(other) => anyhow::bail!(
            "unsupported image format, expected one of: dict_keys(['BMP', 'GIF', 'JPEG', 'PNG', 'TIFF', 'WMF']), got '{other}'"
        ),
        None => anyhow::bail!("cannot identify image file"),
    }
}

/// `shape.image` of a picture: `(filename, blob)`.
fn picture_image<'a>(slide: &SlideContext<'a>, el: &XmlElement) -> anyhow::Result<(String, &'a [u8])> {
    let blip_fill = el.child("p:blipFill").ok_or_else(|| anyhow::anyhow!("required p:blipFill element missing"))?;
    let Some(r_id) = blip_fill.child("a:blip").and_then(|b| b.attr("r:embed")) else {
        anyhow::bail!("no embedded image");
    };
    let (part_name, blob) = slide.related_blob(r_id)?;
    let ext = std::path::Path::new(&part_name).extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
    Ok((format!("image.{ext}"), blob))
}

/// `PptxConverter._is_picture`.
fn is_picture(slide: &SlideContext<'_>, shape: &Shape<'_>) -> anyhow::Result<bool> {
    match shape.shape_type()? {
        Some(ShapeType::Picture) => Ok(true),
        Some(ShapeType::Placeholder) => {
            // `hasattr(shape, "image")`: only picture placeholders have an
            // image, and evaluating it may raise.
            if shape.el.name == "p:pic" {
                picture_image(slide, shape.el)?;
                Ok(true)
            } else {
                Ok(false)
            }
        }
        _ => Ok(false),
    }
}

fn get_shape_content(
    slide: &SlideContext<'_>,
    shape: Shape<'_>,
    title: Option<&XmlElement>,
    options: &ConvertOptions,
    md_content: &mut String,
) -> anyhow::Result<()> {
    // Pictures
    if is_picture(slide, &shape)? {
        let mut llm_description = String::new();
        let mut alt_text = String::new();

        // Potentially generate a description using an LLM
        if let (Some(client), Some(model)) = (options.llm_client.as_ref(), options.llm_model.as_ref()) {
            // Prepare the stream info for the image data
            let (image_filename, blob) = picture_image(slide, shape.el)?;
            let image_extension = std::path::Path::new(&image_filename)
                .extension()
                .map(|e| format!(".{}", e.to_string_lossy()))
                .unwrap_or_default();
            let image_stream_info = StreamInfo {
                mimetype: Some(image_content_type(blob)?.to_string()),
                extension: Some(image_extension),
                filename: Some(image_filename),
                ..StreamInfo::default()
            };

            // Caption the image; failures leave the description empty.
            if let Ok(Some(description)) =
                llm_caption(blob, &image_stream_info, client.as_ref(), model, options.llm_prompt.as_deref())
            {
                llm_description = description;
            }
        }

        // Also grab any description embedded in the deck
        if let Some(descr) = shape.el.elements().next().and_then(|nv| nv.child("p:cNvPr")).and_then(|c| c.attr("descr")) {
            alt_text = descr.to_string();
        }

        // Prepare the alt, escaping any special characters
        let joined = format!("{llm_description}\n{alt_text}");
        let alt_text = if joined.is_empty() { shape_name(shape.el) } else { joined };
        let alt_text = RE_ALT_SPECIAL.replace_all(&alt_text, " ");
        let alt_text = RE_WHITESPACE.replace_all(&alt_text, " ").trim().to_string();

        // A placeholder name
        let filename = format!("{}.jpg", RE_NON_WORD.replace_all(&shape_name(shape.el), ""));
        md_content.push_str(&format!("\n![{alt_text}]({filename})\n"));
    }

    // Tables
    if shape.shape_type()? == Some(ShapeType::Table) {
        md_content.push_str(&convert_table_to_markdown(shape.el));
    }

    // Charts
    if shape.has_chart()? {
        md_content.push_str(&convert_chart_to_markdown(slide, shape.el)?);
    }
    // Text areas
    else if shape.has_text_frame() {
        let text = text_frame_text(shape.el.child("p:txBody"));
        if title.is_some_and(|t| std::ptr::eq(t, shape.el)) {
            md_content.push_str(&format!("# {}\n", text.trim_start()));
        } else {
            md_content.push_str(&format!("{text}\n"));
        }
    }

    // Group Shapes
    if shape.shape_type()? == Some(ShapeType::Group) {
        let members: Vec<Shape<'_>> =
            shape_elements(shape.el).into_iter().map(|el| Shape { el, slide_level: false }).collect();
        for subshape in sort_shapes(slide, members)? {
            get_shape_content(slide, subshape, title, options, md_content)?;
        }
    }
    Ok(())
}

/// Write the table as HTML, then convert it to Markdown.
fn convert_table_to_markdown(graphic_frame: &XmlElement) -> String {
    let mut html_table = String::from("<html><body><table>");
    if let Some(tbl) = graphic_frame.path(&["a:graphic", "a:graphicData", "a:tbl"]) {
        let mut first_row = true;
        for row in tbl.children_named("a:tr") {
            html_table.push_str("<tr>");
            for cell in row.children_named("a:tc") {
                let text = html_escape(&text_frame_text(cell.child("a:txBody")));
                if first_row {
                    html_table.push_str(&format!("<th>{text}</th>"));
                } else {
                    html_table.push_str(&format!("<td>{text}</td>"));
                }
            }
            html_table.push_str("</tr>");
            first_row = false;
        }
    }
    html_table.push_str("</table></body></html>");
    format!("{}\n", convert_html(&html_table).text_content.trim())
}

// -- charts -----------------------------------------------------------------------

/// Why a chart could not be rendered.
enum ChartError {
    /// A `ValueError("unsupported plot type ...")`, or any non-`ValueError`
    /// failure: rendered as "[unsupported chart]".
    Unsupported,
    /// Any other `ValueError`: MarkItDown's handler then returns `None`, and
    /// appending it to the Markdown fails the conversion.
    Value(String),
}

const SUPPORTED_PLOTS: &[&str] = &[
    "c:areaChart",
    "c:area3DChart",
    "c:barChart",
    "c:bubbleChart",
    "c:doughnutChart",
    "c:lineChart",
    "c:pieChart",
    "c:radarChart",
    "c:scatterChart",
];

const PLOT_TAGS: &[&str] = &[
    "c:area3DChart",
    "c:areaChart",
    "c:bar3DChart",
    "c:barChart",
    "c:bubbleChart",
    "c:doughnutChart",
    "c:line3DChart",
    "c:lineChart",
    "c:ofPieChart",
    "c:pie3DChart",
    "c:pieChart",
    "c:radarChart",
    "c:scatterChart",
    "c:stockChart",
    "c:surface3DChart",
    "c:surfaceChart",
];

/// Value cells of the chart table (Python `str()` of each).
enum Cell {
    Text(String),
    Number(Option<f64>),
}

impl Cell {
    fn py_str(&self) -> String {
        match self {
            Cell::Text(s) => s.clone(),
            Cell::Number(Some(v)) => crate::parsers::json_parser::py_float_repr(*v),
            Cell::Number(None) => "None".to_string(),
        }
    }
}

fn convert_chart_to_markdown(slide: &SlideContext<'_>, graphic_frame: &XmlElement) -> anyhow::Result<String> {
    match chart_markdown(slide, graphic_frame) {
        Ok(md) => Ok(md),
        Err(ChartError::Unsupported) => Ok("\n\n[unsupported chart]\n\n".to_string()),
        Err(ChartError::Value(message)) => {
            anyhow::bail!("can only concatenate str (not \"NoneType\") to str (chart error: {message})")
        }
    }
}

/// `c:pt[@idx=idx]` descendants' first match.
fn find_pt(el: &XmlElement, idx: i64) -> Option<&XmlElement> {
    el.descendants("c:pt").into_iter().find(|pt| pt.attr("idx").and_then(|v| v.trim().parse::<f64>().ok()) == Some(idx as f64))
}

fn chart_markdown(slide: &SlideContext<'_>, graphic_frame: &XmlElement) -> Result<String, ChartError> {
    let unsupported = |_| ChartError::Unsupported;
    let r_id = graphic_frame
        .path(&["a:graphic", "a:graphicData", "c:chart"])
        .and_then(|c| c.attr("r:id"))
        .ok_or(ChartError::Unsupported)?;
    let (part_name, _) = slide.related_blob(r_id).map_err(unsupported)?;
    let chart_space = slide.package.read_xml(&part_name).map_err(unsupported)?;
    let chart = chart_space.child("c:chart").ok_or(ChartError::Unsupported)?;

    let mut md = String::from("\n\n### Chart");
    if let Some(title) = chart.child("c:title") {
        let text = text_frame_text(title.path(&["c:tx", "c:rich"]));
        md.push_str(&format!(": {text}"));
    }
    md.push_str("\n\n");

    let plot_area = chart.child("c:plotArea").ok_or(ChartError::Unsupported)?;
    let x_charts: Vec<&XmlElement> = plot_area.elements().filter(|e| PLOT_TAGS.contains(&e.name.as_str())).collect();

    // chart.plots[0].categories
    let first_plot = x_charts.first().ok_or(ChartError::Unsupported)?;
    if !SUPPORTED_PLOTS.contains(&first_plot.name.as_str()) {
        return Err(ChartError::Unsupported);
    }
    let category_names = category_labels(first_plot)?;

    // chart.series: every plot's series in c:order sequence.
    let mut series: Vec<&XmlElement> = Vec::new();
    for x_chart in &x_charts {
        let mut sers: Vec<(i64, &XmlElement)> = Vec::new();
        for ser in x_chart.children_named("c:ser") {
            let order = ser
                .child("c:order")
                .and_then(|o| o.attr("val"))
                .and_then(|v| v.trim().parse::<i64>().ok())
                .ok_or(ChartError::Unsupported)?;
            sers.push((order, ser));
        }
        sers.sort_by_key(|(order, _)| *order);
        series.extend(sers.into_iter().map(|(_, s)| s));
    }

    let mut series_names = Vec::new();
    let mut series_values = Vec::new();
    for ser in &series {
        let parent_tag = x_charts
            .iter()
            .find(|x| x.children_named("c:ser").any(|s| std::ptr::eq(s, *ser)))
            .map(|x| x.name.as_str())
            .unwrap_or("");
        let values_tag = match parent_tag {
            "c:areaChart" | "c:barChart" | "c:doughnutChart" | "c:lineChart" | "c:pieChart" | "c:radarChart" => "c:val",
            "c:bubbleChart" | "c:scatterChart" => "c:yVal",
            // series class not yet implemented
            _ => return Err(ChartError::Unsupported),
        };
        let name = ser
            .child("c:tx")
            .map(|tx| {
                tx.descendants("c:pt")
                    .into_iter()
                    .filter_map(|pt| pt.child("c:v"))
                    .map(element_text)
                    .find(|t| !t.is_empty())
                    .unwrap_or_default()
            })
            .unwrap_or_default();
        series_names.push(name);
        series_values.push(series_point_values(ser.child(values_tag))?);
    }

    let mut data: Vec<Vec<Cell>> = Vec::new();
    let mut header = vec![Cell::Text("Category".to_string())];
    header.extend(series_names.into_iter().map(Cell::Text));
    data.push(header);

    for (idx, category) in category_names.into_iter().enumerate() {
        let mut row = vec![Cell::Text(category)];
        for values in &series_values {
            let value = values.get(idx).ok_or(ChartError::Unsupported)?;
            row.push(Cell::Number(*value));
        }
        data.push(row);
    }

    let markdown_table: Vec<String> = data
        .iter()
        .map(|row| format!("| {} |", row.iter().map(Cell::py_str).collect::<Vec<_>>().join(" | ")))
        .collect();
    let separator = format!("|{}|", vec!["---"; data[0].len()].join("|"));
    let mut lines = vec![markdown_table[0].clone(), separator];
    lines.extend(markdown_table[1..].iter().cloned());
    Ok(md + &lines.join("\n"))
}

/// Category labels of a plot (from the first series' `c:cat`).
fn category_labels(x_chart: &XmlElement) -> Result<Vec<String>, ChartError> {
    let count = x_chart
        .children_named("c:ser")
        .filter_map(|ser| ser.child("c:cat"))
        .flat_map(|cat| cat.descendants("c:ptCount"))
        .next()
        .map(|pc| pc.attr("val").and_then(|v| v.trim().parse::<i64>().ok()).ok_or(ChartError::Unsupported))
        .transpose()?
        .unwrap_or(0);

    let cat = x_chart.children_named("c:ser").next().and_then(|ser| ser.child("c:cat"));
    let mut pts: Vec<&XmlElement> = Vec::new();
    if let Some(cat) = cat {
        // `c:cat//c:lvl[1]/c:pt`: only the first level of multi-level categories.
        let mut scopes = vec![cat];
        scopes.extend(cat.descendants_all());
        for scope in scopes {
            if let Some(first_lvl) = scope.children_named("c:lvl").next() {
                pts.extend(first_lvl.children_named("c:pt"));
            }
        }
        if pts.is_empty() {
            pts = cat.descendants("c:pt");
        }
    }

    let mut by_idx: HashMap<i64, &XmlElement> = HashMap::new();
    for pt in pts {
        let idx = pt.attr("idx").and_then(|v| v.trim().parse::<i64>().ok()).ok_or(ChartError::Unsupported)?;
        by_idx.insert(idx, pt);
    }

    let mut labels = Vec::new();
    for idx in 0..count {
        labels.push(match by_idx.get(&idx) {
            None => String::new(),
            Some(pt) => {
                let v = pt.child("c:v").ok_or(ChartError::Unsupported)?;
                let has_text = v.children.iter().any(|c| matches!(c, XmlNode::Text(_)));
                if has_text { element_text(v) } else { "None".to_string() }
            }
        });
    }
    Ok(labels)
}

/// `series.values`: one float (or `None`) per point of `ptCount`.
fn series_point_values(val: Option<&XmlElement>) -> Result<Vec<Option<f64>>, ChartError> {
    let Some(val) = val else { return Ok(Vec::new()) };
    let count = val
        .descendants("c:ptCount")
        .first()
        .and_then(|pc| pc.attr("val"))
        .map(|v| v.trim().parse::<i64>().map_err(|_| ChartError::Value(format!("invalid literal for int() with base 10: '{v}'"))))
        .transpose()?
        .unwrap_or(0);
    let mut values = Vec::new();
    for idx in 0..count {
        values.push(match find_pt(val, idx) {
            None => None,
            Some(pt) => {
                let v = pt.child("c:v").ok_or(ChartError::Unsupported)?;
                let text = element_text(v);
                Some(
                    text.trim()
                        .parse::<f64>()
                        .map_err(|_| ChartError::Value(format!("could not convert string to float: '{text}'")))?,
                )
            }
        });
    }
    Ok(values)
}
