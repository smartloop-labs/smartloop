//! Port of `tests/unit/test_scanned_pdf_vision_ocr.py`.
//!
//! Validate that scanned PDFs and images are OCR'd through the vision model.
//!
//! The document parser extracts text with MarkItDown for normal docs. Images
//! and scanned / image-only PDF pages carry no extractable text, so they are
//! sent to the vision-capable model (selected by
//! `ModelService::get_vision_llm()`) through an in-process client that speaks
//! MarkItDown's OpenAI `llm_client` interface:
//!
//! * images go through MarkItDown's image converter, and
//! * scanned PDF pages are rendered with MuPDF and transcribed page-by-page.
//!
//! These tests assert those paths are exercised — and that they are NOT
//! exercised for text PDFs or when no vision model is available.

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use serial_test::serial;
use smartloop::parsers::document_parser::{
    set_vision_available_override, set_vision_client_override, VisionClient, VisionClientPair, VisionRequest,
};
use smartloop::llms::cloud_llm::local_only;
use smartloop::parsers::parser_factory::ParserFactory;
use smartloop::tasks::indexer_task::parse_for_index;

// --- fakes for the in-process vision client ----------------------------------

/// Records `create()` calls and returns a canned transcription.
struct FakeCompletions {
    content: String,
    calls: Mutex<Vec<VisionRequest>>,
}

impl VisionClient for FakeCompletions {
    fn create(&self, request: VisionRequest) -> anyhow::Result<String> {
        self.calls.lock().push(request);
        Ok(self.content.clone())
    }
}

/// OpenAI-`llm_client`-shaped fake (what MarkItDown / `ocr_page` call).
struct FakeVisionClient {
    completions: Arc<FakeCompletions>,
}

impl FakeVisionClient {
    fn new(content: &str) -> Self {
        Self { completions: Arc::new(FakeCompletions { content: content.to_string(), calls: Mutex::new(Vec::new()) }) }
    }

    fn default_content() -> Self {
        Self::new("# Page\nTranscribed by vision model.")
    }

    fn calls(&self) -> Vec<VisionRequest> {
        self.completions.calls.lock().clone()
    }

    /// `monkeypatch.setattr(dp, "_vision_client", lambda: (fake, "vision"))`.
    fn install(&self) {
        let completions = self.completions.clone();
        set_vision_client_override(Some(Arc::new(move || -> Option<VisionClientPair> {
            Some((completions.clone() as Arc<dyn VisionClient>, "vision".to_string()))
        })));
    }
}

/// Clears the monkeypatched seams when a test ends.
struct ResetOverrides;

impl Drop for ResetOverrides {
    fn drop(&mut self) {
        set_vision_client_override(None);
        set_vision_available_override(None);
    }
}

/// `monkeypatch.setattr(dp, "_vision_available", lambda logger: value)`.
fn set_vision_available(value: bool) {
    set_vision_available_override(Some(Arc::new(move || value)));
}

// --- helpers -----------------------------------------------------------------

/// A minimal PDF: one page per content stream, A4 (595x842) pages, with the
/// standard Helvetica font available as `/helv`.
fn build_pdf(contents: &[String]) -> Vec<u8> {
    let mut objects: Vec<String> = Vec::new();
    let page_count = contents.len();
    // 1: catalog, 2: pages, 3: resources, 4: font, then (page, content) pairs.
    let kids: Vec<String> = (0..page_count).map(|i| format!("{} 0 R", 5 + i * 2)).collect();
    objects.push("<</Type/Catalog/Pages 2 0 R>>".to_string());
    objects.push(format!("<</Type/Pages/Count {page_count}/Kids[{}]>>", kids.join(" ")));
    objects.push("<</Font<</helv 4 0 R>>>>".to_string());
    objects.push("<</Type/Font/Subtype/Type1/BaseFont/Helvetica/Encoding/WinAnsiEncoding>>".to_string());
    for (i, content) in contents.iter().enumerate() {
        let content_id = 6 + i * 2;
        objects.push(format!(
            "<</Type/Page/MediaBox[0 0 595 842]/Rotate 0/Resources 3 0 R/Parent 2 0 R/Contents {content_id} 0 R>>"
        ));
        objects.push(format!("<</Length {}>>\nstream\n{content}\nendstream", content.len()));
    }

    let mut pdf = b"%PDF-1.7\n".to_vec();
    let mut offsets = Vec::new();
    for (i, object) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n{object}\nendobj\n", i + 1).as_bytes());
    }
    let xref_offset = pdf.len();
    pdf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes());
    for offset in offsets {
        pdf.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(
        format!("trailer\n<</Size {}/Root 1 0 R>>\nstartxref\n{xref_offset}\n%%EOF\n", objects.len() + 1).as_bytes(),
    );
    pdf
}

fn write_temp(name: &str, bytes: &[u8]) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).expect("write temp file");
    (dir, path)
}

/// An image-only PDF: filled rectangles, no text layer.
fn scanned_pdf(pages: usize) -> (tempfile::TempDir, PathBuf) {
    // Rect(40, 40, 520, 760) in top-left page coordinates, filled dark grey.
    let content = "q 0 0 0 RG 0.1 0.1 0.1 rg 40 82 480 720 re B Q".to_string();
    write_temp("scanned.pdf", &build_pdf(&vec![content; pages]))
}

/// A text PDF with plenty of extractable text per page.
fn text_pdf(pages: usize) -> (tempfile::TempDir, PathBuf) {
    let contents: Vec<String> = (0..pages)
        .map(|i| {
            // well over the threshold
            let text = format!("Page {i} of the quarterly report. ").repeat(12);
            format!("BT /helv 11 Tf 72 770 Td ({text}) Tj ET")
        })
        .collect();
    write_temp("textual.pdf", &build_pdf(&contents))
}

/// A small standalone PNG image (no text layer — only the vision model can
/// describe it).
fn png() -> (tempfile::TempDir, PathBuf) {
    let mut img = image::RgbImage::from_pixel(200, 200, image::Rgb([255, 255, 255]));
    for y in 20..180 {
        for x in 20..180 {
            img.put_pixel(x, y, image::Rgb([51, 102, 153]));
        }
    }
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png)
        .expect("encode png");
    write_temp("image.png", &bytes)
}

fn parse(mime: &str, path: &std::path::Path) -> String {
    let parser = ParserFactory::new(None).get_parser(mime, path.to_str().expect("utf-8 path")).expect("parser");
    parser.parse().expect("parse")
}

/// The `content` parts of the first message of a vision call.
fn content_parts(call: &VisionRequest) -> Vec<serde_json::Value> {
    call.messages[0]["content"].as_array().cloned().unwrap_or_default()
}

// --- tests -------------------------------------------------------------------

/// Each sparse page is sent to the vision model as an image.
#[test]
#[serial(vision_ocr)]
fn test_scanned_pdf_is_transcribed_via_vision_model() {
    let _reset = ResetOverrides;
    let fake = FakeVisionClient::new("Vision OCR: revenue grew 20%.");
    fake.install();
    set_vision_available(true);

    let (_dir, path) = scanned_pdf(3);
    let text = parse("application/pdf", &path);

    let create_calls = fake.calls();
    // One vision call per scanned page.
    assert_eq!(create_calls.len(), 3, "expected 3 OCR calls, got {}", create_calls.len());

    // Every call must carry the page image as a base64 data URI.
    for call in &create_calls {
        let parts = content_parts(call);
        let image_parts: Vec<&serde_json::Value> =
            parts.iter().filter(|p| p.get("type").and_then(|t| t.as_str()) == Some("image_url")).collect();
        assert!(!image_parts.is_empty(), "vision call missing image content");
        assert!(image_parts[0]["image_url"]["url"].as_str().unwrap_or("").starts_with("data:image/png;base64,"));
    }

    assert!(text.contains("Vision OCR: revenue grew 20%."));
}

/// No vision model → fast fallback, zero vision calls.
#[test]
#[serial(vision_ocr)]
fn test_scanned_pdf_skips_ocr_when_vision_unavailable() {
    let _reset = ResetOverrides;
    let fake = FakeVisionClient::default_content();
    fake.install();
    set_vision_available(false);

    let (_dir, path) = scanned_pdf(4);
    let text = parse("application/pdf", &path);

    assert!(fake.calls().is_empty(), "must not call vision when unavailable");
    // Scanned pages have no text, so the result is empty — but it returns fast.
    assert_eq!(text, "");
}

/// Text-rich PDFs use the fast extraction path with no vision calls.
#[test]
#[serial(vision_ocr)]
fn test_text_pdf_does_not_use_vision_model() {
    let _reset = ResetOverrides;
    let fake = FakeVisionClient::default_content();
    fake.install();
    set_vision_available(true);

    let (_dir, path) = text_pdf(2);
    let text = parse("application/pdf", &path);

    assert!(fake.calls().is_empty(), "text PDF must not invoke the vision model");
    assert!(text.to_lowercase().contains("quarterly report"));
}

/// A standalone image is OCR'd/described through MarkItDown's image
/// converter, backed by the in-process vision client.
#[test]
#[serial(vision_ocr)]
fn test_image_is_described_via_vision_model() {
    let _reset = ResetOverrides;
    let fake = FakeVisionClient::new("A blue square. Text: HELLO WORLD");
    fake.install();

    let (_dir, path) = png();
    let text = parse("image/png", &path);

    // The image reached the vision model as an image_url part...
    let calls = fake.calls();
    assert!(!calls.is_empty(), "image must be sent to the vision model");
    let parts = content_parts(&calls[0]);
    assert!(parts.iter().any(|p| p.get("type").and_then(|t| t.as_str()) == Some("image_url")));
    // ...and its description is in the parser output.
    assert!(text.contains("A blue square. Text: HELLO WORLD"));
}

/// Records whether each OCR call resolved its model on device only.
struct ScopeProbe(Mutex<Vec<bool>>);

impl VisionClient for ScopeProbe {
    fn create(&self, _request: VisionRequest) -> anyhow::Result<String> {
        self.0.lock().push(local_only());
        Ok("Transcribed.".to_string())
    }
}

/// Indexing OCRs every page in the background, so the vision model it
/// reaches is resolved on device only; an attachment the user adds is not
/// held to that.
#[test]
#[serial(vision_ocr)]
fn test_only_indexing_ocr_resolves_on_device() {
    let _reset = ResetOverrides;
    let probe = Arc::new(ScopeProbe(Mutex::new(Vec::new())));
    let client = probe.clone();
    set_vision_client_override(Some(Arc::new(move || -> Option<VisionClientPair> {
        Some((client.clone() as Arc<dyn VisionClient>, "vision".to_string()))
    })));
    set_vision_available(true);
    let (_dir, path) = scanned_pdf(2);

    parse_for_index(None, "application/pdf", path.to_str().expect("utf-8 path")).expect("parse");
    assert_eq!(std::mem::take(&mut *probe.0.lock()), vec![true, true]);
    assert!(!local_only(), "the scope ends with the parse");

    parse("application/pdf", &path);
    assert_eq!(*probe.0.lock(), vec![false, false]);
}
