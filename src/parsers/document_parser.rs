//! Port of `smartloop/parsers/document_parser.py`.
//!
//! Document & image → Markdown conversion (hand-ported MarkItDown).
//!
//! The MarkItDown port handles fast, CPU-light text extraction for office
//! formats (PDF, DOCX, PPTX, XLSX, HTML). Image understanding / OCR is
//! delegated to a vision-capable model selected by
//! `ModelService::get_vision_llm()` (the same `"vision"` capability the
//! orchestrator uses), reached through a small in-process client that speaks
//! an OpenAI `chat.completions.create`-shaped interface — no HTTP:
//!
//! * Standalone images (PNG/JPEG) and images embedded in office docs are
//!   described/transcribed via that client.
//! * Scanned / image-only PDF pages are rendered to images with MuPDF and
//!   transcribed page-by-page through the same client.
//!
//! Tests replace the vision client / availability probe through
//! [`set_vision_client_override`] and [`set_vision_available_override`]
//! (Python monkeypatched `_vision_client` / `_vision_available`).

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use base64::Engine as _;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde_json::{json, Value};

use crate::api::services::get_model_service;
use crate::api::services::model_service::ModelService;
use crate::config::app_settings::AppSettings;
use crate::llms::cloud_llm::CloudLLM;
use crate::llms::llama::ChatCompletionRequest;
use crate::llms::thinking::strip_thinking;
use crate::parsers::base_parser::{BaseParser, SharedLLM};

pub use crate::parsers::markitdown::{DocumentConverterResult, MarkItDown};

/// Logger name used by the Python module.
const LOGGER: &str = "smartloop.parsers.document_parser";

/// Prompt used to transcribe a rendered PDF page.
pub const OCR_PROMPT: &str = "Transcribe ALL text and tables from this document page into clean GitHub-\
flavored Markdown. Preserve headings, lists and table structure. Output \
only the transcription, with no commentary.";

/// Prompt used to describe a standalone / embedded image.
pub const IMAGE_PROMPT: &str = "Describe this image. Transcribe any text, tables or diagrams it contains \
into clean GitHub-flavored Markdown.";

/// Page-render DPI for PDF OCR. A full page image consumes a large share of the
/// vision model's (small) context — at 150 DPI a letter page is ~2800 image
/// tokens, which alone overflows a 2048 ctx. 96 DPI keeps a page near ~1150
/// tokens so it plus the transcription fit within vision_n_ctx (4096), and
/// shrinks the patch grid fed to the native clip encoder, while staying legible.
pub const OCR_DPI: u32 = 96;
/// Output token budget for one OCR'd page.
pub const OCR_MAX_TOKENS: u32 = 1536;
/// Cap the longest edge of rendered page images so the CLIP patch grid stays
/// small (~10x14 patches for a letter page); 560 px keeps it under ~200 patches.
pub const OCR_MAX_PIXELS: u32 = 560;
/// If the largest image on a page covers more than this fraction of the page
/// area, treat the page as a scan even when an embedded text layer exists (the
/// text layer is often garbage from a prior OCR pass).
pub const SCAN_IMAGE_COVERAGE: f64 = 0.80;

/// Arguments of an OpenAI-shaped `chat.completions.create(...)` call.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionRequest {
    pub model: Option<String>,
    /// OpenAI chat messages (content may be a list of `text` / `image_url` parts).
    pub messages: Vec<Value>,
    pub max_tokens: u32,
    pub temperature: f32,
}

impl VisionRequest {
    /// `create(model=, messages=)` with the defaults of
    /// `_VisionChatCompletions.create` (`max_tokens=2048`, `temperature=0.0`).
    pub fn new(model: Option<String>, messages: Vec<Value>) -> Self {
        Self { model, messages, max_tokens: 2048, temperature: 0.0 }
    }
}

/// `client.chat.completions` interface used for image/page OCR. Returns the
/// assistant message content (`choices[0].message.content or ""`).
pub trait VisionClient: Send + Sync {
    fn create(&self, request: VisionRequest) -> anyhow::Result<String>;
}

/// Holds one slot of the model service's inference semaphore
/// (`with self._model_service._inference_lock:`).
struct InferenceSlot<'a>(&'a ModelService);

impl<'a> InferenceSlot<'a> {
    fn acquire(service: &'a ModelService) -> Self {
        service.inference_lock.acquire();
        Self(service)
    }
}

impl Drop for InferenceSlot<'_> {
    fn drop(&mut self) {
        self.0.inference_lock.release();
    }
}

/// `chat.completions` over the in-process vision LLM (Python
/// `_VisionChatCompletions`).
///
/// This is the interface MarkItDown's image converter calls, so the same
/// object backs both MarkItDown image OCR and the PDF page-OCR loop. The
/// vision model is resolved lazily on first `create()` so parsing a text-only
/// office doc never forces the (heavy) vision model to load.
pub struct VisionChatCompletions {
    model_service: Arc<ModelService>,
}

impl VisionChatCompletions {
    pub fn new(model_service: Arc<ModelService>) -> Self {
        Self { model_service }
    }
}

impl VisionClient for VisionChatCompletions {
    /// Runs `create_chat_completion` under the model service's inference lock,
    /// logs timings (`[vision-ocr] ...`) and strips thinking from the output.
    /// Errors with "No vision-capable model available" when none loads.
    fn create(&self, request: VisionRequest) -> anyhow::Result<String> {
        let Some(llm) = self.model_service.get_vision_llm(None) else {
            anyhow::bail!("No vision-capable model available");
        };
        // A cloud vision model reads the page over HTTP.
        if let Some(cloud) = llm.as_any().downcast_ref::<CloudLLM>() {
            let t0 = Instant::now();
            let response =
                cloud.create_chat_completion(request.messages, Some(request.max_tokens as usize), request.temperature)?;
            tracing::info!(
                target: LOGGER,
                "[vision-ocr] {} completed in {:.1}s",
                cloud.model_name(),
                t0.elapsed().as_secs_f64()
            );
            let content = response.pointer("/choices/0/message/content").and_then(Value::as_str).unwrap_or("");
            return Ok(strip_thinking(content));
        }
        let Some(model) = llm.llama_model() else {
            anyhow::bail!("'{}' object has no attribute 'model'", llm.llm_type());
        };
        let n_gpu = model.n_gpu_layers();
        tracing::info!(target: LOGGER, "[vision-ocr] create_chat_completion n_gpu_layers={}", n_gpu);

        let t0 = Instant::now();
        let response = {
            let _slot = InferenceSlot::acquire(&self.model_service);
            model
                .create_chat_completion(ChatCompletionRequest {
                    messages: request.messages,
                    max_tokens: Some(request.max_tokens as usize),
                    temperature: request.temperature,
                    stream: false,
                    ..ChatCompletionRequest::default()
                })?
                .into_response()
        };
        let elapsed = t0.elapsed().as_secs_f64();
        tracing::info!(target: LOGGER, "[vision-ocr] completed in {:.1}s", elapsed);

        let content = response
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .unwrap_or("");
        // Reasoning models are loaded with thinking off for vision work, but a
        // model whose own chat template thinks by default would otherwise write
        // its monologue into the extracted document text.
        Ok(strip_thinking(content))
    }
}

/// Minimal OpenAI-client stand-in backed by the in-process vision model.
pub struct VisionChatClient {
    pub completions: VisionChatCompletions,
}

impl VisionChatClient {
    pub fn new(model_service: Arc<ModelService>) -> Self {
        Self { completions: VisionChatCompletions::new(model_service) }
    }
}

impl VisionClient for VisionChatClient {
    fn create(&self, request: VisionRequest) -> anyhow::Result<String> {
        self.completions.create(request)
    }
}

/// A vision client plus the model name handed to it (Python `(client, "vision")`).
pub type VisionClientPair = (Arc<dyn VisionClient>, String);
/// Replacement for [`vision_client`] (test seam).
pub type VisionClientFactory = Arc<dyn Fn() -> Option<VisionClientPair> + Send + Sync>;
/// Replacement for [`vision_available`] (test seam).
pub type VisionAvailableFn = Arc<dyn Fn() -> bool + Send + Sync>;

static VISION_CLIENT_OVERRIDE: Lazy<RwLock<Option<VisionClientFactory>>> = Lazy::new(|| RwLock::new(None));
static VISION_AVAILABLE_OVERRIDE: Lazy<RwLock<Option<VisionAvailableFn>>> = Lazy::new(|| RwLock::new(None));

/// Install (or clear with `None`) a replacement for [`vision_client`].
pub fn set_vision_client_override(f: Option<VisionClientFactory>) {
    *VISION_CLIENT_OVERRIDE.write() = f;
}

/// Install (or clear with `None`) a replacement for [`vision_available`].
pub fn set_vision_available_override(f: Option<VisionAvailableFn>) {
    *VISION_AVAILABLE_OVERRIDE.write() = f;
}

/// Return (vision client, model_name) for OCR, or `None`.
///
/// The client speaks MarkItDown's OpenAI `llm_client` interface but runs the
/// local multimodal model in-process. Returns `None` when vision is disabled
/// (`AppSettings.vision_enabled`). The client is lazy: no vision model is
/// loaded until an image is actually transcribed.
pub fn vision_client() -> Option<VisionClientPair> {
    let factory = VISION_CLIENT_OVERRIDE.read().clone();
    if let Some(factory) = factory {
        return factory();
    }
    if !AppSettings::new().vision_enabled {
        return None;
    }
    let client: Arc<dyn VisionClient> = Arc::new(VisionChatClient::new(get_model_service()));
    Some((client, "vision".to_string()))
}

/// True when a multimodal vision model is loaded/loadable for OCR.
///
/// Triggers the (lazy, cached) model load via `get_vision_llm()` so the PDF
/// path can skip the per-page OCR loop and fall back to text-only extraction
/// when no usable vision model exists (e.g. mmproj GGUF missing).
pub fn vision_available() -> bool {
    let probe = VISION_AVAILABLE_OVERRIDE.read().clone();
    if let Some(probe) = probe {
        return probe();
    }
    get_model_service().get_vision_llm(None).is_some()
}

/// Convert a document or image to markdown via MarkItDown (+ PDF vision OCR).
pub struct DocumentParser {
    pub llm: Option<SharedLLM>,
    pub file_path: String,
}

impl DocumentParser {
    pub fn new(llm: Option<SharedLLM>, file_path: impl Into<String>) -> Self {
        Self { llm, file_path: file_path.into() }
    }

    /// Convert a document/image to markdown via MarkItDown.
    ///
    /// When a vision client is available it is handed to MarkItDown as the
    /// `llm_client` so images (standalone files and images embedded in office
    /// docs) are described/transcribed by the vision model. The client is
    /// lazy, so a text-only document never loads the vision model. Returns ""
    /// on failure (logged as "MarkItDown conversion failed for {path}: {e}").
    pub fn parse_with_markitdown(&self) -> String {
        let md = match vision_client() {
            Some((client, vision_model)) => MarkItDown::with_llm(client, vision_model, IMAGE_PROMPT),
            None => MarkItDown::new(),
        };
        match md.convert(&self.file_path) {
            Ok(result) => result.text_content.trim().to_string(),
            Err(e) => {
                tracing::warn!(target: LOGGER, "MarkItDown conversion failed for {}: {}", self.file_path, e);
                String::new()
            }
        }
    }

    /// Parse a PDF, OCR'ing scanned/sparse pages via the vision model.
    ///
    /// MuPDF probes per-page text *first* (milliseconds). Text-rich PDFs return
    /// that text directly; scanned / image-only pages are transcribed per page
    /// by the vision model instead (or returned text-only when no vision model
    /// is available). Falls back to MarkItDown if the PDF can't be opened.
    pub fn parse_pdf(&self) -> anyhow::Result<String> {
        let doc = match mupdf::Document::open(self.file_path.as_str()) {
            Ok(doc) => doc,
            Err(e) => {
                tracing::warn!(target: LOGGER, "Could not open PDF, using MarkItDown only: {}", e);
                return Ok(self.parse_with_markitdown());
            }
        };

        let name = Path::new(&self.file_path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let min_chars = AppSettings::new().vision_ocr_min_chars_per_page;

        let mut pages = Vec::new();
        for page in doc.pages()? {
            pages.push(page?);
        }
        let mut page_texts = Vec::with_capacity(pages.len());
        for page in &pages {
            page_texts.push(page_text(page)?.trim().to_string());
        }

        let mut sparse = Vec::new();
        for (i, t) in page_texts.iter().enumerate() {
            if (t.chars().count() as i64) < min_chars || Self::page_is_scan(&pages[i])? {
                sparse.push(i);
            }
        }

        if sparse.is_empty() {
            return Ok(join_non_empty(&page_texts));
        }

        let (client, vision_model) = match vision_client() {
            Some(pair) if vision_available() => pair,
            _ => {
                tracing::warn!(
                    target: LOGGER,
                    "{}/{} scanned page(s) in {} but vision OCR is unavailable — returning text-only extraction",
                    sparse.len(),
                    page_texts.len(),
                    name
                );
                return Ok(join_non_empty(&page_texts));
            }
        };

        tracing::info!(target: LOGGER, "Vision OCR: {}/{} scanned page(s) in {}", sparse.len(), page_texts.len(), name);

        let mut parts = Vec::with_capacity(page_texts.len());
        for (i, text) in page_texts.iter().enumerate() {
            if sparse.contains(&i) {
                let ocr = Self::ocr_page(&pages[i], i, client.as_ref(), &vision_model);
                parts.push(if ocr.is_empty() { text.clone() } else { ocr });
            } else {
                parts.push(text.clone());
            }
        }
        Ok(join_non_empty(&parts))
    }

    /// True when the page is dominated by a single image (scanned page).
    ///
    /// Scanned PDFs often carry a low-quality embedded OCR text layer that
    /// passes the character-count threshold but produces garbage. Detecting
    /// the full-page image lets us route these pages through vision OCR.
    pub fn page_is_scan(page: &mupdf::Page) -> anyhow::Result<bool> {
        let rect = page.bounds()?;
        let page_area = (rect.x1 as f64 - rect.x0 as f64) * (rect.y1 as f64 - rect.y0 as f64);
        if page_area <= 0.0 {
            return Ok(false);
        }
        // PyMuPDF `page.get_image_info()`: the image blocks of a text page
        // extracted with TEXT_PRESERVE_IMAGES.
        let text_page = page.to_text_page(mupdf::TextPageFlags::PRESERVE_IMAGES)?;
        for block in text_page.blocks() {
            if block.r#type() != mupdf::text_page::TextBlockType::Image {
                continue;
            }
            let bbox = block.bounds();
            let img_area = (bbox.x1 as f64 - bbox.x0 as f64) * (bbox.y1 as f64 - bbox.y0 as f64);
            if img_area / page_area >= SCAN_IMAGE_COVERAGE {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Render a single PDF page to a capped-size PNG and transcribe it.
    ///
    /// The render DPI is reduced so the longest edge stays at most
    /// [`OCR_MAX_PIXELS`] px. This keeps the CLIP patch grid small (e.g.
    /// ~15x20 instead of 26x37), which is the dominant cost in the vision
    /// encode step. Returns "" on failure (logged as
    /// "Vision OCR failed for page {n}: {e}").
    pub fn ocr_page(page: &mupdf::Page, page_number: usize, client: &dyn VisionClient, model: &str) -> String {
        let attempt = || -> anyhow::Result<String> {
            let rect = page.bounds()?;
            let width = rect.x1 as f64 - rect.x0 as f64;
            let height = rect.y1 as f64 - rect.y0 as f64;
            let long_edge_pts = width.max(height);
            let base_px = long_edge_pts * OCR_DPI as f64 / 72.0;
            let dpi = if base_px > OCR_MAX_PIXELS as f64 {
                (OCR_DPI as f64 * OCR_MAX_PIXELS as f64 / base_px) as i64
            } else {
                OCR_DPI as i64
            };
            let png = render_page_png(page, dpi)?;
            let b64 = base64::engine::general_purpose::STANDARD.encode(png);
            let content = client.create(VisionRequest {
                model: Some(model.to_string()),
                messages: vec![json!({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_PROMPT},
                        {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{b64}")}},
                    ],
                })],
                max_tokens: OCR_MAX_TOKENS,
                temperature: 0.0,
            })?;
            Ok(content.trim().to_string())
        };
        match attempt() {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!(target: LOGGER, "Vision OCR failed for page {}: {}", page_number, e);
                String::new()
            }
        }
    }
}

/// `"\n\n".join(p for p in parts if p).strip()`.
fn join_non_empty(parts: &[String]) -> String {
    parts.iter().filter(|p| !p.is_empty()).cloned().collect::<Vec<_>>().join("\n\n").trim().to_string()
}

/// PyMuPDF `page.get_text("text")`: every character of every text line that
/// overlaps the page rectangle, each line terminated by a newline (unless it
/// already ends with one).
pub fn page_text(page: &mupdf::Page) -> anyhow::Result<String> {
    let rect = page.bounds()?;
    let flags = mupdf::TextPageFlags::PRESERVE_LIGATURES
        | mupdf::TextPageFlags::PRESERVE_WHITESPACE
        | mupdf::TextPageFlags::USE_CID_FOR_UNKNOWN_UNICODE;
    let text_page = page.to_text_page(flags)?;
    let infinite = rect.x0 > rect.x1 || rect.y0 > rect.y1;

    let mut out = String::new();
    for block in text_page.blocks() {
        if block.r#type() != mupdf::text_page::TextBlockType::Text {
            continue;
        }
        for line in block.lines() {
            let mut last_char: u32 = 0;
            for ch in line.chars() {
                let bbox = mupdf::Rect::from(ch.quad());
                let overlaps =
                    !(bbox.x0 >= rect.x1 || bbox.y0 >= rect.y1 || bbox.x1 <= rect.x0 || bbox.y1 <= rect.y0);
                if infinite || overlaps {
                    let c = ch.char().unwrap_or(char::REPLACEMENT_CHARACTER);
                    last_char = c as u32;
                    out.push(c);
                }
            }
            if last_char != 10 && last_char > 0 {
                out.push('\n');
            }
        }
    }
    Ok(out)
}

/// PyMuPDF `page.get_pixmap(dpi=dpi).tobytes("png")`: an RGB render without
/// alpha at `dpi / 72` zoom.
pub fn render_page_png(page: &mupdf::Page, dpi: i64) -> anyhow::Result<Vec<u8>> {
    let zoom = dpi as f32 / 72.0;
    let pixmap = page.to_pixmap(&mupdf::Matrix::new_scale(zoom, zoom), &mupdf::Colorspace::device_rgb(), false, true)?;
    let mut png = Vec::new();
    pixmap.write_to(&mut png, mupdf::ImageFormat::PNG)?;
    Ok(png)
}

impl BaseParser for DocumentParser {
    fn llm(&self) -> Option<&SharedLLM> {
        self.llm.as_ref()
    }
    fn file_path(&self) -> &str {
        &self.file_path
    }
    fn type_name(&self) -> &'static str {
        "DocumentParser"
    }
    fn parse(&self) -> anyhow::Result<String> {
        let is_pdf = Path::new(&self.file_path)
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase() == "pdf")
            .unwrap_or(false);
        if is_pdf {
            self.parse_pdf()
        } else {
            Ok(self.parse_with_markitdown())
        }
    }
}
