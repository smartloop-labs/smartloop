//! Port of `smartloop/model/generic.py`.
//!
//! [`GenericModel`] is the shared implementation behind every concrete model:
//! weight download, document ingestion (files, directories, URLs, Google
//! Drive links), skills, status, train/build and the cached LLM instance.  The
//! per-model facts (id, family version, capabilities, ...) come from the
//! concrete model's [`ModelClass`].

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use comfy_table::{CellAlignment, ContentLineStyle, LineStyle, Table, TableStyle};
use md5::{Digest, Md5};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use regex::Regex;

use crate::llms::base_llm::BaseLLM;
use crate::loader::ModelLoader;
use crate::model::downloader::slp_downloader::{SLPDownloader, SharedLogger, SharedWorkspace};
use crate::model::slp_model::{DocumentProgressCallback, DocumentSource, ModelClass, SLPModel, TrainProgressCallback};
use crate::paths::expand_home;
use crate::trainer::{Builder, Trainer, TrainerNotInstalled};
use crate::types::{ModelCapability, ModelType};
use crate::utils::normalize_path;
use crate::workspace::types::{Document, DocumentStatus, Skill};

/// PrettyTable's default look: `+---+` borders, a `-` rule under the header
/// and no rules between rows.
const PRETTY_TABLE_STYLE: TableStyle = TableStyle::new()
    .top_border(LineStyle::new('+', '-', '+', '+'))
    .header_lines(ContentLineStyle::new('|', '|', '|'))
    .header_separator(LineStyle::new('+', '-', '+', '+'))
    .content_lines(ContentLineStyle::new('|', '|', '|'))
    .bottom_border(LineStyle::new('+', '-', '+', '+'));

/// Timeout of the document downloads (`timeout=60`).
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(60);

static GDRIVE_ID_PATTERNS: Lazy<[Regex; 2]> = Lazy::new(|| {
    [
        Regex::new(r"/file/d/([a-zA-Z0-9_-]+)").expect("valid regex"),
        Regex::new(r"id=([a-zA-Z0-9_-]+)").expect("valid regex"),
    ]
});

static CONTENT_DISPOSITION_FILENAME: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"filename="?(.+)"?"#).expect("valid regex"));

static FILENAME_JUNK: Lazy<Regex> = Lazy::new(|| Regex::new(r"[^\w\s-]").expect("valid regex"));

/// The pieces of `urllib.parse.urlsplit` this module reads.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SplitUrl {
    pub scheme: String,
    pub netloc: String,
    pub path: String,
}

/// Port of `urllib.parse.urlsplit` (scheme, netloc and path only).  Errors
/// where Python raises `ValueError` (an unbalanced IPv6 bracket in the netloc).
pub fn url_split(url: &str) -> Result<SplitUrl, String> {
    // WHATWG: leading C0 control-or-space is stripped, and tab/CR/LF are
    // removed wherever they occur.
    let url: String = url
        .trim_start_matches(|c: char| c <= ' ')
        .chars()
        .filter(|c| !matches!(c, '\t' | '\r' | '\n'))
        .collect();
    let mut rest = url.as_str();
    let mut scheme = String::new();
    if let Some(i) = rest.find(':') {
        let candidate = &rest[..i];
        let mut chars = candidate.chars();
        let valid = chars.next().is_some_and(|c| c.is_ascii_alphabetic())
            && candidate.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
        if i > 0 && valid {
            scheme = candidate.to_ascii_lowercase();
            rest = &rest[i + 1..];
        }
    }
    let mut netloc = String::new();
    if let Some(after) = rest.strip_prefix("//") {
        let delim = after.find(['/', '?', '#']).unwrap_or(after.len());
        netloc = after[..delim].to_string();
        rest = &after[delim..];
        if netloc.contains('[') != netloc.contains(']') {
            return Err("Invalid IPv6 URL".to_string());
        }
    }
    let rest = rest.split_once('#').map_or(rest, |(before, _)| before);
    let path = rest.split_once('?').map_or(rest, |(before, _)| before);
    Ok(SplitUrl { scheme, netloc, path: path.to_string() })
}

/// `pathlib.PurePosixPath(path).name`: the last non-empty, non-`.` component.
fn pure_name(path: &str) -> String {
    path.split('/').rev().find(|part| !part.is_empty() && *part != ".").unwrap_or("").to_string()
}

/// `pathlib`'s `(stem, suffix)` split of a file name.
fn stem_and_suffix(name: &str) -> (String, String) {
    match name.rfind('.') {
        Some(i) if i > 0 && i < name.len() - 1 => (name[..i].to_string(), name[i..].to_string()),
        _ => (name.to_string(), String::new()),
    }
}

/// The final component of `path` as a string (`Path.name`).
fn file_name(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// `Path.resolve()` (non-strict): canonicalize what exists, keep the rest.
fn resolve(path: &Path) -> PathBuf {
    if let Ok(real) = fs::canonicalize(path) {
        return real;
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().map(|cwd| cwd.join(path)).unwrap_or_else(|_| path.to_path_buf())
    };
    match (absolute.parent(), absolute.file_name()) {
        (Some(parent), Some(name)) => resolve(parent).join(name),
        _ => absolute,
    }
}

/// `tempfile.mkdtemp()`: a fresh directory the caller owns and removes.
fn mkdtemp() -> std::io::Result<PathBuf> {
    Ok(tempfile::Builder::new().prefix("tmp").tempdir()?.keep())
}

/// `shutil.copy2`: copy the bytes and permissions, then the timestamps.
fn copy2(src: &Path, dest: &Path) -> std::io::Result<()> {
    fs::copy(src, dest)?;
    let meta = fs::metadata(src)?;
    let mut times = fs::FileTimes::new();
    if let Ok(modified) = meta.modified() {
        times = times.set_modified(modified);
    }
    if let Ok(accessed) = meta.accessed() {
        times = times.set_accessed(accessed);
    }
    fs::OpenOptions::new().write(true).open(dest)?.set_times(times)
}

/// `DocumentStatus(...).name`: the enum member name (`"PENDING"`, ...).
fn status_name(status: DocumentStatus) -> String {
    status.as_str().to_uppercase()
}

/// The whole response body.  Read in pieces so the 60s timeout bounds each
/// read (httpx's per-operation timeout) rather than the whole transfer.
fn read_body(mut response: reqwest::blocking::Response) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut content = Vec::new();
    response.read_to_end(&mut content)?;
    Ok(content)
}

fn http_client() -> anyhow::Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .timeout(DOWNLOAD_TIMEOUT)
        .redirect(reqwest::redirect::Policy::limited(20))
        .build()?)
}

/// Shared implementation of [`SLPModel`] for all concrete models.
pub struct GenericModel {
    pub workspace: SharedWorkspace,
    pub logger: SharedLogger,
    pub downloader: Arc<dyn SLPDownloader>,
    /// The concrete model's static facts (the Python subclass's static methods).
    pub class: ModelClass,
    /// The concrete model's modality (the Python `model_type` property).
    pub model_type: ModelType,
    /// Cache for the loaded model.
    instance_cache: Mutex<Option<Arc<dyn BaseLLM>>>,
}

impl GenericModel {
    /// Build a generic model for `class` with modality `model_type`.
    pub fn new(
        workspace: SharedWorkspace,
        downloader: Arc<dyn SLPDownloader>,
        logger: SharedLogger,
        class: ModelClass,
        model_type: ModelType,
    ) -> Self {
        Self { workspace, logger, downloader, class, model_type, instance_cache: Mutex::new(None) }
    }

    /// MD5 hex digest of a file's contents.
    fn calculate_file_hash(&self, file_path: &Path) -> anyhow::Result<String> {
        let mut hasher = Md5::new();
        hasher.update(fs::read(file_path)?);
        Ok(hex::encode(hasher.finalize()))
    }

    /// Whether `source` parses as a URL with both scheme and netloc.
    pub fn is_url(&self, source: &str) -> bool {
        match url_split(source) {
            Ok(parts) => !parts.scheme.is_empty() && !parts.netloc.is_empty(),
            Err(_) => false,
        }
    }

    /// Whether `url` is a Google Drive link.
    pub fn is_google_drive_url(&self, url: &str) -> bool {
        url.contains("drive.google.com")
    }

    /// Extract the file ID from a Google Drive URL (`/file/d/<id>` or
    /// `id=<id>`); errors with "Could not extract file ID from Google Drive
    /// URL: <url>".
    pub fn extract_google_drive_id(&self, url: &str) -> anyhow::Result<String> {
        // Format 1: https://drive.google.com/file/d/FILE_ID/view
        // Format 2: https://drive.google.com/open?id=FILE_ID
        for pattern in GDRIVE_ID_PATTERNS.iter() {
            if let Some(caps) = pattern.captures(url) {
                return Ok(caps[1].to_string());
            }
        }
        anyhow::bail!("Could not extract file ID from Google Drive URL: {url}")
    }

    /// Copy the original file into `dest_dir`, appending `_<n>` to the stem on
    /// name collisions; returns the destination path.
    pub fn copy_to_project_dir(&self, file_path: &Path, dest_dir: &Path) -> anyhow::Result<PathBuf> {
        fs::create_dir_all(dest_dir)?;
        let name = file_name(file_path);
        let mut dest = dest_dir.join(&name);
        // Handle name collisions by appending a counter
        if dest.exists() {
            let (stem, suffix) = stem_and_suffix(&name);
            let mut counter = 1;
            while dest.exists() {
                dest = dest_dir.join(format!("{stem}_{counter}{suffix}"));
                counter += 1;
            }
        }
        copy2(file_path, &dest)?;
        self.logger.info(&format!("Copied to project dir: {}", file_name(&dest)));
        Ok(dest)
    }

    /// Use the LLM to produce a short descriptive snake_case filename from the
    /// first 512 characters of the file; `None` on any failure.
    pub fn generate_filename(&self, file_path: &Path, llm: &dyn BaseLLM) -> Option<String> {
        let bytes = fs::read(file_path).ok()?;
        let content: String = String::from_utf8_lossy(&bytes).chars().take(512).collect();

        if content.trim().is_empty() {
            return None;
        }

        let prompt = format!(
            "Generate a short, descriptive filename (2-5 words, snake_case, no extension) \
             for a document that starts with:\n\n{content}\n\nReply with ONLY the filename, nothing else."
        );
        let raw = llm.invoke(&prompt, None, None);
        let name = raw.trim().trim_matches(|c| c == '"' || c == '\'');
        let (stem, _) = stem_and_suffix(&pure_name(name));
        let cleaned = FILENAME_JUNK.replace_all(&stem, "");
        let name: String = cleaned.trim().replace(' ', "_").chars().take(60).collect();
        if !name.is_empty() {
            self.logger.info(&format!("[add_document] Generated filename: {name}"));
            return Some(name);
        }
        None
    }

    /// Download a Google Drive file (handling the large-file confirm cookie)
    /// into a fresh temp directory.
    fn download_from_google_drive(&self, url: &str) -> anyhow::Result<PathBuf> {
        let result = (|| -> anyhow::Result<PathBuf> {
            let tmp_dir = mkdtemp()?;

            let file_id = self.extract_google_drive_id(url)?;
            let mut download_url = format!("https://drive.google.com/uc?export=download&id={file_id}");

            self.logger.info(&format!("Downloading from Google Drive: {file_id}"));

            let client = http_client()?;
            let mut response = client.get(&download_url).send()?;

            // Check if we need to confirm download for large files
            let confirm_token = response.headers().get_all(reqwest::header::SET_COOKIE).iter().find_map(|value| {
                let cookie = value.to_str().ok()?;
                let pair = cookie.split(';').next()?.trim();
                let (name, value) = pair.split_once('=')?;
                name.trim().starts_with("download_warning").then(|| value.trim().to_string())
            });

            if let Some(confirm_token) = confirm_token {
                download_url =
                    format!("https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}");
                response = client.get(&download_url).send()?;
            }

            let mut filename = response
                .headers()
                .get(reqwest::header::CONTENT_DISPOSITION)
                .and_then(|v| v.to_str().ok())
                .and_then(|content_disp| {
                    CONTENT_DISPOSITION_FILENAME
                        .captures(content_disp)
                        .map(|caps| caps[1].trim_matches('"').to_string())
                })
                .unwrap_or_default();

            if filename.is_empty() {
                filename = format!("gdrive_{file_id}.pdf");
            }

            let download_path = tmp_dir.join(&filename);
            let content = read_body(response)?;
            fs::write(&download_path, &content)?;

            self.logger.info(&format!("Downloaded to: {}", download_path.display()));

            Ok(download_path)
        })();
        if let Err(e) = &result {
            self.logger.error(&format!("Failed to download from Google Drive {url}: {e}"));
        }
        result
    }

    /// Download a URL into a fresh temp directory.
    fn download_url(&self, url: &str) -> anyhow::Result<PathBuf> {
        let result = (|| -> anyhow::Result<PathBuf> {
            let tmp_dir = mkdtemp()?;

            // Get filename from URL or generate one
            let parsed_path = url_split(url).map(|p| p.path).unwrap_or_default();
            let mut filename = pure_name(&parsed_path);
            if filename.is_empty() {
                let digest = hex::encode(Md5::digest(url.as_bytes()));
                filename = format!("download_{}", &digest[..8]);
            }

            let download_path = tmp_dir.join(&filename);

            self.logger.info(&format!("Downloading from URL: {url}"));

            let response = http_client()?.get(url).send()?;
            let response = response.error_for_status()?;
            let content = read_body(response)?;
            fs::write(&download_path, &content)?;

            self.logger.info(&format!("Downloaded to: {}", download_path.display()));

            Ok(download_path)
        })();
        if let Err(e) = &result {
            self.logger.error(&format!("Failed to download from URL {url}: {e}"));
        }
        result
    }

    /// Emit `{"stage", "filename", "message"}` if a callback is provided.
    fn emit_progress(&self, callback: Option<&DocumentProgressCallback>, stage: &str, filename: &str, message: &str) {
        if let Some(callback) = callback {
            callback(serde_json::json!({"stage": stage, "filename": filename, "message": message}));
        }
    }

    /// Register one local file: copy it into `dest_dir` and describe it as a
    /// PENDING document.
    fn register_file(
        &self,
        file_path: &Path,
        dest_dir: &Path,
        progress_callback: Option<&DocumentProgressCallback>,
    ) -> anyhow::Result<Document> {
        let name = file_name(file_path);
        self.emit_progress(progress_callback, "registering", &name, &format!("Registering {name}..."));
        let dest_path = self.copy_to_project_dir(file_path, dest_dir)?;
        Ok(Document {
            id: self.calculate_file_hash(file_path)?,
            path: dest_path.to_string_lossy().into_owned(),
            name: Some(name),
            summary: None,
            status: DocumentStatus::Pending,
        })
    }

    /// Download `source` (a URL) and register the downloaded file.
    fn register_url(
        &self,
        source: &str,
        dest_dir: &Path,
        progress_callback: Option<&DocumentProgressCallback>,
    ) -> anyhow::Result<Document> {
        self.emit_progress(progress_callback, "downloading", source, "Downloading from URL...");
        let downloaded_path =
            if self.is_google_drive_url(source) { self.download_from_google_drive(source)? } else { self.download_url(source)? };

        let name = file_name(&downloaded_path);
        self.emit_progress(progress_callback, "registering", &name, &format!("Registering {name}..."));
        let dest_path = self.copy_to_project_dir(&downloaded_path, dest_dir)?;

        let document = Document {
            id: self.calculate_file_hash(&downloaded_path)?,
            path: dest_path.to_string_lossy().into_owned(),
            name: Some(name),
            summary: None,
            status: DocumentStatus::Pending,
        };

        // Clean up temp download directory
        if let Some(tmp_dir) = downloaded_path.parent() {
            let _ = fs::remove_dir_all(tmp_dir);
        }
        Ok(document)
    }

    /// Rename freshly added `.txt` documents to an LLM-generated name.
    fn rename_with_generated_names(&self, new_docs: &mut [Document]) -> anyhow::Result<()> {
        let llm = SLPModel::get_instance(self)?;
        let mut batch_paths: HashSet<PathBuf> = new_docs.iter().map(|d| resolve(Path::new(&d.path))).collect();
        for doc in new_docs.iter_mut() {
            let doc_path = PathBuf::from(&doc.path);
            if stem_and_suffix(&file_name(&doc_path)).1.to_lowercase() != ".txt" {
                continue;
            }
            let Some(generated) = self.generate_filename(&doc_path, llm.as_ref()) else {
                continue;
            };
            let parent = doc_path.parent().map(Path::to_path_buf).unwrap_or_default();
            let mut new_path = parent.join(format!("{generated}.txt"));
            if resolve(&new_path) != resolve(&doc_path) && new_path.exists() && !batch_paths.contains(&resolve(&new_path)) {
                let mut counter = 1;
                while new_path.exists() && !batch_paths.contains(&resolve(&new_path)) {
                    new_path = parent.join(format!("{generated}_{counter}.txt"));
                    counter += 1;
                }
            }
            batch_paths.remove(&resolve(&doc_path));
            batch_paths.insert(resolve(&new_path));
            fs::rename(&doc_path, &new_path)?;
            doc.path = new_path.to_string_lossy().into_owned();
            self.logger.info(&format!(
                "[add_document] Renamed {} → {}",
                file_name(&doc_path),
                file_name(&new_path)
            ));
        }
        Ok(())
    }

    /// Error unless `model_dir` contains `*.safetensors` files.
    pub fn check_safetensors(&self, model_dir: &Path) -> anyhow::Result<()> {
        let has_safetensors = model_dir.is_dir()
            && fs::read_dir(model_dir)
                .map(|entries| {
                    entries.flatten().any(|entry| entry.file_name().to_string_lossy().ends_with(".safetensors"))
                })
                .unwrap_or(false);
        if !has_safetensors {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "No safetensors files found in {}. Training and building require the full model weights. \
                     Download them using an HF token: slp init --hf-token <token>",
                    model_dir.display()
                ),
            )
            .into());
        }
        Ok(())
    }
}

impl SLPModel for GenericModel {
    fn workspace(&self) -> &SharedWorkspace {
        &self.workspace
    }

    fn logger(&self) -> &SharedLogger {
        &self.logger
    }

    fn init(&self) -> anyhow::Result<()> {
        self.logger.info("[info] Initializing model...");
        self.logger.info(&"=".repeat(50));

        self.downloader.download(&SLPModel::model_id(self), None)?;
        Ok(())
    }

    /// Copies the original file(s) into the project's documents directory and
    /// registers them with PENDING status (duplicates by hash are skipped).
    /// The indexer handles conversion, indexing, and summary generation in
    /// the background.
    fn add_document(
        &self,
        source: DocumentSource,
        progress_callback: Option<DocumentProgressCallback>,
        cwd: Option<&str>,
        generate_filename: bool,
    ) -> anyhow::Result<Vec<Document>> {
        // Documents are project state, so they always land under the project's
        // documents dir — never under a caller-supplied cwd.  `cwd` is the
        // user's own folder for file/bash tools; writing ingested copies there
        // put project state outside the framework's home, where it could not
        // follow SLP_HOME.
        let _ = cwd;
        let progress_callback = progress_callback.as_ref();

        println!("\n{}", console::style(" Adding files to be processed:").italic());

        let (project_id, dest_dir, existing_ids) = {
            let ws = self.workspace.read();
            let Some(current_project) = ws.current_project() else {
                self.logger.error("[error] No current project found.");
                return Ok(Vec::new());
            };
            let existing: HashSet<String> = current_project.documents.iter().map(|d| d.id.clone()).collect();
            (current_project.id.clone(), PathBuf::from(current_project.documents_dir()), existing)
        };

        let mut documents: Vec<Document> = Vec::new();

        match source {
            // Handle URL
            DocumentSource::Str(ref url) if self.is_url(url) => match self.register_url(url, &dest_dir, progress_callback) {
                Ok(document) => documents.push(document),
                Err(_) => {
                    self.logger.error(&format!("[error] Failed to download URL: {url}"));
                    return Ok(Vec::new());
                }
            },
            source => {
                let source_path = match source {
                    DocumentSource::Str(raw) => {
                        // Normalize pasted file paths
                        let mut s = raw.trim().trim_matches(|c| c == '\'' || c == '"').to_string();
                        s = s.replace("\\ ", " ");
                        if s.to_lowercase().starts_with("file://") {
                            s = s.chars().skip(7).collect();
                        }
                        s = normalize_path(&s);
                        PathBuf::from(expand_home(&s))
                    }
                    DocumentSource::Path(path) => path,
                };

                if source_path.is_dir() {
                    for entry in fs::read_dir(&source_path)? {
                        let file_path = source_path.join(entry?.file_name());
                        if file_path.is_file() {
                            match self.register_file(&file_path, &dest_dir, progress_callback) {
                                Ok(document) => documents.push(document),
                                Err(e) => {
                                    self.logger.error(&format!("Failed to register {}: {e}", file_name(&file_path)))
                                }
                            }
                        }
                    }
                } else if source_path.is_file() {
                    match self.register_file(&source_path, &dest_dir, progress_callback) {
                        Ok(document) => documents.push(document),
                        Err(e) => {
                            self.logger
                                .error(&format!("[error] Failed to register file: {}: {e}", source_path.display()));
                            return Err(e);
                        }
                    }
                } else {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("Invalid file path: {}", source_path.display()),
                    )
                    .into());
                }
            }
        }

        let total_found = documents.len();
        let mut new_docs: Vec<Document> = documents.into_iter().filter(|doc| !existing_ids.contains(&doc.id)).collect();
        let skipped = total_found - new_docs.len();
        if skipped > 0 {
            self.logger.info(&format!("[add_document] {skipped} document(s) skipped (duplicate hash)"));
        }

        if generate_filename && !new_docs.is_empty() {
            if let Err(e) = self.rename_with_generated_names(&mut new_docs) {
                self.logger.warning(&format!("[add_document] Filename generation failed: {e}"));
            }
        }

        for doc in &new_docs {
            let doc_name = file_name(Path::new(&doc.path));
            println!("\t{} ", console::style(format!(" added: {doc_name}")).green());
        }

        let total = {
            let mut ws = self.workspace.write();
            match ws.project_mut(&project_id) {
                Some(project) => {
                    project.documents.extend(new_docs.iter().cloned());
                    project.documents.len()
                }
                None => new_docs.len(),
            }
        };
        self.logger.info(&format!(
            "[add_document] project={project_id}, dest_dir={}, {} new, {total} total documents",
            dest_dir.display(),
            new_docs.len()
        ));
        Ok(new_docs)
    }

    fn add_skill(&self, name: &str, content: &str) -> anyhow::Result<()> {
        let mut ws = self.workspace.write();
        let Some(current_project) = ws.current_project_mut() else {
            self.logger.error("[error] No current project found.");
            return Ok(());
        };

        let content = content.trim();
        if content.is_empty() {
            self.logger.error("[error] Skill content cannot be empty.");
            return Ok(());
        }

        if content.chars().count() < 10 {
            self.logger.warning("[warn] Skill content is very short. Consider adding more details.");
        }

        let id: String = uuid::Uuid::new_v4().to_string().chars().take(8).collect();
        let skill = Skill { id, name: name.to_string(), description: None, content: content.to_string(), system: false };
        current_project.skills.push(skill);
        let total = current_project.skills.len();
        drop(ws);

        self.logger.info(&format!("[ok] Skill '{name}' added"));
        self.logger.info(&format!("Total skills: {total}"));
        Ok(())
    }

    fn status(&self) -> anyhow::Result<()> {
        self.logger.info("[info] Project Status:");
        println!();

        let ws = self.workspace.read();
        let Some(current_project) = ws.current_project() else {
            self.logger.error("[error] No project found.");
            return Ok(());
        };

        let mut table = Table::new();
        table.load_style(PRETTY_TABLE_STYLE);
        table.set_header(vec!["File", "Status"]);
        for column in table.column_iter_mut() {
            column.set_cell_alignment(CellAlignment::Left);
        }

        let mut to_be_processed = 0usize;

        for doc in &current_project.documents {
            let name = file_name(Path::new(&doc.path));
            match doc.status {
                DocumentStatus::Pending => {
                    if Path::new(&doc.path).exists() {
                        table.add_row(vec![name, status_name(DocumentStatus::Pending)]);
                        to_be_processed += 1;
                    }
                }
                DocumentStatus::Indexed => {
                    table.add_row(vec![name, status_name(DocumentStatus::Indexed)]);
                }
                DocumentStatus::Deleted => {
                    table.add_row(vec![name, status_name(DocumentStatus::Deleted)]);
                }
                other => {
                    table.add_row(vec![name, status_name(other)]);
                }
            }

            // Printed once per document, as the Python loop does.
            println!("{table}");
        }

        if to_be_processed == 0 {
            self.logger.info("\n- No files to process.");
        } else {
            self.logger.info(&format!("\n({to_be_processed}) files to process."));
        }
        Ok(())
    }

    fn build(&self, progress_callback: Option<TrainProgressCallback>) -> anyhow::Result<()> {
        // `from smartloop.trainer import Builder` runs first: the builder is an
        // optional component, and without it the call fails before anything
        // else happens — as the Python import does without smartloop-trainer.
        let builder = Builder::new()?;

        self.logger.info("[info] Building model...");
        self.logger.info(&"=".repeat(50));

        let ws_model = self.workspace.read().get_model(None);
        let model_id = ws_model.as_ref().map(|m| m.model_id.clone()).unwrap_or_default();

        let model_dir = self.workspace.read().model_dir_for(&model_id);
        self.check_safetensors(Path::new(&model_dir))?;

        let Some(ws_model) = ws_model else {
            anyhow::bail!("'NoneType' object has no attribute 'bin_dir'");
        };
        fs::create_dir_all(ws_model.bin_dir())?;
        let quantize_type = ws_model.config.as_ref().map(|c| c.quantize.clone()).unwrap_or_else(|| "q8_0".to_string());
        // `Builder(workspace, model_id, logger, output_dir=bin_dir).build(quantize_type=..., progress_callback=...)`:
        // the builder this build ships exposes no build step.
        let _ = (builder, quantize_type, progress_callback);
        Err(TrainerNotInstalled.into())
    }

    fn train(&self, progress_callback: Option<TrainProgressCallback>) -> anyhow::Result<()> {
        // `from smartloop.trainer import Trainer` runs first (see `build`).
        let trainer = Trainer::new()?;

        self.logger.info("[info] Training model...");
        self.logger.info(&"=".repeat(50));

        let (has_project, model_dir) = {
            let ws = self.workspace.read();
            let model_id = ws.get_model(None).map(|m| m.model_id).unwrap_or_default();
            (ws.current_project().is_some(), ws.model_dir_for(&model_id))
        };
        if !has_project {
            anyhow::bail!("No active project. Please create a project first.");
        }

        self.check_safetensors(Path::new(&model_dir))?;

        // `Trainer(workspace, logger).train(progress_callback=...)`: the trainer
        // this build ships exposes no training step.
        let _ = (trainer, progress_callback);
        Err(TrainerNotInstalled.into())
    }

    fn model_type(&self) -> ModelType {
        self.model_type
    }

    fn model_id(&self) -> String {
        (self.class.model_id)()
    }

    fn model_version(&self) -> i64 {
        (self.class.model_version)()
    }

    fn param_size_in_b(&self) -> f64 {
        (self.class.param_size_in_b)()
    }

    fn default_capabilities(&self) -> Vec<ModelCapability> {
        (self.class.default_capabilities)()
    }

    fn thinking_default(&self) -> bool {
        (self.class.thinking_default)()
    }

    fn vision_chat_handler(&self) -> Option<String> {
        (self.class.vision_chat_handler)()
    }

    fn get_instance(&self) -> anyhow::Result<Arc<dyn BaseLLM>> {
        let mut cache = self.instance_cache.lock();
        if let Some(instance) = cache.as_ref() {
            return Ok(Arc::clone(instance));
        }
        let model_id = SLPModel::model_id(self);
        // Cascade: use project-level GGUF if it exists, else fall back to workspace GGUF
        let (gguf_path, snapshot) = {
            let ws = self.workspace.read();
            (ws.resolve_gguf_path(&model_id), Arc::new(ws.clone()))
        };

        self.logger.info(&format!("Loading model instance from GGUF: {gguf_path}"));
        let model_loader = ModelLoader::new(snapshot, Some(gguf_path));
        let instance: Arc<dyn BaseLLM> = model_loader.load()?;
        *cache = Some(Arc::clone(&instance));
        Ok(instance)
    }
}

/// Define a concrete model type wrapping [`GenericModel`]: a newtype with
/// inherent static accessors mirroring the Python static methods (these are
/// what the config registry's [`ModelClass`] points at), `Deref` to the
/// generic model, and an [`SLPModel`] impl.
#[macro_export]
macro_rules! generic_model_type {
    (
        $(#[$meta:meta])*
        $name:ident {
            model_id: $model_id:expr,
            model_type: $model_type:expr,
            model_version: $version:expr,
            param_size_in_b: $size:expr,
            default_capabilities: $caps:expr,
            thinking_default: $thinking:expr,
            vision_chat_handler: $handler:expr $(,)?
        }
    ) => {
        $(#[$meta])*
        pub struct $name(pub $crate::model::generic::GenericModel);

        #[allow(non_camel_case_types)]
        impl $name {
            /// Construct the model (`Model(workspace, downloader, logger)`).
            pub fn new(
                workspace: $crate::model::downloader::slp_downloader::SharedWorkspace,
                downloader: ::std::sync::Arc<dyn $crate::model::downloader::slp_downloader::SLPDownloader>,
                logger: $crate::model::downloader::slp_downloader::SharedLogger,
            ) -> Self {
                Self($crate::model::generic::GenericModel::new(workspace, downloader, logger, Self::class(), $model_type))
            }

            /// This model's class descriptor.
            pub fn class() -> $crate::model::slp_model::ModelClass {
                $crate::model::slp_model::ModelClass {
                    class_name: stringify!($name),
                    model_id: Self::model_id,
                    model_version: Self::model_version,
                    param_size_in_b: Self::param_size_in_b,
                    default_capabilities: Self::default_capabilities,
                    thinking_default: Self::thinking_default,
                    vision_chat_handler: Self::vision_chat_handler,
                }
            }

            /// The HuggingFace-style model id.
            pub fn model_id() -> String {
                ($model_id).to_string()
            }
            /// Model family version.
            pub fn model_version() -> i64 {
                $version
            }
            /// Parameter count in billions.
            pub fn param_size_in_b() -> f64 {
                $size
            }
            /// Declared capabilities.
            pub fn default_capabilities() -> Vec<$crate::types::ModelCapability> {
                $caps
            }
            /// Whether reasoning defaults on.
            pub fn thinking_default() -> bool {
                $thinking
            }
            /// Vision chat-handler name.
            pub fn vision_chat_handler() -> Option<String> {
                let handler: Option<&str> = $handler;
                handler.map(str::to_string)
            }
        }

        impl ::std::ops::Deref for $name {
            type Target = $crate::model::generic::GenericModel;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl $crate::model::slp_model::SLPModel for $name {
            fn workspace(&self) -> &$crate::model::downloader::slp_downloader::SharedWorkspace {
                $crate::model::slp_model::SLPModel::workspace(&self.0)
            }
            fn logger(&self) -> &$crate::model::downloader::slp_downloader::SharedLogger {
                $crate::model::slp_model::SLPModel::logger(&self.0)
            }
            fn init(&self) -> ::anyhow::Result<()> {
                $crate::model::slp_model::SLPModel::init(&self.0)
            }
            fn add_document(
                &self,
                source: $crate::model::slp_model::DocumentSource,
                progress_callback: Option<$crate::model::slp_model::DocumentProgressCallback>,
                cwd: Option<&str>,
                generate_filename: bool,
            ) -> ::anyhow::Result<Vec<$crate::workspace::types::Document>> {
                $crate::model::slp_model::SLPModel::add_document(&self.0, source, progress_callback, cwd, generate_filename)
            }
            fn add_skill(&self, name: &str, content: &str) -> ::anyhow::Result<()> {
                $crate::model::slp_model::SLPModel::add_skill(&self.0, name, content)
            }
            fn status(&self) -> ::anyhow::Result<()> {
                $crate::model::slp_model::SLPModel::status(&self.0)
            }
            fn build(&self, progress_callback: Option<$crate::model::slp_model::TrainProgressCallback>) -> ::anyhow::Result<()> {
                $crate::model::slp_model::SLPModel::build(&self.0, progress_callback)
            }
            fn train(&self, progress_callback: Option<$crate::model::slp_model::TrainProgressCallback>) -> ::anyhow::Result<()> {
                $crate::model::slp_model::SLPModel::train(&self.0, progress_callback)
            }
            fn model_type(&self) -> $crate::types::ModelType {
                $crate::model::slp_model::SLPModel::model_type(&self.0)
            }
            fn model_id(&self) -> String {
                $crate::model::slp_model::SLPModel::model_id(&self.0)
            }
            fn model_version(&self) -> i64 {
                $crate::model::slp_model::SLPModel::model_version(&self.0)
            }
            fn param_size_in_b(&self) -> f64 {
                $crate::model::slp_model::SLPModel::param_size_in_b(&self.0)
            }
            fn default_capabilities(&self) -> Vec<$crate::types::ModelCapability> {
                $crate::model::slp_model::SLPModel::default_capabilities(&self.0)
            }
            fn thinking_default(&self) -> bool {
                $crate::model::slp_model::SLPModel::thinking_default(&self.0)
            }
            fn vision_chat_handler(&self) -> Option<String> {
                $crate::model::slp_model::SLPModel::vision_chat_handler(&self.0)
            }
            fn get_instance(&self) -> ::anyhow::Result<::std::sync::Arc<dyn $crate::llms::base_llm::BaseLLM>> {
                $crate::model::slp_model::SLPModel::get_instance(&self.0)
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_split_matches_urllib() {
        let parts = url_split("https://example.com/a/b.pdf?x=1#frag").unwrap();
        assert_eq!(parts.scheme, "https");
        assert_eq!(parts.netloc, "example.com");
        assert_eq!(parts.path, "/a/b.pdf");

        // A Windows drive letter parses as a scheme with no netloc.
        let parts = url_split("C:\\Users\\me\\doc.pdf").unwrap();
        assert_eq!(parts.scheme, "c");
        assert_eq!(parts.netloc, "");

        // file:///x has an empty netloc.
        assert_eq!(url_split("file:///tmp/x.txt").unwrap().netloc, "");
        assert!(url_split("http://[::1/x").is_err());
    }

    #[test]
    fn pathlib_name_helpers() {
        assert_eq!(pure_name("/a/b.pdf"), "b.pdf");
        assert_eq!(pure_name("/a/b/"), "b");
        assert_eq!(pure_name("/"), "");
        assert_eq!(stem_and_suffix("report.final.pdf"), ("report.final".to_string(), ".pdf".to_string()));
        assert_eq!(stem_and_suffix(".bashrc"), (".bashrc".to_string(), String::new()));
        assert_eq!(stem_and_suffix("trailing."), ("trailing.".to_string(), String::new()));
    }
}
