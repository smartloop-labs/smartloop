//! Port of `smartloop/db/workspace_indexer.py`: indexes local project files
//! into a dedicated collection (`workspace_files`) for language-agnostic
//! semantic search.
//!
//! Files are split into overlapping chunks (≤ [`CHUNK_SIZE`] chars) so they
//! fit within the embedding model's token window.  A SHA-256 manifest
//! persisted alongside the database tracks which files are up-to-date so only
//! changed or new files are re-indexed on subsequent runs.

use std::collections::{BTreeMap, HashSet};
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use md5::Md5;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::db::embeddings::indexing_burst;
use crate::db::store::{Chunk, ChunkResult, VectorStore};
use crate::db::store_factory::StoreFactory;
use crate::utils::json_to_table;
use crate::workspace::workspace::PROJECT_STATE_FILE;

/// Logger name of the Python module.
pub const LOG_TARGET: &str = "smartloop.db.workspace_indexer";

/// Collection (index `type`) holding workspace files.
pub const WORKSPACE_COLLECTION: &str = "workspace_files";

/// Characters per chunk (~500 tokens for English text, fits the 512-token
/// embedding model window).  Overlap avoids losing context at boundaries.
pub const CHUNK_SIZE: usize = 2000;
/// Overlap between consecutive chunks.
pub const CHUNK_OVERLAP: usize = 200;

/// File extensions that are indexed (lower-case, with leading dot).
pub const INDEX_EXTENSIONS: &[&str] = &[
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", ".cs",
    ".rb", ".php", ".swift", ".kt", ".scala", ".yaml", ".yml", ".toml", ".json", ".cfg", ".ini",
    ".md", ".txt", ".rst", ".html", ".css", ".sql", ".proto", ".graphql", ".sh", ".csv",
];

/// Documents indexed through a [`DocumentConverter`], when the indexer has
/// one: binary formats that have to be parsed to text first.
pub const DOCUMENT_EXTENSIONS: &[&str] = &[".pdf", ".docx", ".pptx", ".xlsx"];

/// Documents larger than this are left out rather than parsed.
pub const MAX_DOCUMENT_BYTES: u64 = 50 * 1024 * 1024;

/// Parses a document (see [`DOCUMENT_EXTENSIONS`]) to text.
pub type DocumentConverter = Arc<dyn Fn(&Path) -> anyhow::Result<String> + Send + Sync>;

/// Directory names skipped anywhere in the tree (plus `*.egg-info`).
pub const SKIP_DIRS: &[&str] = &[
    ".git", "__pycache__", "node_modules", "venv", ".venv", "dist", "build", ".tox", ".mypy_cache",
    ".ruff_cache", ".eggs", ".pytest_cache", ".out", "skills",
];

/// The indexer's own manifest file, under `project_dir`.
pub const MANIFEST_FILENAME: &str = ".workspace_manifest.json";

/// Directories holding the framework's own state, skipped when they sit at
/// the root of the scan when the scan root is `project_dir` (an unbound
/// project, see the indexer task), where the vector store's own files are.
///
/// `documents` holds the uploaded documents, which the document indexer
/// already indexes.  Scanning it too indexed every `.md`/`.txt` upload a
/// second time as workspace rows, so a 374-document project kept writing
/// chunks long after indexing finished and searches returned each twice.
pub const SKIP_ROOT_DIRS: &[&str] = &["db", "documents"];

/// Files holding the framework's own state, skipped at the root of the scan
/// for the same reason as [`SKIP_ROOT_DIRS`] — and for the reason hidden
/// files are skipped, which `project.json` shares without being hidden.
///
/// It is the project's own record, rewritten on every turn (a new session, a
/// document, an MCP registration), and it grows with use: by a few hundred
/// conversations it was a megabyte, 567 chunks.  Indexing it meant every turn
/// changed its hash, so every scan deleted and re-embedded all of it — a
/// minute of GPU spent re-learning the app's own bookkeeping, competing with
/// the reply the user was waiting for, and never answering a question anyone
/// would ask of a workspace.
pub const SKIP_ROOT_FILES: &[&str] = &[PROJECT_STATE_FILE];

/// What the manifest records for a file: its content hash, and the size and
/// modified time (nanoseconds since the epoch) it had when hashed.  While the
/// size and modified time still match, the file is taken as unchanged without
/// being read, so a pass over a large folder costs a `stat` per file rather
/// than a full read.  A write that keeps the size and lands within the
/// filesystem's timestamp resolution goes unnoticed, the trade every
/// mtime-based tool (make, git's index) makes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestEntry {
    pub hash: String,
    pub size: u64,
    pub mtime: u64,
}

/// `path`'s `(size, mtime)`, or `None` when it can't be read.
fn stat(path: &Path) -> Option<(u64, u64)> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos();
    Some((meta.len(), u64::try_from(mtime).ok()?))
}

/// Result of an [`WorkspaceIndexer::index`] pass:
/// `{"indexed": N, "skipped": N, "removed": N}`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct IndexStats {
    pub indexed: usize,
    pub skipped: usize,
    pub removed: usize,
}

/// SHA-256 hex digest of a file.
pub fn file_hash(path: &str) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Split `text` into overlapping chunks of at most `size` characters
/// (Python defaults: [`CHUNK_SIZE`], [`CHUNK_OVERLAP`]).
pub fn chunk_text(text: &str, size: usize, overlap: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= size {
        return vec![text.to_string()];
    }
    // A non-positive step would never advance (Python loops forever there);
    // step at least one character.
    let step = size.saturating_sub(overlap).max(1);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < chars.len() {
        let end = (start + size).min(chars.len());
        chunks.push(chars[start..end].iter().collect());
        start += step;
    }
    chunks
}

/// Stable, short document ID (MD5 hex) derived from a relative file path.
pub fn stable_id(rel_path: &str) -> String {
    hex::encode(Md5::digest(rel_path.as_bytes()))
}

/// `Path.read_text(encoding="utf-8", errors="replace")`: lossy UTF-8 with
/// universal-newline translation.
fn read_text_lossy(path: &Path) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    let text = String::from_utf8_lossy(&bytes);
    Ok(text.replace("\r\n", "\n").replace('\r', "\n"))
}

/// Indexes workspace files into a dedicated collection.
///
/// The indexer is *incremental*: on each `index()` call it reads the on-disk
/// manifest, hashes every candidate file, and only re-indexes files whose
/// hash has changed.  Deleted files are removed from the collection.
///
/// ```ignore
/// let indexer = WorkspaceIndexer::new("/path/to/.slp/my-project", None)?;
/// indexer.index("/path/to/workspace"); // incremental
/// let results = indexer.search("authentication logic", 10, 0.65);
/// ```
pub struct WorkspaceIndexer {
    project_dir: String,
    store: Arc<dyn VectorStore>,
    manifest_path: String,
    /// rel_path → what it was when last hashed.
    manifest: Mutex<BTreeMap<String, ManifestEntry>>,
    /// Serialises concurrent `index()` calls.
    lock: Mutex<()>,
    /// Parses documents; without one they are left out of the index.
    converter: Option<DocumentConverter>,
    /// What is happening to a document right now, for progress: `Reading
    /// <file>` while it is parsed, `Indexing <file>` while it is embedded.
    activity: Mutex<Option<String>>,
}

impl WorkspaceIndexer {
    /// Create an indexer backed by the project's store and load its manifest.
    pub fn new(project_dir: &str, base_dir: Option<&str>) -> anyhow::Result<Self> {
        let store = StoreFactory::create_store(project_dir, WORKSPACE_COLLECTION, base_dir, None)?;
        Ok(Self::with_store(project_dir, store))
    }

    /// Build an indexer over an explicit store (test seam); loads the manifest
    /// from `<project_dir>/.workspace_manifest.json`.
    pub fn with_store(project_dir: &str, store: Arc<dyn VectorStore>) -> Self {
        let manifest_path = Path::new(project_dir)
            .join(MANIFEST_FILENAME)
            .to_string_lossy()
            .into_owned();
        let manifest = Self::load_manifest(&manifest_path);
        Self {
            project_dir: project_dir.to_string(),
            store,
            manifest_path,
            manifest: Mutex::new(manifest),
            lock: Mutex::new(()),
            converter: None,
            activity: Mutex::new(None),
        }
    }

    /// What is happening to a document right now, if anything.
    pub fn activity(&self) -> Option<String> {
        self.activity.lock().clone()
    }

    /// Index documents too, parsed to text by `converter`.
    pub fn with_converter(mut self, converter: DocumentConverter) -> Self {
        self.converter = Some(converter);
        self
    }

    /// Project directory this indexer belongs to.
    pub fn project_dir(&self) -> &str {
        &self.project_dir
    }

    /// Path of the manifest file.
    pub fn manifest_path(&self) -> &str {
        &self.manifest_path
    }

    /// Number of files currently recorded in the manifest.
    pub fn file_count(&self) -> usize {
        self.manifest.lock().len()
    }

    /// The in-memory manifest's hashes (rel_path → sha256).
    pub fn manifest(&self) -> BTreeMap<String, String> {
        self.manifest.lock().iter().map(|(rel, entry)| (rel.clone(), entry.hash.clone())).collect()
    }

    /// Replace the in-memory manifest (test seam for assigning `_manifest`).
    /// No size or modified time is recorded, so each file is hashed again.
    pub fn set_manifest(&self, manifest: BTreeMap<String, String>) {
        *self.manifest.lock() = manifest
            .into_iter()
            .map(|(rel, hash)| (rel, ManifestEntry { hash, size: 0, mtime: 0 }))
            .collect();
    }

    /// Walk `work_dir` and incrementally index changed/new files inside an
    /// indexing burst.  Thread-safe — concurrent calls are serialised.
    pub fn index(&self, work_dir: &str) -> IndexStats {
        let _guard = self.lock.lock();
        let _burst = indexing_burst();
        self.index_inner(work_dir)
    }

    /// Semantic search over indexed workspace files (Python defaults:
    /// `n_results=10`, `max_distance=0.65`), ordered by relevance (closest
    /// distance first).  Failures are logged and yield an empty list.
    pub fn search(&self, query: &str, n_results: usize, max_distance: f64) -> Vec<ChunkResult> {
        match self.store.query_with_metadata(query, Some(n_results), max_distance, None) {
            Ok(results) => results,
            Err(e) => {
                tracing::warn!(target: LOG_TARGET, "[workspace-index] Search failed: {e}");
                Vec::new()
            }
        }
    }

    /// Load the manifest (empty on missing/corrupt file, with a warning).
    fn load_manifest(manifest_path: &str) -> BTreeMap<String, ManifestEntry> {
        if Path::new(manifest_path).is_file() {
            let loaded = std::fs::read_to_string(manifest_path)
                .map_err(anyhow::Error::from)
                .and_then(|raw| Ok(serde_json::from_str::<BTreeMap<String, ManifestEntry>>(&raw)?));
            match loaded {
                Ok(manifest) => return manifest,
                Err(e) => {
                    tracing::warn!(target: LOG_TARGET, "[workspace-index] Could not load manifest: {e}");
                }
            }
        }
        BTreeMap::new()
    }

    /// Persist the manifest (`indent=2` JSON); failures are logged.
    fn save_manifest(&self, manifest: &BTreeMap<String, ManifestEntry>) {
        let written = py_pretty_json(manifest)
            .and_then(|json| Ok(std::fs::write(&self.manifest_path, json)?));
        if let Err(e) = written {
            tracing::warn!(target: LOG_TARGET, "[workspace-index] Could not save manifest: {e}");
        }
    }

    /// The indexing pass itself (Python `_index`), without the lock or the
    /// burst scope.  Returns `{"indexed": N, "skipped": N, "removed": N}`.
    pub fn index_inner(&self, work_dir: &str) -> IndexStats {
        let work_dir: PathBuf =
            std::fs::canonicalize(work_dir).unwrap_or_else(|_| PathBuf::from(work_dir));
        let mut stats = IndexStats::default();
        let old_manifest = self.manifest.lock().clone();

        // The framework's own state sits at the root only when the scan is of
        // `project_dir` itself; a user's folder may well have a `documents/`.
        let own_state = std::fs::canonicalize(&self.project_dir).is_ok_and(|dir| dir == work_dir);

        // Collect all current candidate files: rel_path -> abs_path
        let mut current_files: Vec<(String, PathBuf)> = Vec::new();
        walk(&work_dir, &work_dir, own_state, &mut current_files);
        let current_set: HashSet<&str> = current_files.iter().map(|(rel, _)| rel.as_str()).collect();

        let mut new_manifest: BTreeMap<String, ManifestEntry> = BTreeMap::new();

        // Index new / changed files
        for (rel_path, abs_path) in &current_files {
            let document = is_document(abs_path);
            if document
                && (self.converter.is_none()
                    || std::fs::metadata(abs_path).map_or(true, |m| m.len() > MAX_DOCUMENT_BYTES))
            {
                continue;
            }
            // Same size and modified time as when it was last hashed: taken
            // as unchanged without reading it.
            let (size, mtime) = stat(abs_path).unwrap_or_default();
            if let Some(old) = old_manifest.get(rel_path)
                && mtime != 0
                && (old.size, old.mtime) == (size, mtime)
            {
                new_manifest.insert(rel_path.clone(), old.clone());
                stats.skipped += 1;
                continue;
            }

            let hash = match file_hash(&abs_path.to_string_lossy()) {
                Ok(h) => h,
                Err(e) => {
                    tracing::debug!(target: LOG_TARGET, "[workspace-index] Cannot hash {rel_path}: {e}");
                    continue;
                }
            };

            new_manifest.insert(rel_path.clone(), ManifestEntry { hash: hash.clone(), size, mtime });

            if old_manifest.get(rel_path).is_some_and(|old| old.hash == hash) {
                stats.skipped += 1;
                continue;
            }

            // A file to embed: not while a turn is answered, which would
            // share the device with it; picked up once the turn is done.
            if crate::tasks::turn_gate::wait_for_turns() {
                tracing::debug!(target: LOG_TARGET, "[workspace-index] Resumed after a turn at {rel_path}");
            }
            let _work = crate::tasks::turn_gate::index_work("workspace", rel_path);

            let read = match &self.converter {
                Some(convert) if document => {
                    tracing::info!(target: LOG_TARGET, "[workspace-index] Parsing {rel_path}");
                    *self.activity.lock() = Some(format!("Reading {rel_path}"));
                    let parsed = convert(abs_path);
                    *self.activity.lock() = None;
                    parsed
                }
                _ => read_text_lossy(abs_path).map_err(anyhow::Error::from),
            };
            let mut text = match read {
                Ok(t) => t,
                Err(e) => {
                    // Kept in the manifest, so a document that won't parse
                    // isn't parsed again every pass until it changes.
                    if document {
                        tracing::warn!(target: LOG_TARGET, "[workspace-index] Cannot parse {rel_path}: {e}");
                    } else {
                        tracing::debug!(target: LOG_TARGET, "[workspace-index] Cannot read {rel_path}: {e}");
                    }
                    continue;
                }
            };

            if text.trim().is_empty() {
                stats.skipped += 1;
                continue;
            }

            // Structured JSON documents (an array of objects) are converted
            // to a markdown table BEFORE chunking so the workspace index
            // holds clean table rows instead of fragmented JSON text —
            // matching the main document indexer's behavior.
            if let Some(table) = json_to_table(&text) {
                tracing::info!(
                    target: LOG_TARGET,
                    "[workspace-index] {}: converted JSON → table ({} chars)",
                    rel_path,
                    table.chars().count()
                );
                text = table;
            }

            let doc_id = stable_id(rel_path);
            let chunks = chunk_text(&text, CHUNK_SIZE, CHUNK_OVERLAP);

            // Remove stale chunks before re-adding
            if old_manifest.contains_key(rel_path) {
                let _ = self.store.delete_document(&doc_id);
            }

            let chunk_dicts: Vec<Chunk> = chunks
                .into_iter()
                .enumerate()
                .map(|(i, chunk)| {
                    let mut d = Map::new();
                    d.insert("text".to_string(), Value::String(chunk));
                    d.insert("document_id".to_string(), Value::String(doc_id.clone()));
                    d.insert("chunk_index".to_string(), Value::from(i));
                    d.insert("document_name".to_string(), Value::String(rel_path.clone()));
                    d
                })
                .collect();

            if document {
                *self.activity.lock() = Some(format!("Indexing {rel_path}"));
            }
            let added = self.store.add_chunks(&chunk_dicts);
            *self.activity.lock() = None;
            match added {
                Ok(()) => stats.indexed += 1,
                Err(e) => {
                    tracing::warn!(target: LOG_TARGET, "[workspace-index] Failed to index {rel_path}: {e}");
                }
            }
        }

        // Remove deleted files from the index
        for rel_path in old_manifest.keys() {
            if !current_set.contains(rel_path.as_str())
                && self.store.delete_document(&stable_id(rel_path)).is_ok()
            {
                stats.removed += 1;
            }
        }

        let total = new_manifest.len();
        self.save_manifest(&new_manifest);
        *self.manifest.lock() = new_manifest;

        tracing::info!(
            target: LOG_TARGET,
            "[workspace-index] Complete — indexed={}, skipped={}, removed={}, total={} files",
            stats.indexed,
            stats.skipped,
            stats.removed,
            total
        );
        stats
    }
}

/// `os.walk(work_dir, topdown=True)` with the indexer's pruning: hidden,
/// skipped and `*.egg-info` directories anywhere, [`SKIP_ROOT_DIRS`] at the root,
/// directories visited in sorted order, symlinked directories not followed.
/// [`SKIP_ROOT_DIRS`] and [`SKIP_ROOT_FILES`] apply only when `own_state`.
fn walk(root: &Path, work_dir: &Path, own_state: bool, out: &mut Vec<(String, PathBuf)>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    let at_root = own_state && std::fs::canonicalize(root).map(|r| r == work_dir).unwrap_or(false);

    let mut dirs: Vec<(String, bool)> = Vec::new();
    let mut files: Vec<String> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.path().is_dir() {
            let is_symlink = entry.file_type().map(|t| t.is_symlink()).unwrap_or(false);
            dirs.push((name, is_symlink));
        } else {
            files.push(name);
        }
    }

    for fname in files {
        // Hidden files are state, not content — and one of them is this
        // indexer's own manifest, written at the end of every run.  Indexing
        // it made every run dirty the input to the next one: a file
        // re-embedded every scan, forever, for nothing.
        if fname.starts_with('.') {
            continue;
        }
        if at_root && SKIP_ROOT_FILES.contains(&fname.as_str()) {
            continue;
        }
        let ext = Path::new(&fname)
            .extension()
            .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
            .unwrap_or_default();
        if !INDEX_EXTENSIONS.contains(&ext.as_str()) && !DOCUMENT_EXTENSIONS.contains(&ext.as_str()) {
            continue;
        }
        let abs_path = root.join(&fname);
        let rel_path = abs_path
            .strip_prefix(work_dir)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| abs_path.to_string_lossy().into_owned());
        out.push((rel_path, abs_path));
    }

    // Hidden directories are state too: `~/.smartloop`, `.cache`, `.config`
    // when the folder is the home one.
    dirs.retain(|(d, _)| {
        !d.starts_with('.')
            && !SKIP_DIRS.contains(&d.as_str())
            && !d.ends_with(".egg-info")
            && !(at_root && SKIP_ROOT_DIRS.contains(&d.as_str()))
    });
    dirs.sort();
    for (d, is_symlink) in dirs {
        // os.walk(followlinks=False) lists a symlinked directory but does not
        // descend into it.
        if is_symlink {
            continue;
        }
        walk(&root.join(d), work_dir, own_state, out);
    }
}

/// Whether `path` is one of [`DOCUMENT_EXTENSIONS`].
fn is_document(path: &Path) -> bool {
    let ext = path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
        .unwrap_or_default();
    DOCUMENT_EXTENSIONS.contains(&ext.as_str())
}

/// `json.dump(obj, f, indent=2)`: two-space indentation, `": "` separators,
/// non-ASCII escaped.
fn py_pretty_json<T: Serialize>(manifest: &T) -> anyhow::Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, AsciiPrettyFormatter::new());
    manifest.serialize(&mut ser)?;
    Ok(out)
}

/// `serde_json`'s pretty formatter with Python's `ensure_ascii=True`.
struct AsciiPrettyFormatter<'a> {
    inner: serde_json::ser::PrettyFormatter<'a>,
}

impl AsciiPrettyFormatter<'_> {
    fn new() -> Self {
        Self { inner: serde_json::ser::PrettyFormatter::with_indent(b"  ") }
    }
}

impl serde_json::ser::Formatter for AsciiPrettyFormatter<'_> {
    fn begin_array<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        self.inner.begin_array(w)
    }
    fn end_array<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        self.inner.end_array(w)
    }
    fn begin_array_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        self.inner.begin_array_value(w, first)
    }
    fn end_array_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        self.inner.end_array_value(w)
    }
    fn begin_object<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        self.inner.begin_object(w)
    }
    fn end_object<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        self.inner.end_object(w)
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        self.inner.begin_object_key(w, first)
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        self.inner.begin_object_value(w)
    }
    fn end_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        self.inner.end_object_value(w)
    }
    fn write_string_fragment<W: ?Sized + std::io::Write>(&mut self, w: &mut W, fragment: &str) -> std::io::Result<()> {
        for ch in fragment.chars() {
            if (' '..='~').contains(&ch) {
                w.write_all(&[ch as u8])?;
            } else {
                let mut buf = [0u16; 2];
                for unit in ch.encode_utf16(&mut buf) {
                    write!(w, "\\u{:04x}", unit)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_text_overlaps_like_python() {
        assert_eq!(chunk_text("short", CHUNK_SIZE, CHUNK_OVERLAP), vec!["short".to_string()]);
        // size 4, overlap 1: starts at 0, 3, 6, 9.
        assert_eq!(chunk_text("abcdefghij", 4, 1), vec!["abcd", "defg", "ghij", "j"]);
    }

    #[test]
    fn stable_id_is_the_md5_hex_of_the_path() {
        assert_eq!(stable_id("a"), "0cc175b9c0f1b6a831c399e269772661");
    }

    #[test]
    fn manifest_is_written_like_json_dump_indent_2() {
        let manifest = BTreeMap::from([
            ("a.md".to_string(), "h1".to_string()),
            ("\u{e9}.md".to_string(), "h2".to_string()),
        ]);
        let out = String::from_utf8(py_pretty_json(&manifest).unwrap()).unwrap();
        // ensure_ascii: the accented key is escaped (backslash, then u00e9).
        let backslash = char::from(92u8);
        let expected = format!("{{\n  \"a.md\": \"h1\",\n  \"{backslash}u00e9.md\": \"h2\"\n}}");
        assert_eq!(out, expected);
    }
}
