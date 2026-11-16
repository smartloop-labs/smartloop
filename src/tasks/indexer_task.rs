//! Port of `smartloop/tasks/indexer_task.py`.
//!
//! Indexer Task — indexes documents into the vector store via a persistent
//! event loop.
//!
//! Each enqueued document is dispatched to a worker pool sized to the CPU
//! count so uploads can be processed in parallel.  A single background thread
//! blocks on an internal queue; when a document is uploaded its ID is pushed
//! onto the queue and the loop wakes immediately to process it.
//!
//! Documents always stay in `<project_dir>/documents/` (the ingest folder).
//! Conversion to markdown happens in-memory; `doc.path` is never updated to an
//! internal `sources/` path.
//!
//! A JSONL manifest at `<project_dir>/documents.jsonl` tracks every indexed
//! file.  The scan loop uses it to detect deletions: files in the manifest but
//! missing from the folder are removed from the vector store and workspace
//! config.

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use md5::{Digest, Md5};
use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use regex::Regex;
use serde_json::{json, Map, Value};

use crate::api::services::get_model_service;
use crate::db::embeddings::device_idle_seconds;
use crate::db::store::Chunk;
use crate::db::store_factory::StoreFactory;
use crate::db::workspace_indexer::{DocumentConverter, IndexStats, WorkspaceIndexer};
use crate::llms::base_llm::BaseLLM;
use crate::parsers::base_parser::{ParserError, SharedLLM};
use crate::parsers::parser_factory::ParserFactory;
use crate::tasks::base_task::{
    panic_message, BaseTask, TaskCore, TaskKwargs, TaskLogger, TaskOptions, TaskResult, TaskStatus,
    WorkQueue,
};
use crate::tasks::documents_manifest::DocumentsManifest;
use crate::utils::doc_splitter::{json_to_table, DocumentSplitter};
use crate::utils::path_utils::normalize_path;
use crate::workspace::context::ProjectBoundExecutor;
use crate::workspace::context::{pinned_project_id, use_project};
use crate::workspace::types::{Document, DocumentStatus};
use crate::workspace::workspace::{Project, SharedWorkspace, Workspace};

/// Progress listener; receives
/// `{"stage", "filename", "current", "total", "message"}`.
pub type IndexProgressCallback = Arc<dyn Fn(Value) + Send + Sync>;

/// Seam returning seconds since the embedding device last generated
/// (defaults to [`device_idle_seconds`]; tests inject a fixed reading).
pub type IdleSecondsProvider = Arc<dyn Fn() -> f64 + Send + Sync>;

/// Seam returning the project a pass indexes (defaults to the workspace
/// lookup in `project()`; tests inject a fixed project).
pub type ProjectProvider = Arc<dyn Fn() -> Option<Project> + Send + Sync>;

/// The slice of [`WorkspaceIndexer`] the indexer task drives; a trait so tests
/// can substitute a recording fake.
pub trait WorkspaceIndex: Send + Sync {
    /// Run one hash-incremental pass over `work_dir`.
    fn index(&self, work_dir: &str) -> IndexStats;
    /// Number of files currently indexed.
    fn file_count(&self) -> usize;
    /// What is happening to a document right now (`Reading <file>`,
    /// `Indexing <file>`), if anything.
    fn activity(&self) -> Option<String> {
        None
    }
}

impl WorkspaceIndex for WorkspaceIndexer {
    fn index(&self, work_dir: &str) -> IndexStats {
        WorkspaceIndexer::index(self, work_dir)
    }
    fn activity(&self) -> Option<String> {
        WorkspaceIndexer::activity(self)
    }
    fn file_count(&self) -> usize {
        WorkspaceIndexer::file_count(self)
    }
}

/// Seconds on a monotonic clock (Python `time.monotonic()`).
fn monotonic() -> f64 {
    static BASE: Lazy<Instant> = Lazy::new(Instant::now);
    BASE.elapsed().as_secs_f64()
}

/// Seconds since the UNIX epoch (Python `time.time()`).
fn epoch_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// `s[:n]` by characters.
fn prefix_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

/// Collapse `//` and interior `.` components the way `pathlib.Path` does.
fn pathlib_normalize(path: &str) -> PathBuf {
    Path::new(path).components().collect()
}

/// Python `mimetypes.guess_type(path)[0]`, rendered as Python formats a
/// missing value (`None`) so a parser lookup failure reads the same.
fn guess_mime_type(path: &Path) -> String {
    mime_guess::from_path(path)
        .first_raw()
        .map(str::to_string)
        .unwrap_or_else(|| "None".to_string())
}

/// Background task that processes documents from a queue in a persistent
/// event loop.
pub struct IndexerTask {
    core: TaskCore,
    me: Weak<IndexerTask>,
    pub workspace: RwLock<SharedWorkspace>,
    /// The project this indexer serves.  A per-project agent child passes its
    /// own id, so its loop indexes that project whether or not it is the one
    /// the UI has selected; left unset the task follows the selection.
    pub project_id: Option<String>,
    pub progress_callback: Option<IndexProgressCallback>,
    loop_thread: Mutex<Option<JoinHandle<()>>>,
    queue: WorkQueue<String>,
    max_workers: usize,
    executor: Mutex<Option<Arc<ProjectBoundExecutor>>>,
    workspace_lock: Mutex<()>,
    inflight: Mutex<HashSet<String>>,
    /// See `MCPTask` — same guard, same two callers, same race.
    start_lock: Mutex<()>,
    /// Monotonic seconds of the last documents-folder scan (`0.0` = never).
    last_scan: Mutex<f64>,
    /// Wall-clock (epoch) seconds of the last dispatched workspace scan
    /// (`0.0` = never / requested).
    last_workspace_scan: Mutex<f64>,
    workspace_indexer: Mutex<Option<Arc<dyn WorkspaceIndex>>>,
    workspace_indexer_dir: Mutex<Option<String>>,
    workspace_scan_inflight: AtomicBool,
    /// Wall-clock time and stats of the last completed workspace pass,
    /// surfaced via `workspace_status` for `/health`.
    last_workspace_scan_at: Mutex<Option<f64>>,
    last_workspace_stats: Mutex<Option<IndexStats>>,
    /// True once a document has been processed since the queue last went
    /// idle; used to release the on-demand vision model exactly once when the
    /// queue drains (so only the orchestrator stays resident).
    processed_since_idle: AtomicBool,
    idle_seconds_provider: RwLock<Option<IdleSecondsProvider>>,
    project_provider: RwLock<Option<ProjectProvider>>,
}

impl IndexerTask {
    /// Documents-folder scan cadence (seconds).
    pub const SCAN_INTERVAL: f64 = 5.0;
    /// Workspace (project working-directory) scan cadence (seconds) — slower
    /// than the documents folder; the repo is larger and its index is only
    /// consulted opportunistically during chat.
    pub const WORKSPACE_SCAN_INTERVAL: f64 = 30.0;
    /// How long the model has to have been quiet before a workspace pass
    /// starts.  Covers the gaps *within* a turn, not just an active call —
    /// see the check in [`IndexerTask::scan_workspace_files`].
    pub const WORKSPACE_QUIET_SECONDS: f64 = 5.0;

    /// Task name `"indexer"`, `auto=true`.  `project_id` defaults to
    /// [`pinned_project_id`].
    pub fn new(
        workspace: SharedWorkspace,
        logger: TaskLogger,
        progress_callback: Option<IndexProgressCallback>,
        project_id: Option<String>,
        options: TaskOptions,
    ) -> Arc<Self> {
        // The project this indexer serves.  A per-project agent child passes
        // its own id, so its loop indexes that project whether or not it is
        // the one the UI has selected; left unset (the single-process case)
        // the task keeps following the selection, as it always has.
        let project_id = project_id.filter(|id| !id.is_empty()).or_else(pinned_project_id);
        let max_workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).max(1);
        Arc::new_cyclic(|me| Self {
            core: TaskCore::new("indexer", logger, true, options),
            me: me.clone(),
            workspace: RwLock::new(workspace),
            project_id,
            progress_callback,
            loop_thread: Mutex::new(None),
            queue: WorkQueue::new(),
            max_workers,
            executor: Mutex::new(None),
            workspace_lock: Mutex::new(()),
            inflight: Mutex::new(HashSet::new()),
            start_lock: Mutex::new(()),
            last_scan: Mutex::new(0.0),
            last_workspace_scan: Mutex::new(0.0),
            workspace_indexer: Mutex::new(None),
            workspace_indexer_dir: Mutex::new(None),
            workspace_scan_inflight: AtomicBool::new(false),
            last_workspace_scan_at: Mutex::new(None),
            last_workspace_stats: Mutex::new(None),
            processed_since_idle: AtomicBool::new(false),
            idle_seconds_provider: RwLock::new(None),
            project_provider: RwLock::new(None),
        })
    }

    // ---------------------------------------------------------------
    // Test / injection seams
    // ---------------------------------------------------------------

    /// Override the idle-seconds reading (`None` restores the default).
    pub fn set_idle_seconds_provider(&self, provider: Option<IdleSecondsProvider>) {
        *self.idle_seconds_provider.write() = provider;
    }

    /// Override the project lookup (`None` restores the default).
    pub fn set_project_provider(&self, provider: Option<ProjectProvider>) {
        *self.project_provider.write() = provider;
    }

    /// Install a workspace indexer bound to `project_dir`.
    pub fn set_workspace_indexer(&self, indexer: Option<Arc<dyn WorkspaceIndex>>, project_dir: Option<String>) {
        *self.workspace_indexer.lock() = indexer;
        *self.workspace_indexer_dir.lock() = project_dir;
    }

    /// Wall-clock (epoch) seconds of the last dispatched workspace scan.
    pub fn last_workspace_scan(&self) -> f64 {
        *self.last_workspace_scan.lock()
    }

    pub fn set_last_workspace_scan(&self, value: f64) {
        *self.last_workspace_scan.lock() = value;
    }

    /// The workspace handle currently in use.
    fn workspace(&self) -> SharedWorkspace {
        self.workspace.read().clone()
    }

    // ---------------------------------------------------------------
    // Public queue API
    // ---------------------------------------------------------------

    /// Push one or more documents onto the indexer queue; returns the count.
    pub fn enqueue(&self, documents: &[Document]) -> usize {
        let mut count = 0;
        for doc in documents {
            self.queue.put(doc.id.clone());
            count += 1;
        }
        if count > 0 {
            self.core
                .logger
                .info(&format!("[Indexer] {count} document(s) enqueued for indexing"));
        }
        count
    }

    pub fn queue_size(&self) -> usize {
        self.queue.qsize()
    }

    /// Queue the project's documents a previous run registered but never
    /// finished: `pending`, `converting` or `indexing` when the loop starts.
    ///
    /// The queue lives in memory and the folder scan only enqueues files it
    /// has not seen, so without this a restart mid-index leaves every
    /// unfinished document in its status forever.  Returns the count.
    pub fn requeue_unfinished(&self) -> usize {
        let Some(project) = self.project() else {
            return 0;
        };
        let unfinished: Vec<Document> = self
            .project_documents(&project)
            .into_iter()
            .filter(|d| {
                matches!(
                    d.status,
                    DocumentStatus::Pending | DocumentStatus::Converting | DocumentStatus::Indexing
                )
            })
            .collect();
        if !unfinished.is_empty() {
            self.core.logger.info(&format!(
                "[Indexer] Resuming {} unfinished document(s)",
                unfinished.len()
            ));
        }
        self.enqueue(&unfinished)
    }

    // ---------------------------------------------------------------
    // Event loop
    // ---------------------------------------------------------------

    /// Start the `index-loop` thread and `index-worker` pool (idempotent).
    pub fn start_loop(&self) {
        let _guard = self.start_lock.lock();
        let alive = self
            .loop_thread
            .lock()
            .as_ref()
            .is_some_and(|handle| !handle.is_finished());
        if alive {
            return;
        }
        let Some(me) = self.me.upgrade() else {
            return;
        };
        self.core.stop_event.clear();
        *self.executor.lock() =
            Some(Arc::new(ProjectBoundExecutor::new(self.max_workers, Some("index-worker"))));
        let spawned = std::thread::Builder::new()
            .name("index-loop".to_string())
            .spawn(move || me.event_loop());
        match spawned {
            Ok(handle) => {
                *self.loop_thread.lock() = Some(handle);
                self.core.logger.info(&format!(
                    "[Indexer] Event loop started ({} worker(s))",
                    self.max_workers
                ));
            }
            Err(exc) => self.core.logger.error(&format!("[Indexer] Event loop error: {exc}")),
        }
    }

    /// Alias for [`IndexerTask::start_loop`].
    pub fn start_watcher(&self) {
        self.start_loop()
    }

    fn event_loop(self: Arc<Self>) {
        self.core.set_status(TaskStatus::Pending, Some("Waiting for documents"));
        self.requeue_unfinished();

        while !self.core.stop_event.is_set() {
            let tick = catch_unwind(AssertUnwindSafe(|| self.tick()));
            let failure = match tick {
                Ok(Ok(())) => None,
                Ok(Err(exc)) => Some(exc.to_string()),
                Err(payload) => Some(panic_message(&*payload)),
            };
            if let Some(exc) = failure {
                self.core.logger.error(&format!("[Indexer] Event loop error: {exc}"));
            }
        }

        if let Some(executor) = self.executor.lock().take() {
            executor.shutdown(false);
        }
        self.core.set_status(TaskStatus::Completed, Some("Event loop stopped"));
    }

    /// One iteration of the loop body.
    fn tick(self: &Arc<Self>) -> anyhow::Result<()> {
        let now = monotonic();
        let last_scan = *self.last_scan.lock();
        if last_scan == 0.0 || now - last_scan >= Self::SCAN_INTERVAL {
            self.scan_documents()?;
            *self.last_scan.lock() = now;
        }

        // Incrementally index the project's working directory into the
        // workspace collection.  Run it on the executor so a large first-time
        // scan never blocks document processing.
        //
        // The workspace cadence runs on the wall clock: the deferral in
        // `scan_workspace_files` re-arms it in wall-clock time, and one clock
        // for both sides keeps a deferral a short postponement.
        let executor = self.executor.lock().clone();
        if let Some(executor) = executor.as_ref() {
            let wall_now = epoch_seconds();
            let due = {
                let last = *self.last_workspace_scan.lock();
                last == 0.0 || wall_now - last >= Self::WORKSPACE_SCAN_INTERVAL
            };
            if !self.workspace_scan_inflight.load(Ordering::SeqCst) && due {
                *self.last_workspace_scan.lock() = wall_now;
                self.workspace_scan_inflight.store(true, Ordering::SeqCst);
                let me = self.clone();
                let _ = executor.submit(move || me.run_workspace_scan());
            }
        }

        let Some(doc_id) = self.queue.get(Duration::from_secs(1)) else {
            let idle = self.inflight.lock().is_empty();
            if idle {
                self.core.set_status(TaskStatus::Pending, Some("Waiting for documents"));
                // Stage boundary: all queued documents are indexed, so the
                // on-demand vision/OCR model is no longer needed — free its
                // VRAM (only the orchestrator stays resident).  Done once per
                // drain to avoid reload churn while idle.
                if self.processed_since_idle.swap(false, Ordering::SeqCst) {
                    self.release_on_demand_models();
                }
            }
            return Ok(());
        };

        {
            let mut inflight = self.inflight.lock();
            if inflight.contains(&doc_id) {
                return Ok(());
            }
            inflight.insert(doc_id.clone());
        }
        self.core.set_status(TaskStatus::Running, Some("Processing document"));
        self.processed_since_idle.store(true, Ordering::SeqCst);
        match executor {
            Some(executor) => {
                let me = self.clone();
                let _ = executor.submit(move || me.worker_run(&doc_id));
                Ok(())
            }
            None => {
                self.inflight.lock().remove(&doc_id);
                Err(anyhow::anyhow!("worker pool is shut down"))
            }
        }
    }

    fn worker_run(&self, doc_id: &str) {
        // Bind the task's project for the whole pass, so anything the parser
        // or model service resolves downstream agrees with the project this
        // document actually belongs to.  A no-op when the task is unscoped.
        // Not while a turn is answered: embedding would share the device
        // with the reply. It's picked up once the turn is done.
        crate::tasks::turn_gate::wait_for_turns();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let _binding = use_project(self.project_id.as_deref());
            self.process_document_by_id(doc_id)
        }));
        let failure = match outcome {
            Ok(Ok(())) => None,
            Ok(Err(exc)) => Some(exc.to_string()),
            Err(payload) => Some(panic_message(&*payload)),
        };
        if let Some(exc) = failure {
            self.core.logger.error(&format!(
                "[Indexer] Worker error for {}: {exc}",
                prefix_chars(doc_id, 8)
            ));
        }
        self.inflight.lock().remove(doc_id);
    }

    // ---------------------------------------------------------------
    // Single-document processing
    // ---------------------------------------------------------------

    fn process_document_by_id(&self, doc_id: &str) -> anyhow::Result<()> {
        let Some(project) = self.project() else {
            self.core.logger.warning("[Indexer] No active project — skipping document");
            return Ok(());
        };

        let Some(doc) = project.documents.iter().find(|d| d.id == doc_id).cloned() else {
            self.core.logger.warning(&format!(
                "[Indexer] Document {} not found in project",
                prefix_chars(doc_id, 8)
            ));
            return Ok(());
        };

        if !matches!(
            doc.status,
            DocumentStatus::Pending | DocumentStatus::Converting | DocumentStatus::Indexing
        ) {
            return Ok(());
        }

        self.process_single_document(&doc, &project);
        Ok(())
    }

    /// Convert and index a document (by id, within `project`).
    ///
    /// `doc.path` always points to the original file in the project's
    /// `<project_dir>/documents/` folder and is never changed.  Conversion to
    /// markdown is done in-memory so no intermediate file is written and the
    /// original is never moved.
    fn process_single_document(&self, doc: &Document, project: &Project) {
        let doc_name = match doc.name.as_deref() {
            Some(name) if !name.is_empty() => name.to_string(),
            _ => Path::new(&doc.path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
        };
        self.core.logger.info(&format!("[Indexer] Processing {doc_name}"));

        if let Some(callback) = self.progress_callback.as_ref() {
            callback(json!({
                "stage": "indexing",
                "filename": doc_name,
                "current": 1,
                "total": 1 + self.queue.qsize(),
                "message": format!("Indexing {doc_name}..."),
            }));
        }

        let manifest = self.get_manifest(project);

        if let Err(exc) = self.convert_and_index(doc, &doc_name, project, &manifest) {
            self.core
                .logger
                .error(&format!("[Indexer] Failed to process {doc_name}: {exc}"));
        }
    }

    /// The body of `_process_single_document`'s `try` block.
    fn convert_and_index(
        &self,
        doc: &Document,
        doc_name: &str,
        project: &Project,
        manifest: &DocumentsManifest,
    ) -> anyhow::Result<()> {
        let doc_path = Path::new(&doc.path);
        if !doc_path.exists() {
            self.core
                .logger
                .warning(&format!("[Indexer] Source file missing, skipping: {}", doc.path));
            return Ok(());
        }

        // Ensure document is in the manifest so the scan loop can detect
        // deletions later (API-uploaded docs skip the scan that normally
        // creates manifest entries).
        manifest.upsert(
            &doc.id,
            doc_name,
            &doc.path,
            DocumentStatus::Converting.as_str(),
            &doc.id,
        );

        self.set_document_status(&project.id, &doc.id, DocumentStatus::Converting)?;

        // --- Convert to plain text / markdown in-memory ---
        let mime_type = guess_mime_type(doc_path);
        let suffix = doc_path
            .extension()
            .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
            .unwrap_or_default();
        let mut text_content = if suffix == ".txt" || suffix == ".md" {
            String::from_utf8_lossy(&std::fs::read(doc_path)?).into_owned()
        } else {
            self.core.logger.info(&format!("[Indexer] Converting {doc_name} to markdown"));
            match self.convert_to_text(&mime_type, &doc.path) {
                Ok(text) => text,
                Err(exc) if is_value_error(&exc) => {
                    self.core
                        .logger
                        .warning(&format!("[Indexer] Failed to convert {doc_name}: {exc}"));
                    self.set_document_status(&project.id, &doc.id, DocumentStatus::Failed)?;
                    manifest.update_status(&doc.id, DocumentStatus::Failed.as_str());
                    return Ok(());
                }
                Err(exc) => return Err(exc),
            }
        };

        self.set_document_status(&project.id, &doc.id, DocumentStatus::Indexing)?;
        manifest.update_status(&doc.id, DocumentStatus::Indexing.as_str());

        let workspace_dir = self.workspace().read().workspace_dir.clone();
        let store = StoreFactory::create_store(&project.project_dir, "documents", Some(&workspace_dir), None)?;

        let llm = self.ensure_model_loaded()?;

        // Structured JSON documents (an array of objects) are converted to a
        // markdown table BEFORE chunking, so the vector index holds clean
        // table rows instead of fragmented JSON text — and the query-time
        // extractor can use the normalized table directly (no overflow).
        if let Some(table) = json_to_table(&text_content) {
            self.core.logger.info(&format!(
                "[Indexer] {doc_name}: converted JSON → table ({} chars)",
                table.chars().count()
            ));
            text_content = table;
        }

        // Split the in-memory text — doc.path stays on the original file
        let text_chunks = DocumentSplitter::new(llm, self.core.logger.clone()).split_text(&text_content);

        let total_chunks = text_chunks.len();

        let mut chunk_dicts: Vec<Chunk> = Vec::with_capacity(total_chunks);
        for (idx, chunk) in text_chunks.iter().enumerate() {
            let page_number = Self::estimate_page_number(chunk, idx, total_chunks, &text_content);

            let mut entry = Map::new();
            entry.insert("text".to_string(), Value::String(Self::compact_for_index(chunk)));
            entry.insert("document_id".to_string(), Value::String(doc.id.clone()));
            entry.insert("document_name".to_string(), Value::String(doc_name.to_string()));
            entry.insert("chunk_index".to_string(), json!(idx));
            entry.insert("total_chunks".to_string(), json!(total_chunks));
            entry.insert("page_number".to_string(), json!(page_number));
            entry.insert("source_content".to_string(), Value::String(chunk.clone()));
            chunk_dicts.push(entry);
        }

        store.add_chunks(&chunk_dicts)?;

        self.set_document_status(&project.id, &doc.id, DocumentStatus::Indexed)?;

        manifest.update_status(&doc.id, DocumentStatus::Indexed.as_str());
        self.core.logger.info(&format!("[Indexer] Indexed {doc_name}"));
        Ok(())
    }

    /// Load the model and parse a non-text document to markdown.
    fn convert_to_text(&self, mime_type: &str, path: &str) -> anyhow::Result<String> {
        let llm = self.ensure_model_loaded()?;
        let shared: SharedLLM = llm;
        let parser = ParserFactory::new(Some(shared)).get_parser(mime_type, path)?;
        parser.parse()
    }

    /// Record `status` on the workspace's copy of the document and persist.
    fn set_document_status(&self, project_id: &str, doc_id: &str, status: DocumentStatus) -> anyhow::Result<()> {
        let _guard = self.workspace_lock.lock();
        let workspace = self.workspace();
        let mut ws = workspace.write();
        if let Some(doc) = ws
            .project_mut(project_id)
            .and_then(|p| p.documents.iter_mut().find(|d| d.id == doc_id))
        {
            doc.status = status;
        }
        ws.save()?;
        Ok(())
    }

    /// The workspace's current status for a document, if it still exists.
    fn document_status(&self, project_id: &str, doc_id: &str) -> Option<DocumentStatus> {
        let workspace = self.workspace();
        let ws = workspace.read();
        ws.project(project_id)
            .and_then(|p| p.documents.iter().find(|d| d.id == doc_id))
            .map(|d| d.status)
    }

    // ---------------------------------------------------------------
    // Index text preparation
    // ---------------------------------------------------------------

    /// Strip formatting and compress text for embedding search.
    ///
    /// The indexed text is used only for semantic similarity matching.  The
    /// full original lives in `source_content` metadata for LLM analysis.
    /// Stripping markdown, punctuation noise, and excess whitespace produces
    /// tighter embeddings.
    pub fn compact_for_index(text: &str) -> String {
        static HEADERS: Lazy<Regex> = Lazy::new(|| Regex::new(r"#{1,6}\s*").unwrap());
        static EMPHASIS: Lazy<Regex> = Lazy::new(|| Regex::new(r"\*{1,3}|_{1,3}").unwrap());
        static LINKS: Lazy<Regex> = Lazy::new(|| Regex::new(r"\[([^\]]*)\]\([^)]*\)").unwrap());
        static IMAGES: Lazy<Regex> = Lazy::new(|| Regex::new(r"!\[([^\]]*)\]\([^)]*\)").unwrap());
        static HTML_TAGS: Lazy<Regex> = Lazy::new(|| Regex::new(r"<[^>]+>").unwrap());
        static BULLETS: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?m)^\s*[-*+]\s+").unwrap());
        static NUMBERED: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?m)^\s*\d+\.\s+").unwrap());
        static WHITESPACE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());

        // Remove markdown formatting (headers, bold, italic, links, images)
        let s = HEADERS.replace_all(text, "");
        let s = EMPHASIS.replace_all(&s, "");
        let s = LINKS.replace_all(&s, "$1");
        let s = IMAGES.replace_all(&s, "");
        // Remove HTML tags
        let s = HTML_TAGS.replace_all(&s, "");
        // Remove bullet/list markers
        let s = BULLETS.replace_all(&s, "");
        let s = NUMBERED.replace_all(&s, "");
        // Collapse whitespace
        let s = WHITESPACE.replace_all(&s, " ");
        s.trim().to_string()
    }

    /// Estimate the page number for a chunk.
    ///
    /// Uses form-feed page breaks when present in the full text.  Falls back
    /// to a proportional estimate based on chunk position (~3000 chars per
    /// page).
    pub fn estimate_page_number(chunk: &str, chunk_idx: usize, total_chunks: usize, full_text: &str) -> usize {
        let head = prefix_chars(chunk, 80);
        // Character offset of the chunk's start in the full text (`str.find`).
        let chunk_start = full_text
            .find(head)
            .map(|byte_idx| full_text[..byte_idx].chars().count());

        // Look for form-feed characters (PDF page breaks in raw text)
        let page_breaks: Vec<usize> = full_text
            .chars()
            .enumerate()
            .filter(|(_, c)| *c == '\u{000C}')
            .map(|(i, _)| i)
            .collect();
        if !page_breaks.is_empty() {
            if let Some(start) = chunk_start {
                let mut page = 1;
                for bp in page_breaks {
                    if bp < start {
                        page += 1;
                    } else {
                        break;
                    }
                }
                return page;
            }
        }

        // Fallback: proportional estimate (assume ~1 page per 3000 chars)
        let chars_per_page = 3000;
        let total_pages = (full_text.chars().count() / chars_per_page).max(1);
        if let Some(start) = chunk_start {
            return total_pages.min((start / chars_per_page + 1).max(1));
        }

        // Last resort: proportional from chunk index
        let fraction = chunk_idx as f64 / total_chunks.max(1) as f64;
        ((fraction * total_pages as f64) as usize + 1).max(1)
    }

    // ---------------------------------------------------------------
    // Folder scan — new files and deletions
    // ---------------------------------------------------------------

    /// Scan the project's documents folder, register new files, and remove
    /// deleted ones.
    ///
    /// Uses the JSONL manifest as the reference of what has been indexed so
    /// that deletions from the folder are reflected in the vector store and
    /// workspace config.
    pub fn scan_documents(&self) -> anyhow::Result<()> {
        let Some(project) = self.project() else {
            return Ok(());
        };

        let inbox_dir = pathlib_normalize(&normalize_path(&project.documents_dir()));
        if !inbox_dir.is_dir() {
            return Ok(());
        }

        let manifest = self.get_manifest(&project);
        let manifest_entries = manifest.entries(); // {doc_id: entry}

        // Current files on disk (skip hidden/system files), in listing order.
        let mut disk_files: Vec<(String, PathBuf)> = Vec::new(); // filename → path
        for entry in std::fs::read_dir(&inbox_dir)? {
            let fp = entry?.path();
            let name = fp
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if fp.is_file() && !name.starts_with('.') {
                disk_files.push((name, fp));
            }
        }
        let disk_names: HashSet<&str> = disk_files.iter().map(|(n, _)| n.as_str()).collect();

        // --- Detect deletions (manifest) ---
        for (doc_id, entry) in &manifest_entries {
            let filename = match entry.get("filename") {
                Some(Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
                None => anyhow::bail!("'filename'"),
            };
            if !disk_names.contains(filename.as_str()) {
                self.core.logger.info(&format!(
                    "[Indexer] File removed from folder: {filename} — cleaning up"
                ));
                self.remove_document(doc_id, &project)?;
                manifest.remove(doc_id);
            }
        }

        // --- Detect deletions (workspace docs missing from disk) ---
        // Catches documents added via the API that were indexed before the
        // manifest upsert fix, so they have no manifest entry.
        for doc in self.project_documents(&project) {
            if doc.status != DocumentStatus::Indexed {
                continue;
            }
            if !doc.path.is_empty() && !Path::new(&doc.path).exists() {
                let label = match doc.name.as_deref() {
                    Some(name) if !name.is_empty() => name.to_string(),
                    _ => doc.path.clone(),
                };
                self.core
                    .logger
                    .info(&format!("[Indexer] File missing from disk: {label} — cleaning up"));
                self.remove_document(&doc.id, &project)?;
                manifest.remove(&doc.id);
            }
        }

        // --- Detect new files ---
        let current_docs = self.project_documents(&project);
        let existing_hashes: HashSet<&str> = current_docs.iter().map(|d| d.id.as_str()).collect();
        let existing_paths: HashSet<&str> = current_docs.iter().map(|d| d.path.as_str()).collect();
        // The inbox names registered documents go by: a converted one's path
        // moves to its converted copy, so its original is known by name.
        let mut registered_names: HashSet<String> = current_docs
            .iter()
            .filter_map(|d| d.name.clone().filter(|n| !n.is_empty()))
            .collect();
        registered_names.extend(manifest_entries.values().filter_map(|e| e.get("filename")?.as_str().map(str::to_string)));

        let mut new_docs: Vec<Document> = Vec::new();
        for (filename, file_path) in &disk_files {
            let path_str = file_path.to_string_lossy().into_owned();
            if existing_paths.contains(path_str.as_str()) {
                continue;
            }
            match std::fs::read(file_path) {
                Ok(bytes) => {
                    let doc_hash = hex::encode(Md5::digest(&bytes).as_slice());
                    if existing_hashes.contains(doc_hash.as_str()) {
                        // A second copy of a document already here (an upload
                        // stored as `name_1` beside it) never registers: left
                        // in, it would list as pending for good.
                        if !registered_names.contains(filename) {
                            match std::fs::remove_file(file_path) {
                                Ok(()) => self.core.logger.info(&format!(
                                    "[Indexer] {filename} duplicates a document already in the project — removed"
                                )),
                                Err(exc) => self
                                    .core
                                    .logger
                                    .warning(&format!("[Indexer] Could not remove duplicate {filename}: {exc}")),
                            }
                        }
                        continue;
                    }
                    new_docs.push(Document {
                        id: doc_hash.clone(),
                        path: path_str.clone(),
                        name: Some(filename.clone()),
                        summary: None,
                        status: DocumentStatus::Pending,
                    });
                    manifest.upsert(
                        &doc_hash,
                        filename,
                        &path_str,
                        DocumentStatus::Pending.as_str(),
                        &doc_hash,
                    );
                }
                Err(exc) => {
                    self.core
                        .logger
                        .warning(&format!("[Indexer] Could not register {filename}: {exc}"));
                }
            }
        }

        if !new_docs.is_empty() {
            {
                let _guard = self.workspace_lock.lock();
                let workspace = self.workspace();
                let mut ws = workspace.write();
                if let Some(p) = ws.project_mut(&project.id) {
                    p.documents.extend(new_docs.iter().cloned());
                }
                ws.save()?;
            }
            self.core
                .logger
                .info(&format!("[Indexer] Picked up {} new file(s)", new_docs.len()));
            self.enqueue(&new_docs);
        }
        Ok(())
    }

    /// The documents the workspace currently holds for `project` (falls back
    /// to the snapshot when the workspace no longer has it).
    fn project_documents(&self, project: &Project) -> Vec<Document> {
        let workspace = self.workspace();
        let ws = workspace.read();
        ws.project(&project.id)
            .map(|p| p.documents.clone())
            .unwrap_or_else(|| project.documents.clone())
    }

    /// Remove a document from the vector store and workspace config.
    fn remove_document(&self, doc_id: &str, project: &Project) -> anyhow::Result<()> {
        let workspace_dir = self.workspace().read().workspace_dir.clone();
        let removed = StoreFactory::create_store(&project.project_dir, "documents", Some(&workspace_dir), None)
            .and_then(|store| store.delete_document(doc_id));
        if let Err(exc) = removed {
            self.core.logger.warning(&format!(
                "[Indexer] Could not remove chunks for {}: {exc}",
                prefix_chars(doc_id, 8)
            ));
        }

        let _guard = self.workspace_lock.lock();
        let workspace = self.workspace();
        let mut ws = workspace.write();
        if let Some(p) = ws.project_mut(&project.id) {
            p.documents.retain(|d| d.id != doc_id);
        }
        ws.save()?;
        Ok(())
    }

    // ---------------------------------------------------------------
    // Workspace (project working-directory) indexing
    // ---------------------------------------------------------------

    /// Ask the event loop to run a workspace index pass on its next tick.
    ///
    /// Resetting the timer makes the loop fire a scan within ~1s (the queue
    /// poll interval) instead of waiting out the full scan interval.  The
    /// scan still runs on the executor, so callers are never blocked.  Used on
    /// project switch and at the start of a chat request so freshly-edited
    /// files are picked up promptly.
    pub fn request_workspace_scan(&self) {
        *self.last_workspace_scan.lock() = 0.0;
    }

    /// Index the project's folder now and return once it is done, for a chat
    /// turn that waits on it.  No quiet-period deferral: the turn waiting is
    /// what would keep the model busy.  A background pass already running is
    /// waited out by the indexer's own lock, then this one finds little left.
    pub fn index_workspace_now(&self) -> anyhow::Result<()> {
        self.workspace_scan_inflight.store(true, Ordering::SeqCst);
        let outcome = self.scan_workspace(false);
        self.workspace_scan_inflight.store(false, Ordering::SeqCst);
        outcome
    }

    /// What the workspace pass is doing to a document right now, if anything.
    pub fn workspace_activity(&self) -> Option<String> {
        self.workspace_indexer.lock().as_ref().and_then(|indexer| indexer.activity())
    }

    /// Executor entry point for the background workspace index pass.
    fn run_workspace_scan(&self) {
        let outcome = catch_unwind(AssertUnwindSafe(|| self.scan_workspace_files()));
        let failure = match outcome {
            Ok(Ok(())) => None,
            Ok(Err(exc)) => Some(exc.to_string()),
            Err(payload) => Some(panic_message(&*payload)),
        };
        if let Some(exc) = failure {
            self.core.logger.warning(&format!("[Indexer] Workspace scan failed: {exc}"));
        }
        self.workspace_scan_inflight.store(false, Ordering::SeqCst);
    }

    /// Incrementally index the project's working-directory files.
    ///
    /// Mirrors [`IndexerTask::scan_documents`] but targets the code/workspace
    /// files in the project's own directory, indexing them into the dedicated
    /// `workspace_files` collection.  Indexing here keeps it off the request
    /// path — `ModelService::search_workspace` only queries the store.  The
    /// underlying [`WorkspaceIndexer`] is hash-incremental, so repeat passes
    /// only re-embed changed files.
    pub fn scan_workspace_files(&self) -> anyhow::Result<()> {
        self.scan_workspace(true)
    }

    /// [`IndexerTask::scan_workspace_files`], deferring to a busy model only
    /// when `defer`.
    fn scan_workspace(&self, defer: bool) -> anyhow::Result<()> {
        let Some(project) = self.project() else {
            return Ok(());
        };

        // The folder the project is bound to, when it still is one this
        // process may work in (see `chat::request_cwd`); otherwise the
        // project's own directory, as for a project with no folder.  The
        // index itself always lives under project_dir.
        let work_dir = project
            .work_dir()
            .and_then(|dir| crate::paths::bindable_dir(&dir))
            .unwrap_or_else(|| project.project_dir.clone());
        if work_dir.is_empty() || !Path::new(&work_dir).is_dir() {
            return Ok(());
        }

        // Wait for the model to be quiet before spending the GPU on chunks.
        //
        // Embedding and generation share one device, and a pass over a few
        // hundred chunks drops a streaming reply from ~57 tok/s to ~3 while it
        // runs — the user watches their answer arrive a word at a time so the
        // workspace index can be a minute fresher.  Nothing here is urgent
        // enough to win that trade.
        //
        // A *quiet period* rather than "is it generating right now": one turn
        // is many sequential calls — classify, skill match, tool resolution,
        // judge, then the answer — and a bare busy check would start a pass in
        // the gap between two of them and run straight through the reply.
        // Deferring, not queueing: the loop comes back every
        // `WORKSPACE_SCAN_INTERVAL` and the pass is hash-incremental, so a
        // skipped turn costs nothing but freshness.
        let provider = self.idle_seconds_provider.read().clone();
        let idle = match provider {
            Some(provider) => provider(),
            None => device_idle_seconds(),
        };
        if defer && idle < Self::WORKSPACE_QUIET_SECONDS {
            self.core.logger.debug(&format!(
                "[Indexer] Workspace scan deferred — model busy (idle {idle:.1}s < {:.1}s)",
                Self::WORKSPACE_QUIET_SECONDS
            ));
            // Re-arm for one quiet period rather than a full interval: the
            // dispatcher stamped the clock when it handed this pass over, and
            // leaving it there would turn every deferral into a 30s wait, so a
            // busy stretch would starve the index instead of postponing it.
            *self.last_workspace_scan.lock() =
                epoch_seconds() - Self::WORKSPACE_SCAN_INTERVAL + Self::WORKSPACE_QUIET_SECONDS;
            return Ok(());
        }

        // Recreate the indexer when the active project changes so the store
        // and manifest always point at the current project's directory.
        let indexer = {
            let mut slot = self.workspace_indexer.lock();
            let mut dir = self.workspace_indexer_dir.lock();
            match slot.as_ref() {
                Some(indexer) if dir.as_deref() == Some(project.project_dir.as_str()) => indexer.clone(),
                _ => {
                    let workspace_dir = self.workspace().read().workspace_dir.clone();
                    let converter: DocumentConverter = Arc::new(convert_document);
                    let indexer: Arc<dyn WorkspaceIndex> = Arc::new(
                        WorkspaceIndexer::new(&project.project_dir, Some(&workspace_dir))?.with_converter(converter),
                    );
                    *slot = Some(indexer.clone());
                    *dir = Some(project.project_dir.clone());
                    indexer
                }
            }
        };

        let stats = indexer.index(&work_dir);
        *self.last_workspace_stats.lock() = Some(stats);
        *self.last_workspace_scan_at.lock() = Some(epoch_seconds());
        if stats.indexed > 0 || stats.removed > 0 {
            self.core.logger.info(&format!(
                "[Indexer] Workspace index — indexed={}, skipped={}, removed={}",
                stats.indexed, stats.skipped, stats.removed
            ));
        }
        Ok(())
    }

    /// Snapshot of background workspace-file indexing for `/health`:
    /// `{"scanning", "files_indexed", "last_scan_at", "last_stats",
    /// "scan_interval_seconds"}`.
    ///
    /// Reports whether a scan is in flight, how many files are currently
    /// indexed, when the last pass completed (epoch seconds), the last pass's
    /// stats, and the scan cadence.
    pub fn workspace_status(&self) -> Value {
        let indexer = self.workspace_indexer.lock().clone();
        let stats = *self.last_workspace_stats.lock();
        json!({
            "scanning": self.workspace_scan_inflight.load(Ordering::SeqCst),
            "files_indexed": indexer.map(|i| i.file_count()).unwrap_or(0),
            "last_scan_at": *self.last_workspace_scan_at.lock(),
            "last_stats": stats.map(|s| json!({
                "indexed": s.indexed,
                "skipped": s.skipped,
                "removed": s.removed,
            })),
            "scan_interval_seconds": Self::WORKSPACE_SCAN_INTERVAL,
        })
    }

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    fn get_manifest(&self, project: &Project) -> DocumentsManifest {
        DocumentsManifest::new(Path::new(&project.project_dir).join("documents.jsonl"))
    }

    /// The project this pass indexes, refreshing the workspace first.
    ///
    /// Scoped to `project_id` when the task owns one; otherwise it follows the
    /// selected project, which is what a single shared process has always
    /// done.
    fn project(&self) -> Option<Project> {
        let provider = self.project_provider.read().clone();
        if let Some(provider) = provider {
            return provider();
        }
        self.refresh_workspace();
        let workspace = self.workspace();
        let ws = workspace.read();
        match self.project_id.as_deref() {
            Some(project_id) => ws.projects.iter().find(|p| p.id == project_id).cloned(),
            None => ws.current_project().cloned(),
        }
    }

    fn refresh_workspace(&self) {
        if let Some(current) = Workspace::get_instance() {
            let changed = !Arc::ptr_eq(&current, &*self.workspace.read());
            if changed {
                self.core
                    .logger
                    .info("[Indexer] Workspace singleton changed — refreshing reference");
                *self.workspace.write() = current;
            }
        }
    }

    fn drain_queue(&self) -> HashSet<String> {
        let mut ids = HashSet::new();
        while let Some(id) = self.queue.get_nowait() {
            ids.insert(id);
        }
        ids
    }

    fn ensure_model_loaded(&self) -> anyhow::Result<Arc<dyn BaseLLM>> {
        let model_service = get_model_service();
        match model_service.get_llm() {
            Some(llm) => Ok(llm),
            None => model_service.load_model(),
        }
    }

    /// Free on-demand models when indexing goes idle (orchestrator stays).
    fn release_on_demand_models(&self) {
        let outcome = catch_unwind(AssertUnwindSafe(|| get_model_service().unload_vision_model()));
        if let Err(payload) = outcome {
            self.core.logger.warning(&format!(
                "[Indexer] Failed to release vision model: {}",
                panic_message(&*payload)
            ));
        }
    }
}

/// Parse a document in the project's folder to text for the workspace
/// index, with the parsers uploads go through.
fn convert_document(path: &Path) -> anyhow::Result<String> {
    let model_service = get_model_service();
    let llm = match model_service.get_llm() {
        Some(llm) => llm,
        None => model_service.load_model()?,
    };
    let shared: SharedLLM = llm;
    let path = path.to_string_lossy();
    ParserFactory::new(Some(shared)).get_parser(&guess_mime_type(Path::new(path.as_ref())), &path)?.parse()
}

/// Whether `exc` is the Rust form of a Python `ValueError` from parsing.
fn is_value_error(exc: &anyhow::Error) -> bool {
    exc.chain()
        .any(|e| matches!(e.downcast_ref::<ParserError>(), Some(ParserError::Value(_))))
}

impl BaseTask for IndexerTask {
    fn core(&self) -> &TaskCore {
        &self.core
    }

    /// Batch fallback for `TaskQueue::submit()` compatibility.
    fn execute(&self, _kwargs: &TaskKwargs) -> anyhow::Result<TaskResult> {
        let Some(project) = self.project() else {
            return Ok(TaskResult::failure("No active project"));
        };

        let queued_ids = self.drain_queue();
        if queued_ids.is_empty() {
            return Ok(TaskResult::ok(json!({ "processed": 0 })));
        }

        let doc_by_id: HashMap<&str, &Document> =
            project.documents.iter().map(|d| (d.id.as_str(), d)).collect();
        let mut processed = 0;
        for doc_id in &queued_ids {
            if self.core.stop_event.is_set() {
                break;
            }
            let Some(doc) = doc_by_id.get(doc_id.as_str()) else {
                continue;
            };
            if matches!(
                doc.status,
                DocumentStatus::Pending | DocumentStatus::Converting | DocumentStatus::Indexing
            ) {
                self.process_single_document(doc, &project);
                if self.document_status(&project.id, &doc.id) == Some(DocumentStatus::Indexed) {
                    processed += 1;
                }
            }
        }

        Ok(TaskResult::ok(json!({ "processed": processed })))
    }

    fn should_run(&self) -> bool {
        !self.queue.is_empty()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn into_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn queue_size(&self) -> Option<usize> {
        Some(IndexerTask::queue_size(self))
    }
    fn workspace_status(&self) -> Option<Value> {
        Some(IndexerTask::workspace_status(self))
    }
}
