//! Port of `smartloop/utils/logging_setup.py`.
//!
//! Centralized logging setup for the API server.
//!
//! All threads (task queue jobs, indexer workers, DAG executor, hybrid
//! executor I/O pool) emit through a single queue handler on the root logger.
//! A [`QueueListener`] thread owns the real handlers (rotating file +
//! optional console) and writes records in arrival order so concurrent
//! producers never tear a line.
//!
//! The queue is a plain in-process queue.  It used to be a
//! `multiprocessing.Queue` in the Python app so that spawn workers could write
//! to it from another process, at a cost of three POSIX named semaphores per
//! startup — semaphores that had to be unlinked before the process died, and
//! that `sem_open` refuses outright under the macOS App Sandbox.  That worker
//! pool is gone, so every producer is a thread in this process and the queue
//! owns no OS resources.
//!
//! Rust code logs with `tracing` macros (`target: "smartloop.x.y"`, the Python
//! logger name).  [`install_tracing_bridge`] installs a global `tracing`
//! subscriber that turns every event into a [`LogRecord`] (its target, with
//! `::` mapped to `.`, as the logger name), filters it through the logger
//! hierarchy and hands it to the root handlers.  This module also hosts a
//! small stand-in for Python's logger hierarchy ([`get_logger`] /
//! [`Logger::set_level`]) so per-logger levels such as the `smartloop`
//! WARNING gag keep working.
//!
//! Used by `crate::api::main::run_server` / `start_server_foreground`, which
//! install the queue + listener.

use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::fmt::Write as _;
use std::io::{IsTerminal, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Once};
use std::thread::JoinHandle;
use std::time::Duration;

use once_cell::sync::Lazy;
use parking_lot::{Condvar, Mutex, MutexGuard, RwLock};

/// Python logging numeric levels.
pub const NOTSET: i32 = 0;
pub const DEBUG: i32 = 10;
pub const INFO: i32 = 20;
pub const WARNING: i32 = 30;
pub const ERROR: i32 = 40;
pub const CRITICAL: i32 = 50;

/// Level assigned to `tracing::Level::TRACE` events (below DEBUG, so they are
/// dropped unless a logger is explicitly opened up to it).
pub const TRACE: i32 = 5;

pub const DEFAULT_FORMAT: &str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s";
/// 10 MB.
pub const DEFAULT_MAX_BYTES: u64 = 10 * 1024 * 1024;
pub const DEFAULT_BACKUP_COUNT: u32 = 5;

/// A level given either as a number or a (case-insensitive) name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LevelSpec {
    Num(i32),
    Name(String),
}

impl From<i32> for LevelSpec {
    fn from(v: i32) -> Self {
        LevelSpec::Num(v)
    }
}

impl From<&str> for LevelSpec {
    fn from(v: &str) -> Self {
        LevelSpec::Name(v.to_string())
    }
}

impl From<String> for LevelSpec {
    fn from(v: String) -> Self {
        LevelSpec::Name(v)
    }
}

impl From<&String> for LevelSpec {
    fn from(v: &String) -> Self {
        LevelSpec::Name(v.clone())
    }
}

/// Python `logging.getLevelName(name)` for a level name (`None` if unknown).
fn level_from_name(name: &str) -> Option<i32> {
    match name {
        "CRITICAL" | "FATAL" => Some(CRITICAL),
        "ERROR" => Some(ERROR),
        "WARNING" | "WARN" => Some(WARNING),
        "INFO" => Some(INFO),
        "DEBUG" => Some(DEBUG),
        "NOTSET" => Some(NOTSET),
        _ => None,
    }
}

/// Resolve a level spec to its numeric value (unknown names -> INFO).
pub fn resolve_level(level: impl Into<LevelSpec>) -> i32 {
    match level.into() {
        LevelSpec::Num(n) => n,
        LevelSpec::Name(name) => level_from_name(&name.trim().to_uppercase()).unwrap_or(INFO),
    }
}

/// Python `logging.getLevelName` for a numeric level.
pub fn level_name(level: i32) -> String {
    match level {
        CRITICAL => "CRITICAL".to_string(),
        ERROR => "ERROR".to_string(),
        WARNING => "WARNING".to_string(),
        INFO => "INFO".to_string(),
        DEBUG => "DEBUG".to_string(),
        NOTSET => "NOTSET".to_string(),
        other => format!("Level {other}"),
    }
}

/// One log event as carried through the queue (Python `LogRecord`).
#[derive(Debug, Clone, PartialEq)]
pub struct LogRecord {
    /// Logger name (the tracing target).
    pub name: String,
    pub levelno: i32,
    pub levelname: String,
    pub message: String,
    pub created: chrono::DateTime<chrono::Local>,
    pub thread_name: Option<String>,
}

impl LogRecord {
    /// Build a record stamped with the current time and thread.
    pub fn new(name: &str, levelno: i32, message: impl Into<String>) -> Self {
        let name = if name.is_empty() { "root".to_string() } else { name.to_string() };
        Self {
            name,
            levelno,
            levelname: level_name(levelno),
            message: message.into(),
            created: chrono::Local::now(),
            thread_name: Some(current_thread_name()),
        }
    }
}

/// Python `threading.current_thread().name`, approximated: the Rust thread
/// name, `MainThread` for the main thread, else the thread id.
fn current_thread_name() -> String {
    let current = std::thread::current();
    match current.name() {
        Some("main") => "MainThread".to_string(),
        Some(name) => name.to_string(),
        None => format!("{:?}", current.id()),
    }
}

/// `logging.Formatter` with `%(name)s`-style placeholders
/// (`asctime`, `name`, `levelname`, `levelno`, `message`, `thread`/`threadName`,
/// `process`).
#[derive(Debug, Clone, PartialEq)]
pub struct Formatter {
    pub fmt: String,
}

impl Default for Formatter {
    fn default() -> Self {
        Self { fmt: DEFAULT_FORMAT.to_string() }
    }
}

impl Formatter {
    pub fn new(fmt: &str) -> Self {
        Self { fmt: fmt.to_string() }
    }

    fn field(record: &LogRecord, key: &str) -> Option<String> {
        Some(match key {
            "asctime" => record.created.format("%Y-%m-%d %H:%M:%S,%3f").to_string(),
            "name" => record.name.clone(),
            "levelname" => record.levelname.clone(),
            "levelno" => record.levelno.to_string(),
            "message" => record.message.clone(),
            "threadName" | "thread" => record.thread_name.clone().unwrap_or_default(),
            "process" => std::process::id().to_string(),
            "created" => {
                format!("{}", record.created.timestamp_millis() as f64 / 1000.0)
            }
            "msecs" => record.created.format("%3f").to_string(),
            _ => return None,
        })
    }

    /// Render a record (asctime as `YYYY-mm-dd HH:MM:SS,mmm`).
    pub fn format(&self, record: &LogRecord) -> String {
        let chars: Vec<char> = self.fmt.chars().collect();
        let mut out = String::with_capacity(self.fmt.len() + record.message.len() + 32);
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c == '%' && chars.get(i + 1) == Some(&'%') {
                out.push('%');
                i += 2;
                continue;
            }
            if c == '%' && chars.get(i + 1) == Some(&'(') {
                if let Some(close) = (i + 2..chars.len()).find(|&j| chars[j] == ')') {
                    let key: String = chars[i + 2..close].iter().collect();
                    // Conversion spec: optional flags/width/precision, then a type char.
                    let mut j = close + 1;
                    let mut left = false;
                    while j < chars.len() && matches!(chars[j], '-' | '+' | ' ' | '#' | '0') {
                        if chars[j] == '-' {
                            left = true;
                        }
                        j += 1;
                    }
                    let width_start = j;
                    while j < chars.len() && chars[j].is_ascii_digit() {
                        j += 1;
                    }
                    let width: usize = chars[width_start..j].iter().collect::<String>().parse().unwrap_or(0);
                    if j < chars.len() && chars[j] == '.' {
                        j += 1;
                        while j < chars.len() && chars[j].is_ascii_digit() {
                            j += 1;
                        }
                    }
                    if j < chars.len() && chars[j].is_ascii_alphabetic() {
                        if let Some(value) = Self::field(record, &key) {
                            let len = value.chars().count();
                            let pad = " ".repeat(width.saturating_sub(len));
                            if left {
                                out.push_str(&value);
                                out.push_str(&pad);
                            } else {
                                out.push_str(&pad);
                                out.push_str(&value);
                            }
                            i = j + 1;
                            continue;
                        }
                    }
                }
            }
            out.push(c);
            i += 1;
        }
        out
    }
}

/// Process-local unbounded FIFO of log records (Python `queue.Queue(-1)`).
/// Cheap to clone; clones share the same queue.
#[derive(Clone, Default)]
pub struct LogQueue {
    inner: Arc<LogQueueInner>,
}

#[derive(Default)]
struct LogQueueInner {
    items: Mutex<VecDeque<LogRecord>>,
    cond: Condvar,
}

impl std::fmt::Debug for LogQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogQueue").field("len", &self.len()).finish_non_exhaustive()
    }
}

impl LogQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enqueue a record.
    pub fn put(&self, record: LogRecord) {
        self.inner.items.lock().push_back(record);
        self.inner.cond.notify_one();
    }

    /// Dequeue a record, waiting up to `timeout`.
    pub fn get_timeout(&self, timeout: Duration) -> Option<LogRecord> {
        let mut items = self.inner.items.lock();
        if let Some(record) = items.pop_front() {
            return Some(record);
        }
        let deadline = std::time::Instant::now() + timeout;
        while items.is_empty() {
            if self.inner.cond.wait_until(&mut items, deadline).timed_out() {
                break;
            }
        }
        items.pop_front()
    }

    pub fn empty(&self) -> bool {
        self.inner.items.lock().is_empty()
    }

    pub fn len(&self) -> usize {
        self.inner.items.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.empty()
    }

    /// Identity comparison (Python `is`).
    pub fn ptr_eq(&self, other: &LogQueue) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

/// A sink owned by a [`QueueListener`] or attached to the root logger
/// (Python `logging.Handler`).
pub trait Handler: Send + Sync {
    /// Format and write the record (no level check; callers filter).
    fn handle(&self, record: &LogRecord);
    /// Handler threshold level.
    fn level(&self) -> i32;
    fn set_level(&self, level: i32);
    fn set_formatter(&self, formatter: Formatter);
    fn flush(&self);
    fn close(&self);
    /// For downcasting to a concrete handler (e.g. [`RotatingFileHandler`]).
    fn as_any(&self) -> &dyn Any;
}

/// An open file stream owned by a [`RotatingFileHandler`].  Its identity
/// changes whenever the handler reopens the file.
pub struct LogStream {
    file: Mutex<Option<std::fs::File>>,
    closed: AtomicBool,
}

impl LogStream {
    /// Open `path` in append mode.
    pub fn open(path: &Path) -> std::io::Result<Arc<LogStream>> {
        let file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Arc::new(LogStream { file: Mutex::new(Some(file)), closed: AtomicBool::new(false) }))
    }

    pub fn write_str(&self, s: &str) -> std::io::Result<()> {
        match self.file.lock().as_mut() {
            Some(f) => f.write_all(s.as_bytes()),
            None => Err(std::io::Error::other("I/O operation on closed file.")),
        }
    }

    pub fn flush(&self) -> std::io::Result<()> {
        match self.file.lock().as_mut() {
            Some(f) => f.flush(),
            None => Ok(()),
        }
    }

    /// Current end-of-file position (Python `stream.seek(0, 2)`).
    fn end_position(&self) -> std::io::Result<u64> {
        match self.file.lock().as_mut() {
            Some(f) => f.seek(SeekFrom::End(0)),
            None => Err(std::io::Error::other("I/O operation on closed file.")),
        }
    }

    pub fn close(&self) {
        let mut file = self.file.lock();
        if let Some(mut f) = file.take() {
            let _ = f.flush();
        }
        self.closed.store(true, Ordering::SeqCst);
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }
}

/// Python `logging.handlers.RotatingFileHandler` (UTF-8, append mode,
/// rotates to `.1`..`.N` when `max_bytes` would be exceeded).
pub struct RotatingFileHandler {
    pub base_filename: PathBuf,
    pub max_bytes: u64,
    pub backup_count: u32,
    level: Mutex<i32>,
    formatter: Mutex<Formatter>,
    /// Handler lock (Python `acquire`/`release`) guarding the stream.
    stream: Mutex<Option<Arc<LogStream>>>,
}

/// Python `os.path.abspath`: absolute without resolving symlinks.
fn abspath(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().map(|cwd| cwd.join(path)).unwrap_or_else(|_| path.to_path_buf())
    }
}

/// Python `os.path.realpath`: resolve symlinks, tolerating a missing tail.
fn realpath(path: &Path) -> PathBuf {
    let abs = abspath(path);
    if let Ok(resolved) = abs.canonicalize() {
        return resolved;
    }
    // Resolve the deepest existing ancestor and re-append the rest.
    let mut tail = Vec::new();
    let mut cur = abs.as_path();
    while let Some(parent) = cur.parent() {
        if let Some(name) = cur.file_name() {
            tail.push(name.to_os_string());
        }
        if let Ok(resolved) = parent.canonicalize() {
            let mut out = resolved;
            for name in tail.iter().rev() {
                out.push(name);
            }
            return out;
        }
        cur = parent;
    }
    abs
}

impl RotatingFileHandler {
    pub fn new(path: &Path, max_bytes: u64, backup_count: u32) -> std::io::Result<Self> {
        let base_filename = abspath(path);
        let stream = LogStream::open(&base_filename)?;
        Ok(Self {
            base_filename,
            max_bytes,
            backup_count,
            level: Mutex::new(NOTSET),
            formatter: Mutex::new(Formatter::default()),
            stream: Mutex::new(Some(stream)),
        })
    }

    /// The currently open stream, if any.
    pub fn stream(&self) -> Option<Arc<LogStream>> {
        self.stream.lock().clone()
    }

    /// Acquire the handler lock (Python `acquire`); the guard gives mutable
    /// access to the stream slot and releases on drop.
    pub fn lock_stream(&self) -> MutexGuard<'_, Option<Arc<LogStream>>> {
        self.stream.lock()
    }

    /// Run `f` with the handler lock held and mutable access to the stream
    /// slot (used by [`truncate_log_file`] to close/reopen).
    pub fn with_stream_locked<R>(&self, f: impl FnOnce(&mut Option<Arc<LogStream>>) -> R) -> R {
        let mut guard = self.stream.lock();
        f(&mut guard)
    }

    /// Open a fresh stream on `base_filename` (Python `_open`).
    pub fn open_stream(&self) -> std::io::Result<Arc<LogStream>> {
        LogStream::open(&self.base_filename)
    }

    /// Whether writing `msg` would push the file past `max_bytes`.  Called
    /// with the handler lock held (`slot` is the locked stream).
    fn should_rollover(&self, slot: &mut Option<Arc<LogStream>>, msg: &str) -> bool {
        // Non-regular files (e.g. /dev/null) are never rotated.
        if self.base_filename.exists() && !self.base_filename.is_file() {
            return false;
        }
        if slot.is_none() {
            match self.open_stream() {
                Ok(stream) => *slot = Some(stream),
                Err(_) => return false,
            }
        }
        if self.max_bytes > 0 {
            if let Some(stream) = slot.as_ref() {
                if let Ok(pos) = stream.end_position() {
                    return pos + msg.len() as u64 >= self.max_bytes;
                }
            }
        }
        false
    }

    /// Rotate `base` -> `base.1` -> ... -> `base.N` and reopen.  Called with
    /// the handler lock held.
    fn do_rollover(&self, slot: &mut Option<Arc<LogStream>>) {
        if let Some(stream) = slot.take() {
            stream.close();
        }
        if self.backup_count > 0 {
            let name = |i: u32| -> PathBuf {
                let mut s = self.base_filename.clone().into_os_string();
                s.push(format!(".{i}"));
                PathBuf::from(s)
            };
            for i in (1..self.backup_count).rev() {
                let sfn = name(i);
                let dfn = name(i + 1);
                if sfn.exists() {
                    if dfn.exists() {
                        let _ = std::fs::remove_file(&dfn);
                    }
                    let _ = std::fs::rename(&sfn, &dfn);
                }
            }
            let dfn = name(1);
            if dfn.exists() {
                let _ = std::fs::remove_file(&dfn);
            }
            if self.base_filename.exists() {
                let _ = std::fs::rename(&self.base_filename, &dfn);
            }
        }
        if let Ok(stream) = self.open_stream() {
            *slot = Some(stream);
        }
    }
}

impl Handler for RotatingFileHandler {
    fn handle(&self, record: &LogRecord) {
        let msg = format!("{}\n", self.formatter.lock().format(record));
        let mut slot = self.stream.lock();
        if self.should_rollover(&mut slot, &msg) {
            self.do_rollover(&mut slot);
        }
        if slot.is_none() {
            match self.open_stream() {
                Ok(stream) => *slot = Some(stream),
                Err(e) => {
                    eprintln!("--- Logging error ---\n{e}");
                    return;
                }
            }
        }
        if let Some(stream) = slot.as_ref() {
            let result = stream.write_str(&msg).and_then(|_| stream.flush());
            if let Err(e) = result {
                eprintln!("--- Logging error ---\n{e}");
            }
        }
    }
    fn level(&self) -> i32 {
        *self.level.lock()
    }
    fn set_level(&self, level: i32) {
        *self.level.lock() = level;
    }
    fn set_formatter(&self, formatter: Formatter) {
        *self.formatter.lock() = formatter;
    }
    fn flush(&self) {
        if let Some(stream) = self.stream.lock().as_ref() {
            let _ = stream.flush();
        }
    }
    fn close(&self) {
        if let Some(stream) = self.stream.lock().take() {
            stream.close();
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Python `logging.StreamHandler`: writes formatted lines to a writer
/// (stdout by default, or any `Write` such as a shared buffer in tests).
pub struct StreamHandler {
    writer: Mutex<Box<dyn Write + Send>>,
    level: Mutex<i32>,
    formatter: Mutex<Formatter>,
}

impl StreamHandler {
    pub fn new(writer: Box<dyn Write + Send>) -> Self {
        Self {
            writer: Mutex::new(writer),
            level: Mutex::new(NOTSET),
            formatter: Mutex::new(Formatter::default()),
        }
    }

    pub fn stdout() -> Self {
        Self::new(Box::new(std::io::stdout()))
    }

    pub fn stderr() -> Self {
        Self::new(Box::new(std::io::stderr()))
    }
}

impl Handler for StreamHandler {
    fn handle(&self, record: &LogRecord) {
        let msg = format!("{}\n", self.formatter.lock().format(record));
        let mut writer = self.writer.lock();
        let _ = writer.write_all(msg.as_bytes());
        let _ = writer.flush();
    }
    fn level(&self) -> i32 {
        *self.level.lock()
    }
    fn set_level(&self, level: i32) {
        *self.level.lock() = level;
    }
    fn set_formatter(&self, formatter: Formatter) {
        *self.formatter.lock() = formatter;
    }
    fn flush(&self) {
        let _ = self.writer.lock().flush();
    }
    fn close(&self) {
        // Python's StreamHandler.close() leaves the underlying stream open.
        self.flush();
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A handler that keeps every record in memory — the stand-in for pytest's
/// `caplog` when a test needs to assert on what was logged.
#[derive(Default)]
pub struct CaptureHandler {
    records: Mutex<Vec<LogRecord>>,
    level: Mutex<i32>,
    formatter: Mutex<Formatter>,
}

impl CaptureHandler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every captured record, in arrival order.
    pub fn records(&self) -> Vec<LogRecord> {
        self.records.lock().clone()
    }

    /// The captured messages (`caplog.messages`).
    pub fn messages(&self) -> Vec<String> {
        self.records.lock().iter().map(|r| r.message.clone()).collect()
    }

    /// The captured records formatted and joined by newlines (`caplog.text`).
    pub fn text(&self) -> String {
        let formatter = self.formatter.lock().clone();
        let mut out = String::new();
        for r in self.records.lock().iter() {
            let _ = writeln!(out, "{}", formatter.format(r));
        }
        out
    }

    pub fn clear(&self) {
        self.records.lock().clear();
    }
}

impl Handler for CaptureHandler {
    fn handle(&self, record: &LogRecord) {
        self.records.lock().push(record.clone());
    }
    fn level(&self) -> i32 {
        *self.level.lock()
    }
    fn set_level(&self, level: i32) {
        *self.level.lock() = level;
    }
    fn set_formatter(&self, formatter: Formatter) {
        *self.formatter.lock() = formatter;
    }
    fn flush(&self) {}
    fn close(&self) {}
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Python `logging.handlers.QueueListener`: a background thread draining a
/// [`LogQueue`] into its handlers.
pub struct QueueListener {
    queue: Mutex<Option<LogQueue>>,
    handlers: RwLock<Vec<Arc<dyn Handler>>>,
    pub respect_handler_level: bool,
    thread: Mutex<Option<JoinHandle<()>>>,
    stop_flag: Arc<AtomicBool>,
}

impl QueueListener {
    pub fn new(
        queue: LogQueue,
        handlers: Vec<Arc<dyn Handler>>,
        respect_handler_level: bool,
    ) -> Arc<Self> {
        Arc::new(Self {
            queue: Mutex::new(Some(queue)),
            handlers: RwLock::new(handlers),
            respect_handler_level,
            thread: Mutex::new(None),
            stop_flag: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Dispatch one record to the handlers (Python `QueueListener.handle`).
    fn handle(&self, record: &LogRecord) {
        let handlers = self.handlers.read().clone();
        for handler in handlers {
            let process = !self.respect_handler_level || record.levelno >= handler.level();
            if process {
                handler.handle(record);
            }
        }
    }

    /// Start the draining thread.
    pub fn start(self: &Arc<Self>) {
        let mut thread = self.thread.lock();
        if thread.is_some() {
            return;
        }
        let Some(queue) = self.queue.lock().clone() else {
            return;
        };
        self.stop_flag.store(false, Ordering::SeqCst);
        let stop_flag = Arc::clone(&self.stop_flag);
        // The thread holds a weak reference so a dropped listener does not
        // keep itself alive; records still drain through the handlers.
        let me = Arc::downgrade(self);
        let handle = std::thread::Builder::new()
            .name("log-queue-listener".into())
            .spawn(move || {
                loop {
                    match queue.get_timeout(Duration::from_millis(50)) {
                        Some(record) => match me.upgrade() {
                            Some(listener) => listener.handle(&record),
                            None => break,
                        },
                        None => {
                            if stop_flag.load(Ordering::SeqCst) {
                                break;
                            }
                        }
                    }
                }
            })
            .expect("spawn log listener thread");
        *thread = Some(handle);
    }

    /// Drain remaining records, stop and join the thread.  Idempotent.
    pub fn stop(&self) {
        let handle = self.thread.lock().take();
        if let Some(handle) = handle {
            self.stop_flag.store(true, Ordering::SeqCst);
            if handle.thread().id() != std::thread::current().id() {
                let _ = handle.join();
            }
        }
    }

    /// The bound queue (`None` once detached by [`shutdown_logging`]).
    pub fn queue(&self) -> Option<LogQueue> {
        self.queue.lock().clone()
    }

    pub fn set_queue(&self, queue: Option<LogQueue>) {
        *self.queue.lock() = queue;
    }

    pub fn handlers(&self) -> Vec<Arc<dyn Handler>> {
        self.handlers.read().clone()
    }

    pub fn set_handlers(&self, handlers: Vec<Arc<dyn Handler>>) {
        *self.handlers.write() = handlers;
    }

    pub fn add_handler(&self, handler: Arc<dyn Handler>) {
        self.handlers.write().push(handler);
    }
}

impl Drop for QueueListener {
    fn drop(&mut self) {
        if let Some(handle) = self.thread.get_mut().take() {
            self.stop_flag.store(true, Ordering::SeqCst);
            if handle.thread().id() != std::thread::current().id() {
                let _ = handle.join();
            }
        }
    }
}

/// A handler attached to the root logger.
#[derive(Clone)]
pub enum RootHandler {
    /// Python `QueueHandler`: enqueue the record for a listener.
    Queue(LogQueue),
    /// Any other handler, called inline in the emitting thread.
    Handler(Arc<dyn Handler>),
}

/// Process-wide singleton — set by [`make_log_queue`].
static LOG_QUEUE: Lazy<Mutex<Option<LogQueue>>> = Lazy::new(|| Mutex::new(None));

/// The listener that owns the real handlers — stashed by [`start_listener`]
/// so [`truncate_log_file`] can reach them.
static LISTENER: Lazy<Mutex<Option<Arc<QueueListener>>>> = Lazy::new(|| Mutex::new(None));

/// Root logger handlers (Python `logging.getLogger().handlers`).
static ROOT_HANDLERS: Lazy<RwLock<Vec<RootHandler>>> = Lazy::new(|| RwLock::new(Vec::new()));

/// Explicit per-logger levels (Python `Logger.setLevel`), keyed by dotted name;
/// the root logger is stored under `""`.
static LOGGER_LEVELS: Lazy<RwLock<HashMap<String, i32>>> = Lazy::new(|| RwLock::new(HashMap::new()));

/// Create the process-wide log queue. Idempotent — returns the existing
/// queue if one was already created.
pub fn make_log_queue() -> LogQueue {
    LOG_QUEUE.lock().get_or_insert_with(LogQueue::new).clone()
}

/// Return the shared log queue, or `None` if logging hasn't been set up yet
/// (e.g. running outside the server, in unit tests, or before startup).
pub fn get_log_queue() -> Option<LogQueue> {
    LOG_QUEUE.lock().clone()
}

/// Reset the singleton (test seam for the Python `_LOG_QUEUE = None` fixture).
pub fn reset_log_queue() {
    *LOG_QUEUE.lock() = None;
}

/// The listener registered by [`start_listener`], if still running
/// (Python `_LISTENER`).
pub fn current_listener() -> Option<Arc<QueueListener>> {
    LISTENER.lock().clone()
}

/// Number of queue handlers currently attached to the root logger.
pub fn root_queue_handler_count() -> usize {
    ROOT_HANDLERS.read().iter().filter(|h| matches!(h, RootHandler::Queue(_))).count()
}

/// Snapshot of the root logger's handlers.
pub fn root_handlers() -> Vec<RootHandler> {
    ROOT_HANDLERS.read().clone()
}

/// Replace the root logger's handlers wholesale (Python
/// `root.handlers[:] = handlers`).
pub fn set_root_handlers(handlers: Vec<RootHandler>) {
    *ROOT_HANDLERS.write() = handlers;
}

/// Attach a handler to the root logger (Python `root.addHandler`).  Also
/// installs the tracing bridge so `tracing` events reach it.
pub fn add_root_handler(handler: Arc<dyn Handler>) {
    install_tracing_bridge();
    ROOT_HANDLERS.write().push(RootHandler::Handler(handler));
}

/// Detach a handler previously attached with [`add_root_handler`].
pub fn remove_root_handler(handler: &Arc<dyn Handler>) {
    ROOT_HANDLERS.write().retain(|h| match h {
        RootHandler::Handler(existing) => !Arc::ptr_eq(existing, handler),
        RootHandler::Queue(_) => true,
    });
}

/// Python `logging.basicConfig(level=..., format=...)`: when the root logger
/// has no handlers yet, attach a stderr handler with `format` and set the
/// root level.  A no-op once root is configured.
pub fn basic_config(level: impl Into<LevelSpec>, format: &str) {
    let level = resolve_level(level);
    {
        let mut handlers = ROOT_HANDLERS.write();
        if !handlers.is_empty() {
            return;
        }
        let handler = StreamHandler::stderr();
        handler.set_formatter(Formatter::new(format));
        handlers.push(RootHandler::Handler(Arc::new(handler)));
    }
    get_logger("").set_level(level);
    install_tracing_bridge();
}

/// Hand a record to the root handlers (Python `Logger.callHandlers`); with
/// none attached, `logging.lastResort` puts WARNING+ messages on stderr.
pub fn dispatch_record(record: LogRecord) {
    let handlers = ROOT_HANDLERS.read().clone();
    if handlers.is_empty() {
        if record.levelno >= WARNING {
            eprintln!("{}", record.message);
        }
        return;
    }
    for handler in handlers {
        match handler {
            RootHandler::Queue(queue) => queue.put(record.clone()),
            RootHandler::Handler(h) => {
                if record.levelno >= h.level() {
                    h.handle(&record);
                }
            }
        }
    }
}

/// Map a `tracing` target to a Python logger name (`a::b` -> `a.b`).
fn logger_name_for_target(target: &str) -> String {
    target.replace("::", ".")
}

fn tracing_level(level: &tracing::Level) -> i32 {
    match *level {
        tracing::Level::ERROR => ERROR,
        tracing::Level::WARN => WARNING,
        tracing::Level::INFO => INFO,
        tracing::Level::DEBUG => DEBUG,
        tracing::Level::TRACE => TRACE,
    }
}

/// Collects an event's `message` plus any other fields (`key=value`).
#[derive(Default)]
struct MessageVisitor {
    message: String,
    extra: Vec<String>,
    /// The original target of a `log`-crate record forwarded by `tracing-log`.
    log_target: Option<String>,
}

impl tracing::field::Visit for MessageVisitor {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        match field.name() {
            "message" => self.message.push_str(value),
            "log.target" => self.log_target = Some(value.to_string()),
            name if name.starts_with("log.") => {}
            name => self.extra.push(format!("{name}={value}")),
        }
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.message, "{value:?}");
        } else if !field.name().starts_with("log.") {
            self.extra.push(format!("{}={:?}", field.name(), value));
        }
    }
}

/// `tracing` layer forwarding events into the root logger.
struct BridgeLayer;

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for BridgeLayer {
    fn register_callsite(&self, _metadata: &'static tracing::Metadata<'static>) -> tracing::subscriber::Interest {
        // Logger levels change at runtime, so decide per event.
        tracing::subscriber::Interest::sometimes()
    }

    fn enabled(&self, metadata: &tracing::Metadata<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) -> bool {
        // Spans are not logged; let them through so events inside still fire.
        if metadata.is_span() {
            return true;
        }
        // `log`-crate records forwarded by tracing-log are checked against
        // their real target here; their event metadata later says "log".
        let name = logger_name_for_target(metadata.target());
        get_logger(&name).is_enabled_for(tracing_level(metadata.level()))
    }

    fn on_event(&self, event: &tracing::Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        let metadata = event.metadata();
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        let target = visitor.log_target.as_deref().unwrap_or(metadata.target());
        let name = logger_name_for_target(target);
        let levelno = tracing_level(metadata.level());
        if !get_logger(&name).is_enabled_for(levelno) {
            return;
        }
        let mut message = visitor.message;
        if !visitor.extra.is_empty() {
            if !message.is_empty() {
                message.push(' ');
            }
            message.push_str(&visitor.extra.join(" "));
        }
        dispatch_record(LogRecord::new(&name, levelno, message));
    }
}

static BRIDGE_INSTALLED: AtomicBool = AtomicBool::new(false);
static BRIDGE_ONCE: Once = Once::new();

/// Install (once) the global `tracing` subscriber that bridges events into
/// the root logger, applying logger-level filtering.  `log`-crate records
/// from dependencies are forwarded too.  Returns `true` when the bridge is
/// the active global subscriber (another subscriber installed first wins).
pub fn install_tracing_bridge() -> bool {
    BRIDGE_ONCE.call_once(|| {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        // try_init also routes `log`-crate records into tracing; when a `log`
        // logger is already set it fails before installing anything, so fall
        // back to installing the subscriber alone.
        let installed = tracing_subscriber::registry().with(BridgeLayer).try_init().is_ok()
            || tracing::subscriber::set_global_default(tracing_subscriber::registry().with(BridgeLayer)).is_ok();
        BRIDGE_INSTALLED.store(installed, Ordering::SeqCst);
    });
    BRIDGE_INSTALLED.load(Ordering::SeqCst)
}

/// Replace root handlers with a single queue handler; set the root level
/// (DEBUG by default in Python callers).
///
/// Root level is set permissively so per-handler levels on the listener side
/// decide what actually gets emitted.
pub fn setup_root_logging(queue: &LogQueue, level: impl Into<LevelSpec>) {
    {
        let mut handlers = ROOT_HANDLERS.write();
        handlers.clear();
        handlers.push(RootHandler::Queue(queue.clone()));
    }
    get_logger("").set_level(resolve_level(level));
    install_tracing_bridge();
}

/// Options for [`start_listener`] (Python keyword arguments).
#[derive(Debug, Clone)]
pub struct ListenerOptions {
    pub home_dir: String,
    pub file_level: LevelSpec,
    pub console_level: LevelSpec,
    /// `None` = auto-detect (stdout is a TTY).
    pub force_console: Option<bool>,
    /// Overrides `<home_dir>/server.log` (e.g. a per-project `agent.log`).
    pub log_path: Option<String>,
}

impl ListenerOptions {
    pub fn new(home_dir: impl Into<String>) -> Self {
        Self {
            home_dir: home_dir.into(),
            file_level: LevelSpec::Name("INFO".into()),
            console_level: LevelSpec::Name("INFO".into()),
            force_console: None,
            log_path: None,
        }
    }

    pub fn file_level(mut self, level: impl Into<LevelSpec>) -> Self {
        self.file_level = level.into();
        self
    }

    pub fn console_level(mut self, level: impl Into<LevelSpec>) -> Self {
        self.console_level = level.into();
        self
    }

    pub fn force_console(mut self, force: Option<bool>) -> Self {
        self.force_console = force;
        self
    }

    pub fn log_path(mut self, path: Option<String>) -> Self {
        self.log_path = path;
        self
    }
}

/// True for the binary the release pipeline ships (`make build-binary` sets
/// `SLP_PACKAGED_BUILD=1` at compile time) — Python's `sys.frozen`.  Not tied
/// to the cargo profile: `make run`/`run-debug` build with `--release` too,
/// and must still log to the terminal like a from-source Python run.
pub fn is_packaged_build() -> bool {
    matches!(option_env!("SLP_PACKAGED_BUILD"), Some(v) if !v.is_empty() && v != "0")
}

/// Start the listener that owns the real handlers.
///
/// Writes to a rotating file handler at `<home_dir>/server.log`, or at
/// `log_path` when one is given.  A per-project agent passes its own
/// `<project_dir>/agent.log`: with a process per project, one shared log
/// would interleave every project's orchestration trace into a file where
/// working out which lines belong to the project you are debugging is the
/// hard part.  Separate files make that question disappear.
///
/// Adds a stdout handler when running interactively (TTY and not a frozen
/// build), unless `force_console` overrides the auto-detection.  "Frozen" is
/// the shipped binary ([`is_packaged_build`]), the counterpart of the Python
/// app's PyInstaller build; anything run from the checkout — `make run-debug`
/// included, which builds with `--release` — is "running from source".
pub fn start_listener(queue: &LogQueue, opts: ListenerOptions) -> std::io::Result<Arc<QueueListener>> {
    let log_path = match &opts.log_path {
        Some(p) if !p.is_empty() => PathBuf::from(p),
        _ => Path::new(&opts.home_dir).join("server.log"),
    };
    if let Some(parent) = log_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let formatter = Formatter::new(DEFAULT_FORMAT);

    let file_handler = RotatingFileHandler::new(&log_path, DEFAULT_MAX_BYTES, DEFAULT_BACKUP_COUNT)?;
    file_handler.set_level(resolve_level(opts.file_level.clone()));
    file_handler.set_formatter(formatter.clone());

    let mut handlers: Vec<Arc<dyn Handler>> = vec![Arc::new(file_handler)];

    let frozen = is_packaged_build();
    let console = match opts.force_console {
        None => std::io::stdout().is_terminal() && !frozen,
        Some(force) => force,
    };

    if console {
        let stream_handler = StreamHandler::stdout();
        stream_handler.set_level(resolve_level(opts.console_level.clone()));
        stream_handler.set_formatter(formatter);
        handlers.push(Arc::new(stream_handler));
    }

    let listener = QueueListener::new(queue.clone(), handlers, true);
    listener.start();
    *LISTENER.lock() = Some(Arc::clone(&listener));
    Ok(listener)
}

/// Stop the listener and detach the shared queue.
///
/// The queue owns no OS resources, so this is only about not leaving a queue
/// handler pointing at a listener that no longer drains it — a record emitted
/// after this would otherwise pile up unread.  Root falls back to the
/// last-resort handler, which still puts WARNING+ on stderr.
///
/// Safe to call more than once.
pub fn shutdown_logging(listener: Option<&Arc<QueueListener>>) {
    if let Some(listener) = listener {
        listener.stop();
        // stop() joins the listener thread but leaves the queue bound.
        listener.set_queue(None);
    }

    // Drop our reference to the stopped listener so nothing tries to act on
    // its handlers afterwards (see truncate_log_file).
    *LISTENER.lock() = None;

    ROOT_HANDLERS.write().retain(|h| !matches!(h, RootHandler::Queue(_)));

    *LOG_QUEUE.lock() = None;
}

/// Empty `<home_dir>/server.log` and return the bytes reclaimed.
///
/// When the listener is up we close its stream, truncate, and reopen.  On
/// POSIX that is belt and braces — the handler opens in append mode, so
/// `O_APPEND` puts every later write at EOF and the rollover check seeks to
/// the end before measuring, which means a bare truncate already resumes at
/// zero without rotating.  Windows is the reason it is here: a file another
/// handle holds open cannot be truncated, so the open-for-write below would
/// fail outright with the handler's stream still attached.
///
/// Falls back to a plain truncate when no listener is registered (CLI, tests).
/// A missing log reports `Ok(0)`; a log that exists but cannot be truncated
/// is an error (Python raises `OSError`).
pub fn truncate_log_file(home_dir: &str) -> std::io::Result<u64> {
    let log_path = Path::new(home_dir).join("server.log");
    let freed = match std::fs::metadata(&log_path) {
        Ok(meta) => meta.len(),
        Err(_) => return Ok(0),
    };

    let listener = current_listener();
    let handlers: Vec<Arc<dyn Handler>> = match &listener {
        Some(listener) => {
            let target = realpath(&log_path);
            listener
                .handlers()
                .into_iter()
                .filter(|h| {
                    h.as_any()
                        .downcast_ref::<RotatingFileHandler>()
                        .map(|rfh| realpath(&rfh.base_filename) == target)
                        .unwrap_or(false)
                })
                .collect()
        }
        None => Vec::new(),
    };
    let rotating: Vec<&RotatingFileHandler> =
        handlers.iter().filter_map(|h| h.as_any().downcast_ref::<RotatingFileHandler>()).collect();

    let mut guards: Vec<MutexGuard<'_, Option<Arc<LogStream>>>> =
        rotating.iter().map(|h| h.lock_stream()).collect();

    for guard in guards.iter_mut() {
        if let Some(stream) = guard.take() {
            stream.close();
        }
    }
    let result = std::fs::OpenOptions::new().read(true).write(true).open(&log_path).and_then(|f| f.set_len(0));
    // Reopen regardless of the outcome (Python's `finally` releases the locks;
    // the reopen sits after the truncate, and emit() reopens a missing stream).
    if result.is_ok() {
        for (guard, handler) in guards.iter_mut().zip(rotating.iter()) {
            match handler.open_stream() {
                Ok(stream) => **guard = Some(stream),
                Err(e) => {
                    drop(guards);
                    return Err(e);
                }
            }
        }
    }
    drop(guards);
    result?;

    Ok(freed)
}

/// Python `logging.getLogger(name)` stand-in: a handle on a named logger in
/// the dotted hierarchy (`""` = root).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Logger {
    pub name: String,
}

/// Get a handle on the named logger (`""` or `"root"` is the root logger).
pub fn get_logger(name: &str) -> Logger {
    let name = if name == "root" { "" } else { name };
    Logger { name: name.to_string() }
}

impl Logger {
    /// Explicit level (NOTSET when unset; the root logger defaults to WARNING).
    pub fn level(&self) -> i32 {
        match LOGGER_LEVELS.read().get(&self.name) {
            Some(level) => *level,
            None if self.name.is_empty() => WARNING,
            None => NOTSET,
        }
    }

    pub fn set_level(&self, level: impl Into<LevelSpec>) {
        let level = resolve_level(level);
        let mut levels = LOGGER_LEVELS.write();
        if level == NOTSET && !self.name.is_empty() {
            levels.remove(&self.name);
        } else {
            levels.insert(self.name.clone(), level);
        }
    }

    /// First explicit level walking up the hierarchy (root default WARNING).
    pub fn effective_level(&self) -> i32 {
        let levels = LOGGER_LEVELS.read();
        let mut name = self.name.as_str();
        while !name.is_empty() {
            if let Some(level) = levels.get(name) {
                if *level != NOTSET {
                    return *level;
                }
            }
            name = match name.rfind('.') {
                Some(idx) => &name[..idx],
                None => "",
            };
        }
        levels.get("").copied().unwrap_or(WARNING)
    }

    pub fn is_enabled_for(&self, level: i32) -> bool {
        level >= self.effective_level()
    }

    /// Emit a message at `level` through the root handlers.
    pub fn log(&self, level: i32, msg: &str) {
        if !self.is_enabled_for(level) {
            return;
        }
        dispatch_record(LogRecord::new(&self.name, level, msg));
    }

    pub fn debug(&self, msg: &str) {
        self.log(DEBUG, msg)
    }

    pub fn info(&self, msg: &str) {
        self.log(INFO, msg)
    }

    pub fn warning(&self, msg: &str) {
        self.log(WARNING, msg)
    }

    pub fn error(&self, msg: &str) {
        self.log(ERROR, msg)
    }
}

impl crate::utils::log_utils::LoggerLike for Logger {
    fn info(&self, msg: &str) {
        Logger::info(self, msg)
    }
    fn warning(&self, msg: &str) {
        Logger::warning(self, msg)
    }
    fn debug(&self, msg: &str) {
        Logger::debug(self, msg)
    }
    fn error(&self, msg: &str) {
        Logger::error(self, msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(msg: &str) -> LogRecord {
        LogRecord {
            name: "a.b".into(),
            levelno: INFO,
            levelname: "INFO".into(),
            message: msg.into(),
            created: chrono::Local::now(),
            thread_name: Some("t".into()),
        }
    }

    #[test]
    fn formatter_renders_python_placeholders() {
        let f = Formatter::new("%(name)s|%(levelname)-8s|%(message)s|%%");
        assert_eq!(f.format(&record("hi")), "a.b|INFO    |hi|%");
        let line = Formatter::default().format(&record("hi"));
        assert!(line.ends_with(" - a.b - INFO - hi"), "{line}");
        // "YYYY-mm-dd HH:MM:SS,mmm"
        assert_eq!(line.find(" - "), Some(23));
        assert_eq!(&line[19..20], ",");
    }

    #[test]
    fn resolves_levels() {
        assert_eq!(resolve_level("debug"), DEBUG);
        assert_eq!(resolve_level("WARN"), WARNING);
        assert_eq!(resolve_level("bogus"), INFO);
        assert_eq!(resolve_level(42), 42);
        assert_eq!(level_name(15), "Level 15");
    }

    #[test]
    fn rotating_handler_rotates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.log");
        let h = RotatingFileHandler::new(&path, 100, 2).unwrap();
        h.set_formatter(Formatter::new("%(message)s"));
        for _ in 0..10 {
            h.handle(&record(&"x".repeat(40)));
        }
        h.close();
        assert!(dir.path().join("x.log.1").exists());
        assert!(dir.path().join("x.log.2").exists());
        assert!(!dir.path().join("x.log.3").exists());
        assert!(std::fs::metadata(&path).unwrap().len() < 100);
    }
}
