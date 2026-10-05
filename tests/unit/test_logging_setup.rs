//! Port of `tests/unit/test_logging_setup.py`.
//!
//! Centralized logging plumbing:
//!
//! * A thread emits through the root queue handler, the listener picks it up,
//!   and the record reaches the attached handler.
//! * `start_listener` always writes to `<home>/server.log`.
//! * The queue owns no OS resources, so a process that dies abruptly leaves
//!   nothing behind for a resource tracker to report.
//!
//! The root logger, the log-queue singleton and the global `tracing` bridge
//! are process-wide, and the `unit` target runs every unit's tests in
//! parallel threads of one process — where any of them may log, or install
//! its own root handlers mid-test.  pytest ran these serially in a process of
//! their own, so each test here re-runs itself in a child copy of the test
//! binary (filtered to just that test) and asserts on the child's outcome.

use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use smartloop::utils::logging_setup::{
    current_listener, get_log_queue, get_logger, make_log_queue, reset_log_queue,
    root_queue_handler_count, setup_root_logging, shutdown_logging, start_listener,
    truncate_log_file, Formatter, Handler, ListenerOptions, LogQueue, QueueListener,
    RotatingFileHandler, StreamHandler, DEBUG, DEFAULT_FORMAT, WARNING,
};

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

/// Env var naming the single test a child process should actually run.
const CHILD_ENV: &str = "SLP_TEST_LOGGING_SETUP_CHILD";
/// Extra argument handed to a child (e.g. a tmp dir).
const CHILD_ARG_ENV: &str = "SLP_TEST_LOGGING_SETUP_ARG";

fn test_path(name: &str) -> String {
    // module_path!() is "unit::test_logging_setup"; libtest names drop the crate.
    let module = module_path!().split_once("::").map(|(_, m)| m).unwrap_or(module_path!());
    format!("{module}::{name}")
}

fn in_child(name: &str) -> bool {
    std::env::var(CHILD_ENV).as_deref() == Ok(name)
}

fn child_command(name: &str) -> Command {
    let mut cmd = Command::new(std::env::current_exe().expect("test binary path"));
    cmd.args([test_path(name).as_str(), "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, name);
    cmd
}

/// Run `body` in an isolated child process (see the module docs), wrapped in
/// the `_reset_log_queue_singleton` autouse fixture.
fn isolated(name: &str, body: impl FnOnce()) {
    if in_child(name) {
        reset_log_queue();
        body();
        reset_log_queue();
        return;
    }
    let out = child_command(name).output().expect("spawn child test process");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "child test {name} failed\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}");
    assert!(stdout.contains("1 passed"), "child test {name} did not run\n--- stdout ---\n{stdout}");
}

/// Block briefly until the listener's queue is drained, then stop it.
fn drain_listener(listener: &Arc<QueueListener>) {
    // The listener processes records in a background thread; give it a
    // moment to flush, then stop.
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline && listener.queue().map(|q| !q.empty()).unwrap_or(false) {
        std::thread::sleep(Duration::from_millis(10));
    }
    listener.stop();
}

/// A cloneable in-memory writer (Python `io.StringIO`).
#[derive(Clone, Default)]
struct SharedBuf(Arc<Mutex<Vec<u8>>>);

impl SharedBuf {
    fn getvalue(&self) -> String {
        String::from_utf8_lossy(&self.0.lock()).into_owned()
    }
}

impl Write for SharedBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Start a listener that writes formatted records into an in-memory buffer.
fn make_capturing_listener(queue: &LogQueue) -> (Arc<QueueListener>, SharedBuf) {
    let buf = SharedBuf::default();
    let handler = StreamHandler::new(Box::new(buf.clone()));
    handler.set_level(DEBUG);
    handler.set_formatter(Formatter::new(DEFAULT_FORMAT));
    let listener = QueueListener::new(queue.clone(), vec![Arc::new(handler)], true);
    listener.start();
    (listener, buf)
}

fn file_size(path: &Path) -> u64 {
    std::fs::metadata(path).expect("log file exists").len()
}

// --------------------------------------------------------------------------
// Thread -> parent listener
// --------------------------------------------------------------------------

/// A worker thread logging via the root logger lands in the listener sink.
#[test]
fn test_thread_log_records_reach_listener_handler() {
    isolated("test_thread_log_records_reach_listener_handler", || {
        let queue = LogQueue::new();
        setup_root_logging(&queue, "DEBUG");
        let (listener, buf) = make_capturing_listener(&queue);

        let t = std::thread::Builder::new()
            .name("probe-thread".into())
            .spawn(|| {
                get_logger("thread.test").info("hello from thread");
                // Rust code logs through `tracing`; the bridge feeds the same root.
                tracing::info!(target: "thread.tracing", "hello from tracing");
            })
            .unwrap();
        t.join().unwrap();
        drain_listener(&listener);

        let value = buf.getvalue();
        assert!(value.contains("hello from thread"), "{value}");
        assert!(value.contains("thread.test"), "{value}");
        assert!(value.contains("hello from tracing"), "{value}");
        assert!(value.contains("thread.tracing"), "{value}");
    });
}

// --------------------------------------------------------------------------
// Listener mode selection
// --------------------------------------------------------------------------

/// `start_listener` should always attach a rotating file handler to <home>/server.log.
#[test]
fn test_start_listener_creates_rotating_file_handler() {
    isolated("test_start_listener_creates_rotating_file_handler", || {
        let tmp = tempfile::tempdir().unwrap();
        let queue = LogQueue::new();
        setup_root_logging(&queue, "DEBUG");
        let listener = start_listener(
            &queue,
            ListenerOptions::new(tmp.path().to_str().unwrap()).file_level("DEBUG").force_console(Some(false)),
        )
        .unwrap();

        get_logger("file.test").info("file probe");
        // Listener thread is async — give it a moment to flush before reading.
        std::thread::sleep(Duration::from_millis(200));
        // Force the handler to flush its buffer to disk.
        for h in listener.handlers() {
            h.flush();
        }
        drain_listener(&listener);

        let log_path = tmp.path().join("server.log");
        assert!(log_path.exists());
        assert!(std::fs::read_to_string(&log_path).unwrap().contains("file probe"));
    });
}

// --------------------------------------------------------------------------
// api_settings.log_level controls whether the smartloop gag is lifted
// --------------------------------------------------------------------------

/// Set api_settings.log_level, install centralized logging, return listener.
fn run_install_with_log_level(tmp: &Path, level: &str) -> Arc<QueueListener> {
    use smartloop::api::config::API_SETTINGS;
    use smartloop::api::main::{install_centralized_logging, APP_SETTINGS, LOG_LISTENER};

    // main.rs clamps smartloop to WARNING; restore that baseline before
    // install_centralized_logging so each case starts from the same state.
    get_logger("smartloop").set_level(WARNING);

    API_SETTINGS.write().log_level = level.to_string();
    // home_dir is computed from slp_home — set the field.
    APP_SETTINGS.write().slp_home = Some(tmp.to_str().unwrap().to_string());

    // Fresh queue per call — the singleton in logging_setup persists otherwise.
    reset_log_queue();

    install_centralized_logging();
    LOG_LISTENER.lock().clone().expect("install_centralized_logging stashes the listener")
}

/// Shipped electron build (log_level=INFO) keeps main.rs's smartloop gag.
#[test]
fn test_log_level_info_keeps_smartloop_at_warning() {
    isolated("test_log_level_info_keeps_smartloop_at_warning", || {
        let tmp = tempfile::tempdir().unwrap();
        let listener = run_install_with_log_level(tmp.path(), "INFO");
        assert_eq!(get_logger("smartloop").level(), WARNING);
        listener.stop();
    });
}

/// Dev mode (log_level=DEBUG) lifts the gag so orchestrator logs reach the file.
#[test]
fn test_log_level_debug_lifts_smartloop_gag() {
    isolated("test_log_level_debug_lifts_smartloop_gag", || {
        let tmp = tempfile::tempdir().unwrap();
        let listener = run_install_with_log_level(tmp.path(), "DEBUG");
        assert_eq!(get_logger("smartloop").level(), DEBUG);
        listener.stop();
    });
}

// --------------------------------------------------------------------------
// Shutdown releases the queue
// --------------------------------------------------------------------------

/// `shutdown_logging` drops the queue handler and detaches the queue.
#[test]
fn test_shutdown_logging_closes_queue_and_detaches_handler() {
    isolated("test_shutdown_logging_closes_queue_and_detaches_handler", || {
        reset_log_queue();
        let queue = make_log_queue();
        setup_root_logging(&queue, "DEBUG");
        let (listener, _) = make_capturing_listener(&queue);

        shutdown_logging(Some(&listener));

        assert_eq!(root_queue_handler_count(), 0);
        assert!(get_log_queue().is_none());
        assert!(listener.queue().is_none());
    });
}

/// Called twice (lifespan shutdown + a retry path) it must not panic.
#[test]
fn test_shutdown_logging_is_idempotent() {
    isolated("test_shutdown_logging_is_idempotent", || {
        reset_log_queue();
        let queue = make_log_queue();
        setup_root_logging(&queue, "DEBUG");
        shutdown_logging(None);
        shutdown_logging(None);
        assert!(get_log_queue().is_none());
    });
}

/// Number of open file descriptors in this process.
#[cfg(unix)]
fn open_fd_count() -> usize {
    std::fs::read_dir("/dev/fd").map(|d| d.count()).unwrap_or(0)
}

/// The shared queue must be an in-process thread queue — no semaphores, no
/// `sem_open`, no OS handles of any kind.
///
/// A multiprocessing queue here cost three POSIX semaphores on every startup
/// in the Python app, which had to be unlinked before the process died or the
/// resource tracker reported them leaked — and `sem_open` is denied under the
/// macOS App Sandbox.  Nothing writes to this queue from another process.
#[test]
fn test_make_log_queue_owns_no_os_resources() {
    isolated("test_make_log_queue_owns_no_os_resources", || {
        #[cfg(unix)]
        let fds_before = open_fd_count();
        let queue: LogQueue = make_log_queue();
        #[cfg(unix)]
        assert_eq!(open_fd_count(), fds_before, "creating the log queue opened OS handles");
        assert!(make_log_queue().ptr_eq(&make_log_queue())); // idempotent singleton
        assert!(queue.ptr_eq(&make_log_queue()));
    });
}

/// A server that dies abruptly must leave nothing for a tracker to warn about.
///
/// `_exit` stands in for the SIGKILL `slp agent stop` escalates to, which
/// skips every exit finalizer.  Nothing is torn down here on purpose: the
/// arrangement must hold no OS resources in the first place.
///
/// stderr goes to a file the child inherits and is polled after the child
/// exits.
#[test]
fn test_server_logging_setup_leaks_no_semaphores() {
    const NAME: &str = "test_server_logging_setup_leaks_no_semaphores";
    if in_child(NAME) {
        let home = std::env::var(CHILD_ARG_ENV).expect("home dir for the child");
        let queue = make_log_queue();
        setup_root_logging(&queue, "DEBUG");
        let _listener = start_listener(&queue, ListenerOptions::new(home).force_console(Some(false))).unwrap();
        unsafe { libc::_exit(0) };
    }

    let tmp = tempfile::tempdir().unwrap();
    let err_path = tmp.path().join("child-stderr.txt");
    let err = std::fs::File::create(&err_path).unwrap();
    let status = child_command(NAME)
        .env(CHILD_ARG_ENV, tmp.path())
        .stdout(std::process::Stdio::null())
        .stderr(err)
        .status()
        .expect("spawn child");

    // No helper process outlives the child here (there is no resource
    // tracker), so a short grace period covers any late write.
    std::thread::sleep(Duration::from_millis(100));
    let stderr = std::fs::read_to_string(&err_path).unwrap_or_default();

    assert!(status.success(), "child exited with {status}: {stderr}");
    assert!(tmp.path().join("server.log").exists(), "child never set up logging: {stderr}");
    assert!(!stderr.contains("leaked semaphore"), "{stderr}");
}

// --------------------------------------------------------------------------
// truncate_log_file against a running listener
// --------------------------------------------------------------------------

/// A started listener writing to `<tmp>/server.log`.
fn listener_on(tmp: &Path) -> Arc<QueueListener> {
    let queue = LogQueue::new();
    setup_root_logging(&queue, "DEBUG");
    start_listener(
        &queue,
        ListenerOptions::new(tmp.to_str().unwrap()).file_level("DEBUG").force_console(Some(false)),
    )
    .unwrap()
}

fn flush(listener: &Arc<QueueListener>) {
    std::thread::sleep(Duration::from_millis(200));
    for h in listener.handlers() {
        h.flush();
    }
}

#[test]
fn test_truncate_reclaims_the_file_while_the_listener_is_up() {
    isolated("test_truncate_reclaims_the_file_while_the_listener_is_up", || {
        let tmp = tempfile::tempdir().unwrap();
        let listener = listener_on(tmp.path());
        let log_path = tmp.path().join("server.log");

        for i in 0..200 {
            get_logger("truncate.test").info(&format!("filler line {i}"));
        }
        flush(&listener);
        let before = file_size(&log_path);
        assert!(before > 0);

        let freed = truncate_log_file(tmp.path().to_str().unwrap()).unwrap();

        assert_eq!(freed, before);
        assert_eq!(file_size(&log_path), 0);
        drain_listener(&listener);
    });
}

/// The contract the sweep depends on: the space actually comes back.
///
/// This pins the observable outcome, not the mechanism.  On POSIX it holds
/// even without the close/reopen — the handler opens in append mode, so
/// `O_APPEND` puts the next write at EOF — which is why
/// `test_truncate_reopens_the_handler_stream` exists alongside it.
#[test]
fn test_logging_resumes_at_zero_after_a_truncate() {
    isolated("test_logging_resumes_at_zero_after_a_truncate", || {
        let tmp = tempfile::tempdir().unwrap();
        let listener = listener_on(tmp.path());
        let log_path = tmp.path().join("server.log");

        for i in 0..500 {
            get_logger("truncate.test").info(&format!("filler line {i}"));
        }
        flush(&listener);
        let before = file_size(&log_path);

        truncate_log_file(tmp.path().to_str().unwrap()).unwrap();
        get_logger("truncate.test").info("after the sweep");
        flush(&listener);

        let after = file_size(&log_path);
        assert!(
            after < 1024,
            "log went back to {after} bytes (was {before}) — the handler resumed writing past a hole instead of at zero"
        );
        assert!(std::fs::read_to_string(&log_path).unwrap().contains("after the sweep"));
        drain_listener(&listener);
    });
}

/// The mechanism itself, which no POSIX size assertion can catch.
///
/// Windows refuses to truncate a file another handle holds open, so the
/// stream must be closed before the truncate and reopened after.  A macOS run
/// would happily pass every byte-count assertion with the dance removed, so
/// pin the swap directly: same handler, a new stream object.
#[test]
fn test_truncate_reopens_the_handler_stream() {
    isolated("test_truncate_reopens_the_handler_stream", || {
        let tmp = tempfile::tempdir().unwrap();
        let listener = listener_on(tmp.path());

        get_logger("truncate.test").info("a line");
        flush(&listener);
        let handlers = listener.handlers();
        let handler = handlers
            .iter()
            .find_map(|h| h.as_any().downcast_ref::<RotatingFileHandler>())
            .expect("a rotating file handler");
        let before = handler.stream().expect("an open stream");

        truncate_log_file(tmp.path().to_str().unwrap()).unwrap();

        let after = handler.stream().expect("a reopened stream");
        assert!(!Arc::ptr_eq(&after, &before), "the stream was never reopened");
        assert!(before.is_closed(), "the old stream was left open");
        assert!(!after.is_closed());
        drain_listener(&listener);
    });
}

/// Handlers are matched by realpath, so an unrelated rotating handler on the
/// same listener keeps its stream and its bytes.
#[test]
fn test_truncate_leaves_a_handler_on_another_file_alone() {
    isolated("test_truncate_leaves_a_handler_on_another_file_alone", || {
        let tmp = tempfile::tempdir().unwrap();
        let listener = listener_on(tmp.path());
        let other_path = tmp.path().join("other.log");
        let other = Arc::new(RotatingFileHandler::new(&other_path, 0, 0).unwrap());
        other.set_formatter(Formatter::new(DEFAULT_FORMAT));
        listener.add_handler(other.clone());

        for i in 0..100 {
            get_logger("truncate.test").info(&format!("filler line {i}"));
        }
        flush(&listener);
        let other_before = file_size(&other_path);
        assert!(other_before > 0);

        truncate_log_file(tmp.path().to_str().unwrap()).unwrap();

        assert_eq!(file_size(&other_path), other_before);
        assert_eq!(file_size(&tmp.path().join("server.log")), 0);
        drain_listener(&listener);
        other.close();
    });
}

#[test]
fn test_truncate_on_a_missing_log_reports_nothing_freed() {
    isolated("test_truncate_on_a_missing_log_reports_nothing_freed", || {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(truncate_log_file(tmp.path().to_str().unwrap()).unwrap(), 0);
    });
}

/// `truncate_log_file` reads the registered listener; a stale one would have
/// it closing and reopening handlers belonging to a stopped listener.
#[test]
fn test_shutdown_clears_the_listener_reference() {
    isolated("test_shutdown_clears_the_listener_reference", || {
        let tmp = tempfile::tempdir().unwrap();
        let listener = listener_on(tmp.path());
        assert!(current_listener().is_some_and(|l| Arc::ptr_eq(&l, &listener)));

        shutdown_logging(Some(&listener));

        assert!(current_listener().is_none());
    });
}
