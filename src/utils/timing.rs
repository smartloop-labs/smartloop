//! Opt-in request timing, to find where a chat turn's time goes.
//!
//! With `SLP_TIMING=1` set (or a file path), events go as JSON lines to
//! `<slp_home>/logs/slp-timing.jsonl`; unset, every call returns at once.
//! The supervisor, the per-project agents and the model host all append to
//! the same file, one whole line per write, and every line names its
//! process and request so `scripts/timing-summary.sh` can join them.
//!
//! A request is opened with [`begin`] and carried as ambient context: a
//! task-local inside async scopes ([`scope`]), a thread-local elsewhere
//! ([`enter`]), and across thread hops by [`crate::workspace::context::bind`],
//! which takes a [`carry`] along with the project binding.  [`phase`] guards
//! time a stretch of work and add it to the request's tally; when the last
//! reference to the request drops, a `request.end` event totals it and one
//! summary line goes to the `smartloop.timing` logger.
//!
//! The logging bridge drops spans, so all of this is events, not
//! `#[instrument]`.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::future::Future;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::ThreadId;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use serde_json::{Value, json};

pub const LOG_TARGET: &str = "smartloop.timing";

/// The header (and gRPC metadata key) a request id travels in between
/// processes.
pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// The log file, relative to `<slp_home>`.
pub const LOG_PATH: &str = "logs/slp-timing.jsonl";

enum Sink {
    File(File),
    #[cfg(test)]
    Memory(Vec<String>),
}

static SINK: Lazy<Mutex<Option<Sink>>> = Lazy::new(|| {
    let sink = open();
    ENABLED.store(sink.is_some(), Ordering::Relaxed);
    Mutex::new(sink)
});

static ENABLED: AtomicBool = AtomicBool::new(false);

/// Which process wrote a line: `supervisor`, `agent:<project>`, `host`.
static PROCESS: RwLock<String> = RwLock::new(String::new());

/// Where a `SLP_TIMING` value says to write, if anywhere.
fn path_for(setting: &str, home: &Path) -> Option<PathBuf> {
    match setting.trim() {
        "" | "0" | "false" | "off" => None,
        "1" | "true" | "on" => Some(home.join(LOG_PATH)),
        file => Some(PathBuf::from(file)),
    }
}

fn open() -> Option<Sink> {
    if cfg!(test) {
        return None;
    }
    let setting = std::env::var("SLP_TIMING").ok()?;
    let path = path_for(&setting, &crate::auth::credential_store::slp_home())?;
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    OpenOptions::new().create(true).append(true).open(&path).ok().map(Sink::File)
}

/// Whether timing is on.
pub fn on() -> bool {
    Lazy::force(&SINK);
    ENABLED.load(Ordering::Relaxed)
}

/// Name this process in every line it writes.
pub fn set_process(name: &str) {
    *PROCESS.write() = name.to_string();
}

/// A duration in ms, to the µs.
pub fn ms(d: Duration) -> f64 {
    (d.as_secs_f64() * 1e6).round() / 1e3
}

/// A fresh request id.
pub fn new_id() -> String {
    format!("req-{}", &uuid::Uuid::new_v4().simple().to_string()[..8])
}

/// One event as its JSON line: the time in µs since the epoch, the process,
/// the request, then `fields`.
fn line(ts: u128, process: &str, req: Option<&str>, ev: &str, fields: Value) -> String {
    let mut event = json!({
        "ts": ts as u64,
        "pid": std::process::id(),
        "proc": process,
        "req": req,
        "ev": ev,
    });
    if let (Some(event), Value::Object(fields)) = (event.as_object_mut(), fields) {
        event.extend(fields);
    }
    event.to_string()
}

fn emit(req: Option<&str>, ev: &str, fields: Value) {
    if !on() {
        return;
    }
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_micros());
    let mut text = line(ts, &PROCESS.read(), req, ev, fields);
    text.push('\n');
    match SINK.lock().as_mut() {
        // One write per line, so lines from other processes never land
        // inside it.
        Some(Sink::File(file)) => {
            let _ = file.write_all(text.as_bytes());
        }
        #[cfg(test)]
        Some(Sink::Memory(lines)) => lines.push(text),
        None => {}
    }
}

/// Write `ev` with `fields`, as part of the current request.
pub fn event(ev: &str, fields: Value) {
    if !on() {
        return;
    }
    let req = current();
    emit(req.as_ref().map(|r| r.id.as_str()), ev, fields);
}

#[derive(Default)]
struct Tally {
    /// Time and count per phase name, in the order first seen.
    phases: Vec<(String, Duration, u64)>,
    counts: BTreeMap<String, u64>,
    marks: BTreeMap<String, Duration>,
}

/// One request's identity and tally.  Shared by every thread working on it;
/// the summary goes out when the last of them lets go.
pub struct Request {
    id: String,
    kind: String,
    started: Instant,
    tally: Mutex<Tally>,
    /// False for a request adopted from another process, which writes the
    /// summary itself.
    summarize: bool,
}

impl Request {
    pub fn id(&self) -> &str {
        &self.id
    }

    fn add_phase(&self, name: &str, took: Duration) {
        let mut tally = self.tally.lock();
        match tally.phases.iter_mut().find(|(n, _, _)| n == name) {
            Some((_, total, count)) => {
                *total += took;
                *count += 1;
            }
            None => tally.phases.push((name.to_string(), took, 1)),
        }
    }

    fn count(&self, key: &str) {
        *self.tally.lock().counts.entry(key.to_string()).or_default() += 1;
    }

    /// Record `name` at the time since the request began, the first time
    /// only; returns that time.
    fn mark(&self, name: &str) -> Duration {
        let elapsed = self.started.elapsed();
        *self.tally.lock().marks.entry(name.to_string()).or_insert(elapsed)
    }

    /// The `request.end` fields and the summary line.
    fn summary(&self) -> (Value, String) {
        let total = self.started.elapsed();
        let tally = self.tally.lock();
        let phases: serde_json::Map<String, Value> = tally
            .phases
            .iter()
            .map(|(name, took, n)| (name.clone(), json!({"ms": ms(*took), "n": n})))
            .collect();
        let marks: serde_json::Map<String, Value> =
            tally.marks.iter().map(|(name, at)| (name.clone(), json!(ms(*at)))).collect();
        let fields = json!({
            "kind": self.kind,
            "ms": ms(total),
            "phases": phases,
            "marks": marks,
            "counts": tally.counts,
        });

        let mut parts = vec![format!("[timing] {} {} {}", self.id, self.kind, format_ms(total))];
        for (name, at) in &tally.marks {
            parts.push(format!("{name}@{}", format_ms(*at)));
        }
        for (name, took, n) in &tally.phases {
            let times = if *n > 1 { format!("×{n}") } else { String::new() };
            parts.push(format!("{name} {}{times}", format_ms(*took)));
        }
        for (key, n) in &tally.counts {
            parts.push(format!("{key}={n}"));
        }
        (fields, parts.join(" · "))
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        if !self.summarize {
            return;
        }
        let (fields, text) = self.summary();
        emit(Some(&self.id), "request.end", fields);
        tracing::info!(target: LOG_TARGET, "{}", text);
    }
}

fn format_ms(d: Duration) -> String {
    if d >= Duration::from_secs(1) {
        format!("{:.2}s", d.as_secs_f64())
    } else {
        format!("{}ms", d.as_millis())
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Arc<Request>>> = const { RefCell::new(None) };
    /// The open phases on this thread, innermost last: who is making a call.
    static CALLERS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

tokio::task_local! {
    static TASK_CURRENT: Option<Arc<Request>>;
}

/// Open a request with `id`, or a fresh one; `None` while timing is off.
pub fn begin(id: Option<&str>, kind: &str, fields: Value) -> Option<Arc<Request>> {
    if !on() {
        return None;
    }
    let id = id.filter(|id| !id.is_empty()).map_or_else(new_id, str::to_string);
    let request = Arc::new(Request {
        id,
        kind: kind.to_string(),
        started: Instant::now(),
        tally: Mutex::new(Tally::default()),
        summarize: true,
    });
    let mut fields = fields;
    if let Value::Object(map) = &mut fields {
        map.insert("kind".into(), json!(kind));
    }
    emit(Some(&request.id), "request.begin", fields);
    Some(request)
}

/// Join request `id`, begun in another process: events carry its id, and
/// that process totals it.
pub fn adopt(id: Option<&str>) -> Option<Arc<Request>> {
    let id = id.filter(|id| !id.is_empty() && on())?;
    Some(Arc::new(Request {
        id: id.to_string(),
        kind: String::new(),
        started: Instant::now(),
        tally: Mutex::new(Tally::default()),
        summarize: false,
    }))
}

/// The request this work belongs to, if any.
pub fn current() -> Option<Arc<Request>> {
    if !on() {
        return None;
    }
    CURRENT
        .with(|c| c.borrow().clone())
        .or_else(|| TASK_CURRENT.try_with(|r| r.clone()).ok().flatten())
}

/// The current request's id, if any.
pub fn current_id() -> Option<String> {
    current().map(|r| r.id.clone())
}

/// Run `fut` with `request` as the current request.
pub async fn scope<F: Future>(request: Option<Arc<Request>>, fut: F) -> F::Output {
    TASK_CURRENT.scope(request, fut).await
}

/// Restores the previous thread-local request on drop.
#[must_use = "the request is unset when the guard is dropped"]
pub struct Entered {
    previous: Option<Option<Arc<Request>>>,
    caller: bool,
}

impl Drop for Entered {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            CURRENT.with(|c| *c.borrow_mut() = previous);
        }
        if self.caller {
            CALLERS.with(|c| c.borrow_mut().pop());
        }
    }
}

/// Make `request` current on this thread for the life of the guard.
pub fn enter(request: Option<Arc<Request>>) -> Entered {
    Carry { request, caller: None }.enter()
}

/// The current request and caller, captured to re-enter on another thread.
#[derive(Default)]
pub struct Carry {
    request: Option<Arc<Request>>,
    caller: Option<String>,
}

/// Capture what [`Carry::enter`] restores elsewhere.
pub fn carry() -> Carry {
    if !on() {
        return Carry::default();
    }
    Carry { request: current(), caller: current_caller() }
}

impl Carry {
    pub fn enter(self) -> Entered {
        if self.request.is_none() && self.caller.is_none() {
            return Entered { previous: None, caller: false };
        }
        let previous = CURRENT.with(|c| std::mem::replace(&mut *c.borrow_mut(), self.request));
        let caller = match self.caller {
            Some(caller) => {
                CALLERS.with(|c| c.borrow_mut().push(caller));
                true
            }
            None => false,
        };
        Entered { previous: Some(previous), caller }
    }
}

/// The innermost open phase on this thread.
pub fn current_caller() -> Option<String> {
    CALLERS.with(|c| c.borrow().last().cloned())
}

/// Name the caller of the work done on this thread, without timing it.
pub fn caller(name: &str) -> Entered {
    if !on() {
        return Entered { previous: None, caller: false };
    }
    CALLERS.with(|c| c.borrow_mut().push(name.to_string()));
    Entered { previous: None, caller: true }
}

/// Count `key` in the current request's tally.
pub fn count(key: &str) {
    if let Some(request) = current() {
        request.count(key);
    }
}

/// Mark `name` in the current request (the first time only) and log it as
/// an event; returns the time since the request began.
pub fn mark(name: &str) -> Option<Duration> {
    let request = current()?;
    let at = request.mark(name);
    emit(Some(&request.id), "mark", json!({"name": name, "ms": ms(at)}));
    Some(at)
}

/// Times a stretch of work until dropped, then logs it and adds it to the
/// request's tally.
#[must_use = "the phase ends when the guard is dropped"]
pub struct Phase {
    inner: Option<PhaseInner>,
}

struct PhaseInner {
    name: String,
    started: Instant,
    request: Option<Arc<Request>>,
    fields: Value,
    /// The thread whose caller stack this phase is on, if any.
    stacked: Option<ThreadId>,
}

impl Phase {
    fn start(name: &str, fields: Value, stacked: bool) -> Self {
        if !on() {
            return Self { inner: None };
        }
        let stacked = stacked.then(|| {
            CALLERS.with(|c| c.borrow_mut().push(name.to_string()));
            std::thread::current().id()
        });
        let fields = if fields.is_object() { fields } else { json!({}) };
        Self {
            inner: Some(PhaseInner {
                name: name.to_string(),
                started: Instant::now(),
                request: current(),
                fields,
                stacked,
            }),
        }
    }

    /// Add `key` to what the phase logs when it ends.
    pub fn field(&mut self, key: &str, value: impl Into<Value>) {
        if let Some(inner) = &mut self.inner {
            inner.fields[key] = value.into();
        }
    }

    /// Time since the phase began.
    pub fn elapsed(&self) -> Option<Duration> {
        self.inner.as_ref().map(|i| i.started.elapsed())
    }
}

impl Drop for Phase {
    fn drop(&mut self) {
        let Some(mut inner) = self.inner.take() else {
            return;
        };
        let took = inner.started.elapsed();
        if inner.stacked == Some(std::thread::current().id()) {
            CALLERS.with(|c| {
                let mut callers = c.borrow_mut();
                if let Some(at) = callers.iter().rposition(|n| *n == inner.name) {
                    callers.remove(at);
                }
            });
        }
        if let Some(request) = &inner.request {
            request.add_phase(&inner.name, took);
        }
        inner.fields["name"] = json!(inner.name);
        inner.fields["ms"] = json!(ms(took));
        emit(inner.request.as_ref().map(|r| r.id.as_str()), "phase", inner.fields);
    }
}

/// Time `name` on this thread; calls made inside name it as their caller.
pub fn phase(name: &str) -> Phase {
    Phase::start(name, json!({}), true)
}

/// [`phase`] with `fields` logged alongside.
pub fn phase_with(name: &str, fields: Value) -> Phase {
    Phase::start(name, fields, true)
}

/// Time `name` across awaits: like [`phase`], but off the caller stack,
/// since an async task can resume on another thread.
pub fn timer(name: &str) -> Phase {
    Phase::start(name, json!({}), false)
}

/// Time one LLM call: a phase named `llm.<capability>`, logged with the
/// phase that made it, the model and the prompt size.
pub fn llm_call(call: &str, capability: Option<&str>, model: Option<&str>, prompt_chars: usize) -> Phase {
    if !on() {
        return Phase { inner: None };
    }
    let capability = capability.filter(|c| !c.is_empty());
    let name = format!("llm.{}", capability.unwrap_or("default"));
    let fields = json!({
        "call": call,
        "caller": current_caller(),
        "capability": capability,
        "model": model.filter(|m| !m.is_empty()),
        "prompt_chars": prompt_chars,
    });
    Phase::start(&name, fields, false)
}

/// Wrap a stream so `phase` logs when its first item came and how many
/// followed; the phase ends when the stream is dropped.
pub fn timed_stream<'a, T: 'a>(
    stream: Box<dyn Iterator<Item = T> + 'a>,
    phase: Phase,
) -> Box<dyn Iterator<Item = T> + 'a> {
    if phase.inner.is_none() {
        return stream;
    }
    let mut phase = phase;
    let mut chunks: u64 = 0;
    Box::new(stream.inspect(move |_| {
        chunks += 1;
        if chunks == 1
            && let Some(first) = phase.elapsed()
        {
            phase.field("first_token_ms", ms(first));
        }
        phase.field("chunks", chunks);
    }))
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    /// Turn timing on for this test binary, writing to memory.
    pub fn capture() {
        let mut sink = SINK.lock();
        if sink.is_none() {
            *sink = Some(Sink::Memory(Vec::new()));
        }
        ENABLED.store(true, Ordering::Relaxed);
    }

    /// The captured events of request `id`.
    pub fn events(id: &str) -> Vec<Value> {
        match SINK.lock().as_ref() {
            Some(Sink::Memory(lines)) => lines
                .iter()
                .filter_map(|l| serde_json::from_str::<Value>(l).ok())
                .filter(|v| v["req"] == id)
                .collect(),
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{capture, events};
    use super::*;

    #[test]
    fn setting_picks_the_file() {
        let home = Path::new("/h");
        assert_eq!(path_for("", home), None);
        assert_eq!(path_for("off", home), None);
        assert_eq!(path_for("1", home), Some(PathBuf::from("/h/logs/slp-timing.jsonl")));
        assert_eq!(path_for("/tmp/t.jsonl", home), Some(PathBuf::from("/tmp/t.jsonl")));
    }

    #[test]
    fn line_has_its_keys_then_fields() {
        let v: Value = serde_json::from_str(&line(12, "agent:p", Some("req-1"), "ev", json!({"ms": 1.5}))).unwrap();
        assert_eq!(v["ts"], 12);
        assert_eq!(v["proc"], "agent:p");
        assert_eq!(v["req"], "req-1");
        assert_eq!(v["ev"], "ev");
        assert_eq!(v["ms"], 1.5);
        let v: Value = serde_json::from_str(&line(0, "", None, "ev", json!({}))).unwrap();
        assert!(v["req"].is_null());
    }

    #[test]
    fn phases_total_up_when_the_request_ends() {
        capture();
        let request = begin(None, "chat", json!({})).unwrap();
        let id = request.id().to_string();
        {
            let _entered = enter(Some(request));
            for _ in 0..2 {
                let _p = phase("chat.plan");
            }
            count("model.load");
            mark("first_token");
        }
        let end = events(&id).into_iter().find(|e| e["ev"] == "request.end").expect("request.end");
        assert_eq!(end["phases"]["chat.plan"]["n"], 2);
        assert_eq!(end["counts"]["model.load"], 1);
        assert!(end["marks"]["first_token"].is_number());
    }

    #[test]
    fn bind_carries_the_request_and_caller_to_a_worker() {
        capture();
        let request = begin(None, "chat", json!({})).unwrap();
        let id = request.id().to_string();
        let caller = {
            let _entered = enter(Some(request));
            let _p = phase("orchestrator.plan");
            let job = crate::workspace::context::bind(|| {
                let _inner = phase("llm.orchestrator");
                current_caller()
            });
            std::thread::spawn(job).join().unwrap()
        };
        assert_eq!(caller.as_deref(), Some("llm.orchestrator"));
        let end = events(&id).into_iter().find(|e| e["ev"] == "request.end").expect("request.end");
        assert_eq!(end["phases"]["llm.orchestrator"]["n"], 1);
        assert_eq!(end["phases"]["orchestrator.plan"]["n"], 1);
    }

    #[tokio::test]
    async fn scope_reaches_spawn_blocking_through_bind() {
        capture();
        let request = begin(Some("req-scoped"), "chat", json!({}));
        let seen = scope(request, async {
            tokio::task::spawn_blocking(crate::workspace::context::bind(current_id)).await.unwrap()
        })
        .await;
        assert_eq!(seen.as_deref(), Some("req-scoped"));
    }

    #[test]
    fn llm_stream_logs_its_caller_and_first_token() {
        capture();
        let request = begin(None, "chat", json!({})).unwrap();
        let id = request.id().to_string();
        {
            let _entered = enter(Some(request));
            let _caller = caller("chat.answer");
            let stream = timed_stream(Box::new(vec![1, 2, 3].into_iter()), llm_call("stream", Some("coding"), None, 10));
            assert_eq!(stream.count(), 3);
        }
        let call = events(&id)
            .into_iter()
            .find(|e| e["ev"] == "phase" && e["name"] == "llm.coding")
            .expect("llm phase");
        assert_eq!(call["caller"], "chat.answer");
        assert_eq!(call["chunks"], 3);
        assert!(call["first_token_ms"].is_number());
    }

    #[test]
    fn phases_pop_off_the_caller_stack() {
        capture();
        {
            let _outer = phase("outer");
            {
                let _inner = phase("inner");
                assert_eq!(current_caller().as_deref(), Some("inner"));
            }
            assert_eq!(current_caller().as_deref(), Some("outer"));
        }
        assert_eq!(current_caller(), None);
    }
}
