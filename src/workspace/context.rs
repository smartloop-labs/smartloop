//! Port of `smartloop/workspace/context.py`.
//!
//! Request-scoped project binding.
//!
//! The project a piece of work belongs to used to be global process state: a
//! `current` flag on one [`Project`], flipped per request by
//! `ModelService::set_current_project`.  Two chats on different projects
//! therefore raced — the second request moved the flag while the first was
//! still resolving documents, skills and MCP servers against it.
//!
//! Here the binding is *scoped* instead of *mutated*.  A request enters
//! [`use_project`] (sync, RAII guard) or [`scope_project`] (async, task-local)
//! for its whole lifetime, and every downstream reader resolves against the
//! context rather than the flag (see [`Workspace::current_project`]).  Two
//! requests on different projects hold different bindings at the same time,
//! so neither can move the other's.
//!
//! Three layers resolve, in order:
//!
//! 1. the active-project binding — the request's own binding.  Inside an
//!    async scope it is a tokio task-local (authoritative while in scope, and
//!    the sync guards write into it there); otherwise a thread-local set by
//!    the RAII guards;
//! 2. the *process pin* ([`pin_project`]) — a per-project agent child fixes its
//!    project once at spawn from `SLP_PROJECT_ID` and never varies it;
//! 3. `None` — the caller falls back to the `current` flag, which means only
//!    "the project the UI has selected".
//!
//! # Ownership
//!
//! Binding says which project a piece of work is *about*; [`delegate_projects`]
//! and [`own_project`] say whose state a process may *write*.  The supervisor
//! delegates every project's own file to the child pinned to it, and claims one
//! back only for the operations that straddle a child's lifetime — create,
//! import, delete.  [`Workspace::save`] reads both.
//!
//! # Threads
//!
//! A binding does not cross a thread boundary on its own, and the chat path
//! hands work to executors several times per request.  [`bind`] captures the
//! binding at the submitting side and re-establishes it in the worker, and
//! [`ProjectBoundExecutor`] applies it to every `submit`.  Binding is copied
//! rather than entered, so the same binding can be live in any number of
//! worker threads at once.

use std::cell::RefCell;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread::ThreadId;

use parking_lot::{Mutex, RwLock};

use crate::workspace::workspace::{Project, Workspace};

/// Logger name used by the Python module.
const LOG_TARGET: &str = "smartloop.workspace.context";

thread_local! {
    /// The project this unit of work belongs to (sync/thread scope). `None`
    /// means "unbound" — fall through to the process pin, then to the
    /// `current` flag.
    static ACTIVE_PROJECT_ID: RefCell<Option<String>> = const { RefCell::new(None) };

    /// The project whose file this unit of work is entitled to write, in a
    /// process that has otherwise delegated them (sync/thread scope).
    static OWNED_PROJECT_ID: RefCell<Option<String>> = const { RefCell::new(None) };
}

tokio::task_local! {
    /// Async-scope counterpart of the active-project binding.  While in scope
    /// it is authoritative (including an explicit `None` from
    /// [`unbind_project_scope`]); the sync guards and tokens write into it
    /// rather than into the thread-local, so a binding made inside a task
    /// follows the task across worker threads — `ContextVar` semantics.
    static TASK_ACTIVE_PROJECT_ID: RefCell<Option<String>>;

    /// Async-scope counterpart of the owned-project claim.
    static TASK_OWNED_PROJECT_ID: RefCell<Option<String>>;
}

/// Process-wide pin, set once at startup by a per-project agent child.  Not a
/// scoped binding: it is a property of the process, not of a request, and it
/// must survive threads that were never bound (background tasks, the task
/// queue) without every one of them having to opt in.
static PINNED_PROJECT_ID: RwLock<Option<String>> = RwLock::new(None);

/// Whether this process has handed every project's own state to someone else.
/// Set by the supervisor, which spawns a child per project: from that moment
/// the files under each `project_dir` belong to those children, and the
/// supervisor keeps only `config.json`.
static PROJECTS_DELEGATED: RwLock<bool> = RwLock::new(false);

/// Serializes writes to the process-wide pin / delegation flags.
static PIN_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BindingKind {
    Active,
    Owned,
}

/// Where a scoped binding was written: the enclosing task's task-local, or
/// (outside any task scope) one thread's thread-local.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Store {
    Task,
    Thread(ThreadId),
}

fn task_key(kind: BindingKind) -> &'static tokio::task::LocalKey<RefCell<Option<String>>> {
    match kind {
        BindingKind::Active => &TASK_ACTIVE_PROJECT_ID,
        BindingKind::Owned => &TASK_OWNED_PROJECT_ID,
    }
}

fn thread_key(kind: BindingKind) -> &'static std::thread::LocalKey<RefCell<Option<String>>> {
    match kind {
        BindingKind::Active => &ACTIVE_PROJECT_ID,
        BindingKind::Owned => &OWNED_PROJECT_ID,
    }
}

/// The binding of `kind` visible from here.
fn read_binding(kind: BindingKind) -> Option<String> {
    match task_key(kind).try_with(|cell| cell.borrow().clone()) {
        Ok(value) => value,
        Err(_) => thread_key(kind).with(|cell| cell.borrow().clone()),
    }
}

/// Replace the binding of `kind` visible from here; returns the previous
/// value and the store it was written to.
fn swap_binding(kind: BindingKind, value: Option<String>) -> (Option<String>, Store) {
    let mut pending = Some(value);
    if let Ok(previous) = task_key(kind).try_with(|cell| cell.replace(pending.take().flatten())) {
        return (previous, Store::Task);
    }
    let previous = thread_key(kind).with(|cell| cell.replace(pending.take().flatten()));
    (previous, Store::Thread(std::thread::current().id()))
}

/// Restore `value` into `store`, tolerating a store that is no longer
/// reachable from here — a guard or token unwound from another context, where
/// Python's `ContextVar.reset` raises and there is nothing to repair.
fn restore_binding(kind: BindingKind, store: Store, value: Option<String>) {
    match store {
        Store::Task => {
            let _ = task_key(kind).try_with(|cell| {
                cell.replace(value);
            });
        }
        Store::Thread(thread_id) => {
            if std::thread::current().id() == thread_id {
                thread_key(kind).with(|cell| {
                    cell.replace(value);
                });
            }
        }
    }
}

/// Fix this process to `project_id` for its whole lifetime.
///
/// Used by a per-project agent child, which is spawned for exactly one project
/// and has no reason to ever look at another.  The supervisor process leaves
/// this unset, so its own reads still resolve through the `current` flag.
pub fn pin_project(project_id: Option<&str>) {
    {
        let _lock = PIN_LOCK.lock();
        *PINNED_PROJECT_ID.write() = project_id.map(str::to_string);
    }
    if let Some(project_id) = project_id.filter(|p| !p.is_empty()) {
        tracing::info!(
            target: LOG_TARGET,
            "[project-context] process pinned to project {}",
            project_id
        );
    }
}

/// The project this process is pinned to, or `None` when unpinned.
pub fn pinned_project_id() -> Option<String> {
    PINNED_PROJECT_ID.read().clone()
}

/// Whether this process owns workspace-level state.
///
/// The positive form of `pinned_project_id().is_none()`.  True in the
/// supervisor, in a `--no-supervised` single server, and in the CLI; false in
/// a per-project agent child, which is pinned to one project and shares
/// `config.json`, `models_dir` and the credential store with N siblings it
/// has no way to coordinate with.
///
/// Ownership, not binding.  [`pinned_project_id`] says which project this
/// process serves; this says whether it may act for the workspace as a whole.
/// Workspace-level background work is gated on it (the token refresh and
/// release-poll loops), and so is redeeming the refresh token.
pub fn owns_workspace() -> bool {
    PINNED_PROJECT_ID.read().is_none()
}

/// Declare that per-project state is owned by other processes.
///
/// Called by the supervisor when its pool comes up, and cleared when it goes
/// down.  [`Workspace::save`] reads this to decide whether it may write every
/// project's file or only the workspace record — see [`own_project`] for the
/// exception.
pub fn delegate_projects(delegated: bool) {
    {
        let _lock = PIN_LOCK.lock();
        *PROJECTS_DELEGATED.write() = delegated;
    }
    tracing::info!(
        target: LOG_TARGET,
        "[project-context] per-project state delegated={}",
        if delegated { "True" } else { "False" }
    );
}

/// Whether another process owns each project's own state.
pub fn projects_delegated() -> bool {
    *PROJECTS_DELEGATED.read()
}

/// RAII guard restoring the previous binding on drop.
///
/// Returned by [`use_project`], [`unbind_project`], [`own_project`] and
/// [`ProjectContext::bound`].
#[must_use = "the binding is undone when the guard is dropped"]
pub struct ProjectBinding {
    kind: BindingKind,
    /// `None` for a no-op guard; otherwise the binding to restore on drop.
    previous: Option<Option<String>>,
    project_id: Option<String>,
    store: Option<Store>,
}

impl ProjectBinding {
    /// The project id this guard bound (the value `with ... as pid` yielded in
    /// Python), or `None` for a no-op / unbinding guard.
    pub fn project_id(&self) -> Option<&str> {
        self.project_id.as_deref()
    }

    fn noop(kind: BindingKind) -> Self {
        Self { kind, previous: None, project_id: None, store: None }
    }

    fn enter(kind: BindingKind, value: Option<String>) -> Self {
        let (previous, store) = swap_binding(kind, value.clone());
        Self { kind, previous: Some(previous), project_id: value, store: Some(store) }
    }
}

impl Drop for ProjectBinding {
    fn drop(&mut self) {
        if let (Some(previous), Some(store)) = (self.previous.take(), self.store) {
            restore_binding(self.kind, store, previous);
        }
    }
}

impl std::fmt::Debug for ProjectBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProjectBinding")
            .field("kind", &self.kind)
            .field("project_id", &self.project_id)
            .finish()
    }
}

/// Claim `project_id`'s own state for the lifetime of the returned guard.
///
/// The supervisor delegates project files to its children, but three
/// operations happen either side of a child's existence and must still write
/// one: creating a project, importing one, and deleting one.  Claiming it says
/// which file this save is allowed to touch, so the claim stays as narrow as
/// the operation instead of re-opening the whole workspace to writes.  A falsy
/// id is a no-op claim.
pub fn own_project(project_id: Option<&str>) -> ProjectBinding {
    match project_id.filter(|p| !p.is_empty()) {
        None => ProjectBinding::noop(BindingKind::Owned),
        Some(project_id) => ProjectBinding::enter(BindingKind::Owned, Some(project_id.to_string())),
    }
}

/// Async form of [`own_project`]: run `fut` with `project_id` claimed.
pub async fn own_project_scope<F: Future>(project_id: Option<String>, fut: F) -> F::Output {
    match project_id.filter(|p| !p.is_empty()) {
        None => fut.await,
        Some(project_id) => TASK_OWNED_PROJECT_ID.scope(RefCell::new(Some(project_id)), fut).await,
    }
}

/// The project claimed by [`own_project`], if any.
pub fn owned_project_id() -> Option<String> {
    read_binding(BindingKind::Owned)
}

/// The project bound to the current request or process, if any.
///
/// Returns `None` when nothing is bound — callers then fall back to the
/// `current` flag, which is what a supervisor process and the CLI want.
pub fn active_project_id() -> Option<String> {
    match read_binding(BindingKind::Active) {
        Some(project_id) if !project_id.is_empty() => Some(project_id),
        _ => pinned_project_id(),
    }
}

/// Bind `project_id` for the lifetime of the returned guard (sync scope).
///
/// A falsy `project_id` is a no-op binding: the caller did not name a project,
/// so resolution should keep falling through rather than pin to nothing.
pub fn use_project(project_id: Option<&str>) -> ProjectBinding {
    match project_id.filter(|p| !p.is_empty()) {
        None => ProjectBinding::noop(BindingKind::Active),
        Some(project_id) => ProjectBinding::enter(BindingKind::Active, Some(project_id.to_string())),
    }
}

/// Async form of [`use_project`]: run `fut` with `project_id` bound as a
/// task-local.  A falsy id runs `fut` without changing the binding.
pub async fn scope_project<F: Future>(project_id: Option<String>, fut: F) -> F::Output {
    match project_id.filter(|p| !p.is_empty()) {
        None => fut.await,
        Some(project_id) => TASK_ACTIVE_PROJECT_ID.scope(RefCell::new(Some(project_id)), fut).await,
    }
}

/// Resolve as if no project were named, for the lifetime of the guard.
///
/// [`use_project`] deliberately treats a falsy id as a no-op — a caller that
/// named no project should keep falling through rather than pin to nothing —
/// so this is the explicit form for the one case that means it: work that has
/// just removed the project it is bound to.
///
/// Deleting a project arrives on a path that names it, so the request stays
/// bound to it after the removal.  Anything resolving that binding then warns
/// about an absence the request was asked to create, and [`Workspace::save`]
/// does resolve it.  The warning is the same sentence a genuinely stale
/// binding produces, which is reason enough not to emit it where nothing is
/// wrong.
///
/// The process pin is left alone: a pinned child never reaches an operation
/// like this, because removing a project is the workspace owner's work.
pub fn unbind_project() -> ProjectBinding {
    ProjectBinding::enter(BindingKind::Active, None)
}

/// Async form of [`unbind_project`].
pub async fn unbind_project_scope<F: Future>(fut: F) -> F::Output {
    TASK_ACTIVE_PROJECT_ID.scope(RefCell::new(None), fut).await
}

/// Token returned by [`set_active_project`]; holds the previous binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveProjectToken {
    previous: Option<String>,
    store: Store,
}

/// Bind `project_id` and return the token to reset with.
///
/// The imperative form of [`use_project`], for the streaming chat path, where
/// the binding has to outlive the function that establishes it (the stream is
/// resumed by the runtime long after the setup call returns).
pub fn set_active_project(project_id: Option<&str>) -> ActiveProjectToken {
    let (previous, store) = swap_binding(BindingKind::Active, project_id.map(str::to_string));
    ActiveProjectToken { previous, store }
}

/// Undo a [`set_active_project`], tolerating an already-unwound / absent token.
///
/// A streaming response can be abandoned by the client mid-generation, which
/// unwinds the stream from a different task than the one that set the
/// binding; a reset from the wrong context is ignored rather than corrupting
/// anything, and there is nothing to repair when it happens.
pub fn reset_active_project(token: Option<ActiveProjectToken>) {
    let Some(token) = token else {
        return;
    };
    restore_binding(BindingKind::Active, token.store, token.previous);
}

/// Wrap `f` so it runs under the caller's project binding.
///
/// Capture happens now, on the submitting thread; the binding is applied
/// inside the worker (even when it is `None`, so the worker resolves exactly
/// as the submitter did).  Use for any hop into a thread that will resolve a
/// project — `spawn_blocking`, `std::thread::spawn`, a pool `submit`.
pub fn bind<F, R>(f: F) -> impl FnOnce() -> R + Send + 'static
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let project_id = active_project_id();
    move || {
        let _guard = ProjectBinding::enter(BindingKind::Active, project_id);
        f()
    }
}

type Job = Box<dyn FnOnce() + Send + 'static>;

/// A thread pool that carries the submitter's project binding into workers.
///
/// Drop-in for Python's `ThreadPoolExecutor` anywhere the submitted work
/// resolves a project — the orchestrator's DAG and I/O pools, the MCP
/// explorer's fan-out.  Without it those workers would read whichever project
/// happened to be bound to the pool thread from an earlier request.
///
/// Like `ThreadPoolExecutor`, worker threads start lazily — one per
/// submission while no worker is idle, up to `max_workers`.
pub struct ProjectBoundExecutor {
    sender: Mutex<Option<mpsc::Sender<Job>>>,
    workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
    max_workers: usize,
    receiver: Arc<Mutex<mpsc::Receiver<Job>>>,
    /// Workers waiting for a job (Python's `_idle_semaphore`).
    idle: Arc<AtomicUsize>,
    thread_name_prefix: String,
}

/// Handle to a job submitted to a [`ProjectBoundExecutor`] (a
/// `concurrent.futures.Future`).
pub struct PoolFuture<R> {
    receiver: mpsc::Receiver<std::thread::Result<R>>,
}

/// Payload reported for a job that never ran because the pool was shut down.
const SHUTDOWN_MESSAGE: &str = "cannot schedule new futures after shutdown";

impl<R> PoolFuture<R> {
    /// Block until the job finishes and return its result; `Err` carries the
    /// panic payload when the job panicked (or a message when it never ran
    /// because the pool was shut down).
    pub fn result(self) -> std::thread::Result<R> {
        match self.receiver.recv() {
            Ok(result) => result,
            Err(_) => Err(Box::new(SHUTDOWN_MESSAGE.to_string())),
        }
    }

    /// [`PoolFuture::result`] with a timeout (`future.result(timeout=...)`);
    /// `None` when the job has not finished in time.
    pub fn result_timeout(&self, timeout: std::time::Duration) -> Option<std::thread::Result<R>> {
        match self.receiver.recv_timeout(timeout) {
            Ok(result) => Some(result),
            Err(mpsc::RecvTimeoutError::Timeout) => None,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Some(Err(Box::new(SHUTDOWN_MESSAGE.to_string())))
            }
        }
    }
}

/// Distinguishes the default thread names of separate pools.
static POOL_COUNTER: AtomicUsize = AtomicUsize::new(0);

impl ProjectBoundExecutor {
    /// Create a pool of up to `max_workers` threads named `<prefix>_<n>`.
    pub fn new(max_workers: usize, thread_name_prefix: Option<&str>) -> Self {
        let (sender, receiver) = mpsc::channel::<Job>();
        let thread_name_prefix = match thread_name_prefix {
            Some(prefix) if !prefix.is_empty() => prefix.to_string(),
            _ => format!("ThreadPoolExecutor-{}", POOL_COUNTER.fetch_add(1, Ordering::Relaxed)),
        };
        Self {
            sender: Mutex::new(Some(sender)),
            workers: Mutex::new(Vec::new()),
            max_workers: max_workers.max(1),
            receiver: Arc::new(Mutex::new(receiver)),
            idle: Arc::new(AtomicUsize::new(0)),
            thread_name_prefix,
        }
    }

    /// The configured worker count.
    pub fn max_workers(&self) -> usize {
        self.max_workers
    }

    /// Submit `f`, bound to the submitter's project (see [`bind`]).
    pub fn submit<F, R>(&self, f: F) -> PoolFuture<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (result_tx, result_rx) = mpsc::channel();
        let bound = bind(f);
        let job: Job = Box::new(move || {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(bound));
            let _ = result_tx.send(outcome);
        });

        let sender = self.sender.lock();
        // After shutdown the job is dropped unrun, and its future reports it.
        if let Some(sender) = sender.as_ref()
            && sender.send(job).is_ok()
        {
            self.adjust_thread_count();
        }
        PoolFuture { receiver: result_rx }
    }

    /// Start another worker when none is idle and the pool is below its cap.
    fn adjust_thread_count(&self) {
        // An idle worker will pick the job up: claim it and stop there.
        if self
            .idle
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1))
            .is_ok()
        {
            return;
        }
        let mut workers = self.workers.lock();
        if workers.len() >= self.max_workers {
            return;
        }
        let receiver = Arc::clone(&self.receiver);
        let idle = Arc::clone(&self.idle);
        let name = format!("{}_{}", self.thread_name_prefix, workers.len());
        let spawned = std::thread::Builder::new().name(name).spawn(move || {
            loop {
                let job = {
                    let receiver = receiver.lock();
                    receiver.recv()
                };
                match job {
                    Ok(job) => {
                        job();
                        idle.fetch_add(1, Ordering::AcqRel);
                    }
                    Err(_) => break,
                }
            }
        });
        match spawned {
            Ok(handle) => workers.push(handle),
            Err(err) => {
                tracing::warn!(target: LOG_TARGET, "could not start a pool worker: {}", err);
            }
        }
    }

    /// Stop accepting work; when `wait`, join the worker threads.  Jobs
    /// already queued still run first, as with `shutdown(wait=True)`.
    pub fn shutdown(&self, wait: bool) {
        self.sender.lock().take();
        if !wait {
            return;
        }
        let handles: Vec<_> = std::mem::take(&mut *self.workers.lock());
        let me = std::thread::current().id();
        for handle in handles {
            // A job dropping its own pool must not wait on itself.
            if handle.thread().id() == me {
                continue;
            }
            let _ = handle.join();
        }
    }
}

impl Drop for ProjectBoundExecutor {
    fn drop(&mut self) {
        self.shutdown(true);
    }
}

/// An explicit `(workspace, project)` pair, resolved once.
///
/// Passed where a callee needs the project as a value rather than as ambient
/// state — a background task owns one for its whole run, and a per-project
/// child builds one at startup.  Resolution is late (`project` is looked up on
/// each access) so a reload of the workspace is picked up rather than pinning
/// a stale object.
#[derive(Clone)]
pub struct ProjectContext {
    pub workspace: Arc<RwLock<Workspace>>,
    pub project_id: String,
}

impl ProjectContext {
    /// Pair `workspace` with `project_id`.
    pub fn new(workspace: Arc<RwLock<Workspace>>, project_id: impl Into<String>) -> Self {
        Self { workspace, project_id: project_id.into() }
    }

    /// Build a context from whatever is bound, or `None` if nothing is.
    pub fn resolve(workspace: Option<Arc<RwLock<Workspace>>>) -> Option<Self> {
        let project_id = active_project_id().filter(|p| !p.is_empty())?;
        let workspace = workspace?;
        Some(Self::new(workspace, project_id))
    }

    /// A snapshot of the project this context names, if the workspace still
    /// holds it.
    pub fn project(&self) -> Option<Project> {
        self.workspace.read().project(&self.project_id).cloned()
    }

    /// The project's `project_dir`, if the workspace still holds it.
    pub fn project_dir(&self) -> Option<String> {
        self.workspace.read().project(&self.project_id).map(|p| p.project_dir.clone())
    }

    /// Make this context the ambient binding for the lifetime of the guard.
    pub fn bound(&self) -> ProjectBinding {
        use_project(Some(&self.project_id))
    }

    /// Async form of [`ProjectContext::bound`].
    pub async fn bound_scope<F: Future>(&self, fut: F) -> F::Output {
        scope_project(Some(self.project_id.clone()), fut).await
    }
}

impl std::fmt::Debug for ProjectContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ProjectContext(project_id={:?})", self.project_id)
    }
}
