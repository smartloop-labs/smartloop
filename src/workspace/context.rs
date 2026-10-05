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
//! context rather than the flag (see [`Workspace::current_project`]).
//!
//! Three layers resolve, in order:
//!
//! 1. the active-project binding — the request's own binding.  In async code it
//!    is a tokio task-local (authoritative while in scope); otherwise a
//!    thread-local set by the RAII guards;
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
//! A binding does not cross a thread boundary on its own.  [`bind`] captures
//! the binding at the submitting side and re-establishes it in the worker, and
//! [`ProjectBoundExecutor`] applies it to every `submit`.

use std::cell::RefCell;
use std::future::Future;
use std::sync::Arc;
use std::sync::mpsc;

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
    /// [`unbind_project_scope`]).
    static TASK_ACTIVE_PROJECT_ID: Option<String>;

    /// Async-scope counterpart of the owned-project claim.
    static TASK_OWNED_PROJECT_ID: Option<String>;
}

/// Process-wide pin, set once at startup by a per-project agent child.  Not a
/// scoped binding: it is a property of the process, not of a request, and it
/// must survive threads that were never bound.
static PINNED_PROJECT_ID: RwLock<Option<String>> = RwLock::new(None);

/// Whether this process has handed every project's own state to someone else.
static PROJECTS_DELEGATED: RwLock<bool> = RwLock::new(false);

/// Serializes writes to the process-wide pin / delegation flags.
static PIN_LOCK: Mutex<()> = Mutex::new(());

/// Fix this process to `project_id` for its whole lifetime.
///
/// Used by a per-project agent child, which is spawned for exactly one project
/// and has no reason to ever look at another.  The supervisor process leaves
/// this unset, so its own reads still resolve through the `current` flag.
pub fn pin_project(project_id: Option<&str>) {
    todo!()
}

/// The project this process is pinned to, or `None` when unpinned.
pub fn pinned_project_id() -> Option<String> {
    todo!()
}

/// Whether this process owns workspace-level state.
///
/// The positive form of `pinned_project_id().is_none()`.  True in the
/// supervisor, in a `--no-supervised` single server, and in the CLI; false in
/// a per-project agent child.
pub fn owns_workspace() -> bool {
    todo!()
}

/// Declare that per-project state is owned by other processes.
///
/// Called by the supervisor when its pool comes up, and cleared when it goes
/// down.  [`Workspace::save`] reads this to decide whether it may write every
/// project's file or only the workspace record — see [`own_project`] for the
/// exception.
pub fn delegate_projects(delegated: bool) {
    todo!()
}

/// Whether another process owns each project's own state.
pub fn projects_delegated() -> bool {
    todo!()
}

/// RAII guard restoring the previous thread-local binding on drop.
///
/// Returned by [`use_project`], [`unbind_project`], [`own_project`] and
/// [`ProjectContext::bound`].
#[must_use = "the binding is undone when the guard is dropped"]
pub struct ProjectBinding {
    kind: BindingKind,
    previous: Option<Option<String>>,
    project_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BindingKind {
    Active,
    Owned,
}

impl ProjectBinding {
    /// The project id this guard bound (the value `with ... as pid` yielded in
    /// Python), or `None` for a no-op / unbinding guard.
    pub fn project_id(&self) -> Option<&str> {
        self.project_id.as_deref()
    }
}

impl Drop for ProjectBinding {
    fn drop(&mut self) {
        todo!()
    }
}

/// Claim `project_id`'s own state for the lifetime of the returned guard.
///
/// The supervisor delegates project files to its children, but three
/// operations happen either side of a child's existence and must still write
/// one: creating a project, importing one, and deleting one.  A falsy id is a
/// no-op claim.
pub fn own_project(project_id: Option<&str>) -> ProjectBinding {
    todo!()
}

/// Async form of [`own_project`]: run `fut` with `project_id` claimed.
pub async fn own_project_scope<F: Future>(project_id: Option<String>, fut: F) -> F::Output {
    todo!()
}

/// The project claimed by [`own_project`], if any.
pub fn owned_project_id() -> Option<String> {
    todo!()
}

/// The project bound to the current request or process, if any.
///
/// Returns `None` when nothing is bound — callers then fall back to the
/// `current` flag, which is what a supervisor process and the CLI want.
pub fn active_project_id() -> Option<String> {
    todo!()
}

/// Bind `project_id` for the lifetime of the returned guard (sync scope).
///
/// A falsy `project_id` is a no-op binding: the caller did not name a project,
/// so resolution should keep falling through rather than pin to nothing.
pub fn use_project(project_id: Option<&str>) -> ProjectBinding {
    todo!()
}

/// Async form of [`use_project`]: run `fut` with `project_id` bound as a
/// task-local.  A falsy id runs `fut` without changing the binding.
pub async fn scope_project<F: Future>(project_id: Option<String>, fut: F) -> F::Output {
    todo!()
}

/// Resolve as if no project were named, for the lifetime of the guard.
///
/// [`use_project`] deliberately treats a falsy id as a no-op, so this is the
/// explicit form for the one case that means it: work that has just removed
/// the project it is bound to.  The process pin is left alone.
pub fn unbind_project() -> ProjectBinding {
    todo!()
}

/// Async form of [`unbind_project`].
pub async fn unbind_project_scope<F: Future>(fut: F) -> F::Output {
    todo!()
}

/// Token returned by [`set_active_project`]; holds the previous binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveProjectToken {
    previous: Option<String>,
}

/// Bind `project_id` and return the token to reset with.
///
/// The imperative form of [`use_project`], for the streaming chat path, where
/// the binding has to outlive the function that establishes it.
pub fn set_active_project(project_id: Option<&str>) -> ActiveProjectToken {
    todo!()
}

/// Undo a [`set_active_project`], tolerating an already-unwound / absent token.
pub fn reset_active_project(token: Option<ActiveProjectToken>) {
    todo!()
}

/// Wrap `f` so it runs under the caller's project binding.
///
/// Capture happens now, on the submitting thread; the binding is applied
/// inside the worker.  Use for any hop into a thread that will resolve a
/// project — `spawn_blocking`, `std::thread::spawn`, a pool `submit`.
pub fn bind<F, R>(f: F) -> impl FnOnce() -> R + Send + 'static
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let project_id = active_project_id();
    move || {
        let _guard = use_project(project_id.as_deref());
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
pub struct ProjectBoundExecutor {
    sender: Mutex<Option<mpsc::Sender<Job>>>,
    workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
    max_workers: usize,
}

/// Handle to a job submitted to a [`ProjectBoundExecutor`] (a
/// `concurrent.futures.Future`).
pub struct PoolFuture<R> {
    receiver: mpsc::Receiver<std::thread::Result<R>>,
}

impl<R> PoolFuture<R> {
    /// Block until the job finishes and return its result; `Err` carries the
    /// panic payload when the job panicked.
    pub fn result(self) -> std::thread::Result<R> {
        todo!()
    }
}

impl ProjectBoundExecutor {
    /// Create a pool of `max_workers` threads named `<prefix>_<n>`.
    pub fn new(max_workers: usize, thread_name_prefix: Option<&str>) -> Self {
        todo!()
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
        todo!()
    }

    /// Stop accepting work; when `wait`, join the worker threads.
    pub fn shutdown(&self, wait: bool) {
        todo!()
    }
}

impl Drop for ProjectBoundExecutor {
    fn drop(&mut self) {
        todo!()
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
        todo!()
    }

    /// A snapshot of the project this context names, if the workspace still
    /// holds it.
    pub fn project(&self) -> Option<Project> {
        todo!()
    }

    /// The project's `project_dir`, if the workspace still holds it.
    pub fn project_dir(&self) -> Option<String> {
        todo!()
    }

    /// Make this context the ambient binding for the lifetime of the guard.
    pub fn bound(&self) -> ProjectBinding {
        todo!()
    }

    /// Async form of [`ProjectContext::bound`].
    pub async fn bound_scope<F: Future>(&self, fut: F) -> F::Output {
        todo!()
    }
}

impl std::fmt::Debug for ProjectContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ProjectContext(project_id={:?})", self.project_id)
    }
}
