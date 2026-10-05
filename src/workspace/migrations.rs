//! Port of `smartloop/workspace/migrations.py`.
//!
//! One-time on-disk layout migrations, run at server startup.
//!
//! Two of them, both moving state to where the process that owns it can reach
//! it: [`migrate_project_state`] relocates skills and documents off a
//! client-supplied path, and [`split_project_state`] breaks the single
//! `config.json` into a workspace record plus one file per project.
//!
//! Skills and documents used to live under a *user-space* folder —
//! `project.cwd`, defaulting to `~/smartloop/<name>`.  That path is
//! client-supplied and absolute, so it does not follow the framework's home:
//!
//!   * under the macOS App Sandbox `~` is redirected to the app container, but
//!     an absolute path persisted before sandboxing (or written by the
//!     un-sandboxed CLI) keeps pointing at the real home the sandbox denies —
//!     the app then reads one workspace and writes another;
//!   * `SLP_HOME` moved the framework's state but left skills and documents
//!     behind.
//!
//! They now live under `project_dir` (`<home_dir>/<workspace_id>/<project_id>`),
//! which is derived from `AppSettings.home_dir` and therefore always travels
//! with the framework.  This module moves anything left in the old location.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::paths::{confined_home, expand_home, real_home};
use crate::workspace::workspace::{Project, Workspace, join, legacy_layout, project_state_path};

/// Logger name used by the Python module.
const LOG_TARGET: &str = "smartloop.workspace.migrations";

/// Copy of the single-document `config.json` taken before it is first split.
/// The rollback path: restore this over `config.json` and set
/// `SLP_LEGACY_CONFIG=1`, and the workspace is exactly as it was.
pub const MONOLITH_BACKUP_SUFFIX: &str = ".pre-split.bak";

/// Subdirectories that moved from the user-space folder into `project_dir`.
const MIGRATED_SUBDIRS: [&str; 2] = ["skills", "documents"];

/// Dropped in the old home once its contents have been brought forward, so a
/// second start does not walk the tree again.  Names the destination, which
/// is the only thing anyone reading it afterwards wants to know.
pub const ADOPTED_MARKER: &str = ".adopted-into-real-home";

/// `{project_id: {subdir: count}}` of entries moved by [`migrate_project_state`].
pub type MigratedCounts = BTreeMap<String, BTreeMap<String, usize>>;

/// Result of [`split_project_state`]: `{"migrated": [...], "backup": ...}`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct SplitResult {
    /// Ids of the projects whose own state file was written.
    pub migrated: Vec<String>,
    /// Path of the pre-split `config.json` backup, if one exists.
    pub backup: Option<String>,
}

/// `os.path.normpath`: collapse redundant separators and `.`/`..` lexically.
fn normpath(path: &str) -> String {
    if path.is_empty() {
        return ".".to_string();
    }
    let mut parts: Vec<String> = Vec::new();
    let mut root = PathBuf::new();
    for component in Path::new(path).components() {
        match component {
            Component::Prefix(_) | Component::RootDir => root.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if parts.last().is_some_and(|p| p != "..") {
                    parts.pop();
                } else if root.as_os_str().is_empty() {
                    parts.push("..".to_string());
                }
            }
            Component::Normal(name) => parts.push(name.to_string_lossy().into_owned()),
        }
    }
    let mut out = root;
    for part in parts {
        out.push(part);
    }
    let out = out.to_string_lossy().into_owned();
    if out.is_empty() { ".".to_string() } else { out }
}

/// Candidate old locations for `project`'s state, most likely first.
///
/// `cwd` is where current builds put it.  `~/smartloop/<name>` covers projects
/// created before `cwd` was persisted unconditionally, whose location was
/// derived from the name at read time.
fn legacy_roots(project: &Project) -> Vec<String> {
    let mut roots: Vec<String> = Vec::new();
    if let Some(cwd) = project.cwd.as_deref().filter(|c| !c.is_empty()) {
        roots.push(expand_home(cwd));
    }
    if let Some(name) = project.name.as_deref().filter(|n| !n.is_empty()) {
        roots.push(join(&join(&real_home(), "smartloop"), name));
    }
    // De-duplicate while preserving order.
    let mut unique: Vec<String> = Vec::new();
    for root in roots {
        if !unique.contains(&root) {
            unique.push(root);
        }
    }
    unique
}

/// True when `path` is a directory this process may actually read.
///
/// Under the App Sandbox a denied path can still pass `isdir` and then fail on
/// access, so the readability check is not redundant.
fn is_readable_dir(path: &str) -> bool {
    if !Path::new(path).is_dir() {
        return false;
    }
    access_rx(path)
}

/// `os.access(path, os.R_OK | os.X_OK)`.
#[cfg(unix)]
fn access_rx(path: &str) -> bool {
    let Ok(c_path) = std::ffi::CString::new(path) else {
        return false;
    };
    // SAFETY: `c_path` is a valid NUL-terminated string for the call's duration.
    unsafe { libc::access(c_path.as_ptr(), libc::R_OK | libc::X_OK) == 0 }
}

/// `os.access(path, os.R_OK | os.X_OK)`: on Windows `os.access` only reports
/// existence for directories, so an actual listing is the honest check.
#[cfg(not(unix))]
fn access_rx(path: &str) -> bool {
    std::fs::read_dir(path).is_ok()
}

/// `shutil.copystat`: carry permissions and access/modification times over.
fn copystat(src: &Path, dest: &Path) -> std::io::Result<()> {
    let meta = std::fs::metadata(src)?;
    std::fs::set_permissions(dest, meta.permissions())?;
    let mut times = std::fs::FileTimes::new();
    if let Ok(accessed) = meta.accessed() {
        times = times.set_accessed(accessed);
    }
    if let Ok(modified) = meta.modified() {
        times = times.set_modified(modified);
    }
    // Directories can be opened read-only to set their times on unix.
    let handle = std::fs::File::open(dest)?;
    handle.set_times(times)
}

/// `shutil.copy2`: copy the file's contents and permission bits, then its
/// timestamps.
fn copy2(src: &Path, dest: &Path) -> std::io::Result<()> {
    std::fs::copy(src, dest)?;
    let meta = std::fs::metadata(src)?;
    let mut times = std::fs::FileTimes::new();
    if let Ok(accessed) = meta.accessed() {
        times = times.set_accessed(accessed);
    }
    if let Ok(modified) = meta.modified() {
        times = times.set_modified(modified);
    }
    let handle = std::fs::OpenOptions::new().write(true).open(dest)?;
    handle.set_times(times)
}

/// `shutil.copytree(src, dest)` (symlinks followed, `copy2` per file): the
/// destination must not exist yet.  Copies every entry it can and reports the
/// first failure at the end, as `shutil.Error` does.
fn copytree(src: &Path, dest: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dest)?;
    let mut first_error: Option<std::io::Error> = None;
    for entry in std::fs::read_dir(src)? {
        let result = entry.and_then(|entry| {
            let from = entry.path();
            let to = dest.join(entry.file_name());
            if from.is_dir() { copytree(&from, &to) } else { copy2(&from, &to) }
        });
        if let Err(err) = result
            && first_error.is_none()
        {
            first_error = Some(err);
        }
    }
    if let Err(err) = copystat(src, dest)
        && first_error.is_none()
    {
        first_error = Some(err);
    }
    match first_error {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

/// `os.listdir` sorted by name; `Err` when the directory cannot be read.
fn sorted_entries(path: &str) -> std::io::Result<Vec<String>> {
    let mut names: Vec<String> = std::fs::read_dir(path)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    Ok(names)
}

/// Move the contents of `src` into `dest`. Returns the number of entries moved.
///
/// Entries already present at the destination win — a partially-completed
/// earlier run must never clobber newer state — and the source is left in
/// place rather than deleted, so a bad migration is always recoverable by hand.
fn migrate_subdir(src: &str, dest: &str) -> usize {
    let mut moved = 0;
    if let Err(exc) = std::fs::create_dir_all(dest) {
        tracing::warn!(target: LOG_TARGET, "[migrate] Could not move {} → {}: {}", src, dest, exc);
        return 0;
    }
    let entries = match sorted_entries(src) {
        Ok(entries) => entries,
        Err(exc) => {
            tracing::warn!(target: LOG_TARGET, "[migrate] Could not move {} → {}: {}", src, dest, exc);
            return 0;
        }
    };
    for entry in entries {
        let src_path = join(src, &entry);
        let dest_path = join(dest, &entry);
        if Path::new(&dest_path).exists() {
            continue;
        }
        let result = if Path::new(&src_path).is_dir() {
            copytree(Path::new(&src_path), Path::new(&dest_path))
        } else {
            copy2(Path::new(&src_path), Path::new(&dest_path))
        };
        match result {
            Ok(()) => moved += 1,
            Err(exc) => {
                tracing::warn!(
                    target: LOG_TARGET,
                    "[migrate] Could not move {} → {}: {}",
                    src_path,
                    dest_path,
                    exc
                );
            }
        }
    }
    moved
}

/// Move skills/documents from the legacy user-space folder into `project_dir`.
///
/// Idempotent and best-effort: every filesystem touch is guarded, because the
/// legacy path may be unreadable (sandbox) or gone entirely, and neither is a
/// reason to fail startup.  Returns `{project_id: {subdir: count}}` for the
/// entries actually moved.
pub fn migrate_project_state(workspace: Option<&Workspace>) -> MigratedCounts {
    let mut migrated = MigratedCounts::new();
    let Some(workspace) = workspace else {
        return migrated;
    };

    for project in &workspace.projects {
        if project.project_dir.is_empty() {
            continue;
        }
        for root in legacy_roots(project) {
            // Never migrate a folder onto itself.
            if normpath(&root) == normpath(&project.project_dir) {
                continue;
            }
            if !is_readable_dir(&root) {
                continue;
            }
            for sub in MIGRATED_SUBDIRS {
                let src = join(&root, sub);
                if !is_readable_dir(&src) {
                    continue;
                }
                let dest = join(&project.project_dir, sub);
                let count = migrate_subdir(&src, &dest);
                if count > 0 {
                    migrated.entry(project.id.clone()).or_default().insert(sub.to_string(), count);
                    tracing::info!(
                        target: LOG_TARGET,
                        "[migrate] {}: moved {} {} entr{} from {} → {}",
                        project.name.as_deref().filter(|n| !n.is_empty()).unwrap_or(&project.id),
                        count,
                        sub,
                        if count == 1 { "y" } else { "ies" },
                        src,
                        dest
                    );
                }
            }
        }
    }
    migrated
}

// ---------------------------------------------------------------------------
// a sandbox container's framework home → the real ~/.smartloop
// ---------------------------------------------------------------------------
//
// The framework home is `~/.smartloop` in every build, so one install serves
// the app and the CLI alike.  Builds that shipped before the home was resolved
// past the sandbox put it under the rewritten `$HOME` instead — inside the App
// Sandbox container, or under `~/snap/smartloop/<revision>` — and those
// installs still have their workspace, projects and model weights there.  This
// brings them forward.

/// `shutil.move`: a rename, falling back to copy-and-delete across devices.
fn shutil_move(src: &str, dest: &str) -> std::io::Result<()> {
    match std::fs::rename(src, dest) {
        Ok(()) => Ok(()),
        Err(err) if err.raw_os_error() == Some(libc::EXDEV) => {
            let src_path = Path::new(src);
            let is_link = std::fs::symlink_metadata(src_path).map(|m| m.file_type().is_symlink()).unwrap_or(false);
            if src_path.is_dir() && !is_link {
                copytree(src_path, Path::new(dest))?;
                std::fs::remove_dir_all(src_path)
            } else {
                copy2(src_path, Path::new(dest))?;
                std::fs::remove_file(src_path)
            }
        }
        Err(err) => Err(err),
    }
}

/// Move one top-level entry of the old home across.  True when it lands.
///
/// A rename, not a copy: the model weights under `<workspace>/models` run to
/// gigabytes, and both sandboxes put their rewritten home on the same volume
/// as the real one, so this costs nothing and needs no free space.  Done per
/// entry rather than per tree so that one unreadable directory leaves the rest
/// of the migration intact — and leaves itself where it was.
fn move_entry(src: &str, dest: &str) -> bool {
    if Path::new(dest).exists() {
        // Already brought forward (or created afresh by a newer build); the
        // copy that is already at the real home is the one in use.
        return false;
    }
    match shutil_move(src, dest) {
        Ok(()) => true,
        Err(err) => {
            tracing::warn!(target: LOG_TARGET, "[adopt] could not move {} → {}: {}", src, dest, err);
            false
        }
    }
}

/// Bring a sandbox-located framework home forward to `dest`.
///
/// `dest` is the real `~/.smartloop`.  Returns the old home (its `.smartloop`
/// directory) when something was actually moved out of it, else `None`.
///
/// Declines rather than guesses in the two cases where it cannot know what the
/// user wants:
///
/// * `SLP_HOME` is set — somebody pinned the home deliberately (a service
///   unit, a test, a packaged build that means to stay in its container) and
///   that instruction outranks this migration;
/// * `dest` already holds a `config.json` — there are two live workspaces,
///   and silently merging them would be a guess about which project list is
///   the real one.  The user keeps both and is told.
///
/// Best-effort throughout: under the App Sandbox the real home is readable
/// only with the entitlement to match, and a denial must leave the install
/// running out of its container rather than fail.
pub fn adopt_confined_home(dest: &str) -> Option<String> {
    if std::env::var("SLP_HOME").is_ok_and(|v| !v.is_empty()) {
        return None;
    }

    let old_home = confined_home().filter(|h| !h.is_empty())?;

    let src = join(&old_home, ".smartloop");
    if normpath(&src) == normpath(dest) {
        return None;
    }
    if !is_readable_dir(&src) {
        return None;
    }
    if Path::new(&join(&src, ADOPTED_MARKER)).exists() {
        return None;
    }

    if Path::new(&join(dest, "config.json")).exists() {
        tracing::warn!(
            target: LOG_TARGET,
            "[adopt] {} already holds a workspace; leaving the sandboxed one at {} alone. Point SLP_HOME at it to use it instead.",
            dest,
            src
        );
        return None;
    }

    let entries = match sorted_entries(&src) {
        Ok(entries) => entries,
        Err(err) => {
            tracing::warn!(target: LOG_TARGET, "[adopt] could not read {}: {}", src, err);
            return None;
        }
    };

    if let Err(err) = std::fs::create_dir_all(dest) {
        // The sandbox denying the real home is the expected failure here, and
        // it is not fatal: the old home is untouched and still serves.
        tracing::warn!(target: LOG_TARGET, "[adopt] could not create {}: {}", dest, err);
        return None;
    }

    let mut moved = 0;
    for name in entries {
        if name == ADOPTED_MARKER {
            continue;
        }
        if move_entry(&join(&src, &name), &join(dest, &name)) {
            moved += 1;
        }
    }

    if moved == 0 {
        return None;
    }

    tracing::info!(
        target: LOG_TARGET,
        "[adopt] moved {} entr{} from {} → {}",
        moved,
        if moved == 1 { "y" } else { "ies" },
        src,
        dest
    );
    if let Err(err) = std::fs::write(join(&src, ADOPTED_MARKER), format!("{dest}\n")) {
        // Only a fast path for the next start; the emptied directory and the
        // config.json now at `dest` both already stop a second run.
        tracing::debug!(target: LOG_TARGET, "[adopt] could not write the marker in {}: {}", src, err);
    }
    Some(src)
}

// ---------------------------------------------------------------------------
// config.json → workspace record + one project.json per project
// ---------------------------------------------------------------------------

/// True when at least one project has no state file of its own yet.
pub fn needs_split(workspace: &Workspace) -> bool {
    workspace
        .projects
        .iter()
        .filter(|project| !project.project_dir.is_empty())
        .any(|project| !Path::new(&project_state_path(&project.project_dir)).is_file())
}

/// Copy `config.json` aside before it is rewritten, once.
///
/// An existing backup is never overwritten: it is the pre-migration state, and
/// a second (already-split) copy on top of it would destroy the only thing a
/// rollback has to restore from.
fn backup_monolith(config_file: &str) -> Option<String> {
    if !Path::new(config_file).is_file() {
        return None;
    }
    let backup = format!("{config_file}{MONOLITH_BACKUP_SUFFIX}");
    if Path::new(&backup).exists() {
        return Some(backup);
    }
    if let Err(exc) = copy2(Path::new(config_file), Path::new(&backup)) {
        tracing::warn!(target: LOG_TARGET, "[migrate] Could not back up {}: {}", config_file, exc);
        return None;
    }
    tracing::info!(target: LOG_TARGET, "[migrate] Backed up the pre-split config to {}", backup);
    Some(backup)
}

/// Move each project's own state out of `config.json` into its directory.
///
/// A workspace used to be one document holding every project.  That made a
/// save a whole-workspace snapshot, so two writers each holding a
/// [`Workspace`] would have the last one silently discard the other project's
/// newly indexed documents and MCP registrations.
///
/// Afterwards `config.json` keeps only workspace state (id, models,
/// models_dir) plus a thin project index, and each `<project_dir>/project.json`
/// holds that project's own record.
///
/// Idempotent, and safe to interrupt: the pre-split `config.json` is copied
/// aside first, every project file is written before the workspace record is
/// rewritten, and a project that already has a file is left alone.
///
/// To roll back, restore the backup over `config.json` and set
/// `SLP_LEGACY_CONFIG=1`; the `project.json` files are then ignored rather
/// than having to be removed.
pub fn split_project_state(workspace: Option<&mut Workspace>) -> SplitResult {
    let mut result = SplitResult::default();
    let Some(workspace) = workspace else {
        return result;
    };
    if legacy_layout() || !needs_split(workspace) {
        return result;
    }

    result.backup = backup_monolith(&Workspace::config_file());

    for project in &workspace.projects {
        if project.project_dir.is_empty() {
            continue;
        }
        let written = std::fs::create_dir_all(&project.project_dir).and_then(|()| project.save_state());
        if let Err(exc) = written {
            tracing::warn!(
                target: LOG_TARGET,
                "[migrate] Could not write state for project {}: {}",
                project.id,
                exc
            );
            continue;
        }
        result.migrated.push(project.id.clone());
    }

    // Only now is the monolith replaced by the thin record — every project's
    // state is already on disk, so an interruption before this point leaves
    // the old file intact and the next run simply redoes the write.
    if let Err(exc) = workspace.save() {
        tracing::warn!(target: LOG_TARGET, "[migrate] Could not rewrite the workspace record: {}", exc);
    }

    tracing::info!(
        target: LOG_TARGET,
        "[migrate] Split config.json — {} project(s) now hold their own state",
        result.migrated.len()
    );
    result
}
