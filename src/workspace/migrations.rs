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
//! `project.cwd`, defaulting to `~/smartloop/<name>`.  They now live under
//! `project_dir` (`<home_dir>/<workspace_id>/<project_id>`), which is derived
//! from `AppSettings.home_dir` and therefore always travels with the
//! framework.  This module moves anything left in the old location.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::paths::{confined_home, expand_home, real_home};
use crate::workspace::workspace::{Project, Workspace, legacy_layout, project_state_path};

/// Logger name used by the Python module.
const LOG_TARGET: &str = "smartloop.workspace.migrations";

/// Copy of the single-document `config.json` taken before it is first split.
/// The rollback path: restore this over `config.json` and set
/// `SLP_LEGACY_CONFIG=1`, and the workspace is exactly as it was.
pub const MONOLITH_BACKUP_SUFFIX: &str = ".pre-split.bak";

/// Subdirectories that moved from the user-space folder into `project_dir`.
const MIGRATED_SUBDIRS: [&str; 2] = ["skills", "documents"];

/// Dropped in the old home once its contents have been brought forward, so a
/// second start does not walk the tree again.  Names the destination.
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

/// Candidate old locations for `project`'s state, most likely first.
///
/// `cwd` is where current builds put it.  `~/smartloop/<name>` covers projects
/// created before `cwd` was persisted unconditionally.  De-duplicated, order
/// preserved.
fn legacy_roots(project: &Project) -> Vec<String> {
    todo!()
}

/// True when `path` is a directory this process may actually read.
///
/// Under the App Sandbox a denied path can still pass `isdir` and then fail on
/// access, so the readability check is not redundant.
fn is_readable_dir(path: &str) -> bool {
    todo!()
}

/// Move the contents of `src` into `dest`. Returns the number of entries moved.
///
/// Entries already present at the destination win, and the source is left in
/// place rather than deleted, so a bad migration is always recoverable by hand.
fn migrate_subdir(src: &str, dest: &str) -> usize {
    todo!()
}

/// Move skills/documents from the legacy user-space folder into `project_dir`.
///
/// Idempotent and best-effort: every filesystem touch is guarded.  Returns
/// `{project_id: {subdir: count}}` for the entries actually moved.
pub fn migrate_project_state(workspace: Option<&Workspace>) -> MigratedCounts {
    todo!()
}

/// Move one top-level entry of the old home across.  True when it lands.
///
/// A rename, not a copy; an existing destination is left alone.
fn move_entry(src: &str, dest: &str) -> bool {
    todo!()
}

/// Bring a sandbox-located framework home forward to `dest`.
///
/// `dest` is the real `~/.smartloop`.  Returns the old home (its `.smartloop`
/// directory) when something was actually moved out of it, else `None`.
///
/// Declines when `SLP_HOME` is set, or when `dest` already holds a
/// `config.json`.  Best-effort throughout.
pub fn adopt_confined_home(dest: &str) -> Option<String> {
    todo!()
}

/// True when at least one project has no state file of its own yet.
fn needs_split(workspace: &Workspace) -> bool {
    todo!()
}

/// Copy `config.json` aside before it is rewritten, once.
///
/// An existing backup is never overwritten.
fn backup_monolith(config_file: &str) -> Option<String> {
    todo!()
}

/// Move each project's own state out of `config.json` into its directory.
///
/// Afterwards `config.json` keeps only workspace state plus a thin project
/// index, and each `<project_dir>/project.json` holds that project's record.
/// Idempotent and safe to interrupt.  Returns the migrated project ids and the
/// backup path.
///
/// To roll back, restore the backup over `config.json` and set
/// `SLP_LEGACY_CONFIG=1`.
pub fn split_project_state(workspace: Option<&mut Workspace>) -> SplitResult {
    todo!()
}
