//! Port of `smartloop/api/services/project_io.py`.
//!
//! Project export/import service.
//!
//! Packages a project into a portable zip archive and restores it on another
//! machine (or the same one).

#![allow(unused_imports)]

use std::path::{Path, PathBuf};

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde_json::{Map, Value};

use crate::config::app_settings::AppSettings;
use crate::workspace::context::own_project;
use crate::workspace::workspace::{Document, MCPServer, Project, Skill, Workspace};

/// Logger name used by the Python module.
#[allow(dead_code)]
const LOG_TARGET: &str = "smartloop.api.services.project_io";

/// Errors raised by export/import, mirroring the Python exception classes the
/// router maps onto status codes (`ValueError` -> 400, `FileNotFoundError` ->
/// 404, anything else -> 500).
#[derive(Debug, thiserror::Error)]
pub enum ProjectIoError {
    /// Python `ValueError`: bad input (invalid zip, missing manifest, no
    /// workspace).
    #[error("{0}")]
    ValueError(String),
    /// Python `FileNotFoundError`: the project does not exist.
    #[error("{0}")]
    FileNotFoundError(String),
    /// Any other failure (I/O, zip, JSON).
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<std::io::Error> for ProjectIoError {
    fn from(e: std::io::Error) -> Self {
        ProjectIoError::Other(e.into())
    }
}

/// Module-level settings (Python `_settings`); tests override it.
static SETTINGS: Lazy<RwLock<Option<AppSettings>>> = Lazy::new(|| RwLock::new(None));

/// Override the module settings (Python tests patch `project_io._settings`).
/// `None` restores `AppSettings::new()`.
pub fn set_settings(settings: Option<AppSettings>) {
    *SETTINGS.write() = settings;
}

/// The SLP home directory from the module settings.
fn home_dir() -> String {
    todo!()
}

/// Return `<home_dir>/conversations/`.
fn conversations_dir() -> String {
    todo!()
}

/// Recursively add `src_dir` contents under `arc_prefix` in `zf`.
fn add_directory_to_zip<W: std::io::Write + std::io::Seek>(
    zf: &mut zip::ZipWriter<W>,
    src_dir: &str,
    arc_prefix: &str,
) -> anyhow::Result<()> {
    let _ = (zf, src_dir, arc_prefix);
    todo!()
}

/// Remove OAuth credentials/tokens from MCP servers in project data.
pub fn strip_oauth(project_data: Map<String, Value>) -> Map<String, Value> {
    let _ = project_data;
    todo!()
}

/// Parse `SKILL.md` files from a skills directory into `Skill` objects.
///
/// Each skill lives in its own subdirectory as a `SKILL.md` file with YAML
/// frontmatter (name, description, tools).
pub fn read_skills_from_dir(skills_dir: &str) -> Vec<Skill> {
    let _ = skills_dir;
    todo!()
}

/// Create a zip archive for the given project.
///
/// Returns the path to the temporary zip file.  The caller is responsible for
/// cleaning it up after the response has been sent.
pub fn export_project(project_id: &str) -> Result<PathBuf, ProjectIoError> {
    let _ = project_id;
    todo!()
}

/// Import a project from a zip archive.
///
/// * `zip_path` — path to the zip file.
/// * `name` — optional new name for the imported project.
/// * `workspace` — the instance to import into.  A caller that holds a live
///   workspace must pass it; importing into a detached `Workspace::load()`
///   leaves that caller's own instance without the project, and its next save
///   drops the imported project's `config.json` entry.  Defaults to a fresh
///   load for the CLI.
///
/// Returns a JSON object with the new project metadata (`id`, `name`, `cwd`,
/// `description`, `current`, `system`, `skills`, `mcp`, `documents`).
pub fn import_project(
    zip_path: &Path,
    name: Option<&str>,
    workspace: Option<&mut Workspace>,
) -> Result<Map<String, Value>, ProjectIoError> {
    let _ = (zip_path, name, workspace);
    todo!()
}
