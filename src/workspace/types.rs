//! Port of `smartloop/workspace/types.py`.
//!
//! Lightweight workspace types shared between the workspace and model layers.
//!
//! Kept in a separate module so the model layer can use these without pulling
//! in the full workspace module.

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::AppSettings;

/// Render `path` relative to the project, else to the framework home.
///
/// Tried in order:
///
/// * relative to `project_dir` — `documents/invoice.pdf`, the form that
///   survives the whole framework being copied to another machine, user, or
///   sandbox container;
/// * relative to `AppSettings.home_dir` — for documents outside the project
///   but still inside the framework;
/// * the bare filename — so a path from somewhere else entirely still tells
///   the client what the file is without disclosing where it lives.
///
/// Always returns POSIX separators so a path rendered on Windows and one
/// rendered on macOS compare equal.  An empty or missing `path` is returned
/// unchanged.
pub fn relative_document_path(path: Option<&str>, project_dir: Option<&str>) -> Option<String> {
    let path = match path {
        None => return None,
        Some("") => return Some(String::new()),
        Some(p) => p,
    };

    let rel = |base: Option<&str>| -> Option<String> {
        let base = base.filter(|b| !b.is_empty())?;
        let resolved = resolve_lenient(Path::new(path))?;
        let resolved_base = resolve_lenient(Path::new(base))?;
        let relative = resolved.strip_prefix(&resolved_base).ok()?;
        Some(as_posix(relative))
    };

    if let Some(resolved) = rel(project_dir) {
        return Some(resolved);
    }

    // `AppSettings()` is built afresh, as in Python, so a changed `SLP_HOME`
    // is honoured; any failure to build it just skips this step.
    let home = AppSettings::try_new().ok().map(|s| s.home_dir());
    if let Some(resolved) = rel(home.as_deref()) {
        return Some(resolved);
    }

    Some(path_name(path))
}

/// `pathlib.PurePath.as_posix()` for a relative path: components joined with
/// `/`, and `.` for the empty path.
fn as_posix(path: &Path) -> String {
    let parts: Vec<String> = path
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    if parts.is_empty() { ".".to_string() } else { parts.join("/") }
}

/// `pathlib.PurePath(path).name`: the final component, `""` when there is
/// none (the root, or an empty path).
pub(crate) fn path_name(path: &str) -> String {
    match Path::new(path).components().next_back() {
        Some(Component::Normal(name)) => name.to_string_lossy().into_owned(),
        Some(Component::ParentDir) => "..".to_string(),
        _ => String::new(),
    }
}

/// `pathlib.Path.resolve()` (non-strict): make `path` absolute, resolve
/// symlinks through the deepest part of it that exists, and normalise the
/// rest lexically.  `None` only when the working directory is unknowable.
fn resolve_lenient(path: &Path) -> Option<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(path)
    };

    let mut resolved = PathBuf::new();
    let mut exists = true;
    for component in absolute.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => resolved.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            Component::Normal(name) => {
                resolved.push(name);
                if exists {
                    match std::fs::canonicalize(&resolved) {
                        Ok(real) => resolved = real,
                        Err(_) => exists = false,
                    }
                }
            }
        }
    }
    Some(resolved)
}

/// Processing state of a document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DocumentStatus {
    #[default]
    Pending,
    Converting,
    Indexing,
    Failed,
    Indexed,
    Deleted,
    Processed,
}

impl DocumentStatus {
    /// The wire value (`"pending"`, `"indexed"`, ...), i.e. Python's `.value`.
    pub fn as_str(&self) -> &'static str {
        match self {
            DocumentStatus::Pending => "pending",
            DocumentStatus::Converting => "converting",
            DocumentStatus::Indexing => "indexing",
            DocumentStatus::Failed => "failed",
            DocumentStatus::Indexed => "indexed",
            DocumentStatus::Deleted => "deleted",
            DocumentStatus::Processed => "processed",
        }
    }
}

impl std::fmt::Display for DocumentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A document registered to a project.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Document {
    pub id: String,
    pub path: String,
    /// Original filename — set once, never overwritten by the indexer.
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub status: DocumentStatus,
}

impl Document {
    /// Return a serialization-ready view of this document.
    ///
    /// `path` is emitted **relative**, never absolute.  An absolute path pins
    /// a client to one machine's layout and discloses the real home directory
    /// to every API consumer.
    pub fn to_info(&self, project_dir: Option<&str>) -> DocumentInfo {
        let filename = match self.name.as_deref() {
            Some(name) if !name.is_empty() => Some(name.to_string()),
            _ if !self.path.is_empty() => Some(path_name(&self.path)),
            _ => None,
        };
        DocumentInfo {
            id: Some(self.id.clone()),
            path: relative_document_path(Some(&self.path), project_dir),
            status: self.status.as_str().to_string(),
            summary: self.summary.clone(),
            filename,
        }
    }
}

/// Serialization-ready view of a Document, suitable for API responses and SSE
/// events.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentInfo {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default = "default_document_info_status")]
    pub status: String,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub filename: Option<String>,
}

fn default_document_info_status() -> String {
    "pending".to_string()
}

impl Default for DocumentInfo {
    fn default() -> Self {
        Self {
            id: None,
            path: None,
            status: default_document_info_status(),
            summary: None,
            filename: None,
        }
    }
}

/// A skill (SKILL.md) available to a project.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Skill {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub content: String,
    #[serde(default)]
    pub system: bool,
}
