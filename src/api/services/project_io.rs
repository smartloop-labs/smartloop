//! Port of `smartloop/api/services/project_io.py`.
//!
//! Project export/import service.
//!
//! Packages a project into a portable zip archive and restores it on another
//! machine (or the same one).

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use chrono::{Datelike, Local, Timelike};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde_json::{json, Map, Value};
use zip::write::SimpleFileOptions;
use zip::CompressionMethod;

use crate::config::app_settings::AppSettings;
use crate::platform::workspace_db_for;
use crate::workspace::context::own_project;
use crate::workspace::workspace::{Document, MCPServer, Project, Skill, Workspace};

use super::{pyjson, utc_now_isoformat};

/// Logger name used by the Python module.
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
    if let Some(settings) = SETTINGS.read().as_ref() {
        return settings.home_dir();
    }
    AppSettings::new().home_dir()
}

/// Return `<home_dir>/conversations/`.
fn conversations_dir() -> String {
    Path::new(&home_dir()).join("conversations").to_string_lossy().into_owned()
}

/// `time.localtime(ts)[:6]` as a zip timestamp (Python's `ZipInfo.date_time`).
fn zip_datetime(ts: std::time::SystemTime) -> zip::DateTime {
    let local: chrono::DateTime<Local> = ts.into();
    zip::DateTime::from_date_and_time(
        local.year().clamp(1980, 2107) as u16,
        local.month() as u8,
        local.day() as u8,
        local.hour() as u8,
        local.minute() as u8,
        local.second().min(59) as u8,
    )
    .unwrap_or_default()
}

/// Options for a member written from a file on disk (`ZipFile.write`): its
/// mtime, its mode bits, deflated, zip64 when it needs it.
fn file_options(meta: &fs::Metadata) -> SimpleFileOptions {
    let mut options = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .large_file(meta.len() >= 0xFFFF_FFFF);
    if let Ok(mtime) = meta.modified() {
        options = options.last_modified_time(zip_datetime(mtime));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        options = options.unix_permissions(meta.permissions().mode() & 0o7777);
    }
    options
}

/// `zf.write(full, arcname)`: copy one file from disk into the archive.
fn write_file_to_zip<W: Write + std::io::Seek>(
    zf: &mut zip::ZipWriter<W>,
    full: &Path,
    arcname: &str,
) -> anyhow::Result<()> {
    // ``ZipFile.write`` stats (and opens) through symlinks.
    let meta = fs::metadata(full)?;
    let mut src = fs::File::open(full)?;
    zf.start_file(arcname, file_options(&meta))?;
    std::io::copy(&mut src, zf)?;
    Ok(())
}

/// Recursively add `src_dir` contents under `arc_prefix` in `zf`.
fn add_directory_to_zip<W: std::io::Write + std::io::Seek>(
    zf: &mut zip::ZipWriter<W>,
    src_dir: &str,
    arc_prefix: &str,
) -> anyhow::Result<()> {
    // ``os.walk`` (followlinks=False): symlinked directories are listed but
    // not descended into; every non-directory entry is written, a symlink to
    // a file contributing its target's bytes.
    let walker = walkdir::WalkDir::new(src_dir).follow_links(false).min_depth(1);
    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            // ``os.walk`` swallows listing errors by default.
            Err(_) => continue,
        };
        let ft = entry.file_type();
        if ft.is_dir() {
            continue;
        }
        if ft.is_symlink() && fs::metadata(entry.path()).map(|m| m.is_dir()).unwrap_or(false) {
            continue;
        }
        let rel = match entry.path().strip_prefix(src_dir) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let rel_parts: Vec<String> =
            rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
        let arcname = format!("{}/{}", arc_prefix, rel_parts.join("/"));
        write_file_to_zip(zf, entry.path(), &arcname)?;
    }
    Ok(())
}

/// Remove OAuth credentials/tokens from MCP servers in project data.
pub fn strip_oauth(mut project_data: Map<String, Value>) -> Map<String, Value> {
    if let Some(Value::Array(servers)) = project_data.get_mut("mcp_servers") {
        for srv in servers.iter_mut() {
            if let Some(srv) = srv.as_object_mut() {
                srv.remove("oauth_credentials");
                if let Some(Value::Object(remote)) = srv.get_mut("remote_config") {
                    remote.remove("token");
                }
            }
        }
    }
    project_data
}

// ── Export ──────────────────────────────────────────────────────────────

/// Parse `SKILL.md` files from a skills directory into `Skill` objects.
///
/// Each skill lives in its own subdirectory as a `SKILL.md` file with YAML
/// frontmatter (name, description, tools).
pub fn read_skills_from_dir(skills_dir: &str) -> Vec<Skill> {
    let dir = Path::new(skills_dir);
    if !dir.is_dir() {
        return Vec::new();
    }

    let mut entries: Vec<String> = match fs::read_dir(dir) {
        Ok(rd) => rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect(),
        Err(_) => return Vec::new(),
    };
    entries.sort();

    let mut skills = Vec::new();
    for entry in entries {
        let entry_path = dir.join(&entry);
        if !entry_path.is_dir() {
            continue;
        }
        let skill_path = entry_path.join("SKILL.md");
        if !skill_path.is_file() {
            continue;
        }

        let raw = match fs::read(&skill_path).map_err(anyhow::Error::from).and_then(|bytes| {
            String::from_utf8(bytes).map_err(anyhow::Error::from)
        }) {
            Ok(raw) => raw,
            Err(e) => {
                tracing::warn!(target: LOG_TARGET, "Could not read skill {}: {}", skill_path.display(), e);
                continue;
            }
        };

        let mut name = entry.clone();
        let mut description: Option<String> = None;
        // Parsed for parity with the frontmatter format; the `Skill` model has
        // no `tools` field, so (as with pydantic's ignored extra) it is dropped.
        let mut _tools: Vec<String> = Vec::new();
        let mut content = raw.clone();

        if raw.starts_with("---") {
            // ``raw.split("---", 2)``
            let parts: Vec<&str> = raw.splitn(3, "---").collect();
            if parts.len() >= 3 {
                let frontmatter = parts[1];
                content = parts[2].trim().to_string();
                for line in frontmatter.trim().lines() {
                    if let Some(rest) = line.strip_prefix("name:") {
                        name = rest.trim().to_string();
                    } else if let Some(rest) = line.strip_prefix("description:") {
                        description = Some(rest.trim().to_string());
                    } else if let Some(rest) = line.strip_prefix("tools:") {
                        let tools_str = rest.trim().trim_matches(|c| c == '[' || c == ']');
                        _tools = tools_str
                            .split(',')
                            .map(|t| t.trim())
                            .filter(|t| !t.is_empty())
                            .map(String::from)
                            .collect();
                    }
                }
            }
        }

        let id: String = uuid::Uuid::new_v4().to_string().chars().take(8).collect();
        skills.push(Skill { id, name, description, content, system: false });
    }

    skills
}

/// Create a zip archive for the given project.
///
/// Returns the path to the temporary zip file.  The caller is responsible for
/// cleaning it up after the response has been sent.
pub fn export_project(project_id: &str) -> Result<PathBuf, ProjectIoError> {
    let workspace = Workspace::load()
        .ok_or_else(|| ProjectIoError::ValueError("Workspace not initialized".to_string()))?;

    let project = workspace
        .projects
        .iter()
        .find(|p| p.id == project_id)
        .ok_or_else(|| ProjectIoError::FileNotFoundError(format!("Project {project_id} not found")))?;

    let dumped = serde_json::to_value(project).map_err(anyhow::Error::from)?;
    let mut project_data = strip_oauth(dumped.as_object().cloned().unwrap_or_default());

    // Skills are exported as files in the project/skills/ directory,
    // not embedded in the manifest.
    project_data.remove("skills");

    let exported_at = utc_now_isoformat();
    let manifest = json!({
        "version": "1.0",
        "exported_at": exported_at,
        "project": Value::Object(project_data),
    });

    let label = match project.name.as_deref() {
        Some(name) if !name.is_empty() => name.to_string(),
        _ => project.id.clone(),
    };
    let tmp = tempfile::Builder::new().prefix(&format!("{label}_")).suffix(".zip").tempfile()?;
    let (file, tmp_path) = tmp.keep().map_err(|e| anyhow::Error::from(e.error))?;

    let written = (|| -> anyhow::Result<()> {
        let mut zf = zip::ZipWriter::new(file);
        // ``writestr`` with a bare name: now, mode 0o600, the archive's compression.
        let manifest_opts = SimpleFileOptions::default()
            .compression_method(CompressionMethod::Deflated)
            .last_modified_time(zip_datetime(std::time::SystemTime::now()))
            .unix_permissions(0o600);
        zf.start_file("manifest.json", manifest_opts)?;
        zf.write_all(pyjson::dumps_indent(&manifest, 2).as_bytes())?;

        // The project's state directory — skills/, documents/, db/ and the
        // manifest all live here now, so this single tree is the whole
        // payload.  Archives from older builds additionally carried a
        // "project" tree (the old ~/smartloop/<name> folder); import still
        // reads that, but export no longer writes it.
        if !project.project_dir.is_empty() && Path::new(&project.project_dir).is_dir() {
            add_directory_to_zip(&mut zf, &project.project_dir, "workspace")?;
        }

        // Conversation history JSONL files
        let conv_dir = conversations_dir();
        for session_id in &project.sessions {
            let jsonl = Path::new(&conv_dir).join(format!("{session_id}.jsonl"));
            if jsonl.is_file() {
                write_file_to_zip(&mut zf, &jsonl, &format!("conversations/{session_id}.jsonl"))?;
            }
        }

        zf.finish()?;
        Ok(())
    })();

    if let Err(e) = written {
        let _ = fs::remove_file(&tmp_path);
        return Err(ProjectIoError::Other(e));
    }

    Ok(tmp_path)
}

// ── Import ──────────────────────────────────────────────────────────────

/// `shutil.copy2`: copy the bytes and the permission bits, then the
/// access/modification times.
fn copy2(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::copy(src, dst)?;
    let meta = fs::metadata(src)?;
    let mut times = fs::FileTimes::new();
    if let Ok(accessed) = meta.accessed() {
        times = times.set_accessed(accessed);
    }
    if let Ok(modified) = meta.modified() {
        times = times.set_modified(modified);
    }
    let f = fs::OpenOptions::new().write(true).open(dst)?;
    f.set_times(times)
}

/// `shutil.copytree(src, dst, dirs_exist_ok=True)` (symlinks followed).
fn copy_tree(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if fs::metadata(&from)?.is_dir() {
            copy_tree(&from, &to)?;
        } else {
            copy2(&from, &to)?;
        }
    }
    Ok(())
}

/// Python `zipfile.BadZipFile`: the archive itself is unreadable.
fn is_bad_zip(e: &zip::result::ZipError) -> bool {
    match e {
        zip::result::ZipError::InvalidArchive(_) | zip::result::ZipError::UnsupportedArchive(_) => true,
        zip::result::ZipError::Io(io) => io.kind() == std::io::ErrorKind::InvalidData,
        _ => false,
    }
}

/// Unpack `zip_path` into `dest` (`ZipFile(zip_path).extractall(dest)`).
fn extract_archive(zip_path: &Path, dest: &Path) -> Result<(), ProjectIoError> {
    let bad = || ProjectIoError::ValueError("Uploaded file is not a valid zip archive".to_string());
    let file = fs::File::open(zip_path)?;
    let mut archive = match zip::ZipArchive::new(file) {
        Ok(a) => a,
        Err(e) if is_bad_zip(&e) => return Err(bad()),
        Err(e) => return Err(ProjectIoError::Other(e.into())),
    };
    match archive.extract(dest) {
        Ok(()) => Ok(()),
        Err(e) if is_bad_zip(&e) => Err(bad()),
        Err(e) => Err(ProjectIoError::Other(e.into())),
    }
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
    let mut loaded: Option<Workspace>;
    let workspace: &mut Workspace = match workspace {
        Some(ws) => ws,
        None => {
            loaded = Workspace::load();
            loaded
                .as_mut()
                .ok_or_else(|| ProjectIoError::ValueError("Workspace not initialized".to_string()))?
        }
    };

    // Removed (errors ignored) when it drops, like ``shutil.rmtree(tmp_dir,
    // ignore_errors=True)`` in the ``finally``.
    let tmp = tempfile::Builder::new().prefix("slp_import_").tempdir()?;
    let tmp_dir = tmp.path();

    // Extract
    extract_archive(zip_path, tmp_dir)?;

    // Read manifest
    let manifest_path = tmp_dir.join("manifest.json");
    if !manifest_path.is_file() {
        return Err(ProjectIoError::ValueError("Zip archive is missing manifest.json".to_string()));
    }

    let mut raw = String::new();
    fs::File::open(&manifest_path)?.read_to_string(&mut raw)?;
    let manifest: Value = serde_json::from_str(&raw).map_err(anyhow::Error::from)?;
    let manifest = manifest
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("'{}' object has no attribute 'get'", py_type_name(&manifest)))?;

    let project_data = match manifest.get("project") {
        Some(v) if py_truthy(v) => v.clone(),
        _ => return Err(ProjectIoError::ValueError("manifest.json is missing project data".to_string())),
    };
    let project_data = project_data
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("'{}' object has no attribute 'get'", py_type_name(&project_data)))?;

    // Generate new ID
    let new_id = uuid::Uuid::new_v4().to_string();
    let new_project_dir = Path::new(&workspace.workspace_dir).join(&new_id);
    fs::create_dir_all(&new_project_dir)?;
    let new_project_dir_str = new_project_dir.to_string_lossy().into_owned();

    // Copy workspace/ contents → new project dir
    let src_workspace = tmp_dir.join("workspace");
    if src_workspace.is_dir() {
        copy_tree(&src_workspace, &new_project_dir)?;
    }

    // Resolve project name (caller override or original)
    let project_name: Option<String> = match name {
        Some(n) if !n.is_empty() => Some(n.to_string()),
        _ => match project_data.get("name") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        },
    };

    // Archives written by older builds carry the project's skills and
    // documents in a separate "project" tree (the old ~/smartloop/<name>
    // folder).  Fold those into the new state directory so a legacy export
    // imports into the current layout.  Newer archives have them under
    // "workspace" already, copied above.
    let src_legacy = tmp_dir.join("project");
    if src_legacy.is_dir() {
        for sub in ["skills", "documents"] {
            let src_sub = src_legacy.join(sub);
            if src_sub.is_dir() {
                copy_tree(&src_sub, &new_project_dir.join(sub))?;
            }
        }
        tracing::info!(target: LOG_TARGET, "[import] Migrated legacy 'project' tree into {}", new_project_dir_str);
    }

    // Copy conversation history JSONL files
    let src_conversations = tmp_dir.join("conversations");
    if src_conversations.is_dir() {
        let conv_dir = PathBuf::from(conversations_dir());
        fs::create_dir_all(&conv_dir)?;
        for entry in fs::read_dir(&src_conversations)? {
            let entry = entry?;
            let fname = entry.file_name().to_string_lossy().into_owned();
            if fname.ends_with(".jsonl") {
                copy2(&entry.path(), &conv_dir.join(&fname))?;
            }
        }
    }

    // Build Project object
    let documents: Vec<Document> = list_field(&project_data, "documents")
        .into_iter()
        .map(serde_json::from_value)
        .collect::<Result<_, _>>()
        .map_err(anyhow::Error::from)?;

    // Read skills from the state directory we just assembled — it holds both
    // the modern "workspace" payload and any folded-in legacy tree.
    let skills = read_skills_from_dir(&new_project_dir.join("skills").to_string_lossy());

    let mut mcp_servers: Vec<MCPServer> = list_field(&project_data, "mcp_servers")
        .into_iter()
        .map(serde_json::from_value)
        .collect::<Result<_, _>>()
        .map_err(anyhow::Error::from)?;

    // Strip OAuth from imported MCP servers
    for srv in mcp_servers.iter_mut() {
        srv.oauth_credentials = None;
        if let Some(remote) = srv.remote_config.as_mut() {
            remote.token = None;
        }
    }

    let description = match project_data.get("description") {
        Some(Value::String(s)) => Some(s.clone()),
        _ => None,
    };
    let system = project_data.get("system").map(py_truthy).unwrap_or(false);
    let sessions: Vec<String> = list_field(&project_data, "sessions")
        .into_iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    let new_project = Project {
        id: new_id.clone(),
        project_dir: new_project_dir_str,
        current: true,
        name: project_name,
        // ``cwd`` is a machine-local working folder, not project state — a
        // path from the exporting machine is meaningless here, so an imported
        // project starts without one until the user picks it.
        cwd: None,
        description,
        system,
        documents,
        skills,
        mcp_servers,
        sessions,
        ..Default::default()
    };

    // Deactivate other projects
    for p in workspace.projects.iter_mut() {
        p.current = false;
    }

    workspace.projects.push(new_project);
    // Seed every model switched off, same as create_project does — an archive
    // carries no model_settings, so without this the project would read
    // through to the workspace record until its first models read.
    workspace.sync_project_models(&new_id);
    // Nothing is serving this project yet, so its first ``project.json`` is
    // ours to write even where per-project files belong to the children.
    {
        let _owned = own_project(Some(&new_id));
        workspace.save()?;
    }
    workspace_db_for(workspace)?.projects().add(&workspace.id, &new_id)?;
    Workspace::reset();

    let new_project = workspace
        .projects
        .iter()
        .find(|p| p.id == new_id)
        .ok_or_else(|| anyhow::anyhow!("imported project {new_id} vanished from the workspace"))?;

    let skills_json: Vec<Value> = new_project
        .get_skills()
        .iter()
        .map(serde_json::to_value)
        .collect::<Result<_, _>>()
        .map_err(anyhow::Error::from)?;
    let mcp_json: Vec<Value> = new_project
        .mcp_servers
        .iter()
        .map(serde_json::to_value)
        .collect::<Result<_, _>>()
        .map_err(anyhow::Error::from)?;
    let documents_json: Vec<Value> = new_project
        .documents
        .iter()
        .map(|d| {
            json!({
                "id": d.id,
                "path": d.path,
                "status": d.status.as_str(),
                "summary": d.summary,
            })
        })
        .collect();

    let mut result = Map::new();
    result.insert("id".into(), Value::String(new_project.id.clone()));
    result.insert("name".into(), json!(new_project.name));
    result.insert("cwd".into(), Value::String(new_project.work_dir().unwrap_or_default()));
    result.insert("description".into(), json!(new_project.description));
    result.insert("current".into(), Value::Bool(new_project.current));
    result.insert("system".into(), Value::Bool(new_project.system));
    result.insert("skills".into(), Value::Array(skills_json));
    result.insert("mcp".into(), Value::Array(mcp_json));
    result.insert("documents".into(), Value::Array(documents_json));

    Ok(result)
}

/// `project_data.get(key, [])` for a list field (a non-list counts as empty).
fn list_field(data: &Map<String, Value>, key: &str) -> Vec<Value> {
    match data.get(key) {
        Some(Value::Array(items)) => items.clone(),
        _ => Vec::new(),
    }
}

/// Python truthiness of a JSON value.
fn py_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// The Python type name a JSON value decodes to (for error text).
fn py_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}
