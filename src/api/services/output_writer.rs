//! Write a model's answer to files in the working directory.
//!
//! When the user asks to create/generate/write something, the answer is saved
//! next to their work.  An answer that lays out files — each fenced block
//! labelled with its path (`File: src/main.rs` on the line above, a backticked
//! path in a lead-in like "Create `Cargo.toml`:", or `rust:src/main.rs` on the
//! fence) — is written as those files; anything else is a single document.

use std::path::{Component, Path, PathBuf};

use once_cell::sync::Lazy;
use regex::Regex;

/// Logger name of this module.
const LOG: &str = "smartloop.api.services.output_writer";

/// Path-like names with no extension that are still files.
const BARE_FILE_NAMES: &[&str] = &[
    "Makefile", "Dockerfile", "LICENSE", "Procfile", "Gemfile", "Rakefile", "Justfile", "Containerfile",
];

/// A label prefix on the line above a block (`File: src/main.rs`).
static LABEL_PREFIX_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)^(?:file ?name|file|path)\s*[:=]\s*").expect("valid LABEL_PREFIX_RE"));

/// A backticked span inside a lead-in line ("Create `Cargo.toml`:").
static BACKTICKED_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"`([^`\s]+)`").expect("valid BACKTICKED_RE"));

/// What [`write_answer`] put on disk.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WriteOutcome {
    /// Files created, as full paths.
    pub written: Vec<String>,
    /// Files the answer named that already existed and were left alone.
    pub skipped: Vec<String>,
}

impl WriteOutcome {
    pub fn is_empty(&self) -> bool {
        self.written.is_empty() && self.skipped.is_empty()
    }
}

/// Whether `s` reads as a relative file path rather than a word or a command.
fn looks_like_path(s: &str) -> bool {
    if s.is_empty() || s.ends_with('/') || s.ends_with('.') || s.contains("://") {
        return false;
    }
    if !s.chars().all(|c| c.is_ascii_alphanumeric() || "._-/\\".contains(c)) {
        return false;
    }
    if !s.chars().any(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    s.contains('.') || s.contains('/') || BARE_FILE_NAMES.contains(&s)
}

/// The path a fence's info string names (```` ```rust:src/main.rs ````,
/// ```` ```toml Cargo.toml ````, ```` ```path=src/lib.rs ````), if any.
fn path_from_info(info: &str) -> Option<String> {
    let mut tokens = info.split(|c: char| c.is_whitespace() || c == ':' || c == '=').filter(|t| !t.is_empty());
    let first = tokens.next()?;
    // A lone token is the language (`toml`) unless it is itself a path.
    let candidate = tokens.last().unwrap_or(first);
    let candidate = candidate.trim_matches(|c| c == '"' || c == '\'');
    looks_like_path(candidate).then(|| candidate.to_string())
}

/// The path the line above a block names, if it is a label for that block.
fn path_from_label(line: &str) -> Option<String> {
    let trimmed = line.trim().trim_start_matches(|c: char| c == '#' || c == '-' || c == '>' || c.is_whitespace());
    let stripped = LABEL_PREFIX_RE.replace(trimmed.trim_matches('*'), "");
    let bare = stripped.trim().trim_matches(|c| c == '*' || c == '`' || c == '_').trim_end_matches(':');
    let bare = bare.trim_matches(|c| c == '*' || c == '`' || c == '_');
    if looks_like_path(bare) {
        return Some(bare.to_string());
    }
    // A lead-in sentence: take its one backticked path.
    if trimmed.chars().count() > 120 {
        return None;
    }
    let mut paths = BACKTICKED_RE
        .captures_iter(trimmed)
        .map(|c| c[1].to_string())
        .filter(|p| looks_like_path(p));
    match (paths.next(), paths.next()) {
        (Some(path), None) => Some(path),
        _ => None,
    }
}

/// The `(path, content)` of every fenced block in `text` that names a file.
/// A later block for the same path replaces an earlier one.
pub fn extract_file_blocks(text: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = text.split('\n').collect();
    let mut blocks: Vec<(String, String)> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim_start();
        let fence_len = line.chars().take_while(|&c| c == '`').count();
        if fence_len < 3 {
            i += 1;
            continue;
        }
        let info = &line[fence_len..];
        let label = lines[..i].iter().rev().take(2).find(|l| !l.trim().is_empty());
        let path = path_from_info(info).or_else(|| label.and_then(|l| path_from_label(l)));

        // The block runs to a fence at least as long as the opening one, so
        // a README can nest ``` blocks inside a ```` block.
        let mut end = i + 1;
        while end < lines.len() {
            let t = lines[end].trim();
            if t.len() >= fence_len && t.chars().all(|c| c == '`') {
                break;
            }
            end += 1;
        }
        if let Some(path) = path {
            let content = lines[i + 1..end.min(lines.len())].join("\n");
            blocks.retain(|(p, _)| p != &path);
            blocks.push((path, content));
        }
        i = end + 1;
    }
    blocks
}

/// `path` resolved under `root`, or `None` when it is absolute or climbs out.
pub fn resolve_under(root: &Path, path: &str) -> Option<PathBuf> {
    let mut resolved = root.to_path_buf();
    let mut depth = 0;
    for component in Path::new(&path.replace('\\', "/")).components() {
        match component {
            Component::Normal(part) => {
                resolved.push(part);
                depth += 1;
            }
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    (depth > 0).then_some(resolved)
}

/// Write each file block under `root`.  An existing file is never
/// overwritten: the answer may be quoting code, not replacing it.
pub fn write_file_blocks(root: &Path, blocks: &[(String, String)]) -> std::io::Result<WriteOutcome> {
    let mut outcome = WriteOutcome::default();
    for (path, content) in blocks {
        let Some(full_path) = resolve_under(root, path) else {
            tracing::warn!(target: LOG, "[auto-write] Ignoring path outside the working directory: {path}");
            continue;
        };
        let shown = full_path.to_string_lossy().into_owned();
        if full_path.exists() {
            tracing::info!(target: LOG, "[auto-write] {shown} exists — left unchanged");
            outcome.skipped.push(shown);
            continue;
        }
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut content = content.clone();
        if !content.ends_with('\n') {
            content.push('\n');
        }
        std::fs::write(&full_path, &content)?;
        tracing::info!(target: LOG, "[auto-write] Saved {shown} ({} chars)", content.chars().count());
        outcome.written.push(shown);
    }
    Ok(outcome)
}

/// Write `content` as `<root>/<name>.md`, adding a counter rather than
/// replacing an existing file.  Returns the full path.
pub fn write_document(root: &Path, name: &str, content: &str) -> std::io::Result<String> {
    std::fs::create_dir_all(root)?;
    let mut full_path = root.join(format!("{name}.md"));
    let mut counter = 1;
    while full_path.exists() {
        full_path = root.join(format!("{name}_{counter}.md"));
        counter += 1;
    }
    let mut content = content.to_string();
    if !content.ends_with('\n') {
        content.push('\n');
    }
    content.push_str("\nCo-Authored By: Smartloop AI\n");
    std::fs::write(&full_path, &content)?;
    let full_path = full_path.to_string_lossy().into_owned();
    tracing::info!(target: LOG, "[auto-write] Saved {full_path} ({} chars)", content.chars().count());
    Ok(full_path)
}
