//! Save a model's answer as a document in the working directory.
//!
//! When the user asks to create/generate/write something, the answer is saved
//! next to their work, under `.out/`.  Writing an answer's files into the
//! project is the client's job (docent takes the reply apart into files,
//! checks and commands); the engine stays generic and only saves the answer.

use std::path::Path;

/// Logger name of this module.
const LOG: &str = "smartloop.api.services.output_writer";

/// What an auto-write put on disk.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WriteOutcome {
    /// Files created, as full paths.
    pub written: Vec<String>,
}

impl WriteOutcome {
    pub fn is_empty(&self) -> bool {
        self.written.is_empty()
    }
}

/// Write `content` as `<root>/<name>.md`, adding a counter rather than
/// replacing an existing file.  Returns the full path.
pub fn write_document(root: &Path, name: &str, content: &str) -> std::io::Result<String> {
    // Under `.out/`, which the folder's index and sync leave alone: a saved
    // answer is the user's to keep, not a document of the project.
    let root = root.join(".out");
    std::fs::create_dir_all(&root)?;
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
