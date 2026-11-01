//! Auto-write: a saved answer goes under `.out/` in the working directory,
//! and the answer links what was written.

use smartloop::api::routers::chat::write_summary;
use smartloop::api::services::output_writer::{write_document, WriteOutcome};

/// A saved answer goes under `.out/`, which the folder's index and sync
/// leave alone, so it never comes back as a document of the project.
#[test]
fn test_write_document_goes_in_out_without_replacing() {
    let dir = tempfile::tempdir().unwrap();
    let first = write_document(dir.path(), "notes", "# One").unwrap();
    let second = write_document(dir.path(), "notes", "# Two").unwrap();
    assert_eq!(first, dir.path().join(".out/notes.md").to_string_lossy());
    assert_eq!(second, dir.path().join(".out/notes_1.md").to_string_lossy());
    assert!(!dir.path().join("notes.md").exists());
}

#[test]
fn test_write_summary_links_files() {
    let single = WriteOutcome { written: vec!["/w/notes.md".into()] };
    assert_eq!(write_summary(&single), "\n\n[notes.md](/w/notes.md)");

    let many = WriteOutcome { written: vec!["/w/Cargo.toml".into(), "/w/src/main.rs".into()] };
    assert_eq!(write_summary(&many), "\n\nWrote 2 files:\n\n- [Cargo.toml](/w/Cargo.toml)\n- [main.rs](/w/src/main.rs)");
    assert_eq!(write_summary(&WriteOutcome::default()), "");
}
