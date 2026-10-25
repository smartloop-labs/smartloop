//! A project bound to the user's folder indexes that folder: its documents
//! through the converter, and none of the root skips meant for the
//! framework's own state.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serial_test::serial;
use smartloop::db::workspace_indexer::{DocumentConverter, WorkspaceIndexer, MANIFEST_FILENAME};

use crate::shared::db::RecordingStore;

/// An indexer whose state lives in `<tmp>/state`, with an empty manifest.
fn indexer(tmp: &Path, converter: Option<DocumentConverter>) -> (WorkspaceIndexer, Arc<RecordingStore>) {
    std::fs::create_dir_all(tmp.join("state")).unwrap();
    let store = RecordingStore::new();
    let mut idx = WorkspaceIndexer::with_store(tmp.join("state").to_str().unwrap(), store.clone());
    if let Some(converter) = converter {
        idx = idx.with_converter(converter);
    }
    idx.set_manifest(BTreeMap::new());
    (idx, store)
}

/// A user folder at `<tmp>/code` with a PDF, a note, and a `documents/` folder.
fn user_folder(tmp: &Path) -> std::path::PathBuf {
    let code = tmp.join("code");
    std::fs::create_dir_all(code.join("documents")).unwrap();
    std::fs::write(code.join("Fund supplement.pdf"), b"%PDF-1.4 fake").unwrap();
    std::fs::write(code.join("notes.md"), "Share classes A and B").unwrap();
    std::fs::write(code.join("documents").join("terms.txt"), "Terms").unwrap();
    code
}

fn keys(idx: &WorkspaceIndexer) -> BTreeSet<String> {
    idx.manifest().into_keys().collect()
}

#[test]
#[serial(embeddings)]
fn test_documents_are_parsed_by_the_converter() {
    let tmp = tempfile::tempdir().unwrap();
    let code = user_folder(tmp.path());
    let parsed: Arc<Mutex<Vec<String>>> = Arc::default();
    let seen = parsed.clone();
    let converter: DocumentConverter = Arc::new(move |path: &Path| {
        seen.lock().push(path.file_name().unwrap().to_string_lossy().into_owned());
        Ok("Class A shares carry voting rights".to_string())
    });
    let (idx, store) = indexer(tmp.path(), Some(converter));

    idx.index(code.to_str().unwrap());

    assert_eq!(*parsed.lock(), vec!["Fund supplement.pdf".to_string()]);
    assert!(keys(&idx).contains("Fund supplement.pdf"));
    let texts: Vec<String> = store
        .added
        .lock()
        .iter()
        .flatten()
        .filter_map(|chunk| chunk.get("text").and_then(|t| t.as_str()).map(str::to_string))
        .collect();
    assert!(texts.iter().any(|t| t.contains("voting rights")));

    // Unchanged on the next pass: not parsed again.
    idx.index(code.to_str().unwrap());
    assert_eq!(parsed.lock().len(), 1);
}

#[test]
#[serial(embeddings)]
fn test_documents_are_left_out_without_a_converter() {
    let tmp = tempfile::tempdir().unwrap();
    let code = user_folder(tmp.path());
    let (idx, _) = indexer(tmp.path(), None);

    idx.index(code.to_str().unwrap());

    assert!(!keys(&idx).contains("Fund supplement.pdf"));
    assert!(keys(&idx).contains("notes.md"));
}

#[test]
#[serial(embeddings)]
fn test_a_documents_folder_in_the_user_folder_is_content() {
    let tmp = tempfile::tempdir().unwrap();
    let code = user_folder(tmp.path());
    let (idx, _) = indexer(tmp.path(), None);

    idx.index(code.to_str().unwrap());

    assert!(keys(&idx).contains(&Path::new("documents").join("terms.txt").to_string_lossy().into_owned()));
}

#[test]
#[serial(embeddings)]
fn test_a_document_that_fails_to_parse_is_not_retried() {
    let tmp = tempfile::tempdir().unwrap();
    let code = user_folder(tmp.path());
    let calls: Arc<Mutex<usize>> = Arc::default();
    let count = calls.clone();
    let converter: DocumentConverter = Arc::new(move |_: &Path| {
        *count.lock() += 1;
        anyhow::bail!("encrypted")
    });
    let (idx, _) = indexer(tmp.path(), Some(converter));

    idx.index(code.to_str().unwrap());
    idx.index(code.to_str().unwrap());

    assert_eq!(*calls.lock(), 1);
}

/// Counts the documents the converter is handed.
fn counting() -> (DocumentConverter, Arc<Mutex<usize>>) {
    let calls: Arc<Mutex<usize>> = Arc::default();
    let count = calls.clone();
    let converter: DocumentConverter = Arc::new(move |_: &Path| {
        *count.lock() += 1;
        Ok("Class A shares carry voting rights".to_string())
    });
    (converter, calls)
}

/// Rewrite `path` with `content`, then set its modified time to `mtime`.
fn rewrite(path: &Path, content: &[u8], mtime: std::time::SystemTime) {
    std::fs::write(path, content).unwrap();
    std::fs::File::options().write(true).open(path).unwrap().set_modified(mtime).unwrap();
}

#[test]
#[serial(embeddings)]
fn test_a_file_with_the_same_size_and_mtime_is_not_read_again() {
    let tmp = tempfile::tempdir().unwrap();
    let code = user_folder(tmp.path());
    let pdf = code.join("Fund supplement.pdf");
    let mtime = std::fs::metadata(&pdf).unwrap().modified().unwrap();
    let (converter, calls) = counting();
    let (idx, _) = indexer(tmp.path(), Some(converter));
    idx.index(code.to_str().unwrap());
    let hash = idx.manifest()["Fund supplement.pdf"].clone();

    // Different bytes, same size and mtime: taken as unchanged.
    rewrite(&pdf, b"%PDF-1.4 FAKE", mtime);
    idx.index(code.to_str().unwrap());

    assert_eq!(*calls.lock(), 1);
    assert_eq!(idx.manifest()["Fund supplement.pdf"], hash);
}

#[test]
#[serial(embeddings)]
fn test_a_new_mtime_gets_the_file_hashed_and_read_again() {
    let tmp = tempfile::tempdir().unwrap();
    let code = user_folder(tmp.path());
    let pdf = code.join("Fund supplement.pdf");
    let (converter, calls) = counting();
    let (idx, _) = indexer(tmp.path(), Some(converter));
    idx.index(code.to_str().unwrap());

    let later = std::fs::metadata(&pdf).unwrap().modified().unwrap() + Duration::from_secs(5);
    rewrite(&pdf, b"%PDF-1.4 FAKE", later);
    idx.index(code.to_str().unwrap());

    assert_eq!(*calls.lock(), 2);
}

#[test]
#[serial(embeddings)]
fn test_a_touched_but_unchanged_file_is_hashed_not_read() {
    let tmp = tempfile::tempdir().unwrap();
    let code = user_folder(tmp.path());
    let pdf = code.join("Fund supplement.pdf");
    let (converter, calls) = counting();
    let (idx, _) = indexer(tmp.path(), Some(converter));
    idx.index(code.to_str().unwrap());

    let later = std::fs::metadata(&pdf).unwrap().modified().unwrap() + Duration::from_secs(5);
    rewrite(&pdf, b"%PDF-1.4 fake", later);
    idx.index(code.to_str().unwrap());

    // The hash still matches, so nothing is parsed again.
    assert_eq!(*calls.lock(), 1);
}

#[test]
#[serial(embeddings)]
fn test_size_and_mtime_carry_over_to_the_next_indexer() {
    let tmp = tempfile::tempdir().unwrap();
    let code = user_folder(tmp.path());
    let pdf = code.join("Fund supplement.pdf");
    let mtime = std::fs::metadata(&pdf).unwrap().modified().unwrap();
    let state = tmp.path().join("state");
    std::fs::create_dir_all(&state).unwrap();
    let (converter, calls) = counting();
    let first = WorkspaceIndexer::with_store(state.to_str().unwrap(), RecordingStore::new()).with_converter(converter.clone());
    first.index(code.to_str().unwrap());

    // A restart: a new indexer over the same state, and a file whose bytes
    // changed behind an unchanged size and mtime.
    rewrite(&pdf, b"%PDF-1.4 FAKE", mtime);
    let second = WorkspaceIndexer::with_store(state.to_str().unwrap(), RecordingStore::new()).with_converter(converter);
    second.index(code.to_str().unwrap());

    assert_eq!(*calls.lock(), 1);
    let manifest = std::fs::read_to_string(state.join(MANIFEST_FILENAME)).unwrap();
    assert!(manifest.contains("\"mtime\""));
}
