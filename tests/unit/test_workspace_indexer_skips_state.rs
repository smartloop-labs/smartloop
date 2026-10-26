//! Port of `tests/unit/test_workspace_indexer_skips_state.py`.
//!
//! The workspace index must not feed on its own output.
//!
//! `IndexerTask` points the walk at `project_dir` — the framework's own state
//! directory — and the indexer writes its manifest there at the end of every
//! run.  Indexing that manifest made each run dirty the input to the next: one
//! file re-embedded every scan, forever, for nothing.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use serial_test::serial;
use smartloop::db::workspace_indexer::{WorkspaceIndexer, MANIFEST_FILENAME};
use smartloop::workspace::workspace::PROJECT_STATE_FILE;

use crate::shared::db::RecordingStore;

/// An indexer over a recording store whose in-memory manifest starts empty
/// (the Python helper builds one with `__new__` and `_manifest = {}`).
fn indexer(project_dir: &Path) -> (WorkspaceIndexer, Arc<RecordingStore>) {
    let store = RecordingStore::new();
    let idx = WorkspaceIndexer::with_store(project_dir.to_str().unwrap(), store.clone());
    assert_eq!(
        Path::new(idx.manifest_path()),
        project_dir.join(MANIFEST_FILENAME).as_path()
    );
    idx.set_manifest(BTreeMap::new());
    (idx, store)
}

fn keys(idx: &WorkspaceIndexer) -> BTreeSet<String> {
    idx.manifest().into_keys().collect()
}

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn write(path: impl AsRef<Path>, content: &str) {
    std::fs::write(path, content).unwrap();
}

fn rel(parts: &[&str]) -> String {
    let mut p = std::path::PathBuf::new();
    for part in parts {
        p.push(part);
    }
    p.to_string_lossy().into_owned()
}

mod test_state_files_are_not_content {
    use super::*;

    #[test]
    #[serial(embeddings)]
    fn test_the_indexer_does_not_index_its_own_manifest() {
        let td = tempfile::tempdir().unwrap();
        write(td.path().join(MANIFEST_FILENAME), r#"{"a": "b"}"#);
        write(td.path().join("notes.md"), "real content");

        let (idx, _store) = indexer(td.path());
        idx.index(td.path().to_str().unwrap());
        assert_eq!(keys(&idx), set(&["notes.md"]));
    }

    #[test]
    #[serial(embeddings)]
    fn test_a_second_run_over_unchanged_content_indexes_nothing() {
        // The loop this exists to prevent: run one wrote the manifest, so run
        // two must still see nothing to do.
        let td = tempfile::tempdir().unwrap();
        write(td.path().join("notes.md"), "real content");

        let (idx, _store) = indexer(td.path());
        let first = idx.index(td.path().to_str().unwrap());
        let second = idx.index(td.path().to_str().unwrap());
        assert_eq!(first.indexed, 1);
        assert_eq!(second.indexed, 0);
        assert_eq!(second.skipped, 1);
    }

    #[test]
    #[serial(embeddings)]
    fn test_other_hidden_files_are_skipped_too() {
        // Other passes write their own manifests into the same directory.
        let td = tempfile::tempdir().unwrap();
        for name in [".mcp_documents_manifest.json", ".env", ".DS_Store"] {
            write(td.path().join(name), "{}");
        }
        let (idx, _store) = indexer(td.path());
        idx.index(td.path().to_str().unwrap());
        assert!(idx.manifest().is_empty());
    }

    #[test]
    #[serial(embeddings)]
    fn test_hidden_directories_are_skipped() {
        // Run from the home folder, the walk met `~/.smartloop` — the
        // framework's own models, db and logs — and indexed them.
        let td = tempfile::tempdir().unwrap();
        for dir in [".smartloop", ".config"] {
            std::fs::create_dir_all(td.path().join(dir)).unwrap();
            write(td.path().join(dir).join("notes.md"), "state");
        }
        write(td.path().join("notes.md"), "real content");
        let (idx, _store) = indexer(td.path());
        idx.index(td.path().to_str().unwrap());
        assert_eq!(keys(&idx), set(&["notes.md"]));
    }

    #[test]
    #[serial(embeddings)]
    fn test_the_vector_store_directory_is_skipped() {
        let td = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(td.path().join("db")).unwrap();
        write(td.path().join("db").join("embedding_model.txt"), "bge-m3");
        let (idx, _store) = indexer(td.path());
        idx.index(td.path().to_str().unwrap());
        assert!(idx.manifest().is_empty());
    }

    #[test]
    #[serial(embeddings)]
    fn test_the_uploaded_documents_directory_is_skipped() {
        // The document indexer owns these; scanning them too indexed every
        // upload a second time and kept writing chunks after indexing was done.
        let td = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(td.path().join("documents")).unwrap();
        write(td.path().join("documents").join("article.md"), "uploaded content");
        write(td.path().join("notes.md"), "real content");
        let (idx, _store) = indexer(td.path());
        idx.index(td.path().to_str().unwrap());
        assert_eq!(keys(&idx), set(&["notes.md"]));
    }

    #[test]
    #[serial(embeddings)]
    fn test_already_indexed_uploads_are_removed_from_the_store() {
        // Existing installs carry them as workspace rows; the first pass after
        // the fix has to drop those duplicates.
        let td = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(td.path().join("documents")).unwrap();
        write(td.path().join("documents").join("article.md"), "uploaded content");
        let (idx, store) = indexer(td.path());
        idx.set_manifest(BTreeMap::from([(rel(&["documents", "article.md"]), "old-hash".to_string())]));

        let stats = idx.index(td.path().to_str().unwrap());

        assert_eq!(stats.removed, 1);
        assert!(idx.manifest().is_empty());
        assert_eq!(store.delete_count(), 1);
    }

    #[test]
    #[serial(embeddings)]
    fn test_a_db_directory_deeper_in_the_tree_is_still_indexed() {
        // Only the scan root's `db` is ours; a project with its own db/ of
        // source files keeps it.
        let td = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(td.path().join("src").join("db")).unwrap();
        write(td.path().join("src").join("db").join("schema.sql"), "create table t (id int);");
        let (idx, _store) = indexer(td.path());
        idx.index(td.path().to_str().unwrap());
        assert_eq!(keys(&idx), set(&[&rel(&["src", "db", "schema.sql"])]));
    }

    #[test]
    #[serial(embeddings)]
    fn test_the_project_state_file_is_not_indexed() {
        // `project.json` is the same mistake as the manifest without the
        // leading dot: the project's own record, rewritten every turn, so
        // indexing it re-embedded a megabyte of bookkeeping on every scan.
        let td = tempfile::tempdir().unwrap();
        write(
            td.path().join(PROJECT_STATE_FILE),
            r#"{"project": {"id": "p1", "documents": []}}"#,
        );
        write(td.path().join("notes.md"), "real content");

        let (idx, _store) = indexer(td.path());
        idx.index(td.path().to_str().unwrap());
        assert_eq!(keys(&idx), set(&["notes.md"]));
    }

    #[test]
    #[serial(embeddings)]
    fn test_rewriting_the_state_file_leaves_the_index_alone() {
        // The loop as the user felt it: every turn rewrites project.json, so
        // every scan used to delete and re-embed all of it.
        let td = tempfile::tempdir().unwrap();
        let state = td.path().join(PROJECT_STATE_FILE);
        write(&state, r#"{"project": {"sessions": []}}"#);
        write(td.path().join("notes.md"), "real content");

        let (idx, _store) = indexer(td.path());
        idx.index(td.path().to_str().unwrap());
        write(&state, r#"{"project": {"sessions": ["a", "b", "c"]}}"#);
        let second = idx.index(td.path().to_str().unwrap());

        assert_eq!(second.indexed, 0);
        assert_eq!(second.removed, 0);
    }

    #[test]
    #[serial(embeddings)]
    fn test_a_state_file_deeper_in_the_tree_is_still_indexed() {
        // Only the scan root's copy is ours — a project that happens to keep
        // a project.json of its own under src/ keeps it.
        let td = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(td.path().join("src")).unwrap();
        let nested = td.path().join("src").join(PROJECT_STATE_FILE);
        write(&nested, r#"{"name": "someone else's file"}"#);
        let (idx, _store) = indexer(td.path());
        idx.index(td.path().to_str().unwrap());
        assert_eq!(keys(&idx), set(&[&rel(&["src", PROJECT_STATE_FILE])]));
    }

    #[test]
    #[serial(embeddings)]
    fn test_an_already_indexed_state_file_is_removed_from_the_store() {
        // Existing installs have it in the store and the manifest; the first
        // pass after the fix has to clean up rather than orphan 567 chunks.
        let td = tempfile::tempdir().unwrap();
        write(td.path().join(PROJECT_STATE_FILE), r#"{"project": {}}"#);
        let (idx, store) = indexer(td.path());
        idx.set_manifest(BTreeMap::from([(
            PROJECT_STATE_FILE.to_string(),
            "whatever-it-hashed-to".to_string(),
        )]));

        let stats = idx.index(td.path().to_str().unwrap());

        assert_eq!(stats.removed, 1);
        assert!(idx.manifest().is_empty());
        assert_eq!(store.delete_count(), 1);
    }
}
