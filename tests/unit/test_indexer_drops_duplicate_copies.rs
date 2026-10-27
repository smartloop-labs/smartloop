//! A second copy of a document already in the project leaves the inbox.
//!
//! An upload whose name is taken lands beside it as `name_1`; with the same
//! content it has the same id, so the scan skipped it, and it sat in the
//! inbox listed as pending for good. A client counting what's still on its
//! way in waited on it forever.

use std::fs;
use std::sync::Arc;

use md5::{Digest, Md5};
use parking_lot::RwLock;
use smartloop::tasks::base_task::{TaskLogger, TaskOptions};
use smartloop::tasks::indexer_task::IndexerTask;
use smartloop::utils::log_utils::LoggerLike;
use smartloop::workspace::types::{Document, DocumentStatus};
use smartloop::workspace::workspace::{Project, Workspace};

struct NullLogger;

impl LoggerLike for NullLogger {
    fn info(&self, _msg: &str) {}
    fn warning(&self, _msg: &str) {}
    fn debug(&self, _msg: &str) {}
    fn error(&self, _msg: &str) {}
}

const CONTENT: &str = "# Readme\n";

/// A project whose inbox holds `README.md`, registered as `path`, and a
/// copy of it as `README_1.md`.
fn scanned(path: impl FnOnce(&std::path::Path) -> String) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let inbox = dir.path().join("documents");
    fs::create_dir_all(&inbox).unwrap();
    fs::write(inbox.join("README.md"), CONTENT).unwrap();
    fs::write(inbox.join("README_1.md"), CONTENT).unwrap();
    let document = Document {
        id: hex::encode(Md5::digest(CONTENT.as_bytes())),
        path: path(dir.path()),
        name: Some("README.md".to_string()),
        summary: None,
        status: DocumentStatus::Indexed,
    };
    let project = Project {
        id: "p1".to_string(),
        project_dir: dir.path().to_string_lossy().into_owned(),
        documents: vec![document],
        ..Default::default()
    };
    let workspace = Arc::new(RwLock::new(Workspace::new("test-workspace")));
    let logger: TaskLogger = Arc::new(NullLogger);
    let task = IndexerTask::new(workspace, logger, None, None, TaskOptions::default());
    task.set_project_provider(Some(Arc::new(move || Some(project.clone()))));
    task.scan_documents().unwrap();
    dir
}

#[test]
fn test_a_copy_of_a_registered_document_is_removed() {
    let dir = scanned(|root| root.join("documents/README.md").to_string_lossy().into_owned());
    assert!(dir.path().join("documents/README.md").exists());
    assert!(!dir.path().join("documents/README_1.md").exists());
}

#[test]
fn test_a_converted_documents_original_stays() {
    // Converted, the document's path is its copy: the original is known by
    // name, and is no duplicate.
    let dir = scanned(|root| {
        let converted = root.join("converted.md");
        fs::write(&converted, CONTENT).unwrap();
        converted.to_string_lossy().into_owned()
    });
    assert!(dir.path().join("documents/README.md").exists());
    assert!(!dir.path().join("documents/README_1.md").exists());
}
