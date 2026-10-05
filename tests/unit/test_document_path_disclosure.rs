//! Port of `tests/unit/test_document_path_disclosure.py`.
//!
//! Document paths leave the API relative, never absolute.
//!
//! An absolute path pins a client to one machine's layout: it breaks the moment
//! `<home_dir>` is copied elsewhere or relocated into the macOS App Sandbox
//! container, and it discloses the operator's real home directory to every API
//! consumer.

use std::fs;
use std::path::{Path, PathBuf};

use serial_test::serial;

use smartloop::workspace::types::{Document, DocumentStatus, relative_document_path};

use crate::shared::workspace_support::EnvGuard;

fn s(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// The `project_dir` fixture (the temp dir is returned to keep it alive).
fn project_dir() -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let d = tmp.path().join("home").join("ws-1").join("proj-1");
    fs::create_dir_all(d.join("documents")).unwrap();
    fs::write(d.join("documents").join("Invoice.pdf"), b"%PDF").unwrap();
    (tmp, d)
}

fn doc(project_dir: &Path, name: &str) -> Document {
    Document {
        id: "d1".to_string(),
        path: s(&project_dir.join("documents").join(name)),
        name: Some(name.to_string()),
        status: DocumentStatus::Indexed,
        ..Default::default()
    }
}

#[test]
#[serial]
fn test_path_is_relative_to_the_project() {
    let (_tmp, project_dir) = project_dir();
    assert_eq!(
        doc(&project_dir, "Invoice.pdf").to_info(Some(&s(&project_dir))).path.as_deref(),
        Some("documents/Invoice.pdf")
    );
}

#[test]
#[serial]
fn test_absolute_path_never_escapes() {
    let (_tmp, project_dir) = project_dir();
    let info = doc(&project_dir, "Invoice.pdf").to_info(Some(&s(&project_dir)));
    let path = info.path.unwrap();
    assert!(!Path::new(&path).is_absolute());
    assert!(!path.contains(&s(&project_dir)));
}

/// Even without a project_dir, the home prefix is stripped.
#[test]
#[serial]
fn test_home_directory_is_not_disclosed() {
    let (_tmp, project_dir) = project_dir();
    let _home = EnvGuard::set("SLP_HOME", project_dir.parent().unwrap().parent().unwrap());
    let info = doc(&project_dir, "Invoice.pdf").to_info(None);
    let path = info.path.unwrap();
    assert_eq!(path, "ws-1/proj-1/documents/Invoice.pdf");
    assert!(!path.split('/').next().unwrap().contains("home"));
}

#[test]
#[serial]
fn test_filename_is_still_reported() {
    let (_tmp, project_dir) = project_dir();
    assert_eq!(
        doc(&project_dir, "Invoice.pdf").to_info(Some(&s(&project_dir))).filename.as_deref(),
        Some("Invoice.pdf")
    );
}

/// A path from somewhere else must not disclose its location.
#[test]
#[serial]
fn test_path_outside_the_framework_degrades_to_a_bare_name() {
    let (tmp, project_dir) = project_dir();
    let outside = tmp.path().join("elsewhere").join("secret-folder").join("leak.pdf");
    fs::create_dir_all(outside.parent().unwrap()).unwrap();
    fs::write(&outside, b"x").unwrap();
    let rendered = relative_document_path(Some(&s(&outside)), Some(&s(&project_dir))).unwrap();
    assert_eq!(rendered, "leak.pdf");
    assert!(!rendered.contains("secret-folder"));
}

#[test]
#[serial]
fn test_none_and_empty_pass_through() {
    assert_eq!(relative_document_path(None, Some("/x")), None);
    assert_eq!(relative_document_path(Some(""), Some("/x")).as_deref(), Some(""));
}

/// A path rendered on Windows must compare equal to one from macOS.
#[test]
#[serial]
fn test_posix_separators_always() {
    let (_tmp, project_dir) = project_dir();
    assert!(
        !doc(&project_dir, "Invoice.pdf")
            .to_info(Some(&s(&project_dir)))
            .path
            .unwrap()
            .contains('\\')
    );
}

/// The same document under a different home renders identically.
#[test]
#[serial]
fn test_relative_form_survives_relocation() {
    let tmp = tempfile::tempdir().unwrap();
    let mut rendered = Vec::new();
    for home in ["home_a", "home_b"] {
        let d = tmp.path().join(home).join("ws-1").join("proj-1");
        fs::create_dir_all(d.join("documents")).unwrap();
        fs::write(d.join("documents").join("Invoice.pdf"), b"%PDF").unwrap();
        rendered.push(doc(&d, "Invoice.pdf").to_info(Some(&s(&d))).path.unwrap());
    }
    assert_eq!(rendered[0], rendered[1]);
    assert_eq!(rendered[1], "documents/Invoice.pdf");
}

/// Rendering must not depend on the file existing on disk.
#[test]
#[serial]
fn test_missing_file_still_renders_relative() {
    let (_tmp, project_dir) = project_dir();
    let doc = Document {
        id: "d2".to_string(),
        path: s(&project_dir.join("documents").join("gone.pdf")),
        name: Some("gone.pdf".to_string()),
        status: DocumentStatus::Indexed,
        ..Default::default()
    };
    assert_eq!(doc.to_info(Some(&s(&project_dir))).path.as_deref(), Some("documents/gone.pdf"));
}
