//! A chat turn works in the client's `cwd` only inside the folder its project
//! is bound to; anywhere else it falls back to `project_dir`.  A project with
//! no folder, or whose folder is gone, is bound by the first usable `cwd`.

use std::fs;

use serde_json::json;
use smartloop::api::model::chat::ChatCompletionRequest;
use smartloop::api::routers::chat::request_cwd;
use smartloop::paths::{bindable_dir, dir_within, reachable_dir};

use crate::shared::api_chat::FakeChatService;

fn request(cwd: Option<&str>) -> ChatCompletionRequest {
    serde_json::from_value(json!({"model": "sl-mini", "messages": [], "cwd": cwd})).unwrap()
}

fn s(path: &std::path::Path) -> String {
    fs::canonicalize(path).unwrap().to_string_lossy().into_owned()
}

/// A service whose project lives in `<tmp>/state` and is bound to `<tmp>/code`.
fn bound(tmp: &std::path::Path) -> FakeChatService {
    fs::create_dir_all(tmp.join("state")).unwrap();
    fs::create_dir_all(tmp.join("code/src")).unwrap();
    let service = FakeChatService::default();
    *service.project_dir.lock() = Some(s(&tmp.join("state")));
    *service.work_dir.lock() = Some(s(&tmp.join("code")));
    service
}

#[test]
fn test_cwd_inside_the_bound_folder_is_used() {
    let tmp = tempfile::tempdir().unwrap();
    let service = bound(tmp.path());

    let root = tmp.path().join("code");
    let sub = tmp.path().join("code/src");
    assert_eq!(request_cwd(&request(Some(&root.to_string_lossy())), &service), Some(s(&root)));
    assert_eq!(request_cwd(&request(Some(&sub.to_string_lossy())), &service), Some(s(&sub)));
}

#[test]
fn test_cwd_outside_the_bound_folder_falls_back() {
    let tmp = tempfile::tempdir().unwrap();
    let service = bound(tmp.path());
    let state = Some(s(&tmp.path().join("state")));

    let escape = tmp.path().join("code/../state");
    assert_eq!(request_cwd(&request(Some(&escape.to_string_lossy())), &service), state);
    assert_eq!(request_cwd(&request(Some("/")), &service), state);
    let missing = tmp.path().join("code/gone");
    assert_eq!(request_cwd(&request(Some(&missing.to_string_lossy())), &service), state);
    assert_eq!(request_cwd(&request(None), &service), state);
}

#[test]
fn test_unbound_project_is_bound_by_the_first_cwd() {
    let tmp = tempfile::tempdir().unwrap();
    let service = bound(tmp.path());
    *service.work_dir.lock() = None;

    let root = tmp.path().join("code");
    assert_eq!(request_cwd(&request(Some(&root.to_string_lossy())), &service), Some(s(&root)));
    assert_eq!(*service.work_dir.lock(), Some(s(&root)));
}

#[test]
fn test_a_project_whose_folder_is_gone_is_bound_again() {
    let tmp = tempfile::tempdir().unwrap();
    let service = bound(tmp.path());
    *service.work_dir.lock() = Some(tmp.path().join("moved-away").to_string_lossy().into_owned());

    let root = tmp.path().join("code");
    assert_eq!(request_cwd(&request(Some(&root.to_string_lossy())), &service), Some(s(&root)));
    assert_eq!(*service.work_dir.lock(), Some(s(&root)));
}

#[test]
fn test_a_bound_project_is_not_moved_to_another_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let service = bound(tmp.path());
    let other = tmp.path().join("other");
    fs::create_dir_all(&other).unwrap();

    assert_eq!(request_cwd(&request(Some(&other.to_string_lossy())), &service), Some(s(&tmp.path().join("state"))));
    assert_eq!(*service.work_dir.lock(), Some(s(&tmp.path().join("code"))));
}

#[test]
fn test_the_home_folder_is_bound_but_root_never() {
    let tmp = tempfile::tempdir().unwrap();
    let service = bound(tmp.path());
    *service.work_dir.lock() = None;

    assert_eq!(request_cwd(&request(Some("/")), &service), Some(s(&tmp.path().join("state"))));
    assert_eq!(*service.work_dir.lock(), None);

    let home = reachable_dir("~");
    assert_eq!(request_cwd(&request(Some("~")), &service), home);
    assert_eq!(*service.work_dir.lock(), home);
}

#[test]
fn test_the_framework_workspace_is_never_bound() {
    let tmp = tempfile::tempdir().unwrap();
    let service = bound(tmp.path());
    *service.work_dir.lock() = None;
    *service.workspace_dir.lock() = Some(s(tmp.path()));

    let inside = tmp.path().join("state");
    assert_eq!(request_cwd(&request(Some(&inside.to_string_lossy())), &service), Some(s(&inside)));
    assert_eq!(*service.work_dir.lock(), None);
}

#[cfg(unix)]
#[test]
fn test_symlink_out_of_the_bound_folder_falls_back() {
    let tmp = tempfile::tempdir().unwrap();
    let service = bound(tmp.path());
    let link = tmp.path().join("code/out");
    std::os::unix::fs::symlink(tmp.path().join("state"), &link).unwrap();

    assert_eq!(request_cwd(&request(Some(&link.to_string_lossy())), &service), Some(s(&tmp.path().join("state"))));
}

#[test]
fn test_reachable_dir_needs_an_existing_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("a.txt");
    fs::write(&file, "x").unwrap();

    assert_eq!(reachable_dir(&tmp.path().to_string_lossy()), Some(s(tmp.path())));
    assert_eq!(reachable_dir(&file.to_string_lossy()), None);
    assert_eq!(reachable_dir(&tmp.path().join("nope").to_string_lossy()), None);
}

#[test]
fn test_dir_within_rejects_siblings_with_a_shared_prefix() {
    let tmp = tempfile::tempdir().unwrap();
    fs::create_dir_all(tmp.path().join("code")).unwrap();
    fs::create_dir_all(tmp.path().join("code-other")).unwrap();

    let root = tmp.path().join("code").to_string_lossy().into_owned();
    assert_eq!(dir_within(&tmp.path().join("code-other").to_string_lossy(), &root), None);
}

#[test]
fn test_bindable_dir_refuses_only_root() {
    let tmp = tempfile::tempdir().unwrap();

    assert_eq!(bindable_dir("~"), reachable_dir("~"));
    assert_eq!(bindable_dir("/"), None);
    assert_eq!(bindable_dir(&tmp.path().to_string_lossy()), Some(s(tmp.path())));
}
