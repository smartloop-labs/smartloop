//! Port of `tests/unit/test_project_store_caches.py`.
//!
//! Store handles are cached per project, not one slot for the whole process.
//!
//! With the project bound per request, two chats hold two different projects
//! at once.  A single cache slot would have them evicting each other's store on
//! every turn — reopening SQLite each time, and in the window between the check
//! and the rebuild, answering one project's query from the other's index.

use std::sync::{Arc, Barrier};

use smartloop::api::services::model_service::{ModelService, ProjectCaches};
use smartloop::db::feedback_indexer::FeedbackIndexer;
use smartloop::db::store::VectorStore;
use smartloop::db::workspace_indexer::WorkspaceIndexer;

use crate::shared::model_service_b::FakeStore;

fn service() -> Arc<ModelService> {
    ModelService::new_bare()
}

/// The label of the fake store held in `project_id`'s slot, if any.
fn store_label(svc: &ModelService, project_id: &str) -> Option<String> {
    let slot = svc.caches(project_id);
    let cache = slot.lock();
    cache.doc_store.as_ref().map(label_of)
}

/// A labelled fake store answers every query with its label.
fn label_of(store: &Arc<dyn VectorStore>) -> String {
    store.query("", None).ok().and_then(|texts| texts.into_iter().next()).unwrap_or_default()
}

fn labelled_store(label: &str) -> Arc<dyn VectorStore> {
    let chunk = smartloop::db::store::ChunkResult { text: label.to_string(), ..Default::default() };
    FakeStore::new(label, vec![chunk])
}

#[test]
fn test_each_project_gets_its_own_slot() {
    let svc = service();
    assert!(!Arc::ptr_eq(&svc.caches("alpha"), &svc.caches("beta")));
}

#[test]
fn test_the_same_project_gets_the_same_slot() {
    let svc = service();
    assert!(Arc::ptr_eq(&svc.caches("alpha"), &svc.caches("alpha")));
}

#[test]
fn test_a_slot_starts_empty() {
    let svc = service();
    let slot = svc.caches("alpha");
    let cache = slot.lock();
    assert!(cache.doc_store.is_none());
    assert!(cache.workspace_indexer.is_none());
    assert!(cache.feedback_indexer.is_none());
}

#[test]
fn test_a_handle_stored_for_one_project_is_invisible_to_another() {
    let svc = service();
    svc.caches("alpha").lock().doc_store = Some(labelled_store("alpha-store"));
    assert!(store_label(&svc, "beta").is_none());
    assert_eq!(store_label(&svc, "alpha").as_deref(), Some("alpha-store"));
}

/// Two requests opening the same project at once must not build two.
#[test]
fn test_concurrent_first_touches_agree_on_one_slot() {
    let svc = service();
    let start = Arc::new(Barrier::new(8));
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let svc = svc.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                start.wait();
                svc.caches("alpha")
            })
        })
        .collect();
    let seen: Vec<_> = handles.into_iter().map(|h| h.join().expect("thread finished")).collect();

    assert_eq!(seen.len(), 8);
    assert!(seen.iter().all(|c| Arc::ptr_eq(c, &seen[0])));
}

#[test]
fn test_dropping_one_project_leaves_the_others_alone() {
    let svc = service();
    svc.caches("alpha").lock().doc_store = Some(labelled_store("alpha-store"));
    svc.caches("beta").lock().doc_store = Some(labelled_store("beta-store"));

    svc.drop_caches(Some("alpha"));

    assert!(store_label(&svc, "alpha").is_none());
    assert_eq!(store_label(&svc, "beta").as_deref(), Some("beta-store"));
}

#[test]
fn test_dropping_everything() {
    let svc = service();
    svc.caches("alpha").lock().doc_store = Some(labelled_store("alpha-store"));
    svc.caches("beta").lock().doc_store = Some(labelled_store("beta-store"));

    svc.drop_caches(None);

    assert!(store_label(&svc, "alpha").is_none());
    assert!(store_label(&svc, "beta").is_none());
}

#[test]
fn test_dropping_a_project_with_no_slot_is_not_an_error() {
    service().drop_caches(Some("never-touched"));
}

/// Python used `__slots__`: one of these exists per open project, so keep it
/// small.  The Rust struct holds exactly its three handles and nothing else.
#[test]
fn test_it_carries_no_per_instance_dict() {
    let handles = std::mem::size_of::<Option<Arc<WorkspaceIndexer>>>()
        + std::mem::size_of::<Option<Arc<FeedbackIndexer>>>()
        + std::mem::size_of::<Option<Arc<dyn VectorStore>>>();
    assert_eq!(std::mem::size_of::<ProjectCaches>(), handles);
    let empty = ProjectCaches::default();
    assert!(empty.doc_store.is_none() && empty.workspace_indexer.is_none() && empty.feedback_indexer.is_none());
}
