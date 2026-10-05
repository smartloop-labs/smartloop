//! Port of `tests/unit/test_feedback_project_scope.py`.
//!
//! A thumbs-up belongs to the project it was given in.
//!
//! `POST /v1/feedback` names its project in the *body*, so nothing in the path
//! or the query string bound it and the supervisor answered it.  The store was
//! written under the explicit project id and so landed correctly — but the
//! *index* was built through `current_project`, which on an unbound process
//! is whichever project the UI happens to have selected.  Rating an answer in
//! one project therefore embedded it into another's feedback index.

use std::sync::Arc;

use parking_lot::Mutex;

use smartloop::api::services::model_service::{ModelService, ModelServiceError, ModelServiceHooks};
use smartloop::db::feedback_indexer::FeedbackIndexer;
use smartloop::db::store::{Chunk, VectorStore};
use smartloop::workspace::{Project, Workspace};

use crate::shared::model_service_a::{attach_workspace, bare_service};

fn project(pid: &str, current: bool) -> Project {
    Project {
        id: pid.to_string(),
        project_dir: format!("/tmp/{pid}"),
        current,
        name: Some(pid.to_string()),
        ..Default::default()
    }
}

fn service() -> Arc<ModelService> {
    let made = bare_service();
    let mut workspace = Workspace::new("ws-1");
    workspace.workspace_dir = "/tmp/ws-1".to_string();
    workspace.projects = vec![project("alpha", true), project("beta", false)];
    attach_workspace(&made, workspace);
    made
}

/// A store that remembers the project it was opened for.
struct RecordingStore {
    project_dir: String,
}

impl VectorStore for RecordingStore {
    fn project_dir(&self) -> &str {
        &self.project_dir
    }

    fn collection_name(&self) -> &str {
        "feedback_pairs"
    }

    fn base_dir(&self) -> Option<&str> {
        None
    }

    fn add_chunks(&self, _chunks: &[Chunk]) -> anyhow::Result<()> {
        Ok(())
    }

    fn query(&self, _query_text: &str, _n_results: Option<usize>) -> anyhow::Result<Vec<String>> {
        Ok(Vec::new())
    }

    fn delete_document(&self, _document_id: &str) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Capture the project_dir every FeedbackIndexer is constructed with.
#[derive(Clone, Default)]
struct Built(Arc<Mutex<Vec<(Arc<FeedbackIndexer>, String)>>>);

impl Built {
    fn install(svc: &ModelService) -> Built {
        let built = Built::default();
        let record = built.0.clone();
        svc.set_hooks(ModelServiceHooks {
            feedback_indexer_factory: Some(Arc::new(move |project_dir: &str, _base_dir: Option<&str>| {
                let store: Arc<dyn VectorStore> = Arc::new(RecordingStore { project_dir: project_dir.to_string() });
                let indexer = Arc::new(FeedbackIndexer::with_store(store));
                record.lock().push((indexer.clone(), project_dir.to_string()));
                Ok(indexer)
            })),
            ..Default::default()
        });
        built
    }

    /// `indexer.project_dir` of an indexer built through the seam.
    fn project_dir(&self, indexer: &Arc<FeedbackIndexer>) -> String {
        self.0
            .lock()
            .iter()
            .find(|(built, _)| Arc::ptr_eq(built, indexer))
            .map(|(_, dir)| dir.clone())
            .expect("indexer was built through the seam")
    }
}

fn projects(svc: &ModelService) -> Vec<Project> {
    svc.workspace_handle().expect("workspace").read().projects.clone()
}

#[test]
fn test_an_explicit_project_wins_over_the_selection() {
    let service = service();
    let built = Built::install(&service);
    let beta = projects(&service)[1].clone();

    let indexer = service.ensure_feedback_indexer(Some(&beta)).expect("indexer");

    assert_eq!(built.project_dir(&indexer), "/tmp/beta");
}

/// The bug, stated directly: alpha is selected, beta was addressed.
#[test]
fn test_it_is_not_the_selected_project() {
    let service = service();
    let built = Built::install(&service);
    let current = service
        .workspace_handle()
        .expect("workspace")
        .read()
        .current_project()
        .map(|p| p.id.clone());
    assert_eq!(current.as_deref(), Some("alpha"));

    let beta = projects(&service)[1].clone();
    let indexer = service.ensure_feedback_indexer(Some(&beta)).expect("indexer");

    assert_ne!(built.project_dir(&indexer), "/tmp/alpha");
}

/// Unchanged for the chat path, where the request is bound.
#[test]
fn test_no_project_still_falls_back_to_the_binding() {
    let service = service();
    let built = Built::install(&service);

    let indexer = service.ensure_feedback_indexer(None).expect("indexer");

    assert_eq!(built.project_dir(&indexer), "/tmp/alpha");
}

#[test]
fn test_each_project_gets_its_own_cached_index() {
    let service = service();
    let _built = Built::install(&service);
    let all = projects(&service);
    let (alpha, beta) = (&all[0], &all[1]);

    let first = service.ensure_feedback_indexer(Some(alpha)).expect("alpha");
    let second = service.ensure_feedback_indexer(Some(beta)).expect("beta");

    assert!(!Arc::ptr_eq(&first, &second));
    assert!(Arc::ptr_eq(&service.ensure_feedback_indexer(Some(alpha)).expect("alpha again"), &first));
}

#[test]
fn test_no_project_at_all_is_an_error() {
    let service = service();
    let _built = Built::install(&service);
    for project in service.workspace_handle().expect("workspace").write().projects.iter_mut() {
        project.current = false;
    }

    let Err(err) = service.ensure_feedback_indexer(None) else {
        panic!("expected an error with no project");
    };

    match err.downcast_ref::<ModelServiceError>() {
        Some(ModelServiceError::Value(msg)) => assert!(msg.contains("No active project")),
        other => panic!("expected ValueError, got {other:?}"),
    }
}
