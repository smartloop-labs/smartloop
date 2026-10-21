//! A restart mid-index resumes the documents it left unfinished.
//!
//! The indexer's queue lives in memory, and its folder scan only enqueues
//! files it has never registered.  An agent stopped while indexing came back
//! with every `pending` document, and the ones caught `indexing`, still in
//! the workspace under that status and nothing left to pick them up: 263 of
//! 402 documents sat there for good.

use std::sync::Arc;

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

fn document(id: &str, status: DocumentStatus) -> Document {
    Document { id: id.to_string(), path: format!("documents/{id}.md"), name: None, summary: None, status }
}

fn task(project: Option<Project>) -> Arc<IndexerTask> {
    let workspace = Arc::new(RwLock::new(Workspace::new("test-workspace")));
    let logger: TaskLogger = Arc::new(NullLogger);
    let task = IndexerTask::new(workspace, logger, None, None, TaskOptions::default());
    task.set_project_provider(Some(Arc::new(move || project.clone())));
    task
}

#[test]
fn test_unfinished_documents_are_queued_again() {
    let project = Project {
        documents: vec![
            document("pending", DocumentStatus::Pending),
            document("converting", DocumentStatus::Converting),
            document("indexing", DocumentStatus::Indexing),
            document("indexed", DocumentStatus::Indexed),
            document("processed", DocumentStatus::Processed),
            document("failed", DocumentStatus::Failed),
        ],
        ..Default::default()
    };
    let task = task(Some(project));

    assert_eq!(task.requeue_unfinished(), 3);
    assert_eq!(task.queue_size(), 3);
}

#[test]
fn test_a_finished_project_queues_nothing() {
    let project = Project { documents: vec![document("indexed", DocumentStatus::Indexed)], ..Default::default() };
    let task = task(Some(project));

    assert_eq!(task.requeue_unfinished(), 0);
    assert_eq!(task.queue_size(), 0);
}

#[test]
fn test_no_project_queues_nothing() {
    let task = task(None);

    assert_eq!(task.requeue_unfinished(), 0);
}
