//! Port of `smartloop/tasks/__init__.py`.
//!
//! Tasks — isolated units of work that run in background threads.
//!
//! Training/build tasks (`BuildTask`, `TrainTask`) live in the optional
//! `smartloop_trainer` package, which has no Rust port; they are absent here
//! exactly as when the Python import fails.

pub mod base_task;
pub mod documents_manifest;
pub mod downloader_task;
pub mod indexer_task;
pub mod mcp_task;
pub mod task_queue;

pub use base_task::{BaseTask, TaskResult, TaskStatus};
pub use documents_manifest::DocumentsManifest;
pub use downloader_task::DownloaderTask;
pub use indexer_task::IndexerTask;
pub use mcp_task::MCPTask;
pub use task_queue::TaskQueue;
