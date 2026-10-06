//! Port of `smartloop/model/workers/worker.py`.

use std::sync::Arc;

use serde_json::{Map, Value};

use crate::model::downloader::slp_downloader::SharedLogger;

/// Opaque handle to a loaded model / tokenizer object (`Any` in Python).
pub type AnyHandle = Arc<dyn std::any::Any + Send + Sync>;

/// Shared state for a worker: the model, its tokenizer and a logger.
#[derive(Clone)]
pub struct WorkerBase {
    pub model: AnyHandle,
    pub tokenizer: AnyHandle,
    pub logger: SharedLogger,
}

impl WorkerBase {
    /// Bundle a model, tokenizer and logger.
    pub fn new(model: AnyHandle, tokenizer: AnyHandle, logger: SharedLogger) -> Self {
        Self { model, tokenizer, logger }
    }
}

/// Abstract worker: runs one unit of work with keyword arguments.
pub trait Worker: Send + Sync {
    /// The worker's model/tokenizer/logger.
    fn base(&self) -> &WorkerBase;

    /// Run the work (`run(**kwargs) -> Any | None`).
    fn run(&self, kwargs: Map<String, Value>) -> anyhow::Result<Option<Value>>;
}
