//! Port of `tests/conftest.py` (shared fixtures).

// ===================================================================== db unit
// Shared fixtures added by the `db` unit: a recording `VectorStore` (the
// Python tests' `MagicMock()` store) and a tracing capture (the Python tests'
// `MagicMock()` logger / `caplog`).

pub mod db {
    use std::sync::Arc;

    use parking_lot::Mutex;
    use serde_json::{Map, Value};
    use smartloop::db::store::{Chunk, ChunkResult, VectorStore};

    /// A `VectorStore` that records every call and stores nothing — the
    /// Python tests' `MagicMock()` store.
    #[derive(Default)]
    pub struct RecordingStore {
        /// Every `add_chunks` call's chunk list, in call order.
        pub added: Mutex<Vec<Vec<Chunk>>>,
        /// Every `delete_document` call's document id, in call order.
        pub deleted: Mutex<Vec<String>>,
        /// Every `query_with_metadata` call's query text.
        pub queries: Mutex<Vec<String>>,
    }

    impl RecordingStore {
        pub fn new() -> Arc<Self> {
            Arc::new(Self::default())
        }

        /// The chunks of the most recent `add_chunks` call (`call_args[0][0]`).
        pub fn last_added(&self) -> Option<Vec<Chunk>> {
            self.added.lock().last().cloned()
        }

        /// Number of `add_chunks` calls.
        pub fn add_count(&self) -> usize {
            self.added.lock().len()
        }

        /// Number of `delete_document` calls.
        pub fn delete_count(&self) -> usize {
            self.deleted.lock().len()
        }
    }

    impl VectorStore for RecordingStore {
        fn project_dir(&self) -> &str {
            ""
        }

        fn collection_name(&self) -> &str {
            "documents"
        }

        fn base_dir(&self) -> Option<&str> {
            None
        }

        fn add_chunks(&self, chunks: &[Chunk]) -> anyhow::Result<()> {
            self.added.lock().push(chunks.to_vec());
            Ok(())
        }

        fn query(&self, query_text: &str, _n_results: Option<usize>) -> anyhow::Result<Vec<String>> {
            self.queries.lock().push(query_text.to_string());
            Ok(Vec::new())
        }

        fn query_with_metadata(
            &self,
            query_text: &str,
            _n_results: Option<usize>,
            _max_distance: f64,
            _where: Option<&Map<String, Value>>,
        ) -> anyhow::Result<Vec<ChunkResult>> {
            self.queries.lock().push(query_text.to_string());
            Ok(Vec::new())
        }

        fn delete_document(&self, document_id: &str) -> anyhow::Result<()> {
            self.deleted.lock().push(document_id.to_string());
            Ok(())
        }
    }

    /// One captured tracing event.
    #[derive(Debug, Clone, PartialEq)]
    pub struct LogRecord {
        pub level: tracing::Level,
        pub target: String,
        pub message: String,
    }

    /// Captured events, shared with the subscriber.
    #[derive(Clone, Default)]
    pub struct LogCapture {
        records: Arc<Mutex<Vec<LogRecord>>>,
    }

    impl LogCapture {
        /// All captured records.
        pub fn records(&self) -> Vec<LogRecord> {
            self.records.lock().clone()
        }

        /// Messages logged at `level` (optionally restricted to `target`).
        pub fn messages_at(&self, level: tracing::Level, target: Option<&str>) -> Vec<String> {
            self.records
                .lock()
                .iter()
                .filter(|r| r.level == level && target.is_none_or(|t| r.target == t))
                .map(|r| r.message.clone())
                .collect()
        }

        /// `logger.info.assert_any_call(message)`.
        pub fn has_info(&self, message: &str) -> bool {
            self.messages_at(tracing::Level::INFO, None).iter().any(|m| m == message)
        }
    }

    struct MessageVisitor<'a>(&'a mut String);

    impl tracing::field::Visit for MessageVisitor<'_> {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" {
                *self.0 = format!("{value:?}");
            }
        }

        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            if field.name() == "message" {
                *self.0 = value.to_string();
            }
        }
    }

    struct CaptureLayer {
        records: Arc<Mutex<Vec<LogRecord>>>,
    }

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for CaptureLayer {
        fn on_event(&self, event: &tracing::Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
            let mut message = String::new();
            event.record(&mut MessageVisitor(&mut message));
            self.records.lock().push(LogRecord {
                level: *event.metadata().level(),
                target: event.metadata().target().to_string(),
                message,
            });
        }
    }

    /// Run `f` with every tracing event on this thread captured.
    pub fn capture_logs<R>(f: impl FnOnce() -> R) -> (R, LogCapture) {
        use tracing_subscriber::layer::SubscriberExt;

        let capture = LogCapture::default();
        let subscriber = tracing_subscriber::registry()
            .with(CaptureLayer { records: capture.records.clone() });
        let result = tracing::subscriber::with_default(subscriber, f);
        (result, capture)
    }
}
// ================================================================= end db unit
