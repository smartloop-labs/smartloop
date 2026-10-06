//! Port of `tests/unit/test_embedding_memory.py`.
//!
//! Embedding memory behaviour: bounded encode batches, and unload when idle.
//!
//! An indexing pass used to push memory over: it embedded every chunk in a
//! single llama.cpp call, and the model then stayed resident with nothing to
//! do.  These cover both halves.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ndarray::{Array2, Axis};
use parking_lot::Mutex;
use serial_test::serial;
use smartloop::db::embeddings::{self, EmbeddingModel};
use smartloop::db::llama_embeddings::{EmbedBackend, LlamaEmbedder, MAX_ENCODE_BATCH};

/// Records the size of each embed() call.
#[derive(Clone, Default)]
struct FakeModel {
    calls: Arc<Mutex<Vec<usize>>>,
    closed: Arc<AtomicBool>,
}

impl EmbedBackend for FakeModel {
    fn embed(&mut self, batch: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.calls.lock().push(batch.len());
        Ok(batch.iter().map(|t| vec![t.chars().count() as f32, 1.0]).collect())
    }

    fn close(&mut self) -> anyhow::Result<()> {
        self.closed.store(true, Ordering::SeqCst);
        Ok(())
    }
}

/// The same fake as the singleton holds it (`embeddings._model`).
impl EmbeddingModel for FakeModel {
    fn encode(&self, texts: &[String]) -> anyhow::Result<Array2<f32>> {
        self.calls.lock().push(texts.len());
        Ok(Array2::from_shape_fn((texts.len(), 2), |(i, j)| {
            if j == 0 { texts[i].chars().count() as f32 } else { 1.0 }
        }))
    }

    fn close(&self) -> anyhow::Result<()> {
        self.closed.store(true, Ordering::SeqCst);
        Ok(())
    }
}

mod test_encode_batching {
    use super::*;

    #[test]
    fn test_large_input_is_sliced_into_bounded_calls() {
        let model = FakeModel::default();
        let embedder = LlamaEmbedder::new(Box::new(model.clone()), "test");

        let n = MAX_ENCODE_BATCH * 2 + 5;
        let texts: Vec<String> = (0..n).map(|i| format!("chunk {i}")).collect();
        let out = embedder.encode(&texts).unwrap();

        let calls = model.calls.lock().clone();
        assert!(calls.iter().copied().max().unwrap() <= MAX_ENCODE_BATCH);
        assert_eq!(calls.iter().sum::<usize>(), n);
        assert_eq!(out.dim(), (n, 2));
    }

    /// Vectors must come back in input order, one per text, normalized.
    #[test]
    fn test_slicing_does_not_change_the_result() {
        let texts: Vec<String> = (1..MAX_ENCODE_BATCH * 2).map(|i| "x".repeat(i)).collect();
        let embedder = LlamaEmbedder::new(Box::new(FakeModel::default()), "test");

        let out = embedder.encode(&texts).unwrap();

        assert_eq!(out.nrows(), texts.len());
        for row in out.axis_iter(Axis(0)) {
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "row norm {norm}");
        }
        // The fake encodes length in the first component, so order is
        // checkable after normalization by comparing the ratio it implies.
        assert!(out[[0, 0]] < out[[out.nrows() - 1, 0]]);
    }

    #[test]
    fn test_single_string_still_returns_one_vector() {
        let out = LlamaEmbedder::new(Box::new(FakeModel::default()), "test").encode_one("hello").unwrap();
        assert_eq!(out.ndim(), 1);
    }

    #[test]
    fn test_empty_input_does_not_call_the_model() {
        let model = FakeModel::default();
        let out = LlamaEmbedder::new(Box::new(model.clone()), "test").encode(&[]).unwrap();

        assert!(model.calls.lock().is_empty());
        assert_eq!(out.nrows(), 0);
    }
}

mod test_unload {
    use super::*;

    /// Never leave a fake in the module singleton (the Python autouse
    /// `_restore` fixture).
    struct Restore {
        before: (Option<Arc<dyn EmbeddingModel>>, bool),
    }

    impl Restore {
        fn new() -> Self {
            Self { before: (embeddings::model(), embeddings::model_load_attempted()) }
        }
    }

    impl Drop for Restore {
        fn drop(&mut self) {
            embeddings::set_model(self.before.0.take());
            embeddings::set_model_load_attempted(self.before.1);
        }
    }

    #[test]
    #[serial(embeddings)]
    fn test_unload_closes_and_clears_the_singleton() {
        let _restore = Restore::new();
        let model = FakeModel::default();
        embeddings::set_model(Some(Arc::new(model.clone())));
        embeddings::set_model_load_attempted(true);

        embeddings::unload();

        assert!(model.closed.load(Ordering::SeqCst));
        assert!(embeddings::model().is_none());
    }

    /// The 'already tried' flag must be cleared, or unloading is permanent
    /// and everything afterwards silently falls back to TF-IDF.
    #[test]
    #[serial(embeddings)]
    fn test_unload_allows_a_later_reload() {
        let _restore = Restore::new();
        embeddings::set_model(Some(Arc::new(FakeModel::default())));
        embeddings::set_model_load_attempted(true);

        embeddings::unload();

        assert!(!embeddings::model_load_attempted());
    }

    #[test]
    #[serial(embeddings)]
    fn test_unload_is_safe_when_nothing_is_loaded() {
        let _restore = Restore::new();
        embeddings::set_model(None);
        embeddings::set_model_load_attempted(false);

        embeddings::unload(); // must not raise
    }

    #[test]
    #[serial(embeddings)]
    fn test_unload_survives_a_failing_close() {
        struct Broken;

        impl EmbeddingModel for Broken {
            fn encode(&self, _texts: &[String]) -> anyhow::Result<Array2<f32>> {
                unreachable!("not encoded")
            }

            fn close(&self) -> anyhow::Result<()> {
                anyhow::bail!("metal teardown blew up")
            }
        }

        let _restore = Restore::new();
        embeddings::set_model(Some(Arc::new(Broken)));
        embeddings::unload();

        assert!(embeddings::model().is_none());
    }
}

/// No sequence may exceed n_ubatch — llama.cpp ggml_abort()s rather than
/// degrading, which takes the process with it.
mod test_input_clamp {
    use smartloop::db::llama_embeddings::{EMBED_N_CTX, MAX_INPUT_CHARS};
    use smartloop::db::workspace_indexer::CHUNK_SIZE;

    use super::*;

    #[test]
    fn test_oversized_input_is_truncated_before_reaching_the_model() {
        struct Seen(Arc<Mutex<Vec<usize>>>);

        impl EmbedBackend for Seen {
            fn embed(&mut self, batch: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
                self.0.lock().extend(batch.iter().map(|t| t.chars().count()));
                Ok(batch.iter().map(|_| vec![1.0, 0.0]).collect())
            }
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        LlamaEmbedder::new(Box::new(Seen(seen.clone())), "test")
            .encode(&["x".repeat(MAX_INPUT_CHARS * 3), "short".to_string()])
            .unwrap();

        assert!(seen.lock().iter().copied().max().unwrap() <= MAX_INPUT_CHARS);
    }

    /// The clamp and the indexers' CHUNK_SIZE must stay consistent: a chunk
    /// that fits the indexer must fit the embedder untruncated.
    #[test]
    #[allow(clippy::absurd_extreme_comparisons)]
    fn test_context_is_large_enough_for_a_full_chunk() {
        assert!(MAX_INPUT_CHARS >= CHUNK_SIZE);
        // Worst case is roughly one token per character; the context must
        // still cover it or llama.cpp aborts.
        assert!(EMBED_N_CTX as usize >= MAX_INPUT_CHARS);
    }
}

mod test_unload_after_response {
    use smartloop::api::services::model_service::ModelService;

    use super::*;

    /// Clears the unload override however the test ends.
    struct ClearHook;

    impl Drop for ClearHook {
        fn drop(&mut self) {
            embeddings::set_unload_hook(None);
        }
    }

    /// Retrieval finishes long before the response does, so the embedder is
    /// idle from then until the next question — same as the auxiliary LLMs.
    #[test]
    #[serial(embeddings)]
    fn test_auxiliary_unload_also_drops_the_embedder() {
        let calls: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
        let _clear = ClearHook;
        let recorder = calls.clone();
        embeddings::set_unload_hook(Some(Arc::new(move || {
            recorder.lock().push("unload");
            Ok(())
        })));

        // A bare instance has no vision/large/capability model loaded, so
        // those unloads are no-ops (the Python test stubs them out).
        let svc = ModelService::new_bare();
        svc.unload_auxiliary_models(true);

        assert_eq!(*calls.lock(), vec!["unload"]);
    }

    #[test]
    #[serial(embeddings)]
    fn test_a_failing_unload_does_not_break_the_response() {
        let _clear = ClearHook;
        embeddings::set_unload_hook(Some(Arc::new(|| anyhow::bail!("teardown blew up"))));

        let svc = ModelService::new_bare();
        svc.unload_auxiliary_models(true); // must not raise
    }
}
