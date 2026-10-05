//! Port of `tests/unit/test_embedding_device.py`.
//!
//! Regression tests for the llama.cpp embedding backend.
//!
//! Dense embeddings run on llama.cpp (GGUF), not PyTorch/sentence-transformers,
//! so the embedding hot path never touches torch's MPS shader JIT — the exact
//! code that segfaulted inside `libtorch_cpu.dylib` in the packaged build.
//!
//! These tests exercise the device→offload mapping and the loader wiring
//! without loading a real GGUF (the backend constructor is stubbed through
//! `load_embedder_with` and the GGUF is pre-placed in the cache dir).

use std::sync::Arc;

use parking_lot::Mutex;
use smartloop::db::llama_embeddings::{
    self, BUNDLE_SUBDIR, EmbedBackend, LlamaEmbedder, LlamaLoadParams, POOLING_TYPE_UNSPECIFIED,
};

/// A backend that never loads anything (the `_FakeLlama` stub).
struct FakeBackend;

impl EmbedBackend for FakeBackend {
    fn embed(&mut self, batch: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(batch.iter().map(|_| vec![0.0, 1.0]).collect())
    }
}

/// A loader that records the parameters it was called with.
fn capturing_loader(
    captured: Arc<Mutex<Option<LlamaLoadParams>>>,
) -> impl Fn(&LlamaLoadParams) -> anyhow::Result<Box<dyn EmbedBackend>> {
    move |params: &LlamaLoadParams| {
        *captured.lock() = Some(params.clone());
        Ok(Box::new(FakeBackend) as Box<dyn EmbedBackend>)
    }
}

mod test_offload_mapping {
    use super::*;

    #[test]
    fn test_cpu_uses_no_gpu_layers() {
        assert_eq!(llama_embeddings::n_gpu_layers_for("cpu"), 0);
    }

    #[test]
    fn test_cpu_is_case_insensitive() {
        assert_eq!(llama_embeddings::n_gpu_layers_for("  CPU "), 0);
    }

    #[test]
    fn test_non_cpu_offloads_fully() {
        // "auto"/"cuda"/anything non-cpu → full Metal/Vulkan offload (-1).
        assert_eq!(llama_embeddings::n_gpu_layers_for("auto"), -1);
        assert_eq!(llama_embeddings::n_gpu_layers_for("cuda"), -1);
    }
}

mod test_load_embedder {
    use super::*;

    #[test]
    fn test_builds_llama_with_embedding_and_mean_pooling() {
        let tmp = tempfile::tempdir().unwrap();
        // Already-cached GGUF: resolution short-circuits before any network.
        std::fs::write(tmp.path().join("bge-m3-Q8_0.gguf"), b"gguf").unwrap();
        let captured = Arc::new(Mutex::new(None));
        let loader = capturing_loader(Arc::clone(&captured));

        let emb = llama_embeddings::load_embedder_with(
            Some(tmp.path().to_str().unwrap()),
            "gpustack/bge-m3-GGUF",
            "bge-m3-Q8_0.gguf",
            "cpu",
            "https://dl.smartloop.ai/embeddings",
            &loader,
        );

        let emb = emb.expect("an embedder");
        let captured = captured.lock().clone().expect("the backend was constructed");
        // The Rust backend is always built in embedding mode; what the
        // loader is handed is the rest of the Python `Llama(...)` kwargs.
        assert_eq!(captured.pooling_type, POOLING_TYPE_UNSPECIFIED);
        assert_eq!(captured.n_gpu_layers, 0); // cpu
        assert_eq!(captured.model_path, tmp.path().join("bge-m3-Q8_0.gguf").to_string_lossy());
        assert_eq!(emb.identity, "llama-cpp:gpustack/bge-m3-GGUF/bge-m3-Q8_0.gguf");
    }

    #[test]
    fn test_unavailable_gguf_returns_none() {
        let loader = |_: &LlamaLoadParams| -> anyhow::Result<Box<dyn EmbedBackend>> {
            anyhow::bail!("the stub cannot load anything")
        };

        // No cache dir to download into, and the Hub is deliberately not
        // consulted — None so callers fall back to TF-IDF rather than
        // crashing.
        assert!(llama_embeddings::load_embedder_with(None, "r", "f.gguf", "cpu", "", &loader).is_none());
    }
}

mod test_bundled_path {
    use super::*;

    #[test]
    fn test_no_meipass_returns_none() {
        // The test binary is not a packaged build: nothing is bundled next
        // to it.
        assert_eq!(llama_embeddings::bundled_gguf_path("bge-m3-Q4_K_M.gguf"), None);
    }

    /// Removes the simulated bundle when the test ends, pass or fail.
    struct BundledFile {
        file: std::path::PathBuf,
        dir: std::path::PathBuf,
        created_dir: bool,
    }

    impl Drop for BundledFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.file);
            if self.created_dir {
                let _ = std::fs::remove_dir(&self.dir);
            }
        }
    }

    #[test]
    fn test_bundled_file_is_preferred_over_download() {
        // Simulate a packaged build whose bundle contains the GGUF: the
        // bundle lives next to the executable, here the test binary.  The
        // name is unique so concurrent runs cannot see each other's file.
        let exe_dir = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
        let bundle = exe_dir.join(BUNDLE_SUBDIR);
        let created_dir = !bundle.exists();
        std::fs::create_dir_all(&bundle).unwrap();
        let filename = format!("bge-m3-bundled-test-{}.gguf", std::process::id());
        let gguf = bundle.join(&filename);
        std::fs::write(&gguf, b"gguf").unwrap();
        let _cleanup = BundledFile { file: gguf.clone(), dir: bundle, created_dir };

        let captured = Arc::new(Mutex::new(None));
        let loader = capturing_loader(Arc::clone(&captured));

        // No cache dir and no download URL: had the bundle been skipped there
        // would be nowhere to get the file from, so it must not download.
        let emb = llama_embeddings::load_embedder_with(None, "gpustack/bge-m3-GGUF", &filename, "cpu", "", &loader);
        assert!(emb.is_some());
        let captured = captured.lock().clone().expect("the backend was constructed");
        assert_eq!(captured.model_path, gguf.to_string_lossy());
    }
}

mod test_encode_normalizes {
    use super::*;

    struct NormFive;

    impl EmbedBackend for NormFive {
        fn embed(&mut self, _batch: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(vec![vec![3.0, 4.0]]) // norm 5
        }
    }

    #[test]
    fn test_encode_l2_normalizes() {
        let emb = LlamaEmbedder::new(Box::new(NormFive), "id");
        let out = emb.encode_one("hello").unwrap();
        let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() <= 1e-5, "norm was {norm}");
    }
}

mod test_get_model_wiring {
    /// `embeddings._get_model` hands the loader the configured repo, file
    /// and bucket.  The singleton's loader is not injectable in the Rust
    /// port, so this pins the settings it forwards — the values the Python
    /// test observed through its stub.
    #[test]
    fn test_get_model_uses_llama_backend() {
        let settings = smartloop::config::AppSettings::defaults();
        assert_eq!(settings.embedding_gguf_repo, "gpustack/bge-m3-GGUF");
        assert_eq!(settings.embedding_gguf_file, "bge-m3-Q4_K_M.gguf");
        // The GGUF comes from our bucket, not the token-gated Hub.
        assert_eq!(settings.embedding_gguf_base_url, "https://dl.smartloop.ai/embeddings");
    }
}

mod test_active_identity {
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_identity_encodes_engine_and_model() {
        let saved: Vec<_> = ["SLP_EMBEDDING_GGUF_REPO", "SLP_EMBEDDING_GGUF_FILE"]
            .iter()
            .map(|k| (*k, std::env::var_os(k)))
            .collect();
        for (k, _) in &saved {
            // SAFETY: `#[serial]`.
            unsafe { std::env::remove_var(k) };
        }

        let ident = smartloop::db::embeddings::active_embedding_identity();

        for (k, v) in saved {
            // SAFETY: `#[serial]`.
            unsafe {
                if let Some(v) = v {
                    std::env::set_var(k, v);
                }
            }
        }
        assert_eq!(ident, "llama-cpp:gpustack/bge-m3-GGUF/bge-m3-Q4_K_M.gguf");
    }
}
