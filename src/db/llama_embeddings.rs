//! Port of `smartloop/db/llama_embeddings.py`: llama.cpp-backed dense
//! embeddings.
//!
//! Runs a GGUF embedding model on the same llama.cpp backend used for
//! generation.  Configured via `slp_embedding_gguf_repo` /
//! `slp_embedding_gguf_file` and `slp_embedding_device`.

use ndarray::{Array1, Array2};
use parking_lot::Mutex;

use crate::db::embeddings::EmbeddingModel;

/// Subdir next to the executable (packaged builds) where a bundled GGUF lives.
pub const BUNDLE_SUBDIR: &str = "embeddings";

/// `LLAMA_POOLING_TYPE_UNSPECIFIED`: let the GGUF drive pooling (CLS for
/// BGE-M3, MEAN for E5/GTE).
pub const POOLING_TYPE_UNSPECIFIED: i32 = -1;

/// Context window (and n_batch / n_ubatch) for the embedding model.  Safe only
/// because [`LlamaEmbedder::encode`] truncates inputs to [`MAX_INPUT_CHARS`].
pub const EMBED_N_CTX: u32 = 2048;

/// Hard cap on a single input, in characters, so no sequence can exceed
/// n_ubatch (which would `ggml_abort()` the process).
pub const MAX_INPUT_CHARS: usize = 2000;

/// Chunks handed to the backend per `embed()` call, bounding peak memory.
pub const MAX_ENCODE_BATCH: usize = 16;

/// First-run GGUF download parameters.
pub const DOWNLOAD_CHUNK_SIZE: usize = 8 * 1024 * 1024;
pub const DOWNLOAD_CONNECT_TIMEOUT: f64 = 30.0;
pub const DOWNLOAD_READ_TIMEOUT: f64 = 60.0;

/// The raw embedding backend `LlamaEmbedder` wraps (a llama.cpp context in
/// production, a fake in tests).  Returns one pooled vector per input.
pub trait EmbedBackend: Send {
    fn embed(&mut self, batch: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;

    /// Free the backend's resources.
    fn close(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Parameters `load_embedder` passes to the backend constructor (mirrors the
/// Python `Llama(...)` kwargs).
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaLoadParams {
    pub model_path: String,
    pub pooling_type: i32,
    pub n_gpu_layers: i32,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
}

/// Constructor for an embedding backend from load parameters (test seam for
/// stubbing `llama_cpp.Llama`).
pub type BackendLoader = dyn Fn(&LlamaLoadParams) -> anyhow::Result<Box<dyn EmbedBackend>>;

/// Adapts a llama.cpp embedding context to the `.encode()` call site.
pub struct LlamaEmbedder {
    /// A llama.cpp context is NOT thread-safe; concurrent embed() calls are
    /// serialised through this lock.
    model: Mutex<Option<Box<dyn EmbedBackend>>>,
    /// Stored in the sqlite index so switching model/engine triggers a re-embed.
    pub identity: String,
}

impl LlamaEmbedder {
    pub fn new(model: Box<dyn EmbedBackend>, identity: impl Into<String>) -> Self {
        Self { model: Mutex::new(Some(model)), identity: identity.into() }
    }

    /// Encode a batch into an `(N, dim)` L2-normalised matrix.  Inputs longer
    /// than [`MAX_INPUT_CHARS`] are truncated (with a warning); batches are
    /// sliced into [`MAX_ENCODE_BATCH`]-sized backend calls.  An empty input
    /// returns a `(0, 0)` matrix without calling the backend.
    pub fn encode(&self, texts: &[String]) -> anyhow::Result<Array2<f32>> {
        let _ = (texts, &self.model);
        todo!()
    }

    /// Encode a single string into one 1-D vector.
    pub fn encode_one(&self, text: &str) -> anyhow::Result<Array1<f32>> {
        let _ = text;
        todo!()
    }

    /// Free the llama.cpp context.  Errors from the backend are swallowed.
    pub fn close(&self) {
        todo!()
    }
}

impl EmbeddingModel for LlamaEmbedder {
    fn encode(&self, texts: &[String]) -> anyhow::Result<Array2<f32>> {
        LlamaEmbedder::encode(self, texts)
    }

    fn close(&self) -> anyhow::Result<()> {
        LlamaEmbedder::close(self);
        Ok(())
    }
}

/// Map the `embedding_device` setting to llama.cpp offload: `cpu` → 0 layers,
/// anything else → -1 (offload fully).
pub fn n_gpu_layers_for(device: &str) -> i32 {
    let _ = device;
    todo!()
}

/// Return the packaged GGUF path (`<exe dir>/embeddings/<filename>`) if it
/// exists, else `None`.
pub fn bundled_gguf_path(filename: &str) -> Option<String> {
    let _ = filename;
    todo!()
}

/// An already-downloaded copy in the HuggingFace cache layout
/// (`<cache_dir>/models--<org>--<repo>/snapshots/<rev>/<filename>`), or `None`.
/// Filesystem lookup only.
pub fn hf_cached_gguf_path(cache_dir: Option<&str>, repo_id: &str, filename: &str) -> Option<String> {
    let _ = (cache_dir, repo_id, filename);
    todo!()
}

/// Download `filename` from our public bucket (`base_url`) into `cache_dir`,
/// or `None` on any failure.  Uses a `.part` file renamed on success and
/// checks `Content-Length`; an existing cached file short-circuits the network.
/// Blocking.
pub fn cdn_gguf_path(cache_dir: Option<&str>, base_url: &str, filename: &str) -> Option<String> {
    let _ = (cache_dir, base_url, filename);
    todo!()
}

/// Load the GGUF embedding model — bundled copy first, else an existing HF
/// cache copy, else our bucket.  Returns `None` on any failure so callers fall
/// back to TF-IDF.  (Python defaults: `device="cpu"`, `base_url=""`.)
pub fn load_embedder(
    cache_dir: Option<&str>,
    repo_id: &str,
    filename: &str,
    device: &str,
    base_url: &str,
) -> Option<LlamaEmbedder> {
    let _ = (cache_dir, repo_id, filename, device, base_url);
    todo!()
}

/// [`load_embedder`] with an injectable backend constructor (test seam).
pub fn load_embedder_with(
    cache_dir: Option<&str>,
    repo_id: &str,
    filename: &str,
    device: &str,
    base_url: &str,
    loader: &BackendLoader,
) -> Option<LlamaEmbedder> {
    let _ = (cache_dir, repo_id, filename, device, base_url, loader);
    todo!()
}
