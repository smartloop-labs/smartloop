//! Port of `smartloop/db/llama_embeddings.py`: llama.cpp-backed dense
//! embeddings.
//!
//! Runs a GGUF embedding model on the same llama.cpp backend used for
//! generation, so embeddings offload to the GPU like the generative models
//! do.  Configured via `slp_embedding_gguf_repo` / `slp_embedding_gguf_file`
//! and `slp_embedding_device`.
//!
//! The public surface mirrors the single method the sqlite store uses —
//! `encode(texts)` returning a float32 matrix — so `embeddings::get_model`
//! can hand it out behind the [`EmbeddingModel`] trait.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::Duration;

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::LlamaCppError;
use ndarray::{Array1, Array2};
use once_cell::sync::OnceCell;
use parking_lot::Mutex;

use crate::db::embeddings::EmbeddingModel;

/// Logger name of the Python module.
const LOG_TARGET: &str = "smartloop.db.llama_embeddings";

/// Subdir next to the executable (packaged builds) where a bundled GGUF lives.
pub const BUNDLE_SUBDIR: &str = "embeddings";

/// `LLAMA_POOLING_TYPE_UNSPECIFIED`: let the GGUF drive pooling (CLS for
/// BGE-M3, MEAN for E5/GTE).  Forcing a specific type (e.g. MEAN) yields
/// wrong vectors for CLS models like BGE-M3.
pub const POOLING_TYPE_UNSPECIFIED: i32 = -1;

/// Context window (and n_batch / n_ubatch) for the embedding model.
///
/// BGE-M3 trains to 8192, but llama.cpp allocates its compute buffer for
/// n_ubatch tokens at load and holds it, so a context sized to the model's
/// maximum cost gigabytes resident regardless of what was being embedded.
/// Retrieval chunks are capped at 2000 characters by both indexers
/// (`CHUNK_SIZE`), well under 2048 tokens even for scripts that tokenize near
/// one-to-one.
///
/// Lowering this is only safe because [`LlamaEmbedder::encode`] truncates to
/// [`MAX_INPUT_CHARS`]: BGE-M3 is a non-causal encoder, so llama.cpp runs each
/// sequence in a single micro-batch and `ggml_abort()`s if one exceeds
/// n_ubatch.
pub const EMBED_N_CTX: u32 = 2048;

/// Hard cap on a single input, in characters.  Chosen so the worst-case token
/// count stays under [`EMBED_N_CTX`] even when a text tokenizes at roughly one
/// token per character (CJK, dense code).  Both chunkers already cap at this,
/// so in practice nothing is truncated — this exists to make the abort
/// impossible rather than to reshape normal input.
pub const MAX_INPUT_CHARS: usize = 2000;

/// Chunks handed to llama.cpp per embed call.  Slicing bounds the peak memory
/// to a fixed cost regardless of how much is being indexed; the model is
/// loaded once either way, so the only price is a few more calls.
pub const MAX_ENCODE_BATCH: usize = 16;

/// First-run GGUF download.  The read timeout bounds the idle gap between
/// received bytes (a stalled transfer is detected instead of hanging), not
/// the total transfer time — a ~400MB model may take as long as it takes.
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
    /// A llama.cpp context is NOT thread-safe: the retrieval fan-out queries
    /// the documents/feedback/workspace stores from parallel DAG worker
    /// threads, all sharing this one embedder — concurrent embed calls on a
    /// single context segfault.  Serialised through this lock (embedding is
    /// fast).  `None` once closed.
    model: Mutex<Option<Box<dyn EmbedBackend>>>,
    /// Stored in the sqlite index so switching model/engine triggers a
    /// re-embed (the vector space is not comparable across models).
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
        if texts.is_empty() {
            return Ok(Array2::zeros((0, 0)));
        }

        // Guarantee no sequence exceeds n_ubatch.  Exceeding it is not a bad
        // embedding, it is ggml_abort() taking the process down, so this is
        // enforced here rather than trusted to every caller.
        let mut clipped = 0usize;
        let batch: Vec<String> = texts
            .iter()
            .map(|text| {
                if text.chars().count() > MAX_INPUT_CHARS {
                    clipped += 1;
                    text.chars().take(MAX_INPUT_CHARS).collect()
                } else {
                    text.clone()
                }
            })
            .collect();
        if clipped > 0 {
            tracing::warn!(
                target: "smartloop.db.llama_embeddings",
                "[embeddings] truncated {} input(s) to {} chars",
                clipped,
                MAX_INPUT_CHARS
            );
        }

        // One vector per input (pooling is enabled).  Sliced so peak memory
        // tracks MAX_ENCODE_BATCH rather than the batch length.
        let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(batch.len());
        {
            let mut guard = self.model.lock();
            let model = guard
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("embedding model is closed"))?;
            for slice in batch.chunks(MAX_ENCODE_BATCH) {
                vecs.extend(model.embed(slice)?);
            }
        }

        let dim = vecs.first().map(Vec::len).unwrap_or(0);
        if vecs.iter().any(|v| v.len() != dim) {
            anyhow::bail!("embedding backend returned vectors of differing lengths");
        }
        let mut arr = Array2::<f32>::zeros((vecs.len(), dim));
        for (i, v) in vecs.iter().enumerate() {
            // L2-normalise so cosine similarity == dot product (the sqlite
            // store ranks by dot/cosine), matching what sentence-transformers
            // returns for these retrieval models.
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for (j, x) in v.iter().enumerate() {
                arr[[i, j]] = x / norm;
            }
        }
        Ok(arr)
    }

    /// Encode a single string into one 1-D vector.
    pub fn encode_one(&self, text: &str) -> anyhow::Result<Array1<f32>> {
        let arr = self.encode(&[text.to_string()])?;
        Ok(arr.row(0).to_owned())
    }

    /// Free the llama.cpp context.  Errors from the backend are swallowed.
    ///
    /// Must run at shutdown when the model is GPU-offloaded: a GPU context
    /// still live at process exit after multi-threaded use can abort during
    /// teardown.  Closing explicitly avoids it.
    pub fn close(&self) {
        let mut guard = self.model.lock();
        if let Some(mut model) = guard.take() {
            let _ = model.close();
        }
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

/// Map the `embedding_device` setting to llama.cpp offload.
///
/// `cpu` keeps embeddings off the GPU (0 layers) so they don't compete with
/// the generative model copies for VRAM; anything else offloads fully (-1).
pub fn n_gpu_layers_for(device: &str) -> i32 {
    if device.trim().to_lowercase() == "cpu" {
        0
    } else {
        -1
    }
}

/// Return the packaged GGUF path (`<exe dir>/embeddings/<filename>`) if it
/// exists, else `None`.
///
/// A packaged build places the embedding GGUF next to the executable; when
/// present it is used directly so the install needs no first-run download
/// and works fully offline.
pub fn bundled_gguf_path(filename: &str) -> Option<String> {
    let exe = std::env::current_exe().ok()?;
    let base = exe.parent()?;
    let path = base.join(BUNDLE_SUBDIR).join(filename);
    if path.is_file() {
        Some(path.to_string_lossy().into_owned())
    } else {
        None
    }
}

/// An already-downloaded copy in the HuggingFace cache layout, or `None`.
///
/// Installs that predate the bucket download hold the GGUF under
/// `<cache_dir>/models--<org>--<repo>/snapshots/<rev>/<filename>`.  Reusing
/// it keeps upgrades from re-fetching ~400MB that is already on disk.  This
/// is a filesystem lookup only — no Hub call, no token.
pub fn hf_cached_gguf_path(cache_dir: Option<&str>, repo_id: &str, filename: &str) -> Option<String> {
    let cache_dir = cache_dir.filter(|c| !c.is_empty())?;
    let repo_dir = Path::new(cache_dir).join(format!("models--{}", repo_id.replace('/', "--")));
    let snapshots = repo_dir.join("snapshots");
    if !snapshots.is_dir() {
        return None;
    }
    let mut revs: Vec<String> = std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    revs.sort();
    revs.into_iter()
        .map(|rev| snapshots.join(rev).join(filename))
        .find(|candidate| candidate.is_file())
        .map(|p| p.to_string_lossy().into_owned())
}

/// Download `filename` from our public bucket (`base_url`) into `cache_dir`,
/// or `None` on any failure.  Blocking.
///
/// Preferred over the HuggingFace Hub: the Hub gates GGUF downloads behind a
/// user token and is slow from most regions, whereas this is an
/// unauthenticated object served from the same bucket as the release
/// archives.  Falls through (returns `None`) on any failure.
///
/// The file lands via a `.part` file renamed on success, so an interrupted
/// download never leaves a truncated GGUF that llama.cpp would later fail to
/// load — and the size is checked against `Content-Length` first.  An
/// existing cached file short-circuits the network.
pub fn cdn_gguf_path(cache_dir: Option<&str>, base_url: &str, filename: &str) -> Option<String> {
    let cache_dir = cache_dir.filter(|c| !c.is_empty())?;
    if base_url.is_empty() {
        return None;
    }

    let target = Path::new(cache_dir).join(filename);
    if target.is_file() {
        tracing::info!(target: LOG_TARGET, "[embeddings] Using cached GGUF {}", target.display());
        return Some(target.to_string_lossy().into_owned());
    }

    let url = format!("{}/{}", base_url.trim_end_matches('/'), filename);
    let partial = PathBuf::from(format!("{}.part", target.to_string_lossy()));
    match download_to(cache_dir, &url, &partial, &target, filename) {
        Ok(written) => {
            tracing::info!(
                target: LOG_TARGET,
                "[embeddings] Downloaded GGUF {} ({} bytes)",
                filename,
                written
            );
            Some(target.to_string_lossy().into_owned())
        }
        Err(exc) => {
            tracing::warn!(
                target: LOG_TARGET,
                "[embeddings] Failed to fetch GGUF from {}: {}",
                url,
                exc
            );
            let _ = std::fs::remove_file(&partial);
            None
        }
    }
}

/// Stream `url` into `partial`, verify the length, then move it to `target`.
/// Returns the number of bytes written.
///
/// Runs the async client on a dedicated thread with its own runtime so it is
/// safe to call from any context (including from inside a tokio runtime).
fn download_to(
    cache_dir: &str,
    url: &str,
    partial: &Path,
    target: &Path,
    filename: &str,
) -> anyhow::Result<u64> {
    let _ = filename;
    std::fs::create_dir_all(cache_dir)?;
    tracing::info!(target: LOG_TARGET, "[embeddings] Downloading GGUF from {}", url);

    let url_owned = url.to_string();
    let partial_owned = partial.to_path_buf();
    let handle = std::thread::Builder::new()
        .name("slp-gguf-download".to_string())
        .spawn(move || -> anyhow::Result<u64> {
            let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
            runtime.block_on(stream_download(&url_owned, &partial_owned))
        })?;
    let written = handle
        .join()
        .map_err(|_| anyhow::anyhow!("download thread panicked"))??;

    std::fs::rename(partial, target)?;
    Ok(written)
}

/// The download itself: follow redirects, fail on an error status, write the
/// body in chunks and check it against `Content-Length`.
async fn stream_download(url: &str, partial: &Path) -> anyhow::Result<u64> {
    use std::io::Write as _;

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs_f64(DOWNLOAD_CONNECT_TIMEOUT))
        .read_timeout(Duration::from_secs_f64(DOWNLOAD_READ_TIMEOUT))
        // Transparent decompression would make the body length disagree
        // with Content-Length; a GGUF is fetched byte for byte.
        .no_gzip()
        .build()?;
    let mut response = client.get(url).send().await?.error_for_status()?;
    let expected: Option<u64> = response
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()))
        .and_then(|s| s.parse().ok());

    let mut written: u64 = 0;
    {
        let file = std::fs::File::create(partial)?;
        let mut writer = std::io::BufWriter::with_capacity(DOWNLOAD_CHUNK_SIZE, file);
        while let Some(chunk) = response.chunk().await? {
            writer.write_all(&chunk)?;
            written += chunk.len() as u64;
        }
        writer.flush()?;
    }

    if let Some(expected) = expected {
        if written != expected {
            anyhow::bail!("truncated download: got {written} of {expected} bytes");
        }
    }
    Ok(written)
}

/// Load the GGUF embedding model — bundled copy first, else download.
///
/// Resolution order is bundled → an existing HuggingFace-cache copy on disk →
/// our public bucket (`base_url`).  The Hub itself is deliberately not
/// called: it gates GGUF downloads behind a user token and is slow from most
/// regions, which is the whole reason the model is mirrored to our bucket.
/// Returns `None` on any failure so callers fall back to TF-IDF.  (Python
/// defaults: `device="cpu"`, `base_url=""`.)
pub fn load_embedder(
    cache_dir: Option<&str>,
    repo_id: &str,
    filename: &str,
    device: &str,
    base_url: &str,
) -> Option<LlamaEmbedder> {
    load_embedder_with(cache_dir, repo_id, filename, device, base_url, &|params| {
        LlamaCppEmbedBackend::load(params).map(|b| Box::new(b) as Box<dyn EmbedBackend>)
    })
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
    let mut model_path = bundled_gguf_path(filename);
    if model_path.is_some() {
        tracing::info!(target: LOG_TARGET, "[embeddings] Using bundled GGUF {}", filename);
    } else {
        model_path = hf_cached_gguf_path(cache_dir, repo_id, filename)
            .or_else(|| cdn_gguf_path(cache_dir, base_url, filename));
    }
    let Some(model_path) = model_path else {
        tracing::warn!(
            target: LOG_TARGET,
            "[embeddings] No GGUF available for {} — falling back to TF-IDF",
            filename
        );
        return None;
    };

    let params = LlamaLoadParams {
        model_path,
        pooling_type: POOLING_TYPE_UNSPECIFIED,
        n_gpu_layers: n_gpu_layers_for(device),
        n_ctx: EMBED_N_CTX,
        // BGE-M3 is a non-causal encoder: llama.cpp processes each sequence
        // in a single micro-batch (no KV-cache splitting), so n_ubatch must
        // be >= the longest input or ggml_abort()s.  encode() truncates to
        // MAX_INPUT_CHARS to hold that invariant.
        n_batch: EMBED_N_CTX,
        n_ubatch: EMBED_N_CTX,
    };
    let backend = match loader(&params) {
        Ok(backend) => backend,
        Err(exc) => {
            tracing::warn!(
                target: LOG_TARGET,
                "[embeddings] Failed to load GGUF embedder {}: {}",
                filename,
                exc
            );
            return None;
        }
    };

    let identity = format!("llama-cpp:{repo_id}/{filename}");
    tracing::info!(
        target: LOG_TARGET,
        "[embeddings] Loaded {} (llama-cpp, n_gpu_layers={})",
        identity,
        n_gpu_layers_for(device)
    );
    Some(LlamaEmbedder::new(backend, identity))
}

// ------------------------------------------------------------ llama.cpp

/// The process-wide llama.cpp backend.
///
/// llama.cpp may be initialised only once per process.  If another part of
/// the crate already initialised it, `LlamaBackend` is a zero-sized proof
/// token, so one is minted here and kept in this static — never dropped, so
/// it never frees the backend out from under its owner.
pub fn shared_backend() -> anyhow::Result<&'static LlamaBackend> {
    static BACKEND: OnceCell<LlamaBackend> = OnceCell::new();
    BACKEND.get_or_try_init(|| match LlamaBackend::init() {
        Ok(backend) => Ok(backend),
        Err(LlamaCppError::BackendAlreadyInitialized) => Ok(LlamaBackend {}),
        Err(e) => Err(anyhow::anyhow!("llama.cpp backend init failed: {e}")),
    })
}

/// A llama.cpp embedding context (`Llama(embedding=True, ...)`).
pub struct LlamaCppEmbedBackend {
    /// Declared before `model` so it drops first: the context borrows the
    /// model.
    ctx: Option<LlamaContext<'static>>,
    model: Option<Box<LlamaModel>>,
    n_batch: usize,
}

// SAFETY: the context is only ever used through `&mut self`, and the
// embedder wraps the backend in a mutex, so it is never touched from two
// threads at once.  The model itself is `Send + Sync`.
unsafe impl Send for LlamaCppEmbedBackend {}

impl LlamaCppEmbedBackend {
    /// Load the GGUF at `params.model_path` with an embedding context.
    pub fn load(params: &LlamaLoadParams) -> anyhow::Result<Self> {
        let backend = shared_backend()?;
        if !Path::new(&params.model_path).exists() {
            anyhow::bail!("model path does not exist: {}", params.model_path);
        }
        // -1 means "offload every layer".
        let n_gpu_layers = u32::try_from(params.n_gpu_layers).unwrap_or(u32::MAX);
        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = Box::new(LlamaModel::load_from_file(backend, &params.model_path, &model_params)?);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(params.n_ctx))
            .with_n_batch(params.n_batch)
            .with_n_ubatch(params.n_ubatch)
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::from(params.pooling_type));

        // SAFETY: the model lives in a Box whose heap address never moves and
        // which is kept alive (and dropped after the context) by this struct.
        let model_ref: &'static LlamaModel = unsafe { &*(model.as_ref() as *const LlamaModel) };
        let ctx = model_ref.new_context(backend, ctx_params)?;

        Ok(Self { ctx: Some(ctx), model: Some(model), n_batch: params.n_batch as usize })
    }
}

impl EmbedBackend for LlamaCppEmbedBackend {
    fn embed(&mut self, batch: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let (Some(ctx), Some(model)) = (self.ctx.as_mut(), self.model.as_ref()) else {
            anyhow::bail!("llama.cpp context is closed");
        };
        let mut out = Vec::with_capacity(batch.len());
        for text in batch {
            let tokens = model.vocab().tokenize(text.as_bytes(), true, false);
            if tokens.len() > self.n_batch {
                anyhow::bail!(
                    "Requested tokens ({}) exceed batch size of {}",
                    tokens.len(),
                    self.n_batch
                );
            }
            ctx.clear_kv_cache();
            let mut llama_batch = LlamaBatch::new(self.n_batch.max(1), 1);
            // Every token's output is requested so a model without pooling
            // can still be averaged below.
            llama_batch.add_sequence(&tokens, 0, true)?;
            ctx.decode(&mut llama_batch)?;

            let vector = match ctx.embeddings_seq_ith(0) {
                Ok(pooled) => pooled.to_vec(),
                Err(_) => {
                    // No pooling in the GGUF: mean-pool the token vectors.
                    let mut sum: Vec<f32> = Vec::new();
                    let n = tokens.len().max(1);
                    for i in 0..tokens.len() {
                        let tok = ctx.embeddings_ith(i as i32)?;
                        if sum.is_empty() {
                            sum = vec![0.0; tok.len()];
                        }
                        for (s, v) in sum.iter_mut().zip(tok) {
                            *s += v;
                        }
                    }
                    sum.iter().map(|v| v / n as f32).collect()
                }
            };
            out.push(vector);
        }
        Ok(out)
    }

    fn close(&mut self) -> anyhow::Result<()> {
        self.ctx = None;
        self.model = None;
        Ok(())
    }
}

impl Drop for LlamaCppEmbedBackend {
    fn drop(&mut self) {
        // Explicit order: the context borrows the model.
        self.ctx = None;
        self.model = None;
    }
}
