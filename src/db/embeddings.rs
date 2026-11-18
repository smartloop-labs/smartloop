//! Port of `smartloop/db/embeddings.py`: dense embedding helpers backed by
//! llama.cpp.
//!
//! Provides a lazily-loaded singleton GGUF embedding model (see
//! [`crate::db::llama_embeddings`]) plus utilities for encoding texts and
//! serialising embeddings to/from SQLite BLOBs.  Embeddings run on the same
//! llama.cpp backend as generation.
//!
//! When the model cannot be loaded (missing GGUF, download failure, etc.) all
//! public functions gracefully return `None` so callers fall back to TF-IDF.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::Value;

use crate::api::services::model_service::ModelService;
use crate::config::AppSettings;
use crate::db::llama_embeddings::load_embedder;
use crate::model_host::client::{is_device_owner, ModelClient, RemoteModelClient};

/// Logger name of the Python module.
const LOG_TARGET: &str = "smartloop.db.embeddings";

/// Cache the GGUF under its own subdir so switching models doesn't collide on
/// disk: `<base_dir>/models/embeddings/`.
pub const CACHE_SUBDIR: &str = "models/embeddings";

/// A loaded embedding model as the singleton holds it.
///
/// `LlamaEmbedder` is the production implementation; tests install fakes via
/// [`set_model`].
pub trait EmbeddingModel: Send + Sync {
    /// Encode `texts` into an `(N, dim)` L2-normalised float32 matrix.
    fn encode(&self, texts: &[String]) -> anyhow::Result<Array2<f32>>;

    /// Free the underlying model/context.  Errors are logged by callers.
    fn close(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Signature of an [`encode_texts`] override (test seam replacing Python's
/// `monkeypatch.setattr("smartloop.db.embeddings.encode_texts", ...)`).
pub type EncodeTextsHook =
    Arc<dyn Fn(&[String], Option<&str>, Option<f64>) -> Option<Array2<f32>> + Send + Sync>;

/// Signature of an [`unload`] override (test seam replacing Python's
/// `monkeypatch.setattr("smartloop.db.embeddings.unload", ...)`).  An `Err`
/// stands for the patched function raising.
pub type UnloadHook = Arc<dyn Fn() -> anyhow::Result<()> + Send + Sync>;

/// Singleton state: the loaded model (if any) and whether a load was
/// attempted.  The mutex serialises the one-time load so parallel DAG worker
/// threads don't build the model twice.
#[derive(Default)]
struct ModelState {
    model: Option<Arc<dyn EmbeddingModel>>,
    load_attempted: bool,
}

static MODEL: Lazy<Mutex<ModelState>> = Lazy::new(|| Mutex::new(ModelState::default()));

/// Serialises the one-time load (and [`unload`]) — Python `_load_lock`.  Kept
/// apart from [`MODEL`] so a caller arriving mid-load sees "already tried"
/// and falls back at once instead of waiting on the download.
static LOAD_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

/// Connection to the model host, when one owns the embedder for this process.
static REMOTE: Lazy<Mutex<Option<Arc<RemoteModelClient>>>> = Lazy::new(|| Mutex::new(None));

/// Active override for [`encode_texts`], if any.
static ENCODE_HOOK: Lazy<Mutex<Option<EncodeTextsHook>>> = Lazy::new(|| Mutex::new(None));

/// Active override for [`unload`], if any.
static UNLOAD_HOOK: Lazy<Mutex<Option<UnloadHook>>> = Lazy::new(|| Mutex::new(None));

/// Depth of nested [`indexing_burst`] scopes.  Only the outermost unloads: an
/// indexer that calls another would otherwise drop the model out from under
/// its caller and force a mid-burst reload.
static BURST_DEPTH: AtomicUsize = AtomicUsize::new(0);

/// Identity of the active embedding model, stored in the index.
///
/// Encodes the engine + model (`llama-cpp:<repo>/<file>`) so switching either
/// invalidates the stored vectors (they live in different, non-comparable
/// spaces) and the sqlite store recomputes.  Mirrors the model selection in
/// [`get_model`].  Falls back to `"llama-cpp:unknown"` when settings cannot be
/// read.
pub fn active_embedding_identity() -> String {
    match AppSettings::try_new() {
        Ok(settings) => format!(
            "llama-cpp:{}/{}",
            settings.embedding_gguf_repo, settings.embedding_gguf_file
        ),
        Err(_) => "llama-cpp:unknown".to_string(),
    }
}

/// The model host to embed through, or `None` to embed in this process.
///
/// BGE-M3 is ~417 MB.  Loading it in every per-project agent would multiply
/// that by the number of open projects for no benefit — the vectors are
/// identical whoever computes them — so a child hands the work to the process
/// that already holds the model.
///
/// Two processes have nobody to hand it to, and both are this recursion's
/// base case:
///
/// * **The host itself.**  It *is* the embedder; delegating would be a call
///   to itself — `EncodeTexts` answered by issuing another `EncodeTexts`, one
///   gRPC worker per hop, until the pool was full and every thread was parked
///   on a call that only it could serve.
/// * **A process that was given no host address.**  The supervisor sets
///   `SLP_MODEL_HOST` for every child it starts, so an absent address means
///   there is no host to share — the `--no-supervised` server, the CLI — and
///   the work is done here rather than sent to a socket nobody listens on.
fn remote_embedder() -> Option<Arc<RemoteModelClient>> {
    let address = std::env::var("SLP_MODEL_HOST").unwrap_or_default().trim().to_string();
    if address.is_empty() || is_device_owner() {
        return None;
    }
    let mut remote = REMOTE.lock();
    match remote.as_ref() {
        Some(client) if client.address == address => Some(client.clone()),
        _ => {
            let client = Arc::new(RemoteModelClient::new(address));
            *remote = Some(client.clone());
            Some(client)
        }
    }
}

/// Return the cached embedding model, loading on first call.
///
/// The GGUF is cached under `<base_dir>/models/embeddings/`.  Returns `None`
/// if the model cannot be loaded (callers then fall back to TF-IDF); a failed
/// load is not retried until [`unload`] clears the flag.
fn get_model(base_dir: Option<&str>) -> Option<Arc<dyn EmbeddingModel>> {
    {
        let state = MODEL.lock();
        if let Some(model) = state.model.as_ref() {
            return Some(model.clone());
        }
        if state.load_attempted {
            return None;
        }
    }

    // The load lock serialises the one-time load so parallel DAG worker
    // threads don't build the model twice (or race the llama.cpp init).
    let _load = LOAD_LOCK.lock();
    {
        // Re-check inside the lock: another thread may have finished loading
        // while we waited.
        let mut state = MODEL.lock();
        if let Some(model) = state.model.as_ref() {
            return Some(model.clone());
        }
        if state.load_attempted {
            return None;
        }
        state.load_attempted = true;
    }

    let settings = AppSettings::new();

    let mut cache_dir: Option<String> = None;
    if let Some(base) = base_dir.filter(|b| !b.is_empty()) {
        let dir = Path::new(base).join(CACHE_SUBDIR);
        if let Err(e) = std::fs::create_dir_all(&dir) {
            tracing::warn!(
                target: LOG_TARGET,
                "[embeddings] Could not create cache dir {}: {e}",
                dir.display()
            );
        }
        cache_dir = Some(dir.to_string_lossy().into_owned());
    }

    let load_timer = crate::utils::timing::phase_with("model.load", serde_json::json!({"slot": "embedding"}));
    let model = load_embedder(
        cache_dir.as_deref(),
        &settings.embedding_gguf_repo,
        &settings.embedding_gguf_file,
        &settings.embedding_device,
        &settings.embedding_gguf_base_url,
    )
    .map(|m| Arc::new(m) as Arc<dyn EmbeddingModel>);
    drop(load_timer);
    MODEL.lock().model = model.clone();
    model
}

/// Current singleton model, if loaded (test inspection seam for `_model`).
pub fn model() -> Option<Arc<dyn EmbeddingModel>> {
    MODEL.lock().model.clone()
}

/// Replace the singleton model (test seam for assigning `_model`).
pub fn set_model(model: Option<Arc<dyn EmbeddingModel>>) {
    MODEL.lock().model = model;
}

/// Whether a load has been attempted (`_model_load_attempted`).
pub fn model_load_attempted() -> bool {
    MODEL.lock().load_attempted
}

/// Set the "load attempted" flag (test seam).
pub fn set_model_load_attempted(value: bool) {
    MODEL.lock().load_attempted = value;
}

/// Install (or clear with `None`) an override for [`encode_texts`].
pub fn set_encode_texts_hook(hook: Option<EncodeTextsHook>) {
    *ENCODE_HOOK.lock() = hook;
}

/// Install (or clear with `None`) an override for [`unload`].
pub fn set_unload_hook(hook: Option<UnloadHook>) {
    *UNLOAD_HOOK.lock() = hook;
}

/// Release the embedding model, freeing its weights and llama.cpp context.
///
/// Indexing is bursty — a workspace pass embeds a few hundred chunks and then
/// nothing touches the model for minutes — but the singleton held the GGUF
/// resident for the life of the process.  Dropping it after a burst returns
/// that memory; the next search reloads it lazily.
///
/// The "load attempted" flag is cleared as well as the model.  It exists to
/// stop retrying a load that failed, and leaving it set here would make the
/// unload permanent: every later call would see the "already tried" flag and
/// fall back to TF-IDF for the rest of the process.  A failing `close()` is
/// logged, never raised.
pub fn unload() {
    let hook = UNLOAD_HOOK.lock().clone();
    if let Some(hook) = hook {
        if let Err(e) = hook() {
            tracing::warn!(target: LOG_TARGET, "[embeddings] unload failed: {e}");
        }
        return;
    }

    let model = {
        let _load = LOAD_LOCK.lock();
        let mut state = MODEL.lock();
        state.load_attempted = false;
        state.model.take()
    };

    let Some(model) = model else {
        return;
    };
    let _t = crate::utils::timing::phase_with("model.unload", serde_json::json!({"slot": "embedding"}));
    match model.close() {
        Ok(()) => tracing::info!(target: LOG_TARGET, "[embeddings] model unloaded"),
        Err(e) => tracing::warn!(target: LOG_TARGET, "[embeddings] unload failed: {e}"),
    }
}

/// RAII guard returned by [`indexing_burst`]; unloads the model when the
/// outermost guard drops (including on panic/early return).
#[must_use = "the burst ends when this guard is dropped"]
pub struct IndexingBurst {
    _private: (),
}

impl Drop for IndexingBurst {
    fn drop(&mut self) {
        // Unloads on drop: an indexing pass that failed has still loaded the
        // model.
        let outermost = BURST_DEPTH.fetch_sub(1, Ordering::SeqCst) == 1;
        if outermost {
            unload();
        }
    }
}

/// Release the embedding model when this indexing scope finishes.
///
/// Indexing is bursty — a document or workspace pass embeds a few hundred
/// chunks and then nothing touches the model for minutes — so holding a GGUF
/// and its llama.cpp context resident afterwards is pure cost.  Wrapping the
/// entry points keeps that decision in one place; adding an indexer means
/// wrapping it, not remembering to call [`unload`].
///
/// Nested scopes only unload at the outermost exit.  Search is deliberately
/// *not* wrapped — it queries repeatedly, and unloading between queries would
/// just thrash.
pub fn indexing_burst() -> IndexingBurst {
    BURST_DEPTH.fetch_add(1, Ordering::SeqCst);
    IndexingBurst { _private: () }
}

/// Close the GPU-offloaded embedder at process exit (Python `atexit` hook).
///
/// Without this a GPU context still live at exit can abort during teardown —
/// a dirty shutdown that a supervisor may read as a crash.  Closing explicitly
/// makes exit clean.  Errors are ignored.
pub fn shutdown_embeddings() {
    let model = MODEL.lock().model.take();
    if let Some(model) = model {
        let _ = model.close();
    }
}

/// Pre-load the embedding model so the first search doesn't pay the cost.
///
/// A no-op when a model host owns the embedder: it is already warm there, and
/// loading a second copy here is the cost this avoids.
///
/// Loading the GGUF only allocates the llama.cpp context — the GPU kernels
/// aren't compiled until the first embed call, which is exactly the latency
/// this exists to hide.  So a trivial encode is run here too, mirroring how
/// the model service primes the KV cache with a real generation rather than
/// just loading the LLM weights.
pub fn warmup(base_dir: Option<&str>) {
    if remote_embedder().is_some() {
        return;
    }
    let Some(model) = get_model(base_dir) else {
        return;
    };
    if let Err(e) = model.encode(&["warmup".to_string()]) {
        tracing::warn!(target: LOG_TARGET, "[embeddings] warmup encode failed: {e}");
    }
}

/// Seconds since the process that runs our embeddings last generated.
///
/// `0.0` while it is generating right now, `f64::INFINITY` when it has not
/// generated at all.  Asked of the *host* when one owns the embedder, because
/// that is where both the embedding and the generation actually happen: a
/// child polling its own idle time would always read "quiet" and answer for
/// the wrong device.
///
/// Batch embedding and generation are the same GPU, and a bge-m3 pass over a
/// few hundred chunks takes the token rate of a streaming reply from ~57/s to
/// ~3/s.  Background indexing reads this to wait for a quiet moment.
///
/// Unknowable means idle: a host that cannot be reached is not one to defer
/// to, and an indexer that never runs is worse than one that overlaps.
pub fn device_idle_seconds() -> f64 {
    if let Some(remote) = remote_embedder() {
        return match remote.idle_seconds() {
            Ok(seconds) => seconds,
            Err(exc) => {
                tracing::debug!(
                    target: LOG_TARGET,
                    "[embeddings] could not read host idle time: {exc}"
                );
                f64::INFINITY
            }
        };
    }

    match ModelService::existing_instance() {
        Some(service) => {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| service.idle_seconds()))
                .unwrap_or(f64::INFINITY)
        }
        None => f64::INFINITY,
    }
}

/// Encode `texts` into a dense `(N, dim)` embedding matrix.
///
/// Returns `None` when the model is unavailable.  Routed to the model host
/// when one owns the embedder, so every project agent shares a single
/// resident copy.  `timeout` (seconds) is the per-call RPC deadline passed on
/// to the host when routing remotely.
pub fn encode_texts(
    texts: &[String],
    base_dir: Option<&str>,
    timeout: Option<f64>,
) -> Option<Array2<f32>> {
    let hook = ENCODE_HOOK.lock().clone();
    if let Some(hook) = hook {
        return hook(texts, base_dir, timeout);
    }

    if let Some(remote) = remote_embedder() {
        return match remote.encode_texts_with_timeout(texts, base_dir, timeout) {
            Ok(result) => result,
            Err(exc) => {
                // Same contract as a failed local load: callers fall back to
                // TF-IDF rather than the child quietly loading its own model.
                tracing::warn!(
                    target: LOG_TARGET,
                    "[embeddings] model host encoding failed: {exc}"
                );
                None
            }
        };
    }

    encode_local(texts, base_dir)
}

/// Encode `texts` with this process's own model, routing nowhere.
///
/// What the host calls to serve an `EncodeTexts` request.  It is a separate
/// entry point rather than a flag on [`encode_texts`] so that the serving path
/// cannot re-enter the routing decision at all: the host asking
/// [`encode_texts`] to embed is the host asking who should embed, and one
/// wrong answer there is a request that answers itself forever.
pub fn encode_local(texts: &[String], base_dir: Option<&str>) -> Option<Array2<f32>> {
    let model = get_model(base_dir)?;
    match model.encode(texts) {
        Ok(embeddings) => Some(embeddings),
        Err(exc) => {
            tracing::warn!(target: LOG_TARGET, "[embeddings] Encoding failed: {exc}");
            None
        }
    }
}

// ------------------------------------------------------------------ BLOB I/O

/// Serialise a 1-D vector to raw little-endian float32 bytes for SQLite BLOB
/// storage (byte-compatible with numpy `astype(float32).tobytes()`).
pub fn embedding_to_blob(vec: ArrayView1<'_, f32>) -> Vec<u8> {
    let mut out = Vec::with_capacity(vec.len() * 4);
    for v in vec.iter() {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Deserialise a BLOB back to a 1-D float32 vector.  Trailing bytes that do
/// not make up a whole float are ignored.
pub fn blob_to_embedding(blob: &[u8]) -> Array1<f32> {
    blob.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Represent a BLOB inside a free-form JSON dict — e.g. a citation's
/// `_vectors` entries, which were Python `bytes` objects — as an array of
/// byte values (serde's own encoding of `Vec<u8>`).
pub fn blob_to_value(blob: &[u8]) -> Value {
    Value::Array(blob.iter().map(|b| Value::from(*b)).collect())
}

/// Read back a BLOB stored by [`blob_to_value`] (a base64 string is accepted
/// too).  `None` for `null` or anything that is not a byte sequence.
pub fn value_to_blob(value: &Value) -> Option<Vec<u8>> {
    match value {
        Value::Array(items) => items
            .iter()
            .map(|v| v.as_u64().filter(|b| *b <= 255).map(|b| b as u8))
            .collect(),
        Value::String(s) => {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD.decode(s).ok()
        }
        _ => None,
    }
}

// --------------------------------------------------------------- similarity

/// Cosine distance (`1 - similarity`) from `query_vec` to each row of
/// `matrix`.
///
/// The one place the comparison is defined, so everything that asks "is this
/// text about the same thing as that text?" — chunk retrieval ranking a
/// corpus against the user's question, citation verification checking a
/// source against the finished answer — agrees on what a distance of 0.55
/// means.  Rows and the query are re-normalised here (zero norms treated as
/// 1) rather than trusted: vectors come both from the embedder (already
/// L2-normalised) and from SQLite BLOBs written by older index versions.
///
/// # Panics
///
/// When the query length differs from the matrix width (numpy raises there
/// too); [`try_cosine_distances`] returns an error instead.
pub fn cosine_distances(query_vec: ArrayView1<'_, f32>, matrix: ArrayView2<'_, f32>) -> Array1<f32> {
    match try_cosine_distances(query_vec, matrix) {
        Ok(distances) => distances,
        Err(e) => panic!("{e}"),
    }
}

/// [`cosine_distances`], returning an error on a dimension mismatch.
pub fn try_cosine_distances(
    query_vec: ArrayView1<'_, f32>,
    matrix: ArrayView2<'_, f32>,
) -> anyhow::Result<Array1<f32>> {
    if matrix.ncols() != query_vec.len() && matrix.nrows() > 0 {
        anyhow::bail!(
            "shapes ({},{}) and ({},) not aligned: {} (dim 1) != {} (dim 0)",
            matrix.nrows(),
            matrix.ncols(),
            query_vec.len(),
            matrix.ncols(),
            query_vec.len()
        );
    }
    let q_norm = l2_norm(query_vec.iter());
    let q_norm = if q_norm == 0.0 { 1.0 } else { q_norm };
    let q: Vec<f32> = query_vec.iter().map(|v| v / q_norm).collect();
    let mut out = Array1::<f32>::zeros(matrix.nrows());
    for (i, row) in matrix.axis_iter(Axis(0)).enumerate() {
        let norm = l2_norm(row.iter());
        let norm = if norm == 0.0 { 1.0 } else { norm };
        let dot: f32 = row.iter().zip(q.iter()).map(|(a, b)| (a / norm) * b).sum();
        out[i] = 1.0 - dot;
    }
    Ok(out)
}

/// Euclidean norm of a float32 sequence.
fn l2_norm<'a>(values: impl Iterator<Item = &'a f32>) -> f32 {
    values.map(|v| v * v).sum::<f32>().sqrt()
}
