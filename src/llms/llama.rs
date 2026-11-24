//! Hand-port of the parts of `llama-cpp-python` (`llama_cpp.Llama` and
//! `llama_cpp.llama_chat_format`) the app relies on, built on `llama-cpp-2`.
//!
//! The Python wrapper ([`crate::llms::llama_cpp::LlamaCpp`]) and the factory
//! ([`crate::llms::llm_factory::LlmFactory`]) talk to a model through the
//! [`LlamaModel`] trait, so tests can substitute a fake context exactly as the
//! Python tests subclassed `Llama`.  Re-exported from `crate::llms::llama_cpp`.
//!
//! What is ported, and from where (llama-cpp-python 0.3.x):
//!
//! * `Llama.__init__` — model/context parameters, LoRA, GGUF metadata, the
//!   chat-format resolution (`tokenizer.chat_template*` → `Jinja2ChatFormatter`
//!   handlers registered in `_chat_handlers`, `guess_chat_format_from_gguf_metadata`,
//!   the `llama-2` fallback).
//! * `Llama.tokenize` / `detokenize` / `eval` / `generate` (with the KV-cache
//!   prefix reuse) / `_create_completion` (stop sequences, the multibyte fix,
//!   the streaming hold-back of partial stop sequences, `length`/`stop` finish
//!   reasons) and the conversion of completions to chat completions.
//! * `Jinja2ChatFormatter` (rendered with minijinja, configured like the
//!   Jinja2 `ImmutableSandboxedEnvironment(trim_blocks, lstrip_blocks)` with the
//!   Transformers `tojson`, `raise_exception`, `strftime_now`, loop controls and
//!   the `{% generation %}` pass-through) and the built-in `llama-2`,
//!   `llama-3`, `chatml` and `mistral-instruct` formats.
//! * The multimodal handlers: `MTMDChatHandler` / `Gemma4ChatHandler` (render
//!   the model's own template) and the `Llava15ChatHandler` family
//!   (`Llava16`, `MiniCPMv26`, `NanoLlava`, `Moondream`, `Qwen25VL`), on
//!   llama.cpp's `mtmd` library.
//!
//! Not ported because nothing in the app uses them: logprobs, grammars /
//! `response_format`, tool-choice grammars, suffix/infill, draft models and
//! the prompt state cache.

pub mod batch;

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::CString;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaLoraAdapter, LlamaModel as NativeModel};
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunks, MtmdInputText, MtmdTokenizeError, mtmd_default_marker,
};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken as NativeToken;
use minijinja::value::Kwargs;
use parking_lot::{Mutex, MutexGuard, ReentrantMutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

/// A llama.cpp token id.
pub type LlamaToken = i32;

/// A stream of OpenAI-shaped chat-completion chunks
/// (`{"choices": [{"delta": {"content": ...}}]}`).
pub type ChatCompletionChunks<'a> = Box<dyn Iterator<Item = Value> + 'a>;

/// What `create_chat_completion` returns: one response dict (`stream=False`)
/// or a chunk iterator (`stream=True`).
pub enum ChatCompletionOutput<'a> {
    Response(Value),
    Stream(ChatCompletionChunks<'a>),
}

impl<'a> ChatCompletionOutput<'a> {
    /// Iterate the output as chunks; a full response yields itself once.
    pub fn into_chunks(self) -> ChatCompletionChunks<'a> {
        match self {
            ChatCompletionOutput::Response(v) => Box::new(std::iter::once(v)),
            ChatCompletionOutput::Stream(s) => s,
        }
    }

    /// The single response; for a stream, the chunks' text joined into a
    /// `{"choices": [{"message": {"role": "assistant", "content": ...}}]}` dict.
    pub fn into_response(self) -> Value {
        match self {
            ChatCompletionOutput::Response(v) => v,
            ChatCompletionOutput::Stream(chunks) => {
                let mut content = String::new();
                let mut finish_reason = Value::Null;
                for chunk in chunks {
                    content.push_str(&choice_text(&chunk));
                    if let Some(reason) = chunk.pointer("/choices/0/finish_reason") {
                        if !reason.is_null() {
                            finish_reason = reason.clone();
                        }
                    }
                }
                json!({
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason,
                    }]
                })
            }
        }
    }
}

/// Text carried by the first choice of an OpenAI-shaped response or chunk
/// (`delta.content`, `message.content` or `text`), `""` when there is none.
fn choice_text(chunk: &Value) -> String {
    let Some(choice) = chunk.get("choices").and_then(Value::as_array).and_then(|c| c.first()) else {
        return String::new();
    };
    let content = if let Some(c) = choice.get("delta").and_then(|d| d.get("content")) {
        c
    } else if let Some(c) = choice.get("message").and_then(|m| m.get("content")) {
        c
    } else if let Some(c) = choice.get("text") {
        c
    } else {
        return String::new();
    };
    content.as_str().unwrap_or_default().to_string()
}

/// Arguments of `Llama.create_chat_completion`, with llama-cpp-python defaults.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionRequest {
    /// OpenAI-shaped messages; `content` may be a string or a list of parts
    /// (text / `image_url`) for vision models.
    pub messages: Vec<Value>,
    pub max_tokens: Option<usize>,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    /// DRY ("don't repeat yourself") strength; 0.0 = off.  Unlike
    /// `repeat_penalty`, which sees only the last 64 tokens, DRY penalizes
    /// extending any sequence the output has already produced, so it reaches
    /// loops whose cycle is longer than that window.
    #[serde(default)]
    pub dry_multiplier: f32,
    pub stop: Vec<String>,
    pub seed: Option<u32>,
    pub stream: bool,
    /// OpenAI function definitions rendered as the template's `tools`
    /// variable, for a model that calls functions natively; `None` renders
    /// the template without tools.
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
}

impl Default for ChatCompletionRequest {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            max_tokens: None,
            temperature: 0.2,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.05,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            dry_multiplier: 0.0,
            stop: Vec::new(),
            seed: None,
            stream: false,
            tools: None,
        }
    }
}

/// What a chat handler is called with (`handler(llama=..., messages=..., **kwargs)`).
pub struct ChatHandlerRequest<'a> {
    /// The model the call names (`llama=`).
    pub llama: Option<&'a dyn LlamaModel>,
    pub request: ChatCompletionRequest,
    /// Extra variables passed through to the chat-template render
    /// (e.g. `enable_thinking`).
    pub template_kwargs: Map<String, Value>,
}

/// A llama-cpp-python chat handler: renders the template and generates.
pub type ChatHandler =
    Arc<dyn for<'a> Fn(ChatHandlerRequest<'a>) -> anyhow::Result<ChatCompletionOutput<'a>> + Send + Sync>;

/// Mutable per-context state the Python code stashed as attributes on the
/// `Llama` object: the generation lock (`_slp_context_lock`), the bound
/// reasoning flag (`_slp_thinking`) and the chat-handler resolution fields
/// (`chat_handler`, `chat_format`, `_chat_handlers`).
pub struct LlamaState {
    context_lock: ReentrantMutex<()>,
    thinking: RwLock<Option<bool>>,
    chat_handler: RwLock<Option<ChatHandler>>,
    chat_format: RwLock<Option<String>>,
    chat_handlers: Option<RwLock<HashMap<String, ChatHandler>>>,
}

impl Default for LlamaState {
    fn default() -> Self {
        Self::new(None, None)
    }
}

impl LlamaState {
    /// State with an optional explicit chat handler and chat format, and an
    /// empty `_chat_handlers` registry.
    pub fn new(chat_handler: Option<ChatHandler>, chat_format: Option<String>) -> Self {
        Self {
            context_lock: ReentrantMutex::new(()),
            thinking: RwLock::new(None),
            chat_handler: RwLock::new(chat_handler),
            chat_format: RwLock::new(chat_format),
            chat_handlers: Some(RwLock::new(HashMap::new())),
        }
    }

    /// State whose `_chat_handlers` registry is absent (a llama-cpp that
    /// resolves handlers differently).
    pub fn without_handler_registry(chat_handler: Option<ChatHandler>, chat_format: Option<String>) -> Self {
        Self { chat_handlers: None, ..Self::new(chat_handler, chat_format) }
    }

    /// The reentrant lock serializing generation on this context.
    pub fn context_lock(&self) -> &ReentrantMutex<()> {
        &self.context_lock
    }

    /// `_slp_thinking`: `None` when the flag was never bound.
    pub fn thinking(&self) -> Option<bool> {
        *self.thinking.read()
    }

    /// Set (or unbind with `None`) `_slp_thinking`.
    pub fn set_thinking(&self, value: Option<bool>) {
        *self.thinking.write() = value;
    }

    /// The explicit chat handler (`llm.chat_handler`).
    pub fn chat_handler(&self) -> Option<ChatHandler> {
        self.chat_handler.read().clone()
    }

    /// Replace the explicit chat handler.
    pub fn set_chat_handler(&self, handler: Option<ChatHandler>) {
        *self.chat_handler.write() = handler;
    }

    /// `llm.chat_format`.
    pub fn chat_format(&self) -> Option<String> {
        self.chat_format.read().clone()
    }

    /// Set `llm.chat_format`.
    pub fn set_chat_format(&self, chat_format: Option<String>) {
        *self.chat_format.write() = chat_format;
    }

    /// `llm._chat_handlers.get(name)`; `Err` when the registry is absent.
    pub fn registered_handler(&self, name: &str) -> anyhow::Result<Option<ChatHandler>> {
        match &self.chat_handlers {
            Some(registry) => Ok(registry.read().get(name).cloned()),
            None => Err(anyhow::anyhow!("'Llama' object has no attribute '_chat_handlers'")),
        }
    }

    /// `llm._chat_handlers[name] = handler`; `Err` when the registry is absent.
    pub fn register_handler(&self, name: impl Into<String>, handler: ChatHandler) -> anyhow::Result<()> {
        match &self.chat_handlers {
            Some(registry) => {
                registry.write().insert(name.into(), handler);
                Ok(())
            }
            None => Err(anyhow::anyhow!("'Llama' object has no attribute '_chat_handlers'")),
        }
    }

    /// Names in the `_chat_handlers` registry (empty when absent).
    pub fn registered_handler_names(&self) -> Vec<String> {
        match &self.chat_handlers {
            Some(registry) => registry.read().keys().cloned().collect(),
            None => Vec::new(),
        }
    }
}

/// The surface of `llama_cpp.Llama` the app uses.
pub trait LlamaModel: Send + Sync {
    /// Per-context mutable state (lock, thinking flag, chat handlers).
    fn state(&self) -> &LlamaState;

    /// Runtime context size (`Llama.n_ctx()`).
    fn n_ctx(&self) -> usize;

    /// GGUF metadata key/value strings (`Llama.metadata`).
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    /// Tokenize UTF-8 bytes (`Llama.tokenize`).
    fn tokenize(&self, text: &[u8]) -> anyhow::Result<Vec<LlamaToken>>;

    /// Detokenize to bytes (`Llama.detokenize`).
    fn detokenize(&self, tokens: &[LlamaToken]) -> anyhow::Result<Vec<u8>>;

    /// `Llama.create_chat_completion`.  Must resolve the handler the same way
    /// llama-cpp-python does (see [`resolve_chat_handler`]).
    fn create_chat_completion(&self, request: ChatCompletionRequest) -> anyhow::Result<ChatCompletionOutput<'_>>;

    /// `Llama.model_path`.
    fn model_path(&self) -> String {
        String::new()
    }

    /// `Llama.model_params.n_gpu_layers`.
    fn n_gpu_layers(&self) -> i32 {
        0
    }

    /// Whether flash attention was enabled at load.
    fn flash_attn(&self) -> bool {
        false
    }

    /// Release native resources (`Llama.close`).
    fn close(&self) {}

    /// The in-process llama.cpp model behind this handle, if any.  The
    /// built-in chat handlers generate through it; a fake model returns `None`.
    fn as_llama(&self) -> Option<&Llama> {
        None
    }
}

/// The constructor arguments of `Llama(**config)` the factory passes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlamaParams {
    pub model_path: String,
    pub n_ctx: usize,
    pub n_batch: usize,
    pub n_threads: usize,
    pub n_gpu_layers: i32,
    pub verbose: bool,
    pub n_parallel: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_path: Option<String>,
    #[serde(default)]
    pub flash_attn: bool,
    /// Explicit chat format name; `None` derives it from the GGUF template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_format: Option<String>,
}

impl Default for LlamaParams {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            n_ctx: 512,
            n_batch: 512,
            n_threads: 4,
            n_gpu_layers: 0,
            verbose: true,
            n_parallel: 1,
            lora_path: None,
            flash_attn: false,
            chat_format: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

/// The process-wide llama.cpp backend.  Statics are never dropped, so the
/// backend is never freed while a model might still use it.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

/// The process-wide llama.cpp backend, initialized on first use.
///
/// llama.cpp's backend may only be initialized once per process; every
/// in-process llama.cpp user (models, embedders) must go through this.  When
/// it was already initialized elsewhere the proof token is recreated (the
/// backend is global state in llama.cpp, the token carries no data).
/// llama.cpp logs are routed to `tracing` and enabled only when `verbose`
/// (llama-cpp-python's `verbose=False` silences them).
pub fn llama_backend_with_logs(verbose: bool) -> &'static LlamaBackend {
    BACKEND.get_or_init(|| {
        llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default().with_logs_enabled(verbose));
        match LlamaBackend::init() {
            Ok(backend) => backend,
            // Already initialized by another component: the backend is
            // process-global, so a fresh proof token is equivalent.
            Err(_) => LlamaBackend {},
        }
    })
}

/// [`llama_backend_with_logs`] with llama.cpp logging off.
pub fn llama_backend() -> &'static LlamaBackend {
    llama_backend_with_logs(false)
}

/// `llama_cpp.llama_supports_gpu_offload()`.
pub fn llama_supports_gpu_offload() -> bool {
    llama_backend().supports_gpu_offload()
}

/// `LLAMA_DEFAULT_SEED`.
const LLAMA_DEFAULT_SEED: u32 = 0xFFFF_FFFF;

/// Sentinel for KV positions that hold no text token (image/audio embeddings).
const NO_TOKEN: LlamaToken = -1;

/// `Llama.last_n_tokens_size` default.
const LAST_N_TOKENS_SIZE: i32 = 64;

/// DRY growth per token a repeat runs past `DRY_ALLOWED_LENGTH`.
const DRY_BASE: f32 = 1.75;

/// Repeats up to this many tokens go unpenalized.  Above the usual 2, so an
/// extract can name the same party or clause twice without being pushed to
/// misspell it the second time.
const DRY_ALLOWED_LENGTH: i32 = 4;

/// Tokens a repeat cannot span: a new line or field starts a new match, so
/// list items and table rows that share a shape are not penalized.
const DRY_SEQ_BREAKERS: [&str; 4] = ["\n", ":", "\"", "*"];

/// Longest cycle the repetition guard looks for, in tokens — a few sentences.
const REPEAT_MAX_PERIOD: usize = 256;

/// Shortest repeated run that ends a generation.  A run counts only once its
/// cycle has come round three times, and never below this, so a short
/// legitimate repeat (a table divider, a list of similar lines) is left alone.
const REPEAT_MIN_RUN: usize = 128;

/// Length of the run of tokens at the end of `tokens` that only repeats an
/// earlier cycle, when the model has fallen into a loop: the last `p` tokens
/// have come round at least three times in a row, over `REPEAT_MIN_RUN`
/// tokens or more.  Dropping that many tokens leaves one copy of the cycle.
///
/// A small model that loops never emits end-of-generation, so without this it
/// runs until `max_tokens` — the rest of the context window when uncapped.
fn repetition_run(tokens: &[LlamaToken]) -> Option<usize> {
    let n = tokens.len();
    for p in 1..=REPEAT_MAX_PERIOD.min(n / 3) {
        // Positions at the end that match the token one cycle back.
        let run = (p..n).rev().take_while(|&i| tokens[i] == tokens[i - p]).count();
        if run >= (2 * p).max(REPEAT_MIN_RUN) {
            return Some(run);
        }
    }
    None
}

/// Decode bytes as UTF-8, dropping invalid sequences (`errors="ignore"`).
pub fn decode_utf8_ignore(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for chunk in bytes.utf8_chunks() {
        out.push_str(chunk.valid());
    }
    out
}

/// Deterministic derivation of the next sampling seed from the current one
/// (llama-cpp-python: `random.Random(self._seed).randint(0, 2**32)`).
fn derive_seed(seed: u32) -> u32 {
    // splitmix64
    let mut z = (seed as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) as u32
}

fn now_epoch_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Llama
// ---------------------------------------------------------------------------

/// The mutable native generation state: the context (KV cache, logits), the
/// tokens it holds and the sampling seed.
struct Engine {
    /// Borrows `Llama::model`; declared first so it drops before the model
    /// (see [`Llama`]).
    ctx: LlamaContext<'static>,
    /// Tokens currently evaluated into the KV cache (`input_ids[:n_tokens]`).
    input_ids: Vec<LlamaToken>,
    /// Restored or truncated state must decode before sampling.
    requires_eval: bool,
    /// Used by the sampler.
    seed: u32,
    _lora: Option<LlamaLoraAdapter>,
}

// SAFETY: the context and adapter are only touched while holding the
// `Llama::engine` mutex, which serializes all access to them.
unsafe impl Send for Engine {}

/// A llama.cpp model loaded in-process — the Rust `llama_cpp.Llama`.
///
/// Field order matters: fields drop in declaration order, so the chat
/// handlers in `state` (which may own a multimodal projector context) and the
/// native context in `engine` are released before the model they reference.
pub struct Llama {
    params: LlamaParams,
    state: LlamaState,
    metadata: HashMap<String, String>,
    engine: Mutex<Engine>,
    /// Slots past which a batch step stops speeding up on this device;
    /// benchmarked on the first batch (see [`batch`]).
    batch_knee: Mutex<Option<usize>>,
    n_ctx: usize,
    n_batch: usize,
    model: Box<NativeModel>,
}

impl Llama {
    /// Load a model (`Llama(**config)`), with an optional multimodal chat handler.
    pub fn new(params: LlamaParams, chat_handler: Option<ChatHandler>) -> anyhow::Result<Self> {
        let backend = llama_backend_with_logs(params.verbose);

        if !std::path::Path::new(&params.model_path).exists() {
            anyhow::bail!("Model path does not exist: {}", params.model_path);
        }

        // Model params.  -1 offloads every layer (llama.cpp clamps INT32_MAX).
        let n_gpu_layers: u32 = if params.n_gpu_layers < 0 { u32::MAX } else { params.n_gpu_layers as u32 };
        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(n_gpu_layers)
            .with_use_mmap(params.lora_path.is_none());

        let model = NativeModel::load_from_file(backend, &params.model_path, &model_params)
            .map_err(|e| anyhow::anyhow!("Failed to load model from file: {} ({e})", params.model_path))?;
        let model = Box::new(model);
        // SAFETY: the model is boxed (stable address) and owned by the
        // returned `Llama`, whose field order drops the context first.
        let model_ref: &'static NativeModel = unsafe { &*(model.as_ref() as *const NativeModel) };

        let n_ctx_req = if params.n_ctx == 0 { model.n_ctx_train() as usize } else { params.n_ctx };
        let n_batch = params.n_batch.min(n_ctx_req).max(1);
        let n_ubatch = n_batch.min(512);
        let n_threads_batch = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let n_threads = if params.n_threads == 0 { (n_threads_batch / 2).max(1) } else { params.n_threads };

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx_req as u32))
            .with_n_batch(n_batch as u32)
            .with_n_ubatch(n_ubatch as u32)
            .with_n_threads(n_threads as i32)
            .with_n_threads_batch(n_threads_batch as i32)
            .with_flash_attention_policy(if params.flash_attn { 1 } else { 0 });

        let ctx = model_ref
            .new_context(backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("Failed to create llama_context ({e})"))?;

        let lora = match &params.lora_path {
            Some(lora_path) => {
                let mut adapter = model_ref.lora_adapter_init(lora_path).map_err(|e| {
                    anyhow::anyhow!("Failed to initialize LoRA adapter from lora path: {lora_path} ({e})")
                })?;
                ctx.lora_adapter_set(&mut adapter, 1.0)
                    .map_err(|e| anyhow::anyhow!("Failed to set LoRA adapter from lora path: {lora_path} ({e})"))?;
                Some(adapter)
            }
            None => None,
        };

        let n_ctx = ctx.n_ctx() as usize;
        let metadata = read_model_metadata(&model);

        let state = LlamaState::new(chat_handler, params.chat_format.clone());
        let mut llama = Self {
            params,
            state,
            metadata,
            engine: Mutex::new(Engine {
                ctx,
                input_ids: Vec::new(),
                requires_eval: true,
                seed: LLAMA_DEFAULT_SEED,
                _lora: lora,
            }),
            batch_knee: Mutex::new(None),
            n_ctx,
            n_batch,
            model,
        };
        llama.register_chat_formats();
        Ok(llama)
    }

    /// The `_chat_handlers` registration and chat-format resolution of
    /// `Llama.__init__`.
    fn register_chat_formats(&mut self) {
        let eos_id = self.token_eos();
        let bos_id = self.token_bos();
        let eos_token = if eos_id != -1 { self.token_get_text(eos_id) } else { String::new() };
        let bos_token = if bos_id != -1 { self.token_get_text(bos_id) } else { String::new() };

        // `tokenizer.chat_template.<name>` → `<name>`, plus the default.
        let mut template_choices: Vec<(String, String)> = self
            .metadata
            .iter()
            .filter(|(k, _)| k.starts_with("tokenizer.chat_template."))
            .map(|(k, v)| (k["tokenizer.".len()..].to_string(), v.clone()))
            .collect();
        template_choices.sort();
        if let Some(default) = self.metadata.get("tokenizer.chat_template") {
            template_choices.push(("chat_template.default".to_string(), default.clone()));
        }

        for (name, template) in &template_choices {
            match Jinja2ChatFormatter::new(template, &eos_token, &bos_token, true, Some(vec![eos_id])) {
                Ok(formatter) => {
                    let _ = self.state.register_handler(name.clone(), formatter.to_chat_handler());
                }
                Err(e) => {
                    tracing::warn!(target: "smartloop.llms.llama", "Failed to compile chat template {name}: {e}");
                }
            }
        }

        let has_default = template_choices.iter().any(|(n, _)| n == "chat_template.default");
        if self.state.chat_format().is_none() && self.state.chat_handler().is_none() && has_default {
            match guess_chat_format_from_gguf_metadata(&self.metadata) {
                Some(guessed) => {
                    if self.params.verbose {
                        tracing::info!(target: "smartloop.llms.llama", "Guessed chat format: {guessed}");
                    }
                    self.state.set_chat_format(Some(guessed.to_string()));
                }
                None => {
                    if self.params.verbose {
                        tracing::info!(target: "smartloop.llms.llama", "Using chat eos_token: {eos_token}");
                        tracing::info!(target: "smartloop.llms.llama", "Using chat bos_token: {bos_token}");
                    }
                    self.state.set_chat_format(Some("chat_template.default".to_string()));
                }
            }
        }

        if self.state.chat_format().is_none() && self.state.chat_handler().is_none() {
            self.state.set_chat_format(Some("llama-2".to_string()));
            if self.params.verbose {
                tracing::info!(target: "smartloop.llms.llama", "Using fallback chat format: llama-2");
            }
        }
    }

    /// The parameters this model was loaded with.
    pub fn params(&self) -> &LlamaParams {
        &self.params
    }

    /// Prefill batch size actually used (`min(n_ctx, n_batch)`).
    pub fn n_batch(&self) -> usize {
        self.n_batch
    }

    /// Threads used for generation.
    pub fn n_threads(&self) -> usize {
        self.params.n_threads
    }

    /// `Llama.token_eos()`.
    pub fn token_eos(&self) -> LlamaToken {
        self.model.vocab().eos().0
    }

    /// `Llama.token_bos()`.
    pub fn token_bos(&self) -> LlamaToken {
        self.model.vocab().bos().0
    }

    /// Vocabulary size.
    pub fn n_vocab(&self) -> i32 {
        self.model.n_vocab()
    }

    /// The raw text of a vocabulary entry (`_model.token_get_text`).
    pub fn token_get_text(&self, token: LlamaToken) -> String {
        self.model
            .vocab()
            .text(NativeToken(token))
            .map(|t| t.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    /// Whether `token` ends generation (`llama_vocab_is_eog`).
    pub fn is_eog(&self, token: LlamaToken) -> bool {
        self.model.vocab().is_eog(NativeToken(token))
    }

    /// `Llama.tokenize(text, add_bos, special)`.
    pub fn tokenize_ext(&self, text: &[u8], add_bos: bool, special: bool) -> Vec<LlamaToken> {
        self.model.vocab().tokenize(text, add_bos, special).into_iter().map(|t| t.0).collect()
    }

    /// `Llama.detokenize(tokens, special=special)`: pieces concatenated, with
    /// the leading space after a BOS token removed.
    pub fn detokenize_ext(&self, tokens: &[LlamaToken], special: bool) -> Vec<u8> {
        let vocab = self.model.vocab();
        let mut output = Vec::new();
        for &token in tokens {
            output.extend(vocab.token_to_piece(NativeToken(token), special, None));
        }
        if !tokens.is_empty() && tokens[0] == self.token_bos() && output.first() == Some(&b' ') {
            output.remove(0);
        }
        output
    }

    /// Lock the native context.
    fn engine(&self) -> MutexGuard<'_, Engine> {
        self.engine.lock()
    }

    /// `Llama.create_completion` over pre-tokenized prompt tokens, converted
    /// to a chat completion (`_convert_completion_to_chat`).
    fn create_chat_completion_from_tokens(
        &self,
        prompt_tokens: Vec<LlamaToken>,
        request: &ChatCompletionRequest,
        stop: Vec<String>,
    ) -> anyhow::Result<ChatCompletionOutput<'_>> {
        if batch::try_capture(&prompt_tokens, request, &stop) {
            return Ok(ChatCompletionOutput::Response(Value::Null));
        }
        let engine = self.engine();
        self.create_completion_locked(engine, prompt_tokens, request, stop)
    }

    /// The body of `_create_completion` once the engine is locked.
    fn create_completion_locked<'a>(
        &'a self,
        mut engine: MutexGuard<'a, Engine>,
        prompt_tokens: Vec<LlamaToken>,
        request: &ChatCompletionRequest,
        stop: Vec<String>,
    ) -> anyhow::Result<ChatCompletionOutput<'a>> {
        if prompt_tokens.len() >= self.n_ctx {
            anyhow::bail!(
                "Requested tokens ({}) exceed context window of {}",
                prompt_tokens.len(),
                self.n_ctx
            );
        }

        let mut max_tokens = match request.max_tokens {
            Some(m) if m > 0 => m,
            // Unlimited, depending on n_ctx.
            _ => self.n_ctx - prompt_tokens.len(),
        };
        // Truncate max_tokens if requested tokens would exceed the context window.
        if max_tokens + prompt_tokens.len() >= self.n_ctx {
            max_tokens = self.n_ctx - prompt_tokens.len();
        }

        engine.seed = match request.seed {
            Some(seed) => seed,
            None => derive_seed(engine.seed),
        };

        let sampler = self.init_sampler(request, engine.seed);
        let token_gen = TokenGen::start(self, &mut engine, &prompt_tokens, sampler);

        // If the prompt is empty, initialize the completion with BOS so
        // detokenization does not start with a space.
        let completion_tokens = if prompt_tokens.is_empty() { vec![self.token_bos()] } else { Vec::new() };

        let run = CompletionRun {
            llama: self,
            engine,
            token_gen,
            stream: request.stream,
            id: format!("cmpl-{}", uuid::Uuid::new_v4()),
            created: now_epoch_secs(),
            model: self.params.model_path.clone(),
            prompt_tokens,
            completion_tokens,
            returned_tokens: 0,
            stop_sequences: stop.iter().filter(|s| !s.is_empty()).map(|s| s.as_bytes().to_vec()).collect(),
            max_tokens,
            multibyte_fix: 0,
            finish_reason: "length",
            text: Vec::new(),
            generating: true,
            pending: VecDeque::new(),
        };

        if request.stream {
            Ok(ChatCompletionOutput::Stream(Box::new(ChatChunks::new(run))))
        } else {
            Ok(ChatCompletionOutput::Response(run.run_to_completion()?))
        }
    }

    /// `Llama._init_sampler`.
    fn init_sampler(&self, request: &ChatCompletionRequest, seed: u32) -> LlamaSampler {
        let mut chain = vec![LlamaSampler::penalties(
            self.n_vocab(),
            LAST_N_TOKENS_SIZE,
            request.repeat_penalty,
            request.frequency_penalty,
            request.presence_penalty,
        )];
        if request.dry_multiplier > 0.0 {
            // -1: the whole context, which only ever holds generated tokens —
            // the sampler accepts what it samples, not the prompt, so copying
            // from a source document is never penalized.
            chain.push(LlamaSampler::dry(
                &self.model,
                request.dry_multiplier,
                DRY_BASE,
                DRY_ALLOWED_LENGTH,
                -1,
                DRY_SEQ_BREAKERS,
            ));
        }
        let temp = request.temperature;
        if temp < 0.0 {
            chain.push(LlamaSampler::dist(seed));
        } else if temp == 0.0 {
            chain.push(LlamaSampler::greedy());
        } else {
            let min_keep = 1;
            chain.push(LlamaSampler::top_k(request.top_k));
            chain.push(LlamaSampler::typical(1.0, min_keep));
            chain.push(LlamaSampler::top_p(request.top_p, min_keep));
            chain.push(LlamaSampler::min_p(request.min_p, min_keep));
            chain.push(LlamaSampler::temp(temp));
            chain.push(LlamaSampler::dist(seed));
        }
        LlamaSampler::chain_simple(chain)
    }

    /// The multimodal prompt path of the vision handlers: reset the context,
    /// evaluate the text and media chunks, then generate from the result.
    fn create_chat_completion_from_mtmd(
        &self,
        mtmd: &MtmdContext,
        chunks: &MtmdInputChunks,
        request: &ChatCompletionRequest,
    ) -> anyhow::Result<ChatCompletionOutput<'_>> {
        let mut engine = self.engine();
        // Reset llama context.
        self.reset(&mut engine);
        engine.ctx.clear_kv_cache();

        let total = chunks.total_tokens();
        if total > self.n_ctx {
            anyhow::bail!("Prompt exceeds n_ctx: {} > {}", total, self.n_ctx);
        }

        let n_past = chunks
            .eval_chunks(mtmd, &engine.ctx, 0, 0, self.n_batch as i32, true)
            .map_err(|e| anyhow::anyhow!("Failed to evaluate chunk: {e}"))?;
        let n_past = usize::try_from(n_past).unwrap_or(0);

        // The KV cache now holds `n_past` positions, some of them media
        // embeddings with no token; the sentinel keeps a later text prompt from
        // prefix-matching them, and lets this prompt count as fully cached.
        engine.input_ids = vec![NO_TOKEN; n_past];
        engine.requires_eval = false;
        let prompt = engine.input_ids.clone();
        self.create_completion_locked(engine, prompt, request, request.stop.clone())
    }

    /// `Llama.reset`.
    fn reset(&self, engine: &mut Engine) {
        engine.input_ids.clear();
        engine.requires_eval = true;
        if self.model.is_recurrent() || self.model.is_hybrid() {
            engine.ctx.clear_kv_cache();
        }
    }

    /// `Llama.eval`.
    fn eval(&self, engine: &mut Engine, tokens: &[LlamaToken]) -> anyhow::Result<()> {
        let n_tokens = engine.input_ids.len() as u32;
        let _ = engine.ctx.clear_kv_cache_seq(None, Some(n_tokens), None);
        for batch_tokens in tokens.chunks(self.n_batch) {
            let n_past = engine.input_ids.len();
            let mut batch = LlamaBatch::new(self.n_batch, 1);
            let last = batch_tokens.len() - 1;
            for (i, &token) in batch_tokens.iter().enumerate() {
                batch.add(NativeToken(token), (n_past + i) as i32, &[0], i == last)?;
            }
            engine.ctx.decode(&mut batch).map_err(|e| anyhow::anyhow!("llama_decode returned {e}"))?;
            engine.input_ids.extend_from_slice(batch_tokens);
            engine.requires_eval = false;
        }
        Ok(())
    }
}

impl LlamaModel for Llama {
    fn state(&self) -> &LlamaState {
        &self.state
    }

    fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    fn metadata(&self) -> HashMap<String, String> {
        self.metadata.clone()
    }

    fn tokenize(&self, text: &[u8]) -> anyhow::Result<Vec<LlamaToken>> {
        Ok(self.tokenize_ext(text, true, false))
    }

    fn detokenize(&self, tokens: &[LlamaToken]) -> anyhow::Result<Vec<u8>> {
        Ok(self.detokenize_ext(tokens, false))
    }

    fn create_chat_completion(&self, request: ChatCompletionRequest) -> anyhow::Result<ChatCompletionOutput<'_>> {
        let handler = resolve_chat_handler(self)?;
        let mut template_kwargs = Map::new();
        if let Some(tools) = request.tools.clone() {
            template_kwargs.insert("tools".to_string(), Value::Array(tools));
        }
        handler(ChatHandlerRequest { llama: Some(self), request, template_kwargs })
    }

    fn model_path(&self) -> String {
        self.params.model_path.clone()
    }

    fn n_gpu_layers(&self) -> i32 {
        self.params.n_gpu_layers
    }

    fn flash_attn(&self) -> bool {
        self.params.flash_attn
    }

    fn as_llama(&self) -> Option<&Llama> {
        Some(self)
    }
}

/// Every GGUF metadata key/value as strings (`_model.metadata()`).
fn read_model_metadata(model: &NativeModel) -> HashMap<String, String> {
    let mut metadata = HashMap::new();
    for i in 0..model.meta_count() {
        let (Ok(key), Ok(value)) = (model.meta_key_by_index(i), model.meta_val_str_by_index(i)) else {
            continue;
        };
        metadata.insert(key, value);
    }
    metadata
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// `Llama.generate`: evaluates the prompt (reusing a matching KV-cache prefix)
/// and then one token per step.
struct TokenGen {
    sampler: LlamaSampler,
    /// Tokens to evaluate before the next sample.
    pending: Vec<LlamaToken>,
}

impl TokenGen {
    fn start(llama: &Llama, engine: &mut Engine, prompt: &[LlamaToken], sampler: LlamaSampler) -> Self {
        let mut tokens = prompt.to_vec();
        let mut reset = true;

        // Check for kv cache prefix match.
        let n_tokens = engine.input_ids.len();
        if n_tokens > 0 {
            let longest_prefix =
                engine.input_ids.iter().zip(tokens.iter()).take_while(|(a, b)| a == b).count();
            let prompt_consumed = longest_prefix == tokens.len();
            let exact_prompt_cached = n_tokens == tokens.len() && prompt_consumed;

            let reuse_prefix;
            // Exact cache hits can sample immediately only when the current
            // logits were produced by a live decode, not restored state.
            if exact_prompt_cached && !engine.requires_eval {
                reset = false;
                tokens.clear();
                reuse_prefix = 0;
            } else {
                // If there is no suffix to decode, replay one token to refresh
                // logits after truncating to a valid prefix.
                reuse_prefix = if prompt_consumed { longest_prefix.saturating_sub(1) } else { longest_prefix };
            }

            // Prefix hits can reuse memory because the suffix decode refreshes
            // logits before sampling.
            if reuse_prefix > 0 {
                if let Ok(true) = engine.ctx.clear_kv_cache_seq(None, Some(reuse_prefix as u32), None) {
                    reset = false;
                    tokens.drain(..reuse_prefix);
                    engine.input_ids.truncate(reuse_prefix);
                    engine.requires_eval = true;
                }
            }
        }

        if reset {
            llama.reset(engine);
        }

        Self { sampler, pending: tokens }
    }

    /// Evaluate what is pending and sample the next token.
    fn next(&mut self, llama: &Llama, engine: &mut Engine) -> anyhow::Result<LlamaToken> {
        let pending = std::mem::take(&mut self.pending);
        if !pending.is_empty() {
            llama.eval(engine, &pending)?;
        }
        if engine.input_ids.is_empty() {
            anyhow::bail!("nothing evaluated to sample from");
        }
        let token = self.sampler.sample(&engine.ctx, -1).0;
        self.pending = vec![token];
        Ok(token)
    }
}

/// One `_create_completion` run: generation state plus the text-completion
/// chunks waiting to be emitted.
struct CompletionRun<'a> {
    llama: &'a Llama,
    engine: MutexGuard<'a, Engine>,
    token_gen: TokenGen,
    stream: bool,
    id: String,
    created: i64,
    model: String,
    prompt_tokens: Vec<LlamaToken>,
    completion_tokens: Vec<LlamaToken>,
    returned_tokens: usize,
    stop_sequences: Vec<Vec<u8>>,
    max_tokens: usize,
    multibyte_fix: i32,
    finish_reason: &'static str,
    text: Vec<u8>,
    generating: bool,
    pending: VecDeque<Value>,
}

impl CompletionRun<'_> {
    fn detok(&self, tokens: &[LlamaToken]) -> Vec<u8> {
        self.llama.detokenize_ext(tokens, false)
    }

    fn text_chunk(&self, text: String, finish_reason: Option<&str>) -> Value {
        json!({
            "id": self.id,
            "object": "text_completion",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "text": text,
                "index": 0,
                "logprobs": Value::Null,
                "finish_reason": finish_reason,
            }],
        })
    }

    /// One iteration of the token loop.  Returns `false` once generation ended.
    fn step(&mut self) -> anyhow::Result<bool> {
        let token = {
            let llama = self.llama;
            let engine: &mut Engine = &mut self.engine;
            self.token_gen.next(llama, engine)?
        };

        if self.llama.is_eog(token) {
            self.text = self.detok(&self.completion_tokens);
            self.finish_reason = "stop";
            return Ok(false);
        }

        self.completion_tokens.push(token);
        let all_text = self.detok(&self.completion_tokens);

        // Contains multi-byte UTF8.
        let tail_start = all_text.len().saturating_sub(3);
        for (k, &ch) in all_text[tail_start..].iter().enumerate() {
            let k = 3 - k as i32;
            for (num, pattern) in [(2, 192u8), (3, 224u8), (4, 240u8)] {
                if num > k && pattern & ch == pattern {
                    self.multibyte_fix = num - k;
                }
            }
        }
        // Stop incomplete bytes from passing.
        if self.multibyte_fix > 0 {
            self.multibyte_fix -= 1;
            return Ok(true);
        }

        if let Some(first_stop) = self.stop_sequences.iter().find(|s| find_bytes(&all_text, s).is_some()) {
            let at = find_bytes(&all_text, first_stop).unwrap_or(all_text.len());
            self.text = all_text[..at].to_vec();
            self.finish_reason = "stop";
            return Ok(false);
        }

        if self.stream {
            self.emit_safe_text();
        }

        if self.completion_tokens.len() >= self.max_tokens {
            self.text = self.detok(&self.completion_tokens);
            self.finish_reason = "length";
            return Ok(false);
        }

        // Checked every 16 tokens: a loop is caught a few tokens late, at a
        // sixteenth of the cost.
        if self.completion_tokens.len() % 16 == 0 {
            if let Some(run) = repetition_run(&self.completion_tokens) {
                // Streamed text already sent cannot be taken back.
                let keep = (self.completion_tokens.len() - run).max(self.returned_tokens);
                tracing::warn!(
                    target: "smartloop.llms.llama",
                    "Generation stopped: output repeating ({run} tokens looped after {keep})"
                );
                self.completion_tokens.truncate(keep);
                self.text = self.detok(&self.completion_tokens);
                self.finish_reason = "length";
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Queue the newly decodable text, holding back anything that could be
    /// the start of a stop sequence.
    fn emit_safe_text(&mut self) {
        let mut remaining_tokens: Vec<LlamaToken> = self.completion_tokens[self.returned_tokens..].to_vec();
        let remaining_text = self.detok(&remaining_tokens);
        let remaining_length = remaining_text.len();

        // We want to avoid yielding any characters from the generated text if
        // they are part of a stop sequence.
        let mut first_stop_position = 0;
        for s in &self.stop_sequences {
            for i in (1..=s.len().min(remaining_length)).rev() {
                if remaining_text.ends_with(&s[..i]) {
                    if i > first_stop_position {
                        first_stop_position = i;
                    }
                    break;
                }
            }
        }

        let mut token_end_position = 0;
        while !remaining_tokens.is_empty() {
            let mut decoded: Option<(usize, Vec<u8>, String)> = None;
            for i in 1..=remaining_tokens.len() {
                let bs = self.detok(&remaining_tokens[..i]);
                if let Ok(ts) = String::from_utf8(bs.clone()) {
                    decoded = Some((i, bs, ts));
                    break;
                }
            }
            // All remaining tokens cannot be decoded to a UTF-8 character.
            let Some((i, bs, ts)) = decoded else { break };
            token_end_position += bs.len();
            if token_end_position > remaining_length - first_stop_position {
                break;
            }
            remaining_tokens.drain(..i);
            self.returned_tokens += i;
            let chunk = self.text_chunk(ts, None);
            self.pending.push_back(chunk);
        }
    }

    /// The end of a streamed completion: release what was held back (up to
    /// any stop sequence) and the final chunk carrying the finish reason.
    fn finish_stream(&mut self) {
        let remaining_tokens: Vec<LlamaToken> = self.completion_tokens[self.returned_tokens..].to_vec();
        let remaining_text = self.detok(&remaining_tokens);
        let end = self
            .stop_sequences
            .iter()
            .filter_map(|s| find_bytes(&remaining_text, s))
            .min()
            .unwrap_or(remaining_text.len());

        let mut token_end_position = 0;
        for &token in &remaining_tokens {
            let piece = self.detok(&[token]);
            token_end_position += piece.len();
            if token_end_position >= end {
                self.returned_tokens += 1;
                let keep = piece.len().saturating_sub(token_end_position - end);
                let chunk = self.text_chunk(decode_utf8_ignore(&piece[..keep]), None);
                self.pending.push_back(chunk);
                break;
            }
            self.returned_tokens += 1;
            let chunk = self.text_chunk(decode_utf8_ignore(&piece), None);
            self.pending.push_back(chunk);
        }
        let last = self.text_chunk(String::new(), Some(self.finish_reason));
        self.pending.push_back(last);
    }

    /// Run a non-streamed completion and build the chat-completion response.
    fn run_to_completion(mut self) -> anyhow::Result<Value> {
        while self.step()? {}
        let text = decode_utf8_ignore(&self.text);
        let prompt_tokens = self.prompt_tokens.len();
        let completion_tokens = self.completion_tokens.len();
        Ok(json!({
            "id": format!("chat{}", self.id),
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "logprobs": Value::Null,
                "finish_reason": self.finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }))
    }

    /// The next text-completion chunk of a streamed run.
    fn next_text_chunk(&mut self) -> Option<Value> {
        loop {
            if let Some(chunk) = self.pending.pop_front() {
                return Some(chunk);
            }
            if !self.generating {
                return None;
            }
            match self.step() {
                Ok(true) => {}
                Ok(false) => {
                    self.generating = false;
                    self.finish_stream();
                }
                Err(e) => {
                    tracing::error!(target: "smartloop.llms.llama", "llama.cpp generation failed: {e}");
                    self.generating = false;
                    self.finish_stream();
                }
            }
        }
    }
}

/// `_convert_text_completion_chunks_to_chat`: a role chunk first, then each
/// text chunk as a `delta`.
struct ChatChunks<'a> {
    run: CompletionRun<'a>,
    first: bool,
    queued: VecDeque<Value>,
}

impl<'a> ChatChunks<'a> {
    fn new(run: CompletionRun<'a>) -> Self {
        Self { run, first: true, queued: VecDeque::new() }
    }
}

impl Iterator for ChatChunks<'_> {
    type Item = Value;

    fn next(&mut self) -> Option<Value> {
        if let Some(v) = self.queued.pop_front() {
            return Some(v);
        }
        let chunk = self.run.next_text_chunk()?;
        let id = format!("chat{}", chunk["id"].as_str().unwrap_or_default());
        let model = chunk["model"].clone();
        let created = chunk["created"].clone();
        let choice = &chunk["choices"][0];
        let finish_reason = choice["finish_reason"].clone();
        if self.first {
            self.first = false;
            self.queued.push_back(json!({
                "id": id,
                "model": model,
                "created": created,
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "logprobs": Value::Null,
                    "finish_reason": Value::Null,
                }],
            }));
        }
        let delta = if finish_reason.is_null() { json!({"content": choice["text"].clone()}) } else { json!({}) };
        self.queued.push_back(json!({
            "id": id,
            "model": model,
            "created": created,
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": delta,
                "logprobs": Value::Null,
                "finish_reason": finish_reason,
            }],
        }));
        self.queued.pop_front()
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

// ---------------------------------------------------------------------------
// Chat formats
// ---------------------------------------------------------------------------

/// `CHATML_CHAT_TEMPLATE`.
pub const CHATML_CHAT_TEMPLATE: &str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
/// `MISTRAL_INSTRUCT_CHAT_TEMPLATE`.
pub const MISTRAL_INSTRUCT_CHAT_TEMPLATE: &str = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}";
/// `MIXTRAL_INSTRUCT_CHAT_TEMPLATE`.
pub const MIXTRAL_INSTRUCT_CHAT_TEMPLATE: &str = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}";
/// `LLAMA3_INSTRUCT_CHAT_TEMPLATE`.
pub const LLAMA3_INSTRUCT_CHAT_TEMPLATE: &str = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}";

/// `guess_chat_format_from_gguf_metadata`.
pub fn guess_chat_format_from_gguf_metadata(metadata: &HashMap<String, String>) -> Option<&'static str> {
    let template = metadata.get("tokenizer.chat_template")?;
    if template == CHATML_CHAT_TEMPLATE {
        return Some("chatml");
    }
    if template == MISTRAL_INSTRUCT_CHAT_TEMPLATE || template == MIXTRAL_INSTRUCT_CHAT_TEMPLATE {
        return Some("mistral-instruct");
    }
    if template == LLAMA3_INSTRUCT_CHAT_TEMPLATE {
        return Some("llama-3");
    }
    None
}

/// `ChatFormatterResponse`.
#[derive(Debug, Clone, Default)]
struct ChatFormatterResponse {
    prompt: String,
    stop: Option<Vec<String>>,
    added_special: bool,
}

type ChatFormatter = Arc<dyn Fn(&[Value], &Map<String, Value>) -> anyhow::Result<ChatFormatterResponse> + Send + Sync>;

/// `chat_formatter_to_chat_completion_handler`.
fn chat_formatter_to_chat_completion_handler(formatter: ChatFormatter) -> ChatHandler {
    Arc::new(move |req: ChatHandlerRequest<'_>| -> anyhow::Result<ChatCompletionOutput<'_>> {
        let llama = req
            .llama
            .and_then(|l| l.as_llama())
            .ok_or_else(|| anyhow::anyhow!("chat handler requires an in-process llama.cpp model"))?;
        let result = formatter(&req.request.messages, &req.template_kwargs)?;
        let prompt = llama.tokenize_ext(result.prompt.as_bytes(), !result.added_special, true);
        let mut stop = req.request.stop.clone();
        if let Some(extra) = result.stop {
            stop.extend(extra);
        }
        llama.create_chat_completion_from_tokens(prompt, &req.request, stop)
    })
}

/// A Jinja2 environment configured like llama-cpp-python's chat formatters:
/// `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)` with
/// loop controls, Python string/dict methods, and (for model templates) the
/// Transformers `tojson` and the `{% generation %}` pass-through.
fn chat_template_env(source: &str, transformers_compat: bool) -> anyhow::Result<minijinja::Environment<'static>> {
    let mut env = minijinja::Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    let source = if transformers_compat {
        env.add_filter("tojson", tojson_filter);
        strip_generation_tags(source)
    } else {
        source.to_string()
    };
    env.add_function("raise_exception", |message: String| -> Result<String, minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, message))
    });
    // Jinja2 doesn't treat None as iterable, but minijinja does; without
    // this, `tools is iterable and tools | length > 0` (Qwen3-Coder) fails
    // on `length` of none whenever a request carries no tools.
    env.add_test("iterable", |value: minijinja::Value| !value.is_none() && value.try_iter().is_ok());
    env.add_function("strftime_now", |format: String| -> String { chrono::Local::now().format(&format).to_string() });
    env.add_template_owned("chat_template", source)?;
    Ok(env)
}

/// Pass-through for HuggingFace's `{% generation %}` chat-template tag.
fn strip_generation_tags(source: &str) -> String {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"\{%-?\s*(?:end)?generation\s*-?%\}").expect("valid regex"));
    re.replace_all(source, "").into_owned()
}

/// Transformers' chat-template `tojson`: `json.dumps(x, ensure_ascii=False,
/// indent=None, separators=None, sort_keys=False)`.
fn tojson_filter(value: minijinja::Value, kwargs: Kwargs) -> Result<minijinja::Value, minijinja::Error> {
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let indent: Option<minijinja::Value> = kwargs.get("indent")?;
    let separators: Option<minijinja::Value> = kwargs.get("separators")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    kwargs.assert_all_used()?;

    let indent = match indent {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => match v.as_str() {
            Some(s) => Some(s.to_string()),
            None => Some(" ".repeat(usize::try_from(v)?)),
        },
    };
    let (item_sep, key_sep) = match separators {
        Some(v) if !v.is_none() => {
            let parts: Vec<String> = v.try_iter()?.map(|p| p.to_string()).collect();
            if parts.len() != 2 {
                return Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    "separators must be a pair of strings",
                ));
            }
            (parts[0].clone(), parts[1].clone())
        }
        _ => {
            if indent.is_some() {
                (",".to_string(), ": ".to_string())
            } else {
                (", ".to_string(), ": ".to_string())
            }
        }
    };
    let json_value = serde_json::to_value(&value)
        .map_err(|e| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string()))?;
    let opts = PyJsonOptions {
        indent,
        item_sep,
        key_sep,
        sort_keys: sort_keys.unwrap_or(false),
        ensure_ascii: ensure_ascii.unwrap_or(false),
    };
    let mut out = String::new();
    py_json_dumps(&json_value, &opts, 0, &mut out);
    Ok(minijinja::Value::from_safe_string(out))
}

struct PyJsonOptions {
    indent: Option<String>,
    item_sep: String,
    key_sep: String,
    sort_keys: bool,
    ensure_ascii: bool,
}

/// Python's `json.dumps` layout over a JSON value.
fn py_json_dumps(value: &Value, opts: &PyJsonOptions, level: usize, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => py_json_string(s, opts.ensure_ascii, out),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(&opts.item_sep);
                }
                py_json_newline(opts, level + 1, out);
                py_json_dumps(item, opts, level + 1, out);
            }
            py_json_newline(opts, level, out);
            out.push(']');
        }
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            if opts.sort_keys {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            out.push('{');
            for (i, (k, v)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(&opts.item_sep);
                }
                py_json_newline(opts, level + 1, out);
                py_json_string(k, opts.ensure_ascii, out);
                out.push_str(&opts.key_sep);
                py_json_dumps(v, opts, level + 1, out);
            }
            py_json_newline(opts, level, out);
            out.push('}');
        }
    }
}

fn py_json_newline(opts: &PyJsonOptions, level: usize, out: &mut String) {
    if let Some(indent) = &opts.indent {
        out.push('\n');
        for _ in 0..level {
            out.push_str(indent);
        }
    }
}

fn py_json_string(s: &str, ensure_ascii: bool, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ensure_ascii && (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `Jinja2ChatFormatter`: renders the GGUF's own chat template.
pub struct Jinja2ChatFormatter {
    env: minijinja::Environment<'static>,
    eos_token: String,
    bos_token: String,
    add_generation_prompt: bool,
    #[allow(dead_code)]
    stop_token_ids: Option<HashSet<LlamaToken>>,
}

impl Jinja2ChatFormatter {
    /// Compile `template`.
    pub fn new(
        template: &str,
        eos_token: &str,
        bos_token: &str,
        add_generation_prompt: bool,
        stop_token_ids: Option<Vec<LlamaToken>>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            env: chat_template_env(template, true)?,
            eos_token: eos_token.to_string(),
            bos_token: bos_token.to_string(),
            add_generation_prompt,
            stop_token_ids: stop_token_ids.map(|ids| ids.into_iter().collect()),
        })
    }

    /// Render the prompt for `messages`, with `kwargs` passed through to the
    /// template (e.g. `enable_thinking`).
    pub fn render(&self, messages: &[Value], kwargs: &Map<String, Value>) -> anyhow::Result<String> {
        let mut ctx: Map<String, Value> = Map::new();
        ctx.insert("messages".into(), Value::Array(messages.to_vec()));
        ctx.insert("eos_token".into(), Value::String(self.eos_token.clone()));
        ctx.insert("bos_token".into(), Value::String(self.bos_token.clone()));
        ctx.insert("add_generation_prompt".into(), Value::Bool(self.add_generation_prompt));
        for key in ["functions", "function_call", "tools", "tool_choice"] {
            ctx.insert(key.into(), Value::Null);
        }
        for (k, v) in kwargs {
            ctx.insert(k.clone(), v.clone());
        }
        let template = self.env.get_template("chat_template")?;
        Ok(template.render(minijinja::Value::from_serialize(&ctx))?)
    }

    fn format(&self, messages: &[Value], kwargs: &Map<String, Value>) -> anyhow::Result<ChatFormatterResponse> {
        // The `stop_token_ids` stopping criterion of llama-cpp-python inspects
        // the last *evaluated* token, which is never the EOS it lists (an EOS
        // ends generation before it is evaluated), so the end-of-generation
        // check in the completion loop is the effective stop.
        Ok(ChatFormatterResponse {
            prompt: self.render(messages, kwargs)?,
            stop: Some(vec![self.eos_token.clone()]),
            added_special: true,
        })
    }

    /// `to_chat_handler`.
    pub fn to_chat_handler(self) -> ChatHandler {
        let formatter = Arc::new(self);
        chat_formatter_to_chat_completion_handler(Arc::new(move |messages, kwargs| formatter.format(messages, kwargs)))
    }
}

/// `_get_system_message`: the first system message's content.
fn get_system_message(messages: &[Value]) -> String {
    for message in messages {
        if message.get("role").and_then(Value::as_str) == Some("system") {
            return message.get("content").and_then(Value::as_str).unwrap_or_default().to_string();
        }
    }
    String::new()
}

/// `_map_roles`: `(mapped role, string content or None)` for mapped roles.
fn map_roles(messages: &[Value], role_map: &[(&str, &str)]) -> Vec<(String, Option<String>)> {
    let mut output = Vec::new();
    for message in messages {
        let role = message.get("role").and_then(Value::as_str).unwrap_or_default();
        if let Some((_, mapped)) = role_map.iter().find(|(r, _)| *r == role) {
            let content = message.get("content").and_then(Value::as_str).map(str::to_string);
            output.push((mapped.to_string(), content));
        }
    }
    output
}

fn format_llama2(messages: &[Value]) -> ChatFormatterResponse {
    let roles = [("user", "<s>[INST]"), ("assistant", "[/INST]")];
    let mapped = map_roles(messages, &roles);
    let mut system_message = get_system_message(messages);
    if !system_message.is_empty() {
        system_message = format!("[INST] <<SYS>>\n{system_message}\n<</SYS>>");
    }
    let seps = [" ", "</s>"];
    let mut ret = format!("{system_message} ");
    for (i, (role, message)) in mapped.iter().enumerate() {
        if !system_message.is_empty() && i == 0 {
            ret.push_str(message.as_deref().unwrap_or_default());
            ret.push_str(seps[i % 2]);
        } else if let Some(m) = message.as_deref().filter(|m| !m.is_empty()) {
            ret.push_str(&format!("{role}{m} {}", seps[i % 2]));
        } else {
            ret.push_str(&format!("{role} "));
        }
    }
    ret.push_str("[/INST]");
    ChatFormatterResponse { prompt: ret, stop: None, added_special: false }
}

fn format_llama3(messages: &[Value]) -> ChatFormatterResponse {
    let roles = [
        ("system", "<|start_header_id|>system<|end_header_id|>\n\n"),
        ("user", "<|start_header_id|>user<|end_header_id|>\n\n"),
        ("assistant", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    ];
    let sep = "<|eot_id|>";
    let mut mapped = map_roles(messages, &roles);
    mapped.push((roles[2].1.to_string(), None));
    // _format_no_colon_single
    let mut ret = String::new();
    for (role, message) in mapped {
        match message.filter(|m| !m.is_empty()) {
            Some(m) => ret.push_str(&format!("{role}{m}{sep}")),
            None => ret.push_str(&role),
        }
    }
    ChatFormatterResponse { prompt: ret, stop: Some(vec![sep.to_string()]), added_special: false }
}

fn format_chatml(messages: &[Value]) -> ChatFormatterResponse {
    let system_message = format!("<|im_start|>system\n{}", get_system_message(messages));
    let roles = [("user", "<|im_start|>user"), ("assistant", "<|im_start|>assistant")];
    let sep = "<|im_end|>";
    let mut mapped = map_roles(messages, &roles);
    mapped.push((roles[1].1.to_string(), None));
    let mut ret = if system_message.is_empty() { String::new() } else { format!("{system_message}{sep}\n") };
    for (role, message) in mapped {
        match message.filter(|m| !m.is_empty()) {
            Some(m) => ret.push_str(&format!("{role}\n{m}{sep}\n")),
            None => ret.push_str(&format!("{role}\n")),
        }
    }
    ChatFormatterResponse { prompt: ret, stop: Some(vec![sep.to_string()]), added_special: false }
}

fn format_mistral_instruct(messages: &[Value]) -> ChatFormatterResponse {
    let eos = "</s>";
    let mut prompt = String::new();
    for message in messages {
        let role = message.get("role").and_then(Value::as_str).unwrap_or_default();
        let content = message.get("content");
        if role == "user" {
            if let Some(c) = content.and_then(Value::as_str) {
                prompt.push_str("[INST] ");
                prompt.push_str(c);
            }
        } else if role == "assistant" && content.is_some_and(|c| !c.is_null()) {
            prompt.push_str(" [/INST]");
            prompt.push_str(&content.map(json_str).unwrap_or_default());
            prompt.push_str(eos);
        }
    }
    prompt.push_str(" [/INST]");
    ChatFormatterResponse { prompt, stop: Some(vec![eos.to_string()]), added_special: false }
}

fn json_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// The built-in (non-template) chat formats this port provides.
const BUILTIN_CHAT_FORMATS: &[&str] = &["llama-2", "llama-3", "chatml", "mistral-instruct"];

/// `llama_chat_format.get_chat_completion_handler(name)`: the built-in handler
/// for a named chat format.
pub fn get_chat_completion_handler(chat_format: Option<&str>) -> anyhow::Result<ChatHandler> {
    let formatter: fn(&[Value]) -> ChatFormatterResponse = match chat_format {
        Some("llama-2") => format_llama2,
        Some("llama-3") => format_llama3,
        Some("chatml") => format_chatml,
        Some("mistral-instruct") => format_mistral_instruct,
        other => {
            anyhow::bail!(
                "Invalid chat handler: {} (valid formats: {:?})",
                other.unwrap_or("None"),
                BUILTIN_CHAT_FORMATS
            )
        }
    };
    Ok(chat_formatter_to_chat_completion_handler(Arc::new(move |messages, _kwargs| Ok(formatter(messages)))))
}

/// Resolve the handler for a completion, in llama-cpp-python order:
/// `chat_handler`, then `_chat_handlers[chat_format]`, then the built-in
/// handler for `chat_format` ([`get_chat_completion_handler`]).
pub fn resolve_chat_handler(llama: &dyn LlamaModel) -> anyhow::Result<ChatHandler> {
    let state = llama.state();
    if let Some(handler) = state.chat_handler() {
        return Ok(handler);
    }
    let chat_format = state.chat_format();
    if let Some(name) = &chat_format {
        if let Some(handler) = state.registered_handler(name)? {
            return Ok(handler);
        }
    } else {
        // `_chat_handlers.get(None)`: still requires the registry.
        state.registered_handler("")?;
    }
    get_chat_completion_handler(chat_format.as_deref())
}

// ---------------------------------------------------------------------------
// Vision (multimodal) chat handlers
// ---------------------------------------------------------------------------

/// Names of the vision chat handlers this build provides (the
/// `llama_chat_format` classes the factory looks up by name).
pub const VISION_CHAT_HANDLERS: &[&str] = &[
    "MTMDChatHandler",
    "Gemma4ChatHandler",
    "Qwen25VLChatHandler",
    "MiniCPMv26ChatHandler",
    "NanoLlavaChatHandler",
    "MoondreamChatHandler",
    "Llava16ChatHandler",
];

/// Whether `name` is a vision chat handler available in this build.
pub fn vision_chat_handler_available(name: &str) -> bool {
    VISION_CHAT_HANDLERS.contains(&name)
}

/// Whether handler `name` takes a `use_gpu` constructor argument
/// (the `MTMDChatHandler` family).
pub fn vision_chat_handler_takes_use_gpu(name: &str) -> bool {
    matches!(name, "MTMDChatHandler" | "Gemma4ChatHandler")
}

/// Set once by [`force_clip_on_cpu`]: every later projector context is
/// created with `use_gpu = false`.
static CLIP_FORCED_TO_CPU: AtomicBool = AtomicBool::new(false);

/// Force every later projector (CLIP/mmproj) context onto CPU — the port of
/// the `mtmd_init_from_file` patch.  Idempotent; returns `true` the first time.
pub fn force_clip_on_cpu() -> bool {
    !CLIP_FORCED_TO_CPU.swap(true, Ordering::SeqCst)
}

const LLAVA15_SYSTEM: &str = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.";

const LLAVA15_CHAT_FORMAT: &str = concat!(
    "{% for message in messages %}",
    "{% if message.role == 'system' %}",
    "{{ message.content }}",
    "{% endif %}",
    "{% if message.role == 'user' %}",
    "{% if message.content is string %}",
    "\nUSER: {{ message.content }}",
    "{% endif %}",
    "{% if message.content is iterable %}",
    "\nUSER: ",
    "{% for content in message.content %}",
    "{% if content.type == 'image_url' and content.image_url is string %}",
    "{{ content.image_url }}",
    "{% endif %}",
    "{% if content.type == 'image_url' and content.image_url is mapping %}",
    "{{ content.image_url.url }}",
    "{% endif %}",
    "{% endfor %}",
    "{% for content in message.content %}",
    "{% if content.type == 'text' %}",
    "{{ content.text }}",
    "{% endif %}",
    "{% endfor %}",
    "{% endif %}",
    "{% endif %}",
    "{% if message.role == 'assistant' and message.content is not none %}",
    "\nASSISTANT: {{ message.content }}",
    "{% endif %}",
    "{% endfor %}",
    "{% if add_generation_prompt %}",
    "\nASSISTANT: ",
    "{% endif %}",
);

const MOONDREAM_CHAT_FORMAT: &str = concat!(
    "{% for message in messages %}",
    "{% if message.role == 'user' %}",
    "{% if message.content is iterable %}",
    "{% for content in message.content %}",
    "{% if content.type == 'image_url' %}",
    "{% if content.image_url is string %}",
    "{{ content.image_url }}\n\n",
    "{% endif %}",
    "{% if content.image_url is mapping %}",
    "{{ content.image_url.url }}\n\n",
    "{% endif %}",
    "{% endif %}",
    "{% endfor %}",
    "{% for content in message.content %}",
    "{% if content.type == 'text' %}",
    "Question: {{ content.text }}\n\n",
    "{% endif %}",
    "{% endfor %}",
    "{% endif %}",
    "{% if message.content is string %}",
    "Question: {{ message.content }}\n\n",
    "{% endif %}",
    "{% endif %}",
    "{% if message.role == 'assistant' %}",
    "Answer:{{ message.content }}\n\n",
    "{% endif %}",
    "{% endfor %}",
    "{% if add_generation_prompt %}",
    "Answer:",
    "{% endif %}",
);

const LLAVA16_SYSTEM: &str = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. ";

const LLAVA16_CHAT_FORMAT: &str = concat!(
    "{% for message in messages %}",
    "{% if message.role == 'system' %}",
    "{{ message.content }}",
    "{% endif %}",
    "{% if message.role == 'user' %}",
    "{% if message.content is iterable %}",
    "{% for content in message.content %}",
    "{% if content.type == 'image_url' %}",
    "{% if content.image_url is string %}",
    "{{ content.image_url }}\n",
    "{% endif %}",
    "{% if content.image_url is mapping %}",
    "{{ content.image_url.url }}\n",
    "{% endif %}",
    "{% endif %}",
    "{% endfor %}",
    "{% for content in message.content %}",
    "{% if content.type == 'text' %}",
    "{{ content.text }}",
    "{% endif %}",
    "{% endfor %}",
    "{% endif %}",
    "{% if message.content is string %}",
    "{{ message.content }}",
    "{% endif %}",
    "{% endif %}",
    "{% if message.role == 'assistant' %}",
    "{{ message.content }}",
    "{% endif %}",
    "{% endfor %}",
    "{% if add_generation_prompt %}",
    "Answer:",
    "{% endif %}",
);

const NANOLLAVA_SYSTEM: &str = "Answer the question";

const NANOLLAVA_CHAT_FORMAT: &str = concat!(
    "{% for message in messages %}",
    "{% if message.role == 'system' %}",
    "<|im_start|>system\n",
    "{{ message.content }}",
    "<|im_end|>",
    "{% endif %}",
    "{% if message.role == 'user' %}",
    "<|im_start|>user\n",
    "{% if message.content is string %}",
    "{{ message.content }}",
    "{% endif %}",
    "{% if message.content is iterable %}",
    "{% for content in message.content %}",
    "{% if content.type == 'image_url' and content.image_url is string %}",
    "{{ content.image_url }}",
    "{% endif %}",
    "{% if content.type == 'image_url' and content.image_url is mapping %}",
    "{{ content.image_url.url }}",
    "{% endif %}",
    "{% endfor %}",
    "{% for content in message.content %}",
    "{% if content.type == 'text' %}",
    "{{ content.text }}",
    "{% endif %}",
    "{% endfor %}",
    "{% endif %}",
    "<|im_end|>",
    "{% endif %}",
    "{% if message.role == 'assistant' %}",
    "<|im_start|>assistant\n",
    "{{ message.content }}",
    "<|im_end|>",
    "{% endif %}",
    "{% endfor %}",
    "{% if add_generation_prompt %}",
    "<|im_start|>assistant\n",
    "{% endif %}",
);

const HELPFUL_ASSISTANT_SYSTEM: &str = "You are a helpful assistant.";

const MINICPMV26_CHAT_FORMAT: &str = concat!(
    "{% for message in messages %}",
    "{% if loop.first and messages[0]['role'] != 'system' %}",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
    "{% endif %}",
    "<|im_start|>{{ message['role'] }}\n",
    "{% if message['content'] is iterable %}",
    "{% for content in message['content'] %}",
    "{% if content.type == 'image_url' %}",
    "{% if content.image_url is string %}",
    "{{ content.image_url }}",
    "{% endif %}",
    "{% if content.image_url is mapping %}",
    "{{ content.image_url.url }}",
    "{% endif %}",
    "{% endif %}",
    "{% endfor %}",
    "{% for content in message['content'] %}",
    "{% if content.type == 'text' %}",
    "{{ content.text }}",
    "{% endif %}",
    "{% endfor %}",
    "{% endif %}",
    "{% if message['content'] is string %}",
    "{{ message['content'] }}",
    "{% endif %}",
    "<|im_end|>\n",
    "{% endfor %}",
    "{% if add_generation_prompt %}",
    "<|im_start|>assistant\n",
    "{% endif %}",
);

const QWEN25VL_CHAT_FORMAT: &str = concat!(
    "{% for message in messages %}",
    "{% if loop.first and message['role'] != 'system' %}",
    "<|im_start|>system\n",
    "{{ self.DEFAULT_SYSTEM_MESSAGE }}<|im_end|>\n",
    "{% endif %}",
    "<|im_start|>{{ message['role'] }}\n",
    "{% if message['content'] is string %}",
    "{{ message['content'] }}<|im_end|>\n",
    "{% else %}",
    "{% for content in message['content'] %}",
    "{% if content['type'] == 'image_url' %}",
    "{% if content.image_url is string %}",
    "{{ content.image_url }}",
    "{% else %}",
    "{{ content.image_url.url }}",
    "{% endif %}",
    "{% elif content['type'] == 'text' %}",
    "{{ content['text'] }}",
    "{% endif %}",
    "{% endfor %}",
    "<|im_end|>\n",
    "{% endif %}",
    "{% endfor %}",
    "<|im_start|>assistant\n",
);

/// How a vision handler builds its prompt.
#[derive(Clone, Copy)]
enum VisionKind {
    /// `MTMDChatHandler`: renders the model's own chat template.
    Mtmd,
    /// The `Llava15ChatHandler` family: a fixed template and default system
    /// message.
    Llava { default_system: Option<&'static str>, chat_format: &'static str },
}

fn vision_kind(name: &str) -> Option<VisionKind> {
    Some(match name {
        "MTMDChatHandler" | "Gemma4ChatHandler" => VisionKind::Mtmd,
        "Qwen25VLChatHandler" => {
            VisionKind::Llava { default_system: Some(HELPFUL_ASSISTANT_SYSTEM), chat_format: QWEN25VL_CHAT_FORMAT }
        }
        "MiniCPMv26ChatHandler" => {
            VisionKind::Llava { default_system: Some(HELPFUL_ASSISTANT_SYSTEM), chat_format: MINICPMV26_CHAT_FORMAT }
        }
        "NanoLlavaChatHandler" => {
            VisionKind::Llava { default_system: Some(NANOLLAVA_SYSTEM), chat_format: NANOLLAVA_CHAT_FORMAT }
        }
        "MoondreamChatHandler" => {
            VisionKind::Llava { default_system: Some(LLAVA15_SYSTEM), chat_format: MOONDREAM_CHAT_FORMAT }
        }
        "Llava16ChatHandler" => {
            VisionKind::Llava { default_system: Some(LLAVA16_SYSTEM), chat_format: LLAVA16_CHAT_FORMAT }
        }
        "Llava15ChatHandler" => {
            VisionKind::Llava { default_system: Some(LLAVA15_SYSTEM), chat_format: LLAVA15_CHAT_FORMAT }
        }
        _ => return None,
    })
}

/// A multimodal chat handler with its lazily created projector context.
struct VisionChatHandler {
    name: String,
    kind: VisionKind,
    clip_model_path: String,
    use_gpu: bool,
    mtmd: Mutex<Option<MtmdContext>>,
}

impl VisionChatHandler {
    /// `_init_mtmd_context`.
    fn init_mtmd_context<'g>(
        &self,
        guard: &'g mut MutexGuard<'_, Option<MtmdContext>>,
        llama: &Llama,
    ) -> anyhow::Result<&'g MtmdContext> {
        if guard.is_none() {
            let use_gpu = self.use_gpu && !CLIP_FORCED_TO_CPU.load(Ordering::SeqCst);
            let params = MtmdContextParams {
                use_gpu,
                print_timings: llama.params.verbose,
                n_threads: llama.n_threads() as i32,
                media_marker: CString::new(mtmd_default_marker())?,
                ..MtmdContextParams::default()
            };
            let ctx = MtmdContext::init_from_file(&self.clip_model_path, &llama.model, &params)
                .map_err(|_| anyhow::anyhow!("Failed to load mtmd context from: {}", self.clip_model_path))?;
            if !ctx.support_vision() {
                anyhow::bail!("Vision is not supported by this model");
            }
            **guard = Some(ctx);
        }
        Ok(guard.as_ref().expect("initialized above"))
    }

    fn call<'a>(&self, req: ChatHandlerRequest<'a>) -> anyhow::Result<ChatCompletionOutput<'a>> {
        let llama = req
            .llama
            .and_then(|l| l.as_llama())
            .ok_or_else(|| anyhow::anyhow!("{} requires an in-process llama.cpp model", self.name))?;

        let mut guard = self.mtmd.lock();
        let mtmd = self.init_mtmd_context(&mut guard, llama)?;
        let media_marker = mtmd_default_marker().to_string();

        let mut messages = req.request.messages.clone();
        let image_urls = get_image_urls(&messages);

        let text = match self.kind {
            VisionKind::Llava { default_system, chat_format } => {
                if let Some(system) = default_system {
                    if get_system_message(&messages).is_empty() {
                        messages.insert(0, json!({"role": "system", "content": system}));
                    }
                }
                let env = chat_template_env(chat_format, false)?;
                let mut ctx = Map::new();
                ctx.insert("messages".into(), Value::Array(messages));
                ctx.insert("add_generation_prompt".into(), Value::Bool(true));
                ctx.insert("eos_token".into(), Value::String(decode_utf8_ignore(&llama.detokenize_ext(&[llama.token_eos()], false))));
                ctx.insert("bos_token".into(), Value::String(decode_utf8_ignore(&llama.detokenize_ext(&[llama.token_bos()], false))));
                // `self` in the template is Jinja's template reference: any
                // attribute on it is undefined and renders empty.
                ctx.insert("self".into(), Value::Object(Map::new()));
                let text = env.get_template("chat_template")?.render(minijinja::Value::from_serialize(&ctx))?;
                replace_urls(text, &image_urls, &media_marker)
            }
            VisionKind::Mtmd => {
                let template = llama
                    .metadata
                    .get("tokenizer.chat_template")
                    .filter(|t| !t.is_empty())
                    .ok_or_else(|| anyhow::anyhow!("{} requires tokenizer.chat_template metadata", self.name))?;
                let env = chat_template_env(template, true)?;
                let template_messages: Vec<Value> =
                    messages.iter().map(|m| convert_message_for_template(m, &media_marker)).collect();
                let mut ctx = Map::new();
                ctx.insert("messages".into(), Value::Array(template_messages));
                ctx.insert("add_generation_prompt".into(), Value::Bool(true));
                ctx.insert("eos_token".into(), Value::String(decode_utf8_ignore(&llama.detokenize_ext(&[llama.token_eos()], false))));
                ctx.insert("bos_token".into(), Value::String(decode_utf8_ignore(&llama.detokenize_ext(&[llama.token_bos()], false))));
                for key in ["functions", "function_call", "tools", "tool_choice"] {
                    ctx.insert(key.into(), Value::Null);
                }
                for (k, v) in &req.template_kwargs {
                    ctx.insert(k.clone(), v.clone());
                }
                let text = env.get_template("chat_template")?.render(minijinja::Value::from_serialize(&ctx))?;
                replace_urls(text, &image_urls, &media_marker)
            }
        };

        if llama.params.verbose {
            tracing::debug!(target: "smartloop.llms.llama", "{text}");
        }

        let mut bitmaps = Vec::with_capacity(image_urls.len());
        for url in &image_urls {
            let bytes = load_image(url)?;
            let bitmap = MtmdBitmap::from_buffer(mtmd, &bytes, false)
                .map_err(|_| anyhow::anyhow!("Failed to create bitmap from image bytes"))?;
            bitmaps.push(bitmap);
        }
        let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
        let chunks = mtmd
            .tokenize(MtmdInputText { text, add_special: true, parse_special: true }, &bitmap_refs)
            .map_err(|e| {
                let code = match e {
                    MtmdTokenizeError::BitmapCountMismatch => 1,
                    MtmdTokenizeError::ImagePreprocessingError => 2,
                    MtmdTokenizeError::UnknownError(code) => code,
                    _ => -1,
                };
                anyhow::anyhow!("Failed to tokenize input: error code {code}")
            })?;

        llama.create_chat_completion_from_mtmd(mtmd, &chunks, &req.request)
    }
}

/// `_convert_message_for_template`: image parts become the media marker.
fn convert_message_for_template(message: &Value, media_marker: &str) -> Value {
    let mut message = message.clone();
    if let Some(Value::Array(parts)) = message.get_mut("content") {
        for part in parts.iter_mut() {
            if part.get("type").and_then(Value::as_str) == Some("image_url") {
                *part = json!({"type": "text", "text": media_marker});
            }
        }
    }
    message
}

fn replace_urls(mut text: String, image_urls: &[String], media_marker: &str) -> String {
    for url in image_urls {
        text = text.replace(url.as_str(), media_marker);
    }
    text
}

/// `get_image_urls`: every `image_url` part of the user messages.
fn get_image_urls(messages: &[Value]) -> Vec<String> {
    let mut urls = Vec::new();
    for message in messages {
        if message.get("role").and_then(Value::as_str) != Some("user") {
            continue;
        }
        let Some(Value::Array(parts)) = message.get("content") else { continue };
        for part in parts {
            if part.get("type").and_then(Value::as_str) == Some("image_url") {
                match part.get("image_url") {
                    Some(Value::Object(obj)) if obj.contains_key("url") => {
                        urls.push(json_str(&obj["url"]));
                    }
                    Some(other) => urls.push(json_str(other)),
                    None => {}
                }
            }
        }
    }
    urls
}

/// `_load_image`: a `data:` URL is base64-decoded, anything else fetched.
fn load_image(image_url: &str) -> anyhow::Result<Vec<u8>> {
    use base64::Engine as _;
    if image_url.starts_with("data:") {
        let payload = image_url
            .split(',')
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("list index out of range"))?;
        return Ok(base64::engine::general_purpose::STANDARD.decode(payload)?);
    }
    // Fetched on a scoped thread: the blocking client must not run on an
    // async runtime worker.
    let url = image_url.to_string();
    std::thread::scope(|s| {
        s.spawn(move || -> anyhow::Result<Vec<u8>> {
            let resp = reqwest::blocking::get(&url)?.error_for_status()?;
            Ok(resp.bytes()?.to_vec())
        })
        .join()
        .map_err(|_| anyhow::anyhow!("image fetch thread panicked"))?
    })
}

/// Construct vision chat handler `name` over the CLIP/mmproj at
/// `clip_model_path`.  `use_gpu: None` uses the handler's default.
pub fn make_vision_chat_handler(name: &str, clip_model_path: &str, use_gpu: Option<bool>) -> anyhow::Result<ChatHandler> {
    let kind = vision_kind(name).ok_or_else(|| anyhow::anyhow!("module 'llama_chat_format' has no attribute '{name}'"))?;
    if use_gpu.is_some() && !vision_chat_handler_takes_use_gpu(name) {
        anyhow::bail!("{name}.__init__() got an unexpected keyword argument 'use_gpu'");
    }
    if !std::path::Path::new(clip_model_path).exists() {
        anyhow::bail!("Clip model path does not exist: {clip_model_path}");
    }
    let handler = Arc::new(VisionChatHandler {
        name: name.to_string(),
        kind,
        clip_model_path: clip_model_path.to_string(),
        // The Llava15 family hardcodes `use_gpu = True`.
        use_gpu: use_gpu.unwrap_or(true),
        mtmd: Mutex::new(None),
    });
    Ok(Arc::new(move |req: ChatHandlerRequest<'_>| handler.call(req)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repetition_run_catches_a_loop_and_keeps_one_copy() {
        let cycle: Vec<LlamaToken> = (100..140).collect();
        let mut tokens: Vec<LlamaToken> = (0..50).collect();
        for _ in 0..5 {
            tokens.extend(&cycle);
        }
        let run = repetition_run(&tokens).expect("loop detected");
        assert_eq!(&tokens[..tokens.len() - run], &[(0..50).collect::<Vec<_>>(), cycle].concat()[..]);
    }

    #[test]
    fn repetition_run_ignores_short_or_varied_repeats() {
        // Three copies of a short cycle: under REPEAT_MIN_RUN.
        let short: Vec<LlamaToken> = (0..3).flat_map(|_| 1..20).collect();
        assert_eq!(repetition_run(&short), None);
        // Table-like rows: a shared shape, different values.
        let rows: Vec<LlamaToken> = (0..40).flat_map(|r| [7, 8, 1000 + r, 8, 2000 + r, 8, 9]).collect();
        assert_eq!(repetition_run(&rows), None);
        // One token over and over.
        assert_eq!(repetition_run(&[5; 200]), Some(199));
    }

    #[test]
    fn decode_ignore_drops_invalid_bytes() {
        assert_eq!(decode_utf8_ignore(b"ab\xffcd"), "abcd");
        assert_eq!(decode_utf8_ignore("é".as_bytes()), "é");
    }

    #[test]
    fn tojson_matches_python_json_dumps() {
        let env = chat_template_env(
            "{{ x|tojson }}|{{ x|tojson(indent=2) }}|{{ 'a<b'|tojson }}",
            true,
        )
        .unwrap();
        let out = env
            .get_template("chat_template")
            .unwrap()
            .render(minijinja::context! { x => json!({"a": [1, 2], "b": {}}) })
            .unwrap();
        assert_eq!(out, "{\"a\": [1, 2], \"b\": {}}|{\n  \"a\": [\n    1,\n    2\n  ],\n  \"b\": {}\n}|\"a<b\"");
    }

    #[test]
    fn none_tools_are_not_iterable() {
        let template = "{% if tools is iterable and tools | length > 0 %}tools{% else %}none{% endif %}|{{ messages is iterable }}|{{ 'ab' is iterable }}";
        let f = Jinja2ChatFormatter::new(template, "", "", true, None).unwrap();
        assert_eq!(f.render(&[], &Map::new()).unwrap(), "none|True|True");
        let mut kwargs = Map::new();
        kwargs.insert("tools".into(), json!([{"type": "function"}]));
        assert_eq!(f.render(&[], &kwargs).unwrap(), "tools|True|True");
    }

    #[test]
    fn jinja_formatter_renders_with_kwargs_and_python_methods() {
        let template = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content.strip() }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}";
        let f = Jinja2ChatFormatter::new(template, "<|im_end|>", "", true, None).unwrap();
        let mut kwargs = Map::new();
        kwargs.insert("enable_thinking".into(), Value::Bool(false));
        let out = f.render(&[json!({"role": "user", "content": "  hi  "})], &kwargs).unwrap();
        assert_eq!(out, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }

    #[test]
    fn generation_tags_pass_through() {
        assert_eq!(strip_generation_tags("a{% generation %}b{%- endgeneration -%}c"), "abc");
    }

    #[test]
    fn llama2_format_matches_python() {
        let r = format_llama2(&[json!({"role": "system", "content": "S"}), json!({"role": "user", "content": "U"})]);
        // The system message is embedded and the first mapped message (the
        // user turn) is appended raw.
        assert_eq!(r.prompt, "[INST] <<SYS>>\nS\n<</SYS>> U [/INST]");
    }

    #[test]
    fn guess_chat_format() {
        let mut m = HashMap::new();
        m.insert("tokenizer.chat_template".to_string(), CHATML_CHAT_TEMPLATE.to_string());
        assert_eq!(guess_chat_format_from_gguf_metadata(&m), Some("chatml"));
        m.insert("tokenizer.chat_template".to_string(), "custom".to_string());
        assert_eq!(guess_chat_format_from_gguf_metadata(&m), None);
    }

    #[test]
    fn image_urls_and_template_conversion() {
        let messages = vec![json!({"role": "user", "content": [
            {"type": "text", "text": "what"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]})];
        assert_eq!(get_image_urls(&messages), vec!["data:image/png;base64,AAAA".to_string()]);
        let converted = convert_message_for_template(&messages[0], "<__media__>");
        assert_eq!(converted["content"][1], json!({"type": "text", "text": "<__media__>"}));
        assert_eq!(load_image("data:image/png;base64,AAAA").unwrap(), vec![0, 0, 0]);
    }

    #[test]
    fn llava_template_renders_with_marker() {
        let env = chat_template_env(LLAVA16_CHAT_FORMAT, false).unwrap();
        let messages = json!([
            {"role": "system", "content": "SYS "},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "IMG"}},
                {"type": "text", "text": "Describe"},
            ]},
        ]);
        let out = env
            .get_template("chat_template")
            .unwrap()
            .render(minijinja::context! { messages => messages, add_generation_prompt => true })
            .unwrap();
        assert_eq!(replace_urls(out, &["IMG".to_string()], "<M>"), "SYS <M>\nDescribeAnswer:");
    }

    #[test]
    fn clip_force_is_idempotent() {
        let first = force_clip_on_cpu();
        assert!(!force_clip_on_cpu());
        let _ = first;
    }
}
