//! Hand-port of the parts of `llama-cpp-python` (`llama_cpp.Llama` and
//! `llama_cpp.llama_chat_format`) the app relies on, built on `llama-cpp-2`.
//!
//! The Python wrapper ([`crate::llms::llama_cpp::LlamaCpp`]) and the factory
//! ([`crate::llms::llm_factory::LlmFactory`]) talk to a model through the
//! [`LlamaModel`] trait, so tests can substitute a fake context exactly as the
//! Python tests subclassed `Llama`.  Re-exported from `crate::llms::llama_cpp`.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::{ReentrantMutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

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
        todo!()
    }
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
    pub stop: Vec<String>,
    pub seed: Option<u32>,
    pub stream: bool,
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
            stop: Vec::new(),
            seed: None,
            stream: false,
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

    /// `llm._chat_handlers.get(name)`; `Err` when the registry is absent.
    pub fn registered_handler(&self, name: &str) -> anyhow::Result<Option<ChatHandler>> {
        let _ = name;
        todo!()
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

/// Native state of a loaded model (backend, model, context, sampler, LoRA).
struct LlamaInner {}

/// A llama.cpp model loaded in-process — the Rust `llama_cpp.Llama`.
pub struct Llama {
    params: LlamaParams,
    state: LlamaState,
    metadata: HashMap<String, String>,
    #[allow(dead_code)]
    inner: parking_lot::Mutex<LlamaInner>,
}

impl Llama {
    /// Load a model (`Llama(**config)`), with an optional multimodal chat handler.
    pub fn new(params: LlamaParams, chat_handler: Option<ChatHandler>) -> anyhow::Result<Self> {
        let _ = (params, chat_handler);
        todo!()
    }

    /// The parameters this model was loaded with.
    pub fn params(&self) -> &LlamaParams {
        &self.params
    }
}

impl LlamaModel for Llama {
    fn state(&self) -> &LlamaState {
        &self.state
    }

    fn n_ctx(&self) -> usize {
        todo!()
    }

    fn metadata(&self) -> HashMap<String, String> {
        self.metadata.clone()
    }

    fn tokenize(&self, text: &[u8]) -> anyhow::Result<Vec<LlamaToken>> {
        let _ = text;
        todo!()
    }

    fn detokenize(&self, tokens: &[LlamaToken]) -> anyhow::Result<Vec<u8>> {
        let _ = tokens;
        todo!()
    }

    fn create_chat_completion(&self, request: ChatCompletionRequest) -> anyhow::Result<ChatCompletionOutput<'_>> {
        let _ = request;
        todo!()
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
}

/// Resolve the handler for a completion, in llama-cpp-python order:
/// `chat_handler`, then `_chat_handlers[chat_format]`, then the built-in
/// handler for `chat_format` ([`get_chat_completion_handler`]).
pub fn resolve_chat_handler(llama: &dyn LlamaModel) -> anyhow::Result<ChatHandler> {
    let _ = llama;
    todo!()
}

/// `llama_chat_format.get_chat_completion_handler(name)`: the built-in handler
/// rendering the model's own (Jinja) chat template.
pub fn get_chat_completion_handler(chat_format: Option<&str>) -> anyhow::Result<ChatHandler> {
    let _ = chat_format;
    todo!()
}

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
    let _ = name;
    todo!()
}

/// Construct vision chat handler `name` over the CLIP/mmproj at
/// `clip_model_path`.  `use_gpu: None` uses the handler's default.
pub fn make_vision_chat_handler(name: &str, clip_model_path: &str, use_gpu: Option<bool>) -> anyhow::Result<ChatHandler> {
    let _ = (name, clip_model_path, use_gpu);
    todo!()
}

/// Force every later projector (CLIP/mmproj) context onto CPU — the port of
/// the `mtmd_init_from_file` patch.  Idempotent; returns `true` the first time.
pub fn force_clip_on_cpu() -> bool {
    todo!()
}

/// `llama_cpp.llama_supports_gpu_offload()`.
pub fn llama_supports_gpu_offload() -> bool {
    todo!()
}
