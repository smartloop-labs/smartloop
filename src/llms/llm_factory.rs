//! Factory to create LLM wrapper instances.
//!
//! Port of `smartloop/llms/llm_factory.py`.
//!
//! The crash probe that Python ran as `sys.executable -c <script>` re-executes
//! the `slp` binary (`std::env::current_exe()`, overridable with `SLP_EXE`) with
//! the hidden subcommand [`PROBE_SUBCOMMAND`] and the JSON-encoded
//! [`LlamaParams`]; `main.rs` dispatches it to [`run_probe`].

use once_cell::sync::Lazy;
use parking_lot::Mutex;

#[allow(unused_imports)]
use crate::config::app_settings::AppSettings;
#[allow(unused_imports)]
use crate::llms::llama::{ChatHandler, Llama, LlamaModel, LlamaParams};
use crate::llms::llama_cpp::LlamaCpp;
#[allow(unused_imports)]
use crate::utils::device_utils::get_device_config;

/// Hidden `slp` subcommand that test-loads a model from a JSON config.
pub const PROBE_SUBCOMMAND: &str = "__probe-model";

/// Seconds before a probe load is treated as a crash.
pub const PROBE_TIMEOUT_SECS: u64 = 120;

/// Stderr tail of the last failed probe (`LlmFactory._last_probe_stderr`).
static LAST_PROBE_STDERR: Lazy<Mutex<String>> = Lazy::new(|| Mutex::new(String::new()));

/// True when the linked llama.cpp was built with the CUDA backend.
#[allow(dead_code)]
pub fn llama_cpp_has_cuda_backend() -> bool {
    todo!()
}

/// Keyword options of `LlmFactory.get_llama_cpp_instance`.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaInstanceOptions {
    /// Test-load the model in a throwaway subprocess first (default true).
    pub probe: bool,
    /// Upper bound for the auto-sized context window; `None`/0 = no cap.
    pub max_n_ctx: Option<usize>,
    /// CLIP/mmproj GGUF; loads the model as a multimodal instance.
    pub mmproj_path: Option<String>,
    /// Keep the CLIP/mmproj encoder on CPU.
    pub clip_on_cpu: bool,
    /// Vision chat-handler name declared by the model class (authoritative).
    pub vision_handler: Option<String>,
    /// Turn the `<think>` block on/off; `None` keeps the template default.
    pub thinking: Option<bool>,
    /// The name the model was selected under.
    pub model_name: Option<String>,
}

impl Default for LlamaInstanceOptions {
    fn default() -> Self {
        Self {
            probe: true,
            max_n_ctx: None,
            mmproj_path: None,
            clip_on_cpu: false,
            vision_handler: None,
            thinking: None,
            model_name: None,
        }
    }
}

/// Factory to create instances of LLM classes.
pub struct LlmFactory;

impl LlmFactory {
    /// Map a model's GGUF `general.architecture` to the vision chat handler
    /// that knows its image-token / prompt format.
    pub const VISION_HANDLER_BY_ARCH: &'static [(&'static [&'static str], &'static str)] = &[
        (&["qwen2vl", "qwen2.5vl", "qwen2_5vl", "qwen3vl", "qwenvl"], "Qwen25VLChatHandler"),
        (&["minicpmv", "minicpm-v", "minicpm"], "MiniCPMv26ChatHandler"),
        (&["nanollava"], "NanoLlavaChatHandler"),
        (&["moondream"], "MoondreamChatHandler"),
        (&["llava"], "Llava16ChatHandler"),
    ];

    /// Stderr tail of the last failed probe.
    pub fn last_probe_stderr() -> String {
        LAST_PROBE_STDERR.lock().clone()
    }

    /// Try loading the model in a subprocess to detect crashes.
    ///
    /// `Some(true)` if the probe succeeded, `Some(false)` if it crashed (signal,
    /// timeout, spawn failure), `None` if it failed with a normal error exit.
    pub fn probe_model_load(path: &str, config: &LlamaParams) -> Option<bool> {
        let _ = (path, config, &LAST_PROBE_STDERR);
        todo!()
    }

    /// Body of the probe subprocess: load the model from `config_json` and
    /// drop it.  Returns the process exit code.
    pub fn run_probe(config_json: &str) -> i32 {
        let _ = config_json;
        todo!()
    }

    /// A model GGUF's `general.architecture` (e.g. `qwen2vl`).
    pub fn read_gguf_arch(path: &str) -> Option<String> {
        let _ = path;
        todo!()
    }

    /// Name of the supported vision handler for this model, or `None`.
    pub fn vision_handler_class_name(model_path: &str) -> Option<String> {
        let _ = model_path;
        todo!()
    }

    /// True when this build can run this model's vision input.
    pub fn vision_handler_supported(model_path: &str, handler_name: Option<&str>) -> bool {
        let _ = (model_path, handler_name);
        todo!()
    }

    /// Make the multimodal (CLIP/mmproj) encoder load on CPU.  Idempotent.
    pub fn force_clip_on_cpu() {
        todo!()
    }

    /// Build the multimodal chat handler for this model, or `None` (load
    /// text-only) when no supported handler exists or it fails to initialize.
    pub fn make_vision_chat_handler(
        mmproj_path: &str,
        model_path: &str,
        clip_on_cpu: bool,
        handler_name: Option<&str>,
    ) -> Option<ChatHandler> {
        let _ = (mmproj_path, model_path, clip_on_cpu, handler_name);
        todo!()
    }

    /// True when `model_name` declares `ModelCapability::Thinking`.
    pub fn supports_thinking(model_name: Option<&str>) -> bool {
        let _ = model_name;
        todo!()
    }

    /// `(n_batch, n_threads)` for this load: `(1024, 4)` offloaded,
    /// `(512, cpu_count)` on CPU; overridable with `SLP_LLM_N_BATCH` /
    /// `SLP_LLM_N_THREADS`.
    pub fn batch_and_threads(n_gpu_layers: i32) -> (usize, usize) {
        let _ = n_gpu_layers;
        todo!()
    }

    /// True when the loaded GGUF's own chat template reads `enable_thinking`.
    pub fn template_reads_thinking(llm: &dyn LlamaModel) -> bool {
        let _ = llm;
        todo!()
    }

    /// Whether `model_name` should reason, or `None` to leave it alone:
    /// `override_`, then `slp_thinking_enabled`, then `thinking_default()`.
    pub fn thinking_for(model_name: Option<&str>, override_: Option<bool>) -> Option<bool> {
        let _ = (model_name, override_);
        todo!()
    }

    /// Render every chat completion with `enable_thinking` read from the
    /// model's `_slp_thinking` flag (initially `enabled`).  Best-effort: a
    /// failure to resolve the handler is logged and leaves the model as is.
    pub fn bind_thinking(llm: &dyn LlamaModel, enabled: bool) {
        let _ = (llm, enabled);
        todo!()
    }

    /// Load a GGUF and return the configured [`LlamaCpp`] wrapper.
    pub fn get_llama_cpp_instance(
        path: &str,
        project_id: &str,
        lora_path: Option<&str>,
        options: LlamaInstanceOptions,
    ) -> anyhow::Result<LlamaCpp> {
        let _ = (path, project_id, lora_path, options);
        todo!()
    }
}
