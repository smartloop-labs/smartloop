//! Application settings configuration.
//!
//! Port of `smartloop/config/app_settings.py` (pydantic-settings
//! `AppSettings`).  Each field is read, case-insensitively, from the process
//! environment or a `.env` file in the working directory (environment wins),
//! under the same alias the Python model declared; unknown keys are ignored.

use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use crate::auth::credential_store;
#[allow(unused_imports)]
use crate::constants::{canonical_model_name, BOOTSTRAP_MODEL};
#[allow(unused_imports)]
use crate::paths::{expand_home, real_home};

/// Application settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AppSettings {
    // LLM Configuration
    /// `SLP_BASE_MODEL_NAME`
    pub model_name: String,
    /// `SLP_PROJECT_ID`
    pub project_id: Option<String>,
    /// `SLP_LOAD_QUANTIZED`
    pub load_quantized: bool,
    /// `SLP_SENTENCE_TRANSFORMER`
    pub sentence_transformer: String,
    /// `HF_TOKEN`
    pub hf_token: String,
    /// `SLP_CHAT_HISTORY_TOKEN_SIZE`
    pub chat_history_token_size: i64,
    /// `SLP_SECTION_TOKEN_SIZE`
    pub section_token_size: i64,
    /// `LOG_LEVEL`
    pub log_level: String,
    /// `SLP_API_BASE` — Smartloop Platform API.
    pub slp_api_base: String,
    /// `SLP_DEVELOPER_TOKEN`; resolved from the credential store when unset.
    pub developer_token: Option<String>,
    /// `SLP_RELEASE_CHANNELS` (comma-separated); first entry is the default.
    pub release_channels_raw: String,
    /// Set from the crate version after init (`version` env/alias).
    pub version: Option<String>,
    /// `SLP_HOME` — workspace root; defaults to `~/.smartloop`.
    pub slp_home: Option<String>,

    // API Server Configuration
    /// `SLP_API_HOST`
    pub api_host: String,
    /// `SLP_API_PORT`
    pub api_port: u16,
    /// `SLP_API_PRELOAD_MODEL`
    pub preload_model: bool,
    /// `SLP_API_BASE_MODEL_NAME`
    pub default_model: String,
    /// `SLP_API_DEBUG_MODE`
    pub debug_mode: bool,
    /// `SLP_API_CORS_ORIGINS`
    pub cors_origins: String,
    /// `SLP_N_PARALLEL` — parallel inference slots (llama.cpp n_parallel).
    pub n_parallel: i64,
    /// `SLP_LLM_N_BATCH` — llama.cpp n_batch; 0 = choose from the device.
    pub llm_n_batch: i64,
    /// `SLP_LLM_N_THREADS` — llama.cpp n_threads; 0 = choose from the device.
    pub llm_n_threads: i64,
    /// `SLP_LLM_MAIN_N_CTX_CAP` — max context for the main model; 0 = no cap.
    pub llm_main_n_ctx_cap: i64,
    /// `SLP_LLM_LARGE_N_CTX_CAP` — max context for the large agent model; 0 = no cap.
    pub llm_large_n_ctx_cap: i64,
    /// `SLP_LLM_LARGE_FIT_RESERVE_GB` — VRAM kept free when fit-checking the large model.
    pub llm_large_fit_reserve_gb: f64,
    /// `SLP_DOC_EXTRACT_MAX_CHARS` — hard cap on fan-out extraction chars; 0 = derive.
    pub doc_extract_max_chars: i64,
    /// `SLP_EMBEDDING_DEVICE` — `"auto"` (GPU offload) or `"cpu"`.
    pub embedding_device: String,
    /// `SLP_EMBEDDING_GGUF_REPO` — HF repo id of the GGUF embedding model.
    pub embedding_gguf_repo: String,
    /// `SLP_EMBEDDING_GGUF_FILE` — GGUF filename within the repo.
    pub embedding_gguf_file: String,
    /// `SLP_EMBEDDING_GGUF_BASE_URL` — download base; empty falls back to HF.
    pub embedding_gguf_base_url: String,
    /// `SLP_VISION_ENABLED`
    pub vision_enabled: bool,
    /// `SLP_VISION_OCR_MIN_CHARS_PER_PAGE`
    pub vision_ocr_min_chars_per_page: i64,
    /// `SLP_VISION_N_CTX`
    pub vision_n_ctx: i64,
    /// `SLP_VISION_CLIP_DEVICE` — `"auto"`, `"gpu"` or `"cpu"`.
    pub vision_clip_device: String,
    /// `SLP_THINKING_ENABLED` — override reasoning models' `<think>` block.
    pub thinking_enabled: Option<bool>,
    /// `SLP_DECISION_MODEL` — small model for decision nodes.
    pub decision_model: Option<String>,
    /// `SLP_ORCHESTRATE_TIMEOUT_S` — 0 disables.
    pub orchestrate_timeout_s: f64,
    /// `SLP_STREAM_STALL_TIMEOUT_S` — 0 disables.
    pub stream_stall_timeout_s: f64,
    /// `SLP_AGENT_IDLE_TTL_S` — 0 = never stop idle agents.
    pub agent_idle_ttl_s: f64,
    /// `SLP_AGENT_MEMORY_LIMIT_MB` — 0 = no ceiling.
    pub agent_memory_limit_mb: i64,
}

impl AppSettings {
    /// Every field at its declared default, with no env/.env reading and no
    /// post-validation (`developer_token` stays `None`).
    pub fn defaults() -> Self {
        AppSettings {
            model_name: BOOTSTRAP_MODEL.to_string(),
            project_id: Some("d4c8f5a2-7b3e-4f91-a6d2-8e9c1b4a7f3d".to_string()),
            load_quantized: false,
            sentence_transformer: "sentence-transformers/all-mpnet-base-v2".to_string(),
            hf_token: "token".to_string(),
            chat_history_token_size: 4096,
            section_token_size: 2048,
            log_level: "INFO".to_string(),
            slp_api_base: "https://api.smartloop.ai".to_string(),
            developer_token: None,
            release_channels_raw: "stable,beta".to_string(),
            version: None,
            slp_home: None,
            api_host: "0.0.0.0".to_string(),
            api_port: 38540,
            preload_model: true,
            default_model: BOOTSTRAP_MODEL.to_string(),
            debug_mode: false,
            cors_origins: "*".to_string(),
            n_parallel: 1,
            llm_n_batch: 0,
            llm_n_threads: 0,
            llm_main_n_ctx_cap: 0,
            llm_large_n_ctx_cap: 0,
            llm_large_fit_reserve_gb: 1.0,
            doc_extract_max_chars: 0,
            embedding_device: "auto".to_string(),
            embedding_gguf_repo: "gpustack/bge-m3-GGUF".to_string(),
            embedding_gguf_file: "bge-m3-Q4_K_M.gguf".to_string(),
            embedding_gguf_base_url: "https://dl.smartloop.ai/embeddings".to_string(),
            vision_enabled: true,
            vision_ocr_min_chars_per_page: 100,
            vision_n_ctx: 8192,
            vision_clip_device: "auto".to_string(),
            thinking_enabled: None,
            decision_model: None,
            orchestrate_timeout_s: 180.0,
            stream_stall_timeout_s: 300.0,
            agent_idle_ttl_s: 900.0,
            agent_memory_limit_mb: 0,
        }
    }

    /// Load settings from the environment and `./.env`, then resolve
    /// dependent defaults (`model_validator(mode="after")`).
    ///
    /// Panics on a value that fails to parse, as pydantic raises a
    /// `ValidationError`; use [`AppSettings::try_new`] to handle it.
    pub fn new() -> Self {
        todo!()
    }

    /// Fallible form of [`AppSettings::new`].
    pub fn try_new() -> anyhow::Result<Self> {
        todo!()
    }

    /// Resolve defaults that depend on other fields: fill `developer_token`
    /// from the credential store, canonicalize `model_name`/`default_model`.
    fn resolve_defaults(&mut self) {
        todo!()
    }

    /// Supported release channels; first entry is the default channel.
    /// Falls back to `["stable"]`.
    pub fn release_channels(&self) -> Vec<String> {
        todo!()
    }

    /// The SLP home directory: `SLP_HOME` (tilde-expanded) when set,
    /// otherwise `<real home>/.smartloop`.
    pub fn home_dir(&self) -> String {
        todo!()
    }
}

impl Default for AppSettings {
    /// Same as [`AppSettings::new`] (pydantic's `AppSettings()`).
    fn default() -> Self {
        AppSettings::new()
    }
}
