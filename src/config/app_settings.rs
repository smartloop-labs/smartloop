//! Application settings configuration.
//!
//! Port of `smartloop/config/app_settings.py` (pydantic-settings
//! `AppSettings`).  Each field is read, case-insensitively, from the process
//! environment or a `.env` file in the working directory (environment wins),
//! under the same alias the Python model declared; unknown keys are ignored.
//!
//! The `.env` reader follows python-dotenv's grammar (as pydantic-settings
//! uses it): `export` prefixes, single/double-quoted values with their escape
//! rules, unquoted values with trailing `# comments` removed, and POSIX
//! `${NAME}` / `${NAME:-default}` interpolation.

use std::collections::HashMap;
use std::path::Path;

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::auth::credential_store;
use crate::constants::{canonical_model_name, BOOTSTRAP_MODEL};
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
    /// `SLP_LLM_MAIN_N_CTX_CAP` — max context for the main (routing) model;
    /// 0 = no cap.  Defaults to 4096: the router only makes short routing
    /// decisions, and an auto-sized window would claim memory the capability
    /// models need.
    pub llm_main_n_ctx_cap: i64,
    /// `SLP_LLM_LARGE_N_CTX_CAP` — max context for the large agent model and
    /// the capability models; 0 = no cap.  Defaults to 16384: each of these
    /// auto-sizes from the memory free when it loads, as if it were alone, so
    /// two loaded back to back each claimed a 65536 window (~6GB of KV apiece)
    /// and a document fan-out over a large project got the process killed.
    /// Fan-out extraction needs at most ~5k prompt tokens plus 2048 output.
    pub llm_large_n_ctx_cap: i64,
    /// `SLP_LLM_LARGE_FIT_RESERVE_GB` — VRAM kept free when fit-checking the large model.
    pub llm_large_fit_reserve_gb: f64,
    /// `SLP_LLM_BATCH_SLOTS` — most prompts a batched generation runs at once;
    /// 0 = size automatically (free memory and a throughput benchmark), 1 =
    /// never batch.
    pub llm_batch_slots: i64,
    /// `SLP_LLM_BATCH_RESERVE_GB` — memory left free when sizing batch slots,
    /// for macOS, other apps and allocation spikes.
    pub llm_batch_reserve_gb: f64,
    /// `SLP_MODEL_KEEP_ALIVE` — how long capability models stay loaded once
    /// nothing is running, in seconds (`"30s"`, `"5m"` and `"1h"` are accepted
    /// too).  `0` unloads them as soon as a response completes; a negative
    /// value keeps them loaded until they are deselected or evicted for room.
    pub model_keep_alive: i64,
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
    /// `SLP_CLOUD_FALLBACK` — run a model in the cloud when its local copy
    /// can't (not downloaded, doesn't fit, fails to load) and the catalog
    /// also serves it there.  Off by default: cloud models are paid, so a
    /// model runs in the cloud only when the user picked its cloud variant.
    pub cloud_fallback: bool,
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
    /// `SLP_ORCHESTRATE_TIMEOUT_S` — 0 (the default) disables.
    pub orchestrate_timeout_s: f64,
    /// `SLP_STREAM_STALL_TIMEOUT_S` — 0 (the default) disables.
    pub stream_stall_timeout_s: f64,
    /// `SLP_AGENT_IDLE_TTL_S` — 0 = never stop idle agents.
    pub agent_idle_ttl_s: f64,
    /// `SLP_AGENT_MEMORY_LIMIT_MB` — 0 = no ceiling.
    pub agent_memory_limit_mb: i64,
}

/// A settings value failed validation (pydantic's `ValidationError`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("1 validation error for {model}\n{field}\n  {message} [input_value={input:?}]")]
pub struct SettingsValidationError {
    /// The settings class name.
    pub model: String,
    /// The env alias that failed.
    pub field: String,
    /// What was wrong with it.
    pub message: String,
    /// The raw input.
    pub input: String,
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
            llm_main_n_ctx_cap: 4096,
            llm_large_n_ctx_cap: 16384,
            llm_large_fit_reserve_gb: 1.0,
            llm_batch_slots: 0,
            llm_batch_reserve_gb: 3.0,
            model_keep_alive: 300,
            doc_extract_max_chars: 0,
            embedding_device: "auto".to_string(),
            embedding_gguf_repo: "gpustack/bge-m3-GGUF".to_string(),
            embedding_gguf_file: "bge-m3-Q4_K_M.gguf".to_string(),
            embedding_gguf_base_url: "https://dl.smartloop.ai/embeddings".to_string(),
            vision_enabled: true,
            cloud_fallback: false,
            vision_ocr_min_chars_per_page: 100,
            vision_n_ctx: 8192,
            vision_clip_device: "auto".to_string(),
            thinking_enabled: None,
            decision_model: None,
            orchestrate_timeout_s: 0.0,
            stream_stall_timeout_s: 0.0,
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
        match Self::try_new() {
            Ok(settings) => settings,
            Err(e) => panic!("{e}"),
        }
    }

    /// Fallible form of [`AppSettings::new`].
    pub fn try_new() -> anyhow::Result<Self> {
        let source = SettingsSource::load(".env");
        let mut s = Self::from_source(&source)?;
        s.resolve_defaults();
        Ok(s)
    }

    /// Build settings from an explicit source, without post-validation.
    pub fn from_source(source: &SettingsSource) -> Result<Self, SettingsValidationError> {
        let d = Self::defaults();
        let r = SourceReader { source, model: "AppSettings" };
        Ok(AppSettings {
            model_name: r.string("slp_base_model_name", d.model_name)?,
            project_id: r.opt_string("slp_project_id", d.project_id)?,
            load_quantized: r.boolean("slp_load_quantized", d.load_quantized)?,
            sentence_transformer: r.string("slp_sentence_transformer", d.sentence_transformer)?,
            hf_token: r.string("hf_token", d.hf_token)?,
            chat_history_token_size: r.int("slp_chat_history_token_size", d.chat_history_token_size)?,
            section_token_size: r.int("slp_section_token_size", d.section_token_size)?,
            log_level: r.string("log_level", d.log_level)?,
            slp_api_base: r.string("slp_api_base", d.slp_api_base)?,
            developer_token: r.opt_string("slp_developer_token", d.developer_token)?,
            release_channels_raw: r.string("slp_release_channels", d.release_channels_raw)?,
            version: r.opt_string("version", d.version)?,
            slp_home: r.opt_string("slp_home", d.slp_home)?,
            api_host: r.string("slp_api_host", d.api_host)?,
            api_port: r.port("slp_api_port", d.api_port)?,
            preload_model: r.boolean("slp_api_preload_model", d.preload_model)?,
            default_model: r.string("slp_api_base_model_name", d.default_model)?,
            debug_mode: r.boolean("slp_api_debug_mode", d.debug_mode)?,
            cors_origins: r.string("slp_api_cors_origins", d.cors_origins)?,
            n_parallel: r.int("slp_n_parallel", d.n_parallel)?,
            llm_n_batch: r.int("slp_llm_n_batch", d.llm_n_batch)?,
            llm_n_threads: r.int("slp_llm_n_threads", d.llm_n_threads)?,
            llm_main_n_ctx_cap: r.int("slp_llm_main_n_ctx_cap", d.llm_main_n_ctx_cap)?,
            llm_large_n_ctx_cap: r.int("slp_llm_large_n_ctx_cap", d.llm_large_n_ctx_cap)?,
            llm_large_fit_reserve_gb: r.float("slp_llm_large_fit_reserve_gb", d.llm_large_fit_reserve_gb)?,
            llm_batch_slots: r.int("slp_llm_batch_slots", d.llm_batch_slots)?,
            llm_batch_reserve_gb: r.float("slp_llm_batch_reserve_gb", d.llm_batch_reserve_gb)?,
            model_keep_alive: r.duration_secs("slp_model_keep_alive", d.model_keep_alive)?,
            doc_extract_max_chars: r.int("slp_doc_extract_max_chars", d.doc_extract_max_chars)?,
            embedding_device: r.string("slp_embedding_device", d.embedding_device)?,
            embedding_gguf_repo: r.string("slp_embedding_gguf_repo", d.embedding_gguf_repo)?,
            embedding_gguf_file: r.string("slp_embedding_gguf_file", d.embedding_gguf_file)?,
            embedding_gguf_base_url: r.string("slp_embedding_gguf_base_url", d.embedding_gguf_base_url)?,
            vision_enabled: r.boolean("slp_vision_enabled", d.vision_enabled)?,
            cloud_fallback: r.boolean("slp_cloud_fallback", d.cloud_fallback)?,
            vision_ocr_min_chars_per_page: r.int("slp_vision_ocr_min_chars_per_page", d.vision_ocr_min_chars_per_page)?,
            vision_n_ctx: r.int("slp_vision_n_ctx", d.vision_n_ctx)?,
            vision_clip_device: r.string("slp_vision_clip_device", d.vision_clip_device)?,
            thinking_enabled: r.opt_boolean("slp_thinking_enabled", d.thinking_enabled)?,
            decision_model: r.opt_string("slp_decision_model", d.decision_model)?,
            orchestrate_timeout_s: r.float("slp_orchestrate_timeout_s", d.orchestrate_timeout_s)?,
            stream_stall_timeout_s: r.float("slp_stream_stall_timeout_s", d.stream_stall_timeout_s)?,
            agent_idle_ttl_s: r.float("slp_agent_idle_ttl_s", d.agent_idle_ttl_s)?,
            agent_memory_limit_mb: r.int("slp_agent_memory_limit_mb", d.agent_memory_limit_mb)?,
        })
    }

    /// Resolve defaults that depend on other fields: fill `developer_token`
    /// from the credential store, canonicalize `model_name`/`default_model`.
    fn resolve_defaults(&mut self) {
        if self.developer_token.is_none() {
            self.developer_token = credential_store::get_active_token();
        }
        // An existing .env may still name the orchestrator model by its old
        // id; normalize here so the rest of the framework only ever sees the
        // canonical name.
        self.model_name = canonical_model_name(&self.model_name);
        self.default_model = canonical_model_name(&self.default_model);
    }

    /// Supported release channels; first entry is the default channel.
    /// Falls back to `["stable"]`.
    pub fn release_channels(&self) -> Vec<String> {
        let channels: Vec<String> = self
            .release_channels_raw
            .split(',')
            .map(py_strip)
            .filter(|c| !c.is_empty())
            .map(str::to_string)
            .collect();
        if channels.is_empty() {
            vec!["stable".to_string()]
        } else {
            channels
        }
    }

    /// The SLP home directory: `SLP_HOME` (tilde-expanded) when set,
    /// otherwise `<real home>/.smartloop`.
    ///
    /// Uses the SLP_HOME env var when set (e.g. by a systemd service).
    pub fn home_dir(&self) -> String {
        match &self.slp_home {
            Some(home) if !home.is_empty() => expand_home(home),
            _ => {
                let home = real_home();
                if home.is_empty() || home.ends_with('/') {
                    format!("{home}.smartloop")
                } else {
                    format!("{home}/.smartloop")
                }
            }
        }
    }
}

impl Default for AppSettings {
    /// Same as [`AppSettings::new`] (pydantic's `AppSettings()`).
    fn default() -> Self {
        AppSettings::new()
    }
}

/// Python `str.strip()` (whitespace, including the ASCII separators
/// `\x1c`-`\x1f` Python treats as whitespace).
fn py_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c))
}

// ---------------------------------------------------------------------------
// Settings sources (environment + .env)
// ---------------------------------------------------------------------------

/// The merged, case-insensitive view pydantic-settings validates against:
/// `.env` values overlaid by the process environment (environment wins).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SettingsSource {
    values: HashMap<String, String>,
}

impl SettingsSource {
    /// Read `env_file` (relative to the working directory; a missing file is
    /// no error) and the process environment.
    pub fn load(env_file: impl AsRef<Path>) -> Self {
        let mut values = HashMap::new();
        for (key, value) in read_dotenv(env_file.as_ref()) {
            if let Some(value) = value {
                values.insert(key.to_lowercase(), value);
            }
        }
        for (key, value) in std::env::vars_os() {
            if let (Some(key), Some(value)) = (key.to_str(), value.to_str()) {
                values.insert(key.to_lowercase(), value.to_string());
            }
        }
        SettingsSource { values }
    }

    /// A source built from explicit `(key, value)` pairs (keys are matched
    /// case-insensitively).
    pub fn from_pairs<K: AsRef<str>, V: Into<String>>(pairs: impl IntoIterator<Item = (K, V)>) -> Self {
        SettingsSource {
            values: pairs.into_iter().map(|(k, v)| (k.as_ref().to_lowercase(), v.into())).collect(),
        }
    }

    /// The raw value for `alias` (case-insensitive), if provided.
    pub fn get(&self, alias: &str) -> Option<&str> {
        self.values.get(&alias.to_lowercase()).map(String::as_str)
    }
}

/// Typed field access over a [`SettingsSource`], producing pydantic-style
/// validation errors.
struct SourceReader<'a> {
    source: &'a SettingsSource,
    model: &'static str,
}

impl SourceReader<'_> {
    fn error(&self, field: &str, message: &str, input: &str) -> SettingsValidationError {
        SettingsValidationError {
            model: self.model.to_string(),
            field: field.to_string(),
            message: message.to_string(),
            input: input.to_string(),
        }
    }

    fn string(&self, alias: &str, default: String) -> Result<String, SettingsValidationError> {
        Ok(self.source.get(alias).map(str::to_string).unwrap_or(default))
    }

    fn opt_string(&self, alias: &str, default: Option<String>) -> Result<Option<String>, SettingsValidationError> {
        Ok(match self.source.get(alias) {
            Some(v) => Some(v.to_string()),
            None => default,
        })
    }

    fn boolean(&self, alias: &str, default: bool) -> Result<bool, SettingsValidationError> {
        match self.source.get(alias) {
            None => Ok(default),
            Some(raw) => parse_bool(raw)
                .ok_or_else(|| self.error(alias, "Input should be a valid boolean, unable to interpret input", raw)),
        }
    }

    fn opt_boolean(&self, alias: &str, default: Option<bool>) -> Result<Option<bool>, SettingsValidationError> {
        match self.source.get(alias) {
            None => Ok(default),
            Some(raw) => parse_bool(raw)
                .map(Some)
                .ok_or_else(|| self.error(alias, "Input should be a valid boolean, unable to interpret input", raw)),
        }
    }

    fn int(&self, alias: &str, default: i64) -> Result<i64, SettingsValidationError> {
        match self.source.get(alias) {
            None => Ok(default),
            Some(raw) => parse_int(raw)
                .ok_or_else(|| self.error(alias, "Input should be a valid integer, unable to parse string as an integer", raw)),
        }
    }

    fn duration_secs(&self, alias: &str, default: i64) -> Result<i64, SettingsValidationError> {
        match self.source.get(alias) {
            None => Ok(default),
            Some(raw) => parse_duration_secs(raw).ok_or_else(|| {
                self.error(alias, "Input should be a duration in seconds, or a number with s, m or h", raw)
            }),
        }
    }

    fn port(&self, alias: &str, default: u16) -> Result<u16, SettingsValidationError> {
        match self.source.get(alias) {
            None => Ok(default),
            Some(raw) => {
                let value = parse_int(raw).ok_or_else(|| {
                    self.error(alias, "Input should be a valid integer, unable to parse string as an integer", raw)
                })?;
                u16::try_from(value)
                    .map_err(|_| self.error(alias, "Input should be a valid port number between 0 and 65535", raw))
            }
        }
    }

    fn float(&self, alias: &str, default: f64) -> Result<f64, SettingsValidationError> {
        match self.source.get(alias) {
            None => Ok(default),
            Some(raw) => parse_float(raw)
                .ok_or_else(|| self.error(alias, "Input should be a valid number, unable to parse string as a number", raw)),
        }
    }
}

/// Pydantic's lax string → bool coercion.
pub fn parse_bool(raw: &str) -> Option<bool> {
    match py_strip(raw).to_lowercase().as_str() {
        "1" | "on" | "t" | "true" | "y" | "yes" => Some(true),
        "0" | "off" | "f" | "false" | "n" | "no" => Some(false),
        _ => None,
    }
}

/// Pydantic's lax string → int coercion (surrounding whitespace allowed,
/// `_` digit separators allowed, an integral float like `"8.0"` accepted).
/// A duration in whole seconds: a plain integer, or one suffixed with `s`,
/// `m` or `h` (`"90"`, `"30s"`, `"5m"`, `"1h"`).  The sign is kept, so `"-1"`
/// still reads as "forever" wherever a negative value means that.
pub fn parse_duration_secs(raw: &str) -> Option<i64> {
    let s = py_strip(raw).to_ascii_lowercase();
    let (number, unit) = match s.chars().last()? {
        's' => (&s[..s.len() - 1], 1),
        'm' => (&s[..s.len() - 1], 60),
        'h' => (&s[..s.len() - 1], 3600),
        _ => (s.as_str(), 1),
    };
    parse_int(number)?.checked_mul(unit)
}

pub fn parse_int(raw: &str) -> Option<i64> {
    let s = py_strip(raw);
    if s.is_empty() {
        return None;
    }
    let cleaned = strip_digit_separators(s)?;
    if let Ok(v) = cleaned.parse::<i64>() {
        return Some(v);
    }
    let f: f64 = cleaned.parse().ok()?;
    if f.is_finite() && f.fract() == 0.0 && f.abs() < 9.2e18 {
        Some(f as i64)
    } else {
        None
    }
}

/// Pydantic's lax string → float coercion.
pub fn parse_float(raw: &str) -> Option<f64> {
    let s = py_strip(raw);
    if s.is_empty() {
        return None;
    }
    strip_digit_separators(s)?.parse::<f64>().ok()
}

/// Remove single `_` separators between digits (Python numeric literal
/// rules); `None` for a misplaced underscore.
fn strip_digit_separators(s: &str) -> Option<String> {
    if !s.contains('_') {
        return Some(s.to_string());
    }
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for (i, c) in chars.iter().enumerate() {
        if *c == '_' {
            let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let next_digit = chars.get(i + 1).is_some_and(|n| n.is_ascii_digit());
            if !(prev_digit && next_digit) {
                return None;
            }
        } else {
            out.push(*c);
        }
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// .env parsing (python-dotenv)
// ---------------------------------------------------------------------------

static POSIX_VARIABLE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\$\{(?P<name>[^\}:]*)(?::-(?P<default>[^\}]*))?\}").expect("valid regex"));

/// Read a `.env` file the way `dotenv_values(path)` does: parsed bindings in
/// file order with variables interpolated (earlier `.env` values override the
/// process environment during interpolation).  A key without `=` maps to
/// `None`.  A missing or unreadable file yields no values.
pub fn read_dotenv(path: &Path) -> Vec<(String, Option<String>)> {
    let Ok(bytes) = std::fs::read(path) else {
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&bytes);
    let raw = parse_dotenv(&text);
    resolve_variables(raw)
}

/// Interpolate `${NAME}` / `${NAME:-default}` in every value.
fn resolve_variables(values: Vec<(String, Option<String>)>) -> Vec<(String, Option<String>)> {
    let mut new_values: Vec<(String, Option<String>)> = Vec::new();
    let mut resolved: HashMap<String, Option<String>> = HashMap::new();
    for (name, value) in values {
        let result = value.map(|value| {
            let lookup = |var: &str| -> Option<String> {
                if let Some(v) = resolved.get(var) {
                    return v.clone();
                }
                std::env::var(var).ok()
            };
            let mut out = String::new();
            let mut cursor = 0;
            for caps in POSIX_VARIABLE.captures_iter(&value) {
                let m = caps.get(0).expect("whole match");
                out.push_str(&value[cursor..m.start()]);
                let var = caps.name("name").map(|n| n.as_str()).unwrap_or("");
                let default = caps.name("default").map(|d| d.as_str().to_string()).unwrap_or_default();
                let has_key = resolved.contains_key(var) || std::env::var_os(var).is_some();
                let replacement = if has_key { lookup(var).unwrap_or_default() } else { default };
                out.push_str(&replacement);
                cursor = m.end();
            }
            out.push_str(&value[cursor..]);
            out
        });
        resolved.insert(name.clone(), result.clone());
        new_values.push((name, result));
    }
    new_values
}

/// Python `\s` for the dotenv grammar.
fn is_ws(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// Whitespace other than line breaks (`[^\S\r\n]`).
fn is_inline_ws(c: char) -> bool {
    is_ws(c) && c != '\r' && c != '\n'
}

/// Character cursor over the `.env` text (python-dotenv's `Reader`).
struct Reader {
    chars: Vec<char>,
    pos: usize,
}

impl Reader {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn has_next(&self) -> bool {
        self.pos < self.chars.len()
    }

    fn skip_while(&mut self, pred: impl Fn(char) -> bool) {
        while let Some(c) = self.peek() {
            if !pred(c) {
                break;
            }
            self.pos += 1;
        }
    }

    /// `(?:export[^\S\r\n]+)?`
    fn skip_export(&mut self) {
        let word: Vec<char> = "export".chars().collect();
        if self.chars.len() >= self.pos + word.len() && self.chars[self.pos..self.pos + word.len()] == word[..] {
            let after = self.pos + word.len();
            if self.chars.get(after).is_some_and(|c| is_inline_ws(*c)) {
                self.pos = after;
                self.skip_while(is_inline_ws);
            }
        }
    }

    /// `[^\r\n]*(?:\r|\n|\r\n)?` — recover from a malformed line.
    fn skip_rest_of_line(&mut self) {
        self.skip_while(|c| c != '\r' && c != '\n');
        if self.peek() == Some('\r') {
            self.pos += 1;
        } else if self.peek() == Some('\n') {
            self.pos += 1;
        }
    }

    /// `[^\S\r\n]*(?:\r\n|\n|\r|$)`; `false` when something else follows.
    fn end_of_line(&mut self) -> bool {
        self.skip_while(is_inline_ws);
        match self.peek() {
            None => true,
            Some('\r') => {
                self.pos += 1;
                if self.peek() == Some('\n') {
                    self.pos += 1;
                }
                true
            }
            Some('\n') => {
                self.pos += 1;
                true
            }
            Some(_) => false,
        }
    }

    /// Read a quoted body up to the closing `quote`, allowing `\<quote>`.
    fn quoted(&mut self, quote: char) -> Option<String> {
        if self.peek() != Some(quote) {
            return None;
        }
        let start = self.pos;
        self.pos += 1;
        let mut body = String::new();
        loop {
            match self.peek() {
                None => {
                    self.pos = start;
                    return None;
                }
                Some('\\') if self.chars.get(self.pos + 1) == Some(&quote) => {
                    body.push('\\');
                    body.push(quote);
                    self.pos += 2;
                }
                Some(c) if c == quote => {
                    self.pos += 1;
                    return Some(body);
                }
                Some(c) => {
                    body.push(c);
                    self.pos += 1;
                }
            }
        }
    }
}

/// Decode `\\` and `\'` in a single-quoted value.
fn decode_single_quoted(value: &str) -> String {
    let mut out = String::new();
    let mut chars = value.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(&n) = chars.peek() {
                if n == '\\' || n == '\'' {
                    out.push(n);
                    chars.next();
                    continue;
                }
            }
        }
        out.push(c);
    }
    out
}

/// Decode `\\ \' \" \a \b \f \n \r \t \v` in a double-quoted value.
fn decode_double_quoted(value: &str) -> String {
    let mut out = String::new();
    let mut chars = value.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(&n) = chars.peek() {
                let decoded = match n {
                    '\\' => Some('\\'),
                    '\'' => Some('\''),
                    '"' => Some('"'),
                    'a' => Some('\x07'),
                    'b' => Some('\x08'),
                    'f' => Some('\x0c'),
                    'n' => Some('\n'),
                    'r' => Some('\r'),
                    't' => Some('\t'),
                    'v' => Some('\x0b'),
                    _ => None,
                };
                if let Some(d) = decoded {
                    out.push(d);
                    chars.next();
                    continue;
                }
            }
        }
        out.push(c);
    }
    out
}

static TRAILING_COMMENT: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+#.*").expect("valid regex"));

/// Parse `.env` text into `(key, value)` bindings (python-dotenv grammar).
fn parse_dotenv(text: &str) -> Vec<(String, Option<String>)> {
    let mut reader = Reader { chars: text.chars().collect(), pos: 0 };
    let mut out = Vec::new();
    loop {
        reader.skip_while(is_ws);
        if !reader.has_next() {
            break;
        }
        let mark = reader.pos;
        match parse_binding(&mut reader) {
            Some(Some((key, value))) => out.push((key, value)),
            Some(None) => {}
            None => {
                reader.pos = mark;
                reader.skip_rest_of_line();
            }
        }
    }
    out
}

/// One binding: `Some(Some(..))` for a key, `Some(None)` for a comment line,
/// `None` for a malformed line.
#[allow(clippy::option_option)]
fn parse_binding(reader: &mut Reader) -> Option<Option<(String, Option<String>)>> {
    reader.skip_export();
    // parse_key
    let key = match reader.peek() {
        Some('#') => None,
        Some('\'') => {
            // '([^']+)'
            let start = reader.pos + 1;
            let end = reader.chars[start..].iter().position(|c| *c == '\'')? + start;
            if end == start {
                return None;
            }
            reader.pos = end + 1;
            Some(reader.chars[start..end].iter().collect::<String>())
        }
        _ => {
            let start = reader.pos;
            reader.skip_while(|c| c != '=' && c != '#' && !is_ws(c));
            if reader.pos == start {
                return None;
            }
            Some(reader.chars[start..reader.pos].iter().collect::<String>())
        }
    };
    reader.skip_while(is_inline_ws);
    let value = if reader.peek() == Some('=') {
        reader.pos += 1;
        reader.skip_while(is_inline_ws);
        Some(match reader.peek() {
            Some('\'') => decode_single_quoted(&reader.quoted('\'')?),
            Some('"') => decode_double_quoted(&reader.quoted('"')?),
            None | Some('\n') | Some('\r') => String::new(),
            Some(_) => {
                let start = reader.pos;
                reader.skip_while(|c| c != '\r' && c != '\n');
                let part: String = reader.chars[start..reader.pos].iter().collect();
                let stripped = TRAILING_COMMENT.replace_all(&part, "");
                stripped.trim_end_matches(is_ws).to_string()
            }
        })
    } else {
        None
    };
    // (?:[^\S\r\n]*#[^\r\n]*)?
    let save = reader.pos;
    reader.skip_while(is_inline_ws);
    if reader.peek() == Some('#') {
        reader.skip_while(|c| c != '\r' && c != '\n');
    } else {
        reader.pos = save;
    }
    if !reader.end_of_line() {
        return None;
    }
    Some(key.map(|k| (k, value)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dotenv_grammar() {
        let text = "# comment\nexport A=1\nB = 'x # y'\nC=\"line\\nbreak\" # trailing\nD=plain value # note\nE\nF=${A}-${MISSING:-dflt}\n";
        let parsed = resolve_variables(parse_dotenv(text));
        assert_eq!(
            parsed,
            vec![
                ("A".to_string(), Some("1".to_string())),
                ("B".to_string(), Some("x # y".to_string())),
                ("C".to_string(), Some("line\nbreak".to_string())),
                ("D".to_string(), Some("plain value".to_string())),
                ("E".to_string(), None),
                ("F".to_string(), Some("1-dflt".to_string())),
            ]
        );
    }

    #[test]
    fn lax_coercions() {
        assert_eq!(parse_bool(" YES "), Some(true));
        assert_eq!(parse_bool("0"), Some(false));
        assert_eq!(parse_bool("maybe"), None);
        assert_eq!(parse_int(" 42 "), Some(42));
        assert_eq!(parse_int("1_000"), Some(1000));
        assert_eq!(parse_int("abc"), None);
        assert_eq!(parse_float("2.5"), Some(2.5));
    }

    #[test]
    fn source_overrides_and_canonicalizes() {
        let source = SettingsSource::from_pairs([
            ("SLP_BASE_MODEL_NAME", "slp-orchestra-mini"),
            ("slp_api_port", "0"),
            ("SLP_DEVELOPER_TOKEN", "sk"),
            ("SLP_RELEASE_CHANNELS", " beta , ,stable"),
        ]);
        let mut s = AppSettings::from_source(&source).unwrap();
        s.resolve_defaults();
        assert_eq!(s.model_name, BOOTSTRAP_MODEL);
        assert_eq!(s.api_port, 0);
        assert_eq!(s.developer_token.as_deref(), Some("sk"));
        assert_eq!(s.release_channels(), vec!["beta", "stable"]);
        let bad = SettingsSource::from_pairs([("SLP_API_PORT", "nope")]);
        assert!(AppSettings::from_source(&bad).is_err());
    }
}
