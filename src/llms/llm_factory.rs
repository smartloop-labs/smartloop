//! Factory to create LLM wrapper instances.
//!
//! Port of `smartloop/llms/llm_factory.py`.
//!
//! The crash probe that Python ran as `sys.executable -c <script>` re-executes
//! the `slp` binary (`std::env::current_exe()`, overridable with `SLP_EXE`) with
//! the hidden subcommand [`PROBE_SUBCOMMAND`] and the JSON-encoded
//! [`LlamaParams`]; `main.rs` dispatches it to [`LlmFactory::run_probe`].

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::Value;

use crate::config::app_settings::AppSettings;
use crate::config::supported_models::SUPPORTED_MODELS;
use crate::llms::llama::{self, ChatHandler, ChatHandlerRequest, Llama, LlamaModel, LlamaParams};
use crate::llms::llama_cpp::LlamaCpp;
use crate::types::ModelCapability;
use crate::utils::device_utils::get_device_config;

/// Hidden `slp` subcommand that test-loads a model from a JSON config.
pub const PROBE_SUBCOMMAND: &str = "__probe-model";

/// Seconds before a probe load is treated as a crash.
pub const PROBE_TIMEOUT_SECS: u64 = 120;

/// Logger name of the factory.
const LOG_TARGET: &str = "smartloop.llms.llm_factory";

/// Stderr tail of the last failed probe (`LlmFactory._last_probe_stderr`).
static LAST_PROBE_STDERR: Lazy<Mutex<String>> = Lazy::new(|| Mutex::new(String::new()));

/// Python's `str(bool)`, for log lines that interpolated a flag.
fn py_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

/// `os.cpu_count() or 4`.
pub fn cpu_count() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

/// True when the linked llama.cpp was built with the CUDA backend.
///
/// The host's other GPU stacks say nothing about llama.cpp's build — a
/// Vulkan-only llama.cpp must not get CUDA-specific load options — so the
/// answer comes from llama.cpp itself: the CUDA feature it was compiled with,
/// or a registered CUDA backend device.
pub fn llama_cpp_has_cuda_backend() -> bool {
    if cfg!(feature = "cuda") {
        return true;
    }
    let _ = llama::llama_backend();
    llama_cpp_2::list_llama_ggml_backend_devices()
        .iter()
        .any(|d| d.backend.eq_ignore_ascii_case("cuda"))
}

/// The binary the probe re-executes: `SLP_EXE`, else this executable.
fn probe_executable() -> std::io::Result<std::path::PathBuf> {
    match std::env::var_os("SLP_EXE") {
        Some(exe) if !exe.is_empty() => Ok(exe.into()),
        _ => std::env::current_exe(),
    }
}

/// The last `n` characters of `s`.
fn tail_chars(s: &str, n: usize) -> String {
    let count = s.chars().count();
    if count <= n {
        return s.to_string();
    }
    s.chars().skip(count - n).collect()
}

/// How a probe subprocess ended.
enum ProbeExit {
    /// Exit code (Python `returncode >= 0`).
    Code(i32),
    /// Killed by a signal (Python `returncode < 0`).
    Signal(i32),
}

/// Spawn `exe PROBE_SUBCOMMAND <config>` and wait up to the probe timeout.
/// `Ok(None)` on timeout (the child is killed).
fn run_probe_process(config_json: &str) -> std::io::Result<Option<(ProbeExit, String)>> {
    let exe = probe_executable()?;
    let mut child = Command::new(exe)
        .arg(PROBE_SUBCOMMAND)
        .arg(config_json)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    // Drain both pipes on their own threads so a chatty llama.cpp cannot fill
    // a pipe buffer and stall the child.
    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();
    let out_reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(s) = stdout.as_mut() {
            let _ = s.read_to_end(&mut buf);
        }
        buf
    });
    let err_reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(s) = stderr.as_mut() {
            let _ = s.read_to_end(&mut buf);
        }
        buf
    });

    let deadline = Instant::now() + Duration::from_secs(PROBE_TIMEOUT_SECS);
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break Some(status);
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let _ = out_reader.join();
    let stderr_bytes = err_reader.join().unwrap_or_default();

    let Some(status) = status else {
        return Ok(None);
    };
    let stderr_text = String::from_utf8_lossy(&stderr_bytes).trim().to_string();

    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return Ok(Some((ProbeExit::Signal(signal), stderr_text)));
        }
    }
    Ok(Some((ProbeExit::Code(status.code().unwrap_or(1)), stderr_text)))
}

/// Read one string-valued key from a GGUF file's metadata header without
/// loading the model (`gguf.GGUFReader(path).fields[key]`).
pub fn read_gguf_string_field(path: &str, wanted: &str) -> Option<String> {
    let file = File::open(path).ok()?;
    let mut reader = BufReader::new(file);

    fn read_u32(r: &mut impl Read) -> Option<u32> {
        let mut b = [0u8; 4];
        r.read_exact(&mut b).ok()?;
        Some(u32::from_le_bytes(b))
    }
    fn read_u64(r: &mut impl Read) -> Option<u64> {
        let mut b = [0u8; 8];
        r.read_exact(&mut b).ok()?;
        Some(u64::from_le_bytes(b))
    }
    fn read_string(r: &mut impl Read) -> Option<Vec<u8>> {
        let len = read_u64(r)?;
        // A metadata string longer than this is a corrupt header.
        if len > (1 << 30) {
            return None;
        }
        let mut buf = vec![0u8; len as usize];
        r.read_exact(&mut buf).ok()?;
        Some(buf)
    }
    /// Byte width of a fixed-size GGUF value type.
    fn scalar_size(ty: u32) -> Option<i64> {
        Some(match ty {
            0 | 1 | 7 => 1, // u8, i8, bool
            2 | 3 => 2,     // u16, i16
            4..=6 => 4,     // u32, i32, f32
            10..=12 => 8,   // u64, i64, f64
            _ => return None,
        })
    }
    fn skip_value<R: Read + Seek>(r: &mut R, ty: u32) -> Option<()> {
        match ty {
            8 => {
                let len = read_u64(r)?;
                r.seek(SeekFrom::Current(i64::try_from(len).ok()?)).ok()?;
            }
            9 => {
                let item_ty = read_u32(r)?;
                let count = read_u64(r)?;
                if let Some(size) = scalar_size(item_ty) {
                    let bytes = i64::try_from(count).ok()?.checked_mul(size)?;
                    r.seek(SeekFrom::Current(bytes)).ok()?;
                } else {
                    for _ in 0..count {
                        skip_value(r, item_ty)?;
                    }
                }
            }
            other => {
                let size = scalar_size(other)?;
                r.seek(SeekFrom::Current(size)).ok()?;
            }
        }
        Some(())
    }

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let version = read_u32(&mut reader)?;
    if version < 2 {
        // GGUF v1 used 32-bit counts; nothing we ship is that old.
        return None;
    }
    let _tensor_count = read_u64(&mut reader)?;
    let kv_count = read_u64(&mut reader)?;
    for _ in 0..kv_count {
        let key = read_string(&mut reader)?;
        let ty = read_u32(&mut reader)?;
        if key == wanted.as_bytes() && ty == 8 {
            let value = read_string(&mut reader)?;
            return Some(String::from_utf8_lossy(&value).into_owned());
        }
        skip_value(&mut reader, ty)?;
    }
    None
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
    /// that knows its image-token / prompt format.  Using the WRONG handler
    /// (or none) for a model crashes the native clip code, so an arch without
    /// an entry here is treated as unsupported (text-only / skipped) rather
    /// than guessed.  `gemma3`/`gemma4` are absent because their model classes
    /// declare the handler explicitly via `vision_chat_handler()`, which is
    /// authoritative and bypasses this arch-sniffing fallback.
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

    /// Try loading the model in a subprocess to detect crashes (SIGABRT/segfault).
    ///
    /// `Some(true)` if the probe succeeded, `Some(false)` if it crashed (signal,
    /// timeout, spawn failure), `None` if it failed with a normal error exit.
    pub fn probe_model_load(path: &str, config: &LlamaParams) -> Option<bool> {
        let _ = path;
        let config_json = match serde_json::to_string(config) {
            Ok(json) => json,
            Err(e) => {
                *LAST_PROBE_STDERR.lock() = e.to_string();
                tracing::warn!(target: LOG_TARGET, "Probe load error: {e}");
                return Some(false);
            }
        };
        match run_probe_process(&config_json) {
            Ok(Some((exit, stderr))) => {
                let failed = match exit {
                    ProbeExit::Code(0) => {
                        LAST_PROBE_STDERR.lock().clear();
                        return Some(true);
                    }
                    other => other,
                };
                let tail = if stderr.is_empty() { "no output".to_string() } else { tail_chars(&stderr, 500) };
                *LAST_PROBE_STDERR.lock() = tail.clone();
                match failed {
                    // A signal (SIGABRT, SIGSEGV) would crash the main process.
                    ProbeExit::Signal(signal) => {
                        tracing::warn!(target: LOG_TARGET, "Probe crashed with signal {signal}: {tail}");
                        Some(false)
                    }
                    // A normal error exit is an ordinary load failure we can
                    // handle in-process.
                    ProbeExit::Code(code) => {
                        tracing::warn!(target: LOG_TARGET, "Probe raised exception (exit code {code}): {tail}");
                        None
                    }
                }
            }
            Ok(None) => {
                *LAST_PROBE_STDERR.lock() = format!("probe timed out after {PROBE_TIMEOUT_SECS}s");
                tracing::warn!(target: LOG_TARGET, "Probe load timed out");
                Some(false)
            }
            Err(e) => {
                *LAST_PROBE_STDERR.lock() = e.to_string();
                tracing::warn!(target: LOG_TARGET, "Probe load error: {e}");
                Some(false)
            }
        }
    }

    /// Body of the probe subprocess: load the model from `config_json` and
    /// drop it.  Returns the process exit code (0 on success, 1 on an error).
    pub fn run_probe(config_json: &str) -> i32 {
        let result = serde_json::from_str::<LlamaParams>(config_json)
            .map_err(anyhow::Error::from)
            .and_then(|config| Llama::new(config, None));
        match result {
            Ok(llm) => {
                drop(llm);
                0
            }
            Err(e) => {
                eprintln!("{e:#}");
                1
            }
        }
    }

    /// A model GGUF's `general.architecture` (e.g. `qwen2vl`).
    pub fn read_gguf_arch(path: &str) -> Option<String> {
        read_gguf_string_field(path, "general.architecture")
    }

    /// Name of the supported vision handler for this model, or `None`.
    pub fn vision_handler_class_name(model_path: &str) -> Option<String> {
        let arch = Self::read_gguf_arch(model_path).unwrap_or_default().to_lowercase();
        Self::VISION_HANDLER_BY_ARCH
            .iter()
            .find(|(keys, _)| keys.iter().any(|k| arch.contains(k)))
            .map(|(_, name)| name.to_string())
    }

    /// True when this build can run this model's vision input.
    ///
    /// `handler_name` is the handler the model class declares
    /// (`vision_chat_handler()`); when given it is authoritative and the
    /// GGUF-arch guess is skipped.
    pub fn vision_handler_supported(model_path: &str, handler_name: Option<&str>) -> bool {
        let name = match handler_name.filter(|n| !n.is_empty()) {
            Some(name) => name.to_string(),
            None => match Self::vision_handler_class_name(model_path) {
                Some(name) => name,
                None => return false,
            },
        };
        llama::vision_chat_handler_available(&name)
    }

    /// Make the multimodal (CLIP/mmproj) encoder load on CPU.  Idempotent.
    ///
    /// The `Llava15`-derived vision handlers always put the projector on the
    /// GPU; on a VRAM-constrained device that OOMs *inside the native clip
    /// encode* — an uncatchable crash.  This flips every later projector
    /// context to CPU; the LLM itself still runs on GPU, and encoding one page
    /// image on CPU is fine for batch OCR.
    pub fn force_clip_on_cpu() {
        if llama::force_clip_on_cpu() {
            tracing::info!(target: LOG_TARGET, "[vision] CLIP/mmproj forced to CPU (avoids GPU OOM in clip encode)");
        }
    }

    /// Build the multimodal chat handler for this model, or `None` (load
    /// text-only) when no supported handler exists or it fails to initialize.
    ///
    /// `handler_name` is the handler declared by the model class and is
    /// authoritative when provided; otherwise the handler is inferred from the
    /// GGUF architecture.  `clip_on_cpu` forces the CLIP/mmproj encoder onto
    /// CPU; the caller sets it only when GPU VRAM is too tight to run the
    /// projector without an OOM crash.
    pub fn make_vision_chat_handler(
        mmproj_path: &str,
        model_path: &str,
        clip_on_cpu: bool,
        handler_name: Option<&str>,
    ) -> Option<ChatHandler> {
        let cls_name = match handler_name.filter(|n| !n.is_empty()) {
            Some(name) => name.to_string(),
            None => match Self::vision_handler_class_name(model_path) {
                Some(name) => name,
                None => {
                    let arch = match Self::read_gguf_arch(model_path) {
                        Some(arch) => format!("'{arch}'"),
                        None => "None".to_string(),
                    };
                    tracing::warn!(
                        target: LOG_TARGET,
                        "No supported vision chat handler for architecture {arch}; loading text-only (image OCR disabled for this model)"
                    );
                    return None;
                }
            },
        };
        if !llama::vision_chat_handler_available(&cls_name) {
            tracing::warn!(target: LOG_TARGET, "Vision handler {cls_name} not in installed llama-cpp; loading text-only");
            return None;
        }
        // Keep the projector off the GPU only when asked — its image-encode
        // pass crashes the native clip code on a tight-VRAM device.  With
        // enough VRAM it stays on GPU (faster).  The MTMD handlers take
        // `use_gpu` as a constructor argument, which scopes the choice to THIS
        // handler; the global switch is the fallback for the older
        // Llava15-derived handlers.  Prefer the argument: the switch is global
        // and permanent, so one tight-VRAM vision load would otherwise pin
        // every later projector in the process to CPU.
        let mut use_gpu = None;
        if clip_on_cpu {
            if llama::vision_chat_handler_takes_use_gpu(&cls_name) {
                use_gpu = Some(false);
            } else {
                Self::force_clip_on_cpu();
            }
        }
        match llama::make_vision_chat_handler(&cls_name, mmproj_path, use_gpu) {
            Ok(handler) => Some(handler),
            Err(e) => {
                tracing::warn!(target: LOG_TARGET, "Failed to init vision handler {cls_name} ({e}); loading text-only");
                None
            }
        }
    }

    /// True when `model_name` declares `ModelCapability::Thinking`.
    ///
    /// That capability is the single source of truth for "this model's chat
    /// template reads `enable_thinking`" — it is what the model picker shows
    /// and what gates the per-request `thinking` flag.  False for a model
    /// outside `SUPPORTED_MODELS` (user-added): binding the flag onto a
    /// template that ignores it does nothing.
    pub fn supports_thinking(model_name: Option<&str>) -> bool {
        let Some(model_name) = model_name.filter(|n| !n.is_empty()) else {
            return false;
        };
        match SUPPORTED_MODELS.get(model_name) {
            Some(model_cls) => (model_cls.default_capabilities)().contains(&ModelCapability::Thinking),
            None => false,
        }
    }

    /// `(n_batch, n_threads)` for this load.
    ///
    /// `n_batch` is the prefill batch: a 3k-token prompt went through 512 six
    /// times where a GPU can take it in two, and the buffer it grows is tens of
    /// megabytes.  `n_threads` only drives the layers that are *not* offloaded,
    /// so 4 costs little on a GPU load and most of the machine on a CPU one.
    ///
    /// Overridable with `SLP_LLM_N_BATCH` / `SLP_LLM_N_THREADS` for a device
    /// where the defaults are wrong.
    pub fn batch_and_threads(n_gpu_layers: i32) -> (usize, usize) {
        let offloaded = n_gpu_layers != 0;
        let (mut n_batch, mut n_threads) = if offloaded { (1024, 4) } else { (512, cpu_count()) };
        if let Ok(settings) = AppSettings::try_new() {
            if settings.llm_n_batch > 0 {
                n_batch = settings.llm_n_batch as usize;
            }
            if settings.llm_n_threads > 0 {
                n_threads = settings.llm_n_threads as usize;
            }
        }
        tracing::info!(
            target: LOG_TARGET,
            "[model] n_batch={n_batch}, n_threads={n_threads} (n_gpu_layers={n_gpu_layers})"
        );
        (n_batch, n_threads)
    }

    /// True when the loaded GGUF's own chat template reads `enable_thinking`.
    ///
    /// `supports_thinking` answers from `SUPPORTED_MODELS`, which knows nothing
    /// about a model the user built or merged themselves — and a merge of a
    /// reasoning base keeps its template.  The template is the ground truth.
    pub fn template_reads_thinking(llm: &dyn LlamaModel) -> bool {
        llm.metadata()
            .get("tokenizer.chat_template")
            .is_some_and(|template| template.contains("enable_thinking"))
    }

    /// Whether `model_name` should reason, or `None` to leave it alone.
    ///
    /// `None` means the model does not support reasoning, so nothing is bound.
    /// Otherwise the answer is, most specific first: the caller's `override_`
    /// (the request's `thinking` field), then `slp_thinking_enabled`, then the
    /// model class's `thinking_default()`.
    pub fn thinking_for(model_name: Option<&str>, override_: Option<bool>) -> Option<bool> {
        if !Self::supports_thinking(model_name) {
            return None;
        }
        if let Some(value) = override_ {
            return Some(value);
        }
        if let Some(setting) = AppSettings::new().thinking_enabled {
            return Some(setting);
        }
        let default = model_name
            .and_then(|name| SUPPORTED_MODELS.get(name))
            .map(|model_cls| (model_cls.thinking_default)())
            .unwrap_or(false);
        Some(default)
    }

    /// Render every chat completion with `enable_thinking=enabled`.
    ///
    /// Hybrid-reasoning models read that variable from their own chat template
    /// to decide whether to open a `<think>` block.  The flag is bound onto the
    /// resolved chat handler once at load time — handlers pass their template
    /// kwargs straight to the render — and on the model rather than at each
    /// call site, so the direct `create_chat_completion` callers (vision Q&A,
    /// OCR) are covered too.
    ///
    /// The bound value is read from the model's `_slp_thinking` flag on every
    /// call rather than baked in, so a caller can turn reasoning off for one
    /// completion — see `LlamaCpp::thinking_for_budget`.  A caller that passes
    /// `enable_thinking` itself keeps its value.
    ///
    /// Best-effort: a model that resolves handlers differently keeps its
    /// template default rather than failing the load.
    pub fn bind_thinking(llm: &dyn LlamaModel, enabled: bool) {
        // Same resolution order as `create_chat_completion`.
        let handler = match llama::resolve_chat_handler(llm) {
            Ok(handler) => handler,
            Err(e) => {
                tracing::warn!(target: LOG_TARGET, "[thinking] Could not set enable_thinking={}: {e}", py_bool(enabled));
                return;
            }
        };

        let with_thinking: ChatHandler = Arc::new(move |mut req: ChatHandlerRequest<'_>| {
            // Read the flag off the model the call names, not a captured
            // reference — the handler must not keep the model alive.
            if !req.template_kwargs.contains_key("enable_thinking") {
                let value = req.llama.and_then(|target| target.state().thinking()).unwrap_or(enabled);
                req.template_kwargs.insert("enable_thinking".to_string(), Value::Bool(value));
            }
            handler(req)
        });

        let state = llm.state();
        state.set_thinking(Some(enabled));
        state.set_chat_handler(Some(with_thinking));
        tracing::info!(target: LOG_TARGET, "[thinking] enable_thinking={} bound to chat template", py_bool(enabled));
    }

    /// Load a GGUF and return the configured [`LlamaCpp`] wrapper.
    ///
    /// * `probe` — when true (default) the model is first test-loaded in a
    ///   throwaway subprocess so a hard crash can't take down the caller.  Set
    ///   false only from a caller that is itself disposable, where a nested
    ///   probe would just add another concurrent model load.
    /// * `max_n_ctx` — upper bound for the auto-sized context window.  The
    ///   auto-sizing otherwise assumes the model is alone on the device and
    ///   reserves nearly all memory for it, starving the on-demand models.
    /// * `mmproj_path` — CLIP/mmproj GGUF.  When set, the model is loaded as a
    ///   multimodal (vision) instance with a vision chat handler; the
    ///   subprocess probe is skipped and the load is forced in-process.
    /// * `vision_handler` — chat-handler name declared by the model class;
    ///   authoritative when set, otherwise inferred from the GGUF architecture.
    /// * `thinking` — turn the model's `<think>` block on/off; `None` keeps the
    ///   chat template's own default.  The block itself is stripped from
    ///   replies by [`crate::llms::thinking`].
    pub fn get_llama_cpp_instance(
        path: &str,
        project_id: &str,
        lora_path: Option<&str>,
        options: LlamaInstanceOptions,
    ) -> anyhow::Result<LlamaCpp> {
        let LlamaInstanceOptions { mut probe, max_n_ctx, mmproj_path, clip_on_cpu, vision_handler, mut thinking, model_name } =
            options;

        let mut chat_handler: Option<ChatHandler> = None;
        if let Some(mmproj) = mmproj_path.as_deref() {
            chat_handler = Self::make_vision_chat_handler(mmproj, path, clip_on_cpu, vision_handler.as_deref());
            if chat_handler.is_some() {
                // The handler holds a loaded CLIP model; it can't be handed to
                // the probe subprocess, so load directly in-process.
                probe = false;
            }
        }

        let app_cfg = AppSettings::new();
        let device_config = get_device_config();
        let mut llama_config = device_config.get_llama_cpp_config(u32::try_from(app_cfg.n_parallel.max(1)).unwrap_or(1));

        // Model file size, to estimate memory requirements.
        let model_size_gb = std::fs::metadata(path)
            .map_err(|e| anyhow::anyhow!("[Errno 2] {e}: '{path}'"))?
            .len() as f64
            / 1024f64.powi(3);

        // Calculate optimal context size based on available memory.
        let mut n_ctx: usize = 8192; // Default context size

        // Adjust GPU layers and context based on available memory.  VRAM is not
        // probed: GPU offload is decided by llama.cpp and device memory is
        // managed by the backend, so discrete-GPU boxes keep full offload with
        // the default context (the subprocess probe below reduces the context
        // or falls back to CPU if it doesn't fit).
        if device_config.device.value() == "mps" {
            // Apple Silicon — unified memory shared between CPU and GPU.
            // Metal handles GPU offloading via n_gpu_layers=-1 (already set).
            let (available_mem_gb, total_mem_gb) = crate::utils::device_utils::available_memory_gb();

            // Use available memory minus model size and a 2GB OS/app overhead.
            let available_for_kv = available_mem_gb - model_size_gb - 2.0;

            if available_for_kv > 0.0 {
                let kv_per_8k = if model_size_gb < 3.0 {
                    0.75
                } else if model_size_gb < 6.0 {
                    1.5
                } else {
                    2.0
                };
                let estimated_ctx = ((available_for_kv / kv_per_8k) * 8192.0) as i64;
                // Apple Silicon can handle larger contexts with unified memory.
                let max_ctx: i64 = if total_mem_gb >= 32.0 {
                    131072
                } else if total_mem_gb >= 16.0 {
                    65536
                } else {
                    32768
                };
                n_ctx = estimated_ctx.min(max_ctx).max(4096) as usize;
            } else {
                n_ctx = 4096;
            }

            tracing::info!(
                target: LOG_TARGET,
                "Apple Silicon: {total_mem_gb:.0}GB total, {available_mem_gb:.1}GB available, {model_size_gb:.1}GB model — using n_ctx={n_ctx}, n_gpu_layers={}",
                llama_config.n_gpu_layers
            );
        } else if llama::llama_supports_gpu_offload() {
            // GPU offload via the Vulkan (or other) llama.cpp backend.  VRAM is
            // not queryable, so keep the full offload with the default
            // context; the probe below degrades to reduced-context / CPU if it
            // doesn't fit.
            tracing::info!(
                target: LOG_TARGET,
                "GPU offload backend available (VRAM managed by the backend) — using n_ctx={n_ctx}, n_gpu_layers={}",
                llama_config.n_gpu_layers
            );
        } else {
            // No GPU available — CPU only.
            llama_config.n_gpu_layers = 0;
            tracing::info!(
                target: LOG_TARGET,
                "No GPU available, offloading entire model to CPU with {} threads",
                cpu_count()
            );
        }

        // Cap the auto-sized context for worker copies so multiple model
        // instances (each with its own KV cache) fit on a single GPU.  A cap
        // of 0/None means no cap — the model uses its full auto-sized window.
        if let Some(cap) = max_n_ctx.filter(|c| *c != 0) {
            if n_ctx > cap {
                tracing::info!(target: LOG_TARGET, "Capping context {n_ctx} → {cap} (shared-GPU worker copy)");
                n_ctx = cap;
            }
        }

        let (n_batch, n_threads) = Self::batch_and_threads(llama_config.n_gpu_layers);
        let mut config = LlamaParams {
            model_path: path.to_string(),
            n_ctx,
            n_batch,
            n_threads,
            n_gpu_layers: llama_config.n_gpu_layers,
            verbose: llama_config.verbose,
            n_parallel: llama_config.n_parallel as usize,
            lora_path: None,
            flash_attn: false,
            chat_format: None,
        };

        if let Some(lora) = lora_path {
            config.lora_path = Some(lora.to_string());
            tracing::info!(target: LOG_TARGET, "Loading LoRA adapter: {lora}");
        }

        if chat_handler.is_some() {
            // Flash attention — VISION LOAD, CUDA ONLY.  Without it the image
            // encode allocates a fixed ~6GB compute buffer that crashes the
            // native clip code on a tight CUDA GPU; flash_attn brings the
            // per-page peak to ~3GB.  On Metal and Vulkan flash_attn is
            // unsupported and makes the load fail, so gate on llama.cpp's
            // compiled backend.
            if llama_cpp_has_cuda_backend() {
                config.flash_attn = true;
            }
            tracing::info!(
                target: LOG_TARGET,
                "Loading vision model with mmproj: {} (flash_attn={})",
                mmproj_path.as_deref().unwrap_or_default(),
                py_bool(config.flash_attn)
            );
        }

        tracing::info!(
            target: LOG_TARGET,
            "Loading model with config: n_ctx={n_ctx}, n_gpu_layers={}, model_size={model_size_gb:.1}GB",
            config.n_gpu_layers
        );

        // Probe in a subprocess first to detect crashes (SIGABRT/segfault) that
        // would kill the main process.  A minimal context is enough — the
        // probe only needs to know whether the model loads at all.
        // Returns: Some(true) success, Some(false) crash/signal, None error exit.
        // Skipped when probe=false: a caller that is already a disposable
        // subprocess gains nothing from a nested probe.
        let probe_result = if probe {
            Self::probe_model_load(path, &LlamaParams { n_ctx: 512, ..config.clone() })
        } else {
            None
        };
        if probe_result == Some(false) {
            // The subprocess crashed — try progressively safer configs.
            let mut fallback_configs: Vec<(&str, LlamaParams)> = Vec::new();
            if config.n_ctx > 2048 {
                fallback_configs.push(("reduced context", LlamaParams { n_ctx: 2048, ..config.clone() }));
            }
            if config.n_gpu_layers != 0 {
                fallback_configs.push((
                    "CPU-only",
                    LlamaParams { n_ctx: config.n_ctx.min(4096), n_gpu_layers: 0, ..config.clone() },
                ));
            }
            fallback_configs.push(("minimal", LlamaParams { n_ctx: 2048, n_gpu_layers: 0, ..config.clone() }));

            let mut probed = false;
            let mut last_stderr = String::new();
            for (label, fallback) in fallback_configs {
                tracing::warn!(
                    target: LOG_TARGET,
                    "Trying {label} config: n_ctx={}, n_gpu_layers={}",
                    fallback.n_ctx,
                    fallback.n_gpu_layers
                );
                let result = Self::probe_model_load(path, &LlamaParams { n_ctx: 512, ..fallback.clone() });
                if result != Some(false) {
                    // Some(true) = clean success, None = an error exit (safe to
                    // try in-process).
                    config = fallback;
                    probed = true;
                    break;
                }
                // Capture stderr for diagnostics on final failure.
                last_stderr = Self::last_probe_stderr();
            }

            if !probed {
                let diag = if last_stderr.is_empty() { String::new() } else { format!(" Last error: {last_stderr}") };
                anyhow::bail!(
                    "Model at {path} cannot be loaded — all configurations crashed. Check GPU drivers and available memory.{diag}"
                );
            }
        }
        // When the probe failed with an ordinary error exit, the in-process
        // load below fails the same way, with a proper error and full context.

        let llm = match Llama::new(config.clone(), chat_handler.clone()) {
            Ok(llm) => llm,
            Err(e) => {
                tracing::error!(
                    target: LOG_TARGET,
                    "Failed to load model (n_gpu_layers={}): {e}",
                    config.n_gpu_layers
                );
                if config.n_ctx > 2048 {
                    // Fallback 1: reduce context size and retry with GPU.
                    let reduced_ctx = (config.n_ctx / 2).max(2048);
                    tracing::warn!(target: LOG_TARGET, "Retrying with reduced context: {reduced_ctx} tokens...");
                    config.n_ctx = reduced_ctx;
                    match Llama::new(config.clone(), chat_handler.clone()) {
                        Ok(llm) => llm,
                        Err(e2) => {
                            tracing::error!(target: LOG_TARGET, "Failed to load model with reduced context: {e2}");
                            // Fallback 2: CPU-only mode with reduced context.
                            if config.n_gpu_layers != 0 {
                                tracing::warn!(
                                    target: LOG_TARGET,
                                    "FALLING BACK TO CPU-ONLY (n_gpu_layers 0) — GPU load failed"
                                );
                                config.n_gpu_layers = 0;
                                Llama::new(config.clone(), chat_handler.clone())?
                            } else {
                                return Err(e2);
                            }
                        }
                    }
                } else if config.n_gpu_layers != 0 {
                    // Fallback: CPU-only mode if GPU loading fails.
                    tracing::warn!(target: LOG_TARGET, "FALLING BACK TO CPU-ONLY (n_gpu_layers 0) — GPU load failed");
                    config.n_gpu_layers = 0;
                    Llama::new(config.clone(), chat_handler.clone())?
                } else {
                    return Err(e);
                }
            }
        };

        if thinking.is_none() {
            // The registry could not answer — a user-built or merged model is
            // not in SUPPORTED_MODELS, so nothing declares THINKING for it.
            // The GGUF itself can: its chat template either reads
            // `enable_thinking` or it doesn't.  Bind it off by default, which
            // keeps the visible answer as it is and still leaves the flag
            // present, so a caller that wants reasoning for its own call can
            // turn it on.
            if Self::template_reads_thinking(&llm) {
                tracing::info!(
                    target: LOG_TARGET,
                    "[thinking] chat template reads enable_thinking — binding the flag for a model the registry does not describe"
                );
                thinking = Some(AppSettings::new().thinking_enabled.unwrap_or(false));
            }
        }

        if let Some(enabled) = thinking {
            Self::bind_thinking(&llm, enabled);
        }

        Ok(LlamaCpp::new(Arc::new(llm), project_id, false, model_name))
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn gguf_string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    #[test]
    fn reads_architecture_from_a_gguf_header() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes()); // tensors
        bytes.extend(3u64.to_le_bytes()); // kv pairs
        // An array of strings to skip.
        gguf_string(&mut bytes, "tokenizer.ggml.tokens");
        bytes.extend(9u32.to_le_bytes());
        bytes.extend(8u32.to_le_bytes());
        bytes.extend(2u64.to_le_bytes());
        gguf_string(&mut bytes, "a");
        gguf_string(&mut bytes, "bc");
        // A scalar to skip.
        gguf_string(&mut bytes, "general.alignment");
        bytes.extend(4u32.to_le_bytes());
        bytes.extend(32u32.to_le_bytes());
        gguf_string(&mut bytes, "general.architecture");
        bytes.extend(8u32.to_le_bytes());
        gguf_string(&mut bytes, "qwen2vl");

        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        let path = file.path().to_str().unwrap();
        assert_eq!(LlmFactory::read_gguf_arch(path).as_deref(), Some("qwen2vl"));
        assert_eq!(LlmFactory::vision_handler_class_name(path).as_deref(), Some("Qwen25VLChatHandler"));
        assert!(LlmFactory::vision_handler_supported(path, None));
        assert!(LlmFactory::vision_handler_supported("/nonexistent", Some("Gemma4ChatHandler")));
        assert!(!LlmFactory::vision_handler_supported("/nonexistent", None));
        assert_eq!(LlmFactory::read_gguf_arch("/nonexistent"), None);
    }

    #[test]
    fn tail_keeps_the_last_characters() {
        assert_eq!(tail_chars("abcdef", 3), "def");
        assert_eq!(tail_chars("ab", 3), "ab");
    }
}
