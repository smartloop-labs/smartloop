//! Port of `smartloop/utils/device_utils.py`.
//!
//! Device detection and configuration utilities for cross-platform
//! compatibility.  GPU availability is decided by llama.cpp's backend.

use serde::{Deserialize, Serialize};

/// Logger name of the Python module.
const LOGGER: &str = "smartloop.utils.device_utils";

/// Detected compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    #[serde(rename = "cpu")]
    Cpu,
    #[serde(rename = "cuda")]
    Cuda,
    #[serde(rename = "mps")]
    Mps,
}

impl DeviceType {
    /// The Python enum `.value` string.
    pub fn value(&self) -> &'static str {
        match self {
            DeviceType::Cpu => "cpu",
            DeviceType::Cuda => "cuda",
            DeviceType::Mps => "mps",
        }
    }
}

/// LlamaCpp configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlamaCppConfig {
    /// -1 = offload all layers to GPU.
    #[serde(default = "default_n_gpu_layers")]
    pub n_gpu_layers: i32,
    #[serde(default)]
    pub verbose: bool,
    /// Number of parallel inference sequences (continuous batching).
    #[serde(default = "default_n_parallel")]
    pub n_parallel: u32,
}

fn default_n_gpu_layers() -> i32 {
    -1
}

fn default_n_parallel() -> u32 {
    1
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self { n_gpu_layers: -1, verbose: false, n_parallel: 1 }
    }
}

/// Device configuration that handles CUDA, MPS (Apple Metal), and CPU.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceConfig {
    pub device: DeviceType,
    pub attn_implementation: String,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Total system memory in GiB (psutil `virtual_memory().total / 1024**3`),
/// or `None` when it cannot be determined.
fn total_memory_gb() -> Option<f64> {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let total = sys.total_memory();
    if total == 0 {
        return None;
    }
    Some(total as f64 / (1024f64 * 1024.0 * 1024.0))
}

impl DeviceConfig {
    /// Detect the device and attention implementation.
    pub fn new() -> Self {
        let device = Self::detect_device();
        let attn_implementation = Self::get_attention_implementation(device);
        Self { device, attn_implementation }
    }

    /// Detect the best available device via llama.cpp.
    fn detect_device() -> DeviceType {
        if llama_gpu_offload() {
            return if cfg!(target_os = "macos") { DeviceType::Mps } else { DeviceType::Cuda };
        }
        DeviceType::Cpu
    }

    /// Human-readable description of the device.
    pub fn get_device_description(&self) -> String {
        match self.device {
            DeviceType::Cuda => "NVIDIA GPU",
            DeviceType::Mps => "Apple Metal GPU",
            DeviceType::Cpu => "CPU",
        }
        .to_string()
    }

    /// Best attention implementation for the device ("sdpa" or "eager").
    fn get_attention_implementation(device: DeviceType) -> String {
        match device {
            DeviceType::Cuda | DeviceType::Mps => "sdpa",
            DeviceType::Cpu => "eager",
        }
        .to_string()
    }

    /// Configuration for llama.cpp based on the device.
    pub fn get_llama_cpp_config(&self, n_parallel: u32) -> LlamaCppConfig {
        let n_gpu_layers = if crate::llms::llama::llama_supports_gpu_offload() { -1 } else { 0 };
        LlamaCppConfig { n_gpu_layers, verbose: false, n_parallel: n_parallel.max(1) }
    }

    /// Optimal batch size based on device.
    pub fn get_batch_size(&self) -> usize {
        match self.device {
            DeviceType::Cuda => 4,
            DeviceType::Mps => match total_memory_gb() {
                Some(total_memory_gb) => ((total_memory_gb / 4.0) as usize).clamp(2, 8),
                None => 4,
            },
            DeviceType::Cpu => 2,
        }
    }

    /// Log the current device configuration.
    pub fn log_config(&self) {
        tracing::info!(target: LOGGER, "Device Configuration:");
        tracing::info!(target: LOGGER, "  Device: {}", self.device.value());
        tracing::info!(target: LOGGER, "  Attention implementation: {}", self.attn_implementation);
    }
}

/// Get a device configuration instance (and log it).
pub fn get_device_config() -> DeviceConfig {
    let config = DeviceConfig::new();
    config.log_config();
    config
}

/// True when the linked llama.cpp was built with a GPU backend.
///
/// This is the single source of truth for "is a GPU usable" across the app.
/// GPU availability and device memory are managed by llama.cpp / the backend
/// (Vulkan, Metal, CUDA) — VRAM is never probed directly.
pub fn llama_gpu_offload() -> bool {
    // Python wraps the llama_cpp import + call in `except Exception: return False`.
    std::panic::catch_unwind(crate::llms::llama::llama_supports_gpu_offload).unwrap_or(false)
}

/// Check if running on Apple Silicon (macOS).
pub fn is_apple_silicon() -> bool {
    cfg!(target_os = "macos")
}

/// Upper bound (tokens) the llama.cpp factory will auto-size a model's
/// context window to on this device when no explicit cap is set.
///
/// Mirrors the Apple Silicon logic in `LlmFactory::get_llama_cpp_instance`
/// (131072 on >=32GB, 65536 on >=16GB, else 32768; the factory then picks the
/// smaller of this bound and the memory-derived estimate).  Memory estimators
/// (worker-pool sizing, large-model fit guard) use this so "no cap" still
/// accounts for a realistic KV-cache footprint instead of zero.
pub fn max_auto_context_tokens() -> u32 {
    if is_apple_silicon() {
        if let Some(total_gb) = total_memory_gb() {
            return if total_gb >= 32.0 {
                131072
            } else if total_gb >= 16.0 {
                65536
            } else {
                32768
            };
        }
    }
    8192
}

/// True on Apple Silicon with a llama.cpp Metal (GPU offload) backend.
pub fn is_mps() -> bool {
    cfg!(target_os = "macos") && llama_gpu_offload()
}
