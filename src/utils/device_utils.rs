//! Port of `smartloop/utils/device_utils.py`.
//!
//! Device detection and configuration utilities for cross-platform
//! compatibility.  GPU availability is decided by llama.cpp's backend.

use serde::{Deserialize, Serialize};

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

impl DeviceConfig {
    /// Detect the device and attention implementation.
    pub fn new() -> Self {
        todo!()
    }

    /// Detect the best available device via llama.cpp.
    fn detect_device() -> DeviceType {
        todo!()
    }

    /// Human-readable description of the device.
    pub fn get_device_description(&self) -> String {
        todo!()
    }

    /// Best attention implementation for the device ("sdpa" or "eager").
    fn get_attention_implementation(device: DeviceType) -> String {
        let _ = device;
        todo!()
    }

    /// Configuration for llama.cpp based on the device.
    pub fn get_llama_cpp_config(&self, n_parallel: u32) -> LlamaCppConfig {
        let _ = n_parallel;
        todo!()
    }

    /// Optimal batch size based on device.
    pub fn get_batch_size(&self) -> usize {
        todo!()
    }

    /// Log the current device configuration.
    pub fn log_config(&self) {
        todo!()
    }
}

/// Get a device configuration instance (and log it).
pub fn get_device_config() -> DeviceConfig {
    todo!()
}

/// True when the linked llama.cpp was built with a GPU backend.
///
/// This is the single source of truth for "is a GPU usable" across the app.
/// GPU availability and device memory are managed by llama.cpp / the backend
/// (Vulkan, Metal, CUDA) — VRAM is never probed directly.
pub fn llama_gpu_offload() -> bool {
    todo!()
}

/// Check if running on Apple Silicon (macOS).
pub fn is_apple_silicon() -> bool {
    todo!()
}

/// Upper bound (tokens) the llama.cpp factory will auto-size a model's
/// context window to on this device when no explicit cap is set
/// (131072 on >=32GB, 65536 on >=16GB, else 32768 on Apple Silicon; 8192
/// elsewhere).
pub fn max_auto_context_tokens() -> u32 {
    todo!()
}

/// True on Apple Silicon with a llama.cpp Metal (GPU offload) backend.
pub fn is_mps() -> bool {
    todo!()
}
