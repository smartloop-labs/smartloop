"""Device detection and configuration utilities for cross-platform compatibility"""
import sys
import enum
import logging

from pydantic import BaseModel
from llama_cpp import llama_supports_gpu_offload

logger = logging.getLogger(__name__)

class DeviceType(enum.Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

class LlamaCppConfig(BaseModel):
    """LlamaCpp configuration class"""
    n_gpu_layers: int = -1  # -1 = offload all layers to GPU
    verbose: bool = False
    n_parallel: int = 1  # number of parallel inference sequences (continuous batching)


class DeviceConfig:
    """Device configuration class that handles CUDA, MPS (Apple Metal), and CPU"""

    def __init__(self):
        self.device = self._detect_device()
        self.attn_implementation = self._get_attention_implementation()

    def _detect_device(self) -> DeviceType:
        """Detect the best available device via llama_cpp"""
        if llama_gpu_offload():
            return DeviceType.MPS if sys.platform == 'darwin' else DeviceType.CUDA
        return DeviceType.CPU

    def get_device_description(self) -> str:
        """Get a human-readable description of the device"""
        if self.device == DeviceType.CUDA:
            return "NVIDIA GPU"
        elif self.device == DeviceType.MPS:
            return "Apple Metal GPU"
        return "CPU"

    def _get_attention_implementation(self) -> str:
        """Get the best attention implementation for the device"""
        if self.device in (DeviceType.CUDA, DeviceType.MPS):
            return "sdpa"
        return "eager"

    def get_llama_cpp_config(self, n_parallel: int = 1) -> LlamaCppConfig:
        """Get configuration for llama-cpp-python based on device"""
        n_gpu_layers = -1 if llama_supports_gpu_offload() else 0
        return LlamaCppConfig(
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            n_parallel=max(1, n_parallel),
        )

    def get_batch_size(self) -> int:
        """Get optimal batch size based on device"""
        if self.device == DeviceType.CUDA:
            return 4
        elif self.device == DeviceType.MPS:
            try:
                import psutil
                total_memory_gb = psutil.virtual_memory().total / (1024**3)
                return min(8, max(2, int(total_memory_gb / 4)))
            except Exception:
                return 4
        return 2

    def log_config(self):
        """Log the current device configuration"""
        logger.info("Device Configuration:")
        logger.info(f"  Device: {self.device.value}")
        logger.info(f"  Attention implementation: {self.attn_implementation}")


def get_device_config() -> DeviceConfig:
    """Get a device configuration instance"""
    config = DeviceConfig()
    config.log_config()
    return config


def llama_gpu_offload() -> bool:
    """True when the installed llama.cpp was built with a GPU backend.

    This is the single source of truth for "is a GPU usable" across the app.
    GPU availability and device memory are managed by llama.cpp / the backend
    (Vulkan, Metal, CUDA) — VRAM is never probed directly.
    """
    try:
        from llama_cpp import llama_supports_gpu_offload
        return bool(llama_supports_gpu_offload())
    except Exception:
        return False


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M1/M2/M3 etc.)"""
    return sys.platform == 'darwin'


def max_auto_context_tokens() -> int:
    """Upper bound (tokens) the llama.cpp factory will auto-size a model's
    context window to on this device when no explicit cap is set.

    Mirrors the Apple Silicon logic in ``LlmFactory.get_llama_cpp_instance``
    (131072 on ≥32GB, 65536 on ≥16GB, else 32768; the factory then picks the
    smaller of this bound and the memory-derived estimate).  Memory estimators
    (worker-pool sizing, large-model fit guard) use this so "no cap" still
    accounts for a realistic KV-cache footprint instead of zero.
    """
    try:
        if is_apple_silicon():
            import psutil
            total_gb = psutil.virtual_memory().total / (1024 ** 3)
            return 131072 if total_gb >= 32 else 65536 if total_gb >= 16 else 32768
    except Exception:
        pass
    return 8192


def is_mps() -> bool:
    """True on Apple Silicon with a llama.cpp Metal (GPU offload) backend."""
    return sys.platform == 'darwin' and llama_gpu_offload()
