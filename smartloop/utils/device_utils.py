"""Device detection and configuration utilities for cross-platform compatibility"""
import enum
import torch
import logging
from typing import TYPE_CHECKING, Dict, Any, Optional
from pydantic import BaseModel
import sys

if TYPE_CHECKING:
    import flash_attn  # type: ignore

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
        self.compute_dtype = self._get_compute_dtype()
        self.attn_implementation = self._get_attention_implementation()

    def _detect_device(self) -> DeviceType:
        """Detect the best available device via llama_cpp"""
        from llama_cpp import llama_supports_gpu_offload
        if llama_supports_gpu_offload():
            return DeviceType.MPS if sys.platform == 'darwin' else DeviceType.CUDA
        return DeviceType.CPU

    def get_device_description(self) -> str:
        """Get a human-readable description of the device"""
        if self.device == DeviceType.CUDA:
            return "NVIDIA GPU"
        elif self.device == DeviceType.MPS:
            return "Apple Metal GPU"
        return "CPU"

    def _get_compute_dtype(self) -> torch.dtype:
        """Get the appropriate compute dtype for the device"""
        if self.device in (DeviceType.CUDA, DeviceType.MPS):
            return torch.bfloat16
        return torch.float16

    def _get_attention_implementation(self) -> str:
        """Get the best attention implementation for the device"""
        if self.device in (DeviceType.CUDA, DeviceType.MPS):
            return "sdpa"
        return "eager"

    def get_llama_cpp_config(self, n_parallel: int = 1) -> LlamaCppConfig:
        """Get configuration for llama-cpp-python based on device"""
        from llama_cpp import llama_supports_gpu_offload
        n_gpu_layers = -1 if llama_supports_gpu_offload() else 0
        return LlamaCppConfig(
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            n_parallel=max(1, n_parallel),
        )

    def get_transformers_config(self, is_training: bool = False) -> Dict[str, Any]:
        """Get configuration for transformers models

        Args:
            is_training: If True, forces all layers onto a single device (required
                by SFTTrainer / Trainer which cannot handle device_map='auto').
        """
        config: Dict[str, Any] = {
            'attn_implementation': self.attn_implementation,
            'low_cpu_mem_usage': True,
            'torch_dtype': self.compute_dtype,
        }

        if self.device == DeviceType.CUDA:
            config['device_map'] = {'': 0} if is_training else 'auto'
        # MPS and CPU: no device_map

        return config

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
        logger.info(f"  Compute dtype: {self.compute_dtype}")
        logger.info(f"  Attention implementation: {self.attn_implementation}")


def get_device_config() -> DeviceConfig:
    """Get a device configuration instance"""
    config = DeviceConfig()
    config.log_config()
    return config


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M1/M2/M3 etc.)"""
    return sys.platform == 'darwin'
