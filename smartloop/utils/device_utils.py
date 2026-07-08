"""Device detection and configuration utilities for cross-platform compatibility"""
import enum
import os
import subprocess
import torch
import logging
from pathlib import Path
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
    OPENVINO = "openvino"  # Intel iGPU/NPU via the ggml-openvino backend


def _backend_dll_present(stem: str) -> bool:
    """True when the given ggml backend library ships in the llama_cpp lib dir."""
    try:
        import llama_cpp
        lib_dir = Path(llama_cpp.__file__).parent / "lib"
        return any(
            (lib_dir / name).exists()
            for name in (f"{stem}.dll", f"lib{stem}.so", f"{stem}.so")
        )
    except Exception:
        return False


def _cuda_gpu_present() -> bool:
    """True when an NVIDIA GPU is usable (torch CUDA, or nvidia-smi for CPU-only torch wheels)."""
    try:
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    try:
        return subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, timeout=5
        ).returncode == 0
    except Exception:
        return False


_intel_devices_cache: dict = {}


def available_intel_devices() -> list:
    """Ordered Intel accelerators to try (plain ``NPU``/``GPU``), or [] when
    none exist (or the OpenVINO runtime is unavailable).

    NPU is preferred over GPU (product decision: keep GPUs free and run
    sustained LLM load on the power-efficient NPU); GPU stays in the list as
    a fallback for models whose ops don't compile on the NPU.  An NVIDIA GPU
    does NOT disable this list — when an OpenVINO IR model is available,
    Intel acceleration is preferred over CUDA by design.
    ``SMARTLOOP_OV_DEVICE`` overrides the list with exactly that device.
    Memoised — the OpenVINO Core() probe costs a few hundred ms.
    """
    override = os.environ.get("SMARTLOOP_OV_DEVICE")
    if override:
        return [override.upper()]
    if "devices" in _intel_devices_cache:
        return _intel_devices_cache["devices"]

    try:
        from openvino import Core
        bases = {d.split(".")[0].upper() for d in Core().available_devices}
        devices = [n for n in ("NPU", "GPU") if n in bases]
    except Exception:
        devices = []
    _intel_devices_cache["devices"] = devices
    return devices


def preferred_intel_device() -> Optional[str]:
    """First entry of available_intel_devices(), or None."""
    devices = available_intel_devices()
    return devices[0] if devices else None


def ensure_openvino_device_env() -> Optional[str]:
    """Export GGML_OPENVINO_DEVICE before the first llama.cpp backend init.

    The ggml-openvino backend reads GGML_OPENVINO_DEVICE exactly once, when
    the backend registers (the first llama_cpp backend init anywhere in the
    process); setting it later has no effect.  It also only accepts the plain
    device names CPU/GPU/NPU — a numbered name such as GPU.0 is silently
    ignored and the backend falls back to its (buggy) intel_cpu device.

    Called from DeviceConfig._detect_device(), the common chokepoint of every
    model-load path, so it runs before anything initialises llama_cpp.
    Returns the exported device name, or None when OpenVINO routing does not
    apply (no OV build, working NVIDIA GPU present, or no Intel accelerator).
    """
    existing = os.environ.get("GGML_OPENVINO_DEVICE")
    if existing:
        return existing
    if not _backend_dll_present("ggml-openvino"):
        return None
    # Unlike the GenAI route, the ggml-openvino backend stays behind CUDA:
    # it cannot decode several architectures, so a working NVIDIA GPU wins
    # for GGUF loads (its layers get pinned to CUDA0 by the factory).
    if _backend_dll_present("ggml-cuda") and _cuda_gpu_present():
        return None
    device = preferred_intel_device()
    if device:
        os.environ["GGML_OPENVINO_DEVICE"] = device
    return device


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
        # Must run before llama_cpp initialises its backends — the OpenVINO
        # device choice is latched at backend registration.
        ov_device = ensure_openvino_device_env()
        from llama_cpp import llama_supports_gpu_offload
        if llama_supports_gpu_offload():
            if sys.platform == 'darwin':
                return DeviceType.MPS
            # ensure_openvino_device_env only exports a device when no
            # working NVIDIA GPU is present, so it doubles as the CUDA-vs-
            # OpenVINO discriminator here.
            if ov_device:
                return DeviceType.OPENVINO
            return DeviceType.CUDA
        return DeviceType.CPU

    def get_device_description(self) -> str:
        """Get a human-readable description of the device"""
        if self.device == DeviceType.CUDA:
            return "NVIDIA GPU"
        elif self.device == DeviceType.MPS:
            return "Apple Metal GPU"
        elif self.device == DeviceType.OPENVINO:
            ov = os.environ.get("GGML_OPENVINO_DEVICE", "auto")
            return f"Intel {ov} (OpenVINO)"
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
