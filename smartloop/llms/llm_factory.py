"""Factory to create LLM wrapper classes"""
import os
import json
import sys
import subprocess
import tempfile
from logging import Logger
from pathlib import Path
from smartloop.utils.device_utils import get_device_config


def _get_cuda_vram_gb() -> float | None:
    """Return total VRAM in GB for CUDA device 0, or None if unavailable.

    Tries torch first (fast), falls back to nvidia-smi when torch was
    installed without CUDA support (CPU-only wheel).
    """
    import torch
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().splitlines()[0]) / 1024  # MiB → GiB
    except Exception:
        pass
    return None


class LlmFactory:
    """Factory to create instances of LLM classes"""

    _last_probe_stderr: str = ""

    @staticmethod
    def _probe_model_load(path: str, config: dict, logger: Logger) -> bool | None:
        """Try loading the model in a subprocess to detect crashes (SIGABRT/segfault).

        Returns True if the probe succeeded, False if the subprocess crashed,
        or None if the subprocess raised a normal Python exception (exit code > 0).
        """
        if getattr(sys, 'frozen', False):
            # PyInstaller bundle: sys.executable is the app binary, not Python.
            # Skip subprocess probe; fall through to in-process load.
            return None

        probe_script = (
            "import sys, json\n"
            "from llama_cpp import Llama\n"
            "config = json.loads(sys.argv[1])\n"
            "llm = Llama(**config)\n"
            "del llm\n"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", probe_script, json.dumps(config)],
                timeout=120,
                capture_output=True,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode().strip()
                LlmFactory._last_probe_stderr = stderr[-500:] if stderr else "no output"
                # Negative exit codes indicate signals (SIGABRT=-6, SIGSEGV=-11)
                # which would crash the main process. Positive codes (1, 2) are
                # normal Python exceptions that we can handle in-process.
                if result.returncode < 0:
                    logger.warning(f"Probe crashed with signal {-result.returncode}: {LlmFactory._last_probe_stderr}")
                    return False
                else:
                    logger.warning(f"Probe raised exception (exit code {result.returncode}): {LlmFactory._last_probe_stderr}")
                    return None
            LlmFactory._last_probe_stderr = ""
            return True
        except subprocess.TimeoutExpired:
            LlmFactory._last_probe_stderr = "probe timed out after 120s"
            logger.warning("Probe load timed out")
            return False
        except Exception as e:
            LlmFactory._last_probe_stderr = str(e)
            logger.warning(f"Probe load error: {e}")
            return False

    @staticmethod
    def get_llama_cpp_instance(
        path: str, logger: Logger, project_id: str, lora_path: str | None = None,
        *, probe: bool = True, max_n_ctx: int | None = None,
    ):
        """Returns an instance of the configured LlamaCpp wrapper class.

        ``probe``: when True (default, main process) the model is first
        test-loaded in a throwaway subprocess so a hard crash (SIGABRT/
        segfault) can't take down the caller.  Set False inside a dedicated
        LLM worker process — the worker IS disposable, and spawning a nested
        probe there only adds another concurrent model load.

        ``max_n_ctx``: upper bound for the auto-sized context window.  Worker
        copies pass a small value so several models' KV caches fit on one GPU
        (the context auto-sizing otherwise assumes the model is alone on the
        device and reserves nearly all VRAM for a single copy).
        """
        import torch
        from llama_cpp import Llama
        from smartloop.llms.llama_cpp import LlamaCpp

        from smartloop.config.app_settings import AppSettings as _AppSettings
        _app_cfg = _AppSettings()
        device_config = get_device_config()
        llama_config = device_config.get_llama_cpp_config(n_parallel=_app_cfg.n_parallel)

        # Get model file size to estimate memory requirements
        model_size_gb = Path(path).stat().st_size / (1024**3)

        # Calculate optimal context size based on available memory
        n_ctx = 8192  # Default context size

        # Adjust GPU layers and context based on available memory
        vram_gb = _get_cuda_vram_gb()
        if vram_gb is not None:
            # Calculate available memory after model loading
            # Reserve memory for: model weights + KV cache + overhead
            available_for_kv = vram_gb - model_size_gb - 0.5  # 0.5GB overhead

            if available_for_kv > 0:
                # Estimate context size based on available memory
                # KV cache memory per 8k tokens (conservative estimates for quantized models)
                kv_per_8k = 0.25 if model_size_gb < 3 else 0.5 if model_size_gb < 6 else 1.0
                estimated_ctx = int((available_for_kv / kv_per_8k) * 8192)

                # Allow larger contexts — cap at model's trained context length or hardware limit
                max_ctx = 32768 if model_size_gb < 3 else 16384 if model_size_gb < 6 else 8192
                n_ctx = max(2048, min(estimated_ctx, max_ctx))
                logger.info(f"Calculated optimal context: {n_ctx} tokens for {vram_gb:.1f}GB VRAM, {model_size_gb:.1f}GB model")
            else:
                # Limited VRAM - use smaller context
                n_ctx = 2048
                logger.warning(f"Limited VRAM ({vram_gb:.1f}GB) for {model_size_gb:.1f}GB model, using context={n_ctx}")

            # If model is larger than available VRAM, reduce GPU layers
            if model_size_gb > vram_gb * 0.8:  # Use up to 80% VRAM for model weights
                # Estimate layers that can fit (assuming ~30-40 layers for 4B models)
                estimated_layers = int((vram_gb * 0.7) / model_size_gb * 35)
                llama_config.n_gpu_layers = max(0, estimated_layers)
                logger.warning(
                    f"Model size ({model_size_gb:.1f}GB) requires conservative GPU allocation. "
                    f"Using {llama_config.n_gpu_layers} GPU layers (rest on CPU)"
                )
        elif torch.backends.mps.is_available():
            # Apple Silicon — unified memory shared between CPU and GPU.
            # Metal handles GPU offloading via n_gpu_layers=-1 (already set).
            import psutil
            total_mem_gb = psutil.virtual_memory().total / (1024**3)
            available_mem_gb = psutil.virtual_memory().available / (1024**3)

            # Use available memory minus model size and a 2GB OS/app overhead
            available_for_kv = available_mem_gb - model_size_gb - 2.0

            if available_for_kv > 0:
                kv_per_8k = 0.75 if model_size_gb < 3 else 1.5 if model_size_gb < 6 else 2.0
                estimated_ctx = int((available_for_kv / kv_per_8k) * 8192)
                # Apple Silicon can handle larger contexts with unified memory
                max_ctx = 131072 if total_mem_gb >= 32 else 65536 if total_mem_gb >= 16 else 32768
                n_ctx = max(4096, min(estimated_ctx, max_ctx))
            else:
                n_ctx = 4096

            logger.info(
                f"Apple Silicon: {total_mem_gb:.0f}GB total, {available_mem_gb:.1f}GB available, "
                f"{model_size_gb:.1f}GB model — using n_ctx={n_ctx}, n_gpu_layers={llama_config.n_gpu_layers}"
            )
        else:
            # No GPU available — CPU only
            llama_config.n_gpu_layers = 0
            n_threads = os.cpu_count() or 4
            logger.info(f"No GPU available, offloading entire model to CPU with {n_threads} threads")

        # Cap the auto-sized context for worker copies so multiple model
        # instances (each with its own KV cache) fit on a single GPU.
        if max_n_ctx is not None and n_ctx > max_n_ctx:
            logger.info(
                f"Capping context {n_ctx} → {max_n_ctx} (shared-GPU worker copy)"
            )
            n_ctx = max_n_ctx

        config = {
            'model_path': path,
            'n_ctx': n_ctx,
            'n_batch': 512,  # Smaller batch size for memory efficiency
            'n_threads': 4,  # Limit CPU threads
            **llama_config.model_dump()
        }

        if lora_path:
            config['lora_path'] = lora_path
            logger.info(f"Loading LoRA adapter: {lora_path}")

        logger.info(f"Loading model with config: n_ctx={n_ctx}, n_gpu_layers={config['n_gpu_layers']}, model_size={model_size_gb:.1f}GB")

        # Probe in subprocess first to detect crashes (SIGABRT/segfault)
        # that would kill the main process.
        # Use a minimal context for the probe — we only need to know if the
        # model can be loaded at all, not whether the full context fits.
        # Returns: True (success), False (crash/signal), None (Python exception)
        # Skipped inside LLM worker processes (probe=False): the worker is
        # already a disposable subprocess, so a nested probe only adds another
        # concurrent model load competing for the GPU.
        probe_config = {**config, 'n_ctx': 512}
        probe_result = LlmFactory._probe_model_load(path, probe_config, logger) if probe else None
        if probe_result is False:
            # Subprocess crashed with a signal — try progressively safer configs
            fallback_configs = []
            if config['n_ctx'] > 2048:
                fallback_configs.append(('reduced context', {**config, 'n_ctx': 2048}))
            if config['n_gpu_layers'] != 0:
                fallback_configs.append(('CPU-only', {**config, 'n_ctx': min(config['n_ctx'], 4096), 'n_gpu_layers': 0}))
            fallback_configs.append(('minimal', {**config, 'n_ctx': 2048, 'n_gpu_layers': 0}))

            probed = False
            last_stderr = ""
            for label, fallback in fallback_configs:
                logger.warning(f"Trying {label} config: n_ctx={fallback['n_ctx']}, n_gpu_layers={fallback['n_gpu_layers']}")
                result = LlmFactory._probe_model_load(path, {**fallback, 'n_ctx': 512}, logger)
                if result is not False:
                    # True = clean success, None = Python exception (safe to try in-process)
                    config = fallback
                    probed = True
                    break
                # Capture stderr for diagnostics on final failure
                last_stderr = LlmFactory._last_probe_stderr

            if not probed:
                diag = f" Last error: {last_stderr}" if last_stderr else ""
                raise RuntimeError(
                    f"Model at {path} cannot be loaded — all configurations "
                    f"crashed. Check GPU drivers and available memory.{diag}"
                )
        # If probe_result is None, the probe raised a Python exception.
        # This won't crash the main process, so proceed and let the
        # in-process Llama() call produce a proper error with full context.

        try:
            llm = Llama(**config)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

            # Fallback 1: reduce context size and retry with GPU
            if config['n_ctx'] > 2048:
                reduced_ctx = max(2048, config['n_ctx'] // 2)
                logger.warning(f"Retrying with reduced context: {reduced_ctx} tokens...")
                config['n_ctx'] = reduced_ctx
                try:
                    llm = Llama(**config)
                except Exception as e2:
                    logger.error(f"Failed to load model with reduced context: {e2}")
                    # Fallback 2: try CPU-only mode with reduced context
                    if config['n_gpu_layers'] != 0:
                        logger.warning("Retrying with CPU-only mode...")
                        config['n_gpu_layers'] = 0
                        llm = Llama(**config)
                    else:
                        raise
            elif config['n_gpu_layers'] != 0:
                # Fallback: try CPU-only mode if GPU loading fails
                logger.warning("Retrying with CPU-only mode...")
                config['n_gpu_layers'] = 0
                llm = Llama(**config)
            else:
                raise

        return LlamaCpp(
            model=llm,
            tokenizer=llm.tokenizer,
            project_id=project_id,
            logger=logger,
        )
