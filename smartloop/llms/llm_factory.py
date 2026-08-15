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
    """Return total VRAM in GB for NVIDIA GPU 0, or None if unavailable.

    GPU availability is decided by llama.cpp and the backend is Vulkan, so
    torch's CUDA view is irrelevant (and torch isn't installed by default).
    Query nvidia-smi directly for the physical GPU's VRAM.
    """
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


def _llama_cpp_has_cuda_backend() -> bool:
    """True when the installed llama.cpp was built with the CUDA backend.

    ``torch.cuda.is_available()`` reflects *torch's* build, not llama.cpp's —
    a Vulkan-only llama.cpp next to a CUDA torch wheel must not get
    CUDA-specific load options.  The ggml CUDA backend ships as its own
    shared library next to the llama library, so its presence is the flag.
    """
    try:
        import llama_cpp
        lib_dir = Path(llama_cpp.__file__).parent / "lib"
        return any(lib_dir.glob("*ggml-cuda*"))
    except Exception:
        return False


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

    # Map a model's GGUF ``general.architecture`` to the llama-cpp vision chat
    # handler that knows its image-token / prompt format.  Using the WRONG
    # handler (or none) for a model crashes the native clip code, so an arch
    # without an entry here is treated as unsupported (text-only / skipped)
    # rather than guessed.  ``gemma3``/``gemma4`` are absent here because their
    # model classes declare the handler explicitly via ``vision_chat_handler()``
    # (the generic MTMDChatHandler / its Gemma4ChatHandler subclass), which is
    # authoritative and bypasses this arch-sniffing fallback.
    _VISION_HANDLER_BY_ARCH = [
        (("qwen2vl", "qwen2.5vl", "qwen2_5vl", "qwen3vl", "qwenvl"), "Qwen25VLChatHandler"),
        (("minicpmv", "minicpm-v", "minicpm"), "MiniCPMv26ChatHandler"),
        (("nanollava",), "NanoLlavaChatHandler"),
        (("moondream",), "MoondreamChatHandler"),
        (("llava",), "Llava16ChatHandler"),
    ]

    @staticmethod
    def _read_gguf_arch(path: str) -> str | None:
        """Return a model GGUF's ``general.architecture`` (e.g. 'qwen2vl')."""
        try:
            import gguf
            reader = gguf.GGUFReader(path)
            field = reader.fields.get("general.architecture")
            if field:
                return bytes(field.parts[field.data[0]]).decode("utf-8", "ignore")
        except Exception:
            pass
        return None

    @staticmethod
    def _vision_handler_class_name(model_path: str) -> str | None:
        """Name of the supported vision handler for this model, or None."""
        arch = (LlmFactory._read_gguf_arch(model_path) or "").lower()
        for keys, cls_name in LlmFactory._VISION_HANDLER_BY_ARCH:
            if any(k in arch for k in keys):
                return cls_name
        return None

    @staticmethod
    def vision_handler_supported(model_path: str, handler_name: str | None = None) -> bool:
        """True when the installed llama-cpp can run this model's vision input.

        ``handler_name`` is the handler the model class declares
        (``vision_chat_handler()``); when given it is authoritative and the
        GGUF-arch guess is skipped. Falls back to arch sniffing for models
        that don't declare one.
        """
        cls_name = handler_name or LlmFactory._vision_handler_class_name(model_path)
        if not cls_name:
            return False
        try:
            import llama_cpp.llama_chat_format as chf
            return getattr(chf, cls_name, None) is not None
        except Exception:
            return False

    @staticmethod
    def _force_clip_on_cpu(logger: Logger) -> None:
        """Make the multimodal (CLIP/mmproj) encoder load on CPU.

        llama-cpp-python's vision handler hardcodes ``ctx_params.use_gpu = True``
        for the mmproj context (``_init_mtmd_context``), so the projector and its
        image activations always land on the GPU.  On a VRAM-constrained device
        that OOMs *inside the native clip encode* — an uncatchable SIGSEGV that
        core-dumps the server (the subprocess crash-probe is skipped for vision
        models).  We patch ``mtmd_init_from_file`` to flip ``use_gpu`` off so the
        projector runs on CPU; the LLM itself still runs on GPU, and encoding one
        page image on CPU is fine for batch OCR.  Idempotent and version-tolerant
        (depends only on the call signature + the ``use_gpu`` field).
        """
        try:
            import llama_cpp.mtmd_cpp as mtmd_cpp
        except Exception:
            return
        if getattr(mtmd_cpp, "_slp_clip_cpu", False):
            return
        _orig = mtmd_cpp.mtmd_init_from_file

        def _cpu_init(clip_path, model, ctx_params):
            try:
                ctx_params.use_gpu = False
            except Exception:
                pass
            return _orig(clip_path, model, ctx_params)

        mtmd_cpp.mtmd_init_from_file = _cpu_init
        mtmd_cpp._slp_clip_cpu = True
        logger.info("[vision] CLIP/mmproj forced to CPU (avoids GPU OOM in clip encode)")

    @staticmethod
    def _make_vision_chat_handler(mmproj_path: str, model_path: str, logger: Logger,
                                  clip_on_cpu: bool = False,
                                  handler_name: str | None = None):
        """Build the multimodal chat handler for this model, or None.

        ``handler_name`` is the handler declared by the model class
        (``vision_chat_handler()``) and is authoritative when provided;
        otherwise the handler is inferred from the GGUF architecture. Returns
        None (caller loads text-only / skips) when neither yields a handler
        available in the installed llama-cpp — this is what prevents a crash
        on e.g. gemma3.

        ``clip_on_cpu`` forces the CLIP/mmproj encoder onto CPU (see
        ``_force_clip_on_cpu``); the caller sets it only when GPU VRAM is too
        tight to run the projector without an OOM segfault.
        """
        cls_name = handler_name or LlmFactory._vision_handler_class_name(model_path)
        if not cls_name:
            arch = LlmFactory._read_gguf_arch(model_path)
            logger.warning(
                f"No supported vision chat handler for architecture {arch!r}; "
                f"loading text-only (image OCR disabled for this model)"
            )
            return None
        try:
            import llama_cpp.llama_chat_format as chf
            handler_cls = getattr(chf, cls_name, None)
            if handler_cls is None:
                logger.warning(
                    f"Vision handler {cls_name} not in installed llama-cpp; loading text-only"
                )
                return None
            # Keep the projector off the GPU only when asked — its image-encode
            # pass OOM-segfaults the (uncatchable) native clip code on a
            # tight-VRAM device.  With enough VRAM it stays on GPU (faster).
            if clip_on_cpu:
                LlmFactory._force_clip_on_cpu(logger)
            return handler_cls(clip_model_path=mmproj_path, verbose=False)
        except Exception as e:
            logger.warning(f"Failed to init vision handler {cls_name} ({e}); loading text-only")
            return None

    @staticmethod
    def get_llama_cpp_instance(
        path: str, logger: Logger, project_id: str, lora_path: str | None = None,
        *, probe: bool = True, max_n_ctx: int | None = None,
        mmproj_path: str | None = None, clip_on_cpu: bool = False,
        vision_handler: str | None = None,
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

        ``mmproj_path``: path to a CLIP/mmproj GGUF.  When set, the model is
        loaded as a multimodal (vision) instance with a vision chat handler so
        it can describe / OCR images.  The subprocess probe is skipped (the
        handler object isn't JSON-serializable) and the load is forced
        in-process.

        ``vision_handler``: chat-handler class name declared by the model
        class (``vision_chat_handler()``); authoritative when set, otherwise
        the handler is inferred from the GGUF architecture.
        """
        from llama_cpp import Llama, llama_supports_gpu_offload
        from smartloop.llms.llama_cpp import LlamaCpp

        chat_handler = None
        if mmproj_path:
            chat_handler = LlmFactory._make_vision_chat_handler(
                mmproj_path, path, logger, clip_on_cpu=clip_on_cpu,
                handler_name=vision_handler)
            if chat_handler is not None:
                # Handler holds a loaded CLIP model; it can't be pickled into
                # the probe subprocess, so load directly in-process.
                probe = False

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
        elif device_config.device.value == 'mps':
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
        elif llama_supports_gpu_offload():
            # GPU backend torch can't see — e.g. Vulkan on an AMD/Intel GPU,
            # or an NVIDIA card with a CPU-only torch wheel and no nvidia-smi.
            # Neither torch nor the llama-cpp-python bindings can report this
            # device's VRAM, so keep the full offload from
            # get_llama_cpp_config() with the default context; the subprocess
            # probe below degrades to reduced-context / CPU if it doesn't fit.
            logger.info(
                f"GPU offload backend available but VRAM unknown (non-CUDA backend, "
                f"e.g. Vulkan) — using n_ctx={n_ctx}, n_gpu_layers={llama_config.n_gpu_layers}"
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

        if chat_handler is not None:
            config['chat_handler'] = chat_handler
            # Flash attention — VISION LOAD, CUDA ONLY.  Without it the image
            # encode allocates a fixed ~6GB compute buffer that OOM-segfaults
            # the native clip code on a tight CUDA GPU; flash_attn brings the
            # per-page peak to ~3GB.  On Metal (MPS) and Vulkan flash_attn is
            # unsupported and causes the load to fail, silently falling back
            # to CPU-only — so gate on llama.cpp's compiled backend, not
            # torch's (a Vulkan llama.cpp can sit next to a CUDA torch wheel).
            if _llama_cpp_has_cuda_backend():
                config['flash_attn'] = True
            logger.info(
                f"Loading vision model with mmproj: {mmproj_path} "
                f"(flash_attn={config.get('flash_attn', False)})"
            )

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
            logger.error(f"Failed to load model (n_gpu_layers={config['n_gpu_layers']}): {e}")

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
                        logger.warning("FALLING BACK TO CPU-ONLY (n_gpu_layers 0) — GPU load failed")
                        config['n_gpu_layers'] = 0
                        llm = Llama(**config)
                    else:
                        raise
            elif config['n_gpu_layers'] != 0:
                # Fallback: try CPU-only mode if GPU loading fails
                logger.warning("FALLING BACK TO CPU-ONLY (n_gpu_layers 0) — GPU load failed")
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
