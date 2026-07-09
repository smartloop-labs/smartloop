"""Factory to create LLM wrapper classes"""
import os
import json
import sys
import time
import subprocess
import tempfile
from logging import Logger
from pathlib import Path
from smartloop.utils.device_utils import (
    get_device_config, ensure_openvino_device_env, available_intel_devices,
)


def _has_openvino_backend() -> bool:
    """True when the ggml-openvino backend DLL/SO is present in the llama_cpp lib dir."""
    try:
        import llama_cpp
        lib_dir = Path(llama_cpp.__file__).parent / "lib"
        return any(
            (lib_dir / name).exists()
            for name in ("ggml-openvino.dll", "libggml-openvino.so", "ggml-openvino.so")
        )
    except Exception:
        return False


def _has_cuda_backend() -> bool:
    """True when the ggml-cuda backend DLL/SO is present in the llama_cpp lib dir.

    Guards against the case where nvidia-smi reports a GPU but the llama-cpp
    build was compiled without CUDA support (e.g. only -DGGML_OPENVINO=ON).
    Without this check n_gpu_layers is set non-zero for a missing backend,
    causing ggml graph allocation to fail and llama_decode to return -3.
    """
    try:
        import llama_cpp
        lib_dir = Path(llama_cpp.__file__).parent / "lib"
        return any(
            (lib_dir / name).exists()
            for name in ("ggml-cuda.dll", "libggml-cuda.so", "ggml-cuda.so")
        )
    except Exception:
        return False


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
            # Run one decode to catch backend failures (e.g. OpenVINO\n"
            # scatter_update assertion on Gemma 3) before the main process\n"
            # commits to this config.\n"
            "llm('Hi', max_tokens=1, echo=False)\n"
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
                    # Positive exit code = Python exception.  Treat a decode
                    # backend error (e.g. OpenVINO scatter_update) the same as
                    # a hard crash so the caller can fall back to native CPU.
                    if "llama_decode returned" in LlmFactory._last_probe_stderr:
                        logger.warning(f"Probe decode failed (backend error): {LlmFactory._last_probe_stderr}")
                        return False
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
    # rather than guessed.  ``gemma3`` is deliberately absent: llama-cpp-python
    # has no Gemma3 handler and the legacy clip path aborts on its projector.
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
    def vision_handler_supported(model_path: str) -> bool:
        """True when the installed llama-cpp can run this model's vision input."""
        cls_name = LlmFactory._vision_handler_class_name(model_path)
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
                                  clip_on_cpu: bool = False):
        """Build the architecture-appropriate multimodal chat handler, or None.

        Returns None (caller loads text-only / skips) when the model's
        architecture has no supported handler in the installed llama-cpp — this
        is what prevents a crash on e.g. gemma3.

        ``clip_on_cpu`` forces the CLIP/mmproj encoder onto CPU (see
        ``_force_clip_on_cpu``); the caller sets it only when GPU VRAM is too
        tight to run the projector without an OOM segfault.
        """
        cls_name = LlmFactory._vision_handler_class_name(model_path)
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
    def _is_openvino_ir_dir(path: str) -> bool:
        """True when *path* is a directory holding an OpenVINO IR export."""
        return bool(path) and (Path(path) / "openvino_model.xml").exists()

    @staticmethod
    def _find_openvino_ir_dir(gguf_path: str) -> str | None:
        """Locate an OpenVINO IR export shipped alongside a GGUF model.

        The IR follows the workspace model layout: the GGUF lives in
        ``models/<org>/<model-name>/<model-name>-<quant>.gguf`` and its IR
        export sits beside it, named after the model (not the quantized
        file).  Both an INT8 export (``<model-name>-ov-int8/``) and the
        default export (``<model-name>-ov/``) are recognized, the INT8 one
        taking precedence.  A directory qualifies when it contains
        ``openvino_model.xml``.
        """
        model_dir = Path(gguf_path).parent
        for suffix in ("-ov-int8", "-ov"):
            ir_dir = model_dir / f"{model_dir.name}{suffix}"
            if (ir_dir / "openvino_model.xml").exists():
                return str(ir_dir)
        return None

    @staticmethod
    def _ir_npu_compatible(ir_dir: str) -> bool:
        """False when the IR cannot compile on the NPU, so the load loop can
        skip it without paying a doomed compile attempt.

        Models with active sliding-window attention (e.g. Gemma-3,
        sliding_window=512) bake a windowed mask Slice into the IR that the
        NPU's static-shape SDPA validation rejects ("Attention mask input
        shape not compatible with other inputs").
        """
        try:
            cfg = json.loads((Path(ir_dir) / "config.json").read_text())
        except Exception:
            return True  # unknown — let the NPU compile attempt decide
        if cfg.get("use_sliding_window") is False:
            return True  # declared but disabled (e.g. Qwen2.5)
        return not cfg.get("sliding_window")

    @staticmethod
    def get_openvino_genai_instance(ir_dir: str, device: str, logger: Logger,
                                    project_id: str):
        """Load an OpenVINO IR model with openvino_genai on an Intel GPU/NPU.

        Used instead of llama.cpp on Intel machines: the ggml-openvino
        backend cannot decode several architectures (Gemma-3 included) on
        any device, while GenAI runs INT4 IR exports natively.
        """
        import openvino_genai as ov_genai
        from smartloop.llms.openvino_genai_llm import OpenVinoGenAI

        # Effective context: the model's trained maximum, capped so the
        # KV cache stays reasonable on shared-memory Intel devices.
        max_context = 8192
        try:
            cfg = json.loads((Path(ir_dir) / "config.json").read_text())
            trained = int(cfg.get("max_position_embeddings", max_context))
            max_context = max(2048, min(trained, 8192))
        except Exception:
            pass

        properties = {}
        if device.upper().startswith("GPU"):
            # Cache compiled blobs next to the model to cut reload time.
            properties["CACHE_DIR"] = str(Path(ir_dir) / ".ov_cache")
        elif device.upper().startswith("NPU"):
            # The NPU pipeline compiles for a fixed prompt budget
            # (default 1024) — raise it so real RAG prompts fit.
            properties["MAX_PROMPT_LEN"] = min(max_context, 4096)

        start = time.time()
        pipe = ov_genai.LLMPipeline(ir_dir, device, **properties)
        logger.info(
            f"OpenVINO GenAI loaded {ir_dir} on {device} "
            f"in {time.time() - start:.1f}s (max_context={max_context})"
        )

        # Some devices compile the model but abort on the very first decode —
        # notably the Intel iGPU OpenCL runtime on hybrid NVIDIA+Intel hosts,
        # which throws "clEnqueueMapBuffer ... CL_INVALID_VALUE" the moment
        # inference starts.  Without this check the pipeline is returned as
        # "loaded" and every real request then fails at generation (an empty
        # response), with no chance to fall back.  Run a one-token decode here
        # (the GenAI analogue of _probe_model_load's decode probe) so a device
        # that can't actually generate raises now — the caller's load loop then
        # advances to the next Intel device and ultimately to llama.cpp.
        smoke = ov_genai.GenerationConfig()
        smoke.max_new_tokens = 1
        smoke.do_sample = False
        pipe.generate("Hi", smoke)
        logger.info(f"OpenVINO GenAI decode smoke-test passed on {device}")

        return OpenVinoGenAI(
            model=pipe,
            tokenizer=pipe.get_tokenizer(),
            logger=logger,
            project_id=project_id,
            max_context=max_context,
        )

    @staticmethod
    def load_openvino_ir(ir_dir: str, logger: Logger, project_id: str):
        """Load an OpenVINO IR across the available Intel accelerators.

        Tries each Intel device in preference order (NPU first, then Intel
        GPUs), skipping the NPU for sliding-window models it cannot compile and
        advancing past any device whose compile or first decode fails.  Returns
        the loaded ``OpenVinoGenAI`` instance, or ``None`` when no Intel device
        could load it — the caller decides whether to fall back to llama.cpp
        (GGUF present) or surface the failure (IR-only model).
        """
        for intel_device in available_intel_devices():
            if intel_device.upper() == "CPU":
                continue
            if (intel_device.upper() == "NPU"
                    and not LlmFactory._ir_npu_compatible(ir_dir)):
                logger.info(
                    "Skipping NPU: model uses sliding-window "
                    "attention, which the NPU cannot compile"
                )
                continue
            try:
                return LlmFactory.get_openvino_genai_instance(
                    ir_dir, intel_device, logger, project_id)
            except Exception as e:
                reason = str(e).strip().splitlines()
                logger.warning(
                    f"OpenVINO GenAI load failed on {intel_device} "
                    f"({reason[0] if reason else e}); trying next device"
                )
        if available_intel_devices():
            logger.warning("OpenVINO GenAI unavailable on all Intel devices")
        return None

    @staticmethod
    def get_llama_cpp_instance(
        path: str, logger: Logger, project_id: str, lora_path: str | None = None,
        *, probe: bool = True, max_n_ctx: int | None = None,
        mmproj_path: str | None = None, clip_on_cpu: bool = False,
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
        """
        # Intel machines (no NVIDIA GPU): prefer OpenVINO GenAI when an IR
        # export ships alongside the GGUF — the ggml-openvino llama.cpp
        # backend cannot decode several architectures (Gemma-3 included) on
        # any OpenVINO device.  LoRA and vision loads stay on llama.cpp.
        if not mmproj_path and not lora_path:
            ir_dir = LlmFactory._find_openvino_ir_dir(path)
            if ir_dir:
                instance = LlmFactory.load_openvino_ir(ir_dir, logger, project_id)
                if instance is not None:
                    return instance
                if available_intel_devices():
                    logger.warning(
                        "OpenVINO GenAI unavailable on all Intel devices; "
                        "falling back to llama.cpp"
                    )

        import torch
        from llama_cpp import Llama
        from smartloop.llms.llama_cpp import LlamaCpp

        chat_handler = None
        if mmproj_path:
            chat_handler = LlmFactory._make_vision_chat_handler(
                mmproj_path, path, logger, clip_on_cpu=clip_on_cpu)
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

        # Adjust GPU layers and context based on available memory.
        # Priority: MPS (Mac) → CUDA (NVIDIA) → OpenVINO AUTO (Intel GPU/NPU) → CPU.
        # OpenVINO is placed after CUDA because llama.cpp always selects the CUDA
        # backend for GPU layers when both are compiled in — context must be sized
        # from VRAM, not system RAM, to avoid KV-cache allocation failures (-3).
        # OpenVINO AUTO is the right choice when CUDA is absent (Intel iGPU/NPU).
        vram_gb = _get_cuda_vram_gb()
        # Extra Llama() kwargs a branch needs to pin device placement.
        placement_kwargs: dict = {}

        if torch.backends.mps.is_available():
            # Apple Silicon — unified memory shared between CPU and GPU.
            import psutil
            total_mem_gb = psutil.virtual_memory().total / (1024**3)
            available_mem_gb = psutil.virtual_memory().available / (1024**3)
            available_for_kv = available_mem_gb - model_size_gb - 2.0
            if available_for_kv > 0:
                kv_per_8k = 0.75 if model_size_gb < 3 else 1.5 if model_size_gb < 6 else 2.0
                estimated_ctx = int((available_for_kv / kv_per_8k) * 8192)
                max_ctx = 131072 if total_mem_gb >= 32 else 65536 if total_mem_gb >= 16 else 32768
                n_ctx = max(4096, min(estimated_ctx, max_ctx))
            else:
                n_ctx = 4096
            logger.info(
                f"Apple Silicon: {total_mem_gb:.0f}GB total, {available_mem_gb:.1f}GB available, "
                f"{model_size_gb:.1f}GB model — using n_ctx={n_ctx}, n_gpu_layers={llama_config.n_gpu_layers}"
            )

        elif vram_gb is not None and _has_cuda_backend():
            # NVIDIA CUDA — context sized from VRAM so the KV cache fits.
            available_for_kv = vram_gb - model_size_gb - 0.5  # 0.5 GB overhead
            if available_for_kv > 0:
                kv_per_8k = 0.25 if model_size_gb < 3 else 0.5 if model_size_gb < 6 else 1.0
                estimated_ctx = int((available_for_kv / kv_per_8k) * 8192)
                max_ctx = 32768 if model_size_gb < 3 else 16384 if model_size_gb < 6 else 8192
                n_ctx = max(2048, min(estimated_ctx, max_ctx))
                logger.info(f"CUDA: {vram_gb:.1f}GB VRAM, {model_size_gb:.1f}GB model — n_ctx={n_ctx}")
            else:
                n_ctx = 2048
                logger.warning(f"Limited VRAM ({vram_gb:.1f}GB) for {model_size_gb:.1f}GB model, using context={n_ctx}")
            if model_size_gb > vram_gb * 0.8:
                estimated_layers = int((vram_gb * 0.7) / model_size_gb * 35)
                llama_config.n_gpu_layers = max(0, estimated_layers)
                logger.warning(
                    f"Model size ({model_size_gb:.1f}GB) requires conservative GPU allocation. "
                    f"Using {llama_config.n_gpu_layers} GPU layers (rest on CPU)"
                )
            if _has_openvino_backend():
                # Combined CUDA+OpenVINO build: the scheduler otherwise splits
                # layers between CUDA0 and OPENVINO0 (which advertises system
                # RAM as free device memory) and llama_decode fails with -3.
                # Pin every offloaded layer to CUDA0.
                import llama_cpp as _llama_cpp
                placement_kwargs = {
                    'split_mode': getattr(_llama_cpp, 'LLAMA_SPLIT_MODE_NONE', 0),
                    'main_gpu': 0,
                }
                logger.info("Dual CUDA+OpenVINO build — pinning all layers to CUDA0 (split_mode=NONE)")

        elif _has_openvino_backend():
            # No CUDA — OpenVINO backend present.
            # Only use OpenVINO when real hardware acceleration (iGPU/NPU) is
            # available.  OpenVINO's intel_cpu plugin has a scatter_update
            # assertion bug with Gemma 3 (and potentially other architectures).
            # If OpenVINO only exposes CPU, fall through to llama.cpp's native
            # CPU backend which handles Gemma 3 correctly.
            # The device (plain GPU/NPU — numbered names are ignored by the
            # backend) was exported before the first llama_cpp backend init
            # by DeviceConfig._detect_device(); calling again just reads it.
            ov_device = ensure_openvino_device_env()

            if ov_device and ov_device.upper() != "CPU":
                llama_config.n_gpu_layers = -1  # offload all layers to iGPU/NPU via OpenVINO

                import psutil
                avail_gb = psutil.virtual_memory().available / (1024**3)
                available_for_kv = avail_gb - model_size_gb - 1.0
                if available_for_kv > 0:
                    kv_per_8k = 0.5 if model_size_gb < 3 else 1.0 if model_size_gb < 6 else 1.5
                    n_ctx = max(2048, min(int(available_for_kv / kv_per_8k * 8192), 32768))
                else:
                    n_ctx = 2048
                logger.info(
                    f"OpenVINO backend — GGML_OPENVINO_DEVICE={ov_device}, "
                    f"n_ctx={n_ctx}, n_gpu_layers={llama_config.n_gpu_layers}"
                )
            else:
                # OpenVINO exposes no Intel GPU/NPU (or the openvino python
                # package is missing) — its intel_cpu plugin has known bugs;
                # use llama.cpp native CPU backend instead.
                llama_config.n_gpu_layers = 0
                logger.info(
                    f"OpenVINO without usable GPU/NPU (device={ov_device}) — "
                    f"using native CPU backend to avoid intel_cpu plugin bugs"
                )

        else:
            # No GPU of any kind — pure CPU.
            llama_config.n_gpu_layers = 0
            logger.info(f"No GPU available, offloading entire model to CPU ({os.cpu_count() or 4} threads)")

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
            **llama_config.model_dump(),
            **placement_kwargs,
        }

        if lora_path:
            config['lora_path'] = lora_path
            logger.info(f"Loading LoRA adapter: {lora_path}")

        if chat_handler is not None:
            config['chat_handler'] = chat_handler
            # Flash attention — VISION LOAD, CUDA ONLY.  Without it the image
            # encode allocates a fixed ~6GB compute buffer that OOM-segfaults
            # the native clip code on a tight CUDA GPU; flash_attn brings the
            # per-page peak to ~3GB.  On Metal (MPS) flash_attn is unsupported
            # and causes the load to fail, silently falling back to CPU-only.
            if torch.cuda.is_available():
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
