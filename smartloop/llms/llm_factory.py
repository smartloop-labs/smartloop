"""Factory to create LLM wrapper classes"""
import inspect
import os
import json
import sys
import subprocess
import tempfile
from logging import Logger
from pathlib import Path
from smartloop.utils.device_utils import get_device_config


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

        Also the only thing standing between a GPU-init hang and a project
        agent that never comes up: nothing here bounds ``Llama(**config)`` in
        the main process, so a driver that hangs instead of failing (Vulkan on
        Windows has done this on some Intel iGPU drivers) blocks forever with
        no fallback. ``subprocess.run``'s ``timeout`` kills a probe that hangs
        this way, which is what turns "stuck loading the model forever" into
        "falls back to CPU and starts".  A frozen (PyInstaller) build used to
        skip this probe entirely — ``sys.executable`` there is the app binary,
        not a Python interpreter, so ``-c <script>`` could not run — but the
        binary already knows how to re-exec ``-m <module>`` (``main.py``, used
        to spawn the project agent and model host the same way), so the probe
        runs as a module for exactly the same reason those do, and a frozen
        Windows build gets the same protection as a dev one.
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "smartloop.llms.model_probe", json.dumps(config)],
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
            # MTMDChatHandler takes ``use_gpu`` as a constructor argument, which
            # scopes the choice to THIS handler; the module-level patch is the
            # fallback for the older Llava15-derived handlers, which hardcode
            # ``use_gpu = True``.  Prefer the argument: the patch is global and
            # permanent, so one tight-VRAM vision load would otherwise pin every
            # later projector in the process to CPU.
            if clip_on_cpu:
                if "use_gpu" in inspect.signature(handler_cls.__init__).parameters:
                    return handler_cls(clip_model_path=mmproj_path, verbose=False,
                                       use_gpu=False)
                LlmFactory._force_clip_on_cpu(logger)
            return handler_cls(clip_model_path=mmproj_path, verbose=False)
        except Exception as e:
            logger.warning(f"Failed to init vision handler {cls_name} ({e}); loading text-only")
            return None

    @staticmethod
    def supports_thinking(model_name: str | None) -> bool:
        """True when ``model_name`` declares ``ModelCapability.THINKING``.

        That capability is the single source of truth for "this model's chat
        template reads ``enable_thinking``" — it is what the model picker shows
        and what gates the per-request ``thinking`` flag.  Treated as False for
        a model outside ``SUPPORTED_MODELS`` (user-added) and for one whose
        class declares no capabilities at all (the bootstrap model): binding
        the flag onto a template that ignores it does nothing.
        """
        if not model_name:
            return False
        try:
            from smartloop.config.supported_models import SUPPORTED_MODELS
            from smartloop.types import ModelCapability
            model_cls = SUPPORTED_MODELS.get(model_name)
        except Exception:
            return False
        # Not every model class declares capabilities — SLPModel has no
        # default, and callers guard with hasattr throughout (see
        # WorkspaceModel.effective_capabilities).
        declared = getattr(model_cls, "default_capabilities", None)
        if not callable(declared):
            return False
        try:
            return ModelCapability.THINKING in declared()
        except Exception:
            return False

    @staticmethod
    def _batch_and_threads(n_gpu_layers: int, logger: Logger) -> tuple[int, int]:
        """``(n_batch, n_threads)`` for this load.

        Both were fixed at 512 and 4 "for memory efficiency", and both are paid
        on every prompt.  ``n_batch`` is the prefill batch: a 3k-token prompt —
        a tool catalog, a document — went through it six times where a GPU can
        take it in two, and the buffer it grows is tens of megabytes, not
        gigabytes.  ``n_threads`` only drives the layers that are *not*
        offloaded, so 4 costs little on a GPU load and most of the machine on a
        CPU one.

        Overridable with ``SLP_LLM_N_BATCH`` / ``SLP_LLM_N_THREADS`` for a
        device where the defaults are wrong.
        """
        offloaded = n_gpu_layers != 0
        n_batch, n_threads = (1024, 4) if offloaded else (512, os.cpu_count() or 4)
        try:
            from smartloop.config.app_settings import AppSettings
            settings = AppSettings()
            n_batch = settings.llm_n_batch or n_batch
            n_threads = settings.llm_n_threads or n_threads
        except Exception:
            pass
        logger.info(
            "[model] n_batch=%d, n_threads=%d (n_gpu_layers=%d)",
            n_batch, n_threads, n_gpu_layers,
        )
        return n_batch, n_threads

    @staticmethod
    def _template_reads_thinking(llm) -> bool:
        """True when the loaded GGUF's own chat template can reason.

        ``supports_thinking`` answers from ``SUPPORTED_MODELS``, which knows
        nothing about a model the user built or merged themselves — and those
        are exactly the models this misses, since a merge of a reasoning base
        keeps its template.  The template is the ground truth: it is what reads
        ``enable_thinking`` when the handler passes it.
        """
        try:
            template = (llm.metadata or {}).get("tokenizer.chat_template", "")
        except Exception:
            return False
        return "enable_thinking" in str(template)

    @staticmethod
    def thinking_for(model_name: str | None, override: bool | None = None) -> bool | None:
        """Whether ``model_name`` should reason, or ``None`` to leave it alone.

        ``None`` means the model does not support reasoning, so nothing is
        bound.  Otherwise the answer is, most specific first: the caller's
        ``override`` (the request's ``thinking`` field), then
        ``slp_thinking_enabled``, then the model class's
        ``thinking_default()``.
        """
        if not LlmFactory.supports_thinking(model_name):
            return None
        if override is not None:
            return override
        from smartloop.config.app_settings import AppSettings
        setting = AppSettings().thinking_enabled
        if setting is not None:
            return setting
        from smartloop.config.supported_models import SUPPORTED_MODELS
        model_cls = SUPPORTED_MODELS.get(model_name)
        default = getattr(model_cls, "thinking_default", None)
        return bool(default()) if callable(default) else False

    @staticmethod
    def _bind_thinking(llm, enabled: bool, logger: Logger) -> None:
        """Render every chat completion with ``enable_thinking=enabled``.

        Hybrid-reasoning models read that variable from their own chat template
        to decide whether to open a ``<think>`` block.  ``Llama.create_chat_completion``
        has no ``**kwargs`` to forward it through, so the flag is bound onto
        the resolved chat handler once at load time — handlers do take
        ``**kwargs`` and pass them straight to the template render.  Binding it
        on the ``Llama`` object rather than at each call site means the direct
        ``create_chat_completion`` callers (vision Q&A, OCR) are covered too.

        The bound value is read from ``Llama._slp_thinking`` on every call
        rather than baked in, so a caller can turn reasoning off for the
        duration of one completion — see ``LlamaCpp._thinking_for_budget``.

        Best-effort: a llama-cpp that resolves handlers differently leaves the
        model loaded with its template default rather than failing the load.
        """
        try:
            import llama_cpp.llama_chat_format as llama_chat_format
            # Same resolution order as Llama.create_chat_completion.
            handler = (
                llm.chat_handler
                or llm._chat_handlers.get(llm.chat_format)
                or llama_chat_format.get_chat_completion_handler(llm.chat_format)
            )

            def _with_thinking(**kwargs):
                # Read the flag off the Llama the call names, not a captured
                # reference — the handler must not keep the model alive.
                target = kwargs.get("llama")
                kwargs.setdefault(
                    "enable_thinking", getattr(target, "_slp_thinking", enabled)
                )
                return handler(**kwargs)

            llm._slp_thinking = enabled
            llm.chat_handler = _with_thinking
            logger.info(f"[thinking] enable_thinking={enabled} bound to chat template")
        except Exception as e:
            logger.warning(f"[thinking] Could not set enable_thinking={enabled}: {e}")

    @staticmethod
    def get_llama_cpp_instance(
        path: str, logger: Logger, project_id: str, lora_path: str | None = None,
        *, probe: bool = True, max_n_ctx: int | None = None,
        mmproj_path: str | None = None, clip_on_cpu: bool = False,
        vision_handler: str | None = None, thinking: bool | None = None,
        model_name: str | None = None,
    ):
        """Returns an instance of the configured LlamaCpp wrapper class.

        ``probe``: when True (default) the model is first test-loaded in a
        throwaway subprocess so a hard crash (SIGABRT/segfault) can't take
        down the caller.  Set False only from a caller that is itself
        disposable, where a nested probe would just add another concurrent
        model load.

        ``max_n_ctx``: upper bound for the auto-sized context window.  The
        context auto-sizing otherwise assumes the model is alone on the device
        and reserves nearly all VRAM for it, which starves the on-demand
        large/vision models.

        ``mmproj_path``: path to a CLIP/mmproj GGUF.  When set, the model is
        loaded as a multimodal (vision) instance with a vision chat handler so
        it can describe / OCR images.  The subprocess probe is skipped (the
        handler object isn't JSON-serializable) and the load is forced
        in-process.

        ``vision_handler``: chat-handler class name declared by the model
        class (``vision_chat_handler()``); authoritative when set, otherwise
        the handler is inferred from the GGUF architecture.

        ``thinking``: turn the model's ``<think>`` reasoning block on/off (see
        ``SLPModel.thinking_default()`` and ``LlmFactory.thinking_for``);
        ``None`` keeps the chat template's own default.  The block itself is
        stripped from replies by :mod:`smartloop.llms.thinking`.
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

        # Adjust GPU layers and context based on available memory. VRAM is not
        # probed: GPU offload is decided by llama.cpp and device memory is
        # managed by the Vulkan backend, so discrete-GPU boxes keep full offload
        # with the default context (the subprocess probe below reduces the
        # context or falls back to CPU if it doesn't fit).
        if device_config.device.value == 'mps':
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
            # GPU offload via the Vulkan (or other) llama.cpp backend. VRAM is
            # not queryable, so keep the full offload from get_llama_cpp_config()
            # with the default context; the subprocess probe below degrades to
            # reduced-context / CPU if it doesn't fit.
            logger.info(
                f"GPU offload backend available (VRAM managed by the backend) — "
                f"using n_ctx={n_ctx}, n_gpu_layers={llama_config.n_gpu_layers}"
            )
        else:
            # No GPU available — CPU only
            llama_config.n_gpu_layers = 0
            n_threads = os.cpu_count() or 4
            logger.info(f"No GPU available, offloading entire model to CPU with {n_threads} threads")

        # Cap the auto-sized context for worker copies so multiple model
        # instances (each with its own KV cache) fit on a single GPU.  A cap
        # of 0/None means no cap — the model uses its full auto-sized window.
        if max_n_ctx not in (None, 0) and n_ctx > max_n_ctx:
            logger.info(
                f"Capping context {n_ctx} → {max_n_ctx} (shared-GPU worker copy)"
            )
            n_ctx = max_n_ctx

        n_batch, n_threads = LlmFactory._batch_and_threads(
            llama_config.n_gpu_layers, logger,
        )
        config = {
            'model_path': path,
            'n_ctx': n_ctx,
            'n_batch': n_batch,
            'n_threads': n_threads,
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
        # Skipped when probe=False: a caller that is already a disposable
        # subprocess gains nothing from a nested probe, which would only add
        # another concurrent model load competing for the GPU.
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

        if thinking is None:
            # The registry could not answer — a user-built or merged model is
            # not in SUPPORTED_MODELS, so nothing declares THINKING for it.
            # The GGUF itself can: its chat template either reads
            # ``enable_thinking`` or it doesn't.  Bind off in that case, which
            # keeps the visible answer as it is today and still leaves the flag
            # present, so a caller that wants reasoning for its own call can
            # turn it on (see ``MCPAgent._thinking_on``).
            if LlmFactory._template_reads_thinking(llm):
                logger.info(
                    "[thinking] chat template reads enable_thinking — binding "
                    "the flag for a model the registry does not describe"
                )
                from smartloop.config.app_settings import AppSettings
                thinking = bool(AppSettings().thinking_enabled)

        if thinking is not None:
            LlmFactory._bind_thinking(llm, thinking, logger)

        return LlamaCpp(
            model=llm,
            tokenizer=llm.tokenizer,
            project_id=project_id,
            logger=logger,
            model_name=model_name,
        )
