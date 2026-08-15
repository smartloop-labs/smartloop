"""LLMExecutor — threads for I/O, processes for LLM.

Dual-pool resource manager that dispatches I/O-bound work (search, MCP
calls) to threads and LLM inference to separate processes (each with its
own model instance).
"""
import logging
import os
import subprocess
import sys
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from smartloop.agents.llm_worker import init_worker, run_inference

logger = logging.getLogger(__name__)


def _llama_gpu_offload() -> bool:
    """True when the installed llama.cpp was built with a GPU backend."""
    try:
        from llama_cpp import llama_supports_gpu_offload
        return bool(llama_supports_gpu_offload())
    except Exception:
        return False


def _nvidia_smi_gpus() -> list:
    """Return ``[(name, free_gb, total_gb), ...]`` for NVIDIA GPUs, or ``[]``.

    GPU availability is decided by llama.cpp; the backend is Vulkan, so torch's
    CUDA view is irrelevant (and torch isn't installed by default). nvidia-smi
    reports the physical NVIDIA GPU(s) directly. Non-NVIDIA Vulkan GPUs
    (AMD/Intel) have no nvidia-smi → the caller falls back to RAM sizing.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                name, free_mib, total_mib = parts
                gpus.append((name, float(free_mib) / 1024, float(total_mib) / 1024))
        return gpus
    except Exception:
        return []


def _gpu_device_cap() -> int | None:
    """Max concurrent GPU model copies (one per GPU), or None for CPU-only.

    One copy per physical NVIDIA GPU; any other llama.cpp GPU backend
    (Vulkan on AMD/Intel, Metal) caps at a single device.
    """
    gpus = _nvidia_smi_gpus()
    if gpus:
        return len(gpus)
    return 1 if _llama_gpu_offload() else None


def _cuda_mem_info_gb():
    """Return ``(name, free_gb, total_gb)`` for NVIDIA GPU 0, or None."""
    gpus = _nvidia_smi_gpus()
    return gpus[0] if gpus else None


def _is_mps() -> bool:
    """True on Apple Silicon with a llama.cpp Metal (GPU offload) backend."""
    return sys.platform == "darwin" and _llama_gpu_offload()


class LLMExecutor:
    """Dual-pool executor: threads for I/O, processes for LLM inference.

    Parameters
    ----------
    max_io_workers:
        Size of the thread pool used for search, MCP calls, and other
        I/O-bound work.
    n_llm_workers:
        Number of worker processes for LLM inference.  Each process loads
        its own copy of the model, so RAM/VRAM usage scales linearly.
        Set to 0 (default) to disable — ``submit_llm`` falls back to the
        thread pool, which serializes LLM calls as before.
    model_path:
        Path to the GGUF model file.  Required when *n_llm_workers* > 0.
    project_id:
        Project identifier passed to the worker initializer.
    """

    def __init__(
        self,
        max_io_workers: int = 6,
        n_llm_workers: int = 0,
        model_path: str | None = None,
        project_id: str | None = None,
        log_queue=None,
    ):
        self._io_pool = ThreadPoolExecutor(max_workers=max_io_workers, thread_name_prefix="io")
        self._llm_pool: ProcessPoolExecutor | None = None

        # When the parent process has set up centralized logging, hand the
        # shared queue to spawn workers so their records flow back to the
        # listener.  Fall back to looking it up from logging_setup so callers
        # don't have to thread it through.
        if log_queue is None:
            try:
                from smartloop.utils.logging_setup import get_log_queue
                log_queue = get_log_queue()
            except Exception:
                log_queue = None
        self._log_queue = log_queue

        if n_llm_workers > 0 and model_path:
            # ``n_llm_workers`` is the on/off switch + preferred count; the
            # actual number of worker copies is sized dynamically to what fits
            # in free VRAM (see _resolve_worker_count) so we neither crash by
            # over-allocating nor leave the GPU underutilized.
            effective_workers = self._resolve_worker_count(model_path, n_llm_workers)

            if effective_workers <= 1:
                # A single worker copy is just a duplicate of the already-loaded
                # main model on the same device.  On one GPU those copies don't
                # run concurrently anyway (the driver serializes kernels across
                # processes without CUDA MPS), so an extra copy only multiplies
                # weights + KV memory and load time for no throughput gain.
                # Skip the pool and reuse the shared model — concurrency still
                # comes from n_parallel continuous batching within that one
                # instance, and ``submit_llm`` falls back to it.
                logger.info(
                    "Skipping LLM worker pool — reusing the shared model "
                    "(1 copy is enough on this device; extra copies serialize "
                    "on a single GPU). Concurrency comes from n_parallel batching."
                )
                return

            # Use the "spawn" start method, not the Linux default "fork".
            # The parent process has already initialized CUDA (torch), and a
            # forked child cannot re-initialize CUDA ("Cannot re-initialize
            # CUDA in forked subprocess"), which crashes the worker. Spawn
            # starts each worker as a fresh process with no inherited CUDA
            # state. (On macOS/Windows spawn is already the default.)
            mp_context = multiprocessing.get_context("spawn")
            logger.info(
                f"Creating ProcessPoolExecutor with {effective_workers} LLM worker(s) "
                f"(model={model_path}, start_method=spawn)"
            )
            self._llm_pool = ProcessPoolExecutor(
                max_workers=effective_workers,
                mp_context=mp_context,
                initializer=init_worker,
                initargs=(model_path, project_id or "", self._log_queue),
            )

    @staticmethod
    def _resolve_worker_count(model_path: str, requested: int) -> int:
        """Return how many LLM worker copies to spawn, sized to available memory.

        Each worker holds its own model copy (weights + a KV cache that scales
        with ``slp_llm_worker_n_ctx``).  ``requested`` is the upper bound; the
        count is then scaled *down* to what fits in the live free memory of the
        active device — CUDA VRAM, Apple MPS unified memory, or CPU RAM — so
        workers don't crash on load.  Extra free memory never raises the count
        above ``requested``.  The estimate is deliberately conservative since
        under-counting a worker's footprint crashes the last copy to load.

        On GPUs the count is also capped at one copy per GPU: separate
        processes can't run GPU kernels concurrently on the same device (the
        driver serializes them without CUDA MPS), so the parallel ceiling is
        the number of physical NVIDIA GPUs (nvidia-smi), or 1 for any other
        llama.cpp GPU backend (Vulkan, Metal).  A single GPU therefore caps at
        1 and the caller reuses the shared model.  CPU has no such cap —
        independent cores give real parallelism, sized by memory below.
        """
        try:
            # One model copy per GPU is the most that can run in parallel;
            # same-GPU copies serialize.  A non-NVIDIA GPU backend (Vulkan on
            # AMD/Intel, Metal) isn't enumerable, so assume one device.  CPU is
            # uncapped here (cores parallelize) → memory is the only limit.
            gpu_cap = _gpu_device_cap()
            if gpu_cap is not None and gpu_cap < requested:
                logger.info(
                    "[device] %d GPU(s) — capping worker copies at one per GPU "
                    "(%d); same-GPU copies serialize. Concurrency within a GPU "
                    "comes from n_parallel continuous batching.",
                    gpu_cap, gpu_cap,
                )
                requested = gpu_cap

            model_gb = os.path.getsize(model_path) / (1024 ** 3)

            from smartloop.config.app_settings import AppSettings
            _settings = AppSettings()
            worker_ctx = _settings.llm_worker_n_ctx
            # Per-copy footprint: weights + KV cache + per-process overhead.
            # The estimate is deliberately conservative — under-counting any
            # term makes ``fits`` round up to a worker that then fails
            # ``llama_context`` allocation.  Overhead must cover the CUDA
            # context that every spawned process allocates (~0.6GB before any
            # weights) plus compute/scratch buffers, so we budget 0.7GB.  The
            # KV estimate scales with worker_ctx (~0.5GB / 4k tokens).
            kv_gb = 0.5 * (worker_ctx / 4096)
            per_copy_gb = model_gb + kv_gb + 0.7

            # The main model is the dominant consumer (weights + a KV cache
            # sized to ``llm_main_n_ctx_cap``), and a flat reserve badly
            # under-counts it — that's what let this over-provision workers and
            # OOM the last copy.  Estimate its footprint the same way as a copy.
            main_kv_gb = 0.5 * (_settings.llm_main_n_ctx_cap / 4096)
            main_footprint_gb = model_gb + main_kv_gb + 0.7

            # Resolve (device_label, free_gb, total_gb, reserve_gb) for whichever
            # accelerator is active.  reserve keeps memory for the main model,
            # fragmentation headroom and the OS so workers don't claim
            # everything and crash the last copy that tries to load.
            cuda_info = _cuda_mem_info_gb()
            if cuda_info is not None:
                # mem_get_info()/nvidia-smi is a single snapshot taken before any
                # worker loads; workers spawn concurrently, so keep a VRAM reserve
                # (was 0.0) to absorb that drift + allocator fragmentation.
                name, free_gb, total_gb = cuda_info
                device, reserve_gb = name, 1.5
            elif _is_mps():
                # Apple Silicon: unified memory shared with the OS/main model.
                import psutil
                vm = psutil.virtual_memory()
                device, total_gb = "Apple MPS (unified)", vm.total / 1024**3
                free_gb, reserve_gb = vm.available / 1024**3, 3.0  # leave headroom
            else:
                import psutil
                vm = psutil.virtual_memory()
                device, total_gb = "CPU", vm.total / 1024**3
                free_gb, reserve_gb = vm.available / 1024**3, 2.0

            # Charge for the main model only to the extent it isn't already
            # resident.  ``mem_get_info`` is timing-dependent: if the executor
            # is built *after* the model loads, ``free_gb`` already excludes it
            # (already_used covers it → unaccounted ≈ 0); if built *before*,
            # already_used ≈ 0 so we reserve the full footprint.  Either way we
            # never double-charge and never ignore it.
            already_used_gb = max(0.0, total_gb - free_gb)
            unaccounted_main_gb = max(0.0, main_footprint_gb - already_used_gb)

            usable_gb = max(0.0, free_gb - reserve_gb - unaccounted_main_gb)
            fits = int(usable_gb // per_copy_gb) if per_copy_gb > 0 else 0
            # Never spawn more than requested: free VRAM is not a reason to
            # exceed the configured parallelism, and capping here removes the
            # surprise over-allocation path.  Scale down when fewer fit, but
            # keep ≥1 since the pool was explicitly enabled.
            effective = max(1, min(requested, fits))

            logger.info(
                f"[device] {device}: {free_gb:.1f}GB free / {total_gb:.1f}GB total "
                f"(reserve {reserve_gb:.1f}GB + main ~{unaccounted_main_gb:.2f}GB "
                f"unaccounted of {main_footprint_gb:.2f}GB). Per worker copy "
                f"~{per_copy_gb:.2f}GB (weights {model_gb:.2f} + KV {kv_gb:.2f} "
                f"@ n_ctx<={worker_ctx} + 0.7). "
                f"Fits ~{fits} copy(ies); requested {requested} → using {effective}."
            )
            if effective != requested:
                logger.warning(
                    f"[device] Adjusted LLM workers {requested} → {effective} based on "
                    f"{free_gb:.1f}GB free memory. Tune with SLP_LLM_WORKER_N_CTX "
                    f"(smaller = more copies) or free memory."
                )
            return effective
        except Exception as exc:
            logger.info(
                f"[device] Could not estimate memory capacity ({exc}); "
                f"using requested {requested} worker(s)"
            )
            return requested

    # -- Public dispatch API ------------------------------------------------

    @property
    def has_llm_pool(self) -> bool:
        """True when a dedicated LLM worker process pool is available."""
        return self._llm_pool is not None

    def submit_io(self, fn, *args, **kwargs):
        """Submit I/O-bound work (search, MCP calls) to the thread pool."""
        return self._io_pool.submit(fn, *args, **kwargs)

    def submit_llm(self, prompt: str):
        """Submit LLM inference to the worker process pool.

        Only valid when a pool exists — callers must gate on ``has_llm_pool``
        and otherwise run inference on the shared in-process model.  (The
        workers' model lives in their own processes; ``run_inference`` has no
        model to use in the main process, so there is no in-process fallback.)
        """
        if self._llm_pool is None:
            raise RuntimeError(
                "submit_llm called without a worker pool; gate on has_llm_pool "
                "and use the shared in-process model instead."
            )
        return self._llm_pool.submit(run_inference, prompt)

    # -- Lifecycle ----------------------------------------------------------

    def shutdown(self, wait: bool = False) -> None:
        """Shut down both pools."""
        self._io_pool.shutdown(wait=wait)
        if self._llm_pool is not None:
            self._llm_pool.shutdown(wait=wait)
