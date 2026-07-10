"""Singleton model service for managing LLM lifecycle"""

import json
import logging
import mimetypes
import os
import re
import shutil
import tempfile
import threading
import gc
import sys


from pathlib import Path
from typing import Dict, Optional, Iterator, List
from uuid import uuid4

import yaml
import markdownify

from langchain_core.outputs import GenerationChunk

from smartloop.api.services.project_io import resolve_visible_dir
from smartloop.model_factory import ModelFactory
from smartloop.config import ModelConfig, SUPPORTED_MODELS
from smartloop.llms.base_llm import BaseLLM
from smartloop.workspace import DocumentInfo, DocumentStatus, MCPServer, Workspace
from smartloop.workspace import Project, MCPServer, MCPServerConfig, MCPToolDefinition, Skill, MCPServerType, RemoteMCPServerConfig
from smartloop.db import ChunkResult, StoreFactory
from smartloop.db.workspace_indexer import WorkspaceIndexer, WORKSPACE_COLLECTION
from smartloop.db.feedback_indexer import FeedbackIndexer
from smartloop.feedback_store import FeedbackStore, FeedbackEntry, ConversationTurn
from smartloop.model import Model
from smartloop.mcp.local_client import LocalMCPClient
from smartloop.mcp.client import MCPLaunchError, MCPDiscoveryError
from smartloop.agents.orchestrator import SLPOrchestrator
from smartloop.agents.planner import SYSTEM_TOOLS_SERVER_ID, SYSTEM_TOOLS_SERVER_NAME, FETCH_SERVER_ID
from smartloop.agents.mcp.skill_generator import MCPSkillGenerator

from smartloop.parsers.parser_factory import ParserFactory
from smartloop import ConversationStore, MessageInput

from smartloop.api.config import api_settings
from smartloop.config.app_settings import AppSettings

from smartloop.auth import credential_store
from smartloop.model.downloader import SLPArtifactDownloader, HFDownloader
from smartloop.tasks import (
    TaskQueue,
    DownloaderTask,
    BuildTask,
    TrainTask,
    IndexerTask,
)
from smartloop.utils import normalize_path

_settings = AppSettings()

logger = logging.getLogger(__name__)

# Exceptions that indicate code defects (not recoverable runtime/IO
# failures).  These re-raise instead of degrading to an empty result so
# bugs surface loudly rather than silently returning no context.
_PROGRAMMING_ERRORS = (NameError, AttributeError, TypeError, ImportError)

DEFAULT_MAX_CONTEXT_TOKENS = 8192

class ModelService:
    """Singleton service to manage model lifecycle and inference"""

    _instance: Optional["ModelService"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._llm: Optional[BaseLLM] = None
        self._model: Optional[Model] = None  # Store Model object to access workspace
        self._model_name: Optional[str] = None
        # Decision model — smaller LLM for classify / match_skills nodes.
        # Falls back to main model when not configured.
        self._decision_llm: Optional[BaseLLM] = None
        self._decision_model_name: Optional[str] = None
        self._large_llm: Optional[BaseLLM] = None
        self._large_model_name: Optional[str] = None
        # Multimodal vision model for document OCR (needs main + mmproj GGUF).
        self._vision_llm: Optional[BaseLLM] = None
        self._vision_model_name: Optional[str] = None
        self._vision_unavailable: bool = False
        # Set once the large model has been considered (loaded, skipped for
        # fit, or absent), so we don't re-probe VRAM on every agent request.
        self._large_model_attempted: bool = False
        # Shared, long-lived executor whose LLM worker processes load the
        # model once and are reused across requests (used for parallel
        # document fan-out).  Created lazily; reset on model load/unload.
        self._llm_executor = None
        # Use a semaphore sized to n_parallel so agents can run concurrently
        # when the model is loaded with n_parallel > 1 (continuous batching).
        self._inference_lock = threading.Semaphore(_settings.n_parallel)
        self._sessions: dict[str, List[MessageInput]] = {}  # session_id -> chat history (in-memory cache)
        self._conversation_store = ConversationStore(_settings.home_dir)
        self._task_queue = TaskQueue(logger)
        self._workspace_indexer: Optional[WorkspaceIndexer] = None
        self._feedback_indexer: Optional[FeedbackIndexer] = None
        self._doc_store: Optional[Any] = None
        self._doc_store_project_id: Optional[str] = None
        self._initialized = True

    def get_session_history(self, session_id: str) -> List[MessageInput]:
        """Get chat history for a session, loading from disk on first access."""
        if session_id not in self._sessions:
            self._sessions[session_id] = self._conversation_store.load(session_id)
        return self._sessions[session_id]

    def get_relevant_history(
        self,
        session_id: str,
        user_query: str,
        max_turns: int = 4,
    ) -> List[Dict[str, str]]:
        """Return the most recent conversation turns as dicts.

        Returns the last *max_turns* messages from the session with
        strict user/assistant alternation so the LLM never sees
        consecutive messages from the same role.
        """
        history = self.get_session_history(session_id)
        messages = [
            {"role": m.role, "content": m.content}
            for m in history if m.content
        ]

        # Enforce strict alternation: collapse consecutive same-role
        # messages and ensure the sequence starts with user.
        alternating: List[Dict[str, str]] = []
        for msg in messages:
            if alternating and alternating[-1]["role"] == msg["role"]:
                # Merge into the previous message of the same role
                alternating[-1]["content"] += "\n\n" + msg["content"]
            else:
                alternating.append(dict(msg))

        # Trim to last max_turns pairs (user+assistant = 2 messages each)
        if len(alternating) > max_turns * 2:
            alternating = alternating[-(max_turns * 2):]

        # Must start with a user message for proper alternation
        if alternating and alternating[0]["role"] != "user":
            alternating = alternating[1:]

        return alternating

    def append_to_session(self, session_id: str, role: str, content: str, citations: list | None = None, message_id: str | None = None):
        """Append a message to session history and persist to disk."""
        if session_id not in self._sessions:
            self._sessions[session_id] = self._conversation_store.load(session_id)
        message = MessageInput(role=role, content=content, message_id=message_id, citations=citations)
        self._sessions[session_id].append(message)
        self._conversation_store.append(session_id, message)

        # Track session on the current project
        if self._model and self._model.workspace:
            project = self._model.workspace.current_project
            if project and session_id not in project.sessions:
                project.sessions.append(session_id)
                self._model.save()

    def _ensure_model(self) -> None:
        """Ensure the default orchestrator model is initialized."""
        default = api_settings.default_model

        if self._model is not None and self._model_name == default:
            return

        workspace = self._ensure_workspace(ModelConfig(
            model_name=default,
            hf_token=api_settings.hf_token,
            developer_token=api_settings.developer_token,
        ))

        ws_model = workspace.get_model(default)
        if not ws_model:
            model_type = SUPPORTED_MODELS.get(default)
            if not model_type:
                raise ValueError(f"Default model {default} not found in workspace")
            ws_model = workspace.add_model(default, model_type.model_id(), config=ModelConfig(
                model_name=default,
                hf_token=api_settings.hf_token,
                developer_token=api_settings.developer_token,
            ))

        model_config = ModelConfig(
            model_name=default,
            hf_token=api_settings.hf_token,
            developer_token=api_settings.developer_token,
        )

        logger.info(f"[_ensure_model] loading default model {default}")
        self._model = ModelFactory.get_model(model_config)
        self._model_name = default
 
    def _ensure_workspace(self, model_config: Optional[ModelConfig] = None) -> Workspace:
        """Return the workspace, creating it if necessary.

        Returns the ``Workspace`` instance.  If a model pipeline has
        already been initialized (``self._model`` is set), its workspace
        is returned directly.  Otherwise the workspace is loaded from
        disk, falling back to ``Workspace.init()`` when no workspace
        exists yet.
        """
        if self._model is not None:
            return self._model.workspace

        workspace = Workspace.load()
        if workspace is not None:
            return workspace

        if model_config is None:
            model_config = ModelConfig(
                model_name=api_settings.default_model,
                hf_token=api_settings.hf_token,
                developer_token=api_settings.developer_token,
            )

        return Workspace.init(model_config)

    def _initialize_workspace(self) -> None:
        """Ensure the workspace is loaded, raising if it cannot be resolved."""
        self._ensure_model()
        if self._model is None:
            raise RuntimeError("Model not loaded")

    def _get_project(self, project_id: str) -> "Project":
        """Find a project by ID, raising ValueError if not found."""
        self._initialize_workspace()
        workspace = self._model.workspace
        project = next((p for p in workspace.projects if p.id == project_id), None)
        if project is None:
            # The cached workspace singleton can be stale when another process
            # (e.g. the CLI/TUI) created the project and wrote config.json after
            # this server loaded. Re-read from disk once before giving up.
            workspace.reload()
            project = next((p for p in workspace.projects if p.id == project_id), None)
        if project is None:
            raise ValueError(f"Project not found: {project_id}")
        return project

    def load_model(self) -> BaseLLM:
        """
        Load and cache the LLM model for the current project.

        Uses the workspace's active model, falling back to api_settings.default_model.

        Returns:
            The loaded BaseLLM instance
        """
        workspace = self._ensure_workspace()
        current_project = workspace.current_project
        # The routing/main model is always the small base model
        # (api_settings.default_model), independent of the workspace's active
        # selection.  A larger active/downloaded model is loaded lazily as the
        # on-demand LARGE agent model only when a plan step needs it (see
        # _detect_large_model / _get_large_llm).  This keeps startup VRAM free
        # for the small orchestrator model + its worker-pool copies.
        target_model = api_settings.default_model
        logger.info(f"[load_model] target={target_model} (orchestrator model)")

        # Return cached model if already loaded
        if self._llm is not None and self._model_name == target_model:
            logger.info(f"Using cached model: {target_model}")
            return self._llm

        # If a different model is loaded, unload it first
        if self._llm is not None and self._model_name != target_model:
            logger.info(f"Switching model from {self._model_name} to {target_model}")
            self._llm = None
            self._model = None
            self._model_name = None
            self._decision_llm = None
            self._decision_model_name = None
            Workspace.reset()

        self._ensure_model()

        project_name = self.get_current_project_name() or "default"
        logger.info(f"Loading model: {target_model} (project: {project_name})")

        # Require an active project before attempting to load
        if not workspace.current_project:
            raise ValueError(
                "No active project. Please create a project first."
            )

        current_project = workspace.current_project
        if current_project:
            self._setup_project_tools(current_project)

        self._llm = self._model.get_instance(debug=api_settings.debug_mode)
        self._model_name = target_model

        # Cache the GGUF path so the orchestrator can pass it to LLM worker
        # processes (each process needs to load its own model instance).
        # Resolve the orchestrator model (target_model) explicitly rather than
        # the workspace-active model, so the worker pool copies the small
        # orchestrator model — not a larger active model that's meant to load
        # lazily on demand.
        try:
            ws_model = self._model.workspace.get_model(target_model)
            model_id = ws_model.model_id if ws_model else ""
            self._model_gguf_path = self._model.workspace.resolve_gguf_path(model_id)
        except Exception:
            self._model_gguf_path = None

        # Rebuild the shared LLM worker pool so its processes load this model.
        self._reset_llm_executor()

        logger.info(f"Model loaded successfully: {target_model} (project: {project_name})")

        # Auto-load decision model if configured
        if _settings.decision_model and not self._decision_llm:
            self._load_decision_model(_settings.decision_model)

        # The larger (e.g. 4B) agent model is loaded lazily on first agent use
        # (see _get_large_llm) rather than eagerly here — it's only needed for
        # agent/skill tasks, so deferring it keeps startup VRAM free for the
        # main model + worker pool.  Reset the per-model attempt flag so the
        # next agent request re-evaluates against the newly active model.
        self._large_llm = None
        self._large_model_name = None
        self._large_model_attempted = False
        # Vision (multimodal) model is also loaded lazily on first OCR use.
        self._vision_llm = None
        self._vision_model_name = None
        self._vision_unavailable = False

        self._warmup_embeddings()
        self._warmup_inference()

        return self._llm

    def _load_decision_model(self, model_name: str) -> None:
        """Load a secondary small model for decision nodes.

        Called automatically during :meth:`load_model` when
        ``SLP_DECISION_MODEL`` is set.  The decision model shares the
        same inference semaphore as the main model.
        """
        if self._decision_llm is not None and self._decision_model_name == model_name:
            return
        try:
            logger.info(f"[decision-model] Loading decision model: {model_name}")
            decision_model_obj = Model(model_name)
            self._decision_llm = decision_model_obj.get_instance(debug=api_settings.debug_mode)
            self._decision_model_name = model_name
            logger.info(f"[decision-model] Decision model loaded: {model_name}")
        except Exception as e:
            logger.warning(
                f"[decision-model] Failed to load '{model_name}', "
                f"falling back to main model: {e}"
            )
            self._decision_llm = None
            self._decision_model_name = None

    def _get_large_llm(self) -> Optional[BaseLLM]:
        """Return the large (agent) model, loading it lazily on first use.

        Loaded on demand — only when an agent task actually needs it — rather
        than at startup, so the main model + worker pool own VRAM by default.
        Attempted at most once per active model (``_large_model_attempted``):
        if it loads it's cached; if it's absent or skipped for fit we return
        ``None`` and the caller falls back to the main/decision model.
        """
        if self._large_llm is not None:
            return self._large_llm
        if self._large_model_attempted:
            return None
        self._large_model_attempted = True
        self._detect_large_model()
        return self._large_llm

    @staticmethod
    def _find_model_ggufs(model_dir: str) -> tuple[Optional[str], Optional[str]]:
        """Return (main_gguf, mmproj_gguf) found directly in ``model_dir``.

        The mmproj / CLIP projector ships as a separate ``*mmproj*.gguf`` (or
        ``*clip*.gguf``) alongside the model GGUF; it must not be mistaken for
        the main weights, so it is matched by name and excluded from the main
        pick.
        """
        main_gguf = None
        mmproj_gguf = None
        if os.path.isdir(model_dir):
            for f in sorted(os.listdir(model_dir)):
                if not f.endswith(".gguf"):
                    continue
                lower = f.lower()
                if "mmproj" in lower or "clip" in lower:
                    if mmproj_gguf is None:
                        mmproj_gguf = os.path.join(model_dir, f)
                elif main_gguf is None:
                    main_gguf = os.path.join(model_dir, f)
        return main_gguf, mmproj_gguf

    def get_vision_llm(self) -> Optional[BaseLLM]:
        """Return a loaded *multimodal* vision LLM, or None when unavailable.

        Vision-capable models are considered largest-first (like the
        orchestrator), but a model is only usable for OCR when ALL of these
        hold, so unrunnable candidates are skipped rather than crashing:
          * its mmproj / CLIP projector GGUF is present (else it's blind), and
          * the installed llama-cpp has a chat handler for its architecture
            (e.g. gemma3 has none in llama-cpp-python and aborts natively), and
          * it fits in available memory.
        Returns None when no candidate qualifies (callers fall back to text).
        """
        from smartloop.config.app_settings import AppSettings
        if not AppSettings().vision_enabled:
            return None
        if self._vision_llm is not None:
            return self._vision_llm
        if not self._model or not self._model.workspace:
            return None

        from smartloop.llms.llm_factory import LlmFactory
        workspace = self._model.workspace
        candidates = [
            m for m in workspace.models
            if m.enabled and m.downloaded
            and "vision" in [c.lower() for c in m.effective_capabilities]
        ]
        # Smallest VL model first: OCR quality is adequate from a lighter model,
        # and a smaller one is far more likely to fit on GPU alongside the
        # orchestrator + on-demand agent models (e.g. Qwen2-VL-2B over -3B).
        candidates.sort(key=lambda m: m.param_size_b or 0.0)

        # OCR is a latency-tolerant batch task, so give the vision model a small
        # dedicated context and waive the agent-model reserve — it only needs to
        # fit the image tokens plus a short transcription, and shouldn't be
        # blocked by headroom kept for the (already-resident) agent model.
        cap = AppSettings().vision_n_ctx
        # Track whether any candidate was rejected for a *transient* reason
        # (not enough free VRAM right now, or a load error) versus a *permanent*
        # one (no mmproj, no handler).  A transient failure must NOT latch
        # ``_vision_unavailable`` — free VRAM recovers once the indexing worker
        # copies are released, so the next document should retry rather than be
        # stuck on text-only for the rest of the process lifetime.
        transient_failure = False
        for ws_model in candidates:
            model_dir = workspace.model_dir_for(ws_model.model_id)
            gguf_path, mmproj_path = self._find_model_ggufs(model_dir)
            if not gguf_path or not mmproj_path:
                logger.info(
                    "[vision] %s skipped — missing %s",
                    ws_model.model_name,
                    "weights GGUF" if not gguf_path else "mmproj/CLIP GGUF",
                )
                continue
            if not LlmFactory.vision_handler_supported(gguf_path):
                logger.warning(
                    "[vision] %s skipped — installed llama-cpp has no vision handler "
                    "for its architecture (%s)",
                    ws_model.model_name, LlmFactory._read_gguf_arch(gguf_path),
                )
                continue
            if not self._large_model_fits(gguf_path, cap, ws_model.model_name,
                                          reserve_gb=0.0):
                logger.warning(
                    "[vision] %s skipped — won't fit in available memory right now "
                    "(will retry on the next document)",
                    ws_model.model_name,
                )
                transient_failure = True
                continue
            clip_on_cpu = self._clip_on_cpu(gguf_path, cap)
            try:
                llm = LlmFactory.get_llama_cpp_instance(
                    gguf_path, logger, self.get_project_id() or "",
                    max_n_ctx=cap, mmproj_path=mmproj_path,
                    clip_on_cpu=clip_on_cpu,
                )
                self._vision_llm = llm
                self._vision_model_name = ws_model.model_name
                logger.info(
                    "[vision] Loaded %s (+mmproj) for image OCR", ws_model.model_name
                )
                return llm
            except Exception as e:
                logger.warning("[vision] Failed to load %s: %s", ws_model.model_name, e)
                transient_failure = True

        if transient_failure:
            logger.warning(
                "[vision] No vision model could be loaded right now — text-only "
                "for this document; will retry later"
            )
        else:
            logger.warning("[vision] No runnable vision model available — image OCR disabled")
        return None

    def _get_llm_for_capability(self, capability: str) -> Optional[BaseLLM]:
        """Return a loaded LLM matching the requested capability.

        Uses ``workspace.best_model_for(capability)`` to find the best
        enabled model that declares the capability.  Loads the GGUF
        lazily and caches the result.  Returns ``None`` when no matching
        model is available (caller should fall back to the default large
        model).
        """
        if not self._model or not self._model.workspace:
            logger.warning(
                "[capability-model] No workspace loaded — cannot resolve "
                "capability '%s'", capability,
            )
            return None
        workspace = self._model.workspace
        ws_model = workspace.best_model_for(capability)
        if not ws_model:
            logger.warning(
                "[capability-model] No enabled+downloaded model declares "
                "capability '%s'", capability,
            )
            return None
        cap_lower = capability.lower()
        if cap_lower not in [c.lower() for c in ws_model.effective_capabilities]:
            logger.warning(
                "[capability-model] Best model %s does not declare "
                "capability '%s' (has %s)",
                ws_model.model_name, capability, ws_model.effective_capabilities,
            )
            return None

        cache_key = f"_cap_llm_{ws_model.model_name}"
        cached = getattr(self, cache_key, None)
        if cached is not None:
            return cached

        try:
            model_dir = workspace.model_dir_for(ws_model.model_id)
            # _find_model_ggufs excludes the mmproj/CLIP GGUF from the main pick.
            # This path is text-only capability routing (orchestrator); image OCR
            # goes through get_vision_llm(), which loads the multimodal handler.
            gguf_path, _mmproj = self._find_model_ggufs(model_dir)
            if not gguf_path:
                logger.warning(
                    "[capability-model] No GGUF found for %s at %s — "
                    "capability '%s' unavailable",
                    ws_model.model_name, model_dir, capability,
                )
                return None

            from smartloop.config.app_settings import AppSettings
            _s = AppSettings()
            cap = _s.llm_large_n_ctx_cap
            # Same fit guard as the default large model: skip the capability
            # (4B) model when it won't fit, so the plan step falls back to the
            # default routing model instead of OOM-ing.
            if not self._large_model_fits(gguf_path, cap, ws_model.model_name):
                logger.warning(
                    "[capability-model] %s doesn't fit alongside the main "
                    "model at n_ctx=%d — capability '%s' unavailable",
                    ws_model.model_name, cap, capability,
                )
                return None
            from smartloop.llms.llm_factory import LlmFactory
            llm = LlmFactory.get_llama_cpp_instance(
                gguf_path, logger, self.get_project_id() or "",
                max_n_ctx=cap,
            )
            setattr(self, cache_key, llm)
            logger.info(
                "[capability-model] Loaded %s for capability '%s'",
                ws_model.model_name, capability,
            )
            return llm
        except Exception as e:
            logger.warning(
                "[capability-model] Failed to load %s for capability '%s': %s",
                ws_model.model_name, capability, e,
            )
            return None

    @staticmethod
    def _device_free_total_gb() -> Optional[tuple[float, float]]:
        """Return (free_gb, total_gb) for the active accelerator, or None.

        None on CPU (RAM is plentiful; no fit guard needed) or when the device
        can't be queried.
        """
        try:
            import torch
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                return free_b / 1024**3, total_b / 1024**3
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                import psutil
                vm = psutil.virtual_memory()
                return vm.available / 1024**3, vm.total / 1024**3
        except Exception:
            pass
        return None

    def _large_model_fits(self, gguf_path: str, cap: int, label: str = "",
                          reserve_gb: Optional[float] = None) -> bool:
        """True if a large model's capped footprint fits in free VRAM.

        Estimates the model's footprint (weights + KV cache @ ``cap`` +
        per-process overhead) and compares it against free device memory after
        a reserve.  ``reserve_gb`` defaults to the configured
        ``llm_large_fit_reserve_gb``; callers that load a latency-tolerant model
        (e.g. the on-demand vision OCR model) may pass a smaller value (or 0) to
        let it use the headroom the reserve normally protects.  Always True on
        CPU (RAM is plentiful) or when the device can't be queried.  When it
        returns False the caller skips the load and falls back to the default
        routing model instead of risking an OOM / starving the worker pool.
        """
        ft = self._device_free_total_gb()
        if ft is None:
            return True
        free_gb, total_gb = ft
        try:
            model_gb = os.path.getsize(gguf_path) / 1024**3
        except OSError:
            return True
        from smartloop.config.app_settings import AppSettings
        if reserve_gb is None:
            reserve_gb = AppSettings().llm_large_fit_reserve_gb
        kv_gb = 0.5 * (cap / 4096)
        footprint_gb = model_gb + kv_gb + 0.7
        if footprint_gb + reserve_gb > free_gb:
            logger.warning(
                "[large-model] %s needs ~%.1fGB (+%.1f reserve) but only %.1fGB "
                "free of %.1fGB — falling back to the default model. Lower "
                "SLP_LLM_LARGE_N_CTX_CAP or free VRAM to enable it.",
                label or os.path.basename(gguf_path),
                footprint_gb, reserve_gb, free_gb, total_gb,
            )
            return False
        return True

    def _clip_on_cpu(self, gguf_path: str, cap: int) -> bool:
        """Decide whether the vision CLIP/mmproj encoder must run on CPU.

        GPU is preferred (faster), but the projector + per-image activations
        OOM the native clip code — an *uncatchable* SIGSEGV — when VRAM is too
        tight.  ``vision_clip_device`` forces the choice ("gpu"/"cpu"); in
        "auto" the projector stays on GPU only when free VRAM comfortably
        exceeds the weights + KV + projector/activation headroom, else CPU.
        """
        from smartloop.config.app_settings import AppSettings
        pref = (AppSettings().vision_clip_device or "auto").lower()
        if pref == "gpu":
            return False
        if pref == "cpu":
            return True
        ft = self._device_free_total_gb()
        if ft is None:
            return False  # no CUDA device — use_gpu is moot (ggml runs clip on CPU)
        free_gb, _ = ft
        try:
            model_gb = os.path.getsize(gguf_path) / 1024**3
        except OSError:
            return True
        # Projector weights (~0.8GB) + per-image encode activation headroom.
        need_gb = model_gb + 0.5 * (cap / 4096) + 1.8
        if free_gb < need_gb:
            logger.info(
                "[vision] CLIP on CPU — only %.1fGB free, GPU clip needs ~%.1fGB "
                "(set SLP_VISION_CLIP_DEVICE=gpu to force, or free VRAM)",
                free_gb, need_gb,
            )
            return True
        return False

    def _detect_large_model(self) -> None:
        """Load the largest downloaded model for agent/skill tasks, if it fits.

        Skips the load when the model's capped footprint won't fit in free
        VRAM (after a reserve for the main model + worker pool), so loading the
        large model never OOM-crashes the process or starves the workers.  The
        active (smaller) model stays as the routing LLM.
        """
        if self._large_llm is not None:
            return
        try:
            workspace = self._model.workspace
            # Compare against the orchestrator model (the loaded small model),
            # not the workspace-active model: the large model is whatever
            # downloaded model is bigger than the orchestrator model (e.g. the
            # user's active 4B).
            orchestrator = workspace.get_model(self._model_name)
            orchestrator_size = orchestrator.param_size_b if orchestrator else 0
            # Pick the agent/large model by *capability*, not raw size: this
            # model drives skills/synthesis, so it must declare the "skills"
            # capability.  Selecting by size alone would load a vision-only VL
            # model (e.g. qwen2-vl-2b) here as a redundant text copy — wasting
            # VRAM that get_vision_llm needs for the actual OCR instance.  Among
            # the skills-capable models, prefer the largest.
            skills_models = [
                m for m in workspace.models
                if m.enabled and m.downloaded
                and m.model_name != self._model_name
                and "skills" in [c.lower() for c in m.effective_capabilities]
            ]
            if not skills_models:
                logger.info(
                    "[large-model] No skills-capable model downloaded — using the "
                    "router model for skills/synthesis"
                )
                return
            largest = max(skills_models, key=lambda m: m.param_size_b or 0.0)
            largest.models_dir = workspace.models_dir
            if largest.param_size_b <= orchestrator_size:
                return

            model_dir = workspace.model_dir_for(largest.model_id)
            # Exclude the mmproj/CLIP GGUF — it is not loadable as a main model
            # (e.g. gemma-3-4b-it ships one upstream even though we don't use it
            # for vision); loading it as weights crashes the native loader.
            gguf_path, _mmproj = self._find_model_ggufs(model_dir)
            if not gguf_path:
                logger.info("[large-model] GGUF not found for %s, skipping", largest.model_name)
                return

            from smartloop.config.app_settings import AppSettings
            _s = AppSettings()
            cap = _s.llm_large_n_ctx_cap

            # Fit guard: skip the load when the large model's capped footprint
            # won't fit in free VRAM after the reserve.  free VRAM already
            # excludes the resident main model, so this is the room genuinely
            # left for the agent model.
            if not self._large_model_fits(gguf_path, cap, largest.model_name):
                return

            logger.info(
                "[large-model] Loading larger model on demand: %s (%.0fB) > router "
                "%s (%.0fB), n_ctx<=%d — %s",
                largest.model_name, largest.param_size_b,
                self._model_name or "?", orchestrator_size, cap, gguf_path,
            )
            from smartloop.llms.llm_factory import LlmFactory
            self._large_llm = LlmFactory.get_llama_cpp_instance(
                gguf_path, logger, self.get_project_id() or "", max_n_ctx=cap,
            )
            self._large_model_name = largest.model_name
            logger.info("[large-model] Loaded %s for agent/skill tasks", largest.model_name)
        except Exception as e:
            logger.warning("[large-model] Failed to load larger model: %s", e)
            self._large_llm = None
            self._large_model_name = None

    def _warmup_embeddings(self) -> None:
        """Pre-load the embedding model in a background thread."""
        base_dir = getattr(self._model, "workspace", None)
        base_dir = base_dir.workspace_dir if base_dir else None

        def _load():
            try:
                from smartloop.db.embeddings import warmup
                warmup(base_dir)
            except Exception as e:
                logger.warning(f"[embeddings] Warmup failed: {e}")

        threading.Thread(target=_load, name="embedding-warmup", daemon=True).start()

    def _warmup_inference(self) -> None:
        """Run a trivial inference to prime the KV cache after model load."""
        if self._llm is None:
            return
        try:
            self._llm.invoke("<system>Reply OK</system><user>hi</user>")
            logger.info("[warmup] Inference warmup completed")
        except Exception as e:
            logger.warning(f"[warmup] Inference warmup failed: {e}")

    @property
    def decision_llm(self) -> Optional[BaseLLM]:
        """Return the decision LLM, or None to fall back to the main model."""
        return self._decision_llm

    def search_workspace(self, query: str, work_dir: str, n_results: int = 10) -> list:
        """Index workspace files on demand then search.

        Indexing is incremental (hash-based) so repeated calls are cheap.
        Returns a list of :class:`~smartloop.db.store.ChunkResult` ordered
        by relevance.
        """
        if self._model is None or not work_dir or not os.path.isdir(work_dir):
            return []
        try:
            if self._workspace_indexer is None:
                current_project = self._model.workspace.current_project
                if not current_project:
                    return []
                self._workspace_indexer = WorkspaceIndexer(
                    project_dir=current_project.project_dir,
                    logger=logger,
                    base_dir=self._model.workspace.workspace_dir,
                )
            self._workspace_indexer.index(work_dir)
            return self._workspace_indexer.search(query, n_results=n_results)
        except Exception as e:
            logger.warning(f"[workspace-index] search_workspace failed: {e}")
            self._workspace_indexer = None
            return []

    # --------------------------------------------------------------- feedback

    def _ensure_feedback_indexer(self) -> FeedbackIndexer:
        if self._feedback_indexer is None:
            current_project = self._model.workspace.current_project
            if not current_project:
                raise ValueError("No active project")
            self._feedback_indexer = FeedbackIndexer(
                project_dir=current_project.project_dir,
                logger=logger,
                base_dir=self._model.workspace.workspace_dir,
            )
        return self._feedback_indexer

    def submit_feedback(self, session_id: str, message_id: str, project_id: str, rating: str) -> dict:
        self._ensure_model()

        history = self.get_session_history(session_id)
        if not history:
            raise ValueError(f"Session not found: {session_id}")

        target_idx = None
        for i, m in enumerate(history):
            if m.message_id == message_id:
                target_idx = i
                break

        if target_idx is None:
            raise ValueError(f"Message not found: {message_id}")

        target = history[target_idx]
        if target.role != "assistant":
            raise ValueError(f"Message {message_id} is not an assistant message")

        question = ""
        for i in range(target_idx - 1, -1, -1):
            if history[i].role == "user":
                question = history[i].content
                break

        answer = target.content
        conversation = [
            ConversationTurn(role=m.role, content=m.content)
            for m in history[:target_idx + 1]
        ]

        selected_project = self._get_project(project_id)

        feedback_store = FeedbackStore(selected_project.project_dir)
        indexer = self._ensure_feedback_indexer()

        # Update rating on the conversation message
        target.rating = rating
        self._conversation_store.save(session_id, history)

        if rating == "up":
            existing = feedback_store.find(session_id, message_id)
            if existing:
                indexer.remove_feedback(existing.feedback_id)
                feedback_store.remove(existing.feedback_id)

            entry = FeedbackEntry.create(
                session_id=session_id,
                message_id=message_id,
                rating="up",
                question=question,
                answer=answer,
                conversation=conversation,
            )
            feedback_store.append(entry)
            if question and answer:
                indexer.index_feedback(entry)

            return {
                "feedback_id": entry.feedback_id,
                "session_id": session_id,
                "message_id": message_id,
                "rating": "up",
                "indexed": bool(question and answer),
            }
        else:
            existing = feedback_store.find(session_id, message_id)
            if existing:
                indexer.remove_feedback(existing.feedback_id)
                feedback_store.remove(existing.feedback_id)

            entry = FeedbackEntry.create(
                session_id=session_id,
                message_id=message_id,
                rating="down",
                question=question,
                answer=answer,
                conversation=conversation,
            )
            feedback_store.append(entry)

            return {
                "feedback_id": entry.feedback_id,
                "session_id": session_id,
                "message_id": message_id,
                "rating": "down",
                "indexed": False,
            }

    def get_feedback_ratings(self, session_id: str, project_id: str) -> dict[str, str]:
        self._ensure_model()
        
        project = self._get_project(project_id)

        if not project:
            raise ValueError(f"Project not found: {project_id}")
        
        feedback_store = FeedbackStore(project.project_dir)
        
        return {
            e.message_id: e.rating
            for e in feedback_store.load()
            if e.session_id == session_id
        }

    def delete_feedback_ratings(self, session_id: str, message_id: str, project_id: str | None = None) -> None:
        """Delete feedback for a specific message and remove it from the index."""
        self._ensure_model()

        if project_id:
            project = self._get_project(project_id)
        else:
            raise ValueError("No active project")

        feedback_store = FeedbackStore(project.project_dir)
        existing = feedback_store.find(session_id, message_id)
        if existing:
            indexer = self._ensure_feedback_indexer()
            indexer.remove_feedback(existing.feedback_id)
            feedback_store.remove(existing.feedback_id)

        # Clear the rating on the in-memory session message
        history = self.get_session_history(session_id)
        for m in history:
            if m.message_id == message_id:
                m.rating = None
                break
        self._conversation_store.save(session_id, history)

    def search_feedback(self, query: str, n_results: int = 5) -> list:
        if self._model is None:
            return []
        try:
            indexer = self._ensure_feedback_indexer()
            return indexer.search(query, n_results=n_results)
        except Exception as e:
            logger.warning(f"[feedback-index] search_feedback failed: {e}")
            self._feedback_indexer = None
            return []

    def build_model(self, model_name: Optional[str] = None, quantize: Optional[str] = None, developer_token: Optional[str] = None, progress_callback=None) -> None:
        """
        Build the GGUF model.

        Resolves model name in order: explicit argument → current project's
        model_name → current active workspace model → api_settings.default_model.

        1. Unloads the inference LLM to free memory.
        2. Ensures the workspace/Model object is initialized.
        3. Calls model.build() to convert to GGUF format.

        NOTE: No download step — all artifacts must already be on disk.
        Use the downloader agent to download model artifacts first.
        """

        self.unload_model()
        self._ensure_model()

        if not self._model.workspace.current_project:
            raise ValueError(
                "No active project. Please create a project first."
            )

        if quantize:
            workspace = self._model.workspace
            ws_model = workspace.get_model()
            if ws_model and ws_model.config:
                ws_model.config.quantize = quantize

        logger.info("Starting model build…")
        try:
            self._model.build(progress_callback=progress_callback)
            logger.info("Model build completed.")
        finally:
            self._llm = None
            self._model_name = None
            self._decision_llm = None
            self._decision_model_name = None

    def train_model(self, model_name: Optional[str] = None, progress_callback=None) -> None:
        """
        Run the full training pipeline.

        1. Unloads the inference LLM to free memory.
        2. Ensures the workspace/Model object is initialised.
        3. Calls model.train() (downloads artefacts + runs LoRA training).
        4. Unloads the workspace so a fresh reload picks up new weights.
        """
        # 1. Free inference model from memory first
        self.unload_model()

        # 2. Initialise workspace (no LLM loaded)
        self._ensure_model()

        logger.info("Starting model training…")
        try:
            self._model.train(progress_callback=progress_callback)
            logger.info("Model training completed.")
            # Persist updated document statuses (PENDING → PROCESSED) to disk
            # so the auto-train check in unload_model() does not re-fire.
            try:
                self._model.save()
            except Exception as e:
                logger.warning(f"Could not save model after training: {e}")
        finally:
            # Reset entirely so next load_model() re-initialises with new weights
            self._llm = None
            self._model = None
            self._model_name = None
            self._decision_llm = None
            self._decision_model_name = None
            Workspace.reset()

    def get_llm(self) -> Optional[BaseLLM]:
        """Get the currently loaded LLM instance"""
        return self._llm

    def get_current_project_name(self) -> Optional[str]:
        """Get the current project name"""
        if self._model is None:
            return None
        try:
            current_project = self._model.workspace.current_project
            return current_project.name if current_project else None
        except Exception:
            return None

    def get_project_id(self) -> Optional[str]:
        """Get the current project ID"""
        if self._model is None:
            return None
        try:
            current_project = self._model.workspace.current_project
            return current_project.id if current_project else None
        except Exception:
            return None

    _SYSTEM_TOOL_DEFS = [
        MCPToolDefinition(
            name="bash",
            description="Execute a shell command (git, echo, etc.) on the host machine",
            input_schema={"type": "object", "properties": {"command": {"type": "string", "description": "Shell command to execute"}}, "required": ["command"]},
        ),
        MCPToolDefinition(
            name="file-read",
            description="Read a file and return its content",
            input_schema={"type": "object", "properties": {"path": {"type": "string", "description": "File path to read"}}, "required": ["path"]},
        ),
        MCPToolDefinition(
            name="file-write",
            description="Write content to a file in the working directory",
            input_schema={"type": "object", "properties": {"path": {"type": "string", "description": "File path to write"}, "content": {"type": "string", "description": "Content to write"}}, "required": ["path", "content"]},
        ),
        MCPToolDefinition(
            name="grep",
            description="Recursively search files in the working directory for a text pattern or regex",
            input_schema={"type": "object", "properties": {"pattern": {"type": "string", "description": "Text or regex to search for"}, "glob": {"type": "string", "description": "Optional filename filter, e.g. '*.py'"}}, "required": ["pattern"]},
        ),
        MCPToolDefinition(
            name="explore",
            description="Explore the local project directory, read files, and produce an LLM-enriched summary of the codebase",
            input_schema={"type": "object", "properties": {"query": {"type": "string", "description": "What to look for in the codebase"}}, "required": ["query"]},
        ),
    ]

    def install_system_tools_server(self, project_id: str) -> None:
        """Register the System Tools MCP server on a project if not already present."""
        project = self._get_project(project_id)
        existing = next(
            (s for s in project.mcp_servers if s.name == SYSTEM_TOOLS_SERVER_NAME), None,
        )
        if existing is not None:
            # Refresh the advertised tool list so projects created before a
            # tool was added (e.g. grep) pick it up on next load.
            current = [t.name for t in (existing.tools or [])]
            wanted = [t.name for t in self._SYSTEM_TOOL_DEFS]
            if current != wanted:
                existing.tools = list(self._SYSTEM_TOOL_DEFS)
                self._model.save()
                logger.info(f"Refreshed System Tools server tools: {wanted}")
            else:
                logger.info("System Tools server already registered")
            return

        server = MCPServer(
            id=SYSTEM_TOOLS_SERVER_ID,
            name=SYSTEM_TOOLS_SERVER_NAME,
            description="Built-in tools for file operations, search, and shell commands",
            server_type=MCPServerType.LOCAL,
            config=MCPServerConfig(
                command=sys.executable,
                args=["-m", "smartloop.api.mcp.system_server"],
            ),
            enabled=True,
            system=True,
            tools=list(self._SYSTEM_TOOL_DEFS),
        )
        project.mcp_servers.append(server)
        self._model.save()
        logger.info(f"Registered System Tools server on project {project_id}")

    FETCH_SERVER_NAME = "fetch"

    _FETCH_TOOL_DEFS = [
        MCPToolDefinition(
            name="fetch",
            description="Fetches a URL from the internet and extracts its contents as markdown",
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "max_length": {"type": "integer", "description": "Maximum number of characters to return", "default": sys.maxsize},
                    "start_index": {"type": "integer", "description": "Start content from this character index", "default": 0},
                    "raw": {"type": "boolean", "description": "Get raw content without markdown conversion", "default": False},
                },
                "required": ["url"],
            },
        ),
    ]

    def install_fetch_server(self, project_id: str) -> None:
        """Register the fetch MCP server on a project if not already present."""
        project = self._get_project(project_id)
        if any(s.name == self.FETCH_SERVER_NAME for s in project.mcp_servers):
            logger.info("Fetch server already registered")
            return

        server = MCPServer(
            id=FETCH_SERVER_ID,
            name=self.FETCH_SERVER_NAME,
            pkg="mcp-server-fetch",
            description="Fetches URLs from the internet and extracts content as markdown",
            server_type=MCPServerType.LOCAL,
            config=MCPServerConfig(
                command=sys.executable,
                args=["-m", "smartloop.api.mcp.fetch_server"],
            ),
            enabled=True,
            system=False,
            tools=list(self._FETCH_TOOL_DEFS),
        )
        project.mcp_servers.append(server)
        self._model.save()
        logger.info(f"Registered fetch server on project {project_id}")

    def _migrate_servers_to_local(self, project: "Project") -> None:
        """Migrate legacy REMOTE system servers to LOCAL (stdio subprocess)."""
        migrated = False
        _LAUNCHERS = {"uvx", "npx", "pipx", "deno", "bunx"}
        for server in project.mcp_servers:
            if server.id == FETCH_SERVER_ID and server.server_type == MCPServerType.REMOTE:
                server.server_type = MCPServerType.LOCAL
                server.config = MCPServerConfig(
                    command=sys.executable,
                    args=["-m", "smartloop.api.mcp.fetch_server"],
                )
                server.remote_config = None
                migrated = True
                logger.info(f"Migrated fetch server to LOCAL on project {project.id}")
            if server.id == SYSTEM_TOOLS_SERVER_ID and server.server_type == MCPServerType.REMOTE:
                server.server_type = MCPServerType.LOCAL
                server.config = MCPServerConfig(
                    command=sys.executable,
                    args=["-m", "smartloop.api.mcp.system_server"],
                )
                server.remote_config = None
                migrated = True
                logger.info(f"Migrated system tools server to LOCAL on project {project.id}")
            # Fix stale user-registered local servers saved with empty args
            # (registered before args validation was enforced). Attempt to
            # recover using the saved pkg or server name so existing projects
            # keep working without requiring a manual re-registration.
            if (
                server.server_type == MCPServerType.LOCAL
                and server.config
                and server.config.command in _LAUNCHERS
                and not server.config.args
            ):
                pkg_name = server.pkg or server.name
                server.config.args = ["--from", pkg_name, pkg_name]
                migrated = True
                logger.info(
                    "Fixed empty args for local MCP server '%s' → %s %s",
                    server.name, server.config.command, server.config.args,
                )
        if migrated:
            self._model.save()

    _DEFAULT_SKILLS = [
        Skill(
            id="fetch-url",
            name="fetch-url",
            description="Use when the user requests to find information from a URL — extracts data from the provided URL and returns it.",
            content="when user requests information from an url use fetch tool extract information from the URL\n\n```mcp\nfetch.fetch(url)\n```",
        ),
        Skill(
            id="file-read",
            name="file-read",
            description="Use when the user wants to read, view, or inspect the contents of a file.",
            content="read the file at the given path and return its content\n\n```mcp\nsmartloop.file-read(path)\n```",
        ),
        Skill(
            id="file-write",
            name="file-write",
            description="Use when the user wants to write, save, or create a file with specific content.",
            content="write content to a file at the given path\n\n```mcp\nsmartloop.file-write(path, content)\n```",
        ),
        Skill(
            id="bash",
            name="bash",
            description="Use when the user wants to run a shell command, execute a script, or perform system operations.",
            content="execute the shell command on the host machine\n\n```mcp\nsmartloop.bash(command)\n```",
        ),
    ]

    def _setup_project_tools(self, project: "Project") -> None:
        """Install system MCP servers, generate workspace docs, and register default skills."""
        project_id = project.id
        self._migrate_servers_to_local(project)
        self.install_system_tools_server(project_id)
        self.install_fetch_server(project_id)
        # Re-fetch: install_* may have replaced self._model via _ensure_model(),
        # so the passed `project` object can be stale (different workspace instance).
        project = self._get_project(project_id)
        self._install_default_skills(project)

    def _install_default_skills(self, project: "Project") -> None:
        """Write default skill SKILL.md files if they don't already exist."""
        existing = {s.name for s in self._read_skills_from_disk(project)}
        for skill in self._DEFAULT_SKILLS:
            if skill.name not in existing:
                self._write_skill_md(project, skill)
                logger.info(f"Installed default skill '{skill.name}' on project {project.id}")

    def _get_system_project(self) -> Optional["Project"]:
        """Return the system project (general_chat), if it exists."""
        workspace = self._model.workspace
        return next((p for p in workspace.projects if p.system), None)

    def get_project_skills(self) -> List[Skill]:
        """Get project skills for the current project (read from disk)."""
        if self._model is None:
            self._ensure_model()
        if self._model is None:
            return []
        try:
            current_project = self._model.workspace.current_project
            if current_project and current_project.name:
                return current_project.get_skills()
        except Exception as e:
            logger.warning(f"Could not get project skills: {e}")
        return []

    def add_skill(
        self,
        project_id: str,
        content: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> dict:
        """Add a skill to a project and persist as SKILL.md."""
        project = self._get_project(project_id)

        body, meta = self._parse_skill_content(content)
        final_description = description or meta.get("description")

        resolved_name = name or meta.get("name")
        already_complete = resolved_name and final_description

        if already_complete:
            generated = {}
        else:
            if self._llm is None:
                self.load_model()
            generated = self._generate_skill_metadata(body, project)

        if generated.get("body"):
            body = generated["body"]

        mcp_block = generated.get("mcp_block")
        if mcp_block and "```mcp" not in body:
            body = body.rstrip() + "\n\n" + mcp_block

        raw_name = resolved_name or generated.get("name") or self._extract_skill_name(body)
        sanitized_name = re.sub(r"[^a-z0-9]+", "-", raw_name.lower()).strip("-") or f"skill-{str(uuid4())[:8]}"

        skill = Skill(
            id=sanitized_name,
            name=sanitized_name,
            description=final_description or generated.get("description"),
            content=body,
        )

        self._write_skill_md(project, skill)
        skill.content = self._render_skill_content(skill)
        return skill.model_dump()

    @staticmethod
    def _parse_skill_content(raw: str) -> tuple[str, dict]:
        """Split raw SKILL.md text into (body, frontmatter_fields).

        If the text starts with ``---`` frontmatter, the metadata fields
        (description, tools) are extracted and the body is returned without
        the fences.  Otherwise the full text is returned as-is.
        """
        meta: dict = {}
        body = raw
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                body = parts[2].strip()
                for line in parts[1].strip().splitlines():
                    if line.startswith("name:"):
                        meta["name"] = line[len("name:"):].strip()
                    elif line.startswith("description:"):
                        meta["description"] = line[len("description:"):].strip()
                    elif line.startswith("tools:"):
                        tools_str = line[len("tools:"):].strip().strip("[]")
                        meta["tools"] = [t.strip() for t in tools_str.split(",") if t.strip()]
        return body, meta

    def update_skill(self, project_id: str, skill_id: str, content: str) -> dict:
        """Update a skill on disk.

        *content* is the full SKILL.md text including YAML frontmatter
        (name, description, tools).
        """
        project = self._get_project(project_id)
        if not project.name:
            raise ValueError(f"Skill not found: {skill_id}")

        # Find existing skill so we can detect renames.
        old_skill = None
        for s in self._read_skills_from_disk(project):
            if s.name == skill_id or s.id == skill_id:
                old_skill = s
                break
        if old_skill is None:
            raise ValueError(f"Skill not found: {skill_id}")

        old_name = old_skill.name

        # Parse frontmatter from the incoming content.
        body, meta = self._parse_skill_content(content)
        new_name = meta.get("name", old_name)

        skill = Skill(
            id=new_name,
            name=new_name,
            description=meta.get("description", old_skill.description),
            content=body,
        )

        # If name changed, remove the old directory.
        if new_name != old_name:
            old_dir = os.path.join(self._skills_dir(project), old_name)
            if os.path.isdir(old_dir):
                shutil.rmtree(old_dir)

        self._write_skill_md(project, skill)

        skill.content = self._render_skill_content(skill)
        return skill.model_dump()

    @staticmethod
    def _extract_skill_name(content: str) -> str:
        """Extract a 1-2 word skill name from content as a fallback.

        Looks for markdown headings or the first meaningful line.
        """
        text = markdownify.markdownify(content, strip=["img", "script", "style"])
        match = re.search(r"^#{1,6}\s+(.+)", text, re.MULTILINE)
        if match:
            words = match.group(1).strip().split()[:2]
            return " ".join(words)
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and len(stripped) > 3 and not stripped.startswith("---"):
                words = stripped.split()[:2]
                return " ".join(words)
        return "skill"

    @staticmethod
    def _match_mcp_tool_from_content(
        content: str,
        project: Optional["Project"],
    ) -> Optional[str]:
        """Eagerly scan skill content for mentions of known server/tool names.

        Returns a ready ```mcp\\n...\\n``` block string when a
        single unambiguous tool match is found, otherwise ``None``.

        Matching strategy (highest confidence first):
        1. Exact ``server_name.tool_name`` already present in content.
        2. Tool name mentioned AND server name mentioned in content.
        3. Tool name alone mentioned (unique across all servers).
        4. Server name alone mentioned (single-tool server).
        """
        if not project:
            return None

        content_lower = content.lower()

        # Build index: (server_name, tool_name, schema)
        server_tools: list[tuple[str, str, dict]] = []
        for server in project.mcp_servers:
            if not server.tools or server.id == SYSTEM_TOOLS_SERVER_ID:
                continue
            for tool in server.tools:
                server_tools.append((server.name, tool.name, tool.input_schema or {}))

        if not server_tools:
            return None

        def _call(server_name: str, tool_name: str, schema: dict) -> str:
            required = schema.get("required", [])
            props = schema.get("properties", {})
            if required:
                param_names = required
            else:
                import re as _re
                from smartloop.agents.mcp.argument_extractor import (
                    ID_PARAM_PATTERNS, CURSOR_PARAM_PATTERNS,
                )
                def _is_skip(p: str) -> bool:
                    return (
                        any(_re.match(pat, p) for pat in ID_PARAM_PATTERNS)
                        or any(_re.match(pat, p) for pat in CURSOR_PARAM_PATTERNS)
                    )
                param_names = [p for p in props if not _is_skip(p)]
            # Bare tokens → inferred_args (LLM extracts values at runtime)
            params = ", ".join(param_names)
            return f"```mcp\n{server_name}.{tool_name}({params})\n```"

        def _tool_words_in_content(tool_name: str) -> bool:
            """Check if ALL words of a tool name appear in the content.

            Splits on ``_`` and ``-`` and requires every word (len >= 3) to
            appear individually in the content.  This handles "save issue"
            matching ``save_issue`` even though the exact substring
            ``"save_issue"`` does not appear.
            """
            words = [w for w in re.split(r'[_\-]', tool_name.lower()) if len(w) >= 3]
            return bool(words) and all(w in content_lower for w in words)

        # 1. Exact dotted reference already in content
        for server_name, tool_name, schema in server_tools:
            if f"{server_name.lower()}.{tool_name.lower()}" in content_lower:
                return _call(server_name, tool_name, schema)

        # 2. Both server name AND tool name mentioned (exact substring)
        matches = [
            (sn, tn, sc) for sn, tn, sc in server_tools
            if sn.lower() in content_lower and tn.lower() in content_lower
        ]
        if len(matches) == 1:
            return _call(*matches[0])

        # 2b. Both server name AND tool words mentioned
        if not matches:
            matches = [
                (sn, tn, sc) for sn, tn, sc in server_tools
                if sn.lower() in content_lower and _tool_words_in_content(tn)
            ]
            if len(matches) == 1:
                return _call(*matches[0])

        # 3. Tool name alone — exact substring, unique across all servers
        tool_matches = [
            (sn, tn, sc) for sn, tn, sc in server_tools
            if tn.lower() in content_lower
        ]
        if len(tool_matches) == 1:
            return _call(*tool_matches[0])

        # 3b. Tool words alone — all words present, unique across all servers
        if not tool_matches:
            tool_matches = [
                (sn, tn, sc) for sn, tn, sc in server_tools
                if _tool_words_in_content(tn)
            ]
            if len(tool_matches) == 1:
                return _call(*tool_matches[0])

        # 4. Server name alone — only one tool on that server
        server_matches = [
            (sn, tn, sc) for sn, tn, sc in server_tools
            if sn.lower() in content_lower
        ]
        mentioned_servers = {sn for sn, _, _ in server_matches}
        if len(mentioned_servers) == 1:
            tools_for_server = [(sn, tn, sc) for sn, tn, sc in server_matches]
            if len(tools_for_server) == 1:
                return _call(*tools_for_server[0])

        return None

    @staticmethod
    def _build_mcp_tools_catalogue(project: Optional["Project"]) -> str:
        """Return a formatted list of available MCP tools for the prompt."""
        if not project:
            return ""
        lines = []
        for server in project.mcp_servers:
            # Exclude the internal Smartloop system tools (bash, file-ops, explore)
            # but include fetch and all user-registered servers.
            if not server.tools or server.id == SYSTEM_TOOLS_SERVER_ID:
                continue
            for tool in server.tools:
                desc = tool.description or ""
                # Summarise required parameters from the input schema
                params = ""
                schema = tool.input_schema or {}
                required = schema.get("required", [])
                props = schema.get("properties", {})
                if required:
                    params = ", ".join(
                        f"{p}=<{props.get(p, {}).get('type', 'value')}>"
                        for p in required
                    )
                elif props:
                    params = ", ".join(
                        f"{p}=<{info.get('type', 'value')}>"
                        for p, info in list(props.items())[:4]
                    )
                lines.append(
                    f"  {server.name}.{tool.name}({params}) — {desc}"
                )
        if not lines:
            return ""
        return "Available MCP tools:\n" + "\n".join(lines)

    def _generate_skill_metadata(
        self,
        content: str,
        project: Optional["Project"] = None,
    ) -> dict:
        """Generate skill name, description, elaborated body and optional MCP call block.

        Returns ``{"name": str, "description": str, "body": str, "mcp_block": str | None}``.
        """
        if self._llm is None:
            logger.warning("No LLM available for skill metadata generation")
            return {}

        mcp_catalogue = self._build_mcp_tools_catalogue(project)
        already_has_mcp = "```mcp" in content

        mcp_field = ""
        if mcp_catalogue and not already_has_mcp:
            mcp_field = (
                '- "mcp_call": the single best matching tool call as a plain string '
                'in the exact form server_name.tool_name(param). '
                'The server_name prefix is REQUIRED — always include it. '
                'Copy both the server_name and tool_name EXACTLY as they appear in the '
                'available MCP tools list below (for example, if the list shows '
                'weather.get_forecast, write weather.get_forecast(city=Paris)). '
                'Do NOT wrap them in angle brackets and do NOT invent or rename the server. '
                'Use fixed values for params known from context; bare param names for runtime values. '
                'If the skill says "save" or "create", pick a create/save tool, NOT a list/read tool. '
                'Omit if no tool clearly matches.\n\n'
                f'{mcp_catalogue}\n'
            )

        prompt = (
            "<system>\n"
            "You are a skill generator. Given raw user input, output a JSON object.\n"
            "Reply with ONLY the JSON — no markdown, no explanation.\n"
            "</system>\n"
            "<user>\n"
            "Generate a JSON object with:\n"
            '- "name": 1-2 word lowercase hyphen-separated identifier derived from the input.\n'
            '- "description": one sentence in the format "When the user asks to <action>, '
            '<what this skill does>." that clearly describes when this skill should be '
            "selected and what it accomplishes.\n"
            '- "body": expand the input into clear, structured skill instructions. '
            "Preserve all specific values mentioned. Do NOT invent facts.\n"
            + mcp_field
            + f"User input:\n{content[:2000]}\n"
            "</user>"
        )

        skill_llm = self._get_large_llm() or self._llm
        try:
            with self._inference_lock:
                raw = str(skill_llm.invoke(prompt)).strip()
            logger.info("Skill metadata LLM response: %s", raw[:200])

            # Strip markdown fences
            if raw.startswith("```"):
                raw = "\n".join(ln for ln in raw.split("\n") if not ln.strip().startswith("```")).strip()

            # Extract JSON object
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]

            # Fix invalid JSON escapes that LLMs produce (e.g. \_ \*)
            raw = re.sub(r'\\([^"\\/bfnrtu])', r'\1', raw)
            meta = json.loads(raw)

            # Wrap mcp_call string into a fenced ```mcp block
            mcp_call = meta.pop("mcp_call", None)
            if mcp_call:
                mcp_call = re.sub(r'```\w*', '', mcp_call).strip()
                # The LLM may emit a hallucinated server name. Re-bind the call
                # to the server that actually owns the matching tool.
                mcp_call = self._correct_mcp_call_server(mcp_call, project)
                meta["mcp_block"] = f"```mcp\n{mcp_call}\n```"
            else:
                meta.pop("mcp_block", None)

            return meta
        except Exception as e:
            logger.warning("Failed to generate skill metadata: %s", e)
            return {}

    @staticmethod
    def _correct_mcp_call_server(mcp_call: str, project: Optional["Project"]) -> str:
        """Re-bind a generated ``Server.tool(args)`` call to the real server.

        The skill-generation LLM is free-text and frequently emits a server
        name that does not match any registered MCP server. Since tool names
        are unambiguous within a project, we look the call's tool up in the
        registered servers and, when found, rewrite the server portion to the
        server that actually owns that tool. Returns the call unchanged when
        no registered server exposes the tool.
        """
        if not project:
            return mcp_call

        # Split the call head ("server.tool") from its argument list. The model
        # sometimes wraps the head in angle brackets/backticks (e.g.
        # "<server.tool>(args)") — strip those from the head only, leaving any
        # "<value>" runtime-param markers inside the parens untouched.
        head, sep, tail = mcp_call.partition("(")
        if sep:
            head = head.strip().strip("<>`").strip()
            mcp_call = f"{head}{sep}{tail}"

        # Parse "[server.]tool(args)" — the tool is the identifier directly
        # before the argument list; everything before the last dot, if present,
        # is the (possibly wrong) server name. The model sometimes omits the
        # server prefix entirely, so it is optional here.
        m = re.match(r'^\s*(?:(.+)\.)?([\w-]+)\s*\((.*)\)\s*$', mcp_call, re.DOTALL)
        if not m:
            return mcp_call

        emitted_server, tool_name, args_str = m.group(1), m.group(2), m.group(3)

        # Find the registered server that actually owns this tool.
        for server in project.mcp_servers:
            if not server.tools or server.id == SYSTEM_TOOLS_SERVER_ID:
                continue
            if any(t.name == tool_name for t in server.tools):
                if not emitted_server:
                    logger.info(
                        "Bound generated mcp_call (no server prefix) to server "
                        "'%s' (owns tool '%s')", server.name, tool_name,
                    )
                elif server.name != emitted_server:
                    logger.info(
                        "Re-bound generated mcp_call from server '%s' to '%s' "
                        "(owns tool '%s')",
                        emitted_server, server.name, tool_name,
                    )
                return f"{server.name}.{tool_name}({args_str})"

        logger.warning(
            "Generated mcp_call references tool '%s' not found on any "
            "registered server; leaving as-is", tool_name,
        )
        return mcp_call

    @staticmethod
    def _visible_project_dir(project: "Project") -> str:
        """Return the visible project directory.

        Delegates to ``Project.visible_dir``.
        """
        return normalize_path(project.visible_dir)

    @staticmethod
    def _render_skill_content(skill: Skill) -> str:
        """Render a Skill as full SKILL.md text with YAML frontmatter."""
        lines = ["---", f"name: {skill.name}"]
        if skill.description:
            lines.append(f"description: {skill.description}")
        lines.append("---")
        # Strip any existing frontmatter from content before prepending.
        body, _ = ModelService._parse_skill_content(skill.content)
        return "\n".join(lines) + "\n\n" + body

    @staticmethod
    def _write_skill_md(project: "Project", skill: Skill) -> None:
        """Write a SKILL.md file with YAML frontmatter for the given skill."""
        base_dir = ModelService._visible_project_dir(project)
        skill_dir = os.path.join(base_dir, "skills", skill.name)
        os.makedirs(skill_dir, exist_ok=True)
        skill_path = os.path.join(skill_dir, "SKILL.md")
        Path(skill_path).write_text(ModelService._render_skill_content(skill), encoding="utf-8")

    @staticmethod
    def _skills_dir(project: "Project") -> str:
        """Return the skills directory for a project."""
        return os.path.join(ModelService._visible_project_dir(project), "skills")

    @staticmethod
    def _read_skills_from_disk(project: "Project") -> List[Skill]:
        """Read all skills from the project skills directory.

        Each subdirectory containing a SKILL.md with YAML frontmatter is
        parsed into a ``Skill`` object.
        """
        skills_dir = ModelService._skills_dir(project)
        if not os.path.isdir(skills_dir):
            return []

        skills: List[Skill] = []
        for entry in sorted(os.listdir(skills_dir)):
            entry_path = os.path.join(skills_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            skill_path = os.path.join(entry_path, "SKILL.md")
            if not os.path.isfile(skill_path):
                continue

            try:
                raw = Path(skill_path).read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Could not read skill {skill_path}: {e}")
                continue

            name = entry
            description = None
            content = raw

            if raw.startswith("---"):
                parts = raw.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    content = parts[2].strip()
                    for line in frontmatter.strip().splitlines():
                        if line.startswith("name:"):
                            name = line[len("name:"):].strip()
                        elif line.startswith("description:"):
                            description = line[len("description:"):].strip()

            skills.append(Skill(
                id=name,
                name=name,
                description=description,
                content=raw,
            ))

        return skills

    def get_skill(self, project_id: str, skill_id: str) -> dict:
        """Get a single skill by name from disk."""
        project = self._get_project(project_id)
        for skill in self._read_skills_from_disk(project):
            if skill.name == skill_id or skill.id == skill_id:
                return skill.model_dump()
        raise ValueError(f"Skill not found: {skill_id}")

    def list_skills(self, project_id: str) -> list[dict]:
        """List all skills for a project, read from disk."""
        project = self._get_project(project_id)
        return [s.model_dump() for s in self._read_skills_from_disk(project)]

    def delete_skill(self, project_id: str, skill_id: str) -> dict:
        """Delete a skill by removing its SKILL.md directory from disk."""
        project = self._get_project(project_id)

        # Verify it exists on disk
        skill = None
        for s in self._read_skills_from_disk(project):
            if s.name == skill_id or s.id == skill_id:
                skill = s
                break
        if skill is None:
            raise ValueError(f"Skill not found: {skill_id}")

        skill_dir = os.path.join(self._skills_dir(project), skill.name)
        if os.path.isdir(skill_dir):
            shutil.rmtree(skill_dir)
            logger.info(f"Deleted skill directory: {skill_dir}")

        return {"id": skill_id, "deleted": True, "message": "Skill deleted successfully"}

    def get_project_info(self) -> Optional[str]:
        """Build a project info summary for context injection."""
        if self._model is None:
            return None

        project = self._model.workspace.current_project
        if not project:
            return None

        lines = []
        lines.append(f"Project: {project.name or project.id}")
        if project.description:
            lines.append(f"Description: {project.description}")

        # Documents summary — only list indexed documents, omit status
        docs = [d for d in project.documents if d.status == DocumentStatus.INDEXED]
        if docs:
            lines.append(f"\nDocuments ({len(docs)}):")
            for doc in docs:
                doc_name = Path(doc.path).name
                lines.append(f"  - {doc_name}")

        # MCP servers & tools summary
        servers = [s for s in project.mcp_servers if s.enabled and not s.system]
        lines.append(f"\nMCP Servers ({len(servers)}):")
        for srv in servers:
            tool_names = [t.name for t in srv.tools]
            desc = f" - {srv.description}" if srv.description else ""
            lines.append(f"  - {srv.name}{desc}")
            if tool_names:
                lines.append(f"    Tools: {', '.join(tool_names)}")

        return "\n".join(lines)

    def get_document_context(self, query: Optional[str] = None) -> Optional[str]:
        """Perform vector search on indexed documents and return relevant chunks.

        Returns None when there are no documents, no relevant results, or
        the results don't address the user's intent — allowing the caller
        to fall back to local explore or the model's trained knowledge.
        """
        result = self.get_document_context_with_citations(query)
        if result is None:
            return None
        # Result is (context, citations)
        return result[0]

    def _vector_search_grouped(
        self, query: Optional[str] = None, skill_instruction: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[tuple]:
        """Core vector search → ``(doc_groups, citations)`` or ``None``.

        ``doc_groups`` is an ``OrderedDict`` mapping document name → combined
        source text for that document, ordered by relevance.  ``citations``
        is a list of per-document citation dicts.

        Shared by :meth:`get_document_context_with_citations` (which joins
        the groups into one string) and :meth:`get_document_groups` (which
        exposes them per-document so the orchestrator can fan out one DAG
        node per document/batch and reduce them into a single result).

        Summaries are stored as vector content at index time, so semantic
        search matches against summaries.  The original text is retrieved
        from ``source_content`` metadata — no LLM calls are needed.

        Args:
            query: The user's search query.
            skill_instruction: Optional skill instruction that refines the
                search query toward the data the skill needs.
            conversation_history: Optional conversation history used to
                enrich the search query with prior context.
        """
        if self._model is None or not query:
            logger.info("[vector-search] Skipped — no model or query")
            return None

        try:
            workspace = self._model.workspace
            current_project = workspace.current_project
            if not current_project:
                logger.info("[vector-search] Skipped — no active project")
                return None

            # Reuse a single store per project to avoid SQLite
            # lock conflicts with the background indexer.
            if self._doc_store is None or self._doc_store_project_id != current_project.id:
                self._doc_store = StoreFactory.create_store(
                    current_project.project_dir, logger=logger, base_dir=workspace.workspace_dir
                )
                self._doc_store_project_id = current_project.id
            store = self._doc_store

            search_query = query
            if conversation_history:
                search_query = self._build_history_search_query(query, conversation_history)

            # Semantic discovery: find which documents are relevant.
            # The goal is document-level recall — we need enough chunks
            # to cover all potentially relevant documents, not just the
            # top few.  With multi-chunk documents a small n_results
            # can be dominated by chunks from 1-2 documents, hiding
            # the rest.  Use a generous limit; the expansion step below
            # deduplicates by document_id anyway.
            indexed_doc_count = len([
                doc for doc in (current_project.documents or [])
                if doc.status in (DocumentStatus.INDEXED, DocumentStatus.PENDING)
            ])
            n_results = max(indexed_doc_count * 5, 20)

            logger.info(
                f"[vector-search] Querying {indexed_doc_count} indexed document(s) "
                f"with: {search_query[:120]}"
            )

            discovery: List[ChunkResult] = store.query_with_metadata(
                search_query, n_results=n_results, max_distance=0.8,
            )

            # Fallback: if the enriched query missed, retry with
            # just the raw user query as a safety net.
            if not discovery and search_query != query:
                logger.info(
                    f"[vector-search] Enriched query returned no results, "
                    f"retrying with raw query: {query[:120]}"
                )
                discovery = store.query_with_metadata(
                    query, n_results=n_results, max_distance=0.8,
                )

            if not discovery:
                logger.info("[vector-search] No chunks found")
                return None

            # Absolute quality floor: if even the closest chunk is only
            # weakly related, treat the whole result set as no match.
            # Without this gate, weak matches populate vector_context and
            # force the orchestrator's tier-1 classifier onto the EXPLORE
            # path (planner.py `has_document_context` check), so a purely
            # conversational prompt gets wrapped in DOCUMENT CONTEXT
            # scaffolding pointing at irrelevant docs.
            _MIN_BEST_DISTANCE = 0.55
            best_distance = discovery[0].distance
            if best_distance > _MIN_BEST_DISTANCE:
                logger.info(
                    f"[vector-search] Best match distance {best_distance:.3f} "
                    f"exceeds quality floor {_MIN_BEST_DISTANCE} — "
                    f"no relevant documents"
                )
                return None

            # Relevance cutoff: keep only chunks within a reasonable
            # margin of the best match.  This avoids pulling in weakly
            # related documents when a few strong matches exist.
            _DISTANCE_CUTOFF_MARGIN = 0.15
            cutoff = best_distance + _DISTANCE_CUTOFF_MARGIN
            before_count = len(discovery)
            discovery = [r for r in discovery if r.distance <= cutoff]
            if len(discovery) < before_count:
                logger.info(
                    f"[vector-search] Cutoff {cutoff:.3f} "
                    f"(best={best_distance:.3f} + {_DISTANCE_CUTOFF_MARGIN}) "
                    f"trimmed {before_count - len(discovery)} chunk(s)"
                )

            # Log distances to help diagnose relevance issues
            unique_docs = list({r.document_name or r.document_id for r in discovery})
            dist_summary = ", ".join(
                f"{r.document_name or r.document_id}={r.distance:.3f}"
                for r in discovery[:6]
            )
            logger.info(
                f"[vector-search] Discovery returned {len(discovery)} chunk(s) "
                f"from {len(unique_docs)} document(s) — "
                f"distances: [{dist_summary}]"
            )

            # Build a distance map from discovery: best (lowest) distance
            # per document so we can order everything by relevance.
            doc_distance: dict[str, float] = {}
            doc_id_distance: dict[str, float] = {}
            for r in discovery:
                key = r.document_name or r.document_id
                if key not in doc_distance or r.distance < doc_distance[key]:
                    doc_distance[key] = r.distance
                if r.document_id:
                    if r.document_id not in doc_id_distance or r.distance < doc_id_distance[r.document_id]:
                        doc_id_distance[r.document_id] = r.distance

            # Expansion: fetch ALL chunks from every matched document.
            # Nothing is dropped — all documents are passed through.
            # Documents are fetched in parallel batches ordered by
            # similarity so the most relevant documents are ready first.
            matched_doc_ids = list({r.document_id for r in discovery if r.document_id})
            # Sort doc IDs by similarity (closest first).
            matched_doc_ids.sort(key=lambda d: doc_id_distance.get(d, float("inf")))

            _DOC_BATCH_SIZE = 4
            chunk_results: List[ChunkResult] = []
            remaining = list(matched_doc_ids)
            from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed
            with _TPE(max_workers=4) as pool:
                while remaining:
                    batch = remaining[:_DOC_BATCH_SIZE]
                    remaining = remaining[_DOC_BATCH_SIZE:]
                    futures = {
                        pool.submit(store.get_by_document_ids, [doc_id]): doc_id
                        for doc_id in batch
                    }
                    for fut in as_completed(futures):
                        chunk_results.extend(fut.result())

            if not chunk_results:
                chunk_results = discovery

            if not chunk_results:
                return None

            # Group source text by document, ordered by relevance.  Both the
            # concatenated-context and per-document (fan-out) callers build
            # their output from these groups.
            from collections import OrderedDict
            grouped: "OrderedDict[str, list]" = OrderedDict()
            for r in chunk_results:
                key = r.document_name or r.document_id
                content = r.source_content or r.text
                grouped.setdefault(key, []).append(content)
            doc_groups: "OrderedDict[str, str]" = OrderedDict(
                (name, "\n".join(texts)) for name, texts in grouped.items()
            )

            # Build citations directly from chunk metadata — no separate
            # citation field needed.  Deduplicate by document, accumulate
            # page numbers from all matched chunks, ordered by similarity.
            seen_docs: dict = {}  # document_id -> citation dict
            for chunk in discovery:
                if not chunk.document_id:
                    continue
                meta = chunk.metadata or {}
                page = meta.get("page_number")
                if chunk.document_id not in seen_docs:
                    citation = {
                        "document_id": chunk.document_id,
                        "document_name": chunk.document_name or meta.get("document_name"),
                    }
                    if page:
                        citation["page_numbers"] = [page]
                    seen_docs[chunk.document_id] = citation
                elif page:
                    existing = seen_docs[chunk.document_id]
                    pages = existing.get("page_numbers", [])
                    if page not in pages:
                        pages.append(page)
                        existing["page_numbers"] = sorted(pages)

            # seen_docs preserves discovery (similarity) order.
            all_citations = list(seen_docs.values())
            return doc_groups, all_citations
        except _PROGRAMMING_ERRORS:
            raise
        except Exception as e:
            logger.warning(f"Vector search failed: {e}", exc_info=True)

        return None

    def get_document_context_with_citations(
        self, query: Optional[str] = None, skill_instruction: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[tuple]:
        """Vector search → ``(context_str, citations_list)`` or ``None``.

        Joins all matched documents into a single context string.  Returns
        ``None`` when there are no documents or no relevant results.
        """
        result = self._vector_search_grouped(
            query, skill_instruction, conversation_history,
        )
        if not result:
            return None
        doc_groups, citations = result
        context = "\n\n---\n\n".join(
            f"=== Document: {name} ===\n{content}"
            for name, content in doc_groups.items()
        )
        return context, citations

    def get_document_groups(
        self, query: Optional[str] = None, skill_instruction: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[tuple]:
        """Vector search → ``(documents, citations)`` for fan-out, or ``None``.

        ``documents`` is a list of ``{"document_name", "content"}`` dicts —
        one entry per matched document, ordered by relevance — so the
        orchestrator can process each document (or batch) in its own DAG
        node and reduce them into a single synthesized result.
        """
        result = self._vector_search_grouped(
            query, skill_instruction, conversation_history,
        )
        if not result:
            return None
        doc_groups, citations = result
        documents = [
            {"document_name": name, "content": content}
            for name, content in doc_groups.items()
        ]
        return documents, citations

    def _build_skill_search_query(self, user_query: str, skill_instruction: str) -> str:
        """Use the LLM to synthesize a vector-search query from the user
        message and the matched skill instruction.

        The generated query targets the *data* the skill needs rather than
        the user's high-level request.  For example:
          user_query:        "generate a report"
          skill_instruction: "create a table of unique items and total
                              quantity purchased from the invoices"
          result:            "invoice items quantities purchased totals"
        """
        prompt = (
            "Given the user's request and a skill instruction, generate a short "
            "search query (max 15 words) that will find the relevant document "
            "content in a vector database. Focus on the data and entities the "
            "skill needs, not the action the user asked for.\n\n"
            f"User request: {user_query}\n"
            f"Skill instruction: {skill_instruction}\n\n"
            "Search query:"
        )
        try:
            result = self.generate_completion(prompt).strip().strip('"\'')
            if result:
                logger.info(f"[skill-search] Rewrote query: {user_query!r} → {result!r}")
                return result
        except Exception as e:
            logger.warning(f"[skill-search] Query rewrite failed: {e}")
        return user_query

    # Reference words that signal the latest message depends on prior turns
    # (pronouns / ellipsis).  Only then does history need to resolve the query.
    _REFERENCE_WORDS = frozenset({
        "it", "its", "this", "that", "these", "those", "they", "them", "their",
        "he", "she", "him", "her", "his", "hers", "one", "ones", "same",
        "above", "previous", "former", "latter", "then",
    })

    def _build_history_search_query(
        self, user_query: str, conversation_history: List[Dict[str, str]]
    ) -> str:
        """Resolve references in the search query using conversation history.

        Conversation history is **context only** — it must not bias or narrow
        the current query toward earlier topics.  A self-contained message is
        used verbatim; history is consulted solely to resolve references
        (pronouns, omitted subjects) in short/ambiguous messages.

        The *conversation_history* is expected to be pre-filtered for
        relevance by the caller — no additional windowing is applied.
        """
        if len(conversation_history) <= 1:
            return user_query

        # Self-contained query → use as-is.  Only rewrite when the message is
        # short or actually references prior turns, so history can't drag in
        # entities/topics the user didn't ask about this turn.
        words = re.findall(r"[a-zA-Z']+", user_query.lower())
        has_reference = bool(set(words) & self._REFERENCE_WORDS)
        if len(words) >= 5 and not has_reference:
            return user_query

        history_text = "\n".join(
            f"{turn['role']}: {turn['content'][:200]}"
            for turn in conversation_history if turn.get("content")
        )
        if not history_text.strip():
            return user_query

        prompt = (
            "Rewrite the user's latest message into a short document-search "
            "query (max 15 words).\n"
            "- If the latest message is already self-contained, return it "
            "UNCHANGED.\n"
            "- Use the conversation history ONLY to resolve references in the "
            "latest message (pronouns like 'it'/'that'/'those', or an omitted "
            "subject).\n"
            "- Do NOT add names, entities, or topics from the history that the "
            "latest message does not refer to. The latest message defines what "
            "to search for; history is only background context.\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Latest message: {user_query}\n\n"
            "Search query:"
        )
        try:
            result = self.generate_completion(prompt).strip().strip('"\'')
            if result:
                logger.info(
                    f"[history-search] Rewrote query: {user_query!r} → {result!r}"
                )
                return result
        except Exception as e:
            logger.warning(f"[history-search] Query rewrite failed: {e}")
        return user_query

    def count_tokens(self, text: str) -> int:
        """Count tokens in *text* using the loaded LLM's tokenizer.

        Returns 0 when no model is loaded so callers always get a safe
        integer back.
        """
        if self._llm is None:
            return 0
        try:
            return self._llm.get_num_tokens(text)
        except Exception:
            return 0

    def get_max_context_tokens(self) -> int:
        """Return the model's maximum context window size in tokens.

        Returns 0 when no model is loaded.
        """
        if self._llm is None:
            return DEFAULT_MAX_CONTEXT_TOKENS
        try:
            return self._llm.get_max_tokens()
        except Exception:
            return DEFAULT_MAX_CONTEXT_TOKENS

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded"""
        return self._llm is not None

    def _llm_for_response(self, capability: Optional[str]) -> Optional[BaseLLM]:
        """Return the LLM to use for the final response.

        Fallback chain when *capability* is set: capability-tagged model →
        largest downloaded model → default orchestrator LLM.  Small models
        struggle to synthesize long tool-result contexts, so we prefer any
        larger model when one is available; falls back cleanly to
        ``self._llm`` when nothing better is present.
        """
        if capability:
            cap_llm = self._get_llm_for_capability(capability)
            if cap_llm is not None and cap_llm is not self._llm:
                logger.info(
                    "[response] Using capability='%s' LLM for final response",
                    capability,
                )
                return cap_llm
            large_llm = self._get_large_llm()
            if large_llm is not None and large_llm is not self._llm:
                logger.info(
                    "[response] Capability='%s' unavailable — falling back to large LLM",
                    capability,
                )
                return large_llm
            logger.info(
                "[response] Capability='%s' unavailable and no larger model — using default LLM",
                capability,
            )
        return self._llm

    def stream_completion(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        cancelled: Optional[threading.Event] = None,
        capability: Optional[str] = None,
    ) -> Iterator[GenerationChunk]:
        """
        Stream completion from the LLM.

        Uses a lock to ensure thread-safe access to llama-cpp which
        does not support concurrent inference.  The caller must call
        ``load_model()`` before invoking this method.

        Args:
            prompt: The formatted prompt string
            model_name: Unused, kept for caller compatibility
            cancelled: Optional event that signals the caller wants to
                       abort inference early.  Checked between chunks.
            capability: When set, prefer the LLM matching this capability
                        over ``self._llm`` for the final response.

        Yields:
            GenerationChunk objects from the LLM
        """
        llm = self._llm_for_response(capability)
        if llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Propagate cancellation event to the LLM so it can stop mid-inference
        llm._cancelled = cancelled

        # Lock around inference - llama-cpp is not thread-safe
        with self._inference_lock:
            try:
                for chunk in llm.stream(prompt):
                    if cancelled is not None and cancelled.is_set():
                        break
                    yield chunk
            finally:
                llm._cancelled = None

    def generate_completion(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        capability: Optional[str] = None,
    ) -> str:
        """
        Generate a non-streaming completion.

        The caller must call ``load_model()`` before invoking this method.

        Args:
            prompt: The formatted prompt string
            model_name: Unused, kept for caller compatibility
            capability: When set, prefer the LLM matching this capability
                        over ``self._llm`` for the final response.

        Returns:
            The generated text response
        """
        llm = self._llm_for_response(capability)
        if llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        with self._inference_lock:
            return llm.invoke(prompt)

    def add_document(self, source: str, project_id: str | None = None, progress_callback=None) -> dict:
        """Add a document source (file path or URL) to a project (by ID or current).

        Delegates to Model.add_document() which processes, indexes, and saves.
        Returns info about the added documents.
        """
        if project_id:
            project = self._get_project(project_id)
            # Set as current temporarily so Model.add_source works against it
            for p in self._model.workspace.projects:
                p.current = (p.id == project_id)
        new_docs = self._model.add_source(source, progress_callback=progress_callback)
        return {
            "added": [doc.to_info().model_dump() for doc in (new_docs or [])]
        }

    def add_document_async(self, source: str, project_id: str | None = None, cwd: str | None = None, generate_filename: bool = False) -> dict:
        """Add a document and return immediately with status=pending.

        The document is registered in the project with PENDING status.
        It is pushed onto the indexer queue so processing starts
        immediately rather than waiting for the next config-poll cycle.
        """
        self._initialize_workspace()

        if project_id:
            project = self._get_project(project_id)
            for p in self._model.workspace.projects:
                p.current = (p.id == project_id)
        else:
            project = self._model.workspace.current_project
            if not project:
                raise ValueError("No active project")

        # Register the document with PENDING status (no heavy processing).
        new_docs = self._model.slp_model.add_document(source, cwd=cwd, generate_filename=generate_filename)

        if not new_docs:
            return {
                "status": "skipped",
                "message": "Document(s) already exist in the project",
                "documents": [],
            }

        # Enqueue BEFORE saving config so that when the watcher detects
        # the config change the queue already contains the document IDs.
        indexer = self._task_queue.get_task("indexer")
        if indexer is not None:
            indexer.enqueue(new_docs)

        # Persist config so the watcher also sees the change as a fallback.
        self._model.save()

        return {
            "status": "pending",
            "message": f"{len(new_docs)} document(s) queued for processing",
            "documents": [doc.to_info().model_dump() for doc in new_docs],
        }

    def upload_document_to_inbox(self, project_id: str, filename: str, file_bytes: bytes, cwd: str = None) -> dict:
        """Drop an uploaded file into the project's documents inbox folder.

        The indexer's ``_scan_documents`` loop picks it up, registers it in
        the workspace config, and starts processing automatically.
        Returns immediately with ``status=accepted``.
        """
        project = self._get_project(project_id)
        if cwd and not project.cwd:
            project.cwd = cwd
            self._model.workspace.save()
        inbox_dir = Path(self._visible_project_dir(project)) / "documents"
        inbox_dir.mkdir(parents=True, exist_ok=True)

        safe_name = Path(filename).name or "upload"
        dest = inbox_dir / safe_name
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            counter = 1
            while dest.exists():
                dest = inbox_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        dest.write_bytes(file_bytes)
        logger.info(f"[upload] Dropped {dest.name} into inbox for project {project_id}")
        return {"status": "accepted", "filename": dest.name, "path": str(dest)}

    def get_document_statuses(self, doc_ids: list[str], project_id: str | None = None) -> list[DocumentInfo]:
        """Return fresh status for each document ID by re-reading the workspace from disk.

        Always reads from disk so the caller gets the latest state written by
        the indexer background thread.
        """
        workspace = Workspace.load()
        if workspace is None:
            return [DocumentInfo(id=did, status="unknown") for did in doc_ids]

        if project_id:
            project = next((p for p in workspace.projects if p.id == project_id), None)
        else:
            project = workspace.current_project

        if project is None:
            return [DocumentInfo(id=did, status="unknown") for did in doc_ids]

        doc_map = {d.id: d for d in project.documents}
        return [
            doc_map[did].to_info() if did in doc_map else DocumentInfo(id=did, status="unknown")
            for did in doc_ids
        ]

    # --------------------------------------------------------------- Task Queue

    def _ensure_downloader_task(self, model_id: str, progress_callback=None) -> DownloaderTask:
        """Register and return the downloader task."""
        existing = self._task_queue.get_task("downloader")
        if existing is not None:
            return existing

        workspace = self._model.workspace if self._model else self._ensure_workspace()

        downloader_cls = SLPArtifactDownloader if credential_store.has_credentials() else HFDownloader
        downloader = downloader_cls(workspace, logger)

        task = DownloaderTask(
            workspace=workspace,
            downloader=downloader,
            logger=logger,
            model_id=model_id,
            progress_callback=progress_callback,
        )
        self._task_queue.register(task)
        return task

    def setup_tasks(self, workspace: Optional[Workspace] = None) -> None:
        """Set up all tasks for the workspace and start auto tasks.

        Called during API server lifespan startup.
        """
        if workspace is None:
            workspace = self._model.workspace if self._model else self._ensure_workspace()

        # Migrate legacy ChromaDB data into the SQLite IndexStore — but ONLY
        # when sqlite is the active backend.  This migration reads (and then
        # archives) chroma.sqlite3; running it while ChromaDB is the active
        # backend would move the data out from under the store that reads it.
        from smartloop.db.store_factory import StoreType, _resolve_default_store_type
        if _resolve_default_store_type() == StoreType.SQLite:
            from smartloop.db.migrate_chroma import migrate_chroma_to_index_store
            for project in workspace.projects:
                if migrate_chroma_to_index_store(project.project_dir):
                    logger.info(f"Migrated ChromaDB data for project '{project.name or project.id}'")

        # Document conversion now uses MarkItDown (lightweight, no model
        # pre-download).  OCR / image understanding is handled on demand by the
        # vision-capable model loaded in-process via get_vision_llm().

        # Indexer task (auto=True — watches for pending documents)
        indexer = IndexerTask(workspace=workspace, logger=logger)
        self._task_queue.register(indexer)

        # Build task (auto=False — triggered explicitly)
        if BuildTask is not None:
            build = BuildTask(workspace=workspace, logger=logger)
            self._task_queue.register(build)

        # Train task (auto=False — triggered explicitly)
        if TrainTask is not None:
            train = TrainTask(workspace=workspace, logger=logger)
            self._task_queue.register(train)

        # Start auto tasks (indexer watches for changes)
        self._task_queue.start_auto_tasks()

        # Start the indexer watcher thread
        indexer.start_watcher()

        logger.info("All tasks initialized and auto-tasks started")

    def shutdown_tasks(self) -> None:
        """Gracefully shut down all tasks.  Called during API server shutdown."""
        self._task_queue.shutdown()
        self._reset_llm_executor()

    def delete_document(self, document_id: str, project_id: str | None = None) -> dict:
        """Delete a document by ID from a project (by ID or current).

        Removes from vector store, deletes parquet and source files,
        removes from project document list, and saves workspace.
        """
        if project_id:
            current_project = self._get_project(project_id)
        else:
            if self._model is None:
                raise RuntimeError("Model not loaded")
            current_project = self._model.workspace.current_project
            if not current_project:
                raise ValueError("No current project found")

        # Find the document
        doc = next((d for d in current_project.documents if d.id == document_id), None)
        if doc is None:
            raise ValueError(f"Document not found: {document_id}")

        # 1. Delete from vector store
        store = StoreFactory.create_store(current_project.project_dir, logger=logger, base_dir=self._model.workspace.workspace_dir)
        store.delete_document(document_id)

        # 2. Delete parquet file
        parquet_path = Path(current_project.project_dir) / "datasets" / f"{document_id}.parquet"
        if parquet_path.exists():
            parquet_path.unlink()
            logger.info(f"Deleted parquet: {parquet_path}")

        # 3. Delete the original file from the user-visible documents folder
        if doc.path and Path(doc.path).exists():
            Path(doc.path).unlink()
            logger.info(f"Deleted document file: {doc.path}")

        # 4. Remove from the JSONL manifest
        from smartloop.tasks.documents_manifest import DocumentsManifest
        manifest = DocumentsManifest(Path(current_project.project_dir) / "documents.jsonl")
        manifest.remove(document_id)

        # 5. Remove from project documents list
        current_project.documents = [d for d in current_project.documents if d.id != document_id]

        # 6. Save workspace
        self._model.save()

        return {"id": document_id, "deleted": True, "message": "Document deleted successfully"}

    def get_documents(self) -> list[dict]:
        """List all documents in the current project."""
        if self._model is None:
            raise RuntimeError("Model not loaded")

        current_project = self._model.workspace.current_project
        if not current_project:
            return []

        return [doc.to_info().model_dump() for doc in current_project.documents]

    def get_documents_for_project(self, project_id: str) -> list[dict]:
        """List documents for a project.

        Reads the visible ~/smartloop/<project-name>/documents/ folder as the
        source of truth for which files exist, then enriches each entry with
        status information from the workspace config.  Files that have been
        dropped into the folder but not yet picked up by the indexer appear
        with status=pending and no id.
        """
        project = self._get_project(project_id)
        visible_dir = Path(self._visible_project_dir(project)) / "documents"

        # Index registered docs by their original inbox path for fast lookup.
        # The indexer may update doc.path to a converted sources/ path, so we
        # also track the original filename so we can match on that.
        registered_by_path: dict[str, "Document"] = {}
        registered_by_name: dict[str, "Document"] = {}
        for doc in project.documents:
            registered_by_path[doc.path] = doc
            # Use doc.name (original filename) for lookup; fall back to path stem
            lookup_name = doc.name or Path(doc.path).name
            registered_by_name[lookup_name] = doc

        results: list[dict] = []
        seen_ids: set[str] = set()

        if visible_dir.is_dir():
            for file_path in sorted(visible_dir.iterdir()):
                if not file_path.is_file():
                    continue
                doc = registered_by_path.get(str(file_path)) or registered_by_name.get(file_path.name)
                if doc:
                    seen_ids.add(doc.id)
                    results.append(doc.to_info().model_dump())
                else:
                    # File is in the inbox but not yet registered by the indexer
                    results.append({
                        "id": None,
                        "path": str(file_path),
                        "filename": file_path.name,
                        "status": "pending",
                        "summary": None,
                    })

        # Append any registered docs whose source is not in the visible folder
        # (e.g. URLs or external paths added before this change).
        for doc in project.documents:
            if doc.id not in seen_ids:
                results.append(doc.to_info().model_dump())

        return results

    def create_project(self, name: str, description: Optional[str] = None, system: bool = False, skills: Optional[List[Skill]] = None, cwd: Optional[str] = None) -> dict:
        """Create a new project with the given name and description.

        Ensures the workspace exists and creates the project entry.
        The project uses the workspace's current active model.
        Model download is handled separately via the ``/v1/init`` endpoint.
        """
        self._ensure_model()

        workspace = self._model.workspace

        project_id = str(uuid4())
        project_dir = os.path.join(self._model.workspace.workspace_dir, project_id)
        os.makedirs(project_dir, exist_ok=True)

        # Unset current on all existing projects
        for p in self._model.workspace.projects:
            p.current = False

        project = Project(
            id=project_id,
            project_dir=project_dir,
            current=True,
            name=name,
            description=description,
            system=system,
            skills=skills or [],
        )
        self._model.workspace.projects.append(project)
        self._model.save()

        # Working directory for the project's file/bash tools.
        # If the caller supplied a cwd (e.g. the directory the CLI was invoked
        # from), use it directly so the project operates on the user's actual
        # folder. Otherwise fall back to a fresh ~/smartloop/<name> folder
        # (unique-ified to <name>_1, <name>_2, … so we never reuse stale
        # content from a deleted/same-named project).
        if cwd:
            visible_dir = os.path.abspath(os.path.expanduser(cwd))
        else:
            visible_dir, _ = resolve_visible_dir(name)
        project.cwd = visible_dir
        self._model.save()
        os.makedirs(visible_dir, exist_ok=True)
        logger.info(f"[create_project] Visible dir: {visible_dir}")

        # Reset cached indexers/stores so they get re-created for the new project
        self._workspace_indexer = None
        self._feedback_indexer = None
        self._doc_store = None
        self._doc_store_project_id = None

        # Initialize the index store so the schema is ready before
        # the first query.
        try:
            from smartloop.db.store_factory import StoreFactory
            store = StoreFactory.create_store(
                project_dir=project_dir,
                logger=logger,
                base_dir=self._model.workspace.workspace_dir,
            )
            logger.info(f"Initialized vector store for project {project_id}")
        except Exception as e:
            logger.warning(f"Could not initialize vector store: {e}")

        self._setup_project_tools(project)

        # Write explicitly provided skills to disk
        if skills:
            for skill in skills:
                self._write_skill_md(project, skill)

        Workspace.reset()

        return self._project_to_dict(project)

    def update_project(
        self,
        project_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        skills: Optional[List[Skill]] = None,
    ) -> dict:
        """Update an existing project."""
        self._ensure_model()

        workspace = self._model.workspace
        project = next((p for p in workspace.projects if p.id == project_id), None)
        if project is None:
            raise ValueError(f"Project not found: {project_id}")

        # Rename the visible project directory when the name changes.
        if name and name != project.name:
            old_visible = self._visible_project_dir(project)
            new_visible = os.path.expanduser(
                os.path.join("~/smartloop", name)
            )
            if os.path.isdir(old_visible) and not os.path.exists(new_visible):
                os.rename(old_visible, new_visible)
                logger.info(
                    f"Renamed visible project dir: {old_visible} -> {new_visible}"
                )
            project.name = name
            project.cwd = new_visible

        if description is not None:
            project.description = description

        if skills is not None:
            # Write skills to disk rather than config.json
            for skill in skills:
                self._write_skill_md(project, skill)

        self._model.save()

        return self._project_to_dict(project)

    def _project_to_dict(self, p: Project) -> dict:
        """Convert a Project to a dict for API responses."""
        skills = self._read_skills_from_disk(p) if p.name else []
        visible = self._visible_project_dir(p)
        return {
            "id": p.id,
            "name": p.name,
            "cwd": visible,
            "description": p.description,
            "model_name": self._model_name,
            "current": p.current,
            "system": p.system,
            "skills": [s.model_dump() for s in skills],
            "mcp": [self._mcp_server_to_dict(s) for s in p.mcp_servers],
            "documents": [doc.to_info().model_dump() for doc in p.documents],
        }

    def list_projects(self) -> list[dict]:
        """List all projects in the workspace."""
        ws = Workspace.load()

        if not ws or not ws.projects:
            return []

        return [self._project_to_dict(p) for p in ws.projects]

    def delete_project(self, project_id: str) -> dict:
        """Delete a project by ID."""
        self._ensure_model()

        workspace = self._model.workspace
        project = next((p for p in workspace.projects if p.id == project_id), None)
        if project is None:
            raise ValueError(f"Project not found: {project_id}")

        if project.system:
            raise ValueError("System projects cannot be deleted")

        was_current = project.current

        # Remove internal project directory
        if project.project_dir and os.path.exists(project.project_dir):
            shutil.rmtree(project.project_dir)
            logger.info(f"Deleted project directory: {project.project_dir}")

        # Remove visible project folder (~/smartloop/<folder_name>)
        visible = self._visible_project_dir(project)
        if visible and os.path.exists(visible):
            shutil.rmtree(visible)
            logger.info(f"Deleted visible project folder: {visible}")

        # Remove from workspace
        workspace.projects = [p for p in workspace.projects if p.id != project_id]

        # If deleted project was current, set the first remaining project as current
        if was_current and workspace.projects:
            workspace.projects[0].current = True

        # Reset cached indexers/stores since the project they pointed to may be gone
        self._workspace_indexer = None
        self._feedback_indexer = None
        self._doc_store = None
        self._doc_store_project_id = None

        self._model.save()

        return {"id": project_id, "deleted": True, "message": "Project deleted successfully"}

    def set_current_project(self, project_id: str) -> None:
        """Switch the active project.

        All projects use the workspace's active model, so no model
        switching is needed on project change.
        """
        self._ensure_model()

        workspace = self._model.workspace
        target = next((p for p in workspace.projects if p.id == project_id), None)
        if target is None:
            raise ValueError(f"Project not found: {project_id}")

        for p in workspace.projects:
            p.current = (p.id == project_id)

        # Reset cached indexers/stores so they get re-created for the new project
        self._workspace_indexer = None
        self._feedback_indexer = None
        self._doc_store = None
        self._doc_store_project_id = None

        self._model.save()

    def _cleanup_gpu_cache(self):
        """Clear GPU cache to free up memory after unloading models."""
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared GPU cache")
        except ImportError:
            pass

    def unload_model(self):
        """Unload the current model to free resources."""
        if self._llm is not None:
            project_name = self.get_current_project_name() or "default"
            logger.info(f"Unloading model: {self._model_name} (project: {project_name})")
            self._llm = None
            self._model_name = None
            self._decision_llm = None
            self._decision_model_name = None

            self._reset_llm_executor()
            self._cleanup_gpu_cache()

    def unload_vision_model(self) -> None:
        """Release the on-demand vision/OCR model and return its VRAM.

        Only the orchestrator model stays resident; the multimodal vision model
        is loaded on demand by ``get_vision_llm`` for document OCR and freed here
        once a stage completes (e.g. the indexer queue drains).  Dropping the
        reference frees the llama.cpp context + its CUDA buffers (~1.4GB for
        Qwen2-VL-2B; verified it returns to the GPU).  Clearing ``_vision_llm``
        re-arms lazy reload, and ``_vision_unavailable`` is reset so a fresh
        attempt re-evaluates fit rather than staying latched off.
        """
        if self._vision_llm is None:
            return
        with self._inference_lock:
            if self._vision_llm is None:
                return
            logger.info("[vision] Unloading vision model %s — freeing VRAM",
                        self._vision_model_name or "?")
            self._vision_llm = None
            self._vision_model_name = None
            self._vision_unavailable = False
            self._cleanup_gpu_cache()

    def unload_large_model(self) -> None:
        """Release the on-demand large agent/skills model and return its VRAM.

        Like the vision model, the larger skills model is loaded on demand
        (``_get_large_llm``) and is not part of the always-resident footprint —
        freeing it here keeps only the orchestrator resident between stages.
        """
        if self._large_llm is None:
            return
        with self._inference_lock:
            if self._large_llm is None:
                return
            logger.info("[large-model] Unloading large agent model — freeing VRAM")
            self._large_llm = None
            self._large_model_attempted = False
            self._cleanup_gpu_cache()

    # --------------------------------------------------------------- MCP servers

    def add_remote_mcp_server(
        self,
        name: str,
        server_url: str,
        auth_type: str = "none",
        tools: list[dict] | None = None,
        token: str | None = None,
        token_endpoint: str | None = None,
        authorization_endpoint: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        access_token: str | None = None,
        refresh_token: str | None = None,
        expires_at: float | None = None,
        token_type: str = "Bearer",
        project_id: str | None = None,
        server_info: dict | None = None,
        description: str | None = None,
        pkg: str | None = None,
    ) -> dict:
        """Register a remote MCP server to a project (by ID or current)."""
        from smartloop.workspace.workspace import (
            MCPServer, MCPServerType, MCPAuthType,
            RemoteMCPServerConfig, MCPOAuthCredentials, MCPToolDefinition,
        )
        if project_id:
            current_project = self._get_project(project_id)
        else:
            self._initialize_workspace()
            current_project = self._model.workspace.current_project
            if not current_project:
                raise ValueError("No active project")

        mcp_auth = MCPAuthType(auth_type) if auth_type in MCPAuthType._value2member_map_ else MCPAuthType.NONE
        oauth_creds = None
        if client_id:
            oauth_creds = MCPOAuthCredentials(
                client_id=client_id,
                client_secret=client_secret,
                access_token=access_token,
                refresh_token=refresh_token,
                expires_at=expires_at,
                token_type=token_type,
            )

        server = MCPServer(
            id=str(uuid4()),
            name=name,
            pkg=pkg,
            description=description,
            server_type=MCPServerType.REMOTE,
            remote_config=RemoteMCPServerConfig(
                server_url=server_url,
                auth_type=mcp_auth,
                token=token,
                token_endpoint=token_endpoint,
                authorization_endpoint=authorization_endpoint,
            ),
            oauth_credentials=oauth_creds,
            tools=[
                MCPToolDefinition(
                    name=t["name"],
                    description=t.get("description"),
                    input_schema=t.get("input_schema"),
                )
                for t in (tools or [])
            ],
            server_info=server_info,
        )
        current_project.mcp_servers.append(server)
        self._model.save()

        result = self._mcp_server_to_dict(server)

        return result

    def add_local_mcp_server(
        self,
        command: str,
        args: list[str] | None = None,
        name: str | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        description: str | None = None,
        project_id: str | None = None,
    ) -> dict:
        """Register a local (stdio) MCP server to a project (by ID or current).

        Follows the standard MCP stdio config: the caller supplies the
        ``command`` (e.g. ``uvx``, ``npx``) and ``args``. The command must be
        installed and on PATH on the host system — it is launched as-is, no
        rewriting. Built-in Smartloop servers use FastMCP modules launched via
        ``<python> -m smartloop...`` and are registered separately.
        """
        if not command:
            raise ValueError("'command' is required to register a local MCP server")

        args = list(args or [])

        # If the caller passed the full command as a single string
        # (e.g. "uvx yahoo-finance-mcp"), split it into command + args.
        # Only applies when args is still empty.
        if not args and " " in command:
            parts = command.split()
            command = parts[0]
            args = parts[1:]

        if not args:
            raise ValueError(
                f"'args' is required to register a local MCP server (command='{command}'). "
                "Provide the package or module name, e.g. args=['yahoo-finance-mcp']."
            )

        # Normalise bare `uvx <pkg>` → `uvx --from <pkg> <pkg>` so uvx
        # always knows both the package to install and the executable to run.
        if (
            command == "uvx"
            and len(args) == 1
            and not args[0].startswith("-")
            and "--from" not in args
        ):
            pkg = args[0]
            args = ["--from", pkg, pkg]

        # `uvx` is only a ~340KB trampoline that execs the real `uv` binary,
        # which it locates *solely* as a sibling file next to itself (it does
        # not search PATH). A system with `uvx` but no adjacent `uv` therefore
        # fails. Since `uvx <args>` is exactly `uv tool run <args>`, prefer the
        # self-contained `uv` binary when it is available on PATH.
        if command == "uvx" and shutil.which("uv"):
            command = "uv"
            args = ["tool", "run", *args]

        if name is None:
            # Derive a display name from the package/script, i.e. the first
            # non-flag arg (e.g. "uvx mcp-server-fetch" → "mcp-server-fetch"),
            # falling back to the command itself.
            pkg_arg = next((a for a in args if not a.startswith("-")), command)
            name = pkg_arg.split("/")[-1]

        if project_id:
            current_project = self._get_project(project_id)
        else:
            self._initialize_workspace()
            current_project = self._model.workspace.current_project
            if not current_project:
                raise ValueError("No current project found")

        server_id = str(uuid4())[:8]
        resolved_cwd = str(Path(cwd).expanduser()) if cwd else None
        resolved_env = env or {}

        server = MCPServer(
            id=server_id,
            name=name,
            pkg=name,
            description=description,
            config=MCPServerConfig(command=command, args=args, env=resolved_env, cwd=resolved_cwd),
        )

        # Discover tools by launching the process and doing the MCP handshake.
        # Any failure here is fatal: we do not persist a server whose tools we
        # could not enumerate, and the error is surfaced to the API caller.
        discovery_error: str | None = None
        try:
            with LocalMCPClient(command=command, args=args, env=resolved_env, cwd=resolved_cwd) as client:
                result = client.connect_and_discover_tools()
        except MCPLaunchError as exc:
            # Both `uvx` and `uv tool run` print a hint like
            # "Use `uvx --from <pkg> <exec>` instead" (or the `uv tool run`
            # form) when the package name and executable name differ. Parse the
            # suggested args and auto-retry under whichever launcher we used.
            hint = re.search(r"Use `uv(?:x| tool run) (.*?)` instead", str(exc))
            if hint:
                suggested = hint.group(1).split()
                # Preserve the `tool run` prefix when launching via the real
                # `uv` binary; for bare `uvx` there is no prefix.
                prefix = args[:2] if command == "uv" and args[:2] == ["tool", "run"] else []
                fixed_args = [*prefix, *suggested]
                logger.info(
                    "Retrying %s with corrected args %s (original: %s)", command, fixed_args, args
                )
                args = fixed_args
                if "--from" in suggested:
                    from_idx = suggested.index("--from")
                    fixed_pkg = suggested[from_idx + 1] if from_idx + 1 < len(suggested) else None
                else:
                    fixed_pkg = next((a for a in suggested if not a.startswith("-")), None)
                if fixed_pkg:
                    server.name = fixed_pkg.split("/")[-1]
                    server.pkg = server.name
                    name = server.name
                server.config = MCPServerConfig(command=command, args=args, env=resolved_env, cwd=resolved_cwd)
                try:
                    with LocalMCPClient(command=command, args=args, env=resolved_env, cwd=resolved_cwd) as client:
                        result = client.connect_and_discover_tools()
                except Exception as retry_exc:
                    result = None
                    discovery_error = str(retry_exc)
            else:
                result = None
                discovery_error = str(exc)
        except Exception as exc:
            result = None
            discovery_error = str(exc)

        if result and result.success:
            server.tools = result.tools
            si = result.server_info
            if si:
                if not server.description and si.description:
                    server.description = si.description
                si_dict = {"name": si.name, "title": si.title, "version": si.version}
                if si.description:
                    si_dict["description"] = si.description
                if si.icons:
                    si_dict["icons"] = si.icons
                server.server_info = si_dict
            logger.info(
                "Discovered %d tool(s) from local MCP server '%s'",
                len(result.tools), name,
            )
        elif result is not None:
            discovery_error = result.error or "tool discovery failed"

        if discovery_error is not None:
            message = (
                f"Could not discover tools from local MCP server '{name}': {discovery_error}"
            )
            logger.warning(message)
            raise MCPDiscoveryError(message)

        current_project.mcp_servers.append(server)
        self._model.save()

        result = self._mcp_server_to_dict(server)

        return result

    def _write_mcp_doc(
        self,
        project: "Project",
        server: "MCPServer",
        content: str,
    ) -> bool:
        """Write MCP tool description to disk for workspace indexer discovery."""
        try:
            base_dir = Path(self._visible_project_dir(project))
            base_dir.mkdir(parents=True, exist_ok=True)

            safe_name = re.sub(r"[^a-z0-9]+", "-", server.name.lower()).strip("-")
            filename = f"mcp-{safe_name}.md"
            doc_path = base_dir / filename

            if doc_path.exists():
                logger.info("Workspace doc '%s' already exists, skipping", filename)
                return False

            doc_path.write_text(content, encoding="utf-8")
            logger.info("Workspace doc '%s' created for MCP server '%s'", filename, server.name)
            return True
        except Exception as e:
            logger.warning("Failed to write workspace doc for MCP server '%s': %s", server.name, e)
            return False

    @staticmethod
    def _mcp_server_to_dict(s: MCPServer) -> dict:
        """Convert an MCPServer to a dict for API responses."""
        result = {
            "id": s.id,
            "name": s.name,
            "description": s.description,
            "summary": s.summary,
            "server_type": s.server_type.value,
            "enabled": s.enabled,
            "system": s.system,
            "tools": [t.model_dump() for t in s.tools],
        }
        if s.config:
            result["command"] = s.config.command
            result["args"] = s.config.args
            if s.config.cwd:
                result["cwd"] = s.config.cwd
        if s.remote_config:
            result["server_url"] = s.remote_config.server_url
        if s.server_info:
            result["server_info"] = s.server_info
        return result

    def list_mcp_servers(self) -> list[dict]:
        """List MCP servers for the current project."""
        self._initialize_workspace()

        current_project = self._model.workspace.current_project
        if not current_project:
            return []

        return [
            self._mcp_server_to_dict(s)
            for s in current_project.mcp_servers
            if not s.system
        ]

    def delete_mcp_server(self, server_id: str, project_id: str | None = None) -> dict:
        """Remove an MCP server from a project (by ID or current)."""
        if project_id:
            current_project = self._get_project(project_id)
        else:
            self._initialize_workspace()
            current_project = self._model.workspace.current_project
            if not current_project:
                raise ValueError("No current project found")

        server = next(
            (s for s in current_project.mcp_servers if s.id == server_id), None
        )
        if server is None:
            # The server may have been registered by another process since the
            # last workspace load.  Reload from disk and retry once.
            self._model.workspace.reload()
            current_project = next(
                (p for p in self._model.workspace.projects if p.id == (project_id or (current_project.id if current_project else ""))),
                current_project,
            )
            server = next(
                (s for s in current_project.mcp_servers if s.id == server_id), None
            )
        if server is None:
            raise ValueError(f"MCP server not found: {server_id}")

        current_project.mcp_servers = [
            s for s in current_project.mcp_servers if s.id != server_id
        ]
        self._model.save()

        return {
            "id": server_id,
            "deleted": True,
            "message": "MCP server deleted successfully",
        }

    # --------------------------------------------------------------- MCP agents

    def _get_llm_executor(self):
        """Return the shared LLMExecutor, creating it lazily.

        Worker processes (each loads its own model copy) are spawned lazily
        by the pool on first use, so this is cheap until a parallel
        document fan-out actually runs.  The pool is rebuilt after a model
        load/unload so workers always load the current GGUF.
        """
        if self._llm_executor is None:
            from smartloop.agents.hybrid_executor import LLMExecutor
            from smartloop.config.app_settings import AppSettings
            settings = AppSettings()
            model_path = getattr(self, "_model_gguf_path", None)
            self._llm_executor = LLMExecutor(
                max_io_workers=6,
                n_llm_workers=settings.n_llm_workers if model_path else 0,
                model_path=model_path,
                project_id=self.get_project_id(),
            )
        return self._llm_executor

    def _reset_llm_executor(self) -> None:
        """Shut down the shared executor so it's rebuilt for the new model."""
        if self._llm_executor is not None:
            try:
                self._llm_executor.shutdown(wait=False)
            except Exception:
                pass
            self._llm_executor = None

    def orchestrate(
        self,
        user_message: str,
        skills: Optional[list] = None,
        cwd: Optional[str] = None,
        conversation_history: Optional[list] = None,
        has_attachments: bool = False,
    ) -> tuple[Optional[str], bool, Optional[str], Optional[list], Optional[str], bool]:
        """Full orchestrator entry point for a user message.

        Runs classification, context enrichment, DAG execution, capability
        routing and — when the plan calls for it — MCP tool calls.  An MCP
        tool call is one possible outcome, not the point of this method;
        CONVERSATION intents route to a capability-tagged model instead.

        Args:
            conversation_history: Recent ``[{"role": ..., "content": ...}]``
                turns passed to the ContextExplorer for history insight.

        Returns:
            (context, matched, error, citations, skill_instruction, mutation)
            — *context* is the formatted tool output,
            *matched* is True when the orchestrator produced content,
            *error* is a user-facing error message (or None on success),
            *citations* is a list of source document dicts (or None),
            *skill_instruction* is the matched skill content (or None),
            *mutation* signals the router to stream *context* verbatim
            instead of running it through another LLM.
        """
        logger.info("[orchestrator] entered orchestrate: %r", user_message[:120])
        if self._model is None or self._llm is None:
            logger.warning(
                "[orchestrator] BYPASSED — model/llm not loaded "
                "(model=%s, llm=%s). Falling back to raw stream_completion.",
                self._model is not None, self._llm is not None,
            )
            return None, False, None, None, None, False

        # Reload workspace from disk to pick up any changes made by
        # background indexers or project setup since the last request.
        self._model.workspace.reload()

        current_project = self._model.workspace.current_project
        if not current_project:
            logger.warning(
                "[orchestrator] BYPASSED — no current project. Falling back "
                "to raw stream_completion.",
            )
            return None, False, None, None, None, False

        # Migrate legacy REMOTE system servers to LOCAL on first access
        self._migrate_servers_to_local(current_project)

        # Filter to enabled servers that have tools (both local and remote)
        eligible_servers = [
            s for s in current_project.mcp_servers
            if s.enabled and s.tools
        ]

        # Always read skills from disk — the single source of truth.
        # The ``skills`` parameter is accepted for backward compat but
        # disk takes precedence when available.
        disk_skills = current_project.get_skills()
        all_skills = disk_skills or (skills or [])
        skill_contents = None
        if all_skills:
            skill_contents = []
            for s in all_skills:
                header = f"---\nname: {s.name}\n"
                if s.description:
                    header += f"description: {s.description}\n"
                header += "---\n\n"
                skill_contents.append(header + s.content)

        # Use the visible project directory (~/smartloop/<name>) so the
        # orchestrator's skill scanner finds SKILL.md files written there.
        visible_project_dir = self._visible_project_dir(current_project)

        try:
            orchestrator = SLPOrchestrator(
                orchestrator_llm=self._llm,
                agent_llm=self._decision_llm or self._llm,
                large_llm_provider=self._get_large_llm,
                capability_llm_provider=self._get_llm_for_capability,
                model_path=getattr(self, '_model_gguf_path', None),
                project_id=self.get_project_id(),
                executor=self._get_llm_executor(),
            )
            workspace = self._model.workspace
            caps: list[str] = []
            for m in workspace.models:
                if m.enabled and m.downloaded:
                    caps.extend(m.effective_capabilities)
            available_capabilities = list(dict.fromkeys(caps))
            logger.info(
                "[caps] models=%s available_capabilities=%s",
                [(m.model_name, m.enabled, m.downloaded, m.effective_capabilities)
                 for m in workspace.models],
                available_capabilities,
            )

            result = orchestrator.execute(
                user_message, eligible_servers,
                skills=skill_contents, cwd=cwd,
                has_attachments=has_attachments,
                project_dir=visible_project_dir,
                model_service=self,
                conversation_history=conversation_history,
                available_capabilities=available_capabilities or None,
            )

            # Save workspace in case tokens were refreshed during the call
            try:
                self._model.save()
            except Exception:
                pass

            if not result.handled:
                return None, False, None, None, None, False

            if result.error:
                error_msg = f"Tool '{result.tool_name}' failed: {result.error}"
                logger.warning(f"MCP tool call failed ({result.tool_name}): {result.error}")
                return None, True, error_msg, None, None, False

            if result.content:
                logger.info(
                    f"MCP tool '{result.tool_name}' returned "
                    f"{len(result.content)} chars of context"
                )
                return result.content, True, None, result.citations, result.skill_instruction, result.mutation

            return None, True, None, None, result.skill_instruction, False

        except Exception as e:
            error_msg = f"Tool call failed: {e}"
            logger.warning(f"MCP tool call attempt failed: {e}")
            return None, True, error_msg, None, None, False

    # -------------------------------------------------------------- auto-write

    _WRITE_INTENT_RE = re.compile(
        r"\b(create|generate|write|save|export|produce|make|build|draft)\b",
        re.IGNORECASE,
    )

    # Matches bare "write" / "save" / "write it" / "save that" commands
    _BARE_WRITE_RE = re.compile(
        r"^\s*(write|save|write\s+(it|that|this)|save\s+(it|that|this))\s*[.!]?\s*$",
        re.IGNORECASE,
    )

    def maybe_write_output(
        self,
        user_message: str,
        llm_response: str,
        cwd: Optional[str],
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """Write LLM output to a file if the user asked to create/generate/write.

        Supports two modes:
        1. Bare "write" / "save" command — writes the last assistant response
           from session history (enables back-and-forth refinement then save).
        2. Inline write intent — writes the current llm_response.

        Returns a confirmation string on success, or None if no write was needed.
        """
        if not cwd or self._llm is None:
            return None

        cwd = os.path.expanduser(cwd)
        content_to_write = llm_response

        # Check for bare "write" command — write last assistant response from history
        if self._BARE_WRITE_RE.match(user_message):
            if session_id:
                last_response = self._get_last_assistant_response(session_id)
                if last_response:
                    content_to_write = last_response
                    logger.info(f"[auto-write] Bare write command — using last assistant response ({len(content_to_write)} chars)")
                else:
                    logger.info("[auto-write] Bare write command but no prior assistant response found")
                    return None
            elif not llm_response:
                return None
        elif not llm_response or not self._WRITE_INTENT_RE.search(user_message):
            return None

        if not content_to_write:
            return None

        content_to_write = self._extract_document_content(content_to_write)
        if not content_to_write:
            return None

        # Ask LLM only for the filename — write the actual content
        prompt = (
            f"User request: {user_message}\n\n"
            "Pick a short descriptive filename (without extension) for saving this output. "
            'Examples: "project_summary", "api_notes", "readme_draft".\n\n'
            "Reply with ONLY the filename, nothing else."
        )
        try:
            file_name = str(self._llm.invoke(prompt)).strip().strip('"\'`')
            if not file_name:
                return None

            safe_name = re.sub(r"[^\w\-]", "_", file_name).strip("_")
            out_dir = os.path.join(cwd, ".out")
            os.makedirs(out_dir, exist_ok=True)
            out_path = f"{safe_name}.md"
            full_path = os.path.join(out_dir, out_path)

            # If file already exists, append a counter
            counter = 1
            while os.path.exists(full_path):
                out_path = f"{safe_name}_{counter}.md"
                full_path = os.path.join(out_dir, out_path)
                counter += 1

            if not content_to_write.endswith("\n"):
                content_to_write += "\n"
            content_to_write += "\nCo-Authored By: Smartloop AI\n"

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content_to_write)
            logger.info(f"[auto-write] Saved to {full_path} ({len(content_to_write)} chars)")
            return full_path
        except Exception as e:
            logger.warning(f"[auto-write] Failed: {e}")
            return None

    def _get_last_assistant_response(self, session_id: str) -> Optional[str]:
        """Get the last assistant message from session history (excluding the current turn)."""
        history = self.get_session_history(session_id)
        # Walk backwards to find the last assistant message
        for msg in reversed(history):
            role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if role == "assistant" and content:
                return content
        return None

    _PREAMBLE_RE = re.compile(
        r"^(?:(?:sure|okay|of course|absolutely|certainly|here(?:'s| is| are| you go))"
        r"[^\n]*\n+)+",
        re.IGNORECASE,
    )
    _POSTAMBLE_RE = re.compile(
        r"\n+(?:(?:i hope|let me know|feel free|enjoy|bon appétit|happy|hope this|"
        r"if you (?:have|need|want)|don't hesitate|"
        r"would you like|do you want|shall i|"
        r"you (?:can|could|may|might) (?:also|adjust|modify|change|customize|tweak)|"
        r"this (?:can be|could be|is just|is a) (?:adjusted|modified|customized|starting)|"
        r"note:|please note|disclaimer|tips?:)[^\n]*\n*)+\s*$",
        re.IGNORECASE,
    )

    @classmethod
    def _extract_document_content(cls, text: str) -> str:
        """Strip conversational preamble/postamble, keeping only the document body."""
        lines = text.split("\n")

        # If the response contains a markdown heading, start from the first one
        first_heading = None
        for i, line in enumerate(lines):
            if line.startswith("#"):
                first_heading = i
                break

        if first_heading is not None and first_heading > 0:
            preamble = "\n".join(lines[:first_heading]).strip()
            if len(preamble) < 200:
                text = "\n".join(lines[first_heading:])
            else:
                text = cls._PREAMBLE_RE.sub("", text)
        else:
            text = cls._PREAMBLE_RE.sub("", text)

        text = cls._POSTAMBLE_RE.sub("", text)
        return text.strip()

    # ------------------------------------------------------------------ assets

    def _assets_dir(self) -> Path:
        """Return (and create if needed) the directory used for asset storage."""
        path = Path(os.path.join(_settings.home_dir, "assets"))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def upload_asset(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: Optional[str] = None,
    ) -> dict:
        """Save an uploaded file as an asset, converting to markdown when possible.

        Mirrors GenericModel._process_document: uses ParserFactory with the loaded
        LLM instance so the same parser pipeline as document-upload is applied.

        Returns a dict with: asset_id, filename, content_type, size, markdown.
        """
        asset_id = str(uuid4())  # full UUID
        ext = Path(filename).suffix.lower()

        if not content_type:
            content_type, _ = mimetypes.guess_type(filename)
            content_type = content_type or "application/octet-stream"

        asset_dir = self._assets_dir() / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        # Write the original file to a temp location for parsing
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        markdown_content = None
        is_markdown = False

        try:
            mime_type = content_type or (mimetypes.guess_type(filename)[0] or "application/octet-stream")
            llm = self.load_model()

            # get_parser raises ValueError for unsupported mime types — treat as raw storage
            try:
                parser = ParserFactory(llm).get_parser(mime_type, tmp_path, logger)
            except ValueError:
                parser = None

            if parser is not None:
                markdown_content = parser.parse()
                is_markdown = True

                # Enforce token limit using the loaded LLM's tokenizer
                _MAX_TOKENS = llm.get_max_tokens() if llm else 8192
                estimated_tokens = llm.get_num_tokens(markdown_content)
                if estimated_tokens > _MAX_TOKENS:
                    raise ValueError(
                        f"Attachment too large: ~{estimated_tokens:,} tokens estimated "
                        f"(maximum is {_MAX_TOKENS:,}). "
                        "Please split the document into smaller sections."
                    )

        except ValueError:
            raise
        except Exception as parse_err:
            logger.error(f"Parser conversion failed for {filename}: {parse_err}; storing raw bytes")
        finally:
            os.unlink(tmp_path)

        # Persist metadata
        metadata = {
            "asset_id": asset_id,
            "filename": filename,
            "content_type": content_type,
            "size": len(file_bytes),
            "markdown": is_markdown,
        }
        (asset_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

        # Persist content
        if markdown_content is not None:
            (asset_dir / "content.md").write_text(markdown_content, encoding="utf-8")

        logger.info(f"Asset stored: {asset_id} ({filename}, markdown={is_markdown})")
        return metadata

    def get_asset(self, asset_id: str) -> dict:
        """Retrieve asset metadata and content by ID."""
        asset_dir = Path(self._assets_dir(), asset_id)
        meta_path = asset_dir / "metadata.json"
        if not meta_path.exists():
            raise ValueError(f"Asset not found: {asset_id}")

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        content_path = asset_dir / "content.md"
        if content_path.exists():
            metadata["content"] = content_path.read_text(encoding="utf-8")
        else:
            metadata["content"] = None

        return metadata

    def delete_asset(self, asset_id: str) -> dict:
        """Delete an asset and all its stored files."""
        asset_dir = Path(self._assets_dir(), asset_id)
        if not asset_dir.exists():
            raise ValueError(f"Asset not found: {asset_id}")

        shutil.rmtree(asset_dir)
        logger.info(f"Asset deleted: {asset_id}")
        return {"asset_id": asset_id, "deleted": True, "message": "Asset deleted successfully"}

    def resolve_attachment_context(self, asset_ids: list[str]) -> str:
        """Return a combined markdown string for a list of asset IDs.

        Used to inject attachment content into the chat prompt.
        """
        parts = []
        for aid in asset_ids:
            try:
                asset = self.get_asset(aid)
                content = asset.get("content") or ""
                if content:
                    parts.append(f"ATTACHMENT [{asset.get('filename', aid)}]:\n{content}")
            except ValueError:
                logger.warning(f"Attachment asset not found: {aid}")
        return "\n\n---\n\n".join(parts)


