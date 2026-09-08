"""Singleton model service for managing LLM lifecycle"""

import base64
import json
import logging
import mimetypes
import os
import re
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
import gc
import sys


from pathlib import Path
from typing import Callable, Dict, Optional, Iterator, List
from uuid import uuid4

import markdownify

from langchain_core.outputs import GenerationChunk

from smartloop.model_factory import ModelFactory
from smartloop.config import ModelConfig, SUPPORTED_MODELS
from smartloop.llms.base_llm import BaseLLM
from smartloop.llms.thinking import strip_thinking
from smartloop.types import ModelCapability
from smartloop.workspace import DocumentInfo, DocumentStatus, MCPServer, Workspace
from smartloop.workspace import Project, MCPServer, MCPServerConfig, MCPServerStatus, MCPToolDefinition, Skill, MCPServerType, RemoteMCPServerConfig
from smartloop.db import ChunkResult, StoreFactory
from smartloop.db.workspace_indexer import WorkspaceIndexer, WORKSPACE_COLLECTION
from smartloop.db.feedback_indexer import FeedbackIndexer
from smartloop.feedback_store import FeedbackStore, FeedbackEntry, ConversationTurn
from smartloop.model.model_manager import Model
from smartloop.mcp.definitions import split_command_line
from smartloop.mcp.local_client import LocalMCPClient
from smartloop.mcp.client import MCPLaunchError, MCPDiscoveryError
from smartloop.agents.orchestrator import SLPOrchestrator
from smartloop.agents.orchestration.model_router import ModelRouter
from smartloop.agents.planner import SYSTEM_TOOLS_SERVER_ID, SYSTEM_TOOLS_SERVER_NAME, FETCH_SERVER_ID

from smartloop.constants import canonical_model_name, display_model_label
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
    MCPTask,
)
from smartloop.utils import normalize_path
from smartloop.db import embeddings

_settings = AppSettings()

logger = logging.getLogger(__name__)


def _n_ctx_desc(cap: int) -> str:
    """How to describe an ``llm_large_n_ctx_cap`` value in a log line.

    ``0`` is the sentinel for *no cap* — the factory auto-sizes the context
    from live available memory — but printing it raw read as a context window
    of zero tokens, which is both alarming and wrong.
    """
    return f"n_ctx={cap}" if cap > 0 else "an auto-sized context"

# Exceptions that indicate code defects (not recoverable runtime/IO
# failures).  These re-raise instead of degrading to an empty result so
# bugs surface loudly rather than silently returning no context.
_PROGRAMMING_ERRORS = (NameError, AttributeError, TypeError, ImportError)

# Cache sentinel: a larger response model (capability/large) that passed the
# load-time fit guard but then exhausted GPU memory during generation (e.g.
# Vulkan ErrorOutOfDeviceMemory).  Stored in place of the cached LLM so later
# requests skip straight to the default router model instead of re-loading a
# model that won't fit.  See ModelService._evict_response_llm.
_GPU_FAULTED = object()

DEFAULT_MAX_CONTEXT_TOKENS = 8192

# Shared vector-retrieval selection criteria — applied identically to
# document and workspace search so both rank and select content the same
# way (broad recall, quality floor, relevance cutoff) instead of each path
# using its own divergent thresholds.
_RECALL_MAX_DISTANCE = 0.8      # broad initial recall before selection
_MIN_BEST_DISTANCE = 0.55       # quality floor: reject a weak best match
_DISTANCE_CUTOFF_MARGIN = 0.15  # keep chunks within this margin of the best

# How many matched chunks a citation carries for the router to verify against.
# Closest-first, so this keeps the passages the document was actually retrieved
# for and drops its long tail of weaker hits — which is both the cheaper and
# the stricter thing to check the finished answer against.
_MAX_CITATION_CHUNKS = 5


def _select_relevant_chunks(chunks: list, label: str = "vector-search") -> list:
    """Apply the shared relevance-selection criteria to ranked chunks.

    *chunks* must be ordered closest-first (ascending distance).  Returns the
    kept chunks, or ``[]`` when the best match fails the quality floor.
    Shared by document and workspace retrieval so both apply the same
    emphasis/thresholds rather than duplicating (and drifting) the logic.
    """
    if not chunks:
        return []
    best = chunks[0].distance
    if best > _MIN_BEST_DISTANCE:
        logger.info(
            f"[{label}] best match distance {best:.3f} exceeds quality floor "
            f"{_MIN_BEST_DISTANCE} — no relevant matches"
        )
        return []
    cutoff = best + _DISTANCE_CUTOFF_MARGIN
    kept = [c for c in chunks if c.distance <= cutoff]
    if len(kept) < len(chunks):
        logger.info(
            f"[{label}] cutoff {cutoff:.3f} (best={best:.3f}+{_DISTANCE_CUTOFF_MARGIN}) "
            f"trimmed {len(chunks) - len(kept)} chunk(s)"
        )
    return kept


class _ProjectCaches:
    """The per-project store handles a request resolves against.

    These used to be one slot each on :class:`ModelService`, implicitly keyed to
    whichever project the ``current`` flag last named.  With the project bound
    per request rather than per process, two concurrent chats hold two
    different projects at once, and a single slot would have them evicting each
    other's store on every turn — reopening SQLite each time, and in the window
    between the check and the rebuild answering one project's query from the
    other's index.  One slot per project removes the interleaving entirely.
    """

    __slots__ = ("workspace_indexer", "feedback_indexer", "doc_store")

    def __init__(self) -> None:
        self.workspace_indexer: Optional[WorkspaceIndexer] = None
        self.feedback_indexer: Optional[FeedbackIndexer] = None
        self.doc_store: Optional[object] = None


class ModelService:
    """Singleton service to manage model lifecycle and inference"""

    _instance: Optional["ModelService"] = None
    _lock = threading.Lock()

    # Class-level so they exist on an instance built with ``__new__`` alone —
    # which several call sites and tests do — rather than blowing up the first
    # time anything asks how this process reaches the model.  The service is a
    # singleton, so one construction lock for the class is one per process.
    _model_client = None
    _model_client_lock = threading.Lock()

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
        # Latched after a GPU clip encode crashes (native OOM on a large
        # image): subsequent vision loads keep the CLIP encoder on CPU.
        self._vision_clip_cpu: bool = False
        # Set once the large model has been considered (loaded, skipped for
        # fit, or absent), so we don't re-probe VRAM on every agent request.
        self._large_model_attempted: bool = False
        # Shared, long-lived executor whose LLM worker processes load the
        # model once and are reused across requests (used for parallel
        # document fan-out).  Created lazily; reset on model load/unload.
        self._llm_executor = None
        # Use a semaphore sized to n_parallel so agents can run concurrently
        # when the model is loaded with n_parallel > 1 (continuous batching).
        # Always taken through :meth:`_inference_slot`, which is what makes
        # "is the model answering someone right now" an answerable question —
        # background work waits for a quiet moment rather than competing with
        # a reply the user is watching stream.
        self._inference_lock = threading.Semaphore(_settings.n_parallel)
        self._inference_active = 0
        self._inference_state_lock = threading.Lock()
        self._last_inference_at = 0.0
        self._sessions: dict[str, List[MessageInput]] = {}  # session_id -> chat history (in-memory cache)
        self._conversation_store = ConversationStore(_settings.home_dir)
        self._task_queue = TaskQueue(logger)
        # Store handles, one set per project (see :class:`_ProjectCaches`).
        self._project_caches: Dict[str, _ProjectCaches] = {}
        self._project_caches_lock = threading.Lock()
        self._initialized = True

    # ------------------------------------------------------------ model client

    @property
    def model_client(self):
        """How this process reaches the generative model.

        In-process by default — the same ``stream_completion`` /
        ``generate_completion`` calls, behind one interface — so the seam costs
        a method call and nothing else.  When ``SLP_MODEL_HOST`` names a socket
        the very same calls go to the process that owns the device instead,
        which is what lets several per-project agents share one resident model
        rather than each loading its own copy into the same VRAM.

        Resolved once: the transport is a property of how this process was
        started, and re-reading the environment per call would let it change
        underneath an in-flight generation.
        """
        client = self._model_client
        if client is None:
            # Built on first use — importing the client pulls in the
            # transport, which a process that never generates should not pay
            # for.
            from smartloop.model_host.client import get_model_client

            with self._model_client_lock:
                if self._model_client is None:
                    # Set on the instance, shadowing the class default, so a
                    # test that swaps one in does not leak into the next.
                    self._model_client = get_model_client(self)
                client = self._model_client
        return client

    def _remote_device(self):
        """The model client when the weights live in another process, else ``None``.

        In-process — the supervisor, and the ``--no-supervised`` fallback —
        this is ``None`` and every path below behaves exactly as it always
        has.  In a per-project agent it is the host, and the service builds
        handles to it rather than loading a second copy of the weights into
        the same device.
        """
        from smartloop.model_host.client import RemoteModelClient

        client = self.model_client
        return client if isinstance(client, RemoteModelClient) else None

    def _remote_llm(self, capability: Optional[str] = None):
        """A handle to the host's model for *capability*."""
        from smartloop.model_host.remote_llm import RemoteLLM

        return RemoteLLM(
            self._remote_device(),
            project_id=self.get_project_id() or "",
            capability=capability,
        )

    # ------------------------------------------------------- per-project caches

    def _caches(self, project_id: str) -> _ProjectCaches:
        """Return (creating if needed) the cache slot for *project_id*."""
        with self._project_caches_lock:
            cache = self._project_caches.get(project_id)
            if cache is None:
                cache = _ProjectCaches()
                self._project_caches[project_id] = cache
            return cache

    def _drop_caches(self, project_id: Optional[str] = None) -> None:
        """Discard cached stores for *project_id*, or for every project.

        Called when a project's on-disk state changes underneath the handles
        (create, delete, UI switch), so the next read reopens rather than
        querying an index that no longer describes the directory.
        """
        with self._project_caches_lock:
            if project_id is None:
                self._project_caches.clear()
            else:
                self._project_caches.pop(project_id, None)

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
        self._sweep_legacy_fetch_server(project)
        return project

    def _sweep_legacy_fetch_server(self, project: "Project") -> None:
        """Drop the obsolete built-in ``__fetch__`` MCP server from a project.

        URL fetching moved to the in-process FetchAgent, but projects created
        before that change still carry a persisted ``__fetch__`` entry that shows
        up in MCP listings. ``remove_fetch_server`` only sweeps the actively
        loaded project (via ``_setup_project_tools``), so wild projects that are
        merely listed never got cleaned. Sweep on every project access so they
        self-heal. Idempotent — only writes when an entry is actually removed,
        so it costs one save per stale project and nothing thereafter.
        """
        if not any(s.id == FETCH_SERVER_ID for s in project.mcp_servers):
            return
        project.mcp_servers = [
            s for s in project.mcp_servers if s.id != FETCH_SERVER_ID
        ]
        try:
            self._model.save()
            logger.info("Swept legacy fetch server from project %s", project.id)
        except Exception as e:
            # Never let a persistence hiccup break project lookup.
            logger.warning(
                "Failed to persist legacy fetch sweep for %s: %s", project.id, e
            )

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

        if self._remote_device() is not None:
            # The device is the model host's.  Every role the orchestrator
            # asks for — main, decision, large, capability — becomes a handle
            # to it; the host does the selection, because it is the only side
            # that can see what fits.
            self._llm = self._remote_llm()
            self._decision_llm = None
            self._model_name = target_model
            logger.info(
                "[load_model] using the model host for %s (project: %s)",
                target_model, project_name,
            )
            return self._llm

        self._llm = self._model.get_instance()
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
        # Clear any capability-model GPU-fault evictions so a fresh model load
        # re-evaluates them against the newly active configuration.
        for attr in list(vars(self)):
            if attr.startswith("_cap_llm_") and getattr(self, attr) is _GPU_FAULTED:
                setattr(self, attr, None)

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
            self._decision_llm = decision_model_obj.get_instance()
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
        if self._remote_device() is not None:
            return self._remote_llm(ModelCapability.SKILLS)

        """Return the large (agent) model, loading it lazily on first use.

        Loaded on demand — only when an agent task actually needs it — rather
        than at startup, so the main model + worker pool own VRAM by default.
        Attempted at most once per active model (``_large_model_attempted``):
        if it loads it's cached; if it's absent or skipped for fit we return
        ``None`` and the caller falls back to the main/decision model.

        With exactly one model selected that model IS the agent model, even
        when it does not out-rank the bootstrap router (``_detect_large_model``
        only promotes a bigger model, so a small selected model would otherwise
        leave agent work on the router the user did not pick) — unless it is a
        vision-only selection, which cannot run agent work at all and so leaves
        detection to find a text model.  A pin that fails to load returns
        ``None`` here rather than falling through to detection: the pinned
        model is the only candidate detection could find, so retrying it would
        just fail the same fit guard a second time.
        The caller drops to the default model, as it does for any other
        unavailable large model.
        """
        if self._pinned_model() is not None:
            return self._pinned_llm()
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

    def get_vision_llm(
        self, on_model_selected: Optional[Callable[[str], None]] = None,
    ) -> Optional[BaseLLM]:
        """Return a loaded *multimodal* vision LLM, or None when unavailable.

        Vision-capable models are considered largest-first (like the
        orchestrator), but a model is only usable for OCR when ALL of these
        hold, so unrunnable candidates are skipped rather than crashing:
          * its mmproj / CLIP projector GGUF is present (else it's blind), and
          * the installed llama-cpp has a chat handler for its architecture
            (the model-declared handler, e.g. the generic MTMDChatHandler, or an
            arch-sniffed one; a missing handler is skipped, never crashed), and
          * it fits in available memory.
        Returns None when no candidate qualifies (callers fall back to text).

        ``on_model_selected(model_name)`` — when supplied — fires with the name
        of the model chosen, right before it is loaded (or immediately for an
        already-resident one), so callers can surface it before the heavy load.
        """
        from smartloop.config.app_settings import AppSettings
        if not AppSettings().vision_enabled:
            return None
        if self._vision_llm is not None:
            if on_model_selected and self._vision_model_name:
                on_model_selected(self._vision_model_name)
            return self._vision_llm
        if not self._model or not self._model.workspace:
            return None

        from smartloop.llms.llm_factory import LlmFactory
        workspace = self._model.workspace
        project = workspace.current_project
        candidates = [
            m for m in workspace.models
            if workspace.effective_model_enabled(m.model_name, project)
            and workspace.effective_model_downloaded(m.model_name)
            and ModelCapability.VISION in m.effective_capabilities
        ]
        # Smallest first: lowest latency OCR; same-size ties broken by cap overlap then version.
        candidates = self._ranked_candidates(
            candidates, ref_caps={ModelCapability.VISION}, descending=False,
        )

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
            # The handler the model class declares alongside its VISION
            # capability (falls back to GGUF-arch sniffing when absent).
            model_cls = SUPPORTED_MODELS.get(ws_model.model_name)
            vision_handler = model_cls.vision_chat_handler() if model_cls else None
            if not LlmFactory.vision_handler_supported(gguf_path, vision_handler):
                logger.warning(
                    "[vision] %s skipped — installed llama-cpp has no vision handler "
                    "(declared=%s, arch=%s)",
                    ws_model.model_name, vision_handler,
                    LlmFactory._read_gguf_arch(gguf_path),
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
            # Announce the chosen model before the (heavy) load so callers can
            # surface it immediately — this candidate passed every guard above,
            # so it's the one that will load (barring a load-time exception).
            if on_model_selected:
                on_model_selected(ws_model.model_name)
            try:
                llm = LlmFactory.get_llama_cpp_instance(
                    gguf_path, logger, self.get_project_id() or "",
                    max_n_ctx=cap, mmproj_path=mmproj_path,
                    clip_on_cpu=clip_on_cpu, vision_handler=vision_handler,
                    model_name=ws_model.model_name,
                    # Reasoning is off for the vision copy whatever the model
                    # declares: transcribing a page and describing an image are
                    # perception, not deliberation, and the small vision_n_ctx
                    # has to hold several thousand image tokens — a <think>
                    # block would spend the transcription's output budget on
                    # itself.
                    thinking=False,
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

    @staticmethod
    def _ranked_candidates(
        candidates: list,
        ref_caps: set,
        descending: bool = True,
    ) -> list:
        """Sort WorkspaceModel candidates by (size, cap-overlap, version).

        ``ref_caps`` is the reference capability set used to score overlap so
        that same-size models with broader coverage rank above narrower ones.
        Pass ``descending=True`` (default) to walk largest→smallest; pass
        ``False`` to walk smallest→largest.
        """
        return sorted(
            candidates,
            key=lambda m: (
                m._rank[0],
                len(ref_caps & set(m.effective_capabilities)),
                m._rank[1],
            ),
            reverse=descending,
        )

    def _selected_models(self) -> List:
        """The models the user has actually chosen — enabled *and* downloaded.

        The bootstrap/orchestrator model is excluded.  It is always enabled,
        is filtered out of the model list the user picks from (see
        ``list_workspace_models``) and cannot be configured per project, so it
        is infrastructure rather than a selection — counting it would mean the
        user could never get down to a single selected model.  Resolution is
        project-scoped, since per-project settings override the workspace
        defaults.
        """
        if not self._model or not self._model.workspace:
            return []
        workspace = self._model.workspace
        project = workspace.current_project
        bootstrap = canonical_model_name(api_settings.default_model)
        return [
            m for m in workspace.models
            if canonical_model_name(m.model_name) != bootstrap
            and workspace.effective_model_enabled(m.model_name, project)
            and workspace.effective_model_downloaded(m.model_name)
        ]

    def _pinned_model(self, capability: Optional[str] = None):
        """The sole selected model, or ``None`` when it cannot serve the step.

        With two or more models selected the choice between them is ours to
        make, and capability routing makes it per step.  With exactly one the
        choice is the user's and it is explicit, so that model runs everything
        — routing has nothing to route between, and second-guessing the
        selection because the model doesn't declare some capability tag would
        answer on the bootstrap model the user never picked.

        The one exception is a selection that cannot hold a text conversation
        at all.  A vision-only model (qwen2-vl-2b declares ``VISION`` and
        nothing else) exists to look at an image; pinning it to the whole
        request puts the planner, the skill matcher, the relevance judge and
        plain chat turns on a model that cannot do them.  So a selection
        without ``GENERAL`` is pinned only to the steps whose *capability* it
        actually declares — the image work it was chosen for — and everything
        else falls back to the default model.
        """
        selected = self._selected_models()
        if len(selected) != 1:
            return None
        ws_model = selected[0]

        caps = {
            self._cap_name(c).lower()
            for c in (ws_model.effective_capabilities or [])
        }
        if ModelCapability.GENERAL.value in caps:
            return ws_model

        cap = self._cap_name(capability).lower() if capability else ""
        if cap and cap in caps:
            return ws_model

        logger.info(
            "[models] Sole selected model %s declares %s but not 'general' — "
            "%s runs on the default model",
            ws_model.model_name, sorted(caps) or "no capability",
            f"capability '{cap}'" if cap else "this step",
        )
        return None

    def _pinned_llm(self, capability: Optional[str] = None) -> Optional[BaseLLM]:
        """Loaded instance of the pinned model, or ``None``.

        ``None`` means the user has not pinned the selection to one model, that
        the pin does not cover *capability* (see :meth:`_pinned_model`), or
        that the pinned model could not be loaded (no GGUF, or it does not
        fit) — callers then take their existing fallback path.
        """
        ws_model = self._pinned_model(capability)
        if ws_model is None:
            return None
        if self._remote_device() is not None:
            # The pin says *which* model; putting it in memory is the host's
            # business.  Loading the GGUF here would be a second copy of the
            # weights in the same device — and a fit check made by a process
            # that cannot see the device.
            return self._remote_llm(capability)
        return self._load_workspace_llm(ws_model, "pinned")

    # Why the most recent capability model could not be loaded, as
    # ``(model_name, cause, detail)`` where cause is one of "memory",
    # "missing", "faulted" or "error" and detail is a short human phrase or
    # None.  Advisory only — it feeds the progress event that explains a
    # fallback, so a race between concurrent fan-out threads costs at most a
    # wrong model name in one status line.
    _capability_skip: Optional[tuple] = None

    # Figures from the last failed fit check, as ``(needed_gb, free_gb)``.
    # Set by _large_model_fits and read by the caller that acts on its verdict.
    _last_fit_shortfall: Optional[tuple] = None

    def last_capability_skip(self) -> Optional[tuple]:
        """``(model_name, cause, detail)`` for the last capability model that
        could not be loaded, or ``None`` when the last resolution succeeded.

        Lets the caller explain a fallback rather than silently answering on a
        smaller model: "not enough memory for gemma4-e2b" is actionable — the
        user can close something or deselect a model — where a bare "Selected
        sl-mini" looks like the model they chose was simply ignored.
        """
        return self._capability_skip

    def _fit_shortfall_detail(self) -> Optional[str]:
        """``"needs ~4.9GB, 4.4GB free"`` from the last failed fit check.

        The numbers are the actionable part: they say how much to free, which
        "not enough memory" on its own does not.
        """
        shortfall = self._last_fit_shortfall
        if not shortfall:
            return None
        try:
            needed_gb, free_gb = shortfall
        except (TypeError, ValueError):
            return None
        return f"needs ~{needed_gb:.1f}GB, {free_gb:.1f}GB free"

    def _load_workspace_llm(self, ws_model, reason: str) -> Optional[BaseLLM]:
        """Return a resident (or freshly loaded) instance of one workspace
        model, or ``None`` when it has no GGUF, faulted, or won't fit.

        Shared by capability routing and the pinned-model path: consults the
        per-model cache, honours a GPU-fault eviction, applies the fit guard
        (evicting prior-step models to make room) and loads.  ``reason`` names
        the caller in the logs — a capability, or ``pinned``.
        """
        workspace = self._model.workspace
        cache_key = f"_cap_llm_{ws_model.model_name}"
        cached = getattr(self, cache_key, None)
        if cached is _GPU_FAULTED:
            # Evicted after a runtime GPU OOM — don't reload it.  That is a
            # memory failure too, just one discovered during generation rather
            # than at load time.
            self._capability_skip = (ws_model.model_name, "memory", None)
            return None
        if cached is not None:
            # Resident and already passed the fit guard when loaded.
            self._capability_skip = None
            return cached

        # A step routed to the orchestrator model (or the on-demand large one)
        # must reuse the resident instance rather than build a second llama.cpp
        # context over the same GGUF.  Duplicating the weights and KV cache of
        # a model already in memory is what trips a shared-memory GPU over its
        # allocation limit:
        #
        #     ggml_vulkan: Failed to allocate pinned memory
        #     (Requested buffer size exceeds device buffer size limit)
        #
        # Canonicalize both sides so a legacy workspace name (e.g. the pre-
        # rename ``slp-orchestra-mini``) still resolves to the resident model.
        canonical = canonical_model_name(ws_model.model_name)
        resident_orchestrator = getattr(self, "_llm", None)
        if (
            resident_orchestrator is not None
            and canonical_model_name(getattr(self, "_model_name", None)) == canonical
        ):
            self._capability_skip = None
            return resident_orchestrator
        resident_large = getattr(self, "_large_llm", None)
        if (
            resident_large is not None
            and canonical_model_name(getattr(self, "_large_model_name", None)) == canonical
        ):
            self._capability_skip = None
            return resident_large

        from smartloop.config.app_settings import AppSettings
        cap = AppSettings().llm_large_n_ctx_cap
        try:
            model_dir = workspace.model_dir_for(ws_model.model_id)
            # _find_model_ggufs excludes the mmproj/CLIP GGUF from the main
            # pick.  This path is text-only capability routing; image OCR
            # goes through get_vision_llm(), which loads the multimodal
            # handler.
            gguf_path, _mmproj = self._find_model_ggufs(model_dir)
            if not gguf_path:
                logger.warning(
                    "[capability-model] No GGUF found for %s at %s (%s)",
                    ws_model.model_name, model_dir, reason,
                )
                self._capability_skip = (ws_model.model_name, "missing", None)
                return None

            # Fit guard: skip a model that won't fit alongside the main
            # model.  First eagerly release prior steps' capability models
            # (and the on-demand large model) and retry — that frees enough
            # VRAM to run this model on a memory-tight device.
            if not self._large_model_fits(gguf_path, cap, ws_model.model_name):
                if not (
                    self._evict_capability_models(keep_model_name=ws_model.model_name)
                    and self._large_model_fits(gguf_path, cap, ws_model.model_name)
                ):
                    logger.warning(
                        "[capability-model] %s doesn't fit alongside the "
                        "main model at %s (%s)",
                        ws_model.model_name, _n_ctx_desc(cap), reason,
                    )
                    self._capability_skip = (
                        ws_model.model_name, "memory", self._fit_shortfall_detail(),
                    )
                    return None
                logger.info(
                    "[capability-model] Freed prior-step models to fit %s (%s)",
                    ws_model.model_name, reason,
                )
            from smartloop.llms.llm_factory import LlmFactory
            llm = LlmFactory.get_llama_cpp_instance(
                gguf_path, logger, self.get_project_id() or "",
                max_n_ctx=cap,
                thinking=self._thinking_for(ws_model.model_name),
                model_name=ws_model.model_name,
            )
            setattr(self, cache_key, llm)
            logger.info(
                "[capability-model] Loaded %s (%s)", ws_model.model_name, reason,
            )
            self._capability_skip = None
            return llm
        except Exception as e:
            logger.warning(
                "[capability-model] Failed to load %s (%s): %s",
                ws_model.model_name, reason, e,
            )
            self._capability_skip = (ws_model.model_name, "error", None)
            return None

    def _thinking_for(self, model_name: str) -> Optional[bool]:
        """Whether ``model_name`` should reason, per the picker's setting.

        Reasoning is a property of the model, not of a request: it is stored
        per model (and per project) like ``enabled`` and set from the model
        picker, so a model the user turned thinking on for reasons on every
        call without the client having to ask each time.  ``None`` means the
        model does not support it and nothing is bound.
        """
        from smartloop.llms.llm_factory import LlmFactory
        try:
            workspace = self._ensure_workspace()
        except Exception:
            return LlmFactory.thinking_for(model_name)
        return workspace.effective_model_thinking(
            model_name, workspace.current_project,
        )

    def apply_thinking_setting(self, model_name: str) -> bool:
        """Push a just-changed reasoning setting onto a resident model.

        The setting otherwise only takes effect on the next load, which for an
        always-resident model is never.  Re-flagging is a single attribute
        write that the bound chat handler reads on its next call (see
        ``LlmFactory._bind_thinking``) — no reload, no VRAM churn.  Returns
        True when a resident instance was updated.
        """
        value = self._thinking_for(model_name)
        if value is None:
            return False
        updated = False
        for name, llm in self._loaded_thinking_models():
            if name != model_name:
                continue
            try:
                setattr(llm.model, "_slp_thinking", value)
                updated = True
            except Exception as e:  # never fail a settings change over a hint
                logger.debug("[thinking] could not re-flag %s: %s", name, e)
        if updated:
            logger.info("[thinking] %s → enable_thinking=%s", model_name, value)
        return updated

    def _loaded_thinking_models(self):
        """(model_name, llm) for every resident model that can reason."""
        from smartloop.llms.llm_factory import LlmFactory
        pairs = []
        if self._model_name and self._llm is not None:
            pairs.append((self._model_name, self._llm))
        if self._large_model_name and self._large_llm is not None:
            pairs.append((self._large_model_name, self._large_llm))
        for attr, cached in list(self.__dict__.items()):
            if not attr.startswith("_cap_llm_") or cached in (None, _GPU_FAULTED):
                continue
            pairs.append((attr[len("_cap_llm_"):], cached))
        return [(n, l) for n, l in pairs
                if l is not None and LlmFactory.supports_thinking(n)]

    @staticmethod
    def _cap_name(capability) -> str:
        """The capability's value, for logs and messages.

        ``ModelCapability`` is a ``str`` enum, but on Python 3.11 formatting a
        member goes through ``Enum.__str__`` — so an f-string yields
        ``ModelCapability.GENERAL`` where every consumer of the value sees
        ``general``.  Consumers are unaffected (they call ``.lower()``), but a
        log line naming a member no one can grep for is worse than useless.
        """
        return str(getattr(capability, "value", capability))

    def supports_thinking(self, capability: Optional[str] = None) -> bool:
        """Whether the model that would answer for *capability* can reason.

        Asked by a project agent through the model host, which is why it lives
        here rather than being read off an LLM handle: in that split the
        weights are in this process and the caller only holds a socket.
        """
        llm = self._llm_for_response(capability)
        return bool(getattr(llm, "supports_thinking", False))

    def has_capability_model(self, capability: str) -> bool:
        """True when some selected model declares *capability*.

        A cheap workspace lookup — it loads nothing.  The orchestrator uses it
        to decide whether to announce a capability model load at all: the
        "Loading model for X…" event has to fire *before* the slow load, so
        without this check it promised a load on workspaces where no selected
        model provides X (only one supported model declares ``skills``, for
        instance) and the step then quietly fell back to the bootstrap model.
        """
        if not self._model or not self._model.workspace:
            return False
        # A pinned model serves every capability it can — declared or not for
        # a general-capable selection, only its own for a specialist one.
        if self._pinned_model(capability) is not None:
            return True
        workspace = self._model.workspace
        return bool(
            workspace.ranked_models_for(capability, workspace.current_project)
        )

    def _get_llm_for_capability(self, capability: str) -> Optional[BaseLLM]:
        if self._remote_device() is not None:
            # The host owns model selection: it can see the device, and a
            # child cannot, so "which model serves this capability" is not a
            # question this side is able to answer.
            return self._remote_llm(capability)

        """Return a loaded LLM matching the requested capability.

        When the user has selected exactly one model, that model is returned
        for every capability it can serve — see :meth:`_pinned_model`.

        Otherwise walks the capability's models largest-first and returns the
        first that is already resident or fits in memory — evicting prior-step
        capability models to make room when needed.  When the largest doesn't
        fit it promotes down to the next-largest same-capability model that
        does, rather than dropping straight to the tiny default.  Returns
        ``None`` only when no capable model fits (caller then falls back to the
        default/large model).
        """
        if not self._model or not self._model.workspace:
            logger.warning(
                "[capability-model] No workspace loaded — cannot resolve "
                "capability '%s'", capability,
            )
            return None
        workspace = self._model.workspace

        # One selected model runs every step, whatever capability the step
        # asked for.  Declared capabilities decide which of several selected
        # models handles a step; with a single selection there is no such
        # choice to make, and treating a missing 'dp' tag as "no capable
        # model" would hand the step to the bootstrap model instead of the
        # one the user picked.  A selection that declares no 'general' at all
        # is the exception: it covers only its own capability, so a vision-only
        # model gets the image step and not the conversation around it.
        pinned = self._pinned_model(capability)
        if pinned is not None:
            llm = self._load_workspace_llm(
                pinned, f"pinned, capability '{self._cap_name(capability)}'",
            )
            if llm is None:
                logger.warning(
                    "[capability-model] Sole selected model %s unavailable for "
                    "'%s' — falling back to the default model",
                    pinned.model_name, capability,
                )
            return llm

        # Candidate selection is project-scoped (per-project model settings
        # override the workspace defaults); re-rank so same-size models with
        # broader capability coverage lead.
        ranked = self._ranked_candidates(
            workspace.ranked_models_for(capability, workspace.current_project),
            ref_caps={capability},
        )
        if not ranked:
            logger.warning(
                "[capability-model] No enabled+downloaded model declares "
                "capability '%s'", capability,
            )
            return None

        for ws_model in ranked:
            # Walk down to the next-smaller capable model when this one has no
            # GGUF, faulted, or doesn't fit — rather than the tiny default.
            llm = self._load_workspace_llm(
                ws_model, f"capability '{self._cap_name(capability)}'",
            )
            if llm is not None:
                return llm

        logger.info(
            "[capability-model] No model declaring capability '%s' fits in "
            "memory — falling back to the default/large model", capability,
        )
        return None

    @staticmethod
    def _device_free_total_gb() -> Optional[tuple[float, float]]:
        """Return (free_gb, total_gb) for the active accelerator, or None.

        VRAM is not probed: for a discrete GPU the Vulkan backend manages device
        memory, so only Apple Silicon (unified memory) reports a fit budget.
        None otherwise (CPU/discrete GPU) — llama.cpp handles offload/fit.
        """
        try:
            from smartloop.utils.device_utils import is_mps
            if is_mps():
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
        # 0 = no cap — the factory auto-sizes the model's context from live
        # available memory (see LlmFactory.get_llama_cpp_instance), so the KV
        # cache fits by construction: it simply loads at a smaller window
        # instead of OOMing.  Only the bare weights + overhead footprint can
        # veto here; estimate no KV cache.  (Estimating it at the device bound
        # instead over-counts and rejects every capable model on a tight box.)
        est_ctx = cap if cap > 0 else 0
        kv_gb = 0.5 * (est_ctx / 4096)
        footprint_gb = model_gb + kv_gb + 0.7
        if footprint_gb + reserve_gb > free_gb:
            # Only suggest the cap when there is a cap to lower.  Uncapped, the
            # KV cache is already estimated at zero and the window auto-sizes,
            # so the figure above is bare weights plus overhead — lowering the
            # setting cannot move it, and offering the knob sends whoever reads
            # this after the one lever that will not help.
            remedy = (
                "Lower SLP_LLM_LARGE_N_CTX_CAP or free memory to enable it."
                if cap > 0 else
                "Free memory to enable it — SLP_LLM_LARGE_N_CTX_CAP is unset, "
                "so the context window already auto-sizes and this is the "
                "weights alone."
            )
            logger.warning(
                "[large-model] %s needs ~%.1fGB (+%.1f reserve) but only %.1fGB "
                "free of %.1fGB — falling back to the default model. %s",
                label or os.path.basename(gguf_path),
                footprint_gb, reserve_gb, free_gb, total_gb, remedy,
            )
            # Carry the figures out for the progress event.  The reserve is
            # folded into the requirement because that is the threshold that
            # actually gated the load — quoting the footprint alone would show
            # a number smaller than the free memory and read as a contradiction.
            self._last_fit_shortfall = (footprint_gb + reserve_gb, free_gb)
            return False
        self._last_fit_shortfall = None
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
        if pref == "cpu" or self._vision_clip_cpu:
            return True
        ft = self._device_free_total_gb()
        if ft is None:
            # VRAM unknown (e.g. a Vulkan llama.cpp build beside a CPU-only
            # torch wheel).  Follow the build: when llama.cpp supports GPU
            # offload the projector stays on GPU with the model (Vulkan
            # included); a CPU-only build runs it on CPU regardless of use_gpu.
            try:
                from llama_cpp import llama_supports_gpu_offload
                return not llama_supports_gpu_offload()
            except Exception:
                return False
        free_gb, _ = ft
        try:
            model_gb = os.path.getsize(gguf_path) / 1024**3
        except OSError:
            return True
        # Projector weights (~0.8GB) + per-image encode activation headroom.
        # flash_attn is CUDA-only (see get_llama_cpp_instance); without it the
        # clip encode of a large image allocates a ~6GB compute buffer instead
        # of ~3GB, so non-CUDA backends need correspondingly more free VRAM.
        from smartloop.llms.llm_factory import _llama_cpp_has_cuda_backend
        act_gb = 1.8 if _llama_cpp_has_cuda_backend() else 6.0
        need_gb = model_gb + 0.5 * (cap / 4096) + act_gb
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
            project = workspace.current_project
            # Compare against the orchestrator model (the loaded small model),
            # not the workspace-active model: the large model is whatever
            # downloaded model is bigger than the orchestrator model (e.g. the
            # user's active 4B).
            orchestrator = workspace.get_model(self._model_name)
            orchestrator_size = orchestrator.param_size_b if orchestrator else 0
            orchestrator_rank = orchestrator._rank if orchestrator else (0, 0.0)
            # Pick the agent/large model from any general-capable (text) model —
            # filtering by GENERAL excludes vision-only models (e.g. qwen2-vl-2b
            # which only declares VISION) without requiring a specific SKILLS tag.
            # Walk largest-first; when the biggest doesn't fit, fall to the next
            # lower tier (e.g. gemma4-4b → gemma4-2b) rather than the tiny router.
            _text_caps = {ModelCapability.GENERAL, ModelCapability.SKILLS, ModelCapability.DP}
            agent_models = [
                m for m in workspace.models
                if workspace.effective_model_enabled(m.model_name, project)
                and workspace.effective_model_downloaded(m.model_name)
                and m.model_name != self._model_name
                and _text_caps & set(m.effective_capabilities)
            ]
            if not agent_models:
                logger.info(
                    "[large-model] No general-capable model downloaded — using the "
                    "router model for agent/synthesis tasks"
                )
                return
            # Rank largest→smallest; same-size models with more cap overlap win.
            agent_models = self._ranked_candidates(
                agent_models, ref_caps=_text_caps, descending=True,
            )

            from smartloop.config.app_settings import AppSettings
            cap = AppSettings().llm_large_n_ctx_cap

            for largest in agent_models:
                largest.models_dir = workspace.models_dir
                # Sorted largest-first: once a candidate no longer out-ranks the
                # router, none of the remaining (smaller) ones will either.
                if largest._rank <= orchestrator_rank:
                    break

                model_dir = workspace.model_dir_for(largest.model_id)
                # Exclude the mmproj/CLIP GGUF — it is not loadable as a main
                # model (e.g. gemma-3-4b-it ships one upstream even though we
                # don't use it for vision); loading it as weights crashes the
                # native loader.
                gguf_path, _mmproj = self._find_model_ggufs(model_dir)
                if not gguf_path:
                    logger.info(
                        "[large-model] GGUF not found for %s, trying next-smaller",
                        largest.model_name,
                    )
                    continue

                # Fit guard: skip a model whose capped footprint won't fit in
                # free VRAM after the reserve (free VRAM already excludes the
                # resident main model).  When it doesn't fit, promote down to the
                # next-largest model instead of dropping straight to the router.
                if not self._large_model_fits(gguf_path, cap, largest.model_name):
                    logger.info(
                        "[large-model] %s doesn't fit — trying next-smaller "
                        "model", largest.model_name,
                    )
                    continue

                logger.info(
                    "[large-model] Loading larger model on demand: %s (%.0fB) > "
                    "router %s (%.0fB), n_ctx<=%d — %s",
                    largest.model_name, largest.param_size_b,
                    self._model_name or "?", orchestrator_size, cap, gguf_path,
                )
                from smartloop.llms.llm_factory import LlmFactory
                self._large_llm = LlmFactory.get_llama_cpp_instance(
                    gguf_path, logger, self.get_project_id() or "", max_n_ctx=cap,
                    thinking=self._thinking_for(largest.model_name),
                    model_name=largest.model_name,
                )
                self._large_model_name = largest.model_name
                logger.info("[large-model] Loaded %s for agent/skill tasks", largest.model_name)
                return
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

    def search_workspace(
        self,
        query: str,
        work_dir: str,
        n_results: int = 10,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> list:
        """Search the workspace-files collection.

        Uses the *same* selection criteria as document search
        (:meth:`_vector_search_grouped`): the query is enriched from history
        for reference/keyword emphasis, recalled broadly, then filtered by
        the shared quality-floor + relevance-cutoff helper — instead of a
        divergent fixed threshold.  Indexing is done in the background by
        :class:`IndexerTask`, so this only queries the store.

        Returns a list of :class:`~smartloop.db.store.ChunkResult` ordered
        by relevance.
        """
        if self._model is None or not work_dir or not os.path.isdir(work_dir):
            return []
        current_project = self._model.workspace.current_project
        if not current_project:
            return []
        cache = self._caches(current_project.id)
        try:
            if cache.workspace_indexer is None:
                cache.workspace_indexer = WorkspaceIndexer(
                    project_dir=current_project.project_dir,
                    logger=logger,
                    base_dir=self._model.workspace.workspace_dir,
                )
            indexer = cache.workspace_indexer

            # Same emphasis step as documents: resolve references / keywords
            # against history so follow-ups are grounded.
            search_query = query
            if conversation_history:
                search_query = self._build_history_search_query(query, conversation_history)

            recall = max(n_results, 20)
            chunks = indexer.search(
                search_query, n_results=recall, max_distance=_RECALL_MAX_DISTANCE,
            )
            # Raw-query fallback, mirroring the document path.
            if not chunks and search_query != query:
                chunks = indexer.search(
                    query, n_results=recall, max_distance=_RECALL_MAX_DISTANCE,
                )
            return _select_relevant_chunks(chunks, label="workspace-search")
        except Exception as e:
            logger.warning(f"[workspace-index] search_workspace failed: {e}")
            cache.workspace_indexer = None
            return []

    # --------------------------------------------------------------- feedback

    def _ensure_feedback_indexer(self) -> FeedbackIndexer:
        current_project = self._model.workspace.current_project
        if not current_project:
            raise ValueError("No active project")
        cache = self._caches(current_project.id)
        if cache.feedback_indexer is None:
            cache.feedback_indexer = FeedbackIndexer(
                project_dir=current_project.project_dir,
                logger=logger,
                base_dir=self._model.workspace.workspace_dir,
            )
        return cache.feedback_indexer

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
        """Search stored feedback Q&A using the shared relevance criteria.

        Feedback was the one retrieval path with no quality floor: it recalled
        its nearest neighbours out to a fixed 0.65 distance and returned them
        all.  A single stored Q&A on an unrelated topic therefore matched every
        question, poisoning the context and — because any non-empty local hit
        counts as a primary source — suppressing web search.  Recall broadly
        and then apply the same floor/cutoff as documents and workspace files.
        """
        if self._model is None:
            return []
        try:
            indexer = self._ensure_feedback_indexer()
            chunks = indexer.search(
                query, n_results=max(n_results, 20),
                max_distance=_RECALL_MAX_DISTANCE,
            )
            kept = _select_relevant_chunks(chunks, label="feedback-search")
            return kept[:n_results]
        except Exception as e:
            logger.warning(f"[feedback-index] search_feedback failed: {e}")
            project = self._model.workspace.current_project
            if project:
                self._caches(project.id).feedback_indexer = None
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
        # Match on the stable id, not the display name: records saved under
        # an earlier name (e.g. "Smartloop") would otherwise never be found,
        # so their tool list stays stale and a duplicate server with the
        # same id gets appended — tool resolution then keeps hitting the
        # stale record, which wins the id-first find_server lookup.
        existing = next(
            (s for s in project.mcp_servers if s.id == SYSTEM_TOOLS_SERVER_ID), None,
        )
        if existing is not None:
            # Refresh the advertised name/tool list so projects created
            # before a rename or before a tool was added (e.g. grep) pick
            # up the changes on next load.
            current = [t.name for t in (existing.tools or [])]
            wanted = [t.name for t in self._SYSTEM_TOOL_DEFS]
            if current != wanted or existing.name != SYSTEM_TOOLS_SERVER_NAME:
                existing.name = SYSTEM_TOOLS_SERVER_NAME
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

    # Skills that shipped with the old fetch MCP server; obsolete now that
    # fetching is handled in-process by the FetchAgent.
    _STALE_FETCH_SKILLS = ("fetch-mcp", "fetch-url")

    def remove_fetch_server(self, project_id: str) -> None:
        """Remove the legacy fetch MCP server — fetching is now the FetchAgent.

        URL fetching moved from an MCP server (``__fetch__``) to an in-process
        agent resolved as a plan step, so any previously installed fetch server
        is dropped from the project.
        """
        project = self._get_project(project_id)
        before = len(project.mcp_servers)
        project.mcp_servers = [
            s for s in project.mcp_servers if s.id != FETCH_SERVER_ID
        ]
        if len(project.mcp_servers) != before:
            self._model.save()
            logger.info(f"Removed legacy fetch server from project {project_id}")
        self._remove_stale_fetch_skills(project)

    def _remove_stale_fetch_skills(self, project: "Project") -> None:
        """Delete leftover fetch skill directories from disk.

        Projects created before the FetchAgent change may still have
        ``fetch-mcp``/``fetch-url`` SKILL.md directories on disk. They reference
        the removed ``__fetch__`` server, so drop them.
        """
        skills_dir = self._skills_dir(project)
        for name in self._STALE_FETCH_SKILLS:
            skill_dir = os.path.join(skills_dir, name)
            if os.path.isdir(skill_dir):
                shutil.rmtree(skill_dir, ignore_errors=True)
                logger.info(
                    f"Removed stale fetch skill '{name}' from project {project.id}"
                )

    def _migrate_servers_to_local(self, project: "Project") -> None:
        """Migrate legacy REMOTE system servers to LOCAL (stdio subprocess)."""
        migrated = False
        # command → how that launcher takes a package name.
        _LAUNCHERS = {
            "uvx": lambda pkg: ["--from", pkg, pkg],
            "uv": lambda pkg: ["tool", "run", "--from", pkg, pkg],
            "npx": lambda pkg: ["-y", pkg],
            "bunx": lambda pkg: [pkg],
            "pipx": lambda pkg: ["run", pkg],
            "deno": lambda pkg: ["run", "-A", pkg],
        }
        for server in project.mcp_servers:
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
            #
            # How a package name is passed is the launcher's own convention —
            # `--from <pkg> <pkg>` is uv's way of naming package and executable
            # separately, and handing those three arguments to npx or deno
            # produces a command that cannot start.
            if (
                server.server_type == MCPServerType.LOCAL
                and server.config
                and server.config.command in _LAUNCHERS
                and not server.config.args
            ):
                pkg_name = server.pkg or server.name
                server.config.args = _LAUNCHERS[server.config.command](pkg_name)
                migrated = True
                logger.info(
                    "Fixed empty args for local MCP server '%s' → %s %s",
                    server.name, server.config.command, server.config.args,
                )
        if migrated:
            self._model.save()

    # Built-in tool/system skills that back the orchestrator's routing. The
    # orchestrator still uses them, but they must never surface in the
    # user-facing skill list (nor be user-selectable): file/shell tool skills
    # plus any ``__``-prefixed internal skill (e.g. ``__system_mcp__``).
    _SYSTEM_SKILL_NAMES = frozenset({
        "file-read", "file-write", "bash", "grep", "explore",
    })

    @classmethod
    def _is_system_skill(cls, name: str) -> bool:
        """True for built-in tool skills and ``__``-prefixed internal skills."""
        return bool(name) and (
            name in cls._SYSTEM_SKILL_NAMES or name.startswith("__")
        )

    _DEFAULT_SKILLS = [
        Skill(
            id="file-read",
            name="file-read",
            description="Use when the user wants to read, view, or inspect the contents of a file.",
            content="read the file at the given path and return its content",
        ),
        Skill(
            id="file-write",
            name="file-write",
            description="Use when the user wants to write, save, or create a file with specific content.",
            content="write content to a file at the given path",
        ),
        Skill(
            id="bash",
            name="bash",
            description="Use when the user wants to run a shell command, execute a script, or perform system operations.",
            content="execute the shell command on the host machine",
        ),
    ]

    def _setup_project_tools(self, project: "Project") -> None:
        """Install system MCP servers, generate workspace docs, and register default skills."""
        project_id = project.id
        self._migrate_servers_to_local(project)
        self.install_system_tools_server(project_id)
        self.remove_fetch_server(project_id)
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

        # Skills are authored directly (indexed and matched by user intent);
        # nothing is LLM-generated here. Persist the provided content/metadata.
        resolved_name = name or meta.get("name")
        raw_name = resolved_name or self._extract_skill_name(body)
        sanitized_name = re.sub(r"[^a-z0-9]+", "-", raw_name.lower()).strip("-") or f"skill-{str(uuid4())[:8]}"

        skill = Skill(
            id=sanitized_name,
            name=sanitized_name,
            description=final_description,
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
    def _project_documents_dir(project: "Project") -> str:
        """Return the project's documents directory (under ``project_dir``)."""
        return normalize_path(project.documents_dir)

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
        skill_dir = os.path.join(ModelService._skills_dir(project), skill.name)
        os.makedirs(skill_dir, exist_ok=True)
        skill_path = os.path.join(skill_dir, "SKILL.md")
        Path(skill_path).write_text(ModelService._render_skill_content(skill), encoding="utf-8")

    @staticmethod
    def _skills_dir(project: "Project") -> str:
        """Return the skills directory for a project (under ``project_dir``)."""
        return normalize_path(project.skills_dir)

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
        """List a project's user-created skills, read from disk.

        Built-in tool skills (file-read/file-write/bash/grep/explore) and
        ``__``-prefixed internal skills back the orchestrator's routing but are
        never exposed here, so the client only sees skills the user created.
        """
        project = self._get_project(project_id)
        return [
            s.model_dump()
            for s in self._read_skills_from_disk(project)
            if not self._is_system_skill(s.name)
        ]

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
            cache = self._caches(current_project.id)
            if cache.doc_store is None:
                cache.doc_store = StoreFactory.create_store(
                    current_project.project_dir, logger=logger, base_dir=workspace.workspace_dir
                )
            store = cache.doc_store

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

            # Apply the shared relevance-selection criteria (quality floor +
            # relevance cutoff).  Without the floor, weak matches populate
            # vector_context and force the orchestrator's tier-1 classifier
            # onto the EXPLORE path, wrapping a conversational prompt in
            # DOCUMENT CONTEXT scaffolding pointing at irrelevant docs.  The
            # same helper drives workspace search so both select identically.
            discovery = _select_relevant_chunks(discovery, label="vector-search")
            if not discovery:
                return None

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
                    # Carry the exact text this document contributed to the
                    # prompt, privately (``_`` keys are stripped before a
                    # citation reaches a client).  Being retrieved is not
                    # evidence a document was used — nearest neighbour always
                    # returns something — so the router checks the finished
                    # answer against this before crediting the source.
                    extract = doc_groups.get(chunk.document_name or chunk.document_id)
                    if extract:
                        citation["_extract"] = extract
                    citation["_chunks"] = []
                    citation["_vectors"] = []
                    if page:
                        citation["page_numbers"] = [page]
                    seen_docs[chunk.document_id] = citation
                elif page:
                    existing = seen_docs[chunk.document_id]
                    pages = existing.get("page_numbers", [])
                    if page not in pages:
                        pages.append(page)
                        existing["page_numbers"] = sorted(pages)

                # The chunks that actually matched, closest first — the
                # passages this document was retrieved *for*.  ``_extract``
                # above is the whole document (the expansion step fetches every
                # chunk of every matched file), and checking a finished answer
                # against a whole document credits almost anything: a long file
                # covers enough ground that some page of it looks related to
                # any answer.  These are what the router scores instead.
                #
                # Their vectors ride along too.  They were computed when the
                # document was indexed and read back a moment ago to rank this
                # very search, so the router can check them against the answer
                # without encoding a single character of source text — the
                # answer itself is the only thing left to embed.
                existing = seen_docs[chunk.document_id]
                chunk_text = (chunk.text or "").strip()
                if chunk_text and len(existing["_chunks"]) < _MAX_CITATION_CHUNKS:
                    existing["_chunks"].append(chunk_text)
                    existing["_vectors"].append(chunk.embedding)

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
    def _build_history_search_query(
        self, user_query: str, conversation_history: List[Dict[str, str]]
    ) -> str:
        """Resolve references in the search query using conversation history.

        Whenever the session has prior turns, history is always consulted to
        resolve references (pronouns, omitted subjects, definite-article
        follow-ups like "the place") in the latest message.  The rewrite
        prompt returns a self-contained message unchanged, so there is no
        semantic gate here — every follow-up in a session gets the chance to
        be grounded against its history.

        The *conversation_history* is expected to be pre-filtered for
        relevance by the caller — no additional windowing is applied.
        """
        if len(conversation_history) <= 1:
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
        remote = self._remote_device()
        if remote is not None:
            # A child has no tokenizer of its own; the host's is the only one
            # that matches the model the prompt will actually be run on.
            try:
                return int(remote.count_tokens(text))
            except Exception:
                return 0
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
        remote = self._remote_device()
        if remote is not None:
            try:
                return int(remote.max_context_tokens()) or DEFAULT_MAX_CONTEXT_TOKENS
            except Exception:
                return DEFAULT_MAX_CONTEXT_TOKENS
        if self._llm is None:
            return DEFAULT_MAX_CONTEXT_TOKENS
        try:
            return self._llm.get_max_tokens()
        except Exception:
            return DEFAULT_MAX_CONTEXT_TOKENS

    def get_response_max_context_tokens(self, capability: Optional[str] = None) -> int:
        """Max context window (tokens) of the model that will synthesize the
        final response — capability-tagged → largest → default.

        The default model's window (see ``get_max_context_tokens``) is often
        larger than the capability model that actually consumes the prompt.
        Sizing the response context against it lets web/document context
        overflow the response model: generation aborts, the request falls back
        to the default model, and the search content never reaches the LLM.
        Return 0 when no model is loaded.
        """
        llm = self._llm_for_response(capability)
        if llm is None:
            return self.get_max_context_tokens()
        try:
            return llm.get_max_tokens()
        except Exception:
            return DEFAULT_MAX_CONTEXT_TOKENS

    def is_loaded(self) -> bool:
        """Whether this service can serve a request right now.

        With the device in another process, that is two questions and both
        have to hold: the host must have weights, *and* this process must hold
        a handle to them.  Answering with the host's half alone tells the
        caller there is nothing to load — so ``load_model`` never runs here,
        ``self._llm`` stays ``None``, and the orchestrator then finds no LLM
        and silently bypasses itself into raw generation.
        """
        remote = self._remote_device()
        if remote is not None:
            if self._llm is None:
                return False
            try:
                return bool(remote.is_loaded())
            except Exception:
                return False
        return self._llm is not None

    @staticmethod
    def _model_label(llm: BaseLLM) -> str:
        """Human-readable name for a loaded LLM, from its GGUF metadata.

        Falls back to the class name when metadata is unavailable so the
        response log always names *something* concrete.  The metadata name is
        mapped to the product's own (see ``display_model_label``) so logs and
        events name the model the user sees in the picker.
        """
        # A handle to a model in another process carries the label its owner
        # reports; there is no local GGUF to read metadata from, and falling
        # through would put the class name on a user-facing status event.
        # Must be a real string: ``getattr`` on a test double answers every
        # attribute with a truthy mock, which would shadow the metadata path
        # for every locally-loaded model.
        explicit = getattr(llm, "model_label", None)
        if isinstance(explicit, str) and explicit:
            return explicit
        try:
            meta = llm.model.metadata
            name = display_model_label(meta.get("general.name", ""))
            size = meta.get("general.size_label", "")
            if name:
                return f"{name} ({size})" if size else name
        except Exception:
            pass
        return type(llm).__name__

    def response_model_label(self, capability: Optional[str] = None) -> str:
        """Name the model that will actually write the response.

        Deliberately the *same* resolution the response itself goes through —
        :meth:`_llm_for_response`, with its pinned-model, capability, large
        fallback and default chain — and not a cheaper "what is resident right
        now" approximation.  This label goes on the progress event a user
        reads, so it has to name the model that answers; an approximation is
        free to name a different one, which is worse than no label at all.

        The selection is what the single-process server ran anyway, and it is
        idempotent for a resident model, so asking it here costs nothing the
        request was not about to spend.
        """
        return self._model_label(self._llm_for_response(capability))

    # What the user is told the response model is being resolved for.  The
    # capability does the routing; this is the label on the event.
    _RESPONSE_PURPOSE = "the response"

    @property
    def _response_router(self) -> ModelRouter:
        """Reports response-model selection, the same way the orchestrator does.

        The response model is resolved here rather than inside the orchestrator
        — it is picked after orchestration finishes, by the streaming layer —
        so without this it was the one model resolution that surfaced nothing:
        a cold load landed between the last orchestration event and the first
        token, as dead air exactly where the answer was expected to start.

        Routing it through :class:`ModelRouter` rather than emitting ad hoc
        means it inherits the same rules as every other resolution: no load is
        announced unless one can happen, the event names the work rather than
        the capability tag, and a fallback says why.

        Held on the service (the streaming layer has no orchestrator to borrow
        one from) and re-armed per request by :meth:`begin_response_request`.
        """
        router = getattr(self, "_response_router_instance", None)
        if router is None:
            router = ModelRouter(
                orchestrator_llm=self._llm,
                agent_llm=self._llm,
                capability_llm_provider=self._get_llm_for_capability,
                capability_available=self.has_capability_model,
                capability_skip_reason=self.last_capability_skip,
            )
            self._response_router_instance = router
        # The resident model can change between requests (a model switch
        # reloads it), so keep the router pointing at the current one.
        router.orchestrator_llm = self._llm
        router.agent_llm = self._llm
        return router

    def begin_response_request(
        self, progress_callback: Optional[Callable[[str, str, str], None]] = None,
    ) -> None:
        """Install this request's progress sink for response-model events.

        Mirrors :meth:`ModelRouter.begin_request`, which the orchestrator calls
        for its own phase.  Clearing per request also resets the once-per-
        request dedupe, so a second question reports its model again.
        """
        self._response_router.begin_request(progress_callback)

    def announce_response_model(self, capability: Optional[str]) -> None:
        """Announce a pending response-model load *before* it blocks.

        Resolution happens inside a synchronous call the streaming generator
        cannot interleave with, so the warning has to be emitted ahead of it.
        Only when something could actually load — see
        :meth:`has_capability_model`.
        """
        if self._pinned_model(capability) is not None or (
            capability and self.has_capability_model(capability)
        ):
            self._response_router.emit_model_loading(
                capability, purpose=self._RESPONSE_PURPOSE,
            )

    def _report_response_model(
        self, capability: Optional[str], llm: Optional[BaseLLM],
    ) -> None:
        """Name the model that will write the answer.

        Silent when the always-resident router model answers and no capability
        was asked for — nothing was loaded and nothing was overridden, so there
        is nothing to report.  When a capability *was* asked for and we still
        landed on the router model, that is a fallback and says why.
        """
        if llm is None:
            return
        if llm is not self._llm:
            self._response_router.emit_model_selected(
                capability, llm, purpose=self._RESPONSE_PURPOSE,
            )
        elif capability:
            self._response_router.emit_model_fallback(
                capability, llm, purpose=self._RESPONSE_PURPOSE,
            )

    def _llm_for_response(self, capability: Optional[str]) -> Optional[BaseLLM]:
        """Return the LLM to use for the final response.

        With exactly one model selected, that model writes the response —
        including when *capability* is None, which otherwise answers on the
        bootstrap orchestrator LLM (see :meth:`_pinned_model`; a vision-only
        selection writes only the response to a vision-tagged step).  When it
        cannot be loaded the default model answers instead: the rest of the
        chain below would only rediscover the same unavailable model, and an
        answer on the default beats no answer at all.

        Fallback chain when *capability* is set: capability-tagged model →
        largest downloaded model → default orchestrator LLM.  Small models
        struggle to synthesize long tool-result contexts, so we prefer any
        larger model when one is available; falls back cleanly to
        ``self._llm`` when nothing better is present.
        """
        pinned_model = self._pinned_model(capability)
        if pinned_model is not None:
            pinned = self._pinned_llm(capability)
            if pinned is not None:
                # DEBUG, not INFO: this resolution runs for *every* generation
                # the process serves, not only the final answer — with a model
                # host in front of it, that is each of a project agent's
                # planning, matching and resolution calls, each logging
                # "[response] … (capability='None')" for a decision that was
                # never about the response.  Which model wrote the answer is
                # reported once, by `_report_response_model`.
                logger.debug(
                    "[response] Using the sole selected model (capability='%s')",
                    self._cap_name(capability),
                )
                self._report_response_model(capability, pinned)
                return pinned
            logger.warning(
                "[response] Sole selected model %s unavailable — answering on "
                "the default model", pinned_model.model_name,
            )
            self._report_response_model(capability, self._llm)
            return self._llm

        if capability:
            # These three are DEBUG for the same reason the sole-selected-model
            # line above is: this resolution answers every question asked of
            # the service, not just "who writes the answer".  A project agent's
            # remote handle asks for the capability's context window, whether
            # it reasons and how to label it before generating anything, so one
            # MCP tool-resolution call printed the same decision four to eight
            # times — and printed it as "for final response", which that call
            # is not.  The outcome is reported once, where it belongs:
            # ``[models] RESPONSE → …`` for the answer, ``[models] RESOLVE(…)``
            # for a tool step, and the progress SSE for the user.
            cap_llm = self._get_llm_for_capability(capability)
            if cap_llm is not None and cap_llm is not self._llm:
                logger.debug(
                    "[response] Using capability='%s' LLM",
                    self._cap_name(capability),
                )
                self._report_response_model(capability, cap_llm)
                return cap_llm
            large_llm = self._get_large_llm()
            if large_llm is not None and large_llm is not self._llm:
                logger.debug(
                    "[response] Capability='%s' unavailable — falling back to large LLM",
                    self._cap_name(capability),
                )
                self._report_response_model(capability, large_llm)
                return large_llm
            logger.debug(
                "[response] Capability='%s' unavailable and no larger model — using default LLM",
                self._cap_name(capability),
            )
        self._report_response_model(capability, self._llm)
        return self._llm

    def _default_fallback_llm(self, failed_llm: BaseLLM) -> Optional[BaseLLM]:
        """The default router LLM to fall back to when a larger response model
        fails during generation, or None when the failed model already IS the
        default (there's nothing smaller/lighter to retry on)."""
        if failed_llm is self._llm:
            return None
        return self._llm

    def _evict_response_llm(self, llm: BaseLLM) -> None:
        """Stop routing responses to a model that faulted during generation.

        A larger model can pass the load-time fit guard yet still exhaust GPU
        memory once generation allocates its compute buffers (e.g. Vulkan
        ``ErrorOutOfDeviceMemory``).  Drop it from the response caches so later
        requests route straight to the default model instead of re-loading a
        model that won't fit the GPU.
        """
        if llm is self._large_llm:
            logger.warning(
                "[response] Large model %s faulted on GPU — evicting; "
                "routing responses to the default model",
                self._large_model_name,
            )
            self._large_llm = None
            self._large_model_name = None
            # Latch so _get_large_llm doesn't re-attempt the load this session.
            self._large_model_attempted = True
            return
        for attr in list(vars(self)):
            if attr.startswith("_cap_llm_") and getattr(self, attr) is llm:
                logger.warning(
                    "[response] Capability model %s faulted on GPU — evicting; "
                    "routing responses to the default model",
                    attr[len("_cap_llm_"):],
                )
                setattr(self, attr, _GPU_FAULTED)
                return

    def stream_completion(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        cancelled: Optional[threading.Event] = None,
        capability: Optional[str] = None,
        thinking: Optional[bool] = None,
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
            thinking: Force the reasoning block on/off for this generation.
                      The answer passes False — see the chat router.

        Yields:
            GenerationChunk objects from the LLM
        """
        llm = self._llm_for_response(capability)
        if llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info(
            "[models] RESPONSE → %s (capability=%s)",
            self._model_label(llm), self._cap_name(capability),
        )

        # Lock around inference - llama-cpp is not thread-safe
        with self._inference_slot():
            emitted = 0
            try:
                # Propagate cancellation event so the LLM can stop mid-inference.
                llm._cancelled = cancelled
                for chunk in llm.stream(prompt, thinking=thinking):
                    if cancelled is not None and cancelled.is_set():
                        break
                    emitted += 1
                    yield chunk
            except Exception as ex:
                fallback = self._default_fallback_llm(llm)
                # A larger response model can OOM the GPU once generation
                # allocates its compute buffers (Vulkan ErrorOutOfDeviceMemory).
                # Re-route to the default router model, which is small enough to
                # fit — but only when nothing has been streamed yet, since a
                # mid-stream restart would duplicate the emitted text (the GPU
                # OOM aborts before the first token, so this holds in practice).
                if emitted == 0 and fallback is not None:
                    logger.warning(
                        "[response] Response model failed during generation (%s) "
                        "— falling back to the default model", ex,
                    )
                    self._evict_response_llm(llm)
                    llm._cancelled = None
                    try:
                        fallback._cancelled = cancelled
                        for chunk in fallback.stream(prompt):
                            if cancelled is not None and cancelled.is_set():
                                break
                            yield chunk
                    finally:
                        fallback._cancelled = None
                else:
                    raise
            finally:
                llm._cancelled = None

    def generate_completion(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        capability: Optional[str] = None,
        max_tokens: Optional[int] = None,
        thinking: Optional[bool] = None,
    ) -> str:
        """
        Generate a non-streaming completion.

        The caller must call ``load_model()`` before invoking this method.

        Args:
            prompt: The formatted prompt string
            model_name: Unused, kept for caller compatibility
            capability: When set, prefer the LLM matching this capability
                        over ``self._llm`` for the final response.
            max_tokens: Optional cap on the generated tokens.
            thinking: Force the reasoning block on/off for this call. Both of
                      these belong to the request, and travel with it from a
                      project agent through the model host (see
                      ``smartloop.model_host.client``).

        Returns:
            The generated text response
        """
        llm = self._llm_for_response(capability)
        if llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        with self._inference_slot():
            try:
                return llm.invoke(prompt, max_tokens=max_tokens, thinking=thinking)
            except Exception as ex:
                # Same GPU-OOM fallback as stream_completion: retry the
                # non-streaming response on the default router model.
                fallback = self._default_fallback_llm(llm)
                if fallback is None:
                    raise
                logger.warning(
                    "[response] Response model failed during generation (%s) "
                    "— falling back to the default model", ex,
                )
                self._evict_response_llm(llm)
                return fallback.invoke(prompt)

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
            "added": [doc.to_info(project.project_dir).model_dump() for doc in (new_docs or [])]
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
            "documents": [doc.to_info(project.project_dir).model_dump() for doc in new_docs],
        }

    def upload_document_to_inbox(self, project_id: str, filename: str, file_bytes: bytes, cwd: str = None) -> dict:
        """Drop an uploaded file into the project's documents inbox folder.

        The indexer's ``_scan_documents`` loop picks it up, registers it in
        the workspace config, and starts processing automatically.
        Returns immediately with ``status=accepted``.
        """
        project = self._get_project(project_id)
        # ``cwd`` is accepted for API compatibility and ignored: uploads always
        # land in the project's own documents folder.
        inbox_dir = Path(self._project_documents_dir(project))
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
            doc_map[did].to_info(project.project_dir) if did in doc_map
            else DocumentInfo(id=did, status="unknown")
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

        def _on_download_complete(task, status, _message):
            from smartloop.tasks.base_task import TaskStatus
            if status == TaskStatus.COMPLETED:
                # Re-arm large-model detection so the just-downloaded model is
                # considered on the next orchestration request.
                self._large_model_attempted = False

        task = DownloaderTask(
            workspace=workspace,
            downloader=downloader,
            logger=logger,
            model_id=model_id,
            progress_callback=progress_callback,
            on_status_change=_on_download_complete,
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

        # MCP task (auto=True — owns background provisioning of newly
        # registered servers)
        mcp = MCPTask(
            logger=logger,
            provisioner=self._provision_mcp_server,
            pending_provider=self.pending_mcp_servers,
        )
        self._task_queue.register(mcp)

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

        # Start the MCP loop: it adopts any server left pending by a previous
        # run and runs the first content refresh cycle immediately, so a
        # session opens with content that tracks the source.
        mcp.start_loop()

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

        return [doc.to_info(current_project.project_dir).model_dump()
                for doc in current_project.documents]

    def get_documents_for_project(self, project_id: str) -> list[dict]:
        """List documents for a project.

        Reads the project's ``documents/`` folder as the source of truth for
        which files exist, then enriches each entry with
        status information from the workspace config.  Files that have been
        dropped into the folder but not yet picked up by the indexer appear
        with status=pending and no id.
        """
        project = self._get_project(project_id)
        inbox_dir = Path(self._project_documents_dir(project))

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

        if inbox_dir.is_dir():
            for file_path in sorted(inbox_dir.iterdir()):
                if not file_path.is_file():
                    continue
                doc = registered_by_path.get(str(file_path)) or registered_by_name.get(file_path.name)
                if doc:
                    seen_ids.add(doc.id)
                    results.append(doc.to_info(project.project_dir).model_dump())
                else:
                    # File is in the inbox but not yet registered by the indexer
                    results.append({
                        "id": None,
                        "path": str(file_path),
                        "filename": file_path.name,
                        "status": "pending",
                        "summary": None,
                    })

        # Append any registered docs whose source is not in the inbox folder
        # (e.g. URLs or external paths added before this change).
        for doc in project.documents:
            if doc.id not in seen_ids:
                results.append(doc.to_info(project.project_dir).model_dump())

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
        # Seed the project's model settings from the workspace (app-default)
        # model registry so the project sidebar starts from the defaults.
        workspace.sync_project_models(project)
        self._model.workspace.projects.append(project)
        self._model.save()

        # A caller-supplied ``cwd`` is recorded for reference only — nothing is
        # ever located through it.  Skills, documents, MCP docs and the
        # workspace index all live under ``project_dir``, so the framework
        # stays self-contained: copy <home_dir> anywhere and it runs as-is.
        if cwd:
            project.cwd = os.path.abspath(os.path.expanduser(cwd))
        os.makedirs(project.skills_dir, exist_ok=True)
        os.makedirs(project.documents_dir, exist_ok=True)
        self._model.save()
        logger.info(f"[create_project] State dir: {project.project_dir}")

        # Drop any slot left over from a project that reused this id, so the
        # fresh directory is opened rather than a handle to the old one.
        self._drop_caches(project_id)

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
        web_search_enabled: Optional[bool] = None,
    ) -> dict:
        """Update an existing project."""
        self._ensure_model()

        workspace = self._model.workspace
        project = next((p for p in workspace.projects if p.id == project_id), None)
        if project is None:
            raise ValueError(f"Project not found: {project_id}")

        # Renaming only changes the label.  Skills and documents live under
        # ``project_dir`` (keyed by project id, not name), so nothing moves on
        # disk, and ``cwd`` is the user's own folder — renaming a project must
        # never relocate it.
        if name and name != project.name:
            project.name = name

        if description is not None:
            project.description = description

        if skills is not None:
            # Write skills to disk rather than config.json
            for skill in skills:
                self._write_skill_md(project, skill)

        if web_search_enabled is not None:
            project.web_search_enabled = web_search_enabled
            logger.info(
                "[web-search] project %s enabled=%s", project.id, web_search_enabled
            )

        self._model.save()

        return self._project_to_dict(project)

    def _project_to_dict(self, p: Project) -> dict:
        """Convert a Project to a dict for API responses."""
        skills = self._read_skills_from_disk(p) if p.name else []
        return {
            "id": p.id,
            "name": p.name,
            "cwd": p.work_dir or "",
            "description": p.description,
            "model_name": self._model_name,
            "current": p.current,
            "system": p.system,
            "skills": [s.model_dump() for s in skills],
            "mcp": [self._mcp_server_to_dict(s) for s in p.mcp_servers],
            "documents": [doc.to_info(p.project_dir).model_dump() for doc in p.documents],
            "web_search_enabled": bool(p.web_search_enabled),
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

        # ``cwd`` is the user's own folder (often a code repo they opened the
        # project on) and holds no framework state — skills and documents live
        # under ``project_dir``, already removed above.  Deleting a project
        # must never delete the user's files.

        # Remove from workspace
        workspace.projects = [p for p in workspace.projects if p.id != project_id]

        # If deleted project was current, set the first remaining project as current
        if was_current and workspace.projects:
            workspace.projects[0].current = True

        # Release the deleted project's store handles; its directory is gone.
        self._drop_caches(project_id)

        self._model.save()

        return {"id": project_id, "deleted": True, "message": "Project deleted successfully"}

    def set_current_project(self, project_id: str) -> None:
        """Record which project the UI has selected.

        This is no longer how a request chooses its project — that is bound
        per request by :func:`~smartloop.workspace.context.use_project`, so
        this call cannot move the project out from under a chat that is
        already running.  What remains is the *selection*: the project a
        client with no ``project_id`` gets, and the one ``slp agent status``
        reports.

        All projects use the workspace's active model, so no model switching
        is needed on selection change.
        """
        self._ensure_model()

        workspace = self._model.workspace
        target = next((p for p in workspace.projects if p.id == project_id), None)
        if target is None:
            raise ValueError(f"Project not found: {project_id}")

        # Already selected — nothing to persist.  Clients re-POST
        # /v1/models/load on unrelated setting changes (e.g. toggling web
        # search), and this is on that path.
        if target.current:
            return

        for p in workspace.projects:
            p.current = (p.id == project_id)

        # Store handles are keyed by project, so a selection change no longer
        # invalidates anything — the incoming project's slot is already right.

        self._model.save()

    def _free_llm(self, llm) -> None:
        """Explicitly close the native llama-cpp context so VRAM is freed immediately."""
        if llm is None:
            return
        # Never close the always-resident orchestrator model.
        if llm is self._llm:
            return
        try:
            if hasattr(llm, 'model') and llm.model is not None and hasattr(llm.model, 'close'):
                llm.model.close()
        except Exception:
            pass

    def mark_model_downloaded(self, model_name: str) -> None:
        """Record that *model_name*'s weights are present in the workspace.

        Weights live in one shared ``models_dir`` and are added or removed for
        the whole workspace, so this is a single workspace-level flag — there
        is no per-project download state to keep in step with it.
        """
        workspace = self._model.workspace if self._model is not None else self._ensure_workspace()
        ws = workspace.get_model(model_name)
        if ws is not None:
            ws.downloaded = True

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
            self._free_llm(self._llm)
            self._llm = None
            self._model_name = None
            self._free_llm(self._decision_llm)
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
        with self._inference_slot():
            if self._vision_llm is None:
                return
            logger.info("[vision] Unloading vision model %s — freeing VRAM",
                        self._vision_model_name or "?")
            self._free_llm(self._vision_llm)
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
        with self._inference_slot():
            if self._large_llm is None:
                return
            logger.info("[large-model] Unloading large agent model — freeing VRAM")
            self._free_llm(self._large_llm)
            self._large_llm = None
            self._large_model_attempted = False
            self._cleanup_gpu_cache()

    def _evict_capability_models(self, keep_model_name: Optional[str] = None) -> bool:
        """Free resident per-capability models to reclaim VRAM between steps.

        A single request runs several steps (document processing, skills,
        vision), each of which may use a *different* capability model. On a
        memory-tight device those must not all stay resident, so when the next
        step's model won't fit we drop the others here — every
        ``_cap_llm_<name>`` cache except ``keep_model_name`` (the one about to be
        used), plus the on-demand large agent model and the vision/OCR model —
        then reclaim the GPU.

        The default orchestrator model is never evicted, whether it is reached
        via ``self._llm`` or via its own ``_cap_llm_<default>`` slot: it is
        always-resident, so dropping it frees nothing durable, and a live
        stream may still be decoding on it.

        Idempotent; returns True if anything was actually freed. Not held under
        ``_inference_lock``: it runs on the load path, before the requesting
        step infers, and only evicts models other than the one being loaded.
        """
        freed = False
        keep_key = f"_cap_llm_{keep_model_name}" if keep_model_name else None
        default_key = f"_cap_llm_{self._model_name}" if self._model_name else None
        for attr in list(self.__dict__):
            if not attr.startswith("_cap_llm_") or attr == keep_key:
                continue
            # Never evict the default orchestrator model.  When the capability
            # ranking picks the default, its cache slot holds a *separate*
            # Llama instance for the same weights, so _free_llm's identity
            # guard against self._llm does not cover it — closing it here tears
            # down the context a live stream is still decoding on (a native
            # use-after-free that takes down the whole process, not just the
            # request).  It is the always-resident model anyway, so freeing it
            # buys nothing: the next step reloads it immediately.
            if attr == default_key:
                continue
            cached = getattr(self, attr, None)
            if cached is not None and cached is not _GPU_FAULTED:
                logger.info(
                    "[mem] Evicting prior-step capability model %s to free VRAM",
                    attr[len("_cap_llm_"):],
                )
                self._free_llm(cached)
                setattr(self, attr, None)
                freed = True
        if self._large_llm is not None and self._large_model_name != keep_model_name:
            logger.info("[mem] Evicting on-demand large model to free VRAM")
            self._free_llm(self._large_llm)
            self._large_llm = None
            self._large_model_name = None
            self._large_model_attempted = False
            freed = True
        # The vision/OCR model is on-demand and per-step exactly like the
        # capability models above, but it lives in its own slot and used to be
        # freed only at end-of-request.  So a document request that ran OCR
        # first left the vision model resident, the next step's 'dp' model
        # failed its fit guard against the VRAM it held, and extraction
        # dropped onto the bootstrap router instead.  OCR is finished by the
        # time a later step needs the room, and get_vision_llm reloads lazily
        # (unload clears _vision_unavailable) if another page needs it.
        if (self._vision_llm is not None
                and self._vision_model_name != keep_model_name):
            logger.info(
                "[mem] Evicting prior-step vision model %s to free VRAM",
                self._vision_model_name or "?",
            )
            self.unload_vision_model()
            freed = True
        if freed:
            self._cleanup_gpu_cache()
        return freed

    def unload_embedding_model(self) -> None:
        """Release the embedding model and free its VRAM."""
        try:
            embeddings.unload()
        except Exception as e:  # cleanup must never fail a response
            logger.warning("[models] embedding unload failed: %s", e)

    def unload_auxiliary_models(self, include_embedding: bool = True) -> None:
        """Release every on-demand auxiliary model, keeping only the resident
        orchestrator.

        The vision/OCR model and the large agent/skills model are both loaded
        on demand for a single response and are not part of the always-resident
        footprint. Called at the end of each orchestration/response cycle so
        their VRAM is returned between chats rather than left resident. Both
        underlying unloads are idempotent, so this is safe to call whether or
        not either model was actually loaded for the request.

        The embedding model goes with them by default, on the same reasoning:
        holding its weights and llama.cpp context until the next question is
        the same idle cost as the auxiliary LLMs.

        ``include_embedding=False`` for the one caller where that reasoning
        does not hold. Retrieval is finished long before the response is, but
        citation verification is not — it scores the sources against the
        finished answer, so the embedding model is wanted again *after*
        generation ends. Unloading it here would mean reloading it seconds
        later, and this GGUF takes about three seconds to come back, paid in
        full where the user is already waiting for citations to appear. That
        caller unloads it itself once verification is done.

        Each unload is guarded on its own so one failure still returns the
        other models' VRAM instead of skipping the rest of the cleanup.
        """
        unloads = [
            ("vision", self.unload_vision_model),
            ("large", self.unload_large_model),
            ("capability", self._evict_capability_models),
        ]
        if include_embedding:
            unloads.append(("embedding", self.unload_embedding_model))
        for name, unload in unloads:
            try:
                unload()
            except Exception as e:  # cleanup must never fail a response
                logger.warning("[models] %s unload failed: %s", name, e)

    def unload_model_by_name(self, model_name: str) -> bool:
        """Free VRAM held by a specific model, e.g. when it's deselected.

        Deselecting a model in the UI flips its ``enabled`` flag so future
        loads skip it, but an instance already resident in a cache slot keeps
        its llama.cpp context + GPU buffers until dropped here.  Clears every
        slot backed by ``model_name`` — the on-demand large/vision caches and
        any per-capability (``_cap_llm_<name>``) cache — then reclaims the GPU.
        Idempotent; returns True if anything was actually unloaded.
        """
        with self._inference_slot():
            unloaded = False

            if self._large_llm is not None and self._large_model_name == model_name:
                logger.info("[unload] Releasing deselected large model %s", model_name)
                self._free_llm(self._large_llm)
                self._large_llm = None
                self._large_model_name = None
                self._large_model_attempted = False
                unloaded = True

            if self._vision_llm is not None and self._vision_model_name == model_name:
                logger.info("[unload] Releasing deselected vision model %s", model_name)
                self._free_llm(self._vision_llm)
                self._vision_llm = None
                self._vision_model_name = None
                self._vision_unavailable = False
                unloaded = True

            cache_key = f"_cap_llm_{model_name}"
            cached = getattr(self, cache_key, None)
            if cached is not None and cached is not _GPU_FAULTED:
                logger.info("[unload] Releasing deselected capability model %s", model_name)
                self._free_llm(cached)
                setattr(self, cache_key, None)
                unloaded = True

            if unloaded:
                self._cleanup_gpu_cache()

        return unloaded

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
        resource: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        access_token: str | None = None,
        refresh_token: str | None = None,
        expires_at: float | None = None,
        token_type: str = "Bearer",
        project_id: str | None = None,
        server_info: dict | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        catalog_slug: str | None = None,
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
            tags=list(tags or []),
            catalog_slug=catalog_slug,
            server_type=MCPServerType.REMOTE,
            remote_config=RemoteMCPServerConfig(
                server_url=server_url,
                auth_type=mcp_auth,
                token=token,
                token_endpoint=token_endpoint,
                authorization_endpoint=authorization_endpoint,
                resource=resource,
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
        # Registration persists the connection and returns.  Filling in any
        # missing tools and generating metadata with the LLM both happen on the
        # MCP task (see :meth:`_provision_mcp_server`), so no client waits on
        # the network or the model to add a connection.
        server.status = MCPServerStatus.PENDING
        current_project.mcp_servers.append(server)
        self._model.save()
        self._queue_mcp_provisioning(current_project, server)

        return self._mcp_server_to_dict(server)

    @staticmethod
    def _mcp2_incompat_error(msg: str) -> bool:
        """True when the discovery failure is the MCP SDK 2.x breakage.

        ``mcp`` 2.0.0 (2026-07-28) removed ``mcp.server.fastmcp`` (renamed to
        ``mcp.server.mcpserver.MCPServer``) and dropped several top-level
        exports (e.g. ``McpError``). Packages that still import the old path or
        symbols die at startup, because their unbounded ``mcp>=1.x`` requirement
        resolves to 2.x. We do *not* match ``mcp_types`` — that is a 2.x-only
        dependency and pinning below 2 would be wrong for it.
        """
        low = msg.lower()
        return (
            re.search(r"no module named ['\"]?mcp(\.|['\" ])", low) is not None
            or "mcp.server.fastmcp" in low
            or "mcp.shared" in low
            or re.search(r"cannot import name .* from ['\"]?mcp(\.|['\" ])", low) is not None
            or re.search(r"module ['\"]?mcp['\"]? has no attribute", low) is not None
            # 2.x also *kept* the low-level Server class and removed the
            # decorators every 1.x server hangs its handlers off, so a package
            # can import cleanly and then die registering its tools.  That
            # failure names no module at all, which is why the import patterns
            # above miss it.
            or re.search(
                r"'(?:server|fastmcp)' object has no attribute "
                r"'(?:list_tools|call_tool|list_resources|read_resource|"
                r"list_prompts|get_prompt)'",
                low,
            ) is not None
        )

    @staticmethod
    def _pin_mcp1x_args(command: str, args: list[str]) -> list[str] | None:
        """Return ``args`` with ``--with mcp<2`` injected for uv/uvx launchers.

        This pins the ephemeral environment to the MCP SDK 1.x line so servers
        that still import ``mcp.server.fastmcp`` start again. Returns ``None``
        when not applicable (non-uv launcher, or a ``--with`` flag is already
        present so we never re-pin or loop forever).
        """
        if command not in ("uv", "uvx"):
            return None
        if any(a == "--with" for a in args) or any(a.startswith("--with=") for a in args):
            return None
        pin = ["--with", "mcp<2"]
        if command == "uvx":
            return [*pin, *args]
        if args[:2] == ["tool", "run"]:
            return [*args[:2], *pin, *args[2:]]
        if args[:1] == ["run"]:
            return [*args[:1], *pin, *args[1:]]
        return None

    def add_local_mcp_server(
        self,
        command: str,
        args: list[str] | None = None,
        name: str | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        catalog_slug: str | None = None,
        project_id: str | None = None,
    ) -> dict:
        """Register a local (stdio) MCP server to a project (by ID or current).

        Follows the standard MCP stdio config: the caller supplies the
        ``command`` (``uvx``, ``npx``, ``python``, a path to a program, …) and
        ``args``, and they are persisted and launched exactly as given.  The
        command must be installed on the host; it is resolved to its full path
        at launch (``LocalMCPClient._resolve_command``) but never rewritten.

        Nothing here knows what a launcher wants.  Registration used to fold
        ``uvx <pkg>`` into ``uvx --from <pkg> <pkg>`` and swap ``uvx`` for
        ``uv tool run``, which meant the command a user configured was not the
        command that ran — invisible in the UI, wrong for every launcher that
        is not uv, and a second place to keep in step with a tool's own CLI.
        A configuration that does not work is now a configuration that can be
        read and corrected.

        Built-in Smartloop servers use FastMCP modules launched via
        ``<python> -m smartloop...`` and are registered separately.
        """
        if not command:
            raise ValueError("'command' is required to register a local MCP server")

        args = list(args or [])

        # If the caller passed the full command as a single string
        # (e.g. "uvx yahoo-finance-mcp", "npx -y @foo/bar-mcp"), split it into
        # command + args.  Only applies when args is still empty.
        if not args and " " in command:
            command, args = split_command_line(command)

        # A launcher with nothing to launch cannot start, and that is the
        # mistake worth catching here.  A command that is a path to a program
        # (`/opt/bin/my-mcp`, `./server`) is complete on its own — requiring an
        # argument from it only rejected a server that would have run.
        if not args and not (os.path.isabs(command) or os.sep in command):
            raise ValueError(
                f"'args' is required to register a local MCP server (command='{command}'). "
                "Provide the package or module name, e.g. args=['yahoo-finance-mcp'] "
                "for uvx, args=['-y', '@scope/pkg'] for npx, or args=['-m', 'module']."
            )

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
            tags=list(tags or []),
            catalog_slug=catalog_slug,
            config=MCPServerConfig(command=command, args=args, env=resolved_env, cwd=resolved_cwd),
        )

        # Registration persists the connection and returns immediately.  Tool
        # discovery launches a process, enrichment calls the LLM and indexing
        # embeds vectors — none of that belongs in front of a UI click, so it
        # runs on the MCP task (see :meth:`_provision_mcp_server`).  A server
        # stays ``pending`` until that pass finishes and records ``ready`` or
        # ``failed`` with a reason clients can show.
        server.status = MCPServerStatus.PENDING
        current_project.mcp_servers.append(server)
        self._model.save()
        self._queue_mcp_provisioning(current_project, server)

        return self._mcp_server_to_dict(server)



    def _discover_local_mcp_tools(self, server: "MCPServer") -> None:
        """Launch a local MCP server and record the tools it advertises.

        Runs on the MCP task, never on the registration request: the handshake
        means starting a process (often a ``uvx``/``npx`` install first), which
        is seconds at best and a hung subprocess at worst.

        Mutates *server* in place — ``tools``, ``server_info``, and, when a
        retry fixed the launch, ``config``/``name``/``pkg``.  Raises
        :class:`MCPDiscoveryError` with a human-readable reason when the tools
        cannot be enumerated at all.
        """
        config = server.config
        if config is None:
            raise MCPDiscoveryError("This connection has no launch command.")
        command = config.command
        args = list(config.args or [])
        resolved_env = dict(config.env or {})
        resolved_cwd = config.cwd
        name = server.name
        # Launch the process and do the MCP handshake.  A server whose tools
        # we cannot enumerate is left registered but marked ``failed`` with the
        # reason, so the client can show it and offer a retry — the connection
        # the user typed is not silently discarded.
        discovery_error: str | None = None
        try:
            with LocalMCPClient(command=command, args=args, env=resolved_env, cwd=resolved_cwd) as client:
                result = client.connect_and_discover_tools()
        except MCPLaunchError as exc:
            # Candidate fixes to retry, tried in order until one succeeds:
            # 1. Both `uvx` and `uv tool run` print a hint like
            #    "Use `uvx --from <pkg> <exec>` instead" (or the `uv tool run`
            #    form) when the package name and executable name differ. Parse
            #    the suggested args and auto-retry under our launcher.
            # 2. MCP SDK 2.x removed `mcp.server.fastmcp`; pin the ephemeral env
            #    to `mcp<2` so legacy servers start again (see _pin_mcp1x_args).
            fixes: list[list[str]] = []
            hint = re.search(r"Use `uv(?:x| tool run) (.*?)` instead", str(exc))
            if hint:
                suggested = hint.group(1).split()
                # Preserve the `tool run` prefix when launching via the real
                # `uv` binary; for bare `uvx` there is no prefix.
                prefix = args[:2] if command == "uv" and args[:2] == ["tool", "run"] else []
                fixes.append([*prefix, *suggested])
            if self._mcp2_incompat_error(str(exc)):
                pinned = self._pin_mcp1x_args(command, args)
                if pinned is not None:
                    logger.info(
                        "Retrying %s with --with mcp<2 (MCP SDK 2.x incompatibility; "
                        "original: %s)", command, args,
                    )
                    fixes.append(pinned)
            result = None
            discovery_error = str(exc)
            for fixed_args in fixes:
                try:
                    with LocalMCPClient(command=command, args=fixed_args, env=resolved_env, cwd=resolved_cwd) as client:
                        result = client.connect_and_discover_tools()
                except Exception as retry_exc:
                    result = None
                    discovery_error = str(retry_exc)
                    # The retry can fail with a *different* failure than the
                    # original (e.g. the uvx hint fix swapped in the right
                    # executable, but the server then dies importing the MCP
                    # SDK 2.x symbols). Re-check for the mcp<2 incompatibility
                    # here too; _pin_mcp1x_args returns None when `--with` is
                    # already present, so this never re-pins or loops forever.
                    if self._mcp2_incompat_error(str(retry_exc)):
                        pinned = self._pin_mcp1x_args(command, fixed_args)
                        if pinned is not None:
                            logger.info(
                                "Retrying %s with --with mcp<2 (MCP SDK 2.x incompatibility; "
                                "original: %s)", command, fixed_args,
                            )
                            fixes.append(pinned)
                    continue
                if not result or not result.success:
                    discovery_error = result.error if result and result.error else "tool discovery failed"
                    continue
                args = fixed_args
                server.config = MCPServerConfig(command=command, args=args, env=resolved_env, cwd=resolved_cwd)
                if hint and suggested:
                    if "--from" in suggested:
                        from_idx = suggested.index("--from")
                        fixed_pkg = suggested[from_idx + 1] if from_idx + 1 < len(suggested) else None
                    else:
                        fixed_pkg = next((a for a in suggested if not a.startswith("-")), None)
                    if fixed_pkg:
                        server.name = fixed_pkg.split("/")[-1]
                        server.pkg = server.name
                        name = server.name
                discovery_error = None
                break
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
            # Keep the full raw error (often a multi-line traceback) in the logs,
            # but hand the client a short, human-readable reason.
            logger.warning(
                "Could not discover tools from local MCP server '%s': %s",
                name, discovery_error,
            )
            raise MCPDiscoveryError(self._friendly_discovery_error(discovery_error))

    # ------------------------------------------------- background provisioning

    def _mcp_task(self) -> Optional["MCPTask"]:
        """Return the registered MCP task, or None before setup_tasks() ran."""
        task = self._task_queue.get_task("mcp")
        return task if isinstance(task, MCPTask) else None

    def _queue_mcp_provisioning(self, project: "Project", server: "MCPServer") -> None:
        """Hand a freshly registered server to the MCP task.

        Falls back to provisioning inline on a thread when the task isn't
        registered (CLI use, tests): the point is that the caller doesn't wait,
        not that a particular task exists.
        """
        task = self._mcp_task()
        if task is not None:
            task.enqueue(project.id, server.id)
            return
        threading.Thread(
            target=self._provision_mcp_server,
            args=(project.id, server.id),
            name=f"mcp-provision-{server.id}",
            daemon=True,
        ).start()

    def pending_mcp_servers(self) -> list[tuple[str, str]]:
        """``(project_id, server_id)`` for every server still awaiting provisioning.

        Used by the MCP task at startup to adopt servers whose provisioning was
        interrupted (the app quit between registration and discovery).
        """
        self._initialize_workspace()
        pending: list[tuple[str, str]] = []
        for project in self._model.workspace.projects:
            for server in project.mcp_servers:
                if server.system:
                    continue
                if server.status == MCPServerStatus.PENDING:
                    pending.append((project.id, server.id))
        return pending

    def _provision_mcp_server(self, project_id: str, server_id: str) -> dict:
        """Discover and enrich one registered server.

        The whole slow half of registration, run off the request path:

        1. discover the tools when registration couldn't supply them (a local
           process launch, or a remote handshake) — that list is the catalog a
           step is later resolved against, so nothing else has to index it;
        2. enrich blank metadata via the LLM (skipped for curated catalog
           entries, which already carry it).

        Records ``ready`` or ``failed`` + ``last_error`` on the server, so the
        state a client polls always reflects a finished pass.
        """
        try:
            project = self._get_project(project_id)
        except Exception:
            logger.warning("[mcp-provision] Project %s is gone — skipping", project_id)
            return {"status": "skipped", "reason": "project not found"}

        server = next((s for s in project.mcp_servers if s.id == server_id), None)
        if server is None:
            # Registered by another process, or deleted before we got to it.
            self._model.workspace.reload()
            project = self._get_project(project_id)
            server = next((s for s in project.mcp_servers if s.id == server_id), None)
        if server is None:
            logger.info("[mcp-provision] Server %s is gone — skipping", server_id)
            return {"status": "skipped", "reason": "server not found"}

        logger.info(
            "[mcp-provision] Provisioning '%s' (%s, %d tool(s) known)",
            server.name, server.server_type.value, len(server.tools or []),
        )

        try:
            if not server.tools:
                self._discover_mcp_tools(server)
        except MCPDiscoveryError as exc:
            server.status = MCPServerStatus.FAILED
            server.last_error = str(exc)
            self._model.save()
            logger.warning(
                "[mcp-provision] '%s' failed: %s", server.name, server.last_error,
            )
            return {"status": "failed", "error": server.last_error}
        except Exception as exc:
            server.status = MCPServerStatus.FAILED
            server.last_error = self._friendly_discovery_error(str(exc))
            self._model.save()
            logger.warning(
                "[mcp-provision] '%s' failed: %s", server.name, exc,
            )
            return {"status": "failed", "error": server.last_error}

        # Metadata for the UI and the tool router.  An LLM call, which is why it
        # is here and not on the registration request.
        self._enrich_mcp_server(server)

        server.status = MCPServerStatus.READY
        server.last_error = None
        self._model.save()

        self._probe_mcp_account(server)

        logger.info(
            "[mcp-provision] '%s' ready — %d tool(s) discovered",
            server.name, len(server.tools or []),
        )
        return {"status": "ready", "tools": len(server.tools or [])}

    def _probe_mcp_account(self, server: "MCPServer") -> None:
        """Record which account *server* is linked as, when it can say.

        One read-only call on the provisioning pass — so it costs a request per
        connect or re-index, never per page load.  Best-effort: a server with no
        identity tool leaves the field alone.
        """
        try:
            from smartloop.mcp.account import probe_account
            account = probe_account(server)
        except Exception as e:
            logger.debug("[mcp-account] probe failed for '%s': %s", server.name, e)
            return
        if account and account != server.account:
            server.account = account
            self._model.save()

    def _discover_mcp_tools(self, server: "MCPServer") -> None:
        """Fill in a server's tools, launching or connecting as needed."""
        if server.server_type == MCPServerType.LOCAL:
            self._discover_local_mcp_tools(server)
            return

        server_url = server.remote_config.server_url if server.remote_config else None
        if not server_url:
            raise MCPDiscoveryError("This connection has no server URL.")

        from smartloop.mcp import MCPClient
        token = None
        if server.oauth_credentials and server.oauth_credentials.access_token:
            token = server.oauth_credentials.access_token
        elif server.remote_config and server.remote_config.token:
            token = server.remote_config.token

        with MCPClient() as client:
            result = client.connect_and_discover_tools(server_url, access_token=token)
        if not result or not result.success:
            raise MCPDiscoveryError(
                self._friendly_discovery_error(
                    (result.error if result else None) or "tool discovery failed"
                )
            )
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
            "Discovered %d tool(s) from remote MCP server '%s'",
            len(result.tools), server.name,
        )

    # ---------------------------------------------------------- busy / idle

    @contextmanager
    def _inference_slot(self):
        """Hold an inference slot, and record that the model is in use.

        Wraps ``_inference_lock`` so the count is maintained wherever the lock
        is taken.  Background work reads it through :meth:`idle_seconds`.
        """
        with self._inference_lock:
            with self._inference_state_lock:
                self._inference_active += 1
            try:
                yield
            finally:
                with self._inference_state_lock:
                    self._inference_active -= 1
                    self._last_inference_at = time.time()

    def idle_seconds(self) -> float:
        """Seconds since the model last finished answering; 0.0 while it is.

        Background work uses this to stay out of the way: it competes for CPU
        and network with a reply the user is waiting for.
        """
        with self._inference_state_lock:
            if self._inference_active > 0:
                return 0.0
            if not self._last_inference_at:
                # Nothing has run this session — idle since startup.
                return float("inf")
            return max(0.0, time.time() - self._last_inference_at)

    def retry_mcp_server(
        self, server_id: str, project_id: str | None = None
    ) -> dict:
        """Re-run provisioning for a server (a failed discovery, or a refresh).

        Returns as soon as the work is queued — the pass itself runs on the MCP
        task, and its outcome shows up in the server's ``status``.
        """
        self._initialize_workspace()
        project = (
            self._get_project(project_id) if project_id
            else self._model.workspace.current_project
        )
        if project is None:
            raise ValueError("No current project found")
        server = next((s for s in project.mcp_servers if s.id == server_id), None)
        if server is None:
            raise ValueError(f"MCP server not found: {server_id}")

        server.status = MCPServerStatus.PENDING
        server.last_error = None
        self._model.save()
        self._queue_mcp_provisioning(project, server)
        return self._mcp_server_to_dict(server)

    @staticmethod
    def _mcp_provisioning_state(server: MCPServer) -> dict:
        """Whether a connection is ready to answer with, and why not.

        One background pass stands between registering a server and being able
        to call it: provisioning discovers its tools (a process launch or a
        remote handshake) and generates the metadata the UI shows.  It runs off
        the request path, so a client that just added a connection sees one
        that knows nothing until it finishes.

        ``in_progress`` is the flag to drive a spinner off.  ``tools`` is the
        count discovery recorded — the catalog a request is resolved against,
        and 0 on a ready server means the server advertised none.
        """
        state = {
            MCPServerStatus.FAILED: "failed",
            MCPServerStatus.PENDING: "provisioning",
        }.get(server.status, "ready")
        return {
            "state": state,
            "in_progress": state == "provisioning",
            "tools": len(server.tools or []),
            "error": server.last_error,
        }

    def list_mcp_servers_with_status(self, project_id: str) -> list[dict]:
        """Every non-system server in a project, each with its provisioning state.

        The list view is where a just-registered connection is watched, so the
        state travels with the list rather than costing one request per server.
        """
        project = self._get_project(project_id)
        return [
            {
                **self._mcp_server_to_dict(server),
                "indexing": self._mcp_provisioning_state(server),
            }
            for server in project.mcp_servers
            if not server.system
        ]

    def mcp_server_status(self, server_id: str, project_id: str | None = None) -> dict:
        """One server, with the provisioning state a client polls it for."""
        self._initialize_workspace()
        project = (
            self._get_project(project_id) if project_id
            else self._model.workspace.current_project
        )
        if project is None:
            raise ValueError("No current project found")

        server = next(
            (s for s in project.mcp_servers if s.id == server_id), None
        )
        if server is None:
            raise ValueError(f"MCP server not found: {server_id}")

        return {
            **self._mcp_server_to_dict(server),
            "indexing": self._mcp_provisioning_state(server),
        }

    def count_mcp_tools(self, project_id: str | None = None) -> dict:
        """Diagnostic: how many MCP tools a project has available.

        ``tools`` is the total across registered servers, as discovery recorded
        it — the catalog :class:`~smartloop.agents.mcp.mcp_agent.MCPAgent`
        resolves a step against.  A server showing 0 has not been discovered
        yet (or failed): see its ``status``/``last_error``.
        """
        if project_id:
            project = self._get_project(project_id)
        else:
            self._initialize_workspace()
            project = self._model.workspace.current_project
            if not project:
                raise ValueError("No current project found")

        servers = [s for s in project.mcp_servers if not getattr(s, "system", False)]
        return {
            "project": project.name,
            "servers": len(servers),
            "tools": sum(len(s.tools or []) for s in servers),
            "by_server": {s.name: len(s.tools or []) for s in servers},
        }

    def _enrich_mcp_server(self, server: "MCPServer") -> None:
        """Enrich a hand-added MCP server via the local LLM when fields are
        blank: a short ``description`` and ``tags`` for the UI, and a ``summary``
        that the tool selector injects to route requests to this server
        (server_selector.py builds a "Server summaries" section from it — without
        it the router only has bare tool names to match). Best-effort — never
        raises, so a failure (or no loaded model) leaves the connection as-is.

        This exists for the hand-added case (``uvx``/``npx`` packages that ship
        no description and are not in our catalog). Servers installed from the
        curated catalog arrive with ``description`` and ``tags`` already set, so
        they skip the LLM entirely — generating metadata for them would burn two
        inference calls (and possibly a model load) to overwrite curated copy.
        """
        try:
            if not server.tools:
                return
            # Curated server — nothing to generate. Two signals: a resolved
            # catalog_slug is provenance the backend established itself, and is
            # authoritative even if the entry left a field blank; description +
            # tags both present means there is simply nothing left to write, so
            # a third party curating their own metadata isn't billed for an LLM
            # call either. The catalog has no summary field, so reuse the
            # description as the selector's capability text rather than paying
            # to generate one.
            if server.catalog_slug or (server.description and server.tags):
                if not server.summary:
                    server.summary = server.description
                return
            if self._llm is None:
                self.load_model()
            # Always the small router model, never the large one — even when a
            # prior chat request left it resident. The two prompts below are a
            # few thousand tokens over a tool list producing a one-line
            # description, three tags and a short capability blurb; the mini is
            # the right size for that, and reaching for the large model here
            # means competing for memory (and the inference slot) with whatever
            # the user is actually doing.
            llm = self._llm
            if llm is None:
                return
            from smartloop.agents.mcp.server_summarizer import MCPServerSummarizer
            summarizer = MCPServerSummarizer(llm)
            with self._inference_slot():
                # Description + tags (UI) — one JSON call when either is missing.
                if not server.description or not server.tags:
                    meta = summarizer.generate_metadata(server)
                    if not server.description and meta.get("description"):
                        server.description = meta["description"]
                    if not server.tags and meta.get("tags"):
                        server.tags = list(meta["tags"])[:3]
                # Summary (tool selection) — free-text call, the highest-value
                # field for routing. Kept separate so a JSON parse flake above
                # doesn't cost us the summary, and vice versa. Falls back to the
                # generated description so the selector always has capability
                # text to match against, even if the summary call comes back empty.
                if not server.summary:
                    summary = summarizer.invoke(server)
                    server.summary = summary or server.description or None
        except Exception as e:
            logger.warning(
                "MCP enrichment failed for '%s': %s",
                getattr(server, "name", "?"), e,
            )

    @staticmethod
    def _friendly_discovery_error(raw: str) -> str:
        """Map a raw MCP discovery failure (often a uvx/npx Python traceback)
        to a short, human-readable message. Falls back to the first line of the
        raw error when nothing matches. Kept in sync with the client so the UI
        doesn't have to re-parse tracebacks.
        """
        msg = (raw or "").strip()
        if not msg:
            return "Couldn't start this connection."
        low = msg.lower()
        # Package deps unpinned/out of date vs the current MCP SDK
        # (e.g. "No module named 'mcp.server.fastmcp'" or
        # "cannot import name 'McpError' from 'mcp'").
        if re.search(
            r"no module named ['\"]?mcp[.'\"]|mcp\.server\.fastmcp|mcp\.shared|"
            r"cannot import name .* from ['\"]?mcp",
            low,
        ):
            return (
                "This package isn't compatible with the current MCP SDK — its "
                "dependencies are out of date. Try another package, or ask its "
                "maintainer to update it."
            )
        # Package name doesn't resolve to anything published.
        if re.search(
            r"not found in the package registry|no solution found when resolving|"
            r"no matching distribution|could not find a version",
            low,
        ):
            return (
                "Package not found. Double-check the name — it must be a published "
                "package (PyPI for uvx, npm for npx)."
            )
        # uv / uvx (or npx) missing from PATH.
        if re.search(r"command not found|uvx: not found|no such file or directory[^\n]*uvx|executable .* not found", low):
            return "Couldn't run uvx. Make sure uv is installed and on your PATH."
        # Any other import/startup crash from the package process.
        if re.search(r"modulenotfounderror|importerror|traceback \(most recent call last\)|process exited", low):
            return (
                "This package failed to start (a dependency or startup error). "
                "It may be broken or incompatible."
            )
        first = msg.split("\n")[0].strip()
        return f"{first[:180]}…" if len(first) > 180 else first

    # --------------------------------------------------------- web search (pro)

    def get_web_search_config(self, project_id: str) -> dict:
        """Return whether web search is enabled for the given project."""
        project = self._get_project(project_id)
        return {"project_id": project.id, "enabled": bool(project.web_search_enabled)}

    def set_web_search_enabled(self, project_id: str, enabled: bool) -> dict:
        """Enable/disable web search for the given project."""
        project = self._get_project(project_id)
        project.web_search_enabled = enabled
        self._model.save()
        logger.info("[web-search] project %s enabled=%s", project.id, enabled)
        return {"project_id": project.id, "enabled": bool(project.web_search_enabled)}

    def _write_mcp_doc(
        self,
        project: "Project",
        server: "MCPServer",
        content: str,
    ) -> bool:
        """Write MCP tool description to disk for workspace indexer discovery."""
        try:
            # Framework state, written where the workspace indexer scans.
            base_dir = Path(project.project_dir)
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
            "tags": s.tags,
            "catalog_slug": s.catalog_slug,
            "server_type": s.server_type.value,
            "enabled": s.enabled,
            "system": s.system,
            "tools": [t.model_dump() for t in s.tools],
            # Background provisioning state — clients poll this instead of
            # blocking on discovery at registration time.
            "status": s.status.value,
            "last_error": s.last_error,
            "account": s.account,
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
        self._remove_mcp_definition_file(server)

        return {
            "id": server_id,
            "deleted": True,
            "message": "MCP server deleted successfully",
        }

    def _remove_mcp_definition_file(self, server: "MCPServer") -> None:
        """Delete the dropped-in file that defined *server*, if it was one.

        Without this a deleted connection leaves its file behind, and re-adding
        a server by the same name silently picks up the old one's command, env
        and cwd.  Deleting a connection should leave nothing of it.

        Only files under ``mcp.d`` are touched, and only when no remaining
        server matches the same definition — two connections can share one.
        """
        try:
            from smartloop.mcp.definitions import definition_for
            from smartloop.mcp.definition_loader import (
                USER_DEFINITIONS_DIR, reload_definitions,
            )

            definition = definition_for(server)
            if definition is None:
                return

            for project in self._model.workspace.projects:
                for other in project.mcp_servers:
                    if other.id == server.id:
                        continue
                    match = definition_for(other)
                    if match is not None and match.key == definition.key:
                        return

            path = os.path.join(
                _settings.home_dir, USER_DEFINITIONS_DIR, f"{definition.key}.toml",
            )
            if os.path.isfile(path):
                os.remove(path)
                reload_definitions()
                logger.info("[mcp-definition] Removed %s", path)
        except Exception as e:
            logger.debug("[mcp-definition] Could not remove the definition file: %s", e)

    # --------------------------------------------------------------- MCP agents

    def _get_llm_executor(self):
        """Return the shared LLMExecutor (an I/O thread pool), creating it lazily."""
        if self._llm_executor is None:
            from smartloop.agents.hybrid_executor import LLMExecutor
            self._llm_executor = LLMExecutor(max_io_workers=6)
        return self._llm_executor

    def _reset_llm_executor(self) -> None:
        """Shut down the shared executor so it's rebuilt for the new model."""
        if self._llm_executor is not None:
            try:
                self._llm_executor.shutdown(wait=False)
            except Exception:
                pass
            self._llm_executor = None

    # Signals that an image-attachment request wants to *use* indexed documents
    # (cross-validate / answer with them), not just read the image. When none of
    # these appear, an attached image is treated as the sole source and the
    # retrieved documents are discarded so they can't mis-attribute its contents.
    _IMAGE_CROSS_REF_RE = re.compile(
        r"\b(compare|comparison|match(?:es|ing)?|validate|verif(?:y|ies|ication)|"
        r"cross[-\s]?(?:check|validat\w*|referenc\w*)|reconcile|consistent|discrepanc\w*|"
        r"recommend|suggest|against (?:our|the|my)|based on (?:our|the|my)|"
        r"using (?:our|the|my)|according to (?:our|the|my)|"
        r"(?:in|from|with) (?:our|the|my|these|those) (?:record|document|doc|file|data|report)s?)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _emit_progress(
        progress_callback: Optional[Callable[[str, str, str], None]],
        step: str, status: str, message: str,
    ) -> None:
        """Fire a progress/SSE event, swallowing any callback error.

        A no-op when no callback is supplied; a failing callback must never
        break orchestration.
        """
        if progress_callback is None:
            return
        try:
            progress_callback(step, status, message)
        except Exception:
            logger.debug("[progress] callback failed", exc_info=True)

    def orchestrate(
        self,
        user_message: str,
        skills: Optional[list] = None,
        cwd: Optional[str] = None,
        conversation_history: Optional[list] = None,
        has_attachments: bool = False,
        images: Optional[list[tuple[bytes, str]]] = None,
        progress_callback: Optional[Callable[[str, str, str], None]] = None,
        cancelled: Optional[threading.Event] = None,
        requested_skill: Optional[str] = None,
        current_date: Optional[str] = None,
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
        # A missing handle is recoverable when the weights are the host's —
        # building one is a constructor call, not a load.  Worth doing here
        # rather than trusting every caller to have gone through load_model,
        # because the alternative is this method quietly degrading to raw
        # generation and taking skills, tools and planning with it.
        if self._llm is None and self._remote_device() is not None:
            self._llm = self._remote_llm()
            self._model_name = self._model_name or api_settings.default_model
            logger.info("[orchestrator] bound to the model host")

        if self._model is None or self._llm is None:
            logger.warning(
                "[orchestrator] BYPASSED — model/llm not loaded "
                "(model=%s, llm=%s). Falling back to raw stream_completion.",
                self._model is not None, self._llm is not None,
            )
            return None, False, None, None, None, False

        # Image attachment → a vision-capable model first *perceives* the image
        # (a concrete description scoped to the user's request) and that
        # perception is folded into the message as an "[Image context: …]" block.
        # The rest of the pipeline — document retrieval, capability routing, MCP,
        # synthesis — then plans and answers grounded in what the image shows, so
        # e.g. "recommend products for this nail color" can search the product
        # docs.  When there is no project to orchestrate over, the perception is
        # streamed back directly (mutation=True).  Falls through untouched when no
        # vision model is available or perception comes back empty.
        image_perception: Optional[str] = None
        image_sole_source: bool = False
        if images:
            # Surface the vision model on the SSE stream naming the model that
            # was actually picked, fired right before the (heavy) load. The
            # vision path loads its model here, outside _llm_for_tier, so it
            # must emit its own event.
            def _on_vision_model(name: str) -> None:
                self._emit_progress(progress_callback, "model", "processing",
                                    f"Processing image with {name}")
            try:
                image_perception = self.vision_perceive(
                    user_message, images, on_model_selected=_on_vision_model,
                )
            except Exception as e:
                logger.warning("[orchestrator] vision perception failed (%s) — "
                               "continuing without image context", e)
            finally:
                # The vision model has done its one job for this request — the
                # perception is folded into the text below and the rest of the
                # pipeline is text-only. Release the auxiliary model now so its
                # VRAM is freed for the downstream synthesis (and not left
                # resident between chats) rather than waiting for idle unload.
                self.unload_vision_model()
            if image_perception:
                logger.info("[orchestrator] image perception: %r",
                            image_perception[:200])
                # Decide by user intent whether the image is the SOLE source or
                # documents should also inform the answer. A bare extract/read of
                # the attached image discards the retrieved docs (they only cause
                # mis-attribution); a request to compare, validate, or answer
                # using indexed content keeps them for cross-validation.
                image_sole_source = not self._IMAGE_CROSS_REF_RE.search(user_message or "")
                logger.info("[orchestrator] image sole-source=%s", image_sole_source)
                user_message = f"{user_message}\n\n[Image context: {image_perception}]"
            else:
                logger.warning("[orchestrator] vision perception empty — "
                               "continuing without image context")

        def _with_image_context(ctx: Optional[str]) -> Optional[str]:
            """Prepend the image description to the returned context.

            The perception is folded into ``user_message`` for planning, but the
            final response prompt is rebuilt from the original messages + this
            returned context — so the image description must ride along here or
            the answering model won't see it (and will deny an image was sent).

            The block is phrased as an instruction, not passive context: the
            answering model is text-only, so without being told the image was
            already analyzed it fixates on the user's "attached image" wording
            and replies that it cannot view images. This makes the description
            authoritative and tells it to answer directly.
            """
            if not image_perception:
                return ctx
            authority = (
                "the SOLE source for answering this request"
                if image_sole_source
                else "an authoritative source for this request"
            )
            block = (
                "The user attached an image. A vision model has already analyzed "
                "it — treat the description below as the actual contents of that "
                f"image and {authority}. Answer the user's request directly; do "
                "NOT say you cannot see or view the image.\n\n"
                f"Image contents: {image_perception}"
            )
            if not ctx:
                return block
            if not image_sole_source:
                # Cross-validation intent — the image is authoritative but the
                # retrieved documents may corroborate or help answer it.
                return f"{block}\n\n{ctx}"
            # Pure extract/read of the image — the retrieved documents are
            # background only. Without this the answering model merges similar
            # indexed docs into the answer and mis-attributes the image's own
            # fields to them (e.g. a driver-license image cited to an unrelated
            # sample PDF).
            return (
                f"{block}\n\n---\n\n"
                "The material below was retrieved from indexed documents as "
                "background only. Do NOT answer from it, cite it, or attribute "
                "the image's contents to any document — the image above is the "
                "sole source for this request.\n\n"
                f"{ctx}"
            )

        # Reload workspace from disk to pick up any changes made by
        # background indexers or project setup since the last request.
        self._model.workspace.reload()

        current_project = self._model.workspace.current_project
        if not current_project:
            if image_perception:
                logger.info(
                    "[orchestrator] no current project — streaming image "
                    "perception directly.",
                )
                return image_perception, True, None, None, None, True
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

        # The orchestrator resolves its working directory as
        # ``cwd or project_dir or os.getcwd()``.  Point it at the framework's
        # own per-project dir so tools and scans stay self-contained.
        orchestrator_cwd = current_project.project_dir

        # With exactly one model selected, it runs the routing and decision
        # nodes too — not just the steps that resolve a capability.  Those
        # nodes otherwise stay on the bootstrap model, which is never a user
        # selection, so an explicit single choice would still leave the
        # planner, the skill matcher and the relevance judge on another model.
        # Asked without a capability, so a vision-only selection declines these
        # text-only nodes and they stay on the default model.
        pinned_llm = self._pinned_llm()
        if pinned_llm is not None:
            logger.info(
                "[models] Single selected model pinned for the whole request: %s",
                self._model_label(pinned_llm),
            )

        try:
            orchestrator = SLPOrchestrator(
                orchestrator_llm=pinned_llm or self._llm,
                agent_llm=pinned_llm or self._decision_llm or self._llm,
                large_llm_provider=self._get_large_llm,
                capability_llm_provider=self._get_llm_for_capability,
                capability_available=self.has_capability_model,
                capability_skip_reason=self.last_capability_skip,
                model_path=getattr(self, '_model_gguf_path', None),
                project_id=self.get_project_id(),
                executor=self._get_llm_executor(),
            )
            workspace = self._model.workspace
            project = workspace.current_project
            caps: list[str] = []
            for m in workspace.models:
                if (workspace.effective_model_enabled(m.model_name, project)
                        and workspace.effective_model_downloaded(m.model_name)):
                    caps.extend(m.effective_capabilities)
            available_capabilities = list(dict.fromkeys(caps))
            logger.info(
                "[caps] models=%s available_capabilities=%s",
                [(m.model_name,
                  workspace.effective_model_enabled(m.model_name, project),
                  workspace.effective_model_downloaded(m.model_name),
                  m.effective_capabilities)
                 for m in workspace.models],
                available_capabilities,
            )

            # Propagate the cancellation event onto the orchestrator's
            # primary LLMs so short planning steps (skill matching, routing)
            # can be interrupted between tokens instead of blocking the whole
            # request until they finish generating.
            cancel_llms = [
                llm for llm in {id(self._llm): self._llm,
                                id(self._decision_llm): self._decision_llm,
                                id(pinned_llm): pinned_llm}.values()
                if llm is not None
            ]
            if cancelled is not None:
                for llm in cancel_llms:
                    llm._cancelled = cancelled

            try:
                result = orchestrator.execute(
                    user_message, eligible_servers,
                    skills=skill_contents, cwd=cwd,
                    has_attachments=has_attachments,
                    project_dir=orchestrator_cwd,
                    model_service=self,
                    conversation_history=conversation_history,
                    available_capabilities=available_capabilities or None,
                    progress_callback=progress_callback,
                    requested_skill=requested_skill,
                    current_date=current_date,
                )
            finally:
                if cancelled is not None:
                    for llm in cancel_llms:
                        llm._cancelled = None

            # Save workspace in case tokens were refreshed during the call
            try:
                self._model.save()
            except Exception:
                pass

            if not result.handled:
                # No tool handled it, but if we perceived an image its
                # description must still reach the final response as context.
                if image_perception:
                    return _with_image_context(None), True, None, None, None, False
                return None, False, None, None, None, False

            if result.error:
                error_msg = f"Tool '{result.tool_name}' failed: {result.error}"
                logger.warning(f"Tool call failed ({result.tool_name}): {result.error}")
                return None, True, error_msg, None, None, False

            if result.content:
                logger.info(
                    f"Tool '{result.tool_name}' returned "
                    f"{len(result.content)} chars of context"
                )
                # Drop document citations when the image is the sole source —
                # the answer comes from the image, so citing retrieved docs would
                # mis-attribute the image's contents to unrelated files.
                _citations = None if image_sole_source else result.citations
                return _with_image_context(result.content), True, None, _citations, result.skill_instruction, result.mutation

            return _with_image_context(None), True, None, None, result.skill_instruction, False

        except Exception as e:
            error_msg = f"Tool call failed: {e}"
            logger.warning(f"Tool call attempt failed: {e}")
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

        markdown_content = None
        is_markdown = False
        is_image = (content_type or "").startswith("image/")

        if is_image:
            # Images are NOT pre-processed to text here: OCR-at-upload throws
            # away the visual content, so "describe the person" can't be
            # answered later.  Persist the original pixels instead and let a
            # vision-capable model read them at query time with the user's
            # actual question (see ``vision_answer`` / ``orchestrate``).
            (asset_dir / f"original{ext or '.img'}").write_bytes(file_bytes)
        else:
            # Write the original file to a temp location for parsing
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

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
            "image": is_image,
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

    def get_asset_images(self, asset_ids: list[str]) -> list[tuple[bytes, str]]:
        """Return ``[(bytes, mime_type)]`` for the image assets among ``asset_ids``.

        Only assets stored as images (``metadata['image']`` true, with an
        ``original.*`` file on disk) are returned; document assets are ignored
        so the caller can route image-attachment turns to the vision model.
        """
        images: list[tuple[bytes, str]] = []
        for aid in asset_ids or []:
            asset_dir = Path(self._assets_dir(), aid)
            meta_path = asset_dir / "metadata.json"
            if not meta_path.exists():
                logger.warning(f"Attachment asset not found: {aid}")
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            content_type = meta.get("content_type") or "application/octet-stream"
            if not (meta.get("image") or content_type.startswith("image/")):
                continue
            original = next(iter(asset_dir.glob("original.*")), None)
            if original is not None:
                images.append((original.read_bytes(), content_type))
        return images

    # Anti-hallucination guidance for image Q&A, sent as the system message.
    # (An earlier version folded this into the *user* message to avoid sending a
    # system message at all, on the theory that a system message made Qwen2-VL
    # emit EOS immediately.  That theory was wrong: Qwen25VLChatHandler always
    # injects a default "You are a helpful assistant." system message when the
    # caller supplies none — the same shape MarkItDown's working OCR path uses —
    # so a system message is present either way.  Supplying our own simply
    # replaces that default with useful steering.)
    _VISION_QA_GUIDANCE = (
        "Answer using only what is actually visible in the image. If the answer "
        "is not visible, say so plainly instead of guessing. Do not invent "
        "people, text, or details that are not there."
    )

    # Perception prompt for the orchestrator's image path.  Unlike direct Q&A,
    # this asks the vision model only to *describe* what it sees — scoped to the
    # user's request so the description surfaces the details that matter (colors,
    # objects, visible text) — because the perception is fed back into the text
    # pipeline (retrieval, capability routing, MCP, synthesis), which then does
    # the actual answering with documents/tools the vision model can't see.
    _VISION_PERCEPTION_PROMPT = (
        "Describe what is visible in this image, focusing on the details that "
        "are relevant to the following request. Be concrete and specific about "
        "colors, objects, and any visible text. Describe only what is visible; "
        "do not answer the request or make recommendations.\n\nRequest: {request}"
    )

    def vision_perceive(
        self,
        user_message: str,
        images: list[tuple[bytes, str]],
        max_tokens: int = 512,
        on_model_selected: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Return a concrete description of ``images`` scoped to ``user_message``.

        This is the orchestrator's image entry point: the returned text is
        folded into the user message as an ``[Image context: …]`` block so the
        downstream text pipeline can plan and answer against documents, MCP and
        capability models grounded in what the image shows.  It reuses
        ``vision_answer``'s inference path (and its GPU-clip-OOM recovery) with a
        description-only prompt — see ``_VISION_PERCEPTION_PROMPT``.
        """
        prompt = self._VISION_PERCEPTION_PROMPT.format(request=user_message)
        return self.vision_answer(
            prompt, images, max_tokens=max_tokens,
            on_model_selected=on_model_selected,
        )

    def vision_answer(
        self,
        prompt: str,
        images: list[tuple[bytes, str]],
        max_tokens: int = 1024,
        on_model_selected: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Return a vision-model answer to ``prompt`` about the given ``images``.

        The image pixels and the user's own prompt go straight to the multimodal
        model via ``create_chat_completion`` (non-streaming — the proven path
        the OCR pipeline uses), so the model reads exactly what the user asked
        about with no lossy upload-time transcription in between.  ``images`` is
        a list of ``(bytes, mime_type)``.

        The anti-hallucination steering goes in the system message and the
        user's own prompt (plus the image) in a single user message — see
        ``_VISION_QA_GUIDANCE``.

        Raises ``RuntimeError`` when no vision-capable model is available so the
        caller can surface actionable guidance instead of a silent empty reply.
        """
        llm = self.get_vision_llm(on_model_selected=on_model_selected)
        if llm is None:
            raise RuntimeError("No vision-capable model available")

        content: list[dict] = [{"type": "text", "text": prompt}]
        for data, mime in images:
            b64 = base64.b64encode(data).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        messages = [
            {"role": "system", "content": self._VISION_QA_GUIDANCE},
            {"role": "user", "content": content},
        ]

        def _ask(vision_llm):
            with self._inference_slot():
                return vision_llm.model.create_chat_completion(
                    messages=messages, max_tokens=max_tokens,
                    temperature=0.0, stream=False,
                )

        try:
            result = _ask(llm)
        except Exception as e:
            # The GPU clip encode of a large image can exhaust device memory
            # inside native code (no flash-attn on non-CUDA backends), which
            # surfaces as an access-violation OSError.  Reload once with the
            # CLIP encoder on CPU — the LLM itself stays on GPU — and retry
            # instead of degrading to the blind text pipeline.
            if self._vision_clip_cpu:
                raise
            logger.warning(
                "[vision-answer] encode failed with GPU clip (%s) — "
                "reloading with CLIP on CPU and retrying", e,
            )
            self._vision_clip_cpu = True
            self.unload_vision_model()
            llm = self.get_vision_llm()
            if llm is None:
                raise
            result = _ask(llm)
        choice0 = (result.get("choices") or [{}])[0]
        answer = (choice0.get("message", {}).get("content") or "").strip()
        # This path calls create_chat_completion directly, so it does its own
        # <think> filtering — a reasoning model loaded here with thinking left
        # on (template default) must not leak its monologue into the answer.
        answer = strip_thinking(answer)
        # Qwen2-VL sometimes prefixes its reply with a stray "label:" (often an
        # empty label, leaving a leading ":"); drop it so the answer reads clean.
        answer = answer.lstrip(":").strip()
        logger.info(
            "[vision-answer] %d char(s), finish=%s, preview=%r",
            len(answer), choice0.get("finish_reason"), answer[:120],
        )
        return answer


