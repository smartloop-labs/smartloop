
from __future__ import annotations

import json
import logging
import os
import threading
from enum import Enum
from typing import Any, ClassVar, Dict, Optional, Set, List
from pathlib import Path

from uuid import uuid4
from pydantic import BaseModel, computed_field, ConfigDict, Field
from smartloop.config import ModelConfig, AppSettings
from smartloop.config import SUPPORTED_MODELS
from smartloop.types import ModelCapability
from smartloop.workspace.types import Document, DocumentInfo, DocumentStatus, Skill

_settings = AppSettings()

def _is_mmproj_gguf(filename: str) -> bool:
    """True when a GGUF is a vision mmproj / CLIP projector (not main weights)."""
    name = os.path.basename(filename).lower()
    return "mmproj" in name or "clip" in name

class MCPToolDefinition(BaseModel):
    """Tool definition discovered from an MCP server."""
    name: str
    description: Optional[str] = None
    input_schema: Optional[dict] = None


class MCPServerConfig(BaseModel):
    """Configuration for launching a local MCP server process."""
    command: str  # e.g. "npx", "uvx"
    args: List[str] = []
    env: Dict[str, str] = {}
    cwd: Optional[str] = None  # working directory; supports ~/


class MCPServerType(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"


class MCPAuthType(str, Enum):
    OAUTH = "oauth"
    TOKEN_URL = "token_url"
    TOKEN = "token"
    NONE = "none"


class RemoteMCPServerConfig(BaseModel):
    """Configuration for a remote MCP server."""
    server_url: str
    auth_type: MCPAuthType = MCPAuthType.NONE
    token_endpoint: Optional[str] = None
    authorization_endpoint: Optional[str] = None
    token: Optional[str] = None


class MCPOAuthCredentials(BaseModel):
    """OAuth credentials for a remote MCP server."""
    client_id: str
    client_secret: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[float] = None
    token_type: str = "Bearer"


class MCPServer(BaseModel):
    """An MCP server registered to a project, persisted in config.json."""
    id: str
    name: str
    pkg: Optional[str] = None
    description: Optional[str] = None
    summary: Optional[str] = None
    config: Optional[MCPServerConfig] = None
    server_type: MCPServerType = MCPServerType.LOCAL
    remote_config: Optional[RemoteMCPServerConfig] = None
    oauth_credentials: Optional[MCPOAuthCredentials] = None
    enabled: bool = True
    system: bool = False
    tools: List[MCPToolDefinition] = []
    server_info: Optional[Dict[str, Any]] = None


class WorkspaceModel(BaseModel):
    """A model registered in the workspace."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str        # key name e.g. "gemma3_it_text"
    model_id: str          # HF id e.g. "google/gemma-3-1b-it"
    downloaded: bool = False
    is_active: bool = False
    enabled: bool = True
    capabilities: List[str] = Field(default_factory=list)
    config: Optional[ModelConfig] = None
    models_dir: str = Field(default="", exclude=True)

    @property
    def model_dir(self) -> str:
        return os.path.join(self.models_dir, self.model_id) if self.models_dir else ""

    @property
    def model_path(self) -> str:
        if not self.models_dir:
            return ""
        return os.path.join(self.models_dir, self.model_id, f"{Path(self.model_id).name}.gguf")

    @property
    def bin_dir(self) -> str:
        """Directory where the built GGUF is written for this model."""
        return os.path.join(self.model_dir, "bin") if self.models_dir else ""

    def gguf_path(self) -> str:
        """GGUF file path under the model's bin directory."""
        if not self.models_dir:
            return ""
        return os.path.join(self.bin_dir, f"{Path(self.model_id).name}.gguf")

    @property
    def param_size_b(self) -> float:
        """Parameter count in billions, taken from the model class.

        Sourced from the ``param_size_in_b()`` declared on the model class
        (the authoritative source). Returns 0 for models without a
        ``SUPPORTED_MODELS`` entry or an explicit size.
        """
        model_cls = SUPPORTED_MODELS.get(self.model_name)
        if model_cls is not None:
            return float(model_cls.param_size_in_b())
        return 0.0

    @property
    def model_version(self) -> int:
        """Model family version declared by the model class (0 if unknown).

        Selection prefers a higher version over a lower one even when the
        higher-version model has fewer parameters.
        """
        model_cls = SUPPORTED_MODELS.get(self.model_name)
        if model_cls is not None:
            return int(model_cls.model_version())
        return 0

    @property
    def _rank(self) -> tuple[float, int]:
        """Selection rank: larger size wins first, then newer model version.

        Size is the primary key so the most capable (largest) model is always
        preferred; the model version only breaks a tie between two models of the
        same parameter count.  (A newer but *smaller* model therefore does NOT
        outrank a larger older one — selection favors capacity over generation.)
        """
        return (self.param_size_b, self.model_version)

    @property
    def effective_capabilities(self) -> List[str]:
        """Return capabilities declared by the model class — read-only.

        The stored ``capabilities`` field is ignored so class defaults
        stay authoritative and updates to any model's capabilities take
        effect immediately on next request (no per-install migration).
        Models without a ``SUPPORTED_MODELS`` entry fall back to
        metadata-inferred capabilities.
        """
        model_cls = SUPPORTED_MODELS.get(self.model_name)
        if model_cls and hasattr(model_cls, "default_capabilities"):
            return list(model_cls.default_capabilities())
        return self._infer_capabilities()

    def _infer_capabilities(self) -> List[str]:
        combined = f"{self.model_name} {self.model_id}".lower()
        caps: List[str] = []
        size = self.param_size_b

        caps.append(ModelCapability.GENERAL)

        if any(kw in combined for kw in ("code", "coder", "starcoder", "deepseek-coder")):
            caps.append(ModelCapability.CODING)

        # Vision is inferred only from an explicit vision marker in the name
        # (e.g. Qwen2.5-VL).  gemma-3-4b is NOT inferred as vision: llama-cpp has
        # no Gemma3 vision handler, so tagging it vision only produces a model
        # that's considered for OCR and then skipped every time.
        if any(kw in combined for kw in ("vision", "vlm", "vl")):
            caps.append(ModelCapability.VISION)

        if size >= 3:
            # GENERAL already added above; larger models additionally handle
            # skills and document processing.
            caps.append(ModelCapability.SKILLS)
            caps.append(ModelCapability.DP)

        return caps

class Project (BaseModel):
    id: str
    project_dir: str
    current: bool
    name: Optional[str] = None
    description: Optional[str] = None
    system: bool = False
    cwd: Optional[str] = None
    documents: List[Document] = []
    skills: List[Skill] = []
    mcp_servers: List[MCPServer] = []
    adapters: Set[str] = set()
    sessions: List[str] = []

    @property
    def visible_dir(self) -> str:
        """Return the user-visible project directory.

        Uses ``cwd`` (always set on new projects).  Falls back to
        ``~/smartloop/<name>`` for legacy projects created before
        ``cwd`` was persisted unconditionally.
        """
        if self.cwd:
            return os.path.expanduser(self.cwd)
        return os.path.expanduser(
            os.path.join("~/smartloop", self.name or self.id)
        )

    def get_skills(self) -> List["Skill"]:
        """Read skills from disk (SKILL.md files), the single source of truth."""
        skills_dir = os.path.join(self.visible_dir, "skills")
        if not os.path.isdir(skills_dir):
            return []
        result: List[Skill] = []
        for entry in sorted(os.listdir(skills_dir)):
            entry_path = os.path.join(skills_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            skill_path = os.path.join(entry_path, "SKILL.md")
            if not os.path.isfile(skill_path):
                continue
            try:
                raw = open(skill_path, encoding="utf-8").read()
            except Exception:
                continue
            name = entry
            description = None
            if raw.startswith("---"):
                parts = raw.split("---", 2)
                if len(parts) >= 3:
                    for line in parts[1].strip().splitlines():
                        if line.startswith("name:"):
                            name = line[len("name:"):].strip()
                        elif line.startswith("description:"):
                            description = line[len("description:"):].strip()
            result.append(Skill(
                id=name, name=name, description=description,
                content=raw,
            ))
        return result

    @property
    def folder_name(self) -> str:
        """Basename of the visible directory on disk."""
        return os.path.basename(self.visible_dir)

    def model_build_dir(self) -> str:
        """Per-project directory where the built GGUF is written."""
        return os.path.join(self.project_dir, "model")

    def model_gguf_path(self, model_id: str) -> str:
        """Per-project GGUF file path (produced by build)."""
        return os.path.join(self.project_dir, "model", f"{Path(model_id).name}.gguf")

class Workspace(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='ignore')

    _instance: ClassVar[Optional['Workspace']] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    config_file: ClassVar[str] = os.path.join(_settings.home_dir, "config.json")

    id: str
    projects: List[Project] = []
    models: List[WorkspaceModel] = []
    adapters: List = []
    models_dir: str = ""
    workspace_dir: str = ""
    merge_model_dir: Optional[str] = None

    def __init__(self, id: str, *,
                 projects: List[Project] = None,
                 workspace_dir: str = "",
                 merge_model_dir: Optional[str] = None,
                 models_dir: str = "",
                 models: Optional[List[WorkspaceModel]] = None,
                 adapters: Optional[List] = None,
                 config: Optional[dict] = None,
        ):
        super().__init__(id=id, projects=projects or [], merge_model_dir=merge_model_dir, workspace_dir=workspace_dir, models_dir=models_dir, models=models or [], adapters=adapters or [])

    @property
    def active_model(self) -> Optional[WorkspaceModel]:
        """Return the currently active WorkspaceModel, or None."""
        return self.get_model()

    def set_active_model(self, model_name: str) -> None:
        """Mark the given model as active, deactivating all others."""
        for m in self.models:
            m.is_active = (m.model_name == model_name)

    def set_model_enabled(self, model_name: str, enabled: bool) -> None:
        """Enable or disable a model. Raises ValueError if not found."""
        for m in self.models:
            if m.model_name == model_name:
                m.enabled = enabled
                return
        raise ValueError(f"Model not found: {model_name}")

    def get_model(self, model_name: Optional[str] = None) -> Optional[WorkspaceModel]:
        """Look up a registered model by its short name.
        If *model_name* is ``None``, returns the default model.
        """
        result = None
        if model_name is None:
            model_name = AppSettings().default_model
        model_name = SUPPORTED_MODELS.get(model_name)
        for m in self.models:
            if SUPPORTED_MODELS.get(m.model_name) == model_name:
                result = m
                break
        if result is not None:
            result.models_dir = self.models_dir
        
        return result

    def add_model(self, model_name: str, model_id: str, config: Optional[ModelConfig] = None) -> WorkspaceModel:
        """Register a model in the workspace without changing the active model."""
        existing = self.get_model(model_name)
        if existing is not None:
            if config is not None:
                if existing.config:
                    merged = existing.config.model_dump()
                    for key, value in config.model_dump().items():
                        if value is not None:
                            merged[key] = value
                    existing.config = ModelConfig.model_validate(merged)
                else:
                    existing.config = config
            return existing
        has_active = any(m.is_active for m in self.models)
        entry = WorkspaceModel(model_name=model_name, model_id=model_id, config=config, is_active=not has_active, models_dir=self.models_dir)
        self.models.append(entry)
        return entry

    def largest_downloaded_model(self) -> Optional[WorkspaceModel]:
        """Return the best enabled, downloaded model, or None.

        Ranks by parameter count first (the largest model wins), then by model
        version as a tie-breaker between same-size models.
        """
        candidates = [m for m in self.models if m.enabled and m.downloaded and m.param_size_b > 0]
        if not candidates:
            return None
        candidates.sort(key=lambda m: m._rank, reverse=True)
        best = candidates[0]
        best.models_dir = self.models_dir
        return best

    def ranked_models_for(self, capability: str) -> List["WorkspaceModel"]:
        """Enabled, downloaded models declaring *capability*, best-first.

        Ranked by :attr:`WorkspaceModel._rank` (larger size first, then newer
        version), so the most capable model leads. Callers walk this list down to
        a smaller same-capability model when the largest doesn't fit in memory,
        rather than dropping straight to the tiny default. Empty when no
        downloaded model declares the capability.
        """
        cap = capability.lower()
        candidates = [
            m for m in self.models
            if m.enabled and m.downloaded and m.param_size_b > 0
            and cap in [c.lower() for c in m.effective_capabilities]
        ]
        candidates.sort(key=lambda m: m._rank, reverse=True)
        for m in candidates:
            m.models_dir = self.models_dir
        return candidates

    def best_model_for(self, capability: str) -> Optional[WorkspaceModel]:
        """Return the single best model for a capability — the largest (by
        size, then newer version) that declares it — or the largest downloaded
        model when none declares it.

        Memory is NOT considered here; callers that must fit the model in VRAM
        use :meth:`ranked_models_for` and walk down to a smaller candidate.
        """
        ranked = self.ranked_models_for(capability)
        if ranked:
            return ranked[0]
        return self.largest_downloaded_model()

    def model_dir_for(self, model_id: str) -> str:
        """Return the workspace-level directory for a given model_id."""
        return os.path.join(self.models_dir, model_id)

    def model_path_for(self, model_id: str) -> str:
        """Return the workspace-level GGUF file path for a given model_id."""
        return os.path.join(self.models_dir, model_id, f"{Path(model_id).name}.gguf")

    def resolve_gguf_path(self, model_id: str) -> str:
        """Cascade: quantize-matched bin GGUF → exact bin GGUF → exact workspace GGUF → first GGUF in model dir."""
        # Honor the requested model_id for the bin_dir cascade (fall back to the
        # active model) so resolving a non-active model — e.g. the small
        # orchestrator model while a larger model is active — returns that
        # model's GGUF, not the active one's.
        ws_model = next((m for m in self.models if m.model_id == model_id), None)
        if ws_model is not None:
            ws_model.models_dir = self.models_dir
        else:
            ws_model = self.get_model()
        if ws_model and ws_model.bin_dir and os.path.isdir(ws_model.bin_dir):
            quantize = (ws_model.config.quantize if ws_model.config else "").upper()
            gguf_files = [f for f in os.listdir(ws_model.bin_dir)
                          if f.endswith(".gguf") and not _is_mmproj_gguf(f)]
            if quantize:
                for f in gguf_files:
                    if f.upper().endswith(f"-{quantize}.GGUF"):
                        return os.path.join(ws_model.bin_dir, f)
            if gguf_files:
                return os.path.join(ws_model.bin_dir, gguf_files[0])
        exact = self.model_path_for(model_id)
        if os.path.exists(exact):
            return exact
        # Fallback: find any main .gguf in the model directory.  The mmproj /
        # CLIP projector GGUF is excluded — it is not a loadable main model.
        model_dir = self.model_dir_for(model_id)
        if os.path.isdir(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith(".gguf") and not _is_mmproj_gguf(f):
                    return os.path.join(model_dir, f)
        return exact

    def save(self) -> None:
        """Persist workspace state to config.json."""
        ws_data = self.model_dump(mode="json")
        ws_data.pop("model_dir", None)
        ws_data.pop("model_path", None)
        ws_data.pop("current_project", None)
        ws_data.pop("model_id", None)
        active = self.active_model
        data = {"id": active.model_id if active else "", "workspace": ws_data}
        with open(self.config_file, "w") as f:
            json.dump(data, f, indent=4)
    
    @classmethod
    def _create_new_workspace(cls, config: ModelConfig) -> 'Workspace':
        workspace_id = str(uuid4())
        workspace_dir = os.path.join(_settings.home_dir, workspace_id)
        models_dir = os.path.join(workspace_dir, "models")

        os.makedirs(models_dir, exist_ok=True)

        workspace = Workspace(
            id=workspace_id,
            workspace_dir=workspace_dir,
            models_dir=models_dir,
            projects=[],
            merge_model_dir=None,
        )
        return workspace

    @classmethod
    def init(cls, config: ModelConfig) -> 'Workspace':
        """Load the workspace singleton from config.json on disk.

        Returns the cached instance if already loaded, otherwise reads
        from ``<base_dir>/config.json``.  Returns ``None`` when no
        config file exists yet (workspace not initialized).
        """
        with cls._lock:
            if cls._instance is not None:
                return cls._instance

            if not os.path.exists(cls.config_file):
                cls._instance = cls._create_new_workspace(config)
                return cls._instance

            try:
                with open(cls.config_file, "r") as f:
                    content = f.read().strip()
                    if not content:
                        cls._instance = cls._create_new_workspace(config)
                        return cls._instance
                    data = json.loads(content)

                ws_data = data.get("workspace", {})
                # Strip computed/runtime fields that shouldn't be in __init__
                ws_data.pop("model_dir", None)
                ws_data.pop("model_path", None)
                ws_data.pop("current_project", None)
                ws_data.pop("model_id", None)

                workspace = Workspace.model_validate(ws_data)

                # Re-resolve paths against current base_dir
                workspace.workspace_dir = os.path.join(_settings.home_dir, workspace.id)
                workspace.models_dir = os.path.join(workspace.workspace_dir, "models")

                # Merge config into the requested WorkspaceModel (without changing active)
                ws_model = workspace.get_model(config.model_name)
                if ws_model:
                    if ws_model.config:
                        merged = ws_model.config.model_dump()
                        for key, value in config.model_dump().items():
                            if value is not None:
                                merged[key] = value
                        ws_model.config = ModelConfig.model_validate(merged)
                    else:
                        ws_model.config = config

                cls._instance = workspace
                return cls._instance
            except (json.JSONDecodeError, ValueError) as e:
                logging.getLogger(__name__).warning(f"Workspace.init: failed to load config.json, creating new workspace: {e}")
                cls._instance = cls._create_new_workspace(config)
                return cls._instance
    @classmethod
    def load(cls) -> Optional['Workspace']:
        """Load the workspace directly from config.json on disk.

        Unlike :py:meth:`init`, this method does *not* require a
        :class:`~smartloop.config.ModelConfig` and does not merge any model
        config into the returned workspace.  It also does *not* update the
        singleton cache.

        Returns ``None`` when the config file is missing, empty, or invalid.
        """
        if not os.path.exists(cls.config_file):
            return None
        try:
            with open(cls.config_file, "r") as f:
                content = f.read().strip()
                if not content:
                    return None
                data = json.loads(content)

            ws_data = data.get("workspace", {})
            ws_data.pop("model_dir", None)
            ws_data.pop("model_path", None)
            ws_data.pop("current_project", None)
            ws_data.pop("model_id", None)

            workspace = Workspace.model_validate(ws_data)

            # Re-resolve paths against home_dir
            workspace.workspace_dir = os.path.join(_settings.home_dir, workspace.id)
            workspace.models_dir = os.path.join(workspace.workspace_dir, "models")

            return workspace
        except (json.JSONDecodeError, ValueError):
            return None

    def reload(self) -> None:
        """Re-read workspace state from disk into this instance.

        Refreshes project list and document statuses so queries see
        documents that were indexed by background workers since the
        last load.
        """
        fresh = Workspace.load()
        if fresh is None:
            return
        # Update projects in-place so references stay valid
        self.projects = fresh.projects

    @classmethod
    def reset(cls) -> None:
        """Clear the cached singleton (useful after model switch or tests)."""
        with cls._lock:
            cls._instance = None

    @classmethod
    def get_instance(cls) -> Optional['Workspace']:
        """Return the current singleton instance, or None."""
        return cls._instance

    @computed_field(repr=False)
    @property
    def current_project(self) -> Optional['Project']:
        """Get the current active project"""
        for project in self.projects:
            if project.current:
                return project
        return None
