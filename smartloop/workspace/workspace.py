
from __future__ import annotations

import json
import logging
import os
import threading
from enum import Enum
from typing import Any, ClassVar, Dict, Optional, Set, List
from pathlib import Path

from uuid import uuid4
from pydantic import (
    BaseModel, computed_field, ConfigDict, Field, PrivateAttr, ValidationError,
    field_validator,
)
from smartloop.config import ModelConfig, AppSettings
from smartloop.config import SUPPORTED_MODELS
from smartloop.constants import canonical_model_name
from smartloop.types import ModelCapability
from smartloop.workspace.context import (
    active_project_id, owned_project_id, pinned_project_id, projects_delegated,
)
from smartloop.workspace.types import Document, DocumentInfo, DocumentStatus, Skill

_settings = AppSettings()
logger = logging.getLogger(__name__)

#: Filename of a project's own state, inside its ``project_dir``.
PROJECT_STATE_FILE = "project.json"

#: Schema version stamped into ``project.json``, so a future layout change can
#: recognise what it is reading rather than guessing from the shape.
PROJECT_STATE_VERSION = 1

#: Fields ``config.json`` stays authoritative for even when ``project.json``
#: also carries them.  Selection is a property of the workspace — which project
#: the UI is looking at — not of any one project, and only the supervisor
#: writes it.
_WORKSPACE_OWNED_PROJECT_FIELDS = ("current",)

#: The thin per-project record kept in ``config.json``.  Enough to list and
#: route projects without opening any of them; everything else lives in the
#: project's own file.
_PROJECT_INDEX_FIELDS = (
    "id", "name", "description", "project_dir", "current", "system",
)


def _home_dir() -> str:
    """The framework home, resolved now rather than at import.

    ``AppSettings`` is instantiated once at import, which pins ``SLP_HOME`` to
    whatever it was then.  A per-project agent child is spawned with its own
    environment, so the home has to be re-read at the moment it is needed.
    """
    home = os.environ.get("SLP_HOME")
    return os.path.expanduser(home) if home else _settings.home_dir


class _ConfigFilePath:
    """``Workspace.config_file``, resolved on access instead of at import.

    It used to be a plain ``ClassVar`` computed when this module was first
    imported, which fixed the path for the life of the process and made it
    impossible to point a child at a different home.  A non-data descriptor
    keeps the same ``Workspace.config_file`` spelling at every call site —
    and stays overridable by assignment, which the tests rely on.
    """

    def __get__(self, obj, owner=None) -> str:
        override = os.environ.get("SLP_CONFIG_FILE")
        if override:
            return os.path.expanduser(override)
        return os.path.join(_home_dir(), "config.json")


def _legacy_layout() -> bool:
    """True when the workspace should be kept as a single ``config.json``.

    The rollback path for the split: set ``SLP_LEGACY_CONFIG=1`` and the
    framework reads and writes the monolith again, ignoring any ``project.json``
    already written beside it.  Nothing has to be undone by hand — the
    pre-migration backup is still there, and the split files are simply not
    consulted.
    """
    return os.environ.get("SLP_LEGACY_CONFIG", "").strip().lower() in ("1", "true", "yes")


def _atomic_write_json(path: str, data: dict) -> None:
    """Write *data* to *path* so a reader sees the old file or the new one.

    Same contract the workspace config has always had: write a sibling
    temporary file, flush it to the platter, then rename over the target.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_file = f"{path}.tmp"
    with open(tmp_file, "w") as fh:
        json.dump(data, fh, indent=4)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_file, path)


def project_state_path(project_dir: str) -> str:
    """Where a project's own state lives."""
    return os.path.join(project_dir, PROJECT_STATE_FILE)


def read_project_state(project_dir: str) -> Optional[dict]:
    """Return a project's persisted record, or ``None`` when there isn't one.

    Never raises: a project whose file is missing, empty or corrupt falls back
    to the thin record in ``config.json`` rather than failing the whole load
    and taking every other project down with it.
    """
    path = project_state_path(project_dir)
    try:
        with open(path, "r") as fh:
            content = fh.read().strip()
    except OSError:
        return None
    if not content:
        return None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("[project-state] %s is not valid JSON; ignoring it", path)
        return None
    project = data.get("project") if isinstance(data, dict) else None
    return project if isinstance(project, dict) else None


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
    # RFC 8707 resource indicator negotiated at connect time. Persisted because
    # refresh requests must carry the same indicator as the original grant.
    resource: Optional[str] = None


class MCPOAuthCredentials(BaseModel):
    """OAuth credentials for a remote MCP server."""
    client_id: str
    client_secret: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[float] = None
    token_type: str = "Bearer"


class MCPServerStatus(str, Enum):
    """Provisioning state of a registered MCP server.

    Registration only persists the connection; discovery and metadata
    enrichment run in the background (see :class:`MCPTask`), so a server is
    ``PENDING`` until that pass finishes.  ``READY`` is the default so
    servers written before this field existed — already discovered — are not
    re-queued on upgrade.
    """
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class MCPServer(BaseModel):
    """An MCP server registered to a project, persisted in config.json."""
    id: str
    name: str
    pkg: Optional[str] = None
    description: Optional[str] = None
    summary: Optional[str] = None
    tags: List[str] = []
    # Slug of the curated catalog entry this server was installed from, when it
    # was. Provenance: it is what marks metadata as ours rather than generated.
    catalog_slug: Optional[str] = None
    config: Optional[MCPServerConfig] = None
    server_type: MCPServerType = MCPServerType.LOCAL
    remote_config: Optional[RemoteMCPServerConfig] = None
    oauth_credentials: Optional[MCPOAuthCredentials] = None
    enabled: bool = True
    system: bool = False
    tools: List[MCPToolDefinition] = []
    server_info: Optional[Dict[str, Any]] = None
    # Background provisioning state (tool discovery, then metadata enrichment).
    # Registration returns as soon as the connection is persisted, so clients
    # poll this instead of blocking on the handshake.
    status: MCPServerStatus = MCPServerStatus.READY
    # Human-readable reason the last provisioning pass failed, if it did.
    last_error: Optional[str] = None
    # Which account this connection is authenticated as ("mehfuz@smartloop.ai",
    # "Smartloop HQ"), probed at provisioning time.  It is what distinguishes
    # two connections to the same product; None when the server ships no
    # identity tool.  See :mod:`smartloop.mcp.account`.
    account: Optional[str] = None


class ProjectModel(BaseModel):
    """A model's settings scoped to a single project.

    Seeded from the workspace defaults (the app-default template) when the
    project is created (see ``Workspace.sync_project_models``), then diverges
    independently per project.  ``settings`` is a catch-all dict for
    project-level inference defaults (temperature, top_p, top_k, …) that will
    be added later.

    Download state deliberately lives only on :class:`WorkspaceModel`: weights
    are downloaded and removed for the whole workspace, so a per-project copy
    would be a cache of a global fact with nothing to invalidate it.  See
    :meth:`Workspace.effective_model_downloaded`.
    """
    model_name: str
    enabled: bool = True
    is_active: bool = False
    # Reason before answering.  None = follow the workspace default (and below
    # that the model class's).  Only meaningful for a model declaring
    # ``ModelCapability.THINKING``; see ``effective_model_thinking``.
    thinking: Optional[bool] = None
    settings: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("model_name")
    @classmethod
    def _canonicalize_model_name(cls, value: str) -> str:
        """Migrate an entry written under a model's old short name.

        Mirrors :meth:`WorkspaceModel._canonicalize_model_name` so a project
        whose ``model_settings`` were written before a rename still resolves
        against the workspace registry instead of silently falling back to the
        workspace default.
        """
        return canonical_model_name(value)


class WorkspaceModel(BaseModel):
    """A model registered in the workspace."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str        # short name e.g. "sl-mini"
    model_id: str          # HF id e.g. "smartloop-ai/slp-orchestra-mini"
    downloaded: bool = False
    is_active: bool = False
    enabled: bool = True
    # Reason before answering; None = the model class's own default.  Set from
    # the model picker for models whose capabilities include "thinking".
    thinking: Optional[bool] = None
    capabilities: List[str] = Field(default_factory=list)
    config: Optional[ModelConfig] = None
    models_dir: str = Field(default="", exclude=True)

    @field_validator("model_name")
    @classmethod
    def _canonicalize_model_name(cls, value: str) -> str:
        """Migrate a record written under a model's old short name.

        Runs on every load of an existing ``config.json``, so a workspace
        created before ``slp-orchestra-mini`` was renamed to ``sl-mini`` reports
        the current name and picks up the new one on the next save.  ``model_id``
        is deliberately left alone — it names the on-disk model directory, so
        rewriting it would orphan already-downloaded weights.
        """
        return canonical_model_name(value)

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
            # reasoning (skill matching, MCP tool resolution, planning) and
            # document processing.
            caps.append(ModelCapability.REASONING)
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
    # Pro/premium feature: when enabled, the orchestrator runs the
    # WebSearchAgent (hosted /v1/websearch) for latest-info queries.
    # On by default, so new projects — and configs written before this flag
    # existed — start enabled; only an explicit ``false`` on disk turns it off.
    # Searching still no-ops without a configured provider/token, so the
    # default costs nothing until a key is connected.
    # Toggled per project via PATCH /v1/projects/{id} (or .../websearch).
    web_search_enabled: bool = True
    model_settings: List[ProjectModel] = []

    @property
    def skills_dir(self) -> str:
        """Where this project's SKILL.md files live.

        Anchored to ``project_dir`` — i.e. under the framework's own home —
        never to a client-supplied path.  ``project_dir`` derives from
        ``AppSettings.home_dir``, so it follows ``SLP_HOME`` and, under the
        macOS App Sandbox, moves into the app container automatically.  A
        client-supplied absolute path would not: it would keep pointing at the
        real home the sandbox denies, leaving the app reading one workspace
        and writing another.
        """
        return os.path.join(self.project_dir, "skills")

    @property
    def documents_dir(self) -> str:
        """Where this project's ingested documents live (see :meth:`skills_dir`)."""
        return os.path.join(self.project_dir, "documents")

    @property
    def work_dir(self) -> Optional[str]:
        """The client-supplied folder this project was opened on, or None.

        Reported back on the API for reference only — **nothing is located
        through it**.  Skills, documents, MCP docs and the workspace index all
        hang off ``project_dir`` so the framework stays self-contained: copy
        ``<home_dir>`` to another machine, user, or sandbox container and it
        runs as-is.  Locating anything through a client path would break that,
        since the path may be unreadable under the App Sandbox or simply
        absent on the copy.
        """
        return os.path.expanduser(self.cwd) if self.cwd else None

    def get_skills(self) -> List["Skill"]:
        """Read skills from disk (SKILL.md files), the single source of truth."""
        skills_dir = self.skills_dir
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

    # ------------------------------------------------------------- persistence

    @property
    def state_file(self) -> str:
        """This project's own ``project.json``, inside its ``project_dir``."""
        return project_state_path(self.project_dir)

    def to_index(self) -> dict:
        """The thin record ``config.json`` keeps for this project.

        Enough to list projects and route a request to one without opening
        any of them.  Documents, skills, MCP servers and model settings are
        deliberately absent: they are the bulk of the old monolith, and the
        reason two processes could not both hold a workspace.
        """
        dumped = self.model_dump(mode="json")
        return {k: dumped[k] for k in _PROJECT_INDEX_FIELDS if k in dumped}

    def save_state(self) -> None:
        """Write this project's full record to its own directory.

        The whole record, not just the fields ``config.json`` leaves out, so
        the directory is self-describing: a ``project_dir`` can be copied to
        another workspace and read back without the file it came from.
        """
        if not self.project_dir:
            return
        _atomic_write_json(self.state_file, {
            "version": PROJECT_STATE_VERSION,
            "project": self.model_dump(mode="json"),
        })

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
    # Serializes save(): background tasks (document indexing, MCP
    # provisioning) each persist from their own thread.
    _save_lock: ClassVar[threading.Lock] = threading.Lock()
    #: Resolved on access, not at import — see :class:`_ConfigFilePath`.
    config_file: ClassVar[Any] = _ConfigFilePath()

    #: Last ``project.json`` mtime this instance has read, per project.  Per
    #: *instance*, not shared: two instances can hold different vintages of the
    #: same project, and a map shared between them would let one skip a re-read
    #: the other had already accounted for.
    _state_mtimes: Dict[str, int] = PrivateAttr(default_factory=dict)

    #: Last ``config.json`` mtime this instance has read its models from.
    #: Per instance for the same reason as :attr:`_state_mtimes`.
    _models_mtime: Optional[int] = PrivateAttr(default=None)

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
                 **_legacy,
        ):
        # ``**_legacy`` swallows keys that were persisted by older builds but
        # have since been removed from the model (e.g. ``web_search``), so
        # loading an existing config.json never fails validation. Anything
        # still supported has to be named explicitly here — a field left to
        # _legacy is silently dropped and reverts to its default on every load.
        super().__init__(id=id, projects=projects or [], merge_model_dir=merge_model_dir, workspace_dir=workspace_dir, models_dir=models_dir, models=models or [], adapters=adapters or [])

    @property
    def active_model(self) -> Optional[WorkspaceModel]:
        """Return the currently active WorkspaceModel, or None."""
        return self.get_model()

    def set_active_model(self, model_name: str) -> None:
        """Mark the given model as active, deactivating all others."""
        target = canonical_model_name(model_name)
        for m in self.models:
            m.is_active = (canonical_model_name(m.model_name) == target)

    def set_model_enabled(self, model_name: str, enabled: bool) -> None:
        """Enable or disable a model. Raises ValueError if not found."""
        target = canonical_model_name(model_name)
        for m in self.models:
            if canonical_model_name(m.model_name) == target:
                m.enabled = enabled
                return
        raise ValueError(f"Model not found: {model_name}")

    # ------------------------------------------------------------- per-project

    def is_bootstrap_model(self, model_name: str) -> bool:
        """True for the always-on bootstrap/orchestrator model.

        That model is always enabled, can't be turned off, and is filtered out
        of every models list response, so it deliberately gets no per-project
        entry: a per-project ``enabled=False`` would shadow its always-on
        workspace state everywhere selection resolves through
        ``effective_model_enabled`` (``ranked_models_for``, the vision and
        agent-model sweeps, the capability sweep).  With no entry it keeps
        falling through to the workspace record, which is the real answer.
        """
        return canonical_model_name(model_name) == canonical_model_name(
            AppSettings().default_model
        )

    def sync_project_models(self, project: Project) -> None:
        """Seed a project's ``model_settings`` from the workspace defaults.

        Adds a ``ProjectModel`` for every registered workspace model the
        project doesn't already have an entry for, copying the workspace-level
        (app-default) enabled/downloaded/active state.  Existing project
        entries are left untouched so per-project overrides survive.  Called
        at project creation and lazily before reads/writes so models added to
        the catalog after a project was created show up too.

        The bootstrap model is excluded (see :meth:`is_bootstrap_model`), and
        any entry a previous build already wrote for it is pruned here so
        existing workspaces heal on next access.
        """
        project.model_settings = [
            m for m in project.model_settings if not self.is_bootstrap_model(m.model_name)
        ]
        existing = {canonical_model_name(m.model_name) for m in project.model_settings}
        for wm in self.models:
            if (canonical_model_name(wm.model_name) in existing
                    or self.is_bootstrap_model(wm.model_name)):
                continue
            project.model_settings.append(ProjectModel(
                model_name=wm.model_name,
                enabled=wm.enabled,
                is_active=wm.is_active,
            ))

    def project_model(self, project: Optional[Project], model_name: str) -> Optional[ProjectModel]:
        """Return a project's ``ProjectModel`` entry for *model_name*, or None.

        No fallback to the workspace default here — callers that want the
        merged view use ``effective_*`` below.
        """
        if project is None:
            return None
        target = canonical_model_name(model_name)
        for m in project.model_settings:
            if canonical_model_name(m.model_name) == target:
                return m
        return None

    def effective_model_enabled(self, model_name: str, project: Optional[Project] = None) -> bool:
        """Whether *model_name* is enabled for *project* (or the workspace default).

        Resolution: project override → workspace (app-default) entry → True.
        """
        pm = self.project_model(project, model_name)
        if pm is not None:
            return pm.enabled
        ws = self.get_model(model_name)
        return ws.enabled if ws is not None else True

    def supports_thinking(self, model_name: str) -> bool:
        """Whether *model_name* declares ``ModelCapability.THINKING``.

        Read straight off the resolved capabilities, so it tracks the model
        class rather than anything persisted.
        """
        ws = self.get_model(model_name)
        caps = ws.effective_capabilities if ws is not None else []
        return ModelCapability.THINKING in [
            c.lower() if isinstance(c, str) else c for c in caps
        ]

    def effective_model_thinking(self, model_name: str, project: Optional[Project] = None) -> Optional[bool]:
        """Whether *model_name* should reason before answering.

        Resolution: project override → workspace (app-default) entry →
        ``slp_thinking_enabled`` → the model class's ``thinking_default()``.

        Returns ``None`` for a model that does not declare
        ``ModelCapability.THINKING``, which means "bind nothing": its chat
        template never reads ``enable_thinking``, so a stored preference for it
        would be a setting that silently does nothing.
        """
        if not self.supports_thinking(model_name):
            return None
        pm = self.project_model(project, model_name)
        if pm is not None and pm.thinking is not None:
            return pm.thinking
        ws = self.get_model(model_name)
        if ws is not None and ws.thinking is not None:
            return ws.thinking
        from smartloop.llms.llm_factory import LlmFactory
        return LlmFactory.thinking_for(model_name)

    def set_project_model_thinking(self, project: Project, model_name: str, thinking: Optional[bool]) -> Optional[ProjectModel]:
        """Set *model_name*'s reasoning preference on *project*.

        ``None`` clears the override so the model falls back to the workspace
        default.  Returns ``None`` without writing anything for the bootstrap
        model, which has no per-project entry.
        """
        if self.is_bootstrap_model(model_name):
            return None
        self.sync_project_models(project)
        pm = self.project_model(project, model_name)
        if pm is None:
            pm = ProjectModel(model_name=model_name, thinking=thinking)
            project.model_settings.append(pm)
        else:
            pm.thinking = thinking
        return pm

    def effective_model_downloaded(self, model_name: str) -> bool:
        """Whether *model_name*'s weights are present in the workspace.

        Downloads are workspace-wide — the weights live in one shared
        ``models_dir`` and are added or removed for every project at once — so
        this takes no ``project`` argument.  Kept alongside the
        ``effective_model_*`` accessors that do vary per project so callers
        filtering on "enabled and downloaded" read consistently.
        """
        ws = self.get_model(model_name)
        return ws.downloaded if ws is not None else False

    def effective_model_active(self, model_name: str, project: Optional[Project] = None) -> bool:
        """Whether *model_name* is the active selection for *project*."""
        pm = self.project_model(project, model_name)
        if pm is not None:
            return pm.is_active
        ws = self.get_model(model_name)
        return ws.is_active if ws is not None else False

    def pin_model_for_existing_projects(self, model_name: str) -> None:
        """Freeze what *model_name* currently resolves to, per project.

        A project with no ``ProjectModel`` entry reads through to the workspace
        app-default (see :meth:`effective_model_enabled`), so changing that
        default would retroactively change the project.  Writing today's answer
        down first means the new default only reaches projects created after.

        Only this model is pinned: a blanket ``sync_project_models`` would also
        detach every unrelated model in every project from future defaults.

        Does nothing in a per-project agent child.  Writing every *sibling*
        project's settings is precisely what a pinned process may not do — its
        ``save()`` writes one file — so pinning there would produce entries that
        are dropped on the way to disk while the caller believed the leak had
        been sealed.
        """
        if pinned_project_id() is not None:
            logger.warning(
                "[workspace] not pinning %s across projects: this process owns "
                "only project %s", model_name, pinned_project_id(),
            )
            return
        if self.is_bootstrap_model(model_name):
            return  # holds no per-project entry by design
        for project in self.projects:
            if self.project_model(project, model_name) is not None:
                continue
            project.model_settings.append(ProjectModel(
                model_name=model_name,
                enabled=self.effective_model_enabled(model_name, project),
                is_active=self.effective_model_active(model_name, project),
                thinking=self.effective_model_thinking(model_name, project),
            ))

    def promote_model_to_workspace_default(self, model_name: str) -> None:
        """Make *model_name* an app-default for projects created from now on.

        One-way: enabling a model in a project promotes it, disabling never
        demotes it, so the default set only grows as models are downloaded.
        Existing projects are pinned first and keep whatever they resolve to
        today — a user who wanted the model there would have enabled it.
        """
        if self.is_bootstrap_model(model_name):
            return  # always enabled, not a configurable default
        if pinned_project_id() is not None:
            # ``enabled`` lives on the workspace record, which a pinned process
            # never writes.  Promoting here would look like it worked and be
            # gone on the next load — along with the pin above that keeps it
            # from leaking into projects that never asked for it.
            logger.warning(
                "[workspace] not promoting %s to a workspace default: this "
                "process owns only project %s", model_name, pinned_project_id(),
            )
            return
        ws = self.get_model(model_name)
        if ws is None or ws.enabled:
            return  # nothing to leak, skip the pin
        self.pin_model_for_existing_projects(model_name)
        ws.enabled = True

    def set_project_model_enabled(self, project: Project, model_name: str, enabled: bool) -> Optional[ProjectModel]:
        """Set *model_name*'s enabled state on *project*, upserting an entry.

        Returns ``None`` without writing anything for the bootstrap model,
        which has no per-project entry (see :meth:`is_bootstrap_model`).
        """
        if self.is_bootstrap_model(model_name):
            return None
        self.sync_project_models(project)
        pm = self.project_model(project, model_name)
        if pm is None:
            pm = ProjectModel(model_name=model_name, enabled=enabled)
            project.model_settings.append(pm)
        else:
            pm.enabled = enabled
        return pm

    def set_project_model_active(self, project: Project, model_name: str, active: bool) -> Optional[ProjectModel]:
        """Set *model_name* as active on *project*, deactivating the others.

        Returns ``None`` without writing anything for the bootstrap model,
        which has no per-project entry (see :meth:`is_bootstrap_model`).
        """
        if self.is_bootstrap_model(model_name):
            return None
        self.sync_project_models(project)
        pm = self.project_model(project, model_name)
        if pm is None:
            pm = ProjectModel(model_name=model_name, is_active=active)
            project.model_settings.append(pm)
        if active:
            for other in project.model_settings:
                other.is_active = (other is pm)
        else:
            pm.is_active = False
        return pm

    def get_model(self, model_name: Optional[str] = None) -> Optional[WorkspaceModel]:
        """Look up a registered model by its short name.

        If *model_name* is ``None``, returns the default model.  Matching is
        alias-aware, so a record written under a model's old name (e.g. a
        workspace from before ``slp-orchestra-mini`` became ``sl-mini``) is found
        by its current name and vice versa.
        """
        if model_name is None:
            model_name = AppSettings().default_model
        target = canonical_model_name(model_name)
        target_cls = SUPPORTED_MODELS.get(target)

        result = None
        for m in self.models:
            # Compare by class where both sides are known models, so two names
            # for the same model still match; otherwise fall back to the name,
            # which keeps unknown (user-added) models from colliding with each
            # other on a shared ``None`` class.
            if canonical_model_name(m.model_name) == target:
                result = m
                break
            if target_cls is not None and SUPPORTED_MODELS.get(m.model_name) is target_cls:
                result = m
                break
        if result is not None:
            result.models_dir = self.models_dir

        return result

    def add_model(self, model_name: str, model_id: str, config: Optional[ModelConfig] = None,
                  enabled: bool = True) -> WorkspaceModel:
        """Register a model in the workspace without changing the active model.

        The record is stored under the canonical name, so a legacy name coming
        from the cloud catalog is not written back into ``config.json``.

        ``enabled`` seeds the app-default that ``sync_project_models`` copies
        into every project, so a caller registering a model on one project's
        behalf should pass ``False`` and enable it on that project itself.
        """
        model_name = canonical_model_name(model_name)
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
        # The bootstrap model gets no per-project entry, so its always-on state
        # resolves through this record (see is_bootstrap_model) — never register
        # it disabled.
        if self.is_bootstrap_model(model_name):
            enabled = True
        entry = WorkspaceModel(model_name=model_name, model_id=model_id, config=config, enabled=enabled, is_active=not has_active, models_dir=self.models_dir)
        self.models.append(entry)
        return entry

    def largest_downloaded_model(self, project: Optional[Project] = None) -> Optional[WorkspaceModel]:
        """Return the best enabled, downloaded model for *project*, or None.

        Ranks by parameter count first (the largest model wins), then by model
        version as a tie-breaker between same-size models.  When *project* is
        given, its per-project model settings override the workspace defaults.
        """
        candidates = [
            m for m in self.models
            if self.effective_model_enabled(m.model_name, project)
            and self.effective_model_downloaded(m.model_name)
            and m.param_size_b > 0
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda m: m._rank, reverse=True)
        best = candidates[0]
        best.models_dir = self.models_dir
        return best

    def ranked_models_for(self, capability: str, project: Optional[Project] = None) -> List["WorkspaceModel"]:
        """Enabled, downloaded models declaring *capability*, best-first.

        Ranked by :attr:`WorkspaceModel._rank` (larger size first, then newer
        version), so the most capable model leads. Callers walk this list down to
        a smaller same-capability model when the largest doesn't fit in memory,
        rather than dropping straight to the tiny default. Empty when no
        downloaded model declares the capability.  Per-project settings
        (``project=``) override the workspace defaults.
        """
        cap = capability.lower()
        candidates = [
            m for m in self.models
            if self.effective_model_enabled(m.model_name, project)
            and self.effective_model_downloaded(m.model_name)
            and m.param_size_b > 0
            and cap in [c.lower() for c in m.effective_capabilities]
        ]
        candidates.sort(key=lambda m: m._rank, reverse=True)
        for m in candidates:
            m.models_dir = self.models_dir
        return candidates

    def best_model_for(self, capability: str, project: Optional[Project] = None) -> Optional[WorkspaceModel]:
        """Return the single best model for a capability — the largest (by
        size, then newer version) that declares it — or the largest downloaded
        model when none declares it.

        Memory is NOT considered here; callers that must fit the model in VRAM
        use :meth:`ranked_models_for` and walk down to a smaller candidate.
        """
        ranked = self.ranked_models_for(capability, project)
        if ranked:
            return ranked[0]
        return self.largest_downloaded_model(project)

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

    def _workspace_record(self, projects: List[dict]) -> dict:
        """The ``config.json`` document, with *projects* as its project list."""
        ws_data = self.model_dump(mode="json")
        ws_data.pop("model_dir", None)
        ws_data.pop("model_path", None)
        ws_data.pop("current_project", None)
        ws_data.pop("model_id", None)
        ws_data["projects"] = projects
        active = self.active_model
        return {"id": active.model_id if active else "", "workspace": ws_data}

    def save(self) -> None:
        """Persist workspace state, each project into its own file.

        Serialized and atomic.  Several background tasks write the workspace
        from their own threads — document indexing, MCP provisioning — and two
        of them writing the same file directly would
        interleave into a truncated or half-updated config.  Every file is
        snapshotted under the lock and swapped in with a rename, so a reader
        (or a crash) sees either the old file or the new one.

        The split matters for more than tidiness.  When every project lived in
        one document, a save was a whole-workspace snapshot: two writers each
        holding a full :class:`Workspace` would have the last one silently
        discard the other project's newly indexed documents and MCP
        registrations.  Now a project's own state is only ever written by
        whoever owns that project.

        What gets written depends on what this process owns:

        * a per-project agent child (pinned, see
          :func:`~smartloop.workspace.context.pin_project`) writes **only** its
          own ``project.json`` — the workspace record belongs to the
          supervisor, and a child writing it would clobber its siblings;
        * the supervisor, which has delegated every project to a child (see
          :func:`~smartloop.workspace.context.delegate_projects`), writes the
          workspace record and *nothing else* — unless it has explicitly
          claimed one project with
          :func:`~smartloop.workspace.context.own_project`, which is how
          create, import and delete still write a file no child owns yet.
          Without this the clobber simply moved tiers: the supervisor holds one
          ``Workspace`` for its whole life, so writing every project's file
          would republish a start-of-process snapshot over the documents and
          MCP servers children had indexed since;
        * any other unpinned process — the ``--no-supervised`` single server,
          the CLI — writes the workspace record and every project's file, which
          is what it has always done: under that fallback it is the only
          process there is.

        ``SLP_LEGACY_CONFIG=1`` restores the single-document format; see
        :func:`_legacy_layout`.
        """
        with self._save_lock:
            if _legacy_layout():
                dumped = self.model_dump(mode="json")
                _atomic_write_json(
                    self.config_file, self._workspace_record(dumped.get("projects") or []),
                )
                return

            pinned = pinned_project_id()
            if pinned:
                project = next((p for p in self.projects if p.id == pinned), None)
                if project is not None:
                    project.save_state()
                else:
                    logger.warning(
                        "[workspace] pinned to project %s, which is not in this "
                        "workspace — nothing saved", pinned,
                    )
                return

            for project in self._writable_projects():
                project.save_state()
            _atomic_write_json(
                self.config_file,
                self._workspace_record([p.to_index() for p in (self.projects or [])]),
            )

    def _writable_projects(self) -> List["Project"]:
        """The projects whose own file this unpinned process may write.

        Every one of them, unless the projects have been delegated to
        per-project children — then only the one explicitly claimed, if any.
        """
        projects = self.projects or []
        if not projects_delegated():
            return list(projects)
        owned = owned_project_id()
        if not owned:
            return []
        return [p for p in projects if p.id == owned]

    @classmethod
    def _hydrate_projects(cls, ws_data: dict) -> dict:
        """Fold each project's own ``project.json`` into the workspace record.

        Called on the raw document before validation, so a project is built
        once from its complete state rather than being validated thin and
        patched afterwards.  ``project_dir`` is derived here the same way
        :meth:`_rebase_paths` derives it — from the current home, never from
        what was persisted — because the file has to be found before the
        project object that would tell us where it is exists.

        A project with no file of its own (not yet migrated, or a workspace
        still in the legacy layout) keeps whatever ``config.json`` holds for
        it, so a half-migrated workspace still loads completely.
        """
        if _legacy_layout():
            return ws_data

        workspace_dir = os.path.join(_home_dir(), ws_data.get("id") or "")
        hydrated: List[dict] = []
        for entry in ws_data.get("projects") or []:
            project_id = entry.get("id") if isinstance(entry, dict) else None
            if not project_id:
                hydrated.append(entry)
                continue
            state = read_project_state(os.path.join(workspace_dir, project_id))
            if not state:
                hydrated.append(entry)
                continue
            # The project's own file wins for its own state; the workspace
            # record stays authoritative for the selection.
            merged = {**entry, **state}
            for field in _WORKSPACE_OWNED_PROJECT_FIELDS:
                if field in entry:
                    merged[field] = entry[field]
            hydrated.append(merged)
        ws_data["projects"] = hydrated
        return ws_data

    @classmethod
    def _create_new_workspace(cls, config: ModelConfig) -> 'Workspace':
        workspace_id = str(uuid4())
        workspace_dir = os.path.join(_home_dir(), workspace_id)
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
                ws_data = cls._hydrate_projects(ws_data)

                workspace = Workspace.model_validate(ws_data)

                # Re-resolve paths against current base_dir
                workspace.workspace_dir = os.path.join(_home_dir(), workspace.id)
                workspace.models_dir = os.path.join(workspace.workspace_dir, "models")
                cls._rebase_paths(workspace)

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
    @staticmethod
    def _rebase_paths(workspace: 'Workspace') -> None:
        """Re-anchor every persisted absolute path onto the current home.

        The framework must be self-contained: copy ``<home_dir>`` somewhere
        else — another machine, another user, a macOS App Sandbox container —
        and it has to run as-is.  ``config.json`` stores absolute paths that
        were correct when written, so nothing persisted may be trusted as a
        location; it is all re-derived from ``AppSettings.home_dir`` here.

        Covers the paths that would otherwise dangle:

        * ``project_dir`` — ``<workspace_dir>/<project_id>`` by construction,
          so re-deriving it can never relocate a normally-created project;
        * ``document.path`` — repointed at the project's own ``documents/``
          folder whenever the file is actually there, since documents are
          project state and travel with it.

        ``project.cwd`` is deliberately left alone: it is the user's own
        folder, not framework state, and a stale one is simply unusable rather
        than wrong.  MCP server commands need no fixing — built-in servers are
        always relaunched under the current interpreter (see
        ``LocalMCPClient._resolve_command``).
        """
        for project in workspace.projects or []:
            if project.id:
                project.project_dir = os.path.join(workspace.workspace_dir, project.id)
            docs_dir = project.documents_dir
            for doc in project.documents or []:
                if not doc.path:
                    continue
                # The framework's own copy always wins, even when the stored
                # path still resolves.  After a migration both exist, and
                # leaving the pointer on the old user-space copy would keep the
                # install dependent on a directory outside <home_dir>.
                candidate = os.path.join(docs_dir, doc.name or os.path.basename(doc.path))
                if os.path.isfile(candidate):
                    doc.path = candidate

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
            ws_data = cls._hydrate_projects(ws_data)

            workspace = Workspace.model_validate(ws_data)

            # Re-resolve paths against home_dir
            workspace.workspace_dir = os.path.join(_home_dir(), workspace.id)
            workspace.models_dir = os.path.join(workspace.workspace_dir, "models")
            cls._rebase_paths(workspace)

            return workspace
        except (json.JSONDecodeError, ValueError):
            return None

    def refresh_project(self, project_id: str) -> bool:
        """Re-read one project's own file if someone has written it since.

        Under supervision a project's own state belongs to the child pinned to
        it, and this process holds a copy taken when it first learned the
        project existed.  Nothing tells it when that copy goes out of date, so
        anything deciding from per-project state here decides from a snapshot:
        a model disabled in a project went on answering that project's
        requests, because the model host runs in the supervisor and resolved
        selection against settings the child had already overwritten.

        Narrower than :meth:`reload`, which re-reads every project: this is
        called on the host's request path, where the bound project is the only
        one that matters.  Gated on the file's mtime, so the usual cost is one
        ``stat`` and the read happens only when a child actually wrote.

        The workspace record stays authoritative for the fields it owns — the
        selection is the workspace's answer, not the project's (see
        :data:`_WORKSPACE_OWNED_PROJECT_FIELDS`) — and the existing
        :class:`Project` is updated in place rather than replaced, so a caller
        already holding it sees the new state instead of a detached object.

        Returns True when something was re-read.
        """
        if not project_id:
            return False
        project = next(
            (p for p in (self.projects or []) if p.id == project_id), None,
        )
        if project is None or not project.project_dir:
            return False
        try:
            mtime = os.stat(project_state_path(project.project_dir)).st_mtime_ns
        except OSError:
            return False
        if self._state_mtimes.get(project_id) == mtime:
            return False
        state = read_project_state(project.project_dir)
        # Record the mtime even for an unreadable file: a project.json that is
        # missing or corrupt is not going to become readable by being stat-ed
        # once per frame, and read_project_state has already logged it.
        self._state_mtimes[project_id] = mtime
        if not state:
            return False
        fresh = Project(**{
            **state, "id": project.id, "project_dir": project.project_dir,
        })
        for name in type(project).model_fields:
            if name in _WORKSPACE_OWNED_PROJECT_FIELDS or name in ("id", "project_dir"):
                continue
            setattr(project, name, getattr(fresh, name))
        return True

    def refresh_models(self) -> bool:
        """Re-read the registered models if the workspace record has moved on.

        The mirror of :meth:`refresh_project`.  That one carries a child's
        write up to the process holding the device; this carries the workspace
        owner's write *down* to a child.  Weights are workspace-wide — one
        shared ``models_dir``, one registry in ``config.json``, written only by
        whoever owns that file — so a model downloaded while a child is running
        is invisible to it: the child holds the workspace it loaded at spawn,
        and ``_ensure_workspace`` re-reads from disk only while no model
        pipeline exists.  :meth:`reload` does not help, by design — it refreshes
        the project list and nothing else.

        The visible symptom is a model that downloaded successfully and then
        cannot be used: absent from ``GET /v1/projects/<id>/models``, and a 404
        from the ``PATCH`` that would enable it, both answered by the child.

        Only the model registry is taken.  The project list is not: a child
        holds project state its own siblings own, and adopting a sibling's
        record here would undo the split that keeps one process from
        overwriting another's work.

        Returns True when something was re-read.
        """
        try:
            mtime = os.stat(self.config_file).st_mtime_ns
        except OSError:
            return False
        if self._models_mtime == mtime:
            return False
        # Record it either way: an unreadable or malformed config will not
        # become readable by being stat-ed again on the next request.
        self._models_mtime = mtime
        try:
            with open(self.config_file, "r") as fh:
                content = fh.read().strip()
            if not content:
                return False
            models = (json.loads(content).get("workspace") or {}).get("models")
        except (OSError, json.JSONDecodeError, ValueError, AttributeError):
            return False
        if not isinstance(models, list):
            return False
        try:
            fresh = [WorkspaceModel.model_validate(m) for m in models]
        except ValidationError:
            logger.warning(
                "[workspace] could not read the model registry from %s; "
                "keeping the one we have", self.config_file,
            )
            return False
        for model in fresh:
            # Same re-anchoring ``load`` does: nothing persisted is trusted as
            # a location, so the weights are found under *this* home.
            model.models_dir = self.models_dir
        self.models = fresh
        return True

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
        """The project this unit of work belongs to.

        Resolution is scoped before it is global.  A request that named a
        ``project_id`` binds it for its own lifetime (and a per-project agent
        child pins one for the whole process), so concurrent chats on different
        projects each resolve their own — where previously both read a single
        ``current`` flag that the later request had already moved.

        The flag remains the fallback, and now means only "the project the UI
        has selected": it answers for unbound callers such as the CLI, the
        supervisor's own reads, and clients that send no ``project_id``.
        """
        bound_id = active_project_id()
        if bound_id:
            for project in self.projects:
                if project.id == bound_id:
                    return project
            # Bound to a project this workspace doesn't have.  Falling back to
            # the flag would silently answer for the wrong project, so say so —
            # but don't raise: a stale binding must not take down the request.
            logger.warning(
                "[project-context] bound project %s is not in this workspace; "
                "falling back to the selected project", bound_id,
            )
        for project in self.projects:
            if project.current:
                return project
        return None
