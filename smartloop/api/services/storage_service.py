"""Disk-storage reporting and cleanup for the SLP home directory.

Reports what ``<home_dir>`` is spending disk on and reclaims the parts that are
safe to lose.  This used to live in the desktop app as Electron JS walking
``~/.smartloop`` with ``fs`` and hand-editing ``config.json`` afterwards; the
framework owns that state, so it does the work here and the client just asks.

Two rules shape everything below:

* **Sizing and deleting read the same list.**  Each ``_scan_*`` helper returns
  candidate rows and both ``compute_usage`` and ``run_cleanup`` work from them,
  so what the UI offers to free is exactly what gets freed.
* **A path is deletable only if it matches a known shape.**  Validation is
  positive (this is a ``models/<org>/<repo>`` directory) rather than a list of
  things to avoid, and every candidate is additionally checked to resolve
  inside ``<home_dir>`` and not to be a symlink.

Models are kept when the workspace says they are in use — ``enabled`` (the
picker's flag), ``is_active``, the default orchestrator, or resident in
``ModelService`` — plus the bundled ``smartloop-ai`` and ``embeddings`` trees,
which the framework cannot run without.
"""

import logging
import os
import re
import shutil
import stat
import sys
import time
import uuid
from typing import Any, Optional

from smartloop.config.app_settings import AppSettings
from smartloop.utils.logging_setup import DEFAULT_BACKUP_COUNT, truncate_log_file
from smartloop.workspace.workspace import Workspace

logger = logging.getLogger(__name__)
_settings = AppSettings()

# Model trees the framework needs to function: the bundled orchestrator and the
# embedding model RAG depends on.  Matched on the org directory so an unregistered
# leftover under them is protected too.
_BUNDLED_ORGS = ("smartloop-ai", "embeddings")

# Installed framework payloads sit in <home>/<x.y.z>/ next to everything else.
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")

# Regenerable caches. Both are written by the desktop app / installer rather than
# by us, but they live under our home and nothing else prunes them.
_CACHE_DIRS = ("cache", "templates")

# Never removed, whatever category is being swept: credentials, the workspace
# record itself, liveness markers, install markers, and the desktop app's own DB.
_NEVER_DELETE = frozenset({
    "config.json", ".credentials", ".key", ".installed-version", ".archive-etag",
    ".binary-hash", "server.pid", "server.port", "version", "installed",
    "studio.db", "workspace.db", "token_usage.jsonl",
})

# Windows can refuse to unlink a GGUF a worker still has mapped, and our own
# executor shuts down without waiting, so give the mapping a moment to drop.
_DELETE_ATTEMPTS = 3
_DELETE_BACKOFF_S = 0.25

# How long to wait for in-flight generation to finish before giving up on a
# model sweep.  Deleting weights out from under a decoding context is a native
# use-after-free, so this is a hard precondition, not an optimization.
_DRAIN_TIMEOUT_S = 30.0

_BUSY_TASKS = ("downloader",)


# ── Paths and sizes ────────────────────────────────────────────────────────

def _home() -> str:
    """The SLP home directory, resolved."""
    return os.path.realpath(_settings.home_dir)


def _walk_size(path: str) -> tuple[int, int]:
    """Return ``(bytes, files)`` for a file or directory tree.

    Symlinks are neither followed nor counted — a link into the user's own
    files must not show up as space we can reclaim.  Missing paths are 0.
    """
    try:
        st = os.lstat(path)
    except OSError:
        return 0, 0
    if stat.S_ISLNK(st.st_mode):
        return 0, 0
    if stat.S_ISREG(st.st_mode):
        return st.st_size, 1
    if not stat.S_ISDIR(st.st_mode):
        return 0, 0

    total = 0
    files = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
                    files += 1
            except OSError:
                continue
    return total, files


def _dirs_in(path: str, *, skip_hidden: bool = False) -> list[str]:
    """Immediate subdirectory names of *path*, skipping symlinks. Sorted."""
    try:
        return sorted(
            e.name for e in os.scandir(path)
            if e.is_dir(follow_symlinks=False) and not e.is_symlink()
            and not (skip_hidden and e.name.startswith("."))
        )
    except OSError:
        return []


def _row(path: str, *, removable: bool, keep_reason: Optional[str] = None, **extra) -> dict:
    """Build a candidate row with its size filled in."""
    size, files = _walk_size(path)
    row = {
        "path": path,
        "bytes": size,
        "files": files,
        "removable": removable and keep_reason is None,
        "keep_reason": keep_reason,
    }
    row.update(extra)
    return row


# ── Workspace and model state ──────────────────────────────────────────────

def _load_workspace(model_service: Any = None) -> Workspace:
    """Return the live workspace, or raise ``RuntimeError``.

    Always the singleton the rest of the process is using — never a fresh
    ``Workspace.load()`` copy, since mutating and saving that would clobber
    whatever the running server has in memory.
    """
    workspace = None
    if model_service is not None:
        try:
            workspace = model_service._ensure_workspace()
        except Exception as e:                       # noqa: BLE001 - reported as 503
            raise RuntimeError(f"Workspace unavailable: {e}") from e
    else:
        workspace = Workspace.load()

    if not isinstance(workspace, Workspace):
        raise RuntimeError("Workspace not initialized")
    return workspace


def _resident_model_names(model_service: Any) -> set[str]:
    """Model names currently holding a loaded context in ``ModelService``.

    Covers the named slots plus the per-capability ``_cap_llm_<name>`` caches.
    A model resident in one of these is being used right now regardless of what
    the workspace flags say, so its weights must stay on disk.
    """
    names: set[str] = set()
    if model_service is None:
        return names

    for attr in ("_model_name", "_decision_model_name", "_large_model_name",
                 "_vision_model_name"):
        value = getattr(model_service, attr, None)
        if isinstance(value, str) and value:
            names.add(value)

    try:
        from smartloop.api.services.model_service import _GPU_FAULTED
    except Exception:                                # noqa: BLE001
        _GPU_FAULTED = None

    for attr, value in list(getattr(model_service, "__dict__", {}).items()):
        if not attr.startswith("_cap_llm_"):
            continue
        # A None slot was never loaded; the faulted sentinel is a latch, not a
        # live context, so neither pins the files.
        if value is None or (_GPU_FAULTED is not None and value is _GPU_FAULTED):
            continue
        names.add(attr[len("_cap_llm_"):])

    return names


def _keep_model_ids(workspace: Workspace, model_service: Any) -> dict[str, str]:
    """Map ``model_id`` → why it must be kept, for every protected model."""
    keep: dict[str, str] = {}

    def mark(model_id: Optional[str], reason: str) -> None:
        if model_id and model_id not in keep:
            keep[model_id] = reason

    for name in _resident_model_names(model_service):
        entry = workspace.get_model(name)
        if entry is not None:
            mark(entry.model_id, "loaded")

    default = workspace.get_model()
    if default is not None:
        mark(default.model_id, "default")

    for m in workspace.models:
        if m.is_active:
            mark(m.model_id, "active")
    for m in workspace.models:
        # ``enabled`` is what the model picker writes, so this is the user's
        # own selection — the thing the old vendor-directory sweep destroyed.
        if m.enabled:
            mark(m.model_id, "enabled")

    return keep


# ── Scanners ───────────────────────────────────────────────────────────────

def _scan_models(workspace: Workspace, model_service: Any) -> list[dict]:
    """One row per ``models/<org>/<repo>`` directory present on disk."""
    models_dir = workspace.models_dir
    if not models_dir or not os.path.isdir(models_dir):
        return []

    keep = _keep_model_ids(workspace, model_service)
    resident = _resident_model_names(model_service)
    by_id = {m.model_id: m for m in workspace.models}

    rows = []
    # Hidden directories are bookkeeping, not models — the HF cache keeps a
    # ".locks" tree under embeddings/.  Their bytes still show up in the
    # category total, they just aren't offered as something to remove.
    for org in _dirs_in(models_dir, skip_hidden=True):
        for repo in _dirs_in(os.path.join(models_dir, org), skip_hidden=True):
            model_id = f"{org}/{repo}"
            entry = by_id.get(model_id)
            keep_reason = "bundled" if org in _BUNDLED_ORGS else keep.get(model_id)
            rows.append(_row(
                os.path.join(models_dir, org, repo),
                removable=True,
                keep_reason=keep_reason,
                model_id=model_id,
                model_name=entry.model_name if entry else None,
                downloaded=bool(entry.downloaded) if entry else False,
                enabled=bool(entry.enabled) if entry else False,
                is_active=bool(entry.is_active) if entry else False,
                loaded=bool(entry and entry.model_name in resident),
                # No workspace entry means an aborted or orphaned download.
                registered=entry is not None,
            ))
    return rows


def _running_versions() -> set[str]:
    """Version directory names that belong to the running install."""
    home = _home()
    running = set()

    for marker in (".installed-version", "version"):
        try:
            with open(os.path.join(home, marker)) as f:
                value = f.read().strip()
            if value:
                running.add(value)
        except OSError:
            continue

    # The binary we are executing is the authority, and the markers can be
    # stale or missing entirely.
    executable = os.path.realpath(sys.executable or "")
    if executable.startswith(home + os.sep):
        first = os.path.relpath(executable, home).split(os.sep)[0]
        if _VERSION_RE.match(first):
            running.add(first)

    return running


def _scan_versions() -> list[dict]:
    """Installed framework payloads other than the running one."""
    home = _home()
    running = _running_versions()
    rows = []
    for name in _dirs_in(home):
        if not _VERSION_RE.match(name):
            continue
        rows.append(_row(
            os.path.join(home, name),
            removable=True,
            keep_reason="running" if name in running else None,
            version=name,
        ))
    return rows


def _scan_logs() -> list[dict]:
    """The server log and its rotated backups."""
    home = _home()
    rows = []

    log_path = os.path.join(home, "server.log")
    if os.path.isfile(log_path):
        # Emptied in place, never unlinked — the handler holds it open.
        rows.append(_row(log_path, removable=True, action="truncate"))

    for n in range(1, DEFAULT_BACKUP_COUNT + 1):
        backup = os.path.join(home, f"server.log.{n}")
        if os.path.isfile(backup):
            rows.append(_row(backup, removable=True, action="delete"))

    return rows


def _scan_cache() -> list[dict]:
    """Contents of the regenerable cache directories (the dirs themselves stay)."""
    home = _home()
    rows = []
    for name in _CACHE_DIRS:
        parent = os.path.join(home, name)
        try:
            entries = sorted(os.scandir(parent), key=lambda e: e.name)
        except OSError:
            continue
        for entry in entries:
            if entry.is_symlink():
                continue
            rows.append(_row(entry.path, removable=True, cache=name))
    return rows


def _scan_orphan_projects(workspace: Workspace) -> list[dict]:
    """Project directories under the workspace that ``config.json`` doesn't know about."""
    workspace_dir = workspace.workspace_dir
    if not workspace_dir or not os.path.isdir(workspace_dir):
        return []

    known = set()
    for project in workspace.projects:
        if project.project_dir:
            known.add(os.path.realpath(project.project_dir))

    rows = []
    for name in _dirs_in(workspace_dir):
        if name == "models":
            continue
        try:
            uuid.UUID(name)
        except ValueError:
            # Only ever consider directories shaped like a project id.
            continue
        path = os.path.join(workspace_dir, name)
        if os.path.realpath(path) in known:
            continue
        rows.append(_row(path, removable=True, project_id=name))
    return rows


def _scan_conversations(workspace: Workspace) -> list[dict]:
    """Session transcripts, reported but never deleted.

    A transcript not referenced by ``project.sessions`` is *not* evidence of an
    orphan: a session id only lands there when an assistant turn arrives while
    that project is current, and the desktop app keeps its own conversation list
    in ``studio.db``.  So an untracked JSONL is usually a chat the UI still
    shows, and deleting it would silently drop that chat's memory.  Deleting
    these needs an explicit keep-list from the client; until then this category
    is informational.
    """
    conversations = os.path.join(_home(), "conversations")
    if not os.path.isdir(conversations):
        return []

    tracked = set()
    for project in workspace.projects:
        tracked.update(getattr(project, "sessions", None) or [])

    rows = []
    try:
        entries = sorted(os.scandir(conversations), key=lambda e: e.name)
    except OSError:
        return []
    for entry in entries:
        if entry.is_symlink() or not entry.name.endswith(".jsonl"):
            continue
        session_id = entry.name[: -len(".jsonl")]
        rows.append(_row(
            entry.path,
            removable=False,
            keep_reason="not deletable without a client keep-list",
            session_id=session_id,
            tracked=session_id in tracked,
        ))
    return rows


# ── Validation ─────────────────────────────────────────────────────────────

def _assert_inside_home(path: str) -> str:
    """Resolve *path* and require it to sit strictly inside the home directory."""
    home = _home()
    resolved = os.path.realpath(path)
    if resolved == home:
        raise ValueError(f"Refusing to act on the home directory itself: {path}")
    if os.path.commonpath([resolved, home]) != home:
        raise ValueError(f"Path escapes {home}: {path}")
    if os.path.basename(resolved) in _NEVER_DELETE:
        raise ValueError(f"Protected path: {path}")
    if os.path.islink(path):
        raise ValueError(f"Refusing to delete through a symlink: {path}")
    return resolved


def _validate(path: str, shape: str, workspace: Workspace) -> str:
    """Require *path* to match the expected shape for its category.

    Positive validation: each category knows precisely what its paths look
    like, so anything else — however harmless it seems — is rejected.
    """
    resolved = _assert_inside_home(path)
    home = _home()

    if shape == "models":
        models_dir = os.path.realpath(workspace.models_dir or "")
        if os.path.dirname(os.path.dirname(resolved)) != models_dir:
            raise ValueError(f"Not a models/<org>/<repo> directory: {path}")
    elif shape == "versions":
        name = os.path.relpath(resolved, home)
        if os.sep in name or not _VERSION_RE.match(name):
            raise ValueError(f"Not a version directory: {path}")
        if name in _running_versions():
            raise ValueError(f"Refusing to delete the running install: {path}")
    elif shape == "logs":
        name = os.path.relpath(resolved, home)
        backup = re.fullmatch(r"server\.log\.(\d+)", name)
        # Compare the index numerically: a character class built from
        # DEFAULT_BACKUP_COUNT would silently mean something else at 10+.
        if name != "server.log" and not (
            backup and 1 <= int(backup.group(1)) <= DEFAULT_BACKUP_COUNT
        ):
            raise ValueError(f"Not a server log file: {path}")
    elif shape == "cache":
        parent = os.path.basename(os.path.dirname(resolved))
        if parent not in _CACHE_DIRS or os.path.dirname(os.path.dirname(resolved)) != home:
            raise ValueError(f"Not a cache entry: {path}")
    elif shape == "orphan_projects":
        workspace_dir = os.path.realpath(workspace.workspace_dir or "")
        if os.path.dirname(resolved) != workspace_dir:
            raise ValueError(f"Not a project directory: {path}")
        uuid.UUID(os.path.basename(resolved))          # raises ValueError
        for project in workspace.projects:
            if project.project_dir and os.path.realpath(project.project_dir) == resolved:
                raise ValueError(f"Refusing to delete a live project: {path}")
    else:
        raise ValueError(f"Unknown category: {shape}")

    return resolved


# ── Removal ────────────────────────────────────────────────────────────────

def _remove(path: str, *, truncate: bool = False) -> tuple[bool, Optional[str]]:
    """Delete (or empty) *path*, retrying briefly. Returns ``(ok, error)``."""
    last_error = None
    for attempt in range(_DELETE_ATTEMPTS):
        try:
            if truncate:
                truncate_log_file(os.path.dirname(path))
            elif os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.unlink(path)
            return True, None
        except OSError as e:
            # A GGUF a spawn worker still has mapped can refuse to go on
            # Windows; the mapping usually drops within a beat.
            last_error = str(e)
            if attempt < _DELETE_ATTEMPTS - 1:
                time.sleep(_DELETE_BACKOFF_S)
    return False, last_error


def _prune_empty_orgs(workspace: Workspace) -> None:
    """Remove ``models/<org>`` directories left empty by a model deletion."""
    models_dir = workspace.models_dir
    if not models_dir or not os.path.isdir(models_dir):
        return
    for org in _dirs_in(models_dir):
        if org in _BUNDLED_ORGS:
            continue
        path = os.path.join(models_dir, org)
        try:
            if not os.listdir(path):
                os.rmdir(path)
        except OSError:
            continue


def _refuse_if_busy(model_service: Any, *, needs_indexer: bool) -> None:
    """Raise ``RuntimeError`` while a task that writes to these trees is running."""
    queue = getattr(model_service, "_task_queue", None)
    get_status = getattr(queue, "get_status", None)
    if not callable(get_status):
        return

    names = list(_BUSY_TASKS) + (["indexer"] if needs_indexer else [])
    for name in names:
        try:
            status = get_status(name)
        except Exception:                            # noqa: BLE001
            continue
        if isinstance(status, dict) and status.get("status") == "running":
            raise RuntimeError(f"Cannot clean up while the {name} task is running")


def _drain_inference(model_service: Any) -> int:
    """Acquire every inference permit so no generation is in flight.

    Returns the number of permits held, for the caller to release.  Raises
    ``RuntimeError`` on timeout rather than deleting weights a live decode
    could still be reading.
    """
    semaphore = getattr(model_service, "_inference_lock", None)
    if semaphore is None or not hasattr(semaphore, "acquire"):
        return 0

    permits = max(1, int(getattr(_settings, "n_parallel", 1) or 1))
    deadline = time.monotonic() + _DRAIN_TIMEOUT_S
    held = 0
    for _ in range(permits):
        remaining = max(0.0, deadline - time.monotonic())
        if not semaphore.acquire(timeout=remaining):
            for _ in range(held):
                semaphore.release()
            raise RuntimeError("Model is busy generating; try again in a moment")
        held += 1
    return held


def _release_inference(model_service: Any, held: int) -> None:
    semaphore = getattr(model_service, "_inference_lock", None)
    if semaphore is None:
        return
    for _ in range(held):
        try:
            semaphore.release()
        except Exception:                            # noqa: BLE001
            break


# ── Public API ─────────────────────────────────────────────────────────────

def _category(rows: list[dict], total_path: Optional[str] = None, **extra) -> dict:
    """Roll candidate rows up into a category report."""
    if total_path is not None:
        total, files = _walk_size(total_path)
    else:
        total = sum(r["bytes"] for r in rows)
        files = sum(r["files"] for r in rows)
    reclaimable = sum(r["bytes"] for r in rows if r["removable"])
    report = {
        "bytes": total,
        "files": files,
        "reclaimable": reclaimable,
        "kept": max(0, total - reclaimable),
        "items": rows,
    }
    report.update(extra)
    return report


def compute_usage(*, model_service: Any = None) -> dict:
    """Report what the home directory is using and what can be reclaimed."""
    workspace = _load_workspace(model_service)
    home = _home()
    total, files = _walk_size(home)

    categories = {
        "models": _category(
            _scan_models(workspace, model_service), total_path=workspace.models_dir
        ),
        "versions": _category(_scan_versions()),
        "logs": _category(_scan_logs()),
        "cache": _category(_scan_cache()),
        "orphan_projects": _category(_scan_orphan_projects(workspace)),
        "conversations": _category(_scan_conversations(workspace)),
    }

    reclaimable = sum(c["reclaimable"] for c in categories.values())
    return {
        "home_dir": home,
        "total": total,
        "files": files,
        "reclaimable": reclaimable,
        "kept": max(0, total - reclaimable),
        "categories": categories,
    }


def _plan_cleanup(
    workspace: Workspace,
    model_service: Any,
    *,
    models: bool,
    model_ids: Optional[list[str]],
    versions: bool,
    logs: bool,
    cache: bool,
    orphan_projects: bool,
) -> list[tuple[str, dict]]:
    """Resolve the requested categories into validated (category, row) work."""
    planned: list[tuple[str, dict]] = []

    if models:
        rows = [r for r in _scan_models(workspace, model_service) if r["removable"]]
        if model_ids is not None:
            wanted = set(model_ids)
            available = {r["model_id"] for r in rows}
            unknown = sorted(wanted - available)
            if unknown:
                raise ValueError(
                    "Not removable (unknown, in use, or bundled): " + ", ".join(unknown)
                )
            rows = [r for r in rows if r["model_id"] in wanted]
        planned += [("models", r) for r in rows]

    if versions:
        planned += [("versions", r) for r in _scan_versions() if r["removable"]]
    if logs:
        planned += [("logs", r) for r in _scan_logs() if r["removable"]]
    if cache:
        planned += [("cache", r) for r in _scan_cache() if r["removable"]]
    if orphan_projects:
        planned += [
            ("orphan_projects", r)
            for r in _scan_orphan_projects(workspace) if r["removable"]
        ]

    for category, row in planned:
        _validate(row["path"], category, workspace)

    return planned


def run_cleanup(
    *,
    model_service: Any,
    models: bool = False,
    model_ids: Optional[list[str]] = None,
    versions: bool = False,
    logs: bool = False,
    cache: bool = False,
    orphan_projects: bool = False,
    dry_run: bool = False,
) -> dict:
    """Reclaim disk space in the selected categories.

    Nothing is deleted unless its category is explicitly requested — there is
    deliberately no "everything" switch.  Model deletion additionally unloads
    the model and rewrites its workspace flags, so the picker stops offering a
    model whose weights are gone.
    """
    workspace = _load_workspace(model_service)
    _refuse_if_busy(model_service, needs_indexer=orphan_projects)

    planned = _plan_cleanup(
        workspace, model_service,
        models=models, model_ids=model_ids, versions=versions,
        logs=logs, cache=cache, orphan_projects=orphan_projects,
    )

    if dry_run:
        return _report(
            workspace, model_service,
            deleted=[{"category": c, "path": r["path"], "bytes": r["bytes"]} for c, r in planned],
            failed=[], models_marked=[], dry_run=True,
        )

    model_rows = [r for c, r in planned if c == "models"]
    # Only a model sweep touches files the inference path has open.
    held = _drain_inference(model_service) if model_rows else 0
    deleted: list[dict] = []
    failed: list[dict] = []
    models_marked: list[str] = []

    try:
        for row in model_rows:
            name = row.get("model_name")
            if name:
                try:
                    model_service.unload_model_by_name(name)
                except Exception:                    # noqa: BLE001
                    logger.exception("[storage] Failed to unload %s before deletion", name)
        if model_rows:
            # The shared executor's workers each loaded their own copy of the
            # weights, so it has to go before the files can.
            reset = getattr(model_service, "_reset_llm_executor", None)
            if callable(reset):
                reset()

        for category, row in planned:
            ok, error = _remove(row["path"], truncate=row.get("action") == "truncate")
            if ok:
                deleted.append({
                    "category": category, "path": row["path"], "bytes": row["bytes"],
                })
                logger.info("[storage] Removed %s (%s bytes)", row["path"], row["bytes"])
            else:
                failed.append({"category": category, "path": row["path"], "error": error})
                logger.warning("[storage] Could not remove %s: %s", row["path"], error)

        if model_rows:
            deleted_paths = {d["path"] for d in deleted}
            deleted_ids = {
                r["model_id"] for r in model_rows if r["path"] in deleted_paths
            }
            models_marked = _mark_models_removed(workspace, deleted_ids)
            _prune_empty_orgs(workspace)
    finally:
        _release_inference(model_service, held)

    return _report(
        workspace, model_service,
        deleted=deleted, failed=failed, models_marked=models_marked, dry_run=False,
    )


def _mark_models_removed(workspace: Workspace, deleted_ids: set[str]) -> list[str]:
    """Flip the workspace flags for models whose weights are gone, and persist.

    The workspace record is the framework's own source of truth for what is
    downloaded; leaving it claiming a deleted model would keep offering it in
    the picker.
    """
    if not deleted_ids:
        return []

    marked = []
    for m in workspace.models:
        if m.model_id in deleted_ids and (m.downloaded or m.enabled or m.is_active):
            m.downloaded = False
            m.enabled = False
            m.is_active = False
            marked.append(m.model_name)

    if marked:
        workspace.save()
        logger.info("[storage] Marked as removed: %s", ", ".join(marked))
    return marked


def _report(
    workspace: Workspace,
    model_service: Any,
    *,
    deleted: list[dict],
    failed: list[dict],
    models_marked: list[str],
    dry_run: bool,
) -> dict:
    freed_by_category: dict[str, int] = {}
    for item in deleted:
        freed_by_category[item["category"]] = (
            freed_by_category.get(item["category"], 0) + item["bytes"]
        )

    return {
        "dry_run": dry_run,
        # Summed from what was actually removed: a before/after comparison of
        # the tree would drift, since the server keeps writing as we work.
        "freed": sum(item["bytes"] for item in deleted),
        "freed_by_category": freed_by_category,
        "deleted": deleted,
        "failed": failed,
        "models_marked": models_marked,
        # A file still mapped by a worker (Windows) needs the agent restarted
        # before it will go.
        "restart_required": bool(failed),
        "usage": compute_usage(model_service=model_service),
    }
