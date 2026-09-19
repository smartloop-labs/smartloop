"""One-time on-disk layout migrations, run at server startup.

Two of them, both moving state to where the process that owns it can reach it:
:func:`migrate_project_state` relocates skills and documents off a
client-supplied path, and :func:`split_project_state` breaks the single
``config.json`` into a workspace record plus one file per project.

Skills and documents used to live under a *user-space* folder — ``project.cwd``,
defaulting to ``~/smartloop/<name>``.  That path is client-supplied and absolute,
so it does not follow the framework's home:

  * under the macOS App Sandbox ``~`` is redirected to the app container, but an
    absolute path persisted before sandboxing (or written by the un-sandboxed
    CLI) keeps pointing at the real home the sandbox denies — the app then reads
    one workspace and writes another;
  * ``SLP_HOME`` moved the framework's state but left skills and documents
    behind.

They now live under ``project_dir`` (``<home_dir>/<workspace_id>/<project_id>``),
which is derived from ``AppSettings.home_dir`` and therefore always travels with
the framework.  This module moves anything left in the old location.
"""

import logging
import os
import shutil
from typing import TYPE_CHECKING, Optional
from smartloop.paths import confined_home, expand_home, real_home

if TYPE_CHECKING:  # pragma: no cover
    from smartloop.workspace.workspace import Project, Workspace

logger = logging.getLogger(__name__)

#: Copy of the single-document ``config.json`` taken before it is first split.
#: The rollback path: restore this over ``config.json`` and set
#: ``SLP_LEGACY_CONFIG=1``, and the workspace is exactly as it was.
MONOLITH_BACKUP_SUFFIX = ".pre-split.bak"

#: Subdirectories that moved from the user-space folder into ``project_dir``.
_MIGRATED_SUBDIRS = ("skills", "documents")


def _legacy_roots(project: "Project") -> list[str]:
    """Candidate old locations for *project*'s state, most likely first.

    ``cwd`` is where current builds put it.  ``~/smartloop/<name>`` covers
    projects created before ``cwd`` was persisted unconditionally, whose
    location was derived from the name at read time.
    """
    roots: list[str] = []
    if project.cwd:
        roots.append(expand_home(project.cwd))
    if project.name:
        roots.append(os.path.join(real_home(), "smartloop", project.name))
    # De-duplicate while preserving order.
    return list(dict.fromkeys(roots))


def _is_readable_dir(path: str) -> bool:
    """True when *path* is a directory this process may actually read.

    Under the App Sandbox a denied path can still pass ``isdir`` and then raise
    on access, so the readability check is not redundant.
    """
    try:
        return os.path.isdir(path) and os.access(path, os.R_OK | os.X_OK)
    except OSError:
        return False


def _migrate_subdir(src: str, dest: str) -> int:
    """Move the contents of *src* into *dest*. Returns the number of entries moved.

    Entries already present at the destination win — a partially-completed
    earlier run must never clobber newer state — and the source is left in
    place rather than deleted, so a bad migration is always recoverable by
    hand.
    """
    moved = 0
    os.makedirs(dest, exist_ok=True)
    for entry in sorted(os.listdir(src)):
        src_path = os.path.join(src, entry)
        dest_path = os.path.join(dest, entry)
        if os.path.exists(dest_path):
            continue
        try:
            shutil.copytree(src_path, dest_path) if os.path.isdir(src_path) \
                else shutil.copy2(src_path, dest_path)
            moved += 1
        except OSError as exc:
            logger.warning("[migrate] Could not move %s → %s: %s", src_path, dest_path, exc)
    return moved


def migrate_project_state(workspace: "Workspace") -> dict:
    """Move skills/documents from the legacy user-space folder into ``project_dir``.

    Idempotent and best-effort: every filesystem touch is guarded, because the
    legacy path may be unreadable (sandbox) or gone entirely, and neither is a
    reason to fail startup.  Returns ``{project_id: {subdir: count}}`` for the
    entries actually moved.
    """
    migrated: dict = {}
    if workspace is None:
        return migrated

    for project in getattr(workspace, "projects", []) or []:
        if not project.project_dir:
            continue
        for root in _legacy_roots(project):
            # Never migrate a folder onto itself.
            if os.path.normpath(root) == os.path.normpath(project.project_dir):
                continue
            if not _is_readable_dir(root):
                continue
            for sub in _MIGRATED_SUBDIRS:
                src = os.path.join(root, sub)
                if not _is_readable_dir(src):
                    continue
                dest = os.path.join(project.project_dir, sub)
                count = _migrate_subdir(src, dest)
                if count:
                    migrated.setdefault(project.id, {})[sub] = count
                    logger.info(
                        "[migrate] %s: moved %d %s entr%s from %s → %s",
                        project.name or project.id, count, sub,
                        "y" if count == 1 else "ies", src, dest,
                    )
    return migrated


# ---------------------------------------------------------------------------
# a sandbox container's framework home → the real ~/.smartloop
# ---------------------------------------------------------------------------
#
# The framework home is ``~/.smartloop`` in every build, so one install serves
# the app and the CLI alike.  Builds that shipped before the home was resolved
# past the sandbox put it under the rewritten ``$HOME`` instead — inside the
# App Sandbox container, or under ``~/snap/smartloop/<revision>`` — and those
# installs still have their workspace, projects and model weights there.  This
# brings them forward.

#: Dropped in the old home once its contents have been brought forward, so a
#: second start does not walk the tree again.  Names the destination, which is
#: the only thing anyone reading it afterwards wants to know.
ADOPTED_MARKER = ".adopted-into-real-home"


def _move_entry(src: str, dest: str) -> bool:
    """Move one top-level entry of the old home across.  True when it lands.

    A rename, not a copy: the model weights under ``<workspace>/models`` run
    to gigabytes, and both sandboxes put their rewritten home on the same
    volume as the real one, so this costs nothing and needs no free space.
    Done per entry rather than per tree so that one unreadable directory
    leaves the rest of the migration intact — and leaves itself where it was,
    rather than half-copied.
    """
    if os.path.exists(dest):
        # Already brought forward (or created afresh by a newer build); the
        # copy that is already at the real home is the one in use.
        return False
    try:
        shutil.move(src, dest)
        return True
    except OSError:
        logger.warning("[adopt] could not move %s → %s", src, dest, exc_info=True)
        return False


def adopt_confined_home(dest: str) -> Optional[str]:
    """Bring a sandbox-located framework home forward to *dest*.

    *dest* is the real ``~/.smartloop``.  Returns the old home when something
    was actually moved out of it, else ``None``.

    Declines rather than guesses in the two cases where it cannot know what
    the user wants:

    * ``SLP_HOME`` is set — somebody pinned the home deliberately (a service
      unit, a test, a packaged build that means to stay in its container) and
      that instruction outranks this migration;
    * *dest* already holds a ``config.json`` — there are two live workspaces,
      and silently merging them would be a guess about which project list is
      the real one.  The user keeps both and is told.

    Best-effort throughout, like every migration here: under the App Sandbox
    the real home is readable only with the entitlement to match, and a denial
    must leave the install running out of its container rather than fail.
    """
    if os.environ.get("SLP_HOME"):
        return None

    old_home = confined_home()
    if not old_home:
        return None

    src = os.path.join(old_home, ".smartloop")
    if os.path.normpath(src) == os.path.normpath(dest):
        return None
    if not _is_readable_dir(src):
        return None
    if os.path.exists(os.path.join(src, ADOPTED_MARKER)):
        return None

    if os.path.exists(os.path.join(dest, "config.json")):
        logger.warning(
            "[adopt] %s already holds a workspace; leaving the sandboxed one "
            "at %s alone. Point SLP_HOME at it to use it instead.", dest, src,
        )
        return None

    try:
        entries = sorted(os.listdir(src))
    except OSError:
        logger.warning("[adopt] could not read %s", src, exc_info=True)
        return None

    try:
        os.makedirs(dest, exist_ok=True)
    except OSError:
        # The sandbox denying the real home is the expected failure here, and
        # it is not fatal: the old home is untouched and still serves.
        logger.warning("[adopt] could not create %s", dest, exc_info=True)
        return None

    moved = 0
    for name in entries:
        if name == ADOPTED_MARKER:
            continue
        if _move_entry(os.path.join(src, name), os.path.join(dest, name)):
            moved += 1

    if not moved:
        return None

    logger.info("[adopt] moved %d entr%s from %s → %s",
                moved, "y" if moved == 1 else "ies", src, dest)
    try:
        with open(os.path.join(src, ADOPTED_MARKER), "w", encoding="utf-8") as fh:
            fh.write(dest + "\n")
    except OSError:
        # Only a fast path for the next start; the emptied directory and the
        # config.json now at *dest* both already stop a second run.
        logger.debug("[adopt] could not write the marker in %s", src, exc_info=True)
    return src


# ---------------------------------------------------------------------------
# config.json → workspace record + one project.json per project
# ---------------------------------------------------------------------------


def _needs_split(workspace: "Workspace") -> bool:
    """True when at least one project has no state file of its own yet."""
    from smartloop.workspace.workspace import project_state_path

    for project in getattr(workspace, "projects", []) or []:
        if not project.project_dir:
            continue
        if not os.path.isfile(project_state_path(project.project_dir)):
            return True
    return False


def _backup_monolith(config_file: str) -> Optional[str]:
    """Copy ``config.json`` aside before it is rewritten, once.

    An existing backup is never overwritten: it is the pre-migration state,
    and a second (already-split) copy on top of it would destroy the only
    thing a rollback has to restore from.
    """
    if not os.path.isfile(config_file):
        return None
    backup = f"{config_file}{MONOLITH_BACKUP_SUFFIX}"
    if os.path.exists(backup):
        return backup
    try:
        shutil.copy2(config_file, backup)
    except OSError as exc:
        logger.warning("[migrate] Could not back up %s: %s", config_file, exc)
        return None
    logger.info("[migrate] Backed up the pre-split config to %s", backup)
    return backup


def split_project_state(workspace: "Workspace") -> dict:
    """Move each project's own state out of ``config.json`` into its directory.

    A workspace used to be one document holding every project.  That made a
    save a whole-workspace snapshot, so two writers each holding a
    :class:`Workspace` would have the last one silently discard the other
    project's newly indexed documents and MCP registrations — the one thing
    that had to be settled before two processes could run side by side.

    Afterwards ``config.json`` keeps only workspace state (id, models,
    models_dir) plus a thin project index, and each
    ``<project_dir>/project.json`` holds that project's own record.

    Idempotent, and safe to interrupt: the pre-split ``config.json`` is copied
    aside first, every project file is written before the workspace record is
    rewritten, and a project that already has a file is left alone.  Returns
    ``{"migrated": [project ids], "backup": path or None}``.

    To roll back, restore the backup over ``config.json`` and set
    ``SLP_LEGACY_CONFIG=1``; the ``project.json`` files are then ignored
    rather than having to be removed.
    """
    from smartloop.workspace.workspace import _legacy_layout

    result: dict = {"migrated": [], "backup": None}
    if workspace is None or _legacy_layout() or not _needs_split(workspace):
        return result

    result["backup"] = _backup_monolith(workspace.config_file)

    for project in getattr(workspace, "projects", []) or []:
        if not project.project_dir:
            continue
        try:
            os.makedirs(project.project_dir, exist_ok=True)
            project.save_state()
        except OSError as exc:
            logger.warning(
                "[migrate] Could not write state for project %s: %s",
                project.id, exc,
            )
            continue
        result["migrated"].append(project.id)

    # Only now is the monolith replaced by the thin record — every project's
    # state is already on disk, so an interruption before this point leaves the
    # old file intact and the next run simply redoes the write.
    workspace.save()

    logger.info(
        "[migrate] Split config.json — %d project(s) now hold their own state",
        len(result["migrated"]),
    )
    return result
