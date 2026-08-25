"""Project export/import service.

Packages a project into a portable zip archive and restores it on another
machine (or the same one).
"""

import json
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from smartloop.config.app_settings import AppSettings
from smartloop.workspace.workspace import Workspace

logger = logging.getLogger(__name__)
_settings = AppSettings()


def _conversations_dir() -> str:
    """Return ``<home_dir>/conversations/``."""
    return os.path.join(_settings.home_dir, "conversations")


def _add_directory_to_zip(zf: zipfile.ZipFile, src_dir: str, arc_prefix: str) -> None:
    """Recursively add *src_dir* contents under *arc_prefix* in *zf*."""
    for root, _dirs, files in os.walk(src_dir):
        for fname in files:
            full = os.path.join(root, fname)
            arcname = os.path.join(arc_prefix, os.path.relpath(full, src_dir))
            zf.write(full, arcname)


def _strip_oauth(project_data: dict) -> dict:
    """Remove OAuth credentials/tokens from MCP servers in project data."""
    for srv in project_data.get("mcp_servers", []):
        srv.pop("oauth_credentials", None)
        remote = srv.get("remote_config")
        if remote and isinstance(remote, dict):
            remote.pop("token", None)
    return project_data


# ── Export ──────────────────────────────────────────────────────────────

def _read_skills_from_dir(skills_dir: str) -> list:
    """Parse SKILL.md files from a skills directory into Skill objects.

    Each skill lives in its own subdirectory as a SKILL.md file with
    YAML frontmatter (name, description, tools).
    """
    from smartloop.workspace.workspace import Skill

    if not os.path.isdir(skills_dir):
        return []

    skills = []
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
        tools: list[str] = []
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
                    elif line.startswith("tools:"):
                        tools_str = line[len("tools:"):].strip().strip("[]")
                        tools = [t.strip() for t in tools_str.split(",") if t.strip()]

        skills.append(Skill(
            id=str(uuid4())[:8],
            name=name,
            description=description,
            content=content,
            tools=tools,
        ))

    return skills


def export_project(project_id: str) -> Path:
    """Create a zip archive for the given project.

    Returns the path to the temporary zip file.  The caller is responsible
    for cleaning it up after the response has been sent.
    """
    workspace = Workspace.load()
    if workspace is None:
        raise ValueError("Workspace not initialized")

    project = next((p for p in workspace.projects if p.id == project_id), None)
    if project is None:
        raise FileNotFoundError(f"Project {project_id} not found")

    project_data = _strip_oauth(project.model_dump(mode="json"))

    # Skills are exported as files in the project/skills/ directory,
    # not embedded in the manifest.
    project_data.pop("skills", None)

    manifest = {
        "version": "1.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "project": project_data,
    }

    tmp = tempfile.NamedTemporaryFile(
        suffix=".zip", prefix=f"{project.name or project.id}_", delete=False
    )
    tmp.close()

    try:
        with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

            # The project's state directory — skills/, documents/, db/ and
            # the manifest all live here now, so this single tree is the whole
            # payload.  Archives from older builds additionally carried a
            # "project" tree (the old ~/smartloop/<name> folder); import still
            # reads that, but export no longer writes it.
            if project.project_dir and os.path.isdir(project.project_dir):
                _add_directory_to_zip(zf, project.project_dir, "workspace")

            # Conversation history JSONL files
            conv_dir = _conversations_dir()
            for session_id in project.sessions:
                jsonl = os.path.join(conv_dir, f"{session_id}.jsonl")
                if os.path.isfile(jsonl):
                    zf.write(jsonl, f"conversations/{session_id}.jsonl")

    except Exception:
        os.unlink(tmp.name)
        raise

    return Path(tmp.name)


# ── Import ──────────────────────────────────────────────────────────────

def import_project(zip_path: Path, name: Optional[str] = None) -> dict:
    """Import a project from a zip archive.

    Args:
        zip_path: Path to the zip file.
        name: Optional new name for the imported project.  When provided the
              imported project is renamed to this instead of the original.

    Returns a dict with the new project metadata.
    """
    workspace = Workspace.load()
    if workspace is None:
        raise ValueError("Workspace not initialized")

    tmp_dir = tempfile.mkdtemp(prefix="slp_import_")
    try:
        # Extract
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_dir)
        except zipfile.BadZipFile:
            raise ValueError("Uploaded file is not a valid zip archive")

        # Read manifest
        manifest_path = os.path.join(tmp_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise ValueError("Zip archive is missing manifest.json")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        project_data = manifest.get("project")
        if not project_data:
            raise ValueError("manifest.json is missing project data")

        # Generate new ID
        new_id = str(uuid4())
        new_project_dir = os.path.join(workspace.workspace_dir, new_id)
        os.makedirs(new_project_dir, exist_ok=True)

        # Copy workspace/ contents → new project dir
        src_workspace = os.path.join(tmp_dir, "workspace")
        if os.path.isdir(src_workspace):
            shutil.copytree(src_workspace, new_project_dir, dirs_exist_ok=True)

        # Resolve project name (caller override or original)
        project_name = name or project_data.get("name")

        # Archives written by older builds carry the project's skills and
        # documents in a separate "project" tree (the old ~/smartloop/<name>
        # folder).  Fold those into the new state directory so a legacy export
        # imports into the current layout.  Newer archives have them under
        # "workspace" already, copied above.
        src_legacy = os.path.join(tmp_dir, "project")
        if os.path.isdir(src_legacy):
            for sub in ("skills", "documents"):
                src_sub = os.path.join(src_legacy, sub)
                if os.path.isdir(src_sub):
                    shutil.copytree(
                        src_sub, os.path.join(new_project_dir, sub),
                        dirs_exist_ok=True,
                    )
            logger.info("[import] Migrated legacy 'project' tree into %s", new_project_dir)

        # Copy conversation history JSONL files
        src_conversations = os.path.join(tmp_dir, "conversations")
        if os.path.isdir(src_conversations):
            conv_dir = _conversations_dir()
            os.makedirs(conv_dir, exist_ok=True)
            for fname in os.listdir(src_conversations):
                if fname.endswith(".jsonl"):
                    shutil.copy2(
                        os.path.join(src_conversations, fname),
                        os.path.join(conv_dir, fname),
                    )

        # Build Project object
        from smartloop.workspace.workspace import (
            Document,
            MCPServer,
            Project,
        )

        documents = [
            Document(**d)
            for d in project_data.get("documents", [])
        ]

        # Read skills from the state directory we just assembled — it holds
        # both the modern "workspace" payload and any folded-in legacy tree.
        skills = _read_skills_from_dir(os.path.join(new_project_dir, "skills"))

        mcp_servers = [MCPServer(**m) for m in project_data.get("mcp_servers", [])]

        # Strip OAuth from imported MCP servers
        for srv in mcp_servers:
            srv.oauth_credentials = None
            if srv.remote_config:
                srv.remote_config.token = None

        new_project = Project(
            id=new_id,
            project_dir=new_project_dir,
            current=True,
            name=project_name,
            # ``cwd`` is a machine-local working folder, not project state —
            # a path from the exporting machine is meaningless here, so an
            # imported project starts without one until the user picks it.
            cwd=None,
            description=project_data.get("description"),
            system=project_data.get("system", False),
            documents=documents,
            skills=skills,
            mcp_servers=mcp_servers,
            sessions=project_data.get("sessions", []),
        )

        # Deactivate other projects
        for p in workspace.projects:
            p.current = False

        workspace.projects.append(new_project)
        # Snapshot the workspace model defaults, same as create_project does —
        # an archive carries no model_settings, so without this the project's
        # defaults would be decided lazily at its first models read.
        workspace.sync_project_models(new_project)
        workspace.save()
        Workspace.reset()

        return {
            "id": new_project.id,
            "name": new_project.name,
            "cwd": new_project.work_dir or "",
            "description": new_project.description,
            "current": new_project.current,
            "system": new_project.system,
            "skills": [s.model_dump(mode="json") for s in new_project.get_skills()],
            "mcp": [m.model_dump(mode="json") for m in new_project.mcp_servers],
            "documents": [
                {"id": d.id, "path": d.path, "status": d.status.value, "summary": d.summary}
                for d in new_project.documents
            ],
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
