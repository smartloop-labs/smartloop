"""End-to-end tests for project export/import."""

import json
import os
import shutil
import zipfile
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from smartloop.config.app_settings import AppSettings
from smartloop.workspace.workspace import (
    Document,
    DocumentStatus,
    MCPOAuthCredentials,
    MCPServer,
    MCPServerConfig,
    Project,
    RemoteMCPServerConfig,
    Skill,
    Workspace,
    WorkspaceModel,
)


@pytest.fixture()
def workspace_env(tmp_path):
    """Set up isolated workspace directories and seed a config.json."""
    home_dir = tmp_path / "dot_smartloop"
    work_dir = tmp_path / "usercode"  # the user's own folder (no framework state)
    home_dir.mkdir()
    work_dir.mkdir()

    workspace_id = str(uuid4())
    workspace_dir = home_dir / workspace_id
    workspace_dir.mkdir()

    # Create a project with skills, documents, and MCP servers
    project_id = str(uuid4())
    project_dir = workspace_dir / project_id
    project_dir.mkdir()

    # Put a sample document file in the project dir
    (project_dir / "docs").mkdir()
    (project_dir / "docs" / "readme.txt").write_text("hello world")

    # Skills live under project_dir, keyed by project id — not by name and
    # not under any client-supplied path.
    project_name = "my-test-project"
    skill_dir = project_dir / "skills" / "greet"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: greet\n---\nSay hello")

    # Create conversation history
    conv_dir = home_dir / "conversations"
    conv_dir.mkdir()
    session_id = str(uuid4())
    (conv_dir / f"{session_id}.jsonl").write_text(
        '{"role":"user","content":"Hello"}\n'
        '{"role":"assistant","content":"Hi there!"}\n'
    )

    project = Project(
        id=project_id,
        project_dir=str(project_dir),
        current=True,
        name=project_name,
        description="A test project",
        documents=[
            Document(id="doc1", path="/some/path.pdf", status=DocumentStatus.INDEXED, summary="A doc"),
        ],
        skills=[
            Skill(id="sk1", name="greet", description="Greet skill", content="Say hello"),
        ],
        mcp_servers=[
            MCPServer(
                id="mcp1",
                name="local-server",
                config=MCPServerConfig(command="npx", args=["@test/server"]),
            ),
            MCPServer(
                id="mcp2",
                name="remote-server",
                server_type="remote",
                remote_config=RemoteMCPServerConfig(
                    server_url="https://mcp.example.com",
                    auth_type="oauth",
                    token="secret-token",
                ),
                oauth_credentials=MCPOAuthCredentials(
                    client_id="cid",
                    client_secret="csecret",
                    access_token="at",
                    refresh_token="rt",
                ),
            ),
        ],
        sessions=[session_id],
    )

    workspace = Workspace(
        id=workspace_id,
        workspace_dir=str(workspace_dir),
        models_dir=str(workspace_dir / "models"),
        projects=[project],
    )

    config_file = home_dir / "config.json"
    ws_data = workspace.model_dump(mode="json")
    ws_data.pop("current_project", None)
    data = {"id": "", "workspace": ws_data}
    config_file.write_text(json.dumps(data, indent=2))

    return {
        "home_dir": str(home_dir),
        "work_dir": str(work_dir),
        "workspace": workspace,
        "project": project,
        "session_id": session_id,
        "config_file": str(config_file),
    }


def _patch_settings(env):
    """Return a stack of patches pointing the framework at the temp home."""
    settings = AppSettings()
    settings.slp_home = env["home_dir"]

    # Need to patch _settings in both project_io and workspace modules
    # so that Workspace.load() re-resolves paths against the temp dir.
    patches = [
        patch("smartloop.api.services.project_io._settings", settings),
        patch("smartloop.workspace.workspace._settings", settings),
        patch.object(Workspace, "config_file", env["config_file"]),
    ]
    return patches


class TestExportProject:
    def test_export_creates_valid_zip(self, workspace_env):
        """Export produces a zip with manifest.json, workspace/, and conversations/."""
        from smartloop.api.services.project_io import export_project

        env = workspace_env
        project = env["project"]

        patches = _patch_settings(env)
        for p in patches:
            p.start()

        try:
            zip_path = export_project(project.id)
            assert zip_path.exists()

            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()

                # manifest.json present
                assert "manifest.json" in names

                # workspace files present
                assert any(n.startswith("workspace/") for n in names)
                assert "workspace/docs/readme.txt" in names

                # Skills ride inside the state tree, not a separate one.
                assert "workspace/skills/greet/SKILL.md" in names
                assert not any(n.startswith("project/") for n in names)

                # conversation history present
                conv_name = f"conversations/{env['session_id']}.jsonl"
                assert conv_name in names
                conv_content = zf.read(conv_name).decode()
                assert '"Hello"' in conv_content
                assert '"Hi there!"' in conv_content

                # Check manifest content
                manifest = json.loads(zf.read("manifest.json"))
                assert manifest["version"] == "1.0"
                assert "exported_at" in manifest
                assert manifest["project"]["id"] == project.id
                assert manifest["project"]["name"] == "my-test-project"
                # Skills are exported as files, not in the manifest
                assert "skills" not in manifest["project"]
                assert len(manifest["project"]["documents"]) == 1
                assert env["session_id"] in manifest["project"]["sessions"]

                # OAuth credentials stripped
                mcp_servers = manifest["project"]["mcp_servers"]
                remote_srv = next(s for s in mcp_servers if s["name"] == "remote-server")
                assert remote_srv.get("oauth_credentials") is None
                assert remote_srv["remote_config"].get("token") is None

            os.unlink(zip_path)
        finally:
            for p in patches:
                p.stop()

    def test_export_nonexistent_project_raises(self, workspace_env):
        """Exporting a non-existent project raises FileNotFoundError."""
        from smartloop.api.services.project_io import export_project

        env = workspace_env
        patches = _patch_settings(env)
        for p in patches:
            p.start()
        try:
            with pytest.raises(FileNotFoundError):
                export_project("nonexistent-id")
        finally:
            for p in patches:
                p.stop()


class TestImportProject:
    def test_import_creates_project(self, workspace_env):
        """Import restores project metadata, files, and conversations."""
        from smartloop.api.services.project_io import export_project, import_project

        env = workspace_env
        patches = _patch_settings(env)
        for p in patches:
            p.start()

        try:
            zip_path = export_project(env["project"].id)

            result = import_project(zip_path)

            # New ID generated
            assert result["id"] != env["project"].id
            assert result["name"] == "my-test-project"
            # cwd is machine-local, so an imported project starts without one.
            assert result["cwd"] == ""
            assert result["description"] == "A test project"
            assert result["current"] is True

            # Documents preserve original status (vector DB is copied with project)
            assert len(result["documents"]) == 1
            assert result["documents"][0]["status"] == "indexed"

            # Skills preserved
            assert len(result["skills"]) == 1
            assert result["skills"][0]["name"] == "greet"

            # Workspace files copied
            new_project_dir = os.path.join(
                env["workspace"].workspace_dir, result["id"]
            )
            assert os.path.isfile(os.path.join(new_project_dir, "docs", "readme.txt"))

            # Skills landed in the new project's state dir.
            assert os.path.isfile(
                os.path.join(new_project_dir, "skills", "greet", "SKILL.md")
            )

            # Conversation history restored
            conv_file = os.path.join(
                env["home_dir"], "conversations", f"{env['session_id']}.jsonl"
            )
            assert os.path.isfile(conv_file)

            # Config updated
            with open(env["config_file"]) as f:
                config = json.load(f)
            ws = config["workspace"]
            assert any(p["id"] == result["id"] for p in ws["projects"])

            os.unlink(zip_path)
        finally:
            for p in patches:
                p.stop()

    def test_import_with_rename(self, workspace_env):
        """Import with a name override renames the project."""
        from smartloop.api.services.project_io import export_project, import_project

        env = workspace_env
        patches = _patch_settings(env)
        for p in patches:
            p.start()

        try:
            zip_path = export_project(env["project"].id)

            result = import_project(zip_path, name="renamed-project")

            assert result["name"] == "renamed-project"

            # Renaming changes the label only — state stays keyed by project id.
            new_project_dir = os.path.join(
                env["workspace"].workspace_dir, result["id"]
            )
            assert os.path.isfile(
                os.path.join(new_project_dir, "skills", "greet", "SKILL.md")
            )
            assert result["skills"][0]["name"] == "greet"

            os.unlink(zip_path)
        finally:
            for p in patches:
                p.stop()

    def test_import_corrupt_zip_raises(self, workspace_env):
        """Importing a corrupt zip raises ValueError."""
        import tempfile

        from smartloop.api.services.project_io import import_project

        env = workspace_env
        patches = _patch_settings(env)
        for p in patches:
            p.start()
        try:
            bad = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
            bad.write(b"not a zip file")
            bad.close()
            with pytest.raises(ValueError, match="not a valid zip"):
                import_project(bad.name)
            os.unlink(bad.name)
        finally:
            for p in patches:
                p.stop()

    def test_import_missing_manifest_raises(self, workspace_env):
        """Importing a zip without manifest.json raises ValueError."""
        import tempfile

        from smartloop.api.services.project_io import import_project

        env = workspace_env
        patches = _patch_settings(env)
        for p in patches:
            p.start()
        try:
            bad = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
            with zipfile.ZipFile(bad.name, "w") as zf:
                zf.writestr("random.txt", "hello")
            with pytest.raises(ValueError, match="missing manifest"):
                import_project(bad.name)
            os.unlink(bad.name)
        finally:
            for p in patches:
                p.stop()


class TestImportingIntoTheCallersWorkspace:
    """Where an import lands when the caller already holds a workspace.

    ``import_project`` used to build its own ``Workspace.load()``, save, and
    return.  The caller's instance — the one the API server serves every
    request from — never learned the project existed.  Every later read
    resolved it to whichever project was selected, and the first save from
    that instance wrote a project list the imported one was missing from,
    taking its ``config.json`` entry with it while the directory, the
    documents and the index stayed on disk.
    """

    def test_the_project_lands_in_the_instance_the_caller_passed(self, workspace_env):
        from smartloop.api.services.project_io import export_project, import_project

        env = workspace_env
        patches = _patch_settings(env)
        for p in patches:
            p.start()
        try:
            live = Workspace.load()
            assert len(live.projects) == 1

            result = import_project(export_project(env["project"].id), workspace=live)

            assert [p.id for p in live.projects if p.id == result["id"]], (
                "the imported project is missing from the instance it was imported into"
            )
            assert len(live.projects) == 2
        finally:
            for p in patches:
                p.stop()

    def test_a_later_save_keeps_the_imported_project(self, workspace_env):
        """The regression: the caller saves, and the import survives it.

        This is the failure as it actually happened — nothing here touches the
        import again.  The caller simply persists something unrelated (a
        selection change) afterwards, which is enough to drop a project the
        instance was never told about.
        """
        from smartloop.api.services.project_io import export_project, import_project

        env = workspace_env
        patches = _patch_settings(env)
        for p in patches:
            p.start()
        try:
            live = Workspace.load()
            imported = import_project(export_project(env["project"].id), workspace=live)

            # Something unrelated moves the selection and persists it.
            live.projects[0].current = True
            live.save()

            on_disk = json.loads(Path(env["config_file"]).read_text())
            ids = [p["id"] for p in on_disk["workspace"]["projects"]]
            assert imported["id"] in ids, (
                "a save from the caller's workspace erased the imported project"
            )
            assert len(ids) == 2
        finally:
            for p in patches:
                p.stop()

    def test_no_workspace_still_loads_one(self, workspace_env):
        """The CLI holds no instance, so the default is unchanged."""
        from smartloop.api.services.project_io import export_project, import_project

        env = workspace_env
        patches = _patch_settings(env)
        for p in patches:
            p.start()
        try:
            result = import_project(export_project(env["project"].id))
            on_disk = json.loads(Path(env["config_file"]).read_text())
            assert result["id"] in [
                p["id"] for p in on_disk["workspace"]["projects"]
            ]
        finally:
            for p in patches:
                p.stop()


class TestExportImportRoundTrip:
    def test_full_cycle(self, workspace_env):
        """Export → import produces a new project with matching data but new ID."""
        from smartloop.api.services.project_io import export_project, import_project

        env = workspace_env
        original = env["project"]
        patches = _patch_settings(env)
        for p in patches:
            p.start()

        try:
            zip_path = export_project(original.id)
            result = import_project(zip_path)

            assert result["id"] != original.id
            assert result["name"] == original.name
            # cwd does not survive a round trip — it is a machine-local
            # working folder, not project state.
            assert result["cwd"] == ""
            assert result["description"] == original.description
            assert len(result["skills"]) == len(original.skills)
            assert len(result["documents"]) == len(original.documents)

            # Document statuses preserved (vector DB copied with project)
            for doc in result["documents"]:
                assert doc["status"] == "indexed"

            os.unlink(zip_path)
        finally:
            for p in patches:
                p.stop()


class TestLegacyArchiveImport:
    """Archives from builds that stored skills in the user-space folder.

    Those zips carry a separate ``project/`` tree alongside ``workspace/``.
    Export no longer writes one, but import must still fold it into the new
    state directory or every previously-exported project loses its skills.
    """

    def test_legacy_project_tree_is_folded_into_state_dir(self, workspace_env):
        from smartloop.api.services.project_io import export_project, import_project

        env = workspace_env
        patches = _patch_settings(env)
        for p in patches:
            p.start()

        try:
            zip_path = export_project(env["project"].id)

            # Rewrite the archive into the old shape: skills under project/.
            legacy = zip_path.with_name("legacy.zip")
            with zipfile.ZipFile(zip_path) as src, zipfile.ZipFile(legacy, "w") as dst:
                for item in src.namelist():
                    data = src.read(item)
                    if item == "workspace/skills/greet/SKILL.md":
                        dst.writestr("project/skills/greet/SKILL.md", data)
                    else:
                        dst.writestr(item, data)

            result = import_project(legacy)

            assert len(result["skills"]) == 1
            assert result["skills"][0]["name"] == "greet"
            new_project_dir = os.path.join(
                env["workspace"].workspace_dir, result["id"]
            )
            assert os.path.isfile(
                os.path.join(new_project_dir, "skills", "greet", "SKILL.md")
            )

            os.unlink(zip_path)
            os.unlink(legacy)
        finally:
            for p in patches:
                p.stop()
