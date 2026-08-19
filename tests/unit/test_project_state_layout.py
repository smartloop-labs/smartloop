"""Skills and documents live under project_dir, never a client-supplied path.

The old layout anchored them to ``project.cwd`` (defaulting to
``~/smartloop/<name>``), an absolute user-space path.  That path does not
follow the framework's home, so under the macOS App Sandbox — where ``~`` is
redirected to the app container — the app could read one workspace and write
another.  ``project_dir`` derives from ``AppSettings.home_dir``, so it always
travels with the framework.
"""

import os

import pytest

from smartloop.workspace.migrations import migrate_project_state
from smartloop.workspace.workspace import Project


def _project(tmp_path, *, cwd=None, name="proj"):
    state = tmp_path / "state" / "p1"
    state.mkdir(parents=True, exist_ok=True)
    return Project(
        id="p1", name=name, project_dir=str(state), current=True,
        cwd=str(cwd) if cwd else None,
    )


def _write_skill(skills_dir, name, body="BODY"):
    d = os.path.join(str(skills_dir), name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "SKILL.md"), "w") as fh:
        fh.write(f"---\nname: {name}\ndescription: d-{name}\n---\n{body}")


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------

def test_state_dirs_hang_off_project_dir(tmp_path):
    p = _project(tmp_path, cwd=tmp_path / "usercode")
    assert p.skills_dir == os.path.join(p.project_dir, "skills")
    assert p.documents_dir == os.path.join(p.project_dir, "documents")


def test_state_dirs_ignore_cwd_entirely(tmp_path):
    """A client-supplied cwd must not influence where state is stored."""
    a = _project(tmp_path, cwd=tmp_path / "one")
    b = _project(tmp_path, cwd=tmp_path / "two")
    assert a.skills_dir == b.skills_dir
    assert a.documents_dir == b.documents_dir


def test_state_dirs_survive_a_missing_cwd(tmp_path):
    """A project with no working folder is still fully usable."""
    p = _project(tmp_path, cwd=None)
    assert p.work_dir is None
    assert p.skills_dir.startswith(p.project_dir)


def test_work_dir_is_the_users_folder(tmp_path):
    p = _project(tmp_path, cwd=tmp_path / "usercode")
    assert p.work_dir == str(tmp_path / "usercode")


def test_work_dir_expands_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    p = Project(id="p", name="n", project_dir=str(tmp_path), current=True,
                cwd="~/repo")
    assert p.work_dir == str(tmp_path / "home" / "repo")


def test_visible_dir_concept_is_gone(tmp_path):
    p = _project(tmp_path)
    assert not hasattr(p, "visible_dir")
    assert not hasattr(p, "folder_name")


def test_get_skills_reads_from_project_dir(tmp_path):
    p = _project(tmp_path, cwd=tmp_path / "usercode")
    _write_skill(p.skills_dir, "greet")
    # A stale copy in the old user-space location must be ignored.
    os.makedirs(tmp_path / "usercode", exist_ok=True)
    _write_skill(tmp_path / "usercode" / "skills", "stale")

    assert [s.name for s in p.get_skills()] == ["greet"]


# --------------------------------------------------------------------------
# migration
# --------------------------------------------------------------------------

class _WS:
    def __init__(self, projects):
        self.projects = projects


def test_migration_moves_legacy_skills_and_documents(tmp_path):
    legacy = tmp_path / "usercode"
    p = _project(tmp_path, cwd=legacy)
    _write_skill(legacy / "skills", "greet")
    os.makedirs(legacy / "documents", exist_ok=True)
    (legacy / "documents" / "notes.txt").write_text("hello")

    moved = migrate_project_state(_WS([p]))

    assert moved["p1"] == {"skills": 1, "documents": 1}
    assert [s.name for s in p.get_skills()] == ["greet"]
    assert os.path.isfile(os.path.join(p.documents_dir, "notes.txt"))


def test_migration_is_idempotent(tmp_path):
    legacy = tmp_path / "usercode"
    p = _project(tmp_path, cwd=legacy)
    _write_skill(legacy / "skills", "greet")

    first = migrate_project_state(_WS([p]))
    second = migrate_project_state(_WS([p]))

    assert first["p1"]["skills"] == 1
    assert second == {}


def test_migration_never_clobbers_newer_state(tmp_path):
    """An already-migrated skill wins over the stale legacy copy."""
    legacy = tmp_path / "usercode"
    p = _project(tmp_path, cwd=legacy)
    _write_skill(p.skills_dir, "greet", body="NEW")
    _write_skill(legacy / "skills", "greet", body="OLD")

    migrate_project_state(_WS([p]))

    assert "NEW" in p.get_skills()[0].content


def test_migration_leaves_the_source_in_place(tmp_path):
    """Recoverable by hand if the move goes wrong."""
    legacy = tmp_path / "usercode"
    p = _project(tmp_path, cwd=legacy)
    _write_skill(legacy / "skills", "greet")

    migrate_project_state(_WS([p]))

    assert os.path.isfile(legacy / "skills" / "greet" / "SKILL.md")


def test_migration_picks_up_legacy_name_based_folder(tmp_path, monkeypatch):
    """Projects predating persisted cwd lived at ~/smartloop/<name>."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    p = _project(tmp_path, cwd=None, name="general_chat")
    _write_skill(home / "smartloop" / "general_chat" / "skills", "greet")

    moved = migrate_project_state(_WS([p]))

    assert moved["p1"]["skills"] == 1
    assert [s.name for s in p.get_skills()] == ["greet"]


def test_migration_survives_an_unreadable_legacy_path(tmp_path):
    """The sandbox case: the old path exists but is denied. Must not raise."""
    legacy = tmp_path / "denied"
    legacy.mkdir()
    _write_skill(legacy / "skills", "greet")
    p = _project(tmp_path, cwd=legacy)
    os.chmod(legacy, 0o000)
    try:
        assert migrate_project_state(_WS([p])) == {}
    finally:
        os.chmod(legacy, 0o755)


def test_migration_handles_a_missing_legacy_path(tmp_path):
    p = _project(tmp_path, cwd=tmp_path / "does-not-exist")
    assert migrate_project_state(_WS([p])) == {}


def test_migration_tolerates_no_workspace():
    assert migrate_project_state(None) == {}


def test_migration_skips_project_without_state_dir(tmp_path):
    p = Project(id="x", name="x", project_dir="", current=True,
                cwd=str(tmp_path))
    assert migrate_project_state(_WS([p])) == {}


# --------------------------------------------------------------------------
# ingest destination
# --------------------------------------------------------------------------

class TestDocumentIngestDestination:
    """Documents must be *created* under project_dir, not created then moved.

    The regression: ``add_document`` preferred ``<cwd>/documents`` and only
    fell back to ``documents_dir``, so an uploaded file landed in the user's
    folder — outside the framework home, where it cannot follow SLP_HOME or
    the App Sandbox container.
    """

    @staticmethod
    def _model(tmp_path, cwd):
        from unittest.mock import MagicMock
        from smartloop.model.generic import GenericModel

        class _Concrete(GenericModel):
            """GenericModel is abstract; only the ingest path is under test."""
            def model_id(self): return "test/model"
            def model_type(self): return "test"

        m = object.__new__(_Concrete)
        m.logger = MagicMock()
        state = tmp_path / "state" / "p1"
        state.mkdir(parents=True, exist_ok=True)
        project = Project(
            id="p1", name="proj", project_dir=str(state), current=True,
            cwd=str(cwd) if cwd else None,
        )
        m.workspace = MagicMock()
        m.workspace.current_project = project
        return m, project

    def test_ingest_writes_under_project_dir(self, tmp_path):
        src = tmp_path / "Invoice.pdf"
        src.write_bytes(b"%PDF-1.4 fake")
        user_folder = tmp_path / "usercode"
        user_folder.mkdir()
        m, project = self._model(tmp_path, user_folder)

        m.add_document(src, cwd=str(user_folder))

        assert os.path.isfile(os.path.join(project.documents_dir, "Invoice.pdf"))

    def test_ingest_never_touches_the_user_folder(self, tmp_path):
        src = tmp_path / "Invoice.pdf"
        src.write_bytes(b"%PDF-1.4 fake")
        user_folder = tmp_path / "usercode"
        user_folder.mkdir()
        m, _ = self._model(tmp_path, user_folder)

        m.add_document(src, cwd=str(user_folder))

        assert not os.path.exists(user_folder / "documents")

    def test_request_cwd_cannot_redirect_storage(self, tmp_path):
        """Even an explicit cwd argument must not move project state."""
        src = tmp_path / "Invoice.pdf"
        src.write_bytes(b"%PDF-1.4 fake")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        m, project = self._model(tmp_path, None)

        m.add_document(src, cwd=str(elsewhere))

        assert os.path.isfile(os.path.join(project.documents_dir, "Invoice.pdf"))
        assert not os.path.exists(elsewhere / "documents")


# --------------------------------------------------------------------------
# self-containment: copy <home_dir> anywhere and it runs as-is
# --------------------------------------------------------------------------

class TestRelocatableFramework:
    """config.json stores absolute paths; none of them may be trusted.

    Copy the framework home to another machine, user, or sandbox container and
    every persisted path is stale.  Workspace.load must re-derive them.
    """

    @staticmethod
    def _seed(home, *, doc_name="Invoice.pdf"):
        """Write a config.json describing one project, plus its files on disk."""
        import json
        ws_id, proj_id = "ws-1", "proj-1"
        proj_dir = home / ws_id / proj_id
        (proj_dir / "skills" / "greet").mkdir(parents=True)
        (proj_dir / "skills" / "greet" / "SKILL.md").write_text(
            "---\nname: greet\n---\nSay hello"
        )
        (proj_dir / "documents").mkdir(parents=True)
        (proj_dir / "documents" / doc_name).write_text("data")
        cfg = {
            "id": "",
            "workspace": {
                "id": ws_id,
                "workspace_dir": "/OLD/HOME/" + ws_id,
                "models_dir": "/OLD/HOME/" + ws_id + "/models",
                "projects": [{
                    "id": proj_id,
                    "project_dir": f"/OLD/HOME/{ws_id}/{proj_id}",
                    "current": True,
                    "name": "proj",
                    "cwd": "/OLD/USER/FOLDER/that/is/gone",
                    "documents": [{
                        "id": "d1",
                        "path": f"/OLD/HOME/{ws_id}/{proj_id}/documents/{doc_name}",
                        "name": doc_name,
                        "status": "indexed",
                    }],
                }],
            },
        }
        (home / "config.json").write_text(json.dumps(cfg))
        return proj_dir

    @staticmethod
    def _load(home):
        from unittest.mock import patch
        from smartloop.config.app_settings import AppSettings
        from smartloop.workspace.workspace import Workspace

        settings = AppSettings()
        settings.slp_home = str(home)
        with patch("smartloop.workspace.workspace._settings", settings), \
             patch.object(Workspace, "config_file", str(home / "config.json")):
            return Workspace.load()

    def test_project_dir_is_rebased_onto_the_new_home(self, tmp_path):
        home = tmp_path / "copied_home"
        home.mkdir()
        proj_dir = self._seed(home)

        ws = self._load(home)

        assert ws.projects[0].project_dir == str(proj_dir)

    def test_skills_are_found_after_relocation(self, tmp_path):
        home = tmp_path / "copied_home"
        home.mkdir()
        self._seed(home)

        ws = self._load(home)

        assert [s.name for s in ws.projects[0].get_skills()] == ["greet"]

    def test_document_paths_are_rebased(self, tmp_path):
        home = tmp_path / "copied_home"
        home.mkdir()
        proj_dir = self._seed(home)

        ws = self._load(home)

        doc = ws.projects[0].documents[0]
        assert doc.path == os.path.join(str(proj_dir), "documents", "Invoice.pdf")
        assert os.path.isfile(doc.path)

    def test_a_dead_user_cwd_does_not_break_anything(self, tmp_path):
        """The stale client path survives in the config but locates nothing."""
        home = tmp_path / "copied_home"
        home.mkdir()
        self._seed(home)

        ws = self._load(home)
        project = ws.projects[0]

        assert not os.path.exists(project.work_dir)
        assert [s.name for s in project.get_skills()] == ["greet"]
        assert project.skills_dir.startswith(str(home))
        assert project.documents_dir.startswith(str(home))

    def test_framework_copy_wins_over_a_still_valid_outside_path(self, tmp_path):
        """After a migration both copies exist; the framework's must win.

        Otherwise the pointer stays on the user-space copy and the install
        keeps depending on a directory outside <home_dir>.
        """
        import json
        home = tmp_path / "copied_home"
        home.mkdir()
        proj_dir = self._seed(home)
        external = tmp_path / "leftover" / "Invoice.pdf"
        external.parent.mkdir()
        external.write_text("stale copy")

        cfg = json.loads((home / "config.json").read_text())
        cfg["workspace"]["projects"][0]["documents"][0]["path"] = str(external)
        (home / "config.json").write_text(json.dumps(cfg))

        ws = self._load(home)

        assert ws.projects[0].documents[0].path == os.path.join(
            str(proj_dir), "documents", "Invoice.pdf"
        )

    def test_unknown_document_keeps_its_path(self, tmp_path):
        """A document with no counterpart in the project is left untouched."""
        import json
        home = tmp_path / "copied_home"
        home.mkdir()
        self._seed(home)
        external = tmp_path / "elsewhere.pdf"
        external.write_text("x")

        cfg = json.loads((home / "config.json").read_text())
        doc = cfg["workspace"]["projects"][0]["documents"][0]
        doc["path"] = str(external)
        doc["name"] = "elsewhere.pdf"
        (home / "config.json").write_text(json.dumps(cfg))

        ws = self._load(home)

        assert ws.projects[0].documents[0].path == str(external)


class TestSkillsAreReadPerProject:
    """Skills come from the project's own dir, via both readers.

    Two readers exist — ``Project.get_skills`` (chat/orchestrator) and
    ``ModelService._read_skills_from_disk`` (API/UI).  They must agree, and
    neither may leak another project's skills.
    """

    @staticmethod
    def _proj(tmp_path, pid, names):
        p = Project(id=pid, name=pid, current=False,
                    project_dir=str(tmp_path / "state" / pid),
                    cwd="/some/client/path/that/is/ignored")
        for n in names:
            d = os.path.join(p.skills_dir, n)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "SKILL.md"), "w") as fh:
                fh.write(f"---\nname: {n}\ndescription: d\n---\nbody")
        return p

    def test_each_project_sees_only_its_own(self, tmp_path):
        a = self._proj(tmp_path, "a", ["fetch-url", "good-morning"])
        b = self._proj(tmp_path, "b", ["other"])

        assert [s.name for s in a.get_skills()] == ["fetch-url", "good-morning"]
        assert [s.name for s in b.get_skills()] == ["other"]

    def test_both_readers_agree(self, tmp_path):
        from smartloop.api.services.model_service import ModelService

        p = self._proj(tmp_path, "a", ["fetch-url", "good-morning"])

        assert [s.name for s in p.get_skills()] == \
               [s.name for s in ModelService._read_skills_from_disk(p)]

    def test_a_skill_created_now_is_visible_immediately(self, tmp_path):
        """No cache anywhere — the next read picks it up."""
        from smartloop.api.services.model_service import ModelService
        from smartloop.workspace.types import Skill

        p = self._proj(tmp_path, "a", [])
        assert p.get_skills() == []

        ModelService._write_skill_md(p, Skill(id="s", name="fetch-url",
                                              description="d", content="body"))

        assert [s.name for s in p.get_skills()] == ["fetch-url"]
        assert [s.name for s in ModelService._read_skills_from_disk(p)] == ["fetch-url"]

    def test_writes_land_in_the_project_dir_not_the_client_path(self, tmp_path):
        from smartloop.api.services.model_service import ModelService
        from smartloop.workspace.types import Skill

        p = self._proj(tmp_path, "a", [])
        ModelService._write_skill_md(p, Skill(id="s", name="fetch-url", content="body"))

        assert os.path.isfile(os.path.join(p.skills_dir, "fetch-url", "SKILL.md"))
        assert not os.path.exists("/some/client/path/that/is/ignored")
