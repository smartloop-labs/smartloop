"""A project's state lives in the project's directory, not in one shared file.

``config.json`` used to hold every project. That made a save a whole-workspace
snapshot: two writers each holding a ``Workspace`` would have the last one
silently discard the other project's newly indexed documents and MCP
registrations — the write-clobber hazard that had to go before two processes
could run side by side.
"""

import json
import os

import pytest

from smartloop.workspace import migrations
from smartloop.workspace.context import pin_project
from smartloop.workspace.migrations import (
    MONOLITH_BACKUP_SUFFIX, split_project_state,
)
from smartloop.workspace.workspace import (
    MCPServer, Project, Workspace, project_state_path, read_project_state,
)


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Point the framework's home at a temp dir, resolved the way it is live."""
    monkeypatch.setenv("SLP_HOME", str(tmp_path))
    monkeypatch.delenv("SLP_LEGACY_CONFIG", raising=False)
    Workspace.reset()
    yield tmp_path
    pin_project(None)
    Workspace.reset()


def _workspace(home, project_ids=("alpha", "beta")) -> Workspace:
    ws_dir = os.path.join(str(home), "ws-1")
    ws = Workspace(id="ws-1", workspace_dir=ws_dir,
                   models_dir=os.path.join(ws_dir, "models"))
    ws.projects = [
        Project(
            id=pid,
            project_dir=os.path.join(ws_dir, pid),
            current=(pid == project_ids[0]),
            name=pid.title(),
            sessions=[f"{pid}-session"],
            mcp_servers=[MCPServer(id=f"{pid}-srv", name=f"{pid} server")],
        )
        for pid in project_ids
    ]
    return ws


def _config(home) -> dict:
    return json.loads((home / "config.json").read_text())


class TestWhatEndsUpWhere:
    def test_config_json_keeps_only_a_thin_index(self, home):
        _workspace(home).save()

        entry = _config(home)["workspace"]["projects"][0]
        assert entry["id"] == "alpha"
        assert entry["current"] is True
        assert entry["project_dir"].endswith(os.path.join("ws-1", "alpha"))
        for bulk in ("documents", "skills", "mcp_servers", "sessions",
                     "model_settings", "web_search_enabled"):
            assert bulk not in entry

    def test_the_project_keeps_its_own_full_record(self, home):
        _workspace(home).save()

        state = read_project_state(os.path.join(str(home), "ws-1", "alpha"))
        assert state["mcp_servers"][0]["name"] == "alpha server"
        assert state["sessions"] == ["alpha-session"]
        # The whole record, so the directory is self-describing.
        assert state["id"] == "alpha"
        assert state["name"] == "Alpha"

    def test_the_file_is_stamped_with_a_version(self, home):
        _workspace(home).save()

        raw = json.loads(open(project_state_path(
            os.path.join(str(home), "ws-1", "alpha"))).read())
        assert raw["version"] == 1

    def test_workspace_level_state_stays_in_config_json(self, home):
        ws = _workspace(home)
        ws.add_model("sl-mini", "smartloop-ai/sl-mini")
        ws.save()

        saved = _config(home)["workspace"]
        assert saved["id"] == "ws-1"
        assert [m["model_name"] for m in saved["models"]] == ["sl-mini"]


class TestItComesBackTheSame:
    def test_a_saved_workspace_round_trips(self, home):
        _workspace(home).save()

        reloaded = Workspace.load()

        assert [p.id for p in reloaded.projects] == ["alpha", "beta"]
        alpha = reloaded.projects[0]
        assert alpha.name == "Alpha"
        assert alpha.sessions == ["alpha-session"]
        assert alpha.mcp_servers[0].name == "alpha server"
        assert reloaded.current_project.id == "alpha"

    def test_the_selection_is_the_workspace_s_to_own(self, home):
        """project.json carries ``current`` too; config.json must still win."""
        ws = _workspace(home)
        ws.save()

        # Move the selection and rewrite only the workspace record, the way a
        # supervisor would.
        for project in ws.projects:
            project.current = project.id == "beta"
        ws.save()

        assert Workspace.load().current_project.id == "beta"

    def test_a_project_with_no_file_of_its_own_still_loads(self, home):
        """A half-migrated workspace must not lose the projects not yet split."""
        _workspace(home).save()
        os.remove(project_state_path(os.path.join(str(home), "ws-1", "beta")))

        reloaded = Workspace.load()

        assert [p.id for p in reloaded.projects] == ["alpha", "beta"]
        assert reloaded.projects[0].sessions == ["alpha-session"]
        assert reloaded.projects[1].name == "Beta"  # from the thin index

    def test_a_corrupt_project_file_does_not_take_the_workspace_down(self, home):
        _workspace(home).save()
        path = project_state_path(os.path.join(str(home), "ws-1", "beta"))
        with open(path, "w") as fh:
            fh.write("{not json")

        reloaded = Workspace.load()

        assert [p.id for p in reloaded.projects] == ["alpha", "beta"]
        assert reloaded.projects[0].mcp_servers[0].name == "alpha server"


class TestOneWriterPerProject:
    """The clobber the split exists to remove."""

    def test_two_workspaces_saving_do_not_discard_each_other_s_project(self, home):
        _workspace(home).save()

        # Two processes, each with its own full copy of the workspace — the
        # exact shape that used to lose data.
        one, two = Workspace.load(), Workspace.load()
        one.projects[0].sessions.append("added-by-one")
        two.projects[1].sessions.append("added-by-two")

        pin_project("alpha")
        one.save()
        pin_project("beta")
        two.save()
        pin_project(None)

        reloaded = Workspace.load()
        assert reloaded.projects[0].sessions == ["alpha-session", "added-by-one"]
        assert reloaded.projects[1].sessions == ["beta-session", "added-by-two"]

    def test_a_pinned_process_writes_only_its_own_project(self, home):
        _workspace(home).save()
        before = (home / "config.json").read_text()

        ws = Workspace.load()
        pin_project("beta")
        ws.projects[0].sessions.append("should-not-persist")
        ws.projects[1].sessions.append("should-persist")
        ws.save()

        assert (home / "config.json").read_text() == before  # untouched
        alpha = read_project_state(os.path.join(str(home), "ws-1", "alpha"))
        beta = read_project_state(os.path.join(str(home), "ws-1", "beta"))
        assert alpha["sessions"] == ["alpha-session"]
        assert beta["sessions"] == ["beta-session", "should-persist"]

    def test_a_pin_to_an_unknown_project_writes_nothing(self, home, caplog):
        _workspace(home).save()
        before = (home / "config.json").read_text()

        ws = Workspace.load()
        pin_project("not-a-project")
        ws.save()

        assert (home / "config.json").read_text() == before
        assert "not in this workspace" in caplog.text


class TestMigratingOffTheMonolith:
    def _write_monolith(self, home) -> Workspace:
        ws = _workspace(home)
        dumped = ws.model_dump(mode="json")
        dumped.pop("current_project", None)
        (home / "config.json").write_text(
            json.dumps({"id": "", "workspace": dumped}, indent=2)
        )
        return ws

    def test_it_splits_an_existing_monolith(self, home):
        self._write_monolith(home)
        ws = Workspace.load()

        result = split_project_state(ws)

        assert sorted(result["migrated"]) == ["alpha", "beta"]
        assert read_project_state(
            os.path.join(str(home), "ws-1", "alpha")
        )["mcp_servers"][0]["name"] == "alpha server"
        assert "mcp_servers" not in _config(home)["workspace"]["projects"][0]

    def test_nothing_is_lost_in_the_split(self, home):
        self._write_monolith(home)
        before = Workspace.load()

        split_project_state(before)
        Workspace.reset()
        after = Workspace.load()

        assert [p.model_dump(mode="json") for p in after.projects] == \
               [p.model_dump(mode="json") for p in before.projects]

    def test_the_pre_split_config_is_kept_for_a_rollback(self, home):
        self._write_monolith(home)
        original = (home / "config.json").read_text()

        result = split_project_state(Workspace.load())

        assert result["backup"] == str(home / "config.json") + MONOLITH_BACKUP_SUFFIX
        assert open(result["backup"]).read() == original

    def test_a_second_run_changes_nothing(self, home):
        self._write_monolith(home)
        split_project_state(Workspace.load())
        Workspace.reset()

        result = split_project_state(Workspace.load())

        assert result["migrated"] == []

    def test_a_second_run_does_not_overwrite_the_backup(self, home):
        self._write_monolith(home)
        original = (home / "config.json").read_text()
        split_project_state(Workspace.load())
        Workspace.reset()

        # Force another pass by removing one project's file.
        os.remove(project_state_path(os.path.join(str(home), "ws-1", "beta")))
        split_project_state(Workspace.load())

        backup = str(home / "config.json") + MONOLITH_BACKUP_SUFFIX
        assert open(backup).read() == original

    def test_an_empty_workspace_is_not_an_error(self, home):
        ws = Workspace(id="ws-1", workspace_dir=str(home / "ws-1"))
        assert split_project_state(ws) == {"migrated": [], "backup": None}

    def test_no_workspace_is_not_an_error(self):
        assert split_project_state(None) == {"migrated": [], "backup": None}


class TestTheRollbackSwitch:
    def test_legacy_mode_writes_the_single_document_again(self, home, monkeypatch):
        monkeypatch.setenv("SLP_LEGACY_CONFIG", "1")

        _workspace(home).save()

        entry = _config(home)["workspace"]["projects"][0]
        assert entry["mcp_servers"][0]["name"] == "alpha server"
        assert not os.path.exists(
            project_state_path(os.path.join(str(home), "ws-1", "alpha"))
        )

    def test_legacy_mode_ignores_project_files_already_written(self, home, monkeypatch):
        _workspace(home).save()  # split layout on disk
        # Restore the pre-split document the way a rollback would.
        ws = _workspace(home)
        dumped = ws.model_dump(mode="json")
        dumped.pop("current_project", None)
        dumped["projects"][0]["sessions"] = ["from-the-backup"]
        (home / "config.json").write_text(json.dumps({"id": "", "workspace": dumped}))

        monkeypatch.setenv("SLP_LEGACY_CONFIG", "1")
        reloaded = Workspace.load()

        assert reloaded.projects[0].sessions == ["from-the-backup"]

    def test_legacy_mode_skips_the_migration_entirely(self, home, monkeypatch):
        monkeypatch.setenv("SLP_LEGACY_CONFIG", "1")
        ws = _workspace(home)

        assert split_project_state(ws)["migrated"] == []


class TestPathsResolveLate:
    def test_config_file_follows_the_home_it_is_asked_in(self, home, monkeypatch, tmp_path):
        assert Workspace.config_file == str(home / "config.json")

        other = tmp_path / "elsewhere"
        monkeypatch.setenv("SLP_HOME", str(other))
        assert Workspace.config_file == str(other / "config.json")

    def test_an_explicit_override_wins(self, home, monkeypatch, tmp_path):
        monkeypatch.setenv("SLP_CONFIG_FILE", str(tmp_path / "pinned.json"))
        assert Workspace.config_file == str(tmp_path / "pinned.json")

    def test_it_is_still_patchable(self, home, monkeypatch):
        """Tests override it by assignment; the descriptor must not block that."""
        monkeypatch.setattr(Workspace, "config_file", "/somewhere/else.json")
        assert Workspace.config_file == "/somewhere/else.json"


class TestNeedsSplitDetection:
    def test_a_project_without_a_file_needs_one(self, home):
        assert migrations._needs_split(_workspace(home)) is True

    def test_a_fully_split_workspace_does_not(self, home):
        ws = _workspace(home)
        ws.save()
        assert migrations._needs_split(ws) is False

    def test_a_project_with_no_directory_is_skipped(self, home):
        ws = _workspace(home)
        for project in ws.projects:
            project.project_dir = ""
        assert migrations._needs_split(ws) is False
