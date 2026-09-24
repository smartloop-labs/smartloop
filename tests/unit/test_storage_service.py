"""Tests for the storage usage/cleanup service.

The safety rails are the point of these tests: a bug here deletes a user's
models, credentials, or project data.  So most of what follows asserts what
must *not* happen.
"""

import json
import logging
import os
import threading
import time
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from smartloop.api.services import storage_service
from smartloop.model.downloader.inflight import downloading
from smartloop.config.app_settings import AppSettings
from smartloop.workspace.workspace import Project, ProjectModel, Workspace, WorkspaceModel

# Mirrors the real config: only the bootstrap orchestrator is is_active, while
# the models the user picked carry enabled=True.  Keying the keep-rule on
# is_active alone would delete the picked ones.  The bootstrap model is recorded
# under its canonical name while its model_id keeps the pre-rename spelling —
# renaming the directory would orphan already-downloaded weights.
MODELS = [
    ("sl-mini", "smartloop-ai/slp-orchestra-mini", True, True),
    ("gemma4-e4b", "google/gemma-4-e4b-it", True, False),
    ("gemma3-4b", "google/gemma-3-4b-it", False, False),
    ("ministral3-3b", "mistralai/Ministral-3-3B-Instruct-2512", False, False),
]

REMOVABLE_IDS = {"google/gemma-3-4b-it", "mistralai/Ministral-3-3B-Instruct-2512"}
CURRENT_VERSION = "1.1.0"
OLD_VERSION = "1.0.9"

# Files that must survive every category, in every combination.
PROTECTED_FILES = (
    "config.json", ".credentials", ".key", ".installed-version", ".binary-hash",
    ".archive-etag", "server.pid", "server.port", "version", "installed",
    "studio.db", "token_usage.jsonl",
)


def _write(path, content="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return path


def _age(path: str, seconds: int = storage_service._ORPHAN_MIN_AGE_S + 3600) -> None:
    """Backdate a path's mtime so the orphan sweep treats it as settled."""
    old = time.time() - seconds
    os.utime(path, (old, old))


@pytest.fixture()
def home(tmp_path):
    """A fake ~/.smartloop with every category populated."""
    home_dir = tmp_path / "dot_smartloop"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("user data")
    home_dir.mkdir()

    workspace_id = str(uuid4())
    workspace_dir = home_dir / workspace_id
    models_dir = workspace_dir / "models"
    models_dir.mkdir(parents=True)

    # Models on disk: every registered one, plus an unregistered leftover.
    for _name, model_id, _enabled, _active in MODELS:
        _write(str(models_dir / model_id / "weights.gguf"), "w" * 100)
    _write(str(models_dir / "Qwen" / "unregistered-model" / "weights.gguf"), "w" * 50)
    # A flat file directly under embeddings/, as the embedding downloader writes.
    _write(str(models_dir / "embeddings" / "bge-m3.gguf"), "e" * 40)
    # A symlink pointing out of home: never sized, never followed.
    os.symlink(str(outside), str(models_dir / "google" / "link"))

    project_id = str(uuid4())
    project_dir = workspace_dir / project_id
    _write(str(project_dir / "documents" / "readme.txt"), "doc")
    orphan_id = str(uuid4())
    _write(str(workspace_dir / orphan_id / "documents" / "stale.txt"), "stale")
    # Backdated past _ORPHAN_MIN_AGE_S: the sweep leaves a recently touched
    # directory alone, so an orphan created here and now would never be offered.
    _age(str(workspace_dir / orphan_id))
    # An orphan by shape but touched moments ago — the age guard keeps it.
    fresh_orphan_id = str(uuid4())
    _write(str(workspace_dir / fresh_orphan_id / "documents" / "live.txt"), "live")
    # A non-uuid directory is not a project shape and must be left alone.
    _write(str(workspace_dir / "not-a-project" / "keep.txt"), "keep")
    _write(str(workspace_dir / "workspace.db"), "db")

    for name in PROTECTED_FILES:
        _write(str(home_dir / name), "protected")
    (home_dir / ".installed-version").write_text(CURRENT_VERSION)
    (home_dir / "version").write_text(CURRENT_VERSION)

    _write(str(home_dir / "server.log"), "l" * 300)
    _write(str(home_dir / "server.log.1"), "l" * 500)
    _write(str(home_dir / "server.log.2"), "l" * 700)

    _write(str(home_dir / CURRENT_VERSION / "slp"), "binary")
    _write(str(home_dir / OLD_VERSION / "slp"), "old binary")

    _write(str(home_dir / "cache" / "slp-1.1.0.tar.gz"), "c" * 200)
    _write(str(home_dir / "templates" / "project_1.zip"), "t" * 100)

    session_id = str(uuid4())
    untracked_id = str(uuid4())
    _write(str(home_dir / "conversations" / f"{session_id}.jsonl"), '{"role":"user"}\n')
    _write(str(home_dir / "conversations" / f"{untracked_id}.jsonl"), '{"role":"user"}\n')

    project = Project(
        id=project_id,
        project_dir=str(project_dir),
        current=True,
        name="test-project",
        sessions=[session_id],
    )
    workspace = Workspace(
        id=workspace_id,
        workspace_dir=str(workspace_dir),
        models_dir=str(models_dir),
        projects=[project],
        models=[
            WorkspaceModel(
                model_name=name, model_id=model_id, downloaded=True,
                enabled=enabled, is_active=active, models_dir=str(models_dir),
            )
            for name, model_id, enabled, active in MODELS
        ],
    )

    config_file = home_dir / "config.json"
    ws_data = workspace.model_dump(mode="json")
    ws_data.pop("current_project", None)
    config_file.write_text(json.dumps({"id": "", "workspace": ws_data}, indent=2))

    return {
        "home_dir": str(home_dir),
        "outside": str(outside),
        "workspace": workspace,
        "workspace_dir": str(workspace_dir),
        "models_dir": str(models_dir),
        "project": project,
        "orphan_id": orphan_id,
        "fresh_orphan_id": fresh_orphan_id,
        "config_file": str(config_file),
        "session_id": session_id,
        "untracked_id": untracked_id,
    }


@pytest.fixture()
def env(home):
    """Point the framework at the fake home and hand back a mock ModelService."""
    settings = AppSettings()
    settings.slp_home = home["home_dir"]

    service = MagicMock()
    service._ensure_workspace.return_value = home["workspace"]
    # No model resident, no task running, permits always available.
    service._model_name = None
    service._decision_model_name = None
    service._large_model_name = None
    service._vision_model_name = None
    service.__dict__["_task_queue"] = MagicMock()
    # Answer only for tasks that are really registered.  A blanket return value
    # let a test name any task and get a reply, which is how two tests came to
    # assert that a "downloader" task guarded downloads when nothing has ever
    # registered one.  Mirrors TaskQueue.get_status.
    service._task_queue.get_status.side_effect = lambda name=None: (
        {"status": "idle"} if name == "indexer"
        else {"error": f"Unknown task: {name}"}
    )
    service._inference_lock = MagicMock()
    service._inference_lock.acquire.return_value = True

    patches = [
        patch("smartloop.api.services.storage_service._settings", settings),
        patch("smartloop.workspace.workspace._settings", settings),
        patch.object(Workspace, "config_file", home["config_file"]),
        # sys.executable points outside the fake home, so version detection has
        # to fall back to the markers.
        patch("smartloop.api.services.storage_service._settings", settings),
    ]
    for p in patches:
        p.start()
    try:
        yield {**home, "service": service, "settings": settings}
    finally:
        for p in patches:
            p.stop()


def _rows(usage, category):
    return usage["categories"][category]["items"]


def _by_model_id(usage):
    return {r["model_id"]: r for r in _rows(usage, "models")}


class TestComputeUsage:
    def test_totals_match_a_walk_of_the_tree(self, env):
        usage = storage_service.compute_usage(model_service=env["service"])

        expected = 0
        for root, dirs, files in os.walk(env["home_dir"]):
            dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))]
            for name in files:
                path = os.path.join(root, name)
                if not os.path.islink(path):
                    expected += os.path.getsize(path)

        assert usage["total"] == expected
        assert usage["home_dir"] == os.path.realpath(env["home_dir"])

    def test_enabled_model_is_never_removable(self, env):
        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        picked = rows["google/gemma-4-e4b-it"]
        assert picked["removable"] is False
        assert picked["keep_reason"] == "enabled"

    def test_bundled_and_active_models_are_kept(self, env):
        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        assert rows["smartloop-ai/slp-orchestra-mini"]["removable"] is False
        assert rows["smartloop-ai/slp-orchestra-mini"]["keep_reason"] == "bundled"

    def test_disabled_models_are_removable(self, env):
        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        assert {mid for mid, r in rows.items() if r["removable"]} == (
            REMOVABLE_IDS | {"Qwen/unregistered-model"}
        )

    def test_resident_model_is_kept_even_when_disabled(self, env):
        env["service"]._cap_llm_gemma3_4b = None
        env["service"].__dict__["_cap_llm_gemma3-4b"] = object()

        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        assert rows["google/gemma-3-4b-it"]["removable"] is False
        assert rows["google/gemma-3-4b-it"]["keep_reason"] == "loaded"

    def test_project_enabled_model_is_kept_though_disabled_app_wide(self, env):
        # A project override resolves ahead of the workspace entry, so this
        # model is in use even though the app-wide picker has it off.
        env["project"].model_settings.append(
            ProjectModel(model_name="gemma3-4b", enabled=True)
        )

        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        assert rows["google/gemma-3-4b-it"]["removable"] is False
        assert rows["google/gemma-3-4b-it"]["keep_reason"] == "enabled"

    def test_project_active_model_is_kept_though_disabled_app_wide(self, env):
        env["project"].model_settings.append(
            ProjectModel(model_name="gemma3-4b", enabled=False, is_active=True)
        )

        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        assert rows["google/gemma-3-4b-it"]["removable"] is False
        assert rows["google/gemma-3-4b-it"]["keep_reason"] == "active"

    def test_model_off_in_every_scope_stays_removable(self, env):
        # The guard against over-keeping: a project entry that says off must not
        # protect anything, or cleanup quietly frees nothing.
        env["project"].model_settings.append(
            ProjectModel(model_name="gemma3-4b", enabled=False)
        )

        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        assert rows["google/gemma-3-4b-it"]["removable"] is True

    def test_a_model_every_project_disabled_is_removable_though_the_default_is_on(self, env):
        """The workspace entry is a template, not a scope.

        It holds the defaults a *new* project is seeded from, so a model it
        enables is not thereby in use by anything. Treating it as a scope of
        its own made the app-default pin weights every existing project had
        turned off — and since the model picker only ever writes project
        overrides, that was disk the UI could never offer to reclaim.
        """
        # gemma4-e4b is enabled on the workspace entry (see MODELS).
        env["project"].model_settings.append(
            ProjectModel(model_name="gemma4-e4b", enabled=False)
        )

        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        assert rows["google/gemma-4-e4b-it"]["removable"] is True
        assert rows["google/gemma-4-e4b-it"]["keep_reason"] is None

    def test_a_project_without_an_override_still_inherits_the_default(self, env):
        # The other half of the same rule: dropping the workspace scope must not
        # cost the template its meaning. This project says nothing about
        # gemma4-e4b, so it reads through to the app-default and keeps it.
        assert not env["project"].model_settings

        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        assert rows["google/gemma-4-e4b-it"]["removable"] is False
        assert rows["google/gemma-4-e4b-it"]["keep_reason"] == "enabled"

    def test_with_no_projects_the_default_is_consulted_directly(self, env):
        # Nothing to read through, so the template is all there is — otherwise
        # a fresh workspace would offer its just-downloaded weights for deletion.
        env["workspace"].projects.clear()

        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        assert rows["google/gemma-4-e4b-it"]["removable"] is False
        assert rows["google/gemma-4-e4b-it"]["keep_reason"] == "enabled"

    def test_every_project_is_refreshed_before_deciding(self, env):
        """Under supervision a project's file is written by its own child.

        This process holds the copy it took when the project first appeared and
        nothing tells it when that went stale, so the sweep would otherwise
        decide what to delete from a snapshot.
        """
        storage_service.compute_usage(model_service=env["service"])

        refreshed = [
            c.args[0] for c in env["service"].refresh_project_state.call_args_list
        ]
        assert refreshed == [env["project"].id]

    def test_a_childs_write_reaches_the_decision(self, env):
        """The direction that actually costs something: a model a child turned
        back *on* must stop being offered, or the sweep deletes weights a
        project is using."""
        # gemma3-4b is off app-wide, so it starts removable.
        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))
        assert rows["google/gemma-3-4b-it"]["removable"] is True

        # A child enables it and writes its own file; the refresh is what picks
        # that up on this side.
        def pick_up_the_write(project_id):
            env["project"].model_settings.append(
                ProjectModel(model_name="gemma3-4b", enabled=True)
            )
            return True

        env["service"].refresh_project_state.side_effect = pick_up_the_write

        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        assert rows["google/gemma-3-4b-it"]["removable"] is False
        assert rows["google/gemma-3-4b-it"]["keep_reason"] == "enabled"

    def test_an_unreadable_project_does_not_fail_the_report(self, env):
        # Keeps the copy already held rather than refusing the whole sweep.
        env["service"].refresh_project_state.side_effect = OSError("gone")

        usage = storage_service.compute_usage(model_service=env["service"])

        assert usage["categories"]["models"]["items"]

    def test_the_bootstrap_model_resolves_under_its_legacy_name(self, env):
        # config.json written before the rename records slp-orchestra-mini;
        # canonicalization has to keep it resolving to the same entry.
        env["workspace"].models[0].model_name = "slp-orchestra-mini"

        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        assert rows["smartloop-ai/slp-orchestra-mini"]["removable"] is False
        assert {mid for mid, r in rows.items() if r["removable"]} == (
            REMOVABLE_IDS | {"Qwen/unregistered-model"}
        )

    def test_unregistered_model_dir_is_removable_and_flagged(self, env):
        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        leftover = rows["Qwen/unregistered-model"]
        assert leftover["registered"] is False
        assert leftover["removable"] is True

    def test_symlink_is_not_counted(self, env):
        usage = storage_service.compute_usage(model_service=env["service"])
        paths = [r["path"] for r in _rows(usage, "models")]

        assert not any(p.endswith("link") for p in paths)
        # The 9-byte file behind the symlink must not appear in any total.
        assert usage["categories"]["models"]["bytes"] < 1000

    def test_current_version_is_kept_old_one_is_not(self, env):
        rows = {r["version"]: r for r in _rows(storage_service.compute_usage(
            model_service=env["service"]), "versions")}

        assert rows[CURRENT_VERSION]["removable"] is False
        assert rows[CURRENT_VERSION]["keep_reason"] == "running"
        assert rows[OLD_VERSION]["removable"] is True

    def test_log_and_backups_are_reported_with_actions(self, env):
        rows = {os.path.basename(r["path"]): r for r in _rows(
            storage_service.compute_usage(model_service=env["service"]), "logs")}

        assert rows["server.log"]["action"] == "truncate"
        assert rows["server.log.1"]["action"] == "delete"
        assert rows["server.log.2"]["action"] == "delete"

    def test_orphan_project_is_found_live_one_is_not(self, env):
        rows = _rows(storage_service.compute_usage(model_service=env["service"]),
                     "orphan_projects")

        assert [r["project_id"] for r in rows] == [env["orphan_id"]]
        assert rows[0]["removable"] is True

    def test_conversations_are_reported_but_never_reclaimable(self, env):
        category = storage_service.compute_usage(
            model_service=env["service"])["categories"]["conversations"]

        assert category["reclaimable"] == 0
        assert category["bytes"] > 0
        assert all(r["removable"] is False for r in category["items"])
        tracked = {r["session_id"]: r["tracked"] for r in category["items"]}
        assert tracked[env["session_id"]] is True
        assert tracked[env["untracked_id"]] is False

    def test_raises_when_workspace_is_unavailable(self, env):
        env["service"]._ensure_workspace.return_value = None

        with pytest.raises(RuntimeError):
            storage_service.compute_usage(model_service=env["service"])


class TestModelUsageGroups:
    """The split the usage panel draws: in use, framework-owned, reclaimable.

    The UI used to show only what it could delete, so every model the user had
    turned on vanished into a "System" remainder alongside config.json.  These
    assert the report names those bytes instead.
    """

    def test_the_groups_partition_the_category(self, env):
        models = storage_service.compute_usage(
            model_service=env["service"]
        )["categories"]["models"]

        assert (
            models["in_use"] + models["system"]
            + models["reclaimable"] + models["overhead"]
        ) == models["bytes"]

    def test_a_bundled_model_is_framework_not_user_footprint(self, env):
        models = storage_service.compute_usage(
            model_service=env["service"]
        )["categories"]["models"]
        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        bootstrap = rows["smartloop-ai/slp-orchestra-mini"]
        # It is the default *and* active, so it would have been held anyway —
        # being bundled has to win, or the orchestrator reads as a choice the
        # user made and shows up under their own usage.
        assert bootstrap["group"] == "system"
        assert models["system"] == bootstrap["bytes"]

    def test_a_model_the_user_turned_on_counts_as_theirs(self, env):
        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        assert rows["google/gemma-4-e4b-it"]["group"] == "in_use"

    def test_unused_is_the_same_bytes_as_reclaimable(self, env):
        usage = storage_service.compute_usage(model_service=env["service"])
        models = usage["categories"]["models"]

        unused = sum(
            r["bytes"] for r in models["items"] if r["group"] == "unused"
        )
        assert unused == models["reclaimable"]
        assert {r["model_id"] for r in models["items"] if r["group"] == "unused"} == (
            REMOVABLE_IDS | {"Qwen/unregistered-model"}
        )

    def test_overhead_catches_what_has_no_row(self, env):
        models = storage_service.compute_usage(
            model_service=env["service"]
        )["categories"]["models"]

        # The embedding GGUF lands as a loose file under embeddings/ rather than
        # in an <org>/<repo> tree, so it is inside the walked total and has no
        # item of its own.  Without overhead those bytes would go unexplained.
        assert models["overhead"] == 40

    def test_an_arriving_download_is_in_use_not_reclaimable(self, env):
        live = _make_partial(env["models_dir"], "google/gemma-3-4b-it")

        with downloading(live):
            rows = _by_model_id(
                storage_service.compute_usage(model_service=env["service"])
            )

            # Growing bytes are not space anyone can offer to free.
            assert rows["google/gemma-3-4b-it"]["group"] == "in_use"

    def test_an_abandoned_download_is_reclaimable(self, env):
        _make_partial(env["models_dir"], "google/gemma-4-e4b-it")

        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        # Enabled, but the weights never arrived — the partial rule wins, and
        # the group follows it rather than the picker's flag.
        assert rows["google/gemma-4-e4b-it"]["group"] == "unused"


class TestOrphanProjectsSettleFirst:
    def test_a_recently_touched_orphan_is_left_alone(self, env):
        rows = _rows(storage_service.compute_usage(model_service=env["service"]),
                     "orphan_projects")

        ids = [r["project_id"] for r in rows]
        assert env["orphan_id"] in ids
        assert env["fresh_orphan_id"] not in ids, (
            "a directory written moments ago was offered for deletion"
        )

    def test_the_sweep_leaves_a_fresh_orphan_on_disk(self, env):
        storage_service.run_cleanup(
            model_service=env["service"], orphan_projects=True,
        )

        assert os.path.isdir(
            os.path.join(env["workspace_dir"], env["fresh_orphan_id"])
        )

    def test_an_orphan_that_has_settled_is_swept(self, env):
        storage_service.run_cleanup(
            model_service=env["service"], orphan_projects=True,
        )

        assert not os.path.exists(
            os.path.join(env["workspace_dir"], env["orphan_id"])
        )


class TestCleanup:
    def test_nothing_happens_without_a_category(self, env):
        result = storage_service.run_cleanup(model_service=env["service"])

        assert result["freed"] == 0
        assert result["deleted"] == []

    def test_dry_run_reports_without_deleting(self, env):
        planned = storage_service.run_cleanup(
            model_service=env["service"], models=True, versions=True, logs=True,
            cache=True, orphan_projects=True, dry_run=True,
        )
        real = storage_service.run_cleanup(
            model_service=env["service"], models=True, versions=True, logs=True,
            cache=True, orphan_projects=True,
        )

        assert planned["dry_run"] is True
        assert planned["freed"] == real["freed"]
        assert {d["path"] for d in planned["deleted"]} == {d["path"] for d in real["deleted"]}

    def test_model_sweep_keeps_what_is_in_use(self, env):
        storage_service.run_cleanup(model_service=env["service"], models=True)

        models_dir = env["models_dir"]
        assert os.path.exists(os.path.join(models_dir, "smartloop-ai/slp-orchestra-mini"))
        assert os.path.exists(os.path.join(models_dir, "google/gemma-4-e4b-it"))
        assert os.path.exists(os.path.join(models_dir, "embeddings"))
        assert not os.path.exists(os.path.join(models_dir, "google/gemma-3-4b-it"))
        assert not os.path.exists(os.path.join(models_dir, "mistralai"))

    def test_model_sweep_keeps_a_project_enabled_model(self, env):
        env["project"].model_settings.append(
            ProjectModel(model_name="gemma3-4b", enabled=True)
        )

        result = storage_service.run_cleanup(model_service=env["service"], models=True)

        assert os.path.exists(os.path.join(env["models_dir"], "google/gemma-3-4b-it"))
        assert "gemma3-4b" not in result["models_marked"]

    def test_model_sweep_prunes_the_emptied_org_dir(self, env):
        storage_service.run_cleanup(model_service=env["service"], models=True)

        # mistralai held only the deleted model; google still has a kept one.
        assert not os.path.exists(os.path.join(env["models_dir"], "mistralai"))
        assert os.path.isdir(os.path.join(env["models_dir"], "google"))

    def test_model_sweep_updates_workspace_flags_in_memory_and_on_disk(self, env):
        result = storage_service.run_cleanup(model_service=env["service"], models=True)

        assert set(result["models_marked"]) == {"gemma3-4b", "ministral3-3b"}
        removed = {
            m.model_name: m for m in env["workspace"].models
            if m.model_id in REMOVABLE_IDS
        }
        for m in removed.values():
            assert (m.downloaded, m.enabled, m.is_active) == (False, False, False)

        on_disk = json.loads(open(env["config_file"]).read())
        flags = {
            m["model_name"]: m for m in on_disk["workspace"]["models"]
        }
        assert flags["gemma3-4b"]["downloaded"] is False
        assert flags["gemma4-e4b"]["downloaded"] is True

    def test_model_sweep_unloads_each_deleted_model_only(self, env):
        storage_service.run_cleanup(model_service=env["service"], models=True)

        calls = env["service"]._unload_model_by_name_unlocked.call_args_list
        assert {c.args[0] for c in calls} == {"gemma3-4b", "ministral3-3b"}
        # unload_model() would evict the always-resident orchestrator.
        env["service"].unload_model.assert_not_called()
        env["service"]._reset_llm_executor.assert_called_once()

    def test_model_sweep_never_takes_the_public_unload(self, env):
        # unload_model_by_name() acquires an inference slot from the very
        # semaphore the sweep has just drained, and threading.Semaphore is not
        # reentrant — calling it here deadlocks the request forever while
        # holding every permit, which wedges inference too.
        storage_service.run_cleanup(model_service=env["service"], models=True)

        env["service"].unload_model_by_name.assert_not_called()

    def test_model_sweep_unloads_without_a_free_permit(self, env):
        # The invariant behind the unlocked unload, against a real semaphore:
        # by the time unload runs, no permit is left to take.
        # Sized as ModelService sizes it, from the same n_parallel the sweep
        # reads when deciding how many permits to drain.
        permits = env["settings"].n_parallel
        lock = threading.Semaphore(permits)
        env["service"]._inference_lock = lock
        acquired = []

        def _unload(_name):
            acquired.append(lock.acquire(timeout=0.25))

        env["service"]._unload_model_by_name_unlocked.side_effect = _unload

        storage_service.run_cleanup(model_service=env["service"], models=True)

        assert acquired and not any(acquired), "the sweep must hold every permit"
        assert not os.path.exists(
            os.path.join(env["models_dir"], "mistralai/Ministral-3-3B-Instruct-2512")
        )
        # Every permit handed back, so inference is not wedged afterwards.
        assert all(lock.acquire(timeout=0.25) for _ in range(permits))

    def test_model_ids_restricts_the_sweep(self, env):
        storage_service.run_cleanup(
            model_service=env["service"], models=True,
            model_ids=["google/gemma-3-4b-it"],
        )

        assert not os.path.exists(os.path.join(env["models_dir"], "google/gemma-3-4b-it"))
        assert os.path.exists(os.path.join(env["models_dir"], "mistralai"))

    @pytest.mark.parametrize("model_id", [
        "google/gemma-4-e4b-it",          # in use
        "smartloop-ai/slp-orchestra-mini",  # bundled
        "../../../etc",                    # traversal
        "no-such/model",
    ])
    def test_model_ids_rejects_anything_not_removable(self, env, model_id):
        with pytest.raises(ValueError):
            storage_service.run_cleanup(
                model_service=env["service"], models=True, model_ids=[model_id],
            )

    def test_log_cleanup_empties_the_log_and_deletes_backups(self, env):
        log = os.path.join(env["home_dir"], "server.log")

        result = storage_service.run_cleanup(model_service=env["service"], logs=True)

        assert os.path.isfile(log), "server.log is emptied, never unlinked"
        assert os.path.getsize(log) == 0
        assert not os.path.exists(log + ".1")
        assert not os.path.exists(log + ".2")
        assert result["freed"] == 300 + 500 + 700

    def test_log_sweep_replays_its_record_after_emptying_the_log(self, env, caplog):
        """Emptying server.log must not erase the record of the same sweep.

        Deletions are logged as they happen, so a sweep that also truncates the
        log destroys its own audit trail — the one thing a "where did my model
        go?" report needs.  The record is therefore written again once the log
        is empty.
        """
        gemma = os.path.join(env["models_dir"], "google", "gemma-3-4b-it")

        with caplog.at_level(logging.INFO, logger=storage_service.logger.name):
            storage_service.run_cleanup(
                model_service=env["service"], models=True, logs=True,
            )

        messages = [r.getMessage() for r in caplog.records]
        replay_at = next(
            (i for i, m in enumerate(messages) if "replayed after server.log" in m),
            None,
        )
        assert replay_at is not None, "no record was replayed after the truncate"

        survives = messages[replay_at:]
        assert any(gemma in m and "Removed" in m for m in survives), (
            "the model deletion is missing from the record that outlives the truncate"
        )
        assert any("server.log" in m and "Removed" in m for m in survives)
        assert any("gemma3-4b" in m for m in survives), (
            "models_marked is part of the record too"
        )

    def test_sweep_without_logs_does_not_replay(self, env, caplog):
        """Nothing truncated the log, so the first record still stands."""
        with caplog.at_level(logging.INFO, logger=storage_service.logger.name):
            storage_service.run_cleanup(model_service=env["service"], models=True)

        messages = [r.getMessage() for r in caplog.records]
        assert not any("replayed" in m for m in messages)
        assert sum(1 for m in messages if "Removed" in m and "gemma-3-4b-it" in m) == 1

    def test_version_cleanup_keeps_the_running_install(self, env):
        storage_service.run_cleanup(model_service=env["service"], versions=True)

        assert os.path.isdir(os.path.join(env["home_dir"], CURRENT_VERSION))
        assert not os.path.exists(os.path.join(env["home_dir"], OLD_VERSION))

    def test_cache_cleanup_empties_contents_but_keeps_the_dirs(self, env):
        storage_service.run_cleanup(model_service=env["service"], cache=True)

        for name in ("cache", "templates"):
            path = os.path.join(env["home_dir"], name)
            assert os.path.isdir(path)
            assert os.listdir(path) == []

    def test_orphan_project_cleanup_leaves_live_project_and_odd_dirs(self, env):
        storage_service.run_cleanup(model_service=env["service"], orphan_projects=True)

        assert not os.path.exists(os.path.join(env["workspace_dir"], env["orphan_id"]))
        assert os.path.isdir(env["project"].project_dir)
        assert os.path.isdir(os.path.join(env["workspace_dir"], "not-a-project"))
        assert os.path.isdir(os.path.join(env["workspace_dir"], "models"))

    def test_conversations_are_never_deleted(self, env):
        storage_service.run_cleanup(
            model_service=env["service"], models=True, versions=True, logs=True,
            cache=True, orphan_projects=True,
        )

        conversations = os.path.join(env["home_dir"], "conversations")
        assert sorted(os.listdir(conversations)) == sorted(
            [f"{env['session_id']}.jsonl", f"{env['untracked_id']}.jsonl"]
        )

    def test_protected_files_survive_every_category(self, env):
        storage_service.run_cleanup(
            model_service=env["service"], models=True, versions=True, logs=True,
            cache=True, orphan_projects=True,
        )

        for name in PROTECTED_FILES:
            assert os.path.isfile(os.path.join(env["home_dir"], name)), name
        assert os.path.isfile(os.path.join(env["workspace_dir"], "workspace.db"))
        assert os.path.isfile(os.path.join(env["outside"], "precious.txt"))

    def test_failure_is_reported_and_asks_for_a_restart(self, env):
        with patch("smartloop.api.services.storage_service.shutil.rmtree",
                   side_effect=PermissionError("still mapped")):
            result = storage_service.run_cleanup(model_service=env["service"], models=True)

        assert result["deleted"] == []
        assert result["restart_required"] is True
        assert {f["error"] for f in result["failed"]} == {"still mapped"}
        # Nothing was marked removed, because nothing was actually removed.
        assert result["models_marked"] == []
        assert os.path.exists(os.path.join(env["models_dir"], "google/gemma-3-4b-it"))

    def test_refuses_while_the_indexer_is_running(self, env):
        env["service"]._task_queue.get_status.side_effect = (
            lambda name=None: {"status": "running"}
        )

        with pytest.raises(RuntimeError, match="indexer"):
            storage_service.run_cleanup(
                model_service=env["service"], orphan_projects=True,
            )

    def test_a_running_indexer_does_not_block_a_model_sweep(self, env):
        """The indexer writes project trees, not model weights."""
        env["service"]._task_queue.get_status.side_effect = (
            lambda name=None: {"status": "running"}
        )

        storage_service.run_cleanup(model_service=env["service"], models=True)

    def test_refuses_when_inference_will_not_drain(self, env):
        env["service"]._inference_lock.acquire.return_value = False

        with pytest.raises(RuntimeError, match="busy"):
            storage_service.run_cleanup(model_service=env["service"], models=True)

    def test_log_only_cleanup_does_not_wait_on_inference(self, env):
        storage_service.run_cleanup(model_service=env["service"], logs=True)

        env["service"]._inference_lock.acquire.assert_not_called()

    def test_report_includes_fresh_usage(self, env):
        result = storage_service.run_cleanup(model_service=env["service"], logs=True)

        assert result["usage"]["categories"]["logs"]["reclaimable"] == 0


def _make_partial(models_dir, model_id, *, keep=(), part=("weights.gguf.part",)):
    """Rewrite *model_id*'s directory to a given set of files.

    *keep* lands as finished files, *part* as unfinished sidecars.  The
    fixture's own ``weights.gguf`` is cleared first so each case states the
    whole directory rather than inheriting a stray complete file.
    """
    target = os.path.join(models_dir, model_id)
    for name in os.listdir(target):
        os.remove(os.path.join(target, name))
    for name in keep:
        _write(os.path.join(target, name), "w" * 100)
    for name in part:
        _write(os.path.join(target, name), "p" * 100)
    return target


class TestAbandonedDownloads:
    """A directory holding only a ``.part`` is a download that died.

    The keep rules all answer "is this model in use?", and none of them can be
    true of bytes that never arrived — ``enabled`` in particular is set when
    the user asks for a model, not when it lands.  Protecting such a tree
    strands the space on a model nothing can load, and on a full disk that is
    a deadlock: the only remedy the UI offers is the download this space
    blocks.

    The bundled orgs are the exception, for the reason the deadlock argument
    turns on: there the retry is not a thing the user might choose, so the
    space comes back on its own.  See
    ``test_a_bundled_partial_is_kept_for_its_retry``.
    """

    def test_abandoned_partial_is_removable_though_enabled(self, env):
        _make_partial(env["models_dir"], "google/gemma-4-e4b-it")

        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        dead = rows["google/gemma-4-e4b-it"]
        assert dead["partial"] is True
        assert dead["removable"] is True
        assert dead["keep_reason"] is None

    def test_abandoned_partial_beats_a_project_pin(self, env):
        # The strongest keep reason there is: off app-wide but pinned on by a
        # project.  Still nothing to protect.
        env["project"].model_settings.append(
            ProjectModel(model_name="gemma3-4b", enabled=True)
        )
        _make_partial(env["models_dir"], "google/gemma-3-4b-it")

        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        assert rows["google/gemma-3-4b-it"]["removable"] is True

    def test_a_bundled_partial_is_kept_for_its_retry(self, env):
        # The inverse of every other case in this class.  A half-downloaded
        # orchestrator cannot run the framework — which is exactly why it will
        # be downloaded again, and why the retry (which unlinks the stale
        # .part itself) reclaims the space without the sweep.  Deleting it
        # buys nothing and risks racing a writer, so bundled wins.
        _make_partial(env["models_dir"], "smartloop-ai/slp-orchestra-mini")

        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        held = rows["smartloop-ai/slp-orchestra-mini"]
        assert held["removable"] is False
        assert held["keep_reason"] == "bundled"
        # Still an honest description of the directory; it just isn't offered.
        assert held["partial"] is True

    def test_a_bundled_partial_survives_the_sweep(self, env):
        held = _make_partial(env["models_dir"], "smartloop-ai/slp-orchestra-mini")
        removable = _make_partial(env["models_dir"], "google/gemma-4-e4b-it")

        result = storage_service.run_cleanup(
            model_service=env["service"], models=True,
        )

        assert os.path.exists(held), "the sweep reclaimed bundled weights"
        assert held not in [d["path"] for d in result["deleted"]]
        assert not os.path.exists(removable), (
            "protecting the bundled tree stopped the rest of the sweep"
        )

    def test_a_live_embedding_fetch_is_never_offered(self, env):
        """Embedding weights land as loose files directly in
        ``models/embeddings/`` — ``llama_embeddings._fetch_gguf`` writes the
        GGUF and its ``.part`` there rather than under an ``<org>/<repo>``
        pair, and it takes no ``downloading`` mark.

        Two things keep them: ``_dirs_in`` enumerates directories, so loose
        files never become rows at all, and the org is bundled so anything that
        *is* a directory under it is held.  The first is an accident of scan
        depth; this test is what fails loudly if that ever changes.
        """
        embeddings = os.path.join(env["models_dir"], "embeddings")
        os.makedirs(embeddings, exist_ok=True)
        _write(os.path.join(embeddings, "bge-m3-Q4_K_M.gguf.part"), "p" * 100)

        usage = storage_service.compute_usage(model_service=env["service"])
        offered = [r["path"] for r in usage["categories"]["models"]["items"]]
        assert not any(p.startswith(embeddings) for p in offered)

        storage_service.run_cleanup(model_service=env["service"], models=True)

        assert os.path.isfile(
            os.path.join(embeddings, "bge-m3-Q4_K_M.gguf.part")
        ), "the sweep deleted an in-flight embedding download"

    def test_weights_that_landed_survive_a_partial_companion(self, env):
        # The interrupted-multi-file case: main weights finished, the vision
        # projector did not.  The downloader checksums the finished file and
        # refetches only the rest, so deleting the tree would throw away real
        # progress — this is the one .part case that must be kept.
        _make_partial(
            env["models_dir"], "google/gemma-4-e4b-it",
            keep=("weights.gguf",), part=("mmproj-gemma-4.gguf.part",),
        )

        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        kept = rows["google/gemma-4-e4b-it"]
        assert kept["partial"] is False
        assert kept["removable"] is False
        assert kept["keep_reason"] == "enabled"

    def test_a_finished_mmproj_alone_is_not_weights(self, env):
        # The mirror image: the projector landed, the weights are still
        # transferring.  A projector is not something that can be served, so
        # this tree is as dead as one holding only .part files.
        _make_partial(
            env["models_dir"], "google/gemma-4-e4b-it",
            keep=("mmproj-gemma-4.gguf",), part=("weights.gguf.part",),
        )

        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        assert rows["google/gemma-4-e4b-it"]["partial"] is True
        assert rows["google/gemma-4-e4b-it"]["removable"] is True

    def test_a_resident_model_is_kept_even_when_partial(self, env):
        # Shouldn't be reachable — weights resolve before a load — but a
        # resident context is the one state where unlinking is a native
        # use-after-free, so it outranks the partial rule.
        env["service"].__dict__["_cap_llm_gemma3-4b"] = object()
        _make_partial(env["models_dir"], "google/gemma-3-4b-it")

        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        assert rows["google/gemma-3-4b-it"]["removable"] is False
        assert rows["google/gemma-3-4b-it"]["keep_reason"] == "loaded"

    def test_complete_models_are_not_flagged_partial(self, env):
        rows = _by_model_id(storage_service.compute_usage(model_service=env["service"]))

        assert all(r["partial"] is False for r in rows.values())

    def test_an_in_flight_download_is_kept_while_the_rest_is_swept(self, env):
        """The race the partial rule opens: those .part files belong to a
        transfer happening right now, and on disk that is indistinguishable
        from one that died.  The registry is what tells them apart.

        Both halves matter.  The live tree survives — and the sweep still runs,
        reclaiming everything else, because protecting a download must not cost
        the user the cleanup they asked for.  That is the difference between
        this and the blanket refusal it replaced.
        """
        live = _make_partial(env["models_dir"], "google/gemma-4-e4b-it")
        removable = os.path.join(env["models_dir"], "google/gemma-3-4b-it")

        with downloading(live):
            result = storage_service.run_cleanup(
                model_service=env["service"], models=True,
            )

        assert os.path.exists(live), "the sweep deleted a running download"
        assert live not in [d["path"] for d in result["deleted"]]
        assert not os.path.exists(removable), (
            "protecting the download stopped the rest of the sweep"
        )

    def test_a_live_download_reports_why_it_was_kept(self, env):
        live = _make_partial(env["models_dir"], "google/gemma-4-e4b-it")

        with downloading(live):
            rows = _by_model_id(
                storage_service.compute_usage(model_service=env["service"])
            )
            held = rows["google/gemma-4-e4b-it"]

            assert held["keep_reason"] == "downloading"
            assert held["removable"] is False
            # Still accurate: it really is a partial tree, it just is not dead.
            assert held["partial"] is True

        after = _by_model_id(
            storage_service.compute_usage(model_service=env["service"])
        )
        assert after["google/gemma-4-e4b-it"]["removable"] is True, (
            "the mark outlived the download"
        )

    def test_a_download_starting_mid_sweep_is_not_deleted(self, env):
        """The scan is a snapshot, so run_cleanup asks again before each
        removal.  Here the download starts after planning, which is the window
        the re-check exists to cover."""
        live = _make_partial(env["models_dir"], "google/gemma-4-e4b-it")
        state = {"started": False}
        real_remove = storage_service._remove

        def remove_and_start_a_download(path, **kwargs):
            state["started"] = True
            return real_remove(path, **kwargs)

        with patch.object(storage_service, "_remove", remove_and_start_a_download), \
             patch.object(
                 storage_service, "is_downloading",
                 lambda p: state["started"] and p == live,
             ):
            result = storage_service.run_cleanup(
                model_service=env["service"], models=True,
            )

        assert os.path.exists(live)
        assert live not in [d["path"] for d in result["deleted"]]
        assert live not in [f["path"] for f in result["failed"]]
        assert result["restart_required"] is False
        assert "gemma4-e4b" not in result["models_marked"], (
            "a skipped model was recorded as removed"
        )

    def test_the_sweep_reclaims_an_abandoned_partial(self, env):
        dead = _make_partial(env["models_dir"], "google/gemma-4-e4b-it")

        result = storage_service.run_cleanup(model_service=env["service"], models=True)

        assert not os.path.exists(dead)
        assert any(d["path"] == dead for d in result["deleted"])

    def test_partial_bytes_are_reported_as_reclaimable(self, env):
        _make_partial(env["models_dir"], "google/gemma-4-e4b-it")

        usage = storage_service.compute_usage(model_service=env["service"])

        assert usage["categories"]["models"]["reclaimable"] >= 100


class TestValidateRefusesBadPaths:
    """The traversal and symlink rails, exercised directly.

    Every other test in this file feeds ``_validate`` paths its own scanners
    produced, so the refusals never fire and the rails are asserted only
    indirectly — "the protected files survived". That is the wrong shape of
    evidence for the guard standing between a POST body and ``shutil.rmtree``.
    Note in particular that the ``"../../../etc"`` case in the model_ids
    parametrize does not reach here at all: ``_plan_cleanup`` rejects it first
    as an unknown model id.
    """

    def test_home_itself_is_refused(self, env):
        with pytest.raises(ValueError, match="home directory itself"):
            storage_service._validate(env["home_dir"], "models", env["workspace"])

    def test_a_path_outside_home_is_refused(self, env):
        escape = os.path.join(env["outside"], "precious.txt")
        with pytest.raises(ValueError, match="escapes"):
            storage_service._validate(escape, "cache", env["workspace"])

    def test_traversal_out_of_the_models_dir_is_refused(self, env):
        escape = os.path.join(env["models_dir"], "..", "..", "..", "outside")
        with pytest.raises(ValueError, match="escapes"):
            storage_service._validate(escape, "models", env["workspace"])

    @pytest.mark.parametrize("name", PROTECTED_FILES)
    def test_a_never_delete_basename_is_refused(self, env, name):
        target = os.path.join(env["home_dir"], name)
        with pytest.raises(ValueError, match="Protected path"):
            storage_service._validate(target, "cache", env["workspace"])

    def test_a_symlink_is_refused(self, env):
        link = os.path.join(env["models_dir"], "google", "link")
        assert os.path.islink(link), "fixture no longer builds the symlink"
        with pytest.raises(ValueError, match="symlink"):
            storage_service._validate(link, "models", env["workspace"])

    def test_a_model_path_at_the_wrong_depth_is_refused(self, env):
        # models/<org> is one level short of a model directory.
        org_only = os.path.join(env["models_dir"], "google")
        with pytest.raises(ValueError, match="models/<org>/<repo>"):
            storage_service._validate(org_only, "models", env["workspace"])

    def test_the_running_install_is_refused(self, env):
        target = os.path.join(env["home_dir"], CURRENT_VERSION)
        with pytest.raises(ValueError, match="running install"):
            storage_service._validate(target, "versions", env["workspace"])

    def test_a_non_version_directory_is_refused(self, env):
        target = os.path.join(env["home_dir"], "cache")
        with pytest.raises(ValueError, match="Not a version directory"):
            storage_service._validate(target, "versions", env["workspace"])

    @pytest.mark.parametrize("name", ["server.log.0", "server.log.6", "server.log.x"])
    def test_a_backup_index_outside_the_kept_range_is_refused(self, env, name):
        # DEFAULT_BACKUP_COUNT is 5, and the bound is compared numerically so
        # that 10+ does not silently become "any digit".
        target = os.path.join(env["home_dir"], name)
        _write(target, "l")
        with pytest.raises(ValueError, match="Not a server log file"):
            storage_service._validate(target, "logs", env["workspace"])

    def test_a_kept_backup_index_is_accepted(self, env):
        target = os.path.join(env["home_dir"], "server.log.1")
        assert storage_service._validate(
            target, "logs", env["workspace"]
        ) == os.path.realpath(target)

    def test_a_path_outside_the_cache_dirs_is_refused(self, env):
        target = os.path.join(env["home_dir"], "conversations")
        with pytest.raises(ValueError, match="Not a cache entry"):
            storage_service._validate(target, "cache", env["workspace"])

    def test_a_live_project_is_refused(self, env):
        with pytest.raises(ValueError, match="live project"):
            storage_service._validate(
                env["project"].project_dir, "orphan_projects", env["workspace"]
            )

    def test_a_non_uuid_directory_is_not_a_project(self, env):
        target = os.path.join(env["workspace_dir"], "not-a-project")
        with pytest.raises(ValueError):
            storage_service._validate(target, "orphan_projects", env["workspace"])

    def test_an_unknown_category_is_refused(self, env):
        target = os.path.join(env["home_dir"], "cache", "slp-1.1.0.tar.gz")
        with pytest.raises(ValueError, match="Unknown category"):
            storage_service._validate(target, "conversations", env["workspace"])


class TestDiscardingACancelledDownload:
    """What a stopped download leaves behind, and what removes it.

    Reached only from the download thread's own unwinding, so every case here
    assumes the transfer has already stopped.
    """

    def test_it_removes_the_tree_and_clears_the_flags(self, env):
        model_id = "google/gemma-3-4b-it"
        path = _make_partial(env["models_dir"], model_id)

        report = storage_service.discard_partial_download(
            model_service=env["service"], model_id=model_id)

        assert report["removed"] is True
        assert report["freed"] > 0
        assert not os.path.exists(path)
        entry = next(m for m in env["workspace"].models if m.model_id == model_id)
        assert not entry.downloaded and not entry.enabled and not entry.is_active
        assert report["models_marked"] == [entry.model_name]

    def test_it_removes_a_vision_model_that_only_half_arrived(self, env):
        """Weights without their projector is the state this exists to prevent.

        ``_gguf_exists`` reads such a directory as "already available", so
        keeping it would leave the model permanently unloadable while the
        picker showed it ready.
        """
        model_id = "google/gemma-3-4b-it"
        path = _make_partial(env["models_dir"], model_id,
                             keep=("weights.gguf",), part=("mmproj-model.gguf.part",))

        report = storage_service.discard_partial_download(
            model_service=env["service"], model_id=model_id)

        assert report["removed"] is True
        assert not os.path.exists(path)

    def test_it_takes_both_part_files_of_a_vision_model(self, env):
        model_id = "google/gemma-3-4b-it"
        path = _make_partial(env["models_dir"], model_id,
                             part=("weights.gguf.part", "mmproj-model.gguf.part"))

        storage_service.discard_partial_download(
            model_service=env["service"], model_id=model_id)

        assert not os.path.exists(path)

    def test_it_prunes_the_org_directory_it_empties(self, env):
        model_id = "mistralai/Ministral-3-3B-Instruct-2512"
        _make_partial(env["models_dir"], model_id)

        storage_service.discard_partial_download(
            model_service=env["service"], model_id=model_id)

        assert not os.path.exists(os.path.join(env["models_dir"], "mistralai"))

    def test_it_leaves_an_org_that_still_holds_a_sibling(self, env):
        _make_partial(env["models_dir"], "google/gemma-3-4b-it")

        storage_service.discard_partial_download(
            model_service=env["service"], model_id="google/gemma-3-4b-it")

        # gemma-4-e4b-it is still there, so google/ stays.
        assert os.path.isdir(os.path.join(env["models_dir"], "google", "gemma-4-e4b-it"))

    def test_a_directory_that_never_existed_is_not_an_error(self, env):
        """Cancelled before a byte landed.  The caller asked for the bytes to
        be absent, and they are."""
        report = storage_service.discard_partial_download(
            model_service=env["service"], model_id="google/never-started")

        assert report["removed"] is False
        assert report["error"] is None
        assert report["freed"] == 0

    def test_it_leaves_every_other_model_alone(self, env):
        _make_partial(env["models_dir"], "google/gemma-3-4b-it")

        storage_service.discard_partial_download(
            model_service=env["service"], model_id="google/gemma-3-4b-it")

        for _name, model_id, _e, _a in MODELS:
            if model_id != "google/gemma-3-4b-it":
                assert os.path.isdir(os.path.join(env["models_dir"], model_id)), model_id

    def test_it_refuses_a_model_id_that_escapes_the_models_dir(self, env):
        with pytest.raises(ValueError):
            storage_service.discard_partial_download(
                model_service=env["service"], model_id="../../..")

    def test_it_refuses_a_shape_that_is_not_org_slash_repo(self, env):
        with pytest.raises(ValueError, match="models/<org>/<repo>"):
            storage_service.discard_partial_download(
                model_service=env["service"], model_id="google")

    def test_it_refuses_to_follow_a_symlink_out_of_home(self, env):
        """The fixture links models/google/link at a directory outside home.

        The containment rail is the one that fires, not the symlink rail:
        ``_assert_inside_home`` resolves the path first, so a link pointing out
        is refused as an escape before anyone asks what kind of file it is.
        Either way the user's own files are untouched, which is the property
        that matters.
        """
        with pytest.raises(ValueError, match="escapes"):
            storage_service.discard_partial_download(
                model_service=env["service"], model_id="google/link")
        assert os.path.exists(os.path.join(env["outside"], "precious.txt"))

    def test_it_does_not_drain_inference(self, env):
        """A button press must not wait 30s on permits, and nothing can have
        weights mapped that never finished downloading."""
        _make_partial(env["models_dir"], "google/gemma-3-4b-it")

        storage_service.discard_partial_download(
            model_service=env["service"], model_id="google/gemma-3-4b-it")

        env["service"]._inference_lock.acquire.assert_not_called()

    def test_it_does_not_unload_or_reset_the_executor(self, env):
        _make_partial(env["models_dir"], "google/gemma-3-4b-it")

        storage_service.discard_partial_download(
            model_service=env["service"], model_id="google/gemma-3-4b-it")

        env["service"]._unload_model_by_name_unlocked.assert_not_called()
        env["service"]._reset_llm_executor.assert_not_called()

    def test_it_does_not_consult_the_indexer_guard(self, env):
        """``run_cleanup`` refuses while the indexer runs; this must not."""
        _make_partial(env["models_dir"], "google/gemma-3-4b-it")
        env["service"]._task_queue.get_status.side_effect = lambda name=None: (
            {"status": "running"} if name == "indexer" else {"error": "unknown"}
        )

        report = storage_service.discard_partial_download(
            model_service=env["service"], model_id="google/gemma-3-4b-it")

        assert report["removed"] is True
