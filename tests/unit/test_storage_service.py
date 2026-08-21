"""Tests for the storage usage/cleanup service.

The safety rails are the point of these tests: a bug here deletes a user's
models, credentials, or project data.  So most of what follows asserts what
must *not* happen.
"""

import json
import os
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from smartloop.api.services import storage_service
from smartloop.config.app_settings import AppSettings
from smartloop.workspace.workspace import Project, Workspace, WorkspaceModel

# Mirrors the real config: only the bootstrap orchestrator is is_active, while
# the models the user picked carry enabled=True.  Keying the keep-rule on
# is_active alone would delete the picked ones.
MODELS = [
    ("slp-orchestra-mini", "smartloop-ai/slp-orchestra-mini", True, True),
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
    service._task_queue.get_status.return_value = {"status": "idle"}
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

        unloaded = {c.args[0] for c in env["service"].unload_model_by_name.call_args_list}
        assert unloaded == {"gemma3-4b", "ministral3-3b"}
        # unload_model() would evict the always-resident orchestrator.
        env["service"].unload_model.assert_not_called()
        env["service"]._reset_llm_executor.assert_called_once()

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

    def test_refuses_while_the_downloader_is_running(self, env):
        env["service"]._task_queue.get_status.return_value = {"status": "running"}

        with pytest.raises(RuntimeError, match="downloader"):
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
