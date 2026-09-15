"""Unit tests for per-project model settings on the Workspace."""

import pytest

from smartloop.config.app_settings import AppSettings
from smartloop.workspace.workspace import Project, ProjectModel, Workspace, WorkspaceModel

BOOTSTRAP_MODEL = AppSettings().default_model


def _make_workspace() -> Workspace:
    """In-memory workspace with two selectable models and one project.

    The always-on bootstrap model is registered too, so the tests below also
    assert it stays out of the per-project settings.
    """
    return Workspace(
        id="ws1",
        models=[
            WorkspaceModel(
                model_name=BOOTSTRAP_MODEL,
                model_id="smartloop/slp-orchestra-mini",
                downloaded=True,
                enabled=True,
            ),
            WorkspaceModel(
                model_name="gemma3-4b",
                model_id="google/gemma-3-4b-it",
                downloaded=True,
                enabled=True,
            ),
            WorkspaceModel(
                model_name="qwen2-vl-2b",
                model_id="Qwen/Qwen2-VL-2B-Instruct",
                downloaded=True,
                enabled=False,
                is_active=True,
            ),
        ],
        projects=[Project(id="p1", project_dir="/tmp/p1", current=True, name="p1")],
    )


class TestSyncProjectModels:
    def test_seeds_project_settings_from_workspace_defaults(self):
        ws = _make_workspace()
        project = ws.projects[0]

        ws.sync_project_models(project)

        entries = {m.model_name: m for m in project.model_settings}
        assert set(entries) == {"gemma3-4b", "qwen2-vl-2b"}
        # Workspace (app-default) state is copied across: disabled stays
        # disabled, active stays active.  Download state is workspace-wide.
        assert entries["gemma3-4b"].enabled is True
        assert entries["qwen2-vl-2b"].enabled is False
        assert entries["qwen2-vl-2b"].is_active is True

    def test_existing_project_entries_are_not_overwritten(self):
        ws = _make_workspace()
        project = ws.projects[0]
        ws.sync_project_models(project)
        # The project overrides the workspace default: enable qwen2-vl-2b.
        ws.set_project_model_enabled(project, "qwen2-vl-2b", True)

        # Re-seeding (e.g. a later lazy sync) must not clobber the override.
        ws.sync_project_models(project)

        assert ws.project_model(project, "qwen2-vl-2b").enabled is True
        # …while the workspace default is untouched.
        assert ws.get_model("qwen2-vl-2b").enabled is False

    def test_sync_is_idempotent(self):
        ws = _make_workspace()
        project = ws.projects[0]
        ws.sync_project_models(project)
        count = len(project.model_settings)

        ws.sync_project_models(project)

        assert len(project.model_settings) == count


class TestEffectiveResolution:
    def test_project_override_wins_over_workspace_default(self):
        ws = _make_workspace()
        project = ws.projects[0]

        assert ws.effective_model_enabled("qwen2-vl-2b") is False          # workspace default
        ws.set_project_model_enabled(project, "qwen2-vl-2b", True)
        assert ws.effective_model_enabled("qwen2-vl-2b", project) is True   # project override

    def test_falls_back_to_workspace_default_when_project_has_no_entry(self):
        ws = _make_workspace()
        # No sync yet — the project has no entries at all.
        project = Project(id="p2", project_dir="/tmp/p2", current=False, name="p2")
        ws.projects.append(project)

        assert ws.effective_model_enabled("gemma3-4b", project) is True
        assert ws.effective_model_downloaded("gemma3-4b") is True
        assert ws.effective_model_active("gemma3-4b", project) is False

    def test_unknown_model_defaults(self):
        ws = _make_workspace()
        project = ws.projects[0]

        assert ws.effective_model_enabled("nope", project) is True
        assert ws.effective_model_downloaded("nope") is False
        assert ws.effective_model_active("nope", project) is False

    def test_no_project_uses_workspace_defaults(self):
        ws = _make_workspace()
        assert ws.effective_model_enabled("gemma3-4b") is True
        assert ws.effective_model_downloaded("gemma3-4b") is True
        assert ws.effective_model_enabled("qwen2-vl-2b") is False


class TestProjectSetters:
    def test_set_enabled_upserts_entry(self):
        ws = _make_workspace()
        project = ws.projects[0]
        assert ws.project_model(project, "gemma3-4b") is None

        pm = ws.set_project_model_enabled(project, "gemma3-4b", False)

        assert pm.enabled is False
        assert ws.effective_model_enabled("gemma3-4b", project) is False
        # Workspace default untouched.
        assert ws.get_model("gemma3-4b").enabled is True

    def test_download_state_is_not_a_per_project_setting(self):
        """Weights are added/removed workspace-wide, so there is no override.

        A project entry carries only enabled/is_active/settings — nothing a
        project writes can change what is on disk.
        """
        ws = _make_workspace()
        project = ws.projects[0]
        ws.sync_project_models(project)

        assert not hasattr(ws, "set_project_model_downloaded")
        assert "downloaded" not in ProjectModel.model_fields
        # Disabling the model for the project leaves disk state alone.
        ws.set_project_model_enabled(project, "gemma3-4b", False)
        assert ws.effective_model_downloaded("gemma3-4b") is True

    def test_set_active_deactivates_other_project_models(self):
        ws = _make_workspace()
        project = ws.projects[0]
        ws.sync_project_models(project)

        ws.set_project_model_active(project, "gemma3-4b", True)

        assert ws.effective_model_active("gemma3-4b", project) is True
        assert ws.effective_model_active("qwen2-vl-2b", project) is False

    def test_projects_are_independent(self):
        ws = _make_workspace()
        p1 = ws.projects[0]
        p2 = Project(id="p2", project_dir="/tmp/p2", current=False, name="p2")
        ws.projects.append(p2)

        ws.set_project_model_enabled(p1, "gemma3-4b", False)
        ws.set_project_model_active(p1, "gemma3-4b", True)

        assert ws.effective_model_enabled("gemma3-4b", p1) is False
        assert ws.effective_model_enabled("gemma3-4b", p2) is True   # still workspace default
        assert ws.effective_model_active("gemma3-4b", p1) is True
        assert ws.effective_model_active("gemma3-4b", p2) is False


class TestPromoteToWorkspaceDefault:
    """Downloading a model makes it a default for projects created after.

    The trigger is the download, not the enable: ``/v1/init`` calls this once
    the weights land (``tests/api/test_init_promotes_a_downloaded_model.py``),
    and the project route deliberately does not
    (``TestEnablingAModelIsProjectScoped`` in
    ``tests/api/test_project_models.py``).
    """

    def test_promote_flips_the_workspace_default(self):
        ws = _make_workspace()
        assert ws.get_model("qwen2-vl-2b").enabled is False

        ws.promote_model_to_workspace_default("qwen2-vl-2b")

        assert ws.get_model("qwen2-vl-2b").enabled is True

    def test_existing_project_without_an_entry_is_pinned_off(self):
        """An untouched project is 'unspecified', not 'off' — write it down.

        Without the pin it would read through to the new default and silently
        gain a model the user never enabled there.
        """
        ws = _make_workspace()
        p2 = Project(id="p2", project_dir="/tmp/p2", current=False, name="p2")
        ws.projects.append(p2)
        assert ws.project_model(p2, "qwen2-vl-2b") is None

        ws.promote_model_to_workspace_default("qwen2-vl-2b")

        assert ws.project_model(p2, "qwen2-vl-2b").enabled is False
        assert ws.effective_model_enabled("qwen2-vl-2b", p2) is False
        # …and a later lazy sync must not undo the pin.
        ws.sync_project_models(p2)
        assert ws.effective_model_enabled("qwen2-vl-2b", p2) is False

    def test_project_created_after_promotion_inherits_it(self):
        ws = _make_workspace()
        ws.promote_model_to_workspace_default("qwen2-vl-2b")

        fresh = Project(id="p3", project_dir="/tmp/p3", current=False, name="p3")
        ws.projects.append(fresh)
        ws.sync_project_models(fresh)

        assert ws.effective_model_enabled("qwen2-vl-2b", fresh) is True

    def test_pin_touches_only_the_promoted_model(self):
        """Pinning one model must not detach the rest from future defaults."""
        ws = _make_workspace()
        p2 = Project(id="p2", project_dir="/tmp/p2", current=False, name="p2")
        ws.projects.append(p2)

        ws.promote_model_to_workspace_default("qwen2-vl-2b")

        assert {m.model_name for m in p2.model_settings} == {"qwen2-vl-2b"}

    def test_bootstrap_model_is_not_promotable(self):
        ws = _make_workspace()
        project = ws.projects[0]

        ws.promote_model_to_workspace_default(BOOTSTRAP_MODEL)

        # Already always-on, and it still gets no per-project entry.
        assert ws.get_model(BOOTSTRAP_MODEL).enabled is True
        assert ws.project_model(project, BOOTSTRAP_MODEL) is None

    def test_already_enabled_model_writes_nothing(self):
        """The early return keeps a no-op enable from pinning every project."""
        ws = _make_workspace()
        p2 = Project(id="p2", project_dir="/tmp/p2", current=False, name="p2")
        ws.projects.append(p2)

        ws.promote_model_to_workspace_default("gemma3-4b")

        assert ws.get_model("gemma3-4b").enabled is True
        assert p2.model_settings == []

    def test_unknown_model_is_a_noop(self):
        ws = _make_workspace()
        project = ws.projects[0]

        ws.promote_model_to_workspace_default("nope")

        assert project.model_settings == []


class TestProjectAwareSelection:
    def test_largest_downloaded_respects_project_override(self):
        ws = _make_workspace()
        p1 = ws.projects[0]
        ws.sync_project_models(p1)
        # Workspace default: gemma3-4b (larger, enabled) wins.
        assert ws.largest_downloaded_model(p1).model_name == "gemma3-4b"

        # Project disables gemma3-4b → qwen2-vl-2b (the only downloaded one) wins.
        ws.set_project_model_enabled(p1, "gemma3-4b", False)
        ws.set_project_model_enabled(p1, "qwen2-vl-2b", True)
        assert ws.largest_downloaded_model(p1).model_name == "qwen2-vl-2b"

    def test_largest_downloaded_ignores_project_when_none_passed(self):
        ws = _make_workspace()
        project = ws.projects[0]
        ws.set_project_model_enabled(project, "gemma3-4b", False)

        # Without a project, selection falls back to workspace defaults.
        assert ws.largest_downloaded_model().model_name == "gemma3-4b"

    def test_ranked_models_for_filters_on_project_state(self):
        ws = _make_workspace()
        project = ws.projects[0]

        # qwen2-vl-2b is the only vision-capable model (gemma3-4b isn't tagged
        # as vision) and it's disabled by default → no candidates.
        assert ws.ranked_models_for("vision", project) == []

        ws.set_project_model_enabled(project, "qwen2-vl-2b", True)
        ranked = ws.ranked_models_for("vision", project)
        assert [m.model_name for m in ranked] == ["qwen2-vl-2b"]


class TestBootstrapModelIsNotAPerProjectSetting:
    """The orchestrator model is always enabled and can't be turned off.

    It is filtered out of every models list response, so it must not get a
    per-project entry either — an ``enabled=False`` override would otherwise
    shadow its always-on workspace state in the selection paths.
    """

    def test_sync_does_not_seed_an_entry_for_it(self):
        ws = _make_workspace()
        project = ws.projects[0]

        ws.sync_project_models(project)

        assert BOOTSTRAP_MODEL not in {m.model_name for m in project.model_settings}

    def test_effective_state_falls_through_to_the_workspace_record(self):
        ws = _make_workspace()
        project = ws.projects[0]
        ws.sync_project_models(project)

        assert ws.project_model(project, BOOTSTRAP_MODEL) is None
        assert ws.effective_model_enabled(BOOTSTRAP_MODEL, project) is True
        assert ws.effective_model_downloaded(BOOTSTRAP_MODEL) is True

    def test_sync_prunes_an_entry_written_by_an_earlier_build(self):
        ws = _make_workspace()
        project = ws.projects[0]
        # A workspace written before the exclusion existed, with the entry
        # already flipped off — the state that would break model selection.
        project.model_settings.append(
            ProjectModel(model_name=BOOTSTRAP_MODEL, enabled=False)
        )

        ws.sync_project_models(project)

        assert ws.project_model(project, BOOTSTRAP_MODEL) is None
        assert ws.effective_model_enabled(BOOTSTRAP_MODEL, project) is True

    @pytest.mark.parametrize("setter,value", [
        ("set_project_model_enabled", False),
        ("set_project_model_active", True),
    ])
    def test_setters_refuse_to_create_an_entry(self, setter, value):
        ws = _make_workspace()
        project = ws.projects[0]
        ws.sync_project_models(project)

        assert getattr(ws, setter)(project, BOOTSTRAP_MODEL, value) is None
        assert ws.project_model(project, BOOTSTRAP_MODEL) is None
        assert ws.effective_model_enabled(BOOTSTRAP_MODEL, project) is True

    def test_selection_still_sees_it_after_a_project_disables_everything_else(self):
        ws = _make_workspace()
        project = ws.projects[0]
        ws.sync_project_models(project)
        ws.set_project_model_enabled(project, "gemma3-4b", False)
        ws.set_project_model_enabled(project, "qwen2-vl-2b", False)

        assert ws.effective_model_enabled(BOOTSTRAP_MODEL, project) is True

    def test_sync_prunes_an_entry_written_under_the_legacy_name(self):
        """A workspace from before the ``sl-mini`` rename must still heal.

        Pre-rename builds wrote ``model_settings`` entries keyed on
        ``slp-orchestra-mini``.  If the bootstrap check were not alias-aware
        the entry would survive the prune and its ``enabled=False`` would
        shadow the always-on workspace record everywhere selection resolves
        through ``effective_model_enabled``.
        """
        ws = _make_workspace()
        project = ws.projects[0]
        project.model_settings.append(
            ProjectModel(model_name="slp-orchestra-mini", enabled=False)
        )

        ws.sync_project_models(project)

        assert ws.project_model(project, BOOTSTRAP_MODEL) is None
        assert ws.project_model(project, "slp-orchestra-mini") is None
        assert ws.effective_model_enabled(BOOTSTRAP_MODEL, project) is True

    def test_is_bootstrap_model_accepts_the_legacy_name(self):
        assert _make_workspace().is_bootstrap_model("slp-orchestra-mini") is True


class TestLegacyModelNames:
    """Per-project settings resolve across a model rename."""

    def test_a_legacy_named_entry_is_migrated_on_load(self):
        # The field validator canonicalizes on construction, mirroring
        # WorkspaceModel, so a legacy config.json record reports the new name.
        assert ProjectModel(model_name="slp-orchestra-mini").model_name == BOOTSTRAP_MODEL

    def test_lookup_is_alias_aware(self):
        ws = _make_workspace()
        project = ws.projects[0]
        ws.sync_project_models(project)
        ws.set_project_model_enabled(project, "gemma3-4b", False)

        # A model with no alias still resolves by its own name.
        assert ws.project_model(project, "gemma3-4b").enabled is False
        assert ws.effective_model_enabled("gemma3-4b", project) is False
