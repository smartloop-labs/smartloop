//! Port of `tests/unit/test_workspace_project_models.py`.
//!
//! Unit tests for per-project model settings on the Workspace.

use std::collections::{BTreeSet, HashMap};

use serial_test::serial;

use smartloop::config::app_settings::AppSettings;
use smartloop::workspace::workspace::{Project, ProjectModel, Workspace, WorkspaceModel};

use crate::shared::workspace_support::ProcessScopeReset;

fn bootstrap_model() -> String {
    AppSettings::new().default_model
}

fn project(pid: &str, current: bool) -> Project {
    Project {
        id: pid.to_string(),
        project_dir: format!("/tmp/{pid}"),
        current,
        name: Some(pid.to_string()),
        ..Default::default()
    }
}

/// In-memory workspace with two selectable models and one project.
///
/// The always-on bootstrap model is registered too, so the tests below also
/// assert it stays out of the per-project settings.
fn make_workspace() -> Workspace {
    Workspace::new("ws1")
        .with_models(vec![
            WorkspaceModel {
                model_name: bootstrap_model(),
                model_id: "smartloop/slp-orchestra-mini".to_string(),
                downloaded: true,
                enabled: true,
                ..Default::default()
            },
            WorkspaceModel {
                model_name: "gemma3-4b".to_string(),
                model_id: "google/gemma-3-4b-it".to_string(),
                downloaded: true,
                enabled: true,
                ..Default::default()
            },
            WorkspaceModel {
                model_name: "qwen2-vl-2b".to_string(),
                model_id: "Qwen/Qwen2-VL-2B-Instruct".to_string(),
                downloaded: true,
                enabled: false,
                is_active: true,
                ..Default::default()
            },
        ])
        .with_projects(vec![project("p1", true)])
}

fn names(project: &Project) -> BTreeSet<String> {
    project.model_settings.iter().map(|m| m.model_name.clone()).collect()
}

mod sync_project_models {
    use super::*;

    #[test]
    #[serial]
    fn test_seeds_project_settings_from_workspace_defaults() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();

        ws.sync_project_models("p1");

        let project = &ws.projects[0];
        let entries: HashMap<&str, &ProjectModel> =
            project.model_settings.iter().map(|m| (m.model_name.as_str(), m)).collect();
        assert_eq!(
            entries.keys().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from(["gemma3-4b", "qwen2-vl-2b"])
        );
        // Workspace (app-default) state is copied across: disabled stays
        // disabled, active stays active.  Download state is workspace-wide.
        assert!(entries["gemma3-4b"].enabled);
        assert!(!entries["qwen2-vl-2b"].enabled);
        assert!(entries["qwen2-vl-2b"].is_active);
    }

    #[test]
    #[serial]
    fn test_existing_project_entries_are_not_overwritten() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();
        ws.sync_project_models("p1");
        // The project overrides the workspace default: enable qwen2-vl-2b.
        ws.set_project_model_enabled("p1", "qwen2-vl-2b", true);

        // Re-seeding (e.g. a later lazy sync) must not clobber the override.
        ws.sync_project_models("p1");

        assert!(ws.project_model(ws.project("p1"), "qwen2-vl-2b").unwrap().enabled);
        // …while the workspace default is untouched.
        assert!(!ws.get_model(Some("qwen2-vl-2b")).unwrap().enabled);
    }

    #[test]
    #[serial]
    fn test_sync_is_idempotent() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();
        ws.sync_project_models("p1");
        let count = ws.projects[0].model_settings.len();

        ws.sync_project_models("p1");

        assert_eq!(ws.projects[0].model_settings.len(), count);
    }
}

mod effective_resolution {
    use super::*;

    #[test]
    #[serial]
    fn test_project_override_wins_over_workspace_default() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();

        assert!(!ws.effective_model_enabled("qwen2-vl-2b", None)); // workspace default
        ws.set_project_model_enabled("p1", "qwen2-vl-2b", true);
        assert!(ws.effective_model_enabled("qwen2-vl-2b", ws.project("p1"))); // project override
    }

    #[test]
    #[serial]
    fn test_falls_back_to_workspace_default_when_project_has_no_entry() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();
        // No sync yet — the project has no entries at all.
        ws.projects.push(project("p2", false));
        let p2 = ws.project("p2");

        assert!(ws.effective_model_enabled("gemma3-4b", p2));
        assert!(ws.effective_model_downloaded("gemma3-4b"));
        assert!(!ws.effective_model_active("gemma3-4b", p2));
    }

    #[test]
    #[serial]
    fn test_unknown_model_defaults() {
        let _reset = ProcessScopeReset::new();
        let ws = make_workspace();
        let p1 = ws.project("p1");

        assert!(ws.effective_model_enabled("nope", p1));
        assert!(!ws.effective_model_downloaded("nope"));
        assert!(!ws.effective_model_active("nope", p1));
    }

    #[test]
    #[serial]
    fn test_no_project_uses_workspace_defaults() {
        let _reset = ProcessScopeReset::new();
        let ws = make_workspace();
        assert!(ws.effective_model_enabled("gemma3-4b", None));
        assert!(ws.effective_model_downloaded("gemma3-4b"));
        assert!(!ws.effective_model_enabled("qwen2-vl-2b", None));
    }
}

mod project_setters {
    use super::*;

    #[test]
    #[serial]
    fn test_set_enabled_upserts_entry() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();
        assert!(ws.project_model(ws.project("p1"), "gemma3-4b").is_none());

        let pm = ws.set_project_model_enabled("p1", "gemma3-4b", false).unwrap();

        assert!(!pm.enabled);
        assert!(!ws.effective_model_enabled("gemma3-4b", ws.project("p1")));
        // Workspace default untouched.
        assert!(ws.get_model(Some("gemma3-4b")).unwrap().enabled);
    }

    /// Weights are added/removed workspace-wide, so there is no override.
    ///
    /// A project entry carries only enabled/is_active/settings — nothing a
    /// project writes can change what is on disk.
    #[test]
    #[serial]
    fn test_download_state_is_not_a_per_project_setting() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();
        ws.sync_project_models("p1");

        // `ProjectModel` has no `downloaded` field (and `Workspace` no
        // `set_project_model_downloaded`) — checked here on the wire form.
        let dumped = serde_json::to_value(ProjectModel::new("gemma3-4b")).unwrap();
        assert!(dumped.get("downloaded").is_none());
        // Disabling the model for the project leaves disk state alone.
        ws.set_project_model_enabled("p1", "gemma3-4b", false);
        assert!(ws.effective_model_downloaded("gemma3-4b"));
    }

    #[test]
    #[serial]
    fn test_set_active_deactivates_other_project_models() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();
        ws.sync_project_models("p1");

        ws.set_project_model_active("p1", "gemma3-4b", true);

        assert!(ws.effective_model_active("gemma3-4b", ws.project("p1")));
        assert!(!ws.effective_model_active("qwen2-vl-2b", ws.project("p1")));
    }

    #[test]
    #[serial]
    fn test_projects_are_independent() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();
        ws.projects.push(project("p2", false));

        ws.set_project_model_enabled("p1", "gemma3-4b", false);
        ws.set_project_model_active("p1", "gemma3-4b", true);

        assert!(!ws.effective_model_enabled("gemma3-4b", ws.project("p1")));
        assert!(ws.effective_model_enabled("gemma3-4b", ws.project("p2"))); // still workspace default
        assert!(ws.effective_model_active("gemma3-4b", ws.project("p1")));
        assert!(!ws.effective_model_active("gemma3-4b", ws.project("p2")));
    }
}

/// Downloading a model makes it a default for projects created after.
///
/// The trigger is the download, not the enable: `/v1/init` calls this once the
/// weights land, and the project route deliberately does not.
mod promote_to_workspace_default {
    use super::*;

    #[test]
    #[serial]
    fn test_promote_flips_the_workspace_default() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();
        assert!(!ws.get_model(Some("qwen2-vl-2b")).unwrap().enabled);

        ws.promote_model_to_workspace_default("qwen2-vl-2b");

        assert!(ws.get_model(Some("qwen2-vl-2b")).unwrap().enabled);
    }

    /// An untouched project is 'unspecified', not 'off' — write it down.
    ///
    /// Without the pin it would read through to the new default and silently
    /// gain a model the user never enabled there.
    #[test]
    #[serial]
    fn test_existing_project_without_an_entry_is_pinned_off() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();
        ws.projects.push(project("p2", false));
        assert!(ws.project_model(ws.project("p2"), "qwen2-vl-2b").is_none());

        ws.promote_model_to_workspace_default("qwen2-vl-2b");

        assert!(!ws.project_model(ws.project("p2"), "qwen2-vl-2b").unwrap().enabled);
        assert!(!ws.effective_model_enabled("qwen2-vl-2b", ws.project("p2")));
        // …and a later lazy sync must not undo the pin.
        ws.sync_project_models("p2");
        assert!(!ws.effective_model_enabled("qwen2-vl-2b", ws.project("p2")));
    }

    #[test]
    #[serial]
    fn test_project_created_after_promotion_inherits_it() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();
        ws.promote_model_to_workspace_default("qwen2-vl-2b");

        ws.projects.push(project("p3", false));
        ws.sync_project_models("p3");

        assert!(ws.effective_model_enabled("qwen2-vl-2b", ws.project("p3")));
    }

    /// Pinning one model must not detach the rest from future defaults.
    #[test]
    #[serial]
    fn test_pin_touches_only_the_promoted_model() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();
        ws.projects.push(project("p2", false));

        ws.promote_model_to_workspace_default("qwen2-vl-2b");

        assert_eq!(names(ws.project("p2").unwrap()), BTreeSet::from(["qwen2-vl-2b".to_string()]));
    }

    #[test]
    #[serial]
    fn test_bootstrap_model_is_not_promotable() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();

        ws.promote_model_to_workspace_default(&bootstrap_model());

        // Already always-on, and it still gets no per-project entry.
        assert!(ws.get_model(Some(&bootstrap_model())).unwrap().enabled);
        assert!(ws.project_model(ws.project("p1"), &bootstrap_model()).is_none());
    }

    /// The early return keeps a no-op enable from pinning every project.
    #[test]
    #[serial]
    fn test_already_enabled_model_writes_nothing() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();
        ws.projects.push(project("p2", false));

        ws.promote_model_to_workspace_default("gemma3-4b");

        assert!(ws.get_model(Some("gemma3-4b")).unwrap().enabled);
        assert!(ws.project("p2").unwrap().model_settings.is_empty());
    }

    #[test]
    #[serial]
    fn test_unknown_model_is_a_noop() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();

        ws.promote_model_to_workspace_default("nope");

        assert!(ws.project("p1").unwrap().model_settings.is_empty());
    }
}

mod project_aware_selection {
    use super::*;

    #[test]
    #[serial]
    fn test_largest_downloaded_respects_project_override() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();
        ws.sync_project_models("p1");
        // Workspace default: gemma3-4b (larger, enabled) wins.
        assert_eq!(ws.largest_downloaded_model(ws.project("p1")).unwrap().model_name, "gemma3-4b");

        // Project disables gemma3-4b → qwen2-vl-2b (the only downloaded one) wins.
        ws.set_project_model_enabled("p1", "gemma3-4b", false);
        ws.set_project_model_enabled("p1", "qwen2-vl-2b", true);
        assert_eq!(ws.largest_downloaded_model(ws.project("p1")).unwrap().model_name, "qwen2-vl-2b");
    }

    #[test]
    #[serial]
    fn test_largest_downloaded_ignores_project_when_none_passed() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();
        ws.set_project_model_enabled("p1", "gemma3-4b", false);

        // Without a project, selection falls back to workspace defaults.
        assert_eq!(ws.largest_downloaded_model(None).unwrap().model_name, "gemma3-4b");
    }

    #[test]
    #[serial]
    fn test_ranked_models_for_filters_on_project_state() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();

        // qwen2-vl-2b is the only vision-capable model (gemma3-4b isn't tagged
        // as vision) and it's disabled by default → no candidates.
        assert!(ws.ranked_models_for("vision", ws.project("p1")).is_empty());

        ws.set_project_model_enabled("p1", "qwen2-vl-2b", true);
        let ranked = ws.ranked_models_for("vision", ws.project("p1"));
        assert_eq!(ranked.iter().map(|m| m.model_name.as_str()).collect::<Vec<_>>(), vec!["qwen2-vl-2b"]);
    }
}

/// The orchestrator model is always enabled and can't be turned off.
///
/// It is filtered out of every models list response, so it must not get a
/// per-project entry either — an `enabled=false` override would otherwise
/// shadow its always-on workspace state in the selection paths.
mod bootstrap_model_is_not_a_per_project_setting {
    use super::*;

    #[test]
    #[serial]
    fn test_sync_does_not_seed_an_entry_for_it() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();

        ws.sync_project_models("p1");

        assert!(!names(ws.project("p1").unwrap()).contains(&bootstrap_model()));
    }

    #[test]
    #[serial]
    fn test_effective_state_falls_through_to_the_workspace_record() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();
        ws.sync_project_models("p1");

        assert!(ws.project_model(ws.project("p1"), &bootstrap_model()).is_none());
        assert!(ws.effective_model_enabled(&bootstrap_model(), ws.project("p1")));
        assert!(ws.effective_model_downloaded(&bootstrap_model()));
    }

    #[test]
    #[serial]
    fn test_sync_prunes_an_entry_written_by_an_earlier_build() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();
        // A workspace written before the exclusion existed, with the entry
        // already flipped off — the state that would break model selection.
        ws.projects[0]
            .model_settings
            .push(ProjectModel { enabled: false, ..ProjectModel::new(&bootstrap_model()) });

        ws.sync_project_models("p1");

        assert!(ws.project_model(ws.project("p1"), &bootstrap_model()).is_none());
        assert!(ws.effective_model_enabled(&bootstrap_model(), ws.project("p1")));
    }

    fn setters_refuse_to_create_an_entry(setter: &str, value: bool) {
        let mut ws = make_workspace();
        ws.sync_project_models("p1");

        let result = match setter {
            "set_project_model_enabled" => ws.set_project_model_enabled("p1", &bootstrap_model(), value),
            "set_project_model_active" => ws.set_project_model_active("p1", &bootstrap_model(), value),
            other => panic!("unknown setter {other}"),
        };
        assert!(result.is_none());
        assert!(ws.project_model(ws.project("p1"), &bootstrap_model()).is_none());
        assert!(ws.effective_model_enabled(&bootstrap_model(), ws.project("p1")));
    }

    #[test]
    #[serial]
    fn test_setters_refuse_to_create_an_entry_set_project_model_enabled() {
        let _reset = ProcessScopeReset::new();
        setters_refuse_to_create_an_entry("set_project_model_enabled", false);
    }

    #[test]
    #[serial]
    fn test_setters_refuse_to_create_an_entry_set_project_model_active() {
        let _reset = ProcessScopeReset::new();
        setters_refuse_to_create_an_entry("set_project_model_active", true);
    }

    #[test]
    #[serial]
    fn test_selection_still_sees_it_after_a_project_disables_everything_else() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();
        ws.sync_project_models("p1");
        ws.set_project_model_enabled("p1", "gemma3-4b", false);
        ws.set_project_model_enabled("p1", "qwen2-vl-2b", false);

        assert!(ws.effective_model_enabled(&bootstrap_model(), ws.project("p1")));
    }

    /// A workspace from before the `sl-mini` rename must still heal.
    ///
    /// Pre-rename builds wrote `model_settings` entries keyed on
    /// `slp-orchestra-mini`.  If the bootstrap check were not alias-aware the
    /// entry would survive the prune and its `enabled=false` would shadow the
    /// always-on workspace record everywhere selection resolves through
    /// `effective_model_enabled`.
    #[test]
    #[serial]
    fn test_sync_prunes_an_entry_written_under_the_legacy_name() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();
        // Written as on disk: the raw legacy name, not canonicalized.
        ws.projects[0].model_settings.push(ProjectModel {
            model_name: "slp-orchestra-mini".to_string(),
            enabled: false,
            ..Default::default()
        });

        ws.sync_project_models("p1");

        assert!(ws.project_model(ws.project("p1"), &bootstrap_model()).is_none());
        assert!(ws.project_model(ws.project("p1"), "slp-orchestra-mini").is_none());
        assert!(ws.effective_model_enabled(&bootstrap_model(), ws.project("p1")));
    }

    #[test]
    #[serial]
    fn test_is_bootstrap_model_accepts_the_legacy_name() {
        let _reset = ProcessScopeReset::new();
        assert!(make_workspace().is_bootstrap_model("slp-orchestra-mini"));
    }
}

/// Per-project settings resolve across a model rename.
mod legacy_model_names {
    use super::*;

    #[test]
    #[serial]
    fn test_a_legacy_named_entry_is_migrated_on_load() {
        let _reset = ProcessScopeReset::new();
        // Canonicalized on construction, mirroring WorkspaceModel, so a legacy
        // config.json record reports the new name.
        assert_eq!(ProjectModel::new("slp-orchestra-mini").model_name, bootstrap_model());
        let loaded: ProjectModel =
            serde_json::from_value(serde_json::json!({"model_name": "slp-orchestra-mini"})).unwrap();
        assert_eq!(loaded.model_name, bootstrap_model());
    }

    #[test]
    #[serial]
    fn test_lookup_is_alias_aware() {
        let _reset = ProcessScopeReset::new();
        let mut ws = make_workspace();
        ws.sync_project_models("p1");
        ws.set_project_model_enabled("p1", "gemma3-4b", false);

        // A model with no alias still resolves by its own name.
        assert!(!ws.project_model(ws.project("p1"), "gemma3-4b").unwrap().enabled);
        assert!(!ws.effective_model_enabled("gemma3-4b", ws.project("p1")));
    }
}
