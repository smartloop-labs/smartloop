//! Port of `tests/unit/test_model_pin_is_workspace_work.py`.
//!
//! Pinning a model across projects is work only the workspace owner can do.
//!
//! `pin_model_for_existing_projects` freezes what a model resolves to in every
//! project *before* the app-default moves, so the new default only reaches
//! projects created after it.  A project with no `ProjectModel` entry reads
//! through to the workspace record, so without the pin an app-default change
//! rewrites the past.
//!
//! It writes every project's `model_settings`, which makes it workspace-owner
//! work in the strict sense: a per-project agent child may write one file, and
//! a supervisor that has delegated its projects may write none of them
//! (`Workspace::writable_projects`).  In either tier the entries appended here
//! are dropped at `save()` while the caller goes on to move the app-default
//! believing the leak was sealed — which is the one outcome the pin exists to
//! prevent.
//!
//! A delegated project pins itself instead, lazily, through
//! `sync_project_models` on its next project-route read or write — where the
//! child that owns the file is the one doing the writing.

use std::collections::BTreeMap;

use serial_test::serial;

use smartloop::config::app_settings::AppSettings;
use smartloop::workspace::context::{delegate_projects, pin_project};
use smartloop::workspace::workspace::{Project, Workspace, WorkspaceModel};

use crate::shared::workspace_support::ProcessScopeReset;

fn bootstrap_model() -> String {
    AppSettings::new().default_model
}

/// Two projects, neither holding an entry for the disabled model.
///
/// "No entry" is the state the pin exists for: both projects currently resolve
/// `qwen2-vl-2b` through the workspace record, so moving that record is what
/// would reach back into them.
fn workspace() -> Workspace {
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
                model_name: "qwen2-vl-2b".to_string(),
                model_id: "Qwen/Qwen2-VL-2B-Instruct".to_string(),
                downloaded: true,
                enabled: false,
                ..Default::default()
            },
        ])
        .with_projects(vec![
            Project {
                id: "alpha".to_string(),
                project_dir: "/tmp/alpha".to_string(),
                current: true,
                name: Some("alpha".to_string()),
                ..Default::default()
            },
            Project {
                id: "beta".to_string(),
                project_dir: "/tmp/beta".to_string(),
                current: false,
                name: Some("beta".to_string()),
                ..Default::default()
            },
        ])
}

fn settings(ws: &Workspace) -> BTreeMap<String, Vec<String>> {
    ws.projects
        .iter()
        .map(|p| (p.id.clone(), p.model_settings.iter().map(|m| m.model_name.clone()).collect()))
        .collect()
}

fn expected(alpha: &[&str], beta: &[&str]) -> BTreeMap<String, Vec<String>> {
    BTreeMap::from([
        ("alpha".to_string(), alpha.iter().map(|s| s.to_string()).collect()),
        ("beta".to_string(), beta.iter().map(|s| s.to_string()).collect()),
    ])
}

mod only_the_workspace_owner_pins {
    use super::*;

    #[test]
    #[serial]
    fn test_a_delegating_supervisor_pins_nothing() {
        let _reset = ProcessScopeReset::new();
        let mut ws = workspace();
        delegate_projects(true);

        ws.pin_model_for_existing_projects("qwen2-vl-2b");

        assert_eq!(settings(&ws), expected(&[], &[]));
    }

    #[test]
    #[serial]
    fn test_a_pinned_child_pins_nothing() {
        let _reset = ProcessScopeReset::new();
        let mut ws = workspace();
        pin_project(Some("alpha"));

        ws.pin_model_for_existing_projects("qwen2-vl-2b");

        assert_eq!(settings(&ws), expected(&[], &[]));
    }

    /// `SLP_SUPERVISOR=0`: unpinned and undelegated, as it always was.
    #[test]
    #[serial]
    fn test_the_single_server_fallback_still_pins() {
        let _reset = ProcessScopeReset::new();
        let mut ws = workspace();

        ws.pin_model_for_existing_projects("qwen2-vl-2b");

        assert_eq!(settings(&ws), expected(&["qwen2-vl-2b"], &["qwen2-vl-2b"]));
        assert!(
            ws.projects
                .iter()
                .all(|p| !ws.project_model(Some(p), "qwen2-vl-2b").unwrap().enabled)
        );
    }

    /// Delegation alone is enough — the supervisor is never also pinned.
    #[test]
    #[serial]
    fn test_the_guard_does_not_depend_on_being_asked_first() {
        let _reset = ProcessScopeReset::new();
        let mut ws = workspace();
        delegate_projects(true);
        pin_project(None);

        ws.pin_model_for_existing_projects("qwen2-vl-2b");

        assert_eq!(settings(&ws), expected(&[], &[]));
    }
}

/// The guard withholds the pin, not the promotion it precedes.
///
/// `promote_model_to_workspace_default` calls the pin and then flips the
/// workspace record.  That record is `config.json`, which is precisely what a
/// delegating supervisor *does* own, so the promotion itself must still land —
/// only the per-project half is someone else's to write.
mod the_app_default_still_moves {
    use super::*;

    #[test]
    #[serial]
    fn test_a_delegating_supervisor_still_promotes() {
        let _reset = ProcessScopeReset::new();
        let mut ws = workspace();
        delegate_projects(true);

        ws.promote_model_to_workspace_default("qwen2-vl-2b");

        assert!(ws.get_model(Some("qwen2-vl-2b")).unwrap().enabled);
        assert_eq!(settings(&ws), expected(&[], &[]));
    }

    /// A child cannot write `config.json` at all, so it must not try.
    #[test]
    #[serial]
    fn test_a_pinned_child_promotes_nothing() {
        let _reset = ProcessScopeReset::new();
        let mut ws = workspace();
        pin_project(Some("alpha"));

        ws.promote_model_to_workspace_default("qwen2-vl-2b");

        assert!(!ws.get_model(Some("qwen2-vl-2b")).unwrap().enabled);
    }

    /// The residual is bounded by `sync_project_models` running first.
    ///
    /// A project that has been through the project route already holds an
    /// entry, and that entry survives the promotion the supervisor goes on to
    /// make — which is the mechanism standing in for the pin under delegation.
    #[test]
    #[serial]
    fn test_a_delegated_project_pins_itself_on_its_next_sync() {
        let _reset = ProcessScopeReset::new();
        let mut ws = workspace();
        assert!(ws.sync_project_models("beta"));
        delegate_projects(true);

        ws.promote_model_to_workspace_default("qwen2-vl-2b");

        assert!(!ws.effective_model_enabled("qwen2-vl-2b", ws.project("beta")));
        // …while a project that never synced reads through to the new default.
        assert!(ws.effective_model_enabled("qwen2-vl-2b", Some(&ws.projects[0])));
    }
}
