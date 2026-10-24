//! Port of `tests/unit/test_model_name_aliases.py`.
//!
//! Tests for the sl-mini rename and its backward-compatibility layer.
//!
//! The orchestrator model's short name changed from `slp-orchestra-mini` to
//! `sl-mini`.  Two things must not move with it: the `model_id` (it names the
//! on-disk model directory, so renaming it would orphan downloaded weights) and
//! the name used against the platform API (`sl-mini` isn't published upstream
//! yet, so artifacts are still fetched under the old name).

use serial_test::serial;

use smartloop::model::spec::MODEL_SPECS;
use smartloop::constants::{
    BOOTSTRAP_MODEL, canonical_model_name, display_model_name, platform_model_name,
};

use crate::shared::workspace_support::EnvGuard;

const LEGACY: &str = "slp-orchestra-mini";

mod canonical_model_name_tests {
    use super::*;

    #[test]
    fn test_legacy_name_maps_to_new_name() {
        assert_eq!(canonical_model_name(LEGACY), "sl-mini");
    }

    #[test]
    fn test_new_name_is_stable() {
        assert_eq!(canonical_model_name("sl-mini"), "sl-mini");
    }

    #[test]
    fn test_bootstrap_constant_is_the_new_name() {
        assert_eq!(BOOTSTRAP_MODEL, "sl-mini");
    }

    #[test]
    fn test_other_models_pass_through() {
        assert_eq!(canonical_model_name("gemma3-4b"), "gemma3-4b");
    }

    /// Parametrized over `[None, ""]`; `None` is not representable for a
    /// `&str` argument, so only the empty string is exercised.
    #[test]
    fn test_empty_passes_through() {
        assert_eq!(canonical_model_name(""), "");
    }
}

mod platform_model_name_tests {
    use super::*;

    #[test]
    fn test_new_name_still_downloads_under_the_old_one() {
        // sl-mini does not exist upstream yet.
        assert_eq!(platform_model_name("sl-mini"), LEGACY);
    }

    #[test]
    fn test_legacy_name_also_maps_to_the_published_name() {
        assert_eq!(platform_model_name(LEGACY), LEGACY);
    }

    #[test]
    fn test_unmapped_models_use_their_own_name() {
        assert_eq!(platform_model_name("gemma3-4b"), "gemma3-4b");
    }
}

mod model_specs {
    use super::*;

    #[test]
    fn test_new_name_resolves() {
        assert!(MODEL_SPECS.get("sl-mini").is_some());
    }

    #[test]
    fn test_legacy_name_resolves_to_the_same_spec() {
        assert_eq!(MODEL_SPECS.get(LEGACY), MODEL_SPECS.get("sl-mini"));
        assert!(MODEL_SPECS.get(LEGACY).is_some());
    }

    #[test]
    fn test_legacy_name_is_a_member() {
        assert!(MODEL_SPECS.contains(LEGACY));
    }

    #[test]
    fn test_alias_is_not_listed_as_a_separate_model() {
        // Iteration must not show the same model twice.
        let keys = MODEL_SPECS.keys();
        assert!(!keys.iter().any(|k| k == LEGACY));
        assert!(keys.iter().any(|k| k == "sl-mini"));
    }

    #[test]
    fn test_model_id_is_unchanged_by_the_rename() {
        // The model_id names the on-disk directory and the upstream bucket path.
        assert_eq!(MODEL_SPECS.get("sl-mini").unwrap().model_id, "smartloop-ai/slp-orchestra-mini");
    }
}

mod display_name {
    use super::*;

    #[test]
    fn test_legacy_record_still_displays_as_sl_mini() {
        assert_eq!(display_model_name(LEGACY), "sl-mini");
    }

    #[test]
    fn test_unknown_model_passes_through() {
        assert_eq!(display_model_name("qwen3-8b"), "qwen3-8b");
    }
}

mod app_settings {
    use super::*;

    use smartloop::config::AppSettings;

    #[test]
    #[serial]
    fn test_defaults_use_the_new_name() {
        let _api = EnvGuard::remove("SLP_API_BASE_MODEL_NAME");
        let _base = EnvGuard::remove("SLP_BASE_MODEL_NAME");
        let settings = AppSettings::new();
        assert_eq!(settings.default_model, "sl-mini");
        assert_eq!(settings.model_name, "sl-mini");
    }

    #[test]
    #[serial]
    fn test_legacy_env_override_is_normalized() {
        // An existing .env may still pin the old name.
        let _api = EnvGuard::set("SLP_API_BASE_MODEL_NAME", LEGACY);
        let _base = EnvGuard::set("SLP_BASE_MODEL_NAME", LEGACY);
        let settings = AppSettings::new();
        assert_eq!(settings.default_model, "sl-mini");
        assert_eq!(settings.model_name, "sl-mini");
    }
}

mod workspace_model_records {
    use super::*;

    use serde_json::json;
    use smartloop::workspace::{Workspace, WorkspaceModel};

    fn legacy_workspace(models: Vec<WorkspaceModel>) -> Workspace {
        Workspace::new("ws")
            .with_workspace_dir("/tmp/ws")
            .with_models_dir("/tmp/ws/models")
            .with_projects(Vec::new())
            .with_models(models)
    }

    fn legacy_record() -> WorkspaceModel {
        // A struct literal skips the canonicalizing validator, so this holds
        // the legacy name exactly as an un-migrated record would.
        WorkspaceModel {
            model_name: LEGACY.to_string(),
            model_id: "smartloop-ai/slp-orchestra-mini".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_legacy_record_is_migrated_on_load() {
        let model: WorkspaceModel = serde_json::from_value(json!({
            "model_name": LEGACY,
            "model_id": "smartloop-ai/slp-orchestra-mini",
            "downloaded": true,
        }))
        .unwrap();
        assert_eq!(model.model_name, "sl-mini");
        // model_id is the on-disk directory — it must survive untouched.
        assert_eq!(model.model_id, "smartloop-ai/slp-orchestra-mini");
    }

    #[test]
    fn test_lookup_finds_a_legacy_record_by_the_new_name() {
        let ws = legacy_workspace(vec![legacy_record()]);
        assert!(ws.get_model(Some("sl-mini")).is_some());
        assert!(ws.get_model(Some(LEGACY)).is_some());
    }

    #[test]
    fn test_add_model_does_not_duplicate_a_legacy_record() {
        let mut ws = legacy_workspace(vec![legacy_record()]);
        ws.add_model("sl-mini", "smartloop-ai/slp-orchestra-mini", None, true);
        assert_eq!(ws.models.len(), 1);
    }

    #[test]
    fn test_add_model_stores_the_canonical_name() {
        let mut ws = legacy_workspace(Vec::new());
        let entry = ws.add_model(LEGACY, "smartloop-ai/slp-orchestra-mini", None, true);
        assert_eq!(entry.model_name, "sl-mini");
    }
}

/// The downloader must ask the platform for `slp-orchestra-mini`.
///
/// `sl-mini` is not published upstream yet, so a request under the new name
/// would 404 — the artifacts still live under the old one.
mod download_uses_the_published_name {
    use super::*;

    use std::sync::Arc;

    use parking_lot::Mutex;
    use serde_json::{Value, json};
    use smartloop::model::downloader::artifact_downloader::SLPArtifactDownloader;
    use smartloop::model::downloader::artifact_source::{
        ArtifactMap, ArtifactSource, BootstrapArtifactSource,
    };
    use smartloop::model::downloader::slp_downloader::SLPDownloader;
    use smartloop::utils::log_utils::ConsoleLogger;

    struct FakeSource {
        asked: Arc<Mutex<Vec<String>>>,
    }

    impl ArtifactSource for FakeSource {
        fn get_artifacts(&self, model_name: &str) -> anyhow::Result<ArtifactMap> {
            self.asked.lock().push(model_name.to_string());
            Ok(ArtifactMap::new()) // no GGUFs -> download() returns early
        }
    }

    struct FakeBootstrapSource {
        published: Vec<Value>,
        asked: Arc<Mutex<Vec<String>>>,
    }

    impl ArtifactSource for FakeBootstrapSource {
        fn get_artifacts(&self, model_name: &str) -> anyhow::Result<ArtifactMap> {
            self.asked.lock().push(model_name.to_string());
            Ok(ArtifactMap::new())
        }
    }

    impl BootstrapArtifactSource for FakeBootstrapSource {
        fn list_models(&self) -> anyhow::Result<Vec<Value>> {
            Ok(self.published.clone())
        }
    }

    #[test]
    fn test_download_requests_the_legacy_name() {
        let tmp = tempfile::tempdir().unwrap();
        let asked = Arc::new(Mutex::new(Vec::new()));
        let downloader = SLPArtifactDownloader::for_tests(
            Some(tmp.path().to_path_buf()),
            Arc::new(ConsoleLogger::new()),
            None,
        )
        .with_model_source(Arc::new(FakeSource { asked: Arc::clone(&asked) }));

        let _ = downloader.download("smartloop-ai/slp-orchestra-mini", None);
        assert_eq!(*asked.lock(), vec![LEGACY.to_string()]);
    }

    #[test]
    fn test_bootstrap_matches_the_published_name() {
        let tmp = tempfile::tempdir().unwrap();
        let published = vec![json!({"name": LEGACY, "artifacts": {"m.gguf": "https://u/m.gguf"}})];
        let asked = Arc::new(Mutex::new(Vec::new()));
        let downloader = SLPArtifactDownloader::for_tests(
            Some(tmp.path().to_path_buf()),
            Arc::new(ConsoleLogger::new()),
            None,
        )
        .with_bootstrap_source(Arc::new(FakeBootstrapSource { published, asked: Arc::clone(&asked) }));

        // Asking for the new name must find the entry published under the old one.
        let err = downloader.bootstrap_download(Some("sl-mini"), None).expect_err("no GGUF artifact");
        assert!(err.to_string().contains("No GGUF artifact"), "unexpected error: {err}");
        assert_eq!(*asked.lock(), vec![LEGACY.to_string()]);
    }
}
