//! Port of `tests/unit/test_model_access_level.py`.
//!
//! One unknown access level must not take down the whole model picker.
//!
//! The catalog is served by the platform and gains tiers without a client
//! release — `internal` arrived that way and failed `ModelEntry` validation,
//! which fails the `list[ModelEntry]` response model for *every* model in the
//! list, so /v1/models 500'd instead of dropping one entry.

use serde_json::json;
use smartloop::api::routers::models::{AccessLevel, ModelEntry};

/// `ModelEntry(name=..., model_id=..., access_level=level)` — built through
/// validation, which is where the unknown-tier tolerance lives.
fn entry(level: &str) -> ModelEntry {
    serde_json::from_value(json!({
        "name": "qwen3.5-2b",
        "model_id": "Qwen/Qwen3.5-2B",
        "access_level": level,
    }))
    .expect("a catalog row with any access level must validate")
}

#[test]
fn test_internal_is_a_known_tier() {
    assert_eq!(entry("internal").access_level, Some(AccessLevel::Internal));
}

#[test]
fn test_the_public_tiers_still_parse() {
    assert_eq!(entry("free").access_level, Some(AccessLevel::Free));
    assert_eq!(entry("pro").access_level, Some(AccessLevel::Paid));
}

/// `premium` never existed on the platform's `ModelAccessLevel`, so it is not
/// a member here — it takes the unknown-tier path.
#[test]
fn test_the_bogus_premium_tier_degrades() {
    assert_eq!(entry("premium").access_level, None);
}

#[test]
fn test_a_future_tier_degrades_instead_of_failing() {
    assert_eq!(entry("enterprise-preview").access_level, None);
}

#[test]
fn test_an_absent_level_stays_absent() {
    assert_eq!(ModelEntry::new("x", "y").access_level, None);
    // And through validation, with the field missing entirely.
    let parsed: ModelEntry = serde_json::from_value(json!({"name": "x", "model_id": "y"})).unwrap();
    assert_eq!(parsed.access_level, None);
}

#[test]
fn test_free_login_is_a_known_tier() {
    assert_eq!(entry("free_login").access_level, Some(AccessLevel::FreeLogin));
}

mod test_model_kind {
    use serde_json::json;
    use smartloop::api::routers::models::ModelEntry;
    use smartloop::types::ModelKind;

    fn kind(row: serde_json::Value) -> ModelEntry {
        serde_json::from_value(row).expect("a catalog row must validate")
    }

    #[test]
    fn test_a_remote_row_carries_its_url_and_context() {
        let parsed = kind(json!({
            "name": "kimi-k2", "model_id": null, "type": "remote",
            "url": "https://api.smartloop.ai", "context_length": 131072,
        }));
        assert_eq!(parsed.kind, ModelKind::Remote);
        assert_eq!(parsed.model_id, "");
        assert_eq!(parsed.url.as_deref(), Some("https://api.smartloop.ai"));
        assert_eq!(parsed.context_length, Some(131072));
    }

    #[test]
    fn test_a_row_without_a_type_is_local() {
        let parsed = kind(json!({"name": "gemma3-4b", "model_id": "google/gemma-3-4b-it"}));
        assert_eq!(parsed.kind, ModelKind::Local);
        assert_eq!(parsed.url, None);
    }

    #[test]
    fn test_an_unknown_type_degrades_to_local() {
        let parsed = kind(json!({"name": "x", "model_id": "y", "type": "hybrid"}));
        assert_eq!(parsed.kind, ModelKind::Local);
    }

    #[test]
    fn test_the_type_serializes_under_its_wire_name() {
        let mut entry = ModelEntry::new("kimi-k2", "");
        entry.kind = ModelKind::Remote;
        let value = serde_json::to_value(&entry).unwrap();
        assert_eq!(value["type"], "remote");
        assert!(value.get("kind").is_none());
    }

    #[test]
    fn test_workspace_and_project_entries_carry_the_remote_fields() {
        use smartloop::api::routers::models::{ProjectModelEntry, WorkspaceModelEntry};

        let workspace = WorkspaceModelEntry {
            model_name: "kimi-k2".into(),
            model_id: String::new(),
            downloaded: true,
            is_active: false,
            enabled: true,
            capabilities: vec!["general".into()],
            thinking: None,
            kind: ModelKind::Remote,
            url: Some("https://api.smartloop.ai".into()),
            context_length: Some(131072),
            param_size_b: Some(1000.0),
        };
        let value = serde_json::to_value(&workspace).unwrap();
        assert_eq!(value["type"], "remote");
        assert_eq!(value["url"], "https://api.smartloop.ai");
        assert_eq!(value["context_length"], 131072);
        assert_eq!(value["param_size_b"], 1000.0);

        let project = ProjectModelEntry {
            model_name: "kimi-k2".into(),
            model_id: String::new(),
            downloaded: true,
            is_active: false,
            enabled: true,
            capabilities: vec![],
            settings: Default::default(),
            kind: ModelKind::Remote,
            url: Some("https://api.smartloop.ai".into()),
            context_length: None,
            param_size_b: None,
        };
        let value = serde_json::to_value(&project).unwrap();
        assert_eq!(value["type"], "remote");
        assert_eq!(value["context_length"], serde_json::Value::Null);
    }
}
