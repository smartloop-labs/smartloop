//! Model specs: the built-in seeds, and catalog rows laid over them.

use serde_json::{json, Value};
use smartloop::model::spec::{seed_specs, CatalogModel, ModelRegistry, ModelSpec};
use smartloop::platform::client::JsonObject;
use smartloop::types::{Attention, ModelCapability, ModelKind, ModelType};

use ModelCapability::{Coding, Dp, General, Reasoning, Thinking, Vision};

fn row(value: Value) -> CatalogModel {
    let Value::Object(obj) = value else { panic!("row must be an object") };
    CatalogModel::from_row(&obj).expect("row parses")
}

fn obj(value: Value) -> JsonObject {
    let Value::Object(obj) = value else { panic!("row must be an object") };
    obj
}

#[test]
fn test_seeds_pin_every_built_in_model() {
    // (name, model_id, version, size, capabilities, handler) for each model
    // this build knows offline; the catalog publishes the same values.
    let expected: Vec<(&str, &str, i64, f64, Vec<ModelCapability>, Option<&str>)> = vec![
        ("sl-mini", "smartloop-ai/slp-orchestra-mini", 3, 1.0, vec![General], None),
        ("gemma3-4b", "google/gemma-3-4b-it", 3, 4.0, vec![General, Dp], Some("MTMDChatHandler")),
        ("gemma4-e2b", "google/gemma-4-e2b-it", 4, 2.0, vec![General, Dp, Vision, Thinking], Some("Gemma4ChatHandler")),
        ("gemma4-e4b", "google/gemma-4-e4b-it", 4, 4.0, vec![General, Dp, Vision, Thinking], Some("Gemma4ChatHandler")),
        ("qwen2-vl-2b", "Qwen/Qwen2-VL-2B-Instruct", 2, 2.0, vec![Vision], Some("Qwen25VLChatHandler")),
        ("ministral3-3b", "mistralai/Ministral-3-3B-Instruct-2512", 3, 3.0, vec![General, Dp, Vision], Some("MTMDChatHandler")),
        (
            "qwen3.5-2b",
            "Qwen/Qwen3.5-2B",
            3,
            2.0,
            vec![General, Reasoning, Dp, Vision, Thinking],
            Some("MTMDChatHandler"),
        ),
    ];
    let seeds = seed_specs();
    assert_eq!(seeds.len(), expected.len());
    for (spec, (name, model_id, version, size, caps, handler)) in seeds.iter().zip(expected) {
        assert_eq!(spec.name, name);
        assert_eq!(spec.model_id, model_id, "{name}");
        assert_eq!(spec.kind, ModelKind::Local, "{name}");
        assert_eq!(spec.model_version, version, "{name}");
        assert_eq!(spec.param_size_b, size, "{name}");
        assert_eq!(spec.capabilities, caps, "{name}");
        assert!(!spec.thinking_default, "{name}");
        assert!(!spec.param_size_estimated, "{name}");
        assert_eq!(spec.vision_chat_handler.as_deref(), handler, "{name}");
    }
}

#[test]
fn test_modality_follows_the_vision_handler() {
    let registry = ModelRegistry::seeded();
    assert_eq!(registry.get("sl-mini").unwrap().model_type(), ModelType::Text);
    assert_eq!(registry.get("gemma4-e2b").unwrap().model_type(), ModelType::MultiModal);
}

#[test]
fn test_lookup_canonicalizes_names() {
    let registry = ModelRegistry::seeded();
    let spec = registry.get("slp-orchestra-mini").unwrap();
    assert_eq!(spec.name, "sl-mini");
    assert_eq!(registry.get_by_id("google/gemma-4-e4b-it").unwrap().name, "gemma4-e4b");
    assert!(registry.get_by_id("").is_none());
    assert!(registry.get("nope").is_none());
}

#[test]
fn test_a_catalog_row_overrides_only_what_it_declares() {
    let mut registry = ModelRegistry::seeded();
    let added = registry.hydrate([row(json!({
        "name": "gemma4-e2b",
        "model_id": "google/gemma-4-e2b-it",
        "param_size_b": 2.5,
        "thinking_default": true,
        "capabilities": [],
        "vision_chat_handler": null,
    }))]);
    assert_eq!(added, 0);

    let spec = registry.get("gemma4-e2b").unwrap();
    assert_eq!(spec.param_size_b, 2.5);
    assert!(spec.thinking_default);
    // Undeclared (absent, null or empty) keeps the seed.
    assert_eq!(spec.model_version, 4);
    assert_eq!(spec.capabilities, vec![General, Dp, Vision, Thinking]);
    assert_eq!(spec.vision_chat_handler.as_deref(), Some("Gemma4ChatHandler"));
}

#[test]
fn test_a_platform_name_lands_on_the_canonical_model() {
    let mut registry = ModelRegistry::seeded();
    let before = registry.len();
    registry.hydrate([row(json!({"name": "slp-orchestra-mini", "model_version": 5}))]);
    assert_eq!(registry.len(), before);
    assert_eq!(registry.get("sl-mini").unwrap().model_version, 5);
}

#[test]
fn test_a_new_local_model_needs_only_a_catalog_row() {
    let mut registry = ModelRegistry::seeded();
    let added = registry.hydrate([row(json!({
        "name": "gemma4-e8b",
        "model_id": "google/gemma-4-e8b-it",
        "type": "local",
        "param_size_b": 8.0,
        "model_version": 4,
        "capabilities": ["general", "dp", "vision"],
        "thinking_default": false,
        "vision_chat_handler": "Gemma4ChatHandler",
    }))]);
    assert_eq!(added, 1);

    let spec = registry.get("gemma4-e8b").unwrap();
    assert_eq!(spec.model_id, "google/gemma-4-e8b-it");
    assert_eq!(spec.kind, ModelKind::Local);
    assert_eq!(spec.param_size_b, 8.0);
    assert_eq!(spec.capabilities, vec![General, Dp, Vision]);
    assert_eq!(spec.model_type(), ModelType::MultiModal);
    assert_eq!(registry.get_by_id("google/gemma-4-e8b-it").unwrap().name, "gemma4-e8b");
}

#[test]
fn test_a_remote_model_carries_size_and_endpoint() {
    let mut registry = ModelRegistry::seeded();
    registry.hydrate([row(json!({
        "name": "glm-5.3",
        "model_id": null,
        "type": "remote",
        "url": "https://api.smartloop.ai",
        "capabilities": ["general", "coding"],
        "context_length": 1048576,
        "param_size_b": 355.0,
        "param_size_estimated": true,
        "model_version": 5,
    }))]);

    let spec = registry.get("glm-5.3").unwrap();
    assert!(spec.is_remote());
    assert_eq!(spec.model_id, "");
    assert_eq!(spec.url.as_deref(), Some("https://api.smartloop.ai"));
    assert_eq!(spec.context_length, Some(1048576));
    assert_eq!(spec.param_size_b, 355.0);
    assert!(spec.param_size_estimated);
    assert_eq!(spec.capabilities, vec![General, Coding]);
    // No weights, so nothing resolves by an empty id.
    assert!(registry.get_by_id("").is_none());
}

#[test]
fn test_a_mixture_of_experts_model_ranks_on_active_params() {
    let mut registry = ModelRegistry::default();
    registry.hydrate([row(json!({
        "name": "moe", "type": "remote", "param_size_b": 428.0, "active_param_size_b": 23.0,
        "attention": "hybrid",
    }))]);
    let spec = registry.get("moe").unwrap();
    assert_eq!(spec.param_size_b, 428.0);
    assert_eq!(spec.active_param_size_b, Some(23.0));
    assert_eq!(spec.ranking_size_b(), 23.0);
    assert_eq!(spec.attention, Some(Attention::Hybrid));

    // A dense model ranks on its parameter count.
    assert_eq!(ModelRegistry::seeded().get("gemma4-e4b").unwrap().ranking_size_b(), 4.0);
}

#[test]
fn test_a_reasoning_budget_is_carried_only_when_declared() {
    let mut registry = ModelRegistry::default();
    registry.hydrate([
        row(json!({"name": "thinker", "type": "remote", "reasoning_budget_tokens": 1024})),
        row(json!({"name": "plain", "type": "remote"})),
    ]);
    assert_eq!(registry.get("thinker").unwrap().reasoning_budget_tokens, Some(1024));
    assert_eq!(registry.get("plain").unwrap().reasoning_budget_tokens, None);
}

#[test]
fn test_every_attention_type_parses() {
    for (value, attention) in [
        ("full", Attention::Full),
        ("sliding_window", Attention::SlidingWindow),
        ("linear", Attention::Linear),
        ("hybrid", Attention::Hybrid),
        ("sparse", Attention::Sparse),
    ] {
        assert_eq!(Attention::from_value(value), Some(attention), "{value}");
        assert_eq!(serde_json::to_value(attention).unwrap(), json!(value));
    }
}

#[test]
fn test_an_unknown_attention_type_is_ignored() {
    let mut registry = ModelRegistry::default();
    registry.hydrate([row(json!({"name": "x", "attention": "quantum"}))]);
    assert_eq!(registry.get("x").unwrap().attention, None);
}

#[test]
fn test_an_undeclared_model_defaults_to_general() {
    let mut registry = ModelRegistry::default();
    registry.hydrate([row(json!({"name": "mystery", "model_id": "org/mystery"}))]);
    let spec = registry.get("mystery").unwrap();
    assert_eq!(*spec, ModelSpec::local("mystery", "org/mystery"));
    assert_eq!(spec.capabilities, vec![General]);
}

#[test]
fn test_unknown_capabilities_are_dropped() {
    let mut registry = ModelRegistry::default();
    registry.hydrate([row(json!({"name": "x", "capabilities": ["general", "telepathy"]}))]);
    assert_eq!(registry.get("x").unwrap().capabilities, vec![General]);
}

#[test]
fn test_a_replaced_model_id_stops_resolving() {
    let mut registry = ModelRegistry::default();
    registry.hydrate([row(json!({"name": "x", "model_id": "org/old"}))]);
    registry.hydrate([row(json!({"name": "x", "model_id": "org/new"}))]);
    assert!(registry.get_by_id("org/old").is_none());
    assert_eq!(registry.get_by_id("org/new").unwrap().name, "x");
    assert_eq!(registry.len(), 1);
}

#[test]
fn test_a_later_row_replaces_what_an_earlier_one_said() {
    let mut registry = ModelRegistry::seeded();
    registry.hydrate([row(json!({"name": "remote-x", "type": "remote", "url": "https://old.example"}))]);
    registry.hydrate([row(json!({"name": "remote-x", "type": "remote"}))]);
    assert_eq!(registry.get("remote-x").unwrap().url, None);

    // A seeded model falls back to its seed, not to the earlier row.
    registry.hydrate([row(json!({"name": "gemma4-e2b", "param_size_b": 9.0}))]);
    registry.hydrate([row(json!({"name": "gemma4-e2b"}))]);
    assert_eq!(registry.get("gemma4-e2b").unwrap().param_size_b, 2.0);
}

#[test]
fn test_models_the_catalog_omits_are_kept() {
    let mut registry = ModelRegistry::seeded();
    let before = registry.len();
    registry.hydrate([]);
    assert_eq!(registry.len(), before);
}

#[test]
fn test_unparseable_rows_are_skipped() {
    assert!(CatalogModel::from_row(&obj(json!({"model_id": "org/x"}))).is_none());
    assert!(CatalogModel::from_row(&obj(json!({"name": "x", "param_size_b": "big"}))).is_none());
    assert!(CatalogModel::from_row(&obj(json!({"name": "x", "description": "extra fields pass"}))).is_some());
}
