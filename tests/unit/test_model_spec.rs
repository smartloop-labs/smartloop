//! Model specs: the built-in seeds, and catalog rows laid over them.

use serde_json::{json, Value};
use smartloop::model::spec::{seed_specs, CatalogModel, ModelRegistry, ModelSpec};
use smartloop::platform::client::JsonObject;
use smartloop::types::{Attention, ModelCapability, ModelKind, ModelType};

use ModelCapability::{Reasoning, Text, Thinking, Tools, Vision};

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
        ("sl-mini", "smartloop-ai/slp-orchestra-mini", 3, 1.0, vec![Text], None),
        ("gemma3-4b", "google/gemma-3-4b-it", 3, 4.0, vec![Text], Some("MTMDChatHandler")),
        ("gemma4-e2b", "google/gemma-4-e2b-it", 4, 2.0, vec![Text, Vision, Thinking], Some("Gemma4ChatHandler")),
        ("gemma4-e4b", "google/gemma-4-e4b-it", 4, 4.0, vec![Text, Vision, Thinking], Some("Gemma4ChatHandler")),
        ("qwen2-vl-2b", "Qwen/Qwen2-VL-2B-Instruct", 2, 2.0, vec![Vision], Some("Qwen25VLChatHandler")),
        ("ministral3-3b", "mistralai/Ministral-3-3B-Instruct-2512", 3, 3.0, vec![Text, Vision], Some("MTMDChatHandler")),
        ("qwen3.5-2b", "Qwen/Qwen3.5-2B", 3, 2.0, vec![Text, Reasoning, Vision, Thinking, Tools], Some("MTMDChatHandler")),
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
    assert_eq!(spec.capabilities, vec![Text, Vision, Thinking]);
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
    // "general" and "dp" each said the model holds a text conversation:
    // they read as `text`, once.
    assert_eq!(spec.capabilities, vec![Text, Vision]);
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
    // A catalog row's "general" and "coding" each said the model holds a
    // text conversation: `text`, once.
    assert_eq!(spec.capabilities, vec![Text]);
    // No weights, so nothing resolves by an empty id.
    assert!(registry.get_by_id("").is_none());
}

#[test]
fn test_a_mixture_of_experts_model_ranks_on_total_params() {
    let mut registry = ModelRegistry::default();
    registry.hydrate([row(json!({
        "name": "moe", "type": "remote", "param_size_b": 428.0, "active_param_size_b": 23.0,
        "attention": "hybrid",
    }))]);
    let spec = registry.get("moe").unwrap();
    assert_eq!(spec.param_size_b, 428.0);
    assert_eq!(spec.active_param_size_b, Some(23.0));
    // Ranked on how big it is, not on the share active per token.
    assert_eq!(spec.ranking_size_b(), 428.0);
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
fn test_an_undeclared_model_declares_nothing() {
    let mut registry = ModelRegistry::default();
    registry.hydrate([row(json!({"name": "mystery", "model_id": "org/mystery"}))]);
    let spec = registry.get("mystery").unwrap();
    assert_eq!(*spec, ModelSpec::local("mystery", "org/mystery"));
    assert!(spec.capabilities.is_empty());
}

#[test]
fn test_unknown_capabilities_are_dropped() {
    let mut registry = ModelRegistry::default();
    registry.hydrate([row(json!({"name": "x", "capabilities": ["vision", "telepathy"]}))]);
    assert_eq!(registry.get("x").unwrap().capabilities, vec![Vision]);
    // The retired "general", "dp" and "coding" each read as `text`, once.
    registry.hydrate([row(json!({"name": "x", "capabilities": ["general", "dp", "coding"]}))]);
    assert_eq!(registry.get("x").unwrap().capabilities, vec![Text]);
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

/// A model that declares nothing but vision is the one kind that holds no
/// text conversation; a vision model whose row also says `general`, `dp` or
/// `coding` — the whole cloud catalog — is a text model, though only
/// `vision` survives as a capability.
#[test]
fn test_only_a_model_declared_vision_alone_is_not_a_text_model() {
    let mut registry = ModelRegistry::seeded();
    registry.hydrate([
        row(json!({
            "name": "minimax-m3", "model_id": null, "type": "remote", "url": "https://api.smartloop.ai",
            "capabilities": ["general", "dp", "vision", "coding"], "param_size_b": 428.0, "active_param_size_b": 23.0,
        })),
        row(json!({
            "name": "ocr-only", "model_id": "acme/ocr-only", "type": "local",
            "capabilities": ["vision"], "vision_chat_handler": "MTMDChatHandler", "param_size_b": 2.0,
        })),
    ]);
    let text = |spec: &ModelSpec| ModelCapability::text_model(&spec.capability_values());
    let minimax = registry.get("minimax-m3").unwrap();
    assert_eq!(minimax.capabilities, vec![Text, Vision]);
    assert!(text(&minimax), "the legacy tags beside vision say it chats");
    let ocr = registry.get("ocr-only").unwrap();
    assert_eq!(ocr.capabilities, vec![Vision]);
    assert!(!text(&ocr));
    // The seeds agree: Qwen2-VL is declared vision alone; the text models are text.
    assert!(!text(&registry.get("qwen2-vl-2b").unwrap()));
    assert!(text(&registry.get("qwen3.5-2b").unwrap()));
    assert!(text(&registry.get("slp-orchestra-mini").unwrap()));
}

#[test]
fn test_a_name_listed_both_ways_is_one_local_spec_with_a_cloud_variant() {
    use std::sync::Arc;
    let remote = row(json!({
        "name": "qwen3.8-27b",
        "model_id": null,
        "type": "remote",
        "url": "https://api.smartloop.ai",
        "capabilities": ["text", "reasoning", "vision", "tools"],
        "context_length": 262144,
        "param_size_b": 27.0,
    }));
    let local = row(json!({
        "name": "qwen3.8-27b",
        "model_id": "Qwen/Qwen3.8-27B",
        "type": "local",
        "capabilities": ["text", "reasoning", "vision", "thinking", "tools"],
        "param_size_b": 27.3,
        "vision_chat_handler": "MTMDChatHandler",
    }));
    // Either order: the API lists by creation time.
    for rows in [vec![remote.clone(), local.clone()], vec![local.clone(), remote.clone()]] {
        let mut registry = ModelRegistry::seeded();
        registry.hydrate(rows);
        let spec = registry.get("qwen3.8-27b").unwrap();
        assert!(!spec.is_remote());
        assert!(spec.has_cloud());
        assert_eq!(spec.model_id, "Qwen/Qwen3.8-27B");
        assert!(spec.supports(Thinking));
        assert_eq!(registry.specs().iter().filter(|s| s.name == "qwen3.8-27b").count(), 1);

        let cloud = spec.for_kind(ModelKind::Remote);
        assert!(cloud.is_remote());
        assert_eq!(cloud.model_id, "");
        assert_eq!(cloud.url.as_deref(), Some("https://api.smartloop.ai"));
        assert!(!cloud.supports(Thinking));
        assert!(Arc::ptr_eq(&spec.for_kind(ModelKind::Local), &spec));
    }
}

#[test]
fn test_a_later_catalog_without_the_remote_row_drops_the_cloud_variant() {
    let mut registry = ModelRegistry::seeded();
    registry.hydrate([
        row(json!({"name": "qwen3.8-27b", "model_id": "Qwen/Qwen3.8-27B", "type": "local"})),
        row(json!({"name": "qwen3.8-27b", "model_id": null, "type": "remote"})),
    ]);
    assert!(registry.get("qwen3.8-27b").unwrap().cloud.is_some());
    registry.hydrate([row(json!({"name": "qwen3.8-27b", "model_id": "Qwen/Qwen3.8-27B", "type": "local"}))]);
    assert!(registry.get("qwen3.8-27b").unwrap().cloud.is_none());
}

#[test]
fn test_a_remote_only_model_has_no_cloud_variant() {
    let mut registry = ModelRegistry::seeded();
    registry.hydrate([row(json!({"name": "glm-5.3", "model_id": null, "type": "remote"}))]);
    let spec = registry.get("glm-5.3").unwrap();
    assert!(spec.is_remote() && spec.has_cloud() && spec.cloud.is_none());
}
