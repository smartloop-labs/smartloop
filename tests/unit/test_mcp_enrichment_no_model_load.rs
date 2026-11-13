//! Port of `tests/unit/test_mcp_enrichment_no_model_load.py`.
//!
//! Registering an MCP server must never *load* a model.
//!
//! `enrich_mcp_server` used to call `get_large_llm()`, which lazily loads the
//! largest downloaded model — with an auto-sized context that can swallow
//! ~10GB of KV cache — and nothing on the registration path ever unloads it.
//! Enrichment now uses the small router model unconditionally: the prompts are
//! short and low-stakes, so they are not worth the large model's memory even
//! when a prior chat request already left it resident.
//!
//! The Python tests patched `MCPServerSummarizer` to capture the LLM it was
//! built with; here the recording LLMs say which one the summarizer drove.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;

use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::llms::base_llm::BaseLLM;
use smartloop::workspace::workspace::{MCPServer, MCPToolDefinition};

use crate::shared::model_service_b::FakeLLM;

fn tools() -> Vec<MCPToolDefinition> {
    vec![MCPToolDefinition {
        name: "get_weather".into(),
        description: Some("d".into()),
        input_schema: json!({"type": "object"}).as_object().cloned(),
        ..Default::default()
    }]
}

/// Hooks recording every request for the large model.
fn large_lookups(svc: &ModelService) -> Arc<AtomicUsize> {
    let asked = Arc::new(AtomicUsize::new(0));
    let counter = asked.clone();
    let mut hooks = svc.hooks.read().clone();
    hooks.get_large_llm = Some(Arc::new(move || {
        counter.fetch_add(1, Ordering::SeqCst);
        None
    }));
    svc.set_hooks(hooks);
    asked
}

/// Run the real `enrich_mcp_server` on `svc`.
fn run_enrichment(svc: &ModelService) -> MCPServer {
    let mut server = MCPServer { id: "x".into(), name: "demo".into(), tools: tools(), ..MCPServer::default() };
    svc.enrich_mcp_server(&mut server);
    server
}

/// The regression test for the 11GB blowup: with no large model resident,
/// enrichment must use the router model and must not ask for one to be
/// loaded.
#[test]
fn test_cold_process_uses_the_router_model_and_loads_nothing() {
    let svc = ModelService::new_bare();
    let router = FakeLLM::new("router");
    *svc.llm.write() = Some(router.dyn_llm());
    *svc.large_llm.write() = None;
    let large_asked = large_lookups(&svc);

    run_enrichment(&svc);

    assert!(router.calls() > 0, "the summarizer must run on the router model");
    assert_eq!(large_asked.load(Ordering::SeqCst), 0);
    // `_detect_large_model` is reached only through the large-model lookup,
    // which latches this flag first.
    assert!(!svc.large_model_attempted.load(Ordering::SeqCst));
}

/// Even when a prior chat request left the large model resident,
/// enrichment sticks to the router model.
#[test]
fn test_resident_large_model_is_still_not_used() {
    let svc = ModelService::new_bare();
    let router = FakeLLM::new("router");
    let large = FakeLLM::new("large");
    *svc.llm.write() = Some(router.dyn_llm());
    *svc.large_llm.write() = Some(large.dyn_llm());
    let large_asked = large_lookups(&svc);

    run_enrichment(&svc);

    assert!(router.calls() > 0);
    assert_eq!(large.calls(), 0);
    assert_eq!(large_asked.load(Ordering::SeqCst), 0);
}

/// The router model is the always-resident one; enrichment may load it
/// (normally a no-op — lifespan preloads it at startup).
#[test]
fn test_router_model_is_loaded_when_absent() {
    let svc = ModelService::new_bare();
    *svc.llm.write() = None;
    *svc.large_llm.write() = None;
    let loaded = FakeLLM::new("router");
    let loads = Arc::new(AtomicUsize::new(0));

    let weak = Arc::downgrade(&svc);
    let to_load: Arc<dyn BaseLLM> = loaded.dyn_llm();
    let load_count = loads.clone();
    let mut hooks = ModelServiceHooks::default();
    hooks.load_model = Some(Arc::new(move || {
        load_count.fetch_add(1, Ordering::SeqCst);
        let svc = weak.upgrade().expect("service alive");
        *svc.llm.write() = Some(to_load.clone());
        Ok(to_load.clone())
    }));
    svc.set_hooks(hooks);
    let large_asked = large_lookups(&svc);

    run_enrichment(&svc);

    assert!(loaded.calls() > 0, "the summarizer must run on the freshly loaded router model");
    assert_eq!(loads.load(Ordering::SeqCst), 1);
    assert_eq!(large_asked.load(Ordering::SeqCst), 0);
}

/// Belt-and-braces against someone reinstating the lazy load — the
/// behavioural assertions above would silently pass if the call moved into a
/// helper.
#[test]
fn test_source_does_not_reference_the_lazy_loader() {
    let source = include_str!("../../src/api/services/model_service/part2.rs");
    let start = source.find("    pub fn enrich_mcp_server(").expect("enrich_mcp_server is defined");
    let body = &source[start..];
    let end = body.find("\n    }\n").expect("enrich_mcp_server has a body");
    let code: Vec<&str> = body[..end].lines().filter(|line| !line.trim_start().starts_with("//")).collect();
    let code = code.join("\n");
    assert!(!code.contains("get_large_llm"));
    assert!(!code.contains("large_llm"));
}
