//! Port of `tests/unit/test_mcp_enrichment_skips_curated.py`.
//!
//! LLM enrichment must only run for hand-added MCP servers.
//!
//! `enrich_mcp_server` exists for the `uvx`/`npx` case: a package that ships
//! no description and is not in our curated catalog.  Servers installed from
//! the catalog arrive with `description` and `tags` already set, and running
//! enrichment on those burns two inference calls (plus possibly a model load)
//! only to overwrite curated copy with generated copy.
//!
//! Python patched `MCPServerSummarizer` with a stub; here the real summarizer
//! runs against a stub model whose two prompts are told apart by their system
//! line, so `calls` records which summarizer methods — i.e. which LLM
//! round-trips — actually fired.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Map, Value};

use smartloop::api::model::mcp::MCPRegisterRequest;
use smartloop::api::services::model_service::ModelService;
use smartloop::workspace::workspace::{MCPServer, MCPToolDefinition};

use crate::shared::agents_mcp_agent::StubLLM;

const METADATA_PROMPT: &str = "You describe MCP tool servers.";
const SUMMARY_PROMPT: &str = "You summarize MCP tool servers";

fn tools() -> Vec<MCPToolDefinition> {
    let mut schema = Map::new();
    schema.insert("type".to_string(), json!("object"));
    vec![MCPToolDefinition {
        name: "get_weather".to_string(),
        description: Some("d".to_string()),
        input_schema: Some(schema),
        ..Default::default()
    }]
}

/// The summarizer round-trips a stub model saw, by name.
fn summarizer_calls(llm: &Arc<StubLLM>) -> Vec<&'static str> {
    llm.calls()
        .iter()
        .map(|c| {
            if c.prompt.contains(METADATA_PROMPT) {
                "generate_metadata"
            } else if c.prompt.contains(SUMMARY_PROMPT) {
                "invoke"
            } else {
                "other"
            }
        })
        .collect()
}

/// A model answering the metadata prompt with generated copy and the summary
/// prompt with a generated summary.
fn generating_llm() -> Arc<StubLLM> {
    StubLLM::with_reply_fn(Arc::new(|prompt: &str, _max, _thinking| {
        if prompt.contains(METADATA_PROMPT) {
            r#"{"description": "GEN-DESC", "tags": ["gen1", "gen2"]}"#.to_string()
        } else {
            "GEN-SUMMARY".to_string()
        }
    }))
}

/// Run the real `enrich_mcp_server` against a stub model.
///
/// Returns `(server, calls)` where `calls` records which summarizer methods
/// actually fired.
fn run_enrichment(fields: impl FnOnce(&mut MCPServer)) -> (MCPServer, Vec<&'static str>) {
    let mut server = MCPServer {
        id: "x".to_string(),
        name: "demo".to_string(),
        tools: tools(),
        ..Default::default()
    };
    fields(&mut server);
    let llm = generating_llm();
    let svc = ModelService::new_bare();
    *svc.llm.write() = Some(llm.as_llm());

    // Best-effort: enrichment never fails the caller.
    let _ = svc.enrich_mcp_server(&mut server);
    let calls = summarizer_calls(&llm);
    (server, calls)
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|s| s.to_string()).collect()
}

// ------------------------------------------------------ TestCuratedServersSkipEnrichment

#[test]
fn test_curated_description_and_tags_make_no_llm_calls() {
    let (server, calls) = run_enrichment(|s| {
        s.description = Some("Curated desc".to_string());
        s.tags = strings(&["weather", "api"]);
    });
    assert!(calls.is_empty());
    assert_eq!(server.description.as_deref(), Some("Curated desc"));
    assert_eq!(server.tags, strings(&["weather", "api"]));
}

/// The catalog has no summary field, so the selector reuses the description
/// rather than paying for a generated one.
#[test]
fn test_missing_summary_falls_back_to_description_for_free() {
    let (server, calls) = run_enrichment(|s| {
        s.description = Some("Curated desc".to_string());
        s.tags = strings(&["weather"]);
    });
    assert!(calls.is_empty());
    assert_eq!(server.summary.as_deref(), Some("Curated desc"));
}

#[test]
fn test_curated_summary_is_left_alone() {
    let (server, calls) = run_enrichment(|s| {
        s.description = Some("Curated desc".to_string());
        s.tags = strings(&["weather"]);
        s.summary = Some("Curated summary".to_string());
    });
    assert!(calls.is_empty());
    assert_eq!(server.summary.as_deref(), Some("Curated summary"));
}

// ---------------------------------------------------------- TestHandAddedServersStillEnrich

#[test]
fn test_bare_uvx_add_generates_everything() {
    let (server, calls) = run_enrichment(|_| {});
    assert_eq!(calls, vec!["generate_metadata", "invoke"]);
    assert_eq!(server.description.as_deref(), Some("GEN-DESC"));
    assert_eq!(server.tags, strings(&["gen1", "gen2"]));
    assert_eq!(server.summary.as_deref(), Some("GEN-SUMMARY"));
}

/// A package that advertises its own description but is not in our listings
/// has no tags — it still needs them generated.
#[test]
fn test_description_without_tags_still_enriches() {
    let (server, calls) = run_enrichment(|s| {
        s.description = Some("Package's own desc".to_string());
    });
    assert!(calls.contains(&"generate_metadata"));
    assert_eq!(server.description.as_deref(), Some("Package's own desc")); // not overwritten
    assert_eq!(server.tags, strings(&["gen1", "gen2"]));
}

#[test]
fn test_no_tools_discovered_skips_entirely() {
    let mut server = MCPServer {
        id: "x".to_string(),
        name: "demo".to_string(),
        tools: vec![],
        ..Default::default()
    };
    let llm = generating_llm();
    let svc = ModelService::new_bare();
    *svc.llm.write() = Some(llm.as_llm());

    let _ = svc.enrich_mcp_server(&mut server);

    // No summarizer was built, so no model was asked anything.
    assert_eq!(llm.call_count(), 0);
}

// ----------------------------------------------------- TestCatalogProvenanceSkipsEnrichment
// A resolved catalog slug is provenance the backend established itself, so it
// short-circuits enrichment even when the entry left a field blank.

#[test]
fn test_slug_alone_skips_enrichment() {
    let (_server, calls) = run_enrichment(|s| {
        s.catalog_slug = Some("linear".to_string());
    });
    assert!(calls.is_empty());
}

/// A curated entry with no tags must not fall through to the LLM.
#[test]
fn test_slug_skips_even_without_tags() {
    let (server, calls) = run_enrichment(|s| {
        s.description = Some("Curated desc".to_string());
        s.catalog_slug = Some("linear".to_string());
    });
    assert!(calls.is_empty());
    assert_eq!(server.description.as_deref(), Some("Curated desc"));
}

#[test]
fn test_no_slug_and_nothing_curated_still_enriches() {
    let (server, calls) = run_enrichment(|_| {});
    assert_eq!(calls, vec!["generate_metadata", "invoke"]);
    assert_eq!(server.catalog_slug, None);
}

// ------------------------------------------------------ TestCuratedMetadataCanReachTheServer
// The skip above is only reachable if the register API carries the slug
// through — without it the record always lands bare.

#[test]
fn test_register_request_accepts_catalog_slug() {
    let request: MCPRegisterRequest =
        serde_json::from_value(json!({"server_url": "https://x.test"})).expect("valid request");
    assert_eq!(request.catalog_slug, None);
    let request: MCPRegisterRequest =
        serde_json::from_value(json!({"server_url": "https://x.test", "catalog_slug": "linear"}))
            .expect("valid request");
    assert_eq!(request.catalog_slug.as_deref(), Some("linear"));
}

// `inspect.signature(...).parameters` has no runtime equivalent: each case
// below passes the curated field in its own position, so the test target only
// compiles while the add method accepts it.  The closures are never run.

#[test]
fn test_service_add_methods_accept_curated_fields_add_local_mcp_server_tags() {
    let _accepts = |svc: &ModelService| {
        svc.add_local_mcp_server("uvx", None, None, None, None, None, Some(vec!["weather".to_string()]), None, None)
    };
}

#[test]
fn test_service_add_methods_accept_curated_fields_add_local_mcp_server_catalog_slug() {
    let _accepts = |svc: &ModelService| {
        svc.add_local_mcp_server("uvx", None, None, None, None, None, None, Some("linear"), None)
    };
}

#[test]
fn test_service_add_methods_accept_curated_fields_add_local_mcp_server_description() {
    let _accepts = |svc: &ModelService| {
        svc.add_local_mcp_server("uvx", None, None, None::<HashMap<String, String>>, None, Some("Curated"), None, None, None)
    };
}

/// Calls `add_remote_mcp_server` with the three curated fields given.
fn remote_with(
    svc: &ModelService,
    description: Option<&str>,
    tags: Option<Vec<String>>,
    catalog_slug: Option<&str>,
) -> anyhow::Result<Value> {
    svc.add_remote_mcp_server(
        "linear",
        "https://x.test/mcp",
        "none",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "Bearer",
        None,
        None,
        None,
        description,
        tags,
        catalog_slug,
        None,
    )
}

#[test]
fn test_service_add_methods_accept_curated_fields_add_remote_mcp_server_tags() {
    let _accepts = |svc: &ModelService| remote_with(svc, None, Some(vec!["weather".to_string()]), None);
}

#[test]
fn test_service_add_methods_accept_curated_fields_add_remote_mcp_server_catalog_slug() {
    let _accepts = |svc: &ModelService| remote_with(svc, None, None, Some("linear"));
}

#[test]
fn test_service_add_methods_accept_curated_fields_add_remote_mcp_server_description() {
    let _accepts = |svc: &ModelService| remote_with(svc, Some("Curated"), None, None);
}

#[test]
fn test_server_record_persists_provenance() {
    let server = MCPServer {
        id: "x".to_string(),
        name: "demo".to_string(),
        catalog_slug: Some("linear".to_string()),
        ..Default::default()
    };
    let stored = serde_json::to_value(&server).expect("serializable");
    assert_eq!(stored.get("catalog_slug"), Some(&json!("linear")));
    let restored: MCPServer = serde_json::from_value(stored).expect("deserializable");
    assert_eq!(restored.catalog_slug.as_deref(), Some("linear"));
}
