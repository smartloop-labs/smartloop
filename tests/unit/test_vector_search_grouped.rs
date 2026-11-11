//! Port of `tests/unit/test_vector_search_grouped.py`.
//!
//! Regression tests for ModelService vector-search grouping.
//!
//! `vector_search_grouped` powers both `get_document_context_with_citations`
//! (concatenated string) and `get_document_groups` (per-document fan-out).
//! Its failures are swallowed and turned into `None`, so a defect there fails
//! silently at runtime — these tests exercise the real code path with a fake
//! store.  The Python tests patched `StoreFactory.create_store`; here the fake
//! store is placed in the project's cache slot, which the search reuses.

use std::collections::HashSet;
use std::sync::Arc;

use serde_json::{Map, Value, json};

use smartloop::api::services::model_service::ModelService;
use smartloop::db::store::{ChunkResult, VectorStore};
use smartloop::workspace::types::Document;

use crate::shared::model_service_b::{FakeStore, project, service_with_workspace};

/// A bare service wired to a project and `store`.  Store handles are per
/// project, so the store goes into the project's slot.
fn service_with_store(store: Arc<dyn VectorStore>) -> Arc<ModelService> {
    let svc = service_with_workspace("/tmp", vec![project("proj-1", "/tmp/proj-1", true)], vec![]);
    svc.caches("proj-1").lock().doc_store = Some(store);
    svc
}

fn page(n: i64) -> Map<String, Value> {
    json!({"page_number": n}).as_object().cloned().unwrap_or_default()
}

fn chunks() -> Vec<ChunkResult> {
    vec![
        ChunkResult {
            text: "t1".into(),
            document_id: "d1".into(),
            document_name: Some("alpha.pdf".into()),
            distance: 0.10,
            metadata: page(1),
            source_content: Some("Alpha body".into()),
            embedding: None,
        },
        ChunkResult {
            text: "t2".into(),
            document_id: "d2".into(),
            document_name: Some("beta.pdf".into()),
            distance: 0.20,
            metadata: page(2),
            source_content: Some("Beta body".into()),
            embedding: None,
        },
    ]
}

fn fake_store() -> Arc<dyn VectorStore> {
    FakeStore::new("docs", chunks())
}

fn names(items: &[Value]) -> HashSet<String> {
    items.iter().filter_map(|d| d["document_name"].as_str().map(str::to_string)).collect()
}

#[test]
fn test_get_document_groups_returns_documents_and_citations() {
    let svc = service_with_store(fake_store());

    let result = svc.get_document_groups(Some("alpha beta"), None, None);

    let (documents, citations) = result.expect("vector search returned None (swallowed failure?)");
    assert_eq!(names(&documents), HashSet::from(["alpha.pdf".to_string(), "beta.pdf".to_string()]));
    assert!(documents.iter().all(|d| d.get("content").is_some()));
    // Citations are built (this is the regression: all_citations must exist).
    assert_eq!(names(&citations), HashSet::from(["alpha.pdf".to_string(), "beta.pdf".to_string()]));
    assert!(citations.iter().any(|c| c.get("page_numbers").is_some_and(|p| match p {
        Value::Array(a) => !a.is_empty(),
        Value::Null => false,
        _ => true,
    })));
}

#[test]
fn test_get_document_context_with_citations_tuple_shape() {
    let svc = service_with_store(fake_store());

    let result = svc.get_document_context_with_citations(Some("alpha beta"), None, None);

    let (context, citations) = result.expect("a context");
    assert!(context.contains("=== Document: alpha.pdf ==="));
    assert!(context.contains("Alpha body"));
    assert_eq!(citations.len(), 2);
}

/// Store where every chunk is only weakly related (distance > 0.55).
fn weak_match_store() -> Arc<dyn VectorStore> {
    FakeStore::new(
        "weak",
        vec![ChunkResult {
            text: "t1".into(),
            document_id: "d1".into(),
            document_name: Some("unrelated.pdf".into()),
            distance: 0.72,
            metadata: page(1),
            source_content: Some("Unrelated body".into()),
            embedding: None,
        }],
    )
}

/// Best-match distance above the quality floor → no documents surfaced, so
/// purely conversational prompts don't get routed through EXPLORE.
#[test]
fn test_weak_best_match_returns_none() {
    let svc = service_with_store(weak_match_store());
    assert!(svc.get_document_groups(Some("places to visit in madrid"), None, None).is_none());
    assert!(svc.get_document_context_with_citations(Some("places to visit in madrid"), None, None).is_none());
}

/// Each citation ships the text its document contributed to the prompt.
///
/// Without it the router has no evidence to check the finished answer
/// against, so every retrieved document gets credited whether or not the
/// answer drew on it.
#[test]
fn test_citations_carry_the_document_text_as_a_private_extract() {
    let svc = service_with_store(fake_store());

    let (_, citations) = svc.get_document_groups(Some("alpha beta"), None, None).expect("citations");

    let by_name = |name: &str| citations.iter().find(|c| c["document_name"] == name).cloned().unwrap_or(Value::Null);
    assert_eq!(by_name("alpha.pdf")["_extract"], "Alpha body");
    assert_eq!(by_name("beta.pdf")["_extract"], "Beta body");
}

/// A citation says where its document's file is, relative to the project as
/// the documents API gives it, so a client can open the source; a document
/// the project doesn't list has none.
#[test]
fn test_citations_carry_the_document_path_relative_to_the_project() {
    let mut proj = project("proj-1", "/tmp/proj-1", true);
    proj.documents = vec![Document {
        id: "d1".into(),
        path: "/tmp/proj-1/documents/alpha.pdf".into(),
        name: Some("alpha.pdf".into()),
        ..Document::default()
    }];
    let svc = service_with_workspace("/tmp", vec![proj], vec![]);
    svc.caches("proj-1").lock().doc_store = Some(fake_store());

    let (_, citations) = svc.get_document_groups(Some("alpha beta"), None, None).expect("citations");

    let by_name = |name: &str| citations.iter().find(|c| c["document_name"] == name).cloned().unwrap_or(Value::Null);
    assert_eq!(by_name("alpha.pdf")["document_path"], "documents/alpha.pdf");
    assert!(by_name("beta.pdf").get("document_path").is_none());
}

/// A retrieved document's citation says it is one, so a client opens it as
/// a file rather than guessing from which keys it has.
#[test]
fn test_citations_name_their_source_as_a_document() {
    let svc = service_with_workspace("/tmp", vec![project("proj-1", "/tmp/proj-1", true)], vec![]);
    svc.caches("proj-1").lock().doc_store = Some(fake_store());

    let (_, citations) = svc.get_document_groups(Some("alpha beta"), None, None).expect("citations");

    assert!(!citations.is_empty());
    assert!(citations.iter().all(|c| c["source"] == "document"));
}
