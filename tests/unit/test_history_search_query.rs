//! Port of `tests/unit/test_history_search_query.py`.
//!
//! Conversation history always grounds the search query for a session.
//!
//! `build_history_search_query` no longer applies a semantic gate: whenever
//! the session has prior turns it consults history via an LLM rewrite.  The
//! rewrite prompt is responsible for returning a self-contained message
//! UNCHANGED and for not dragging in entities the latest message does not
//! reference.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};

use crate::shared::model_service_a::bare_service;

/// A minimal service whose `generate_completion` returns `rewrite_return`,
/// plus the number of times it was called.
fn svc(rewrite_return: &str) -> (Arc<ModelService>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let value = rewrite_return.to_string();
    let svc = bare_service();
    svc.set_hooks(ModelServiceHooks {
        generate_completion: Some(Arc::new(move |_prompt: &str| {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(value.clone())
        })),
        ..Default::default()
    });
    (svc, calls)
}

fn turn(role: &str, content: &str) -> HashMap<String, String> {
    let mut t = HashMap::new();
    t.insert("role".to_string(), role.to_string());
    t.insert("content".to_string(), content.to_string());
    t
}

fn history() -> Vec<HashMap<String, String>> {
    vec![
        turn("user", "tell me about Adam Hart's invoices"),
        turn("assistant", "Adam Hart purchased chairs and a printer."),
    ]
}

/// A self-contained query is passed through the rewrite; the LLM returns it
/// unchanged (its prompt says to), so the output matches the input.
#[test]
fn test_self_contained_query_still_consults_history() {
    let query = "what items did Aaron Bergman purchase";
    let (svc, calls) = svc(query);
    let out = svc.build_history_search_query(query, &history());
    assert_eq!(out, query);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// A query that references prior turns ('those') is rewritten.
#[test]
fn test_referential_query_is_resolved_via_history() {
    let (svc, calls) = svc("Adam Hart invoice items");
    let out = svc.build_history_search_query("what about those", &history());
    assert_eq!(out, "Adam Hart invoice items");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// A multi-word follow-up with no pronoun ('the place') — previously skipped
/// by the word-count gate — now gets grounded against history.
#[test]
fn test_definite_article_followup_is_resolved_via_history() {
    let (svc, calls) = svc("key highlights of the report");
    let out = svc.build_history_search_query("what are the key highlights of the place", &history());
    assert_eq!(out, "key highlights of the report");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn test_single_turn_history_returns_query_unchanged() {
    let (svc, calls) = svc("REWRITTEN");
    let out = svc.build_history_search_query("hello", &[turn("user", "hello")]);
    assert_eq!(out, "hello");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
