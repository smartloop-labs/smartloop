//! Port of `tests/unit/test_document_fanout.py`.
//!
//! Tests for the document map-reduce fan-out in SLPOrchestrator.
//!
//! When several documents match the search, the orchestrator extends the DAG
//! at runtime with one node per document/batch (parallel extraction) plus a
//! single reduce node that merges them into one synthesized document.  The
//! downstream skill/classify pipeline then hangs off that reduce node.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use serde_json::{json, Map, Value};

use smartloop::agents::mcp::explorer::SearchContext;
use smartloop::agents::nodes::{NodeResult, RouteNode};
use smartloop::agents::orchestration::context as octx;
use smartloop::agents::orchestration::document_fanout::{DocumentFanoutStep, BATCH_GROUP_SIZE};
use smartloop::agents::orchestration::step::{Step, StepNodes};
use smartloop::agents::orchestrator::{
    CapabilityLlmProvider, ExecuteOptions, SLPOrchestrator, SLPOrchestratorOptions, SharedCell, SharedSearchContext,
};

use crate::shared::agents_core::{llm_8k, obj, orchestrator_over, FakeLlm};

fn make_orchestrator() -> (SLPOrchestrator, Arc<FakeLlm>) {
    let llm = llm_8k("OK");
    (orchestrator_over(&llm), llm)
}

#[test]
fn test_frame_web_results_instructs_model_not_to_disclaim() {
    let (orch, _llm) = make_orchestrator();
    let _ = orch.response();
    let framed = smartloop::agents::orchestration::response_builder::ResponseBuilder::frame_web_results(
        "## [1] https://canyon.com\n\nSome bikes",
    );
    assert!(framed.contains("live web search was just performed"));
    assert!(framed.contains("real-time search results"));
    assert!(framed.to_lowercase().contains("do not say you lack") || framed.contains("Do NOT say you lack"));
    assert!(framed.contains("WEB SEARCH RESULTS:\n## [1] https://canyon.com"));
}

fn docs(n: usize, size: usize) -> Vec<Map<String, Value>> {
    (0..n)
        .map(|i| obj(json!({"document_name": format!("doc{i}.pdf"), "content": format!("{} body {i}", "X".repeat(size))})))
        .collect()
}

/// `_ctx(orch)` in the step's context type.
fn ctx(orch: &SLPOrchestrator, user_message: &str) -> octx::SharedOrchestratorContext {
    let value = orch
        .build_context(user_message, &[], &ExecuteOptions::default())
        .unwrap()
        .with_ref(|c| c.clone());
    SharedCell::wrap(value)
}

fn summarize_ctx(orch: &SLPOrchestrator) -> octx::SharedOrchestratorContext {
    ctx(orch, "summarize the documents")
}

// ------------------------------------------------------------------ #
// _batch_documents packing                                            #
// ------------------------------------------------------------------ #

mod batch_documents {
    use super::*;

    /// Documents are grouped so a node can hand the model their prompts
    /// together; each is still extracted on its own (see `fanout_handlers`).
    #[test]
    fn test_documents_are_grouped_up_to_the_group_size() {
        let (orch, _llm) = make_orchestrator();
        let batches = orch.doc_fanout().batch_documents(&docs(5, 50));
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 5);
    }

    #[test]
    fn test_all_documents_processed_regardless_of_count() {
        let (orch, _llm) = make_orchestrator();
        let docs = docs(50, 100);
        let batches = orch.doc_fanout().batch_documents(&docs);
        assert_eq!(batches.len(), 50usize.div_ceil(BATCH_GROUP_SIZE));
        assert!(batches.iter().all(|b| b.len() <= BATCH_GROUP_SIZE));
        let names: Vec<&Value> = batches.iter().flatten().map(|d| &d["document_name"]).collect();
        let expected: Vec<&Value> = docs.iter().map(|d| &d["document_name"]).collect();
        assert_eq!(names, expected);
    }
}

// ------------------------------------------------------------------ #
// Route-level wiring                                                  #
// ------------------------------------------------------------------ #

mod route_fanout {
    //! Python replaced `PipelineBuilder._build_unified_pipeline` with a fake
    //! that records the parent it is handed, then inspected the route node's
    //! `_extend_dag`; here the fake goes in through the builder's
    //! unified-pipeline seam.

    use super::*;

    use parking_lot::Mutex;
    use smartloop::agents::mcp::explorer::EnrichedContext;
    use smartloop::agents::nodes::{AgentNode, CollectNode, SkillMatchNode};

    use crate::shared::agents_core::class_name;

    /// `{type(n).__name__: n for n, _ in res.data["_extend_dag"]}` plus the
    /// parent the downstream pipeline was handed.
    fn route(
        orch: &SLPOrchestrator,
        sctx: SearchContext,
        matched_skills: Option<Value>,
    ) -> (HashMap<&'static str, AgentNode>, Option<AgentNode>) {
        let (res, parent) = route_result(orch, sctx, matched_skills);
        let nodes: HashMap<&'static str, AgentNode> =
            res.extend_dag.iter().map(|e| (class_name(&e.node), e.node.clone())).collect();
        (nodes, parent)
    }

    /// The route node's own result plus the parent the pipeline was handed.
    fn route_result(
        orch: &SLPOrchestrator,
        sctx: SearchContext,
        matched_skills: Option<Value>,
    ) -> (NodeResult, Option<AgentNode>) {
        let ctx = summarize_ctx(orch);

        let skill_match_node = SkillMatchNode::new(&[]);
        let collect_node = CollectNode::new(&[&skill_match_node]);
        let route_node = RouteNode::new(&[&skill_match_node, &collect_node]);

        ctx.write().enriched = Some(EnrichedContext::default());

        let captured: Arc<Mutex<Option<AgentNode>>> = Arc::new(Mutex::new(None));
        let seen = Arc::clone(&captured);
        orch.pipeline().set_unified_pipeline_hook(Some(Arc::new(move |_c, _ms, _sc, _ec, parent| {
            *seen.lock() = parent.cloned();
            Vec::new()
        })));

        let mut deps = HashMap::new();
        deps.insert(
            skill_match_node.node_id.clone(),
            NodeResult::with_data(
                skill_match_node.node_id.clone(),
                json!({"matched_skills": matched_skills.unwrap_or(Value::Null)}),
            ),
        );
        deps.insert(
            collect_node.node_id.clone(),
            NodeResult::with_data(
                collect_node.node_id.clone(),
                json!({"search_ctx": serde_json::to_value(&sctx).unwrap()}),
            )
            .with_payload(sctx),
        );

        let res = orch.pipeline().handle_route(&ctx, &deps, &route_node, &skill_match_node, &collect_node);
        let parent = captured.lock().clone();
        (res, parent)
    }

    /// The executor deep-clones a result into every dependant, one per
    /// retrieved document under the fan-out.  Serialized into `data`, a
    /// 213-document context became ~32GB of copies; it must ride only as
    /// the shared payload.
    #[test]
    fn test_route_result_does_not_copy_the_search_context() {
        let (orch, _llm) = make_orchestrator();
        let sctx = SearchContext { vector_context: "raw".to_string(), documents: docs(3, 100), ..Default::default() };
        let (res, _) = route_result(&orch, sctx, None);

        assert!(res.data.get("search_ctx").is_none());
        let shared = res.payload_as::<SharedSearchContext>().expect("the shared context");
        assert_eq!(shared.lock().documents.len(), 3);
    }

    #[test]
    fn test_fanout_injected_for_multiple_documents() {
        let (orch, _llm) = make_orchestrator();
        let sctx = SearchContext { vector_context: "raw".to_string(), documents: docs(3, 100), ..Default::default() };
        let (nodes, captured) = route(&orch, sctx, None);

        let batch_nodes: Vec<&AgentNode> = nodes.values().filter(|n| n.is_llm_decision()).collect();
        let reduce_nodes: Vec<&AgentNode> = nodes.values().filter(|n| n.is_transform()).collect();

        assert!(!batch_nodes.is_empty());
        assert!(!reduce_nodes.is_empty());
        assert!(captured.expect("the downstream parent").is_transform());
    }

    /// DOC_FANOUT_THRESHOLD = 1: even a single document goes through the
    /// same map-reduce structure so every retrieved document is processed
    /// by its own DAG node.  With one doc that's one batch + one reduce.
    #[test]
    fn test_single_document_uses_one_batch_node() {
        let (orch, _llm) = make_orchestrator();
        let sctx = SearchContext { vector_context: "raw".to_string(), documents: docs(1, 100), ..Default::default() };
        let (nodes, captured) = route(&orch, sctx, None);

        let batch_nodes: Vec<&AgentNode> = nodes.values().filter(|n| n.is_llm_decision()).collect();
        let reduce_nodes: Vec<&AgentNode> = nodes.values().filter(|n| n.is_transform()).collect();

        assert_eq!(batch_nodes.len(), 1);
        assert!(!reduce_nodes.is_empty());
        assert!(captured.expect("the downstream parent").is_transform());
    }

    #[test]
    fn test_no_fanout_when_no_documents() {
        let (orch, _llm) = make_orchestrator();
        let sctx = SearchContext { vector_context: "raw".to_string(), ..Default::default() };
        let (nodes, captured) = route(&orch, sctx, None);

        assert!(!nodes.values().any(|n| n.is_transform() && Some(n) != captured.as_ref()));
        assert_eq!(class_name(&captured.expect("the downstream parent")), "RouteNode");
    }
}

// ------------------------------------------------------------------ #
// Batch + reduce handler behavior                                     #
// ------------------------------------------------------------------ #

/// Build the fan-out over `sctx` with a route node as parent.
fn build(orch: &SLPOrchestrator, sctx: &SearchContext) -> StepNodes {
    let ctx = summarize_ctx(orch);
    let parent = RouteNode::new(&[]);
    let shared: SharedSearchContext = Arc::new(parking_lot::Mutex::new(sctx.clone()));
    orch.doc_fanout().build(&ctx, Some(&shared), &parent)
}

/// Run every batch handler, then the reduce handler over their results.
fn run_fanout(built: &StepNodes) -> (HashMap<String, NodeResult>, NodeResult) {
    let mut results = HashMap::new();
    let mut reduce_handler = None;
    for (node, handler) in &built.pairs {
        if node.is_transform() {
            reduce_handler = Some(handler.clone());
            continue;
        }
        results.insert(node.node_id.clone(), handler(HashMap::new()).unwrap());
    }
    let reduce_result = reduce_handler.expect("a reduce node")(results.clone()).unwrap();
    (results, reduce_result)
}

fn content(result: &NodeResult) -> String {
    result.data["content"].as_str().unwrap_or("").to_string()
}

/// Each surviving document's extract in a group node's result.
fn extracts(result: &NodeResult) -> Vec<String> {
    match result.data["extracts"].as_array() {
        Some(list) => list.iter().map(|e| e["content"].as_str().unwrap_or("").to_string()).collect(),
        None => vec![content(result)].into_iter().filter(|c| !c.is_empty()).collect(),
    }
}

/// The search context the reduce handed downstream.
fn reduced(result: &NodeResult) -> SearchContext {
    result
        .payload_as::<SharedSearchContext>()
        .map(|shared| shared.lock().clone())
        .or_else(|| result.payload_as::<SearchContext>().cloned())
        .expect("the reduced search context")
}

mod fanout_handlers {
    use super::*;

    #[test]
    fn test_reduce_merges_extracts_and_replaces_context() {
        let (orch, llm) = make_orchestrator();
        llm.set_responder(|p| Ok(format!("EXTRACT::{}", p.chars().last().map(String::from).unwrap_or_default())));

        let sctx = SearchContext {
            vector_context: "ORIGINAL RAW CONCAT".to_string(),
            documents: docs(3, 100),
            ..Default::default()
        };

        let built = build(&orch, &sctx);
        assert!(built.parent.is_transform());

        let (results, reduce_result) = run_fanout(&built);
        let all: Vec<String> = results.values().flat_map(extracts).collect();
        assert_eq!(all.len(), 3);
        assert!(all.iter().all(|e| e.starts_with("EXTRACT::")));

        let merged = content(&reduce_result);
        assert_eq!(merged.matches("EXTRACT::").count(), all.len());
        let after = reduced(&reduce_result);
        assert_eq!(after.vector_context, merged);
        assert_ne!(after.vector_context, "ORIGINAL RAW CONCAT");
    }

    #[test]
    fn test_reduce_restricts_citations_to_surviving_documents() {
        let (orch, llm) = make_orchestrator();
        llm.set_responder(|prompt| {
            Ok(if prompt.contains("body 0") { "Relevant extract.".to_string() } else { "NONE".to_string() })
        });

        let sctx = SearchContext {
            documents: docs(2, 5000),
            citations: Some(vec![
                obj(json!({"document_id": "0", "document_name": "doc0.pdf"})),
                obj(json!({"document_id": "1", "document_name": "doc1.pdf"})),
            ]),
            ..Default::default()
        };

        let (_, reduce_result) = run_fanout(&build(&orch, &sctx));
        let after = reduced(&reduce_result);
        let cited: HashSet<String> = after
            .citations
            .unwrap_or_default()
            .iter()
            .map(|c| c["document_name"].as_str().unwrap_or("").to_string())
            .collect();
        assert_eq!(cited, HashSet::from(["doc0.pdf".to_string()]));
        assert_eq!(after.cited_documents, Some(vec!["doc0.pdf".to_string()]));
    }

    /// Every batch extracts on the shared in-process model.
    ///
    /// Extraction used to be dispatched to an LLM worker *process* when a
    /// pool existed, with `parallel_inference=True` to opt the node out of the
    /// DAG executor's LLM lock.  The pool is gone (same-device model copies
    /// serialize in the driver anyway), so there is one inference path and
    /// every LLMDecisionNode is serialized.
    ///
    /// Python replaced the router's `llm_for_tier`; here the DP-capability
    /// model is supplied the way the service supplies it, through the
    /// orchestrator's capability provider.
    #[test]
    fn test_extraction_uses_the_in_process_dp_model() {
        let llm = llm_8k("OK");
        let dp_llm = llm_8k("DP EXTRACT");
        let provider_llm = dp_llm.shared();
        let provider: CapabilityLlmProvider = Arc::new(move |capability: &str| {
            Ok(if capability == "dp" { Some(Arc::clone(&provider_llm)) } else { None })
        });
        let orch = SLPOrchestrator::new(
            llm.shared(),
            SLPOrchestratorOptions { capability_llm_provider: Some(provider), ..Default::default() },
        );

        let sctx = SearchContext { documents: docs(3, 100), ..Default::default() };
        let built = build(&orch, &sctx);

        let batch: Vec<_> = built.pairs.iter().filter(|(n, _)| n.is_llm_decision()).collect();
        assert!(!batch.is_empty());
        // The opt-out flag is gone: an AgentNode carries no such field, so the
        // executor's lock always applies to these nodes.

        let mut n = 0;
        for (_, handler) in &batch {
            for extract in extracts(&handler(HashMap::new()).unwrap()) {
                assert_eq!(extract, "DP EXTRACT");
                n += 1;
            }
        }
        // One extraction per document, whatever the grouping.
        assert_eq!(n, 3);
        assert_eq!(dp_llm.call_count(), 3);
    }

    fn first_batch(orch: &SLPOrchestrator, sctx: &SearchContext) -> NodeResult {
        let built = build(orch, sctx);
        let (_, handler) = &built.pairs[0];
        handler(HashMap::new()).unwrap()
    }

    #[test]
    fn test_batch_drops_document_when_llm_returns_none() {
        let (orch, llm) = make_orchestrator();
        llm.set_reply("NONE");

        let result = first_batch(&orch, &SearchContext { documents: docs(2, 100), ..Default::default() });
        assert!(extracts(&result).is_empty());
        assert_eq!(content(&result), "");
    }

    #[test]
    fn test_batch_falls_back_to_raw_when_llm_call_fails() {
        let (orch, llm) = make_orchestrator();
        llm.set_failure("boom");

        let result = first_batch(&orch, &SearchContext { documents: docs(2, 100), ..Default::default() });
        let all = extracts(&result);
        assert_eq!(all.len(), 2);
        assert!(all.iter().all(|e| e.contains("body")));
    }

    /// llama.cpp's empty-output fallback string must never be surfaced as
    /// extracted document content — treat it like a failure and keep a raw
    /// excerpt so the document isn't silently replaced by placeholder text.
    #[test]
    fn test_batch_falls_back_to_raw_when_llm_returns_placeholder() {
        let (orch, llm) = make_orchestrator();
        llm.set_reply("No response generated");

        let result = first_batch(&orch, &SearchContext { documents: docs(2, 100), ..Default::default() });
        let all = extracts(&result);
        assert!(all.iter().all(|e| !e.contains("No response generated")));
        assert!(all[0].contains("body 0"));
    }

    #[test]
    fn test_batch_skips_llm_when_document_is_json_table() {
        let (orch, llm) = make_orchestrator();
        llm.set_failure("LLM must not run for JSON docs");

        let doc = obj(json!({
            "document_name": "stats.json",
            "content": json!({
                "rows": [
                    {"name": "Drake Maye", "team": "NE", "values": {"epa": 0.296484, "td": 31}},
                    {"name": "Jordan Love", "team": "GB", "values": {"epa": 0.240092, "td": 23}},
                ]
            }).to_string(),
        }));
        let result = first_batch(&orch, &SearchContext { documents: vec![doc], ..Default::default() });
        let content = content(&result);
        assert_eq!(llm.call_count(), 0);
        assert!(content.contains("| name | team | values.epa | values.td |"));
        assert!(content.contains("| Drake Maye | NE | 0.2965 | 31 |"));
        assert!(content.contains("| Jordan Love | GB | 0.2401 | 23 |"));
    }

    /// Content already normalized to a table at index time must be used
    /// as-is — no LLM extraction (which could overflow the model window).
    #[test]
    fn test_batch_uses_existing_markdown_table_directly() {
        let (orch, llm) = make_orchestrator();
        llm.set_failure("LLM must not run for tables");

        let doc = obj(json!({
            "document_name": "stats.txt",
            "content": "| name | team | td |\n|---|---|---|\n| Drake Maye | NE | 31 |\n| Jordan Love | GB | 23 |\n",
        }));
        let result = first_batch(&orch, &SearchContext { documents: vec![doc], ..Default::default() });
        assert_eq!(llm.call_count(), 0);
        assert!(content(&result).contains("| Drake Maye | NE | 31 |"));
        assert!(content(&result).contains("| Jordan Love | GB | 23 |"));
    }
}

// ------------------------------------------------------------------ #
// _json_to_table                                                      #
// ------------------------------------------------------------------ #

mod json_to_table {
    use super::*;

    fn table(text: &str) -> String {
        DocumentFanoutStep::json_to_table(text).expect("a table")
    }

    #[test]
    fn test_top_level_array() {
        let table = table(r#"[{"name": "A", "v": 1}, {"name": "B", "v": 2}]"#);
        assert!(table.contains("| name | v |"));
        assert!(table.contains("| A | 1 |"));
        assert!(table.contains("| B | 2 |"));
    }

    #[test]
    fn test_rows_container() {
        let table = table(r#"{"rows": [{"name": "A", "v": 1}]}"#);
        assert!(table.contains("| name | v |"));
        assert!(table.contains("| A | 1 |"));
    }

    #[test]
    fn test_nested_values_flattened_with_dotted_keys() {
        let table = table(r#"[{"name": "A", "values": {"epa": 0.296484, "td": 31}}]"#);
        assert!(table.contains("| name | values.epa | values.td |"));
        assert!(table.contains("| A | 0.2965 | 31 |"));
    }

    #[test]
    fn test_float_compaction() {
        let table = table(r#"[{"v": 4394.0, "epa": 0.296484}]"#);
        assert!(table.contains("| 4394 | 0.2965 |"));
    }

    #[test]
    fn test_returns_none_for_non_json() {
        assert_eq!(DocumentFanoutStep::json_to_table("just plain text"), None);
    }

    #[test]
    fn test_returns_none_for_json_without_rows() {
        assert_eq!(DocumentFanoutStep::json_to_table(r#"{"category": "EPA/play"}"#), None);
    }
}

// ------------------------------------------------------------------ #
// Full execute() integration                                         #
// ------------------------------------------------------------------ #

mod fanout_integration {
    use super::*;

    #[test]
    fn test_execute_triggers_fanout_extraction() {
        let extraction_calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&extraction_calls);

        let llm = llm_8k("OK");
        llm.set_responder(move |prompt| {
            let p = prompt.to_lowercase();
            if p.contains("extract the information relevant") {
                counter.fetch_add(1, Ordering::SeqCst);
                return Ok("Extracted fact from document.".to_string());
            }
            if p.contains("classify") || p.contains("intent") {
                return Ok("conversation".to_string());
            }
            Ok("OK".to_string())
        });

        let orch = orchestrator_over(&llm);
        let sctx: SharedSearchContext = SharedCell::wrap(SearchContext {
            vector_context: "raw concat".to_string(),
            documents: docs(3, 100),
            citations: Some(
                (0..3).map(|i| obj(json!({"document_id": format!("d{i}"), "document_name": format!("doc{i}.pdf")}))).collect(),
            ),
            ..Default::default()
        });

        let collected = sctx.clone();
        orch.set_collect_override(Some(Arc::new(move |_ctx| collected.clone())));
        let result = orch.execute(
            "summarize all the documents",
            &[],
            ExecuteOptions { skills: None, cwd: Some("/tmp".to_string()), ..Default::default() },
        );

        assert!(extraction_calls.load(Ordering::SeqCst) >= 1);
        assert!(sctx.with_ref(|s| s.vector_context.contains("Extracted fact from document.")));
        assert!(result.handled);
    }
}

/// A model without batched generation answers `invoke_batch` one prompt at a
/// time, in prompt order.
#[test]
fn test_invoke_batch_defaults_to_one_at_a_time_in_order() {
    use smartloop::llms::base_llm::{BaseLLM, BatchPrompt};
    let llm = llm_8k("unused");
    llm.set_responder(|p| Ok(format!("echo:{}", p.chars().last().unwrap_or(' '))));
    let prompts: Vec<BatchPrompt> = ["a", "b", "c"].iter().map(|p| BatchPrompt::new(*p, None)).collect();
    assert_eq!(llm.shared().invoke_batch(&prompts, None), vec!["echo:a", "echo:b", "echo:c"]);
    assert_eq!(llm.call_count(), 3);
}

/// A cancelled batch stops asking the model: prompts after the cancel come
/// back empty.
#[test]
fn test_invoke_batch_stops_once_cancelled() {
    use smartloop::llms::base_llm::{BaseLLM, BatchPrompt};
    use std::sync::atomic::AtomicBool;
    let llm = llm_8k("unused");
    let cancelled = Arc::new(AtomicBool::new(false));
    let flag = cancelled.clone();
    llm.set_responder(move |p| {
        flag.store(true, Ordering::SeqCst);
        Ok(format!("echo:{}", p.chars().last().unwrap_or(' ')))
    });
    let prompts: Vec<BatchPrompt> = ["a", "b", "c"].iter().map(|p| BatchPrompt::new(*p, None)).collect();
    assert_eq!(llm.shared().invoke_batch(&prompts, Some(cancelled)), vec!["echo:a", "", ""]);
    assert_eq!(llm.call_count(), 1);
}

/// Once the request is cancelled, a fan-out node skips its documents instead
/// of extracting them for nobody.
#[test]
fn test_fanout_node_skips_work_after_cancel() {
    use std::sync::atomic::AtomicBool;
    let (orch, llm) = make_orchestrator();
    llm.set_reply("EXTRACT");
    let c = summarize_ctx(&orch);
    c.write().cancelled = Some(Arc::new(AtomicBool::new(true)));
    let sctx = SearchContext { documents: docs(3, 100), ..Default::default() };
    let shared: SharedSearchContext = Arc::new(parking_lot::Mutex::new(sctx));
    let built = orch.doc_fanout().build(&c, Some(&shared), &smartloop::agents::nodes::AgentNode::default());
    for (node, handler) in &built.pairs {
        if node.is_llm_decision() {
            let result = handler(HashMap::new()).unwrap();
            assert_eq!(result.data["extracts"], json!([]));
        }
    }
    assert_eq!(llm.call_count(), 0);
}
