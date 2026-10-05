//! Port of `smartloop/agents/orchestration/document_fanout.py`.
//!
//! Document map-reduce: one DAG node per retrieved document.
//!
//! EVERY retrieved document gets its own node, so the answer covers them all
//! and nothing is dropped for volume.  Processing each document independently
//! is also what makes per-document relevance possible: a document the query
//! doesn't ask about is filtered out by its own `NONE` extraction instead of
//! leaking into a co-batched document's result.
//!
//! The reduce step merges the surviving extracts and narrows the citation list
//! to exactly the documents that contributed.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::agents::mcp::explorer::SearchContext;
use crate::agents::nodes::{AgentNode, LLMDecisionNode, NodeHandler, NodeResult, TransformNode};
use crate::agents::orchestration::context::{OrchestratorContext, SharedOrchestratorContext};
use crate::agents::orchestration::model_router::ModelRouter;
use crate::agents::orchestration::replies::{is_none_extract, is_placeholder_output};
use crate::agents::orchestration::step::{NodePair, SharedSearchContext, Step, StepNodes};
use crate::agents::planner::ModelTier;
use crate::config::app_settings::AppSettings;
use crate::types::ModelCapability;

/// Python logger name of this module.
pub const LOGGER: &str = "smartloop.agents.orchestration.document_fanout";

/// Node id of the reduce node's result.
pub const DOC_REDUCE_NODE: &str = "doc_reduce";

/// Conservative chars/token floor used to size extraction input for
/// data-dense content.  Prose runs closer to 4 chars/token but dense data
/// (stats, JSON tables) can drop to ~2, and an optimistic estimate overflows
/// the window and kills extraction entirely.
pub const CHARS_PER_TOKEN: f64 = 2.0;

/// Tokens reserved for prompt scaffolding + output when sizing extraction.
pub const OUTPUT_RESERVE_TOKENS: usize = 1024;

/// Default extraction input cap when the model's window is unknown.
pub const DEFAULT_MAX_CHARS: usize = 16000;

/// Raw-excerpt length used when the extract is unusable.
pub const RAW_EXCERPT_CHARS: usize = 2000;

/// The first `n` characters of `text` (Python `text[:n]`).
fn char_prefix(text: &str, n: usize) -> &str {
    match text.char_indices().nth(n) {
        Some((idx, _)) => &text[..idx],
        None => text,
    }
}

/// Python `str(d.get(key, ""))` for a document field: a missing key is `""`,
/// a string is itself, anything else its Python-ish rendering.
fn doc_field(doc: &Map<String, Value>, key: &str) -> String {
    match doc.get(key) {
        None => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) => "None".to_string(),
        Some(Value::Bool(true)) => "True".to_string(),
        Some(Value::Bool(false)) => "False".to_string(),
        Some(other) => other.to_string(),
    }
}

/// Builds and runs the per-document extract → reduce sub-DAG.
///
/// Cheap to clone, so DAG handlers capture a clone.
#[derive(Clone)]
pub struct DocumentFanoutStep {
    models: Arc<ModelRouter>,
}

impl DocumentFanoutStep {
    /// Step name.
    pub const NAME: &'static str = "documents";

    /// When at least this many documents match, each document is processed in
    /// its own DAG node and reduced into a single result.  The number
    /// processed concurrently is bounded by the LLM worker pool; overflow
    /// queues.
    pub const THRESHOLD: usize = 1;

    /// Build the step over the shared model router.
    pub fn new(models: Arc<ModelRouter>) -> Self {
        Self { models }
    }

    /// Map each document to its own batch (one document per fan-out node).
    ///
    /// EVERY retrieved document gets its own node so the answer covers them
    /// all — nothing is dropped regardless of count.  Processing each document
    /// independently also means a document the query doesn't ask about (e.g.
    /// another person's invoice) is filtered out by its own `NONE` extraction
    /// instead of leaking into a co-batched document's result.
    ///
    /// The number processed concurrently is bounded by the LLM worker pool
    /// (and the DAG thread pool); any overflow simply waits in the pool's
    /// queue and is processed as workers free up.
    pub fn batch_documents(&self, documents: &[Map<String, Value>]) -> Vec<Vec<Map<String, Value>>> {
        documents.iter().map(|doc| vec![doc.clone()]).collect()
    }

    /// See [`crate::utils::doc_splitter::json_to_table`].
    pub fn json_to_table(text: &str) -> Option<String> {
        crate::utils::json_to_table(text)
    }

    /// True when the text contains a complete markdown table.  See
    /// [`crate::utils::doc_splitter::is_markdown_table`].
    pub fn is_markdown_table(text: &str) -> bool {
        crate::utils::is_markdown_table(text)
    }

    /// Create a handler that extracts query-relevant content from one batch.
    ///
    /// Inference uses the shared in-process model, serialized upstream by the
    /// DAG executor's LLM lock.  Structured (JSON / markdown-table) documents
    /// are used deterministically; otherwise the DP-capability model extracts,
    /// sized to its own window.
    pub fn make_batch_handler(
        &self,
        ctx: &SharedOrchestratorContext,
        node_id: &str,
        batch: Vec<Map<String, Value>>,
        skill_instruction: Option<String>,
    ) -> NodeHandler {
        let step = self.clone();
        let ctx = ctx.clone();
        let node_id = node_id.to_string();
        Arc::new(move |_deps: HashMap<String, NodeResult>| {
            Ok(step.run_batch(&ctx, &node_id, &batch, skill_instruction.as_deref()))
        })
    }

    /// Body of a batch handler (see [`Self::make_batch_handler`]).
    fn run_batch(
        &self,
        ctx: &SharedOrchestratorContext,
        node_id: &str,
        batch: &[Map<String, Value>],
        skill_instruction: Option<&str>,
    ) -> NodeResult {
        let doc_names: Vec<String> = batch.iter().map(|d| doc_field(d, "document_name")).collect();
        let named: Vec<&str> =
            doc_names.iter().map(String::as_str).filter(|n| !n.is_empty()).collect();
        let reading = if named.is_empty() {
            "Reading document".to_string()
        } else {
            format!("Reading document: {}", named.join(", "))
        };
        self.models.emit("read", "processing", &reading);
        let combined = batch
            .iter()
            .map(|d| {
                format!(
                    "=== Document: {} ===\n{}",
                    doc_field(d, "document_name"),
                    doc_field(d, "content")
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        // Structured data documents are used deterministically — no LLM
        // context window, so large files can't overflow the extractor, and
        // exact values are preserved.  Two shapes are handled:
        //   * raw JSON (an array of objects, e.g. a {"rows": [...]} dump)
        //     → converted to a table on the fly;
        //   * content already normalized to a markdown table at index time
        //     → used directly.
        // Both skip loading the DP model entirely.
        let mut table_parts: Vec<String> = Vec::new();
        for d in batch {
            let doc_name = doc_field(d, "document_name");
            let content = doc_field(d, "content");
            let mut table = Self::json_to_table(&content);
            if table.is_none() && Self::is_markdown_table(&content) {
                table = Some(content.clone());
            }
            if let Some(t) = table {
                table_parts.push(if doc_name.is_empty() { t } else { format!("{doc_name}\n{t}") });
            }
        }
        if !table_parts.is_empty() {
            let table = table_parts.join("\n\n---\n\n");
            tracing::info!(
                target: LOGGER,
                "[doc-batch] {}: json→table conversion → {} chars",
                node_id,
                table.chars().count()
            );
            return NodeResult::with_data(
                node_id,
                json!({"doc_names": doc_names, "content": table}),
            );
        }

        // Size the extraction input to the EXTRACTION model's actual context
        // window — not the router's.  The router's max_tokens can exceed the
        // DP model's window (e.g. 8192 vs 6080), so sizing from it overflows
        // the extractor and generation silently fails.
        let mut max_chars = DEFAULT_MAX_CHARS;
        let dp_llm = self.models.llm_for_tier(ModelTier::Large, Some(ModelCapability::Dp.value()), None);
        tracing::info!(
            target: LOGGER,
            "[doc-batch] {} → {}",
            node_id,
            ModelRouter::model_label(dp_llm.as_ref())
        );
        let ctx_tokens = dp_llm.get_max_tokens();
        if ctx_tokens > 0 {
            let usable = std::cmp::max(512, ctx_tokens.saturating_sub(OUTPUT_RESERVE_TOKENS));
            max_chars = (usable as f64 * CHARS_PER_TOKEN) as usize;
        }
        match AppSettings::try_new() {
            Ok(settings) => {
                if settings.doc_extract_max_chars > 0 {
                    max_chars = max_chars.min(settings.doc_extract_max_chars as usize);
                }
            }
            Err(e) => {
                tracing::warn!(target: LOGGER, "[doc-batch] {} context sizing failed: {}", node_id, e);
            }
        }

        let user_message = ctx.read().user_message.clone();
        let guidance = match skill_instruction.filter(|s| !s.is_empty()) {
            Some(instruction) => format!("Guidance:\n{instruction}\n\n"),
            None => String::new(),
        };
        let prompt = format!(
            "<system>\n\
             You extract the information relevant to the user's request from \
             the document(s) below. Preserve concrete facts, numbers, names, \
             dates, and identifiers exactly as written. Be faithful to the \
             source — do not invent or infer beyond it. Return a concise \
             factual extract; if nothing is relevant, reply with NONE.\n\
             </system>\n\
             <user>\n\
             User request: {user_message}\n\n\
             {guidance}\
             Documents:\n{}\n\
             </user>",
            char_prefix(&combined, max_chars)
        );

        let mut extract = String::new();
        let mut llm_failed = false;
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            dp_llm.invoke(&prompt, None, None)
        })) {
            Ok(reply) => extract = reply.trim().to_string(),
            Err(panic) => {
                llm_failed = true;
                let reason = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                tracing::warn!(target: LOGGER, "[doc-batch] {} extraction failed: {}", node_id, reason);
            }
        }

        if is_none_extract(&extract) {
            // The LLM judged this document irrelevant to the request (it was
            // told to reply NONE).  Respect that and drop the batch so an
            // entity-specific query ("items Adam Hart purchased") doesn't pull
            // in other people's documents.
            tracing::info!(
                target: LOGGER,
                "[doc-batch] {}: not relevant → dropped ({})",
                node_id,
                named.join(", ")
            );
            extract = String::new();
        } else if extract.is_empty() || llm_failed || is_placeholder_output(&extract) {
            // No usable extract: the model errored, returned empty, or
            // produced only a placeholder (e.g. llama.cpp's empty-response
            // fallback "No response generated" — which also happens when the
            // extraction prompt overflows the model's context window and
            // generation silently fails).  Fall back to a raw excerpt so a
            // degenerate model never replaces the document with placeholder
            // text.
            tracing::info!(
                target: LOGGER,
                "[doc-batch] {}: extract unusable → falling back to raw excerpt ({} chars)",
                node_id,
                extract.chars().count()
            );
            extract = char_prefix(&combined, RAW_EXCERPT_CHARS).to_string();
        }

        tracing::info!(
            target: LOGGER,
            "[doc-batch] {}: {} doc(s) ({}) → {} chars",
            node_id,
            batch.len(),
            named.join(", "),
            extract.chars().count()
        );
        NodeResult::with_data(node_id, json!({"doc_names": doc_names, "content": extract}))
    }

    /// Create the reduce handler that merges batch extracts into one document,
    /// narrowing citations to the surviving documents.
    ///
    /// The shared [`SearchContext`] is updated in place (Python mutated the
    /// one request-wide object) and also travels as the result payload
    /// (Python's `data["search_ctx"]`), as a [`SharedSearchContext`].
    pub fn make_reduce_handler(
        &self,
        ctx: &SharedOrchestratorContext,
        search_ctx: SharedSearchContext,
        batch_node_ids: Vec<String>,
    ) -> NodeHandler {
        let _ = ctx;
        Arc::new(move |deps: HashMap<String, NodeResult>| {
            Ok(Self::reduce(&search_ctx, &batch_node_ids, &deps))
        })
    }

    /// Body of the reduce handler (see [`Self::make_reduce_handler`]).
    fn reduce(
        search_ctx: &SharedSearchContext,
        batch_node_ids: &[String],
        deps: &HashMap<String, NodeResult>,
    ) -> NodeResult {
        // Merge in batch order (most relevant first) for determinism.  Each
        // extract is prefixed with its source document name(s) so the
        // downstream LLM can attribute every fact to the right document
        // instead of anchoring the whole answer on a single source.
        let mut parts: Vec<String> = Vec::new();
        // Docs that actually contributed.
        let mut surviving_names: Vec<String> = Vec::new();
        // name -> the extract fed to the model for that document.  Carried on
        // the citation so the router can check the finished answer actually
        // drew on it before crediting the source.
        let mut extract_by_name: HashMap<String, String> = HashMap::new();
        for node_id in batch_node_ids {
            let Some(dep_result) = deps.get(node_id) else {
                continue;
            };
            if !dep_result.success {
                continue;
            }
            let Some(data) = dep_result.data.as_object() else {
                continue;
            };
            let content = data.get("content").and_then(Value::as_str).unwrap_or("");
            if content.is_empty() {
                continue;
            }
            let names: Vec<String> = data
                .get("doc_names")
                .and_then(Value::as_array)
                .map(|names| {
                    names
                        .iter()
                        .filter_map(Value::as_str)
                        .filter(|n| !n.is_empty())
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            surviving_names.extend(names.iter().cloned());
            for n in &names {
                extract_by_name.insert(n.clone(), content.to_string());
            }
            let header = if names.is_empty() { "Unknown source".to_string() } else { names.join(", ") };
            parts.push(format!("=== Source: {header} ===\n{content}"));
        }

        let merged = parts.join("\n\n---\n\n");
        let citations = {
            let mut sc = search_ctx.lock();
            // Replace the raw concatenated context with the reduced,
            // query-focused document so every downstream step sees it.
            //
            // An empty merge means the per-document judge dropped every
            // retrieved document as irrelevant (each replied NONE).  That
            // verdict has to clear the context, not leave it alone: keeping
            // the raw concatenation resurrected exactly the documents the
            // fan-out just rejected, and the answer was then framed as
            // DOCUMENT CONTEXT built from them — so the model reported it had
            // no context for the question rather than answering it directly.
            if !merged.is_empty() {
                sc.vector_context = merged.clone();
            } else {
                sc.vector_context = String::new();
                sc.citations = None;
                sc.cited_documents = None;
                tracing::info!(
                    target: LOGGER,
                    "[doc-reduce] every retrieved document judged irrelevant — cleared document context"
                );
            }

            // Bundle citations from ONLY the surviving (relevant) documents.
            // Build the list directly from the surviving doc names (deduped,
            // in relevance order) rather than filtering the full citation list
            // — so we never accidentally keep citations for documents the
            // fan-out dropped.  Reuse page info from the original citations
            // when a name matches; synthesize a minimal entry otherwise.
            // Record the surviving set so the final citation list stays
            // restricted even if a later EXPLORE step re-collects.
            if !surviving_names.is_empty() {
                let mut seen: HashSet<&str> = HashSet::new();
                let ordered: Vec<String> = surviving_names
                    .iter()
                    .filter(|n| seen.insert(n.as_str()))
                    .cloned()
                    .collect();
                sc.cited_documents = Some(ordered.clone());
                let mut by_name: HashMap<String, Map<String, Value>> = HashMap::new();
                for c in sc.citations.iter().flatten() {
                    if let Some(name) = c.get("document_name").and_then(Value::as_str) {
                        if !name.is_empty() && !by_name.contains_key(name) {
                            by_name.insert(name.to_string(), c.clone());
                        }
                    }
                }
                let mut new_citations: Vec<Map<String, Value>> = Vec::new();
                for name in &ordered {
                    let mut citation = by_name.get(name).cloned().unwrap_or_else(|| {
                        let mut m = Map::new();
                        m.insert("document_name".into(), Value::String(name.clone()));
                        m
                    });
                    // Only fill in an extract when the citation has no source
                    // text yet.  The retrieval citation already carries the
                    // document's own words, which is the better evidence: this
                    // extract is the DP model's answer *to the user's query*,
                    // so it echoes the query's wording and would match the
                    // final answer no matter which document it came from.
                    // Private, stripped before the citation reaches a client.
                    let has_extract = citation.get("_extract").is_some_and(|v| match v {
                        Value::Null => false,
                        Value::Bool(b) => *b,
                        Value::String(s) => !s.is_empty(),
                        Value::Array(a) => !a.is_empty(),
                        Value::Object(o) => !o.is_empty(),
                        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
                    });
                    if !has_extract {
                        if let Some(extract) = extract_by_name.get(name).filter(|e| !e.is_empty()) {
                            citation.insert("_extract".into(), Value::String(extract.clone()));
                        }
                    }
                    new_citations.push(citation);
                }
                sc.citations = Some(new_citations);
            }
            sc.citations.clone()
        };

        tracing::info!(
            target: LOGGER,
            "[doc-reduce] merged {} batch extract(s) → {} chars; cited {} surviving doc(s)",
            parts.len(),
            merged.chars().count(),
            citations.as_ref().map(Vec::len).unwrap_or(0)
        );
        let citations_value = match citations {
            Some(list) => Value::Array(list.into_iter().map(Value::Object).collect()),
            None => Value::Null,
        };
        NodeResult::with_data(
            DOC_REDUCE_NODE,
            json!({"content": merged, "citations": citations_value}),
        )
        .with_payload(search_ctx.clone())
    }
}

impl Step for DocumentFanoutStep {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    /// True when the retrieved documents warrant the fan-out.
    fn applies_to(&self, _ctx: &OrchestratorContext, search_ctx: Option<&SearchContext>) -> bool {
        search_ctx.is_some_and(|sc| sc.documents.len() >= Self::THRESHOLD)
    }

    /// Build the document map-reduce sub-DAG.
    ///
    /// The reduce node becomes the parent for everything downstream, so later
    /// steps see the merged extract rather than racing the individual batches.
    fn build(
        &self,
        ctx: &SharedOrchestratorContext,
        search_ctx: Option<&SharedSearchContext>,
        parent: &AgentNode,
    ) -> StepNodes {
        let Some(search_ctx) = search_ctx else {
            return StepNodes::none(parent.clone());
        };
        let skill_instruction = ctx.read().skill_context.clone();

        let documents = search_ctx.lock().documents.clone();
        let batches = self.batch_documents(&documents);
        tracing::info!(
            target: LOGGER,
            "[doc-fanout] {} document(s) → {} batch node(s) (serialized in-thread)",
            documents.len(),
            batches.len()
        );

        let mut nodes_and_handlers: Vec<NodePair> = Vec::new();
        let mut batch_nodes: Vec<AgentNode> = Vec::new();
        for batch in batches {
            let node = LLMDecisionNode::new(&[parent]);
            let handler =
                self.make_batch_handler(ctx, &node.node_id, batch, skill_instruction.clone());
            nodes_and_handlers.push((node.clone(), handler));
            batch_nodes.push(node);
        }

        let batch_node_ids: Vec<String> = batch_nodes.iter().map(|n| n.node_id.clone()).collect();
        let batch_refs: Vec<&AgentNode> = batch_nodes.iter().collect();
        let reduce_node = TransformNode::new(&batch_refs);
        let reduce_handler = self.make_reduce_handler(ctx, search_ctx.clone(), batch_node_ids);
        nodes_and_handlers.push((reduce_node.clone(), reduce_handler));

        StepNodes::new(nodes_and_handlers, reduce_node)
    }
}

#[cfg(test)]
mod tests {
    use parking_lot::{Mutex, RwLock};

    use super::*;
    use crate::agents::orchestration::model_router::test_llm::ScriptedLlm;
    use crate::agents::orchestration::model_router::ModelRouterOptions;

    fn docs(n: usize) -> Vec<Map<String, Value>> {
        (0..n)
            .map(|i| {
                let mut d = Map::new();
                d.insert("document_name".into(), Value::String(format!("doc{i}.pdf")));
                d.insert("content".into(), Value::String(format!("{} body {i}", "X".repeat(100))));
                d
            })
            .collect()
    }

    fn step(reply: impl Fn(&str) -> String + Send + Sync + 'static) -> DocumentFanoutStep {
        let llm = ScriptedLlm::new(reply);
        DocumentFanoutStep::new(Arc::new(ModelRouter::new(llm.clone(), llm, ModelRouterOptions::default())))
    }

    fn ctx() -> SharedOrchestratorContext {
        Arc::new(RwLock::new(OrchestratorContext::new("summarize the documents", vec![], vec![], "")))
    }

    fn reduce_result(content: &str, names: &[&str]) -> NodeResult {
        NodeResult::with_data("batch_0", json!({"doc_names": names, "content": content}))
    }

    #[test]
    fn each_document_gets_its_own_batch() {
        let batches = step(|_| "OK".into()).batch_documents(&docs(5));
        assert_eq!(batches.len(), 5);
        assert!(batches.iter().all(|b| b.len() == 1));
    }

    #[test]
    fn reduce_restricts_citations_to_surviving_documents() {
        let fanout = step(|p| if p.contains("body 0") { "Relevant extract.".into() } else { "NONE".into() });
        let mut c0 = Map::new();
        c0.insert("document_id".into(), json!("0"));
        c0.insert("document_name".into(), json!("doc0.pdf"));
        let mut c1 = Map::new();
        c1.insert("document_id".into(), json!("1"));
        c1.insert("document_name".into(), json!("doc1.pdf"));
        let sc: SharedSearchContext = Arc::new(Mutex::new(SearchContext {
            vector_context: "ORIGINAL RAW CONCAT".into(),
            documents: docs(2),
            citations: Some(vec![c0, c1]),
            ..Default::default()
        }));
        let ctx = ctx();
        let parent = AgentNode::default();
        assert!(fanout.applies_to(&ctx.read(), Some(&sc.lock())));
        let built = fanout.build(&ctx, Some(&sc), &parent);
        assert!(built.parent.is_transform());
        assert_eq!(built.pairs.len(), 3);

        let mut results = HashMap::new();
        let mut reduce = None;
        for (node, handler) in &built.pairs {
            if node.is_transform() {
                reduce = Some(handler.clone());
                continue;
            }
            assert_eq!(node.depends_on, vec![parent.node_id.clone()]);
            results.insert(node.node_id.clone(), handler(HashMap::new()).unwrap());
        }
        let out = (reduce.unwrap())(results).unwrap();
        assert_eq!(out.node_id, DOC_REDUCE_NODE);
        let sc_now = sc.lock();
        assert_eq!(sc_now.vector_context, out.data["content"].as_str().unwrap());
        assert!(sc_now.vector_context.contains("=== Source: doc0.pdf ===\nRelevant extract."));
        assert_eq!(sc_now.cited_documents, Some(vec!["doc0.pdf".to_string()]));
        let cites = sc_now.citations.clone().unwrap();
        assert_eq!(cites.len(), 1);
        assert_eq!(cites[0]["document_id"], json!("0"));
        assert_eq!(cites[0]["_extract"], json!("Relevant extract."));
        assert!(out.payload_as::<SharedSearchContext>().is_some());
    }

    #[test]
    fn reduce_clears_context_when_every_document_is_dropped() {
        let fanout = step(|_| "OK".into());
        let mut c = Map::new();
        c.insert("document_name".into(), json!("invoice.pdf"));
        let sc: SharedSearchContext = Arc::new(Mutex::new(SearchContext {
            vector_context: "RAW: Q3 invoice".into(),
            citations: Some(vec![c]),
            ..Default::default()
        }));
        let handler = fanout.make_reduce_handler(&ctx(), sc.clone(), vec!["batch_0".into()]);
        let mut deps = HashMap::new();
        deps.insert("batch_0".to_string(), reduce_result("", &["invoice.pdf"]));
        handler(deps).unwrap();
        let sc = sc.lock();
        assert_eq!(sc.vector_context, "");
        assert!(sc.citations.is_none());
        assert!(sc.cited_documents.is_none());
    }

    #[test]
    fn batch_drops_none_and_falls_back_on_placeholders_and_failures() {
        let sc: SharedSearchContext =
            Arc::new(Mutex::new(SearchContext { documents: docs(1), ..Default::default() }));
        let parent = AgentNode::default();
        let run = |fanout: DocumentFanoutStep| {
            let built = fanout.build(&ctx(), Some(&sc), &parent);
            (built.pairs[0].1)(HashMap::new()).unwrap().data["content"].as_str().unwrap().to_string()
        };
        assert_eq!(run(step(|_| "NONE".into())), "");
        let placeholder = run(step(|_| "No response generated".into()));
        assert!(!placeholder.contains("No response generated"));
        assert!(placeholder.contains("body 0"));
        assert!(run(step(|_| panic!("boom"))).contains("body 0"));
        assert_eq!(run(step(|_| "  DP EXTRACT ".into())), "DP EXTRACT");
    }

    #[test]
    fn no_fanout_without_documents() {
        let fanout = step(|_| "OK".into());
        let c = ctx();
        assert!(!fanout.applies_to(&c.read(), Some(&SearchContext::default())));
        assert!(!fanout.applies_to(&c.read(), None));
    }
}
