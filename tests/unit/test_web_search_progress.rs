//! Port of `tests/unit/test_web_search_progress.py`.
//!
//! Web search reports its progress on the SSE stream.
//!
//! Since web search moved into `Explorer::collect_parallel` it ran with no
//! progress events at all — only the legacy `WEB_SEARCH` plan step emitted
//! one.  It is also the longest leg of a request (a provider round-trip plus
//! one serial fetch per result), so the client sat on the previous step's
//! status line for the whole search.

use std::sync::Arc;

use parking_lot::Mutex;

use smartloop::agents::fetch_agent::UrlFetcher;
use smartloop::agents::mcp::explorer::{ProgressCallback, SearchContext};
use smartloop::agents::web_search_agent::{ProgressCallback as AgentProgress, WebSearchAgent};

use crate::shared::explorer_fixtures::{
    bare_explorer, messages, planned, recording_sink, EmptySearch, FakeWebSearch, ProgressEvent,
    ScriptedResolver,
};

// ── the agent reports each page as it is fetched ──────────────────────────
//
// Python patched `WebSearchAgent.is_enabled` → True and `_fetch_links` → the
// links below, so `search()` went straight to fetching them.  That remaining
// leg is `fetch_top`, called here with the agent's `top_k` / `num_candidates`
// exactly as `search()` passes them.

/// Fetches every URL as a page long enough to be usable.
struct PageFetcher;

impl UrlFetcher for PageFetcher {
    fn fetch(&self, url: &str, _max_length: usize, _start_index: usize, _raw: bool) -> String {
        format!("content of {url} ").repeat(40)
    }
}

/// `WebSearchAgent(fetch_agent=fetch, top_k=..., num_candidates=2, llm=None)`:
/// llm=None → no rerank, so the endpoint order is kept and the assertions
/// below can name the sources they expect.
fn agent(top_k: usize) -> WebSearchAgent {
    WebSearchAgent::new(Some(Arc::new(PageFetcher)), top_k, 2, None)
}

/// `agent.search("canyon bikes", progress=sink)` with `links` as the results.
fn search(agent: &WebSearchAgent, top_k: usize, links: &[&str], progress: Option<&AgentProgress>) -> String {
    let links: Vec<String> = links.iter().map(|l| l.to_string()).collect();
    agent.fetch_top("canyon bikes", &links, top_k, 2, progress)
}

fn events(sink: &Arc<Mutex<Vec<ProgressEvent>>>) -> Vec<String> {
    messages(sink)
}

#[test]
fn test_each_fetched_page_is_reported() {
    let (sink, seen) = recording_sink();
    let links = ["https://example.com/a", "https://other.org/b"];

    search(&agent(2), 2, &links, Some(&sink));

    let messages = events(&seen);
    assert!(messages.contains(&"Reading example.com".to_string()), "{messages:?}");
    assert!(messages.contains(&"Reading other.org".to_string()), "{messages:?}");
}

#[test]
fn test_a_summary_is_reported_when_the_search_finishes() {
    let (sink, seen) = recording_sink();
    search(&agent(2), 2, &["https://example.com/a", "https://other.org/b"], Some(&sink));

    assert_eq!(events(&seen).last().map(String::as_str), Some("Read 2 web sources"));
}

#[test]
fn test_the_summary_is_singular_for_one_source() {
    let (sink, seen) = recording_sink();
    // `agent._top_k = 1`
    search(&agent(1), 1, &["https://example.com/a"], Some(&sink));

    assert_eq!(events(&seen).last().map(String::as_str), Some("Read 1 web source"));
}

#[test]
fn test_events_use_the_web_search_step_name() {
    let (sink, seen) = recording_sink();
    search(&agent(2), 2, &["https://example.com/a"], Some(&sink));

    let seen = seen.lock();
    assert!(!seen.is_empty());
    assert!(seen.iter().all(|(step, _, _)| step == "web_search"), "{seen:?}");
    assert!(seen.iter().all(|(_, status, _)| status == "processing"), "{seen:?}");
}

/// Progress is cosmetic; the answer is not.
#[test]
fn test_a_failing_sink_never_breaks_the_search() {
    let sink: AgentProgress = Arc::new(|_: &str, _: &str, _: &str| panic!("client went away"));
    let content = search(&agent(2), 2, &["https://example.com/a"], Some(&sink));

    assert!(content.contains("example.com"), "{content}");
}

#[test]
fn test_search_without_a_sink_still_works() {
    assert!(!search(&agent(2), 2, &["https://example.com/a"], None).is_empty());
}

// ── collect never searches ────────────────────────────────────────────────
//
// The web is searched only when the model asks, through the client's
// `web_search` tool, so collect neither searches nor announces a search, even
// with search on for the project and nothing local to answer from.

fn collect(progress: Option<ProgressCallback>) -> (Arc<FakeWebSearch>, SearchContext) {
    let web = FakeWebSearch::new(true, "WEB", &["https://example.com/a"]);
    let ex = bare_explorer(None, None, ScriptedResolver::returning(None), web.clone(), planned(vec![]));
    let ctx = ex.collect_parallel(
        "what did they announce today",
        Some(Arc::new(EmptySearch)),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        progress,
        None,
    );
    (web, ctx)
}

#[test]
fn test_collect_neither_searches_nor_announces_a_search() {
    let (sink, seen) = recording_sink();
    let (web, ctx) = collect(Some(sink));

    assert!(seen.lock().is_empty());
    assert!(web.searched_with.lock().is_empty());
    assert_eq!(ctx.web_context, "");
}
