//! Web search agent — pulls the latest information/events from the web.
//!
//! Port of `smartloop/agents/web_search_agent.py`.
//!
//! Web search is a Pro/premium feature. When enabled on the project, this agent
//! calls the hosted Smartloop endpoint `GET {slp_api_base}/v1/websearch` (which
//! returns a JSON array of result links), then fetches up to `num_candidates`
//! links (via the [`FetchAgent`], HTML/PDF → markdown), LLM-reranks them by
//! relevance to the query, and returns the top `top_k` sources so the LLM can
//! reason over up-to-date information.
//!
//! Enablement is per project (`Project.web_search_enabled`), toggled via
//! `PATCH /v1/projects/{id}/websearch`. Whether the endpoint will actually serve
//! a search is decided entirely server-side and is read up front from
//! `GET /v1/websearch/status` so a search that cannot run is skipped. Any
//! failure (disabled, 401/403, no credentials, request error, no results) is a
//! silent skip that returns `""` so the model falls back to its own knowledge.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use crate::agents::agent::SharedLLM;
use crate::agents::fetch_agent::{FetchAgent, UrlFetcher};
#[allow(unused_imports)]
use crate::search::providers::{self, SearchProviderError};
#[allow(unused_imports)]
use crate::workspace::workspace::Workspace;

/// `progress(step, status, message)` sink reporting live activity.
pub type ProgressCallback = Arc<dyn Fn(&str, &str, &str) + Send + Sync>;

/// Leading conversational filler stripped before the search query hits the
/// hosted endpoint.
pub const WEBSEARCH_FILLERS: &[&str] = &[
    "give me a list of ",
    "give me a list ",
    "give a list of ",
    "give me ",
    "can you give me ",
    "could you give me ",
    "can you tell me ",
    "could you tell me ",
    "tell me about ",
    "tell me ",
    "show me ",
    "can you show me ",
    "can you ",
    "could you ",
    "i want to know about ",
    "i want to know ",
    "i want to ",
    "i would like to know ",
    "i would like ",
    "i'd like to know ",
    "i'd like ",
    "do you know about ",
    "do you know ",
    "do you have any info on ",
    "do you have ",
    "what are the best ",
    "what are the latest ",
    "what are ",
    "what is the latest ",
    "what is ",
    "whats the latest ",
    "whats ",
    "any info on ",
    "info on ",
    "about ",
    "a list of ",
    "list of ",
    "please ",
];

/// Trailing politeness particles trimmed from the query.
pub const WEBSEARCH_TRAILERS: &[&str] = &[" please", " please?", " thanks", " thank you"];

/// Generic / temporal tokens that add no retrieval signal to a web query.
pub const WEBSEARCH_STOPWORDS: &[&str] = &[
    "the", "a", "an", "for", "of", "to", "and", "or", "in", "on", "at", "with", "current",
    "latest", "today", "todays", "now", "please",
];

/// Distill a conversational message into a web-search query.
///
/// `"give me a list latest canyon bikes"` → `"canyon bikes"`.  Falls back to
/// the raw message when nothing meaningful remains.
pub fn normalize_search_query(raw: &str) -> String {
    let _ = raw;
    todo!()
}

/// The web-search surface the orchestrator and explorer use (a seam so tests
/// can substitute a fake searcher).
pub trait WebSearcher: Send + Sync {
    /// Search the web for `query` and return fetched result content, or `""`.
    fn search(
        &self,
        query: &str,
        top_k: Option<usize>,
        num_candidates: Option<usize>,
        progress: Option<&ProgressCallback>,
    ) -> String;

    /// URLs of the sources kept after the last search (reranked order).
    fn last_sources(&self) -> Vec<String>;

    /// url → the page text handed to the model for that source.
    fn last_source_extracts(&self) -> HashMap<String, String>;

    /// True when the orchestrator should run a web search.
    fn is_enabled(&self) -> bool;
}

/// Search the web via the hosted endpoint and return fetched content.
pub struct WebSearchAgent {
    fetch_agent: Mutex<Option<Arc<dyn UrlFetcher>>>,
    top_k: usize,
    num_candidates: usize,
    llm: Option<SharedLLM>,
    /// URLs of the sources kept after the last `search()` (top_k, reranked).
    pub last_sources: RwLock<Vec<String>>,
    /// url → the page text handed to the model for that source.
    pub last_source_extracts: RwLock<HashMap<String, String>>,
}

impl Default for WebSearchAgent {
    fn default() -> Self {
        Self::new(None, 3, 10, None)
    }
}

impl WebSearchAgent {
    /// Build the agent.  `fetch_agent` defaults lazily to a [`FetchAgent`];
    /// `llm` (optional) judges the rerank.  Python defaults: `top_k=3`,
    /// `num_candidates=10`.
    pub fn new(
        fetch_agent: Option<Arc<dyn UrlFetcher>>,
        top_k: usize,
        num_candidates: usize,
        llm: Option<SharedLLM>,
    ) -> Self {
        Self {
            fetch_agent: Mutex::new(fetch_agent),
            top_k,
            num_candidates,
            llm,
            last_sources: RwLock::new(Vec::new()),
            last_source_extracts: RwLock::new(HashMap::new()),
        }
    }

    /// The fetcher, creating a default [`FetchAgent`] on first use.
    pub fn fetch_agent(&self) -> Arc<dyn UrlFetcher> {
        self.fetch_agent
            .lock()
            .get_or_insert_with(|| Arc::new(FetchAgent::new()) as Arc<dyn UrlFetcher>)
            .clone()
    }

    /// The rerank LLM, if any.
    pub fn llm(&self) -> Option<&SharedLLM> {
        self.llm.as_ref()
    }

    /// Search the web for `query` and return fetched result content.
    ///
    /// Returns `""` (never fails) when web search is disabled, forbidden,
    /// unauthenticated, or fails.  `progress` reports each page as it is
    /// fetched; a failing sink never breaks the search.
    pub fn search(
        &self,
        query: &str,
        top_k: Option<usize>,
        num_candidates: Option<usize>,
        progress: Option<&ProgressCallback>,
    ) -> String {
        let _ = (query, top_k, num_candidates, progress);
        todo!()
    }

    /// Return result URLs for `query` from the hosted endpoint.
    pub fn fetch_links(&self, query: &str, count: usize) -> Result<Vec<String>, SearchProviderError> {
        let _ = (query, count);
        todo!()
    }

    /// Report a search step, never letting a bad sink break the search.
    pub fn emit(progress: Option<&ProgressCallback>, message: &str) {
        let _ = (progress, message);
        todo!()
    }

    /// Host of `url` for display, falling back to the raw URL.
    pub fn source_label(url: &str) -> String {
        let _ = url;
        todo!()
    }

    /// Fetch the links, then LLM-rerank and keep the `top_k` most relevant.
    pub fn fetch_top(
        &self,
        query: &str,
        links: &[String],
        top_k: usize,
        num_candidates: usize,
        progress: Option<&ProgressCallback>,
    ) -> String {
        let _ = (query, links, top_k, num_candidates, progress);
        todo!()
    }

    /// Re-order `candidates` `(url, content)` by LLM-judged relevance; falls
    /// back to the input order when no LLM is wired in or the call fails.
    pub fn rerank(&self, query: &str, candidates: Vec<(String, String)>) -> Vec<(String, String)> {
        let _ = (query, candidates);
        todo!()
    }

    /// True when fetched content is real page text (not empty / JS-only).
    pub fn is_usable(content: &str) -> bool {
        let _ = content;
        todo!()
    }

    /// True when the orchestrator should run a web search: the project flag
    /// (read from disk) and then the hosted status endpoint.
    pub fn is_enabled() -> bool {
        todo!()
    }
}

impl WebSearcher for WebSearchAgent {
    fn search(
        &self,
        query: &str,
        top_k: Option<usize>,
        num_candidates: Option<usize>,
        progress: Option<&ProgressCallback>,
    ) -> String {
        WebSearchAgent::search(self, query, top_k, num_candidates, progress)
    }

    fn last_sources(&self) -> Vec<String> {
        self.last_sources.read().clone()
    }

    fn last_source_extracts(&self) -> HashMap<String, String> {
        self.last_source_extracts.read().clone()
    }

    fn is_enabled(&self) -> bool {
        WebSearchAgent::is_enabled()
    }
}
