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
//! a search is decided entirely server-side — signed in, with a search provider
//! registered on the account (managed from the web app) — and is read up front
//! from `GET /v1/websearch/status` so a search that cannot run is skipped rather
//! than attempted and discarded. This side never sees a provider name or an API
//! key. Web search is part of explore: any failure (disabled, 401/403, no
//! credentials, request error, no results) is a silent skip that returns `""`
//! so the model simply falls back to its own knowledge — no error or upsell is
//! injected into the answer.

use std::collections::{HashMap, HashSet};
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use regex::Regex;

use crate::llms::base_llm::BaseLLM;
use crate::agents::fetch_agent::{FetchAgent, UrlFetcher};
use crate::search::providers::{self, SearchProviderError};
use crate::workspace::workspace::Workspace;

/// Python logger name.
const LOG_TARGET: &str = "smartloop.agents.web_search_agent";

/// `progress(step, status, message)` sink reporting live activity.
pub type ProgressCallback = Arc<dyn Fn(&str, &str, &str) + Send + Sync>;

/// Leading conversational filler stripped before the search query hits the
/// hosted endpoint.  A raw message like "give me a list latest canyon bikes"
/// otherwise matches on the filler word ("give") and returns dictionary pages
/// instead of the actual subject.
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
/// The hosted endpoint otherwise latches onto them: "current stock prices for
/// apple" returns current.com pages, not AAPL quotes.  Dropped across the
/// WHOLE query (not just the leading filler).  A single-subject query like
/// "electric current" is kept intact via the >= 2 remaining words floor.
pub const WEBSEARCH_STOPWORDS: &[&str] = &[
    "the", "a", "an", "for", "of", "to", "and", "or", "in", "on", "at", "with", "current",
    "latest", "today", "todays", "now", "please",
];

static WHITESPACE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());
static DIGITS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\d+").unwrap());
static SCHEME_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[A-Za-z][A-Za-z0-9+\-.]*:").unwrap());

/// Characters Python's `.strip("?.,;:!")` removes.
const PUNCT: &[char] = &['?', '.', ',', ';', ':', '!'];

/// `re.sub(r"\s+", " ", s.strip()).strip("?.,;:!").strip()`.
fn squash(s: &str) -> String {
    WHITESPACE_RE.replace_all(s.trim(), " ").trim_matches(PUNCT).trim().to_string()
}

/// Python `s[n:]` by characters.
fn drop_chars(s: &str, n: usize) -> String {
    s.chars().skip(n).collect()
}

/// Python `repr()` of a string, for log lines that used `%r`.
fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Distill a conversational message into a web-search query.
///
/// `"give me a list latest canyon bikes"` → `"canyon bikes"`.  Polite framing
/// and filler is stripped from the front and trailing particles trimmed, then
/// weak/temporal stopwords ("the", "for", "current", "latest") are dropped
/// anywhere so the endpoint matches the actual subject instead of a filler
/// word.  Falls back to the raw message when nothing meaningful remains.
pub fn normalize_search_query(raw: &str) -> String {
    if raw.is_empty() {
        return raw.to_string();
    }
    let mut q = squash(raw);
    // Strip leading filler iteratively — "please tell me about the weather"
    // sheds "please", then "tell me", then "about".
    for _ in 0..6 {
        let low = format!(" {} ", q.to_lowercase());
        let hit = WEBSEARCH_FILLERS.iter().find(|f| low.starts_with(&format!(" {}", f)));
        let Some(hit) = hit else { break };
        q = drop_chars(&q, hit.chars().count());
    }
    q = squash(&q);
    for tail in WEBSEARCH_TRAILERS {
        if q.to_lowercase().ends_with(tail) {
            let keep = q.chars().count().saturating_sub(tail.chars().count());
            let cut: String = q.chars().take(keep).collect();
            q = cut.trim_matches(PUNCT).trim().to_string();
            break;
        }
    }
    let words: Vec<&str> =
        q.split_whitespace().filter(|w| !WEBSEARCH_STOPWORDS.contains(&w.to_lowercase().as_str())).collect();
    if words.len() >= 2 {
        q = words.join(" ");
    }
    if q.is_empty() { raw.trim().to_string() } else { q }
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

/// Resolves the rerank model when a search runs, so the choice follows the
/// request's selection and memory rather than whatever was resident when
/// the agent was built.  `None` falls back to the agent's fixed `llm`.
pub type LlmResolver = Arc<dyn Fn() -> Option<Arc<dyn BaseLLM>> + Send + Sync>;

/// Search the web via the hosted endpoint and return fetched content.
pub struct WebSearchAgent {
    fetch_agent: Mutex<Option<Arc<dyn UrlFetcher>>>,
    top_k: usize,
    num_candidates: usize,
    llm: Option<Arc<dyn BaseLLM>>,
    llm_resolver: Option<LlmResolver>,
    /// URLs of the sources kept after the last `search()` (top_k, reranked
    /// order).  Read by the orchestrator so the sources surface as links in
    /// the final answer rather than only as context the model may not echo.
    pub last_sources: RwLock<Vec<String>>,
    /// url → the page text handed to the model for that source.  Lets the
    /// router confirm the answer actually drew on a page before crediting it
    /// as a reference.  Empty when the fetches failed and only bare links
    /// were passed through.
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
    ///
    /// `top_k` links are returned; `num_candidates` is how many result links
    /// are requested and fetched so the search survives forbidden/JS-only
    /// pages (some links 401/403 or need JavaScript — they are skipped) and
    /// can then LLM-rerank the survivors by relevance.  Without an `llm` the
    /// endpoint's positional order is kept.
    pub fn new(
        fetch_agent: Option<Arc<dyn UrlFetcher>>,
        top_k: usize,
        num_candidates: usize,
        llm: Option<Arc<dyn BaseLLM>>,
    ) -> Self {
        Self {
            fetch_agent: Mutex::new(fetch_agent),
            top_k,
            num_candidates,
            llm,
            llm_resolver: None,
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

    /// Resolve the rerank model per search with `resolver` (see
    /// [`LlmResolver`]); the fixed `llm` stays the fallback.
    pub fn with_llm_resolver(mut self, resolver: LlmResolver) -> Self {
        self.llm_resolver = Some(resolver);
        self
    }

    /// The rerank LLM, if any.
    pub fn llm(&self) -> Option<&Arc<dyn BaseLLM>> {
        self.llm.as_ref()
    }

    /// The model this search reranks with: the resolver's pick, else the
    /// fixed `llm`.
    fn rerank_llm(&self) -> Option<Arc<dyn BaseLLM>> {
        self.llm_resolver.as_ref().and_then(|resolve| resolve()).or_else(|| self.llm.clone())
    }

    /// Search the web for `query` and return fetched result content.
    ///
    /// Returns `""` (never fails, never a notice) when web search is disabled
    /// for the project, forbidden (401/403 — no plan or no provider registered
    /// on the account), unauthenticated, or fails — so it silently skips and
    /// the model falls back to its own knowledge instead of surfacing an
    /// error/upsell in the answer.
    ///
    /// `progress(step, status, message)` — when supplied — reports each page
    /// as it is fetched.  This is the slowest part of a request (one network
    /// round-trip per result, serially), so without it the client sits on a
    /// stale status line for the whole search.  Passed in per call rather than
    /// held on the agent, which is shared across concurrent requests.  A
    /// failing sink must never break the search, so it is called defensively.
    pub fn search(
        &self,
        query: &str,
        top_k: Option<usize>,
        num_candidates: Option<usize>,
        progress: Option<&ProgressCallback>,
    ) -> String {
        if !Self::is_enabled() {
            tracing::info!(target: LOG_TARGET, "[web_search] disabled for this project — skipping");
            return String::new();
        }

        let want = top_k.filter(|k| *k != 0).unwrap_or(self.top_k);
        let n = num_candidates.filter(|k| *k != 0).unwrap_or(self.num_candidates);
        self.last_sources.write().clear();
        self.last_source_extracts.write().clear();

        // Search on the distilled subject, not the raw conversational message
        // ("give me a list latest canyon bikes" → "latest canyon bikes").
        let query = normalize_search_query(query);

        let links = match self.fetch_links(&query, n) {
            Ok(links) => links,
            Err(SearchProviderError::SearchAuthError(_)) => {
                tracing::info!(
                    target: LOG_TARGET,
                    "[web_search] provider rejected credentials (401/403) — skipping"
                );
                return String::new();
            }
            Err(SearchProviderError::NoSearchCredentials(_)) => {
                tracing::info!(target: LOG_TARGET, "[web_search] no credentials available — skipping");
                return String::new();
            }
            Err(e) => {
                tracing::warn!(target: LOG_TARGET, "[web_search] request failed, skipping: {}", e);
                return String::new();
            }
        };

        if links.is_empty() {
            tracing::info!(target: LOG_TARGET, "[web_search] no results for {} — skipping", py_repr(&query));
            return String::new();
        }

        self.fetch_top(&query, &links, want, n, progress)
    }

    // ---------------------------------------------------------------- endpoint

    /// Return result URLs for `query` from the hosted endpoint.
    ///
    /// Fails with `SearchAuthError` when the endpoint refuses the request (no
    /// plan or no registered provider) and `NoSearchCredentials` when there is
    /// no Smartloop token to call it with.
    pub fn fetch_links(&self, query: &str, count: usize) -> Result<Vec<String>, SearchProviderError> {
        let provider = providers::resolve_provider();
        tracing::info!(
            target: LOG_TARGET,
            "[web_search] provider={} for {}",
            py_repr(provider.name()),
            py_repr(query)
        );
        provider.search(query, count)
    }

    // ---------------------------------------------------------------- fetching

    /// Report a search step, never letting a bad sink break the search.
    pub fn emit(progress: Option<&ProgressCallback>, message: &str) {
        let Some(progress) = progress else { return };
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| progress("web_search", "processing", message)));
        if outcome.is_err() {
            tracing::debug!(target: LOG_TARGET, "[web_search] progress callback failed");
        }
    }

    /// Host of `url` for display (`urlparse(url).netloc`), falling back to the
    /// raw URL.
    pub fn source_label(url: &str) -> String {
        let rest = match SCHEME_RE.find(url) {
            Some(m) => &url[m.end()..],
            None => url,
        };
        let netloc = match rest.strip_prefix("//") {
            Some(after) => after.split(['/', '?', '#']).next().unwrap_or(""),
            None => "",
        };
        if netloc.is_empty() { url.to_string() } else { netloc.to_string() }
    }

    /// Fetch the links, then LLM-rerank and keep the `top_k` most relevant.
    ///
    /// Fetches up to `num_candidates` usable sources (skipping forbidden /
    /// JS-only pages), re-orders them by LLM-judged relevance to `query`, and
    /// returns the top `top_k`.  Falls back to the endpoint's positional order
    /// when no LLM is wired in or the rerank call fails.
    pub fn fetch_top(
        &self,
        query: &str,
        links: &[String],
        top_k: usize,
        num_candidates: usize,
        progress: Option<&ProgressCallback>,
    ) -> String {
        let mut fetched: Vec<(String, String)> = Vec::new(); // (url, content)
        let mut skipped: Vec<String> = Vec::new();
        let budget = num_candidates.max(top_k);
        let fetcher = self.fetch_agent();
        for url in links {
            if fetched.len() >= budget {
                break;
            }
            Self::emit(progress, &format!("Reading {}", Self::source_label(url)));
            let content = match std::panic::catch_unwind(AssertUnwindSafe(|| fetcher.fetch(url, 4000, 0, false))) {
                Ok(content) => content,
                Err(_) => {
                    tracing::info!(target: LOG_TARGET, "[web_search] fetch failed for {}: fetch panicked", url);
                    String::new()
                }
            };
            if Self::is_usable(&content) {
                fetched.push((url.clone(), content.trim().to_string()));
            } else {
                skipped.push(url.clone());
                tracing::info!(target: LOG_TARGET, "[web_search] skipping unusable/forbidden link: {}", url);
            }
        }

        if fetched.is_empty() {
            // Nothing fetched cleanly — fall back to the bare link list so the
            // LLM at least has the sources.
            let kept: Vec<String> = links.iter().take(top_k).cloned().collect();
            let listing: Vec<String> = kept.iter().map(|u| format!("- {}", u)).collect();
            *self.last_sources.write() = kept;
            return format!("Web search results for: {}\n\n{}", query, listing.join("\n"));
        }

        let fetched_count = fetched.len();
        let ranked = self.rerank(query, fetched);
        let kept: Vec<(String, String)> = ranked.into_iter().take(top_k).collect();
        *self.last_sources.write() = kept.iter().map(|(url, _)| url.clone()).collect();
        *self.last_source_extracts.write() = kept.iter().cloned().collect();
        let mut parts = vec![format!("Web search results for: {}", query)];
        for (i, (url, content)) in kept.iter().enumerate() {
            parts.push(format!("## [{}] {}\n\n{}", i + 1, url, content));
        }
        tracing::info!(
            target: LOG_TARGET,
            "[web_search] fetched {}/{} usable source(s), kept top {} ({} skipped) for {}",
            fetched_count,
            num_candidates,
            kept.len(),
            skipped.len(),
            py_repr(query)
        );
        Self::emit(
            progress,
            &format!("Read {} web source{}", kept.len(), if kept.len() != 1 { "s" } else { "" }),
        );
        parts.join("\n\n")
    }

    /// Re-order `candidates` `(url, content)` by LLM-judged relevance.
    ///
    /// Asks the model for a comma-separated ranking of the candidate indices
    /// and re-orders accordingly.  Falls back to the input (endpoint) order
    /// when no LLM is wired in or the call fails, so web search never degrades
    /// below the positional ranking.
    pub fn rerank(&self, query: &str, candidates: Vec<(String, String)>) -> Vec<(String, String)> {
        if candidates.len() <= 1 {
            return candidates;
        }
        // Resolved only once there is something to rank, so a single result
        // never loads a model.
        let Some(llm) = self.rerank_llm() else {
            return candidates;
        };
        // Judge relevance on a short snippet of each source — full content is
        // too much for the rerank model's window and adds no signal.
        let lines: Vec<String> = candidates
            .iter()
            .enumerate()
            .map(|(i, (url, content))| {
                let snippet: String = WHITESPACE_RE.replace_all(content, " ").chars().take(400).collect();
                format!("{}. {} — {}", i + 1, url, snippet)
            })
            .collect();
        let prompt = format!(
            "Rank the web results below by relevance to the user's query, most \
             relevant first. Respond with ONLY the numbers as a comma-separated \
             list, e.g. \"3,1,5\". No explanation.\n\nQuery: {}\n\n{}",
            query,
            lines.join("\n")
        );
        let order: Vec<usize> =
            match std::panic::catch_unwind(AssertUnwindSafe(|| llm.invoke(&prompt, Some(32), None))) {
                Ok(resp) => DIGITS_RE
                    .find_iter(resp.trim())
                    .filter_map(|m| m.as_str().parse::<usize>().ok())
                    .collect(),
                Err(_) => {
                    tracing::info!(
                        target: LOG_TARGET,
                        "[web_search] LLM rerank failed (invoke panicked) — using endpoint order"
                    );
                    return candidates;
                }
            };
        let mut seen: HashSet<String> = HashSet::new();
        let mut ranked: Vec<(String, String)> = Vec::new();
        for i in order {
            if (1..=candidates.len()).contains(&i) {
                let url = &candidates[i - 1].0;
                if seen.insert(url.clone()) {
                    ranked.push(candidates[i - 1].clone());
                }
            }
        }
        // Any candidates the model skipped stay at the end.
        for c in &candidates {
            if seen.insert(c.0.clone()) {
                ranked.push(c.clone());
            }
        }
        tracing::info!(
            target: LOG_TARGET,
            "[web_search] LLM reranked {} source(s) for {}",
            ranked.len(),
            py_repr(query)
        );
        ranked
    }

    /// True when fetched content is real page text (not empty / JS-only).
    pub fn is_usable(content: &str) -> bool {
        if content.trim().is_empty() {
            return false;
        }
        let low = content.to_lowercase();
        if low.contains("requires javascript") || low.contains("javascript-rendered pages cannot") {
            return false;
        }
        content.trim().chars().count() >= 200
    }

    // ------------------------------------------------------------------ config

    /// True when the orchestrator should run a web search.
    ///
    /// Two questions, in cost order. The project flag is the user's own
    /// choice, read off disk and free — it is what the Cmd/Ctrl+Alt+S shortcut
    /// toggles. Then the account's own state, from `GET /v1/websearch/status`:
    /// signed in, with a provider registered.
    ///
    /// Asking the endpoint up front is the point. Web search is otherwise
    /// discovered to be unavailable only after normalizing the query, calling
    /// the endpoint and throwing away a 403 — work done every turn to reach the
    /// same answer. The status answer is cached briefly, and fails open, so a
    /// health check that cannot complete never suppresses a search that would
    /// have worked.
    pub fn is_enabled() -> bool {
        let check = || -> bool {
            // Read the on-disk state first: agents can run in worker processes
            // whose cached singleton predates the toggle, so trusting the
            // in-memory copy would keep searching after the user disabled it.
            let ws = Workspace::load().or_else(|| Workspace::get_instance().map(|shared| shared.read().clone()));
            let enabled = ws
                .as_ref()
                .and_then(|ws| ws.current_project())
                .map(|project| project.web_search_enabled)
                .unwrap_or(false);
            if !enabled {
                return false;
            }
            providers::hosted_search_available(None)
        };
        match std::panic::catch_unwind(AssertUnwindSafe(check)) {
            Ok(enabled) => enabled,
            Err(payload) => {
                let message = payload
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown error".to_string());
                tracing::warn!(target: LOG_TARGET, "[web_search] enablement check failed: {}", message);
                false
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_label_is_the_netloc() {
        assert_eq!(WebSearchAgent::source_label("https://canyon.com/bikes?x=1"), "canyon.com");
        assert_eq!(WebSearchAgent::source_label("a.com"), "a.com");
        assert_eq!(WebSearchAgent::source_label("http://u@h:8080/p"), "u@h:8080");
    }

    #[test]
    fn usable_content_needs_real_text() {
        assert!(!WebSearchAgent::is_usable("   "));
        assert!(!WebSearchAgent::is_usable(&format!("{} requires JavaScript", "x".repeat(300))));
        assert!(!WebSearchAgent::is_usable("short"));
        assert!(WebSearchAgent::is_usable(&"x".repeat(200)));
    }
}
