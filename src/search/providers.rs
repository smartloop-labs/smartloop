//! Web search client for the WebSearchAgent.
//!
//! Port of `smartloop/search/providers.py`.
//!
//! Search runs against Smartloop's hosted `/v1/websearch` endpoint, which runs
//! the query server-side on Smartloop's own credentials.  Providers used to be
//! configured here — a Brave API key in the local credential store — but that
//! tied web search to one machine and left the hosted API unable to tell
//! whether search was usable.  Nothing is configured on this side any more: it
//! asks for links and treats a 401/403 (unavailable) or 429 (the month's
//! allowance is spent) as "search is off".
//!
//! The HTTP itself belongs to `SLPPlatformClient` like every other platform
//! call; what lives here is the search-specific policy — when a failure means
//! "off" versus "unknown", and how long a status answer is trusted.

use std::sync::Arc;
use std::time::{Duration, Instant};

use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::config::app_settings::AppSettings;
use crate::platform::auth as platform_auth;
use crate::platform::client::{PlatformError, SEARCH_STATUS_TIMEOUT, SEARCH_TIMEOUT, SLPPlatformClient};

/// Logger name (`logging.getLogger(__name__)`).
const LOG_TARGET: &str = "smartloop.search.providers";

/// How long a status answer is trusted, in seconds.  The orchestrator asks
/// once per query, so an uncached check would put a network round-trip in
/// front of every turn; a provider being connected or removed is not something
/// that needs to be noticed within the minute.
pub const STATUS_TTL_SECONDS: f64 = 60.0;

/// `(checked_at, enabled)` of the last status answer.
static STATUS_CACHE: Lazy<Mutex<Option<(Instant, bool)>>> = Lazy::new(|| Mutex::new(None));

/// How a platform call failed, as `SLPPlatformClient` reports it.
#[derive(Debug, thiserror::Error)]
pub enum WebSearchCallError {
    /// 401/403/429 — Python's `PermissionError`: "stop searching".
    #[error("{0}")]
    Permission(String),
    /// Anything else (network, 5xx) — may be transient.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<PlatformError> for WebSearchCallError {
    fn from(e: PlatformError) -> Self {
        if e.is_permission() { WebSearchCallError::Permission(e.message()) } else { WebSearchCallError::Other(e.into()) }
    }
}

/// The two platform calls web search needs (`SLPPlatformClient.web_search`
/// and `.web_search_status`).  A seam so tests can stub the platform.
pub trait WebSearchClient: Send + Sync {
    /// `GET /v1/websearch`: up to `count` result URLs for `query`.
    fn web_search(&self, query: &str, count: usize) -> Result<Vec<String>, WebSearchCallError>;
    /// `GET /v1/websearch/status`: whether search would serve.
    fn web_search_status(&self) -> Result<bool, WebSearchCallError>;
}

impl WebSearchClient for SLPPlatformClient {
    fn web_search(&self, query: &str, count: usize) -> Result<Vec<String>, WebSearchCallError> {
        let count = u32::try_from(count).unwrap_or(u32::MAX);
        Ok(SLPPlatformClient::web_search(self, query, count, SEARCH_TIMEOUT)?)
    }

    fn web_search_status(&self) -> Result<bool, WebSearchCallError> {
        Ok(SLPPlatformClient::web_search_status(self, SEARCH_STATUS_TIMEOUT)?)
    }
}

/// Factory producing the platform client (`_platform_client`).
pub type PlatformClientFactory = Arc<dyn Fn() -> Box<dyn WebSearchClient> + Send + Sync>;

/// Credential check (`platform_auth.has_credentials`).
pub type CredentialsCheck = Arc<dyn Fn() -> bool + Send + Sync>;

/// Stand-in for the whole of [`hosted_search_available`].
pub type AvailabilityCheck = Arc<dyn Fn() -> bool + Send + Sync>;

static CLIENT_FACTORY: Lazy<Mutex<Option<PlatformClientFactory>>> = Lazy::new(|| Mutex::new(None));
static CREDENTIALS_CHECK: Lazy<Mutex<Option<CredentialsCheck>>> = Lazy::new(|| Mutex::new(None));
static AVAILABILITY_OVERRIDE: Lazy<Mutex<Option<AvailabilityCheck>>> = Lazy::new(|| Mutex::new(None));

/// Replace (or with `None`, restore) the platform-client factory.
pub fn set_platform_client_factory(factory: Option<PlatformClientFactory>) {
    *CLIENT_FACTORY.lock() = factory;
}

/// Replace (or with `None`, restore) the credential check.
pub fn set_credentials_check(check: Option<CredentialsCheck>) {
    *CREDENTIALS_CHECK.lock() = check;
}

/// Replace (or with `None`, restore) [`hosted_search_available`] wholesale —
/// the Python tests' `monkeypatch.setattr(providers, "hosted_search_available", ...)`.
pub fn set_hosted_search_available_override(check: Option<AvailabilityCheck>) {
    *AVAILABILITY_OVERRIDE.lock() = check;
}

/// Return a platform client bound to the configured API base.
pub fn platform_client() -> Box<dyn WebSearchClient> {
    let factory = CLIENT_FACTORY.lock().clone();
    match factory {
        Some(factory) => factory(),
        None => Box::new(SLPPlatformClient::new(&AppSettings::new().slp_api_base)),
    }
}

/// Whether Smartloop credentials are available.
pub fn has_credentials() -> bool {
    let check = CREDENTIALS_CHECK.lock().clone();
    match check {
        Some(check) => check(),
        None => platform_auth::has_credentials(),
    }
}

/// Web-search failures (`SearchProviderError` and its subclasses).
#[derive(Debug, thiserror::Error)]
pub enum SearchProviderError {
    /// The endpoint refused the request (401/403): web search is unavailable
    /// for this account — search is not configured on the server, or the
    /// month's allowance is spent.  Either way the agent silently skips
    /// searching.
    #[error("{0}")]
    SearchAuthError(String),
    /// No Smartloop credentials available to call the endpoint.
    #[error("{0}")]
    NoSearchCredentials(String),
    /// Any other failure from the platform call (Python let the client's
    /// `ConnectionError` propagate unchanged).
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Python's `repr()` of a string, for log lines that used `%r`.
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

/// Smartloop's hosted `/v1/websearch` endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct HostedSearchProvider;

impl HostedSearchProvider {
    /// Provider name.
    pub const NAME: &'static str = "hosted";

    /// Construct the provider.
    pub fn new() -> Self {
        Self
    }

    /// Provider name.
    pub fn name(&self) -> &'static str {
        Self::NAME
    }

    /// Return up to `count` (clamped to 1..=20) result URLs for `query`.
    pub fn search(&self, query: &str, count: usize) -> Result<Vec<String>, SearchProviderError> {
        if !has_credentials() {
            return Err(SearchProviderError::NoSearchCredentials(
                "no Smartloop credentials available to call the hosted endpoint".to_string(),
            ));
        }

        let links = match platform_client().web_search(query, count.clamp(1, 20)) {
            Ok(links) => links,
            Err(WebSearchCallError::Permission(msg)) => return Err(SearchProviderError::SearchAuthError(msg)),
            Err(WebSearchCallError::Other(e)) => return Err(SearchProviderError::Other(e)),
        };

        tracing::info!(
            target: LOG_TARGET,
            "[search] hosted endpoint returned {} link(s) for {}",
            links.len(),
            py_repr(query)
        );
        Ok(links)
    }
}

/// Return the search client.  Always the hosted endpoint.
pub fn resolve_provider() -> HostedSearchProvider {
    HostedSearchProvider::new()
}

/// Whether the hosted endpoint would actually serve a search.
///
/// Asks `GET /v1/websearch/status`, which answers for the whole account:
/// signed in, with searches left this month.  Checking it up front means the
/// orchestrator skips the search step outright rather than running the query
/// normalizer, calling the endpoint and discarding a 403.  The answer is cached
/// for `ttl` seconds ([`STATUS_TTL_SECONDS`] when `None`).
///
/// Fails open.  A status check that cannot complete says nothing about whether
/// search works, and the search itself already degrades safely — it returns
/// `""` on a 403.  Refusing to search because a health check timed out would
/// be the worse failure.  Only an explicit 401/403 (`PermissionError`) is taken
/// as a real "no".
pub fn hosted_search_available(ttl: Option<f64>) -> bool {
    let override_check = AVAILABILITY_OVERRIDE.lock().clone();
    if let Some(check) = override_check {
        return check();
    }

    let ttl = Duration::from_secs_f64(ttl.unwrap_or(STATUS_TTL_SECONDS).max(0.0));
    let now = Instant::now();
    {
        let cache = STATUS_CACHE.lock();
        if let Some((checked_at, enabled)) = *cache {
            if now.saturating_duration_since(checked_at) < ttl {
                return enabled;
            }
        }
    }

    if !has_credentials() {
        *STATUS_CACHE.lock() = Some((now, false));
        return false;
    }

    let enabled = match platform_client().web_search_status() {
        Ok(enabled) => enabled,
        Err(WebSearchCallError::Permission(_)) => false,
        Err(WebSearchCallError::Other(e)) => {
            tracing::info!(target: LOG_TARGET, "[search] status check failed ({e}) — assuming available");
            return true;
        }
    };

    *STATUS_CACHE.lock() = Some((now, enabled));
    if !enabled {
        tracing::info!(target: LOG_TARGET, "[search] hosted endpoint reports web search unavailable");
    }
    enabled
}

/// Forget the cached status — used by tests and after re-authenticating.
pub fn reset_status_cache() {
    *STATUS_CACHE.lock() = None;
}

#[cfg(test)]
mod tests {
    use super::py_repr;

    #[test]
    fn repr_matches_python() {
        assert_eq!(py_repr("bikes"), "'bikes'");
        assert_eq!(py_repr("it's"), "\"it's\"");
        assert_eq!(py_repr("a'b\"c"), "'a\\'b\"c'");
    }
}
