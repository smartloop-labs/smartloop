//! Web search client for the WebSearchAgent.
//!
//! Port of `smartloop/search/providers.py`.
//!
//! Search runs against Smartloop's hosted `/v1/websearch` endpoint, which runs
//! the query server-side on Smartloop's own credentials.  Nothing is configured
//! on this side: it asks for links and treats a 401/403 (unavailable) or 429
//! (the month's allowance is spent) as "search is off".
//!
//! The HTTP itself belongs to `SLPPlatformClient` like every other platform
//! call; what lives here is the search-specific policy — when a failure means
//! "off" versus "unknown", and how long a status answer is trusted.

use std::sync::Arc;
use std::time::Instant;

use once_cell::sync::Lazy;
use parking_lot::Mutex;

#[allow(unused_imports)]
use crate::config::app_settings::AppSettings;
#[allow(unused_imports)]
use crate::platform::auth as platform_auth;
#[allow(unused_imports)]
use crate::platform::client::SLPPlatformClient;

/// How long a status answer is trusted, in seconds.
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

/// The two platform calls web search needs (`SLPPlatformClient.web_search`
/// and `.web_search_status`).  A seam so tests can stub the platform.
pub trait WebSearchClient: Send + Sync {
    /// `GET /v1/websearch`: up to `count` result URLs for `query`.
    fn web_search(&self, query: &str, count: usize) -> Result<Vec<String>, WebSearchCallError>;
    /// `GET /v1/websearch/status`: whether search would serve.
    fn web_search_status(&self) -> Result<bool, WebSearchCallError>;
}

/// Factory producing the platform client (`_platform_client`).
pub type PlatformClientFactory = Arc<dyn Fn() -> Box<dyn WebSearchClient> + Send + Sync>;

/// Credential check (`platform_auth.has_credentials`).
pub type CredentialsCheck = Arc<dyn Fn() -> bool + Send + Sync>;

static CLIENT_FACTORY: Lazy<Mutex<Option<PlatformClientFactory>>> = Lazy::new(|| Mutex::new(None));
static CREDENTIALS_CHECK: Lazy<Mutex<Option<CredentialsCheck>>> = Lazy::new(|| Mutex::new(None));

/// Replace (or with `None`, restore) the platform-client factory.
pub fn set_platform_client_factory(factory: Option<PlatformClientFactory>) {
    *CLIENT_FACTORY.lock() = factory;
}

/// Replace (or with `None`, restore) the credential check.
pub fn set_credentials_check(check: Option<CredentialsCheck>) {
    *CREDENTIALS_CHECK.lock() = check;
}

/// Return a platform client bound to the configured API base.
pub fn platform_client() -> Box<dyn WebSearchClient> {
    todo!()
}

/// Whether Smartloop credentials are available.
pub fn has_credentials() -> bool {
    todo!()
}

/// Web-search failures (`SearchProviderError` and its subclasses).
#[derive(Debug, thiserror::Error)]
pub enum SearchProviderError {
    /// The endpoint refused the request (401/403): web search is unavailable
    /// for this account.  The agent silently skips searching.
    #[error("{0}")]
    SearchAuthError(String),
    /// No Smartloop credentials available to call the endpoint.
    #[error("{0}")]
    NoSearchCredentials(String),
    /// Any other failure from the platform call.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
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

    /// Up to `count` (clamped to 1..=20) result URLs for `query`.
    pub fn search(&self, query: &str, count: usize) -> Result<Vec<String>, SearchProviderError> {
        let _ = (query, count);
        todo!()
    }
}

/// Return the search client.  Always the hosted endpoint.
pub fn resolve_provider() -> HostedSearchProvider {
    HostedSearchProvider::new()
}

/// Whether the hosted endpoint would actually serve a search, cached for
/// `ttl` seconds ([`STATUS_TTL_SECONDS`] by default).  Fails open: only an
/// explicit 401/403 counts as "no"; signed out is "no" without a request.
pub fn hosted_search_available(ttl: Option<f64>) -> bool {
    let _ = (ttl, &STATUS_CACHE);
    todo!()
}

/// Forget the cached status — used by tests and after re-authenticating.
pub fn reset_status_cache() {
    *STATUS_CACHE.lock() = None;
}
