//! Port of `smartloop/platform/token_agent.py`.
//!
//! Keeps the access token fresh.
//!
//! Two entry points over the same operation: a background loop started from
//! the API server lifespan, and [`refresh_access_token_now`] for callers that
//! have just been told their access token is dead and want it back immediately.
//!
//! Both run in the workspace owner and nowhere else.  Under supervision this
//! process is one of N+1 sharing a single credential file, and the refresh
//! token is single-use: redeeming it twice gets the loser a 401, which reads as
//! "the refresh token is dead" and clears both credentials.  The guard is
//! ownership rather than locking; see [`refresh_access_token_now`].

use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::Instant;

use futures::future::BoxFuture;
use once_cell::sync::Lazy;
use parking_lot::Mutex;

#[allow(unused_imports)]
use crate::auth::credential_store;
#[allow(unused_imports)]
use crate::config::AppSettings;
use crate::platform::client::PlatformApi;
#[allow(unused_imports)]
use crate::platform::client::SLPPlatformClient;
#[allow(unused_imports)]
use crate::workspace::context::{owns_workspace, pinned_project_id};

/// Python logger name for this module.
pub const LOGGER: &str = "smartloop.platform.token_agent";

/// 45 minutes.
pub const REFRESH_INTERVAL_SECONDS: u64 = 45 * 60;

/// How long a failed redemption suppresses the next attempt.  The lock alone
/// only collapses callers when the refresh *succeeds*; one attempt per window
/// is enough to discover the platform is still down.
pub const FAILED_REFRESH_COOLDOWN_SECONDS: f64 = 30.0;

/// Serializes redemption and records when the last one failed (`None` = no
/// recent failure; Python `_last_failure_at = 0.0`).  Two callers reach this —
/// the background loop and a request thread whose call just came back 401.
static REFRESH_LOCK: Lazy<Mutex<Option<Instant>>> = Lazy::new(|| Mutex::new(None));

/// Forget the last failed redemption so the cooldown no longer applies
/// (tests: `token_agent._last_failure_at = 0.0`).
pub fn reset_failure_cooldown() {
    *REFRESH_LOCK.lock() = None;
}

/// The `exp` claim of a JWT, or `None` if it has none to read.
///
/// Claims only — the signature is the server's to verify.
pub fn jwt_exp(token: Option<&str>) -> Option<f64> {
    let _ = token;
    todo!()
}

/// True when the stored access token will not survive until the next tick.
pub fn needs_refresh() -> bool {
    todo!()
}

/// Redeem the stored refresh token for a new access token.
///
/// Returns true when a new access token is in place.  Concurrent callers
/// collapse onto one redemption.  A process pinned to a project (not the
/// workspace owner) declines immediately, before touching the credential
/// store.  *client* defaults to a fresh `SLPPlatformClient` for
/// `AppSettings::slp_api_base`.
pub fn refresh_access_token_now(client: Option<&dyn PlatformApi>) -> bool {
    let _ = client;
    todo!()
}

/// Do the actual redemption.  Callers hold [`REFRESH_LOCK`].
///
/// A `PermissionError` from the refresh endpoint clears both tokens; any other
/// failure leaves credentials alone to be retried later.
fn redeem(client: Option<&dyn PlatformApi>) -> bool {
    let _ = client;
    todo!()
}

/// Injectable seams for [`run_token_refresh_loop_with`] (the Python tests patch
/// `_needs_refresh`, `refresh_access_token_now`, `asyncio.sleep` and
/// `SLPPlatformClient` on the module).
#[derive(Clone)]
pub struct TokenRefreshLoopHooks {
    /// Whether a refresh is due; an `Err` is a failed cycle (logged, loop continues).
    pub needs_refresh: Arc<dyn Fn() -> anyhow::Result<bool> + Send + Sync>,
    /// Performs the refresh (run on the blocking pool).
    pub refresh: Arc<dyn Fn(Arc<dyn PlatformApi>) -> bool + Send + Sync>,
    /// Sleeps between cycles; `Break` stops the loop.
    pub sleep: Arc<dyn Fn(u64) -> BoxFuture<'static, ControlFlow<()>> + Send + Sync>,
    /// Builds the client used for every cycle.
    pub make_client: Arc<dyn Fn() -> Arc<dyn PlatformApi> + Send + Sync>,
}

impl Default for TokenRefreshLoopHooks {
    /// The production hooks: [`needs_refresh`], [`refresh_access_token_now`],
    /// `tokio::time::sleep`, and an `SLPPlatformClient` for the configured base.
    fn default() -> Self {
        todo!()
    }
}

/// Renew the access token before it expires, starting immediately (the sleep
/// is at the end of each cycle).  Never returns.
pub async fn run_token_refresh_loop() {
    run_token_refresh_loop_with(TokenRefreshLoopHooks::default()).await
}

/// [`run_token_refresh_loop`] with injectable seams; returns only when the
/// sleep hook yields `ControlFlow::Break`.
pub async fn run_token_refresh_loop_with(hooks: TokenRefreshLoopHooks) {
    let _ = hooks;
    todo!()
}
