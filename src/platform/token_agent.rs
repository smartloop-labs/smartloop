//! Port of `smartloop/platform/token_agent.py`.
//!
//! Keeps the access token fresh.
//!
//! Two entry points over the same operation: a background loop started from
//! the API server lifespan, and [`refresh_access_token_now`] for callers that
//! have just been told their access token is dead and want it back immediately.
//!
//! Both run in the workspace owner and nowhere else.  Under supervision this
//! process is one of N+1 sharing a single credential file — a supervisor plus
//! an agent child per project — and the refresh token is single-use: redeeming
//! it twice gets the loser a 401, which reads as "the refresh token is dead"
//! and clears both credentials.  Nothing here can see another process, so the
//! guard is ownership rather than locking; see [`refresh_access_token_now`].

use std::ops::ControlFlow;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::{Duration, Instant};

use base64::Engine as _;
use futures::future::BoxFuture;
use futures::FutureExt as _;
use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use serde_json::Value;

use crate::auth::credential_store;
use crate::config::AppSettings;
use crate::platform::client::PlatformApi;
use crate::platform::client::SLPPlatformClient;
use crate::workspace::context::{owns_workspace, pinned_project_id};

/// Python logger name for this module.
pub const LOGGER: &str = "smartloop.platform.token_agent";

/// 45 minutes.
pub const REFRESH_INTERVAL_SECONDS: u64 = 45 * 60;

/// How long a failed redemption suppresses the next attempt.
///
/// The lock alone only collapses callers when the refresh *succeeds* — on
/// failure the stored token is unchanged, so every queued waiter falls through
/// and pays the full timeout in turn.  With the auth endpoints served from the
/// blocking pool, ten concurrent 401s against a degraded platform meant ten
/// serialized redemptions holding ten pool slots.  One attempt per window is
/// enough to discover the platform is still down.
pub const FAILED_REFRESH_COOLDOWN_SECONDS: f64 = 30.0;

/// Serializes redemption and records when the last one failed (`None` = no
/// recent failure; Python `_last_failure_at = 0.0`).
///
/// Two callers reach this — the background loop, and a request thread whose
/// call just came back 401 — and with single-use rotating refresh tokens the
/// second redemption of the same token is refused, which clears both
/// credentials and logs the user out.
///
/// Two, and only two, because a pinned process declines before reaching here
/// ([`refresh_access_token_now`]).  Without that gate there would be one more
/// per agent child, on a lock that does not span processes and so would not
/// order them.
static REFRESH_LOCK: Lazy<Mutex<Option<Instant>>> = Lazy::new(|| Mutex::new(None));

/// Forget the last failed redemption so the cooldown no longer applies
/// (tests: `token_agent._last_failure_at = 0.0`).
pub fn reset_failure_cooldown() {
    *REFRESH_LOCK.lock() = None;
}

/// Observer of this module's credential-store reads (`"access_token"` /
/// `"refresh_token"`).  A test seam standing in for monkeypatching
/// `credential_store.get_access_token`: it lets a test prove a pinned process
/// declines before touching the credential file at all.
pub type CredentialReadObserver = Arc<dyn Fn(&str) + Send + Sync>;

static CREDENTIAL_READ_OBSERVER: Lazy<RwLock<Option<CredentialReadObserver>>> =
    Lazy::new(|| RwLock::new(None));

/// Install (or with `None`, remove) the [`CredentialReadObserver`].
pub fn set_credential_read_observer(observer: Option<CredentialReadObserver>) {
    *CREDENTIAL_READ_OBSERVER.write() = observer;
}

fn notify_read(what: &str) {
    let observer = CREDENTIAL_READ_OBSERVER.read().clone();
    if let Some(observer) = observer {
        observer(what);
    }
}

/// The stored access token (`None` when absent or empty).
fn read_access_token() -> Option<String> {
    notify_read("access_token");
    credential_store::get_access_token().filter(|t| !t.is_empty())
}

/// The stored refresh token (`None` when absent or empty).
fn read_refresh_token() -> Option<String> {
    notify_read("refresh_token");
    credential_store::get_refresh_token().filter(|t| !t.is_empty())
}

/// Python `float(x)` for a JSON claim value.
fn claim_as_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

/// Decode a JWT segment the way Python's `base64.urlsafe_b64decode` does after
/// the caller pads it: characters outside the alphabet are discarded, and a
/// length that cannot be valid base64 is an error.
fn urlsafe_b64decode_lenient(segment: &str) -> Option<Vec<u8>> {
    let cleaned: String = segment
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    if cleaned.len() % 4 == 1 {
        return None;
    }
    base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(cleaned.as_bytes()).ok()
}

/// The `exp` claim of a JWT, or `None` if it has none to read.
///
/// Claims only — the signature is the server's to verify.  We are deciding
/// whether to bother asking for a new token, not whether to trust this one.
pub fn jwt_exp(token: Option<&str>) -> Option<f64> {
    let token = token.filter(|t| !t.is_empty())?;
    if token.matches('.').count() != 2 {
        return None;
    }
    let payload = token.split('.').nth(1)?;
    let decoded = urlsafe_b64decode_lenient(payload)?;
    let claims: Value = serde_json::from_slice(&decoded).ok()?;
    let exp = claims.as_object()?.get("exp")?;
    if exp.is_null() {
        return None;
    }
    claim_as_f64(exp)
}

/// True when the stored access token will not survive until the next tick.
///
/// The loop wakes on a fixed interval, so anything expiring inside that
/// interval has to be renewed now — there is no later chance before it dies.
pub fn needs_refresh() -> bool {
    if read_refresh_token().is_none() {
        return false;
    }
    let Some(access) = read_access_token() else {
        // Nothing to use.  This is the state a 401 used to leave behind, and
        // it is exactly when a refresh is most needed.
        return true;
    };
    match jwt_exp(Some(&access)) {
        // Opaque token — no expiry to read, so renew on the interval as before.
        None => true,
        Some(exp) => exp - crate::platform::service::now() <= REFRESH_INTERVAL_SECONDS as f64,
    }
}

/// Redeem the stored refresh token for a new access token.
///
/// Returns true when a new access token is in place.  Concurrent callers
/// collapse onto one redemption: whoever holds the lock does the work, and the
/// rest find the token already replaced and report success without spending
/// the refresh token a second time.
///
/// Workspace-owner work.  Everything here that keeps two *threads* from
/// double-redeeming is process-local, so it says nothing about two
/// *processes* — and under supervision there are N+1 of them against one
/// credential file.  A child racing the supervisor's background loop spends
/// the rotated token twice, and the refusal that comes back is read by
/// [`redeem`] as a dead refresh token, clearing both credentials: a full
/// logout of every project, which is the failure this function exists to
/// prevent.
///
/// So a pinned process declines, and it costs it nothing.  The retry in
/// [`SLPPlatformClient::authed`] is keyed on the credentials having *changed*,
/// not on what this returned, so declining leaves the caller with the 401 it
/// already had.  And because the credential store caches nothing, a child
/// picks up a token the owner rotated on its very next read, with no
/// redemption and no message passing.
///
/// This is a statement about process topology, not a safety property: it
/// holds because only the owner writes, not because writing is safe.
/// *client* defaults to a fresh `SLPPlatformClient` for
/// `AppSettings::slp_api_base`.
pub fn refresh_access_token_now(client: Option<&dyn PlatformApi>) -> bool {
    if !owns_workspace() {
        // At info, like the matching decline in `auth::resolve_user`.  This is
        // the guard that keeps one project's 401 from clearing the whole
        // workspace's credentials, so "did it fire?" is a question worth being
        // able to answer from a log a user sent in.
        tracing::info!(
            target: "smartloop.platform.token_agent",
            "[token] refresh declined in project agent {} — redemption belongs to the workspace owner",
            pinned_project_id().unwrap_or_else(|| "None".to_string())
        );
        return false;
    }

    let before = read_access_token();
    let mut last_failure = REFRESH_LOCK.lock();
    let current = read_access_token();
    if current.is_some() && current != before {
        // Someone refreshed while we waited for the lock.  Their token is the
        // live one; redeeming again would burn a rotated refresh token for a
        // result we already have.
        return true;
    }

    if let Some(failed_at) = *last_failure {
        if failed_at.elapsed() < Duration::from_secs_f64(FAILED_REFRESH_COOLDOWN_SECONDS) {
            // A redemption failed moments ago.  Whatever stopped it — the
            // platform is down, the refresh token is gone — will not have been
            // fixed by the time it took us to reach the lock.
            return false;
        }
    }

    let ok = redeem(client);
    *last_failure = if ok { None } else { Some(Instant::now()) };
    ok
}

/// Do the actual redemption.  Callers hold [`REFRESH_LOCK`].
///
/// A permission error from the refresh endpoint means the refresh token
/// itself is dead — retrying cannot help, so both tokens are cleared and the
/// user has to log in again.  Any other failure leaves the credentials alone
/// to be retried later; `raise_for_status` guarantees a 429 or 5xx arrives as
/// a `ConnectionError` rather than as the `PermissionError` that would clear
/// them.
fn redeem(client: Option<&dyn PlatformApi>) -> bool {
    let Some(refresh_token) = read_refresh_token() else {
        return false;
    };
    let owned;
    let client: &dyn PlatformApi = match client {
        Some(client) => client,
        None => {
            owned = SLPPlatformClient::new(&AppSettings::new().slp_api_base);
            &owned
        }
    };
    let data = match client.refresh_access_token(&refresh_token) {
        Ok(data) => data,
        Err(err) if err.is_permission() => {
            credential_store::clear_tokens();
            tracing::info!(
                target: "smartloop.platform.token_agent",
                "Cleared tokens after 401 from refresh endpoint — re-login required"
            );
            return false;
        }
        Err(err) => {
            tracing::warn!(
                target: "smartloop.platform.token_agent",
                "Token refresh failed — will retry next cycle: {}",
                err
            );
            return false;
        }
    };

    let access = match data.get("access_token") {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        _ => {
            tracing::warn!(target: "smartloop.platform.token_agent", "Refresh endpoint returned no access token");
            return false;
        }
    };
    let rotated = match data.get("refresh_token") {
        Some(Value::String(s)) => Some(s.as_str()),
        _ => None,
    };
    credential_store::set_tokens(&access, rotated);
    tracing::info!(target: "smartloop.platform.token_agent", "Tokens refreshed via /v1/auth/jwt/refresh");
    true
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
        Self {
            needs_refresh: Arc::new(|| Ok(needs_refresh())),
            refresh: Arc::new(|client: Arc<dyn PlatformApi>| refresh_access_token_now(Some(&*client))),
            sleep: Arc::new(|seconds| {
                async move {
                    tokio::time::sleep(Duration::from_secs(seconds)).await;
                    ControlFlow::Continue(())
                }
                .boxed()
            }),
            make_client: Arc::new(|| {
                let settings = AppSettings::new();
                Arc::new(SLPPlatformClient::new(&settings.slp_api_base)) as Arc<dyn PlatformApi>
            }),
        }
    }
}

/// Renew the access token before it expires, starting immediately.
///
/// The sleep is at the *end*: sleeping first meant nothing refreshed for the
/// first interval after start, so an access token that expired in that window
/// was never renewed — and the 401 it caused deleted it.  Checking on entry
/// also repairs that state on the next restart.
///
/// Skips silently when no refresh token is present (user hasn't logged in).
/// Never returns.
pub async fn run_token_refresh_loop() {
    run_token_refresh_loop_with(TokenRefreshLoopHooks::default()).await
}

/// One cycle's failure, for the log line.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic".to_string()
    }
}

/// [`run_token_refresh_loop`] with injectable seams; returns only when the
/// sleep hook yields `ControlFlow::Break`.
pub async fn run_token_refresh_loop_with(hooks: TokenRefreshLoopHooks) {
    let client = (hooks.make_client)();

    loop {
        // A bad cycle must not kill the loop: errors and panics alike are
        // logged and the next cycle runs on schedule.
        let due = std::panic::catch_unwind(AssertUnwindSafe(|| (hooks.needs_refresh)()));
        match due {
            Ok(Ok(true)) => {
                // Off the event loop.  Redemption is a synchronous HTTP POST
                // with a 10s budget, taken behind a lock a request thread may
                // already hold.  Awaiting it inline stalls this loop for as
                // long as that takes — and under supervision this loop belongs
                // to the supervisor, which is proxying every project's chat
                // stream through it byte by byte.  One expired token would
                // pause generation for every project at once.
                let refresh = hooks.refresh.clone();
                let client = client.clone();
                if let Err(e) = tokio::task::spawn_blocking(move || refresh(client)).await {
                    let detail = if e.is_panic() {
                        panic_message(&*e.into_panic())
                    } else {
                        e.to_string()
                    };
                    tracing::warn!(target: "smartloop.platform.token_agent", "Token refresh cycle failed: {}", detail);
                }
            }
            Ok(Ok(false)) => {}
            Ok(Err(e)) => {
                tracing::warn!(target: "smartloop.platform.token_agent", "Token refresh cycle failed: {:#}", e);
            }
            Err(payload) => {
                tracing::warn!(
                    target: "smartloop.platform.token_agent",
                    "Token refresh cycle failed: {}",
                    panic_message(&*payload)
                );
            }
        }
        if (hooks.sleep)(REFRESH_INTERVAL_SECONDS).await.is_break() {
            return;
        }
    }
}
