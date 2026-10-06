//! Port of `tests/unit/test_token_refresh.py`.
//!
//! The access token is renewed before it dies, and on demand when it already
//! has.
//!
//! A 401 used to delete the access token and leave the refresh token
//! untouched, with nothing to redeem it until the background loop's next tick
//! — up to 45 minutes of being logged out while a valid refresh token sat on
//! disk.  The loop also slept *first*, so nothing refreshed at all during the
//! first interval after start.

use std::collections::VecDeque;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use futures::FutureExt as _;
use parking_lot::Mutex;
use serde_json::json;
use serial_test::serial;
use smartloop::auth::credential_store;
use smartloop::platform::client::{JsonObject, PlatformApi, PlatformError, PlatformResult};
use smartloop::platform::token_agent::{
    self, jwt_exp, needs_refresh, refresh_access_token_now, TokenRefreshLoopHooks, REFRESH_INTERVAL_SECONDS,
};

use crate::shared::platform_support::{autouse, obj, Conftest};

fn time_now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

/// A JWT-shaped token whose exp is *offset* seconds from now.
fn jwt(exp_offset_seconds: i64) -> String {
    let claims = json!({"exp": (time_now() as i64) + exp_offset_seconds}).to_string();
    let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(claims.as_bytes());
    format!("header.{payload}.signature")
}

fn clear_all() {
    credential_store::clear_access_token();
    credential_store::clear_refresh_token();
    credential_store::clear_token();
}

/// The conftest fixtures plus this module's `_clean_credentials`.
struct Clean {
    _conftest: Conftest,
}

impl Drop for Clean {
    fn drop(&mut self) {
        clear_all();
    }
}

fn clean_credentials() -> Clean {
    let conftest = autouse();
    clear_all();
    Clean { _conftest: conftest }
}

/// `MagicMock()` client whose `refresh_access_token` is scripted: each call
/// takes the next answer (`side_effect` list), the last one repeating.
struct MockClient {
    answers: Mutex<VecDeque<PlatformResult<JsonObject>>>,
    calls: Mutex<Vec<String>>,
    hook: Option<Box<dyn Fn(&str) -> PlatformResult<JsonObject> + Send + Sync>>,
}

impl MockClient {
    fn returning(answer: PlatformResult<JsonObject>) -> Self {
        Self::sequence(vec![answer])
    }

    fn sequence(answers: Vec<PlatformResult<JsonObject>>) -> Self {
        MockClient { answers: Mutex::new(answers.into()), calls: Mutex::new(Vec::new()), hook: None }
    }

    fn with_hook(hook: impl Fn(&str) -> PlatformResult<JsonObject> + Send + Sync + 'static) -> Self {
        MockClient { answers: Mutex::new(VecDeque::new()), calls: Mutex::new(Vec::new()), hook: Some(Box::new(hook)) }
    }

    fn call_count(&self) -> usize {
        self.calls.lock().len()
    }
}

impl PlatformApi for MockClient {
    fn refresh_access_token(&self, refresh_token: &str) -> PlatformResult<JsonObject> {
        self.calls.lock().push(refresh_token.to_string());
        if let Some(hook) = &self.hook {
            return hook(refresh_token);
        }
        let mut answers = self.answers.lock();
        if answers.len() > 1 {
            answers.pop_front().unwrap()
        } else {
            answers.front().cloned().unwrap_or_else(|| Ok(JsonObject::new()))
        }
    }
}

mod test_reading_the_expiry {
    use super::*;

    #[test]
    fn test_a_jwt_exp_is_read() {
        let exp = jwt_exp(Some(&jwt(600))).unwrap();
        assert!((exp - (time_now() + 600.0)).abs() <= 5.0);
    }

    #[test]
    fn test_anything_that_is_not_a_jwt_reads_as_unknown() {
        for token in [Some(""), Some("opaque-token"), Some("a.b"), Some("a.b.c.d"), None] {
            assert_eq!(jwt_exp(token), None, "{token:?}");
        }
    }

    #[test]
    fn test_undecodable_claims_read_as_unknown() {
        assert_eq!(jwt_exp(Some("header.!!!not-base64!!!.sig")), None);
    }
}

mod test_deciding_whether_to_refresh {
    use super::*;

    #[test]
    #[serial]
    fn test_no_refresh_token_means_nothing_to_do() {
        let _c = clean_credentials();
        credential_store::set_access_token(&jwt(-60));
        assert!(!needs_refresh());
    }

    /// The state a 401 used to leave behind — and never recover from.
    #[test]
    #[serial]
    fn test_a_missing_access_token_is_refreshed() {
        let _c = clean_credentials();
        credential_store::set_refresh_token("rt");
        assert!(needs_refresh());
    }

    #[test]
    #[serial]
    fn test_a_token_expiring_before_the_next_tick_is_refreshed() {
        let _c = clean_credentials();
        credential_store::set_refresh_token("rt");
        credential_store::set_access_token(&jwt(REFRESH_INTERVAL_SECONDS as i64 - 60));
        assert!(needs_refresh());
    }

    #[test]
    #[serial]
    fn test_an_already_expired_token_is_refreshed() {
        let _c = clean_credentials();
        credential_store::set_refresh_token("rt");
        credential_store::set_access_token(&jwt(-60));
        assert!(needs_refresh());
    }

    #[test]
    #[serial]
    fn test_a_token_that_outlives_the_next_tick_is_left_alone() {
        let _c = clean_credentials();
        credential_store::set_refresh_token("rt");
        credential_store::set_access_token(&jwt(REFRESH_INTERVAL_SECONDS as i64 + 600));
        assert!(!needs_refresh());
    }

    /// No expiry to read — renew on schedule, as it always did.
    #[test]
    #[serial]
    fn test_an_opaque_token_is_refreshed_on_the_interval() {
        let _c = clean_credentials();
        credential_store::set_refresh_token("rt");
        credential_store::set_access_token("opaque");
        assert!(needs_refresh());
    }
}

mod test_redeeming_the_refresh_token {
    use super::*;

    #[test]
    #[serial]
    fn test_a_new_access_token_is_stored() {
        let _c = clean_credentials();
        credential_store::set_refresh_token("old-rt");
        let client = MockClient::returning(Ok(obj(json!({"access_token": "fresh", "refresh_token": "new-rt"}))));

        assert!(refresh_access_token_now(Some(&client)));
        assert_eq!(credential_store::get_access_token().as_deref(), Some("fresh"));
        assert_eq!(credential_store::get_refresh_token().as_deref(), Some("new-rt"));
    }

    #[test]
    #[serial]
    fn test_rotation_is_optional() {
        let _c = clean_credentials();
        credential_store::set_refresh_token("old-rt");
        let client = MockClient::returning(Ok(obj(json!({"access_token": "fresh"}))));

        assert!(refresh_access_token_now(Some(&client)));
        assert_eq!(credential_store::get_refresh_token().as_deref(), Some("old-rt"));
    }

    #[test]
    #[serial]
    fn test_a_rejected_refresh_token_clears_both() {
        let _c = clean_credentials();
        credential_store::set_access_token("stale");
        credential_store::set_refresh_token("dead-rt");
        let client = MockClient::returning(Err(PlatformError::permission("401")));

        assert!(!refresh_access_token_now(Some(&client)));
        assert_eq!(credential_store::get_access_token(), None);
        assert_eq!(credential_store::get_refresh_token(), None);
    }

    /// A network failure is not evidence the token is bad.
    #[test]
    #[serial]
    fn test_an_outage_keeps_the_credentials_for_a_retry() {
        let _c = clean_credentials();
        credential_store::set_access_token("stale");
        credential_store::set_refresh_token("rt");
        let client = MockClient::returning(Err(PlatformError::connection("unreachable")));

        assert!(!refresh_access_token_now(Some(&client)));
        assert_eq!(credential_store::get_refresh_token().as_deref(), Some("rt"));
        assert_eq!(credential_store::get_access_token().as_deref(), Some("stale"));
    }

    #[test]
    #[serial]
    fn test_a_reply_with_no_token_changes_nothing() {
        let _c = clean_credentials();
        credential_store::set_refresh_token("rt");
        let client = MockClient::returning(Ok(JsonObject::new()));

        assert!(!refresh_access_token_now(Some(&client)));
        assert_eq!(credential_store::get_access_token(), None);
        assert_eq!(credential_store::get_refresh_token().as_deref(), Some("rt"));
    }

    #[test]
    #[serial]
    fn test_nothing_to_redeem_is_not_an_error() {
        let _c = clean_credentials();
        let client = MockClient::returning(Ok(JsonObject::new()));

        assert!(!refresh_access_token_now(Some(&client)));
        assert_eq!(client.call_count(), 0);
    }

    /// Only a 401 may clear the credentials.
    ///
    /// `raise_for_status` used to give every non-2xx the endpoint's error
    /// class, so a 500 from the refresh endpoint arrived as `PermissionError`
    /// and landed in the branch that clears *both* tokens.  A bad minute on
    /// the platform logged the user out with no way back but a re-login.
    fn check_a_platform_outage_is_not_a_rejection(status: u16) {
        let _c = clean_credentials();
        credential_store::set_access_token("stale");
        credential_store::set_refresh_token("rt");
        let exc = PlatformError::Connection { message: format!("API error ({status})"), status_code: Some(status) };
        let client = MockClient::returning(Err(exc));

        assert!(!refresh_access_token_now(Some(&client)));
        assert_eq!(credential_store::get_access_token().as_deref(), Some("stale"));
        assert_eq!(credential_store::get_refresh_token().as_deref(), Some("rt"));
    }

    #[test]
    #[serial]
    fn test_a_platform_outage_is_not_a_rejection_429() {
        check_a_platform_outage_is_not_a_rejection(429);
    }

    #[test]
    #[serial]
    fn test_a_platform_outage_is_not_a_rejection_500() {
        check_a_platform_outage_is_not_a_rejection(500);
    }

    #[test]
    #[serial]
    fn test_a_platform_outage_is_not_a_rejection_502() {
        check_a_platform_outage_is_not_a_rejection(502);
    }

    #[test]
    #[serial]
    fn test_a_platform_outage_is_not_a_rejection_503() {
        check_a_platform_outage_is_not_a_rejection(503);
    }
}

/// Two callers reach the refresh agent: the background loop and a request
/// thread whose call just came back 401.  With single-use rotating refresh
/// tokens, the second redemption of the same token is *refused* — which takes
/// the 401 branch and clears both credentials, logging the user out on what
/// was a successful refresh.
///
/// Two, because these run in the workspace owner.  A pinned agent child is a
/// third caller the lock could not order, and it declines instead — see
/// `test_token_refresh_is_workspace_work`.
mod test_concurrent_callers_redeem_once {
    use super::*;

    #[test]
    #[serial]
    fn test_the_refresh_token_is_spent_once() {
        let _c = clean_credentials();
        credential_store::set_refresh_token("old-rt");
        let started = Arc::new(Barrier::new(2));
        let redeemed = Arc::new(AtomicUsize::new(0));

        let client = {
            let redeemed = redeemed.clone();
            Arc::new(MockClient::with_hook(move |_rt| {
                let n = redeemed.fetch_add(1, Ordering::SeqCst) + 1;
                // hold the lock long enough for the other to queue
                std::thread::sleep(Duration::from_millis(50));
                Ok(obj(json!({"access_token": format!("fresh-{n}"), "refresh_token": "rotated"})))
            }))
        };

        let handles: Vec<_> = (0..2)
            .map(|_| {
                let started = started.clone();
                let client = client.clone();
                std::thread::spawn(move || {
                    started.wait();
                    refresh_access_token_now(Some(&*client))
                })
            })
            .collect();
        let results: Vec<bool> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        assert_eq!(results, vec![true, true]);
        let calls = client.calls.lock().clone();
        assert_eq!(calls, vec!["old-rt".to_string()], "refresh token redeemed {} times", calls.len());
        assert_eq!(credential_store::get_access_token().as_deref(), Some("fresh-1"));
        assert_eq!(credential_store::get_refresh_token().as_deref(), Some("rotated"));
    }

    /// The lock alone only collapses callers when the refresh *succeeds*.
    ///
    /// On failure the stored token is unchanged, so every waiter falls through
    /// and pays the full timeout in turn.  With the auth endpoints served from
    /// the blocking pool, ten concurrent 401s against a degraded platform
    /// meant ten serialized redemptions holding ten pool slots.
    #[test]
    #[serial]
    fn test_a_failed_redemption_holds_off_the_queue_behind_it() {
        let _c = clean_credentials();
        credential_store::set_refresh_token("rt");
        let client = MockClient::returning(Err(PlatformError::connection("platform unreachable")));

        assert!(!refresh_access_token_now(Some(&client)));
        assert!(!refresh_access_token_now(Some(&client)));
        assert!(!refresh_access_token_now(Some(&client)));

        assert_eq!(client.call_count(), 1);
        // Nothing was thrown away — the credentials are still there to retry with.
        assert_eq!(credential_store::get_refresh_token().as_deref(), Some("rt"));
    }

    #[test]
    #[serial]
    fn test_the_hold_off_lifts_once_one_succeeds() {
        let _c = clean_credentials();
        credential_store::set_refresh_token("rt");
        let client = MockClient::returning(Err(PlatformError::connection("down")));
        assert!(!refresh_access_token_now(Some(&client)));

        token_agent::reset_failure_cooldown(); // window elapsed
        let client = MockClient::returning(Ok(obj(json!({"access_token": "fresh"}))));
        assert!(refresh_access_token_now(Some(&client)));
        assert_eq!(credential_store::get_access_token().as_deref(), Some("fresh"));
    }

    /// The short-circuit is "someone else just refreshed", not "ever refreshed".
    #[test]
    #[serial]
    fn test_a_caller_that_genuinely_needs_one_still_gets_one() {
        let _c = clean_credentials();
        credential_store::set_refresh_token("rt");
        let client = MockClient::sequence(vec![
            Ok(obj(json!({"access_token": "first"}))),
            Ok(obj(json!({"access_token": "second"}))),
        ]);

        assert!(refresh_access_token_now(Some(&client)));
        assert!(refresh_access_token_now(Some(&client)));
        assert_eq!(credential_store::get_access_token().as_deref(), Some("second"));
    }
}

/// A client stand-in for `patch.object(token_agent, "SLPPlatformClient", MagicMock())`.
struct NullClient;

impl PlatformApi for NullClient {}

fn null_client() -> Arc<dyn PlatformApi> {
    Arc::new(NullClient)
}

mod test_the_loop_runs_before_it_sleeps {
    use super::*;

    /// Sleeping first left the first whole interval unprotected.
    #[tokio::test]
    #[serial]
    async fn test_the_first_check_happens_immediately() {
        let _c = clean_credentials();
        let calls = Arc::new(Mutex::new(Vec::<&'static str>::new()));

        let hooks = TokenRefreshLoopHooks {
            needs_refresh: Arc::new(|| Ok(true)),
            refresh: {
                let calls = calls.clone();
                Arc::new(move |_client| {
                    calls.lock().push("refreshed");
                    true
                })
            },
            // Stop after the first cycle.
            sleep: Arc::new(|_seconds| async { ControlFlow::Break(()) }.boxed()),
            make_client: Arc::new(null_client),
        };
        token_agent::run_token_refresh_loop_with(hooks).await;

        assert_eq!(*calls.lock(), vec!["refreshed"]);
    }

    #[tokio::test]
    #[serial]
    async fn test_a_failing_cycle_does_not_kill_the_loop() {
        let _c = clean_credentials();
        let cycles = Arc::new(AtomicUsize::new(0));

        let hooks = TokenRefreshLoopHooks {
            needs_refresh: Arc::new(|| Err(anyhow::anyhow!("credential store unreadable"))),
            refresh: Arc::new(|_client| true),
            sleep: {
                let cycles = cycles.clone();
                Arc::new(move |_seconds| {
                    let n = cycles.fetch_add(1, Ordering::SeqCst) + 1;
                    async move { if n >= 2 { ControlFlow::Break(()) } else { ControlFlow::Continue(()) } }.boxed()
                })
            },
            make_client: Arc::new(null_client),
        };
        token_agent::run_token_refresh_loop_with(hooks).await;

        assert_eq!(cycles.load(Ordering::SeqCst), 2);
    }
}

/// Redemption is synchronous; the loop that drives it must not be.
///
/// `refresh_access_token_now` is a blocking HTTP POST with a 10s budget, taken
/// behind a lock a request thread may already hold.  Awaited inline it stalls
/// the whole runtime — and under supervision that runtime is the supervisor's,
/// which proxies every project's chat stream through it byte by byte, so one
/// expired token would pause generation everywhere at once.
mod test_the_loop_does_not_block_the_event_loop {
    use super::*;

    /// Ordering, not elapsed time.
    ///
    /// A blocking call does not make this hang — the redemption still returns
    /// eventually and everything queued behind it still runs.  What gives it
    /// away is *when*: inline, the refresh finishes before any other task gets
    /// a turn.  So the assertion is that the other task ran first, not that
    /// the test was quick.
    #[tokio::test]
    #[serial]
    async fn test_other_work_runs_while_a_redemption_is_in_flight() {
        let _c = clean_credentials();
        credential_store::set_refresh_token("rt");
        credential_store::set_access_token(&jwt(-10)); // expired: force a refresh

        let (redeeming_tx, redeeming_rx) = std::sync::mpsc::channel::<()>();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let release_rx = Arc::new(Mutex::new(release_rx));
        let redeeming_tx = Arc::new(Mutex::new(redeeming_tx));
        let order = Arc::new(Mutex::new(Vec::<&'static str>::new()));

        let slow_refresh = {
            let order = order.clone();
            Arc::new(move |_client: Arc<dyn PlatformApi>| {
                let _ = redeeming_tx.lock().send(());
                let _ = release_rx.lock().recv_timeout(Duration::from_secs(5));
                order.lock().push("refresh-done");
                true
            })
        };

        let hooks = TokenRefreshLoopHooks { refresh: slow_refresh, ..TokenRefreshLoopHooks::default() };
        let loop_task = tokio::spawn(token_agent::run_token_refresh_loop_with(hooks));
        let other = {
            let order = order.clone();
            tokio::spawn(async move {
                for _ in 0..3 {
                    tokio::task::yield_now().await;
                    order.lock().push("tick");
                }
            })
        };

        tokio::task::spawn_blocking(move || redeeming_rx.recv_timeout(Duration::from_secs(5)))
            .await
            .unwrap()
            .expect("the redemption started");
        other.await.unwrap();

        let seen = order.lock().clone();
        assert_eq!(
            seen,
            vec!["tick", "tick", "tick"],
            "the redemption ran to completion before the runtime got a turn — it is blocking \
             the loop (order was {seen:?})"
        );

        release_tx.send(()).unwrap();
        loop_task.abort();
        let _ = loop_task.await;
    }
}
