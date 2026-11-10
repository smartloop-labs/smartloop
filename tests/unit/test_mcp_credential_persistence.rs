//! Port of `tests/unit/test_mcp_credential_persistence.py`.
//!
//! A refreshed MCP token has to reach the workspace.
//!
//! `MCPToolCaller::refresh_access_token` stored the new credentials on the
//! in-memory server object and stopped there.  Only the content crawl wrote
//! them back, through its own `on_server_updated` hook — so whether a
//! connection survived a restart depended on which code path happened to
//! refresh it last.  With a provider that rotates refresh tokens, the workspace
//! kept one the provider had already retired, and the connection could never
//! be renewed again.
//!
//! The Python tests patched `httpx.Client`; here the token endpoint is a local
//! wiremock server the connection's `token_endpoint` points at.

use std::collections::HashMap;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde_json::{json, Value};
use serial_test::serial;
use smartloop::agents::mcp::argument_extractor::MCPArgumentExtractor;
use smartloop::agents::mcp::mcp_tool_caller::{self as module, reset_refresh_state, MCPToolCaller, PersistFn, ServerChange};
use smartloop::agents::mcp::result_mapper::MCPResultMapper;
use smartloop::workspace::workspace::{
    MCPAuthType, MCPOAuthCredentials, MCPServer, MCPServerType, RemoteMCPServerConfig,
};
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

use crate::shared::agents_mcp_args::ScriptedLLM;

/// The token endpoint the fixture server uses when no request is expected.
const MONDAY_TOKEN_ENDPOINT: &str = "https://auth.monday.com/oauth_ms/oauth/token";

/// conftest `_reset_mcp_refresh_state`: forget locks and cooldowns around a test.
struct RefreshStateGuard;

impl RefreshStateGuard {
    fn new() -> Self {
        reset_refresh_state();
        Self
    }
}

impl Drop for RefreshStateGuard {
    fn drop(&mut self) {
        reset_refresh_state();
    }
}

fn server_with_endpoint(token_endpoint: &str) -> MCPServer {
    MCPServer {
        id: "s1".to_string(),
        name: "monday".to_string(),
        server_type: MCPServerType::Remote,
        remote_config: Some(RemoteMCPServerConfig {
            server_url: "https://mcp.monday.com/mcp".to_string(),
            auth_type: MCPAuthType::Oauth,
            token_endpoint: Some(token_endpoint.to_string()),
            ..Default::default()
        }),
        oauth_credentials: Some(MCPOAuthCredentials {
            client_id: "cid".to_string(),
            access_token: Some("expired".to_string()),
            refresh_token: Some("old-rt".to_string()),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn caller(persist: Option<PersistFn>) -> MCPToolCaller {
    let extractor = Arc::new(MCPArgumentExtractor::new(Arc::new(ScriptedLLM::new(""))));
    MCPToolCaller::new(extractor, Arc::new(Mutex::new(MCPResultMapper::new())), persist)
}

fn creds(server: &MCPServer) -> &MCPOAuthCredentials {
    server.oauth_credentials.as_ref().unwrap()
}

/// A token endpoint served by wiremock (stands in for the patched `httpx.Client`).
struct TokenEndpoint {
    server: MockServer,
    rt: tokio::runtime::Runtime,
}

/// Responds through a closure over the posted form.
struct FormResponder(Arc<dyn Fn(&HashMap<String, String>) -> ResponseTemplate + Send + Sync>);

impl Respond for FormResponder {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let form: HashMap<String, String> = url::form_urlencoded::parse(&request.body)
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();
        (self.0)(&form)
    }
}

impl TokenEndpoint {
    fn start(responder: FormResponder) -> Self {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let server = rt.block_on(async {
            let server = MockServer::start().await;
            Mock::given(method("POST")).respond_with(responder).mount(&server).await;
            server
        });
        Self { server, rt }
    }

    /// `_reply`: the refresh grant returns `payload`.
    fn reply(payload: Value) -> Self {
        Self::start(FormResponder(Arc::new(move |_| {
            ResponseTemplate::new(200).set_body_json(payload.clone())
        })))
    }

    /// `_reply_from`: like [`Self::reply`], but `grant` is called with the posted form.
    fn reply_from(
        grant: impl Fn(&HashMap<String, String>) -> Value + Send + Sync + 'static,
        delay: Duration,
    ) -> Self {
        Self::start(FormResponder(Arc::new(move |form| {
            ResponseTemplate::new(200).set_body_json(grant(form)).set_delay(delay)
        })))
    }

    /// `_stalling`: the endpoint does not answer usefully for `delay`, then fails.
    fn stalling(delay: Duration) -> Self {
        Self::start(FormResponder(Arc::new(move |_| {
            ResponseTemplate::new(504)
                .set_body_string("the token endpoint is not answering")
                .set_delay(delay)
        })))
    }

    fn url(&self) -> String {
        format!("{}/oauth_ms/oauth/token", self.server.uri())
    }

    /// How many grants were posted.
    fn post_count(&self) -> usize {
        self.rt
            .block_on(self.server.received_requests())
            .map(|r| r.len())
            .unwrap_or(0)
    }
}

mod test_the_refresh_is_written_back {
    use super::*;

    #[test]
    #[serial]
    fn test_rotated_credentials_are_persisted() {
        let _guard = RefreshStateGuard::new();
        let endpoint = TokenEndpoint::reply(json!({"access_token": "fresh", "refresh_token": "rotated"}));
        let mut server = server_with_endpoint(&endpoint.url());
        let saves = Arc::new(Mutex::new(Vec::<ServerChange>::new()));
        let saves_hook = saves.clone();
        let caller = caller(Some(Arc::new(move |_s: &MCPServer, change: ServerChange| {
            saves_hook.lock().push(change);
            Ok(())
        })));

        assert_eq!(caller.refresh_access_token(&mut server, None).as_deref(), Some("fresh"));

        assert_eq!(creds(&server).access_token.as_deref(), Some("fresh"));
        assert_eq!(creds(&server).refresh_token.as_deref(), Some("rotated"));
        assert_eq!(*saves.lock(), vec![ServerChange::Credentials], "the rotated refresh token never reached the workspace");
    }

    /// The token in memory is good — the call this refresh was for proceeds.
    #[test]
    #[serial]
    fn test_a_failed_write_does_not_sink_the_call() {
        let _guard = RefreshStateGuard::new();
        let endpoint = TokenEndpoint::reply(json!({"access_token": "fresh"}));
        let mut server = server_with_endpoint(&endpoint.url());

        let caller = caller(Some(Arc::new(|_s: &MCPServer, _change: ServerChange| {
            Err(anyhow::anyhow!("read-only file system"))
        })));
        assert_eq!(caller.refresh_access_token(&mut server, None).as_deref(), Some("fresh"));
    }

    #[test]
    #[serial]
    fn test_nothing_is_persisted_when_the_refresh_fails() {
        let _guard = RefreshStateGuard::new();
        // no access_token
        let endpoint = TokenEndpoint::reply(json!({"token_type": "Bearer"}));
        let mut server = server_with_endpoint(&endpoint.url());
        let saves = Arc::new(Mutex::new(Vec::<i32>::new()));
        let saves_hook = saves.clone();
        let caller = caller(Some(Arc::new(move |_s: &MCPServer, _change: ServerChange| {
            saves_hook.lock().push(1);
            Ok(())
        })));

        assert_eq!(caller.refresh_access_token(&mut server, None), None);

        assert!(saves.lock().is_empty());
        assert_eq!(creds(&server).access_token.as_deref(), Some("expired"));
    }

    /// The content crawl persists through its own `on_server_updated`.
    #[test]
    #[serial]
    fn test_a_caller_without_a_hook_still_refreshes() {
        let _guard = RefreshStateGuard::new();
        let endpoint = TokenEndpoint::reply(json!({"access_token": "fresh"}));
        let mut server = server_with_endpoint(&endpoint.url());
        let caller = caller(None);

        assert_eq!(caller.refresh_access_token(&mut server, None).as_deref(), Some("fresh"));
    }

    fn nothing_to_refresh_with(missing: &str) {
        let _guard = RefreshStateGuard::new();
        let mut server = server_with_endpoint(MONDAY_TOKEN_ENDPOINT);
        if missing == "refresh_token" {
            server.oauth_credentials.as_mut().unwrap().refresh_token = None;
        } else {
            server.remote_config.as_mut().unwrap().token_endpoint = None;
        }

        let caller = caller(Some(Arc::new(|_s: &MCPServer, _change: ServerChange| -> anyhow::Result<()> {
            panic!("no refresh happened")
        })));
        assert_eq!(caller.refresh_access_token(&mut server, None), None);
    }

    #[test]
    #[serial]
    fn test_nothing_to_refresh_with_is_not_an_error_refresh_token() {
        nothing_to_refresh_with("refresh_token");
    }

    #[test]
    #[serial]
    fn test_nothing_to_refresh_with_is_not_an_error_token_endpoint() {
        nothing_to_refresh_with("token_endpoint");
    }
}

/// Concurrent 401s must not spend the same refresh token twice.
///
/// Parallel step groups and tool fan-out share one caller, so several calls
/// can come back 401 together.  Each used to redeem independently — and a
/// provider that issues single-use refresh tokens invalidates the grant on the
/// second attempt, killing the connection the refresh was meant to save.
mod test_one_redemption_per_rotation {
    use super::*;

    #[test]
    #[serial]
    fn test_the_loser_of_the_race_reuses_the_winners_token() {
        let _guard = RefreshStateGuard::new();
        let endpoint = TokenEndpoint::reply(json!({"access_token": "never", "refresh_token": "never"}));
        let mut server = server_with_endpoint(&endpoint.url());
        let caller = caller(None);

        // The state after another thread has already refreshed: the access
        // token on the server is no longer the one this caller's request
        // failed with.
        server.oauth_credentials.as_mut().unwrap().access_token = Some("fresh".to_string());

        let token = caller.refresh_access_token(&mut server, Some("expired"));

        assert_eq!(token.as_deref(), Some("fresh"));
        assert_eq!(endpoint.post_count(), 0);
        assert_eq!(creds(&server).refresh_token.as_deref(), Some("old-rt"));
    }

    #[test]
    #[serial]
    fn test_concurrent_callers_redeem_exactly_once() {
        let _guard = RefreshStateGuard::new();
        let grants = Arc::new(Mutex::new(Vec::<String>::new()));
        let grants_seen = grants.clone();
        // The round trip is what makes the window real: without the lock every
        // thread clears the stale check while the first is still waiting on
        // the network, and they all redeem.
        let endpoint = TokenEndpoint::reply_from(
            move |payload| {
                let mut g = grants_seen.lock();
                g.push(payload.get("refresh_token").cloned().unwrap_or_default());
                let n = g.len();
                json!({"access_token": format!("fresh-{n}"), "refresh_token": "new-rt"})
            },
            Duration::from_millis(50),
        );

        // One server object shared by every caller, as in the Python test.
        let server = Arc::new(Mutex::new(server_with_endpoint(&endpoint.url())));
        let caller = Arc::new(caller(None));
        let barrier = Arc::new(Barrier::new(4));
        let results = Arc::new(Mutex::new(Vec::<Option<String>>::new()));

        let threads: Vec<_> = (0..4)
            .map(|_| {
                let (server, caller, barrier, results) =
                    (server.clone(), caller.clone(), barrier.clone(), results.clone());
                std::thread::spawn(move || {
                    barrier.wait();
                    let token = {
                        let mut s = server.lock();
                        caller.refresh_access_token(&mut s, Some("expired"))
                    };
                    results.lock().push(token);
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }

        let grants = grants.lock().clone();
        assert_eq!(grants.len(), 1, "redeemed {} times: {:?}", grants.len(), grants);
        let distinct: std::collections::HashSet<Option<String>> = results.lock().iter().cloned().collect();
        assert_eq!(distinct, [Some("fresh-1".to_string())].into_iter().collect());
    }

    /// Callers that cannot say what they failed with are not blocked.
    #[test]
    #[serial]
    fn test_no_stale_token_still_refreshes() {
        let _guard = RefreshStateGuard::new();
        let endpoint = TokenEndpoint::reply(json!({"access_token": "fresh", "refresh_token": "new-rt"}));
        let mut server = server_with_endpoint(&endpoint.url());
        let caller = caller(None);
        server.oauth_credentials.as_mut().unwrap().access_token = Some("something-else".to_string());

        assert_eq!(caller.refresh_access_token(&mut server, None).as_deref(), Some("fresh"));
    }
}

/// Serializing redemptions must not serialize their timeouts.
///
/// Waiters queue behind the caller that is talking to the token endpoint, so
/// where they each used to time out in parallel they now time out one after
/// another — turning one 30-second stall into two minutes of a chat turn on a
/// four-call fan-out.  A refusal stands briefly, keyed on the token that was
/// refused so a reconnection is never held back by it.
mod test_a_refusal_is_not_re_paid_by_every_waiter {
    use super::*;

    #[test]
    #[serial]
    fn test_only_the_first_waiter_pays_the_timeout() {
        let _guard = RefreshStateGuard::new();
        let stall = Duration::from_millis(200);
        let endpoint = TokenEndpoint::stalling(stall);
        let server = Arc::new(Mutex::new(server_with_endpoint(&endpoint.url())));
        let caller = Arc::new(caller(None));
        let barrier = Arc::new(Barrier::new(4));

        let started = Instant::now();
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let (server, caller, barrier) = (server.clone(), caller.clone(), barrier.clone());
                std::thread::spawn(move || {
                    barrier.wait();
                    let mut s = server.lock();
                    caller.refresh_access_token(&mut s, Some("expired"));
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let elapsed = started.elapsed();

        assert_eq!(endpoint.post_count(), 1);
        // Serialized it would be four stalls; the cooldown collapses it to one.
        assert!(
            elapsed < stall * 2,
            "{:.2}s for a {:.2}s stall",
            elapsed.as_secs_f64(),
            stall.as_secs_f64()
        );
    }

    /// The cooldown is keyed on the refused token, not on the server.
    #[test]
    #[serial]
    fn test_a_reconnection_is_not_held_back() {
        let _guard = RefreshStateGuard::new();
        let stalling = TokenEndpoint::stalling(Duration::ZERO);
        let mut server = server_with_endpoint(&stalling.url());
        let caller = caller(None);

        assert_eq!(caller.refresh_access_token(&mut server, Some("expired")), None);

        // Re-authorizing mints a different refresh token.
        server.oauth_credentials.as_mut().unwrap().refresh_token = Some("brand-new-rt".to_string());
        let working = TokenEndpoint::reply(json!({"access_token": "fresh", "refresh_token": "newer-rt"}));
        server.remote_config.as_mut().unwrap().token_endpoint = Some(working.url());
        assert_eq!(
            caller.refresh_access_token(&mut server, Some("expired")).as_deref(),
            Some("fresh")
        );
    }

    #[test]
    #[serial]
    fn test_a_success_clears_the_hold() {
        let _guard = RefreshStateGuard::new();
        let stalling = TokenEndpoint::stalling(Duration::ZERO);
        let mut server = server_with_endpoint(&stalling.url());
        let caller = caller(None);

        assert_eq!(caller.refresh_access_token(&mut server, Some("expired")), None);

        let working = TokenEndpoint::reply(json!({"access_token": "fresh"}));
        server.remote_config.as_mut().unwrap().token_endpoint = Some(working.url());
        // Same refresh token, but nothing is holding it back once it works.
        // The hold is shared across callers now, so it is cleared there.
        module::REFRESH_FAILURES.lock().clear();
        assert_eq!(
            caller.refresh_access_token(&mut server, Some("expired")).as_deref(),
            Some("fresh")
        );
        assert!(!module::REFRESH_FAILURES.lock().contains_key(&server.id));
    }
}
