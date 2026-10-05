//! Port of `tests/unit/test_token_refresh_is_workspace_work.py`.
//!
//! Redeeming the refresh token belongs to the process that owns the workspace.
//!
//! Under supervision the server is not one process but N+1: a supervisor plus
//! an agent child per project, all running the same app against one credential
//! file.  Everything that stops two *threads* redeeming the same single-use
//! refresh token — the refresh lock, the failed-refresh cooldown — is
//! process-local, so none of it orders two *processes*.
//!
//! The consequence is not a wasted request.  The second redemption of a
//! rotated token is refused, `redeem` reads that refusal as "the refresh token
//! is dead", and clears both credentials: every project logged out, by the
//! machinery that exists to keep people logged in.  So a pinned process
//! declines, and the workspace owner is the only one that ever spends the token.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::json;
use serial_test::serial;
use smartloop::auth::credential_store;
use smartloop::platform::client::{JsonObject, PlatformApi, PlatformError, PlatformResult, SLPPlatformClient};
use smartloop::platform::token_agent::{refresh_access_token_now, set_credential_read_observer};
use smartloop::workspace::context::pin_project;

use crate::shared::platform_support::{autouse, obj, Conftest, StubServer};

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

/// `MagicMock(spec=SLPPlatformClient)` with a scripted `refresh_access_token`.
struct MockClient {
    answer: PlatformResult<JsonObject>,
    calls: Mutex<Vec<String>>,
}

impl MockClient {
    fn answering(answer: PlatformResult<JsonObject>) -> Self {
        MockClient { answer, calls: Mutex::new(Vec::new()) }
    }
}

impl PlatformApi for MockClient {
    fn refresh_access_token(&self, refresh_token: &str) -> PlatformResult<JsonObject> {
        self.calls.lock().push(refresh_token.to_string());
        self.answer.clone()
    }
}

/// A client whose refresh endpoint hands back a rotated pair.
fn rotating_client() -> MockClient {
    MockClient::answering(Ok(obj(json!({"access_token": "new-at", "refresh_token": "new-rt"}))))
}

mod test_only_the_workspace_owner_redeems {
    use super::*;

    #[test]
    #[serial]
    fn test_a_pinned_child_does_not_redeem() {
        let _c = clean_credentials();
        credential_store::set_refresh_token("rt");
        pin_project(Some("alpha"));
        let client = rotating_client();

        assert!(!refresh_access_token_now(Some(&client)));
        assert!(client.calls.lock().is_empty());
    }

    /// The regression.  A child must not turn its own 401 into a logout.
    ///
    /// Without the gate this is what a child racing the supervisor looks like:
    /// it presents a token the supervisor already spent, the platform refuses
    /// it, and `redeem` deletes both credentials for every project.
    #[test]
    #[serial]
    fn test_a_pinned_child_does_not_clear_on_a_dead_refresh_token() {
        let _c = clean_credentials();
        credential_store::set_access_token("at");
        credential_store::set_refresh_token("rt");
        pin_project(Some("alpha"));

        let client = MockClient::answering(Err(PlatformError::permission("HTTP 401")));

        assert!(!refresh_access_token_now(Some(&client)));
        assert!(client.calls.lock().is_empty());
        assert_eq!(credential_store::get_access_token().as_deref(), Some("at"));
        assert_eq!(credential_store::get_refresh_token().as_deref(), Some("rt"));
    }

    /// The gate is the first statement, ahead of any credential I/O.
    ///
    /// White-box on purpose: a child answering 401s during a chat turn should
    /// not be decrypting the credential file to discover it has nothing to do.
    #[test]
    #[serial]
    fn test_a_pinned_child_reads_no_credentials_at_all() {
        let _c = clean_credentials();
        credential_store::set_refresh_token("rt");
        pin_project(Some("alpha"));

        /// Removes the read observer when dropped (`MonkeyPatch.context()`).
        struct ObserverGuard;
        impl Drop for ObserverGuard {
            fn drop(&mut self) {
                set_credential_read_observer(None);
            }
        }

        let reads = Arc::new(Mutex::new(Vec::<String>::new()));
        let _guard = ObserverGuard;
        {
            let reads = reads.clone();
            set_credential_read_observer(Some(Arc::new(move |what: &str| reads.lock().push(what.to_string()))));
        }
        assert!(!refresh_access_token_now(Some(&rotating_client())));
        assert!(
            reads.lock().is_empty(),
            "a pinned process read the credential store: {:?}",
            reads.lock()
        );
    }

    #[test]
    #[serial]
    fn test_the_workspace_owner_still_redeems() {
        let _c = clean_credentials();
        credential_store::set_access_token("old-at");
        credential_store::set_refresh_token("old-rt");
        let client = rotating_client();

        assert!(refresh_access_token_now(Some(&client)));
        assert_eq!(*client.calls.lock(), vec!["old-rt".to_string()]);
        assert_eq!(credential_store::get_access_token().as_deref(), Some("new-at"));
        assert_eq!(credential_store::get_refresh_token().as_deref(), Some("new-rt"));
    }

    /// The gate covers the path `authed` actually takes.
    ///
    /// `authed` calls `refresh_fn`, which defaults to `refresh_credentials` —
    /// so gating anywhere shallower than `refresh_access_token_now` would leave
    /// this reachable.  The client points at a stub server so any redemption
    /// attempt would be recorded.
    #[test]
    #[serial]
    fn test_the_default_client_hook_is_gated_too() {
        let _c = clean_credentials();
        credential_store::set_refresh_token("rt");
        pin_project(Some("alpha"));

        let server = StubServer::start();
        server.route("/v1/auth/jwt/refresh", 200, json!({"access_token": "new-at"}));
        let client = SLPPlatformClient::new(server.base_url());

        assert!(!client.refresh_credentials());
        assert!(server.requests().is_empty(), "refresh_access_token was called");
    }
}
