//! Port of `tests/unit/test_auth_writes_are_workspace_work.py`.
//!
//! Discarding the shared access token belongs to the workspace owner.
//!
//! `resolve_user` drops a dead access token so later requests fall back to the
//! developer token.  That write is workspace-wide, and routing does not keep it
//! in the supervisor as reliably as it looks: a project is resolved from a
//! `project_id` query parameter on *any* path, so `/v1/auth/status?project_id=X`
//! is proxied to an agent child.  A child acting on its own 401 would delete the
//! credential for every project — and it cannot tell a dead token from one the
//! owner has already replaced, because it never saw the replacement.

use std::sync::Arc;

use serial_test::serial;

use smartloop::api::routers::auth as auth_router;
use smartloop::api::routers::auth::AuthPlatformService;
use smartloop::auth::credential_store;
use smartloop::platform::client::{JsonObject, PlatformResult};
use smartloop::platform::PlatformError;
use smartloop::workspace::context::pin_project;

use crate::shared::misc_isolation::IsolatedHome;

/// The `_signed_in` fixture: an access and a refresh token in an isolated
/// credential store, cleared afterwards.
struct SignedIn {
    _home: IsolatedHome,
}

impl SignedIn {
    fn new() -> Self {
        let home = IsolatedHome::new();
        credential_store::set_access_token("at");
        credential_store::set_refresh_token("rt");
        SignedIn { _home: home }
    }
}

impl Drop for SignedIn {
    fn drop(&mut self) {
        credential_store::clear_access_token();
        credential_store::clear_refresh_token();
        auth_router::set_platform_service_override(None);
    }
}

/// A platform service whose `get_current_user` refuses with *status*.
struct RefusingService {
    status: u16,
}

impl AuthPlatformService for RefusingService {
    fn get_current_user(&self, _force: bool) -> PlatformResult<JsonObject> {
        Err(PlatformError::Permission {
            message: format!("HTTP {}", self.status),
            status_code: Some(self.status),
        })
    }

    fn list_models(&self, _force: bool) -> PlatformResult<Vec<JsonObject>> {
        Ok(Vec::new())
    }

    fn list_mcp_catalog(&self, _force: bool) -> PlatformResult<Vec<JsonObject>> {
        Ok(Vec::new())
    }
}

fn refusing_service(status: u16) -> Arc<dyn AuthPlatformService> {
    Arc::new(RefusingService { status })
}

mod test_clearing_the_access_token_is_owner_work {
    use super::*;

    #[test]
    #[serial]
    fn test_a_pinned_process_does_not_clear_the_access_token() {
        let _signed_in = SignedIn::new();
        pin_project(Some("alpha"));

        auth_router::set_platform_service_override(Some(refusing_service(401)));
        assert_eq!(auth_router::resolve_user(), None);
        auth_router::set_platform_service_override(None);

        assert_eq!(credential_store::get_access_token().as_deref(), Some("at"));
    }

    #[test]
    #[serial]
    fn test_the_workspace_owner_still_clears_a_dead_access_token() {
        let _signed_in = SignedIn::new();

        auth_router::set_platform_service_override(Some(refusing_service(401)));
        assert_eq!(auth_router::resolve_user(), None);
        auth_router::set_platform_service_override(None);

        assert_eq!(credential_store::get_access_token(), None);
        assert_eq!(
            credential_store::get_refresh_token().as_deref(),
            Some("rt"),
            "only the access token is local to auth status"
        );
    }

    /// A refusal that is not about the token is not a reason to drop it.
    ///
    /// `get_current_user` raises a permission error for every non-transient
    /// refusal, so an API deploy that moved a route (404) or a tenant policy
    /// (403) used to log the desktop client out.
    fn a_non_401_leaves_the_credential_alone_in_either_process(pinned: Option<&str>, status: u16) {
        let _signed_in = SignedIn::new();
        pin_project(pinned);

        auth_router::set_platform_service_override(Some(refusing_service(status)));
        assert_eq!(auth_router::resolve_user(), None);
        auth_router::set_platform_service_override(None);

        assert_eq!(credential_store::get_access_token().as_deref(), Some("at"));
    }

    #[test]
    #[serial]
    fn test_a_non_401_leaves_the_credential_alone_in_either_process_none_403() {
        a_non_401_leaves_the_credential_alone_in_either_process(None, 403);
    }

    #[test]
    #[serial]
    fn test_a_non_401_leaves_the_credential_alone_in_either_process_alpha_403() {
        a_non_401_leaves_the_credential_alone_in_either_process(Some("alpha"), 403);
    }

    #[test]
    #[serial]
    fn test_a_non_401_leaves_the_credential_alone_in_either_process_none_404() {
        a_non_401_leaves_the_credential_alone_in_either_process(None, 404);
    }

    #[test]
    #[serial]
    fn test_a_non_401_leaves_the_credential_alone_in_either_process_alpha_404() {
        a_non_401_leaves_the_credential_alone_in_either_process(Some("alpha"), 404);
    }
}
