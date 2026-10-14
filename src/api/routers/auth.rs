//! Port of `smartloop/api/routers/auth.py`.
//!
//! Auth router – credential management and OAuth callback for the local API server.
//!
//! Endpoints:
//!     GET    /v1/auth/status    – check credential state
//!     POST   /v1/auth/callback  – receive tokens from webapp after OAuth flow
//!     GET    /v1/auth/callback  – receive tokens via query params (Safari fallback)
//!     POST   /v1/auth/token     – manually set a token (developer_token or access_token)
//!     DELETE /v1/auth/token     – clear stored credentials

use std::sync::Arc;

use axum::http::header::LOCATION;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::auth::credential_store;
use crate::platform::client::{JsonObject, PlatformResult};
use crate::platform::PlatformService;
use crate::search::providers;
use crate::workspace::context::{owns_workspace, pinned_project_id};

use super::_errors::{run_blocking, validation_error, HTTPException, ValidatedJson, ValidatedQuery};

/// Logger name used by this router.
pub const LOGGER: &str = "smartloop.api.routers.auth";

// -- request / response models ------------------------------------------------

/// Kind of token stored via `POST /v1/auth/token`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum TokenType {
    #[default]
    DeveloperToken,
    AccessToken,
}

/// Which credentials `DELETE /v1/auth/token` clears.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ClearScope {
    #[default]
    All,
    DeveloperToken,
    AccessToken,
}

/// Body of `POST /v1/auth/token`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct TokenSetRequest {
    /// The token value to store (min length 1).
    pub token: String,
    /// developer_token (CLI/scripts) or access_token (OAuth/Electron).
    #[serde(default, rename = "type")]
    pub type_: TokenType,
}

impl TokenSetRequest {
    /// Pydantic's `Field(min_length=1)` on `token`.
    fn validate(&self) -> Result<(), HTTPException> {
        if self.token.chars().count() < 1 {
            return Err(validation_error(vec![json!({
                "type": "string_too_short",
                "loc": ["body", "token"],
                "msg": "String should have at least 1 character",
                "input": self.token,
                "ctx": {"min_length": 1},
            })]));
        }
        Ok(())
    }
}

/// The logged-in user's identity.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct UserInfo {
    pub email: String,
    #[serde(default)]
    pub name: Option<String>,
}

/// Credential state, without token values.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct AuthStatusResponse {
    pub authenticated: bool,
    /// `"developer_token"`, `"access_token"` or null.
    #[serde(default)]
    pub method: Option<TokenType>,
    pub has_developer_token: bool,
    pub has_access_token: bool,
    #[serde(default)]
    pub user: Option<UserInfo>,
}

/// Query of `DELETE /v1/auth/token`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
pub struct ClearTokenQuery {
    /// Which credentials to clear: all, developer_token, or access_token.
    #[serde(default, rename = "type")]
    pub type_: ClearScope,
}

/// Body of `POST /v1/auth/callback`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct OAuthCallbackRequest {
    /// JWT access token from the OAuth flow.
    pub access_token: String,
    /// Refresh token for renewing the access token.
    #[serde(default)]
    pub refresh_token: Option<String>,
}

/// Query of `GET /v1/auth/callback` (Safari fallback).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
pub struct OAuthCallbackQuery {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
}

// -- platform seam --------------------------------------------------------------

/// The two platform-service calls this router makes.
///
/// The router resolves the service through its own module-level
/// [`get_platform_service`] (not through a request dependency), exactly as the
/// Python router did; tests replace it with [`set_platform_service_override`]
/// the way the Python tests patch `smartloop.api.routers.auth.get_platform_service`.
pub trait AuthPlatformService: Send + Sync {
    /// `PlatformService.get_current_user(force=...)`.
    fn get_current_user(&self, force: bool) -> PlatformResult<JsonObject>;
    /// `PlatformService.list_models(force=...)`.
    fn list_models(&self, force: bool) -> PlatformResult<Vec<JsonObject>>;
}

impl AuthPlatformService for PlatformService {
    fn get_current_user(&self, force: bool) -> PlatformResult<JsonObject> {
        PlatformService::get_current_user(self, force)
    }

    fn list_models(&self, force: bool) -> PlatformResult<Vec<JsonObject>> {
        PlatformService::list_models(self, force)
    }
}

static PLATFORM_SERVICE_OVERRIDE: Lazy<RwLock<Option<Arc<dyn AuthPlatformService>>>> =
    Lazy::new(|| RwLock::new(None));

/// Make this router's [`get_platform_service`] return *service* (or, with
/// `None`, the process-wide platform service again).
pub fn set_platform_service_override(service: Option<Arc<dyn AuthPlatformService>>) {
    *PLATFORM_SERVICE_OVERRIDE.write() = service;
}

/// The platform service this router talks to.
pub fn get_platform_service() -> Arc<dyn AuthPlatformService> {
    if let Some(service) = PLATFORM_SERVICE_OVERRIDE.read().clone() {
        return service;
    }
    crate::platform::get_platform_service()
}

/// Replacement for [`resolve_user`] as called by [`auth_status`].
pub type ResolveUserFn = Arc<dyn Fn() -> Option<UserInfo> + Send + Sync>;

static RESOLVE_USER_OVERRIDE: Lazy<RwLock<Option<ResolveUserFn>>> = Lazy::new(|| RwLock::new(None));

/// Make [`auth_status`] resolve the user through *resolver* (or, with `None`,
/// through [`resolve_user`] again) — the Python tests' patch of `_resolve_user`.
pub fn set_resolve_user_override(resolver: Option<ResolveUserFn>) {
    *RESOLVE_USER_OVERRIDE.write() = resolver;
}

/// [`resolve_user`], or the installed override.
fn resolve_user_for_status() -> Option<UserInfo> {
    let resolver = RESOLVE_USER_OVERRIDE.read().clone();
    match resolver {
        Some(resolver) => resolver(),
        None => resolve_user(),
    }
}

// -- endpoints ----------------------------------------------------------------

/// Webapp base URL the Safari fallback redirects to.
///
/// `AppSettings` has no `slp_app_url` field, so this is always the hosted
/// webapp — the Python `hasattr(AppSettings(), "slp_app_url")` check never
/// succeeds.
pub static APP_URL: Lazy<String> = Lazy::new(|| "https://app.smartloop.ai".to_string());

/// The credential value the platform cache was last refreshed for (dedupe).
static LAST_REFRESHED_TOKEN: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));

/// Routes under `/v1/auth` (tag `auth`).
pub fn router() -> Router {
    Router::new()
        .route("/v1/auth/status", get(auth_status_handler))
        .route("/v1/auth/token", post(set_token_handler).delete(clear_token_handler))
        .route("/v1/auth/callback", post(oauth_callback_handler).get(oauth_callback_get_handler))
}

/// Force-refresh the cached model catalog (and optionally the user) on an
/// auth change.
///
/// The catalog always refreshes so each model's `is_accessible` — which the
/// cloud computes on the fly per-user — reflects the current auth state instead
/// of a stale entry.  On logout pass `refresh_user = false`: the cached user
/// profile is intentionally left in place (stale is fine; already-downloaded
/// models stay usable), and we avoid a pointless 401 round-trip.
///
/// Best-effort and independent: a failed user fetch doesn't block the catalog
/// refresh.  A single login delivers the token redundantly (POST + Safari GET
/// fallback, plus retries), so we dedupe on the credential value — only a
/// genuine change triggers a refetch.
pub fn refresh_platform_cache(refresh_user: bool) {
    let token = credential_store::get_access_token().or_else(credential_store::get_token);
    if token.is_some() && *LAST_REFRESHED_TOKEN.lock() == token {
        return;
    }

    // Whether hosted search will serve is an answer about the account, cached for
    // a minute.  Signing in with that cache holding a "no" from the logged-out
    // state left search skipping itself for the first minute of the session —
    // `reset_status_cache` existed for exactly this and nothing called it.
    providers::reset_status_cache();

    let svc = get_platform_service();
    if refresh_user {
        if let Err(e) = svc.get_current_user(true) {
            tracing::warn!(target: LOGGER, "Failed to refresh cached user: {}", e);
        }
    }
    if let Err(e) = svc.list_models(true) {
        tracing::warn!(target: LOGGER, "Failed to refresh cached model catalog: {}", e);
    }
    *LAST_REFRESHED_TOKEN.lock() = token;
}

/// Build a [`UserInfo`] from the platform's user payload, as
/// `UserInfo(email=data["email"], name=data.get("name"))` would.
fn user_info_from(data: &JsonObject) -> Result<UserInfo, String> {
    let email = match data.get("email") {
        Some(Value::String(email)) => email.clone(),
        Some(other) => return Err(format!("email: Input should be a valid string, got {other}")),
        None => return Err("'email'".to_string()),
    };
    let name = match data.get("name") {
        None | Some(Value::Null) => None,
        Some(Value::String(name)) => Some(name.clone()),
        Some(other) => return Err(format!("name: Input should be a valid string, got {other}")),
    };
    Ok(UserInfo { email, name })
}

/// Return the logged-in user's profile, or `None` on failure.
///
/// Resolves the user through the platform service layer so the cached user
/// info is used instead of an uncached client call.
///
/// On a 401 (which already survived the client's refresh-and-retry) the dead
/// access token is dropped — but only by the workspace owner; a pinned project
/// agent leaves the credential alone.  Any other refusal status leaves
/// credentials untouched.
pub fn resolve_user() -> Option<UserInfo> {
    let result = get_platform_service().get_current_user(false);
    match result {
        Ok(data) => match user_info_from(&data) {
            Ok(user) => Some(user),
            Err(e) => {
                tracing::warn!(target: LOGGER, "Could not resolve user from platform API: {}", e);
                None
            }
        },
        Err(exc) if exc.is_permission() => {
            // `get_current_user` raises a permission error for every refusal
            // that is not transient — 403 and 404 as much as 401 — so the status
            // has to be read rather than assumed.  Only a 401 is a statement
            // about the token.  Discarding one on a 404 meant an API deploy that
            // moved a route logged everyone out of the desktop client.
            let status = exc.status_code();
            if status != Some(401) {
                let shown = status.map(|s| s.to_string()).unwrap_or_else(|| "?".to_string());
                tracing::warn!(
                    target: LOGGER,
                    "Could not resolve user (HTTP {}) — leaving credentials alone",
                    shown
                );
                return None;
            }

            // A 401 that survived `SLPPlatformClient::authed`, which redeems the
            // refresh token and retries before any error is raised.  So by the
            // time one arrives here the refresh has already been tried and
            // failed — there was nothing to refresh with, the refresh itself was
            // refused, or the fresh token was refused too.
            //
            // What is left is the part that was always local to auth status:
            // drop the dead access token so later requests fall back to the
            // developer token.  A refresh token that was itself rejected has
            // already been cleared by the refresh agent.
            //
            // Dropping it is workspace-owner work.  Routing normally keeps this
            // endpoint on the supervisor, but a project is resolved from a
            // `project_id` query parameter on *any* path, so a stray
            // `/v1/auth/status?project_id=X` is proxied to an agent child — which
            // would delete the whole workspace's access token on the strength of
            // one project's 401.  A child also cannot tell a dead credential from
            // one the owner has already replaced, because it never saw the
            // replacement.
            if !owns_workspace() {
                tracing::info!(
                    target: LOGGER,
                    "401 from the platform API in project agent {} — leaving the credential to the workspace owner",
                    pinned_project_id().unwrap_or_else(|| "None".to_string())
                );
                return None;
            }

            if credential_store::get_access_token().is_some() {
                credential_store::clear_access_token();
                tracing::warn!(target: LOGGER, "Cleared expired access token after 401 from platform API");
            }
            None
        }
        Err(e) => {
            tracing::warn!(target: LOGGER, "Could not resolve user from platform API: {}", e);
            None
        }
    }
}

/// Current credential state without exposing token values.
///
/// Sync on purpose, along with every other handler body in this router: they
/// resolve the user through the platform API with a synchronous client, and
/// on the worst path — a 401, a refresh, a retry — that is three 30-second
/// calls.  The axum handlers run them on the blocking pool so they never stall
/// a chat stream in flight.
pub fn auth_status() -> AuthStatusResponse {
    let has_dev = credential_store::get_token().is_some();
    let has_access = credential_store::get_access_token().is_some();
    let authenticated = has_dev || has_access;

    // resolve_user may clear an expired access token on 401,
    // so re-read credential state afterward.
    let user = if authenticated { resolve_user_for_status() } else { None };

    let has_dev = credential_store::get_token().is_some();
    let has_access = credential_store::get_access_token().is_some();
    let authenticated = has_dev || has_access;

    let method = if has_access {
        Some(TokenType::AccessToken)
    } else if has_dev {
        Some(TokenType::DeveloperToken)
    } else {
        None
    };

    AuthStatusResponse {
        authenticated,
        method,
        has_developer_token: has_dev,
        has_access_token: has_access,
        user,
    }
}

/// Store a token in the credential store.
///
/// Defaults to developer_token (used by `slp token set`).  Pass
/// `"type": "access_token"` for OAuth/JWT tokens.
pub fn set_token(request: TokenSetRequest) -> AuthStatusResponse {
    if request.type_ == TokenType::DeveloperToken {
        credential_store::set_token(&request.token);
        tracing::info!(target: LOGGER, "Developer token updated via /v1/auth/token");
    } else {
        credential_store::set_access_token(&request.token);
        tracing::info!(target: LOGGER, "Access token updated via /v1/auth/token");
    }

    refresh_platform_cache(true);
    auth_status()
}

/// Clear stored credentials.
///
/// By default clears everything (logout).  Pass `?type=developer_token` or
/// `?type=access_token` to clear only one credential type.
pub fn clear_token(type_: ClearScope) -> AuthStatusResponse {
    if matches!(type_, ClearScope::All | ClearScope::DeveloperToken) {
        credential_store::clear_token();
        tracing::info!(target: LOGGER, "Developer token cleared via /v1/auth/token");
    }

    if matches!(type_, ClearScope::All | ClearScope::AccessToken) {
        credential_store::clear_access_token();
        credential_store::clear_refresh_token();
        tracing::info!(target: LOGGER, "Access/refresh tokens cleared via /v1/auth/token");
    }

    // Refresh the catalog (now anonymous) so model is_accessible reflects the
    // logout.  Keep the cached user profile as-is — stale is acceptable and
    // already-downloaded models remain usable.
    refresh_platform_cache(false);
    auth_status()
}

/// Receive tokens from the webapp after a completed OAuth flow.
///
/// The cloud API handles the actual OAuth code exchange and redirects the
/// browser to the webapp with the minted JWT.  The webapp then POSTs the
/// token(s) here so the local server can store them in the credential store.
pub fn oauth_callback(request: OAuthCallbackRequest) -> AuthStatusResponse {
    credential_store::set_tokens(&request.access_token, request.refresh_token.as_deref());

    log_token_received("POST", &request.access_token, request.refresh_token.as_deref());
    refresh_platform_cache(true);
    auth_status()
}

/// `token[:4]...token[-4:]` when longer than 8 characters, else `***`.
fn redact(token: &str) -> String {
    let chars: Vec<char> = token.chars().collect();
    if chars.len() > 8 {
        let head: String = chars[..4].iter().collect();
        let tail: String = chars[chars.len() - 4..].iter().collect();
        format!("{head}...{tail}")
    } else {
        "***".to_string()
    }
}

/// Log token receipt with partially redacted values for debugging.
pub fn log_token_received(method: &str, access_token: &str, refresh_token: Option<&str>) {
    let at = redact(access_token);
    let rt = match refresh_token {
        Some(rt) if !rt.is_empty() => redact(rt),
        _ => "none".to_string(),
    };
    tracing::info!(
        target: LOGGER,
        "OAuth tokens stored via /v1/auth/callback ({}) access={} refresh={}",
        method,
        at,
        rt
    );
}

/// Safari fallback: store tokens from query params and return the redirect
/// target (`{APP_URL}/redirect/?mode=desktop`).
///
/// Safari blocks mixed-content fetch (HTTPS→HTTP) so the webapp falls back to
/// a full-page navigation with tokens as query params.
pub fn oauth_callback_get(access_token: &str, refresh_token: Option<&str>) -> String {
    credential_store::set_tokens(access_token, refresh_token);

    log_token_received("GET", access_token, refresh_token);
    refresh_platform_cache(true);
    format!("{}/redirect/?mode=desktop", *APP_URL)
}

/// `GET /v1/auth/status`.
#[utoipa::path(
    get,
    path = "/v1/auth/status",
    tag = "auth",
    summary = "Get sign-in status",
    description = "",
    responses((status = 200, description = "OK", body = AuthStatusResponse))
)]
async fn auth_status_handler() -> Result<Json<AuthStatusResponse>, HTTPException> {
    run_blocking(auth_status).await.map(Json)
}

/// `POST /v1/auth/token`.
#[utoipa::path(
    post,
    path = "/v1/auth/token",
    tag = "auth",
    request_body = TokenSetRequest,
    summary = "Store a token",
    description = "",
    responses((status = 200, description = "OK", body = AuthStatusResponse), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
async fn set_token_handler(
    ValidatedJson(request): ValidatedJson<TokenSetRequest>,
) -> Result<Json<AuthStatusResponse>, HTTPException> {
    request.validate()?;
    run_blocking(move || set_token(request)).await.map(Json)
}

/// `DELETE /v1/auth/token`.
#[utoipa::path(
    delete,
    path = "/v1/auth/token",
    tag = "auth",
    params(ClearTokenQuery),
    summary = "Sign out",
    description = "Clears all credentials, or only the developer or access token.",
    responses((status = 200, description = "OK", body = AuthStatusResponse), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
async fn clear_token_handler(
    ValidatedQuery(query): ValidatedQuery<ClearTokenQuery>,
) -> Result<Json<AuthStatusResponse>, HTTPException> {
    run_blocking(move || clear_token(query.type_)).await.map(Json)
}

/// `POST /v1/auth/callback`.
#[utoipa::path(
    post,
    path = "/v1/auth/callback",
    tag = "auth",
    request_body = OAuthCallbackRequest,
    summary = "Complete OAuth sign-in",
    description = "",
    responses((status = 200, description = "OK", body = AuthStatusResponse), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
async fn oauth_callback_handler(
    ValidatedJson(request): ValidatedJson<OAuthCallbackRequest>,
) -> Result<Json<AuthStatusResponse>, HTTPException> {
    run_blocking(move || oauth_callback(request)).await.map(Json)
}

/// `GET /v1/auth/callback` — 307 redirect to the webapp success page
/// (FastAPI's `RedirectResponse` default status).
#[utoipa::path(
    get,
    path = "/v1/auth/callback",
    tag = "auth",
    params(OAuthCallbackQuery),
    summary = "OAuth redirect target",
    description = "Stores the token and redirects to the sign-in success page.",
    responses((status = 307, description = "Redirect to the sign-in success page"), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
async fn oauth_callback_get_handler(
    ValidatedQuery(query): ValidatedQuery<OAuthCallbackQuery>,
) -> Result<Response, HTTPException> {
    let location = run_blocking(move || {
        oauth_callback_get(&query.access_token, query.refresh_token.as_deref())
    })
    .await?;
    Ok((StatusCode::TEMPORARY_REDIRECT, [(LOCATION, location)]).into_response())
}
