//! Payments router – Mac App Store purchases, relayed to the platform API.
//!
//! The desktop app keeps no cloud token (it hands them to this service), so a
//! purchase it completes with StoreKit is verified through here.
//!
//! Builds outside the Mac App Store upgrade through Stripe Checkout instead.
//!
//! Endpoints:
//!     POST /v1/payments/apple/verify – link an App Store transaction to the account
//!     POST /v1/payments/checkout     – start a Stripe Checkout session
//!     POST /v1/payments/refresh      – re-read the plan from the cloud (poll after checkout)

use axum::routing::post;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::platform::client::JsonObject;
use crate::platform::{get_platform_service, PlatformError, PlatformService};
use crate::search::providers;

use super::users::{user_read, UserRead};
use super::_errors::{run_blocking, unhandled_error, HTTPException, ValidatedJson};

/// Logger name used by this router.
pub const LOGGER: &str = "smartloop.api.routers.payments";

/// Body of `POST /v1/payments/apple/verify`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct AppleVerifyRequest {
    /// The StoreKit transaction id (`Transaction.id` / `originalID`).
    pub transaction_id: String,
}

/// The platform's answer, for the OpenAPI document.  The handler relays the
/// platform's body as-is, so a field it adds later reaches the app too.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, utoipa::ToSchema)]
pub struct AppleVerifyResponse {
    pub is_paid: bool,
    pub subscription_status: String,
    /// ISO 8601, or null.
    pub current_period_end: Option<String>,
}

/// The platform's Stripe Checkout answer, for the OpenAPI document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, utoipa::ToSchema)]
pub struct CheckoutResponse {
    pub checkout_url: String,
}

/// Routes under `/v1/payments` (tag `auth`).
pub fn router() -> Router {
    Router::new()
        .route("/v1/payments/apple/verify", post(verify_apple_handler))
        .route("/v1/payments/checkout", post(checkout_handler))
        .route("/v1/payments/refresh", post(refresh_handler))
}

/// Map a payment failure to the status the desktop app acts on.
///
/// A 401 (or a permission error with no status: no credentials at all) is
/// "sign in".  Any other refusal keeps the platform's status and its `detail`,
/// which the app shows as written: 400 invalid transaction / Stripe error,
/// 404 Stripe disabled (the app hides the option), 409 linked to another
/// account, 503 Apple IAP not configured, 429 rate limited.  A
/// transport failure or any other 5xx is 502 "Platform API unavailable", as
/// the users and web search routes report it.
fn payment_failure(exc: &PlatformError) -> HTTPException {
    match (exc.is_permission(), exc.is_connection(), exc.status_code()) {
        (true, _, None | Some(401)) => HTTPException::new(401, "Not authenticated"),
        (true, _, Some(status)) => HTTPException::new(status, exc.message()),
        (_, true, Some(status @ (429 | 503))) => HTTPException::new(status, exc.message()),
        (_, true, _) => HTTPException::new(502, "Platform API unavailable"),
        _ => {
            tracing::error!(target: LOGGER, "Unhandled platform error: {}", exc);
            unhandled_error()
        }
    }
}

/// Refresh what a plan change invalidates, so Pro unlocks without a restart.
///
/// The cached user profile carries `is_paid` (5 minute TTL), the cached
/// catalog carries each model's per-user `is_accessible` (15 minutes), and
/// the hosted search status is cached for a minute.  Best-effort: the
/// purchase itself succeeded, and a failed refresh only means the old answer
/// lives out its TTL.
pub fn refresh_entitlements(svc: &PlatformService) {
    providers::reset_status_cache();
    if let Err(e) = svc.get_current_user(true) {
        tracing::warn!(target: LOGGER, "Failed to refresh cached user after purchase: {}", e);
    }
    if let Err(e) = svc.list_models(true) {
        tracing::warn!(target: LOGGER, "Failed to refresh cached model catalog after purchase: {}", e);
    }
}

/// Verify an App Store transaction with the platform.
///
/// Sync on purpose, like `get_me` in the users router: a 401 costs a token
/// refresh and a retry, which must not stall the async runtime.  422 when
/// `transaction_id` is blank — there is nothing to ask the platform.
pub fn verify_apple(body: AppleVerifyRequest) -> Result<JsonObject, HTTPException> {
    let transaction_id = body.transaction_id.trim();
    if transaction_id.is_empty() {
        return Err(HTTPException::new(422, "transaction_id is required"));
    }

    let svc = get_platform_service();
    let data = svc
        .client()
        .verify_apple_transaction(transaction_id)
        .map_err(|e| payment_failure(&e))?;
    refresh_entitlements(&svc);
    Ok(data)
}

/// `POST /v1/payments/apple/verify`.
#[utoipa::path(
    post,
    path = "/v1/payments/apple/verify",
    tag = "auth",
    request_body = AppleVerifyRequest,
    summary = "Verify a Mac App Store purchase",
    description = "Relays the transaction to the platform, which verifies it with Apple and links the subscription to the signed-in account. On success the cached user profile and model catalog are refreshed so Pro models unlock immediately.",
    responses(
        (status = 200, description = "OK", body = AppleVerifyResponse),
        (status = 400, description = "Invalid transaction (platform `detail` preserved)"),
        (status = 401, description = "Not signed in"),
        (status = 409, description = "Transaction already linked to another account (platform `detail` preserved)"),
        (status = 422, description = "Missing transaction_id"),
        (status = 502, description = "Platform API unavailable"),
        (status = 503, description = "Apple IAP not configured on the platform (platform `detail` preserved)"),
        (status = 500, description = "Server error")
    )
)]
pub async fn verify_apple_handler(
    ValidatedJson(body): ValidatedJson<AppleVerifyRequest>,
) -> Result<Json<JsonObject>, HTTPException> {
    run_blocking(move || verify_apple(body)).await?.map(Json)
}

/// Start a Stripe Checkout session for the Pro upgrade.
///
/// Sync on purpose, as [`verify_apple`].  Nothing is refreshed here: the plan
/// changes when the user completes checkout in the browser, and the cached
/// user/catalog pick that up on their TTL or the next sign-in refresh.
pub fn checkout() -> Result<JsonObject, HTTPException> {
    get_platform_service()
        .client()
        .create_checkout_session()
        .map_err(|e| payment_failure(&e))
}

/// `POST /v1/payments/checkout`.
#[utoipa::path(
    post,
    path = "/v1/payments/checkout",
    tag = "auth",
    summary = "Start a Stripe Checkout for the Pro upgrade",
    description = "Relays the platform's `POST /v1/payments/create-checkout-session` and returns its `checkout_url`. Not for Mac App Store builds, which use `/v1/payments/apple/verify`.",
    responses(
        (status = 200, description = "OK", body = CheckoutResponse),
        (status = 400, description = "Stripe error (platform `detail` preserved)"),
        (status = 401, description = "Not signed in"),
        (status = 404, description = "Stripe checkout is disabled on the platform"),
        (status = 502, description = "Platform API unavailable"),
        (status = 500, description = "Server error")
    )
)]
pub async fn checkout_handler() -> Result<Json<JsonObject>, HTTPException> {
    run_blocking(checkout).await?.map(Json)
}

/// `is_paid` of a user payload; absent or null counts as not paid, as
/// [`user_read`] defaults it.
fn is_paid(user: &JsonObject) -> bool {
    matches!(user.get("is_paid"), Some(Value::Bool(true)))
}

/// Re-read the user's plan from the cloud and return the fresh profile.
///
/// The desktop polls this after sending the user to Stripe Checkout, so
/// unlike [`refresh_entitlements`] it is not best-effort and it never serves
/// the cache: the profile comes from the cloud or the call fails (401 not
/// signed in / token rejected, 502 unreachable — as `GET /v1/users/me`).
/// The cache is written with what came back.
///
/// Kept cheap for polling: the model catalog and the hosted search status
/// are only refreshed when `is_paid` differs from the cached profile (or
/// there was none to compare with).
pub fn refresh() -> Result<UserRead, HTTPException> {
    let svc = get_platform_service();
    let cached_paid = svc.db().and_then(|db| db.get_user().ok().flatten()).map(|u| is_paid(&u));

    let user = svc.client().get_current_user().map_err(|e| {
        if e.is_permission() {
            HTTPException::new(401, "Not authenticated")
        } else if e.is_connection() {
            HTTPException::new(502, "Platform API unavailable")
        } else {
            tracing::error!(target: LOGGER, "Unhandled platform error: {}", e);
            unhandled_error()
        }
    })?;
    let profile = user_read(&user).map_err(|e| {
        tracing::error!(target: LOGGER, "Invalid user payload from the platform API: {}", e);
        unhandled_error()
    })?;

    if let Some(db) = svc.db() {
        if let Err(e) = db.save_user(&user) {
            tracing::warn!(target: LOGGER, "Failed to cache refreshed user: {:#}", e);
        }
    }

    if cached_paid != Some(is_paid(&user)) {
        providers::reset_status_cache();
        if let Err(e) = svc.list_models(true) {
            tracing::warn!(target: LOGGER, "Failed to refresh cached model catalog after plan change: {}", e);
        }
    }
    Ok(profile)
}

/// `POST /v1/payments/refresh`.
#[utoipa::path(
    post,
    path = "/v1/payments/refresh",
    tag = "auth",
    summary = "Re-read the account's plan from the platform",
    description = "Fetches the user profile from the platform (never the cache), stores it, and returns it in the `GET /v1/users/me` shape. When `is_paid` changed, the model catalog and web search status are refreshed too. Poll after Stripe Checkout to see the upgrade land.",
    responses(
        (status = 200, description = "OK", body = UserRead),
        (status = 401, description = "Not signed in"),
        (status = 502, description = "Platform API unavailable"),
        (status = 500, description = "Server error")
    )
)]
pub async fn refresh_handler() -> Result<Json<UserRead>, HTTPException> {
    run_blocking(refresh).await?.map(Json)
}
