//! Port of `smartloop/api/routers/users.py`.
//!
//! Users router – user profile from the Smartloop platform API.
//!
//! Endpoints:
//!     GET  /v1/users/me – fetch authenticated user's full profile
//!     PATCH /v1/users/me – update mutable profile fields (default_temperature, default_memory)
//!     DELETE /v1/users/me – permanently delete the account and sign out

use axum::http::StatusCode;
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::platform::client::JsonObject;
use crate::platform::{get_platform_service, PlatformError};

use super::_errors::{run_blocking, unhandled_error, HTTPException, ValidatedJson};
use super::auth::{clear_token, ClearScope};

/// Logger name used by this router.
pub const LOGGER: &str = "smartloop.api.routers.users";

/// User profile returned by the platform API.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, utoipa::ToSchema)]
pub struct UserRead {
    pub email: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default = "some_false")]
    pub is_premium: Option<bool>,
    #[serde(default = "some_false")]
    pub is_paid: Option<bool>,
    #[serde(default = "some_zero")]
    pub number_of_tokens_processed: Option<i64>,
    #[serde(default = "default_temperature")]
    pub default_temperature: f64,
    #[serde(default = "default_true")]
    pub default_memory: bool,
}

fn some_false() -> Option<bool> {
    Some(false)
}

fn some_zero() -> Option<i64> {
    Some(0)
}

fn default_temperature() -> f64 {
    0.3
}

fn default_true() -> bool {
    true
}

impl Default for UserRead {
    fn default() -> Self {
        Self {
            email: String::new(),
            name: None,
            is_premium: Some(false),
            is_paid: Some(false),
            number_of_tokens_processed: Some(0),
            default_temperature: 0.3,
            default_memory: true,
        }
    }
}

/// Mutable user profile fields (serialized with `exclude_none`).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct UserUpdate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_memory: Option<bool>,
}

/// Routes under `/v1/users` (tag `users`).
pub fn router() -> Router {
    Router::new().route(
        "/v1/users/me",
        get(get_me_handler).patch(update_me_handler).delete(delete_me_handler),
    )
}

/// `data.get(key, default)` validated as `Optional[T]`: an absent key takes
/// the default, an explicit null stays null.
fn optional_field<T>(data: &JsonObject, key: &str, default: Option<T>) -> Result<Option<T>, String>
where
    T: serde::de::DeserializeOwned,
{
    match data.get(key) {
        None => Ok(default),
        Some(Value::Null) => Ok(None),
        Some(value) => serde_json::from_value(value.clone()).map(Some).map_err(|e| format!("{key}: {e}")),
    }
}

/// `data.get(key, default)` validated as a required (non-null) `T`.
fn required_field<T>(data: &JsonObject, key: &str, default: T) -> Result<T, String>
where
    T: serde::de::DeserializeOwned,
{
    match data.get(key) {
        None => Ok(default),
        Some(value) => serde_json::from_value(value.clone()).map_err(|e| format!("{key}: {e}")),
    }
}

/// Build a [`UserRead`] from the platform's user payload, applying defaults.
///
/// `data["email"]` is required; every other field falls back to its default
/// when absent.  A payload that does not validate (no email, a field of the
/// wrong type) is an error, as the `KeyError` / pydantic `ValidationError`
/// raised by `_user_read` was.
pub fn user_read(data: &JsonObject) -> Result<UserRead, String> {
    let email = match data.get("email") {
        Some(Value::String(email)) => email.clone(),
        Some(other) => return Err(format!("email: Input should be a valid string, got {other}")),
        None => return Err("'email'".to_string()),
    };
    Ok(UserRead {
        email,
        name: optional_field(data, "name", None)?,
        is_premium: optional_field(data, "is_premium", Some(false))?,
        is_paid: optional_field(data, "is_paid", Some(false))?,
        number_of_tokens_processed: optional_field(data, "number_of_tokens_processed", Some(0))?,
        default_temperature: required_field(data, "default_temperature", 0.3)?,
        default_memory: required_field(data, "default_memory", true)?,
    })
}

/// Map a platform failure the way both endpoints do: a permission error is
/// 401 "Not authenticated", a connection error is 502 "Platform API
/// unavailable", and anything else escapes the handler as a bare 500.
fn platform_failure(exc: &PlatformError) -> HTTPException {
    if exc.is_permission() {
        HTTPException::new(401, "Not authenticated")
    } else if exc.is_connection() {
        HTTPException::new(502, "Platform API unavailable")
    } else {
        tracing::error!(target: LOGGER, "Unhandled platform error: {}", exc);
        unhandled_error()
    }
}

/// [`user_read`] for a response: a payload that does not validate escapes the
/// handler, which FastAPI answers with a bare 500.
fn user_read_or_500(data: &JsonObject) -> Result<UserRead, HTTPException> {
    user_read(data).map_err(|e| {
        tracing::error!(target: LOGGER, "Invalid user payload from the platform API: {}", e);
        unhandled_error()
    })
}

/// Return the authenticated user's profile from the platform API.
///
/// Sync on purpose: the platform client is synchronous, and a 401 now costs a
/// refresh and a retry on top of the original call.  On the async runtime
/// that would stall every in-flight chat stream, so the handler runs this on
/// the blocking pool.  See `auth_status` in the auth router.
pub fn get_me() -> Result<UserRead, HTTPException> {
    let svc = get_platform_service();
    let data = svc.get_current_user(false).map_err(|e| platform_failure(&e))?;
    user_read_or_500(&data)
}

/// Update the authenticated user's profile via the platform API.
///
/// Sync on purpose, for the same reason as [`get_me`].  422 `"No fields to
/// update"` when the body sets nothing.
pub fn update_me(body: UserUpdate) -> Result<UserRead, HTTPException> {
    let payload = match serde_json::to_value(&body) {
        Ok(Value::Object(map)) => map,
        _ => JsonObject::new(),
    };
    if payload.is_empty() {
        return Err(HTTPException::new(422, "No fields to update"));
    }

    let svc = get_platform_service();
    let data = svc.update_current_user(&payload).map_err(|e| platform_failure(&e))?;
    user_read_or_500(&data)
}

/// Permanently delete the signed-in account via the platform API, then drop
/// the credentials that authorised it.
///
/// Sync on purpose, for the same reason as [`get_me`].  Clearing the stored
/// tokens is what signs the app out: they belong to an account that no longer
/// exists, so keeping them would leave every later platform call failing 401
/// until the person noticed.  The cloud call comes first — a network failure
/// must not sign someone out of an account that is still there.
///
/// Required by App Review guideline 5.1.1(v): an app that offers account
/// creation has to offer account deletion in-app.
pub fn delete_me() -> Result<(), HTTPException> {
    let svc = get_platform_service();
    svc.delete_current_user().map_err(|e| platform_failure(&e))?;
    tracing::info!(target: LOGGER, "Account deleted on the platform; clearing stored credentials");
    clear_token(ClearScope::All);
    Ok(())
}

/// `GET /v1/users/me`.
#[utoipa::path(
    get,
    path = "/v1/users/me",
    tag = "auth",
    summary = "Get the current user",
    description = "",
    responses((status = 200, description = "OK", body = UserRead), (status = 401, description = "Not signed in"), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
async fn get_me_handler() -> Result<Json<UserRead>, HTTPException> {
    run_blocking(get_me).await?.map(Json)
}

/// `PATCH /v1/users/me`.
#[utoipa::path(
    patch,
    path = "/v1/users/me",
    tag = "auth",
    request_body = UserUpdate,
    summary = "Update the current user's preferences",
    description = "",
    responses((status = 200, description = "OK", body = UserRead), (status = 401, description = "Not signed in"), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
async fn update_me_handler(
    ValidatedJson(body): ValidatedJson<UserUpdate>,
) -> Result<Json<UserRead>, HTTPException> {
    run_blocking(move || update_me(body)).await?.map(Json)
}

/// `DELETE /v1/users/me`.
#[utoipa::path(
    delete,
    path = "/v1/users/me",
    tag = "auth",
    summary = "Delete the current user's account",
    description = "Permanently deletes the signed-in account on the platform, then clears the stored credentials so the app is signed out.",
    responses((status = 204, description = "Deleted"), (status = 401, description = "Not signed in"), (status = 500, description = "Server error"))
)]
async fn delete_me_handler() -> Result<StatusCode, HTTPException> {
    run_blocking(delete_me).await??;
    Ok(StatusCode::NO_CONTENT)
}
