//! Port of `smartloop/api/routers/_errors.py`.
//!
//! Turning a framework exception back into the HTTP status it stands for.
//!
//! Shared because the alternative is each router deciding for itself, and the
//! decisions drifted: `PermissionError` means four different things depending on
//! who raised it, and a router that reads it as one of them answers "sign in" to a
//! full disk.
//!
//! Also home to the pieces of FastAPI's request handling the routers rely on:
//!
//! * [`HTTPException`], the Rust stand-in for FastAPI's `HTTPException`: it
//!   renders as `{"detail": ...}` with the given status.
//! * [`ValidatedJson`] / [`ValidatedQuery`], extractors that answer a body or
//!   query string that does not fit its model with FastAPI's `422`
//!   `{"detail": [{"type", "loc", "msg", "input"}]}` shape instead of axum's
//!   plain-text 400/415/422.
//! * [`read_form`], multipart parsing with FastAPI's "missing field is a 422"
//!   behaviour.
//! * [`py_error_kind`], the Python exception class an `anyhow::Error` from a
//!   service stands for, so `except ValueError` / `except RuntimeError` branches
//!   port one-to-one.

use std::error::Error as StdError;
use std::io;

use axum::body::Bytes;
use axum::extract::multipart::MultipartRejection;
use axum::extract::{FromRequest, FromRequestParts, Multipart, Request};
use axum::http::header::CONTENT_TYPE;
use axum::http::request::Parts;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer};
use serde_json::{json, Value};

use crate::api::services::model_service::ModelServiceError;
use crate::api::services::project_io::ProjectIoError;
use crate::api::services::storage_service::StorageError;
use crate::model::downloader::artifact_downloader::{DownloadError, StaleDownloadURLError};
use crate::model::downloader::cancellation::{DownloadCancelled, DownloadInProgress};
use crate::platform::{ModelAccessError, PlatformError};

/// `errno.ENOSPC` — no space left on device.
pub const ENOSPC: i32 = libc::ENOSPC;

/// FastAPI's `HTTPException`: an error answered as `{"detail": <detail>}` with
/// `status_code`.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{status_code}: {detail}")]
pub struct HTTPException {
    /// HTTP status to answer with.
    pub status_code: u16,
    /// The `detail` payload (usually a string).
    pub detail: Value,
}

impl HTTPException {
    /// An exception whose detail is the string *detail*.
    pub fn new(status_code: u16, detail: impl Into<String>) -> Self {
        Self { status_code, detail: Value::String(detail.into()) }
    }

    /// An exception with an arbitrary JSON detail.
    pub fn with_detail(status_code: u16, detail: Value) -> Self {
        Self { status_code, detail }
    }
}

impl IntoResponse for HTTPException {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.status_code)
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, axum::Json(json!({ "detail": self.detail }))).into_response()
    }
}

// ---------------------------------------------------------------------------
// status_for
// ---------------------------------------------------------------------------

/// The `status_code` an error carries on itself (Python's
/// `getattr(exc, "status_code", None)`), seeing through the transparent
/// wrappers Rust adds where Python would have raised the inner error as is.
fn own_status(exc: &(dyn StdError + 'static)) -> Option<u16> {
    if let Some(e) = exc.downcast_ref::<HTTPException>() {
        return Some(e.status_code);
    }
    if let Some(e) = exc.downcast_ref::<PlatformError>() {
        return e.status_code();
    }
    if let Some(e) = exc.downcast_ref::<ModelAccessError>() {
        return Some(e.status_code);
    }
    if let Some(e) = exc.downcast_ref::<DownloadCancelled>() {
        return Some(e.status_code());
    }
    if let Some(e) = exc.downcast_ref::<DownloadInProgress>() {
        return Some(e.status_code());
    }
    if let Some(e) = exc.downcast_ref::<DownloadError>() {
        return match e {
            DownloadError::Cancelled(c) => Some(c.status_code()),
            DownloadError::Source(inner) => own_status(inner.as_ref()),
            _ => None,
        };
    }
    None
}

/// `isinstance(exc, ModelAccessError)`.
fn is_model_access(exc: &(dyn StdError + 'static)) -> bool {
    if exc.downcast_ref::<ModelAccessError>().is_some() {
        return true;
    }
    if let Some(PlatformError::ModelAccess(_)) = exc.downcast_ref::<PlatformError>() {
        return true;
    }
    if let Some(DownloadError::Source(inner)) = exc.downcast_ref::<DownloadError>() {
        return is_model_access(inner.as_ref());
    }
    false
}

/// Python's `ConnectionError` family, on the error itself.
fn is_connection_error(exc: &(dyn StdError + 'static)) -> bool {
    if let Some(e) = exc.downcast_ref::<PlatformError>() {
        return e.is_connection();
    }
    if exc.downcast_ref::<StaleDownloadURLError>().is_some() {
        return true;
    }
    if let Some(e) = exc.downcast_ref::<DownloadError>() {
        return match e {
            DownloadError::Connection { .. } | DownloadError::StaleUrl(_) => true,
            DownloadError::Io(io_err) => is_io_connection_error(io_err),
            DownloadError::Source(inner) => is_connection_error(inner.as_ref()),
            _ => false,
        };
    }
    if let Some(io_err) = exc.downcast_ref::<io::Error>() {
        return is_io_connection_error(io_err);
    }
    false
}

/// The `OSError` subclasses Python files under `ConnectionError`.
fn is_io_connection_error(err: &io::Error) -> bool {
    matches!(
        err.kind(),
        io::ErrorKind::ConnectionRefused
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::BrokenPipe
    )
}

/// An `OSError` whose errno is `ENOSPC`.
fn is_enospc(err: &io::Error) -> bool {
    err.raw_os_error() == Some(ENOSPC) || err.kind() == io::ErrorKind::StorageFull
}

/// Whether this one link of a chain is itself an out-of-space error, looking
/// inside the transparent wrappers that hide their payload from `source()`.
fn link_is_disk_full(exc: &(dyn StdError + 'static)) -> bool {
    if let Some(io_err) = exc.downcast_ref::<io::Error>() {
        if is_enospc(io_err) {
            return true;
        }
    }
    if let Some(e) = exc.downcast_ref::<DownloadError>() {
        let wrapped_full = match e {
            DownloadError::Io(io_err) => is_enospc(io_err),
            DownloadError::Source(inner) => inner.chain().any(link_is_disk_full),
            _ => false,
        };
        if wrapped_full {
            return true;
        }
    }
    if let Some(ProjectIoError::Other(inner)) = exc.downcast_ref::<ProjectIoError>() {
        if inner.chain().any(link_is_disk_full) {
            return true;
        }
    }
    false
}

/// True when *exc* — or the failure it wraps — is an out-of-space OSError.
///
/// The download path wraps a raw ENOSPC in a classified `ConnectionError`
/// (keeping the cause as its source), so the errno is not always on the error
/// itself — follow the chain before declaring it a network failure.
pub fn is_disk_full(exc: &(dyn StdError + 'static)) -> bool {
    let mut current: Option<&(dyn StdError + 'static)> = Some(exc);
    while let Some(e) = current {
        if link_is_disk_full(e) {
            return true;
        }
        current = e.source();
    }
    false
}

/// Classify *exc* as the HTTP status it would have been.
///
/// Platform errors carry the upstream status and are answered with it, so the
/// client never has to match on message text — that is written for a person to
/// read, and it gets reworded.
///
/// Only [`ModelAccessError`] earns a 403.  `PermissionError` used to, which was
/// too wide by half: it is an `OSError` subclass, so an unwritable models
/// directory arrived as "not on your plan" and the desktop client turned that
/// into a sign-in prompt for what is a local disk problem.
pub fn status_for(exc: &(dyn StdError + 'static)) -> u16 {
    if let Some(status) = own_status(exc) {
        return status;
    }
    if is_model_access(exc) {
        return 403;
    }
    if is_disk_full(exc) {
        // The most retryable failure there is (free some space and try again),
        // and not a network problem — so neither 500 nor 502 says it right.
        // 507 is how the client tells "refill the disk" from "it broke".
        return 507;
    }
    // Transfers that died on a dropped socket, a stale signed URL or an upstream
    // 5xx are aggregated into a ConnectionError by `aggregate_error`.
    if is_connection_error(exc) {
        return 502;
    }
    500
}

/// [`status_for`] for an `anyhow::Error`.
pub fn status_for_anyhow(exc: &anyhow::Error) -> u16 {
    let inner: &(dyn StdError + Send + Sync + 'static) = exc.as_ref();
    status_for(inner)
}

// ---------------------------------------------------------------------------
// Python exception classes of service errors
// ---------------------------------------------------------------------------

/// Which Python exception class a service error stands for — what the
/// routers' `except ValueError` / `except RuntimeError` / `except
/// FileNotFoundError` branches match on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyErrorKind {
    /// `ValueError` (bad input, unknown id).
    Value,
    /// `RuntimeError` (not ready, busy).
    Runtime,
    /// `FileNotFoundError`.
    FileNotFound,
    /// `ConnectionError`.
    Connection,
    /// `PermissionError`.
    Permission,
    /// Anything else.
    Other,
}

/// Classify a service error by the Python exception class it ports.
pub fn py_error_kind(err: &anyhow::Error) -> PyErrorKind {
    if err.downcast_ref::<HTTPException>().is_some() {
        return PyErrorKind::Other;
    }
    if let Some(e) = err.downcast_ref::<ModelServiceError>() {
        return match e {
            ModelServiceError::Value(_) => PyErrorKind::Value,
            ModelServiceError::Runtime(_) => PyErrorKind::Runtime,
        };
    }
    if let Some(e) = err.downcast_ref::<StorageError>() {
        return match e {
            StorageError::ValueError(_) => PyErrorKind::Value,
            StorageError::RuntimeError(_) => PyErrorKind::Runtime,
        };
    }
    if let Some(e) = err.downcast_ref::<ProjectIoError>() {
        return match e {
            ProjectIoError::ValueError(_) => PyErrorKind::Value,
            ProjectIoError::FileNotFoundError(_) => PyErrorKind::FileNotFound,
            ProjectIoError::Other(inner) => py_error_kind(inner),
        };
    }
    if let Some(e) = err.downcast_ref::<PlatformError>() {
        return match e {
            PlatformError::Value(_) => PyErrorKind::Value,
            PlatformError::Connection { .. } => PyErrorKind::Connection,
            PlatformError::Permission { .. } | PlatformError::ModelAccess(_) => PyErrorKind::Permission,
            PlatformError::Other(_) => PyErrorKind::Other,
        };
    }
    if err.downcast_ref::<ModelAccessError>().is_some() {
        return PyErrorKind::Permission;
    }
    if err.downcast_ref::<DownloadInProgress>().is_some() {
        // A RuntimeError subclass in Python.
        return PyErrorKind::Runtime;
    }
    if let Some(e) = err.downcast_ref::<DownloadError>() {
        return match e {
            DownloadError::Value(_) => PyErrorKind::Value,
            DownloadError::Runtime { .. } => PyErrorKind::Runtime,
            DownloadError::Connection { .. } | DownloadError::StaleUrl(_) => PyErrorKind::Connection,
            DownloadError::Io(io_err) => io_kind(io_err),
            DownloadError::Source(inner) => py_error_kind(inner),
            _ => PyErrorKind::Other,
        };
    }
    if let Some(io_err) = err.downcast_ref::<io::Error>() {
        return io_kind(io_err);
    }
    if err.downcast_ref::<serde_json::Error>().is_some() {
        // json.JSONDecodeError is a ValueError subclass.
        return PyErrorKind::Value;
    }
    PyErrorKind::Other
}

fn io_kind(err: &io::Error) -> PyErrorKind {
    if is_io_connection_error(err) {
        PyErrorKind::Connection
    } else if err.kind() == io::ErrorKind::NotFound {
        PyErrorKind::FileNotFound
    } else if err.kind() == io::ErrorKind::PermissionDenied {
        PyErrorKind::Permission
    } else {
        PyErrorKind::Other
    }
}

/// `HTTPException(status_code=500, detail=str(e))` — the routers' catch-all.
pub fn internal_error(err: &anyhow::Error) -> HTTPException {
    HTTPException::new(500, err.to_string())
}

// ---------------------------------------------------------------------------
// Running blocking work (FastAPI's threadpool / `run_in_executor`)
// ---------------------------------------------------------------------------

/// The text of a panic payload — what `str(e)` would have been had the
/// failure been a Python exception.
pub fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "Internal Server Error".to_string()
    }
}

/// Run *f* on the blocking pool and wait for it.
///
/// The Rust counterpart of a sync FastAPI handler (run in the threadpool) and
/// of `loop.run_in_executor(None, f)`.  A panic inside *f* is the analogue of
/// an exception escaping the call, and is answered as a 500 carrying its
/// message — what the routers' `except Exception as e` branches do.
pub async fn run_blocking<F, T>(f: F) -> Result<T, HTTPException>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(value) => Ok(value),
        Err(join_error) => {
            let message = if join_error.is_panic() {
                panic_message(join_error.into_panic().as_ref())
            } else {
                join_error.to_string()
            };
            Err(HTTPException::new(500, message))
        }
    }
}

/// [`run_blocking`] on a bounded executor: *gate* holds as many permits as
/// the Python `ThreadPoolExecutor(max_workers=N)` had workers, so at most N
/// calls submitted through it run at once and the rest queue.
pub async fn run_on<F, T>(gate: &tokio::sync::Semaphore, f: F) -> Result<T, HTTPException>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let _permit = gate
        .acquire()
        .await
        .map_err(|e| HTTPException::new(500, e.to_string()))?;
    run_blocking(f).await
}

/// `Model(**result)`: validate a service result into a response model.  A
/// result that does not fit is a 500, as the pydantic error raised inside the
/// handler's `try` would have been.
pub fn parse_model<T: DeserializeOwned>(value: Value) -> Result<T, HTTPException> {
    serde_json::from_value(value).map_err(|e| HTTPException::new(500, e.to_string()))
}

/// FastAPI's answer to an exception no handler caught: a bare 500.
pub fn unhandled_error() -> HTTPException {
    HTTPException::new(500, "Internal Server Error")
}

/// Python's `repr()` of a string (`'abc'`, or `"it's"` when the text holds a
/// single quote and no double quote), as used in f-string `{x!r}` messages.
pub fn py_repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

// ---------------------------------------------------------------------------
// FastAPI-shaped request validation
// ---------------------------------------------------------------------------

/// One entry of FastAPI's 422 `detail` list.
fn validation_entry(kind: &str, loc: Vec<Value>, msg: &str, input: Value) -> Value {
    json!({ "type": kind, "loc": loc, "msg": msg, "input": input })
}

/// A FastAPI `RequestValidationError` response: `422 {"detail": [...]}`.
pub fn validation_error(errors: Vec<Value>) -> HTTPException {
    HTTPException::with_detail(422, Value::Array(errors))
}

/// The field a serde "missing field `x`" message names.
fn missing_field(message: &str) -> Option<String> {
    let rest = message.split("missing field `").nth(1)?;
    let end = rest.find('`')?;
    Some(rest[..end].to_string())
}

/// Translate a serde deserialization failure into FastAPI's 422 entries,
/// located under `source` (`"body"` or `"query"`).
fn serde_validation_errors(source: &str, message: &str, input: Value) -> Vec<Value> {
    if let Some(field) = missing_field(message) {
        return vec![validation_entry(
            "missing",
            vec![json!(source), json!(field)],
            "Field required",
            input,
        )];
    }
    vec![validation_entry("value_error", vec![json!(source)], message, input)]
}

/// Body extractor with FastAPI semantics.
///
/// The body is read as JSON when there is no `Content-Type` or it names JSON
/// (`application/json`, `application/*+json`); an empty body is "missing",
/// undecodable JSON is `json_invalid`, and a body that does not fit `T` is a
/// 422 naming the offending field — never axum's plain-text rejection.
#[derive(Debug, Clone, Copy, Default)]
pub struct ValidatedJson<T>(pub T);

/// Whether a `Content-Type` value is one FastAPI decodes as JSON.
fn is_json_content_type(value: &str) -> bool {
    let mime = value.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
    match mime.split_once('/') {
        Some(("application", subtype)) => subtype == "json" || subtype.ends_with("+json"),
        _ => false,
    }
}

/// Decode a request body the way FastAPI does before validating it.
pub fn decode_body<T: DeserializeOwned>(content_type: Option<&str>, bytes: &[u8]) -> Result<T, HTTPException> {
    if bytes.is_empty() {
        return Err(validation_error(vec![validation_entry(
            "missing",
            vec![json!("body")],
            "Field required",
            Value::Null,
        )]));
    }
    let as_json = match content_type {
        None => true,
        Some(ct) => is_json_content_type(ct),
    };
    if !as_json {
        // FastAPI hands the raw bytes to the model, which cannot take them.
        return Err(validation_error(vec![validation_entry(
            "model_attributes_type",
            vec![json!("body")],
            "Input should be a valid dictionary or object to extract fields from",
            Value::String(String::from_utf8_lossy(bytes).into_owned()),
        )]));
    }
    let value: Value = match serde_json::from_slice(bytes) {
        Ok(v) => v,
        Err(e) => {
            return Err(validation_error(vec![json!({
                "type": "json_invalid",
                "loc": ["body", e.column()],
                "msg": "JSON decode error",
                "input": {},
                "ctx": {"error": e.to_string()},
            })]));
        }
    };
    if !value.is_object() {
        return Err(validation_error(vec![validation_entry(
            "model_attributes_type",
            vec![json!("body")],
            "Input should be a valid dictionary or object to extract fields from",
            value,
        )]));
    }
    serde_json::from_value::<T>(value.clone())
        .map_err(|e| validation_error(serde_validation_errors("body", &e.to_string(), value)))
}

impl<S, T> FromRequest<S> for ValidatedJson<T>
where
    S: Send + Sync,
    T: DeserializeOwned,
{
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let content_type = req
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let bytes = Bytes::from_request(req, state).await.map_err(IntoResponse::into_response)?;
        decode_body::<T>(content_type.as_deref(), &bytes)
            .map(ValidatedJson)
            .map_err(IntoResponse::into_response)
    }
}

/// Query-string extractor with FastAPI semantics: a missing or unparsable
/// parameter is a 422 located under `query`, not axum's plain-text 400.
#[derive(Debug, Clone, Copy, Default)]
pub struct ValidatedQuery<T>(pub T);

/// Parse a query string into `T`, answering a misfit with FastAPI's 422.
pub fn decode_query<T: DeserializeOwned>(query: Option<&str>) -> Result<T, HTTPException> {
    let raw = query.unwrap_or("");
    let uri: axum::http::Uri = format!("/?{raw}")
        .parse()
        .map_err(|_| validation_error(vec![validation_entry("value_error", vec![json!("query")], "Invalid query string", Value::Null)]))?;
    match axum::extract::Query::<T>::try_from_uri(&uri) {
        Ok(axum::extract::Query(value)) => Ok(value),
        Err(rejection) => {
            let text = rejection.body_text();
            let message = text
                .strip_prefix("Failed to deserialize query string: ")
                .unwrap_or(&text)
                .to_string();
            Err(validation_error(serde_validation_errors("query", &message, Value::Null)))
        }
    }
}

impl<S, T> FromRequestParts<S> for ValidatedQuery<T>
where
    S: Send + Sync,
    T: DeserializeOwned,
{
    type Rejection = Response;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        decode_query::<T>(parts.uri.query())
            .map(ValidatedQuery)
            .map_err(IntoResponse::into_response)
    }
}

/// Deserialize a query-string boolean the way FastAPI/Pydantic does:
/// `true/false`, `1/0`, `yes/no`, `on/off`, `t/f`, `y/n`, case-insensitive.
pub fn deserialize_py_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Raw {
        Bool(bool),
        Text(String),
    }
    match Raw::deserialize(deserializer)? {
        Raw::Bool(b) => Ok(b),
        Raw::Text(s) => match s.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "t" | "yes" | "y" | "on" => Ok(true),
            "0" | "false" | "f" | "no" | "n" | "off" => Ok(false),
            _ => Err(serde::de::Error::custom(
                "Input should be a valid boolean, unable to interpret input",
            )),
        },
    }
}

// ---------------------------------------------------------------------------
// Multipart forms
// ---------------------------------------------------------------------------

/// One field of a multipart form (`UploadFile` or a plain form value).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FormField {
    /// Field name.
    pub name: String,
    /// `UploadFile.filename` — present for file parts.
    pub filename: Option<String>,
    /// `UploadFile.content_type`.
    pub content_type: Option<String>,
    /// The field's bytes.
    pub data: Vec<u8>,
}

impl FormField {
    /// The field's value as text (form fields are strings in FastAPI).
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.data).into_owned()
    }
}

/// A parsed multipart form.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Form {
    pub fields: Vec<FormField>,
}

impl Form {
    /// The first field called `name`.
    pub fn get(&self, name: &str) -> Option<&FormField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Fail with FastAPI's 422 when any of `required` is absent.
    pub fn require(&self, required: &[&str]) -> Result<(), HTTPException> {
        let missing: Vec<Value> = required
            .iter()
            .filter(|name| self.get(name).is_none())
            .map(|name| {
                validation_entry("missing", vec![json!("body"), json!(name)], "Field required", Value::Null)
            })
            .collect();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(validation_error(missing))
        }
    }
}

/// Read a whole multipart form.
///
/// A request that is not multipart at all reads as an empty form (Starlette
/// parses no fields from it), so the caller's [`Form::require`] answers 422
/// for the missing fields just as FastAPI would.  A body that claims to be
/// multipart but cannot be parsed is a 400 "There was an error parsing the
/// body", FastAPI's answer for the same failure.
pub async fn read_form(multipart: Result<Multipart, MultipartRejection>) -> Result<Form, HTTPException> {
    let mut multipart = match multipart {
        Ok(m) => m,
        Err(_) => return Ok(Form::default()),
    };
    let mut form = Form::default();
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(_) => return Err(HTTPException::new(400, "There was an error parsing the body")),
        };
        let name = field.name().unwrap_or("").to_string();
        let filename = field.file_name().map(str::to_owned);
        let content_type = field.content_type().map(str::to_owned);
        let data = field
            .bytes()
            .await
            .map_err(|_| HTTPException::new(400, "There was an error parsing the body"))?;
        form.fields.push(FormField { name, filename, content_type, data: data.to_vec() });
    }
    Ok(form)
}

// ---------------------------------------------------------------------------
// Python-compatible JSON text
// ---------------------------------------------------------------------------

/// `serde_json` formatter reproducing Python's `json.dumps` defaults:
/// `", "` / `": "` separators and `ensure_ascii=True`.
struct PythonFormatter;

impl serde_json::ser::Formatter for PythonFormatter {
    fn begin_array_value<W: ?Sized + io::Write>(&mut self, writer: &mut W, first: bool) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + io::Write>(&mut self, writer: &mut W, first: bool) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }

    fn write_string_fragment<W: ?Sized + io::Write>(&mut self, writer: &mut W, fragment: &str) -> io::Result<()> {
        for c in fragment.chars() {
            let code = c as u32;
            if code < 0x7f {
                let mut buf = [0u8; 4];
                writer.write_all(c.encode_utf8(&mut buf).as_bytes())?;
            } else if code <= 0xffff {
                write!(writer, "\\u{code:04x}")?;
            } else {
                let v = code - 0x10000;
                let high = 0xd800 + (v >> 10);
                let low = 0xdc00 + (v & 0x3ff);
                write!(writer, "\\u{high:04x}\\u{low:04x}")?;
            }
        }
        Ok(())
    }
}

/// `json.dumps(value)` with Python's default separators and ASCII escaping.
pub fn json_dumps(value: &Value) -> String {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PythonFormatter);
    serde::Serialize::serialize(value, &mut ser).expect("serializing a JSON value cannot fail");
    String::from_utf8(out).expect("json output is ASCII")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_dumps_matches_python() {
        // ensure_ascii=True: non-ASCII as \uXXXX, astral characters as a
        // surrogate pair, DEL escaped too.
        let v = json!({"a": [1, 2], "b": "x — y", "c": null, "d": "\u{1F600}", "e": "\u{7f}"});
        let bs = char::from(92u8); // a literal backslash
        assert_eq!(
            json_dumps(&v),
            format!(r#"{{"a": [1, 2], "b": "x {bs}u2014 y", "c": null, "d": "{bs}ud83d{bs}ude00", "e": "{bs}u007f"}}"#)
        );
    }

    #[test]
    fn py_repr_quotes_like_python() {
        assert_eq!(py_repr_str("nope"), "'nope'");
        assert_eq!(py_repr_str("it's"), "\"it's\"");
    }

    #[test]
    fn missing_field_is_parsed() {
        assert_eq!(missing_field("missing field `token` at line 1 column 2"), Some("token".into()));
        assert_eq!(missing_field("invalid type"), None);
    }

    #[test]
    fn io_enospc_is_507_and_permission_is_500() {
        let full = io::Error::from_raw_os_error(ENOSPC);
        assert_eq!(status_for(&full), 507);
        let denied = io::Error::new(io::ErrorKind::PermissionDenied, "denied");
        assert_eq!(status_for(&denied), 500);
        let refused = io::Error::new(io::ErrorKind::ConnectionRefused, "refused");
        assert_eq!(status_for(&refused), 502);
    }
}
