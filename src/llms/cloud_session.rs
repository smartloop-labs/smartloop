//! A short-lived credential for calling a cloud model's provider directly.
//!
//! The platform never relays a chat.  It hands out a key instead, scoped the
//! way a pre-signed URL is: it expires, it carries a spend limit (the user's
//! remaining allowance), and the provider refuses any model outside the
//! user's plan.  The framework then sends the chat to the provider itself,
//! so prompts and answers never pass through `api.smartloop.ai`.
//!
//! ## Contract
//!
//! `POST {SLP_API_BASE}/v1/chat/session` (platform auth), body
//! `{"project_id": "...", "replaces": "<session_id>"}` (`replaces` only when
//! the previous session's key was refused, so the server settles it at once
//! instead of holding its budget until it expires), answers:
//!
//! ```json
//! {
//!   "session_id": "...",
//!   "base_url": "https://openrouter.ai/api/v1",
//!   "key": "sk-or-...",
//!   "expires_at": "2026-11-02T18:30:00Z",
//!   "headers": {"HTTP-Referer": "https://smartloop.ai", "X-Title": "Smartloop"},
//!   "models": [
//!     {"name": "kimi-k2", "provider_model": "moonshotai/kimi-k2",
//!      "max_tokens": 16384,
//!      "body": {"provider": {"sort": "price"}, "usage": {"include": true}}}
//!   ]
//! }
//! ```
//!
//! `models[].body` is merged into each request for that model, so the server
//! keeps deciding provider routing and price caps without a client release.
//! A 401, 403 or 429 is reported the way the chat endpoint reported it.
//!
//! The key lives in memory only.  It is never written to disk, logged, or
//! sent anywhere but `base_url`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::llms::cloud_llm::describe_failure;
use crate::platform::client::{Headers, SLPPlatformClient};

const LOG_TARGET: &str = "smartloop.llms.cloud_session";

pub const CHAT_SESSION: &str = "/v1/chat/session";

/// A session this close to expiry is renewed before use, so a request never
/// sets out with a key that dies on the way.
const RENEW_BEFORE: chrono::Duration = chrono::Duration::seconds(60);

const FETCH_TIMEOUT: Duration = Duration::from_secs(15);

/// How long a freshly minted key may be refused while the provider has not
/// caught up with it.  OpenRouter answers `401 User not found` for a key a
/// few seconds old; a refusal inside this window is waited out, not taken
/// as a dead key (each replacement reserves another budget).
pub const KEY_PROPAGATION: Duration = Duration::from_secs(20);

/// One model the session's key may call, and how.
#[derive(Clone, Debug, Deserialize)]
pub struct SessionModel {
    /// The catalog name the app knows the model by.
    pub name: String,
    /// The provider's id for it, sent as `model`.
    pub provider_model: String,
    /// The most answer tokens the server allows a request.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Fields merged into every request body for this model.
    #[serde(default)]
    pub body: Map<String, Value>,
}

/// A provider endpoint and the short-lived key to call it with.
#[derive(Clone, Deserialize)]
pub struct CloudSession {
    /// The server's name for this session, sent as `replaces` when its key is
    /// refused.  Absent from an older server.
    #[serde(default)]
    pub session_id: Option<String>,
    /// When this process received the session.
    #[serde(skip, default = "Instant::now")]
    minted_at: Instant,
    pub base_url: String,
    key: String,
    pub expires_at: DateTime<Utc>,
    #[serde(default)]
    headers: HashMap<String, String>,
    #[serde(default)]
    pub models: Vec<SessionModel>,
}

impl std::fmt::Debug for CloudSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CloudSession")
            .field("base_url", &self.base_url)
            .field("key", &"<redacted>")
            .field("expires_at", &self.expires_at)
            .field("models", &self.models.iter().map(|m| m.name.as_str()).collect::<Vec<_>>())
            .finish()
    }
}

impl CloudSession {
    /// The entry for catalog model `name`, if the key may call it.
    pub fn model(&self, name: &str) -> Option<&SessionModel> {
        self.models.iter().find(|m| m.name == name)
    }

    /// Where chat completions are posted.
    pub fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }

    /// The headers for a provider request: the server's extras, then the
    /// key.  An extra can't replace the authorization.
    pub fn headers(&self) -> Headers {
        let mut headers: Headers = self
            .headers
            .iter()
            .filter(|(name, _)| !name.eq_ignore_ascii_case("authorization"))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        headers.insert("Authorization".to_string(), format!("Bearer {}", self.key));
        headers
    }

    /// Whether the key is young enough that a refusal may only mean the
    /// provider has not caught up with it yet.
    pub fn is_new(&self) -> bool {
        self.minted_at.elapsed() < KEY_PROPAGATION
    }

    /// Whether the session can still be used at `now`.
    pub fn is_fresh(&self, now: DateTime<Utc>) -> bool {
        self.expires_at - RENEW_BEFORE > now
    }

    /// Refuse a session whose key would travel in the clear, or that has no
    /// key at all.  Plain http is allowed only to this machine.
    fn validated(self) -> Result<Self, String> {
        let url = url::Url::parse(&self.base_url).map_err(|_| "The cloud session named no usable endpoint.".to_string())?;
        let loopback = matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "::1" | "[::1]"));
        if url.scheme() != "https" && !(url.scheme() == "http" && loopback) {
            tracing::warn!(target: LOG_TARGET, "[session] refusing endpoint {} (not https)", self.base_url);
            return Err("The cloud session named an insecure endpoint.".to_string());
        }
        if self.key.trim().is_empty() {
            return Err("The cloud session came without a key.".to_string());
        }
        Ok(self)
    }
}

/// Sessions by platform API base.
static SESSIONS: Lazy<Mutex<HashMap<String, Arc<CloudSession>>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// The current session for `client`'s platform, minting one when there is
/// none or it is about to expire.  `model` names the model in errors.
///
/// The lock is held through a fetch, so requests that find the session stale
/// together wait for one mint instead of each making their own.
pub fn session(client: &SLPPlatformClient, project_id: &str, model: &str) -> Result<Arc<CloudSession>, String> {
    let mut sessions = SESSIONS.lock();
    if let Some(current) = sessions.get(client.base_url())
        && current.is_fresh(Utc::now())
    {
        return Ok(current.clone());
    }
    let fresh = Arc::new(fetch(client, project_id, model, None)?);
    tracing::info!(
        target: LOG_TARGET,
        "[session] new cloud session for {} models, expires {}",
        fresh.models.len(),
        fresh.expires_at
    );
    sessions.insert(client.base_url().to_string(), fresh.clone());
    Ok(fresh)
}

/// A session in place of `stale`, whose key was refused.
///
/// If another request already replaced it, that replacement is used.
/// Otherwise the new mint names `stale` as the session it `replaces`, so the
/// server settles it now rather than holding its budget until it expires.
pub fn replace(
    client: &SLPPlatformClient,
    stale: &Arc<CloudSession>,
    project_id: &str,
    model: &str,
) -> Result<Arc<CloudSession>, String> {
    let mut sessions = SESSIONS.lock();
    if let Some(current) = sessions.get(client.base_url())
        && !Arc::ptr_eq(current, stale)
        && current.is_fresh(Utc::now())
    {
        return Ok(current.clone());
    }
    sessions.remove(client.base_url());
    let fresh = Arc::new(fetch(client, project_id, model, stale.session_id.as_deref())?);
    tracing::info!(
        target: LOG_TARGET,
        "[session] replaced a refused cloud session ({} models, expires {})",
        fresh.models.len(),
        fresh.expires_at
    );
    sessions.insert(client.base_url().to_string(), fresh.clone());
    Ok(fresh)
}

/// Drop `stale` once its key was refused.  A session another request has
/// already put in its place is kept.
pub fn invalidate(client: &SLPPlatformClient, stale: &Arc<CloudSession>) {
    let mut sessions = SESSIONS.lock();
    if sessions.get(client.base_url()).is_some_and(|current| Arc::ptr_eq(current, stale)) {
        sessions.remove(client.base_url());
    }
}

/// Forget every session, e.g. on sign-out.  The server revokes the keys;
/// this only stops this process from using them meanwhile.
pub fn clear() {
    SESSIONS.lock().clear();
}

fn fetch(
    client: &SLPPlatformClient,
    project_id: &str,
    model: &str,
    replaces: Option<&str>,
) -> Result<CloudSession, String> {
    let http = reqwest::blocking::Client::builder()
        .connect_timeout(FETCH_TIMEOUT)
        .timeout(FETCH_TIMEOUT)
        .build()
        .map_err(|e| format!("Could not reach {model}: {e}"))?;
    let url = format!("{}{CHAT_SESSION}", client.base_url());
    let mut body = json!({ "project_id": project_id });
    if let Some(replaces) = replaces {
        body["replaces"] = json!(replaces);
    }
    let response = client
        .authed(|headers: &Headers| {
            let mut request = http.post(&url).json(&body);
            for (name, value) in headers {
                request = request.header(name.as_str(), value.as_str());
            }
            request.send()
        })
        .map_err(|e| {
            tracing::warn!(target: LOG_TARGET, "[session] request failed: {e}");
            format!("Could not reach {model}. Check your connection and try again.")
        })?;
    let status = response.status().as_u16();
    if status >= 400 {
        let text = response.text().unwrap_or_default();
        tracing::warn!(target: LOG_TARGET, "[session] refused ({status}): {text}");
        return Err(describe_failure(status, &text, model));
    }
    let session: CloudSession = response.json().map_err(|e| {
        tracing::warn!(target: LOG_TARGET, "[session] unreadable answer: {e}");
        format!("{model} is not available right now.")
    })?;
    session.validated()
}
