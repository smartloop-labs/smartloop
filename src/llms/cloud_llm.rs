//! A remote catalog model, answered over HTTP by the platform API.
//!
//! The prompt is split back into chat messages and sent to
//! `{url}/v1/chat/completions` under the catalog name; the platform forwards
//! it to the model's upstream provider and records token usage itself.
//! Not to be confused with `model_host::remote_llm::RemoteLLM`, which is a
//! local model loaded in the model-host process.
//!
//! Failures surface the way every `BaseLLM` stream reports them: the stream
//! panics with a message meant for the user, and `ModelService` hands that
//! message back to the chat instead of falling back to a local model.

use std::any::Any;
use std::cell::Cell;
use std::io::{BufRead, BufReader};
use std::sync::atomic::Ordering;
use std::time::Duration;

use parking_lot::RwLock;
use serde_json::{json, Value};

use crate::config::app_settings::AppSettings;
use crate::llms::base_llm::{BaseLLM, CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream};
use crate::platform::client::{AuthHeadersFn, Headers, RefreshFn, SLPPlatformClient};

const LOG_TARGET: &str = "smartloop.llms.cloud";

/// Context window assumed when the catalog gives none.
pub const DEFAULT_CONTEXT_LENGTH: usize = 32_768;

const CHAT_PATH: &str = "/v1/chat/completions";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Whole-request budget, streamed body included: long enough for a long
/// answer, short enough that a hung upstream frees the chat.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

thread_local! {
    static LOCAL_ONLY: Cell<bool> = const { Cell::new(false) };
}

/// Restores the previous [`local_only`] setting when dropped.
#[must_use = "the setting is restored when the guard is dropped"]
pub struct LocalOnlyScope(bool);

impl Drop for LocalOnlyScope {
    fn drop(&mut self) {
        LOCAL_ONLY.with(|cell| cell.set(self.0));
    }
}

/// Mark model resolution on this thread as orchestration (`on`) or not, until
/// the guard drops.  Orchestration runs on device: while set, a pinned remote
/// model is ignored and selection falls back to local models.  The final
/// answer is resolved outside any such scope.
pub fn local_only_scope(on: bool) -> LocalOnlyScope {
    LocalOnlyScope(LOCAL_ONLY.with(|cell| cell.replace(on)))
}

/// Whether this thread is resolving a model for orchestration.
pub fn local_only() -> bool {
    LOCAL_ONLY.with(Cell::get)
}

/// Whether our platform credentials may be sent to `url`: only to the
/// configured platform API host or a `smartloop.ai` host.  The catalog names
/// the url, and a user's bearer token must not follow it anywhere else.
pub fn credentials_allowed(url: &str, api_base: &str) -> bool {
    let host = |u: &str| url::Url::parse(u).ok().and_then(|u| u.host_str().map(str::to_lowercase));
    let Some(target) = host(url) else { return false };
    if host(api_base).is_some_and(|base| base == target) {
        return true;
    }
    target == "smartloop.ai" || target.ends_with(".smartloop.ai")
}

/// The message to show for a failed request, from the status and body.
pub fn describe_failure(status: u16, body: &str, model: &str) -> String {
    let detail = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v.get("detail").and_then(Value::as_str).map(str::to_string));
    match status {
        401 => format!("Sign in to use {model}."),
        403 => format!("{model} isn't available on your current plan."),
        404 => format!("{model} is no longer available."),
        // A spent allowance says when it resets; a busy upstream or our own
        // rate limit has no detail worth more than the generic message.
        429 => detail.unwrap_or_else(|| format!("{model} is busy right now. Try again shortly.")),
        400 => detail.unwrap_or_else(|| format!("{model} could not answer this request.")),
        _ => format!("{model} did not answer (HTTP {status})."),
    }
}

/// A remote model as a [`BaseLLM`].
pub struct CloudLLM {
    project_id: String,
    model_name: String,
    context_length: usize,
    client: SLPPlatformClient,
    send_credentials: bool,
    cancelled: RwLock<Option<CancelEvent>>,
}

impl CloudLLM {
    pub fn new(project_id: &str, model_name: &str, url: &str, context_length: Option<u32>) -> Self {
        Self::with_hooks(project_id, model_name, url, context_length, None, None)
    }

    /// [`Self::new`] with the platform client's auth and refresh hooks
    /// replaced (`None` keeps the default).
    pub fn with_hooks(
        project_id: &str,
        model_name: &str,
        url: &str,
        context_length: Option<u32>,
        auth_headers_fn: Option<AuthHeadersFn>,
        refresh_fn: Option<RefreshFn>,
    ) -> Self {
        let api_base = AppSettings::new().slp_api_base;
        let send_credentials = credentials_allowed(url, &api_base);
        if !send_credentials {
            tracing::warn!(target: LOG_TARGET, "[cloud] {model_name} is served from {url}; not sending credentials there");
        }
        Self {
            project_id: project_id.to_string(),
            model_name: model_name.to_string(),
            context_length: context_length.map(|n| n as usize).filter(|n| *n > 0).unwrap_or(DEFAULT_CONTEXT_LENGTH),
            client: SLPPlatformClient::with_hooks(url, auth_headers_fn, refresh_fn),
            send_credentials,
            cancelled: RwLock::new(None),
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn url(&self) -> &str {
        self.client.base_url()
    }

    fn body(&self, messages: &[ChatMessage], stream: bool, options: CallOptions) -> Value {
        let mut body = json!({
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "project_id": self.project_id,
        });
        if let Some(max_tokens) = options.max_tokens {
            body["max_tokens"] = json!(max_tokens);
        }
        body
    }

    /// POST the chat request, redeeming the refresh token once on a 401.
    fn post(&self, body: &Value) -> Result<reqwest::blocking::Response, String> {
        let http = reqwest::blocking::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|e| format!("Could not reach {}: {e}", self.model_name))?;
        let url = format!("{}{CHAT_PATH}", self.client.base_url());
        let send = |headers: &Headers| {
            let mut request = http.post(&url).json(body);
            for (name, value) in headers {
                request = request.header(name.as_str(), value.as_str());
            }
            request.send()
        };
        let sent = if self.send_credentials {
            self.client.authed(send)
        } else {
            send(&self.client.headers(None))
        };
        let response = sent.map_err(|e| {
            tracing::warn!(target: LOG_TARGET, "[cloud] {} request failed: {e}", self.model_name);
            format!("Could not reach {}. Check your connection and try again.", self.model_name)
        })?;
        let status = response.status().as_u16();
        if status >= 400 {
            let text = response.text().unwrap_or_default();
            tracing::warn!(target: LOG_TARGET, "[cloud] {} answered {status}: {text}", self.model_name);
            return Err(describe_failure(status, &text, &self.model_name));
        }
        Ok(response)
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.read().as_ref().is_some_and(|c| c.load(Ordering::SeqCst))
    }
}

/// The text delta of one streamed chunk, or the error it reports.
fn chunk_text(chunk: &Value) -> Result<Option<String>, String> {
    if let Some(error) = chunk.get("error") {
        let message = error.get("message").and_then(Value::as_str).unwrap_or("The model stopped answering.");
        return Err(message.to_string());
    }
    Ok(chunk
        .pointer("/choices/0/delta/content")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string))
}

impl BaseLLM for CloudLLM {
    fn project_id(&self) -> &str {
        &self.project_id
    }

    /// Estimated at four characters a token: the model's own tokenizer is
    /// remote, and this only sizes the prompt against the context window.
    fn get_num_tokens(&self, text: &str) -> usize {
        text.chars().count().div_ceil(4)
    }

    fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        messages
    }

    fn get_max_tokens(&self) -> usize {
        self.context_length
    }

    fn call(&self, prompt: &str, _stop: Option<&[String]>, options: CallOptions) -> anyhow::Result<String> {
        let messages = self.create_messages_from_prompt(prompt);
        let response = self.post(&self.body(&messages, false, options)).map_err(anyhow::Error::msg)?;
        let data: Value = response
            .json()
            .map_err(|e| anyhow::anyhow!("{} sent an unreadable answer: {e}", self.model_name))?;
        Ok(data
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string())
    }

    fn stream<'a>(&'a self, prompt: &str, options: CallOptions) -> GenerationStream<'a> {
        let messages = self.create_messages_from_prompt(prompt);
        let response = match self.post(&self.body(&messages, true, options)) {
            Ok(response) => response,
            Err(message) => std::panic::panic_any(message),
        };
        let mut lines = BufReader::new(response).lines();
        Box::new(std::iter::from_fn(move || loop {
            if self.is_cancelled() {
                return None;
            }
            let line = match lines.next()? {
                Ok(line) => line,
                Err(e) => {
                    tracing::warn!(target: LOG_TARGET, "[cloud] {} stream broke off: {e}", self.model_name);
                    std::panic::panic_any(format!("{} stopped answering.", self.model_name));
                }
            };
            let Some(data) = line.strip_prefix("data:").map(str::trim) else { continue };
            if data == "[DONE]" {
                return None;
            }
            let Ok(chunk) = serde_json::from_str::<Value>(data) else { continue };
            match chunk_text(&chunk) {
                Ok(Some(text)) => return Some(GenerationChunk { text }),
                Ok(None) => continue,
                Err(message) => std::panic::panic_any(message),
            }
        }))
    }

    fn model_label(&self) -> String {
        self.model_name.clone()
    }

    fn cancelled(&self) -> Option<CancelEvent> {
        self.cancelled.read().clone()
    }

    fn set_cancelled(&self, event: Option<CancelEvent>) {
        *self.cancelled.write() = event;
    }

    fn llm_type(&self) -> &str {
        "smartloop-cloud"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
