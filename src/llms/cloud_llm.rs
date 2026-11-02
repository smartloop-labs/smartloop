//! A remote catalog model, answered over HTTP.
//!
//! The prompt is split back into chat messages and sent one of two ways:
//!
//! * **Direct** (`SLP_CLOUD_DIRECT`): to the model's provider, with the
//!   short-lived key a [`cloud_session`](crate::llms::cloud_session) holds.
//!   The platform only issues the key; it never sees the chat.
//! * **Proxy**: to `{url}/v1/chat/completions` under the catalog name; the
//!   platform forwards it upstream and records token usage itself.
//!
//! Not to be confused with `model_host::remote_llm::RemoteLLM`, which is a
//! local model loaded in the model-host process.
//!
//! Failures surface the way every `BaseLLM` stream reports them: the stream
//! panics with a message meant for the user, and `ModelService` hands that
//! message back to the chat instead of falling back to a local model.

use std::any::Any;
use std::cell::Cell;
use std::io::{BufRead, BufReader};
use std::collections::HashMap;
use std::sync::atomic::Ordering;

use once_cell::sync::Lazy;
use parking_lot::Mutex;

/// The most tokens one answer may have at each endpoint and model, as the
/// endpoint said when a request asked for more ("max_tokens is at most
/// N"), kept for the process: a client is made anew for each answer.
static ANSWER_CAPS: Lazy<Mutex<HashMap<String, usize>>> = Lazy::new(|| Mutex::new(HashMap::new()));
use std::time::Duration;

use parking_lot::RwLock;
use serde_json::{json, Value};

use crate::config::app_settings::AppSettings;
use crate::llms::base_llm::{BaseLLM, CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream};
use crate::llms::cloud_session::{self, SessionModel};
use crate::platform::client::{AuthHeadersFn, Headers, RefreshFn, SLPPlatformClient};

const LOG_TARGET: &str = "smartloop.llms.cloud";

/// Context window assumed when the catalog gives none.
pub const DEFAULT_CONTEXT_LENGTH: usize = 32_768;

/// Reasoning allowance added to a capped request when the catalog gives
/// none.  Reasoning models count their thinking against `max_tokens`, and
/// the ones measured spent 50-350 tokens before answering a short prompt.
/// Room given to a model's reasoning on top of the answer's, when its
/// catalog row sets none: the reasoning models served from the cloud think
/// in thousands of tokens before a line of answer.
pub const DEFAULT_REASONING_BUDGET_TOKENS: usize = 4096;

/// Room an answer gets when the caller caps nothing: enough for a reply
/// that gives files back whole. Without a cap the endpoint applies its own,
/// which a long answer, or the reasoning before it, runs through with
/// nothing to show.
pub const DEFAULT_ANSWER_TOKENS: usize = 16_384;

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

/// The message to show for a request the provider refused, called directly
/// with a session key.  Providers report `{"error": {"message": ...}}`.
pub fn describe_provider_failure(status: u16, body: &str, model: &str) -> String {
    let message = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v.pointer("/error/message").and_then(Value::as_str).map(str::to_string));
    match status {
        // The key's spend limit: the allowance this session was given is used.
        402 => "You've used your cloud allowance for now.".to_string(),
        // The key's model allowlist.
        404 => format!("{model} isn't available on your current plan."),
        429 => format!("{model} is busy right now. Try again shortly."),
        401 => format!("{model} refused this session. Try again."),
        400 | 403 => message.unwrap_or_else(|| format!("{model} could not answer this request.")),
        _ => format!("{model} did not answer (HTTP {status})."),
    }
}

/// The provider request for `model`: the server's fields for it, then the
/// app's, under the provider's model id.  The project id stays home, and the
/// answer is held to the server's cap.
fn direct_body(model: &SessionModel, body: &Value) -> Value {
    let mut out = Value::Object(model.body.clone());
    if let Value::Object(fields) = body {
        for (name, value) in fields {
            if name != "project_id" {
                out[name] = value.clone();
            }
        }
    }
    out["model"] = json!(model.provider_model);
    if let (Some(cap), Some(asked)) = (model.max_tokens, out["max_tokens"].as_u64())
        && asked as usize > cap
    {
        out["max_tokens"] = json!(cap);
    }
    if out["stream"] == json!(true) && out.get("stream_options").is_none() {
        out["stream_options"] = json!({"include_usage": true});
    }
    out
}

/// A remote model as a [`BaseLLM`].
pub struct CloudLLM {
    project_id: String,
    model_name: String,
    context_length: usize,
    reasoning_budget_tokens: usize,
    client: SLPPlatformClient,
    send_credentials: bool,
    /// The platform to take session keys from, when calling providers
    /// directly; `None` sends chats to `client`.
    session_client: Option<SLPPlatformClient>,
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
        let settings = AppSettings::new();
        let api_base = settings.slp_api_base;
        let send_credentials = credentials_allowed(url, &api_base);
        if !send_credentials {
            tracing::warn!(target: LOG_TARGET, "[cloud] {model_name} is served from {url}; not sending credentials there");
        }
        // Only a model the platform serves has a session to call it with.
        let session_client = (settings.cloud_direct && send_credentials)
            .then(|| SLPPlatformClient::with_hooks(&api_base, auth_headers_fn.clone(), refresh_fn.clone()));
        Self {
            project_id: project_id.to_string(),
            model_name: model_name.to_string(),
            context_length: context_length.map(|n| n as usize).filter(|n| *n > 0).unwrap_or(DEFAULT_CONTEXT_LENGTH),
            reasoning_budget_tokens: DEFAULT_REASONING_BUDGET_TOKENS,
            client: SLPPlatformClient::with_hooks(url, auth_headers_fn, refresh_fn),
            send_credentials,
            session_client,
            cancelled: RwLock::new(None),
        }
    }

    /// Use the catalog's reasoning budget; `None` keeps the default.
    pub fn with_reasoning_budget(mut self, budget: Option<u32>) -> Self {
        if let Some(budget) = budget {
            self.reasoning_budget_tokens = budget as usize;
        }
        self
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn url(&self) -> &str {
        self.client.base_url()
    }

    /// Whether chats go straight to the provider with a session key.
    pub fn is_direct(&self) -> bool {
        self.session_client.is_some()
    }

    /// The `max_tokens` to send for a caller's cap.  The cap sizes the
    /// answer; the model's reasoning comes on top, or a short cap is spent
    /// thinking and the answer comes back empty.
    /// The cap sent with a request: the caller's, or `DEFAULT_ANSWER_TOKENS`
    /// when it sets none, so the endpoint's own default never cuts an
    /// answer short; the reasoning budget on top either way, and never
    /// past what the endpoint said an answer may have.
    fn request_max_tokens(&self, options: CallOptions) -> Option<usize> {
        Some(self.within_cap(options.max_tokens.unwrap_or(DEFAULT_ANSWER_TOKENS) + self.reasoning_budget_tokens))
    }

    /// `tokens`, or the endpoint's answer cap when that's known and less.
    /// Not the context window: that's `context_length`.
    fn within_cap(&self, tokens: usize) -> usize {
        match ANSWER_CAPS.lock().get(&self.cap_key()) {
            Some(cap) => tokens.min(*cap),
            None => tokens,
        }
    }

    fn cap_key(&self) -> String {
        format!("{}|{}", self.client.base_url(), self.model_name)
    }

    /// The answer cap an endpoint's refusal names ("max_tokens is at most
    /// 16384 for minimax-m3"), if it's that refusal.
    fn cap_in(text: &str) -> Option<usize> {
        let rest = text.split("is at most ").nth(1)?;
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        digits.parse().ok().filter(|n| *n > 0)
    }

    fn body(&self, messages: &[ChatMessage], stream: bool, max_tokens: Option<usize>) -> Value {
        self.body_for(json!(messages), stream, max_tokens)
    }

    /// The request body for already-shaped OpenAI `messages`.
    fn body_for(&self, messages: Value, stream: bool, max_tokens: Option<usize>) -> Value {
        let mut body = json!({
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "project_id": self.project_id,
        });
        if let Some(max_tokens) = max_tokens {
            body["max_tokens"] = json!(max_tokens);
        }
        body
    }

    /// One non-streaming completion: the answer and why it stopped.
    fn complete(&self, messages: &[ChatMessage], max_tokens: Option<usize>) -> anyhow::Result<(String, Option<String>)> {
        let response = self.post(&self.body(messages, false, max_tokens)).map_err(anyhow::Error::msg)?;
        let data: Value = response
            .json()
            .map_err(|e| anyhow::anyhow!("{} sent an unreadable answer: {e}", self.model_name))?;
        let content = data.pointer("/choices/0/message/content").and_then(Value::as_str).unwrap_or_default();
        let finish = data.pointer("/choices/0/finish_reason").and_then(Value::as_str);
        Ok((content.to_string(), finish.map(str::to_string)))
    }

    /// One non-streaming completion over OpenAI-shaped `messages`, whose
    /// `content` may be a list of parts (text / `image_url`) — the vision
    /// path.  Returns the raw response, shaped like a local model's
    /// `create_chat_completion`, so callers read `choices` the same way.
    /// `max_tokens` sizes the answer; the reasoning budget comes on top.
    pub fn create_chat_completion(
        &self,
        messages: Vec<Value>,
        max_tokens: Option<usize>,
        temperature: f32,
    ) -> anyhow::Result<Value> {
        let max_tokens = max_tokens.map(|cap| cap + self.reasoning_budget_tokens);
        let mut body = self.body_for(Value::Array(messages), false, max_tokens);
        body["temperature"] = json!(temperature);
        let response = self.post(&body).map_err(anyhow::Error::msg)?;
        response
            .json()
            .map_err(|e| anyhow::anyhow!("{} sent an unreadable answer: {e}", self.model_name))
    }

    /// Send `body` the way this model is reached; refused for asking more
    /// answer tokens than the endpoint allows, it learns the cap and sends
    /// again within it.
    fn post(&self, body: &Value) -> Result<reqwest::blocking::Response, String> {
        let http = reqwest::blocking::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|e| format!("Could not reach {}: {e}", self.model_name))?;
        match &self.session_client {
            Some(platform) => self.post_direct(&http, platform, body),
            None => self.post_proxy(&http, body),
        }
    }

    /// POST to the platform's chat endpoint, redeeming the refresh token
    /// once on a 401.
    fn post_proxy(&self, http: &reqwest::blocking::Client, body: &Value) -> Result<reqwest::blocking::Response, String> {
        let url = format!("{}{CHAT_PATH}", self.client.base_url());
        let mut body = body.clone();
        loop {
            let send = |headers: &Headers| {
                let mut request = http.post(&url).json(&body);
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
            let response = sent.map_err(|e| self.unreachable(e))?;
            let status = response.status().as_u16();
            if status < 400 {
                return Ok(response);
            }
            let text = response.text().unwrap_or_default();
            tracing::warn!(target: LOG_TARGET, "[cloud] {} answered {status}: {text}", self.model_name);
            if self.learn_cap(&text, &mut body) {
                continue;
            }
            return Err(describe_failure(status, &text, &self.model_name));
        }
    }

    /// POST to the provider with the session's key.  A key the provider
    /// refuses (expired, revoked, or out of allowance) is traded for a new
    /// session once.
    fn post_direct(
        &self,
        http: &reqwest::blocking::Client,
        platform: &SLPPlatformClient,
        body: &Value,
    ) -> Result<reqwest::blocking::Response, String> {
        let mut session = cloud_session::session(platform, &self.project_id, &self.model_name)?;
        let mut renewed = false;
        let mut body = body.clone();
        loop {
            let Some(model) = session.model(&self.model_name) else {
                return Err(format!("{} isn't available on your current plan.", self.model_name));
            };
            if let Some(cap) = model.max_tokens {
                let mut caps = ANSWER_CAPS.lock();
                let known = caps.entry(self.cap_key()).or_insert(cap);
                *known = (*known).min(cap);
            }
            let mut request = http.post(session.chat_url()).json(&direct_body(model, &body));
            for (name, value) in session.headers() {
                request = request.header(name.as_str(), value.as_str());
            }
            let response = request.send().map_err(|e| self.unreachable(e))?;
            let status = response.status().as_u16();
            if status < 400 {
                return Ok(response);
            }
            let text = response.text().unwrap_or_default();
            tracing::warn!(target: LOG_TARGET, "[cloud] {} provider answered {status}: {text}", self.model_name);
            if matches!(status, 401 | 402 | 403) && !renewed {
                renewed = true;
                cloud_session::invalidate(platform, &session);
                session = cloud_session::session(platform, &self.project_id, &self.model_name)?;
                continue;
            }
            if self.learn_cap(&text, &mut body) {
                continue;
            }
            return Err(describe_provider_failure(status, &text, &self.model_name));
        }
    }

    /// The message for a request that never got an answer.
    fn unreachable(&self, e: reqwest::Error) -> String {
        tracing::warn!(target: LOG_TARGET, "[cloud] {} request failed: {e}", self.model_name);
        format!("Could not reach {}. Check your connection and try again.", self.model_name)
    }

    /// When `text` refuses `body` for asking more answer tokens than allowed,
    /// remember the cap and lower `body` to it.  Whether to send again.
    fn learn_cap(&self, text: &str, body: &mut Value) -> bool {
        let asked = body["max_tokens"].as_u64().map(|n| n as usize);
        let (Some(cap), Some(asked)) = (Self::cap_in(text), asked) else { return false };
        if cap >= asked {
            return false;
        }
        tracing::info!(target: LOG_TARGET, "[cloud] {} allows {cap} answer tokens; asking for that", self.model_name);
        ANSWER_CAPS.lock().insert(self.cap_key(), cap);
        body["max_tokens"] = json!(cap);
        true
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

    /// A capped answer that runs out of tokens is retried once with twice
    /// the room.  If it is still cut off it comes back empty: callers treat
    /// an empty reply as unusable, where a truncated one would pass as
    /// complete.
    fn call(&self, prompt: &str, _stop: Option<&[String]>, options: CallOptions) -> anyhow::Result<String> {
        let messages = self.create_messages_from_prompt(prompt);
        let Some(max_tokens) = self.request_max_tokens(options) else {
            let (content, finish) = self.complete(&messages, None)?;
            if finish.as_deref() == Some("length") {
                tracing::warn!(target: LOG_TARGET, "[cloud] {} ran out of context mid-answer", self.model_name);
            }
            return Ok(content);
        };
        let (content, finish) = self.complete(&messages, Some(max_tokens))?;
        if finish.as_deref() != Some("length") {
            return Ok(content);
        }
        // Cut off at the cap: once more with twice the room, as far as the
        // endpoint allows. At its limit already, the answer is what it is.
        let more = self.within_cap(max_tokens * 2);
        if more <= max_tokens {
            tracing::warn!(
                target: LOG_TARGET,
                "[cloud] {} hit max_tokens={max_tokens}, the most an answer may have — keeping the truncated answer",
                self.model_name
            );
            return Ok(content);
        }
        tracing::warn!(
            target: LOG_TARGET,
            "[cloud] {} hit max_tokens={max_tokens} ({} chars answered) — retrying with {}",
            self.model_name,
            content.chars().count(),
            more
        );
        let (content, finish) = self.complete(&messages, Some(more))?;
        if finish.as_deref() != Some("length") {
            return Ok(content);
        }
        tracing::warn!(
            target: LOG_TARGET,
            "[cloud] {} hit max_tokens={} again — discarding the truncated answer",
            self.model_name,
            more
        );
        Ok(String::new())
    }

    fn stream<'a>(&'a self, prompt: &str, options: CallOptions) -> GenerationStream<'a> {
        let messages = self.create_messages_from_prompt(prompt);
        let response = match self.post(&self.body(&messages, true, self.request_max_tokens(options))) {
            Ok(response) => response,
            Err(message) => std::panic::panic_any(message),
        };
        let mut lines = BufReader::new(response).lines();
        // Characters of answer and of reasoning seen, to say where the
        // room went when the stream is cut off.
        let (mut answered, mut reasoned) = (0usize, 0usize);
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
            reasoned += ["/choices/0/delta/reasoning_content", "/choices/0/delta/reasoning"]
                .iter()
                .filter_map(|p| chunk.pointer(p).and_then(Value::as_str))
                .map(|s| s.chars().count())
                .sum::<usize>();
            // Text already streamed can't be taken back, so a cut-off stream
            // is only reported, with where the room went.
            if chunk.pointer("/choices/0/finish_reason").and_then(Value::as_str) == Some("length") {
                tracing::warn!(
                    target: LOG_TARGET,
                    "[cloud] {} stream was cut off at its token limit ({answered} chars of answer, {reasoned} of reasoning)",
                    self.model_name
                );
            }
            match chunk_text(&chunk) {
                Ok(Some(text)) => {
                    answered += text.chars().count();
                    return Some(GenerationChunk { text });
                }
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
