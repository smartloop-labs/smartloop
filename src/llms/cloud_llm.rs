//! A remote catalog model, answered over HTTP.
//!
//! The prompt is split back into chat messages and sent to the model's
//! provider with the short-lived key a
//! [`cloud_session`](crate::llms::cloud_session) holds.  The platform only
//! issues the key and settles what it spent; it never sees the chat.
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
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde_json::{json, Value};

use crate::config::app_settings::AppSettings;
use crate::llms::base_llm::{BaseLLM, CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream};
use crate::llms::cloud_session::{self, SessionModel};
use crate::llms::tool_calls::{openai_messages, parse_tool_calls, render_tool_calls, split_messages, split_tools, NativeToolCall};
use crate::model::spec::MODEL_SPECS;
use crate::platform::client::{AuthHeadersFn, RefreshFn, SLPPlatformClient};
use crate::types::{ModelCapability, ModelKind};

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
/// app's, under the provider's model id, and the answer held to the server's
/// cap.
fn direct_body(model: &SessionModel, body: &Value) -> Value {
    let mut out = Value::Object(model.body.clone());
    if let Value::Object(fields) = body {
        for (name, value) in fields {
            out[name] = value.clone();
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

/// Waits before retrying a key the provider refused while it is new (see
/// [`cloud_session::KEY_PROPAGATION`]), in units of [`KEY_WAIT_UNIT_MS`]:
/// 1s, 2s, 3s — six seconds in all.
const KEY_PROPAGATION_WAITS: [u64; 3] = [1, 2, 3];

/// Milliseconds per unit of [`KEY_PROPAGATION_WAITS`]; tests shorten it.
static KEY_WAIT_UNIT_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1000);

/// Shorten the new-key waits (tests only).
#[doc(hidden)]
pub fn set_key_wait_unit_ms(ms: u64) {
    KEY_WAIT_UNIT_MS.store(ms, std::sync::atomic::Ordering::SeqCst);
}

/// A remote model as a [`BaseLLM`].
pub struct CloudLLM {
    project_id: String,
    model_name: String,
    context_length: usize,
    reasoning_budget_tokens: usize,
    /// Where the catalog says the model is served; only keys learned answer
    /// caps.  The provider to call comes from the session.
    url: String,
    /// The platform API the session keys come from, under the user's own
    /// credentials.  Those never leave for the provider.
    platform: SLPPlatformClient,
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
        Self {
            project_id: project_id.to_string(),
            model_name: model_name.to_string(),
            context_length: context_length.map(|n| n as usize).filter(|n| *n > 0).unwrap_or(DEFAULT_CONTEXT_LENGTH),
            reasoning_budget_tokens: DEFAULT_REASONING_BUDGET_TOKENS,
            url: url.trim_end_matches('/').to_string(),
            platform: SLPPlatformClient::with_hooks(&api_base, auth_headers_fn, refresh_fn),
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
        &self.url
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
        format!("{}|{}", self.url, self.model_name)
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
        });
        if let Some(max_tokens) = max_tokens {
            body["max_tokens"] = json!(max_tokens);
        }
        body
    }

    /// One non-streaming completion: the answer and why it stopped.
    ///
    /// With `tools`, the request offers them and a structured `tool_calls`
    /// reply comes back as text in the canonical
    /// `<tool_call>{…}</tool_call>` shape (see [`crate::llms::tool_calls`]),
    /// so a caller reads a local and a remote model's call the same way.
    fn complete(
        &self,
        messages: &Value,
        max_tokens: Option<usize>,
        tools: Option<&[Value]>,
    ) -> anyhow::Result<(String, Option<String>)> {
        let mut body = self.body_for(messages.clone(), false, max_tokens);
        if let Some(tools) = tools {
            body["tools"] = Value::Array(tools.to_vec());
            body["tool_choice"] = json!("auto");
        }
        let response = self.post(&body).map_err(anyhow::Error::msg)?;
        let data: Value = response
            .json()
            .map_err(|e| anyhow::anyhow!("{} sent an unreadable answer: {e}", self.model_name))?;
        let content = data.pointer("/choices/0/message/content").and_then(Value::as_str).unwrap_or_default();
        let finish = data.pointer("/choices/0/finish_reason").and_then(Value::as_str);
        let calls: Vec<NativeToolCall> = data
            .pointer("/choices/0/message/tool_calls")
            .and_then(Value::as_array)
            .map(|calls| calls.iter().flat_map(|c| parse_tool_calls(&c.to_string())).collect())
            .unwrap_or_default();
        if !calls.is_empty() {
            return Ok((render_tool_calls(&calls), finish.map(str::to_string)));
        }
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

    /// Send `body` to the provider; refused for asking more answer tokens
    /// than it allows, it learns the cap and sends again within it.
    fn post(&self, body: &Value) -> Result<reqwest::blocking::Response, String> {
        let http = reqwest::blocking::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|e| format!("Could not reach {}: {e}", self.model_name))?;
        self.post_direct(&http, &self.platform, body)
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
        let mut young_retries = 0usize;
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
            // A key minted seconds ago is refused until the provider has
            // caught up with it: wait that out with the same key, since a
            // replacement reserves another budget and is just as new.
            if status == 401 && session.is_new() && young_retries < KEY_PROPAGATION_WAITS.len() {
                let unit = KEY_WAIT_UNIT_MS.load(std::sync::atomic::Ordering::SeqCst);
                let wait = Duration::from_millis(KEY_PROPAGATION_WAITS[young_retries] * unit);
                young_retries += 1;
                tracing::info!(
                    target: LOG_TARGET,
                    "[cloud] {} key is new — retrying the same key in {:.1}s",
                    self.model_name,
                    wait.as_secs_f64()
                );
                std::thread::sleep(wait);
                continue;
            }
            if matches!(status, 401 | 402 | 403) && !renewed {
                renewed = true;
                session = cloud_session::replace(platform, &session, &self.project_id, &self.model_name)?;
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

/// How a streamed answer arrived, logged once it ends: time to the first
/// token, how many chunks the text came in, and the provider's own token
/// counts with the rate they make over the time the answer streamed.  The
/// reasoning before the first token is counted, but never shown.
struct StreamTiming {
    started: Instant,
    first: Option<Instant>,
    last: Option<Instant>,
    chunks: usize,
    usage: Option<Value>,
    reported: bool,
}

impl StreamTiming {
    fn new(started: Instant) -> Self {
        Self { started, first: None, last: None, chunks: 0, usage: None, reported: false }
    }

    fn chunk(&mut self) {
        let now = Instant::now();
        self.first.get_or_insert(now);
        self.last = Some(now);
        self.chunks += 1;
    }

    fn report(&mut self, model: &str) {
        if std::mem::replace(&mut self.reported, true) {
            return;
        }
        let ms = |d: Duration| d.as_millis();
        let total = self.started.elapsed();
        let Some(first) = self.first else {
            tracing::info!(target: LOG_TARGET, "[cloud] {model} streamed no text in {} ms", ms(total));
            return;
        };
        let streaming = self.last.map_or(Duration::ZERO, |last| last - first);
        let tokens = |key: &str| self.usage.as_ref().and_then(|u| u.pointer(key)).and_then(Value::as_u64);
        let completion = tokens("/completion_tokens");
        let reasoning = tokens("/completion_tokens_details/reasoning_tokens").unwrap_or(0);
        let rate = completion
            .map(|c| c.saturating_sub(reasoning))
            .filter(|_| streaming.as_secs_f64() > 0.0)
            .map(|answer| format!("{:.0} tok/s", answer as f64 / streaming.as_secs_f64()))
            .unwrap_or_else(|| "rate unknown".to_string());
        tracing::info!(
            target: LOG_TARGET,
            "[cloud] {model} first token after {} ms; {} chunk(s) over {} ms; {} answer + {reasoning} reasoning token(s); {rate}; {} ms in all",
            ms(first - self.started),
            self.chunks,
            ms(streaming),
            completion.map_or("?".to_string(), |c| c.saturating_sub(reasoning).to_string()),
            ms(total),
        );
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

    /// True when the catalog declares `tools` for this model: its endpoint
    /// then takes an OpenAI `tools` list and answers with `tool_calls`.
    fn supports_native_tools(&self) -> bool {
        MODEL_SPECS
            .get(&self.model_name)
            .is_some_and(|spec| spec.for_kind(ModelKind::Remote).supports(ModelCapability::Tools))
    }

    /// A capped answer that runs out of tokens is retried once with twice
    /// the room.  If it is still cut off it comes back empty: callers treat
    /// an empty reply as unusable, where a truncated one would pass as
    /// complete.
    fn call(&self, prompt: &str, _stop: Option<&[String]>, options: CallOptions) -> anyhow::Result<String> {
        let (prompt, tools) = split_tools(prompt);
        let tools = tools.filter(|_| self.supports_native_tools());
        let tools = tools.as_deref();
        // A tool conversation travels as OpenAI messages, sent as they are.
        let messages: Value = match split_messages(prompt).filter(|_| tools.is_some()) {
            Some(conversation) => Value::Array(openai_messages(&conversation)),
            None => json!(self.create_messages_from_prompt(prompt)),
        };
        let Some(max_tokens) = self.request_max_tokens(options) else {
            let (content, finish) = self.complete(&messages, None, tools)?;
            if finish.as_deref() == Some("length") {
                tracing::warn!(target: LOG_TARGET, "[cloud] {} ran out of context mid-answer", self.model_name);
            }
            return Ok(content);
        };
        let (content, finish) = self.complete(&messages, Some(max_tokens), tools)?;
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
        let (content, finish) = self.complete(&messages, Some(more), tools)?;
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
        let started = Instant::now();
        let response = match self.post(&self.body(&messages, true, self.request_max_tokens(options))) {
            Ok(response) => response,
            Err(message) => std::panic::panic_any(message),
        };
        let mut lines = BufReader::new(response).lines();
        // Characters of answer and of reasoning seen, to say where the
        // room went when the stream is cut off.
        let (mut answered, mut reasoned) = (0usize, 0usize);
        let mut timing = StreamTiming::new(started);
        Box::new(std::iter::from_fn(move || loop {
            if self.is_cancelled() {
                return None;
            }
            let Some(line) = lines.next() else {
                timing.report(&self.model_name);
                return None;
            };
            let line = match line {
                Ok(line) => line,
                Err(e) => {
                    tracing::warn!(target: LOG_TARGET, "[cloud] {} stream broke off: {e}", self.model_name);
                    std::panic::panic_any(format!("{} stopped answering.", self.model_name));
                }
            };
            let Some(data) = line.strip_prefix("data:").map(str::trim) else { continue };
            if data == "[DONE]" {
                timing.report(&self.model_name);
                return None;
            }
            let Ok(chunk) = serde_json::from_str::<Value>(data) else { continue };
            if let Some(usage) = chunk.get("usage").filter(|u| u.is_object()) {
                timing.usage = Some(usage.clone());
            }
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
                    timing.chunk();
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
