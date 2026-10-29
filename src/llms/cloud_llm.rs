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

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use regex::Regex;
use serde_json::{json, Value};

use crate::config::app_settings::AppSettings;
use crate::llms::base_llm::{
    clean_reply, BaseLLM, BatchPrompt, CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream,
};
use crate::platform::client::{AuthHeadersFn, Headers, RefreshFn, SLPPlatformClient};
use crate::types::ReasoningEffort;

const LOG_TARGET: &str = "smartloop.llms.cloud";

/// Context window assumed when the catalog gives none.
pub const DEFAULT_CONTEXT_LENGTH: usize = 32_768;

/// Reasoning allowance added to a capped request when the catalog gives
/// none.  Reasoning models count their thinking against `max_tokens`, and
/// the ones measured spent 50-350 tokens before answering a short prompt.
pub const DEFAULT_REASONING_BUDGET_TOKENS: usize = 1024;

/// Most answer tokens one packed batch request asks for, summed over its
/// prompts' caps.  The catalog gives no output limit; the long-window models
/// it lists answer at least this much.
pub const PACK_MAX_ANSWER_TOKENS: usize = 32_768;

/// Characters per token assumed when sizing a packed request against the
/// window.  Tighter than [`BaseLLM::get_num_tokens`]'s four: dense text
/// (tables, JSON) tokenizes closer to two or three, and a request that
/// overflows the window fails outright.
const PACK_CHARS_PER_TOKEN: usize = 3;

static ANSWER_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?s)<answer id="(\d+)">(.*?)</answer>"#).expect("valid regex"));

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
    reasoning_budget_tokens: usize,
    /// Efforts the catalog lists for the model, lowest first.
    reasoning_efforts: Vec<ReasoningEffort>,
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
            reasoning_budget_tokens: DEFAULT_REASONING_BUDGET_TOKENS,
            reasoning_efforts: Vec::new(),
            client: SLPPlatformClient::with_hooks(url, auth_headers_fn, refresh_fn),
            send_credentials,
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

    /// Use the catalog's reasoning efforts.
    pub fn with_reasoning_efforts(mut self, mut efforts: Vec<ReasoningEffort>) -> Self {
        efforts.sort();
        efforts.dedup();
        self.reasoning_efforts = efforts;
        self
    }

    /// The effort to send for a call asking `requested`: the lowest one the
    /// model lists at or above it, or its highest when it lists none that
    /// high.  So asking for `none` gets the least thinking a model allows
    /// (`minimal` on one that always reasons).  `None` for a model that
    /// lists no efforts, leaving the platform's default.
    fn reasoning_effort(&self, requested: Option<ReasoningEffort>) -> Option<ReasoningEffort> {
        let requested = requested?;
        let effort = self
            .reasoning_efforts
            .iter()
            .copied()
            .find(|e| *e >= requested)
            .or_else(|| self.reasoning_efforts.last().copied());
        if effort.is_some_and(|e| e != requested) {
            tracing::debug!(
                target: LOG_TARGET,
                "[cloud] {} takes no '{}' reasoning effort; sending '{}'",
                self.model_name,
                requested.value(),
                effort.map_or("", ReasoningEffort::value)
            );
        }
        effort
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn url(&self) -> &str {
        self.client.base_url()
    }

    /// The `max_tokens` to send for a caller's cap.  The cap sizes the
    /// answer; the model's reasoning comes on top, or a short cap is spent
    /// thinking and the answer comes back empty.  With reasoning off there
    /// is nothing to add.
    fn request_max_tokens(&self, max_tokens: Option<usize>, effort: Option<ReasoningEffort>) -> Option<usize> {
        let budget = if effort == Some(ReasoningEffort::None) { 0 } else { self.reasoning_budget_tokens };
        max_tokens.map(|cap| cap + budget)
    }

    fn body(
        &self,
        messages: &[ChatMessage],
        stream: bool,
        max_tokens: Option<usize>,
        effort: Option<ReasoningEffort>,
    ) -> Value {
        self.body_for(json!(messages), stream, max_tokens, effort)
    }

    /// The request body for already-shaped OpenAI `messages`.
    fn body_for(
        &self,
        messages: Value,
        stream: bool,
        max_tokens: Option<usize>,
        effort: Option<ReasoningEffort>,
    ) -> Value {
        let mut body = json!({
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "project_id": self.project_id,
        });
        if let Some(max_tokens) = max_tokens {
            body["max_tokens"] = json!(max_tokens);
        }
        if let Some(effort) = effort {
            body["reasoning"] = json!({"effort": effort.value()});
        }
        body
    }

    /// One non-streaming completion: the answer and why it stopped.
    fn complete(
        &self,
        messages: &[ChatMessage],
        max_tokens: Option<usize>,
        effort: Option<ReasoningEffort>,
    ) -> anyhow::Result<(String, Option<String>)> {
        let response = self.post(&self.body(messages, false, max_tokens, effort)).map_err(anyhow::Error::msg)?;
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
        let max_tokens = self.request_max_tokens(max_tokens, None);
        let mut body = self.body_for(Value::Array(messages), false, max_tokens, None);
        body["temperature"] = json!(temperature);
        let response = self.post(&body).map_err(anyhow::Error::msg)?;
        response
            .json()
            .map_err(|e| anyhow::anyhow!("{} sent an unreadable answer: {e}", self.model_name))
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

    /// One batch prompt on its own request, answered as `invoke` answers.
    fn invoke_one(&self, prompt: &BatchPrompt) -> String {
        self.invoke_with(
            &prompt.prompt,
            CallOptions { max_tokens: prompt.max_tokens, thinking: prompt.thinking, reasoning_effort: prompt.reasoning_effort },
        )
    }

    /// Split `prompts` into the groups [`BaseLLM::invoke_batch`] sends as one
    /// request each, in order.  Consecutive prompts share a request when
    /// they carry the same instructions and options and fit the window
    /// together, with their summed answer caps within
    /// [`PACK_MAX_ANSWER_TOKENS`].  A prompt with no cap, or more than a
    /// system and a user message, goes alone.
    fn pack(&self, prompts: &[BatchPrompt]) -> Vec<Vec<PackTask>> {
        let mut groups: Vec<Vec<PackTask>> = Vec::new();
        let mut used = 0usize;
        for (index, prompt) in prompts.iter().enumerate() {
            let task = PackTask::new(index, prompt, self.create_messages_from_prompt(&prompt.prompt));
            let fits = |group: &[PackTask]| {
                let first = &group[0];
                let answer = group.iter().map(|t| t.cap).sum::<usize>() + task.cap;
                let effort = self.reasoning_effort(first.options.reasoning_effort);
                let output = self.request_max_tokens(Some(answer), effort).unwrap_or(answer);
                first.packable()
                    && task.packable()
                    && first.system == task.system
                    && first.options == task.options
                    && answer <= PACK_MAX_ANSWER_TOKENS
                    && used + task.tokens + output <= self.context_length
            };
            match groups.last_mut() {
                Some(group) if fits(group) => {
                    used += task.tokens;
                    group.push(task);
                }
                _ => {
                    used = task.system.as_deref().map_or(0, pack_tokens) + task.tokens;
                    groups.push(vec![task]);
                }
            }
        }
        groups
    }

    /// Ask for `tasks` in one request.  Each answer comes back under its
    /// task's position; `None` for one the model left out or was cut off
    /// before finishing.
    fn invoke_packed(&self, tasks: &[PackTask]) -> anyhow::Result<Vec<Option<String>>> {
        let first = &tasks[0];
        let preamble = format!(
            "This request holds {} separate tasks, each in a <task id=\"N\"> block. Do each task on its own, \
             as if it were the only one: nothing in one task may inform another task's answer. Answer every \
             task, in order, inside <answer id=\"N\"></answer> carrying that task's id, and write nothing \
             outside those blocks.",
            tasks.len()
        );
        let system = match &first.system {
            Some(system) => format!("{system}\n\n{preamble}"),
            None => preamble,
        };
        let user = tasks
            .iter()
            .enumerate()
            .map(|(id, task)| format!("<task id=\"{}\">\n{}\n</task>", id + 1, task.user))
            .collect::<Vec<_>>()
            .join("\n\n");
        let messages = [ChatMessage::new("system", system), ChatMessage::new("user", user)];
        let effort = self.reasoning_effort(first.options.reasoning_effort);
        let answer: usize = tasks.iter().map(|t| t.cap).sum();
        let (content, finish) = self.complete(&messages, self.request_max_tokens(Some(answer), effort), effort)?;
        if finish.as_deref() == Some("length") {
            tracing::warn!(
                target: LOG_TARGET,
                "[cloud] {} hit its token limit answering {} packed prompts",
                self.model_name,
                tasks.len()
            );
        }
        let mut answers = vec![None; tasks.len()];
        for caps in ANSWER_PATTERN.captures_iter(&content) {
            let id = caps[1].parse::<usize>().unwrap_or(0);
            if let Some(slot) = id.checked_sub(1).and_then(|i| answers.get_mut(i)) {
                *slot = Some(caps[2].trim().to_string());
            }
        }
        Ok(answers)
    }
}

/// One prompt of a batch, split for packing.
struct PackTask {
    /// Position in the batch.
    index: usize,
    system: Option<String>,
    user: String,
    /// The prompt's answer cap; `0` when it has none.
    cap: usize,
    options: CallOptions,
    /// Estimated tokens of `user` in a packed request.
    tokens: usize,
    /// Whether the prompt is a system and a user message, or a user message alone.
    simple: bool,
}

impl PackTask {
    fn new(index: usize, prompt: &BatchPrompt, messages: Vec<ChatMessage>) -> Self {
        let options = CallOptions {
            max_tokens: prompt.max_tokens,
            thinking: prompt.thinking,
            reasoning_effort: prompt.reasoning_effort,
        };
        let (system, user, simple) = match messages.as_slice() {
            [system, user] if system.role == "system" && user.role == "user" => {
                (Some(system.content.clone()), user.content.clone(), true)
            }
            [user] if user.role == "user" => (None, user.content.clone(), true),
            _ => (None, String::new(), false),
        };
        // The task's wrapper and answer tags ride on top of its text.
        let tokens = pack_tokens(&user) + 16;
        Self { index, system, user, cap: prompt.max_tokens.unwrap_or(0), options, tokens, simple }
    }

    fn packable(&self) -> bool {
        self.simple && self.cap > 0
    }
}

/// Estimated tokens of `text` when sizing a packed request.
fn pack_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(PACK_CHARS_PER_TOKEN)
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
        let effort = self.reasoning_effort(options.reasoning_effort);
        let Some(max_tokens) = self.request_max_tokens(options.max_tokens, effort) else {
            let (content, finish) = self.complete(&messages, None, effort)?;
            if finish.as_deref() == Some("length") {
                tracing::warn!(target: LOG_TARGET, "[cloud] {} ran out of context mid-answer", self.model_name);
            }
            return Ok(content);
        };
        let (content, finish) = self.complete(&messages, Some(max_tokens), effort)?;
        if finish.as_deref() != Some("length") {
            return Ok(content);
        }
        tracing::warn!(
            target: LOG_TARGET,
            "[cloud] {} hit max_tokens={max_tokens} ({} chars answered) — retrying with {}",
            self.model_name,
            content.chars().count(),
            max_tokens * 2
        );
        let (content, finish) = self.complete(&messages, Some(max_tokens * 2), effort)?;
        if finish.as_deref() != Some("length") {
            return Ok(content);
        }
        tracing::warn!(
            target: LOG_TARGET,
            "[cloud] {} hit max_tokens={} again — discarding the truncated answer",
            self.model_name,
            max_tokens * 2
        );
        Ok(String::new())
    }

    fn stream<'a>(&'a self, prompt: &str, options: CallOptions) -> GenerationStream<'a> {
        let messages = self.create_messages_from_prompt(prompt);
        let effort = self.reasoning_effort(options.reasoning_effort);
        let max_tokens = self.request_max_tokens(options.max_tokens, effort);
        let response = match self.post(&self.body(&messages, true, max_tokens, effort)) {
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
            // Text already streamed can't be taken back, so a cut-off stream
            // is only reported.
            if chunk.pointer("/choices/0/finish_reason").and_then(Value::as_str) == Some("length") {
                tracing::warn!(target: LOG_TARGET, "[cloud] {} stream was cut off at its token limit", self.model_name);
            }
            match chunk_text(&chunk) {
                Ok(Some(text)) => return Some(GenerationChunk { text }),
                Ok(None) => continue,
                Err(message) => std::panic::panic_any(message),
            }
        }))
    }

    /// Prompts sharing their instructions go out together, as many to a
    /// request as fit (see [`CloudLLM::pack`]): a request per prompt is a
    /// round trip and a quota hold each, run one after another.  A prompt the
    /// packed reply leaves unanswered is asked again on its own, and a failed
    /// request answers each of its prompts with the error, as `invoke` would.
    fn invoke_batch(&self, prompts: &[BatchPrompt], cancelled: Option<CancelEvent>) -> Vec<String> {
        let is_cancelled = || cancelled.as_ref().is_some_and(|c| c.load(Ordering::SeqCst));
        let mut replies = vec![String::new(); prompts.len()];
        for group in self.pack(prompts) {
            if is_cancelled() {
                break;
            }
            if let [task] = group.as_slice() {
                replies[task.index] = self.invoke_one(&prompts[task.index]);
                continue;
            }
            match self.invoke_packed(&group) {
                Ok(answers) => {
                    let missing = answers.iter().filter(|a| a.is_none()).count();
                    tracing::info!(
                        target: LOG_TARGET,
                        "[cloud] {} answered {} packed prompts in one request ({missing} to ask again)",
                        self.model_name,
                        group.len() - missing
                    );
                    for (task, answer) in group.iter().zip(answers) {
                        replies[task.index] = match answer {
                            Some(answer) => clean_reply(&answer),
                            None if is_cancelled() => String::new(),
                            None => self.invoke_one(&prompts[task.index]),
                        };
                    }
                }
                Err(e) => {
                    tracing::error!(target: LOG_TARGET, "LLM invocation failed: {e}");
                    for task in &group {
                        replies[task.index] = format!("Error generating response: {e}");
                    }
                }
            }
        }
        replies
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
