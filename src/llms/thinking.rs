//! Reasoning-block ("thinking") handling for models that emit one.
//!
//! Port of `smartloop/llms/thinking.py`.
//!
//! Hybrid-reasoning models answer in two parts: an internal monologue, then the
//! reply meant for the reader.  Only the reply may reach callers — everything
//! downstream of the LLM wrapper (the JSON planner, the DAG, retrieval, the chat
//! SSE stream) treats the generated text as the answer, so an unfiltered
//! monologue is displayed to the user and parsed as data.
//!
//! Reasoning is requested per model by `SLPModel::thinking_default` (and per
//! request from the chat endpoint), which sets the `enable_thinking` variable the
//! model's own chat template reads; this module removes what it produces.
//!
//! The delimiters are per-family, read off the templates in the GGUFs we ship:
//!
//! * `<think>…</think>` — Qwen3.5.
//! * `<|channel>thought…<channel|>` — Gemma 4's thought channel.
//!
//! Filtering runs unconditionally — a model that never emits the tags passes
//! through untouched — so a template that reasons without us asking is handled
//! too.  A close tag ahead of any open tag is also handled, because a template
//! can pre-fill the opening marker into the generation prompt, leaving the model
//! to emit only the close.

/// (open, close) per model family.  Order matters only for readability; the
/// filter always acts on whichever marker appears first in the text.
pub const REASONING_BLOCKS: &[(&str, &str)] = &[("<think>", "</think>"), ("<|channel>", "<channel|>")];

/// Every opening marker, in `REASONING_BLOCKS` order.
pub const OPEN_TAGS: &[&str] = &["<think>", "<|channel>"];

/// Every closing marker, in `REASONING_BLOCKS` order.
pub const CLOSE_TAGS: &[&str] = &["</think>", "<channel|>"];

/// All markers, opening first.
const ALL_TAGS: &[&str] = &["<think>", "<|channel>", "</think>", "<channel|>"];

/// The closing marker that matches `open_tag`.
fn close_for(open_tag: &str) -> &'static str {
    REASONING_BLOCKS
        .iter()
        .find(|(open, _)| *open == open_tag)
        .map(|(_, close)| *close)
        .expect("open tag must come from OPEN_TAGS")
}

/// Byte index and identity of the earliest of `tags` in `text`.
fn first_of(text: &str, tags: &[&'static str]) -> Option<(usize, &'static str)> {
    let mut best: Option<(usize, &'static str)> = None;
    for &tag in tags {
        if let Some(at) = text.find(tag) {
            if best.is_none_or(|(b, _)| at < b) {
                best = Some((at, tag));
            }
        }
    }
    best
}

/// Length (in bytes) of the trailing run of `text` that could still become a tag.
///
/// Streamed chunks split anywhere, so `"<thi"` at the end of a chunk may be the
/// start of `<think>` — it has to be held back rather than emitted.  Tags are
/// ASCII, so any matching tail is ASCII and the cut is a char boundary.
fn partial_tag_len(text: &str, tags: &[&str]) -> usize {
    let longest = tags.iter().map(|t| t.len()).max().unwrap_or(0);
    let max = longest.saturating_sub(1).min(text.len());
    for size in (1..=max).rev() {
        let start = text.len() - size;
        if !text.is_char_boundary(start) {
            continue;
        }
        let tail = &text[start..];
        if tags.iter().any(|t| t.starts_with(tail)) {
            return size;
        }
    }
    0
}

/// Incremental filter that drops reasoning blocks from generated text.
///
/// Feed it chunks in order; each [`ThinkingFilter::feed`] returns the text that
/// is safe to show.  Call [`ThinkingFilter::flush`] once at the end to release
/// anything held back (a trailing partial tag that turned out to be ordinary
/// text).
#[derive(Debug, Clone, Default)]
pub struct ThinkingFilter {
    buf: String,
    /// Close tag we are inside of.
    awaiting: Option<&'static str>,
    thought: Vec<String>,
    emitted: bool,
    /// True once any reasoning marker has been seen.
    pub saw_thinking: bool,
}

impl ThinkingFilter {
    /// A fresh filter.
    pub fn new() -> Self {
        Self::default()
    }

    /// True when generation stopped inside a reasoning block.
    ///
    /// Happens when `max_tokens` cuts the model off mid-thought; the reply it
    /// was reasoning toward was never produced.
    pub fn unterminated(&self) -> bool {
        self.awaiting.is_some()
    }

    /// Feed the next chunk; returns the visible text it releases.
    pub fn feed(&mut self, chunk: &str) -> String {
        if chunk.is_empty() {
            return String::new();
        }
        self.buf.push_str(chunk);
        let mut out = String::new();

        while !self.buf.is_empty() {
            if let Some(awaiting) = self.awaiting {
                match self.buf.find(awaiting) {
                    None => {
                        // Withhold the reasoning, keeping only what could still
                        // turn out to be a split closing tag.
                        let hold = partial_tag_len(&self.buf, &[awaiting]);
                        let keep = self.buf.len() - hold;
                        self.thought.push(self.buf[..keep].to_string());
                        self.buf = self.buf[keep..].to_string();
                        break;
                    }
                    Some(at) => {
                        self.thought.push(self.buf[..at].to_string());
                        self.buf = self.buf[at + awaiting.len()..].to_string();
                        self.awaiting = None;
                        continue;
                    }
                }
            }

            let open = first_of(&self.buf, OPEN_TAGS);
            let close = first_of(&self.buf, CLOSE_TAGS);
            // A close tag ahead of any open tag means the chat template
            // pre-filled the opening marker into the prompt: everything up to
            // it is reasoning, not answer.
            if let Some((close_at, close_tag)) = close {
                if open.is_none_or(|(open_at, _)| close_at < open_at) {
                    self.thought.push(self.buf[..close_at].to_string());
                    self.buf = self.buf[close_at + close_tag.len()..].to_string();
                    self.saw_thinking = true;
                    continue;
                }
            }
            match open {
                None => {
                    let hold = partial_tag_len(&self.buf, ALL_TAGS);
                    let keep = self.buf.len() - hold;
                    out.push_str(&self.buf[..keep]);
                    self.buf = self.buf[keep..].to_string();
                    break;
                }
                Some((open_at, open_tag)) => {
                    out.push_str(&self.buf[..open_at]);
                    self.buf = self.buf[open_at + open_tag.len()..].to_string();
                    self.awaiting = Some(close_for(open_tag));
                    self.saw_thinking = true;
                }
            }
        }

        if !out.trim().is_empty() {
            self.emitted = true;
        }
        out
    }

    /// Release held-back text once generation has ended.
    ///
    /// When the model never closed its reasoning block and produced no reply,
    /// the withheld monologue is returned rather than nothing: a truncated
    /// thought is poor output, an empty string is a failed request.
    pub fn flush(&mut self) -> String {
        if self.awaiting.is_some() {
            if self.emitted {
                return String::new();
            }
            return self.thought.concat().trim().to_string();
        }
        std::mem::take(&mut self.buf)
    }
}

/// Return `text` with any reasoning block removed.
pub fn strip_thinking(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let mut f = ThinkingFilter::new();
    let mut out = f.feed(text);
    out.push_str(&f.flush());
    out.trim().to_string()
}
