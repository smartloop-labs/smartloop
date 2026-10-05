//! Port of `tests/unit/test_model_display_name.py`.
//!
//! The public model name shown in SSE streams.
//!
//! The orchestrator model is now named `sl-mini` throughout the framework, but
//! the old `slp-orchestra-mini` still turns up — in workspace records written
//! by an earlier build, in `.env` overrides, and inside the `model_id` that
//! keys the on-disk model directory and the HuggingFace repo (which is
//! deliberately never renamed).  Anything a user reads — the chat completion
//! chunks and the /v1/init progress events — shows `sl-mini` regardless.  See
//! tests/unit/test_model_name_aliases.rs for the rest of the rename.

use serde_json::Value;

use smartloop::constants::display_model_name;

// --------------------------------------------------------------------------
// The mapping itself
// --------------------------------------------------------------------------

#[test]
fn test_orchestrator_id_maps_to_public_name() {
    assert_eq!(display_model_name("slp-orchestra-mini"), "sl-mini");
}

/// Workspace models the user added keep whatever name they were given.
#[test]
fn test_unknown_model_passes_through() {
    assert_eq!(display_model_name("qwen3-8b"), "qwen3-8b");
}

/// The Python test is parametrized over `None` and `""`; the Rust function
/// takes a `&str`, so "no name" is the empty string.
#[test]
fn test_empty_name_passes_through() {
    assert_eq!(display_model_name(""), "");
}

// --------------------------------------------------------------------------
// Chat SSE chunks
// --------------------------------------------------------------------------

fn chat_source() -> String {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/api/routers/chat.rs");
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

/// 1-based line of byte offset *pos* in *src*.
fn line_of(src: &str, pos: usize) -> usize {
    src[..pos].matches('\n').count() + 1
}

/// The text between the brace opening at *open* and its matching close.
fn braced(src: &str, open: usize) -> &str {
    let bytes = src.as_bytes();
    let mut depth = 0usize;
    for (i, &b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &src[open + 1..i];
                }
            }
            _ => {}
        }
    }
    &src[open + 1..]
}

/// The top-level `field: value` entries of a struct literal body.
fn top_level_fields(body: &str) -> Vec<(String, String)> {
    let mut fields = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let bytes = body.as_bytes();
    let mut push = |piece: &str| {
        if let Some((name, value)) = piece.split_once(':') {
            let name = name.trim();
            if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                fields.push((name.to_string(), value.trim().to_string()));
            }
        }
    };
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'{' | b'(' | b'[' => depth += 1,
            b'}' | b')' | b']' => depth -= 1,
            b',' if depth == 0 => {
                push(&body[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    push(&body[start..]);
    fields
}

/// Every `ChatCompletionChunk { ... }` struct literal in *src*, as
/// `(line, body)`.  Type positions (`-> ChatCompletionChunk {`, `impl`,
/// `struct`) are not constructions and are skipped.
fn chunk_literals(src: &str) -> Vec<(usize, String)> {
    let needle = "ChatCompletionChunk";
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(rel) = src[from..].find(needle) {
        let at = from + rel;
        from = at + needle.len();
        let before = src[..at].trim_end();
        let preceded_by_ident = src[..at].chars().last().is_some_and(|c| c.is_alphanumeric() || c == '_');
        let after = &src[from..];
        let trimmed = after.trim_start();
        if preceded_by_ident || !trimmed.starts_with('{') {
            continue;
        }
        if before.ends_with("->") || before.ends_with("impl") || before.ends_with("struct") || before.ends_with("for") {
            continue;
        }
        let open = from + (after.len() - trimmed.len());
        found.push((line_of(src, at), braced(src, open).to_string()));
    }
    found
}

/// No `ChatCompletionChunk` may echo the raw requested id.
///
/// `stream_model` is resolved once at the top of the stream; a new chunk that
/// passes `request.model` straight through would leak `slp-orchestra-mini`
/// back to the client mid-stream.
#[test]
fn test_every_sse_chunk_reports_the_display_name() {
    let src = chat_source();
    let chunks = chunk_literals(&src);
    assert!(!chunks.is_empty(), "expected ChatCompletionChunk constructions in chat.rs");

    for (line, body) in chunks {
        let fields = top_level_fields(&body);
        let model = fields.iter().find(|(name, _)| name == "model");
        let (_, value) = model.unwrap_or_else(|| panic!("chunk at line {line} has no model:"));
        let base = value.split(['.', ' ']).next().unwrap_or_default().trim_start_matches('&');
        assert_eq!(
            base, "stream_model",
            "chunk at line {line} must use stream_model, not the raw request id"
        );
    }
}

/// Telemetry keys off the real model id — the alias is display-only.
#[test]
fn test_token_usage_still_reports_the_internal_id() {
    let src = chat_source();
    let mut calls = Vec::new();
    let mut from = 0;
    while let Some(rel) = src[from..].find("push_token_usage(") {
        let open = from + rel + "push_token_usage".len();
        from = open + 1;
        // Balanced parentheses: the argument list.
        let mut depth = 0usize;
        let mut end = open;
        for (i, c) in src[open..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        calls.push(src[open + 1..end].to_string());
    }
    assert!(
        calls.iter().any(|args| args.contains("request.model")),
        "token usage must be reported under request.model"
    );
}

// --------------------------------------------------------------------------
// /v1/init SSE progress events
// --------------------------------------------------------------------------

/// `Loading orchestrator model ...` names sl-mini, not the internal id.
///
/// The Python test replaces `load_model` with a no-op; `ModelService` has no
/// such seam, so the real one runs against an empty, isolated home with the
/// platform pointed at a closed port — it fails fast instead of loading or
/// downloading anything, and every frame the stream produced up to that point
/// is checked.
#[tokio::test]
#[serial_test::serial]
async fn test_init_load_stream_shows_public_name() {
    use futures::StreamExt;
    use smartloop::api::config::API_SETTINGS;
    use smartloop::api::routers::init;
    use smartloop::api::services::ModelService;

    struct Env {
        _home: crate::shared::misc_isolation::IsolatedHome,
        saved_api_base: Option<String>,
        saved_default_model: String,
    }
    impl Drop for Env {
        fn drop(&mut self) {
            API_SETTINGS.write().default_model = self.saved_default_model.clone();
            // SAFETY: serial test; nothing else reads the environment meanwhile.
            unsafe {
                match &self.saved_api_base {
                    Some(v) => std::env::set_var("SLP_API_BASE", v),
                    None => std::env::remove_var("SLP_API_BASE"),
                }
            }
        }
    }

    let home = crate::shared::misc_isolation::IsolatedHome::new();
    let saved_api_base = std::env::var("SLP_API_BASE").ok();
    // SAFETY: serial test; nothing else reads the environment meanwhile.
    unsafe { std::env::set_var("SLP_API_BASE", "http://127.0.0.1:9") };
    let saved_default_model = API_SETTINGS.read().default_model.clone();
    let _env = Env { _home: home, saved_api_base, saved_default_model };
    API_SETTINGS.write().default_model = "slp-orchestra-mini".to_string();

    let model_service = ModelService::new_bare();
    *model_service.model_name.write() = Some("slp-orchestra-mini".to_string());

    let mut messages: Vec<String> = Vec::new();
    let mut stream = init::load_orchestrator_and_stream(model_service);
    let collect = async {
        while let Some(item) = stream.next().await {
            let Ok(event) = item else { break };
            for line in event.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    let frame: Value = serde_json::from_str(data).expect("data is JSON");
                    messages.push(frame.get("message").and_then(Value::as_str).unwrap_or("").to_string());
                }
            }
        }
    };
    tokio::time::timeout(std::time::Duration::from_secs(120), collect)
        .await
        .expect("the load stream finished");

    let joined = messages.join("\n");
    assert!(joined.contains("sl-mini"), "{joined}");
    assert!(!joined.contains("slp-orchestra-mini"), "{joined}");
}
