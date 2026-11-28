//! The framework's own tools: `web_search` and `web_fetch`.
//!
//! They live here, with the agents that do their work, so every client gets
//! them from one place: the chat endpoint offers them to the model with each
//! turn (always: whether to search is the model's to decide, not a switch's)
//! and runs the calls itself (see `api::routers::chat`), and a client
//! that links the framework can take the same definitions and run them in
//! process.  A client extends the set by sending tools of its own with the
//! request, and replaces one of these by sending a tool of the same name,
//! which it then runs itself.
//!
//! `web_search` asks Smartloop's hosted `/v1/websearch` for links (signed in,
//! within the month's allowance) through [`WebSearchAgent`], then reads the
//! top pages as markdown.  `web_fetch` reads one page the model already has
//! the URL of, through [`FetchAgent`] (HTML or PDF to markdown).
//!
//! Neither fails: why a search or a page couldn't be had is the tool's
//! result, so the model goes on without it.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::agents::fetch_agent::FetchAgent;
use crate::agents::web_search_agent::{ProgressCallback, WebSearchAgent};
use crate::search::providers::SearchProviderError;

/// The search tool's name, as the model calls it.
pub const WEB_SEARCH: &str = "web_search";
/// The fetch tool's name, as the model calls it.
pub const WEB_FETCH: &str = "web_fetch";

/// Pages handed back for one search.
const TOP: usize = 3;
/// Links asked for: spares for pages that won't load.  Only fetched as
/// needed, so spares cost nothing unless a top page fails.
const CANDIDATES: usize = 10;
/// Characters of a page one `web_fetch` call gives back; the rest is had
/// with `start`.
pub const PAGE: usize = 12_000;

/// The search tool, as offered to the model.
pub fn web_search_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": WEB_SEARCH,
            "description": "Search the web and read the top pages, for what the user's own material and your own knowledge don't tell or may have out of date: current events, prices, a product's or library's latest version, an error message, documentation. Returns the pages' text with their URLs; cite the URLs you use.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for, as you'd type it into a search engine"},
                },
                "required": ["query"],
            },
        },
    })
}

/// The fetch tool, as offered to the model.
pub fn web_fetch_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": WEB_FETCH,
            "description": format!("Read one web page or PDF as markdown, by its URL: a page the user gave, or one `{WEB_SEARCH}` turned up that you need more of. Gives up to {PAGE} characters; a longer page says where to go on from, with `start`."),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The page's full http(s) URL"},
                    "start": {"type": "integer", "description": "Character to start from, for the rest of a long page; 0 by default"},
                },
                "required": ["url"],
            },
        },
    })
}

/// Every tool the framework has, as offered to the model.
pub fn definitions() -> Vec<Value> {
    vec![web_search_tool(), web_fetch_tool()]
}

/// Whether `name` is one of the framework's tools.
pub fn is_builtin(name: &str) -> bool {
    name == WEB_SEARCH || name == WEB_FETCH
}

/// The name a tool definition or a call gives its function.
pub fn tool_name(tool: &Value) -> &str {
    tool.pointer("/function/name").and_then(Value::as_str).unwrap_or_default()
}

/// The tools a turn offers the model: the client's own, then the framework's.
///
/// The client's come first and win a clash of names, which is how a client
/// replaces a tool: the framework's tool of that name is left out, and the
/// call goes back to the client like any other of its own.  The second value
/// is what the framework runs itself: the names of its tools that made it in.
pub fn offered(client: &[Value], framework: Vec<Value>) -> (Vec<Value>, Vec<String>) {
    let mut tools = client.to_vec();
    let mut ours = Vec::new();
    for tool in framework {
        let name = tool_name(&tool).to_string();
        if name.is_empty() || client.iter().any(|t| tool_name(t) == name) {
            continue;
        }
        ours.push(name);
        tools.push(tool);
    }
    (tools, ours)
}

/// What the system prompt says of the framework's tools on offer (`ours`,
/// the names it runs this turn), or `None` when it runs none.
///
/// The definitions alone don't do it: the system prompt has the model answer
/// from what it knows, and a small model reads that as "I can't browse" and
/// never looks at the tools.  So it is told outright that it can.
pub fn guidance(ours: &[String]) -> Option<String> {
    let has = |name: &str| ours.iter().any(|n| n == name);
    let mut lines = Vec::new();
    if has(WEB_SEARCH) {
        lines.push(format!(
            "WEB: You can search the web. For anything current (news, events, weather, prices, scores, schedules, \
             releases, versions) or anything you aren't sure of, call the `{WEB_SEARCH}` tool before you answer, and \
             answer from what it returns, citing the URLs. Never say you can't browse the internet or lack real-time \
             access: search instead. One search is usually enough: answer from its results, and search or read \
             again only when they don't answer the question. Don't search for what the conversation or the provided \
             data already answers."
        ));
    }
    if has(WEB_FETCH) {
        lines.push(format!("To read a page or PDF whose URL you have, call the `{WEB_FETCH}` tool."));
    }
    if lines.is_empty() { None } else { Some(lines.join(" ")) }
}

/// `prompt` (the flattened turn, opening with its `<system>` block) with the
/// [`guidance`] for `ours` added to the end of that block.
pub fn with_guidance(prompt: &str, ours: &[String]) -> String {
    let Some(guidance) = guidance(ours) else { return prompt.to_string() };
    match prompt.find("</system>") {
        Some(at) => format!("{}\n{}{}", &prompt[..at], guidance, &prompt[at..]),
        None => format!("<system>{guidance}</system>\n{prompt}"),
    }
}

/// `result` cut to `max_chars`, saying so: a tool's result has to leave the
/// model room to answer in.
pub fn clip(result: String, max_chars: usize) -> String {
    if result.chars().count() <= max_chars {
        return result;
    }
    let kept: String = result.chars().take(max_chars).collect();
    format!("{}\n\n[Cut here to fit. Call `{WEB_FETCH}` on a URL above for more of that page.]", kept.trim_end())
}

/// What to say a call is doing, for the client's progress line.
pub fn describe(name: &str, arguments: &str) -> String {
    match name {
        WEB_SEARCH => match query(arguments) {
            Ok(q) => format!("Searching the web for \"{q}\""),
            Err(_) => "Searching the web".to_string(),
        },
        WEB_FETCH => match fetch_args(arguments) {
            Ok((url, _)) => format!("Reading {}", WebSearchAgent::source_label(&url)),
            Err(_) => "Reading a page".to_string(),
        },
        other => format!("Calling {other}"),
    }
}

/// Run the framework's tool `name` with the model's `arguments` (a JSON
/// object, as text): what goes back to the model.  Blocking, a network round
/// trip a page: run it off the async workers.  `said` hears each step as it
/// happens ("Reading docs.rs").
pub fn run(name: &str, arguments: &str, said: impl Fn(&str) + Send + Sync + 'static) -> String {
    match name {
        WEB_SEARCH => match query(arguments) {
            Ok(q) => search(&q, said),
            Err(e) => e,
        },
        WEB_FETCH => match fetch_args(arguments) {
            Ok((url, start)) => fetch(&url, start),
            Err(e) => e,
        },
        other => format!("Error: there is no `{other}` tool."),
    }
}

/// The query a call of `arguments` asks for, or what to tell the model.
pub fn query(arguments: &str) -> Result<String, String> {
    let arguments: Value = serde_json::from_str(arguments)
        .map_err(|e| format!("Error: the arguments aren't valid JSON ({e}); give {{\"query\": \"...\"}}."))?;
    arguments["query"]
        .as_str()
        .map(str::trim)
        .filter(|q| !q.is_empty())
        .map(str::to_string)
        .ok_or_else(|| "Error: no `query` given; give {\"query\": \"...\"}.".to_string())
}

/// The URL and start a `web_fetch` call of `arguments` asks for, or what to
/// tell the model.
pub fn fetch_args(arguments: &str) -> Result<(String, usize), String> {
    let arguments: Value = serde_json::from_str(arguments)
        .map_err(|e| format!("Error: the arguments aren't valid JSON ({e}); give {{\"url\": \"https://...\"}}."))?;
    let url = arguments["url"].as_str().map(str::trim).filter(|u| !u.is_empty());
    let Some(url) = url else {
        return Err("Error: no `url` given; give {\"url\": \"https://...\"}.".to_string());
    };
    let lower = url.to_ascii_lowercase();
    if !lower.starts_with("http://") && !lower.starts_with("https://") {
        return Err(format!("Error: `{url}` isn't an http(s) URL; give the page's full address."));
    }
    let start = arguments["start"].as_u64().unwrap_or(0) as usize;
    Ok((url.to_string(), start))
}

/// Search the web for `query` and read the top pages.  Never fails: why it
/// couldn't search is the result.
pub fn search(query: &str, said: impl Fn(&str) + Send + Sync + 'static) -> String {
    let agent = WebSearchAgent::new(None, TOP, CANDIDATES, None);
    // What the agent says of each step, a page at a time: "Reading docs.rs".
    let progress: ProgressCallback = Arc::new(move |_, _, message| said(message));
    let links = match agent.fetch_links(query, CANDIDATES) {
        Ok(links) => links,
        Err(SearchProviderError::NoSearchCredentials(_)) => {
            return "Web search isn't available: not signed in to Smartloop. Go on without it.".to_string();
        }
        Err(SearchProviderError::SearchAuthError(_)) => {
            return "Web search isn't available for this account, or this month's searches are used up. Go on \
                    without it."
                .to_string();
        }
        Err(e) => return format!("The web search failed ({e}). Go on without it, or try another query."),
    };
    if links.is_empty() {
        return format!("No web results for \"{query}\"; try other words.");
    }
    // No rerank model, so fetching more than TOP usable pages would only
    // read pages that get thrown away: stop at TOP, skipping failed links.
    agent.fetch_top(query, &links, TOP, TOP, Some(&progress))
}

/// Read the page at `url` from character `start`.  Never fails: why the page
/// couldn't be read is the result.
pub fn fetch(url: &str, start: usize) -> String {
    // One character past the page, to tell whether there's more.
    let text = FetchAgent::new().fetch(url, PAGE + 1, start, false);
    page(url, start, &text)
}

/// What goes back for `text`, read from `url` at `start` with one character
/// past the page asked for.
pub fn page(url: &str, start: usize, text: &str) -> String {
    if text.trim().is_empty() {
        return if start > 0 {
            format!("Nothing more at {url} from character {start}: that was the whole page.")
        } else {
            format!("Couldn't read {url}: it failed to load, refused the request, or has no text. Go on without it.")
        };
    }
    let body: String = text.chars().take(PAGE).collect();
    let more = text.chars().count() > PAGE;
    let mut out = format!("{url}\n\n");
    out.push_str(body.trim_end());
    if more {
        out.push_str(&format!(
            "\n\n[The page goes on: call `{WEB_FETCH}` with \"start\": {} for the next part.]",
            start + PAGE
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_call_gives_its_query() {
        assert_eq!(query(r#"{"query": "  tokio 1.40 select macro "}"#), Ok("tokio 1.40 select macro".into()));
        assert!(query(r#"{"query": ""}"#).unwrap_err().contains("no `query` given"));
        assert!(query("{oops").unwrap_err().contains("aren't valid JSON"));
    }

    #[test]
    fn a_fetch_gives_its_url_and_start() {
        assert_eq!(fetch_args(r#"{"url": " https://docs.rs/tokio "}"#), Ok(("https://docs.rs/tokio".into(), 0)));
        assert_eq!(fetch_args(r#"{"url": "https://a.b/c", "start": 12000}"#), Ok(("https://a.b/c".into(), 12000)));
        assert!(fetch_args(r#"{"url": "file:///etc/passwd"}"#).unwrap_err().contains("isn't an http(s) URL"));
        assert!(fetch_args(r#"{}"#).unwrap_err().contains("no `url` given"));
    }

    #[test]
    fn a_long_page_says_where_it_goes_on() {
        let long = "q".repeat(PAGE + 1);
        let shown = page("https://a.b", 0, &long);
        assert!(shown.contains(&format!("\"start\": {PAGE}")), "{shown}");
        assert_eq!(shown.matches('q').count(), PAGE);
        assert!(!page("https://a.b", 0, "short page").contains("goes on"));
        assert!(page("https://a.b", 0, "  ").starts_with("Couldn't read"));
        assert!(page("https://a.b", PAGE, "").contains("whole page"));
    }

    #[test]
    fn the_framework_has_both_tools() {
        let names: Vec<String> = definitions().iter().map(|t| tool_name(t).to_string()).collect();
        assert_eq!(names, vec![WEB_SEARCH, WEB_FETCH]);
        assert!(is_builtin(WEB_SEARCH) && is_builtin(WEB_FETCH) && !is_builtin("bash"));
    }

    #[test]
    fn a_client_extends_the_tools_and_replaces_one_by_name() {
        let bash = json!({"type": "function", "function": {"name": "bash"}});
        // Its own tool is added to the framework's, which the framework runs.
        let (tools, ours) = offered(std::slice::from_ref(&bash), definitions());
        let names: Vec<&str> = tools.iter().map(tool_name).collect();
        assert_eq!(names, vec!["bash", WEB_SEARCH, WEB_FETCH]);
        assert_eq!(ours, vec![WEB_SEARCH, WEB_FETCH]);

        // One of the framework's names is the client's to run.
        let own_search = json!({"type": "function", "function": {"name": WEB_SEARCH, "description": "mine"}});
        let (tools, ours) = offered(&[bash, own_search], definitions());
        let names: Vec<&str> = tools.iter().map(tool_name).collect();
        assert_eq!(names, vec!["bash", WEB_SEARCH, WEB_FETCH]);
        assert_eq!(tools[1]["function"]["description"], "mine");
        assert_eq!(ours, vec![WEB_FETCH]);

        // No client tools: the framework's alone.
        let (tools, ours) = offered(&[], definitions());
        assert_eq!(tools.len(), 2);
        assert_eq!(ours.len(), 2);
    }

    #[test]
    fn the_model_is_told_it_can_search() {
        let both = vec![WEB_SEARCH.to_string(), WEB_FETCH.to_string()];
        let prompt = with_guidance("<system>Be helpful.</system>\n<user>news?</user>", &both);
        assert!(prompt.starts_with("<system>Be helpful.\nWEB: You can search the web."), "{prompt}");
        assert!(prompt.contains("`web_fetch` tool.</system>\n<user>news?</user>"), "{prompt}");
        // Only what the framework runs: a client that replaced the search says its own.
        let fetch_only = guidance(&[WEB_FETCH.to_string()]).unwrap();
        assert!(!fetch_only.contains("web_search") && fetch_only.contains("web_fetch"));
        assert_eq!(guidance(&[]), None);
        assert_eq!(with_guidance("<system>x</system>", &[]), "<system>x</system>");
    }

    #[test]
    fn a_long_result_is_cut_and_says_so() {
        assert_eq!(clip("short".to_string(), 100), "short");
        let cut = clip("é".repeat(500), 100);
        assert_eq!(cut.matches('é').count(), 100);
        assert!(cut.ends_with("for more of that page.]"), "{cut}");
    }

    #[test]
    fn a_call_says_what_it_is_doing() {
        assert_eq!(describe(WEB_SEARCH, r#"{"query": "rust 2024"}"#), "Searching the web for \"rust 2024\"");
        assert_eq!(describe(WEB_FETCH, r#"{"url": "https://docs.rs/tokio/latest"}"#), "Reading docs.rs");
        assert_eq!(describe(WEB_SEARCH, "{oops"), "Searching the web");
    }

    #[test]
    fn a_bad_call_is_answered_without_the_network() {
        assert!(run(WEB_SEARCH, "{}", |_| {}).contains("no `query` given"));
        assert!(run(WEB_FETCH, r#"{"url": "ftp://a.b"}"#, |_| {}).contains("isn't an http(s) URL"));
        assert!(run("bash", "{}", |_| {}).contains("no `bash` tool"));
    }
}
