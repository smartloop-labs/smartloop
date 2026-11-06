//! A model's own tool loop over the connected MCP servers.
//!
//! For a model that calls functions natively (`BaseLLM::supports_native_tools`)
//! a tool step is not "resolve one call, then run it": the model is given the
//! whole catalog as tool definitions and the request, calls what it needs,
//! sees each result as a `role: tool` message, and calls again — "find Chris,
//! then file the issue assigned to him" is two turns of one conversation, the
//! second filled from the first's result.  It stops when it answers in text
//! (what it did, or why it could not), when a tool fails the same way twice
//! (the model is not correcting itself), or at [`MAX_TURNS`].
//!
//! The loop owns no transport: calls go through the caller's `call` function
//! (the same `MCPToolCaller` path a planned step uses, token refresh
//! included), and progress through `emit`.  The conversation reaches the
//! model as a prompt carrying `<slp_tools>` and `<slp_messages>` blocks
//! ([`crate::llms::tool_calls`]), so it crosses the model host like any
//! prompt.  `None` from [`run_tool_loop`] means the loop could not start —
//! the catalog does not fit, or the first turn failed — and the caller falls
//! back to the single resolve-and-call step.

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Map, Value};

use crate::agents::mcp::mcp_agent::{format_call, MCPAgent};
use crate::llms::tool_calls::{
    assistant_tool_turn, parse_tool_calls, tool_result_turn, with_tools_and_messages, NativeToolCall,
};
use crate::llms::BaseLLM;
use crate::workspace::workspace::MCPServer;

const LOGGER: &str = "smartloop.agents.mcp.tool_loop";

/// Model turns before the loop stops: enough for a lookup, an action and a
/// follow-up, not enough to wander.
pub const MAX_TURNS: usize = 6;

/// Output budget per turn: room to reason and to write a call whose
/// arguments carry prose (an issue description).
pub const TURN_MAX_TOKENS: usize = 2048;

/// The most of one tool result fed back to the model.
pub const MAX_RESULT_CHARS: usize = 6000;

const LOOP_SYSTEM: &str = "You carry out the user's request using the provided tools.

- Call tools as the request needs, one step at a time; you will see each result before you continue. Look up what an action needs before acting (find a person's id before assigning them; find the team before creating an issue in it).
- Take ids, keys and URLs only from tool results or the request. Never invent them.
- Fill an optional argument only when the request or the conversation states or clearly implies its value — for something you create, that means its title, description, assignee, labels and the like.
- When the request is done, or cannot be done with these tools, stop calling tools and answer briefly: what you did, with the identifier or link of anything you created or changed, or what is missing.";

/// One call the loop made.
#[derive(Debug, Clone, PartialEq)]
pub struct LoopCall {
    pub server_id: String,
    pub server_name: String,
    pub tool_name: String,
    pub arguments: Map<String, Value>,
    /// What the call returned (or why it did not), as fed back to the model.
    pub result: String,
    pub ok: bool,
}

/// What the loop did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoopOutcome {
    pub calls: Vec<LoopCall>,
    /// The model's closing answer; empty when it was cut off by [`MAX_TURNS`].
    pub final_text: String,
    pub turns: usize,
}

impl LoopOutcome {
    /// The calls and their results, then the model's account — what the
    /// answer step is given.
    pub fn transcript(&self) -> String {
        let mut parts: Vec<String> = self
            .calls
            .iter()
            .map(|c| {
                format!(
                    "{} → {}",
                    format_call(&c.server_name, &c.tool_name, &c.arguments),
                    take_chars(c.result.trim(), 1500)
                )
            })
            .collect();
        if !self.final_text.trim().is_empty() {
            parts.push(format!("Outcome: {}", self.final_text.trim()));
        } else if self.turns >= MAX_TURNS {
            parts.push(format!("Outcome: stopped after {MAX_TURNS} turns without a final answer."));
        } else if self.calls.last().is_some_and(|c| !c.ok) {
            parts.push("Outcome: stopped — the last call kept failing with the same error.".to_string());
        }
        parts.join("\n\n")
    }

    /// The last call that succeeded, for the step's citation.
    pub fn last_ok(&self) -> Option<&LoopCall> {
        self.calls.iter().rev().find(|c| c.ok)
    }
}

/// Run `llm`'s tool loop for `instruction` over `servers`.
///
/// `call` runs one tool and returns its result text, or why it failed — the
/// model is shown the reason so it can fix its next call; `emit` reports
/// progress.
pub fn run_tool_loop(
    agent: &MCPAgent,
    llm: &dyn BaseLLM,
    instruction: &str,
    servers: &[MCPServer],
    call: &dyn Fn(&MCPServer, &str, &Map<String, Value>) -> Result<String, String>,
    emit: &dyn Fn(&str),
) -> Option<LoopOutcome> {
    let entries = agent.catalog(servers);
    if entries.is_empty() {
        tracing::info!(target: LOGGER, "[tool-loop] no callable tools");
        return None;
    }
    let (tools, names) = agent.native_tools(&entries);
    let size: usize = tools.iter().map(|t| t.to_string().chars().count()).sum();
    let budget = agent.catalog_budget(llm);
    if size > budget {
        tracing::info!(
            target: LOGGER,
            "[tool-loop] {} tool definition(s) need {} chars, over the {} this model can be shown",
            tools.len(),
            size,
            budget
        );
        return None;
    }

    tracing::info!(
        target: LOGGER,
        "[tool-loop] {} over {} tool(s), up to {} turns",
        MCPAgent::model_label(llm),
        tools.len(),
        MAX_TURNS
    );

    let mut messages: Vec<Value> = vec![
        json!({"role": "system", "content": LOOP_SYSTEM}),
        json!({"role": "user", "content": instruction}),
    ];
    let mut outcome = LoopOutcome::default();
    // (tool, error) of every failed call: a second identical failure means
    // the model is repeating itself rather than correcting the call.
    let mut failures: Vec<(String, String)> = Vec::new();

    for turn in 0..MAX_TURNS {
        outcome.turns = turn + 1;
        let prompt = with_tools_and_messages(&tools, &messages);
        let max_tokens = MCPAgent::budget(llm, &prompt, TURN_MAX_TOKENS);
        let text = llm.invoke(&prompt, Some(max_tokens), Some(true));

        if text.starts_with("Error generating response") || text == "No response generated" {
            tracing::warn!(target: LOGGER, "[tool-loop] turn {} failed: {}", turn + 1, take_chars(&text, 200));
            if turn == 0 {
                return None;
            }
            break;
        }

        let mut calls = parse_tool_calls(&text);
        let visible = strip_calls(&text);
        if calls.is_empty() {
            tracing::info!(target: LOGGER, "[tool-loop] done after {} turn(s), {} call(s)", turn + 1, outcome.calls.len());
            outcome.final_text = visible;
            return Some(outcome);
        }

        for (i, c) in calls.iter_mut().enumerate() {
            if c.id.as_deref().is_none_or(str::is_empty) {
                c.id = Some(format!("call_{}_{}", turn + 1, i + 1));
            }
        }
        messages.push(assistant_tool_turn(&visible, &calls));

        let mut repeated = false;
        for c in &calls {
            let id = c.id.clone().unwrap_or_default();
            let made = make_call(agent, &entries, &names, servers, c, call, emit);
            let fed = take_chars(&made.result, MAX_RESULT_CHARS);
            messages.push(tool_result_turn(&id, &c.name, &fed));
            if !made.ok {
                let failure = (made.tool_name.clone(), made.result.clone());
                repeated |= failures.contains(&failure);
                failures.push(failure);
            }
            outcome.calls.push(made);
        }
        if repeated {
            tracing::warn!(
                target: LOGGER,
                "[tool-loop] stopped after {} turn(s): a call failed again with the same error",
                turn + 1
            );
            return Some(outcome);
        }
    }

    tracing::warn!(target: LOGGER, "[tool-loop] stopped at {} turns, {} call(s)", MAX_TURNS, outcome.calls.len());
    Some(outcome)
}

/// Run one call the model asked for, or say why it could not run.
fn make_call(
    agent: &MCPAgent,
    entries: &[crate::agents::mcp::mcp_agent::Entry],
    names: &[String],
    servers: &[MCPServer],
    c: &NativeToolCall,
    call: &dyn Fn(&MCPServer, &str, &Map<String, Value>) -> Result<String, String>,
    emit: &dyn Fn(&str),
) -> LoopCall {
    let Some(index) = MCPAgent::native_entry(&c.name, names, entries) else {
        tracing::info!(target: LOGGER, "[tool-loop] model called unknown tool '{}'", c.name);
        return LoopCall {
            server_id: String::new(),
            server_name: String::new(),
            tool_name: c.name.clone(),
            arguments: c.arguments.clone(),
            result: format!("Error: there is no tool named '{}'.", c.name),
            ok: false,
        };
    };
    let entry = &entries[index];
    let arguments = agent.clean_arguments(&entry.schema, &c.arguments, true);
    let mut made = LoopCall {
        server_id: entry.server_id.clone(),
        server_name: entry.server_name.clone(),
        tool_name: entry.tool_name.clone(),
        arguments: arguments.clone(),
        result: String::new(),
        ok: false,
    };
    let Some(server) = servers.iter().find(|s| s.id == entry.server_id) else {
        made.result = format!("Error: the {} connection is not available.", entry.server_name);
        return made;
    };
    emit(&format!("Calling {}", format_call(&entry.server_name, &entry.tool_name, &arguments)));
    tracing::info!(
        target: LOGGER,
        "[tool-loop] calling {}",
        format_call(&entry.server_name, &entry.tool_name, &arguments)
    );
    match call(server, &entry.tool_name, &arguments) {
        Ok(result) if !result.trim().is_empty() => {
            made.result = result;
            made.ok = true;
        }
        Ok(_) => {
            made.result = "(the tool returned nothing)".to_string();
            made.ok = true;
        }
        Err(why) if !why.trim().is_empty() => made.result = format!("Error: {}", why.trim()),
        Err(_) => made.result = "Error: the call failed and returned no result.".to_string(),
    }
    made
}

/// The model's prose, without its tool-call blocks or reasoning.
fn strip_calls(text: &str) -> String {
    static CALLS: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?s)<tool_call>.*?(?:</tool_call>|$)").expect("valid regex"));
    CALLS.replace_all(text, "").trim().to_string()
}

fn take_chars(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n).collect();
        out.push('…');
        out
    }
}
