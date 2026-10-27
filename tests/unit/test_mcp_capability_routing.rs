//! Port of `tests/unit/test_mcp_capability_routing.py`.
//!
//! An MCP tool call follows the step's capability, like every other node.
//!
//! `pipeline_builder` tags a non-skill TOOL step `SMALL` and *still* sets its
//! capability, but `ModelRouter::llm_for_tier` only consulted the capability
//! provider on the `LARGE` branch — so the tag was read by nothing and both
//! halves of an MCP call (picking the tool, extracting its arguments) ran on
//! the bootstrap router.  It only showed with two or more models selected: a
//! single selection is pinned into `agent_llm` upstream, so the default *was*
//! the user's model.
//!
//! SMALL routes best-effort — `decision_llm` takes the capability model only
//! when it is resident or fits — so honouring the tag never forces a load onto
//! a memory-tight device.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use serial_test::serial;

use smartloop::agents::mcp::mcp_agent::{clear_resolution_cache, MCPAgent};
use smartloop::agents::orchestration::model_router::{ModelRouter, ModelRouterOptions, ProgressCallback};
use smartloop::agents::planner::ModelTier;
use smartloop::llms::BaseLLM;
use smartloop::types::ModelCapability;
use smartloop::workspace::workspace::{MCPServer, MCPToolDefinition};

use crate::shared::agents_mcp_agent::{same_llm, StubLLM};

/// A model named the way the GGUF metadata names it.
fn named_llm(name: &str) -> Arc<StubLLM> {
    StubLLM::labelled(name)
}

/// A progress sink that records every `(step, status, message)`.
fn recording_progress() -> (ProgressCallback, Arc<Mutex<Vec<(String, String, String)>>>) {
    let calls: Arc<Mutex<Vec<(String, String, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = calls.clone();
    let callback: ProgressCallback = Arc::new(move |step: &str, status: &str, message: &str| {
        sink.lock().push((step.to_string(), status.to_string(), message.to_string()));
    });
    (callback, calls)
}

/// The message argument of every progress call (Python's `call.args[2]`).
fn messages(calls: &Arc<Mutex<Vec<(String, String, String)>>>) -> Vec<String> {
    calls.lock().iter().map(|(_, _, m)| m.clone()).collect()
}

/// A router whose capability provider answers `capability_llm` for every
/// capability.
fn router_for(capability_llm: Option<Arc<dyn BaseLLM>>) -> (ModelRouter, Arc<dyn BaseLLM>) {
    let agent_llm = named_llm("Slp Orchestra Mini").as_llm();
    let provided = capability_llm.clone();
    let available = capability_llm.is_some();
    let router = ModelRouter::new(
        named_llm("Slp Orchestra Mini").as_llm(),
        agent_llm.clone(),
        ModelRouterOptions {
            capability_llm_provider: Some(Arc::new(move |_cap: &str| Ok(provided.clone()))),
            capability_available: Some(Arc::new(move |_cap: &str| Ok(available))),
            ..Default::default()
        },
    );
    let (progress, _calls) = recording_progress();
    router.begin_request(Some(progress));
    (router, agent_llm)
}

// ------------------------------------------------------ TestSmallTierHonoursCapability

#[test]
fn test_a_tagged_small_step_gets_the_capability_model() {
    let cap_llm = named_llm("Gemma 3n E2B").as_llm();
    let (router, agent_llm) = router_for(Some(cap_llm.clone()));

    let resolved = router.llm_for_tier(ModelTier::Small, Some(ModelCapability::Coding.value()), None);

    assert!(same_llm(&resolved, &cap_llm));
    assert!(!same_llm(&resolved, &agent_llm));
}

#[test]
fn test_an_untagged_small_step_still_gets_the_agent_model() {
    let (router, agent_llm) = router_for(Some(named_llm("Gemma 3n E2B").as_llm()));

    assert!(same_llm(&router.llm_for_tier(ModelTier::Small, None, None), &agent_llm));
}

/// decision_llm never forces a load — a declined provider lands here.
#[test]
fn test_it_falls_back_when_the_capability_model_will_not_load() {
    let (router, agent_llm) = router_for(None);

    let resolved = router.llm_for_tier(ModelTier::Small, Some(ModelCapability::Coding.value()), None);

    assert!(same_llm(&resolved, &agent_llm));
}

#[test]
fn test_a_provider_failure_does_not_break_the_step() {
    let agent_llm = named_llm("Slp Orchestra Mini").as_llm();
    let router = ModelRouter::new(
        agent_llm.clone(),
        agent_llm.clone(),
        ModelRouterOptions {
            capability_llm_provider: Some(Arc::new(|_cap: &str| Err(anyhow::anyhow!("oom")))),
            ..Default::default()
        },
    );
    let (progress, _calls) = recording_progress();
    router.begin_request(Some(progress));

    assert!(same_llm(
        &router.llm_for_tier(ModelTier::Small, Some(ModelCapability::Dp.value()), None),
        &agent_llm
    ));
}

#[test]
fn test_the_large_branch_is_unchanged() {
    let cap_llm = named_llm("Gemma 3n E2B").as_llm();
    let (router, _agent_llm) = router_for(Some(cap_llm.clone()));

    assert!(same_llm(
        &router.llm_for_tier(ModelTier::Large, Some(ModelCapability::Reasoning.value()), None),
        &cap_llm
    ));
}

// --------------------------------------------------- TestToolResolutionTakesTheOverride
// The pick is the harder judgement of the two — upgrading only the argument
// extraction just got the arguments right for the wrong tool.

fn obj(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(m) => m,
        other => panic!("expected an object, got {other}"),
    }
}

fn agent_with_two_tools(default_llm: &Arc<StubLLM>) -> (MCPAgent, Vec<MCPServer>) {
    let agent = MCPAgent::new(default_llm.as_llm());
    let server = MCPServer {
        id: "srv-1".to_string(),
        name: "linear".to_string(),
        enabled: true,
        system: false,
        tools: vec![
            MCPToolDefinition {
                name: "list_issues".to_string(),
                description: Some("List issues".to_string()),
                input_schema: Some(obj(json!({"type": "object", "properties": {}}))),
            },
            MCPToolDefinition {
                name: "create_issue".to_string(),
                description: Some("Create an issue".to_string()),
                input_schema: Some(obj(json!({
                    "type": "object",
                    "properties": {"title": {"type": "string", "description": "The issue title"}},
                    "required": ["title"],
                }))),
            },
        ],
        ..Default::default()
    };
    (agent, vec![server])
}

#[test]
#[serial]
fn test_the_override_model_resolves_the_call() {
    clear_resolution_cache();
    let default_llm = named_llm("Slp Orchestra Mini");
    default_llm.set_reply(r#"{"tool": 2}"#);
    let capability_llm = named_llm("Gemma 3n E2B");
    capability_llm.set_reply(r#"{"tool": 1}"#);
    let (agent, servers) = agent_with_two_tools(&default_llm);

    let resolved = agent
        .resolve("list my open issues", &servers, None, Some(capability_llm.as_llm()))
        .unwrap();

    assert_eq!(
        (resolved.server_id.as_str(), resolved.tool_name.as_str()),
        ("srv-1", "list_issues")
    );
    assert_eq!(capability_llm.call_count(), 1);
    assert_eq!(default_llm.call_count(), 0);
    clear_resolution_cache();
}

#[test]
#[serial]
fn test_without_an_override_the_agent_model_still_resolves() {
    clear_resolution_cache();
    let default_llm = named_llm("Slp Orchestra Mini");
    default_llm.set_reply(r#"{"tool": 2, "arguments": {"title": "Login is broken"}}"#);
    let (agent, servers) = agent_with_two_tools(&default_llm);

    let resolved = agent
        .resolve("file an issue: Login is broken", &servers, None, None)
        .unwrap();

    assert_eq!(
        (resolved.server_id.as_str(), resolved.tool_name.as_str()),
        ("srv-1", "create_issue")
    );
    // The pick and its required argument come out of the same call.
    assert_eq!(resolved.arguments, obj(json!({"title": "Login is broken"})));
    assert_eq!(default_llm.call_count(), 1);
    clear_resolution_cache();
}

// ------------------------------------------------ TestTheReasoningModelIsAnnouncedAsWork
// What the user is told about the model that resolves a tool call.
//
// `thinking` is an internal routing tag: the API leaves it out of the
// capabilities it reports, and it names a property of the model rather than
// the work in progress.  Surfaced verbatim it read "Selected qwen3.5-2b for
// thinking" — and on the single-model workspace where that model also writes
// the answer, it landed directly beside "Selected qwen3.5-2b for the
// response": one model, twice, under two labels.

type Recorded = Arc<Mutex<Vec<(String, String, String)>>>;

fn reasoning_router(capability_llm: Option<Arc<dyn BaseLLM>>) -> (ModelRouter, Arc<dyn BaseLLM>, Recorded) {
    let agent = named_llm("Qwen3.5 2B");
    agent.set_supports_thinking(false);
    let agent_llm = agent.as_llm();
    let provided = capability_llm.clone();
    let available = capability_llm.is_some();
    let router = ModelRouter::new(
        agent_llm.clone(),
        agent_llm.clone(),
        ModelRouterOptions {
            capability_llm_provider: Some(Arc::new(move |_cap: &str| Ok(provided.clone()))),
            capability_available: Some(Arc::new(move |_cap: &str| Ok(available))),
            ..Default::default()
        },
    );
    let (progress, calls) = recording_progress();
    router.begin_request(Some(progress));
    (router, agent_llm, calls)
}

#[test]
fn test_a_switch_is_announced_as_tool_resolution() {
    let reasoner = named_llm("Qwen3 4B");
    reasoner.set_supports_thinking(true);
    let reasoner = reasoner.as_llm();
    let (router, agent_llm, progress) = reasoning_router(Some(reasoner.clone()));

    assert!(same_llm(&router.reasoning_llm(agent_llm), &reasoner));

    let messages = messages(&progress);
    assert_eq!(messages, vec!["Selected Qwen3 4B for tool resolution".to_string()]);
    assert!(!messages.iter().any(|m| m.contains("thinking")));
}

/// Nothing was switched, so there is nothing to report.
#[test]
fn test_landing_on_the_model_already_in_use_is_not_announced() {
    let agent = named_llm("Qwen3.5 2B");
    agent.set_supports_thinking(false);
    let agent_llm = agent.as_llm();
    let provided = agent_llm.clone();
    let router = ModelRouter::new(
        agent_llm.clone(),
        agent_llm.clone(),
        ModelRouterOptions {
            // The workspace's reasoning model *is* the model this request runs
            // on — the single-model case.
            capability_llm_provider: Some(Arc::new(move |_cap: &str| Ok(Some(provided.clone())))),
            capability_available: Some(Arc::new(|_cap: &str| Ok(true))),
            ..Default::default()
        },
    );
    let (progress, calls) = recording_progress();
    router.begin_request(Some(progress));

    router.reasoning_llm(agent_llm);

    assert!(messages(&calls).is_empty());
}

#[test]
fn test_a_fallback_that_already_reasons_is_kept_and_silent() {
    let reasoner = named_llm("Qwen3 4B");
    reasoner.set_supports_thinking(true);
    let reasoner = reasoner.as_llm();
    let (router, _agent_llm, progress) = reasoning_router(Some(named_llm("Gemma 3n E2B").as_llm()));

    assert!(same_llm(&router.reasoning_llm(reasoner.clone()), &reasoner));
    assert!(messages(&progress).is_empty());
}

#[test]
fn test_no_reasoning_model_in_the_workspace_says_nothing() {
    let (router, agent_llm, progress) = reasoning_router(None);

    assert!(same_llm(&router.reasoning_llm(agent_llm.clone()), &agent_llm));
    assert!(messages(&progress).is_empty());
}

/// Two or more models selected: the request itself runs on the small router
/// model, so a capability model is always a real switch.
fn multi_model_router(
    capability_llm: Arc<dyn Fn(&str) -> Option<Arc<dyn BaseLLM>> + Send + Sync>,
) -> (ModelRouter, Arc<dyn BaseLLM>, Recorded) {
    let router_model = named_llm("Slp Orchestra Mini");
    router_model.set_supports_thinking(false);
    let router_llm = router_model.as_llm();
    let provider = capability_llm.clone();
    let available = capability_llm.clone();
    let router = ModelRouter::new(
        router_llm.clone(),
        router_llm.clone(),
        ModelRouterOptions {
            capability_llm_provider: Some(Arc::new(move |cap: &str| Ok(provider(cap)))),
            capability_available: Some(Arc::new(move |cap: &str| Ok(available(cap).is_some()))),
            ..Default::default()
        },
    );
    let (progress, calls) = recording_progress();
    router.begin_request(Some(progress));
    (router, router_llm, calls)
}

/// One model serving two capabilities is one selection, not two.
///
/// Synthesis names it for `skills`; the tool call lands on the same model a
/// moment later.  Both are switches away from the router model, so neither is
/// suppressed by the "already in use" check — and the user read the same model
/// being picked twice:
///
///     Selected qwen3.5-2b for reasoning
///     Selected qwen3.5-2b for tool resolution
#[test]
fn test_a_model_already_announced_is_not_announced_again() {
    let worker = named_llm("Qwen3.5 2B");
    worker.set_supports_thinking(true);
    let worker = worker.as_llm();
    let (router, router_llm, progress) = multi_model_router(Arc::new(move |_cap: &str| Some(worker.clone())));

    router.llm_for_tier(ModelTier::Large, Some(ModelCapability::Reasoning.value()), None);
    router.reasoning_llm(router_llm);

    let selected: Vec<String> = messages(&progress)
        .into_iter()
        .filter(|m| m.starts_with("Selected"))
        .collect();
    assert_eq!(selected, vec!["Selected Qwen3.5 2B for reasoning".to_string()]);
}

/// Suppression is about repetition, not about the second event: work moving
/// to another model is exactly what the line is for.
#[test]
fn test_a_different_model_for_the_tool_call_is_still_announced() {
    let worker = named_llm("Qwen3.5 2B");
    worker.set_supports_thinking(false);
    let worker = worker.as_llm();
    let reasoner = named_llm("Qwen3 4B");
    reasoner.set_supports_thinking(true);
    let reasoner = reasoner.as_llm();
    let (router, router_llm, progress) = multi_model_router(Arc::new(move |cap: &str| {
        if cap == ModelCapability::Reasoning.value() {
            Some(reasoner.clone())
        } else {
            Some(worker.clone())
        }
    }));

    router.llm_for_tier(ModelTier::Large, Some(ModelCapability::Dp.value()), None);
    router.reasoning_llm(router_llm);

    let selected: Vec<String> = messages(&progress)
        .into_iter()
        .filter(|m| m.starts_with("Selected"))
        .collect();
    assert_eq!(
        selected,
        vec![
            "Selected Qwen3.5 2B for document processing".to_string(),
            "Selected Qwen3 4B for tool resolution".to_string(),
        ]
    );
}

// ------------------------------------------------ TestToolResolutionPromotesToReasoning
// The router model (sl-mini) only routes.  When no thinking model is on
// device, tool resolution moves up to a model declaring `reasoning` — a cloud
// one included — rather than staying on the router model.

/// Each lookup the provider saw: `(capability, local_only)`.
type Lookups = Arc<Mutex<Vec<(String, bool)>>>;

fn promoting_router(
    capability_llm: Arc<dyn Fn(&str) -> Option<Arc<dyn BaseLLM>> + Send + Sync>,
    declared: &'static [&'static str],
) -> (ModelRouter, Arc<dyn BaseLLM>, Recorded, Lookups) {
    let router_model = named_llm("sl-mini");
    router_model.set_supports_thinking(false);
    let router_llm = router_model.as_llm();
    let lookups: Lookups = Arc::new(Mutex::new(Vec::new()));
    let seen = lookups.clone();
    let router = ModelRouter::new(
        router_llm.clone(),
        router_llm.clone(),
        ModelRouterOptions {
            capability_llm_provider: Some(Arc::new(move |cap: &str| {
                seen.lock().push((cap.to_string(), smartloop::llms::cloud_llm::local_only()));
                Ok(capability_llm(cap))
            })),
            capability_available: Some(Arc::new(move |cap: &str| Ok(declared.contains(&cap)))),
            ..Default::default()
        },
    );
    let (progress, calls) = recording_progress();
    router.begin_request(Some(progress));
    (router, router_llm, calls, lookups)
}

#[test]
fn test_tool_resolution_promotes_to_a_remote_reasoning_model() {
    let cloud = named_llm("minimax-m3");
    cloud.set_supports_thinking(false);
    let cloud = cloud.as_llm();
    let provided = cloud.clone();
    let (router, router_llm, progress, lookups) =
        promoting_router(Arc::new(move |_cap: &str| Some(provided.clone())), &["general", "reasoning"]);

    assert!(same_llm(&router.reasoning_llm(router_llm), &cloud));
    // No thinking model is selected, so only the reasoning lookup ran — and
    // it was allowed off device.
    assert_eq!(*lookups.lock(), vec![("reasoning".to_string(), false)]);
    assert_eq!(messages(&progress), vec!["Selected minimax-m3 for tool resolution".to_string()]);
}

/// A reasoning model is asked first — the best one, a cloud one included —
/// so a small local thinking model does not take the call from it.
#[test]
fn test_a_reasoning_model_is_preferred_over_a_thinking_one() {
    let small_thinker = named_llm("qwen3.5-2b");
    small_thinker.set_supports_thinking(true);
    let small_thinker = small_thinker.as_llm();
    let cloud = named_llm("minimax-m3").as_llm();
    let (thinking_pick, reasoning_pick) = (small_thinker, cloud.clone());
    let (router, router_llm, progress, lookups) = promoting_router(
        Arc::new(move |cap: &str| {
            if cap == ModelCapability::Thinking.value() {
                Some(thinking_pick.clone())
            } else {
                Some(reasoning_pick.clone())
            }
        }),
        &["general", "reasoning", "thinking"],
    );

    assert!(same_llm(&router.reasoning_llm(router_llm), &cloud));
    assert_eq!(*lookups.lock(), vec![("reasoning".to_string(), false)]);
    assert_eq!(messages(&progress), vec!["Selected minimax-m3 for tool resolution".to_string()]);
}

/// With no reasoning model selected, a thinking model resolves on device.
/// A thinking lookup that lands on a model that cannot think (the host's
/// fallback to the router model) is neither used nor announced.
#[test]
fn test_a_thinking_model_resolves_when_none_declares_reasoning() {
    let local = named_llm("qwen3-thinker");
    local.set_supports_thinking(true);
    let local = local.as_llm();
    let provided = local.clone();
    let (router, router_llm, _progress, lookups) =
        promoting_router(Arc::new(move |_cap: &str| Some(provided.clone())), &["thinking"]);
    assert!(same_llm(&router.reasoning_llm(router_llm), &local));
    assert_eq!(*lookups.lock(), vec![("thinking".to_string(), true)]);

    let cannot_think = named_llm("sl-mini");
    cannot_think.set_supports_thinking(false);
    let cannot_think = cannot_think.as_llm();
    let (router, router_llm, progress, _lookups) =
        promoting_router(Arc::new(move |_cap: &str| Some(cannot_think.clone())), &["thinking"]);
    assert!(same_llm(&router.reasoning_llm(router_llm.clone()), &router_llm));
    assert!(messages(&progress).is_empty());
}

/// No selected model reasons: the router model keeps the call.
#[test]
fn test_without_a_reasoning_model_the_router_model_resolves() {
    let (router, router_llm, progress, lookups) =
        promoting_router(Arc::new(|_cap: &str| None), &["general"]);

    assert!(same_llm(&router.reasoning_llm(router_llm.clone()), &router_llm));
    assert!(lookups.lock().is_empty());
    assert!(messages(&progress).is_empty());
}
