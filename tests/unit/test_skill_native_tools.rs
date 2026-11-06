//! A matched skill guides tool steps; it must not take them off the model's
//! native tool calling.
//!
//! A skill like "linear-issue" says in prose "pull the latest issues from the
//! Linear MCP".  With that skill in its prompt the planner names a tool for
//! the step, and a named tool used to skip both the native tool loop and
//! native resolution.  `open_skill_tool_steps` leaves such a step open (server
//! kept, the planner's pick as a hint) unless the skill author pinned it.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use smartloop::agents::orchestrator::SLPOrchestrator;
use smartloop::agents::planner::{PlanStep, StepType};

use crate::shared::agents_core::{llm_8k, orchestrator_over};

const LINEAR_SKILL: &str = "# linear-issue\n\n\
    When the user asks about their work, pull the latest issues from the Linear MCP \
    and summarize them. If any issue is overdue, post a comment on it.";

fn orchestrator() -> SLPOrchestrator {
    orchestrator_over(&llm_8k("OK"))
}

fn linear_step(step: i64, tool: &str) -> PlanStep {
    PlanStep {
        step,
        step_type: StepType::Tool,
        tool_name: Some(tool.to_string()),
        server_id: Some("linear-srv".to_string()),
        server_name: Some("linear".to_string()),
        instruction: "Pull the latest issues".to_string(),
        skill_name: Some("linear-issue".to_string()),
        ..Default::default()
    }
}

#[test]
fn skill_planned_tool_step_is_left_open_for_the_model() {
    let orch = orchestrator();
    let mut steps = vec![linear_step(1, "list_issues")];
    orch.pipeline().open_skill_tool_steps(&mut steps, LINEAR_SKILL);

    let step = &steps[0];
    assert_eq!(step.tool_name, None);
    assert_eq!(step.server_id.as_deref(), Some("linear-srv"));
    assert!(step.instruction.starts_with("Pull the latest issues"));
    assert!(step.instruction.contains("`list_issues`"));
    assert!(step.instruction.contains("linear"));
}

#[test]
fn author_pinned_steps_stay_pinned() {
    let orch = orchestrator();
    let mut defaults = BTreeMap::new();
    defaults.insert("team".to_string(), "DEV".to_string());
    let mut steps = vec![
        PlanStep { default_args: Some(defaults), ..linear_step(1, "list_issues") },
        PlanStep { inferred_args: Some(vec!["query".to_string()]), ..linear_step(2, "search_issues") },
        PlanStep { for_each: Some(true), ..linear_step(3, "get_issue") },
        PlanStep { condition: Some("llm:Any overdue?".to_string()), ..linear_step(4, "save_comment") },
    ];
    orch.pipeline().open_skill_tool_steps(&mut steps, LINEAR_SKILL);

    let names: Vec<_> = steps.iter().map(|s| s.tool_name.as_deref()).collect();
    assert_eq!(
        names,
        vec![Some("list_issues"), Some("search_issues"), Some("get_issue"), Some("save_comment")]
    );
}

#[test]
fn builtin_system_steps_stay_pinned() {
    let orch = orchestrator();
    let mut steps = vec![PlanStep {
        server_id: Some("__system_tools__".to_string()),
        ..linear_step(1, "bash")
    }];
    orch.pipeline().open_skill_tool_steps(&mut steps, LINEAR_SKILL);
    assert_eq!(steps[0].tool_name.as_deref(), Some("bash"));
}

#[test]
fn conditional_later_step_keeps_its_tool_and_records_the_condition() {
    let orch = orchestrator();
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    orch.skills().set_condition_extractor(Some(Arc::new(move |_text: &str, tool: &str| {
        seen.fetch_add(1, Ordering::SeqCst);
        (tool == "save_comment").then(|| "Is any issue overdue?".to_string())
    })));

    let mut steps = vec![linear_step(1, "list_issues"), linear_step(2, "save_comment")];
    orch.pipeline().open_skill_tool_steps(&mut steps, LINEAR_SKILL);

    // The first step has nothing upstream to be conditional on.
    assert_eq!(steps[0].tool_name, None);
    assert_eq!(steps[1].tool_name.as_deref(), Some("save_comment"));
    assert_eq!(steps[1].condition.as_deref(), Some("Is any issue overdue?"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn unconditional_later_step_is_opened_too() {
    let orch = orchestrator();
    orch.skills().set_condition_extractor(Some(Arc::new(|_: &str, _: &str| None)));
    let mut steps = vec![linear_step(1, "list_issues"), linear_step(2, "get_issue")];
    orch.pipeline().open_skill_tool_steps(&mut steps, LINEAR_SKILL);
    assert!(steps.iter().all(|s| s.tool_name.is_none()));
}

#[test]
fn non_tool_steps_are_untouched() {
    let orch = orchestrator();
    let explore = PlanStep {
        step: 1,
        step_type: StepType::Explore,
        tool_name: Some("explore".to_string()),
        instruction: "look around".to_string(),
        ..Default::default()
    };
    let mut steps = vec![explore.clone()];
    orch.pipeline().open_skill_tool_steps(&mut steps, LINEAR_SKILL);
    assert_eq!(steps[0], explore);
}

mod step_servers {
    use smartloop::agents::orchestration::pipeline_builder::step_servers;
    use smartloop::agents::planner::{PlanStep, StepType};
    use smartloop::workspace::workspace::MCPServer;

    fn server(id: &str) -> MCPServer {
        MCPServer { id: id.to_string(), name: id.to_string(), ..Default::default() }
    }

    fn step(server_id: Option<&str>) -> PlanStep {
        PlanStep {
            step: 1,
            step_type: StepType::Tool,
            server_id: server_id.map(str::to_string),
            ..Default::default()
        }
    }

    #[test]
    fn a_step_naming_its_server_sees_only_that_server() {
        let servers = vec![server("clera"), server("__skills__"), server("linear")];
        let ids: Vec<_> = step_servers(&step(Some("__skills__")), &servers).into_iter().map(|s| s.id).collect();
        assert_eq!(ids, vec!["__skills__"]);
    }

    #[test]
    fn an_open_or_unknown_server_sees_every_server() {
        let servers = vec![server("clera"), server("linear")];
        assert_eq!(step_servers(&step(None), &servers).len(), 2);
        assert_eq!(step_servers(&step(Some("gone")), &servers).len(), 2);
    }
}
